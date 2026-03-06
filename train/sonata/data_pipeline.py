#!/usr/bin/env python3
"""Multi-dataset encoding pipeline for scaling Sonata training to 10K+ hours.

Supports multiple speech corpora with a unified manifest format.
Encodes each dataset with the Sonata Codec and produces (text, semantic_tokens,
acoustic_latents, prosody_features) for LM and Flow training.

For large datasets, uses sharded output to avoid OOM: each shard is a separate
.pt file. A .shards.txt index file is written for easy loading.

Supported sources:
  - LibriSpeech directory layout (--source librispeech)
  - Common Voice TSV layout    (--source common_voice)
  - Unified JSONL manifest     (--source manifest, from acquire_data.py)

Usage:
  # Encode from unified manifest (recommended — use acquire_data.py first)
  python data_pipeline.py --source manifest \
    --manifest data/manifest_clean.jsonl \
    --codec-ckpt checkpoints/codec/sonata_codec_final.pt \
    --output data/encoded.pt --shard-size 10000

  # Encode LibriSpeech directly
  python data_pipeline.py --source librispeech \
    --split train-clean-100,train-clean-360,train-other-500 \
    --codec-ckpt checkpoints/codec/sonata_codec_final.pt \
    --output data/encoded_ls960.pt
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterator, Tuple

import torch
import torch.nn.functional as F
import soundfile as sf

from config import CodecConfig
from codec import SonataCodec
from encode_dataset import extract_prosody


def iter_librispeech(data_dir: str, splits: list) -> Iterator[dict]:
    """Iterate LibriSpeech / LibriLight entries."""
    for split in splits:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            print(f"[data] WARNING: {split_dir} not found, skipping")
            continue
        for speaker_dir in sorted(split_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            for chapter_dir in sorted(speaker_dir.iterdir()):
                if not chapter_dir.is_dir():
                    continue
                trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
                if not trans_file.exists():
                    continue
                transcripts = {}
                with open(trans_file) as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            transcripts[parts[0]] = parts[1]
                for flac in sorted(chapter_dir.glob("*.flac")):
                    utt_id = flac.stem
                    if utt_id in transcripts:
                        yield {
                            "audio": str(flac),
                            "text": transcripts[utt_id],
                            "utt_id": utt_id,
                            "speaker": speaker_dir.name,
                        }


def iter_common_voice(data_dir: str, splits: list) -> Iterator[dict]:
    """Iterate Mozilla Common Voice TSV entries."""
    for split in splits:
        tsv_path = Path(data_dir) / f"{split}.tsv"
        if not tsv_path.exists():
            print(f"[data] WARNING: {tsv_path} not found, skipping")
            continue
        clips_dir = Path(data_dir) / "clips"
        with open(tsv_path) as f:
            header = f.readline().strip().split("\t")
            path_idx = header.index("path")
            text_idx = header.index("sentence")
            for line in f:
                cols = line.strip().split("\t")
                if len(cols) > max(path_idx, text_idx):
                    audio_path = clips_dir / cols[path_idx]
                    if audio_path.exists():
                        yield {
                            "audio": str(audio_path),
                            "text": cols[text_idx],
                            "utt_id": audio_path.stem,
                        }


def iter_manifest(manifest_path: str) -> Iterator[dict]:
    """Iterate a JSONL manifest with {audio, text, ...} per line."""
    with open(manifest_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "audio" in entry and "text" in entry:
                yield entry


def get_data_iterator(args) -> Iterator[dict]:
    """Get the appropriate data iterator for the source type."""
    splits = [s.strip() for s in args.split.split(",")]
    if args.source == "librispeech":
        data_dir = args.data_dir or "data/LibriSpeech"
        return iter_librispeech(data_dir, splits)
    elif args.source == "common_voice":
        data_dir = args.data_dir or "data/common_voice"
        return iter_common_voice(data_dir, splits)
    elif args.source == "manifest":
        return iter_manifest(args.manifest)
    else:
        raise ValueError(f"Unknown source: {args.source}")


@torch.no_grad()
def encode_pipeline(args):
    """Encode audio to semantic tokens + acoustic latents with sharded output."""
    device = torch.device(args.device)

    ckpt = torch.load(args.codec_ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = CodecConfig(**{k: v for k, v in cfg_dict.items()
                         if k in CodecConfig.__dataclass_fields__})
    model = SonataCodec(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[encode] Loaded codec from {args.codec_ckpt}")
    print(f"[encode] Frame rate: {cfg.frame_rate} Hz, FSQ: {cfg.fsq_codebook_size}")
    print(f"[encode] Shard size: {args.shard_size} utterances")

    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    shard_buf = []
    shard_idx = 0
    total_encoded = 0
    total_samples = 0
    n_errors = 0
    shard_paths = []
    t0 = time.time()

    def flush_shard(buf, idx):
        nonlocal shard_paths
        if not buf:
            return
        if args.shard_size > 0 and args.shard_size < 999_999:
            path = out_base.with_suffix(f".shard{idx:04d}.pt")
        else:
            path = out_base
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(buf, path)
        shard_paths.append(str(path))
        print(f"  [shard {idx}] Saved {len(buf)} entries to {path}")

    def load_audio(entry_with_idx):
        """Load and resample audio in a worker thread (I/O-bound)."""
        idx, entry = entry_with_idx
        try:
            data, sr = sf.read(entry["audio"], dtype='float32')
            audio = torch.from_numpy(data)
            if audio.dim() > 1:
                audio = audio.mean(dim=-1)
            if sr != cfg.sample_rate:
                ratio = cfg.sample_rate / sr
                new_len = int(audio.shape[0] * ratio)
                audio = F.interpolate(
                    audio.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear',
                    align_corners=False,
                ).squeeze()
            if audio.shape[0] < cfg.sample_rate * 0.3:
                return None
            if audio.shape[0] > cfg.sample_rate * args.max_duration:
                audio = audio[:cfg.sample_rate * int(args.max_duration)]
            return (idx, entry, audio)
        except Exception as e:
            return (idx, entry, e)

    from concurrent.futures import ThreadPoolExecutor
    import queue

    prefetch_queue = queue.Queue(maxsize=64)
    io_done = [False]

    def io_producer():
        with ThreadPoolExecutor(max_workers=args.num_io_workers) as executor:
            batch = []
            for i, entry in enumerate(get_data_iterator(args)):
                batch.append((i, entry))
                if len(batch) >= 16:
                    futures = [executor.submit(load_audio, item) for item in batch]
                    for fut in futures:
                        result = fut.result()
                        if result is not None:
                            prefetch_queue.put(result)
                    batch = []
            if batch:
                futures = [executor.submit(load_audio, item) for item in batch]
                for fut in futures:
                    result = fut.result()
                    if result is not None:
                        prefetch_queue.put(result)
        io_done[0] = True

    import threading
    io_thread = threading.Thread(target=io_producer, daemon=True)
    io_thread.start()

    processed = 0
    while not (io_done[0] and prefetch_queue.empty()):
        try:
            item = prefetch_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        idx, entry, audio_or_error = item
        if isinstance(audio_or_error, Exception):
            n_errors += 1
            if n_errors <= 10:
                print(f"[encode] Error on {entry.get('audio', '?')}: {audio_or_error}")
            processed += 1
            continue

        audio = audio_or_error
        try:
            audio_device = audio.unsqueeze(0).to(device)
            semantic_tokens, acoustic_latent, _ = model.encode(audio_device)

            n_frames = semantic_tokens.shape[1]
            hop_length = cfg.sample_rate // cfg.frame_rate if hasattr(cfg, 'frame_rate') and cfg.frame_rate > 0 else 480
            prosody = extract_prosody(audio, cfg.sample_rate, n_frames, hop_length)

            shard_buf.append({
                "text": entry["text"],
                "utt_id": entry.get("utt_id", f"utt_{idx:06d}"),
                "speaker": entry.get("speaker", ""),
                "semantic_tokens": semantic_tokens[0].cpu(),
                "acoustic_latent": acoustic_latent[0].cpu(),
                "prosody_features": prosody,
                "n_frames": n_frames,
                "n_samples": audio_device.shape[1],
            })
            total_encoded += 1
            total_samples += audio_device.shape[1]
        except Exception as e:
            n_errors += 1
            if n_errors <= 10:
                print(f"[encode] Error on {entry.get('audio', '?')}: {e}")

        processed += 1
        if processed % 500 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed
            total_hours = total_samples / cfg.sample_rate / 3600
            print(f"  [{processed}] {total_encoded} encoded, {total_hours:.1f}h, "
                  f"{rate:.0f} utt/s, {n_errors} errors")

        if args.shard_size > 0 and len(shard_buf) >= args.shard_size:
            flush_shard(shard_buf, shard_idx)
            shard_buf = []
            shard_idx += 1

    io_thread.join()
    flush_shard(shard_buf, shard_idx)

    total_sec = total_samples / cfg.sample_rate if total_samples > 0 else 0
    elapsed = time.time() - t0

    print(f"\n[encode] Complete:")
    print(f"  Encoded: {total_encoded:,} utterances ({n_errors} errors)")
    print(f"  Audio:   {total_sec/3600:.1f} hours")
    print(f"  Shards:  {len(shard_paths)}")
    print(f"  Time:    {elapsed:.0f}s ({total_sec/max(elapsed,1):.1f}x realtime)")

    if len(shard_paths) > 1:
        index_path = out_base.with_suffix(".shards.txt")
        with open(index_path, "w") as f:
            for p in shard_paths:
                f.write(p + "\n")
        print(f"  Index:   {index_path}")
        print(f"\n  To train: --data {','.join(shard_paths)}")
    elif shard_paths:
        print(f"  Output:  {shard_paths[0]}")


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset encoding pipeline")
    parser.add_argument("--source", choices=["librispeech", "common_voice", "manifest"],
                        default="manifest")
    parser.add_argument("--data-dir", default=None, help="Root dir for dataset")
    parser.add_argument("--manifest", default=None, help="JSONL manifest path")
    parser.add_argument("--split", default="dev-clean", help="Comma-separated splits")
    parser.add_argument("--codec-ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--shard-size", type=int, default=10000,
                        help="Utterances per shard (0=single file)")
    parser.add_argument("--num-io-workers", type=int, default=8,
                        help="Number of I/O threads for parallel audio loading")
    args = parser.parse_args()
    encode_pipeline(args)


if __name__ == "__main__":
    main()
