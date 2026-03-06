"""Prepare SpeakStream-style interleaved text-speech training data.

Converts (text, audio) manifest pairs into interleaved chunk sequences where
text and speech alternate in fixed-size segments. This teaches the Flow v3
model to generate speech from partial text natively, enabling 30ms streaming
TTS latency.

Pipeline:
  1. Load manifest entry (text + audio path)
  2. Run forced alignment (MFA TextGrid or G2P-based uniform fallback)
  3. Segment into interleaved chunks of N words each
  4. Extract mel spectrogram for each speech chunk
  5. Output JSONL with text_chunks, speech_chunk_frames, alignment

Usage:
  python prepare_interleaved.py \\
    --manifest data/manifest_clean.jsonl \\
    --output data/interleaved_manifest.jsonl \\
    --alignment-dir data/alignments \\
    --chunk-words 6
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch

from config import FlowV3Config, InterleavedTrainingConfig
from mel_utils import mel_spectrogram as extract_mel_from_audio


def load_textgrid_alignment(textgrid_path: str) -> List[Dict]:
    """Parse MFA TextGrid to extract word-level timestamps.

    Returns list of {"word": str, "start": float, "end": float} dicts.
    """
    words = []
    in_words_tier = False
    in_interval = False
    current = {}

    with open(textgrid_path, "r") as f:
        for line in f:
            line = line.strip()
            if '"words"' in line.lower():
                in_words_tier = True
                continue
            if in_words_tier:
                if line.startswith("xmin"):
                    current["start"] = float(line.split("=")[1].strip())
                elif line.startswith("xmax"):
                    current["end"] = float(line.split("=")[1].strip())
                elif line.startswith("text"):
                    text = line.split("=")[1].strip().strip('"')
                    if text and text != "":
                        current["word"] = text
                        words.append(dict(current))
                    current = {}
                elif line.startswith("item"):
                    break  # Next tier
    return words


def uniform_alignment(text: str, duration_sec: float) -> List[Dict]:
    """Fallback: distribute words uniformly across audio duration.

    Used when MFA alignment is not available. Assumes constant speaking rate.
    """
    words = text.split()
    if not words:
        return []
    word_dur = duration_sec / len(words)
    result = []
    for i, word in enumerate(words):
        result.append({
            "word": word,
            "start": i * word_dur,
            "end": (i + 1) * word_dur,
        })
    return result


def segment_into_chunks(
    alignment: List[Dict],
    chunk_words: int = 6,
    overlap_words: int = 1,
    min_chunk_duration_ms: float = 200.0,
) -> List[Dict]:
    """Segment word-level alignment into interleaved text-speech chunks.

    Each chunk contains chunk_words consecutive words and their corresponding
    audio time span. Adjacent chunks overlap by overlap_words for smoother
    transitions at chunk boundaries.

    Returns list of chunks: {"text": str, "words": [...], "start": float, "end": float}
    """
    if not alignment:
        return []

    chunks = []
    step = max(1, chunk_words - overlap_words)
    min_dur_sec = min_chunk_duration_ms / 1000.0

    for i in range(0, len(alignment), step):
        chunk_words_list = alignment[i : i + chunk_words]
        if not chunk_words_list:
            break

        text = " ".join(w["word"] for w in chunk_words_list)
        start = chunk_words_list[0]["start"]
        end = chunk_words_list[-1]["end"]

        # Enforce minimum chunk duration
        if (end - start) < min_dur_sec and i + chunk_words < len(alignment):
            # Extend to meet minimum duration
            j = i + chunk_words
            while j < len(alignment) and (alignment[j]["end"] - start) < min_dur_sec:
                chunk_words_list.append(alignment[j])
                text = " ".join(w["word"] for w in chunk_words_list)
                end = alignment[j]["end"]
                j += 1

        chunks.append({
            "text": text,
            "words": [w["word"] for w in chunk_words_list],
            "start": start,
            "end": end,
        })

    # Don't produce an empty trailing chunk
    if chunks and not chunks[-1]["text"].strip():
        chunks = chunks[:-1]

    return chunks


def extract_chunk_mel(
    audio: torch.Tensor,
    start_sec: float,
    end_sec: float,
    cfg: FlowV3Config,
) -> torch.Tensor:
    """Extract mel spectrogram for a time segment of audio.

    Returns: (T_frames, mel_dim) tensor.
    """
    start_sample = int(start_sec * cfg.sample_rate)
    end_sample = int(end_sec * cfg.sample_rate)
    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(audio.shape[0], end_sample)

    if end_sample <= start_sample:
        # Return a single silent frame
        return torch.zeros(1, cfg.n_mels_extract)

    chunk_audio = audio[start_sample:end_sample]
    mel = extract_mel_from_audio(
        chunk_audio,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels_extract,
        sample_rate=cfg.sample_rate,
    )
    return mel


def process_entry(
    entry: Dict,
    cfg: FlowV3Config,
    interleaved_cfg: InterleavedTrainingConfig,
    alignment_dir: Optional[str] = None,
) -> Optional[Dict]:
    """Process a single manifest entry into interleaved chunks.

    Returns a dict with text_chunks, speech_chunk_frames, and alignment,
    or None if the entry can't be processed.
    """
    audio_path = entry.get("audio", "")
    text = entry.get("text", "")

    if not audio_path or not text:
        return None

    if not os.path.exists(audio_path):
        return None

    # Load audio
    data, sr = sf.read(audio_path, dtype="float32")
    audio = torch.from_numpy(data)
    if audio.dim() > 1:
        audio = audio.mean(dim=-1)

    duration_sec = audio.shape[0] / sr

    # Resample if needed
    if sr != cfg.sample_rate:
        import torch.nn.functional as F
        ratio = cfg.sample_rate / sr
        new_len = int(audio.shape[0] * ratio)
        audio = F.interpolate(
            audio.unsqueeze(0).unsqueeze(0), size=new_len, mode="linear",
            align_corners=False,
        ).squeeze()
        duration_sec = audio.shape[0] / cfg.sample_rate

    # Get alignment
    alignment = None

    # Try MFA TextGrid first
    if alignment_dir:
        stem = Path(audio_path).stem
        tg_path = os.path.join(alignment_dir, f"{stem}.TextGrid")
        if os.path.exists(tg_path):
            alignment = load_textgrid_alignment(tg_path)

    # Check for inline alignment in manifest
    if alignment is None and "word_alignments" in entry:
        alignment = entry["word_alignments"]

    # Fallback to uniform alignment
    if alignment is None:
        alignment = uniform_alignment(text, duration_sec)

    if not alignment:
        return None

    # Segment into chunks
    chunks = segment_into_chunks(
        alignment,
        chunk_words=interleaved_cfg.chunk_words,
        overlap_words=interleaved_cfg.overlap_words,
        min_chunk_duration_ms=interleaved_cfg.min_chunk_duration_ms,
    )

    if not chunks:
        return None

    # Extract mel for each chunk and record frame counts
    text_chunks = []
    speech_chunk_frames = []
    chunk_alignments = []

    for chunk in chunks:
        mel = extract_chunk_mel(audio, chunk["start"], chunk["end"], cfg)
        n_frames = mel.shape[0]

        if n_frames == 0:
            continue

        text_chunks.append(chunk["text"])
        speech_chunk_frames.append(n_frames)
        chunk_alignments.append({
            "start": round(chunk["start"], 4),
            "end": round(chunk["end"], 4),
            "n_words": len(chunk["words"]),
            "n_frames": n_frames,
        })

    if not text_chunks:
        return None

    return {
        "audio": audio_path,
        "text": text,
        "speaker": entry.get("speaker", ""),
        "duration": round(duration_sec, 4),
        "text_chunks": text_chunks,
        "speech_chunk_frames": speech_chunk_frames,
        "alignment": chunk_alignments,
        "n_chunks": len(text_chunks),
        "total_frames": sum(speech_chunk_frames),
    }


def prepare_interleaved_manifest(
    manifest_path: str,
    output_path: str,
    cfg: FlowV3Config,
    interleaved_cfg: InterleavedTrainingConfig,
    alignment_dir: Optional[str] = None,
    max_duration: float = 15.0,
) -> Dict:
    """Process entire manifest into interleaved training data.

    Returns stats dict with counts and diagnostics.
    """
    stats = {
        "total_entries": 0,
        "processed": 0,
        "skipped": 0,
        "total_chunks": 0,
        "total_frames": 0,
        "avg_chunks_per_utt": 0.0,
        "avg_frames_per_chunk": 0.0,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(manifest_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            stats["total_entries"] += 1

            if entry.get("duration", 999) > max_duration:
                stats["skipped"] += 1
                continue

            result = process_entry(entry, cfg, interleaved_cfg, alignment_dir)
            if result is None:
                stats["skipped"] += 1
                continue

            fout.write(json.dumps(result) + "\n")
            stats["processed"] += 1
            stats["total_chunks"] += result["n_chunks"]
            stats["total_frames"] += result["total_frames"]

    if stats["processed"] > 0:
        stats["avg_chunks_per_utt"] = round(
            stats["total_chunks"] / stats["processed"], 2
        )
    if stats["total_chunks"] > 0:
        stats["avg_frames_per_chunk"] = round(
            stats["total_frames"] / stats["total_chunks"], 1
        )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SpeakStream-style interleaved training data"
    )
    parser.add_argument("--manifest", required=True, help="Input manifest JSONL")
    parser.add_argument("--output", required=True, help="Output interleaved JSONL")
    parser.add_argument(
        "--alignment-dir", default="",
        help="Directory with MFA TextGrid files (optional)",
    )
    parser.add_argument("--chunk-words", type=int, default=6)
    parser.add_argument("--overlap-words", type=int, default=1)
    parser.add_argument("--min-chunk-duration-ms", type=float, default=200.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    args = parser.parse_args()

    cfg = FlowV3Config(sample_rate=args.sample_rate)
    interleaved_cfg = InterleavedTrainingConfig(
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        min_chunk_duration_ms=args.min_chunk_duration_ms,
    )

    print(f"\n{'='*60}")
    print(f"  SONATA — INTERLEAVED DATA PREPARATION")
    print(f"{'='*60}")
    print(f"  Input:  {args.manifest}")
    print(f"  Output: {args.output}")
    print(f"  Chunk words: {interleaved_cfg.chunk_words}")
    print(f"  Overlap words: {interleaved_cfg.overlap_words}")
    print(f"  Min chunk duration: {interleaved_cfg.min_chunk_duration_ms}ms")
    if args.alignment_dir:
        print(f"  Alignment dir: {args.alignment_dir}")
    else:
        print(f"  Alignment: uniform fallback (no MFA TextGrids)")

    stats = prepare_interleaved_manifest(
        args.manifest,
        args.output,
        cfg,
        interleaved_cfg,
        alignment_dir=args.alignment_dir or None,
        max_duration=args.max_duration,
    )

    print(f"\n  Results:")
    print(f"    Total entries:     {stats['total_entries']}")
    print(f"    Processed:         {stats['processed']}")
    print(f"    Skipped:           {stats['skipped']}")
    print(f"    Total chunks:      {stats['total_chunks']}")
    print(f"    Total frames:      {stats['total_frames']}")
    print(f"    Avg chunks/utt:    {stats['avg_chunks_per_utt']}")
    print(f"    Avg frames/chunk:  {stats['avg_frames_per_chunk']}")
    print(f"\n  Output: {args.output}")


if __name__ == "__main__":
    main()
