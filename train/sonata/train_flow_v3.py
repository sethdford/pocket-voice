"""Train Sonata Flow v3: interleaved streaming text → mel via causal flow matching.

Upgrades over v2:
  - Causal sliding-window attention (enables streaming inference)
  - Interleaved text-speech training (SpeakStream-style learned alignment)
  - Reference audio prompting (zero-shot voice cloning)
  - Mixed training: alternates standard + interleaved loss

Usage:
  # From prepared .pt files (same data format as v2):
  python train_flow_v3.py --data-dir data/flow_v2_ljspeech --device mps --steps 50000

  # From manifest JSONL:
  python train_flow_v3.py --manifest data/manifest_clean.jsonl --device mps --steps 200000

  # Resume from v2 checkpoint (transfer learning):
  python train_flow_v3.py --data-dir data/flow_v2_ljspeech --resume checkpoints/flow_v2_best.pt --device mps
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from flow_v3 import SonataFlowV3, FlowV3Config
from config import FlowV3LargeConfig, InterleavedTrainingConfig
from modules import TrainingLog

try:
    from perceptual_loss import PerceptualMelLoss
    HAS_PERCEPTUAL = True
except ImportError:
    HAS_PERCEPTUAL = False

try:
    from g2p import PhonemeFrontend
    HAS_G2P = True
except ImportError:
    HAS_G2P = False

try:
    from ema import EMA
    HAS_EMA = True
except ImportError:
    HAS_EMA = False


def count_manifest_speakers(manifest_path: str, max_duration: float = 15.0) -> int:
    """Count unique speakers in manifest (same logic as ManifestDataset)."""
    speakers = set()
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("duration", 999) <= max_duration:
                speakers.add(entry.get("speaker", ""))
    return len(speakers)


def cosine_lr(step, warmup, max_lr, min_lr, total):
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    ratio = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))


from mel_utils import mel_spectrogram as extract_mel_from_audio


class PtDataset(Dataset):
    def __init__(self, data_dir: str, max_frames: int = 800, max_text_len: int = 512):
        self.files = sorted(Path(data_dir).glob("*.pt"))
        self.files = [f for f in self.files if f.name != "meta.pt"]
        self.max_frames = max_frames
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        text = data.get("text", "")
        mel = data["mel"]
        speaker_id = data.get("speaker_id", 0)
        T = min(mel.shape[0], self.max_frames)
        mel = mel[:T]
        char_ids = torch.tensor([ord(c) % 256 for c in text[:self.max_text_len]], dtype=torch.long)
        return char_ids, mel, speaker_id, None  # ref_mel always None for .pt data


class ManifestDataset(Dataset):
    def __init__(self, manifest: str, cfg: FlowV3Config, max_frames: int = 800,
                 g2p=None):
        self.cfg = cfg
        self.max_frames = max_frames
        self.max_ref_frames = getattr(cfg, "max_ref_frames", 200)
        self.g2p = g2p
        self.entries = []
        speakers_seen = {}
        with open(manifest) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("duration", 999) <= 15.0:
                    spk = entry.get("speaker", "")
                    if spk not in speakers_seen:
                        speakers_seen[spk] = len(speakers_seen)
                    self.entries.append(entry)
        self.speaker_map = speakers_seen
        self.speaker_to_indices = {}
        for i, e in enumerate(self.entries):
            spk = e.get("speaker", "")
            self.speaker_to_indices.setdefault(spk, []).append(i)
        mode = "phonemes" if g2p else "characters"
        print(f"  Manifest: {len(self.entries)} utterances, {len(speakers_seen)} speakers ({mode})")

    def __len__(self):
        return len(self.entries)

    def _load_ref_mel(self, ref_idx: int) -> Optional[torch.Tensor]:
        """Load and extract mel from reference audio (max max_ref_frames)."""
        ref_entry = self.entries[ref_idx]
        data, sr = sf.read(ref_entry["audio"], dtype='float32')
        audio = torch.from_numpy(data)
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)
        if sr != self.cfg.sample_rate:
            ratio = self.cfg.sample_rate / sr
            new_len = int(audio.shape[0] * ratio)
            audio = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear',
                align_corners=False
            ).squeeze()
        max_ref_samples = self.max_ref_frames * self.cfg.hop_length
        if audio.shape[0] > max_ref_samples:
            start = torch.randint(0, audio.shape[0] - max_ref_samples, (1,)).item()
            audio = audio[start:start + max_ref_samples]
        ref_mel = extract_mel_from_audio(
            audio,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels_extract,
            sample_rate=self.cfg.sample_rate,
        )
        T = min(ref_mel.shape[0], self.max_ref_frames)
        return ref_mel[:T]

    def __getitem__(self, idx):
        entry = self.entries[idx]
        data, sr = sf.read(entry["audio"], dtype='float32')
        audio = torch.from_numpy(data)
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)
        if sr != self.cfg.sample_rate:
            ratio = self.cfg.sample_rate / sr
            new_len = int(audio.shape[0] * ratio)
            audio = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear',
                align_corners=False
            ).squeeze()

        if self.cfg.speed_perturb and torch.rand(1).item() < 0.5:
            speed = random.choice([0.9, 1.0, 1.1])
            if speed != 1.0:
                new_len = int(audio.shape[0] / speed)
                audio = F.interpolate(
                    audio.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear',
                    align_corners=False
                ).squeeze()

        max_samples = self.max_frames * self.cfg.hop_length
        if audio.shape[0] > max_samples:
            start = torch.randint(0, audio.shape[0] - max_samples, (1,)).item()
            audio = audio[start:start + max_samples]

        mel = extract_mel_from_audio(
            audio,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels_extract,
            sample_rate=self.cfg.sample_rate,
        )

        if self.cfg.spec_augment and torch.rand(1).item() < 0.5:
            mel = self._spec_augment(mel)

        text = entry.get("text", "")
        precomputed = entry.get("phoneme_ids")
        if precomputed is not None:
            char_ids = torch.tensor(precomputed, dtype=torch.long)
        elif self.g2p is not None:
            char_ids = self.g2p.encode(text, add_bos=True, add_eos=True)
        else:
            char_ids = torch.tensor(
                [ord(c) % self.cfg.char_vocab_size for c in text],
                dtype=torch.long
            )
        speaker_id = self.speaker_map.get(entry.get("speaker", ""), 0)
        n_spk = getattr(self.cfg, "n_speakers", 0)
        if n_spk > 0 and speaker_id >= n_spk:
            raise ValueError(
                f"Speaker ID {speaker_id} out of bounds (n_speakers={n_spk}). "
                "Increase --n-speakers to cover all speakers in the manifest."
            )

        ref_mel = None
        if torch.rand(1).item() < 0.5:
            spk = entry.get("speaker", "")
            indices = self.speaker_to_indices.get(spk, [])
            other_indices = [i for i in indices if i != idx]
            if other_indices:
                ref_idx = random.choice(other_indices)
                ref_mel = self._load_ref_mel(ref_idx)

        return char_ids, mel, speaker_id, ref_mel

    @staticmethod
    def _spec_augment(mel, n_freq_masks=2, freq_width=15, n_time_masks=2, time_width=50):
        """SpecAugment: frequency and time masking on mel spectrogram."""
        T, D = mel.shape
        for _ in range(n_freq_masks):
            f = random.randint(0, min(freq_width, D - 1))
            f0 = random.randint(0, D - f)
            mel[..., f0:f0 + f] = 0
        for _ in range(n_time_masks):
            t = random.randint(0, min(time_width, T - 1))
            t0 = random.randint(0, T - t)
            mel[t0:t0 + t, :] = 0
        return mel


class InterleavedManifestDataset(Dataset):
    """Dataset that loads pre-chunked interleaved manifest entries.

    Each entry has text_chunks and speech_chunk_frames from prepare_interleaved.py.
    Returns the full mel + per-chunk boundaries for interleaved training loss.
    """

    def __init__(self, manifest: str, cfg: FlowV3Config,
                 interleaved_cfg: InterleavedTrainingConfig,
                 max_frames: int = 800, g2p=None):
        self.cfg = cfg
        self.interleaved_cfg = interleaved_cfg
        self.max_frames = max_frames
        self.g2p = g2p
        self.entries = []
        speakers_seen = {}
        with open(manifest) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("n_chunks", 0) == 0:
                    continue
                if entry.get("total_frames", 999999) > max_frames:
                    continue
                spk = entry.get("speaker", "")
                if spk not in speakers_seen:
                    speakers_seen[spk] = len(speakers_seen)
                self.entries.append(entry)
        self.speaker_map = speakers_seen
        print(f"  Interleaved manifest: {len(self.entries)} utterances, "
              f"{len(speakers_seen)} speakers")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Load full audio and extract mel
        data, sr = sf.read(entry["audio"], dtype="float32")
        audio = torch.from_numpy(data)
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)
        if sr != self.cfg.sample_rate:
            ratio = self.cfg.sample_rate / sr
            new_len = int(audio.shape[0] * ratio)
            audio = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0), size=new_len, mode="linear",
                align_corners=False,
            ).squeeze()

        mel = extract_mel_from_audio(
            audio,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels_extract,
            sample_rate=self.cfg.sample_rate,
        )

        # Build per-chunk text token sequences and frame boundaries
        text_chunks = entry["text_chunks"]
        alignment = entry["alignment"]
        chunk_char_ids = []
        chunk_frame_starts = []
        chunk_frame_ends = []

        for i, (text_chunk, align) in enumerate(zip(text_chunks, alignment)):
            if i >= self.interleaved_cfg.max_chunks_per_utterance:
                break
            # Encode text chunk
            if self.g2p is not None:
                ids = self.g2p.encode(text_chunk, add_bos=True, add_eos=True)
            else:
                ids = torch.tensor(
                    [ord(c) % self.cfg.char_vocab_size for c in text_chunk],
                    dtype=torch.long,
                )

            # Frame boundaries from alignment timestamps
            start_frame = int(align["start"] * self.cfg.frame_rate)
            end_frame = int(align["end"] * self.cfg.frame_rate)
            end_frame = min(end_frame, mel.shape[0])

            if end_frame <= start_frame:
                continue

            chunk_char_ids.append(ids)
            chunk_frame_starts.append(start_frame)
            chunk_frame_ends.append(end_frame)

        if not chunk_char_ids:
            # Fallback: return full utterance as single chunk
            full_text = entry.get("text", "")
            if self.g2p is not None:
                ids = self.g2p.encode(full_text, add_bos=True, add_eos=True)
            else:
                ids = torch.tensor(
                    [ord(c) % self.cfg.char_vocab_size for c in full_text],
                    dtype=torch.long,
                )
            chunk_char_ids = [ids]
            chunk_frame_starts = [0]
            chunk_frame_ends = [mel.shape[0]]

        speaker_id = self.speaker_map.get(entry.get("speaker", ""), 0)

        return (mel, chunk_char_ids, chunk_frame_starts, chunk_frame_ends,
                speaker_id, len(chunk_char_ids))


def interleaved_collate_fn(batch):
    """Collate interleaved dataset: pads mel and packs chunk boundaries.

    Returns: (mel, mel_mask, speakers, chunk_char_ids_list, chunk_bounds_list, n_chunks_list)
    where chunk_bounds_list[b] = [(start, end), ...] for batch element b.
    """
    mel_list = [item[0] for item in batch]
    chunk_chars_list = [item[1] for item in batch]
    chunk_starts_list = [item[2] for item in batch]
    chunk_ends_list = [item[3] for item in batch]
    spk_list = [item[4] for item in batch]
    n_chunks_list = [item[5] for item in batch]

    B = len(batch)
    max_frames = max(m.shape[0] for m in mel_list)
    mel_dim = mel_list[0].shape[1]

    mel = torch.zeros(B, max_frames, mel_dim)
    mel_mask = torch.zeros(B, max_frames)
    speakers = torch.tensor(spk_list, dtype=torch.long)

    for i, m in enumerate(mel_list):
        mel[i, :m.shape[0]] = m
        mel_mask[i, :m.shape[0]] = 1.0

    # Pack chunk boundaries as list-of-lists (variable per batch element)
    chunk_bounds = []
    for starts, ends in zip(chunk_starts_list, chunk_ends_list):
        chunk_bounds.append(list(zip(starts, ends)))

    return mel, mel_mask, speakers, chunk_chars_list, chunk_bounds, n_chunks_list


def collate_fn(batch):
    char_list, mel_list, spk_list, ref_list = zip(*batch)
    max_chars = max(c.shape[0] for c in char_list)
    max_frames = max(m.shape[0] for m in mel_list)
    mel_dim = mel_list[0].shape[1]
    B = len(batch)

    chars = torch.zeros(B, max_chars, dtype=torch.long)
    mel = torch.zeros(B, max_frames, mel_dim)
    mel_mask = torch.zeros(B, max_frames)
    speakers = torch.tensor(spk_list, dtype=torch.long)

    for i, (c, m, _, _) in enumerate(zip(char_list, mel_list, spk_list, ref_list)):
        chars[i, :c.shape[0]] = c
        mel[i, :m.shape[0]] = m
        mel_mask[i, :m.shape[0]] = 1.0

    ref_mel = None
    ref_mel_mask = None
    refs_with_data = [r for r in ref_list if r is not None]
    if refs_with_data:
        max_ref = max(r.shape[0] for r in refs_with_data)
        ref_mel = torch.zeros(B, max_ref, mel_dim)
        ref_mel_mask = torch.zeros(B, max_ref)
        for i, r in enumerate(ref_list):
            if r is not None:
                ref_mel[i, :r.shape[0]] = r
                ref_mel_mask[i, :r.shape[0]] = 1.0

    return chars, mel, mel_mask, speakers, ref_mel, ref_mel_mask


@torch.no_grad()
def validate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            if n >= max_batches:
                break
            chars, mel, mel_mask, speakers, ref_mel, ref_mel_mask = batch
            chars, mel, mel_mask = chars.to(device), mel.to(device), mel_mask.to(device)
            ref_mel = ref_mel.to(device) if ref_mel is not None else None
            speakers = speakers.to(device) if model.cfg.n_speakers > 0 else None
            loss = model.compute_loss(mel, chars, speaker_ids=speakers,
                                      mel_mask=mel_mask, ref_mel=ref_mel)
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def compute_chunked_interleaved_loss(
    model, mel, mel_mask, speakers, chunk_chars_list, chunk_bounds, device,
):
    """SpeakStream-style chunk-wise interleaved training with KV-cache.

    For each utterance, processes chunks sequentially: text chunk conditions the
    model, speech chunk gets flow-matching loss. KV-cache from previous chunks
    provides autoregressive context, matching SpeakStream's training objective.

    Loss is computed only on speech tokens (text tokens are conditioning).
    BOS/EOS boundaries around each speech chunk are implicit in the chunk structure.
    """
    B = mel.shape[0]
    total_loss = 0.0
    n_chunks_total = 0

    for b in range(B):
        bounds = chunk_bounds[b]
        chars_list = chunk_chars_list[b]
        kv_caches = None
        offset = 0

        for chunk_idx, ((start_frame, end_frame), char_ids) in enumerate(
            zip(bounds, chars_list)
        ):
            chunk_mel = mel[b:b+1, start_frame:end_frame]  # (1, T_chunk, mel_dim)
            T_chunk = chunk_mel.shape[1]
            if T_chunk == 0:
                continue

            chunk_mask = mel_mask[b:b+1, start_frame:end_frame]

            # Text conditioning for this chunk
            char_ids_d = char_ids.unsqueeze(0).to(device)
            text_enc = model.interleaved_enc.encode_text(char_ids_d)
            text_cond = model._align_text_to_mel(text_enc, T_chunk)

            # Flow matching: sample timestep and interpolate
            z = torch.randn(1, device=device) + model.cfg.sway_coefficient
            t = torch.sigmoid(z).clamp(1e-5, 1.0 - 1e-5)

            noise = torch.randn_like(chunk_mel)
            t_expand = t[:, None, None]
            x_t = (1 - t_expand) * noise + t_expand * chunk_mel
            target_velocity = chunk_mel - noise

            # Forward with KV-cache from previous chunks
            spk = speakers[b:b+1] if speakers is not None else None
            predicted, new_kv = model.forward(
                x_t, t, text_cond, spk,
                kv_caches=kv_caches, offset=offset,
            )

            # Masked MSE loss on speech tokens only
            mask = chunk_mask.unsqueeze(-1)
            per_elem = F.mse_loss(predicted, target_velocity, reduction="none") * mask
            n_valid = mask.sum() * predicted.shape[-1]
            chunk_loss = per_elem.sum() / n_valid.clamp(min=1)

            total_loss = total_loss + chunk_loss
            n_chunks_total += 1

            # Update KV-cache for next chunk (teacher forcing: use ground truth mel)
            # Re-encode the ground truth chunk to build cache for the next chunk
            with torch.no_grad():
                gt_enc = model.input_proj(chunk_mel) + text_cond[:, :T_chunk]
                # Build cache by running through blocks
                cache_kv = []
                x_cache = gt_enc
                for i, block in enumerate(model.blocks):
                    layer_cache = kv_caches[i] if kv_caches is not None else None
                    x_cache, new_cache = block(
                        x_cache, torch.zeros(1, model.cfg.cond_dim, device=device),
                        kv_cache=layer_cache, offset=offset,
                    )
                    cache_kv.append(new_cache)
                kv_caches = cache_kv
                offset += T_chunk

    if n_chunks_total > 0:
        total_loss = total_loss / n_chunks_total

    return total_loss


def train(args):
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  SONATA FLOW v3 — INTERLEAVED STREAMING TRAINING")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    g2p = None
    if args.phonemes:
        if not HAS_G2P:
            print("  ERROR: --phonemes requires g2p.py and phonemizer package")
            return
        g2p = PhonemeFrontend()
        print(f"  Phoneme mode: ON ({g2p.vocab_size} tokens)")

    if args.model_size == "large":
        cfg = FlowV3LargeConfig(n_speakers=args.n_speakers)
    else:
        cfg = FlowV3Config(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_speakers=args.n_speakers,
            window_size=args.window_size,
            chunk_size=args.chunk_size,
        )

    if g2p is not None:
        cfg.char_vocab_size = g2p.vocab_size

    # Validate speaker count when using manifest: n_speakers must cover all speakers
    if args.manifest:
        n_speakers_from_data = count_manifest_speakers(args.manifest)
        if n_speakers_from_data > 0:
            if args.n_speakers == 0:
                args.n_speakers = n_speakers_from_data
                cfg.n_speakers = args.n_speakers
                print(f"  Auto-set n_speakers={args.n_speakers} from manifest (was 0)")
            elif n_speakers_from_data > args.n_speakers:
                raise ValueError(
                    f"Manifest has {n_speakers_from_data} speakers but --n-speakers={args.n_speakers}. "
                    f"Increase --n-speakers or use --n-speakers=0 for auto-detect."
                )

    model = SonataFlowV3(cfg, cfg_dropout_prob=args.cfg_dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.1f}M params")
    print(f"  {cfg.n_layers}L × d={cfg.d_model}, H={cfg.n_heads}")
    print(f"  Window: {cfg.window_size}, Chunk: {cfg.chunk_size}")
    print(f"  Output: {cfg.mel_dim}-dim mel spectrogram")
    print(f"  Vocab: {cfg.char_vocab_size} ({'phonemes' if g2p else 'characters'})")
    print(f"  Interleaved training: {'ON' if args.interleaved else 'OFF'}")
    print(f"  Interleave ratio: {args.interleave_ratio:.0%}")

    # Interleaved training config
    interleaved_cfg = InterleavedTrainingConfig(
        chunk_words=args.chunk_words_interleaved,
    )
    if args.interleaved_manifest:
        cfg.interleaved_training = True

    if args.data_dir:
        dataset = PtDataset(args.data_dir, max_frames=args.max_frames)
        print(f"  Dataset: {len(dataset)} utterances from .pt files")
    elif args.manifest:
        dataset = ManifestDataset(args.manifest, cfg, max_frames=args.max_frames, g2p=g2p)
        print(f"  Dataset: {len(dataset)} utterances from manifest")
    else:
        print("  ERROR: Provide --data-dir or --manifest")
        return

    if len(dataset) == 0:
        print("  ERROR: No data found.")
        return

    # Interleaved dataset (separate loader, mixed with standard during training)
    interleaved_loader = None
    if args.interleaved_manifest:
        intlv_dataset = InterleavedManifestDataset(
            args.interleaved_manifest, cfg, interleaved_cfg,
            max_frames=args.max_frames, g2p=g2p,
        )
        if len(intlv_dataset) > 0:
            interleaved_loader = DataLoader(
                intlv_dataset, batch_size=args.batch_size, shuffle=True,
                collate_fn=interleaved_collate_fn,
                num_workers=args.num_workers, drop_last=True,
            )
            print(f"  Interleaved dataset: {len(intlv_dataset)} utterances")

    val_size = min(max(int(len(dataset) * 0.02), 50), 500)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )

    ema = EMA(model, decay=0.999) if HAS_EMA else None

    perceptual_fn = None
    if HAS_PERCEPTUAL and args.perceptual_weight > 0:
        perceptual_fn = PerceptualMelLoss(n_mels=cfg.mel_dim).to(device)
        print(f"  Perceptual loss: ON (weight={args.perceptual_weight})")

    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tlog = TrainingLog(str(ckpt_dir / "losses.jsonl"))

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        model_state = model.state_dict()
        filtered = {}
        skipped_shape = []
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v
            elif k in model_state:
                skipped_shape.append(k)
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        n_loaded = len(filtered)
        n_new = len(missing)
        if skipped_shape:
            print(f"  Transfer: {n_loaded} loaded, {n_new} new, {len(skipped_shape)} shape-mismatched:")
            for k in skipped_shape[:5]:
                print(f"    {k}: ckpt={state[k].shape} → model={model_state[k].shape}")
        elif missing:
            print(f"  Transfer learning: {n_new} new params, {len(unexpected)} skipped")
        else:
            print(f"  Full checkpoint loaded ({n_loaded} params)")
        if not skipped_shape:
            try:
                optimizer.load_state_dict(ckpt.get("optimizer", {}))
            except (ValueError, KeyError, TypeError):
                print(f"  WARNING: Could not load optimizer state")
        start_step = 0 if skipped_shape else ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Starting from step {start_step}")

    step = start_step
    running_loss = 0.0
    running_loss_interleaved = 0.0
    n_interleaved = 0
    t0 = time.time()

    print(f"\n  Training: step {start_step} → {args.steps}, batch={args.batch_size}")
    print(f"  Train/val split: {train_size}/{val_size}")

    model.train()
    while step < args.steps:
        for chars, mel, mel_mask, speakers, ref_mel, ref_mel_mask in loader:
            if step >= args.steps:
                break

            chars = chars.to(device)
            mel = mel.to(device)
            speakers = speakers.to(device) if cfg.n_speakers > 0 else None
            ref_mel = ref_mel.to(device) if ref_mel is not None else None

            lr = cosine_lr(step, args.warmup, args.lr, args.lr * 0.01, args.steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            dur_ramp = 5000
            model._dur_loss_weight = min(0.1, 0.1 * step / max(1, dur_ramp))

            use_interleaved = (args.interleaved and
                               torch.rand(1).item() < args.interleave_ratio)
            use_mas = step >= args.mas_start

            mel_mask_d = mel_mask.to(device)
            use_percep = perceptual_fn is not None and step >= 1000
            if use_interleaved:
                loss = model.compute_interleaved_loss(
                    mel, chars, speaker_ids=speakers, ref_mel=ref_mel,
                    mel_mask=mel_mask_d)
                running_loss_interleaved += loss.item()
                n_interleaved += 1
            elif use_percep:
                loss, mel_denoised = model.compute_loss(
                    mel, chars, speaker_ids=speakers, ref_mel=ref_mel,
                    mel_mask=mel_mask_d, use_mas=use_mas, return_denoised=True)
                p_loss = perceptual_fn(mel_denoised, mel, mel_mask=mel_mask_d)
                loss = loss + args.perceptual_weight * p_loss
            else:
                loss = model.compute_loss(mel, chars, speaker_ids=speakers,
                                          ref_mel=ref_mel,
                                          mel_mask=mel_mask_d, use_mas=use_mas)

            if torch.isnan(loss):
                print(f"  [WARN] step {step}: NaN loss, skipping")
                optimizer.zero_grad()
                continue

            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()

            accum_step = step - start_step
            if (accum_step + 1) % args.grad_accum == 0 or step >= args.steps - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if ema:
                    ema.update()

            running_loss += loss.item()
            step += 1

            if step % args.log_every == 0:
                n = args.log_every
                elapsed = time.time() - t0
                avg = running_loss / n
                avg_i = running_loss_interleaved / max(1, n_interleaved) if args.interleaved else 0
                parts = f"  step {step:6d} | loss={avg:.4f}"
                if args.interleaved:
                    parts += f" intlv={avg_i:.4f}"
                parts += f" | lr={lr:.2e} | {n/elapsed:.1f} steps/s"
                print(parts)
                tlog.log(step=step, loss=avg, intlv=avg_i, lr=lr,
                         steps_per_sec=n / elapsed)
                running_loss = 0.0
                running_loss_interleaved = 0.0
                n_interleaved = 0
                t0 = time.time()

            if step % args.save_every == 0:
                save_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
                    "best_val_loss": best_val_loss,
                }
                if ema:
                    save_dict["ema"] = ema.state_dict()
                path = ckpt_dir / f"flow_v3_step_{step}.pt"
                torch.save(save_dict, path)
                print(f"  [ckpt] {path}")

            if step % args.val_every == 0 and len(val_ds) > 0:
                if ema:
                    ema.apply_shadow()
                val_loss = validate(model, val_loader, device)
                if ema:
                    ema.restore()
                is_best = val_loss < best_val_loss
                tag = " (best!)" if is_best else ""
                print(f"  [val] step {step}: loss={val_loss:.4f}{tag}")
                tlog.log(step=step, val_loss=val_loss, is_best=is_best)
                if is_best:
                    best_val_loss = val_loss
                    save_dict = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
                        "best_val_loss": best_val_loss,
                    }
                    if ema:
                        save_dict["ema"] = ema.state_dict()
                    best_path = ckpt_dir / "flow_v3_best.pt"
                    torch.save(save_dict, best_path)
                    print(f"  [best] {best_path}")

    # Final save
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
        "best_val_loss": best_val_loss,
    }
    if ema:
        save_dict["ema"] = ema.state_dict()
    path = ckpt_dir / "flow_v3_final.pt"
    torch.save(save_dict, path)
    print(f"\n  Final: {path}")
    print(f"  Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sonata Flow v3")
    parser.add_argument("--data-dir", default="", help="Directory with .pt files")
    parser.add_argument("--manifest", default="", help="Manifest JSONL")
    parser.add_argument("--output-dir", default="train/checkpoints/flow_v3")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--max-frames", type=int, default=800)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-speakers", type=int, default=0)
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--interleaved", action="store_true",
                        help="Enable interleaved text-speech training")
    parser.add_argument("--interleave-ratio", type=float, default=0.3,
                        help="Fraction of batches using interleaved loss")
    parser.add_argument("--mas-start", type=int, default=5000,
                        help="Step at which to switch from uniform to MAS alignment")
    parser.add_argument("--perceptual-weight", type=float, default=0.1,
                        help="Weight for perceptual mel loss (0=disabled)")
    parser.add_argument("--model-size", default="base", choices=["base", "large"],
                        help="Model size: base (55M) or large (150M)")
    parser.add_argument("--phonemes", action="store_true",
                        help="Use phoneme conditioning via espeak-ng G2P (recommended)")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--chunk-words-interleaved", type=int, default=6,
                        help="Words per chunk for interleaved training (4-8 recommended)")
    parser.add_argument("--interleaved-manifest", default="",
                        help="Separate manifest for interleaved training data")
    parser.add_argument("--resume", default="", help="Resume from checkpoint (v2 or v3)")
    args = parser.parse_args()
    train(args)
