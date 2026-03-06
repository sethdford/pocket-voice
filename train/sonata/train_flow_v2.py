"""Train Sonata Flow v2: single-stage text → mel via conditional flow matching.

F5-TTS-inspired training with masked mel infilling and OT-CFM loss.
Supports both .pt data format (from prepare_flow_v2_data.py) and manifest JSONL.

Usage:
  # From prepared .pt files:
  python train_flow_v2.py --data-dir data/flow_v2_ljspeech --device mps --steps 50000

  # From manifest JSONL:
  python train_flow_v2.py --manifest data/manifest_clean.jsonl --device mps --steps 200000
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import FlowV2Config
from flow_v2 import SonataFlowV2

try:
    from ema import EMA
    HAS_EMA = True
except ImportError:
    HAS_EMA = False


def cosine_lr(step, warmup, max_lr, min_lr, total):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    ratio = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))


def extract_mel_from_audio(audio: torch.Tensor, n_fft=1024, hop_length=480,
                           n_mels=80, sample_rate=24000) -> torch.Tensor:
    """Extract log-mel spectrogram from raw audio."""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    window = torch.hann_window(n_fft, periodic=True, device=audio.device)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    mag = stft.abs().pow(2)

    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0, sample_rate / 2, n_freqs)
    mel_low = 2595 * math.log10(1 + 0 / 700)
    mel_high = 2595 * math.log10(1 + sample_rate / 2 / 700)
    mels = torch.linspace(mel_low, mel_high, n_mels + 2)
    hz = 700 * (10 ** (mels / 2595) - 1)
    fb = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        low, center, high = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - low) / (center - low + 1e-8)
        down = (high - freqs) / (high - center + 1e-8)
        fb[i] = torch.clamp(torch.min(up, down), min=0)
    fb = fb.to(audio.device)
    mel = torch.matmul(fb, mag.squeeze(0))
    mel = torch.log(mel.clamp(min=1e-5))
    return mel.T  # (T, n_mels)


class PtDataset(Dataset):
    """Loads (text, mel, speaker_id) from prepared .pt files."""

    def __init__(self, data_dir: str, max_frames: int = 800, max_text_len: int = 512):
        self.files = sorted(Path(data_dir).glob("*.pt"))
        self.files = [f for f in self.files if f.name != "meta.pt"]
        self.max_frames = max_frames
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = torch.load(self.files[idx], weights_only=True)
        text = data.get("text", "")
        mel = data["mel"]
        speaker_id = data.get("speaker_id", 0)

        T = min(mel.shape[0], self.max_frames)
        mel = mel[:T]

        char_ids = torch.tensor([ord(c) % 256 for c in text[:self.max_text_len]],
                                dtype=torch.long)
        return char_ids, mel, speaker_id


class ManifestDataset(Dataset):
    """Loads (text, mel) from manifest JSONL (extracts mel on-the-fly)."""

    def __init__(self, manifest: str, cfg: FlowV2Config, max_frames: int = 800):
        self.cfg = cfg
        self.max_frames = max_frames
        self.entries = []
        with open(manifest) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("duration", 999) <= 15.0:
                    self.entries.append(entry)
        print(f"  Manifest: {len(self.entries)} utterances")

    def __len__(self):
        return len(self.entries)

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

        max_samples = self.max_frames * self.cfg.hop_length
        if audio.shape[0] > max_samples:
            start = torch.randint(0, audio.shape[0] - max_samples, (1,)).item()
            audio = audio[start:start + max_samples]

        mel = extract_mel_from_audio(
            audio, self.cfg.n_fft, self.cfg.hop_length,
            self.cfg.n_mels_extract, self.cfg.sample_rate
        )
        char_ids = torch.tensor(
            [ord(c) % self.cfg.char_vocab_size for c in entry.get("text", "")],
            dtype=torch.long
        )
        speaker_id = hash(entry.get("speaker", "")) % 10000
        return char_ids, mel, speaker_id


def collate_fn(batch):
    char_list, mel_list, spk_list = zip(*batch)

    max_chars = max(c.shape[0] for c in char_list)
    max_frames = max(m.shape[0] for m in mel_list)
    mel_dim = mel_list[0].shape[1]
    B = len(batch)

    chars = torch.zeros(B, max_chars, dtype=torch.long)
    mel = torch.zeros(B, max_frames, mel_dim)
    mel_mask = torch.zeros(B, max_frames)
    speakers = torch.tensor(spk_list, dtype=torch.long)

    for i, (c, m, _) in enumerate(zip(char_list, mel_list, spk_list)):
        chars[i, :c.shape[0]] = c
        mel[i, :m.shape[0]] = m
        mel_mask[i, :m.shape[0]] = 1.0

    return chars, mel, mel_mask, speakers


def generate_mask(mel_mask: torch.Tensor, min_mask_ratio: float = 0.7,
                  max_mask_ratio: float = 1.0) -> torch.Tensor:
    """F5-style random masks: 1 = masked (predict), 0 = reference (keep)."""
    B, T = mel_mask.shape
    masks = torch.zeros_like(mel_mask)

    for i in range(B):
        valid_len = int(mel_mask[i].sum().item())
        if valid_len == 0:
            continue
        ratio = torch.rand(1).item() * (max_mask_ratio - min_mask_ratio) + min_mask_ratio
        n_mask = max(1, int(valid_len * ratio))
        start = torch.randint(0, max(1, valid_len - n_mask + 1), (1,)).item()
        masks[i, start:start + n_mask] = 1.0

    return masks * mel_mask


@torch.no_grad()
def validate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    n = 0
    for chars, mel, mel_mask, speakers in val_loader:
        if n >= max_batches:
            break
        chars, mel = chars.to(device), mel.to(device)
        loss = model.compute_loss(mel, chars)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def train(args):
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  SONATA FLOW v2 — SINGLE-STAGE TRAINING")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    cfg = FlowV2Config(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_speakers=args.n_speakers,
    )

    model = SonataFlowV2(cfg, cfg_dropout_prob=args.cfg_dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.1f}M params")
    print(f"  {cfg.n_layers}L × d={cfg.d_model}, H={cfg.n_heads}")
    print(f"  Text encoder: {cfg.text_encoder_layers}L ConvNeXt")
    print(f"  Output: {cfg.mel_dim}-dim mel spectrogram")
    print(f"  Speakers: {cfg.n_speakers}")

    # Dataset: .pt files or manifest JSONL
    if args.data_dir:
        dataset = PtDataset(args.data_dir, max_frames=args.max_frames)
        print(f"  Dataset: {len(dataset)} utterances from .pt files")
    elif args.manifest:
        dataset = ManifestDataset(args.manifest, cfg, max_frames=args.max_frames)
        print(f"  Dataset: {len(dataset)} utterances from manifest")
    else:
        print("  ERROR: Provide --data-dir or --manifest")
        return

    if len(dataset) == 0:
        print("  ERROR: No data found.")
        return

    # Train/val split
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

    ema = None
    if HAS_EMA:
        ema = EMA(model, decay=0.999)

    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError):
            print(f"  WARNING: Could not load optimizer state")
        if ema and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed from step {start_step}")

    step = start_step
    running_loss = 0.0
    t0 = time.time()

    print(f"\n  Training: step {start_step} → {args.steps}, batch={args.batch_size}")
    print(f"  Train/val split: {train_size}/{val_size}")

    model.train()
    while step < args.steps:
        for chars, mel, mel_mask, speakers in loader:
            if step >= args.steps:
                break

            chars = chars.to(device)
            mel = mel.to(device)
            mel_mask = mel_mask.to(device)
            speakers = speakers.to(device) if cfg.n_speakers > 0 else None

            lr = cosine_lr(step, args.warmup, args.lr, args.lr * 0.01, args.steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            loss = model.compute_loss(mel, chars, speaker_ids=speakers)

            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at step {step}, skipping")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema:
                ema.update()

            running_loss += loss.item()
            step += 1

            if step % args.log_every == 0:
                n = args.log_every
                elapsed = time.time() - t0
                print(f"  step {step:6d} | loss={running_loss/n:.4f} | "
                      f"lr={lr:.2e} | {n/elapsed:.1f} steps/s")
                running_loss = 0.0
                t0 = time.time()

            if step % args.save_every == 0:
                save_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": vars(cfg),
                    "best_val_loss": best_val_loss,
                }
                if ema:
                    save_dict["ema"] = ema.state_dict()
                path = ckpt_dir / f"flow_v2_step_{step}.pt"
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
                if is_best:
                    best_val_loss = val_loss
                    save_dict = {
                        "model": model.state_dict(),
                        "step": step,
                        "config": vars(cfg),
                        "best_val_loss": best_val_loss,
                    }
                    if ema:
                        save_dict["ema"] = ema.state_dict()
                    best_path = ckpt_dir / "flow_v2_best.pt"
                    torch.save(save_dict, best_path)
                    print(f"  [best] {best_path}")

    # Final save
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": vars(cfg),
        "best_val_loss": best_val_loss,
    }
    if ema:
        save_dict["ema"] = ema.state_dict()
    path = ckpt_dir / "flow_v2_final.pt"
    torch.save(save_dict, path)
    print(f"\n  Final: {path}")
    print(f"  Best val loss: {best_val_loss:.4f}")

    cfg_path = ckpt_dir / "flow_v2_config.json"
    with open(cfg_path, "w") as f:
        json.dump(vars(cfg), f, indent=2)
    print(f"  Config: {cfg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sonata Flow v2")
    parser.add_argument("--data-dir", default="", help="Directory with .pt files")
    parser.add_argument("--manifest", default="", help="Manifest JSONL with audio+text")
    parser.add_argument("--output-dir", default="train/checkpoints/flow_v2")
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
    parser.add_argument("--resume", default="", help="Resume from checkpoint")
    args = parser.parse_args()
    train(args)
