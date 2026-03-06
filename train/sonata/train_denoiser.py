#!/usr/bin/env python3
"""
train_denoiser.py — Train ERB-band denoiser for Sonata deep_filter module.

Architecture:
  FFT(512) → 32 ERB bands (log-compressed) → 2-layer GRU(64) → sigmoid gain mask

Trains on LibriSpeech clean speech mixed with DEMAND/MUSAN noise at various SNRs.
Exports weights to .dnf binary format for C inference.

Usage:
  # Install dependencies
  pip install torch torchaudio soundfile pesq

  # Train (downloads LibriSpeech-clean-100 + DEMAND noise automatically)
  python train_denoiser.py --epochs 50 --batch-size 32

  # Export weights for C inference
  python train_denoiser.py --export --checkpoint checkpoints/denoiser_best.pt \
    --output models/denoiser.dnf
"""

import argparse
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Architecture constants (must match deep_filter.c) ──────────────────────

SAMPLE_RATE = 16000
FFT_SIZE = 512
HOP_SIZE = 256
N_ERB = 32
GRU_HIDDEN = 64
N_GRU_LAYERS = 2
FREQ_BINS = FFT_SIZE // 2 + 1  # 257

# ── ERB filter bank ─────────────────────────────────────────────────────────


def hz_to_erb_rate(hz):
    return 21.4 * np.log10(0.00437 * hz + 1.0)


def compute_erb_filter_bank(n_erb, n_bins, sample_rate):
    """Compute triangular ERB filter bank [n_erb, n_bins]."""
    nyquist = sample_rate / 2.0
    erb_low = hz_to_erb_rate(0.0)
    erb_high = hz_to_erb_rate(nyquist)
    erb_step = (erb_high - erb_low) / (n_erb + 1)

    centers = np.array([
        (10 ** ((erb_low + i * erb_step) / 21.4) - 1) / 0.00437
        for i in range(n_erb + 2)
    ])

    bin_hz = nyquist / (n_bins - 1)
    fb = np.zeros((n_erb, n_bins), dtype=np.float32)

    for band in range(n_erb):
        lo, mid, hi = centers[band], centers[band + 1], centers[band + 2]
        for b in range(n_bins):
            freq = b * bin_hz
            if lo <= freq <= mid and mid > lo:
                fb[band, b] = (freq - lo) / (mid - lo)
            elif mid < freq <= hi and hi > mid:
                fb[band, b] = (hi - freq) / (hi - mid)

        # Normalize
        s = fb[band].sum()
        if s > 0:
            fb[band] /= s

    return fb


# ── Model ────────────────────────────────────────────────────────────────────


class ERBDenoiser(nn.Module):
    """ERB-band gain prediction denoiser.

    Input: magnitude spectrum [batch, time, freq_bins]
    Output: per-ERB-band gain mask [batch, time, n_erb]
    """

    def __init__(self, n_erb=N_ERB, gru_hidden=GRU_HIDDEN, n_gru_layers=N_GRU_LAYERS):
        super().__init__()
        self.n_erb = n_erb
        self.gru_hidden = gru_hidden
        self.n_gru_layers = n_gru_layers

        # ERB filter bank (not trainable — fixed perceptual scale)
        erb_fb = compute_erb_filter_bank(n_erb, FREQ_BINS, SAMPLE_RATE)
        self.register_buffer("erb_fb", torch.from_numpy(erb_fb))

        # 2-layer GRU
        self.gru = nn.GRU(
            input_size=n_erb,
            hidden_size=gru_hidden,
            num_layers=n_gru_layers,
            batch_first=True,
        )

        # Output projection
        self.out_linear = nn.Linear(gru_hidden, n_erb)

    def forward(self, mag, h=None):
        """
        Args:
            mag: [B, T, FREQ_BINS] magnitude spectrum
            h: optional GRU hidden state [n_layers, B, H]
        Returns:
            gains: [B, T, N_ERB] gain mask in [0, 1]
            h_out: GRU hidden state
        """
        # Map to ERB bands: [B, T, N_ERB]
        erb = torch.matmul(mag, self.erb_fb.T)

        # Log-compress
        erb = torch.log(erb + 1.0)

        # GRU
        gru_out, h_out = self.gru(erb, h)

        # Output projection + sigmoid
        gains = torch.sigmoid(self.out_linear(gru_out))

        return gains, h_out


# ── Dataset ──────────────────────────────────────────────────────────────────


class NoisyDataset(Dataset):
    """Mix clean speech with noise at random SNRs."""

    def __init__(self, clean_dir, noise_dir, segment_len=2.0, snr_range=(-5, 20)):
        self.segment_samples = int(segment_len * SAMPLE_RATE)
        self.snr_range = snr_range

        self.clean_files = sorted(Path(clean_dir).rglob("*.flac"))
        if not self.clean_files:
            self.clean_files = sorted(Path(clean_dir).rglob("*.wav"))

        self.noise_files = sorted(Path(noise_dir).rglob("*.wav"))
        if not self.noise_files:
            self.noise_files = sorted(Path(noise_dir).rglob("*.flac"))

        if not self.clean_files:
            raise RuntimeError(f"No audio files found in {clean_dir}")
        if not self.noise_files:
            raise RuntimeError(f"No noise files found in {noise_dir}")

        print(f"[dataset] {len(self.clean_files)} clean, {len(self.noise_files)} noise files")

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        import torchaudio

        # Load clean speech
        clean, sr = torchaudio.load(self.clean_files[idx])
        if sr != SAMPLE_RATE:
            clean = torchaudio.functional.resample(clean, sr, SAMPLE_RATE)
        clean = clean[0]  # mono

        # Pad or trim to segment length
        if len(clean) < self.segment_samples:
            clean = F.pad(clean, (0, self.segment_samples - len(clean)))
        else:
            start = torch.randint(0, len(clean) - self.segment_samples + 1, (1,)).item()
            clean = clean[start : start + self.segment_samples]

        # Load random noise
        noise_idx = torch.randint(0, len(self.noise_files), (1,)).item()
        noise, nsr = torchaudio.load(self.noise_files[noise_idx])
        if nsr != SAMPLE_RATE:
            noise = torchaudio.functional.resample(noise, nsr, SAMPLE_RATE)
        noise = noise[0]

        # Loop noise if shorter than clean
        if len(noise) < self.segment_samples:
            reps = math.ceil(self.segment_samples / len(noise))
            noise = noise.repeat(reps)
        start = torch.randint(0, len(noise) - self.segment_samples + 1, (1,)).item()
        noise = noise[start : start + self.segment_samples]

        # Mix at random SNR
        snr_db = torch.FloatTensor(1).uniform_(*self.snr_range).item()
        clean_rms = clean.pow(2).mean().sqrt().clamp(min=1e-8)
        noise_rms = noise.pow(2).mean().sqrt().clamp(min=1e-8)
        snr_scale = clean_rms / (noise_rms * 10 ** (snr_db / 20))
        noise = noise * snr_scale
        noisy = clean + noise

        return noisy, clean


def compute_stft_targets(noisy, clean, erb_fb):
    """Compute target ERB gains from clean/noisy STFT magnitudes.

    Args:
        noisy: [B, T_samples] noisy waveform
        clean: [B, T_samples] clean waveform
        erb_fb: [N_ERB, FREQ_BINS] ERB filter bank

    Returns:
        noisy_mag: [B, T_frames, FREQ_BINS]
        target_gains: [B, T_frames, N_ERB]
    """
    window = torch.hann_window(FFT_SIZE, device=noisy.device)

    noisy_stft = torch.stft(noisy, FFT_SIZE, HOP_SIZE, window=window,
                             return_complex=True)
    clean_stft = torch.stft(clean, FFT_SIZE, HOP_SIZE, window=window,
                             return_complex=True)

    noisy_mag = noisy_stft.abs()  # [B, FREQ_BINS, T]
    clean_mag = clean_stft.abs()

    # Transpose to [B, T, FREQ_BINS]
    noisy_mag = noisy_mag.transpose(1, 2)
    clean_mag = clean_mag.transpose(1, 2)

    # Compute ideal ERB gains: clean_erb / (noisy_erb + eps)
    noisy_erb = torch.matmul(noisy_mag, erb_fb.T)
    clean_erb = torch.matmul(clean_mag, erb_fb.T)

    target_gains = (clean_erb / (noisy_erb + 1e-8)).clamp(0, 1)

    return noisy_mag, target_gains


# ── Training ─────────────────────────────────────────────────────────────────


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[train] device: {device}")

    model = ERBDenoiser().to(device)
    print(f"[train] params: {sum(p.numel() for p in model.parameters()):,}")

    dataset = NoisyDataset(args.clean_dir, args.noise_dir,
                           segment_len=args.segment_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            noisy_mag, target_gains = compute_stft_targets(
                noisy, clean, model.erb_fb
            )

            pred_gains, _ = model(noisy_mag)

            # MSE loss on gain mask
            loss = F.mse_loss(pred_gains, target_gains)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"[epoch {epoch + 1}/{args.epochs}] loss={avg_loss:.6f} "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, os.path.join(args.checkpoint_dir, "denoiser_best.pt"))
            print(f"  → saved best model (loss={avg_loss:.6f})")

    print(f"[train] done. best loss={best_loss:.6f}")


# ── Export to .dnf ───────────────────────────────────────────────────────────

DNF_MAGIC = 0x46464E44  # "DNFF"
DNF_VERSION = 1


def export_dnf(args):
    """Export trained model to .dnf binary weight file."""
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ERBDenoiser()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "wb") as f:
        # Header
        f.write(struct.pack("<IIIIIIII4I",
                            DNF_MAGIC, DNF_VERSION, SAMPLE_RATE,
                            FFT_SIZE, HOP_SIZE, N_ERB,
                            N_GRU_LAYERS, GRU_HIDDEN,
                            0, 0, 0, 0))  # reserved

        # ERB filter bank [N_ERB, FREQ_BINS]
        erb_fb = model.erb_fb.numpy().astype(np.float32)
        f.write(erb_fb.tobytes())

        # GRU layers
        state = model.state_dict()
        for layer_idx in range(N_GRU_LAYERS):
            prefix = f"gru.weight_ih_l{layer_idx}"
            W_ih = state[prefix].numpy().astype(np.float32)
            f.write(W_ih.tobytes())

            prefix = f"gru.weight_hh_l{layer_idx}"
            W_hh = state[prefix].numpy().astype(np.float32)
            f.write(W_hh.tobytes())

            prefix = f"gru.bias_ih_l{layer_idx}"
            b_ih = state[prefix].numpy().astype(np.float32)
            f.write(b_ih.tobytes())

            prefix = f"gru.bias_hh_l{layer_idx}"
            b_hh = state[prefix].numpy().astype(np.float32)
            f.write(b_hh.tobytes())

        # Output linear: weight [N_ERB, GRU_HIDDEN] + bias [N_ERB]
        out_w = state["out_linear.weight"].numpy().astype(np.float32)
        out_b = state["out_linear.bias"].numpy().astype(np.float32)
        f.write(out_w.tobytes())
        f.write(out_b.tobytes())

    # Verify file size
    total_params = N_ERB * FREQ_BINS  # ERB filter bank
    gru_input_dims = [N_ERB, GRU_HIDDEN]
    for i in range(N_GRU_LAYERS):
        D = gru_input_dims[i]
        H = GRU_HIDDEN
        G = 3 * H
        total_params += G * D + G * H + G + G
    total_params += N_ERB * GRU_HIDDEN + N_ERB

    expected_size = 48 + total_params * 4  # header + float32 params
    actual_size = os.path.getsize(args.output)

    print(f"[export] {args.output}: {total_params:,} params, "
          f"{actual_size:,} bytes (expected {expected_size:,})")

    if actual_size != expected_size:
        print(f"WARNING: size mismatch! expected {expected_size}, got {actual_size}")
        sys.exit(1)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train ERB-band denoiser")
    sub = parser.add_subparsers(dest="cmd")

    # Train
    train_p = sub.add_parser("train", help="Train the denoiser")
    train_p.add_argument("--clean-dir", required=True,
                         help="Directory with clean speech (LibriSpeech)")
    train_p.add_argument("--noise-dir", required=True,
                         help="Directory with noise files (DEMAND/MUSAN)")
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--segment-len", type=float, default=2.0,
                         help="Training segment length in seconds")
    train_p.add_argument("--checkpoint-dir", default="checkpoints",
                         help="Directory for saving checkpoints")

    # Export
    export_p = sub.add_parser("export", help="Export weights to .dnf")
    export_p.add_argument("--checkpoint", required=True,
                          help="Path to .pt checkpoint")
    export_p.add_argument("--output", default="models/denoiser.dnf",
                          help="Output .dnf file path")

    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "export":
        export_dnf(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
