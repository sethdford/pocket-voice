"""Train Voice Activity Projection (VAP) model for turn-taking prediction.

SOTA neural turn-taking predictor: user+system mel → 4 sigmoid heads
(user_speaking, system_turn, backchannel, eou).

Architecture matches C inference (vap_model.c):
  Input [T, 160] → Linear(160, d_model) → SinusoidalPE
  → N transformer encoder layers (RMSNorm, causal attn, SiLU FFN)
  → Final RMSNorm → 4 Linear heads → sigmoid

Usage:
  # Synthetic mode (no data)
  python train_vap.py --synthetic 500 --steps 5000 --export

  # Real data from JSONL manifest (Fisher, Switchboard, CallHome format)
  python train_vap.py --manifest data/vap_manifest.jsonl --steps 50000 --export

  # Resume training from checkpoint
  python train_vap.py --manifest data/vap_manifest.jsonl --resume checkpoints/vap/step_5000.pt --steps 50000

  # Export only (from checkpoint)
  python train_vap.py --ckpt checkpoints/vap/best.pt --export --output models/vap.vap

Features:
  - Step-based training with cosine LR schedule + warmup
  - 90/10 train/val split with per-head metrics (AUC, accuracy)
  - Checkpoint save every N steps + best checkpoint by val AUC
  - Data augmentation: speed perturbation, additive noise, channel swap
  - Gradient clipping and loss logging to TrainingLog (JSONL)
"""

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, random_split

from modules import TrainingLog, cosine_lr

# VAP binary format (must match vap_model.c)
VAP_MAGIC = 0x00504156  # "VAP\0" little-endian
VAP_VERSION = 1
VAP_INPUT_DIM = 160
VAP_MEL_DIM = 80


def sinusoidal_pe(seq_len: int, d_model: int, device) -> torch.Tensor:
    """Sinusoidal positional encoding."""
    pe = torch.zeros(seq_len, d_model, device=device)
    div = 10000.0
    for i in range(d_model):
        freq = 1.0 / (div ** (i / d_model))
        pe[:, i] = torch.arange(seq_len, device=device, dtype=torch.float32) * freq
    pe[:, 0::2] = pe[:, 0::2].cos()
    pe[:, 1::2] = pe[:, 1::2].sin()
    return pe


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class VAPModel(nn.Module):
    """VAP: Voice Activity Projection for turn-taking prediction.

    Matches C architecture exactly for export compatibility.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_dim: int = 256,
        input_dim: int = VAP_INPUT_DIM,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ff_dim = ff_dim
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            VAPEncoderLayer(d_model, n_heads, self.head_dim, ff_dim)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.head_user = nn.Linear(d_model, 1)
        self.head_system = nn.Linear(d_model, 1)
        self.head_backchannel = nn.Linear(d_model, 1)
        self.head_eou = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, T, 160]
        B, T, _ = x.shape
        x = self.input_proj(x)
        pe = sinusoidal_pe(T, self.d_model, x.device).unsqueeze(0)
        x = x + pe

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        for layer in self.layers:
            x = layer(x, causal_mask) + x

        x = self.final_norm(x)
        p_user = torch.sigmoid(self.head_user(x)).squeeze(-1)
        p_system = torch.sigmoid(self.head_system(x)).squeeze(-1)
        p_backchannel = torch.sigmoid(self.head_backchannel(x)).squeeze(-1)
        p_eou = torch.sigmoid(self.head_eou(x)).squeeze(-1)
        return torch.stack([p_user, p_system, p_backchannel, p_eou], dim=-1)


class VAPEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, ff_dim: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn_up = nn.Linear(d_model, ff_dim)
        self.ffn_down = nn.Linear(ff_dim, d_model)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        n_heads = self.wq.out_features // self.wk.out_features
        head_dim = self.wk.out_features // n_heads

        residual = x
        x = self.attn_norm(x)
        h = self.wq.out_features // head_dim
        q = self.wq(x).view(B, T, h, head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, h, head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, h, head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, h * head_dim)
        out = self.wo(out)

        x = residual + out
        residual = x
        x = self.ffn_norm(x)
        x = F.silu(self.ffn_up(x))
        x = self.ffn_down(x)
        return residual + x


# ═════════════════════════════════════════════════════════════════════════════
# Datasets
# ═════════════════════════════════════════════════════════════════════════════


class SyntheticVAPDataset(Dataset):
    """Synthetic turn-taking data for testing."""

    def __init__(
        self,
        n_samples: int = 500,
        max_frames: int = 250,
        frame_rate: int = 50,
    ):
        self.n_samples = n_samples
        self.max_frames = max_frames
        self.frame_rate = frame_rate

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        T = torch.randint(50, self.max_frames + 1, (1,)).item()
        mel = torch.randn(T, VAP_INPUT_DIM) * 0.5
        # Random binary labels
        labels = torch.randint(0, 2, (T, 4)).float()
        return mel, labels


class VAPManifestDataset(Dataset):
    """Conversational audio from JSONL manifest.

    Manifest format:
    {"audio_path": "conv001.wav", "channels": 2, "annotations": [
        {"start": 0.0, "end": 2.5, "speaker": "A", "type": "speech"},
        {"start": 1.8, "end": 2.0, "speaker": "B", "type": "backchannel"},
        {"start": 2.8, "end": 5.1, "speaker": "B", "type": "speech"}
    ]}

    Generates frame-level labels at 50Hz for horizon 0.5s.
    Supports data augmentation: speed perturbation, additive noise, channel swap.
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 16000,
        frame_rate: int = 50,
        horizon: float = 0.5,
        max_frames: int = 500,
        user_channel: int = 0,
        system_channel: int = 1,
        augment: bool = False,
    ):
        self.frame_rate = frame_rate
        self.horizon = horizon
        self.max_frames = max_frames
        self.user_channel = user_channel
        self.system_channel = system_channel
        self.sample_rate = sample_rate
        self.hop = sample_rate // frame_rate
        self.augment = augment

        self.items = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self.items.append(entry)

    def __len__(self):
        return len(self.items)

    def _speed_perturb(self, audio: torch.Tensor, factor: float = 0.95) -> torch.Tensor:
        """Resample audio by factor (0.9-1.1x)."""
        if factor == 1.0:
            return audio
        new_len = int(audio.shape[0] / factor)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        else:
            audio = audio.unsqueeze(0)
        resampled = F.interpolate(audio, size=new_len, mode="linear", align_corners=False)
        return resampled.squeeze(0).squeeze(0) if resampled.shape[0] == 1 else resampled.squeeze(0)

    def _add_noise(self, audio: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
        """Add Gaussian noise at specified SNR."""
        signal_power = (audio ** 2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = torch.randn_like(audio) * torch.sqrt(noise_power)
        return audio + noise

    def _extract_mel_stft(self, audio: torch.Tensor) -> torch.Tensor:
        """Simple STFT-based log-mel (80 bins)."""
        n_fft = 512
        hop = self.hop
        win = 400
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            return_complex=True,
        )
        mag = spec.abs().clamp(min=1e-5).squeeze(0)
        n_mels = VAP_MEL_DIM
        n_freq = n_fft // 2 + 1
        mel_fb = torch.zeros(n_mels, n_freq)
        for i in range(n_mels):
            low = max(0, int(i * n_freq / (n_mels + 1)))
            mid = min(n_freq, int((i + 1) * n_freq / (n_mels + 1)))
            high = min(n_freq, int((i + 2) * n_freq / (n_mels + 1)))
            if mid > low:
                mel_fb[i, low:mid] = torch.linspace(0, 1, mid - low)
            if high > mid:
                mel_fb[i, mid:high] = torch.linspace(1, 0, high - mid)
        mel = torch.log((mel_fb @ mag).T + 1e-5)
        return mel

    def _annotations_to_labels(
        self,
        annotations: list,
        n_frames: int,
        user_speaker: str = "A",
        system_speaker: str = "B",
    ) -> torch.Tensor:
        """Convert annotations to frame-level labels [T, 4]."""
        labels = torch.zeros(n_frames, 4)
        for ann in annotations:
            start, end = ann["start"], ann["end"]
            speaker = ann.get("speaker", "A")
            typ = ann.get("type", "speech")

            start_frame = int(start * self.frame_rate)
            end_frame = int(end * self.frame_rate)
            horizon_frames = int(self.horizon * self.frame_rate)

            for t in range(n_frames):
                window_end = (t + 1) / self.frame_rate + self.horizon
                window_start = t / self.frame_rate

                if typ == "speech":
                    if speaker == user_speaker:
                        if end >= window_start and start <= window_end:
                            labels[t, 0] = 1.0
                    else:
                        if end >= window_start and start <= window_end:
                            labels[t, 1] = 1.0
                elif typ == "backchannel":
                    if end >= window_start and start <= window_end:
                        labels[t, 2] = 1.0

                if speaker == user_speaker and typ == "speech":
                    if start <= window_end and end >= window_end:
                        labels[t, 3] = 1.0

        return labels

    def __getitem__(self, idx):
        entry = self.items[idx]
        path = entry["audio_path"]
        annotations = entry.get("annotations", [])
        channels = entry.get("channels", 1)

        data, sr = sf.read(path, dtype="float32")
        if data.ndim == 1:
            data = data[:, None]
        user_audio = torch.from_numpy(data[:, self.user_channel % data.shape[1]])
        system_audio = (
            torch.from_numpy(data[:, self.system_channel % data.shape[1]])
            if channels >= 2
            else torch.zeros_like(user_audio)
        )

        if sr != self.sample_rate:
            ratio = self.sample_rate / sr
            new_len = int(user_audio.shape[0] * ratio)
            user_audio = F.interpolate(
                user_audio.unsqueeze(0).unsqueeze(0),
                size=new_len,
                mode="linear",
                align_corners=False,
            ).squeeze()
            system_audio = F.interpolate(
                system_audio.unsqueeze(0).unsqueeze(0),
                size=new_len,
                mode="linear",
                align_corners=False,
            ).squeeze()

        # Data augmentation (during training)
        if self.augment:
            # Speed perturbation: 0.9-1.1x
            speed_factor = torch.empty(1).uniform_(0.9, 1.1).item()
            if abs(speed_factor - 1.0) > 0.01:
                user_audio = self._speed_perturb(user_audio, speed_factor)
                system_audio = self._speed_perturb(system_audio, speed_factor)
            # Additive noise: 5-20 dB SNR
            snr = torch.randint(5, 21, (1,)).item()
            user_audio = self._add_noise(user_audio, snr)
            system_audio = self._add_noise(system_audio, snr)

        user_mel = self._extract_mel_stft(user_audio)
        system_mel = self._extract_mel_stft(system_audio)
        T = min(user_mel.shape[0], system_mel.shape[0], self.max_frames)
        user_mel = user_mel[:T]
        system_mel = system_mel[:T]

        # Channel swap with 10% probability (also swap user/system labels)
        swap_channels = self.augment and torch.rand(1).item() < 0.1
        if swap_channels:
            user_mel, system_mel = system_mel, user_mel

        mel = torch.cat([user_mel, system_mel], dim=-1)
        if mel.shape[1] < VAP_INPUT_DIM:
            mel = F.pad(mel, (0, VAP_INPUT_DIM - mel.shape[1]))
        elif mel.shape[1] > VAP_INPUT_DIM:
            mel = mel[:, :VAP_INPUT_DIM]

        labels = self._annotations_to_labels(annotations, T)

        # If we swapped channels, swap user/system labels
        if swap_channels:
            labels[:, [0, 1]] = labels[:, [1, 0]]

        return mel, labels


# ═════════════════════════════════════════════════════════════════════════════
# Export to .vap binary
# ═════════════════════════════════════════════════════════════════════════════


def export_vap(ckpt_path: str, output_path: str, config: dict | None = None):
    """Export VAP model to .vap binary for C inference."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    cfg = config or ckpt.get("config", {})

    d_model = int(cfg.get("d_model", 128))
    n_layers = int(cfg.get("n_layers", 4))
    n_heads = int(cfg.get("n_heads", 4))
    ff_dim = int(cfg.get("ff_dim", 256))
    head_dim = d_model // n_heads

    def w(name, *shape):
        t = state[name]
        assert t.shape == torch.Size(shape), f"{name}: expected {shape}, got {t.shape}"
        return t.float().numpy().tobytes()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_floats = 0
    total_floats += d_model * VAP_INPUT_DIM + d_model
    for _ in range(n_layers):
        total_floats += d_model + (n_heads * head_dim) * d_model * 4 + d_model * (n_heads * head_dim)
        total_floats += d_model + ff_dim * d_model + ff_dim + d_model * ff_dim + d_model
    total_floats += d_model + d_model * 4 + 4

    with open(output_file, "wb") as f:
        f.write(struct.pack("<I", VAP_MAGIC))
        f.write(struct.pack("<I", VAP_VERSION))
        f.write(struct.pack("<I", d_model))
        f.write(struct.pack("<I", n_layers))
        f.write(struct.pack("<I", n_heads))
        f.write(struct.pack("<I", ff_dim))
        f.write(struct.pack("<Q", total_floats))

        f.write(w("input_proj.weight", d_model, VAP_INPUT_DIM))
        f.write(w("input_proj.bias", d_model))

        for l in range(n_layers):
            prefix = f"layers.{l}."
            f.write(w(f"{prefix}attn_norm.weight", d_model))
            f.write(w(f"{prefix}wq.weight", n_heads * head_dim, d_model))
            f.write(w(f"{prefix}wk.weight", n_heads * head_dim, d_model))
            f.write(w(f"{prefix}wv.weight", n_heads * head_dim, d_model))
            f.write(w(f"{prefix}wo.weight", d_model, n_heads * head_dim))
            f.write(w(f"{prefix}ffn_norm.weight", d_model))
            f.write(w(f"{prefix}ffn_up.weight", ff_dim, d_model))
            f.write(w(f"{prefix}ffn_up.bias", ff_dim))
            f.write(w(f"{prefix}ffn_down.weight", d_model, ff_dim))
            f.write(w(f"{prefix}ffn_down.bias", d_model))

        f.write(w("final_norm.weight", d_model))
        f.write(w("head_user.weight", 1, d_model))
        f.write(w("head_user.bias", 1))
        f.write(w("head_system.weight", 1, d_model))
        f.write(w("head_system.bias", 1))
        f.write(w("head_backchannel.weight", 1, d_model))
        f.write(w("head_backchannel.bias", 1))
        f.write(w("head_eou.weight", 1, d_model))
        f.write(w("head_eou.bias", 1))

    print(f"Exported VAP to {output_path} ({output_file.stat().st_size / 1e6:.1f} MB)")


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════


def compute_binary_accuracy(pred: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute binary accuracy for multi-head predictions."""
    pred_binary = (pred > threshold).float()
    accuracy = (pred_binary == labels).float().mean().item()
    return accuracy


def compute_auc_manual(pred: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute AUC-ROC without sklearn (manual implementation for efficiency)."""
    pred = pred.view(-1)
    labels = labels.view(-1)

    sorted_idx = torch.argsort(pred, descending=True)
    sorted_labels = labels[sorted_idx]

    n_pos = (labels == 1).sum().float()
    n_neg = (labels == 0).sum().float()

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.arange(1, len(sorted_labels) + 1, device=labels.device).float() - tp

    tpr = tp / n_pos
    fpr = fp / n_neg

    auc = torch.trapezoid(tpr, fpr).item()
    return auc


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn,
) -> dict:
    """Run validation loop, return per-head metrics."""
    model.eval()
    total_loss = 0.0
    all_pred = []
    all_labels = []

    with torch.no_grad():
        for mel, labels in val_loader:
            mel, labels = mel.to(device), labels.to(device)
            pred = model(mel)
            loss = loss_fn(pred, labels)
            total_loss += loss.item()
            all_pred.append(pred)
            all_labels.append(labels)

    all_pred = torch.cat(all_pred, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = {
        "loss": total_loss / len(val_loader),
        "accuracy": compute_binary_accuracy(all_pred, all_labels),
        "auc_user": compute_auc_manual(all_pred[..., 0], all_labels[..., 0]),
        "auc_system": compute_auc_manual(all_pred[..., 1], all_labels[..., 1]),
        "auc_backchannel": compute_auc_manual(all_pred[..., 2], all_labels[..., 2]),
        "auc_eou": compute_auc_manual(all_pred[..., 3], all_labels[..., 3]),
    }
    metrics["auc_mean"] = (metrics["auc_user"] + metrics["auc_system"] + metrics["auc_backchannel"] + metrics["auc_eou"]) / 4.0

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════


def train(args):
    """Production-quality training loop with validation, checkpointing, and LR scheduling."""

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = VAPModel(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")

    if args.synthetic:
        dataset = SyntheticVAPDataset(n_samples=args.synthetic, max_frames=250)
        print(f"  Dataset: synthetic ({args.synthetic} samples)")
    elif args.manifest:
        dataset = VAPManifestDataset(
            args.manifest,
            max_frames=args.max_frames,
            augment=args.augment,
        )
        print(f"  Dataset: manifest ({len(dataset)} samples)")
    else:
        raise ValueError("Specify --synthetic N or --manifest path")

    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"  Train/val split: {n_train} / {n_val}")

    def collate_fn(batch):
        mels = torch.nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True)
        return mels, labels

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.max_lr)
    loss_fn = nn.BCELoss()

    tlog = TrainingLog(str(output_dir / "losses.jsonl"))

    start_step = 0
    best_val_auc = 0.0
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        best_val_auc = ckpt.get("best_val_auc", 0.0)
        print(f"  Resumed at step {start_step}, best val AUC {best_val_auc:.4f}")

    config_dict = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "ff_dim": args.ff_dim,
    }

    print(f"  Training for {args.steps} steps")
    print(f"  LR schedule: warmup {args.warmup}, max {args.max_lr:.1e}, min {args.min_lr:.1e}")
    print()

    step = start_step
    epoch = 0
    epoch_start_step = start_step

    while step < args.steps:
        model.train()
        epoch += 1

        for mel, labels in train_loader:
            if step >= args.steps:
                break

            mel, labels = mel.to(device), labels.to(device)

            lr = cosine_lr(step, args.warmup, args.max_lr, args.min_lr, args.steps)
            for param_group in opt.param_groups:
                param_group["lr"] = lr

            pred = model(mel)
            loss = loss_fn(pred, labels)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % args.log_every == 0:
                tlog.log(step=step, loss=loss.item(), lr=lr, epoch=epoch)
                if step % (args.log_every * 10) == 0:
                    print(f"  Step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e}")

            if step % args.val_every == 0 and step > 0:
                val_metrics = validate(model, val_loader, device, loss_fn)
                val_auc = val_metrics["auc_mean"]
                print(f"    Val: loss {val_metrics['loss']:.4f} | acc {val_metrics['accuracy']:.4f} | auc {val_auc:.4f}")
                print(f"         auc_user {val_metrics['auc_user']:.4f} | auc_system {val_metrics['auc_system']:.4f} | auc_bc {val_metrics['auc_backchannel']:.4f} | auc_eou {val_metrics['auc_eou']:.4f}")

                tlog.log(
                    step=step,
                    val_loss=val_metrics["loss"],
                    val_accuracy=val_metrics["accuracy"],
                    val_auc=val_auc,
                    val_auc_user=val_metrics["auc_user"],
                    val_auc_system=val_metrics["auc_system"],
                    val_auc_backchannel=val_metrics["auc_backchannel"],
                    val_auc_eou=val_metrics["auc_eou"],
                )

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_path = output_dir / "best.pt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": opt.state_dict(),
                            "step": step,
                            "config": config_dict,
                            "best_val_auc": best_val_auc,
                        },
                        best_path,
                    )
                    print(f"    Saved best checkpoint (auc {best_val_auc:.4f}) to {best_path.name}")

            if step % args.save_every == 0 and step > 0:
                ckpt_path = output_dir / f"step_{step}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "step": step,
                        "config": config_dict,
                        "best_val_auc": best_val_auc,
                    },
                    ckpt_path,
                )

            step += 1

        print(f"  Epoch {epoch} complete (steps {epoch_start_step}-{step-1})")
        epoch_start_step = step

    print()
    print("  Final validation:")
    val_metrics = validate(model, val_loader, device, loss_fn)
    val_auc = val_metrics["auc_mean"]
    print(f"    Loss: {val_metrics['loss']:.4f}")
    print(f"    Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"    Mean AUC: {val_auc:.4f}")
    print(f"    AUC by head: user {val_metrics['auc_user']:.4f}, system {val_metrics['auc_system']:.4f}, backchannel {val_metrics['auc_backchannel']:.4f}, eou {val_metrics['auc_eou']:.4f}")

    tlog.close()
    print(f"  Training log: {output_dir / 'losses.jsonl'}")
    print(f"  Best checkpoint: {output_dir / 'best.pt'} (val AUC {best_val_auc:.4f})")

    return model, config_dict


def main():
    ap = argparse.ArgumentParser(description="Train VAP model")
    ap.add_argument("--synthetic", type=int, default=0, help="Synthetic mode: number of samples")
    ap.add_argument("--manifest", type=str, help="JSONL manifest path")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size")
    ap.add_argument("--max-frames", type=int, default=500, help="Max frames per sample")
    ap.add_argument("--d-model", type=int, default=128, help="Model dimension")
    ap.add_argument("--n-layers", type=int, default=4, help="Number of encoder layers")
    ap.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    ap.add_argument("--ff-dim", type=int, default=256, help="FFN hidden dimension")
    ap.add_argument("--steps", type=int, default=50000, help="Total training steps")
    ap.add_argument("--warmup", type=int, default=500, help="LR warmup steps")
    ap.add_argument("--max-lr", type=float, default=1e-3, help="Max learning rate")
    ap.add_argument("--min-lr", type=float, default=1e-5, help="Min learning rate")
    ap.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    ap.add_argument("--val-every", type=int, default=500, help="Validate every N steps")
    ap.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    ap.add_argument("--augment", action="store_true", help="Enable data augmentation")
    ap.add_argument("--resume", type=str, help="Resume from checkpoint")
    ap.add_argument("--output-dir", type=str, default="checkpoints/vap", help="Output directory")
    ap.add_argument("--export", action="store_true", help="Export to .vap after training")
    ap.add_argument("--output", type=str, default="models/vap.vap", help="Output .vap path")
    ap.add_argument("--ckpt", type=str, help="Checkpoint path (for export only)")
    args = ap.parse_args()

    if args.manifest or args.synthetic:
        model, config = train(args)
        if args.export:
            export_vap(str(Path(args.output_dir) / "best.pt"), args.output, config)
    elif args.ckpt and args.export:
        export_vap(args.ckpt, args.output)
    else:
        ap.error("Specify --manifest, --synthetic, or use --ckpt with --export")


if __name__ == "__main__":
    main()
