#!/usr/bin/env python3
"""
train_speaker_encoder.py — Train ECAPA-TDNN speaker encoder with GE2E loss.

Architecture: SE-Res2Net blocks + attentive statistics pooling
Loss: Generalized End-to-End (GE2E) contrastive loss
Data: LibriTTS-R (download: https://www.openslr.org/141/)

Usage:
    python3 train_speaker_encoder.py \
      --data-dir /path/to/libritts_r \
      --output-dir ./checkpoints \
      --batch-size 32 \
      --n-epochs 100 \
      --lr 0.001

Exports to safetensors for Rust loading.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    from safetensors.torch import save_file
except ImportError:
    print("Error: safetensors not installed. Run: pip install safetensors")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ── Mel Spectrogram Extraction ─────────────────────────────────────────────

class MelSpectrogramExtractor(nn.Module):
    """Extract 80-bin mel spectrograms for speaker encoding."""

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=80,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=20.0,
            f_max=7600.0,
            center=True,
            pad_mode='reflect'
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] or [T]
        Returns:
            mel_spec: [B, 80, time] or [80, time]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = self.mel_transform(waveform)
        mel = torch.log(mel.clamp(min=1e-9))
        return mel


# ── Batch Normalization 1D ────────────────────────────────────────────────

class BatchNorm1d(nn.Module):
    """Batch normalization that preserves running stats for inference."""

    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels, eps=eps, momentum=momentum, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


# ── SE (Squeeze-and-Excitation) Block ──────────────────────────────────────

class SEBlock(nn.Module):
    """Channel-wise squeeze-and-excitation."""

    def __init__(self, channels: int, se_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, se_channels, 1)
        self.conv2 = nn.Conv1d(se_channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        s = x.mean(dim=2, keepdim=True)  # [B, C, 1]
        s = F.relu(self.conv1(s))
        s = torch.sigmoid(self.conv2(s))
        return x * s


# ── Res2Net Block ───────────────────────────────────────────────────────────

class Res2NetBlock(nn.Module):
    """Multi-branch residual block with dilated convolutions."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, scale: int = 8):
        super().__init__()
        self.scale = scale
        self.width = channels // scale
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # scale-2 convolutions (first chunk is identity, last chunk is identity)
        padding = (kernel_size - 1) * dilation // 2
        for _ in range(scale - 1):
            self.convs.append(
                nn.Conv1d(self.width, self.width, kernel_size,
                         padding=padding, dilation=dilation)
            )
            self.bns.append(BatchNorm1d(self.width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] → split into scale chunks along C
        chunks = x.chunk(self.scale, dim=1)

        outputs = [chunks[0]]  # First chunk identity

        for i in range(1, self.scale):
            if i == 1:
                input_chunk = chunks[i]
            else:
                input_chunk = chunks[i] + outputs[-1]

            out = self.convs[i - 1](input_chunk)
            out = self.bns[i - 1](out)
            out = F.relu(out)
            outputs.append(out)

        return torch.cat(outputs, dim=1)


# ── SE-Res2Net Block (one ECAPA-TDNN block) ──────────────────────────────

class SERes2NetBlock(nn.Module):
    """ECAPA-TDNN building block: SE-Res2Net with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, scale: int = 8, se_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = BatchNorm1d(out_channels)
        self.res2net = Res2NetBlock(out_channels, kernel_size, dilation, scale)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn2 = BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels, se_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x) if self.shortcut else x

        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.res2net(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.se(h)

        return F.relu(h + residual)


# ── Attentive Statistics Pooling ──────────────────────────────────────────

class AttentiveStatisticsPooling(nn.Module):
    """Channel-wise attention pooling with weighted mean and std."""

    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        self.attention = nn.Conv1d(channels, attention_channels, 1)
        self.bn = BatchNorm1d(attention_channels)
        self.proj = nn.Conv1d(attention_channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        alpha = self.attention(x)
        alpha = self.bn(alpha)
        alpha = torch.tanh(alpha)
        alpha = self.proj(alpha)
        alpha = F.softmax(alpha, dim=2)  # [B, C, T]

        # Weighted mean
        mean = (x * alpha).sum(dim=2)  # [B, C]

        # Weighted std
        var = ((x - mean.unsqueeze(2)) ** 2 * alpha).sum(dim=2)
        std = torch.sqrt(var + 1e-8)

        return torch.cat([mean, std], dim=1)  # [B, 2*C]


# ── ECAPA-TDNN Model ──────────────────────────────────────────────────────

class EcapaTdnn(nn.Module):
    """ECAPA-TDNN speaker encoder."""

    def __init__(self, n_mels: int = 80, embedding_dim: int = 256,
                 channels: list = None, kernel_sizes: list = None,
                 dilations: list = None, scale: int = 8,
                 se_channels: int = 128, attention_channels: int = 128):
        super().__init__()

        if channels is None:
            channels = [1024, 1024, 1024, 1024, 1536]
        if kernel_sizes is None:
            kernel_sizes = [5, 3, 3, 3, 1]
        if dilations is None:
            dilations = [1, 2, 3, 4, 1]

        self.input_conv = nn.Conv1d(n_mels, channels[0], kernel_sizes[0],
                                     padding=(kernel_sizes[0] - 1) // 2)
        self.input_bn = BatchNorm1d(channels[0])

        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            block = SERes2NetBlock(
                channels[i], channels[i + 1], kernel_sizes[i + 1],
                dilations[i + 1], scale, se_channels
            )
            self.blocks.append(block)

        # Multi-layer feature aggregation
        # features = [input_conv_out] + [block_i_out for each block]
        # = channels[0] + channels[1] + ... + channels[-1] = sum(channels)
        total_cat = sum(channels)
        last_ch = channels[-1]
        self.mfa_conv = nn.Conv1d(total_cat, last_ch, 1)
        self.mfa_bn = BatchNorm1d(last_ch)

        self.asp = AttentiveStatisticsPooling(last_ch, attention_channels)

        self.final_linear = nn.Linear(last_ch * 2, embedding_dim)
        self.final_bn = BatchNorm1d(embedding_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 80, T]
        Returns:
            embedding: [B, 256] L2-normalized
        """
        x = self.input_conv(mel)
        x = self.input_bn(x)
        x = F.relu(x)

        features = [x]
        for block in self.blocks:
            x = block(x)
            features.append(x)

        # Multi-layer feature aggregation
        cat = torch.cat(features, dim=1)
        x = self.mfa_conv(cat)
        x = self.mfa_bn(x)
        x = F.relu(x)

        # Attentive statistics pooling
        x = self.asp(x)  # [B, 2*C]

        # Final projection + L2 normalize
        x = self.final_linear(x)  # [B, 256]
        x = self.final_bn(x)

        # L2 normalization
        return F.normalize(x, p=2, dim=1)


# ── GE2E Loss ─────────────────────────────────────────────────────────────

class GE2ELoss(nn.Module):
    """Generalized End-to-End speaker verification loss."""

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embeddings: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, 256] L2-normalized speaker embeddings
            speaker_ids: [B] speaker labels (0 to num_speakers-1)
        Returns:
            loss: scalar
        """
        # Remap global speaker IDs to batch-local indices [0, num_speakers)
        unique_ids, local_ids = torch.unique(speaker_ids, sorted=True, return_inverse=True)
        num_speakers = len(unique_ids)

        # Compute speaker centroids
        centroids = []
        for i in range(num_speakers):
            mask = local_ids == i
            centroid = embeddings[mask].mean(dim=0)
            centroid = F.normalize(centroid, p=2, dim=0)
            centroids.append(centroid)

        centroids = torch.stack(centroids)  # [num_speakers, 256]

        # Compute similarities: [B, num_speakers]
        sims = torch.mm(embeddings, centroids.t())
        sims = self.w * sims + self.b

        # Cross-entropy loss with batch-local labels
        loss = F.cross_entropy(sims, local_ids)
        return loss


# ── Dataset ────────────────────────────────────────────────────────────────

class LibriTTSRDataset(torch.utils.data.Dataset):
    """LibriTTS-R dataset for speaker encoder training."""

    def __init__(self, data_dir: str, sample_rate: int = 16000,
                 segment_duration: float = 3.0, mode: str = 'train'):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_duration * sample_rate)
        self.mode = mode

        self.files = []
        self.speaker_ids = {}
        self._build_index()

    # LibriTTS-R directory name mapping
    SUBSET_DIRS = {
        'train': ['train-clean-100', 'train-clean-360', 'train-other-500'],
        'test': ['test-clean', 'test-other'],
    }

    def _build_index(self):
        """Build index of audio files and speaker mappings."""
        subset_dirs = self.SUBSET_DIRS.get(self.mode, self.SUBSET_DIRS['train'])

        found_any = False
        for subset_name in subset_dirs:
            audio_dir = self.data_dir / subset_name
            if not audio_dir.exists():
                logger.warning(f"Dataset directory not found: {audio_dir}")
                continue
            found_any = True
            for wav_file in sorted(audio_dir.rglob('*.wav')):
                # LibriTTS-R structure: subset/speaker_id/chapter_id/file.wav
                parts = wav_file.relative_to(audio_dir).parts
                if len(parts) >= 2:
                    speaker_id = parts[0]
                    if speaker_id not in self.speaker_ids:
                        self.speaker_ids[speaker_id] = len(self.speaker_ids)
                    self.files.append((wav_file, speaker_id))

        if not found_any:
            logger.error(f"No dataset directories found in {self.data_dir} for mode '{self.mode}'. "
                        f"Expected: {subset_dirs}")

        logger.info(f"Indexed {len(self.files)} files from {len(self.speaker_ids)} speakers")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        wav_path, speaker_id = self.files[idx]
        spk_label = self.speaker_ids[speaker_id]

        try:
            waveform, sr = torchaudio.load(str(wav_path))

            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)

            if waveform.shape[0] >= self.segment_samples:
                start = np.random.randint(0, waveform.shape[0] - self.segment_samples)
                waveform = waveform[start:start + self.segment_samples]
            else:
                waveform = F.pad(waveform, (0, self.segment_samples - waveform.shape[0]))

            return waveform, spk_label
        except Exception as e:
            logger.warning(f"Error loading {wav_path}: {e}")
            return torch.randn(self.segment_samples), spk_label


# ── Training ────────────────────────────────────────────────────────────────

def train_epoch(model, mel_extractor, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    mel_extractor.eval()
    total_loss = 0.0

    with tqdm(loader, desc='Training') as pbar:
        for waveforms, speaker_ids in pbar:
            waveforms = waveforms.to(device)
            speaker_ids = speaker_ids.to(device)

            optimizer.zero_grad()

            mel = mel_extractor(waveforms)
            embeddings = model(mel)
            loss = criterion(embeddings, speaker_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)


def validate(model, mel_extractor, loader, criterion, device):
    """Validate model."""
    model.eval()
    mel_extractor.eval()
    total_loss = 0.0

    with torch.no_grad():
        with tqdm(loader, desc='Validating') as pbar:
            for waveforms, speaker_ids in pbar:
                waveforms = waveforms.to(device)
                speaker_ids = speaker_ids.to(device)

                mel = mel_extractor(waveforms)
                embeddings = model(mel)
                loss = criterion(embeddings, speaker_ids)

                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)


def export_to_safetensors(model, config, output_path):
    """Export model weights to safetensors format."""
    state_dict = model.state_dict()
    state_dict = {k: v.cpu().float() for k, v in state_dict.items()}

    save_file(state_dict, output_path)
    logger.info(f"Exported model to {output_path}")

    config_path = output_path.replace('.safetensors', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Exported config to {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Train ECAPA-TDNN speaker encoder')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to LibriTTS-R dataset')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Output dir: {output_dir}")

    logger.info("Loading LibriTTS-R dataset...")
    train_dataset = LibriTTSRDataset(args.data_dir, mode='train')

    if len(train_dataset) == 0:
        logger.error("No training data found! Check --data-dir path.")
        sys.exit(1)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    test_dataset = LibriTTSRDataset(args.data_dir, mode='test')
    test_loader = None
    if len(test_dataset) > 0:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
    else:
        logger.warning("No test data found — using train loss for checkpointing")

    logger.info("Creating ECAPA-TDNN model...")
    model = EcapaTdnn(
        n_mels=80,
        embedding_dim=256,
        channels=[1024, 1024, 1024, 1024, 1536],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        scale=8,
        se_channels=128,
        attention_channels=128
    ).to(device)

    mel_extractor = MelSpectrogramExtractor().to(device)
    criterion = GE2ELoss().to(device)
    optimizer = Adam(list(model.parameters()) + list(criterion.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    best_loss = float('inf')
    for epoch in range(args.n_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.n_epochs}")

        train_loss = train_epoch(model, mel_extractor, train_loader, criterion, optimizer, device)

        if test_loader is not None:
            val_loss = validate(model, mel_extractor, test_loader, criterion, device)
            logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
            track_loss = val_loss
        else:
            logger.info(f"Train loss: {train_loss:.4f}")
            track_loss = train_loss

        scheduler.step()

        if track_loss < best_loss:
            best_loss = track_loss
            checkpoint_path = output_dir / f"speaker_encoder_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Exporting to safetensors...")
    config = {
        'n_mels': 80,
        'channels': [1024, 1024, 1024, 1024, 1536],
        'kernel_sizes': [5, 3, 3, 3, 1],
        'dilations': [1, 2, 3, 4, 1],
        'embedding_dim': 256,
        'res2net_scale': 8,
        'se_channels': 128,
        'attention_channels': 128,
        'sample_rate': 16000,
    }
    export_to_safetensors(model, config, str(output_dir / 'speaker_encoder.safetensors'))
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
