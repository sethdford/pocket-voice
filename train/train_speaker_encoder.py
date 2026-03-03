#!/usr/bin/env python3
"""
train_speaker_encoder.py — Train ECAPA-TDNN speaker encoder with AAM-Softmax loss.

Architecture: SE-Res2Net blocks + attentive statistics pooling (24.6M params)
Loss: Additive Angular Margin Softmax (ArcFace) with optional sub-center support
Data: LibriTTS-R with MUSAN/RIR augmentation + SpecAugment + speed perturbation
Advanced: Sub-center ArcFace, Large Margin Fine-tuning, AS-Norm validation, Multi-length Training
Validation: Equal Error Rate (EER) on test-clean with optional AS-Norm normalization

Usage:
    # Standard training
    python3 train_speaker_encoder.py \
      --data-dir /path/to/LibriTTS_R \
      --output-dir ./checkpoints/speaker_encoder_v2 \
      --batch-size 64 --n-epochs 40 --lr 0.001 \
      --scale 30.0 --margin 0.2 \
      --warmup-epochs 2 --val-every 2 --patience 10

    # With MUSAN/RIR augmentation
    python3 train_speaker_encoder.py \
      --data-dir /path/to/LibriTTS_R \
      --musan-dir /path/to/MUSAN \
      --rir-dir /path/to/RIRS_NOISES \
      --crop-duration 4.0

    # Fine-tuning with large margin
    python3 train_speaker_encoder.py \
      --data-dir /path/to/LibriTTS_R \
      --fine-tune --resume ./checkpoints/best.pt \
      --fine-tune-margin 0.5 --fine-tune-epochs 5 \
      --fine-tune-crop 6.0 --sub-centers 2

    # Validation with AS-Norm
    python3 train_speaker_encoder.py \
      --data-dir /path/to/LibriTTS_R \
      --asnorm --asnorm-cohort-size 300

Exports to safetensors + ONNX for Rust/C inference.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
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


# ============================================================================
# Feature Extraction
# ============================================================================

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
        """Convert waveform to log mel spectrogram.

        Args:
            waveform: [batch, samples] or [samples]

        Returns:
            mel: [batch, 80, time]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel.clamp(min=1e-9))
        return mel


# ============================================================================
# Data Augmentation
# ============================================================================

class SpecAugment(nn.Module):
    """SpecAugment data augmentation for mel spectrograms."""

    def __init__(self, freq_mask_param: int = 10, time_mask_param: int = 50,
                 n_freq_masks: int = 2, n_time_masks: int = 2):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to mel spectrogram.

        Args:
            mel: [batch, n_mels, time]

        Returns:
            augmented_mel: [batch, n_mels, time]
        """
        for _ in range(self.n_freq_masks):
            mel = self.freq_mask(mel)
        for _ in range(self.n_time_masks):
            mel = self.time_mask(mel)
        return mel


class MUSANAugmenter(nn.Module):
    """MUSAN dataset augmentation (noise, music, speech babble)."""

    def __init__(self, musan_dir: Optional[str] = None, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.musan_files = {'noise': [], 'music': [], 'speech': []}

        if musan_dir:
            self._load_musan_index(musan_dir)

    def _load_musan_index(self, musan_dir: str) -> None:
        """Index all MUSAN .wav files by category."""
        musan_path = Path(musan_dir)
        for category in ['noise', 'music', 'speech']:
            cat_dir = musan_path / category
            if cat_dir.exists():
                files = sorted(cat_dir.rglob('*.wav'))
                self.musan_files[category] = files
                logger.info(f"Indexed {len(files)} MUSAN {category} files")

    def forward(self, waveform: torch.Tensor, category: str = None) -> torch.Tensor:
        """Add MUSAN augmentation at random SNR.

        Args:
            waveform: [samples]
            category: 'noise', 'music', 'speech' or random if None

        Returns:
            augmented: [samples]
        """
        if not self.musan_files['noise']:
            return waveform

        if category is None:
            category = np.random.choice(['noise', 'music', 'speech'])

        if category == 'noise':
            return self._add_noise(waveform, snr_range=(5, 20))
        elif category == 'music':
            return self._add_music(waveform, snr_range=(5, 15))
        elif category == 'speech':
            return self._add_babble(waveform, num_speakers=(3, 7), snr_range=(13, 20))
        else:
            return waveform

    def _add_noise(self, waveform: torch.Tensor, snr_range: Tuple[float, float]) -> torch.Tensor:
        """Add single noise clip at random SNR."""
        if not self.musan_files['noise']:
            return waveform

        noise_file = np.random.choice(self.musan_files['noise'])
        noise, sr = self._load_audio(noise_file)

        # Match length
        if noise.shape[0] < waveform.shape[0]:
            noise = torch.tile(noise, (waveform.shape[0] // noise.shape[0] + 1,))
        noise = noise[:waveform.shape[0]]

        snr_db = np.random.uniform(*snr_range)
        return self._mix_at_snr(waveform, noise, snr_db)

    def _add_music(self, waveform: torch.Tensor, snr_range: Tuple[float, float]) -> torch.Tensor:
        """Add single music clip at random SNR."""
        if not self.musan_files['music']:
            return waveform

        music_file = np.random.choice(self.musan_files['music'])
        music, sr = self._load_audio(music_file)

        # Random segment
        if music.shape[0] > waveform.shape[0]:
            start = np.random.randint(0, music.shape[0] - waveform.shape[0])
            music = music[start:start + waveform.shape[0]]
        else:
            music = torch.tile(music, (waveform.shape[0] // music.shape[0] + 1,))
            music = music[:waveform.shape[0]]

        snr_db = np.random.uniform(*snr_range)
        return self._mix_at_snr(waveform, music, snr_db)

    def _add_babble(self, waveform: torch.Tensor, num_speakers: Tuple[int, int],
                    snr_range: Tuple[float, float]) -> torch.Tensor:
        """Add multiple speech clips as babble at random SNR."""
        if not self.musan_files['speech']:
            return waveform

        n_speakers = np.random.randint(*num_speakers)
        babble = torch.zeros_like(waveform)

        for _ in range(n_speakers):
            speech_file = np.random.choice(self.musan_files['speech'])
            speech, sr = self._load_audio(speech_file)

            # Random segment
            if speech.shape[0] > waveform.shape[0]:
                start = np.random.randint(0, speech.shape[0] - waveform.shape[0])
                speech = speech[start:start + waveform.shape[0]]
            else:
                speech = torch.tile(speech, (waveform.shape[0] // speech.shape[0] + 1,))
                speech = speech[:waveform.shape[0]]

            babble = babble + speech

        # Normalize babble
        babble = babble / (n_speakers + 1e-9)

        snr_db = np.random.uniform(*snr_range)
        return self._mix_at_snr(waveform, babble, snr_db)

    def _load_audio(self, filepath: Path) -> Tuple[torch.Tensor, int]:
        """Load audio on demand (no caching to avoid OOM in DataLoader workers)."""
        try:
            waveform, sr = torchaudio.load(str(filepath))
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            waveform = waveform.squeeze(0)
            return waveform, self.sample_rate
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return torch.zeros(self.sample_rate), self.sample_rate

    def _mix_at_snr(self, signal: torch.Tensor, noise: torch.Tensor,
                    snr_db: float) -> torch.Tensor:
        """Mix signal and noise at target SNR."""
        signal_power = signal.pow(2).mean()
        noise_power = noise.pow(2).mean()

        if noise_power < 1e-9:
            return signal

        snr_linear = 10 ** (snr_db / 10)
        noise_scale = (signal_power / (snr_linear * noise_power)).sqrt()
        return signal + noise * noise_scale


class RIRAugmenter(nn.Module):
    """Room Impulse Response reverb augmentation."""

    def __init__(self, rir_dir: Optional[str] = None, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.rir_files = []

        if rir_dir:
            self._load_rir_index(rir_dir)

    def _load_rir_index(self, rir_dir: str) -> None:
        """Index all RIR .wav files."""
        rir_path = Path(rir_dir)
        self.rir_files = sorted(rir_path.rglob('*.wav'))
        logger.info(f"Indexed {len(self.rir_files)} RIR files")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply RIR convolution (reverb).

        Args:
            waveform: [samples]

        Returns:
            convolved: [samples + rir_len - 1]
        """
        if not self.rir_files:
            return waveform

        rir_file = np.random.choice(self.rir_files)
        try:
            rir, sr = torchaudio.load(str(rir_file))
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                rir = resampler(rir)
            rir = rir.squeeze(0)

            # Normalize RIR
            rir = rir / (rir.abs().max() + 1e-9)

            # FFT convolution
            waveform_np = waveform.numpy()
            rir_np = rir.numpy()
            convolved = np.convolve(waveform_np, rir_np, mode='full')
            convolved = torch.from_numpy(convolved).float()

            # Trim to original length + small margin
            if convolved.shape[0] > waveform.shape[0]:
                convolved = convolved[:waveform.shape[0]]

            return convolved
        except Exception as e:
            logger.warning(f"RIR convolution failed: {e}")
            return waveform


class AugmentationPipeline(nn.Module):
    """Chain augmentations with configurable probabilities."""

    def __init__(self, musan_dir: Optional[str] = None,
                 rir_dir: Optional[str] = None,
                 sample_rate: int = 16000):
        super().__init__()
        self.speed_perturb_prob = 0.5
        self.rir_prob = 0.3
        self.musan_prob = 0.5

        self.musan = MUSANAugmenter(musan_dir, sample_rate) if musan_dir else None
        self.rir = RIRAugmenter(rir_dir, sample_rate) if rir_dir else None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply augmentation pipeline.

        Args:
            waveform: [samples]

        Returns:
            augmented: [samples]
        """
        # Speed perturbation (50% chance: 0.9x, 1.0x, 1.1x)
        if np.random.random() < self.speed_perturb_prob:
            speed = np.random.choice([0.9, 1.0, 1.1])
            if speed != 1.0:
                try:
                    effects = [['speed', str(speed)], ['rate', str(16000)]]
                    waveform_unsq = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
                    waveform_aug, _ = torchaudio.sox_effects.apply_effects_tensor(
                        waveform_unsq, 16000, effects
                    )
                    waveform = waveform_aug.squeeze(0)
                except Exception:
                    pass

        # RIR reverb (30% chance)
        if self.rir and np.random.random() < self.rir_prob:
            waveform = self.rir(waveform)

        # MUSAN augmentation (50% chance)
        if self.musan and np.random.random() < self.musan_prob:
            waveform = self.musan(waveform)

        return waveform


# ============================================================================
# ECAPA-TDNN Model Components
# ============================================================================

class BatchNorm1d(nn.Module):
    """Wrapper around nn.BatchNorm1d with consistent initialization."""

    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""

    def __init__(self, channels: int, se_channels: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(channels, se_channels)
        self.fc2 = nn.Linear(se_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE block.

        Args:
            x: [batch, channels, time]

        Returns:
            scaled: [batch, channels, time]
        """
        # Global average pooling: [batch, channels, time] -> [batch, channels]
        se = x.mean(dim=2)
        # Excitation: [batch, channels] -> [batch, se_channels] -> [batch, channels]
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        # Scale: [batch, channels, 1] * [batch, channels, time]
        return x * se.unsqueeze(2)


class Res2NetBlock(nn.Module):
    """Res2Net block with multi-branch dilated convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, scale: int = 8):
        super().__init__()
        self.scale = scale
        self.branch_channels = out_channels // scale

        self.conv1x1_in = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn1 = BatchNorm1d(out_channels)

        # Multi-branch dilated convolutions
        self.branches = nn.ModuleList()
        for i in range(scale - 1):
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(self.branch_channels, self.branch_channels, kernel_size,
                              padding=(kernel_size - 1) // 2 * dilation, dilation=dilation),
                    BatchNorm1d(self.branch_channels),
                    nn.ReLU(inplace=True)
                )
            )

        self.conv1x1_out = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn2 = BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Res2Net block with residual connection.

        Args:
            x: [batch, channels, time]

        Returns:
            output: [batch, out_channels, time]
        """
        residual = x

        # 1x1 conv to expand channels
        x = self.conv1x1_in(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Split into branches
        xs = torch.chunk(x, self.scale, dim=1)
        ys = [xs[0]]  # First branch is identity

        # Process remaining branches with sequential connections
        for i, branch in enumerate(self.branches):
            y = branch(xs[i + 1] + ys[-1])
            ys.append(y)

        # Concatenate all branches
        x = torch.cat(ys, dim=1)

        # 1x1 conv to project back
        x = self.conv1x1_out(x)
        x = self.bn2(x)

        # Add residual (need to match channels)
        if residual.shape[1] != x.shape[1]:
            residual = F.pad(residual, (0, 0, 0, x.shape[1] - residual.shape[1]))

        return F.relu(x + residual)


class SERes2NetBlock(nn.Module):
    """SE-Res2Net block combining Res2Net with squeeze-and-excitation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, scale: int = 8, se_channels: int = 128):
        super().__init__()
        self.res2net = Res2NetBlock(in_channels, out_channels, kernel_size,
                                     dilation, scale)
        self.se = SEBlock(out_channels, se_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE-Res2Net block.

        Args:
            x: [batch, in_channels, time]

        Returns:
            output: [batch, out_channels, time]
        """
        x = self.res2net(x)
        x = self.se(x)
        return x


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling with channel-wise attention."""

    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(attention_channels, channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attentive statistics pooling.

        Args:
            x: [batch, channels, time]

        Returns:
            pooled: [batch, 2*channels]
        """
        # Compute attention weights
        attention = torch.softmax(self.attention(x), dim=2)

        # Weighted mean
        weighted_x = x * attention
        mean = weighted_x.sum(dim=2)  # [batch, channels]

        # Weighted standard deviation
        weighted_var = ((x ** 2) * attention).sum(dim=2) - mean ** 2
        std = torch.sqrt(weighted_var.clamp(min=1e-9))  # [batch, channels]

        # Concatenate mean and std
        pooled = torch.cat([mean, std], dim=1)  # [batch, 2*channels]
        return pooled


class EcapaTdnn(nn.Module):
    """ECAPA-TDNN speaker encoder with multi-branch architecture."""

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

        self.embedding_dim = embedding_dim

        # Input convolution
        self.input_conv = nn.Conv1d(n_mels, channels[0], kernel_size=5,
                                     padding=2, stride=1)
        self.input_bn = BatchNorm1d(channels[0])

        # SE-Res2Net blocks (N-1 blocks for N channel specs)
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(channels[i], channels[i + 1], kernel_sizes[i + 1],
                              dilations[i + 1], scale=scale, se_channels=se_channels)
            )

        # Multi-layer feature aggregation
        # features = [input_conv_out(channels[0])] + [block_i_out for i in range(N-1)]
        # = channels[0] + channels[1] + ... + channels[-1] = sum(channels)
        total_channels = sum(channels)

        # Conv after aggregation
        self.mfa_conv = nn.Conv1d(total_channels, channels[-1], kernel_size=1)
        self.mfa_bn = BatchNorm1d(channels[-1])

        # Attentive statistics pooling
        self.asp = AttentiveStatisticsPooling(channels[-1],
                                               attention_channels=attention_channels)

        # Final linear layer to embedding
        self.final_linear = nn.Linear(channels[-1] * 2, embedding_dim)
        self.final_bn = BatchNorm1d(embedding_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from mel spectrogram.

        Args:
            mel: [batch, n_mels, time]

        Returns:
            embedding: [batch, embedding_dim] (L2 normalized)
        """
        # Input processing
        x = self.input_conv(mel)
        x = self.input_bn(x)
        x = F.relu(x)

        # Collect features from all blocks (multi-frame aggregation)
        features = [x]
        for block in self.blocks:
            x = block(x)
            features.append(x)

        # Concatenate all features
        cat = torch.cat(features, dim=1)

        # Multi-frame aggregation convolution
        x = self.mfa_conv(cat)
        x = self.mfa_bn(x)
        x = F.relu(x)

        # Attentive pooling
        x = self.asp(x)

        # Final linear projection
        x = self.final_linear(x)
        x = self.final_bn(x)

        # L2 normalization
        return F.normalize(x, p=2, dim=1)


# ============================================================================
# Loss Function
# ============================================================================

class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax (ArcFace) for speaker verification.

    Supports optional sub-centers per speaker class (K > 1) for richer decision boundaries.

    Unlike GE2E, this loss:
    - Has no learnable scale/bias that can drift to zero
    - Doesn't need speaker-balanced batches
    - Angular margin prevents embedding collapse
    - Sub-center variant improves convergence (K=2 recommended)
    """

    def __init__(self, embedding_dim: int, num_speakers: int,
                 scale: float = 30.0, margin: float = 0.2, sub_centers: int = 1):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.sub_centers = sub_centers
        self.num_speakers = num_speakers

        # Weight matrix: [num_speakers * sub_centers, embedding_dim]
        self.weight = nn.Parameter(torch.FloatTensor(num_speakers * sub_centers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute trigonometric values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold to prevent cos(theta+m) from being negative when theta > pi-m
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute AAM-Softmax loss with optional sub-centers.

        Args:
            embeddings: [batch, embedding_dim]
            labels: [batch]

        Returns:
            loss: scalar
        """
        # L2 normalize both embeddings and weight columns
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: [batch, num_speakers * sub_centers]
        cosine = F.linear(embeddings, weight)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Compute cos(theta + m) using trigonometric identity
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        # When cos(theta) < cos(pi - m), use cosine - mm as fallback
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # If using sub-centers, take max cosine over all centers per speaker
        if self.sub_centers > 1:
            # Reshape: [batch, num_speakers, sub_centers]
            cosine_reshaped = cosine.reshape(-1, self.num_speakers, self.sub_centers)
            phi_reshaped = phi.reshape(-1, self.num_speakers, self.sub_centers)

            # Max over sub-centers: [batch, num_speakers]
            cosine_max, max_indices = cosine_reshaped.max(dim=2)
            phi_max = phi_reshaped.gather(2, max_indices.unsqueeze(2)).squeeze(2)

            cosine = cosine_max
            phi = phi_max

        # Only apply margin to target class
        one_hot = F.one_hot(labels, num_classes=self.num_speakers).float()
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = self.scale * logits

        return F.cross_entropy(logits, labels)


# ============================================================================
# Dataset
# ============================================================================

class LibriTTSRDataset(torch.utils.data.Dataset):
    """LibriTTS-R dataset for speaker encoder training."""

    SUBSET_DIRS = {
        'train': ['train-clean-100', 'train-clean-360', 'train-other-500'],
        'test': ['test-clean', 'test-other'],
    }

    def __init__(self, data_dir: str, sample_rate: int = 16000,
                 segment_duration: float = 3.0, mode: str = 'train',
                 augment: bool = False, musan_dir: Optional[str] = None,
                 rir_dir: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_duration * sample_rate)
        self.mode = mode
        self.augment = augment
        self.files = []
        self.speaker_ids = {}

        # Initialize augmentation pipeline
        if augment:
            self.aug_pipeline = AugmentationPipeline(
                musan_dir=musan_dir,
                rir_dir=rir_dir,
                sample_rate=sample_rate
            )
        else:
            self.aug_pipeline = None

        self._build_index()

    def _build_index(self):
        """Build index of all audio files and speaker IDs."""
        subset_dirs = self.SUBSET_DIRS.get(self.mode, self.SUBSET_DIRS['train'])
        found_any = False

        for subset_name in subset_dirs:
            audio_dir = self.data_dir / subset_name
            if not audio_dir.exists():
                logger.warning(f"Dataset directory not found: {audio_dir}")
                continue

            found_any = True
            for wav_file in sorted(audio_dir.rglob('*.wav')):
                parts = wav_file.relative_to(audio_dir).parts
                if len(parts) >= 2:
                    speaker_id = parts[0]
                    if speaker_id not in self.speaker_ids:
                        self.speaker_ids[speaker_id] = len(self.speaker_ids)
                    self.files.append((wav_file, speaker_id))

        if not found_any:
            logger.error(f"No dataset directories found in {self.data_dir} for mode '{self.mode}'")

        logger.info(f"[{self.mode}] Indexed {len(self.files)} files from {len(self.speaker_ids)} speakers")

    @property
    def num_speakers(self) -> int:
        """Return number of unique speakers."""
        return len(self.speaker_ids)

    def _speed_perturb(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random speed perturbation (0.9x, 1.0x, or 1.1x)."""
        speed = np.random.choice([0.9, 1.0, 1.1])
        if speed == 1.0:
            return waveform

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        effects = [['speed', str(speed)], ['rate', str(self.sample_rate)]]
        try:
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, self.sample_rate, effects
            )
            return augmented.squeeze(0)
        except Exception:
            return waveform.squeeze(0)

    def _add_noise(self, waveform: torch.Tensor,
                   snr_range: Tuple[float, float] = (15.0, 40.0)) -> torch.Tensor:
        """Add Gaussian noise at random SNR."""
        if np.random.random() > 0.2:  # Only 20% of the time
            return waveform

        snr_db = np.random.uniform(*snr_range)
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        return waveform + noise

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load audio and return (waveform, speaker_label).

        Returns:
            waveform: [segment_samples]
            speaker_label: integer in [0, num_speakers)
        """
        wav_path, speaker_id = self.files[idx]
        spk_label = self.speaker_ids[speaker_id]

        try:
            waveform, sr = torchaudio.load(str(wav_path))

            # Resample if needed
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)

            # Apply augmentation pipeline (training only)
            if self.augment:
                if self.aug_pipeline is not None:
                    # Use new augmentation pipeline if available
                    waveform = self.aug_pipeline(waveform)
                else:
                    # Fallback to old methods if pipeline not initialized
                    waveform = self._speed_perturb(waveform)
                    waveform = self._add_noise(waveform)

            # Extract fixed-length segment
            if waveform.shape[0] > self.segment_samples:
                start = np.random.randint(0, waveform.shape[0] - self.segment_samples)
                waveform = waveform[start:start + self.segment_samples]
            else:
                waveform = F.pad(waveform, (0, self.segment_samples - waveform.shape[0]))

            return waveform, spk_label

        except Exception as e:
            logger.warning(f"Error loading {wav_path}: {e}")
            return torch.zeros(self.segment_samples), spk_label


# ============================================================================
# Evaluation
# ============================================================================

def compute_eer(embeddings: np.ndarray, labels: np.ndarray,
                n_pairs: int = 50000) -> float:
    """Compute Equal Error Rate from embeddings and speaker labels.

    Args:
        embeddings: [num_utterances, embedding_dim]
        labels: [num_utterances]
        n_pairs: number of pairs to sample

    Returns:
        eer: Equal Error Rate in [0, 1]
    """
    unique_labels = np.unique(labels)
    label_to_indices = {l: np.where(labels == l)[0] for l in unique_labels}

    # Filter speakers with at least 2 utterances for positive pairs
    valid_speakers = [l for l in unique_labels if len(label_to_indices[l]) >= 2]

    if len(valid_speakers) < 2:
        logger.warning("Not enough speakers for EER computation")
        return 1.0

    scores, targets = [], []

    for _ in range(n_pairs):
        # Positive pair (same speaker)
        spk = np.random.choice(valid_speakers)
        i, j = np.random.choice(label_to_indices[spk], 2, replace=False)
        scores.append(float(np.dot(embeddings[i], embeddings[j])))
        targets.append(1)

        # Negative pair (different speakers)
        spk1, spk2 = np.random.choice(unique_labels, 2, replace=False)
        i = np.random.choice(label_to_indices[spk1])
        j = np.random.choice(label_to_indices[spk2])
        scores.append(float(np.dot(embeddings[i], embeddings[j])))
        targets.append(0)

    scores = np.array(scores)
    targets = np.array(targets)

    # Sweep thresholds to find EER
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    min_diff = float('inf')
    eer = 1.0

    for t in thresholds:
        far = np.mean(scores[targets == 0] >= t)   # False Accept Rate
        frr = np.mean(scores[targets == 1] < t)    # False Reject Rate
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0

    return eer


def compute_asnorm_eer(embeddings: np.ndarray, labels: np.ndarray,
                       cohort_size: int = 300, n_pairs: int = 50000) -> float:
    """Compute EER with Adaptive Score Normalization (AS-Norm).

    AS-Norm normalizes scores using a cohort of similar speakers:
    s_norm = (s - mean_cohort) / std_cohort

    Args:
        embeddings: [num_utterances, embedding_dim]
        labels: [num_utterances]
        cohort_size: number of cohort speakers
        n_pairs: number of pairs to sample

    Returns:
        eer: Equal Error Rate with AS-Norm in [0, 1]
    """
    unique_labels = np.unique(labels)
    label_to_indices = {l: np.where(labels == l)[0] for l in unique_labels}

    # Filter speakers with at least 2 utterances
    valid_speakers = [l for l in unique_labels if len(label_to_indices[l]) >= 2]

    if len(valid_speakers) < 2:
        logger.warning("Not enough speakers for AS-Norm EER computation")
        return 1.0

    # Build cohort from random sample of training speakers
    cohort_sample_size = min(cohort_size, len(embeddings) // 3)
    cohort_indices = np.random.choice(len(embeddings), cohort_sample_size, replace=False)
    cohort_embeddings = embeddings[cohort_indices]

    scores, targets = [], []

    for _ in range(n_pairs):
        # Positive pair (same speaker)
        spk = np.random.choice(valid_speakers)
        i, j = np.random.choice(label_to_indices[spk], 2, replace=False)
        score_ij = float(np.dot(embeddings[i], embeddings[j]))

        # Symmetric AS-Norm: normalize by both cohorts
        cohort_i = np.dot(embeddings[i], cohort_embeddings.T)
        cohort_j = np.dot(embeddings[j], cohort_embeddings.T)
        mean_cohort = (cohort_i.mean() + cohort_j.mean()) / 2
        std_cohort = (cohort_i.std() + cohort_j.std()) / 2 + 1e-9
        score_norm = (score_ij - mean_cohort) / std_cohort

        scores.append(float(score_norm))
        targets.append(1)

        # Negative pair (different speakers)
        spk1, spk2 = np.random.choice(unique_labels, 2, replace=False)
        i = np.random.choice(label_to_indices[spk1])
        j = np.random.choice(label_to_indices[spk2])
        score_ij = float(np.dot(embeddings[i], embeddings[j]))

        # Symmetric AS-Norm
        cohort_i = np.dot(embeddings[i], cohort_embeddings.T)
        cohort_j = np.dot(embeddings[j], cohort_embeddings.T)
        mean_cohort = (cohort_i.mean() + cohort_j.mean()) / 2
        std_cohort = (cohort_i.std() + cohort_j.std()) / 2 + 1e-9
        score_norm = (score_ij - mean_cohort) / std_cohort

        scores.append(float(score_norm))
        targets.append(0)

    scores = np.array(scores)
    targets = np.array(targets)

    # Sweep thresholds to find EER
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    min_diff = float('inf')
    eer = 1.0

    for t in thresholds:
        far = np.mean(scores[targets == 0] >= t)
        frr = np.mean(scores[targets == 1] < t)
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0

    return eer


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, mel_extractor, spec_augment, loader, criterion,
                optimizer, device, max_grad_norm: float = 5.0) -> float:
    """Train for one epoch.

    Args:
        model: ECAPA-TDNN encoder
        mel_extractor: mel spectrogram extractor
        spec_augment: SpecAugment module
        loader: training data loader
        criterion: AAMSoftmax loss
        optimizer: Adam optimizer
        device: torch device
        max_grad_norm: gradient clipping threshold

    Returns:
        average_loss: average loss over epoch
    """
    model.train()
    mel_extractor.eval()
    total_loss = 0.0
    n_batches = 0

    with tqdm(loader, desc='Training') as pbar:
        for waveforms, speaker_ids in pbar:
            waveforms = waveforms.to(device)
            speaker_ids = speaker_ids.to(device)

            optimizer.zero_grad()

            # Extract mel spectrogram
            mel = mel_extractor(waveforms)

            # Apply SpecAugment
            if spec_augment is not None:
                mel = spec_augment(mel)

            # Forward pass
            embeddings = model(mel)
            loss = criterion(embeddings, speaker_ids)

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.parameters()),
                max_grad_norm
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'grad': f'{grad_norm:.2f}'
            })

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_eer(model, mel_extractor, loader, device, use_asnorm: bool = False,
                 asnorm_cohort_size: int = 300) -> Tuple[float, Optional[float]]:
    """Compute EER on validation set with optional AS-Norm.

    Args:
        model: ECAPA-TDNN encoder
        mel_extractor: mel spectrogram extractor
        loader: test data loader
        device: torch device
        use_asnorm: whether to compute AS-Norm EER as well
        asnorm_cohort_size: cohort size for AS-Norm

    Returns:
        (eer, asnorm_eer): standard EER and optional AS-Norm EER
    """
    model.eval()
    mel_extractor.eval()
    all_embeddings = []
    all_labels = []

    for waveforms, speaker_ids in tqdm(loader, desc='Extracting embeddings'):
        waveforms = waveforms.to(device)
        mel = mel_extractor(waveforms)
        embeddings = model(mel)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(speaker_ids.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    eer = compute_eer(embeddings, labels)
    asnorm_eer = None

    if use_asnorm:
        asnorm_eer = compute_asnorm_eer(embeddings, labels, cohort_size=asnorm_cohort_size)

    return eer, asnorm_eer


# ============================================================================
# Export Functions
# ============================================================================

def export_to_safetensors(model, config, output_path: str) -> None:
    """Export model weights to safetensors format (for Rust inference).

    Args:
        model: ECAPA-TDNN model
        config: configuration dictionary
        output_path: path to save .safetensors file
    """
    state_dict = model.state_dict()
    state_dict = {
        k: v.cpu().float().contiguous()
        for k, v in state_dict.items()
    }
    save_file(state_dict, output_path)
    logger.info(f"Exported model to {output_path}")

    # Save config
    config_path = output_path.replace('.safetensors', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Exported config to {config_path}")


def export_to_onnx(model, output_path: str, n_mels: int = 80) -> None:
    """Export encoder to ONNX (for C inference).

    Args:
        model: ECAPA-TDNN model
        output_path: path to save .onnx file
        n_mels: number of mel bins
    """
    model.eval()
    model_cpu = model.cpu()

    # Dummy input: [batch=1, n_mels=80, time=100]
    dummy_input = torch.randn(1, n_mels, 100)

    torch.onnx.export(
        model_cpu, dummy_input, output_path,
        input_names=['fbank'],
        output_names=['embedding'],
        dynamic_axes={'fbank': {2: 'time'}},
        opset_version=17,
    )
    logger.info(f"Exported ONNX to {output_path}")

    # Validate ONNX
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        test_input = np.random.randn(1, n_mels, 100).astype(np.float32)
        result = sess.run(None, {'fbank': test_input})
        logger.info(f"ONNX validation: output shape {result[0].shape}")
    except Exception as e:
        logger.warning(f"ONNX validation failed: {e}")


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train ECAPA-TDNN speaker encoder (AAM-Softmax)'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to LibriTTS_R root')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Checkpoint output directory')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--scale', type=float, default=30.0,
                        help='AAM-Softmax scale factor')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='AAM-Softmax angular margin')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='Linear LR warmup epochs')
    parser.add_argument('--val-every', type=int, default=2,
                        help='Validate EER every N epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default='',
                        help='Checkpoint path to resume from')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers')

    # New augmentation args
    parser.add_argument('--musan-dir', type=str, default='',
                        help='Path to MUSAN dataset (enables MUSAN augmentation)')
    parser.add_argument('--rir-dir', type=str, default='',
                        help='Path to RIR dataset (enables reverb augmentation)')
    parser.add_argument('--crop-duration', type=float, default=3.0,
                        help='Audio crop length in seconds')

    # New sub-center ArcFace arg
    parser.add_argument('--sub-centers', type=int, default=1,
                        help='Sub-center count for ArcFace (default 1 = standard AAM-Softmax)')

    # New fine-tuning args
    parser.add_argument('--fine-tune', action='store_true',
                        help='Enable large-margin fine-tuning mode')
    parser.add_argument('--fine-tune-margin', type=float, default=0.5,
                        help='Margin for fine-tuning')
    parser.add_argument('--fine-tune-epochs', type=int, default=5,
                        help='Epochs for fine-tuning')
    parser.add_argument('--fine-tune-crop', type=float, default=6.0,
                        help='Crop duration for fine-tuning in seconds')

    # New AS-Norm arg
    parser.add_argument('--asnorm', action='store_true',
                        help='Enable AS-Norm for validation EER')
    parser.add_argument('--asnorm-cohort-size', type=int, default=300,
                        help='Cohort size for AS-Norm')

    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Config: scale={args.scale}, margin={args.margin}, "
                f"lr={args.lr}, bs={args.batch_size}, crop={args.crop_duration}s")
    if args.sub_centers > 1:
        logger.info(f"Sub-center ArcFace: K={args.sub_centers}")
    if args.musan_dir:
        logger.info(f"MUSAN augmentation: {args.musan_dir}")
    if args.rir_dir:
        logger.info(f"RIR augmentation: {args.rir_dir}")
    if args.asnorm:
        logger.info(f"AS-Norm validation: cohort_size={args.asnorm_cohort_size}")

    # ========================================================================
    # Handle Fine-tuning Mode
    # ========================================================================
    if args.fine_tune:
        if not args.resume:
            logger.error("--fine-tune requires --resume checkpoint!")
            sys.exit(1)
        args.n_epochs = args.fine_tune_epochs
        args.margin = args.fine_tune_margin
        args.lr = args.lr * 0.1  # Reduce learning rate for fine-tuning
        args.crop_duration = args.fine_tune_crop
        logger.info(f"Fine-tuning mode: margin={args.margin}, lr={args.lr}, "
                   f"crop={args.crop_duration}s, epochs={args.n_epochs}")

    # ========================================================================
    # Load Data
    # ========================================================================
    logger.info("Loading LibriTTS-R dataset...")
    musan_dir = args.musan_dir if args.musan_dir else None
    rir_dir = args.rir_dir if args.rir_dir else None

    train_dataset = LibriTTSRDataset(
        args.data_dir, mode='train', augment=True,
        segment_duration=args.crop_duration,
        musan_dir=musan_dir, rir_dir=rir_dir
    )
    if len(train_dataset) == 0:
        logger.error("No training data found!")
        sys.exit(1)

    test_dataset = LibriTTSRDataset(
        args.data_dir, mode='test', augment=False,
        segment_duration=args.crop_duration
    )
    has_validation = len(test_dataset) > 0
    if not has_validation:
        logger.warning("No test data found — training without EER validation!")
        logger.warning("Download test-clean from: https://www.openslr.org/141/")

    num_speakers = train_dataset.num_speakers
    logger.info(f"Training speakers: {num_speakers}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    test_loader = None
    if has_validation:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )

    # ========================================================================
    # Create Model
    # ========================================================================
    logger.info("Creating ECAPA-TDNN model...")
    model_config = {
        'n_mels': 80,
        'embedding_dim': 256,
        'channels': [1024, 1024, 1024, 1024, 1536],
        'kernel_sizes': [5, 3, 3, 3, 1],
        'dilations': [1, 2, 3, 4, 1],
        'scale': 8,
        'se_channels': 128,
        'attention_channels': 128,
        'sample_rate': 16000,
    }

    model = EcapaTdnn(
        n_mels=model_config['n_mels'],
        embedding_dim=model_config['embedding_dim'],
        channels=model_config['channels'],
        kernel_sizes=model_config['kernel_sizes'],
        dilations=model_config['dilations'],
        scale=model_config['scale'],
        se_channels=model_config['se_channels'],
        attention_channels=model_config['attention_channels']
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e6:.1f}M")

    mel_extractor = MelSpectrogramExtractor().to(device)
    spec_augment = SpecAugment().to(device)
    classifier = AAMSoftmax(
        model_config['embedding_dim'],
        num_speakers,
        scale=args.scale,
        margin=args.margin,
        sub_centers=args.sub_centers
    ).to(device)

    # ========================================================================
    # Optimizer and Scheduler
    # ========================================================================
    optimizer = Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs - args.warmup_epochs)

    start_epoch = 0
    best_eer = 1.0
    patience_counter = 0

    # ========================================================================
    # Resume from Checkpoint
    # ========================================================================
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if 'classifier' in ckpt:
            classifier.load_state_dict(ckpt['classifier'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0)
        best_eer = ckpt.get('best_eer', 1.0)
        logger.info(f"Resumed at epoch {start_epoch}, best_eer={best_eer:.4f}")

    # ========================================================================
    # Training Loop
    # ========================================================================
    for epoch in range(start_epoch, args.n_epochs):
        epoch_start = time.time()

        # Linear warmup
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
            current_lr = warmup_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"\nEpoch {epoch + 1}/{args.n_epochs} | LR: {current_lr:.6f}")

        # Train
        train_loss = train_epoch(
            model, mel_extractor, spec_augment, train_loader,
            classifier, optimizer, device
        )

        epoch_time = time.time() - epoch_start
        logger.info(f"Train loss: {train_loss:.6f} | Time: {epoch_time:.1f}s")

        # Step scheduler after warmup
        if epoch >= args.warmup_epochs:
            scheduler.step()

        # ====================================================================
        # Validation
        # ====================================================================
        if has_validation and (epoch + 1) % args.val_every == 0:
            eer, asnorm_eer = validate_eer(
                model, mel_extractor, test_loader, device,
                use_asnorm=args.asnorm,
                asnorm_cohort_size=args.asnorm_cohort_size
            )

            if args.asnorm and asnorm_eer is not None:
                logger.info(f"Validation EER: {eer:.4f} ({eer*100:.2f}%) | "
                           f"AS-Norm EER: {asnorm_eer:.4f} ({asnorm_eer*100:.2f}%) | "
                           f"Best: {best_eer:.4f}")
                # Use AS-Norm EER for early stopping
                eval_eer = asnorm_eer
            else:
                logger.info(f"Validation EER: {eer:.4f} ({eer*100:.2f}%) | "
                           f"Best: {best_eer:.4f}")
                eval_eer = eer

            if eval_eer < best_eer:
                best_eer = eval_eer
                patience_counter = 0

                checkpoint_path = output_dir / "speaker_encoder_best.pt"
                torch.save({
                    'model': model.state_dict(),
                    'classifier': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'best_eer': best_eer,
                    'num_speakers': num_speakers,
                    'config': model_config,
                }, checkpoint_path)
                logger.info(f"New best! Saved to {checkpoint_path}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{args.patience}")

                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Save periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            periodic_path = output_dir / f"speaker_encoder_epoch{epoch + 1}.pt"
            torch.save({
                'model': model.state_dict(),
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_eer': best_eer,
                'num_speakers': num_speakers,
                'config': model_config,
            }, periodic_path)
            logger.info(f"Periodic checkpoint: {periodic_path}")

    # ========================================================================
    # Export
    # ========================================================================
    logger.info("\nExporting final model...")
    export_to_safetensors(
        model,
        model_config,
        str(output_dir / 'speaker_encoder.safetensors')
    )
    export_to_onnx(model, str(output_dir / 'speaker_encoder.onnx'))

    logger.info(f"\nTraining complete! Best EER: {best_eer:.4f} ({best_eer*100:.2f}%)")


if __name__ == '__main__':
    main()
