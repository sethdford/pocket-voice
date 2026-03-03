#!/usr/bin/env python3
"""
train_speaker_encoder.py — Train CAM++ speaker encoder with AAM-Softmax loss.

Architecture matches sonata-cam Rust crate:
- MultiScaleFrontend: 3 Conv1d layers (80→64→128→256, kernel=3, padding=1)
- 6 CAMBlocks: attention masking + Conv1d FFN (dim=256, 8 heads)
- AttentiveStatsPooling: weighted mean+std → Linear(512→192)
- Output: 192-dim L2-normalized speaker embedding

Loss: Additive Angular Margin Softmax (ArcFace)
Data: LibriTTS-R with MUSAN/RIR augmentation + SpecAugment + speed perturbation
Validation: Equal Error Rate (EER) on test-clean with optional AS-Norm

Usage:
    # Standard training
    python3 train_speaker_encoder.py \\
      --data-dir /path/to/LibriTTS_R \\
      --output-dir ./checkpoints/speaker_encoder_v2 \\
      --batch-size 64 --n-epochs 40 --lr 0.001

    # With MUSAN/RIR augmentation
    python3 train_speaker_encoder.py \\
      --data-dir /path/to/LibriTTS_R \\
      --musan-dir /path/to/MUSAN \\
      --rir-dir /path/to/RIRS_NOISES

    # Fine-tuning with large margin
    python3 train_speaker_encoder.py \\
      --data-dir /path/to/LibriTTS_R \\
      --fine-tune --resume ./checkpoints/best.pt \\
      --fine-tune-margin 0.5 --fine-tune-epochs 5

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

# Constants matching sonata-common
CAM_DIM = 256
CAM_HEADS = 8
CAM_BLOCKS = 6
SPEAKER_EMBED_DIM = 192
SPEAKER_SAMPLE_RATE = 16000
SPEAKER_MEL_BINS = 80


# ============================================================================
# Feature Extraction
# ============================================================================

class MelSpectrogramExtractor(nn.Module):
    """Extract 80-bin mel spectrograms for speaker encoding."""

    def __init__(self, sample_rate: int = SPEAKER_SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=SPEAKER_MEL_BINS,
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
        for _ in range(self.n_freq_masks):
            mel = self.freq_mask(mel)
        for _ in range(self.n_time_masks):
            mel = self.time_mask(mel)
        return mel


class MUSANAugmenter(nn.Module):
    """MUSAN dataset augmentation (noise, music, speech babble)."""

    def __init__(self, musan_dir: Optional[str] = None, sample_rate: int = SPEAKER_SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self.musan_files = {'noise': [], 'music': [], 'speech': []}
        if musan_dir:
            self._load_musan_index(musan_dir)

    def _load_musan_index(self, musan_dir: str) -> None:
        musan_path = Path(musan_dir)
        for category in ['noise', 'music', 'speech']:
            cat_dir = musan_path / category
            if cat_dir.exists():
                files = sorted(cat_dir.rglob('*.wav'))
                self.musan_files[category] = files
                logger.info(f"Indexed {len(files)} MUSAN {category} files")

    def forward(self, waveform: torch.Tensor, category: str = None) -> torch.Tensor:
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
        return waveform

    def _add_noise(self, waveform, snr_range):
        if not self.musan_files['noise']:
            return waveform
        noise_file = np.random.choice(self.musan_files['noise'])
        noise, sr = self._load_audio(noise_file)
        if noise.shape[0] < waveform.shape[0]:
            noise = torch.tile(noise, (waveform.shape[0] // noise.shape[0] + 1,))
        noise = noise[:waveform.shape[0]]
        snr_db = np.random.uniform(*snr_range)
        return self._mix_at_snr(waveform, noise, snr_db)

    def _add_music(self, waveform, snr_range):
        if not self.musan_files['music']:
            return waveform
        music_file = np.random.choice(self.musan_files['music'])
        music, sr = self._load_audio(music_file)
        if music.shape[0] > waveform.shape[0]:
            start = np.random.randint(0, music.shape[0] - waveform.shape[0])
            music = music[start:start + waveform.shape[0]]
        else:
            music = torch.tile(music, (waveform.shape[0] // music.shape[0] + 1,))
            music = music[:waveform.shape[0]]
        snr_db = np.random.uniform(*snr_range)
        return self._mix_at_snr(waveform, music, snr_db)

    def _add_babble(self, waveform, num_speakers, snr_range):
        if not self.musan_files['speech']:
            return waveform
        n_speakers = np.random.randint(*num_speakers)
        babble = torch.zeros_like(waveform)
        for _ in range(n_speakers):
            speech_file = np.random.choice(self.musan_files['speech'])
            speech, sr = self._load_audio(speech_file)
            if speech.shape[0] > waveform.shape[0]:
                start = np.random.randint(0, speech.shape[0] - waveform.shape[0])
                speech = speech[start:start + waveform.shape[0]]
            else:
                speech = torch.tile(speech, (waveform.shape[0] // speech.shape[0] + 1,))
                speech = speech[:waveform.shape[0]]
            babble = babble + speech
        babble = babble / (n_speakers + 1e-9)
        snr_db = np.random.uniform(*snr_range)
        return self._mix_at_snr(waveform, babble, snr_db)

    def _load_audio(self, filepath):
        try:
            waveform, sr = torchaudio.load(str(filepath))
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            return waveform.squeeze(0), self.sample_rate
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return torch.zeros(self.sample_rate), self.sample_rate

    def _mix_at_snr(self, signal, noise, snr_db):
        signal_power = signal.pow(2).mean()
        noise_power = noise.pow(2).mean()
        if noise_power < 1e-9:
            return signal
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = (signal_power / (snr_linear * noise_power)).sqrt()
        return signal + noise * noise_scale


class RIRAugmenter(nn.Module):
    """Room Impulse Response reverb augmentation."""

    def __init__(self, rir_dir: Optional[str] = None, sample_rate: int = SPEAKER_SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self.rir_files = []
        if rir_dir:
            rir_path = Path(rir_dir)
            self.rir_files = sorted(rir_path.rglob('*.wav'))
            logger.info(f"Indexed {len(self.rir_files)} RIR files")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.rir_files:
            return waveform
        rir_file = np.random.choice(self.rir_files)
        try:
            rir, sr = torchaudio.load(str(rir_file))
            if sr != self.sample_rate:
                rir = T.Resample(sr, self.sample_rate)(rir)
            rir = rir.squeeze(0)
            rir = rir / (rir.abs().max() + 1e-9)
            convolved = np.convolve(waveform.numpy(), rir.numpy(), mode='full')
            convolved = torch.from_numpy(convolved).float()
            if convolved.shape[0] > waveform.shape[0]:
                convolved = convolved[:waveform.shape[0]]
            return convolved
        except Exception as e:
            logger.warning(f"RIR convolution failed: {e}")
            return waveform


class AugmentationPipeline(nn.Module):
    """Chain augmentations with configurable probabilities."""

    def __init__(self, musan_dir=None, rir_dir=None, sample_rate=SPEAKER_SAMPLE_RATE):
        super().__init__()
        self.speed_perturb_prob = 0.5
        self.rir_prob = 0.3
        self.musan_prob = 0.5
        self.musan = MUSANAugmenter(musan_dir, sample_rate) if musan_dir else None
        self.rir = RIRAugmenter(rir_dir, sample_rate) if rir_dir else None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if np.random.random() < self.speed_perturb_prob:
            speed = np.random.choice([0.9, 1.0, 1.1])
            if speed != 1.0:
                try:
                    effects = [['speed', str(speed)], ['rate', str(SPEAKER_SAMPLE_RATE)]]
                    waveform_unsq = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
                    waveform_aug, _ = torchaudio.sox_effects.apply_effects_tensor(
                        waveform_unsq, SPEAKER_SAMPLE_RATE, effects
                    )
                    waveform = waveform_aug.squeeze(0)
                except Exception:
                    pass
        if self.rir and np.random.random() < self.rir_prob:
            waveform = self.rir(waveform)
        if self.musan and np.random.random() < self.musan_prob:
            waveform = self.musan(waveform)
        return waveform


# ============================================================================
# CAM++ Model Components (matches sonata-cam Rust crate)
# ============================================================================

class MultiScaleFrontend(nn.Module):
    """Multi-scale frontend: 3 Conv1d layers (80→64→128→256).

    Matches sonata-cam/src/frontend.rs.
    Key naming: frontend.convs.{i}, frontend.norms.{i}
    """

    def __init__(self, in_channels: int = SPEAKER_MEL_BINS, out_channels: int = CAM_DIM):
        super().__init__()
        channels = [in_channels, 64, 128, out_channels]
        self.convs = nn.ModuleList([
            nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, padding=1)
            for i in range(3)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(channels[i + 1])
            for i in range(3)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process mel spectrogram through multi-scale frontend.

        Args:
            x: [B, 80, T]

        Returns:
            features: [B, 256, T]
        """
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x)))
        return x


class CAMBlock(nn.Module):
    """Context-Aware Masking block for speaker verification.

    Matches sonata-cam/src/cam_block.rs:
    - Attention: Q/K/V/out_proj Linear(dim→dim)
    - Context-aware masking via softmax attention
    - FFN: Sequential Conv1d(dim→4*dim, k=3) → ReLU → Conv1d(4*dim→dim, k=3)
    - 2 LayerNorms

    Key naming: blocks.{i}.q_proj, blocks.{i}.ffn.0, blocks.{i}.norm1

    Args:
        dim: Model dimension (default: 256)
        num_heads: Attention heads (default: 8)
    """

    def __init__(self, dim: int = CAM_DIM, num_heads: int = CAM_HEADS):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Attention with separate Q/K/V projections (matches Rust VarBuilder paths)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Conv1d FFN as Sequential (keys: ffn.0.weight, ffn.2.weight)
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, dim * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim * 4, dim, kernel_size=3, padding=1),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CAM block with residual connection.

        Args:
            x: [B, dim, T]

        Returns:
            output: [B, dim, T]
        """
        residual = x

        # [B, D, T] → [B, T, D] for attention
        x_t = x.permute(0, 2, 1)
        x_t = self.norm1(x_t)

        # Context-aware attention masking
        q = self.q_proj(x_t)
        k = self.k_proj(x_t)
        v = self.v_proj(x_t)

        attn = F.softmax(q * k / (self.dim ** 0.5), dim=-1)
        x_attn = self.out_proj(attn * v)
        x_attn = x_attn.permute(0, 2, 1)  # [B, D, T]

        x = residual + x_attn

        # FFN with residual
        residual = x
        x_t = x.permute(0, 2, 1)
        x_t = self.norm2(x_t)
        x = residual + self.ffn(x_t.permute(0, 2, 1))

        return x


class AttentiveStatsPooling(nn.Module):
    """Attentive statistics pooling matching sonata-cam/src/pooling.rs.

    attention: Linear(in_dim→1), softmax weights
    Weighted mean + weighted std → concat → Linear(in_dim*2→embed_dim)

    Args:
        in_dim: Input channel dimension (default: 256)
        embed_dim: Output embedding dimension (default: 192)
    """

    def __init__(self, in_dim: int = CAM_DIM, embed_dim: int = SPEAKER_EMBED_DIM):
        super().__init__()
        self.attention = nn.Linear(in_dim, 1)
        self.output_proj = nn.Linear(in_dim * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attentive statistics pooling.

        Args:
            x: [B, dim, T]

        Returns:
            embedding: [B, embed_dim]
        """
        # [B, D, T] → [B, T, D]
        x_t = x.permute(0, 2, 1)

        # Attention weights: [B, T, 1] → softmax over T
        attn_weights = F.softmax(self.attention(x_t), dim=1)  # [B, T, 1]

        # Weighted mean: [B, D]
        mean = (x_t * attn_weights).sum(dim=1)

        # Weighted std: [B, D]
        var = ((x_t ** 2) * attn_weights).sum(dim=1) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-9))

        # Concat mean + std → project to embed_dim
        pooled = torch.cat([mean, std], dim=1)  # [B, 2*D]
        return self.output_proj(pooled)  # [B, embed_dim]


class CamPlusPlus(nn.Module):
    """CAM++ speaker encoder matching sonata-cam/src/lib.rs.

    Architecture:
        MultiScaleFrontend(80→256) → 6 CAMBlocks(256, 8 heads) →
        AttentiveStatsPooling(256→192) → L2 normalize

    Args:
        n_mels: Input mel bins (default: 80)
        dim: Model dimension (default: 256)
        num_blocks: Number of CAM blocks (default: 6)
        num_heads: Attention heads (default: 8)
        embed_dim: Output embedding dimension (default: 192)
    """

    def __init__(self, n_mels: int = SPEAKER_MEL_BINS, dim: int = CAM_DIM,
                 num_blocks: int = CAM_BLOCKS, num_heads: int = CAM_HEADS,
                 embed_dim: int = SPEAKER_EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim

        self.frontend = MultiScaleFrontend(n_mels, dim)
        self.blocks = nn.ModuleList([
            CAMBlock(dim, num_heads) for _ in range(num_blocks)
        ])
        self.pooling = AttentiveStatsPooling(dim, embed_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from mel spectrogram.

        Args:
            mel: [B, n_mels, T]

        Returns:
            embedding: [B, embed_dim] (L2 normalized)
        """
        x = self.frontend(mel)  # [B, 256, T]

        for block in self.blocks:
            x = block(x)  # [B, 256, T]

        embedding = self.pooling(x)  # [B, 192]
        return F.normalize(embedding, p=2, dim=1)


# ============================================================================
# Loss Function
# ============================================================================

class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax (ArcFace) for speaker verification.

    Args:
        embedding_dim: Input embedding dimension (192 for CAM++)
        num_speakers: Number of speaker classes
        scale: Logit scale factor
        margin: Angular margin in radians
        sub_centers: Sub-centers per speaker (1 = standard ArcFace)
    """

    def __init__(self, embedding_dim: int, num_speakers: int,
                 scale: float = 30.0, margin: float = 0.2, sub_centers: int = 1):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.sub_centers = sub_centers
        self.num_speakers = num_speakers

        self.weight = nn.Parameter(torch.FloatTensor(num_speakers * sub_centers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weight)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        if self.sub_centers > 1:
            cosine_reshaped = cosine.reshape(-1, self.num_speakers, self.sub_centers)
            phi_reshaped = phi.reshape(-1, self.num_speakers, self.sub_centers)
            cosine_max, max_indices = cosine_reshaped.max(dim=2)
            phi_max = phi_reshaped.gather(2, max_indices.unsqueeze(2)).squeeze(2)
            cosine = cosine_max
            phi = phi_max

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

    def __init__(self, data_dir: str, sample_rate: int = SPEAKER_SAMPLE_RATE,
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

        if augment:
            self.aug_pipeline = AugmentationPipeline(
                musan_dir=musan_dir, rir_dir=rir_dir, sample_rate=sample_rate
            )
        else:
            self.aug_pipeline = None

        self._build_index()

    def _build_index(self):
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
        return len(self.speaker_ids)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        wav_path, speaker_id = self.files[idx]
        spk_label = self.speaker_ids[speaker_id]

        try:
            waveform, sr = torchaudio.load(str(wav_path))
            if sr != self.sample_rate:
                waveform = T.Resample(sr, self.sample_rate)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)

            if self.augment and self.aug_pipeline is not None:
                waveform = self.aug_pipeline(waveform)

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
    """Compute Equal Error Rate from embeddings and speaker labels."""
    unique_labels = np.unique(labels)
    label_to_indices = {l: np.where(labels == l)[0] for l in unique_labels}
    valid_speakers = [l for l in unique_labels if len(label_to_indices[l]) >= 2]

    if len(valid_speakers) < 2:
        logger.warning("Not enough speakers for EER computation")
        return 1.0

    scores, targets = [], []
    for _ in range(n_pairs):
        spk = np.random.choice(valid_speakers)
        i, j = np.random.choice(label_to_indices[spk], 2, replace=False)
        scores.append(float(np.dot(embeddings[i], embeddings[j])))
        targets.append(1)

        spk1, spk2 = np.random.choice(unique_labels, 2, replace=False)
        i = np.random.choice(label_to_indices[spk1])
        j = np.random.choice(label_to_indices[spk2])
        scores.append(float(np.dot(embeddings[i], embeddings[j])))
        targets.append(0)

    scores = np.array(scores)
    targets = np.array(targets)

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


def compute_asnorm_eer(embeddings: np.ndarray, labels: np.ndarray,
                       cohort_size: int = 300, n_pairs: int = 50000) -> float:
    """Compute EER with Adaptive Score Normalization (AS-Norm)."""
    unique_labels = np.unique(labels)
    label_to_indices = {l: np.where(labels == l)[0] for l in unique_labels}
    valid_speakers = [l for l in unique_labels if len(label_to_indices[l]) >= 2]

    if len(valid_speakers) < 2:
        return 1.0

    cohort_sample_size = min(cohort_size, len(embeddings) // 3)
    cohort_indices = np.random.choice(len(embeddings), cohort_sample_size, replace=False)
    cohort_embeddings = embeddings[cohort_indices]

    scores, targets = [], []
    for _ in range(n_pairs):
        spk = np.random.choice(valid_speakers)
        i, j = np.random.choice(label_to_indices[spk], 2, replace=False)
        score_ij = float(np.dot(embeddings[i], embeddings[j]))
        cohort_i = np.dot(embeddings[i], cohort_embeddings.T)
        cohort_j = np.dot(embeddings[j], cohort_embeddings.T)
        mean_cohort = (cohort_i.mean() + cohort_j.mean()) / 2
        std_cohort = (cohort_i.std() + cohort_j.std()) / 2 + 1e-9
        scores.append(float((score_ij - mean_cohort) / std_cohort))
        targets.append(1)

        spk1, spk2 = np.random.choice(unique_labels, 2, replace=False)
        i = np.random.choice(label_to_indices[spk1])
        j = np.random.choice(label_to_indices[spk2])
        score_ij = float(np.dot(embeddings[i], embeddings[j]))
        cohort_i = np.dot(embeddings[i], cohort_embeddings.T)
        cohort_j = np.dot(embeddings[j], cohort_embeddings.T)
        mean_cohort = (cohort_i.mean() + cohort_j.mean()) / 2
        std_cohort = (cohort_i.std() + cohort_j.std()) / 2 + 1e-9
        scores.append(float((score_ij - mean_cohort) / std_cohort))
        targets.append(0)

    scores = np.array(scores)
    targets = np.array(targets)

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
    model.train()
    mel_extractor.eval()
    total_loss = 0.0
    n_batches = 0

    with tqdm(loader, desc='Training') as pbar:
        for waveforms, speaker_ids in pbar:
            waveforms = waveforms.to(device)
            speaker_ids = speaker_ids.to(device)

            optimizer.zero_grad()
            mel = mel_extractor(waveforms)
            if spec_augment is not None:
                mel = spec_augment(mel)

            embeddings = model(mel)
            loss = criterion(embeddings, speaker_ids)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.parameters()),
                max_grad_norm
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'grad': f'{grad_norm:.2f}'})

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_eer(model, mel_extractor, loader, device, use_asnorm=False,
                 asnorm_cohort_size=300):
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
    state_dict = model.state_dict()
    state_dict = {k: v.cpu().float().contiguous() for k, v in state_dict.items()}
    save_file(state_dict, output_path)
    logger.info(f"Exported model to {output_path}")

    config_path = output_path.replace('.safetensors', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Exported config to {config_path}")


def export_to_onnx(model, output_path: str, n_mels: int = SPEAKER_MEL_BINS) -> None:
    model.eval()
    model_cpu = model.cpu()
    dummy_input = torch.randn(1, n_mels, 100)

    torch.onnx.export(
        model_cpu, dummy_input, output_path,
        input_names=['fbank'],
        output_names=['embedding'],
        dynamic_axes={'fbank': {2: 'time'}},
        opset_version=17,
    )
    logger.info(f"Exported ONNX to {output_path}")

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
        description='Train CAM++ speaker encoder (AAM-Softmax)'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to LibriTTS_R root')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Checkpoint output directory')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scale', type=float, default=30.0)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--val-every', type=int, default=2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--musan-dir', type=str, default='')
    parser.add_argument('--rir-dir', type=str, default='')
    parser.add_argument('--crop-duration', type=float, default=3.0)
    parser.add_argument('--sub-centers', type=int, default=1)
    parser.add_argument('--fine-tune', action='store_true')
    parser.add_argument('--fine-tune-margin', type=float, default=0.5)
    parser.add_argument('--fine-tune-epochs', type=int, default=5)
    parser.add_argument('--fine-tune-crop', type=float, default=6.0)
    parser.add_argument('--asnorm', action='store_true')
    parser.add_argument('--asnorm-cohort-size', type=int, default=300)

    args = parser.parse_args()

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

    # Handle Fine-tuning Mode
    if args.fine_tune:
        if not args.resume:
            logger.error("--fine-tune requires --resume checkpoint!")
            sys.exit(1)
        args.n_epochs = args.fine_tune_epochs
        args.margin = args.fine_tune_margin
        args.lr = args.lr * 0.1
        args.crop_duration = args.fine_tune_crop
        logger.info(f"Fine-tuning mode: margin={args.margin}, lr={args.lr}, "
                   f"crop={args.crop_duration}s, epochs={args.n_epochs}")

    # Load Data
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

    # Create Model — CAM++ (matches sonata-cam Rust crate)
    model_config = {
        'architecture': 'cam++',
        'n_mels': SPEAKER_MEL_BINS,
        'dim': CAM_DIM,
        'num_blocks': CAM_BLOCKS,
        'num_heads': CAM_HEADS,
        'embed_dim': SPEAKER_EMBED_DIM,
        'sample_rate': SPEAKER_SAMPLE_RATE,
    }

    logger.info(f"Creating CAM++ model (dim={CAM_DIM}, blocks={CAM_BLOCKS}, "
                f"heads={CAM_HEADS}, embed_dim={SPEAKER_EMBED_DIM})...")

    model = CamPlusPlus(
        n_mels=model_config['n_mels'],
        dim=model_config['dim'],
        num_blocks=model_config['num_blocks'],
        num_heads=model_config['num_heads'],
        embed_dim=model_config['embed_dim'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e6:.1f}M")

    mel_extractor = MelSpectrogramExtractor().to(device)
    spec_augment = SpecAugment().to(device)
    classifier = AAMSoftmax(
        SPEAKER_EMBED_DIM,
        num_speakers,
        scale=args.scale,
        margin=args.margin,
        sub_centers=args.sub_centers
    ).to(device)

    # Optimizer and Scheduler
    optimizer = Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs - args.warmup_epochs)

    start_epoch = 0
    best_eer = 1.0
    patience_counter = 0

    # Resume from Checkpoint
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

    # Training Loop
    for epoch in range(start_epoch, args.n_epochs):
        epoch_start = time.time()

        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
            current_lr = warmup_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"\nEpoch {epoch + 1}/{args.n_epochs} | LR: {current_lr:.6f}")

        train_loss = train_epoch(
            model, mel_extractor, spec_augment, train_loader,
            classifier, optimizer, device
        )

        epoch_time = time.time() - epoch_start
        logger.info(f"Train loss: {train_loss:.6f} | Time: {epoch_time:.1f}s")

        if epoch >= args.warmup_epochs:
            scheduler.step()

        # Validation
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
                eval_eer = asnorm_eer
            else:
                logger.info(f"Validation EER: {eer:.4f} ({eer*100:.2f}%) | Best: {best_eer:.4f}")
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

    # Export
    logger.info("\nExporting final model...")
    export_to_safetensors(model, model_config, str(output_dir / 'speaker_encoder.safetensors'))
    export_to_onnx(model, str(output_dir / 'speaker_encoder.onnx'))

    logger.info(f"\nTraining complete! Best EER: {best_eer:.4f} ({best_eer*100:.2f}%)")


if __name__ == '__main__':
    main()
