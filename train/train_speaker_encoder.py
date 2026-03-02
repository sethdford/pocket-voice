#!/usr/bin/env python3
"""
train_speaker_encoder.py — Train ECAPA-TDNN speaker encoder with AAM-Softmax loss.

Architecture: SE-Res2Net blocks + attentive statistics pooling (24.6M params)
Loss: Additive Angular Margin Softmax (ArcFace) — prevents training collapse
Data: LibriTTS-R with SpecAugment + speed perturbation + noise injection
Validation: Equal Error Rate (EER) on test-clean

Usage:
    python3 train_speaker_encoder.py \
      --data-dir /path/to/LibriTTS_R \
      --output-dir ./checkpoints/speaker_encoder_v2 \
      --batch-size 64 --n-epochs 40 --lr 0.001 \
      --scale 30.0 --margin 0.2 \
      --warmup-epochs 2 --val-every 2 --patience 10

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

        # SE-Res2Net blocks
        self.blocks = nn.ModuleList()
        for i in range(len(channels)):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            out_ch = channels[i]
            self.blocks.append(
                SERes2NetBlock(in_ch, out_ch, kernel_sizes[i], dilations[i],
                              scale=scale, se_channels=se_channels)
            )

        # Multi-frame aggregation (concatenate all block outputs)
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

    Unlike GE2E, this loss:
    - Has no learnable scale/bias that can drift to zero
    - Doesn't need speaker-balanced batches
    - Angular margin prevents embedding collapse
    """

    def __init__(self, embedding_dim: int, num_speakers: int,
                 scale: float = 30.0, margin: float = 0.2):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_speakers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute trigonometric values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold to prevent cos(theta+m) from being negative when theta > pi-m
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute AAM-Softmax loss.

        Args:
            embeddings: [batch, embedding_dim]
            labels: [batch]

        Returns:
            loss: scalar
        """
        # L2 normalize both embeddings and weight columns
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: [batch, num_speakers]
        cosine = F.linear(embeddings, weight)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Compute cos(theta + m) using trigonometric identity
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        # When cos(theta) < cos(pi - m), use cosine - mm as fallback
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Only apply margin to target class
        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
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
                 augment: bool = False):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_duration * sample_rate)
        self.mode = mode
        self.augment = augment
        self.files = []
        self.speaker_ids = {}
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

            # Apply augmentation (training only)
            if self.augment:
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
def validate_eer(model, mel_extractor, loader, device) -> float:
    """Compute EER on validation set.

    Args:
        model: ECAPA-TDNN encoder
        mel_extractor: mel spectrogram extractor
        loader: test data loader
        device: torch device

    Returns:
        eer: Equal Error Rate
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
    return eer


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
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Config: scale={args.scale}, margin={args.margin}, "
                f"lr={args.lr}, bs={args.batch_size}")

    # ========================================================================
    # Load Data
    # ========================================================================
    logger.info("Loading LibriTTS-R dataset...")
    train_dataset = LibriTTSRDataset(args.data_dir, mode='train', augment=True)
    if len(train_dataset) == 0:
        logger.error("No training data found!")
        sys.exit(1)

    test_dataset = LibriTTSRDataset(args.data_dir, mode='test', augment=False)
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
        margin=args.margin
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
            eer = validate_eer(model, mel_extractor, test_loader, device)
            logger.info(f"Validation EER: {eer:.4f} ({eer*100:.2f}%) | "
                       f"Best: {best_eer:.4f}")

            if eer < best_eer:
                best_eer = eer
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
