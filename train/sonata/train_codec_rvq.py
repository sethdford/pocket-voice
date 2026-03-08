"""Train Sonata RVQ Codec for 12.5Hz audio compression with semantic distillation.

Architecture:
  Encode:  Audio (24kHz) → ConvEncoder (1920x downsample) → Transformer bottleneck → RVQ (8x2048)
  Decode:  RVQ codes → Transformer → ConvDecoder (1920x upsample) → Audio (24kHz)

Semantic distillation from WavLM-Large ensures the first RVQ codebook captures
meaningful speech content (similar to Mimi's semantic + acoustic split).

Training data: directory of .wav files or JSONL manifest with audio paths.

Usage:
  python train_codec_rvq.py \
    --manifest data/libritts_r_full_manifest.jsonl \
    --wavlm-model "microsoft/wavlm-large" \
    --device cuda \
    --epochs 100 \
    --batch-size 8 \
    --grad-accum 2 \
    --amp \
    --n-codebooks 8 \
    --codebook-size 2048 \
    --distill-weight 0.5

The encoder uses strided convolutions [4, 8, 5, 4, 3] = 1920x downsample.
The bottleneck is an 8-layer Transformer (512 dim, 8 heads, RoPE, causal).
RVQ has 8 codebooks x 2048 entries each (sparse vocab, ~13-bit per codebook).

Loss combines:
  - Reconstruction (multi-scale STFT + mel spectrogram)
  - RVQ commitment loss (encourages codebook usage)
  - WavLM distillation (MSE + contrastive, for semantic codebook)
  - Optional adversarial (discriminator on reconstruction)
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rvq_module import ResidualVQ
from wavlm_distill import WavLMDistillLoss
from codec import MultiScaleSTFTLoss, MelReconstructionLoss, MelSpectrogram
from config import Codec12HzConfig
from modules import cosine_lr, TrainingLog


# ═══════════════════════════════════════════════════════════════════════════════
# Strided ConvEncoder for RVQ — 1920x downsample
# ═══════════════════════════════════════════════════════════════════════════════

class ConvEncoderBlock(nn.Module):
    """Strided Conv1d downsample + residual dilated units."""

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.downsample = nn.Conv1d(
            in_ch, out_ch, kernel_size=stride * 2,
            stride=stride, padding=stride // 2,
        )
        self.residuals = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 3, dilation=1, padding=1),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, dilation=3, padding=3),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, dilation=9, padding=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        return x + self.residuals(x)  # Residual connection


class RVQEncoder(nn.Module):
    """Conv1d encoder: 24kHz audio → compressed latents at 12.5Hz.

    Strides [4, 8, 5, 4, 3] = 1920x downsample (24000 → 12.5 frames/sec).
    Maps 24kHz waveform to 512-dim bottleneck ready for Transformer + RVQ.
    """

    def __init__(self, enc_dim: int = 512, n_strides: int = 5):
        super().__init__()
        strides = [4, 8, 5, 4, 3]
        assert len(strides) == n_strides, f"Expected {n_strides} strides, got {len(strides)}"

        # Channel progression: [1] → [32] → [64] → [128] → [256] → [512] → [512]
        channels = [enc_dim // 32, enc_dim // 16, enc_dim // 8, enc_dim // 4, enc_dim // 2, enc_dim]

        self.input_conv = nn.Conv1d(1, channels[0], 7, padding=3)

        self.encoder = nn.Sequential(*[
            ConvEncoderBlock(channels[i], channels[i + 1], s)
            for i, s in enumerate(strides)
        ])

        self.output_proj = nn.Conv1d(channels[-1], enc_dim, 1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, T_samples) → latent: (B, T_frames, enc_dim)"""
        x = audio.unsqueeze(1)          # (B, 1, T)
        x = self.input_conv(x)          # (B, C, T)
        x = self.encoder(x)             # (B, C, T/1920)
        x = self.output_proj(x)         # (B, enc_dim, T_frames)
        return x.transpose(1, 2)        # (B, T_frames, enc_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# Transformer Bottleneck with RoPE and Causal Attention
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE rotation frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor) -> tuple:
    """Apply RoPE (rotary position embedding) to query and key."""
    def rotate(x, f):
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        f = f[None, :x_c.shape[1], None, :]
        return torch.view_as_real(x_c * f).flatten(-2).to(x.dtype)
    return rotate(xq, freqs), rotate(xk, freqs)


class TransformerBlock(nn.Module):
    """Single Transformer block with causal attention, RoPE, and SwiGLU FFN."""

    def __init__(self, dim: int = 512, n_heads: int = 8, ff_mult: float = 4.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Self-attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)

        # FFN (SwiGLU-style)
        ff_dim = int(dim * ff_mult)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, ff_dim)
        self.up = nn.Linear(dim, ff_dim)
        self.down = nn.Linear(ff_dim, dim)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        """x: (B, T, dim)"""
        B, T, D = x.shape

        # Attention
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).reshape(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x_norm).reshape(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x_norm).reshape(B, T, self.n_heads, self.head_dim)

        # Apply RoPE
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q_rot, k_rot = apply_rope(q.reshape(B, self.n_heads, T, -1),
                                   k.reshape(B, self.n_heads, T, -1),
                                   rope_freqs)
        q = q_rot.reshape(B, self.n_heads, T, self.head_dim)
        k = k_rot.reshape(B, self.n_heads, T, self.head_dim)

        # Causal attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = attn_weights @ v

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(attn_out)
        x = x + attn_out

        # FFN
        x_norm = self.norm2(x)
        gate_out = F.silu(self.gate(x_norm))
        up_out = self.up(x_norm)
        ffn_out = self.down(gate_out * up_out)
        x = x + ffn_out

        return x


class TransformerBottleneck(nn.Module):
    """Transformer stack with causal attention, RoPE, and optional skip connections."""

    def __init__(self, dim: int = 512, n_layers: int = 8, n_heads: int = 8, ff_mult: float = 4.0):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, ff_mult)
            for _ in range(n_layers)
        ])
        self.max_seq_len = 4096

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, dim) → (B, T, dim)"""
        rope_freqs = precompute_rope_freqs(self.dim // 8, self.max_seq_len, theta=10000.0)
        rope_freqs = rope_freqs.to(x.device)

        for block in self.blocks:
            x = block(x, rope_freqs)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# ConvDecoder for RVQ — 1920x upsample
# ═══════════════════════════════════════════════════════════════════════════════

class UpsampleBlock(nn.Module):
    """ConvTranspose upsample + residual dilated units."""

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=stride * 2,
            stride=stride, padding=stride // 2,
        )
        self.residuals = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 3, dilation=1, padding=1),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, dilation=3, padding=3),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, dilation=9, padding=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return x + self.residuals(x)  # Residual connection


class RVQDecoder(nn.Module):
    """Conv1d decoder: RVQ latents (12.5Hz) → 24kHz audio.

    Mirrors encoder strides in reverse: [3, 4, 5, 8, 4] = 1920x upsample.
    Upsamples from 512-dim bottleneck to 1-channel waveform.
    """

    def __init__(self, enc_dim: int = 512, n_strides: int = 5):
        super().__init__()
        strides = [3, 4, 5, 8, 4]  # Mirror of encoder: [4, 8, 5, 4, 3]
        assert len(strides) == n_strides, f"Expected {n_strides} strides, got {len(strides)}"

        # Channel progression: [512] → [256] → [128] → [64] → [32] → [16] → [1]
        channels = [enc_dim, enc_dim // 2, enc_dim // 4, enc_dim // 8, enc_dim // 16, enc_dim // 32]

        self.input_proj = nn.Conv1d(enc_dim, channels[0], 7, padding=3)

        self.backbone = nn.Sequential(*[
            UpsampleBlock(channels[i], channels[i + 1], s)
            for i, s in enumerate(strides)
        ])

        self.output = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(channels[-1], 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: (B, T_frames, enc_dim) → audio: (B, T_samples)"""
        x = latent.transpose(1, 2)      # (B, enc_dim, T_frames)
        x = self.input_proj(x)          # (B, enc_dim, T_frames)
        x = self.backbone(x)            # (B, C, T_frames * 1920)
        x = self.output(x)              # (B, 1, T_samples)
        return x.squeeze(1)             # (B, T_samples)


# ═══════════════════════════════════════════════════════════════════════════════
# Full RVQ Codec Model
# ═══════════════════════════════════════════════════════════════════════════════

class RVQCodec(nn.Module):
    """Complete RVQ-based audio codec with Transformer bottleneck.

    Audio (24kHz) → ConvEncoder (1920x) → Transformer → RVQ (8x2048)
    RVQ → Transformer → ConvDecoder (1920x) → Audio (24kHz)

    The RVQ codebooks are trained jointly with the encoder/decoder and a
    WavLM semantic distillation loss on the first codebook to encourage
    semantic structure.
    """

    def __init__(self,
                 enc_dim: int = 512,
                 n_codebooks: int = 8,
                 codebook_size: int = 2048,
                 n_transformer_layers: int = 8,
                 n_transformer_heads: int = 8):
        super().__init__()
        self.enc_dim = enc_dim
        self.n_codebooks = n_codebooks

        self.encoder = RVQEncoder(enc_dim=enc_dim)
        self.transformer_enc = TransformerBottleneck(
            dim=enc_dim, n_layers=n_transformer_layers, n_heads=n_transformer_heads
        )
        self.rvq = ResidualVQ(
            input_dim=enc_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=128,
        )
        self.transformer_dec = TransformerBottleneck(
            dim=enc_dim, n_layers=n_transformer_layers, n_heads=n_transformer_heads
        )
        self.decoder = RVQDecoder(enc_dim=enc_dim)

    def encode(self, audio: torch.Tensor) -> tuple:
        """Encode audio to RVQ codes.

        Args:
            audio: (B, T_samples)

        Returns:
            codes: (B, n_codebooks, T_frames)
            latent_post_rvq: (B, T_frames, enc_dim) — quantized latent
        """
        latent = self.encoder(audio)        # (B, T_frames, enc_dim)
        latent = self.transformer_enc(latent)  # (B, T_frames, enc_dim)
        codes, latent_quantized, commit_loss = self.rvq(latent.transpose(1, 2))
        latent_quantized = latent_quantized.transpose(1, 2)  # (B, T_frames, enc_dim)
        return codes, latent_quantized

    def decode(self, latent_quantized: torch.Tensor) -> torch.Tensor:
        """Decode from RVQ quantized latent to audio.

        Args:
            latent_quantized: (B, T_frames, enc_dim)

        Returns:
            audio: (B, T_samples)
        """
        latent = self.transformer_dec(latent_quantized)  # (B, T_frames, enc_dim)
        audio = self.decoder(latent)         # (B, T_samples)
        return audio

    def forward(self, audio: torch.Tensor) -> tuple:
        """Full encode-decode cycle.

        Returns:
            reconstructed: (B, T_samples)
            codes: (B, n_codebooks, T_frames)
            latent_pre_rvq: (B, T_frames, enc_dim) — for distillation
            latent_post_rvq: (B, T_frames, enc_dim)
            commit_loss: scalar
        """
        latent = self.encoder(audio)        # (B, T_frames, enc_dim)
        latent = self.transformer_enc(latent)  # (B, T_frames, enc_dim)

        # RVQ quantization
        codes, latent_quantized, commit_loss = self.rvq(latent.transpose(1, 2))
        latent_quantized = latent_quantized.transpose(1, 2)  # (B, T_frames, enc_dim)

        # Decode
        latent = self.transformer_dec(latent_quantized)
        reconstructed = self.decoder(latent)

        # For distillation, we use pre-RVQ latent or first codebook quantized embedding
        latent_pre_rvq = latent.transpose(1, 2)  # Keep encoder output for distillation

        return reconstructed, codes, latent_pre_rvq, latent_quantized, commit_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Audio Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class AudioDataset(Dataset):
    """Load audio files for RVQ codec training."""

    def __init__(self, manifest: str = None, data_dir: str = None,
                 segment_length: int = 24000, sample_rate: int = 24000):
        if manifest:
            self.files = []
            with open(manifest) as f:
                for line in f:
                    entry = json.loads(line)
                    self.files.append(Path(entry["audio"]))
            print(f"  Loaded {len(self.files)} files from manifest")
        elif data_dir:
            self.files = sorted(Path(data_dir).glob("**/*.wav"))
            if not self.files:
                self.files = sorted(Path(data_dir).glob("**/*.pt"))
            print(f"  Found {len(self.files)} files in {data_dir}")
        else:
            raise ValueError("Must provide --manifest or --data-dir")

        self.segment_length = segment_length
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        if path.suffix == ".pt":
            data = torch.load(path, weights_only=True, map_location="cpu")
            if "audio" in data:
                audio = data["audio"].float()
            elif "waveform" in data:
                audio = data["waveform"].float()
            else:
                raise KeyError(f"Expected 'audio' or 'waveform' key in {path}")
        else:
            import soundfile as sf
            audio_np, sr = sf.read(str(path), dtype="float32")
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            audio = torch.from_numpy(audio_np)
            if sr != self.sample_rate:
                ratio = self.sample_rate / sr
                new_len = int(len(audio) * ratio)
                audio = F.interpolate(
                    audio.unsqueeze(0).unsqueeze(0), size=new_len,
                    mode="linear", align_corners=False
                ).squeeze()

        # Random crop or pad
        if audio.shape[-1] > self.segment_length:
            start = torch.randint(0, audio.shape[-1] - self.segment_length, (1,)).item()
            audio = audio[start:start + self.segment_length]
        else:
            audio = F.pad(audio, (0, self.segment_length - audio.shape[-1]))

        return audio


# ═══════════════════════════════════════════════════════════════════════════════
# WavLM Feature Extractor for Distillation
# ═══════════════════════════════════════════════════════════════════════════════

class WavLMFeatureExtractor(nn.Module):
    """Extract WavLM hidden states for semantic distillation."""

    def __init__(self, model_name: str = "microsoft/wavlm-large"):
        super().__init__()
        try:
            from transformers import WavLMModel
            self.wavlm = WavLMModel.from_pretrained(model_name)
        except ImportError:
            print("  WARNING: transformers not installed, distillation disabled")
            self.wavlm = None

        # Freeze WavLM
        if self.wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False

    def forward(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract WavLM features.

        Args:
            audio: (B, T_samples) at 16kHz

        Returns:
            features: (B, T_frames, 1024) at codec frame rate (~12.5 Hz)
        """
        if self.wavlm is None:
            return None

        with torch.no_grad():
            # WavLM expects 16kHz audio
            outputs = self.wavlm(audio, output_hidden_states=True)
            hidden = outputs.last_hidden_state  # (B, T_wl, 1024)

        return hidden


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device(args.device)

    # Build model
    model = RVQCodec(
        enc_dim=512,
        n_codebooks=args.n_codebooks,
        codebook_size=args.codebook_size,
        n_transformer_layers=8,
        n_transformer_heads=8,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"RVQ Codec: {n_params/1e6:.1f}M params")
    print(f"  Encoder: 1920x downsample (4,8,5,4,3)")
    print(f"  Bottleneck: 8-layer Transformer (512 dim, 8 heads, RoPE)")
    print(f"  RVQ: {args.n_codebooks} codebooks x {args.codebook_size} entries")
    print(f"  Decoder: 1920x upsample (3,4,5,8,4)")

    # Loss functions
    stft_loss = MultiScaleSTFTLoss(
        fft_sizes=(1024, 2048, 4096),
        hop_sizes=(256, 512, 1024),
        win_sizes=(1024, 2048, 4096),
    ).to(device)

    cfg = Codec12HzConfig()
    mel_loss_fn = MelReconstructionLoss(cfg).to(device)

    wavlm_distill = None
    wavlm_extractor = None
    if args.distill_weight > 0:
        wavlm_extractor = WavLMFeatureExtractor(args.wavlm_model).to(device)
        wavlm_distill = WavLMDistillLoss(
            codebook_dim=128,  # RVQ internal codebook dim
            wavlm_dim=1024,
            temperature=0.07,
            mse_weight=1.0,
            contrastive_weight=0.5,
        ).to(device)
        print(f"  WavLM distillation: ON (weight={args.distill_weight})")
    else:
        print(f"  WavLM distillation: OFF")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # AMP
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    if use_amp:
        print(f"  AMP: ON (bfloat16)")

    # Dataset
    dataset = AudioDataset(
        manifest=args.manifest,
        data_dir=args.data_dir,
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    print(f"Dataset: {len(dataset)} files")
    print(f"Batch size: {args.batch_size}, Grad accum: {args.grad_accum}, "
          f"Effective batch: {args.batch_size * args.grad_accum}")
    print(f"Epochs: {args.epochs}")

    os.makedirs(args.output_dir, exist_ok=True)
    tlog = TrainingLog(os.path.join(args.output_dir, "losses.jsonl"))
    total_steps = len(loader) * args.epochs
    step = 0
    t0 = time.time()

    # Resume
    start_epoch = 0
    latest_ckpt = os.path.join(args.output_dir, "codec_rvq_latest.pt")
    if os.path.exists(latest_ckpt):
        print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed at step {step}, epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, audio in enumerate(loader):
            audio = audio.to(device)  # (B, T_samples)

            # Learning rate schedule
            lr = cosine_lr(step, args.warmup_steps, args.lr, args.lr * 0.01, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward pass
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                reconstructed, codes, latent_pre_rvq, latent_post_rvq, commit_loss = model(audio)

                # Reconstruction loss
                stft = stft_loss(reconstructed, audio)
                mel = mel_loss_fn(reconstructed, audio)
                recon_loss = stft + mel

                # RVQ commitment loss
                total_loss = recon_loss + 0.25 * commit_loss

                # WavLM distillation loss (on first codebook)
                distill_loss = 0.0
                if wavlm_distill is not None and args.distill_weight > 0:
                    # Extract WavLM features from original audio
                    # NOTE: WavLM expects 16kHz; resample if needed
                    wavlm_feats = wavlm_extractor(audio)
                    if wavlm_feats is not None:
                        # Get first codebook embeddings from RVQ
                        first_codes = codes[:, 0, :]  # (B, T_frames)
                        first_emb = model.rvq.quantizers[0].codebook(first_codes)
                        first_emb = first_emb.transpose(1, 2)  # (B, dim, T_frames) → (B, T_frames, dim)

                        # Downsample WavLM features to codec frame rate if needed
                        # (simple interpolation; frame rates may differ slightly)
                        if wavlm_feats.shape[1] != first_emb.shape[1]:
                            ratio = first_emb.shape[1] / wavlm_feats.shape[1]
                            wavlm_feats = F.interpolate(
                                wavlm_feats.transpose(1, 2),
                                size=int(wavlm_feats.shape[1] * ratio),
                                mode="linear",
                                align_corners=False
                            ).transpose(1, 2)
                            wavlm_feats = wavlm_feats[:, :first_emb.shape[1], :]

                        distill_loss = wavlm_distill(first_emb.transpose(1, 2), wavlm_feats.transpose(1, 2))
                        total_loss = total_loss + args.distill_weight * distill_loss

            # Backward
            loss_scaled = total_loss / args.grad_accum
            loss_scaled.backward()

            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += total_loss.item()
            step += 1

            if step <= 3 or step % args.log_interval == 0:
                elapsed = time.time() - t0
                sps = args.log_interval / elapsed
                print(f"  step {step}: loss={total_loss.item():.4f} "
                      f"(recon={recon_loss.item():.4f} commit={commit_loss.item():.4f} "
                      f"distill={distill_loss:.4f}) "
                      f"lr={lr:.2e} | {sps:.1f} steps/s")
                tlog.log(step=step, loss=total_loss.item(),
                         recon=recon_loss.item(), commit=commit_loss.item(),
                         distill=distill_loss, lr=lr, steps_per_sec=sps)
                t0 = time.time()

            # Periodic save
            if args.save_steps > 0 and step % args.save_steps == 0:
                ckpt_path = os.path.join(args.output_dir, f"codec_rvq_step{step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch,
                }, ckpt_path)
                print(f"  Saved {ckpt_path}")

                # Update latest
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch,
                }, latest_ckpt)

        # Epoch summary
        n_batches = max(len(loader), 1)
        print(f"Epoch {epoch+1}/{args.epochs}: avg_loss={epoch_loss/n_batches:.4f}")

        # Epoch checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"codec_rvq_epoch{epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "epoch": epoch + 1,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sonata RVQ Codec")
    parser.add_argument("--manifest", default=None,
                        help="JSONL manifest with 'audio' paths")
    parser.add_argument("--data-dir", default=None,
                        help="Directory of .wav files")
    parser.add_argument("--output-dir", default="checkpoints/codec_rvq")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--segment-length", type=int, default=24000)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--n-codebooks", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument("--wavlm-model", default="microsoft/wavlm-large")
    parser.add_argument("--distill-weight", type=float, default=0.5)
    args = parser.parse_args()
    train(args)
