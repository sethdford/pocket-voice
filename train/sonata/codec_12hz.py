"""Sonata Codec 12.5Hz — Low frame rate codec for 4x token reduction.

Encode:  Audio (24kHz) → Mel/Waveform → Encoder → FSQ (semantic tokens) + Acoustic latents
Decode:  (Semantic embedding + Acoustic latents) → ConvDecoder → Audio

Key changes vs 50Hz codec (codec.py):
  - hop_length: 480 → 1920 (12.5Hz vs 50Hz, 4x fewer tokens)
  - n_fft: 1024 → 4096 (better freq resolution for larger frames)
  - n_mels: 80 → 160 (richer per-frame spectral info)
  - enc_dim: 256 → 512 (more capacity per frame)
  - acoustic_dim: 256 → 512 (richer acoustic latent)
  - WaveformEncoder: 4 stages → 5 stages (1920x vs 480x downsample)
  - ConvDecoder: 4 stages → 5 stages (1920x vs 480x upsample)
  - FSQ: [8,8,8,8] = 4096 entries (stable, no-collapse guarantee)
  - Temporal Context Module: multi-scale dilated convs before Conformer

The 4x token reduction compounds with speculative decoding:
  4x fewer tokens × 2.3x faster generation (ReDrafter) = ~9x effective speedup.

Reference architectures:
  - Mimi (Kyutai/Moshi): 12.5Hz, 1.1 kbps, SeaNet encoder, streaming-capable
  - DualCodec: 12.5Hz semantic + acoustic, FSQ-based
  - WavTokenizer: Single codebook at low frame rates, attention decoder
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Codec12HzConfig
from codec import (
    MelSpectrogram,
    ConformerBlock,
    FSQ,
    ConvNeXtBlock,
    ResidualUnit,
    ISTFTHead,
    MultiScaleSTFTLoss,
    MelReconstructionLoss,
    fsq_entropy_loss,
    WavLMPerceptualLoss,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal Context Module — compensates for 4x lower frame rate
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalContextModule(nn.Module):
    """Multi-scale dilated convolution for temporal context aggregation.

    At 12.5Hz, each frame represents 80ms of audio (vs 20ms at 50Hz). The
    Conformer needs wider temporal context per frame to capture the same
    information. This module applies parallel dilated convolutions at multiple
    scales and fuses them, giving each frame access to ~320ms of context
    before the Conformer even sees it.

    Inspired by Mimi's multi-scale encoder and TCN (Temporal Convolutional
    Network) architectures.
    """

    def __init__(self, dim: int, kernel_size: int = 7,
                 dilations: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size,
                          dilation=d, padding=(kernel_size // 2) * d, groups=dim),
                nn.BatchNorm1d(dim),
                nn.SiLU(),
            )
            for d in dilations
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(dim * len(dilations), dim, 1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D) with expanded temporal context."""
        x_t = x.transpose(1, 2)  # (B, D, T)
        branches = [branch(x_t) for branch in self.branches]
        fused = self.fuse(torch.cat(branches, dim=1))
        return fused.transpose(1, 2) + x  # Residual connection


# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced Conformer Encoder for 12.5Hz
# ═══════════════════════════════════════════════════════════════════════════════

class ConformerEncoder12Hz(nn.Module):
    """Conformer encoder with temporal context module for 12.5Hz operation.

    At 12.5Hz, each mel frame encodes 80ms of audio. We:
    1. Project mel (160-dim) to enc_dim (512)
    2. Apply TemporalContextModule for multi-scale context aggregation
    3. Run through deeper Conformer blocks (6 layers vs 4)
    4. Project to FSQ latent dimension
    """

    def __init__(self, cfg: Codec12HzConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.n_mels, cfg.enc_dim)
        self.temporal_ctx = TemporalContextModule(
            cfg.enc_dim, kernel_size=7, dilations=(1, 2, 4, 8)
        )
        self.blocks = nn.ModuleList([
            ConformerBlock(
                cfg.enc_dim, cfg.enc_n_heads, cfg.enc_conv_kernel,
                cfg.enc_ff_mult,
            )
            for _ in range(cfg.enc_n_layers)
        ])
        self.semantic_proj = nn.Linear(cfg.enc_dim, cfg.fsq_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, T_frames) → semantic_latent: (B, T_frames, fsq_dim)"""
        x = mel.transpose(1, 2)        # (B, T, n_mels)
        x = self.input_proj(x)         # (B, T, enc_dim)
        x = self.temporal_ctx(x)       # Multi-scale context
        for block in self.blocks:
            x = block(x)
        return self.semantic_proj(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced Waveform Encoder for 12.5Hz (1920x downsample)
# ═══════════════════════════════════════════════════════════════════════════════

class ConvEncoderBlock(nn.Module):
    """Strided Conv1d downsample + dilated residual units."""

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.downsample = nn.Conv1d(
            in_ch, out_ch, kernel_size=stride * 2,
            stride=stride, padding=stride // 2,
        )
        self.residuals = nn.Sequential(
            ResidualUnit(out_ch, dilation=1),
            ResidualUnit(out_ch, dilation=3),
            ResidualUnit(out_ch, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residuals(self.downsample(x))


class WaveformEncoder12Hz(nn.Module):
    """Conv1d encoder for raw waveform → 12.5Hz acoustic latent.

    Strides [4, 8, 5, 4, 3] = 1920x downsample (24kHz → 12.5Hz).
    5 stages vs 4 in the 50Hz codec, with larger channel widths.
    """

    def __init__(self, cfg: Codec12HzConfig):
        super().__init__()
        strides = cfg.encoder_strides  # [4, 8, 5, 4, 3] = 1920x
        D = cfg.dec_dim
        # 5 stages + input: channels grow progressively
        channels = [D // 32, D // 16, D // 8, D // 4, D // 2, D]

        self.input_conv = nn.Conv1d(1, channels[0], 7, padding=3)

        self.encoder = nn.Sequential(*[
            ConvEncoderBlock(channels[i], channels[i + 1], s)
            for i, s in enumerate(strides)
        ])

        self.output_proj = nn.Conv1d(channels[-1], cfg.acoustic_dim, 1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, T_samples) → acoustic_latent: (B, T_frames, acoustic_dim)"""
        x = audio.unsqueeze(1)          # (B, 1, T)
        x = self.input_conv(x)          # (B, C0, T)
        x = self.encoder(x)             # (B, D, T/1920)
        x = self.output_proj(x)         # (B, acoustic_dim, T_frames)
        return x.transpose(1, 2)        # (B, T_frames, acoustic_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced Upsample Decoder for 12.5Hz (1920x upsample)
# ═══════════════════════════════════════════════════════════════════════════════

class UpsampleBlock(nn.Module):
    """ConvTranspose upsample + residual dilated convolutions."""

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=stride * 2,
            stride=stride, padding=stride // 2,
        )
        self.residuals = nn.Sequential(
            ResidualUnit(out_ch, dilation=1),
            ResidualUnit(out_ch, dilation=3),
            ResidualUnit(out_ch, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residuals(self.upsample(x))


class ConvDecoder12Hz(nn.Module):
    """ConvTranspose decoder for 12.5Hz → 24kHz reconstruction.

    Strides [3, 4, 5, 8, 4] = 1920x upsample.
    Larger channel dimensions and deeper ConvNeXt backbone than 50Hz version
    to compensate for the harder reconstruction task (80ms per frame).
    """

    def __init__(self, cfg: Codec12HzConfig):
        super().__init__()
        input_dim = cfg.fsq_dim + cfg.acoustic_dim
        D = cfg.dec_dim

        # Decoder strides: [3, 4, 5, 8, 4] = 1920x upsample
        strides = cfg.decoder_strides
        channels = [D, D, D // 2, D // 4, D // 8, D // 16]

        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, D, 7, padding=3),
            nn.LeakyReLU(0.1),
        )

        self.backbone = nn.Sequential(*[
            ConvNeXtBlock(D, cfg.dec_conv_kernel, cfg.dec_ff_mult)
            for _ in range(cfg.dec_n_layers)
        ])

        self.upsample = nn.Sequential(*[
            UpsampleBlock(channels[i], channels[i + 1], s)
            for i, s in enumerate(strides)
        ])

        self.output = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels[-1], 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, semantic_codes: torch.Tensor,
                acoustic_latent: torch.Tensor) -> torch.Tensor:
        """Decode from codes + latent → audio waveform."""
        x = torch.cat([semantic_codes, acoustic_latent], dim=-1)
        x = x.transpose(1, 2)           # (B, D_in, T_frames)
        x = self.input_proj(x)          # (B, D, T_frames)
        x = self.backbone(x)            # (B, D, T_frames)
        x = self.upsample(x)            # (B, D//16, T_samples)
        return self.output(x).squeeze(1)  # (B, T_samples)


class VocosDecoder12Hz(nn.Module):
    """Vocos-style iSTFT decoder for 12.5Hz.

    Predicts STFT magnitude and instantaneous frequency, then reconstructs
    via iSTFT. n_fft=4096, hop=1920 for 12.5Hz operation.
    """

    def __init__(self, cfg: Codec12HzConfig):
        super().__init__()
        input_dim = cfg.fsq_dim + cfg.acoustic_dim
        self.input_proj = nn.Conv1d(input_dim, cfg.dec_dim, 7, padding=3)
        self.backbone = nn.Sequential(*[
            ConvNeXtBlock(cfg.dec_dim, cfg.dec_conv_kernel, cfg.dec_ff_mult)
            for _ in range(cfg.dec_n_layers)
        ])
        self.head = ISTFTHead(cfg.dec_dim, cfg.n_fft, cfg.hop_length)

    def forward(self, semantic_codes: torch.Tensor,
                acoustic_latent: torch.Tensor) -> torch.Tensor:
        x = torch.cat([semantic_codes, acoustic_latent], dim=-1)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.backbone(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Full 12.5Hz Codec
# ═══════════════════════════════════════════════════════════════════════════════

class SonataCodec12Hz(nn.Module):
    """Sonata 12.5Hz audio codec — 4x token reduction vs 50Hz baseline.

    Architecture:
      Encode: audio → mel → TemporalContext → Conformer → FSQ (4096 codebook)
              audio → StridedConv (1920x) → acoustic latent (512-dim)
      Decode: (FSQ codes + acoustic latent) → ConvNeXt → ConvTranspose (1920x) → audio

    The semantic tokens (4096 vocab, 12.5 Hz) feed the LM.
    The acoustic latents (512-dim, 12.5 Hz) are predicted by the Flow network.
    """

    def __init__(self, cfg: Codec12HzConfig):
        super().__init__()
        self.cfg = cfg
        self.mel = MelSpectrogram(cfg)
        self.semantic_encoder = ConformerEncoder12Hz(cfg)
        self.acoustic_encoder = WaveformEncoder12Hz(cfg)
        self.fsq = FSQ(cfg.fsq_levels)

        if cfg.decoder_type == 'istft':
            self.decoder = VocosDecoder12Hz(cfg)
        else:
            self.decoder = ConvDecoder12Hz(cfg)

        self.semantic_embed = nn.Embedding(cfg.fsq_codebook_size, cfg.fsq_dim)

    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        audio: (B, T_samples)
        Returns:
            semantic_tokens: (B, T_frames) — discrete indices for LM
            acoustic_latent: (B, T_frames, acoustic_dim) — continuous for flow
            semantic_codes: (B, T_frames, fsq_dim) — quantized for decoder
        """
        mel = self.mel(audio)
        semantic_raw = self.semantic_encoder(mel)
        acoustic_latent = self.acoustic_encoder(audio)
        semantic_codes, semantic_tokens = self.fsq(semantic_raw)

        # Align sequence lengths (mel path and waveform path may differ slightly)
        T_min = min(semantic_codes.shape[1], acoustic_latent.shape[1])
        semantic_codes = semantic_codes[:, :T_min]
        semantic_tokens = semantic_tokens[:, :T_min]
        acoustic_latent = acoustic_latent[:, :T_min]

        return semantic_tokens, acoustic_latent, semantic_codes

    def decode(self, semantic_tokens: torch.Tensor,
               acoustic_latent: torch.Tensor) -> torch.Tensor:
        """Decode from semantic tokens + acoustic latent → audio."""
        semantic_codes = self.fsq.indices_to_codes(semantic_tokens)
        return self.decoder(semantic_codes, acoustic_latent)

    def decode_from_codes(self, semantic_codes: torch.Tensor,
                          acoustic_latent: torch.Tensor) -> torch.Tensor:
        """Decode from raw FSQ codes + acoustic latent → audio."""
        return self.decoder(semantic_codes, acoustic_latent)

    def forward(self, audio: torch.Tensor):
        """Full encode-decode for training. Returns reconstructed audio + tokens."""
        semantic_tokens, acoustic_latent, semantic_codes = self.encode(audio)
        reconstructed = self.decode_from_codes(semantic_codes, acoustic_latent)
        return reconstructed, semantic_tokens, acoustic_latent


# ═══════════════════════════════════════════════════════════════════════════════
# Training Utilities
# ═══════════════════════════════════════════════════════════════════════════════

class Codec12HzLoss(nn.Module):
    """Combined loss for 12.5Hz codec training.

    Multi-scale STFT loss + mel reconstruction + FSQ entropy + optional WavLM perceptual.
    Tuned with wider STFT windows to match the 12.5Hz frame rate.
    """

    def __init__(self, cfg: Codec12HzConfig,
                 stft_weight: float = 1.0,
                 mel_weight: float = 1.0,
                 entropy_weight: float = 0.1,
                 perceptual_weight: float = 0.0):
        super().__init__()
        self.stft_weight = stft_weight
        self.mel_weight = mel_weight
        self.entropy_weight = entropy_weight
        self.perceptual_weight = perceptual_weight

        # Wider STFT windows for 12.5Hz — need to capture 80ms frame detail
        self.stft_loss = MultiScaleSTFTLoss(
            fft_sizes=(1024, 2048, 4096),
            hop_sizes=(256, 512, 1024),
            win_sizes=(1024, 2048, 4096),
        )
        self.mel_loss = MelReconstructionLoss(cfg)
        self.codebook_size = cfg.fsq_codebook_size

        if perceptual_weight > 0:
            self.perceptual_loss = WavLMPerceptualLoss()
        else:
            self.perceptual_loss = None

    def forward(self, predicted: torch.Tensor, target: torch.Tensor,
                semantic_tokens: torch.Tensor) -> dict:
        """Compute all losses, return dict for flexible weighting."""
        # Match lengths
        T = min(predicted.shape[-1], target.shape[-1])
        predicted = predicted[..., :T]
        target = target[..., :T]

        losses = {}
        losses['stft'] = self.stft_loss(predicted, target)
        losses['mel'] = self.mel_loss(predicted, target)
        losses['entropy'] = fsq_entropy_loss(semantic_tokens, self.codebook_size)

        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            losses['perceptual'] = self.perceptual_loss(predicted, target)

        losses['total'] = (
            self.stft_weight * losses['stft'] +
            self.mel_weight * losses['mel'] +
            self.entropy_weight * losses['entropy'] +
            self.perceptual_weight * losses.get('perceptual', 0.0)
        )
        return losses


# ═══════════════════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = Codec12HzConfig()
    model = SonataCodec12Hz(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    enc_params = sum(p.numel() for p in model.semantic_encoder.parameters()) + \
                 sum(p.numel() for p in model.acoustic_encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())

    print(f"Sonata Codec 12.5Hz:")
    print(f"  Encoder:  {enc_params/1e6:.1f}M params")
    print(f"  FSQ:      {cfg.fsq_codebook_size} entries ({cfg.fsq_dim}-dim, levels={cfg.fsq_levels})")
    print(f"  Decoder:  {dec_params/1e6:.1f}M params")
    print(f"  Total:    {n_params/1e6:.1f}M params")
    print(f"  Frame rate: {cfg.frame_rate} Hz")
    print(f"  Acoustic dim: {cfg.acoustic_dim}")
    print(f"  Encoder strides: {cfg.encoder_strides} = {math.prod(cfg.encoder_strides)}x")
    print(f"  Decoder strides: {cfg.decoder_strides} = {math.prod(cfg.decoder_strides)}x")

    # Forward test
    B, T = 2, 24000  # 1 second of audio
    audio = torch.randn(B, T)
    reconstructed, tokens, acoustic = model(audio)

    print(f"\n  Input:  {audio.shape}")
    print(f"  Tokens: {tokens.shape} (range: {tokens.min()}-{tokens.max()})")
    print(f"  Acoustic: {acoustic.shape}")
    print(f"  Output: {reconstructed.shape}")
    print(f"  Tokens/sec: {tokens.shape[1]} (expected ~{cfg.frame_rate:.1f})")

    # Loss test
    loss_fn = Codec12HzLoss(cfg)
    losses = loss_fn(reconstructed, audio, tokens)
    print(f"\n  Total loss: {losses['total']:.4f}")
    print(f"  STFT loss:  {losses['stft']:.4f}")
    print(f"  Mel loss:   {losses['mel']:.4f}")
    print(f"  Entropy:    {losses['entropy']:.4f}")

    # Compare with 50Hz
    print(f"\n  Token reduction vs 50Hz: {50/cfg.frame_rate:.1f}x")
    print(f"  Tokens per second: {cfg.frame_rate:.1f} (was 50)")
    print(f"  Bits per second: {cfg.frame_rate * math.log2(cfg.fsq_codebook_size):.0f} "
          f"(was {50 * math.log2(32768):.0f})")
