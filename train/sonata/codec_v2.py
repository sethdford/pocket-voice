"""Sonata Codec v2 — WavTokenizer-inspired single-codebook codec at 25 tok/sec.

Key improvements over v1:
  1. Extended context window (960 hop → 25 Hz frame rate instead of 480 → 50 Hz)
  2. Broader VQ space (FSQ levels [16,16,16,16] = 65536 entries, 4-dim)
  3. Attention-based decoder (T-Mimi style) instead of ConvNeXt for mobile efficiency
  4. Semantic pre-distillation: contrastive loss aligns codebook with SSL features
  5. Multi-scale discriminator for adversarial training

Based on: WavTokenizer (ICLR 2025), T-Mimi (Jan 2026), STACodec (Feb 2026).
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CodecConfig, CodecV2Config


# ═══════════════════════════════════════════════════════════════════════════════
# Codec V2 Config
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Transformer Decoder Block (T-Mimi style)
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerDecoderBlock(nn.Module):
    """Pure transformer block: LayerNorm → MHSA → LayerNorm → FFN.
    Runs 10x faster than ConvNeXt on mobile (XNNPACK/Metal optimize transformers).
    """

    def __init__(self, dim: int, n_heads: int, ff_mult: float = 4.0, eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        ff_dim = int(dim * ff_mult)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        h, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerDecoder(nn.Module):
    """T-Mimi-style pure transformer decoder with iSTFT head.

    Critical insight from T-Mimi paper: the last 2 layers must stay at
    full precision during quantization to maintain audio quality.
    """

    def __init__(self, cfg: CodecV2Config):
        super().__init__()
        input_dim = cfg.fsq_dim + cfg.acoustic_dim

        self.input_proj = nn.Linear(input_dim, cfg.dec_dim)

        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(cfg.dec_dim, cfg.dec_n_heads, cfg.dec_ff_mult)
            for _ in range(cfg.dec_n_layers)
        ])

        # iSTFT head: predict magnitude and instantaneous frequency
        n_bins = cfg.n_fft // 2 + 1
        self.mag_proj = nn.Linear(cfg.dec_dim, n_bins)
        self.phase_proj = nn.Linear(cfg.dec_dim, n_bins)
        self.register_buffer("window", torch.hann_window(cfg.n_fft))
        self.n_fft = cfg.n_fft
        self.hop_length = cfg.hop_length

        # Track which layers need full precision (last 2)
        self.n_full_precision_layers = 2

    def forward(self, semantic_codes: torch.Tensor,
                acoustic_latent: torch.Tensor) -> torch.Tensor:
        x = torch.cat([semantic_codes, acoustic_latent], dim=-1)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        mag = self.mag_proj(x).clamp(max=15.0).exp().transpose(1, 2)
        inst_freq = self.phase_proj(x).transpose(1, 2)
        phase = torch.cumsum(inst_freq, dim=-1)
        stft_complex = mag * torch.exp(1j * phase)
        return torch.istft(stft_complex, self.n_fft, self.hop_length,
                           window=self.window, length=None)


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Pre-Distillation (STACodec-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticDistillationLoss(nn.Module):
    """Align codec FSQ codes with self-supervised semantic features via contrastive loss.

    During training, a frozen SSL model (WavLM/HuBERT) provides target features.
    The codec encoder learns to produce codes that are semantically rich,
    not just acoustically faithful — this means fewer tokens carry more meaning.

    Based on: STACodec (Feb 2026), SecoustiCodec (2025).
    """

    def __init__(self, projection_dim: int = 256, temperature: float = 0.07,
                 ssl_dim: int = 768, codec_dim: int = 256):
        super().__init__()
        self.temperature = temperature
        self.projection_dim = projection_dim
        self.ssl_proj = nn.Linear(ssl_dim, projection_dim)
        self.codec_proj = nn.Linear(codec_dim, projection_dim)

    def _ensure_projections(self, ssl_dim: int, codec_dim: int, device):
        if self.ssl_proj.in_features != ssl_dim:
            self.ssl_proj = nn.Linear(ssl_dim, self.projection_dim).to(device)
        if self.codec_proj.in_features != codec_dim:
            self.codec_proj = nn.Linear(codec_dim, self.projection_dim).to(device)

    def forward(self, codec_features: torch.Tensor,
                ssl_features: torch.Tensor) -> torch.Tensor:
        """Contrastive alignment between codec and SSL frame features.

        codec_features: (B, T, codec_dim) — from encoder before FSQ
        ssl_features: (B, T', ssl_dim) — from frozen WavLM/HuBERT
        """
        self._ensure_projections(ssl_features.shape[-1], codec_features.shape[-1],
                                 codec_features.device)

        T_min = min(codec_features.shape[1], ssl_features.shape[1])
        c = F.normalize(self.codec_proj(codec_features[:, :T_min]), dim=-1)
        s = F.normalize(self.ssl_proj(ssl_features[:, :T_min].detach()), dim=-1)

        # InfoNCE: positive pairs are temporally aligned frames
        logits = torch.bmm(c, s.transpose(1, 2)) / self.temperature
        labels = torch.arange(T_min, device=logits.device).expand(logits.shape[0], -1)
        return F.cross_entropy(logits.reshape(-1, T_min), labels.reshape(-1))


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Scale Discriminator
# ═══════════════════════════════════════════════════════════════════════════════

class MultiScaleSubDiscriminator(nn.Module):
    """Single-scale 1D conv discriminator."""

    def __init__(self, in_channels: int = 1, channels: int = 64, n_layers: int = 4):
        super().__init__()
        layers = [nn.Conv1d(in_channels, channels, 15, padding=7), nn.LeakyReLU(0.1)]
        ch = channels
        for i in range(n_layers):
            next_ch = min(ch * 2, 512)
            stride = 4 if i < 2 else 2
            layers += [
                nn.Conv1d(ch, next_ch, stride * 2 + 1, stride=stride, padding=stride, groups=4 if i > 0 else 1),
                nn.LeakyReLU(0.1),
            ]
            ch = next_ch
        layers.append(nn.Conv1d(ch, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator operating at 1x, 2x, 4x downsampled audio."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            MultiScaleSubDiscriminator(),
            MultiScaleSubDiscriminator(),
            MultiScaleSubDiscriminator(),
        ])
        self.downsamplers = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, stride=2, padding=1),
            nn.AvgPool1d(8, stride=4, padding=2),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x: (B, 1, T) or (B, T). Returns list of discriminator outputs."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        outputs = []
        for disc, ds in zip(self.discriminators, self.downsamplers):
            x_ds = ds(x)
            outputs.append(disc(x_ds))
        return outputs


# ═══════════════════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = CodecV2Config()
    print(f"Sonata Codec V2:")
    print(f"  Frame rate: {cfg.frame_rate} Hz (vs 50 Hz in v1)")
    print(f"  FSQ: {cfg.fsq_codebook_size} entries ({cfg.fsq_dim}-dim, levels={cfg.fsq_levels})")
    print(f"  Token rate: {cfg.frame_rate} tok/s (vs 50 tok/s in v1)")
    print(f"  Decoder: {cfg.dec_n_layers}L transformer (T-Mimi style)")

    dec = TransformerDecoder(cfg)
    n_dec = sum(p.numel() for p in dec.parameters())
    print(f"  Decoder params: {n_dec/1e6:.1f}M")

    disc = MultiScaleDiscriminator()
    n_disc = sum(p.numel() for p in disc.parameters())
    print(f"  Discriminator params: {n_disc/1e6:.1f}M")

    distill = SemanticDistillationLoss()
    print(f"  Semantic distillation: contrastive alignment (temperature={distill.temperature})")
