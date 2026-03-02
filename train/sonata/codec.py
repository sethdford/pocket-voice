"""Sonata Codec — Conformer encoder + FSQ quantizer + iSTFT decoder.

Encode:  Audio (24kHz) → Mel → Conformer → FSQ (semantic tokens) + Acoustic latents
Decode:  (Semantic embedding + Acoustic latents) → ConvNeXt backbone → iSTFT → Audio

The semantic tokens (4096 vocab, 50 Hz) go to the LM for text-to-speech.
The acoustic latents (256-dim, 50 Hz) are predicted by the Flow network.
The iSTFT decoder reconstructs audio from both, ~100x faster than ConvTranspose.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CodecConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Mel Spectrogram
# ═══════════════════════════════════════════════════════════════════════════════

class MelSpectrogram(nn.Module):
    """Compute log-mel spectrogram from waveform."""

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.n_fft = cfg.n_fft
        self.hop_length = cfg.hop_length
        self.n_mels = cfg.n_mels
        self.sample_rate = cfg.sample_rate

        # Mel filterbank
        mel_fb = self._mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sample_rate)
        self.register_buffer("mel_fb", mel_fb)
        self.register_buffer("window", torch.hann_window(cfg.win_length))

    @staticmethod
    def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> torch.Tensor:
        n_freqs = n_fft // 2 + 1
        low_freq = 0.0
        high_freq = sr / 2.0

        def hz_to_mel(f):
            return 2595.0 * math.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_low = hz_to_mel(low_freq)
        mel_high = hz_to_mel(high_freq)
        mel_points = torch.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        freq_bins = (hz_points / sr * n_fft).long()

        fb = torch.zeros(n_mels, n_freqs)
        for i in range(n_mels):
            lo, mid, hi = freq_bins[i], freq_bins[i + 1], freq_bins[i + 2]
            for j in range(lo, mid):
                fb[i, j] = (j - lo).float() / max(1, (mid - lo).float())
            for j in range(mid, hi):
                fb[i, j] = (hi - j).float() / max(1, (hi - mid).float())
        return fb  # (n_mels, n_freqs)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, T) → mel: (B, n_mels, T_frames)"""
        spec = torch.stft(
            audio, self.n_fft, self.hop_length,
            window=self.window, return_complex=True,
        )
        power = spec.abs().pow(2)    # (B, n_fft/2+1, T_frames)
        mel = torch.matmul(self.mel_fb, power)  # (B, n_mels, T_frames)
        return torch.log(mel.clamp(min=1e-7))


# ═══════════════════════════════════════════════════════════════════════════════
# Conformer Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class ConformerFeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    """Conformer convolution module: pointwise → GLU → depthwise → BN → SiLU → pointwise."""

    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pointwise1 = nn.Conv1d(dim, 2 * dim, 1)
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise2 = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pointwise1(x)
        x = F.glu(x, dim=1)    # (B, D, T)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)  # (B, T, D)


class ConformerMHSA(nn.Module):
    """Multi-head self-attention with relative positional encoding."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm)
        return self.dropout(out)


class ConformerBlock(nn.Module):
    """Conformer block: FF/2 → MHSA → Conv → FF/2 → LayerNorm."""

    def __init__(self, dim: int, n_heads: int, conv_kernel: int = 31,
                 ff_mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ff1 = ConformerFeedForward(dim, ff_mult, dropout)
        self.mhsa = ConformerMHSA(dim, n_heads, dropout)
        self.conv = ConformerConvModule(dim, conv_kernel, dropout)
        self.ff2 = ConformerFeedForward(dim, ff_mult, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class ConformerEncoder(nn.Module):
    """Conformer encoder: mel → conformer blocks → semantic latent (for FSQ)."""

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.n_mels, cfg.enc_dim)
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
        x = mel.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.semantic_proj(x)


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


class WaveformEncoder(nn.Module):
    """Conv1d encoder on raw waveform. Preserves phase info unlike mel-based encoders.
    Mirrors the ConvDecoder with strides [8,5,4,3] for 480x downsample → 50 Hz.
    """

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        strides = [3, 4, 5, 8]
        D = cfg.dec_dim
        channels = [D // 16, D // 8, D // 4, D // 2, D]

        self.input_conv = nn.Conv1d(1, channels[0], 7, padding=3)

        self.encoder = nn.Sequential(*[
            ConvEncoderBlock(channels[i], channels[i + 1], s)
            for i, s in enumerate(strides)
        ])

        self.output_proj = nn.Conv1d(channels[-1], cfg.acoustic_dim, 1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, T_samples) → acoustic_latent: (B, T_frames, acoustic_dim)"""
        x = audio.unsqueeze(1)
        x = self.input_conv(x)
        x = self.encoder(x)
        x = self.output_proj(x)
        return x.transpose(1, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Finite Scalar Quantization (FSQ)
# ═══════════════════════════════════════════════════════════════════════════════

class FSQ(nn.Module):
    """Finite Scalar Quantization — simpler and more stable than VQ/RVQ.

    Each dimension is independently quantized to a finite set of levels.
    No codebook collapse, no EMA updates, no commitment loss needed.
    Straight-through estimator for gradients.

    With levels=[8,8,8,8]: 4096 implicit codebook entries, 4-dim latent.
    """

    def __init__(self, levels: list, input_scale: float = 3.0):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        self.input_scale = input_scale
        self.register_buffer(
            "levels_t", torch.tensor(levels, dtype=torch.float32)
        )
        self.codebook_size = 1
        for l in levels:
            self.codebook_size *= l

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize continuous latent to discrete levels. STE for gradients.

        For L levels per dim, centers are at index - (L-1)/2 for index in [0, L-1].
        E.g. L=8: centers at -3.5,-2.5,...,2.5,3.5 (8 levels). The previous
        round().clamp(-floor((L-1)/2), floor((L-1)/2)) gave only 7 levels for even L.
        """
        half_levels = (self.levels_t - 1) / 2
        z_bounded = torch.tanh(z * self.input_scale)
        z_scaled = z_bounded * half_levels

        # Map to nearest of L discrete levels; index in [0, L-1]
        indices = (z_scaled + half_levels).round()
        max_idx = (self.levels_t - 1).to(device=z.device, dtype=indices.dtype)
        indices = torch.minimum(torch.maximum(indices, indices.new_zeros(1)), max_idx)
        z_quantized = indices - half_levels
        z_quantized = z_scaled + (z_quantized - z_scaled).detach()

        return z_quantized

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert per-dimension codes to flat codebook indices."""
        # codes: (..., dim) with values in [-(L-1)/2, (L-1)/2]
        half_levels = (self.levels_t - 1) / 2
        shifted = (codes + half_levels).long()  # [0, L-1]

        indices = torch.zeros(codes.shape[:-1], dtype=torch.long, device=codes.device)
        multiplier = 1
        for d in reversed(range(self.dim)):
            indices += shifted[..., d] * multiplier
            multiplier *= self.levels[d]
        return indices

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert flat indices back to per-dimension codes."""
        codes = torch.zeros(*indices.shape, self.dim, device=indices.device)
        remainder = indices
        for d in reversed(range(self.dim)):
            codes[..., d] = (remainder % self.levels[d]).float()
            remainder = remainder // self.levels[d]
        half_levels = (self.levels_t - 1) / 2
        return codes - half_levels

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: (B, T, fsq_dim)
        Returns:
            quantized: (B, T, fsq_dim) — quantized latent (STE gradients)
            indices: (B, T) — codebook indices for LM training
        """
        quantized = self.quantize(z)
        indices = self.codes_to_indices(quantized)
        return quantized, indices


# ═══════════════════════════════════════════════════════════════════════════════
# iSTFT Decoder (Vocos-style)
# ═══════════════════════════════════════════════════════════════════════════════

class ConvNeXtBlock(nn.Module):
    """ConvNeXt-v2 block: depthwise conv → LayerNorm → pointwise up → GELU → pointwise down."""

    def __init__(self, dim: int, kernel_size: int = 7, mult: float = 4.0):
        super().__init__()
        inner = int(dim * mult)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, inner)
        self.pwconv2 = nn.Linear(inner, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, T)"""
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, T, D)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        x = x * self.gamma
        x = x.transpose(1, 2)  # (B, D, T)
        return residual + x


class ResidualUnit(nn.Module):
    """Dilated residual unit for upsampling decoder."""

    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(dim, dim, 7, dilation=dilation, padding=3 * dilation),
            nn.LeakyReLU(0.1),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


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


class ISTFTHead(nn.Module):
    """Predict STFT magnitude and phase, then reconstruct via iSTFT."""

    def __init__(self, dim: int, n_fft: int, hop_length: int):
        super().__init__()
        n_bins = n_fft // 2 + 1
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mag_proj = nn.Linear(dim, n_bins)
        self.phase_proj = nn.Linear(dim, n_bins)
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        log_mag = self.mag_proj(x).clamp(-10.0, 10.0)
        mag = log_mag.exp().transpose(1, 2)
        inst_freq = self.phase_proj(x).clamp(-torch.pi, torch.pi).transpose(1, 2)
        phase = torch.cumsum(inst_freq, dim=-1)
        stft_complex = mag * torch.exp(1j * phase)
        audio = torch.istft(stft_complex, self.n_fft, self.hop_length,
                            window=self.window, length=None)
        return audio.clamp(-1.0, 1.0)


class ConvDecoder(nn.Module):
    """ConvTranspose decoder (Encodec/DAC-style). Proven to train fast and produce
    high-quality waveforms. Upsamples from frame rate to sample rate via
    cascaded transposed convolutions with dilated residual blocks.
    """

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        input_dim = cfg.fsq_dim + cfg.acoustic_dim
        D = cfg.dec_dim

        # hop_length=480 factored as 8 × 5 × 4 × 3
        strides = [8, 5, 4, 3]
        channels = [D, D // 2, D // 4, D // 8, D // 16]

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
        x = torch.cat([semantic_codes, acoustic_latent], dim=-1)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.backbone(x)
        x = self.upsample(x)
        return self.output(x).squeeze(1)


class VocosDecoder(nn.Module):
    """Vocos-style decoder (iSTFT). Kept for compatibility."""

    def __init__(self, cfg: CodecConfig):
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
# Full Sonata Codec
# ═══════════════════════════════════════════════════════════════════════════════

class SonataCodec(nn.Module):
    """Complete Sonata audio codec.

    Encode: audio → mel → conformer → (semantic_tokens, acoustic_latent)
    Decode: (semantic_codes, acoustic_latent) → ConvNeXt → iSTFT → audio
    """

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.cfg = cfg
        self.mel = MelSpectrogram(cfg)
        self.semantic_encoder = ConformerEncoder(cfg)
        self.acoustic_encoder = WaveformEncoder(cfg)
        self.fsq = FSQ(cfg.fsq_levels)

        if getattr(cfg, 'decoder_type', 'conv') == 'istft':
            self.decoder = VocosDecoder(cfg)
        else:
            self.decoder = ConvDecoder(cfg)



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
# Loss Functions
# ═══════════════════════════════════════════════════════════════════════════════

class MultiScaleSTFTLoss(nn.Module):
    """Multi-resolution STFT loss with magnitude, phase, and complex components."""

    def __init__(self, fft_sizes=(512, 1024, 2048), hop_sizes=(128, 256, 512),
                 win_sizes=(512, 1024, 2048), phase_weight: float = 0.5,
                 complex_weight: float = 0.5):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.phase_weight = phase_weight
        self.complex_weight = complex_weight

    def _stft_loss(self, x, y, n_fft, hop, win_len):
        window = torch.hann_window(win_len, device=x.device)
        x = x.clamp(-1.0, 1.0)
        y = y.clamp(-1.0, 1.0)
        x_stft = torch.stft(x, n_fft, hop, window=window, return_complex=True)
        y_stft = torch.stft(y, n_fft, hop, window=window, return_complex=True)

        x_mag = x_stft.abs().clamp(min=1e-5)
        y_mag = y_stft.abs().clamp(min=1e-5)

        sc_loss = (x_mag - y_mag).norm(dim=(1, 2)) / y_mag.norm(dim=(1, 2)).clamp(min=1.0)
        log_loss = F.l1_loss(torch.log(x_mag), torch.log(y_mag))
        mag_loss = sc_loss.mean() + log_loss

        phase_diff = torch.angle(x_stft) - torch.angle(y_stft)
        phase_loss = (1 - torch.cos(phase_diff)).mean()

        complex_loss = F.l1_loss(torch.view_as_real(x_stft),
                                 torch.view_as_real(y_stft))

        total = mag_loss + self.phase_weight * phase_loss + self.complex_weight * complex_loss
        return total.clamp(max=50.0)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            loss += self._stft_loss(predicted, target, n_fft, hop, win)
        return loss / len(self.fft_sizes)


class MelReconstructionLoss(nn.Module):
    """L1 loss on mel spectrogram — perceptually weighted."""

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.mel = MelSpectrogram(cfg)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mel = self.mel(predicted)
        tgt_mel = self.mel(target)
        return F.l1_loss(pred_mel, tgt_mel)


def fsq_entropy_loss(indices: torch.Tensor, codebook_size: int) -> torch.Tensor:
    """Encourage uniform codebook usage via negative entropy of code distribution.

    Returns a value in [0, 1] where 0 = perfectly uniform (ideal).
    """
    flat = indices.reshape(-1)
    counts = torch.zeros(codebook_size, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
    probs = counts / counts.sum().clamp(min=1)
    probs = probs[probs > 0]
    entropy = -(probs * probs.log()).sum()
    max_entropy = math.log(codebook_size)
    return 1.0 - entropy / max_entropy


class WavLMPerceptualLoss(nn.Module):
    """Perceptual loss using frozen WavLM features.

    Matches intermediate representations of a pretrained WavLM model between
    original and reconstructed audio. This captures high-level perceptual
    similarity that STFT/mel losses miss — dramatically improves codec quality.

    Used by DAC, Vocos, Encodec v2, FunCodec, and all recent SOTA codecs.
    """

    def __init__(self, layer_ids=(4, 8, 12), sample_rate: int = 16000):
        super().__init__()
        self.layer_ids = layer_ids
        self.sample_rate = sample_rate
        self._model = None

    def _load_model(self, device):
        if self._model is not None:
            return
        try:
            from transformers import WavLMModel
            self._model = WavLMModel.from_pretrained(
                "microsoft/wavlm-base", output_hidden_states=True
            ).to(device)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
            n = sum(p.numel() for p in self._model.parameters())
            print(f"  [WavLM] Loaded microsoft/wavlm-base ({n/1e6:.0f}M params, frozen)")
        except ImportError:
            print("  [WavLM] WARNING: transformers not installed. "
                  "pip install transformers. Falling back to no perceptual loss.")
            self._model = None

    @staticmethod
    def _resample_if_needed(audio, orig_sr, target_sr):
        if orig_sr == target_sr:
            return audio
        ratio = target_sr / orig_sr
        new_len = int(audio.shape[-1] * ratio)
        return torch.nn.functional.interpolate(
            audio.unsqueeze(1), size=new_len, mode='linear', align_corners=False
        ).squeeze(1)

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor,
                orig_sr: int = 24000) -> torch.Tensor:
        self._load_model(original.device)
        if self._model is None:
            return torch.tensor(0.0, device=original.device)

        with torch.no_grad():
            orig_16k = self._resample_if_needed(original.detach(), orig_sr, self.sample_rate)
            orig_out = self._model(orig_16k)
            orig_features = [orig_out.hidden_states[i] for i in self.layer_ids]

        recon_16k = self._resample_if_needed(reconstructed, orig_sr, self.sample_rate)
        recon_out = self._model(recon_16k)
        recon_features = [recon_out.hidden_states[i] for i in self.layer_ids]

        loss = torch.tensor(0.0, device=original.device)
        for orig_feat, recon_feat in zip(orig_features, recon_features):
            T_min = min(orig_feat.shape[1], recon_feat.shape[1])
            loss = loss + torch.nn.functional.l1_loss(
                recon_feat[:, :T_min], orig_feat[:, :T_min]
            )
        return loss / len(self.layer_ids)


# ═══════════════════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = CodecConfig()
    model = SonataCodec(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    enc_params = sum(p.numel() for p in model.semantic_encoder.parameters()) + \
                 sum(p.numel() for p in model.acoustic_encoder.parameters())
    print(f"Sonata Codec:")
    print(f"  Encoder:  {enc_params/1e6:.1f}M params")
    print(f"  FSQ:      {cfg.fsq_codebook_size} entries ({cfg.fsq_dim}-dim, levels={cfg.fsq_levels})")
    print(f"  Decoder:  {sum(p.numel() for p in model.decoder.parameters())/1e6:.1f}M params")
    print(f"  Total:    {n_params/1e6:.1f}M params")
    print(f"  Frame rate: {cfg.frame_rate} Hz")
    print(f"  Acoustic dim: {cfg.acoustic_dim}")

    # Forward test
    B, T = 2, 24000  # 1 second of audio
    audio = torch.randn(B, T)
    reconstructed, tokens, acoustic = model(audio)

    print(f"\n  Input:  {audio.shape}")
    print(f"  Tokens: {tokens.shape} (range: {tokens.min()}-{tokens.max()})")
    print(f"  Acoustic: {acoustic.shape}")
    print(f"  Output: {reconstructed.shape}")

    # Loss test
    ms_loss = MultiScaleSTFTLoss()
    mel_loss = MelReconstructionLoss(cfg)
    l_ms = ms_loss(reconstructed, audio)
    l_mel = mel_loss(reconstructed, audio)
    print(f"\n  Multi-scale STFT loss: {l_ms:.4f}")
    print(f"  Mel reconstruction loss: {l_mel:.4f}")
