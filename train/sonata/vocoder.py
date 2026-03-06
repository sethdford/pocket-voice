"""BigVGAN-lite mel → waveform vocoder for Sonata Flow v3.

Compact GAN vocoder (~14M params) that converts 80-bin mel spectrograms
to 24kHz audio. Designed to pair with Flow v3's mel output.

Architecture (BigVGAN-inspired):
  Generator:  mel → Conv1d → [Upsample + AMPBlock] × 4 → Conv1d → tanh → audio
  Anti-aliased snake activation for periodic inductive bias.

Discriminators:
  MPD: Multi-Period Discriminator (periods 2, 3, 5, 7, 11)
  MSD: Multi-Scale Discriminator (scales 1x, 2x, 4x)

Training losses:
  G: mel reconstruction (L1) + adversarial (hinge) + feature matching
  D: adversarial (hinge)

Based on: BigVGAN (NVIDIA, ICLR 2023), HiFi-GAN (Kong et al., 2020).
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VocoderConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Anti-aliased Snake activation (BigVGAN)
# ═══════════════════════════════════════════════════════════════════════════════

class SnakeAlpha(nn.Module):
    """Snake activation: x + (1/a) * sin^2(a*x). Periodic inductive bias for audio."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-9)) * torch.sin(self.alpha * x).pow(2)


# ═══════════════════════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════════════════════

class AMPBlock(nn.Module):
    """Anti-aliased multi-periodicity residual block.

    Parallel dilated convolutions at multiple kernel sizes, each with snake
    activation, summed and averaged.
    """

    def __init__(self, channels: int, kernel_sizes: List[int],
                 dilations: List[List[int]]):
        super().__init__()
        self.n_blocks = len(kernel_sizes)
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()

        for k, ds in zip(kernel_sizes, dilations):
            layers = nn.ModuleList()
            acts = nn.ModuleList()
            for d in ds:
                padding = (k * d - d) // 2
                layers.append(nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(channels, channels, k, dilation=d, padding=padding)
                ))
                acts.append(SnakeAlpha(channels))
            self.convs.append(layers)
            self.acts.append(acts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for conv_layers, act_layers in zip(self.convs, self.acts):
            h = x
            for conv, act in zip(conv_layers, act_layers):
                h = act(h)
                h = conv(h)
            out = out + h
        return out / self.n_blocks


class VocoderGenerator(nn.Module):
    """BigVGAN-lite generator: mel → waveform via upsampling + AMPBlocks."""

    def __init__(self, cfg: VocoderConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.upsample_initial_channel

        self.input_conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(cfg.n_mels, ch, 7, padding=3)
        )

        self.upsamples = nn.ModuleList()
        self.amp_blocks = nn.ModuleList()

        for i, (u_rate, u_kernel) in enumerate(
            zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)
        ):
            in_ch = ch // (2 ** i)
            out_ch = ch // (2 ** (i + 1))
            self.upsamples.append(nn.utils.parametrizations.weight_norm(
                nn.ConvTranspose1d(in_ch, out_ch, u_kernel,
                                   stride=u_rate,
                                   padding=(u_kernel - u_rate) // 2)
            ))
            self.amp_blocks.append(
                AMPBlock(out_ch, cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes)
            )

        final_ch = ch // (2 ** len(cfg.upsample_rates))
        self.output_act = SnakeAlpha(final_ch)
        self.output_conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(final_ch, 1, 7, padding=3)
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, T_frames) → audio: (B, 1, T_samples)"""
        x = self.input_conv(mel)
        for upsample, amp_block in zip(self.upsamples, self.amp_blocks):
            x = upsample(x)
            x = amp_block(x)
        x = self.output_act(x)
        x = self.output_conv(x)
        return torch.tanh(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Period Discriminator (MPD)
# ═══════════════════════════════════════════════════════════════════════════════

class PeriodDiscriminator(nn.Module):
    """Single-period sub-discriminator."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        channels = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            stride = 3 if i < len(channels) - 2 else 1
            self.convs.append(nn.utils.parametrizations.weight_norm(
                nn.Conv2d(channels[i], channels[i + 1], (5, 1),
                          stride=(stride, 1), padding=(2, 0))
            ))
        self.output = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(1024, 1, (3, 1), padding=(1, 0))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []
        B, C, T = x.shape

        # Pad to multiple of period, reshape to 2D
        pad_len = (self.period - T % self.period) % self.period
        x = F.pad(x, (0, pad_len), mode="reflect")
        x = x.view(B, C, -1, self.period)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.output(x)
        features.append(x)
        return x.flatten(1, -1), features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: List[int]):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outputs, all_features = [], []
        for d in self.discriminators:
            out, feats = d(x)
            outputs.append(out)
            all_features.append(feats)
        return outputs, all_features


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Scale Discriminator (MSD)
# ═══════════════════════════════════════════════════════════════════════════════

class ScaleDiscriminator(nn.Module):
    """Single-scale sub-discriminator."""

    def __init__(self):
        super().__init__()
        channels = [1, 128, 128, 256, 512, 1024, 1024, 1024]
        kernels = [15, 41, 41, 41, 41, 41, 5]
        strides = [1, 2, 2, 4, 4, 1, 1]
        groups_ = [1, 4, 16, 16, 16, 16, 1]

        self.convs = nn.ModuleList()
        for i in range(len(kernels)):
            self.convs.append(nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels[i], channels[i + 1], kernels[i],
                          stride=strides[i], groups=groups_[i],
                          padding=kernels[i] // 2)
            ))
        self.output = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(1024, 1, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.output(x)
        features.append(x)
        return x.flatten(1, -1), features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.downsamplers = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, stride=2, padding=1),
            nn.Sequential(
                nn.AvgPool1d(4, stride=2, padding=1),
                nn.AvgPool1d(4, stride=2, padding=1),
            ),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outputs, all_features = [], []
        for disc, downsample in zip(self.discriminators, self.downsamplers):
            x_d = downsample(x)
            out, feats = disc(x_d)
            outputs.append(out)
            all_features.append(feats)
        return outputs, all_features


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Resolution STFT Discriminator (BigVGAN-v2)
# ═══════════════════════════════════════════════════════════════════════════════

class STFTDiscriminator(nn.Module):
    """Discriminates on magnitude STFT at a single resolution."""

    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n_bins = n_fft // 2 + 1

        self.convs = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(
                nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            nn.utils.parametrizations.weight_norm(
                nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            nn.utils.parametrizations.weight_norm(
                nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            nn.utils.parametrizations.weight_norm(
                nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.output = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = x.squeeze(1) if x.dim() == 3 else x
        window = torch.hann_window(self.n_fft, periodic=True, device=x.device)
        spec = torch.stft(x, self.n_fft, self.hop_length, window=window,
                          return_complex=True)
        mag = spec.abs().unsqueeze(1)

        features = []
        h = mag
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            features.append(h)
        out = self.output(h)
        features.append(out)
        return out.flatten(1, -1), features


class MultiResolutionSTFTDiscriminator(nn.Module):
    """Discriminates at multiple STFT resolutions for broad spectral coverage."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(n_fft=1024, hop_length=256),
            STFTDiscriminator(n_fft=2048, hop_length=512),
            STFTDiscriminator(n_fft=512, hop_length=128),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outputs, all_features = [], []
        for disc in self.discriminators:
            out, feats = disc(x)
            outputs.append(out)
            all_features.append(feats)
        return outputs, all_features


# ═══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════════════════

class MelSpecLoss(nn.Module):
    """Multi-resolution mel spectrogram L1 loss."""

    RESOLUTIONS = [(1024, 480, 1024), (2048, 960, 2048), (512, 240, 512)]

    def __init__(self, sample_rate: int = 24000, n_mels: int = 80):
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        for n_fft, _, _ in self.RESOLUTIONS:
            fb = self._mel_filterbank(n_mels, n_fft, sample_rate)
            self.register_buffer(f"mel_fb_{n_fft}", fb)

    @staticmethod
    def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> torch.Tensor:
        import math as _math
        n_freqs = n_fft // 2 + 1
        def hz_to_mel(f): return 2595.0 * _math.log10(1.0 + f / 700.0)
        def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        low_mel, high_mel = hz_to_mel(0), hz_to_mel(sr / 2)
        mel_points = torch.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = torch.tensor([mel_to_hz(m.item()) for m in mel_points])
        bins = (hz_points / (sr / n_fft)).long().clamp(0, n_freqs - 1)
        fb = torch.zeros(n_mels, n_freqs)
        for i in range(n_mels):
            lo, center, hi = bins[i], bins[i + 1], bins[i + 2]
            if center > lo:
                fb[i, lo:center] = torch.linspace(0, 1, int(center - lo))
            if hi > center:
                fb[i, center:hi] = torch.linspace(1, 0, int(hi - center))
        return fb

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mel_loss = 0.0
        sc_loss = 0.0
        for n_fft, hop, win in self.RESOLUTIONS:
            window = torch.hann_window(win, periodic=True, device=y.device)
            mel_fb = getattr(self, f"mel_fb_{n_fft}").to(y.device)
            y_mag, y_mel = self._spec_and_mel(y.squeeze(1), n_fft, hop, win, window, mel_fb)
            yh_mag, yh_mel = self._spec_and_mel(y_hat.squeeze(1), n_fft, hop, win, window, mel_fb)
            mel_loss = mel_loss + F.l1_loss(yh_mel, y_mel)
            sc_loss = sc_loss + torch.norm(y_mag - yh_mag, p="fro") / torch.norm(y_mag, p="fro").clamp(min=1e-6)
        return (mel_loss + sc_loss) / float(len(self.RESOLUTIONS))

    @staticmethod
    def _spec_and_mel(x, n_fft, hop, win, window, mel_fb):
        spec = torch.stft(x, n_fft, hop, win, window=window, return_complex=True)
        mag = spec.abs().clamp(min=1e-5)
        power = mag.pow(2)
        mel = torch.matmul(mel_fb, power)
        return mag, torch.log(mel.clamp(min=1e-5))


def discriminator_loss(disc_real_outputs, disc_fake_outputs):
    """Hinge discriminator loss."""
    loss = 0.0
    for dr, df in zip(disc_real_outputs, disc_fake_outputs):
        loss += torch.mean(F.relu(1 - dr)) + torch.mean(F.relu(1 + df))
    return loss


def generator_loss(disc_fake_outputs):
    """Hinge generator adversarial loss."""
    loss = 0.0
    for df in disc_fake_outputs:
        loss += torch.mean(-df)
    return loss


def feature_matching_loss(real_features, fake_features):
    """L1 feature matching loss across all discriminator layers."""
    loss = 0.0
    n = 0
    for rf_list, ff_list in zip(real_features, fake_features):
        for rf, ff in zip(rf_list, ff_list):
            loss += F.l1_loss(ff, rf.detach())
            n += 1
    return loss / max(n, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Full Vocoder (Generator + Discriminators)
# ═══════════════════════════════════════════════════════════════════════════════

class SonataVocoder(nn.Module):
    """Complete vocoder: generator + discriminators for training."""

    def __init__(self, cfg: VocoderConfig):
        super().__init__()
        self.cfg = cfg
        self.generator = VocoderGenerator(cfg)
        self.mpd = MultiPeriodDiscriminator(cfg.mpd_periods)
        self.msd = MultiScaleDiscriminator()
        self.mrstft = MultiResolutionSTFTDiscriminator()
        self.mel_loss = MelSpecLoss(cfg.sample_rate, cfg.n_mels)

    def generate(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, T_frames, n_mels) → audio: (B, T_samples)"""
        return self.generator(mel.transpose(1, 2)).squeeze(1)

    def _align_lengths(self, audio_fake: torch.Tensor,
                       audio_real: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Truncate generated audio to match real audio length (STFT framing can add samples)."""
        min_len = min(audio_fake.shape[-1], audio_real.shape[-1])
        return audio_fake[..., :min_len], audio_real[..., :min_len]

    @staticmethod
    def _r1_penalty(real_outputs, real_input):
        """R1 gradient penalty on real data (Mescheder et al., 2018)."""
        total_out = sum(o.sum() for o in real_outputs)
        grads = torch.autograd.grad(
            outputs=total_out,
            inputs=real_input,
            create_graph=True,
            retain_graph=True
        )
        penalty = grads[0].pow(2).sum() / grads[0].shape[0]
        return penalty

    def training_step_d(self, mel: torch.Tensor,
                        audio_real: torch.Tensor,
                        r1_weight: float = 1.0) -> dict:
        """Discriminator training step with R1 gradient penalty."""
        with torch.no_grad():
            audio_fake = self.generator(mel.transpose(1, 2))
        audio_real = audio_real.unsqueeze(1)
        audio_fake, audio_real = self._align_lengths(audio_fake, audio_real)

        audio_real.requires_grad_(True)
        mpd_real, _ = self.mpd(audio_real)
        mpd_fake, _ = self.mpd(audio_fake.detach())
        msd_real, _ = self.msd(audio_real)
        msd_fake, _ = self.msd(audio_fake.detach())
        stft_real, _ = self.mrstft(audio_real)
        stft_fake, _ = self.mrstft(audio_fake.detach())

        d_loss = (discriminator_loss(mpd_real, mpd_fake) +
                  discriminator_loss(msd_real, msd_fake) +
                  discriminator_loss(stft_real, stft_fake))

        r1 = self._r1_penalty(mpd_real + msd_real + stft_real, audio_real)
        d_loss = d_loss + r1_weight * r1

        return {"d_loss": d_loss, "r1": r1.item()}

    def training_step_g(self, mel: torch.Tensor,
                        audio_real: torch.Tensor) -> dict:
        """Generator training step."""
        audio_fake = self.generator(mel.transpose(1, 2))
        audio_real = audio_real.unsqueeze(1)
        audio_fake, audio_real = self._align_lengths(audio_fake, audio_real)

        mpd_real, mpd_real_feats = self.mpd(audio_real)
        mpd_fake, mpd_fake_feats = self.mpd(audio_fake)
        msd_real, msd_real_feats = self.msd(audio_real)
        msd_fake, msd_fake_feats = self.msd(audio_fake)
        stft_real, stft_real_feats = self.mrstft(audio_real)
        stft_fake, stft_fake_feats = self.mrstft(audio_fake)

        adv_loss = (generator_loss(mpd_fake) +
                    generator_loss(msd_fake) +
                    generator_loss(stft_fake))
        fm_loss = (feature_matching_loss(mpd_real_feats, mpd_fake_feats) +
                   feature_matching_loss(msd_real_feats, msd_fake_feats) +
                   feature_matching_loss(stft_real_feats, stft_fake_feats))
        mel_loss = self.mel_loss(audio_fake, audio_real)

        total = (adv_loss +
                 self.cfg.feature_loss_weight * fm_loss +
                 self.cfg.mel_loss_weight * mel_loss)

        return {
            "g_loss": total,
            "adv_loss": adv_loss.item(),
            "fm_loss": fm_loss.item(),
            "mel_loss": mel_loss.item(),
        }


if __name__ == "__main__":
    cfg = VocoderConfig()
    model = SonataVocoder(cfg)

    g_params = sum(p.numel() for p in model.generator.parameters())
    d_params = sum(p.numel() for p in model.mpd.parameters()) + \
               sum(p.numel() for p in model.msd.parameters())

    print(f"Sonata Vocoder (BigVGAN-lite):")
    print(f"  Generator:      {g_params / 1e6:.1f}M params")
    print(f"  Discriminators: {d_params / 1e6:.1f}M params")
    print(f"  Total:          {(g_params + d_params) / 1e6:.1f}M params")
    print(f"  Upsample rates: {cfg.upsample_rates} (product={math.prod(cfg.upsample_rates)})")
    print(f"  Hop length:     {cfg.hop_length}")
    print(f"  Expected:       product × 1 = {math.prod(cfg.upsample_rates)} samples per frame")

    B, T = 2, 50
    mel = torch.randn(B, T, cfg.n_mels)
    audio = model.generate(mel)
    print(f"\n  Input mel:  {mel.shape}")
    print(f"  Output audio: {audio.shape}")
    print(f"  Expected samples: {T * cfg.hop_length} = {T}×{cfg.hop_length}")
    print(f"  Actual samples:   {audio.shape[-1]}")
