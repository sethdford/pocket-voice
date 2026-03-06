"""Multi-Period Discriminator (MPD) + Multi-Scale Discriminator (MSD) for Sonata Codec.

Adversarial training is CRITICAL for human-quality audio. Without it, reconstructed
speech sounds muffled and lacks the crisp transients of natural speech.

MPD captures periodic patterns (pitch harmonics at different periods).
MSD captures multi-scale temporal structure.

Based on HiFi-GAN / EnCodec / Vocos discriminator design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class PeriodDiscriminator(nn.Module):
    """Discriminator for a single period — reshapes 1D audio to 2D and applies 2D convs."""

    def __init__(self, period: int, channels: int = 32, max_channels: int = 512):
        super().__init__()
        self.period = period

        layers = []
        in_ch = 1
        for i in range(4):
            out_ch = min(channels * (2 ** i), max_channels)
            layers.append(nn.Conv2d(in_ch, out_ch, (5, 1), stride=(3, 1), padding=(2, 0)))
            layers.append(nn.LeakyReLU(0.1))
            in_ch = out_ch
        layers.append(nn.Conv2d(in_ch, 1, (3, 1), padding=(1, 0)))
        self.net = nn.ModuleList([l for l in layers if isinstance(l, nn.Module)])
        self.activations = nn.ModuleList([nn.LeakyReLU(0.1)] * 4 + [nn.Identity()])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """x: (B, 1, T) → score + feature maps for feature matching loss."""
        B, C, T = x.shape
        # Pad to multiple of period
        if T % self.period != 0:
            pad = self.period - (T % self.period)
            x = F.pad(x, (0, pad), "reflect")
            T = x.shape[-1]

        x = x.view(B, 1, T // self.period, self.period)

        features = []
        conv_idx = 0
        for module in self.net:
            if isinstance(module, nn.Conv2d):
                x = module(x)
                if conv_idx < 4:
                    x = F.leaky_relu(x, 0.1)
                features.append(x)
                conv_idx += 1

        return x.flatten(1), features


class MultiPeriodDiscriminator(nn.Module):
    """MPD: multiple periods capture different harmonic structures."""

    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(self, x: torch.Tensor):
        """x: (B, T) → list of (score, features) per period."""
        x = x.unsqueeze(1)  # (B, 1, T)
        results = []
        for disc in self.discriminators:
            score, feats = disc(x)
            results.append((score, feats))
        return results


class ScaleDiscriminator(nn.Module):
    """Single-scale discriminator with 1D convolutions."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, channels, 15, stride=1, padding=7),
            nn.Conv1d(channels, channels, 41, stride=2, padding=20, groups=4),
            nn.Conv1d(channels, channels * 2, 41, stride=2, padding=20, groups=16),
            nn.Conv1d(channels * 2, channels * 4, 41, stride=4, padding=20, groups=16),
            nn.Conv1d(channels * 4, channels * 4, 41, stride=4, padding=20, groups=16),
            nn.Conv1d(channels * 4, channels * 4, 5, stride=1, padding=2),
            nn.Conv1d(channels * 4, 1, 3, stride=1, padding=1),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:
                x = F.leaky_relu(x, 0.1)
            features.append(x)
        return x.flatten(1), features


class MultiScaleDiscriminator(nn.Module):
    """MSD: discriminators at 1x, 2x, 4x downsampled rates."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.downsamplers = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(2, stride=2),
            nn.AvgPool1d(4, stride=4),
        ])

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)  # (B, 1, T)
        results = []
        for disc, down in zip(self.discriminators, self.downsamplers):
            x_down = down(x)
            score, feats = disc(x_down)
            results.append((score, feats))
        return results


class ResolutionDiscriminator(nn.Module):
    """Single-resolution STFT discriminator: operates on 2D magnitude spectrogram."""

    def __init__(self, n_fft: int = 1024, hop_length: int = 256, channels: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n_bins = n_fft // 2 + 1

        self.convs = nn.ModuleList([
            nn.Conv2d(1, channels, (3, 9), padding=(1, 4)),
            nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(channels, channels, (3, 3), padding=(1, 1)),
            nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """x: (B, 1, T) → STFT → 2D conv."""
        x_1d = x.squeeze(1)
        window = torch.hann_window(self.n_fft, device=x.device)
        stft = torch.stft(x_1d, self.n_fft, self.hop_length, window=window, return_complex=True)
        mag = stft.abs().unsqueeze(1)

        features = []
        for i, conv in enumerate(self.convs):
            mag = conv(mag)
            if i < len(self.convs) - 1:
                mag = F.leaky_relu(mag, 0.1)
            features.append(mag)
        return mag.flatten(1), features


class MultiResolutionDiscriminator(nn.Module):
    """MRD: discriminators at different STFT resolutions.
    Captures spectral detail that MPD/MSD miss — complements them.
    """

    def __init__(self, resolutions=((1024, 256), (2048, 512), (512, 128))):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ResolutionDiscriminator(n_fft, hop)
            for n_fft, hop in resolutions
        ])

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        results = []
        for disc in self.discriminators:
            score, feats = disc(x)
            results.append((score, feats))
        return results


class SonataDiscriminator(nn.Module):
    """Combined MPD + MSD + optional MRD discriminator."""

    def __init__(self, use_mrd: bool = False):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.mrd = MultiResolutionDiscriminator() if use_mrd else None

    def forward(self, x: torch.Tensor):
        """x: (B, T) → mpd_results, msd_results (+ mrd_results if enabled)."""
        mpd_results = self.mpd(x)
        msd_results = self.msd(x)
        if self.mrd is not None:
            mrd_results = self.mrd(x)
            return mpd_results, msd_results, mrd_results
        return mpd_results, msd_results


# ─── Loss functions ──────────────────────────────────────────────────────────

def discriminator_loss(real_results, fake_results):
    """Hinge loss for discriminator training."""
    loss = 0
    for (real_score, _), (fake_score, _) in zip(real_results, fake_results):
        loss += torch.mean(F.relu(1 - real_score)) + torch.mean(F.relu(1 + fake_score))
    return loss


def generator_adversarial_loss(fake_results):
    """Generator adversarial loss (make discriminator think fake is real)."""
    loss = 0
    for fake_score, _ in fake_results:
        loss += torch.mean(F.relu(1 - fake_score))
    return loss


def feature_matching_loss(real_results, fake_results):
    """L1 feature matching loss between discriminator intermediate features."""
    loss = 0
    count = 0
    for (_, real_feats), (_, fake_feats) in zip(real_results, fake_results):
        for rf, ff in zip(real_feats, fake_feats):
            loss += F.l1_loss(ff, rf.detach())
            count += 1
    return loss / max(count, 1)


def r1_gradient_penalty(disc: nn.Module, real_audio: torch.Tensor) -> torch.Tensor:
    """R1 gradient penalty on real samples — stabilizes GAN training.

    Penalizes the discriminator for having large gradients on real data,
    preventing mode collapse and training divergence (NaN blowups).

    Applied lazily (every N steps) to reduce overhead. Standard in
    StyleGAN2, Vocos, DAC, and all modern GAN-based audio codecs.
    """
    real_audio = real_audio.detach().requires_grad_(True)
    disc_out = disc(real_audio)

    all_results = []
    for item in disc_out:
        if isinstance(item, list):
            all_results.extend(item)

    all_scores = []
    for score, _ in all_results:
        all_scores.append(score.sum())
    total_score = sum(all_scores)

    grad = torch.autograd.grad(
        outputs=total_score, inputs=real_audio,
        create_graph=True, retain_graph=True,
    )[0]
    return grad.pow(2).reshape(grad.shape[0], -1).sum(dim=1).mean()


if __name__ == "__main__":
    disc = SonataDiscriminator()
    n_params = sum(p.numel() for p in disc.parameters())
    print(f"Sonata Discriminator: {n_params/1e6:.1f}M params")

    x = torch.randn(2, 24000)
    mpd_results, msd_results = disc(x)
    print(f"  MPD: {len(mpd_results)} sub-discriminators")
    print(f"  MSD: {len(msd_results)} sub-discriminators")

    d_loss = discriminator_loss(mpd_results + msd_results, mpd_results + msd_results)
    print(f"  D loss: {d_loss:.4f}")
