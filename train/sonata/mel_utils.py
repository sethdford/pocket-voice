"""Canonical mel spectrogram extraction for Sonata Flow v3 + Vocoder.

Shared implementation ensures Flow v3 and Vocoder use identical mel filterbanks.
Any mismatch would cause Flow output to be incompatible with Vocoder input.

Parameters: n_fft=1024, hop_length=480 (50 Hz), n_mels=80, sample_rate=24000.
Uses vocoder-style bin-index-based filterbank for invertibility.
Hann window with periodic=True for spectral accuracy.
"""

import math
from typing import Optional

import torch


def _mel_filterbank(n_mels: int, n_fft: int, sr: int,
                   device: Optional[torch.device] = None) -> torch.Tensor:
    """Build mel filterbank using bin-index-based triangular filters.

    Uses vocoder-compatible construction: mel-to-hz points mapped to FFT bins
    via (hz / (sr / n_fft)).long(), triangular weights between lo/center/hi.
    """

    def hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    low_mel = hz_to_mel(0.0)
    high_mel = hz_to_mel(sr / 2.0)
    mel_points = torch.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = torch.tensor([mel_to_hz(m.item()) for m in mel_points])
    bins = (hz_points / (sr / n_fft)).long().clamp(0, n_freqs - 1)

    fb = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        lo, center, hi = bins[i].item(), bins[i + 1].item(), bins[i + 2].item()
        if center > lo:
            fb[i, lo:center] = torch.linspace(0, 1, int(center - lo))
        if hi > center:
            fb[i, center:hi] = torch.linspace(1, 0, int(hi - center))

    if device is not None:
        fb = fb.to(device)
    return fb


def mel_spectrogram(
    audio: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 480,
    n_mels: int = 80,
    sample_rate: int = 24000,
    win_length: Optional[int] = None,
) -> torch.Tensor:
    """Extract log-mel spectrogram from single audio.

    Args:
        audio: (T_samples,) or (1, T_samples) — mono audio
    Returns:
        (T_frames, n_mels) — log-mel spectrogram (time first)
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    win_length = win_length or n_fft
    window = torch.hann_window(n_fft, periodic=True, device=audio.device)
    stft = torch.stft(
        audio, n_fft, hop_length, win_length=win_length,
        window=window, return_complex=True,
    )
    mag = stft.abs().pow(2)  # power spectrum

    mel_fb = _mel_filterbank(n_mels, n_fft, sample_rate, device=audio.device)
    mel = torch.matmul(mel_fb, mag.squeeze(0))
    mel = torch.log(mel.clamp(min=1e-5))
    return mel.T  # (T_frames, n_mels)


def extract_mel_batch(
    audio: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 480,
    n_mels: int = 80,
    sample_rate: int = 24000,
    win_length: Optional[int] = None,
) -> torch.Tensor:
    """Extract log-mel spectrogram from batch of audio.

    Args:
        audio: (B, T_samples) — batch of mono audio
    Returns:
        (B, T_frames, n_mels) — log-mel spectrograms
    """
    B = audio.shape[0]
    if B == 0:
        return audio.new_empty(0, 0, n_mels)

    win_length = win_length or n_fft
    window = torch.hann_window(n_fft, periodic=True, device=audio.device)
    stft = torch.stft(
        audio, n_fft, hop_length, win_length=win_length,
        window=window, return_complex=True,
    )
    mag = stft.abs().pow(2)  # (B, n_freqs, T_frames)

    mel_fb = _mel_filterbank(n_mels, n_fft, sample_rate, device=audio.device)
    # (n_mels, n_freqs) @ (B, n_freqs, T) -> (B, n_mels, T)
    mel = torch.einsum("mf,bft->bmt", mel_fb, mag)
    mel = torch.log(mel.clamp(min=1e-5))
    return mel.transpose(1, 2)  # (B, T_frames, n_mels)
