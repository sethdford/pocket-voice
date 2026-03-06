"""Perceptual loss modules for Sonata Flow v3 mel-space training.

Provides two loss types:

  1. PerceptualMelLoss: For flow training where we only have mels. Combines L1,
     delta features (temporal derivatives), spectral convergence, and band-weighted
     loss emphasizing the formant region (300-3000 Hz). Used directly on pred/target
     mel spectrograms (B, T, 80).
  2. WavLMPerceptualLoss: For joint fine-tuning when waveforms are available.
     Extracts WavLM features from layers 4, 8, 12 and computes L1. Lazy-loads
     microsoft/wavlm-base. Use with vocoder output (mel → waveform) for perceptual
     refinement.

Both support temporal masking for padded batches and are MPS-efficient.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PerceptualMelLoss", "WavLMPerceptualLoss"]


def _masked_mean(
    x: torch.Tensor, mask: Optional[torch.Tensor], dim: Optional[tuple] = None
) -> torch.Tensor:
    """Compute mean over masked elements. mask: same shape as x, 1=valid 0=pad."""
    if mask is None:
        return x.mean()
    if dim is None:
        dim = tuple(range(x.dim()))
    mask = mask.expand_as(x)
    n = mask.sum().clamp(min=1.0)
    return (x * mask).sum() / n


def _safe_spectral_convergence(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-8
) -> torch.Tensor:
    """Spectral convergence: ||pred - target||_F / (||target||_F + eps)."""
    diff = pred - target
    if mask is not None:
        diff = diff * mask
        target = target * mask
    n = pred.numel() if mask is None else mask.sum().clamp(min=1.0)
    sc_num = (diff.pow(2).sum() / n).sqrt()
    sc_den = (target.pow(2).sum() / n).sqrt().clamp(min=eps)
    return sc_num / sc_den


class PerceptualMelLoss(nn.Module):
    """Mel-space perceptual loss for flow training.

    Combines:
      - L1 mel loss
      - Delta feature loss (1st and 2nd order temporal derivatives)
      - Spectral convergence
      - Band-weighted loss (emphasize formant region, mel bins ~10-50)

    Designed for (B, T, n_mels) log-mel spectrograms. Efficient on MPS.
    """

    def __init__(
        self,
        n_mels: int = 80,
        formant_bin_start: int = 10,
        formant_bin_end: int = 50,
        weight_l1: float = 1.0,
        weight_delta: float = 0.5,
        weight_delta2: float = 0.25,
        weight_spectral_convergence: float = 0.5,
        weight_band: float = 1.0,
        formant_weight: float = 2.0,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.formant_bin_start = formant_bin_start
        self.formant_bin_end = formant_bin_end
        self.weight_l1 = weight_l1
        self.weight_delta = weight_delta
        self.weight_delta2 = weight_delta2
        self.weight_spectral_convergence = weight_spectral_convergence
        self.weight_band = weight_band
        self.formant_weight = formant_weight

        # Band weights: higher for formant region (F1-F3, ~300-3000 Hz)
        band_weights = torch.ones(n_mels)
        band_weights[formant_bin_start:formant_bin_end] = formant_weight
        self.register_buffer("band_weights", band_weights)

    def forward(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
        mel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute perceptual mel loss.

        Args:
            pred_mel: (B, T, n_mels) predicted log-mel spectrogram
            target_mel: (B, T, n_mels) target log-mel spectrogram
            mel_mask: (B, T) optional mask, 1.0=valid 0.0=padded

        Returns:
            Scalar loss (weighted sum of components)
        """
        B, T, D = pred_mel.shape
        if mel_mask is not None:
            mask_3d = mel_mask.unsqueeze(-1)  # (B, T, 1)
        else:
            mask_3d = None

        # 1. L1 mel loss
        l1 = F.l1_loss(pred_mel, target_mel, reduction="none")
        l1_loss = _masked_mean(l1, mask_3d)

        # 2. Delta feature loss (temporal derivatives)
        # 1st order: diff along time
        delta_pred = pred_mel[:, 1:] - pred_mel[:, :-1]  # (B, T-1, D)
        delta_target = target_mel[:, 1:] - target_mel[:, :-1]
        if mel_mask is not None:
            delta_mask = mel_mask[:, 1:] * mel_mask[:, :-1]  # valid where both neighbors valid
            delta_mask = delta_mask.unsqueeze(-1)
        else:
            delta_mask = None
        delta_loss = _masked_mean(F.l1_loss(delta_pred, delta_target, reduction="none"), delta_mask)

        # 2nd order delta
        delta2_pred = delta_pred[:, 1:] - delta_pred[:, :-1]
        delta2_target = delta_target[:, 1:] - delta_target[:, :-1]
        if mel_mask is not None:
            m2 = mel_mask[:, 2:] * mel_mask[:, 1:-1] * mel_mask[:, :-2]
            delta2_mask = m2.unsqueeze(-1)
        else:
            delta2_mask = None
        delta2_loss = _masked_mean(
            F.l1_loss(delta2_pred, delta2_target, reduction="none"), delta2_mask
        )

        # 3. Spectral convergence
        sc_loss = _safe_spectral_convergence(pred_mel, target_mel, mask_3d)

        # 4. Band-weighted loss (formant emphasis)
        w = self.band_weights.to(pred_mel.device).view(1, 1, -1)
        band_l1 = _masked_mean(
            F.l1_loss(pred_mel, target_mel, reduction="none") * w, mask_3d
        )
        band_loss = band_l1 / w.mean()  # normalize so scale matches plain L1

        # 5. Energy envelope loss (loudness contour matching)
        pred_energy = pred_mel.mean(dim=-1)
        target_energy = target_mel.mean(dim=-1)
        energy_loss = _masked_mean(
            F.l1_loss(pred_energy, target_energy, reduction="none"),
            mel_mask if mel_mask is not None else None,
        )

        total = (
            self.weight_l1 * l1_loss
            + self.weight_delta * delta_loss
            + self.weight_delta2 * delta2_loss
            + self.weight_spectral_convergence * sc_loss
            + self.weight_band * band_loss
            + 0.5 * energy_loss
        )
        return total


class WavLMPerceptualLoss(nn.Module):
    """WavLM-based perceptual loss for waveform-level fine-tuning.

    Extracts features from intermediate WavLM layers (4, 8, 12) and computes L1.
    Model is lazy-loaded on first forward. Parameters are frozen (requires_grad=False).
    Gradients flow through pred_audio for training. Use when you have waveforms
    (e.g., from vocoder) for joint flow+vocoder training or vocoder-only refinement.

    Expects 16 kHz mono audio: (B, T) or (B, 1, T). If stereo/multi-channel,
    takes first channel. Input is normalized (zero-mean unit-var) before WavLM.
    """

    LAYERS = (4, 8, 12)
    SAMPLE_RATE = 16000

    def __init__(
        self,
        model_id: str = "microsoft/wavlm-base",
        layers: tuple[int, ...] = (4, 8, 12),
        weight_per_layer: Optional[list[float]] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.layers = layers
        self._model = None
        if weight_per_layer is not None:
            self.register_buffer("layer_weights", torch.tensor(weight_per_layer, dtype=torch.float32))
        else:
            self.register_buffer("layer_weights", torch.ones(len(layers)) / len(layers))

    def _ensure_model(self, device: torch.device):
        if self._model is not None:
            return
        try:
            from transformers import WavLMModel
        except ImportError:
            raise ImportError(
                "WavLMPerceptualLoss requires transformers. Install with: pip install transformers"
            )
        self._model = WavLMModel.from_pretrained(self.model_id)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False
        if str(device) != "cpu":
            self._model = self._model.to(device)

    def _normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Zero-mean unit-variance per sequence (WavLM expects normalized input)."""
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        mean = audio.mean(dim=1, keepdim=True)
        std = audio.std(dim=1, keepdim=True).clamp(min=1e-5)
        return (audio - mean) / std

    def _extract_features(
        self, audio: torch.Tensor, with_grad: bool
    ) -> list[torch.Tensor]:
        """Extract WavLM hidden states for target layers.

        Args:
            audio: (B, T) or (B, 1, T), 16 kHz expected
            with_grad: if True, gradients flow to audio (for pred); if False, no_grad (for target)
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        B, T = audio.shape
        device = audio.device

        self._ensure_model(device)

        audio = self._normalize_audio(audio)
        if T < 320:  # ~20 ms minimum for WavLM
            audio = F.pad(audio, (0, 320 - T))

        if with_grad:
            outputs = self._model(audio, output_hidden_states=True)
        else:
            with torch.no_grad():
                outputs = self._model(audio, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        return [hidden_states[i] for i in self.layers]

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute WavLM perceptual loss.

        Args:
            pred_audio: (B, T) or (B, 1, T) predicted waveform, 16 kHz
            target_audio: (B, T) or (B, 1, T) target waveform, 16 kHz
            mask: (B, T) or (B, T') optional temporal mask. If T' matches WavLM
                  output frames (~50 Hz), used directly. Else downsampled to frame rate.
                  Pass None for no masking.

        Returns:
            Scalar L1 loss on WavLM features
        """
        pred_feats = self._extract_features(pred_audio, with_grad=True)
        target_feats = self._extract_features(target_audio, with_grad=False)

        losses = []
        weights = self.layer_weights.to(pred_audio.device)
        for i, (pf, tf) in enumerate(zip(pred_feats, target_feats)):
            tf = tf.detach()  # ensure no grad to target
            if mask is not None and mask.dim() >= 2:
                T_out = pf.shape[1]
                T_in = mask.shape[1]
                if T_in != T_out:
                    mask_f = mask.float().unsqueeze(1)
                    mask_f = F.interpolate(mask_f, size=T_out, mode="nearest")
                    mask_f = mask_f.squeeze(1).unsqueeze(-1)
                else:
                    mask_f = mask.unsqueeze(-1).expand_as(pf)
                n = mask_f.sum().clamp(min=1.0)
                l = ((pf - tf).abs() * mask_f).sum() / n
            else:
                l = F.l1_loss(pf, tf)
            losses.append(l * weights[i])
        return sum(losses)
