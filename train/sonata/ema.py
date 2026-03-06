"""Exponential Moving Average of model weights for inference quality.

The EMA model almost always produces better output than the raw trained model.
Standard in all SOTA TTS systems (Vocos, F5-TTS, CosyVoice, SoundStorm).
"""

import copy
import torch
import torch.nn as nn


class EMA:
    """Maintains an exponential moving average shadow of model parameters.

    Usage:
        ema = EMA(model, decay=0.999)
        # In training loop after optimizer.step():
        ema.update()
        # For validation/export:
        ema.apply_shadow()  # swap in EMA weights
        validate(model)
        ema.restore()       # swap back training weights
        # Or get state dict directly:
        ema_state = ema.state_dict()
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._init_shadow()

    def _init_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)
