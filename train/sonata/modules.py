"""Shared modules for Sonata models.

Single source of truth for RMSNorm, SwiGLU, AdaLayerNorm, TimestepEmbedding,
RoPE, and other primitives used across codec, LM, flow, SoundStorm, and STT.
"""

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm: modulates with shift + scale from conditioning."""

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_scale = self.proj(cond)
        shift, scale = shift_scale.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        emb = t[:, None] * freqs[None, :]
        emb = torch.cat([emb.cos(), emb.sin()], dim=-1)
        return self.mlp(emb)


def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor):
    def rotate(x, f):
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        f = f[None, :x_c.shape[1], None, :]
        return torch.view_as_real(x_c * f).flatten(-2).to(x.dtype)
    return rotate(xq, freqs), rotate(xk, freqs)


def _mas_numpy(log_prob_np):
    """MAS core in numpy — ~50-100x faster than torch tensor loops."""
    import numpy as np
    T_text, T_mel = log_prob_np.shape
    Q = np.full((T_text, T_mel), -1e9, dtype=np.float32)
    Q[0, 0] = log_prob_np[0, 0]
    for j in range(1, T_mel):
        Q[0, j] = Q[0, j - 1] + log_prob_np[0, j]
    for i in range(1, T_text):
        for j in range(i, T_mel):
            Q[i, j] = log_prob_np[i, j] + max(Q[i - 1, j - 1], Q[i, j - 1])
    path = np.zeros((T_text, T_mel), dtype=np.float32)
    i, j = T_text - 1, T_mel - 1
    path[i, j] = 1.0
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        elif Q[i - 1, j - 1] >= Q[i, j - 1]:
            i -= 1
            j -= 1
        else:
            j -= 1
        path[i, j] = 1.0
    return path


try:
    import numba
    @numba.jit(nopython=True, cache=True)
    def _mas_numba(log_prob_np):
        """MAS with numba JIT — ~500x faster than torch tensor loops."""
        T_text, T_mel = log_prob_np.shape
        Q = np.full((T_text, T_mel), -1e9, dtype=np.float32)
        Q[0, 0] = log_prob_np[0, 0]
        for j in range(1, T_mel):
            Q[0, j] = Q[0, j - 1] + log_prob_np[0, j]
        for i in range(1, T_text):
            for j in range(i, T_mel):
                Q[i, j] = log_prob_np[i, j] + max(Q[i - 1, j - 1], Q[i, j - 1])
        path = np.zeros((T_text, T_mel), dtype=np.float32)
        i, j = T_text - 1, T_mel - 1
        path[i, j] = 1.0
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            elif Q[i - 1, j - 1] >= Q[i, j - 1]:
                i -= 1
                j -= 1
            else:
                j -= 1
            path[i, j] = 1.0
        return path
    _MAS_BACKEND = 'numba'
except ImportError:
    _MAS_BACKEND = 'numpy'


@torch.no_grad()
def monotonic_alignment_search(log_prob: torch.Tensor) -> torch.Tensor:
    """Monotonic Alignment Search (Glow-TTS / VITS).

    Uses numba JIT if available (~500x faster), falls back to numpy (~50x faster).
    Original pure-torch loop was the bottleneck that killed Flow v3 training at step 5000.

    Args:
        log_prob: (T_text, T_mel) log-probability matrix
    Returns:
        path: (T_text, T_mel) binary alignment matrix
    """
    import numpy as np
    lp = log_prob.detach().cpu().numpy().astype(np.float32)
    if _MAS_BACKEND == 'numba':
        path_np = _mas_numba(lp)
    else:
        path_np = _mas_numpy(lp)
    return torch.from_numpy(path_np).to(log_prob.device)


@torch.no_grad()
def mas_durations(text_enc: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
    """Extract character-level durations via MAS from a batch.

    Uses energy-based soft alignment between text and mel energy contours.
    All computation is detached (no gradients needed for GT durations).

    Args:
        text_enc: (B, T_text, D) text encodings
        mel: (B, T_mel, D_mel) mel spectrogram
    Returns:
        durations: (B, T_text) integer durations per character
    """
    B, T_text, D = text_enc.shape
    T_mel = mel.shape[1]
    dev = text_enc.device
    durations = torch.zeros(B, T_text, dtype=torch.long, device=dev)

    te_detach = text_enc.detach()
    mel_detach = mel.detach()

    for b in range(B):
        n_chars = max(1, int((te_detach[b].abs().sum(-1) > 0).sum().item()))
        mel_energy = mel_detach[b].norm(dim=-1)
        text_energy = te_detach[b, :n_chars].norm(dim=-1)

        log_prob = -torch.abs(text_energy.unsqueeze(1) - mel_energy.unsqueeze(0))

        path = monotonic_alignment_search(log_prob)
        dur = path.sum(dim=1).long()

        total = dur.sum().item()
        if total != T_mel:
            diff = T_mel - total
            dur[-1] = max(1, dur[-1].item() + diff)

        durations[b, :n_chars] = dur

    return durations


def epss_schedule(n_steps: int, sway: float = -1.0) -> List[float]:
    """Non-uniform timestep schedule biased toward t=1 (EPSS / Sway Sampling).

    sway < 0: bias toward t=1 (more steps near noise→signal transition)
    sway = 0: uniform
    sway > 0: bias toward t=0

    For n_steps=8, uses the F5-TTS empirical schedule.
    For other step counts, uses logit-normal CDF quantiles with mu=sway.
    """
    EMPIRICAL_SCHEDULES = {
        4: [0.0, 0.30, 0.58, 0.82, 1.0],
        5: [0.0, 0.22, 0.44, 0.66, 0.85, 1.0],
        6: [0.0, 0.16, 0.34, 0.52, 0.70, 0.86, 1.0],
        7: [0.0, 0.12, 0.26, 0.42, 0.58, 0.72, 0.86, 1.0],
        8: [0.0, 0.12, 0.26, 0.42, 0.5, 0.58, 0.72, 0.86, 1.0],
    }
    if n_steps in EMPIRICAL_SCHEDULES:
        return EMPIRICAL_SCHEDULES[n_steps]

    # sway=0 → uniform
    if sway == 0.0:
        return torch.linspace(0.0, 1.0, n_steps + 1).tolist()

    # Logit-normal CDF quantiles: t = sigmoid(-sway + norm_ppf(p))
    # Use -sway so that sway < 0 biases toward t=1, sway > 0 toward t=0
    p = (torch.arange(n_steps + 1, dtype=torch.float32) + 0.5) / (n_steps + 1)
    norm = torch.distributions.Normal(0.0, 1.0)
    z = norm.icdf(p)
    t = torch.sigmoid(-sway + z)
    # Enforce exact 0 and 1 at endpoints
    t[0] = 0.0
    t[-1] = 1.0
    return t.tolist()


class TokenLevelEmoSteer(nn.Module):
    """Per-token emotion steering via activation addition.

    Instead of a single direction vector for the whole utterance,
    this module identifies "emotional tokens" via attention-weighted
    search and applies steering only to those positions.

    Based on: EmoSteer-TTS (arXiv:2508.03543)
    """

    def __init__(self, d_model: int, n_emotions: int = 12):
        super().__init__()
        self.d_model = d_model
        self.n_emotions = n_emotions
        # Emotion direction bank: (n_emotions, d_model)
        self.directions = nn.Parameter(torch.randn(n_emotions, d_model) * 0.01)
        # Token importance scorer
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        emotion_id: Optional[int] = None,
        emotion_ids: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Apply emotion steering to the most emotionally relevant tokens.

        hidden: (B, T, D)
        emotion_id: int — single emotion for whole batch (used when emotion_ids is None)
        emotion_ids: (B,) — per-sample emotion IDs (overrides emotion_id when provided)
        Returns: hidden + weighted emotion direction
        """
        scores = self.scorer(hidden).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=-1)  # (B, T)
        if emotion_ids is not None:
            direction = self.directions[emotion_ids]  # (B, D)
        else:
            direction = self.directions[emotion_id].unsqueeze(0).expand(
                hidden.size(0), -1
            )  # (B, D)
        steering = scale * weights.unsqueeze(-1) * direction.unsqueeze(1)
        return hidden + steering

    def interpolate(
        self,
        hidden: torch.Tensor,
        emo_a: int,
        emo_b: int,
        alpha: float,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Interpolate between two emotions."""
        dir_interp = (1 - alpha) * self.directions[emo_a] + alpha * self.directions[emo_b]
        scores = self.scorer(hidden).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        steering = scale * weights.unsqueeze(-1) * dir_interp.unsqueeze(0).unsqueeze(0)
        return hidden + steering

    def erase(
        self,
        hidden: torch.Tensor,
        emotion_id: int,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Remove an emotion from the hidden state (subtraction)."""
        return self.forward(hidden, emotion_id=emotion_id, scale=-scale)


class BidirectionalAttention(nn.Module):
    """Full bidirectional self-attention (no causal mask)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out(out.transpose(1, 2).reshape(B, T, -1))


class GQAttention(nn.Module):
    """Grouped-query self-attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, freqs, mask=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        q, k = apply_rope(q, k, freqs)
        if self.n_rep > 1:
            k = k[:, :, :, None, :].expand(B, T, self.n_kv_heads, self.n_rep, self.head_dim)
            k = k.reshape(B, T, self.n_heads, self.head_dim)
            v = v[:, :, :, None, :].expand(B, T, self.n_kv_heads, self.n_rep, self.head_dim)
            v = v.reshape(B, T, self.n_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        is_causal = mask is not None and T > 1
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class CrossAttention(nn.Module):
    """Cross-attention with GQA support."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // self.n_kv_heads
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, context):
        B, T_a, _ = x.shape
        _, T_c, _ = context.shape
        q = self.wq(x).view(B, T_a, self.n_heads, self.head_dim)
        k = self.wk(context).view(B, T_c, self.n_kv_heads, self.head_dim)
        v = self.wv(context).view(B, T_c, self.n_kv_heads, self.head_dim)
        if self.n_rep > 1:
            k = k[:, :, :, None, :].expand(B, T_c, self.n_kv_heads, self.n_rep, self.head_dim)
            k = k.reshape(B, T_c, self.n_heads, self.head_dim)
            v = v[:, :, :, None, :].expand(B, T_c, self.n_kv_heads, self.n_rep, self.head_dim)
            v = v.reshape(B, T_c, self.n_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.wo(out.transpose(1, 2).reshape(B, T_a, -1))


class TrainingLog:
    """Append-only JSONL training log for tracking loss curves."""

    def __init__(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.path = path
        self.f = open(path, "a")

    def log(self, **kwargs):
        import json, time
        kwargs["timestamp"] = time.time()
        self.f.write(json.dumps(kwargs) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def cosine_lr(step: int, warmup: int, max_lr: float, min_lr: float, total: int) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    ratio = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))
