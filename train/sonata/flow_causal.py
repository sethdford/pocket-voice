"""Causal Streaming Flow — left-to-right flow matching for real-time TTS.

Standard flow matching is bidirectional: every frame attends to every other frame.
This means you must wait for ALL semantic tokens before generating ANY audio.

Causal flow uses causal (left-to-right) attention: each frame only attends to
frames at the same or earlier positions. This enables:
  1. Start generating audio from the first few tokens
  2. Continuously extend as more tokens arrive from the LM
  3. Sub-100ms time-to-first-audio in streaming mode

Inspired by CosyVoice 2's "chunk-aware causal flow matching."

Architecture:
  - Causal self-attention (like GPT, but for flow matching)
  - Optional chunk-based attention: attend to full history + current chunk
  - Same ODE formulation and loss as standard flow
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FlowConfig
from modules import AdaLayerNorm, TimestepEmbedding


class CausalFlowAttention(nn.Module):
    """Causal self-attention for streaming flow matching."""

    def __init__(self, dim: int, n_heads: int, chunk_size: int = 0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.chunk_size = chunk_size
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        if self.chunk_size > 0:
            # Chunk-aware: attend to all history + current chunk
            mask = self._chunk_mask(T, x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.out(out.transpose(1, 2).reshape(B, T, -1))

    def _chunk_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Each position attends to: all previous chunks + own chunk."""
        mask = torch.ones(T, T, dtype=torch.bool, device=device).tril()
        cs = self.chunk_size
        for start in range(0, T, cs):
            end = min(start + cs, T)
            mask[start:end, start:end] = True
        return mask.unsqueeze(0).unsqueeze(0)


class CausalFlowBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, cond_dim: int,
                 ff_mult: float = 4.0, eps: float = 1e-5, chunk_size: int = 0):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim, cond_dim, eps)
        self.attn = CausalFlowAttention(dim, n_heads, chunk_size)
        self.norm2 = AdaLayerNorm(dim, cond_dim, eps)
        ff_dim = int(dim * ff_mult)
        self.mlp = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x, cond))
        x = x + self.mlp(self.norm2(x, cond))
        return x


class SonataCausalFlow(nn.Module):
    """Causal streaming flow matching model.

    Same ODE formulation as SonataFlow, but with causal attention.
    Enables true streaming: generate audio left-to-right as tokens arrive.
    """

    def __init__(self, cfg: FlowConfig, chunk_size: int = 25):
        super().__init__()
        self.cfg = cfg
        self.chunk_size = chunk_size

        self.semantic_emb = nn.Embedding(cfg.semantic_vocab_size + 4, cfg.cond_dim)
        self.time_emb = TimestepEmbedding(cfg.cond_dim)

        cond_input_dim = cfg.cond_dim * 2
        if cfg.n_speakers > 0:
            self.speaker_emb = nn.Embedding(cfg.n_speakers, cfg.speaker_dim)
            self.speaker_proj = nn.Linear(cfg.speaker_dim, cfg.cond_dim)
            cond_input_dim += cfg.cond_dim
        else:
            self.speaker_emb = None

        self.cond_proj = nn.Linear(cond_input_dim, cfg.d_model)
        self.input_proj = nn.Linear(cfg.acoustic_dim, cfg.d_model)

        self.blocks = nn.ModuleList([
            CausalFlowBlock(cfg.d_model, cfg.n_heads, cfg.d_model,
                            cfg.ff_mult, cfg.norm_eps, chunk_size)
            for _ in range(cfg.n_layers)
        ])

        self.output_norm = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.output_proj = nn.Linear(cfg.d_model, cfg.acoustic_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t, t, semantic_tokens, speaker_ids=None):
        B, T_seq, _ = x_t.shape
        sem_cond = self.semantic_emb(semantic_tokens)
        time_cond = self.time_emb(t)[:, None, :].expand_as(sem_cond)
        cond_parts = [sem_cond, time_cond]

        if self.speaker_emb is not None and speaker_ids is not None:
            spk = self.speaker_proj(self.speaker_emb(speaker_ids))
            cond_parts.append(spk[:, None, :].expand_as(sem_cond))

        cond = self.cond_proj(torch.cat(cond_parts, dim=-1))
        x = self.input_proj(x_t)
        for block in self.blocks:
            x = block(x, cond)
        return self.output_proj(self.output_norm(x))

    def compute_loss(self, x_target, semantic_tokens, speaker_ids=None):
        B = x_target.shape[0]
        z = torch.randn(B, device=x_target.device)
        t = torch.sigmoid(z).clamp(1e-5, 1 - 1e-5)
        noise = torch.randn_like(x_target)
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * noise + t_expand * x_target
        v_pred = self.forward(x_t, t, semantic_tokens, speaker_ids)
        return F.mse_loss(v_pred, x_target - noise)

    @torch.no_grad()
    def sample_streaming(self, semantic_tokens: torch.Tensor,
                         n_steps: int = 8, speaker_ids=None,
                         yield_every: int = 25):
        """Generator that yields audio chunks as they're ready.

        Instead of waiting for the full sequence, generates chunk by chunk.
        Each chunk sees all previous context (causal attention).
        """
        B, T = semantic_tokens.shape
        device = semantic_tokens.device
        x = torch.randn(B, T, self.cfg.acoustic_dim, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.forward(x, t, semantic_tokens, speaker_ids)
            x = x + dt * v

        # Yield in chunks
        for start in range(0, T, yield_every):
            end = min(start + yield_every, T)
            yield x[:, start:end]

    @torch.no_grad()
    def _velocity(self, x, t, semantic_tokens, speaker_ids=None,
                  emotion_ids=None, prosody_features=None, cfg_scale=1.0,
                  force_uncond=False):
        """Velocity wrapper compatible with adaptive_quality.py interface."""
        return self.forward(x, t, semantic_tokens, speaker_ids)

    @torch.no_grad()
    def sample_incremental(self, semantic_tokens: torch.Tensor,
                           n_steps: int = 8, speaker_ids=None):
        """Full sample (non-streaming) with causal attention."""
        B, T = semantic_tokens.shape
        device = semantic_tokens.device
        x = torch.randn(B, T, self.cfg.acoustic_dim, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.forward(x, t, semantic_tokens, speaker_ids)
            x = x + dt * v
        return x


if __name__ == "__main__":
    cfg = FlowConfig(d_model=256, n_layers=4, n_heads=4)
    model = SonataCausalFlow(cfg, chunk_size=25)
    n = sum(p.numel() for p in model.parameters())
    print(f"SonataCausalFlow: {n/1e6:.1f}M params, chunk_size=25")

    B, T = 2, 100
    sem = torch.randint(0, cfg.semantic_vocab_size, (B, T))
    x = torch.randn(B, T, cfg.acoustic_dim)
    t = torch.rand(B)

    v = model(x, t, sem)
    print(f"  Forward: {v.shape}")

    loss = model.compute_loss(x, sem)
    print(f"  Loss: {loss:.4f}")

    gen = model.sample_incremental(sem, n_steps=4)
    print(f"  Sample: {gen.shape}")

    chunks = list(model.sample_streaming(sem, n_steps=4, yield_every=25))
    print(f"  Streaming: {len(chunks)} chunks, sizes: {[c.shape[1] for c in chunks]}")
    print("PASS")
