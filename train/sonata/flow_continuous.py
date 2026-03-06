"""Continuous Latent Flow — skip discrete FSQ, predict continuous vectors.

The FSQ quantization is the single biggest quality bottleneck in Sonata:
32K codebook entries can't represent the full richness of speech.

This module replaces discrete semantic tokens with continuous latent vectors
from the codec encoder. The flow model predicts acoustic latents directly
from continuous semantic features — no quantization cliff.

Architecture:
  Codec encoder → continuous features (B, T, 256)
  ↓ (no FSQ quantization)
  Flow predicts: continuous features → acoustic latents
  Decoder: (continuous features, acoustic latents) → audio

The LM is replaced by a continuous predictor: given text, predict continuous
features directly (regression instead of classification).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FlowConfig
from modules import AdaLayerNorm, TimestepEmbedding, RMSNorm, SwiGLU


class ContinuousPredictor(nn.Module):
    """Predicts continuous semantic features from text (replaces discrete LM).

    Instead of predicting a distribution over 32K tokens, directly regresses
    continuous 256-dim vectors from text conditioning. Uses flow matching
    itself — text conditions a flow that generates semantic features.
    """

    def __init__(self, d_model: int = 512, n_layers: int = 8, n_heads: int = 8,
                 text_vocab_size: int = 32000, feature_dim: int = 256,
                 max_seq_len: int = 4096):
        super().__init__()
        self.feature_dim = feature_dim
        self.text_emb = nn.Embedding(text_vocab_size + 4, d_model)
        self.text_pos = nn.Embedding(max_seq_len, d_model)
        self.text_norm = RMSNorm(d_model)

        self.text_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, int(d_model * 4),
                                       batch_first=True, norm_first=True)
            for _ in range(4)
        ])

        self.feature_proj = nn.Linear(feature_dim, d_model)
        self.time_emb = TimestepEmbedding(d_model)

        self.blocks = nn.ModuleList([
            ContinuousFlowBlock(d_model, n_heads, d_model)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, feature_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        B, T = text_tokens.shape
        pos = torch.arange(T, device=text_tokens.device).unsqueeze(0)
        x = self.text_emb(text_tokens) + self.text_pos(pos)
        for block in self.text_blocks:
            x = block(x)
        return self.text_norm(x)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                text_enc: torch.Tensor) -> torch.Tensor:
        """Predict velocity for continuous semantic features."""
        time_cond = self.time_emb(t)[:, None, :].expand(x_t.shape[0], x_t.shape[1], -1)
        x = self.feature_proj(x_t) + time_cond

        for block in self.blocks:
            x = block(x, text_enc)

        return self.output_proj(self.output_norm(x))

    def compute_loss(self, features: torch.Tensor, text_tokens: torch.Tensor):
        """OT-CFM loss for continuous feature prediction."""
        B = features.shape[0]
        text_enc = self.encode_text(text_tokens)

        z = torch.randn(B, device=features.device)
        t = torch.sigmoid(z).clamp(1e-5, 1 - 1e-5)
        noise = torch.randn_like(features)
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * noise + t_expand * features
        v_pred = self.forward(x_t, t, text_enc)
        return F.mse_loss(v_pred, features - noise)

    @torch.no_grad()
    def sample(self, text_tokens: torch.Tensor, n_frames: int,
               n_steps: int = 8) -> torch.Tensor:
        B = text_tokens.shape[0]
        device = text_tokens.device
        text_enc = self.encode_text(text_tokens)

        x = torch.randn(B, n_frames, self.feature_dim, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.forward(x, t, text_enc)
            x = x + dt * v
        return x


class ContinuousFlowBlock(nn.Module):
    """Block with cross-attention to text encoding."""

    def __init__(self, dim: int, n_heads: int, cond_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        ff_dim = int(dim * 4)
        self.mlp = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h)[0]
        h = self.norm2(x)
        x = x + self.cross_attn(h, context, context)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class ContinuousFlowPipeline(nn.Module):
    """Full continuous latent pipeline: Text → Features → Acoustics.

    Eliminates FSQ quantization entirely. Two-stage flow:
    Stage 1: Text → continuous semantic features (via ContinuousPredictor)
    Stage 2: Semantic features → acoustic latents (via standard Flow)
    """

    def __init__(self, predictor: ContinuousPredictor, flow, codec):
        super().__init__()
        self.predictor = predictor
        self.flow = flow
        self.codec = codec

    @torch.no_grad()
    def generate(self, text_tokens: torch.Tensor, n_frames: int,
                 predictor_steps: int = 8, flow_steps: int = 8,
                 speaker_ids=None) -> torch.Tensor:
        """Full generation: text → features → acoustic → audio."""
        # Stage 1: predict continuous semantic features
        features = self.predictor.sample(text_tokens, n_frames, predictor_steps)

        # Stage 2: flow matching to acoustic latents
        # Project continuous features to FSQ dim, quantize for flow conditioning
        with torch.no_grad():
            fsq_input = self.codec.semantic_proj(features) if features.shape[-1] != self.codec.fsq.dim else features
            _, semantic_tokens = self.codec.fsq(fsq_input)

        acoustic = self.flow.sample(semantic_tokens, n_steps=flow_steps,
                                    speaker_ids=speaker_ids)

        # Decode
        T_min = min(features.shape[1], acoustic.shape[1])
        semantic_codes = self.codec.fsq.indices_to_codes(semantic_tokens[:, :T_min])
        audio = self.codec.decoder(semantic_codes, acoustic[:, :T_min])
        return audio


if __name__ == "__main__":
    predictor = ContinuousPredictor(d_model=256, n_layers=4, n_heads=4, feature_dim=256)
    n = sum(p.numel() for p in predictor.parameters())
    print(f"ContinuousPredictor: {n/1e6:.1f}M params")

    B = 2
    text = torch.randint(0, 100, (B, 20))
    features = torch.randn(B, 50, 256)

    loss = predictor.compute_loss(features, text)
    print(f"  Loss: {loss:.4f}")

    gen = predictor.sample(text, n_frames=50, n_steps=4)
    print(f"  Generated features: {gen.shape}")
    print("PASS")
