"""Mixture of Experts Flow — specialized expert networks for voice diversity.

Instead of one monolithic flow model for all speakers, MoE flow uses
multiple expert FFN networks. A learned router selects 2 experts per token
based on speaker/content features. Each expert can specialize in different
voice characteristics (male/female, age, accent, emotional range).

Same total compute per forward pass, but much higher model capacity.
Switch Transformer-style top-k routing with load balancing loss.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FlowConfig
from modules import AdaLayerNorm, TimestepEmbedding


class ExpertFFN(nn.Module):
    """Single expert feed-forward network."""

    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    """Mixture of Experts with top-k routing and load balancing.

    n_experts total, top_k active per token. Router is a learned linear layer.
    Load balancing loss encourages uniform expert utilization.
    """

    def __init__(self, dim: int, ff_mult: float = 4.0,
                 n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        ff_dim = int(dim * ff_mult)

        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(dim, ff_dim) for _ in range(n_experts)])

        self._aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (B*T, D)

        logits = self.router(x_flat)  # (B*T, n_experts)
        probs = F.softmax(logits, dim=-1)

        top_k_probs, top_k_indices = probs.topk(self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute load balancing auxiliary loss
        # fraction of tokens routed to each expert × average routing probability
        with torch.no_grad():
            mask = F.one_hot(top_k_indices, self.n_experts).float().sum(dim=1)
            f_i = mask.mean(dim=0)                    # fraction routed to expert i
        p_i = probs.mean(dim=0)                       # avg prob for expert i
        self._aux_loss = (f_i * p_i).sum() * self.n_experts

        # Weighted sum of top-k experts
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]
            weight = top_k_probs[:, k:k+1]

            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weight[mask] * expert_output

        return output.view(B, T, D)


class MoEFlowAttention(nn.Module):
    """Bidirectional self-attention (same as standard flow)."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out(out.transpose(1, 2).reshape(B, T, -1))


class MoEFlowBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, cond_dim: int,
                 ff_mult: float = 4.0, eps: float = 1e-5,
                 n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim, cond_dim, eps)
        self.attn = MoEFlowAttention(dim, n_heads)
        self.norm2 = AdaLayerNorm(dim, cond_dim, eps)
        self.moe = MoELayer(dim, ff_mult, n_experts, top_k)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x, cond))
        x = x + self.moe(self.norm2(x, cond))
        return x


class SonataMoEFlow(nn.Module):
    """Flow matching with Mixture of Experts for voice diversity.

    Same architecture as SonataFlow but with MoE FFN layers.
    The router learns to specialize experts for different speaker types.
    """

    def __init__(self, cfg: FlowConfig, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.cfg = cfg
        self.n_experts = n_experts

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
            MoEFlowBlock(cfg.d_model, cfg.n_heads, cfg.d_model,
                         cfg.ff_mult, cfg.norm_eps, n_experts, top_k)
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

    def get_aux_loss(self) -> torch.Tensor:
        """Total load balancing loss across all MoE layers."""
        total = 0.0
        for block in self.blocks:
            total = total + block.moe._aux_loss
        return total / len(self.blocks)

    def compute_loss(self, x_target, semantic_tokens, speaker_ids=None,
                     aux_weight: float = 0.01):
        B = x_target.shape[0]
        z = torch.randn(B, device=x_target.device)
        t = torch.sigmoid(z).clamp(1e-5, 1 - 1e-5)
        noise = torch.randn_like(x_target)
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * noise + t_expand * x_target
        v_pred = self.forward(x_t, t, semantic_tokens, speaker_ids)
        cfm_loss = F.mse_loss(v_pred, x_target - noise)
        return cfm_loss + aux_weight * self.get_aux_loss()

    @torch.no_grad()
    def _velocity(self, x, t, semantic_tokens, speaker_ids=None,
                  emotion_ids=None, prosody_features=None, cfg_scale=1.0,
                  force_uncond=False):
        """Velocity wrapper compatible with adaptive_quality.py interface."""
        return self.forward(x, t, semantic_tokens, speaker_ids)

    @torch.no_grad()
    def sample(self, semantic_tokens, n_steps=8, speaker_ids=None):
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
    model = SonataMoEFlow(cfg, n_experts=8, top_k=2)

    total = sum(p.numel() for p in model.parameters())
    active = total  # top_k/n_experts ratio applies per-token
    print(f"SonataMoEFlow: {total/1e6:.1f}M total, 8 experts, top-2 routing")

    B, T = 2, 50
    sem = torch.randint(0, cfg.semantic_vocab_size, (B, T))
    x = torch.randn(B, T, cfg.acoustic_dim)

    loss = model.compute_loss(x, sem)
    print(f"  Loss (CFM + aux): {loss:.4f}")

    gen = model.sample(sem, n_steps=4)
    print(f"  Sample: {gen.shape}")
    print("PASS")
