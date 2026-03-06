"""SoundStorm — MaskGIT-style parallel masked prediction for semantic tokens.

Replaces the autoregressive LM for 10-50x faster inference. Instead of predicting
one token at a time, predicts ALL tokens simultaneously, then iteratively refines
by re-masking low-confidence predictions.

Architecture: Bidirectional transformer with cross-attention to text encoding.
Training: Masked token prediction (like BERT/MaskGIT).
Inference: Start fully masked → predict all → re-mask lowest confidence → repeat.

Based on: SoundStorm (Google), SpecMaskGIT, G-MLM.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SemanticLMConfig
from modules import RMSNorm, SwiGLU, BidirectionalAttention, CrossAttention


class TextEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, eps: float = 1e-5):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps)
        self.attn = BidirectionalAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model, eps)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SoundStormBlock(nn.Module):
    """Bidirectional block with cross-attention to text + mask-step conditioning."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, eps: float = 1e-5):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps)
        self.self_attn = BidirectionalAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model, eps)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.norm3 = RMSNorm(d_model, eps)
        self.ffn = SwiGLU(d_model, d_ff)
        self.step_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.step_proj.weight)
        nn.init.zeros_(self.step_proj.bias)

    def forward(self, x: torch.Tensor, text_enc: torch.Tensor,
                step_emb: torch.Tensor) -> torch.Tensor:
        x = x + step_emb + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), text_enc)
        x = x + self.ffn(self.norm3(x))
        return x


class SonataStorm(nn.Module):
    """SoundStorm-style parallel masked prediction for semantic tokens.

    Training: randomly mask 10-100% of tokens, predict masked positions.
    Inference: iterative refinement in T steps (typically 8-16).
    """

    MASK_TOKEN = 3  # matches SemanticLMConfig.n_special_tokens MASK=3

    def __init__(self, cfg: SemanticLMConfig, n_text_layers: int = 4):
        super().__init__()
        self.cfg = cfg
        total_semantic = cfg.semantic_vocab_size + cfg.n_special_tokens

        self.text_emb = nn.Embedding(cfg.text_vocab_size + cfg.n_special_tokens, cfg.d_model)
        self.text_pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.text_encoder = nn.ModuleList([
            TextEncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.norm_eps)
            for _ in range(n_text_layers)
        ])
        self.text_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        self.semantic_emb = nn.Embedding(total_semantic, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # Mask step embedding (tells model which iteration we're on)
        self.step_emb = nn.Sequential(
            nn.Linear(1, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, cfg.d_model)
        )

        self.blocks = nn.ModuleList([
            SoundStormBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.norm_eps)
            for _ in range(cfg.n_layers)
        ])

        self.output_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.semantic_vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        B, T = text_tokens.shape
        pos = torch.arange(T, device=text_tokens.device).unsqueeze(0)
        x = self.text_emb(text_tokens) + self.text_pos(pos)
        for block in self.text_encoder:
            x = block(x)
        return self.text_norm(x)

    def forward(self, text_tokens: torch.Tensor, masked_tokens: torch.Tensor,
                mask_ratio: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = masked_tokens.shape
        text_enc = self.encode_text(text_tokens)

        pos = torch.arange(T, device=masked_tokens.device).unsqueeze(0)
        x = self.semantic_emb(masked_tokens) + self.pos_emb(pos)

        if mask_ratio is None:
            mask_ratio = torch.zeros(B, 1, device=x.device)
        step_emb = self.step_emb(mask_ratio.unsqueeze(-1) if mask_ratio.dim() == 1 else mask_ratio.unsqueeze(-1))
        step_emb = step_emb.unsqueeze(1) if step_emb.dim() == 2 else step_emb

        for block in self.blocks:
            x = block(x, text_enc, step_emb)

        return self.head(self.output_norm(x))

    def compute_loss(self, text_tokens: torch.Tensor,
                     target_tokens: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        B, T = target_tokens.shape

        # Random mask ratio per sample (cosine schedule biased toward high masking)
        u = torch.rand(B, device=target_tokens.device)
        mask_ratio = torch.cos(u * math.pi * 0.5)  # [0, 1], biased toward 1.0
        n_mask = (mask_ratio * T).long().clamp(min=1, max=T)

        masked = target_tokens.clone()
        mask = torch.zeros(B, T, dtype=torch.bool, device=target_tokens.device)
        for i in range(B):
            perm = torch.randperm(T, device=target_tokens.device)[:n_mask[i]]
            mask[i, perm] = True
            masked[i, perm] = self.MASK_TOKEN

        logits = self.forward(text_tokens, masked, mask_ratio)

        loss = F.cross_entropy(
            logits[mask].view(-1, logits.size(-1)),
            target_tokens[mask].view(-1),
            ignore_index=0,
        )

        with torch.no_grad():
            preds = logits[mask].argmax(dim=-1)
            acc = (preds == target_tokens[mask]).float().mean().item()

        return loss, {"acc": acc, "mask_ratio": mask_ratio.mean().item()}

    @torch.no_grad()
    def generate(self, text_tokens: torch.Tensor, seq_len: int,
                 n_steps: int = 16, temperature: float = 1.0,
                 cfg_scale: float = 1.0) -> torch.Tensor:
        """Iterative parallel decoding (MaskGIT-style).

        Start fully masked, predict all positions, keep most confident,
        re-mask the rest, repeat for n_steps iterations.
        """
        B = text_tokens.shape[0]
        device = text_tokens.device
        text_enc = self.encode_text(text_tokens)

        tokens = torch.full((B, seq_len), self.MASK_TOKEN, dtype=torch.long, device=device)

        for step in range(n_steps):
            mask_ratio_val = 1.0 - step / n_steps
            mask = (tokens == self.MASK_TOKEN)

            if not mask.any():
                break

            mask_ratio = torch.full((B,), mask_ratio_val, device=device)
            pos = torch.arange(seq_len, device=device).unsqueeze(0)
            x = self.semantic_emb(tokens) + self.pos_emb(pos)
            step_emb = self.step_emb(mask_ratio.unsqueeze(-1)).unsqueeze(1)

            for block in self.blocks:
                x = block(x, text_enc, step_emb)

            logits = self.head(self.output_norm(x))

            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
            else:
                probs = F.one_hot(logits.argmax(-1), logits.size(-1)).float()

            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, seq_len)
            confidence = probs.max(dim=-1).values

            # Only update masked positions
            tokens = torch.where(mask, sampled, tokens)
            confidence = torch.where(mask, confidence, torch.ones_like(confidence) * 1e9)

            # Determine how many to keep (cosine schedule)
            next_ratio = 1.0 - (step + 1) / n_steps
            n_remask = max(int(next_ratio * seq_len), 0)

            if n_remask > 0 and step < n_steps - 1:
                for b in range(B):
                    _, indices = confidence[b].topk(n_remask, largest=False)
                    tokens[b, indices] = self.MASK_TOKEN

        return tokens


if __name__ == "__main__":
    cfg = SemanticLMConfig(d_model=512, n_layers=8, n_heads=8, n_kv_heads=4)
    model = SonataStorm(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"SonataStorm (Parallel Decoder):")
    print(f"  {cfg.n_layers}L × d={cfg.d_model}, H={cfg.n_heads}")
    print(f"  Semantic vocab: {cfg.semantic_vocab_size}")
    print(f"  Total params: {n_params/1e6:.1f}M")

    B, T_text, T_audio = 2, 32, 128
    text = torch.randint(0, 100, (B, T_text))
    target = torch.randint(4, cfg.semantic_vocab_size, (B, T_audio))

    loss, info = model.compute_loss(text, target)
    print(f"\n  Loss: {loss:.4f}, Acc: {info['acc']:.2%}, MaskRatio: {info['mask_ratio']:.2f}")

    generated = model.generate(text, seq_len=T_audio, n_steps=8)
    print(f"  Generated: {generated.shape}, range: [{generated.min()}, {generated.max()}]")
