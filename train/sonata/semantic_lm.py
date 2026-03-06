"""Sonata Semantic LM — Cross-attention autoregressive transformer.

Predicts semantic tokens (from Sonata Codec FSQ) conditioned on text.

Architecture: Text encoder (4 layers) + Audio decoder (16 layers with cross-attention).
Text encoder encodes the full input text once. The audio decoder autoregressively
generates semantic tokens, attending to the text encoding via cross-attention.
This replaces the fixed 4:1 text advance heuristic with learned alignment.

~280M parameters total.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SemanticLMConfig
from modules import RMSNorm, SwiGLU, GQAttention, CrossAttention, precompute_rope_freqs


class TextEncoderBlock(nn.Module):
    """Bidirectional text encoder block (no causal mask, no RoPE — uses absolute pos)."""
    def __init__(self, cfg: SemanticLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, batch_first=True,
        )
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x):
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderBlock(nn.Module):
    """Audio decoder block: causal self-attention + optional cross-attention + FFN."""
    def __init__(self, cfg: SemanticLMConfig, use_cross_attention: bool = True):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.cross_attn = CrossAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, text_enc, freqs, mask=None):
        x = x + self.attn(self.attn_norm(x), freqs, mask)
        if self.use_cross_attention and text_enc is not None:
            x = x + self.cross_attn(self.cross_norm(x), text_enc)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ─── Sonata Semantic LM ──────────────────────────────────────────────────────

PROSODY_DIM = 3  # (log_pitch, energy, speaking_rate)
TEXT_ENCODER_LAYERS = 4


class SonataSemanticLM(nn.Module):
    """Cross-attention LM: text encoder + autoregressive audio decoder.

    The text encoder processes the full text bidirectionally.
    The audio decoder generates semantic tokens autoregressively,
    attending to the text encoding via cross-attention at every layer.
    This replaces the fixed 4:1 text advance with learned alignment.

    Optional prosody conditioning: when prosody_features are provided,
    the model conditions generation on speaking rate, pitch, and energy.
    """

    def __init__(self, cfg: SemanticLMConfig, use_prosody: bool = False,
                 use_cross_attention: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_prosody = use_prosody
        self.use_cross_attention = use_cross_attention
        total_text = cfg.text_vocab_size + cfg.n_special_tokens
        total_semantic = cfg.semantic_vocab_size + cfg.n_special_tokens

        self.text_emb = nn.Embedding(total_text, cfg.d_model)

        if use_cross_attention:
            self.text_pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            self.text_encoder = nn.ModuleList([
                TextEncoderBlock(cfg) for _ in range(TEXT_ENCODER_LAYERS)
            ])
            self.text_encoder_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        self.semantic_emb = nn.Embedding(total_semantic, cfg.d_model)

        if use_prosody:
            self.prosody_proj = nn.Sequential(
                nn.Linear(PROSODY_DIM, cfg.d_model),
                nn.SiLU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )

        self.layers = nn.ModuleList([
            DecoderBlock(cfg, use_cross_attention=use_cross_attention)
            for _ in range(cfg.n_layers)
        ])
        self.output_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        self.semantic_head = nn.Linear(cfg.d_model, cfg.semantic_vocab_size, bias=False)

        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Encode text bidirectionally. Only used with cross-attention."""
        if not self.use_cross_attention:
            return self.text_emb(text_tokens)
        B, T_text = text_tokens.shape
        pos = torch.arange(T_text, device=text_tokens.device).unsqueeze(0)
        x = self.text_emb(text_tokens) + self.text_pos_emb(pos)
        for block in self.text_encoder:
            x = block(x)
        return self.text_encoder_norm(x)

    def forward(
        self,
        text_tokens: torch.Tensor,       # (B, T_text) — text input
        semantic_tokens: torch.Tensor,    # (B, T_audio) — previous semantic tokens (shifted right)
        target_semantic: Optional[torch.Tensor] = None,  # (B, T_audio) — for loss
        prosody_features: Optional[torch.Tensor] = None,  # (B, T_audio, 3)
        text_encoded: Optional[torch.Tensor] = None,  # pre-computed text encoding
    ):
        B, T_audio = semantic_tokens.shape

        if self.use_cross_attention:
            if text_encoded is None:
                text_enc = self.encode_text(text_tokens)
            else:
                text_enc = text_encoded
            x = self.semantic_emb(semantic_tokens)
            if self.use_prosody and prosody_features is not None:
                x = x + self.prosody_proj(prosody_features)
            T_total = T_audio
        else:
            text_enc = None
            text_x = self.text_emb(text_tokens)
            sem_x = self.semantic_emb(semantic_tokens)
            if self.use_prosody and prosody_features is not None:
                sem_x = sem_x + self.prosody_proj(prosody_features)
            x = torch.cat([text_x, sem_x], dim=1)
            T_total = x.shape[1]

        freqs = self.rope_freqs[:T_total]
        causal_mask = torch.tril(torch.ones(T_total, T_total, device=x.device, dtype=torch.bool)) if T_total > 1 else None
        for layer in self.layers:
            x = layer(x, text_enc, freqs, mask=causal_mask)

        if not self.use_cross_attention:
            T_text = text_tokens.shape[1]
            x = x[:, T_text:]

        hidden = self.output_norm(x)
        semantic_logits = self.semantic_head(hidden)

        losses = {}
        if target_semantic is not None:
            losses["semantic"] = F.cross_entropy(
                semantic_logits.reshape(-1, semantic_logits.size(-1)),
                target_semantic.reshape(-1),
                ignore_index=-1,
            )

        return semantic_logits, losses


if __name__ == "__main__":
    cfg = SemanticLMConfig()
    model = SonataSemanticLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    # Count sub-components
    enc_params = sum(p.numel() for n, p in model.named_parameters() if 'text_encoder' in n or 'text_emb' in n or 'text_pos' in n)
    dec_params = n_params - enc_params

    print(f"Sonata Semantic LM (Cross-Attention):")
    print(f"  Text encoder: {TEXT_ENCODER_LAYERS}L, {enc_params/1e6:.1f}M params")
    print(f"  Audio decoder: {cfg.n_layers}L × d={cfg.d_model}, {dec_params/1e6:.1f}M params")
    print(f"  FF dim: {cfg.d_ff}")
    print(f"  Text vocab: {cfg.text_vocab_size}, Semantic vocab: {cfg.semantic_vocab_size}")
    print(f"  Total params: {n_params/1e6:.1f}M")

    B, T_text, T_audio = 2, 32, 128
    text = torch.randint(0, cfg.text_vocab_size, (B, T_text))
    semantic = torch.randint(0, cfg.semantic_vocab_size, (B, T_audio))
    target = torch.randint(0, cfg.semantic_vocab_size, (B, T_audio))

    logits, losses = model(text, semantic, target_semantic=target)
    print(f"\n  logits: {logits.shape}")
    print(f"  loss: {losses['semantic']:.4f}")

    # Verify text encoding caching works
    text_enc = model.encode_text(text)
    logits2, _ = model(text, semantic, text_encoded=text_enc)
    assert torch.allclose(logits, logits2, atol=1e-5)
    print(f"  Text encoding cache: PASS")
