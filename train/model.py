"""
Mimi-Lite TTS Language Model — Llama-style transformer for NeuCodec token prediction.

Architecture (~200M params):
  - 16 layers, d_model=1024, n_heads=16, GQA with 4 KV heads
  - RoPE positional encoding, RMSNorm, SwiGLU FFN
  - Cross-attention every 4 layers (layers 3, 7, 11, 15) for speaker conditioning
  - Input: text_embedding(text_token) + audio_embedding(audio_token)
  - Output: Linear(1024, 65536) → softmax over FSQ vocabulary

Token flow:
  Step t: input = embed_text(text_t) + embed_audio(audio_{t-1})
          → 16-layer transformer → logits → sample → audio_t
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MimiLiteConfig:
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4          # GQA: 4 KV heads shared across 16 query heads
    ffn_mult: float = 2.667      # SwiGLU: d_ff = d_model * ffn_mult * 2/3 ≈ 2730
    max_seq_len: int = 2048      # 2048 tokens = ~40s of audio at 50 tok/s
    text_vocab_size: int = 32000
    audio_vocab_size: int = 65536
    n_special_tokens: int = 4    # PAD=0, BOS=1, EOS=2, TEXT_SEP=3
    cross_attn_layers: tuple = (3, 7, 11, 15)
    speaker_dim: int = 2048      # NeuCodec encoder output dimension
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0

    @property
    def d_ff(self) -> int:
        raw = int(self.d_model * self.ffn_mult)
        return raw - (raw % 256)  # align to 256 for GPU efficiency

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def param_count_estimate(self) -> int:
        d, L, ff = self.d_model, self.n_layers, self.d_ff
        n_h, n_kv = self.n_heads, self.n_kv_heads
        hd = self.head_dim
        sa = d * n_h * hd + 2 * d * n_kv * hd + d * n_h * hd  # QKV + O
        ffn = 3 * d * ff  # gate + up + down (SwiGLU)
        per_layer = sa + ffn + 2 * d  # + 2 norms
        ca_layers = len(self.cross_attn_layers)
        ca = ca_layers * (d * n_h * hd + 2 * d * n_kv * hd + d * n_h * hd + 2 * d)
        emb = self.text_vocab_size * d + self.audio_vocab_size * d
        head = self.audio_vocab_size * d
        return L * per_layer + ca + emb + head


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor):
    B, T, H, D = xq.shape
    xq_c = torch.view_as_complex(xq.float().reshape(B, T, H, D // 2, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(B, T, xk.shape[2], D // 2, 2))
    freqs = freqs[:T].unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class GQAttention(nn.Module):
    """Grouped-Query Attention with KV cache support."""

    def __init__(self, cfg: MimiLiteConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                cache: Optional[tuple] = None) -> tuple:
        B, T, _ = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs)

        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        new_cache = (k, v)

        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(
                B, k.shape[1], self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(
                B, v.shape[1], self.n_heads, self.head_dim)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores.float(), dim=-1).type_as(q)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(out), new_cache


class CrossAttention(nn.Module):
    """Cross-attention for speaker conditioning."""

    def __init__(self, cfg: MimiLiteConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.speaker_dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.speaker_dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        residual = x
        x = self.norm(x)

        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.wk(cond).reshape(B, cond.shape[1], self.n_kv_heads, self.head_dim)
        v = self.wv(cond).reshape(B, cond.shape[1], self.n_kv_heads, self.head_dim)

        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(
                B, k.shape[1], self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(
                B, v.shape[1], self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores.float(), dim=-1).type_as(q)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        return residual + self.wo(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: MimiLiteConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, cfg: MimiLiteConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLUFFN(cfg)

        self.cross_attn = None
        if layer_idx in cfg.cross_attn_layers:
            self.cross_attn = CrossAttention(cfg)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                speaker_cond: Optional[torch.Tensor] = None,
                cache: Optional[tuple] = None) -> tuple:
        h, new_cache = self.attn(self.attn_norm(x), freqs, mask, cache)
        x = x + h
        if self.cross_attn is not None and speaker_cond is not None:
            x = self.cross_attn(x, speaker_cond)
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class MimiLiteLM(nn.Module):
    """TTS Language Model: text + audio tokens → next audio token prediction."""

    def __init__(self, cfg: MimiLiteConfig):
        super().__init__()
        self.cfg = cfg

        self.text_emb = nn.Embedding(cfg.text_vocab_size + cfg.n_special_tokens, cfg.d_model)
        self.audio_emb = nn.Embedding(cfg.audio_vocab_size + cfg.n_special_tokens, cfg.d_model)
        self.speaker_proj = nn.Linear(cfg.speaker_dim, cfg.d_model, bias=False)

        self.layers = nn.ModuleList([
            TransformerBlock(i, cfg) for i in range(cfg.n_layers)
        ])

        self.output_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.output_head = nn.Linear(cfg.d_model, cfg.audio_vocab_size, bias=False)

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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        text_tokens: torch.Tensor,       # (B, T) text token IDs
        audio_tokens: torch.Tensor,       # (B, T) audio token IDs (shifted right)
        speaker_emb: Optional[torch.Tensor] = None,  # (B, S, speaker_dim) NeuCodec encoder output
        cache: Optional[list] = None,
    ) -> tuple:
        B, T = text_tokens.shape

        x = self.text_emb(text_tokens) + self.audio_emb(audio_tokens)

        if speaker_emb is not None:
            speaker_cond = self.speaker_proj(speaker_emb)  # (B, S, d_model)
            # Expand speaker_dim to d_model for cross-attention compatibility
            # The CrossAttention module uses speaker_dim for K,V projections,
            # so we pass the original speaker_emb there
            speaker_for_ca = speaker_emb
        else:
            speaker_cond = None
            speaker_for_ca = None

        freqs = self.rope_freqs

        if cache is not None:
            start_pos = cache[0][0].shape[1] if cache[0] is not None else 0
            freqs = freqs[start_pos:start_pos + T]
        else:
            freqs = freqs[:T]

        mask = None
        if T > 1:
            mask = torch.full((T, T), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)
            if cache is not None and cache[0] is not None:
                cache_len = cache[0][0].shape[1]
                mask = torch.cat([torch.zeros(T, cache_len, device=x.device), mask], dim=-1)
            mask = mask.unsqueeze(0).unsqueeze(0)

        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = layer(x, freqs, mask, speaker_for_ca, layer_cache)
            new_caches.append(new_cache)

        x = self.output_norm(x)
        logits = self.output_head(x)

        return logits, new_caches

    @torch.no_grad()
    def generate(
        self,
        text_tokens: torch.Tensor,       # (B, T_text) full text sequence
        speaker_emb: Optional[torch.Tensor] = None,
        max_audio_tokens: int = 1000,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> list[int]:
        """Autoregressive generation of audio tokens from text."""
        device = text_tokens.device
        B = text_tokens.shape[0]
        assert B == 1, "batch generation not supported yet"

        audio_tokens = [1]  # BOS
        text_pos = 0
        cache = [None] * self.cfg.n_layers

        for step in range(max_audio_tokens):
            t_tok = text_tokens[0, min(text_pos, text_tokens.shape[1] - 1)].unsqueeze(0).unsqueeze(0)
            a_tok = torch.tensor([[audio_tokens[-1]]], device=device)

            logits, cache = self.forward(t_tok, a_tok, speaker_emb, cache)
            logits = logits[0, -1] / temperature

            if top_k > 0:
                topk_vals, topk_idx = logits.topk(top_k)
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(0, topk_idx, topk_vals)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token == 2:  # EOS
                break

            audio_tokens.append(next_token)

            # Advance text pointer heuristically (1 text token per ~5 audio tokens)
            if step > 0 and step % 5 == 0 and text_pos < text_tokens.shape[1] - 1:
                text_pos += 1

        return audio_tokens[1:]  # strip BOS


def create_model(cfg: Optional[MimiLiteConfig] = None) -> MimiLiteLM:
    if cfg is None:
        cfg = MimiLiteConfig()
    model = MimiLiteLM(cfg)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"MimiLiteLM: {param_count:,} params ({param_count/1e6:.1f}M)")
    print(f"  Config estimate: {cfg.param_count_estimate():,}")
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}")
    print(f"  n_kv_heads={cfg.n_kv_heads}, d_ff={cfg.d_ff}, head_dim={cfg.head_dim}")
    print(f"  cross_attn at layers: {cfg.cross_attn_layers}")
    return model


if __name__ == "__main__":
    model = create_model()

    # Quick forward pass test
    B, T = 2, 100
    text = torch.randint(0, 32000, (B, T))
    audio = torch.randint(0, 65536, (B, T))
    speaker = torch.randn(B, 10, 2048)  # 10 frames of NeuCodec encoder output

    logits, _ = model(text, audio, speaker)
    print(f"\nForward pass: text={text.shape}, audio={audio.shape} → logits={logits.shape}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
