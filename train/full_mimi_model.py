"""Full Mimi TTS Model — Main Transformer + DepFormer for multi-codebook prediction.

Architecture matches Kyutai Moshi TTS at a more compact scale:
  Main Transformer: 24 layers, d=1024, 16 heads, 4 KV heads (GQA), SwiGLU, RoPE
  DepFormer: 6 layers, d=512, 8 heads, processes codebooks sequentially

Flow per step:
  1. Embed: text_emb(prev_text) + sum(audio_embs[k](prev_audio[k]) for k in 0..n_q)
  2. Main transformer → hidden state
  3. text_head(hidden) → text logits
  4. DepFormer: hidden → codebook[0] → codebook[1] → ... → codebook[n_q-1]

Training: teacher-forced on (text_tokens, audio_codes[n_q][T]) pairs from Mimi-encoded audio.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FullMimiConfig:
    # Main transformer
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    ffn_mult: float = 2.667
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    # DepFormer
    dep_d_model: int = 512
    dep_n_layers: int = 6
    dep_n_heads: int = 8
    dep_n_kv_heads: int = 4

    # Vocabulary
    text_vocab_size: int = 32000
    audio_vocab_size: int = 2048    # Per-codebook (Mimi RVQ bins)
    n_codebooks: int = 8            # Number of RVQ codebooks
    n_special_tokens: int = 4       # PAD=0, BOS=1, EOS=2, MASK=3
    audio_pad_token: int = 0

    # Speaker conditioning
    speaker_dim: int = 512          # Mimi encoder output dim
    cross_attn_layers: list = field(default_factory=lambda: [5, 11, 17, 23])

    # Interleaver
    text_audio_delay: int = 25      # ~2s delay between text and audio

    dropout: float = 0.0

    @property
    def d_ff(self) -> int:
        raw = int(self.d_model * self.ffn_mult)
        return raw - (raw % 256)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def dep_d_ff(self) -> int:
        raw = int(self.dep_d_model * self.ffn_mult)
        return raw - (raw % 128)

    def param_count_estimate(self) -> dict:
        d, L = self.d_model, self.n_layers
        ff = self.d_ff
        # Main transformer
        main_attn = L * (d * d + 2 * d * (d // (self.n_heads // self.n_kv_heads)) + d * d)
        main_ffn = L * (2 * d * ff + ff * d)
        main_emb = (self.text_vocab_size + self.n_special_tokens) * d + \
                    self.n_codebooks * (self.audio_vocab_size + self.n_special_tokens) * d
        main_total = main_attn + main_ffn + main_emb

        # DepFormer
        dd, dL = self.dep_d_model, self.dep_n_layers
        dff = self.dep_d_ff
        dep_total = dL * (dd * dd * 4 + 2 * dd * dff + dff * dd)
        dep_total += self.n_codebooks * (d * dd + dd * self.audio_vocab_size)

        return {
            "main_transformer": main_total,
            "depformer": dep_total * self.n_codebooks,
            "total": main_total + dep_total * self.n_codebooks,
        }


# ─── Shared Components ──────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


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


class GQAttention(nn.Module):
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

    def forward(self, x: torch.Tensor, freqs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs)

        # GQA expand
        if self.n_rep > 1:
            k = k[:, :, :, None, :].expand(B, T, self.n_kv_heads, self.n_rep, self.head_dim)
            k = k.reshape(B, T, self.n_heads, self.head_dim)
            v = v[:, :, :, None, :].expand(B, T, self.n_kv_heads, self.n_rep, self.head_dim)
            v = v.reshape(B, T, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 d_ff: int, norm_eps: float):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, norm_eps)
        self.attn = GQAttention(d_model, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(d_model, norm_eps)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, freqs, mask=None):
        x = x + self.attn(self.attn_norm(x), freqs, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ─── DepFormer ───────────────────────────────────────────────────────────────

class DepFormerSlice(nn.Module):
    """One slice of the DepFormer — predicts one codebook's token."""
    def __init__(self, cfg: FullMimiConfig, slice_idx: int):
        super().__init__()
        dd = cfg.dep_d_model
        dff = cfg.dep_d_ff

        # Project from main transformer hidden to depformer dim
        self.linear_in = nn.Linear(cfg.d_model, dd, bias=False)

        # Embedding for previous codebook token (or text token for slice 0)
        if slice_idx == 0:
            vocab = cfg.text_vocab_size + cfg.n_special_tokens
        else:
            vocab = cfg.audio_vocab_size + cfg.n_special_tokens
        self.emb = nn.Embedding(vocab, dd)

        self.layers = nn.ModuleList([
            TransformerBlock(dd, cfg.dep_n_heads, cfg.dep_n_kv_heads,
                            dff, cfg.norm_eps)
            for _ in range(cfg.dep_n_layers)
        ])
        self.norm = RMSNorm(dd, cfg.norm_eps)
        self.head = nn.Linear(dd, cfg.audio_vocab_size, bias=False)

    def forward(self, main_hidden: torch.Tensor, prev_token: torch.Tensor,
                freqs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        main_hidden: (B, T, d_model) from main transformer
        prev_token: (B, T) previous codebook/text tokens
        Returns: logits (B, T, audio_vocab_size)
        """
        x = self.linear_in(main_hidden) + self.emb(prev_token)
        for layer in self.layers:
            x = layer(x, freqs, mask)
        return self.head(self.norm(x))


class DepFormer(nn.Module):
    """Depth-wise transformer: predicts n_codebooks tokens sequentially."""
    def __init__(self, cfg: FullMimiConfig):
        super().__init__()
        self.n_codebooks = cfg.n_codebooks
        self.slices = nn.ModuleList([
            DepFormerSlice(cfg, i) for i in range(cfg.n_codebooks)
        ])
        self.register_buffer(
            "dep_freqs",
            precompute_rope_freqs(
                cfg.dep_d_model // cfg.dep_n_heads,
                cfg.n_codebooks + 1,
                cfg.rope_theta,
            ),
            persistent=False,
        )

    def forward(self, main_hidden: torch.Tensor, text_tokens: torch.Tensor,
                audio_targets: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None):
        """
        main_hidden: (B, T, d_model)
        text_tokens: (B, T) for slice 0 input
        audio_targets: (B, n_codebooks, T) teacher-forced targets (training)
        Returns: all_logits list of (B, T, audio_vocab_size) per codebook
        """
        all_logits = []
        prev_token = text_tokens  # Slice 0 gets text tokens

        for k in range(self.n_codebooks):
            logits = self.slices[k](
                main_hidden, prev_token,
                self.dep_freqs[:1],  # Single-step RoPE for depth
                mask=None,  # No causal mask needed for depth (each step independent)
            )
            all_logits.append(logits)

            # Next slice gets current codebook's target (teacher forcing)
            if audio_targets is not None:
                prev_token = audio_targets[:, k, :]
            else:
                prev_token = logits.argmax(dim=-1)

        return all_logits


# ─── Full Mimi TTS Model ────────────────────────────────────────────────────

class FullMimiTTS(nn.Module):
    """Complete TTS: text + multi-codebook audio → next frame prediction."""

    def __init__(self, cfg: FullMimiConfig):
        super().__init__()
        self.cfg = cfg
        total_audio_vocab = cfg.audio_vocab_size + cfg.n_special_tokens

        # Embeddings
        self.text_emb = nn.Embedding(cfg.text_vocab_size + cfg.n_special_tokens, cfg.d_model)
        self.audio_embs = nn.ModuleList([
            nn.Embedding(total_audio_vocab, cfg.d_model)
            for _ in range(cfg.n_codebooks)
        ])

        # Main transformer
        self.layers = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                            cfg.d_ff, cfg.norm_eps)
            for _ in range(cfg.n_layers)
        ])
        self.output_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.text_head = nn.Linear(cfg.d_model, cfg.text_vocab_size + cfg.n_special_tokens, bias=False)

        # DepFormer
        self.depformer = DepFormer(cfg)

        # RoPE
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
        text_tokens: torch.Tensor,       # (B, T)
        audio_codes: torch.Tensor,        # (B, n_codebooks, T) — shifted right
        target_text: Optional[torch.Tensor] = None,   # (B, T) for text loss
        target_audio: Optional[torch.Tensor] = None,   # (B, n_codebooks, T) for audio loss
    ):
        B, T = text_tokens.shape

        # Input = text embedding + sum of all codebook embeddings
        x = self.text_emb(text_tokens)
        for k in range(self.cfg.n_codebooks):
            x = x + self.audio_embs[k](audio_codes[:, k, :])

        # Causal mask
        mask = torch.full((T, T), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Main transformer
        freqs = self.rope_freqs[:T]
        for layer in self.layers:
            x = layer(x, freqs, mask)

        hidden = self.output_norm(x)  # (B, T, d_model)

        # Text logits from main transformer
        text_logits = self.text_head(hidden)  # (B, T, text_vocab)

        # DepFormer: predict audio codebooks
        audio_logits = self.depformer(
            hidden, text_tokens,
            audio_targets=target_audio,  # Teacher forcing during training
        )  # list of n_codebooks × (B, T, audio_vocab)

        # Compute losses if targets provided
        losses = {}
        if target_text is not None:
            losses["text"] = F.cross_entropy(
                text_logits.reshape(-1, text_logits.size(-1)),
                target_text.reshape(-1),
                ignore_index=0,
            )
        if target_audio is not None:
            audio_loss = 0
            for k in range(self.cfg.n_codebooks):
                al = audio_logits[k]
                tgt = target_audio[:, k, :]
                audio_loss += F.cross_entropy(
                    al.reshape(-1, al.size(-1)),
                    tgt.reshape(-1),
                    ignore_index=0,
                )
            losses["audio"] = audio_loss / self.cfg.n_codebooks

        return text_logits, audio_logits, losses


if __name__ == "__main__":
    cfg = FullMimiConfig()
    counts = cfg.param_count_estimate()
    print(f"Full Mimi TTS Config:")
    print(f"  Main: {cfg.n_layers}L × d={cfg.d_model}, H={cfg.n_heads}, KV={cfg.n_kv_heads}, FF={cfg.d_ff}")
    print(f"  DepFormer: {cfg.dep_n_layers}L × d={cfg.dep_d_model}, H={cfg.dep_n_heads}")
    print(f"  Codebooks: {cfg.n_codebooks} × {cfg.audio_vocab_size} vocab")
    print(f"  Estimated params: {counts['total']/1e6:.0f}M (main: {counts['main_transformer']/1e6:.0f}M, dep: {counts['depformer']/1e6:.0f}M)")

    model = FullMimiTTS(cfg)
    actual = sum(p.numel() for p in model.parameters())
    print(f"  Actual params: {actual/1e6:.1f}M")

    # Quick forward test
    B, T = 2, 32
    text = torch.randint(0, cfg.text_vocab_size, (B, T))
    audio = torch.randint(0, cfg.audio_vocab_size, (B, cfg.n_codebooks, T))
    text_logits, audio_logits, losses = model(text, audio, target_text=text, target_audio=audio)
    print(f"  text_logits: {text_logits.shape}")
    print(f"  audio_logits: {len(audio_logits)} × {audio_logits[0].shape}")
    print(f"  losses: text={losses['text']:.3f}, audio={losses['audio']:.3f}")
