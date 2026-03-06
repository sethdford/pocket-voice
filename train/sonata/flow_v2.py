"""Sonata Flow v2 — Single-stage text-to-mel via conditional flow matching.

F5-TTS inspired: text characters padded with filler tokens to mel length,
refined by ConvNeXt, then used as conditioning for the DiT flow backbone.
No semantic tokens, no LM, no codec in the TTS critical path.

Architecture:
  1. Text → char IDs → pad to mel length → ConvNeXt text encoder
  2. Sample t ~ logit-normal, noise ~ N(0,I)
  3. x_t = (1-t) * noise + t * mel_target
  4. DiT blocks: AdaLN(timestep) × (Attention + MLP)
  5. Loss = ||v_pred - (mel_target - noise)||^2

Inference: Euler/Heun ODE from noise → mel → vocoder (Griffin-Lim or iSTFT)
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FlowV2Config


class ConvNeXtTextEncoder(nn.Module):
    """ConvNeXt V2 text encoder: refines padded character embeddings."""

    def __init__(self, cfg: FlowV2Config):
        super().__init__()
        self.char_emb = nn.Embedding(cfg.char_vocab_size + 1, cfg.text_encoder_dim)
        layers = []
        for _ in range(cfg.text_encoder_layers):
            layers.append(ConvNeXtBlock(cfg.text_encoder_dim, cfg.text_encoder_kernel))
        self.blocks = nn.Sequential(*layers)
        self.proj = nn.Linear(cfg.text_encoder_dim, cfg.cond_dim)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        x = self.char_emb(char_ids).transpose(1, 2)
        x = self.blocks(x).transpose(1, 2)
        return self.proj(x)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, ff_mult: float = 4.0):
        super().__init__()
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        ff_dim = int(dim * ff_mult)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x).transpose(1, 2)
        x = self.ff(self.norm(x)).transpose(1, 2)
        return x + residual


from modules import AdaLayerNorm, TimestepEmbedding


class FlowV2Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, use_rope: bool = True, max_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        if use_rope:
            freqs = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            t = torch.arange(max_len).float()
            angles = torch.outer(t, freqs)
            self.register_buffer("rope_freqs", torch.polar(torch.ones_like(angles), angles), persistent=False)

    def _apply_rope(self, q, k):
        def rotate(x, f):
            x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            f = f[None, :x_c.shape[1], None, :]
            return torch.view_as_real(x_c * f).flatten(-2).to(x.dtype)
        return rotate(q, self.rope_freqs), rotate(k, self.rope_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        if self.use_rope:
            q, k = self._apply_rope(q, k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out(out.transpose(1, 2).reshape(B, T, -1))


class FlowV2Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, cond_dim: int,
                 ff_mult: float = 4.0, eps: float = 1e-5):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim, cond_dim, eps)
        self.attn = FlowV2Attention(dim, n_heads, use_rope=True)
        self.norm2 = AdaLayerNorm(dim, cond_dim, eps)
        ff_dim = int(dim * ff_mult)
        self.mlp = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x, cond))
        x = x + self.mlp(self.norm2(x, cond))
        return x


class SonataFlowV2(nn.Module):
    """Single-stage text-to-mel conditional flow matching.

    Training: text + mel_target → OT-CFM loss
    Inference: text → Euler ODE → mel → vocoder
    """

    def __init__(self, cfg: FlowV2Config, cfg_dropout_prob: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.cfg_dropout_prob = cfg_dropout_prob

        self.text_encoder = ConvNeXtTextEncoder(cfg)
        self.time_emb = TimestepEmbedding(cfg.cond_dim)

        cond_input_dim = cfg.cond_dim * 2
        if cfg.n_speakers > 0:
            self.speaker_emb = nn.Embedding(cfg.n_speakers, cfg.speaker_dim)
            self.speaker_proj = nn.Linear(cfg.speaker_dim, cfg.cond_dim)
            cond_input_dim += cfg.cond_dim
        else:
            self.speaker_emb = None

        self.cond_proj = nn.Linear(cond_input_dim, cfg.d_model)

        self.input_proj = nn.Linear(cfg.mel_dim, cfg.d_model)

        self.blocks = nn.ModuleList([
            FlowV2Block(cfg.d_model, cfg.n_heads, cfg.d_model, cfg.ff_mult, cfg.norm_eps)
            for _ in range(cfg.n_layers)
        ])

        self.output_norm = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.output_proj = nn.Linear(cfg.d_model, cfg.mel_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def pad_text_to_mel_length(self, char_ids: torch.Tensor, mel_len: int) -> torch.Tensor:
        """Pad text with filler tokens to match mel sequence length."""
        B, T_text = char_ids.shape
        if T_text >= mel_len:
            return char_ids[:, :mel_len]
        ratio = mel_len / max(T_text, 1)
        padded = torch.full((B, mel_len), self.cfg.filler_token_id,
                            dtype=torch.long, device=char_ids.device)
        for b in range(B):
            text_len = (char_ids[b] != 0).sum().item()
            for i in range(text_len):
                start = int(i * ratio)
                end = int((i + 1) * ratio)
                padded[b, start:end] = char_ids[b, i]
        return padded

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                text_cond: torch.Tensor,
                speaker_ids: Optional[torch.Tensor] = None,
                force_uncond: bool = False) -> torch.Tensor:
        B, T_seq, _ = x_t.shape

        if self.training and self.cfg_dropout_prob > 0 and not force_uncond:
            drop_mask = torch.rand(B, device=x_t.device) < self.cfg_dropout_prob
            if drop_mask.any():
                text_cond = text_cond.clone()
                text_cond[drop_mask] = 0

        if force_uncond:
            text_cond = torch.zeros_like(text_cond)

        time_cond = self.time_emb(t)[:, None, :].expand(-1, T_seq, -1)
        cond_parts = [text_cond[:, :T_seq], time_cond]

        if self.speaker_emb is not None and speaker_ids is not None:
            spk = self.speaker_proj(self.speaker_emb(speaker_ids))
            spk = spk[:, None, :].expand(-1, T_seq, -1)
            if force_uncond:
                spk = torch.zeros_like(spk)
            cond_parts.append(spk)

        cond = self.cond_proj(torch.cat(cond_parts, dim=-1))
        x = self.input_proj(x_t)

        for block in self.blocks:
            x = block(x, cond)

        return self.output_proj(self.output_norm(x))

    def compute_loss(self, mel_target: torch.Tensor, char_ids: torch.Tensor,
                     speaker_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = mel_target.shape

        padded_text = self.pad_text_to_mel_length(char_ids, T)
        text_cond = self.text_encoder(padded_text)

        # Sway sampling: shift logit-normal toward t=1
        z = torch.randn(B, device=mel_target.device) + self.cfg.sway_coefficient
        t = torch.sigmoid(z).clamp(1e-5, 1.0 - 1e-5)

        noise = torch.randn_like(mel_target)
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * noise + t_expand * mel_target
        target_velocity = mel_target - noise

        predicted_velocity = self.forward(x_t, t, text_cond, speaker_ids)
        return F.mse_loss(predicted_velocity, target_velocity)

    @torch.no_grad()
    def sample(self, char_ids: torch.Tensor, n_frames: int,
               n_steps: Optional[int] = None,
               speaker_ids: Optional[torch.Tensor] = None,
               cfg_scale: float = 1.0,
               step_schedule: Optional[List[float]] = None) -> torch.Tensor:
        """Generate mel from text via ODE solver.

        step_schedule: Optional list of n_steps+1 timesteps in [0,1]. If None,
                       uses uniform linspace. Use epss_schedule() for EPSS/Sway Sampling.
        """
        n_steps = n_steps or self.cfg.n_steps_inference
        B = char_ids.shape[0]
        device = char_ids.device

        if step_schedule is not None:
            if len(step_schedule) != n_steps + 1:
                raise ValueError(f"step_schedule must have n_steps+1={n_steps + 1} elements, got {len(step_schedule)}")
            t_schedule = step_schedule
        else:
            t_schedule = [i / n_steps for i in range(n_steps + 1)]

        padded_text = self.pad_text_to_mel_length(char_ids, n_frames)
        text_cond = self.text_encoder(padded_text)

        x = torch.randn(B, n_frames, self.cfg.mel_dim, device=device)

        for i in range(n_steps):
            t_val = t_schedule[i]
            dt = t_schedule[i + 1] - t_schedule[i]
            t = torch.full((B,), t_val, device=device)
            v_cond = self.forward(x, t, text_cond, speaker_ids)
            if cfg_scale > 1.0:
                v_uncond = self.forward(x, t, text_cond, speaker_ids, force_uncond=True)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            x = x + dt * v

        return x


if __name__ == "__main__":
    cfg = FlowV2Config()
    model = SonataFlowV2(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Sonata Flow V2 (Text → Mel):")
    print(f"  {cfg.n_layers}L × d={cfg.d_model}, H={cfg.n_heads}")
    print(f"  Mel dim: {cfg.mel_dim}, Cond dim: {cfg.cond_dim}")
    print(f"  Total params: {n_params/1e6:.1f}M")

    B, T_text, T_mel = 2, 32, 100
    chars = torch.randint(1, 128, (B, T_text))
    mel = torch.randn(B, T_mel, cfg.mel_dim)

    loss = model.compute_loss(mel, chars)
    print(f"\n  Training loss: {loss:.4f}")

    generated = model.sample(chars, n_frames=T_mel, n_steps=4)
    print(f"  Generated: {generated.shape}")
