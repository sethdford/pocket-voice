"""Sonata Flow — Conditional Flow Matching for acoustic latent prediction.

Given semantic tokens → predicts acoustic latents via optimal transport flow matching.
Non-autoregressive: generates ALL acoustic latents in parallel.

Architecture: DiT-style transformer with adaptive layer norm conditioning.
  - Input: noisy acoustic latents + semantic token embeddings (condition)
  - Output: velocity field v(x_t, t) for the flow ODE

Training: OT-CFM loss — match velocity field to linear interpolation.
Inference: 8-step Euler ODE solve from noise → acoustic latents.

This replaces Mimi's sequential DepFormer with a parallel generation model.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FlowConfig
from modules import AdaLayerNorm, TimestepEmbedding, TokenLevelEmoSteer, epss_schedule


# ─── Flow Transformer Block ─────────────────────────────────────────────────

def _precompute_flow_rope(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def _apply_flow_rope(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor):
    def rotate(x, f):
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        f = f[None, :x_c.shape[1], None, :]
        return torch.view_as_real(x_c * f).flatten(-2).to(x.dtype)
    return rotate(q, freqs), rotate(k, freqs)


class FlowAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, use_rope: bool = False, max_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        if use_rope:
            self.register_buffer(
                "rope_freqs",
                _precompute_flow_rope(self.head_dim, max_len),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        if self.use_rope:
            q, k = _apply_flow_rope(q, k, self.rope_freqs[:T])
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out(out.transpose(1, 2).reshape(B, T, -1))


class RefAudioCrossAttention(nn.Module):
    """Cross-attention to reference audio features for voice cloning.

    Instead of a single speaker embedding vector, attends to a sequence of
    features from reference audio — captures timbre, prosody, and style.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, ref_features: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        _, Tr, _ = ref_features.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(ref_features).view(B, Tr, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(ref_features).view(B, Tr, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out(out.transpose(1, 2).reshape(B, T, -1))


class FlowTransformerBlock(nn.Module):
    """DiT block: AdaLN → Attention → (optional RefAudio XAttn) → AdaLN → MLP."""

    def __init__(self, dim: int, n_heads: int, cond_dim: int,
                 ff_mult: float = 4.0, eps: float = 1e-5, use_rope: bool = False,
                 use_ref_audio: bool = False):
        super().__init__()
        ff_dim = int(dim * ff_mult)
        self.norm1 = AdaLayerNorm(dim, cond_dim, eps)
        self.attn = FlowAttention(dim, n_heads, use_rope=use_rope)

        self.use_ref_audio = use_ref_audio
        if use_ref_audio:
            self.ref_norm = nn.LayerNorm(dim, eps=eps)
            self.ref_attn = RefAudioCrossAttention(dim, n_heads)

        self.norm2 = AdaLayerNorm(dim, cond_dim, eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                ref_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x, cond))
        if self.use_ref_audio and ref_features is not None:
            x = x + self.ref_attn(self.ref_norm(x), ref_features)
        x = x + self.mlp(self.norm2(x, cond))
        return x


# ─── Sonata Flow Network ────────────────────────────────────────────────────

class SonataFlow(nn.Module):
    """Conditional Flow Matching: semantic tokens → acoustic latents.

    Training:
      1. Sample t ~ logit-normal (concentrated near t=0 and t=1)
      2. Compute x_t = (1-t) * noise + t * x_target (linear interpolation)
      3. Predict velocity v(x_t, t, condition)
      4. Loss = ||v - (x_target - noise)||^2

    Inference:
      1. Start from noise x_0 ~ N(0, I)
      2. Euler steps: x_{t+dt} = x_t + dt * v(x_t, t, condition)
      3. After n_steps: x_1 ≈ acoustic latents

    CFG: Training with cfg_dropout_prob > 0 randomly drops semantic conditioning.
    At inference, cfg_scale > 1.0 blends conditional and unconditional predictions.
    """

    def __init__(self, cfg: FlowConfig, cfg_dropout_prob: float = 0.0):
        super().__init__()
        self.cfg = cfg
        self.cfg_dropout_prob = cfg_dropout_prob

        # Semantic conditioning
        self.semantic_emb = nn.Embedding(
            cfg.semantic_vocab_size + 4,  # +4 special tokens
            cfg.cond_dim,
        )

        # Timestep conditioning
        self.time_emb = TimestepEmbedding(cfg.cond_dim)

        # Speaker conditioning (optional)
        cond_input_dim = cfg.cond_dim * 2
        if cfg.n_speakers > 0:
            self.speaker_emb = nn.Embedding(cfg.n_speakers, cfg.speaker_dim)
            self.speaker_proj = nn.Linear(cfg.speaker_dim, cfg.cond_dim)
            cond_input_dim += cfg.cond_dim
        else:
            self.speaker_emb = None

        # Emotion conditioning (optional)
        if cfg.n_emotions > 0:
            self.emotion_emb = nn.Embedding(cfg.n_emotions, cfg.emotion_dim)
            self.emotion_proj = nn.Linear(cfg.emotion_dim, cfg.cond_dim)
            cond_input_dim += cfg.cond_dim
        else:
            self.emotion_emb = None

        # Prosody conditioning (optional)
        if cfg.prosody_dim > 0:
            self.prosody_proj = nn.Sequential(
                nn.Linear(cfg.prosody_dim, cfg.cond_dim),
                nn.SiLU(),
                nn.Linear(cfg.cond_dim, cfg.cond_dim),
            )
            cond_input_dim += cfg.cond_dim
        else:
            self.prosody_proj = None

        # Energy predictor (optional) — predicts per-frame log-energy from semantic conditioning
        if cfg.use_energy_predictor:
            self.energy_predictor = nn.Sequential(
                nn.Linear(cfg.cond_dim, cfg.energy_dim),
                nn.SiLU(),
                nn.Linear(cfg.energy_dim, cfg.energy_dim),
                nn.SiLU(),
                nn.Linear(cfg.energy_dim, 1),
            )
        else:
            self.energy_predictor = None

        # Combined conditioning projection
        self.cond_proj = nn.Linear(cond_input_dim, cfg.d_model)

        # Reference audio encoder for zero-shot voice cloning (optional)
        self.use_ref_audio = getattr(cfg, 'use_ref_audio', False)
        if self.use_ref_audio:
            ref_input_dim = getattr(cfg, 'ref_audio_dim', 80)
            self.ref_audio_encoder = nn.Sequential(
                nn.Linear(ref_input_dim, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )

        # Input projection: acoustic_dim → d_model
        self.input_proj = nn.Linear(cfg.acoustic_dim, cfg.d_model)

        use_rope = getattr(cfg, 'use_rope', False)
        self.blocks = nn.ModuleList([
            FlowTransformerBlock(
                cfg.d_model, cfg.n_heads, cfg.d_model,
                cfg.ff_mult, cfg.norm_eps, use_rope=use_rope,
                use_ref_audio=self.use_ref_audio,
            )
            for _ in range(cfg.n_layers)
        ])

        # Token-level emotion steering (optional, applies after transformer blocks)
        if cfg.n_emotions > 0:
            self.token_emosteer = TokenLevelEmoSteer(cfg.d_model, cfg.n_emotions)
        else:
            self.token_emosteer = None

        # Output projection: d_model → acoustic_dim (velocity field)
        self.output_norm = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.output_proj = nn.Linear(cfg.d_model, cfg.acoustic_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                semantic_tokens: torch.Tensor,
                speaker_ids: Optional[torch.Tensor] = None,
                emotion_ids: Optional[torch.Tensor] = None,
                prosody_features: Optional[torch.Tensor] = None,
                ref_audio: Optional[torch.Tensor] = None,
                force_uncond: bool = False) -> torch.Tensor:
        """
        Predict velocity field v(x_t, t, semantic_tokens).

        x_t: (B, T, acoustic_dim) — noisy acoustic latent at time t
        t: (B,) — timestep in [0, 1]
        semantic_tokens: (B, T) — conditioning semantic token indices
        speaker_ids: (B,) — optional speaker ID for multi-speaker conditioning
        emotion_ids: (B,) — optional emotion ID for emotion conditioning
        prosody_features: (B, T, 3) — optional (log_pitch, energy, rate)
        ref_audio: (B, T_ref, ref_dim) — optional reference audio features for voice cloning
        force_uncond: if True, zero out all conditioning (for CFG inference)
        Returns: v (B, T, acoustic_dim) — predicted velocity
        """
        B, T_seq, _ = x_t.shape

        if self.training and self.cfg_dropout_prob > 0 and not force_uncond:
            drop_mask = torch.rand(B, device=x_t.device) < self.cfg_dropout_prob
            if drop_mask.any():
                semantic_tokens = semantic_tokens.clone()
                semantic_tokens[drop_mask] = 0

        if force_uncond:
            semantic_tokens = torch.zeros_like(semantic_tokens)

        sem_cond = self.semantic_emb(semantic_tokens)
        time_cond = self.time_emb(t)[:, None, :].expand_as(sem_cond)

        cond_parts = [sem_cond, time_cond]

        if self.speaker_emb is not None and speaker_ids is not None:
            spk = self.speaker_proj(self.speaker_emb(speaker_ids))
            spk = spk[:, None, :].expand_as(sem_cond)
            if force_uncond:
                spk = torch.zeros_like(spk)
            cond_parts.append(spk)

        if self.emotion_emb is not None:
            if emotion_ids is not None and not force_uncond:
                emo = self.emotion_proj(self.emotion_emb(emotion_ids))
                emo = emo[:, None, :].expand_as(sem_cond)
            else:
                emo = torch.zeros(B, T_seq, self.cfg.cond_dim, device=x_t.device)
            cond_parts.append(emo)

        if self.prosody_proj is not None:
            if prosody_features is not None and not force_uncond:
                pros = self.prosody_proj(prosody_features)
            else:
                pros = torch.zeros(B, T_seq, self.cfg.cond_dim, device=x_t.device)
            cond_parts.append(pros)

        cond = self.cond_proj(torch.cat(cond_parts, dim=-1))

        # Encode reference audio for voice cloning
        ref_features = None
        if self.use_ref_audio and ref_audio is not None and not force_uncond:
            ref_features = self.ref_audio_encoder(ref_audio)

        x = self.input_proj(x_t)

        for block in self.blocks:
            x = block(x, cond, ref_features)

        # Token-level emotion steering (after blocks, before output projection)
        if self.token_emosteer is not None and emotion_ids is not None and not force_uncond:
            x = self.token_emosteer(x, emotion_ids=emotion_ids, scale=1.0)

        return self.output_proj(self.output_norm(x))

    def predict_energy(self, semantic_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """Predict per-frame log-energy from semantic conditioning.

        Returns (B, T, 1) log-energy values, or None if no energy predictor.
        Note: v1 Flow does not have MAS alignment, so this predicts frame energy
        rather than true phoneme durations.
        """
        if self.energy_predictor is None:
            return None
        cond = self.semantic_emb(semantic_tokens)
        return self.energy_predictor(cond)

    @staticmethod
    def sample_logit_normal(batch_size: int, device: torch.device,
                            mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        """Sample timesteps from logit-normal distribution.
        Concentrates samples near t=0 and t=1 where the velocity field
        changes most rapidly, improving training efficiency.
        """
        z = torch.randn(batch_size, device=device) * std + mean
        t = torch.sigmoid(z)
        return t.clamp(1e-5, 1.0 - 1e-5)

    def compute_loss(self, x_target: torch.Tensor,
                     semantic_tokens: torch.Tensor,
                     mask: Optional[torch.Tensor] = None,
                     speaker_ids: Optional[torch.Tensor] = None,
                     emotion_ids: Optional[torch.Tensor] = None,
                     prosody_features: Optional[torch.Tensor] = None,
                     timestep_sampling: str = "logit_normal") -> torch.Tensor:
        """OT-CFM training loss.

        x_target: (B, T, acoustic_dim) — ground truth acoustic latents from codec
        semantic_tokens: (B, T) — corresponding semantic tokens from codec
        mask: (B, T) — 1.0 at valid positions, 0.0 at padding
        speaker_ids: (B,) — optional speaker IDs
        emotion_ids: (B,) — optional emotion IDs
        prosody_features: (B, T, 3) — optional (log_pitch, energy, rate)
        """
        B = x_target.shape[0]
        device = x_target.device

        if timestep_sampling == "logit_normal":
            t = self.sample_logit_normal(B, device)
        else:
            t = torch.rand(B, device=device)

        noise = torch.randn_like(x_target)
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * noise + t_expand * x_target
        target_velocity = x_target - noise

        predicted_velocity = self.forward(
            x_t, t, semantic_tokens, speaker_ids,
            emotion_ids=emotion_ids, prosody_features=prosody_features
        )

        if mask is not None:
            per_elem = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
            denom = (mask.sum() * predicted_velocity.shape[-1]).clamp(min=1)
            loss = (per_elem * mask.unsqueeze(-1)).sum() / denom
        else:
            loss = F.mse_loss(predicted_velocity, target_velocity)

        # Energy predictor loss (if enabled)
        if self.energy_predictor is not None:
            pred_energy = self.predict_energy(semantic_tokens)
            if pred_energy is not None:
                energy = x_target.pow(2).mean(-1, keepdim=True)
                log_energy = torch.log(energy.clamp(min=1e-8))
                if mask is not None:
                    per_elem_energy = F.mse_loss(pred_energy, log_energy, reduction="none")
                    denom_energy = mask.sum().clamp(min=1)
                    loss = loss + 0.1 * (per_elem_energy.squeeze(-1) * mask).sum() / denom_energy
                else:
                    loss = loss + 0.1 * F.mse_loss(pred_energy, log_energy)

        return loss

    @torch.no_grad()
    def _velocity(self, x, t, semantic_tokens, speaker_ids,
                  emotion_ids, prosody_features, cfg_scale, force_uncond=False):
        """Compute velocity with optional CFG."""
        v_cond = self.forward(x, t, semantic_tokens, speaker_ids,
                              emotion_ids=emotion_ids,
                              prosody_features=prosody_features,
                              force_uncond=force_uncond)
        if cfg_scale > 1.0 and not force_uncond:
            v_uncond = self.forward(x, t, semantic_tokens, speaker_ids,
                                    emotion_ids=emotion_ids,
                                    prosody_features=prosody_features,
                                    force_uncond=True)
            return v_uncond + cfg_scale * (v_cond - v_uncond)
        return v_cond

    def sample(self, semantic_tokens: torch.Tensor,
               n_steps: Optional[int] = None,
               speaker_ids: Optional[torch.Tensor] = None,
               emotion_ids: Optional[torch.Tensor] = None,
               prosody_features: Optional[torch.Tensor] = None,
               cfg_scale: float = 1.0,
               use_heun: bool = False,
               step_schedule: Optional[List[float]] = None) -> torch.Tensor:
        """Generate acoustic latents from semantic tokens via ODE solver.

        Args:
            use_heun: Use Heun's 2nd-order method for better quality at same step count.
                      Falls back to Euler for the last step.
            step_schedule: Optional list of n_steps+1 timesteps in [0,1]. If None, uses
                           uniform linspace. Use epss_schedule() for EPSS/Sway Sampling.
        """
        n_steps = n_steps or self.cfg.n_steps_inference
        B, T = semantic_tokens.shape
        device = semantic_tokens.device

        if step_schedule is not None:
            if len(step_schedule) != n_steps + 1:
                raise ValueError(f"step_schedule must have n_steps+1={n_steps + 1} elements, got {len(step_schedule)}")
            t_schedule = step_schedule
        else:
            t_schedule = [i / n_steps for i in range(n_steps + 1)]

        x = torch.randn(B, T, self.cfg.acoustic_dim, device=device)

        for i in range(n_steps):
            t_val = t_schedule[i]
            dt = t_schedule[i + 1] - t_schedule[i]
            t = torch.full((B,), t_val, device=device)
            v1 = self._velocity(x, t, semantic_tokens, speaker_ids,
                                emotion_ids, prosody_features, cfg_scale)

            if use_heun and i < n_steps - 1:
                x_euler = x + dt * v1
                t2_val = t_schedule[i + 1]
                t2 = torch.full((B,), t2_val, device=device)
                v2 = self._velocity(x_euler, t2, semantic_tokens, speaker_ids,
                                    emotion_ids, prosody_features, cfg_scale)
                x = x + dt * 0.5 * (v1 + v2)
            else:
                x = x + dt * v1

        return x

    @torch.no_grad()
    def sample_smooth_cache(self, semantic_tokens: torch.Tensor,
                            n_steps: Optional[int] = None,
                            speaker_ids: Optional[torch.Tensor] = None,
                            emotion_ids: Optional[torch.Tensor] = None,
                            prosody_features: Optional[torch.Tensor] = None,
                            cfg_scale: float = 1.0,
                            cache_interval: int = 2) -> torch.Tensor:
        """SmoothCache: cache velocity outputs across ODE steps.

        Key insight: adjacent ODE steps produce very similar velocities.
        We compute full velocity every `cache_interval` steps and interpolate
        in between. 2-3x speedup with negligible quality loss.

        Based on: "SmoothCache: A Universal Inference Acceleration Technique
        for Diffusion Transformers" (F5-TTS application).
        """
        n_steps = n_steps or self.cfg.n_steps_inference
        B, T = semantic_tokens.shape
        device = semantic_tokens.device

        x = torch.randn(B, T, self.cfg.acoustic_dim, device=device)
        dt = 1.0 / n_steps

        cached_v = None
        prev_v = None

        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)

            if i % cache_interval == 0 or cached_v is None:
                v = self._velocity(x, t, semantic_tokens, speaker_ids,
                                   emotion_ids, prosody_features, cfg_scale)
                prev_v = cached_v
                cached_v = v
            else:
                # Interpolate between the two nearest cache-point velocities
                alpha = (i % cache_interval) / cache_interval
                if prev_v is not None:
                    v = (1 - alpha) * prev_v + alpha * cached_v
                else:
                    v = cached_v

            x = x + dt * v

        return x

    @staticmethod
    def epss_schedule(n_steps: int, sway: float = -1.0) -> List[float]:
        """EPSS: Empirically Pruned Step Sampling — non-uniform timesteps.

        Delegates to modules.epss_schedule. For n_steps=8 uses F5-TTS empirical
        schedule; for other counts uses logit-normal CDF with sway.

        sway < 0: bias toward t=1 (Sway Sampling)
        sway = 0: uniform
        sway > 0: bias toward t=0
        """
        return epss_schedule(n_steps, sway)

    @torch.no_grad()
    def sample_epss(self, semantic_tokens: torch.Tensor,
                    n_steps: Optional[int] = None,
                    speaker_ids: Optional[torch.Tensor] = None,
                    emotion_ids: Optional[torch.Tensor] = None,
                    prosody_features: Optional[torch.Tensor] = None,
                    cfg_scale: float = 1.0,
                    sway: float = -1.0) -> torch.Tensor:
        """Sample with EPSS non-uniform timestep scheduling.

        Calls sample() with step_schedule from epss_schedule(n_steps, sway).
        Uses fewer total steps but concentrates them where they matter most.
        Achieves similar quality to uniform 8-step with fewer steps.
        """
        n_steps = n_steps or self.cfg.n_steps_inference
        return self.sample(
            semantic_tokens,
            n_steps=n_steps,
            speaker_ids=speaker_ids,
            emotion_ids=emotion_ids,
            prosody_features=prosody_features,
            cfg_scale=cfg_scale,
            step_schedule=epss_schedule(n_steps, sway),
        )

    def compute_cfg_distill_loss(self, x_target: torch.Tensor,
                                 semantic_tokens: torch.Tensor,
                                 cfg_scale: float = 2.0,
                                 speaker_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """CFG Distillation: train a single-pass model to approximate the
        dual-pass CFG output. Eliminates the 2x compute cost at inference.

        The teacher signal is: v_uncond + cfg_scale * (v_cond - v_uncond)
        We train the student (same model with a special "distilled" flag)
        to predict this combined velocity in a single forward pass.
        """
        B = x_target.shape[0]
        device = x_target.device

        t = self.sample_logit_normal(B, device)
        noise = torch.randn_like(x_target)
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * noise + t_expand * x_target

        # Teacher: dual forward pass (detached)
        with torch.no_grad():
            v_cond = self.forward(x_t, t, semantic_tokens, speaker_ids)
            v_uncond = self.forward(x_t, t, semantic_tokens, speaker_ids, force_uncond=True)
            v_teacher = v_uncond + cfg_scale * (v_cond - v_uncond)

        # Student: single forward pass (with gradient)
        v_student = self.forward(x_t, t, semantic_tokens, speaker_ids)

        return F.mse_loss(v_student, v_teacher)


if __name__ == "__main__":
    cfg = FlowConfig()
    model = SonataFlow(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"Sonata Flow (Conditional Flow Matching):")
    print(f"  {cfg.n_layers}L × d={cfg.d_model}, H={cfg.n_heads}")
    print(f"  Acoustic dim: {cfg.acoustic_dim}, Cond dim: {cfg.cond_dim}")
    print(f"  Inference steps: {cfg.n_steps_inference}")
    print(f"  Total params: {n_params/1e6:.1f}M")

    B, T = 2, 50  # 50 frames = 1 second
    semantic = torch.randint(0, cfg.semantic_vocab_size, (B, T))
    x_target = torch.randn(B, T, cfg.acoustic_dim)

    # Training loss
    loss = model.compute_loss(x_target, semantic)
    print(f"\n  Training loss: {loss:.4f}")

    # Inference
    generated = model.sample(semantic, n_steps=4)
    print(f"  Generated: {generated.shape}")
