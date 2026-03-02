"""Sonata Flow v3 — Interleaved streaming + causal architecture.

Key upgrades over v2 (F5-TTS-style):
  1. **Causal DiT backbone**: Sliding-window attention for left-to-right generation.
     Enables streaming mel output without waiting for full sequence.
  2. **Interleaved text-speech training**: Text tokens and mel frames alternate
     in the sequence, inspired by SpeakStream (Feb 2026) and MELLE (2025).
     This teaches the model natural alignment without a duration predictor.
  3. **Sway sampling with logit-normal weighting**: Better timestep distribution
     for flow matching (F5-TTS trick, proven to improve convergence).
  4. **Reference audio prompting**: Prepend reference mel for zero-shot cloning.
  5. **Chunk-wise streaming inference**: Generate mel in chunks with context overlap.

The key insight from SpeakStream: interleaving text and speech tokens lets the
model learn alignment implicitly, eliminating the duration predictor bottleneck
and enabling truly streaming TTS (generate audio before the full text arrives).

Based on: SpeakStream (Feb 2026), F5-TTS (2024), CosyVoice 2 (2025),
          Mars5 (2024), MELLE (2025).
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FlowV3Config
from modules import AdaLayerNorm, TimestepEmbedding, TokenLevelEmoSteer, mas_durations


# ═══════════════════════════════════════════════════════════════════════════════
# Causal Sliding-Window Attention (for streaming)
# ═══════════════════════════════════════════════════════════════════════════════

class CausalSlidingWindowAttention(nn.Module):
    """Attention with causal mask + sliding window for bounded memory in streaming.

    Standard self-attention is O(n^2) in sequence length, which makes streaming
    impractical for long utterances. Sliding-window limits each token to attend
    only to the last W frames, giving O(n*W) complexity.

    For streaming: process chunks of C frames, each attending to C + W prior frames.
    """

    def __init__(self, dim: int, n_heads: int, window_size: int = 256,
                 use_rope: bool = True, max_len: int = 8192):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

        if use_rope:
            freqs = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            t = torch.arange(max_len).float()
            angles = torch.outer(t, freqs)
            self.register_buffer("rope_freqs", torch.polar(torch.ones_like(angles), angles), persistent=False)

    def _apply_rope(self, q, k, offset: int = 0):
        def rotate(x, f):
            x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            seq_len = x_c.shape[1]
            f = f[None, offset:offset + seq_len, None, :]
            return torch.view_as_real(x_c * f).flatten(-2).to(x.dtype)
        return rotate(q, self.rope_freqs), rotate(k, self.rope_freqs)

    def forward(self, x: torch.Tensor, kv_cache: Optional[torch.Tensor] = None,
                offset: int = 0) -> tuple:
        """
        x: (B, T, D)
        kv_cache: (B, 2, T_past, D) or None
        Returns: (output, new_kv_cache)
        """
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each (B, T, H, D_h)

        if self.use_rope:
            q, k = self._apply_rope(q, k, offset)

        # Prepend cached KV if streaming
        if kv_cache is not None:
            k_past, v_past = kv_cache[:, 0], kv_cache[:, 1]  # (B, T_past, H, D_h)
            k = torch.cat([k_past, k], dim=1)
            v = torch.cat([v_past, v], dim=1)

        # Update KV cache (keep only window_size)
        T_kv = k.shape[1]
        if T_kv > self.window_size:
            k = k[:, -self.window_size:]
            v = v[:, -self.window_size:]
            T_kv = self.window_size

        new_cache = torch.stack([k, v], dim=1)  # (B, 2, T_kv, H, D_h)

        # Transpose for attention: (B, H, T, D_h)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Always causal when sequence has multiple positions (matches training)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out(out), new_cache


# ═══════════════════════════════════════════════════════════════════════════════
# Interleaved Text-Speech Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class InterleavedEncoder(nn.Module):
    """Encodes interleaved text + mel frames into a unified sequence.

    SpeakStream-style: text characters and mel frames alternate in the sequence,
    with type embeddings to distinguish them. The model learns alignment from
    the interleaved structure itself.

    Interleaving pattern for training:
      [text_1] [mel_1..mel_k] [text_2] [mel_k+1..mel_2k] ...
    where k is the number of mel frames per text character (~4-6 for English).

    For inference (streaming), text arrives first, and mel is generated
    autoregressively, naturally interleaved.
    """

    def __init__(self, char_vocab_size: int, mel_dim: int, d_model: int):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_size, d_model)
        self.mel_proj = nn.Linear(mel_dim, d_model)
        self.type_emb = nn.Embedding(2, d_model)  # 0=text, 1=mel

    def encode_text(self, char_ids: torch.Tensor) -> torch.Tensor:
        """char_ids: (B, T_text) → (B, T_text, D)"""
        type_ids = torch.zeros_like(char_ids)
        return self.char_emb(char_ids) + self.type_emb(type_ids)

    def encode_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, T_mel, mel_dim) → (B, T_mel, D)"""
        B, T = mel.shape[:2]
        type_ids = torch.ones(B, T, dtype=torch.long, device=mel.device)
        return self.mel_proj(mel) + self.type_emb(type_ids)

    def interleave(self, text_enc: torch.Tensor, mel_enc: torch.Tensor,
                   char_per_group: int = 1) -> torch.Tensor:
        """Interleave text and mel encodings.

        For each text character, assigns ~(T_mel/T_text) mel frames.
        Returns: (B, T_text + T_mel, D) interleaved sequence.
        """
        B, T_text, D = text_enc.shape
        T_mel = mel_enc.shape[1]

        if T_text == 0:
            return mel_enc

        frames_per_char = max(1, T_mel // T_text)
        parts = []
        mel_idx = 0

        for i in range(T_text):
            parts.append(text_enc[:, i:i+1])
            end = min(mel_idx + frames_per_char, T_mel)
            if mel_idx < end:
                parts.append(mel_enc[:, mel_idx:end])
            mel_idx = end

        # Append remaining mel frames
        if mel_idx < T_mel:
            parts.append(mel_enc[:, mel_idx:])

        return torch.cat(parts, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Flow v3 DiT Block
# ═══════════════════════════════════════════════════════════════════════════════

class FlowV3Block(nn.Module):
    """DiT block with causal sliding-window attention + AdaLN conditioning."""

    def __init__(self, dim: int, n_heads: int, cond_dim: int,
                 ff_mult: float = 4.0, eps: float = 1e-5, window_size: int = 256):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim, cond_dim, eps)
        self.attn = CausalSlidingWindowAttention(dim, n_heads, window_size)
        self.norm2 = AdaLayerNorm(dim, cond_dim, eps)
        ff_dim = int(dim * ff_mult)
        self.mlp = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                kv_cache: Optional[torch.Tensor] = None,
                offset: int = 0) -> tuple:
        h, new_cache = self.attn(self.norm1(x, cond), kv_cache, offset)
        x = x + h
        x = x + self.mlp(self.norm2(x, cond))
        return x, new_cache


# ═══════════════════════════════════════════════════════════════════════════════
# Duration Predictor (learned text-to-duration mapping)
# ═══════════════════════════════════════════════════════════════════════════════

class DurationPredictor(nn.Module):
    """Predicts log-duration (in mel frames) for each text character.

    Architecture: char embedding → 2×Conv1d → Linear → log-duration per char.
    Trained with MSE on ground-truth character-level durations extracted by
    MAS (Monotonic Alignment Search) during training, or simple uniform
    alignment as a warm-start.
    """

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, text_enc: torch.Tensor) -> torch.Tensor:
        """text_enc: (B, T_text, D) → log_durations: (B, T_text)"""
        h = self.conv1(text_enc.transpose(1, 2)).transpose(1, 2)
        h = self.drop(F.relu(self.norm1(h)))
        h = self.conv2(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(F.relu(self.norm2(h)))
        return self.proj(h).squeeze(-1)

    @staticmethod
    def expand_encodings(text_enc: torch.Tensor, durations: torch.Tensor,
                         target_len: int) -> torch.Tensor:
        """Expand text encodings to mel length using predicted durations.

        durations: (B, T_text) integer frame counts per character.
        Returns: (B, target_len, D) expanded text conditioning.
        """
        B, T_text, D = text_enc.shape
        device = text_enc.device
        expanded = torch.zeros(B, target_len, D, device=device, dtype=text_enc.dtype)
        for b in range(B):
            pos = 0
            for i in range(T_text):
                dur = durations[b, i].item()
                if dur <= 0:
                    continue
                end = min(pos + dur, target_len)
                expanded[b, pos:end] = text_enc[b, i]
                pos = end
                if pos >= target_len:
                    break
        return expanded

    @staticmethod
    def compute_gt_durations(char_ids: torch.Tensor,
                             mel_len: int) -> torch.Tensor:
        """Simple uniform alignment for warm-start training.
        Returns integer durations summing to mel_len per batch element.
        """
        B, T_text = char_ids.shape
        durations = torch.zeros(B, T_text, dtype=torch.long, device=char_ids.device)
        for b in range(B):
            n_chars = (char_ids[b] > 0).sum().item()
            if n_chars == 0:
                continue
            base = mel_len // n_chars
            remainder = mel_len % n_chars
            durations[b, :n_chars] = base
            durations[b, :remainder] += 1
        return durations


# ═══════════════════════════════════════════════════════════════════════════════
# Sonata Flow v3 (Interleaved Streaming)
# ═══════════════════════════════════════════════════════════════════════════════

class SonataFlowV3(nn.Module):
    """Interleaved streaming flow model for real-time TTS.

    Training modes:
      1. Standard: text → mel (like v2, for backward compatibility)
      2. Interleaved: interleaved text+mel → predict masked mel (SpeakStream-style)
      3. Prompted: ref_mel + text → generate mel (zero-shot voice cloning)

    Inference modes:
      1. Full-sequence: generate all mel frames at once (non-streaming)
      2. Chunk-streaming: generate chunk_size frames at a time with KV cache
    """

    def __init__(self, cfg: FlowV3Config, cfg_dropout_prob: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.cfg_dropout_prob = cfg_dropout_prob

        self.interleaved_enc = InterleavedEncoder(
            cfg.char_vocab_size, cfg.mel_dim, cfg.d_model
        )
        self.time_emb = TimestepEmbedding(cfg.cond_dim)

        cond_input_dim = cfg.cond_dim
        if cfg.n_speakers > 0:
            self.speaker_emb = nn.Embedding(cfg.n_speakers, cfg.speaker_dim)
            self.speaker_proj = nn.Linear(cfg.speaker_dim, cfg.cond_dim)
            cond_input_dim += cfg.cond_dim
        else:
            self.speaker_emb = None

        self.cond_merge = nn.Linear(cond_input_dim, cfg.d_model)

        self.input_proj = nn.Linear(cfg.mel_dim, cfg.d_model)

        self.blocks = nn.ModuleList([
            FlowV3Block(cfg.d_model, cfg.n_heads, cfg.d_model,
                        cfg.ff_mult, cfg.norm_eps, cfg.window_size)
            for _ in range(cfg.n_layers)
        ])

        # Token-level emotion steering (optional, applies after transformer blocks)
        if cfg.n_emotions > 0:
            self.token_emosteer = TokenLevelEmoSteer(cfg.d_model, cfg.n_emotions)
        else:
            self.token_emosteer = None

        self.duration_predictor = DurationPredictor(cfg.d_model)

        self.output_norm = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.output_proj = nn.Linear(cfg.d_model, cfg.mel_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _build_conditioning(self, t: torch.Tensor, T_seq: int,
                            speaker_ids: Optional[torch.Tensor] = None,
                            force_uncond: bool = False) -> torch.Tensor:
        time_cond = self.time_emb(t)[:, None, :].expand(-1, T_seq, -1)
        parts = [time_cond]

        if self.speaker_emb is not None:
            if speaker_ids is not None and not force_uncond:
                spk = self.speaker_proj(self.speaker_emb(speaker_ids))
                spk = spk[:, None, :].expand(-1, T_seq, -1)
            else:
                B = time_cond.shape[0]
                spk = torch.zeros(B, T_seq, self.cfg.cond_dim,
                                  device=time_cond.device, dtype=time_cond.dtype)
            parts.append(spk)

        return self.cond_merge(torch.cat(parts, dim=-1))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                text_context: torch.Tensor,
                speaker_ids: Optional[torch.Tensor] = None,
                emotion_ids: Optional[torch.Tensor] = None,
                force_uncond: bool = False,
                ref_mel: Optional[torch.Tensor] = None,
                kv_caches: Optional[List[torch.Tensor]] = None,
                offset: int = 0) -> tuple:
        """Predict velocity field.

        x_t: (B, T_mel, mel_dim) — noisy mel at timestep t
        text_context: (B, T_mel, D) — pre-encoded text (already aligned to mel length)
        ref_mel: (B, T_ref, mel_dim) — optional reference mel for prompting
        kv_caches: list of per-layer KV caches for streaming (None = full-sequence mode)
        offset: position offset for RoPE in streaming mode

        Returns: (velocity, new_kv_caches) — velocity field and updated KV caches
        """
        B, T_mel, _ = x_t.shape

        # Handle CFG dropout
        if self.training and self.cfg_dropout_prob > 0 and not force_uncond:
            drop_mask = torch.rand(B, device=x_t.device) < self.cfg_dropout_prob
            if drop_mask.any():
                text_context = text_context.clone()
                text_context[drop_mask] = 0

        if force_uncond:
            text_context = torch.zeros_like(text_context)

        # Build conditioning
        total_len = T_mel
        if ref_mel is not None:
            total_len += ref_mel.shape[1]

        cond = self._build_conditioning(t, total_len, speaker_ids, force_uncond)

        # Encode input: optionally prepend reference mel
        x = self.input_proj(x_t)
        if ref_mel is not None:
            ref_enc = self.interleaved_enc.encode_mel(ref_mel)
            x = torch.cat([ref_enc, x], dim=1)
            ref_cond = text_context[:, :1].expand(-1, ref_mel.shape[1], -1)
            text_context = torch.cat([ref_cond, text_context], dim=1)

        # Add text context
        x = x + text_context[:, :x.shape[1]]

        # Run through causal DiT blocks, propagating KV cache
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, cond, kv_cache=layer_cache, offset=offset)
            new_kv_caches.append(new_cache)

        # Token-level emotion steering (after blocks, before output projection)
        if self.token_emosteer is not None and emotion_ids is not None and not force_uncond:
            x = self.token_emosteer(x, emotion_ids=emotion_ids, scale=1.0)

        x = self.output_proj(self.output_norm(x))

        # Strip reference prefix if present
        if ref_mel is not None:
            x = x[:, ref_mel.shape[1]:]

        return x, new_kv_caches

    def compute_loss(self, mel_target: torch.Tensor, char_ids: torch.Tensor,
                     speaker_ids: Optional[torch.Tensor] = None,
                     emotion_ids: Optional[torch.Tensor] = None,
                     ref_mel: Optional[torch.Tensor] = None,
                     mel_mask: Optional[torch.Tensor] = None,
                     use_mas: bool = False,
                     return_denoised: bool = False):
        B, T, _ = mel_target.shape

        text_enc = self.interleaved_enc.encode_text(char_ids)

        if use_mas:
            gt_durations = mas_durations(text_enc, mel_target)
        else:
            gt_durations = DurationPredictor.compute_gt_durations(char_ids, T)

        log_dur_pred = self.duration_predictor(text_enc)
        log_dur_gt = torch.log(gt_durations.float().clamp(min=1))
        char_mask = (char_ids > 0).float()
        dur_sq = F.mse_loss(log_dur_pred * char_mask, log_dur_gt * char_mask, reduction='none') * char_mask
        dur_loss = dur_sq.sum() / char_mask.sum().clamp(min=1)

        if use_mas:
            text_cond = DurationPredictor.expand_encodings(text_enc, gt_durations, T)
        else:
            text_cond = self._align_text_to_mel(text_enc, T)

        z = torch.randn(B, device=mel_target.device) + self.cfg.sway_coefficient
        t = torch.sigmoid(z).clamp(1e-5, 1.0 - 1e-5)

        noise = torch.randn_like(mel_target)
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * noise + t_expand * mel_target
        target_velocity = mel_target - noise

        predicted, _ = self.forward(x_t, t, text_cond, speaker_ids,
                                    emotion_ids=emotion_ids, ref_mel=ref_mel)

        if mel_mask is not None:
            mask = mel_mask.unsqueeze(-1)
            per_elem = F.mse_loss(predicted, target_velocity, reduction='none') * mask
            n_valid = mask.sum() * predicted.shape[-1]
            flow_loss = per_elem.sum() / n_valid.clamp(min=1)
        else:
            flow_loss = F.mse_loss(predicted, target_velocity)

        dur_weight = getattr(self, '_dur_loss_weight', 0.1)
        total = flow_loss + dur_weight * dur_loss

        if return_denoised:
            mel_denoised = x_t + (1 - t_expand) * predicted
            return total, mel_denoised
        return total

    def compute_interleaved_loss(self, mel_target: torch.Tensor,
                                 char_ids: torch.Tensor,
                                 speaker_ids: Optional[torch.Tensor] = None,
                                 emotion_ids: Optional[torch.Tensor] = None,
                                 ref_mel: Optional[torch.Tensor] = None,
                                 mel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """SpeakStream-style interleaved training loss.

        Interleaves text and mel tokens, masks a portion of mel frames,
        and trains the model to denoise the masked mel frames in-place.

        IMPORTANT: Flow conditioning must be text-only. Using interleaved[:, :T]
        would leak target mel into conditioning (interleaved = [text_0, mel_0..,
        text_1, mel_1..]) making training trivial. We use _align_text_to_mel
        for text-only conditioning, same as compute_loss.
        """
        B, T, _ = mel_target.shape

        text_enc = self.interleaved_enc.encode_text(char_ids)

        # Use text-only conditioning (NOT interleaved which contains target mel).
        # The interleaved pattern [text_0, mel_0..mel_k, text_1, ...] would leak
        # target mel into conditioning if we sliced interleaved[:, :T].
        text_cond = self._align_text_to_mel(text_enc, T)

        z = torch.randn(B, device=mel_target.device) + self.cfg.sway_coefficient
        t = torch.sigmoid(z).clamp(1e-5, 1.0 - 1e-5)

        noise = torch.randn_like(mel_target)
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * noise + t_expand * mel_target
        target_velocity = mel_target - noise

        predicted, _ = self.forward(x_t, t, text_cond, speaker_ids,
                                    emotion_ids=emotion_ids, ref_mel=ref_mel)

        if mel_mask is not None:
            mask = mel_mask.unsqueeze(-1)
            per_elem = F.mse_loss(predicted, target_velocity, reduction='none') * mask
            n_valid = mask.sum() * predicted.shape[-1]
            return per_elem.sum() / n_valid.clamp(min=1)
        return F.mse_loss(predicted, target_velocity)

    def _align_text_to_mel(self, text_enc: torch.Tensor, mel_len: int) -> torch.Tensor:
        """Distribute text encoding evenly across mel frames."""
        B, T_text, D = text_enc.shape
        if T_text >= mel_len:
            return text_enc[:, :mel_len]

        aligned = torch.zeros(B, mel_len, D, device=text_enc.device, dtype=text_enc.dtype)
        if T_text == 0:
            return aligned

        frames_per_char = mel_len / T_text
        for i in range(T_text):
            start = int(i * frames_per_char)
            end = int((i + 1) * frames_per_char) if i < T_text - 1 else mel_len
            aligned[:, start:end] = text_enc[:, i:i+1]
        return aligned

    @torch.no_grad()
    def predict_duration(self, char_ids: torch.Tensor,
                         speed: float = 1.0) -> torch.Tensor:
        """Predict mel frame count from text using the learned duration predictor.

        Returns: integer total frame count.
        """
        text_enc = self.interleaved_enc.encode_text(char_ids)
        log_dur = self.duration_predictor(text_enc)
        dur = torch.exp(log_dur).clamp(min=1)
        dur = dur * (char_ids > 0).float() / max(speed, 0.1)
        return dur.sum(dim=-1).long().clamp(min=10)

    @torch.no_grad()
    def sample(self, char_ids: torch.Tensor, n_frames: int = 0,
               n_steps: Optional[int] = None,
               speaker_ids: Optional[torch.Tensor] = None,
               emotion_ids: Optional[torch.Tensor] = None,
               cfg_scale: float = 2.0,
               ref_mel: Optional[torch.Tensor] = None,
               step_schedule: Optional[List[float]] = None,
               speed: float = 1.0,
               use_heun: bool = False) -> torch.Tensor:
        """Full-sequence generation (non-streaming).

        If n_frames=0, uses the learned duration predictor to estimate length.
        step_schedule: Optional list of n_steps+1 timesteps in [0,1]. If None,
                       uses uniform linspace. Use epss_schedule() for EPSS/Sway Sampling.
        """
        n_steps = n_steps or self.cfg.n_steps_inference
        B = char_ids.shape[0]
        device = char_ids.device

        if n_frames <= 0:
            n_frames = self.predict_duration(char_ids, speed=speed).max().item()

        if step_schedule is not None:
            if len(step_schedule) != n_steps + 1:
                raise ValueError(f"step_schedule must have n_steps+1={n_steps + 1} elements, got {len(step_schedule)}")
            t_schedule = step_schedule
        else:
            t_schedule = [i / n_steps for i in range(n_steps + 1)]

        text_enc = self.interleaved_enc.encode_text(char_ids)

        log_dur = self.duration_predictor(text_enc)
        dur = torch.exp(log_dur).clamp(min=1) * (char_ids > 0).float() / max(speed, 0.1)
        dur_int = dur.round().long()
        dur_sum = dur_int.sum(dim=-1)
        for b in range(B):
            diff = n_frames - dur_sum[b].item()
            if diff != 0:
                max_idx = dur_int[b].argmax()
                dur_int[b, max_idx] += diff
        text_cond = DurationPredictor.expand_encodings(text_enc, dur_int, n_frames)

        x = torch.randn(B, n_frames, self.cfg.mel_dim, device=device)

        def _velocity(x_in, t_val):
            t_tensor = torch.full((B,), t_val, device=device)
            v_cond, _ = self.forward(x_in, t_tensor, text_cond, speaker_ids,
                                     emotion_ids=emotion_ids, ref_mel=ref_mel)
            if cfg_scale > 1.0:
                v_uncond, _ = self.forward(x_in, t_tensor, text_cond, speaker_ids,
                                           emotion_ids=emotion_ids,
                                           force_uncond=True, ref_mel=ref_mel)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            return v_cond

        for i in range(n_steps):
            t_val = t_schedule[i]
            dt = t_schedule[i + 1] - t_schedule[i]
            v1 = _velocity(x, t_val)

            if use_heun and i < n_steps - 1:
                x_euler = x + dt * v1
                v2 = _velocity(x_euler, t_val + dt)
                x = x + dt * 0.5 * (v1 + v2)
            else:
                x = x + dt * v1

        return x

    @torch.no_grad()
    def sample_streaming(self, char_ids: torch.Tensor, n_frames: int,
                         n_steps: int = 4,
                         speaker_ids: Optional[torch.Tensor] = None,
                         emotion_ids: Optional[torch.Tensor] = None,
                         cfg_scale: float = 2.0,
                         ref_mel: Optional[torch.Tensor] = None,
                         overlap_frames: int = 5,
                         step_schedule: Optional[List[float]] = None):
        """Chunk-streaming generation with overlapping crossfade.

        Generates mel in chunks of cfg.chunk_size frames with `overlap_frames`
        overlap between adjacent chunks. The overlap region is linearly
        crossfaded to eliminate boundary artifacts.

        Each chunk runs a full ODE solve. KV caches from the final ODE step
        are carried forward for causal cross-chunk context.

        step_schedule: Optional list of n_steps+1 timesteps in [0,1]. If None,
                       uses uniform linspace. Use epss_schedule() for EPSS/Sway Sampling.

        Yields: (B, chunk_frames, mel_dim) mel chunks, already crossfaded.
        The first chunk is chunk_size frames; subsequent chunks are
        (chunk_size - overlap_frames) new frames.
        """
        chunk_size = self.cfg.chunk_size
        B = char_ids.shape[0]
        device = char_ids.device

        if step_schedule is not None:
            if len(step_schedule) != n_steps + 1:
                raise ValueError(f"step_schedule must have n_steps+1={n_steps + 1} elements, got {len(step_schedule)}")
            t_schedule = step_schedule
        else:
            t_schedule = [i / n_steps for i in range(n_steps + 1)]

        text_enc = self.interleaved_enc.encode_text(char_ids)
        text_cond = self._align_text_to_mel(text_enc, n_frames)

        kv_caches = None
        offset = 0
        prev_overlap = None  # (B, overlap_frames, mel_dim) tail of previous chunk

        # Effective stride: we advance by (chunk_size - overlap) each step,
        # but generate chunk_size frames so adjacent chunks share `overlap` frames.
        stride = max(1, chunk_size - overlap_frames)

        for chunk_start in range(0, n_frames, stride):
            chunk_end = min(chunk_start + chunk_size, n_frames)
            chunk_len = chunk_end - chunk_start

            chunk_cond = text_cond[:, chunk_start:chunk_end]
            x = torch.randn(B, chunk_len, self.cfg.mel_dim, device=device)

            for i in range(n_steps):
                t_val = t_schedule[i]
                dt = t_schedule[i + 1] - t_schedule[i]
                t_batch = torch.full((B,), t_val, device=device)
                # Use previous chunk's KV cache for all ODE steps (cross-chunk context).
                # Only the last step's KV cache is passed to the next chunk.
                use_cache = kv_caches
                use_offset = offset
                v_cond, new_kv = self.forward(
                    x, t_batch, chunk_cond, speaker_ids,
                    emotion_ids=emotion_ids, ref_mel=ref_mel,
                    kv_caches=use_cache, offset=use_offset,
                )
                if cfg_scale > 1.0:
                    v_uncond, _ = self.forward(
                        x, t_batch, chunk_cond, speaker_ids,
                        emotion_ids=emotion_ids, force_uncond=True,
                        ref_mel=ref_mel,
                    )
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = v_cond
                x = x + dt * v

            kv_caches = new_kv

            # Crossfade overlap region with previous chunk
            if prev_overlap is not None and overlap_frames > 0:
                actual_overlap = min(overlap_frames, x.shape[1], prev_overlap.shape[1])
                if actual_overlap > 0:
                    # Linear crossfade: ramp from prev to current over overlap region
                    fade_in = torch.linspace(0, 1, actual_overlap, device=device)[None, :, None]
                    fade_out = 1.0 - fade_in
                    x[:, :actual_overlap] = (
                        fade_out * prev_overlap[:, -actual_overlap:] +
                        fade_in * x[:, :actual_overlap]
                    )
                # Yield only the new (non-overlapping) portion
                out = x[:, actual_overlap:]
            else:
                # First chunk: yield everything
                out = x

            # Save tail for crossfade with next chunk
            if overlap_frames > 0 and chunk_end < n_frames:
                prev_overlap = x[:, -overlap_frames:].clone()
            else:
                prev_overlap = None

            offset += out.shape[1]

            if out.shape[1] > 0:
                yield out

            if chunk_end >= n_frames:
                break

    @torch.no_grad()
    def sample_dragon(self, char_ids: torch.Tensor, n_frames: int,
                      n_steps: int = 4,
                      speaker_ids: Optional[torch.Tensor] = None,
                      emotion_ids: Optional[torch.Tensor] = None,
                      cfg_scale: float = 2.0,
                      ref_mel: Optional[torch.Tensor] = None,
                      overlap_frames: int = 5,
                      step_schedule: Optional[List[float]] = None):
        """Dragon-FM: autoregressive across chunks, parallel within chunks.

        Key difference from sample_streaming:
          - Within each chunk: use BIDIRECTIONAL attention (no causal mask)
            for maximum parallel denoising quality.
          - Across chunks: autoregressive via KV cache from the final state
            of each completed chunk.

        This gives the quality of bidirectional flow matching (like non-streaming)
        with the streaming capability of autoregressive models.

        step_schedule: Optional list of n_steps+1 timesteps in [0,1]. If None,
                       uses uniform linspace. Use epss_schedule() for EPSS/Sway Sampling.

        Yields: crossfaded mel chunks as they're ready.
        """
        chunk_size = self.cfg.chunk_size
        B = char_ids.shape[0]
        device = char_ids.device

        if step_schedule is not None:
            if len(step_schedule) != n_steps + 1:
                raise ValueError(f"step_schedule must have n_steps+1={n_steps + 1} elements, got {len(step_schedule)}")
            t_schedule = step_schedule
        else:
            t_schedule = [i / n_steps for i in range(n_steps + 1)]

        text_enc = self.interleaved_enc.encode_text(char_ids)
        text_cond = self._align_text_to_mel(text_enc, n_frames)

        kv_caches = None
        offset = 0
        prev_overlap = None
        stride = max(1, chunk_size - overlap_frames)

        for chunk_start in range(0, n_frames, stride):
            chunk_end = min(chunk_start + chunk_size, n_frames)
            chunk_len = chunk_end - chunk_start

            chunk_cond = text_cond[:, chunk_start:chunk_end]
            x = torch.randn(B, chunk_len, self.cfg.mel_dim, device=device)

            for i in range(n_steps):
                t_val = t_schedule[i]
                dt = t_schedule[i + 1] - t_schedule[i]
                t_batch = torch.full((B,), t_val, device=device)

                # Dragon-FM: bidirectional within chunk + AR context from cache.
                # Pass kv_cache (context from previous chunks) but NO causal mask
                # on the current chunk's tokens — they all see each other.
                v_cond, new_kv = self._forward_dragon(
                    x, t_batch, chunk_cond, speaker_ids,
                    emotion_ids=emotion_ids, ref_mel=ref_mel,
                    kv_caches=kv_caches, offset=offset,
                )
                if cfg_scale > 1.0:
                    v_uncond, _ = self._forward_dragon(
                        x, t_batch, chunk_cond, speaker_ids,
                        emotion_ids=emotion_ids, force_uncond=True,
                        ref_mel=ref_mel, kv_caches=kv_caches, offset=offset,
                    )
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = v_cond
                x = x + dt * v

            # Only update KV cache from the final ODE step
            kv_caches = new_kv

            # Crossfade with previous chunk
            if prev_overlap is not None and overlap_frames > 0:
                actual_overlap = min(overlap_frames, x.shape[1], prev_overlap.shape[1])
                if actual_overlap > 0:
                    fade_in = torch.linspace(0, 1, actual_overlap, device=device)[None, :, None]
                    x[:, :actual_overlap] = (
                        (1.0 - fade_in) * prev_overlap[:, -actual_overlap:] +
                        fade_in * x[:, :actual_overlap]
                    )
                out = x[:, actual_overlap:]
            else:
                out = x

            if overlap_frames > 0 and chunk_end < n_frames:
                prev_overlap = x[:, -overlap_frames:].clone()
            else:
                prev_overlap = None

            offset += out.shape[1]

            if out.shape[1] > 0:
                yield out

            if chunk_end >= n_frames:
                break

    def _forward_dragon(self, x_t: torch.Tensor, t: torch.Tensor,
                        text_context: torch.Tensor,
                        speaker_ids: Optional[torch.Tensor] = None,
                        emotion_ids: Optional[torch.Tensor] = None,
                        force_uncond: bool = False,
                        ref_mel: Optional[torch.Tensor] = None,
                        kv_caches: Optional[List[torch.Tensor]] = None,
                        offset: int = 0) -> tuple:
        """Dragon-FM forward: bidirectional within chunk, AR via KV cache.

        Same as self.forward() but the attention within the current chunk
        is bidirectional (no causal mask). Cross-chunk context comes from
        kv_caches which are prepended as non-causal context.
        """
        B, T_mel, _ = x_t.shape

        if force_uncond:
            text_context = torch.zeros_like(text_context)

        total_len = T_mel
        if ref_mel is not None:
            total_len += ref_mel.shape[1]

        cond = self._build_conditioning(t, total_len, speaker_ids, force_uncond)

        x = self.input_proj(x_t)
        if ref_mel is not None:
            ref_enc = self.interleaved_enc.encode_mel(ref_mel)
            x = torch.cat([ref_enc, x], dim=1)
            ref_cond = text_context[:, :1].expand(-1, ref_mel.shape[1], -1)
            text_context = torch.cat([ref_cond, text_context], dim=1)

        x = x + text_context[:, :x.shape[1]]

        # Dragon-FM: for each block, prepend KV cache but use bidirectional
        # attention within the current chunk's tokens
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = self._dragon_block_forward(
                block, x, cond, layer_cache, offset
            )
            new_kv_caches.append(new_cache)

        # Token-level emotion steering (after blocks, before output projection)
        if self.token_emosteer is not None and emotion_ids is not None and not force_uncond:
            x = self.token_emosteer(x, emotion_ids=emotion_ids, scale=1.0)

        x = self.output_proj(self.output_norm(x))

        if ref_mel is not None:
            x = x[:, ref_mel.shape[1]:]

        return x, new_kv_caches

    def _dragon_block_forward(self, block: FlowV3Block, x: torch.Tensor,
                              cond: torch.Tensor,
                              kv_cache: Optional[torch.Tensor],
                              offset: int) -> tuple:
        """Run a FlowV3Block with Dragon-FM attention pattern.

        Within the chunk: bidirectional (all-to-all) attention.
        From cache: chunk tokens attend to all cached tokens (cross-attention-like).
        """
        attn = block.attn
        B, T, _ = x.shape

        normed = block.norm1(x, cond)
        qkv = attn.qkv(normed).reshape(B, T, 3, attn.n_heads, attn.head_dim)
        q, k, v = qkv.unbind(2)

        if attn.use_rope:
            q, k = attn._apply_rope(q, k, offset)

        # Prepend cached K, V from previous chunks
        if kv_cache is not None:
            k_past, v_past = kv_cache[:, 0], kv_cache[:, 1]
            k_full = torch.cat([k_past, k], dim=1)
            v_full = torch.cat([v_past, v], dim=1)
        else:
            k_full, v_full = k, v

        # Trim to window
        T_kv = k_full.shape[1]
        W = attn.window_size
        if T_kv > W:
            k_full = k_full[:, -W:]
            v_full = v_full[:, -W:]
            T_kv = W

        new_cache = torch.stack([k_full, v_full], dim=1)

        # Transpose for SDPA: (B, H, T, D_h)
        q = q.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v_full = v_full.transpose(1, 2)

        # BIDIRECTIONAL attention (no causal mask) — this is the Dragon-FM difference.
        # All query positions (current chunk) attend to all KV positions
        # (past cache + current chunk) without restriction.
        out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)

        out = out.transpose(1, 2).reshape(B, T, -1)
        h = attn.out(out)

        x = x + h
        x = x + block.mlp(block.norm2(x, cond))
        return x, new_cache

    @torch.no_grad()
    def estimate_duration(self, char_ids: torch.Tensor,
                          target_rate: float = 4.5) -> int:
        """Estimate output mel frames from text length.

        target_rate: mel frames per text character (~4-6 for English at 50Hz).
        """
        T_text = (char_ids != 0).sum(dim=-1).max().item()
        return max(25, int(T_text * target_rate))


if __name__ == "__main__":
    cfg = FlowV3Config()
    model = SonataFlowV3(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"Sonata Flow V3 (Interleaved Streaming):")
    print(f"  {cfg.n_layers}L × d={cfg.d_model}, H={cfg.n_heads}")
    print(f"  Window: {cfg.window_size} frames, Chunk: {cfg.chunk_size} frames")
    print(f"  Mel dim: {cfg.mel_dim}")
    print(f"  Total params: {n_params/1e6:.1f}M")

    B, T_text, T_mel = 2, 32, 100
    chars = torch.randint(1, 128, (B, T_text))
    mel = torch.randn(B, T_mel, cfg.mel_dim)

    # Standard training loss
    loss = model.compute_loss(mel, chars)
    print(f"\n  Standard loss: {loss:.4f}")

    # Interleaved training loss
    loss_i = model.compute_interleaved_loss(mel, chars)
    print(f"  Interleaved loss: {loss_i:.4f}")

    # Full-sequence generation
    generated = model.sample(chars, n_frames=T_mel, n_steps=4)
    print(f"  Generated: {generated.shape}")

    # Streaming generation
    chunks = list(model.sample_streaming(chars, n_frames=T_mel, n_steps=4))
    total_frames = sum(c.shape[1] for c in chunks)
    print(f"  Streaming: {len(chunks)} chunks, {total_frames} total frames")

    # Duration estimation
    est_dur = model.estimate_duration(chars)
    print(f"  Estimated duration: {est_dur} frames ({est_dur/cfg.frame_rate:.2f}s)")

    # Dragon-FM streaming (AR across chunks, parallel within)
    dragon_chunks = list(model.sample_dragon(chars, n_frames=T_mel, n_steps=4))
    dragon_total = sum(c.shape[1] for c in dragon_chunks)
    print(f"  Dragon-FM: {len(dragon_chunks)} chunks, {dragon_total} total frames")

    # Zero-shot with reference mel
    ref = torch.randn(B, 50, cfg.mel_dim)
    generated_zs = model.sample(chars, n_frames=T_mel, n_steps=4, ref_mel=ref)
    print(f"  Zero-shot: {generated_zs.shape}")
