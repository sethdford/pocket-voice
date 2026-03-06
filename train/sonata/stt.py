"""Sonata STT — Two-pass speech recognition on Sonata Codec representations.

Pass 1 (SonataCTC): CTC head on codec conformer encoder.
  Audio → Mel → SpecAugment → Conformer (RoPE) → CTC projection → streaming text

Pass 2 (SonataRefiner): Encoder-decoder on discrete semantic tokens.
  Semantic tokens (50Hz, 32768 vocab) → Transformer encoder → Cross-attention decoder → text

Together they form a symmetric counterpart to the TTS pipeline:
  STT: Audio → Codec Encoder → Semantic Tokens → Refiner → Text
  TTS: Text → Sonata LM → Semantic Tokens → Flow → Audio
"""

import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import STTConfig, RefinerConfig, CodecConfig
from codec import MelSpectrogram


# ═════════════════════════════════════════════════════════════════════════════
# Pass 1: CTC on Codec Encoder
# ═════════════════════════════════════════════════════════════════════════════

CTC_VOCAB = [
    "<blank>", " ",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "'", "<eou>",
]

def text_to_ctc_ids(text: str, append_eou: bool = False) -> list:
    text = text.lower().strip()
    ids = []
    for c in text:
        if c == " ":
            ids.append(1)
        elif c == "'":
            ids.append(28)
        elif "a" <= c <= "z":
            ids.append(ord(c) - ord("a") + 2)
    if append_eou:
        ids.append(29)
    return ids


def ctc_ids_to_text(ids: list) -> str:
    chars = []
    prev = -1
    for idx in ids:
        if idx == 0:  # blank
            prev = idx
            continue
        if idx == prev:
            continue
        prev = idx
        if idx == 29:  # <eou>
            break
        if idx == 1:
            chars.append(" ")
        elif idx == 28:
            chars.append("'")
        elif 2 <= idx <= 27:
            chars.append(chr(ord("a") + idx - 2))
    return "".join(chars).strip()


# ─── SentencePiece BPE Subword Tokenizer for CTC ────────────────────────

class SubwordCTCTokenizer:
    """SentencePiece BPE tokenizer for CTC with blank(0) and <eou> tokens.

    Vocab layout: blank(0), <unk>(1), subword tokens(2..V-2), <eou>(V-1).
    The ▁ prefix in SentencePiece tokens represents word boundaries (spaces).
    """

    def __init__(self, model_path: str):
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("pip install sentencepiece")
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self._vocab_size = self.sp.get_piece_size() + 2  # +blank +eou

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def eou_id(self) -> int:
        return self._vocab_size - 1

    def encode(self, text: str, append_eou: bool = False) -> list:
        """Encode text → list of CTC token IDs (offset by 1 for blank at 0)."""
        sp_ids = self.sp.encode(text.lower().strip(), out_type=int)
        ids = [i + 1 for i in sp_ids]  # shift by 1 for blank
        if append_eou:
            ids.append(self.eou_id)
        return ids

    def decode(self, ids: list) -> str:
        """CTC-collapse and decode token IDs → text."""
        collapsed = []
        prev = -1
        for idx in ids:
            if idx == 0:  # blank
                prev = idx
                continue
            if idx == prev:
                continue
            if idx == self.eou_id:
                break
            prev = idx
            collapsed.append(idx - 1)  # undo blank offset
        return self.sp.decode(collapsed)

    def get_vocab_strings(self) -> list:
        """Get vocab as string list for beam decoder initialization."""
        vocab = ["<blank>"]
        for i in range(self.sp.get_piece_size()):
            vocab.append(self.sp.id_to_piece(i))
        vocab.append("<eou>")
        return vocab

    @staticmethod
    def train(text_file: str, output_prefix: str, vocab_size: int = 1024):
        """Train a SentencePiece BPE model from a text file."""
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("pip install sentencepiece")
        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=output_prefix,
            vocab_size=vocab_size - 2,  # reserve blank + eou
            model_type="bpe",
            character_coverage=1.0,
            normalization_rule_name="identity",
            add_dummy_prefix=True,
        )
        print(f"  [SPM] Trained BPE model: {output_prefix}.model"
              f" (vocab={vocab_size}, blank=0, eou={vocab_size-1})")


# ─── SpecAugment ──────────────────────────────────────────────────────────

def spec_augment(mel: torch.Tensor, cfg: STTConfig) -> torch.Tensor:
    """Apply SpecAugment (time + frequency masking) to mel spectrogram.
    mel: (B, n_mels, T_frames) — log-mel spectrogram
    """
    if not cfg.spec_augment:
        return mel
    B, F_dim, T = mel.shape
    mel = mel.clone()
    for b in range(B):
        for _ in range(cfg.n_freq_masks):
            f = random.randint(0, min(cfg.freq_mask_width, F_dim - 1))
            f0 = random.randint(0, F_dim - f)
            mel[b, f0:f0 + f, :] = 0.0
        for _ in range(cfg.n_time_masks):
            t = random.randint(0, min(cfg.time_mask_width, T - 1))
            t0 = random.randint(0, T - t)
            mel[b, :, t0:t0 + t] = 0.0
    return mel


def speed_perturb(audio: torch.Tensor, factor: float) -> torch.Tensor:
    """Resample audio by speed factor. factor>1 = faster, factor<1 = slower."""
    if abs(factor - 1.0) < 1e-4:
        return audio
    B, T = audio.shape
    new_T = int(T / factor)
    return F.interpolate(audio.unsqueeze(1), size=new_T, mode="linear",
                         align_corners=False).squeeze(1)


# ─── Noise and Reverb Augmentation ────────────────────────────────────────

def _additive_noise(audio: torch.Tensor, snr_db_min: float, snr_db_max: float,
                    noise_type: str = "white") -> torch.Tensor:
    """Add random noise at random SNR. Pure torch, no external deps.
    audio: (T,) or (B, T). Returns same shape."""
    dims = audio.dim()
    if dims == 1:
        audio = audio.unsqueeze(0)
    B, T = audio.shape

    # Generate noise
    if noise_type == "pink":
        # Pink-like: convolve white noise with short exponential kernel (low-pass)
        white = torch.randn(B, T, device=audio.device)
        k_len = min(64, T // 4)
        alpha = 0.9
        decay = torch.pow(alpha, torch.arange(k_len, dtype=audio.dtype, device=audio.device))
        kernel = (decay / decay.sum()).unsqueeze(0).unsqueeze(0)
        noise = F.conv1d(white.unsqueeze(1), kernel, padding=k_len // 2).squeeze(1)
        if noise.shape[1] != T:
            noise = F.interpolate(noise.unsqueeze(1), size=T, mode="linear").squeeze(1)
    else:
        noise = torch.randn(B, T, device=audio.device, dtype=audio.dtype)

    # SNR in dB: signal_rms / noise_rms = 10^(snr_db/20)
    sig_rms = audio.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
    snr_db = snr_db_min + (snr_db_max - snr_db_min) * torch.rand(B, 1, device=audio.device)
    noise_scale = sig_rms / (10 ** (snr_db / 20))
    noise_rms = noise.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
    noise = (noise / noise_rms) * noise_scale
    audio = audio + noise

    return audio.squeeze(0) if dims == 1 else audio


def _synthetic_reverb(audio: torch.Tensor, sample_rate: int,
                      rt60_min: float, rt60_max: float) -> torch.Tensor:
    """Simple synthetic reverb: exponentially decaying noise convolution.
    RT60 in seconds. Pure torch."""
    dims = audio.dim()
    if dims == 1:
        audio = audio.unsqueeze(0)
    B, T = audio.shape

    rt60 = rt60_min + (rt60_max - rt60_min) * random.random()
    # Decay: -60 dB at t=RT60 => exp(-6.9 * n / (RT60 * sr)) at sample n
    decay_coef = 6.9 / (rt60 * sample_rate)
    ir_len = min(int(rt60 * sample_rate * 2), T)  # 2x RT60 length
    t = torch.arange(ir_len, dtype=audio.dtype, device=audio.device)
    ir = torch.randn(ir_len, device=audio.device, dtype=audio.dtype) * torch.exp(-decay_coef * t)
    ir = ir / (ir.abs().max() + 1e-8)

    # Convolve per batch (pad to avoid circular wrap)
    pad = ir_len - 1
    padded = F.pad(audio, (pad, 0), mode="reflect")
    ir_b = ir.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
    out = F.conv1d(padded.unsqueeze(1), ir_b, padding=0).squeeze(1)
    out = out[:, :T]
    out = out * (audio.abs().max() / (out.abs().max() + 1e-8))

    return out.squeeze(0) if dims == 1 else out


def _volume_perturb(audio: torch.Tensor, gain_min: float, gain_max: float) -> torch.Tensor:
    """Random gain scaling."""
    gain = gain_min + (gain_max - gain_min) * random.random()
    return audio * gain


def _random_freq_filter(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Random low-pass or high-pass to simulate phone/speaker quality.
    Uses simple moving-average style convolution."""
    dims = audio.dim()
    if dims == 1:
        audio = audio.unsqueeze(0)
    B, T = audio.shape

    is_lowpass = random.random() < 0.5
    # Random cutoff roughly 500 Hz - 12 kHz for 24kHz sr
    fc = 500 + (12000 - 500) * random.random()
    k_len = max(2, min(int(sample_rate / fc), T // 2))
    if k_len % 2 == 0:
        k_len += 1

    kernel = torch.ones(1, 1, k_len, device=audio.device, dtype=audio.dtype) / k_len
    low = F.conv1d(audio.unsqueeze(1), kernel, padding=k_len // 2).squeeze(1)
    if low.shape[1] != T:
        low = F.interpolate(low.unsqueeze(1), size=T, mode="linear").squeeze(1)

    out = low if is_lowpass else (audio - low)
    return out.squeeze(0) if dims == 1 else out


def apply_noise_augmentations(audio: torch.Tensor, sample_rate: int, cfg: STTConfig) -> torch.Tensor:
    """Apply noise, reverb, volume, and filter augmentations with configurable probs.
    audio: (T,) 1D waveform. Returns augmented (T,)."""
    if not cfg.noise_augment:
        return audio
    snr_min, snr_max = cfg.noise_snr_range
    rt60_min, rt60_max = cfg.reverb_rt60_range
    prob = cfg.augment_prob

    if random.random() < prob:
        audio = _additive_noise(audio.unsqueeze(0), snr_min, snr_max,
                               noise_type=random.choice(["white", "pink"])).squeeze(0)
    if random.random() < prob:
        audio = _synthetic_reverb(audio.unsqueeze(0), sample_rate, rt60_min, rt60_max).squeeze(0)
    if random.random() < prob:
        audio = _volume_perturb(audio, 0.5, 2.0)
    if random.random() < prob:
        audio = _random_freq_filter(audio.unsqueeze(0), sample_rate).squeeze(0)

    return audio


# ─── RoPE Conformer Encoder ──────────────────────────────────────────────

def precompute_rope(dim: int, max_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope_qk(q, k, freqs):
    def rotate(x, f):
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        f = f[None, :x_c.shape[1], None, :]
        return torch.view_as_real(x_c * f).flatten(-2).to(x.dtype)
    return rotate(q, freqs), rotate(k, freqs)


class RoPEConformerMHSA(nn.Module):
    """Multi-head self-attention with Rotary Positional Encoding."""

    def __init__(self, dim: int, n_heads: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "rope_freqs",
            precompute_rope(self.head_dim, max_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm(x)
        q = self.wq(h).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(h).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(h).view(B, T, self.n_heads, self.head_dim)

        q, k = apply_rope_qk(q, k, self.rope_freqs[:T])

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        out = self.dropout(out)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class ConformerFeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pointwise1 = nn.Conv1d(dim, 2 * dim, 1)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size,
                                    padding=kernel_size // 2, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise2 = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class STTConformerBlock(nn.Module):
    """Conformer block with RoPE attention: FF/2 → MHSA → Conv → FF/2 → LN."""

    def __init__(self, dim, n_heads, conv_kernel=31, ff_mult=4.0,
                 use_rope=True, dropout=0.0):
        super().__init__()
        self.ff1 = ConformerFeedForward(dim, ff_mult, dropout)
        if use_rope:
            self.mhsa = RoPEConformerMHSA(dim, n_heads, dropout=dropout)
        else:
            from codec import ConformerMHSA
            self.mhsa = ConformerMHSA(dim, n_heads, dropout)
        self.conv = ConformerConvModule(dim, conv_kernel, dropout)
        self.ff2 = ConformerFeedForward(dim, ff_mult, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class STTConformerEncoder(nn.Module):
    """Conformer encoder with RoPE, configurable depth/width for STT."""

    def __init__(self, cfg: STTConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.n_mels, cfg.enc_dim)
        self.blocks = nn.ModuleList([
            STTConformerBlock(
                cfg.enc_dim, cfg.enc_n_heads, cfg.enc_conv_kernel,
                cfg.enc_ff_mult, cfg.use_rope,
            )
            for _ in range(cfg.enc_n_layers)
        ])

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, T) → hidden: (B, T, enc_dim)"""
        x = mel.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x


class SonataCTC(nn.Module):
    """CTC head on Sonata conformer encoder.

    Supports both the codec-derived encoder (4L d=256, frozen) and a
    standalone STT-optimized encoder (12L d=512, trainable). Includes
    SpecAugment, speed perturbation, RoPE attention, and EOU detection.
    """

    def __init__(self, cfg: STTConfig):
        super().__init__()
        self.cfg = cfg

        codec_cfg = CodecConfig(
            n_mels=cfg.n_mels, enc_dim=cfg.enc_dim,
            enc_n_layers=cfg.enc_n_layers, enc_n_heads=cfg.enc_n_heads,
            enc_conv_kernel=cfg.enc_conv_kernel, enc_ff_mult=cfg.enc_ff_mult,
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
            hop_length=cfg.hop_length, win_length=cfg.win_length,
        )
        self.mel = MelSpectrogram(codec_cfg)

        if cfg.use_rope:
            self.encoder = STTConformerEncoder(cfg)
        else:
            from codec import ConformerEncoder
            self.encoder = ConformerEncoder(codec_cfg)

        self.adapter = nn.Sequential(
            nn.LayerNorm(cfg.enc_dim),
            nn.Linear(cfg.enc_dim, cfg.enc_dim),
            nn.SiLU(),
            nn.Dropout(cfg.ctc_dropout),
        )
        self.ctc_proj = nn.Linear(cfg.enc_dim, cfg.text_vocab_size)

    def load_codec_encoder(self, codec_ckpt_path: str, freeze: bool = True):
        """Load conformer encoder weights from a trained codec checkpoint.

        Handles conversion from nn.MultiheadAttention (fused QKV) to
        separate RoPE attention weights (wq, wk, wv, wo).
        """
        ckpt = torch.load(codec_ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt

        enc_state = {}
        mel_state = {}
        for k, v in state.items():
            if k.startswith("semantic_encoder."):
                new_k = k.replace("semantic_encoder.", "")
                enc_state[new_k] = v
            elif k.startswith("mel."):
                new_k = k.replace("mel.", "")
                mel_state[new_k] = v

        if self.cfg.use_rope:
            converted = {}
            skip_keys = set()
            d = self.cfg.enc_dim
            for k, v in enc_state.items():
                if "mhsa.attn.in_proj_weight" in k:
                    prefix = k.replace("attn.in_proj_weight", "")
                    converted[f"{prefix}wq.weight"] = v[:d]
                    converted[f"{prefix}wk.weight"] = v[d:2*d]
                    converted[f"{prefix}wv.weight"] = v[2*d:]
                    skip_keys.add(k)
                elif "mhsa.attn.in_proj_bias" in k:
                    skip_keys.add(k)
                elif "mhsa.attn.out_proj.weight" in k:
                    prefix = k.replace("attn.out_proj.weight", "")
                    converted[f"{prefix}wo.weight"] = v
                    skip_keys.add(k)
                elif "mhsa.attn.out_proj.bias" in k:
                    skip_keys.add(k)

            for k, v in enc_state.items():
                if k not in skip_keys:
                    converted[k] = v
            enc_state = converted

        loaded_enc = self.encoder.load_state_dict(enc_state, strict=False)
        loaded_mel = self.mel.load_state_dict(mel_state, strict=False)
        print(f"  [CTC] Loaded encoder: {len(enc_state)} tensors"
              f" (missing={loaded_enc.missing_keys})")
        print(f"  [CTC] Loaded mel: {len(mel_state)} tensors"
              f" (missing={loaded_mel.missing_keys})")

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.mel.parameters():
                p.requires_grad = False
            print(f"  [CTC] Encoder + mel frozen")

    def forward(self, audio: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """audio: (B, T_samples) → logits: (B, T_frames, text_vocab_size)"""
        mel = self.mel(audio)
        if augment:
            mel = spec_augment(mel, self.cfg)
        if self.cfg.use_rope:
            x = self.encoder(mel)
        else:
            x = mel.transpose(1, 2)
            x = self.encoder.input_proj(x)
            for block in self.encoder.blocks:
                x = block(x)
        x = self.adapter(x)
        return self.ctc_proj(x)

    def recognize(self, audio: torch.Tensor) -> list:
        """Greedy CTC decode → list of text strings."""
        logits = self.forward(audio)
        preds = logits.argmax(dim=-1)
        results = []
        for b in range(preds.shape[0]):
            results.append(ctc_ids_to_text(preds[b].tolist()))
        return results

    def eou_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract per-frame EOU probability from CTC logits.
        logits: (B, T, vocab) → eou_prob: (B, T)
        """
        probs = logits.softmax(dim=-1)
        return probs[:, :, self.cfg.eou_id]


# ═════════════════════════════════════════════════════════════════════════════
# Pass 2: Semantic Token → Text Refiner (Encoder-Decoder)
# ═════════════════════════════════════════════════════════════════════════════

# Semantic pad index for Refiner: must not be a valid FSQ code (0..32767 for 8^5)
SEMANTIC_PAD_ID = 32768

from modules import RMSNorm, precompute_rope_freqs, apply_rope


class RefinerEncoderBlock(nn.Module):
    """Bidirectional transformer block for encoding semantic token sequences."""

    def __init__(self, cfg: RefinerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.enc_d_model, cfg.norm_eps)
        self.attn = nn.MultiheadAttention(
            cfg.enc_d_model, cfg.enc_n_heads,
            dropout=cfg.dropout, batch_first=True,
        )
        self.ffn_norm = RMSNorm(cfg.enc_d_model, cfg.norm_eps)
        inner = cfg.enc_d_ff
        self.ffn = nn.Sequential(
            nn.Linear(cfg.enc_d_model, inner),
            nn.SiLU(),
            nn.Linear(inner, cfg.enc_d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """key_padding_mask: (B, T) True = pad position, should be ignored."""
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x


class RefinerDecoderBlock(nn.Module):
    """Causal decoder with cross-attention to semantic encoder output."""

    def __init__(self, cfg: RefinerConfig):
        super().__init__()
        d = cfg.dec_d_model
        self.self_attn_norm = RMSNorm(d, cfg.norm_eps)

        # Self-attention with GQA
        self.n_heads = cfg.dec_n_heads
        self.n_kv_heads = cfg.dec_n_kv_heads
        self.head_dim = cfg.dec_head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, d, bias=False)

        # Cross-attention
        self.cross_norm = RMSNorm(d, cfg.norm_eps)
        self.cross_q = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.cross_k = nn.Linear(cfg.enc_d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.cross_v = nn.Linear(cfg.enc_d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.cross_o = nn.Linear(self.n_heads * self.head_dim, d, bias=False)

        # FFN
        self.ffn_norm = RMSNorm(d, cfg.norm_eps)
        inner = cfg.dec_d_ff
        self.ffn = nn.Sequential(
            nn.Linear(d, inner),
            nn.SiLU(),
            nn.Linear(inner, d),
            nn.Dropout(cfg.dropout),
        )

    def _self_attn(self, x, freqs, mask):
        B, T, _ = x.shape
        h = self.self_attn_norm(x)
        q = self.wq(h).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(h).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(h).view(B, T, self.n_kv_heads, self.head_dim)
        q, k = apply_rope(q, k, freqs)

        if self.n_rep > 1:
            k = k[:, :, :, None, :].expand(B, T, self.n_kv_heads, self.n_rep, self.head_dim)
            k = k.reshape(B, T, self.n_heads, self.head_dim)
            v = v[:, :, :, None, :].expand(B, T, self.n_kv_heads, self.n_rep, self.head_dim)
            v = v.reshape(B, T, self.n_heads, self.head_dim)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))

    def _cross_attn(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """memory_key_padding_mask: (B, T_enc) True = pad encoder position."""
        B, T_dec, _ = x.shape
        _, T_enc, _ = enc_out.shape
        h = self.cross_norm(x)

        q = self.cross_q(h).view(B, T_dec, self.n_heads, self.head_dim)
        k = self.cross_k(enc_out).view(B, T_enc, self.n_kv_heads, self.head_dim)
        v = self.cross_v(enc_out).view(B, T_enc, self.n_kv_heads, self.head_dim)

        if self.n_rep > 1:
            k = k[:, :, :, None, :].expand(B, T_enc, self.n_kv_heads, self.n_rep, self.head_dim)
            k = k.reshape(B, T_enc, self.n_heads, self.head_dim)
            v = v[:, :, :, None, :].expand(B, T_enc, self.n_kv_heads, self.n_rep, self.head_dim)
            v = v.reshape(B, T_enc, self.n_heads, self.head_dim)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if memory_key_padding_mask is not None:
            scores = scores.masked_fill(
                memory_key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        return self.cross_o(out.transpose(1, 2).reshape(B, T_dec, -1))

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self._self_attn(x, freqs, mask)
        x = x + self._cross_attn(x, enc_out, memory_key_padding_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SonataRefiner(nn.Module):
    """Encoder-decoder transformer: semantic tokens → text.

    The encoder processes the full semantic token sequence bidirectionally.
    The decoder generates text tokens autoregressively with cross-attention
    to the encoder output. Architecturally the inverse of Sonata LM.
    """

    def __init__(self, cfg: RefinerConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder: semantic tokens → contextual representations
        self.sem_emb = nn.Embedding(cfg.semantic_vocab_size + 4, cfg.enc_d_model)
        self.sem_pos = nn.Embedding(cfg.max_audio_len, cfg.enc_d_model)
        self.enc_layers = nn.ModuleList([
            RefinerEncoderBlock(cfg) for _ in range(cfg.enc_n_layers)
        ])
        self.enc_norm = RMSNorm(cfg.enc_d_model, cfg.norm_eps)

        # Decoder: text generation with cross-attention
        self.text_emb = nn.Embedding(cfg.text_vocab_size + 4, cfg.dec_d_model)
        self.dec_layers = nn.ModuleList([
            RefinerDecoderBlock(cfg) for _ in range(cfg.dec_n_layers)
        ])
        self.dec_norm = RMSNorm(cfg.dec_d_model, cfg.norm_eps)
        self.output_proj = nn.Linear(cfg.dec_d_model, cfg.text_vocab_size + 4, bias=False)

        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(cfg.dec_head_dim, cfg.max_text_len, cfg.rope_theta),
            persistent=False,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode_semantic(
        self,
        semantic_tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode semantic tokens bidirectionally. (B, T_audio) → (B, T_audio, D)

        key_padding_mask: (B, T) True = pad position (should not attend to).
        """
        B, T = semantic_tokens.shape
        pos = torch.arange(T, device=semantic_tokens.device).unsqueeze(0)
        x = self.sem_emb(semantic_tokens) + self.sem_pos(pos)
        for layer in self.enc_layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.enc_norm(x)

    def forward(
        self,
        semantic_tokens: torch.Tensor,           # (B, T_audio)
        text_tokens: torch.Tensor,               # (B, T_text) — shifted right (teacher forcing)
        target_text: Optional[torch.Tensor] = None,  # (B, T_text)
        enc_out: Optional[torch.Tensor] = None,
        enc_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        B, T_text = text_tokens.shape

        if enc_out is None:
            enc_out = self.encode_semantic(semantic_tokens, key_padding_mask=enc_padding_mask)

        x = self.text_emb(text_tokens)

        mask = torch.full((T_text, T_text), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        freqs = self.rope_freqs[:T_text]
        for layer in self.dec_layers:
            x = layer(x, enc_out, freqs, mask, memory_key_padding_mask=enc_padding_mask)

        logits = self.output_proj(self.dec_norm(x))

        losses = {}
        if target_text is not None:
            losses["text"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_text.reshape(-1),
                ignore_index=self.cfg.text_pad_id,
            )

        return logits, losses

    @torch.no_grad()
    def generate(self, semantic_tokens: torch.Tensor, max_len: int = 256,
                 temperature: float = 0.0) -> list:
        """Autoregressive greedy/sampling decode. Returns list of token ID lists."""
        enc_padding_mask = None
        if semantic_tokens.dim() == 2:
            enc_padding_mask = semantic_tokens == SEMANTIC_PAD_ID
        enc_out = self.encode_semantic(semantic_tokens, key_padding_mask=enc_padding_mask)
        B = semantic_tokens.shape[0]
        device = semantic_tokens.device

        generated = torch.full((B, 1), self.cfg.text_bos_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            logits, _ = self.forward(semantic_tokens, generated, enc_out=enc_out)
            next_logits = logits[:, -1, :]

            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == self.cfg.text_eos_id).all():
                break

        results = []
        for b in range(B):
            tokens = generated[b, 1:].tolist()  # skip BOS
            if self.cfg.text_eos_id in tokens:
                tokens = tokens[:tokens.index(self.cfg.text_eos_id)]
            results.append(tokens)
        return results


# ═════════════════════════════════════════════════════════════════════════════
# Round-trip quality verification (Sonata TTS → Sonata STT via semantic tokens)
# ═════════════════════════════════════════════════════════════════════════════

def semantic_roundtrip_score(
    text: str,
    semantic_tokens_tts: torch.Tensor,
    refiner: "SonataRefiner",
    tokenizer_decode_fn=None,
) -> dict:
    """Verify round-trip: text → (LM) → semantic_tokens → (Refiner) → text'.

    Uses the shared semantic token representation as the round-trip bridge.
    No audio involved — tests the symbolic pipeline end-to-end.

    Returns dict with: original, reconstructed, char_match_rate.
    """
    with torch.no_grad():
        if semantic_tokens_tts.dim() == 1:
            semantic_tokens_tts = semantic_tokens_tts.unsqueeze(0)
        gen_ids = refiner.generate(semantic_tokens_tts, max_len=256, temperature=0.0)

    if tokenizer_decode_fn:
        reconstructed = tokenizer_decode_fn(gen_ids[0])
    else:
        chars = []
        for idx in gen_ids[0]:
            if idx == 0:
                continue
            if idx == 3:
                chars.append(" ")
            elif idx == 30:
                chars.append("'")
            elif 4 <= idx <= 29:
                chars.append(chr(ord("a") + idx - 4))
        reconstructed = "".join(chars).strip()

    orig_clean = text.lower().strip()
    recon_clean = reconstructed.lower().strip()

    matches = sum(1 for a, b in zip(orig_clean, recon_clean) if a == b)
    max_len = max(len(orig_clean), len(recon_clean), 1)
    char_match = matches / max_len

    return {
        "original": text,
        "reconstructed": reconstructed,
        "char_match_rate": char_match,
        "exact_match": orig_clean == recon_clean,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Quick test
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from config import STTLargeConfig

    print("=" * 60)
    print("  SONATA STT — MODEL TESTS")
    print("=" * 60)

    # Pass 1: CTC (base encoder)
    ctc_cfg = STTConfig()
    ctc = SonataCTC(ctc_cfg)
    ctc_params = sum(p.numel() for p in ctc.parameters())
    trainable = sum(p.numel() for p in ctc.parameters() if p.requires_grad)
    print(f"\n  Pass 1 — SonataCTC (base, RoPE={ctc_cfg.use_rope}):")
    print(f"    Total: {ctc_params/1e6:.1f}M params")
    print(f"    Trainable: {trainable/1e6:.1f}M")
    print(f"    Vocab: {ctc_cfg.text_vocab_size} (incl blank + <eou>)")

    B = 2
    audio = torch.randn(B, 24000)
    logits = ctc(audio)
    print(f"    Input:  {audio.shape}")
    print(f"    Output: {logits.shape}")

    texts = ctc.recognize(audio)
    print(f"    Decoded: {texts}")

    eou = ctc.eou_prob(logits)
    print(f"    EOU prob shape: {eou.shape}, max={eou.max():.4f}")

    # CTC loss test
    targets = torch.tensor(text_to_ctc_ids("hello world", append_eou=True))
    target_len = torch.tensor([len(targets)])
    input_len = torch.tensor([logits.shape[1]])
    loss = F.ctc_loss(
        logits[:1].transpose(0, 1).log_softmax(2),
        targets.unsqueeze(0), input_len, target_len, blank=0,
    )
    print(f"    CTC loss (random): {loss:.4f}")

    # Pass 1b: CTC (large encoder)
    large_cfg = STTLargeConfig()
    ctc_large = SonataCTC(large_cfg)
    large_params = sum(p.numel() for p in ctc_large.parameters())
    print(f"\n  Pass 1b — SonataCTC (large, 12L d=512):")
    print(f"    Total: {large_params/1e6:.1f}M params")

    # SpecAugment test
    mel = ctc.mel(audio)
    mel_aug = spec_augment(mel, ctc_cfg)
    assert mel_aug.shape == mel.shape, "SpecAugment preserves shape"
    print(f"    SpecAugment: {mel.shape} → {mel_aug.shape} ✓")

    # Speed perturbation test
    audio_slow = speed_perturb(audio, 0.9)
    audio_fast = speed_perturb(audio, 1.1)
    print(f"    Speed perturb: {audio.shape[1]} → slow={audio_slow.shape[1]} fast={audio_fast.shape[1]}")

    # Pass 2: Refiner
    ref_cfg = RefinerConfig()
    refiner = SonataRefiner(ref_cfg)
    ref_params = sum(p.numel() for p in refiner.parameters())
    print(f"\n  Pass 2 — SonataRefiner:")
    print(f"    Total: {ref_params/1e6:.1f}M params")
    print(f"    Encoder: {ref_cfg.enc_n_layers}L × d={ref_cfg.enc_d_model}")
    print(f"    Decoder: {ref_cfg.dec_n_layers}L × d={ref_cfg.dec_d_model}")

    sem_tokens = torch.randint(0, 32768, (B, 50))
    text_in = torch.randint(0, 4096, (B, 20))
    text_tgt = torch.randint(0, 4096, (B, 20))
    logits_r, losses_r = refiner(sem_tokens, text_in, target_text=text_tgt)
    print(f"    Input:  semantic={sem_tokens.shape}, text={text_in.shape}")
    print(f"    Output: {logits_r.shape}")
    print(f"    Loss:   {losses_r['text']:.4f}")

    gen = refiner.generate(sem_tokens, max_len=10)
    print(f"    Generated: {[len(g) for g in gen]} tokens")

    # Round-trip test (with random weights — just validates API)
    rt = semantic_roundtrip_score("hello world", sem_tokens[0], refiner)
    print(f"\n  Round-trip (random weights):")
    print(f"    Original:      '{rt['original']}'")
    print(f"    Reconstructed: '{rt['reconstructed']}'")
    print(f"    Char match:    {rt['char_match_rate']:.2%}")

    print(f"\n  Combined: {(ctc_params + ref_params)/1e6:.1f}M total params")
    print(f"  Large STT: {large_params/1e6:.1f}M params")
