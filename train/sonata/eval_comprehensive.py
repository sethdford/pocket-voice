#!/usr/bin/env python3
"""Comprehensive Sonata TTS evaluation suite.

Single-file evaluation covering ALL model types and quality metrics.
Works with trained PyTorch models (not Rust inference engines).

Usage:
  python eval_comprehensive.py --mode quality --model-a checkpoints/flow/flow_best.pt
  python eval_comprehensive.py --mode ab --model-a ckpt_a/ --model-b ckpt_b/
  python eval_comprehensive.py --mode all --codec-ckpt ckpt.pt --lm-ckpt ckpt.pt --flow-ckpt ckpt.pt
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F

# Optional deps with graceful fallbacks
try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from scipy.fft import dct as scipy_dct

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ═══════════════════════════════════════════════════════════════════════════════
# Test sentences
# ═══════════════════════════════════════════════════════════════════════════════

TEST_SENTENCES = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "I need to schedule a meeting for three thirty PM tomorrow.",
    "That sounds absolutely wonderful, I'm so excited!",
    "The total comes to forty seven dollars and ninety nine cents.",
    "Can you repeat that please? I didn't quite catch what you said.",
    "We need to discuss the quarterly revenue projections.",
    "Oh no, that's terrible news. I'm so sorry to hear that.",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Levenshtein distance (for WER/CER)
# ═══════════════════════════════════════════════════════════════════════════════

def levenshtein_distance(seq_a: list, seq_b: list) -> int:
    """Compute Levenshtein edit distance between two sequences (words or chars)."""
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[m, n])


def _normalize_for_wer(text: str) -> str:
    """Lowercase, strip punctuation (except apostrophe), collapse whitespace."""
    out = []
    prev_space = True
    for c in text.lower():
        if c in ".,?!;:\"'\"-":
            continue
        if c.isspace():
            if not prev_space:
                out.append(" ")
                prev_space = True
            continue
        out.append(c)
        prev_space = False
    return "".join(out).strip()


def compute_wer(ref: str, hyp: str) -> float:
    """Word Error Rate (0–1)."""
    ref_n = _normalize_for_wer(ref)
    hyp_n = _normalize_for_wer(hyp)
    ref_words = ref_n.split() if ref_n else []
    hyp_words = hyp_n.split() if hyp_n else []
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def compute_cer(ref: str, hyp: str) -> float:
    """Character Error Rate (0–1)."""
    ref_n = _normalize_for_wer(ref)
    hyp_n = _normalize_for_wer(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    ref_chars = list(ref_n)
    hyp_chars = list(hyp_n)
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return dist / max(len(ref_chars), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Mel spectrogram (for MCD, spectral quality)
# ═══════════════════════════════════════════════════════════════════════════════

def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    mel_low = 2595 * np.log10(1 + 0 / 700)
    mel_high = 2595 * np.log10(1 + sr / 2 / 700)
    mel_pts = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = np.floor((hz_pts / sr) * n_fft).astype(int).clip(0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        for j in range(lo, min(mid, fb.shape[1])):
            fb[i, j] = (j - lo) / max(1, mid - lo)
        for j in range(mid, min(hi, fb.shape[1])):
            fb[i, j] = (hi - j) / max(1, hi - mid)
    return fb


def audio_to_mel(
    audio: np.ndarray,
    sr: int = 24000,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 80,
) -> np.ndarray:
    """Compute log-mel spectrogram."""
    padded = np.pad(audio, (n_fft // 2, n_fft // 2))
    win = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    spec = np.abs(np.fft.rfft(frames * win))
    mel_fb = _mel_filterbank(n_mels, n_fft, sr)
    mel = np.dot(spec, mel_fb.T)
    return np.log(np.clip(mel, 1e-7, None))


def mel_to_mfcc(mel: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """DCT of log-mel → MFCC. Requires scipy for DCT."""
    if HAS_SCIPY:
        return scipy_dct(mel, type=2, n=n_mfcc, axis=-1)
    # Fallback: DCT-II via FFT (avoids scipy dependency)
    N = mel.shape[-1]
    k = np.arange(n_mfcc)
    n = np.arange(N)
    dct_basis = np.cos(np.pi * k[:, None] * (2 * n[None, :] + 1) / (2 * N))
    return mel @ dct_basis.T


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: MCD (Mel Cepstral Distortion)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mcd(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 24000, n_mfcc: int = 13) -> float:
    """Mel Cepstral Distortion between reference and generated audio (lower=better)."""
    min_len = min(len(ref_audio), len(gen_audio))
    if min_len < 512:
        return 99.0
    ref_audio, gen_audio = ref_audio[:min_len], gen_audio[:min_len]
    hop = 256
    ref_mel = audio_to_mel(ref_audio, sr=sr, hop=hop, n_mels=80)
    gen_mel = audio_to_mel(gen_audio, sr=sr, hop=hop, n_mels=80)
    T = min(ref_mel.shape[0], gen_mel.shape[0])
    if T < 1:
        return 99.0
    ref_mfcc = mel_to_mfcc(ref_mel[:T], n_mfcc)
    gen_mfcc = mel_to_mfcc(gen_mel[:T], n_mfcc)
    diff = ref_mfcc[:, 1:] - gen_mfcc[:, 1:]
    mcd = np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=-1)))
    return float((10 / np.log(10)) * mcd)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: STOI (simplified Short-Time Objective Intelligibility)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stoi(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 24000) -> float:
    """Simplified STOI via frame-wise correlation (higher=better, 0–1)."""
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    min_len = min(len(ref_audio), len(gen_audio))
    ref_audio, gen_audio = ref_audio[:min_len], gen_audio[:min_len]
    n_frames = (min_len - frame_len) // hop
    if n_frames < 1:
        return 0.0
    correlations = []
    for i in range(n_frames):
        s = i * hop
        r, e = ref_audio[s : s + frame_len], gen_audio[s : s + frame_len]
        std_r, std_e = np.std(r) + 1e-10, np.std(e) + 1e-10
        c = np.mean((r - np.mean(r)) * (e - np.mean(e))) / (std_r * std_e)
        correlations.append(max(0.0, min(1.0, c)))
    return float(np.mean(correlations))


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Segmental SNR
# ═══════════════════════════════════════════════════════════════════════════════

def compute_snr(ref_audio: np.ndarray, gen_audio: np.ndarray, frame_len: int = 512) -> float:
    """Segmental SNR in dB (higher=better). ref=signal, gen=estimate."""
    min_len = min(len(ref_audio), len(gen_audio))
    ref_audio = ref_audio[:min_len] - np.mean(ref_audio[:min_len])
    gen_audio = gen_audio[:min_len] - np.mean(gen_audio[:min_len])
    n_frames = min_len // frame_len
    if n_frames < 1:
        return 0.0
    snrs = []
    for i in range(n_frames):
        s = i * frame_len
        r, e = ref_audio[s : s + frame_len], gen_audio[s : s + frame_len]
        sig_power = np.mean(r ** 2) + 1e-10
        noise_power = np.mean((r - e) ** 2) + 1e-10
        snrs.append(10 * np.log10(sig_power / noise_power))
    return float(np.mean(np.clip(snrs, -10, 60)))


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Spectral quality heuristic (artifacts, clipping, silence)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_spectral_quality(audio: np.ndarray, sr: int = 24000) -> Dict[str, float]:
    """Mel-based quality heuristic: NaN, clipping, silence ratio, spectral flatness."""
    out = {
        "has_nan": 1.0 if np.any(np.isnan(audio)) else 0.0,
        "has_inf": 1.0 if np.any(np.isinf(audio)) else 0.0,
        "clip_ratio": float(np.mean(np.abs(audio) >= 0.99)),
        "silence_ratio": 0.0,
        "spectral_flatness": 0.0,
        "rms": 0.0,
        "quality_score": 1.0,
    }
    if len(audio) < 256:
        return out
    rms = np.sqrt(np.mean(audio ** 2))
    out["rms"] = float(rms)
    if rms < 1e-8:
        out["silence_ratio"] = 1.0
        out["quality_score"] = 0.0
        return out
    frame_len = 512
    hop = 256
    n_frames = (len(audio) - frame_len) // hop + 1
    if n_frames < 1:
        return out
    energies = []
    flatnesses = []
    for i in range(n_frames):
        s = i * hop
        frame = audio[s : s + frame_len] * np.hanning(frame_len)
        spec = np.abs(np.fft.rfft(frame)) ** 2
        spec = np.clip(spec, 1e-12, None)
        energies.append(np.mean(spec))
        geo = np.exp(np.mean(np.log(spec)))
        arith = np.mean(spec)
        flatnesses.append(geo / (arith + 1e-12))
    out["silence_ratio"] = float(np.mean(np.array(energies) < rms * 1e-4))
    out["spectral_flatness"] = float(np.mean(flatnesses))
    out["quality_score"] = float(
        1.0
        - 0.3 * out["has_nan"]
        - 0.3 * out["has_inf"]
        - 0.2 * min(1.0, out["clip_ratio"] * 10)
        - 0.2 * out["silence_ratio"]
        + 0.1 * min(1.0, out["spectral_flatness"])
    )
    out["quality_score"] = max(0.0, min(1.0, out["quality_score"]))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Speaker similarity (MFCC cosine)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_speaker_similarity(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 24000) -> float:
    """MFCC-based cosine similarity (higher=better, 0–1)."""
    min_len = min(len(ref_audio), len(gen_audio))
    if min_len < 512:
        return 0.0
    hop = 256
    ref_mel = audio_to_mel(ref_audio[:min_len], sr=sr, hop=hop)
    gen_mel = audio_to_mel(gen_audio[:min_len], sr=sr, hop=hop)
    ref_mfcc = mel_to_mfcc(ref_mel, n_mfcc=13)
    gen_mfcc = mel_to_mfcc(gen_mel, n_mfcc=13)
    ref_mean = np.mean(ref_mfcc, axis=0)
    gen_mean = np.mean(gen_mfcc, axis=0)
    norm_ref = np.linalg.norm(ref_mean) + 1e-8
    norm_gen = np.linalg.norm(gen_mean) + 1e-8
    cos_sim = np.dot(ref_mean, gen_mean) / (norm_ref * norm_gen)
    return float(np.clip(cos_sim, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading and TTS generation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TTSBundle:
    """Bundle of codec, LM, flow for TTS generation."""
    codec: Any = None
    codec_cfg: Any = None
    lm: Any = None
    lm_cfg: Any = None
    flow: Any = None
    flow_cfg: Any = None
    soundstorm: Any = None
    soundstorm_cfg: Any = None
    device: Any = None


def _text_to_ids(text: str, device: torch.device) -> torch.Tensor:
    """Simple char-level encoding for LM (ord % 32000)."""
    ids = [min(ord(c) % 32000, 31999) for c in text]
    return torch.tensor([ids], dtype=torch.long, device=device)


def _load_ckpt(path: str, key: str = "model") -> Tuple[dict, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("ema", ckpt.get(key, ckpt))
    cfg = ckpt.get("config", {})
    return state, cfg


def load_tts_bundle(
    codec_ckpt: Optional[str] = None,
    lm_ckpt: Optional[str] = None,
    flow_ckpt: Optional[str] = None,
    soundstorm_ckpt: Optional[str] = None,
    device: str = "cpu",
) -> Optional["TTSBundle"]:
    """Load Codec, LM, Flow (and optionally SoundStorm) from checkpoints."""
    dev = torch.device(device)
    bundle = TTSBundle(device=dev)

    if codec_ckpt and Path(codec_ckpt).exists():
        from config import CodecConfig
        from codec import SonataCodec

        state, cfg = _load_ckpt(codec_ckpt)
        codec_cfg = CodecConfig(**{k: v for k, v in cfg.items() if k in CodecConfig.__dataclass_fields__})
        codec = SonataCodec(codec_cfg).to(dev)
        codec.load_state_dict(state, strict=False)
        codec.eval()
        bundle.codec = codec
        bundle.codec_cfg = codec_cfg

    if lm_ckpt and Path(lm_ckpt).exists():
        from config import SemanticLMConfig
        from semantic_lm import SonataSemanticLM

        state, cfg = _load_ckpt(lm_ckpt)
        lm_cfg = SemanticLMConfig(**{k: v for k, v in cfg.items() if k in SemanticLMConfig.__dataclass_fields__})
        lm = SonataSemanticLM(lm_cfg).to(dev)
        lm.load_state_dict(state, strict=False)
        lm.eval()
        bundle.lm = lm
        bundle.lm_cfg = lm_cfg

    if soundstorm_ckpt and Path(soundstorm_ckpt).exists():
        from config import SemanticLMConfig
        from soundstorm import SonataStorm

        state, cfg = _load_ckpt(soundstorm_ckpt)
        cfg_cls = SemanticLMConfig
        storm_cfg = cfg_cls(**{k: v for k, v in cfg.items() if k in cfg_cls.__dataclass_fields__})
        storm = SonataStorm(storm_cfg).to(dev)
        storm.load_state_dict(state, strict=False)
        storm.eval()
        bundle.soundstorm = storm
        bundle.soundstorm_cfg = storm_cfg

    if flow_ckpt and Path(flow_ckpt).exists():
        from config import FlowConfig
        from flow import SonataFlow

        state, cfg = _load_ckpt(flow_ckpt)
        flow_cfg = FlowConfig(**FlowConfig._normalize_loaded_dict(cfg))
        flow = SonataFlow(flow_cfg).to(dev)
        flow.load_state_dict(state, strict=False)
        flow.eval()
        bundle.flow = flow
        bundle.flow_cfg = flow_cfg

    if not bundle.codec or not bundle.flow:
        return None
    if not bundle.lm and not bundle.soundstorm:
        return None
    return bundle


@torch.no_grad()
def generate_semantic_tokens_ar(lm, text_ids: torch.Tensor, n_frames: int, temperature: float = 0.8, top_k: int = 50):
    """AR LM: autoregressive semantic token generation."""
    B = text_ids.shape[0]
    device = text_ids.device
    if text_ids.shape[1] < n_frames:
        repeats = (n_frames // text_ids.shape[1]) + 1
        text_ids = text_ids.repeat(1, repeats)[:, :n_frames]
    else:
        text_ids = text_ids[:, :n_frames]
    tokens = torch.ones(B, 1, dtype=torch.long, device=device)
    for i in range(n_frames):
        text_in = text_ids[:, : min(text_ids.shape[1], tokens.shape[1])]
        sem_in = tokens
        min_t = min(text_in.shape[1], sem_in.shape[1])
        text_in, sem_in = text_in[:, :min_t], sem_in[:, :min_t]
        logits, _ = lm(text_in, sem_in)
        next_logits = logits[:, -1, :] / temperature
        if top_k > 0:
            topk_vals, _ = next_logits.topk(min(top_k, next_logits.shape[-1]))
            next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens[:, 1:]


@torch.no_grad()
def generate_audio(bundle: "TTSBundle", text: str, sr: int = 24000) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Generate audio from text. Returns (audio_array, latency_info)."""
    if not bundle.codec or not bundle.flow:
        return None, {}
    device = bundle.device
    text_ids = _text_to_ids(text, device)
    n_frames = max(25, len(text) * 3)

    t0 = time.perf_counter()
    if bundle.soundstorm:
        text_ext = text_ids.repeat(1, (n_frames // text_ids.shape[1]) + 1)[:, :n_frames]
        semantic_tokens = bundle.soundstorm.generate(text_ext, n_frames, n_steps=12, temperature=1.0)
    elif bundle.lm:
        semantic_tokens = generate_semantic_tokens_ar(bundle.lm, text_ids, n_frames)
    else:
        return None, {}
    t_lm = time.perf_counter() - t0

    t1 = time.perf_counter()
    acoustic_latents = bundle.flow.sample(semantic_tokens, n_steps=8)
    t_flow = time.perf_counter() - t1

    semantic_codes = bundle.codec.fsq.indices_to_codes(semantic_tokens)
    T_min = min(semantic_codes.shape[1], acoustic_latents.shape[1])
    semantic_codes = semantic_codes[:, :T_min]
    acoustic_latents = acoustic_latents[:, :T_min]

    t2 = time.perf_counter()
    audio = bundle.codec.decoder(semantic_codes, acoustic_latents)
    t_dec = time.perf_counter() - t2

    audio_np = audio.squeeze(0).cpu().numpy()
    duration = len(audio_np) / sr
    total_time = t_lm + t_flow + t_dec

    return audio_np, {
        "ttft_lm_ms": t_lm * 1000,
        "ttfa_flow_ms": (t_lm + t_flow) * 1000,
        "ttfa_total_ms": total_time * 1000,
        "duration_sec": duration,
        "rtf": total_time / max(duration, 0.01),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Quality evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_quality(
    model: "TTSBundle",
    test_texts: List[str],
    reference_wavs: Optional[List[str]] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compute quality metrics for generated audio."""
    metrics = {
        "mcd": [],
        "stoi": [],
        "snr": [],
        "spectral_quality": [],
        "speaker_similarity": [],
        "quality_score": [],
    }
    ref_audios = []
    if reference_wavs and HAS_SOUNDFILE:
        for p in reference_wavs:
            if Path(p).exists():
                a, sr = sf.read(p, dtype="float32")
                if a.ndim > 1:
                    a = a.mean(axis=-1)
                ref_audios.append((a, sr))
    while len(ref_audios) < len(test_texts):
        ref_audios.append((None, 24000))

    for i, text in enumerate(test_texts):
        audio, _ = generate_audio(model, text)
        if audio is None:
            continue
        sr = 24000
        sq = compute_spectral_quality(audio, sr)
        metrics["spectral_quality"].append(sq)
        metrics["quality_score"].append(sq["quality_score"])
        if ref_audios[i][0] is not None:
            ref_a, ref_sr = ref_audios[i]
            if ref_sr != sr:
                ref_a = np.interp(
                    np.linspace(0, len(ref_a), int(len(ref_a) * sr / ref_sr)),
                    np.arange(len(ref_a)),
                    ref_a,
                )
            metrics["mcd"].append(compute_mcd(ref_a, audio, sr))
            metrics["stoi"].append(compute_stoi(ref_a, audio, sr))
            metrics["snr"].append(compute_snr(ref_a, audio))
            metrics["speaker_similarity"].append(compute_speaker_similarity(ref_a, audio, sr))

    def _mean(lst):
        lst = [x for x in lst if not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
        return float(np.mean(lst)) if lst else float("nan")

    return {
        "mcd_mean": _mean(metrics["mcd"]) if metrics["mcd"] else float("nan"),
        "stoi_mean": _mean(metrics["stoi"]) if metrics["stoi"] else float("nan"),
        "snr_mean": _mean(metrics["snr"]) if metrics["snr"] else float("nan"),
        "quality_score_mean": _mean(metrics["quality_score"]),
        "speaker_similarity_mean": _mean(metrics["speaker_similarity"]) if metrics["speaker_similarity"] else float("nan"),
        "n_samples": len(test_texts),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. A/B testing
# ═══════════════════════════════════════════════════════════════════════════════

def ab_test(
    model_a: "TTSBundle",
    model_b: "TTSBundle",
    test_texts: List[str],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compare two models on objective metrics. Returns wins/losses/ties."""
    results_a = {"mcd": [], "stoi": [], "snr": [], "quality_score": [], "rtf": []}
    results_b = {"mcd": [], "stoi": [], "snr": [], "quality_score": [], "rtf": []}

    for text in test_texts:
        aud_a, lat_a = generate_audio(model_a, text)
        aud_b, lat_b = generate_audio(model_b, text)
        if aud_a is None or aud_b is None:
            continue
        sq_a = compute_spectral_quality(aud_a, 24000)
        sq_b = compute_spectral_quality(aud_b, 24000)
        results_a["quality_score"].append(sq_a["quality_score"])
        results_b["quality_score"].append(sq_b["quality_score"])
        results_a["rtf"].append(lat_a.get("rtf", 0))
        results_b["rtf"].append(lat_b.get("rtf", 0))
        results_a["mcd"].append(compute_mcd(aud_a, aud_b, 24000))
        results_b["mcd"].append(compute_mcd(aud_b, aud_a, 24000))
        results_a["stoi"].append(compute_stoi(aud_a, aud_b, 24000))
        results_b["stoi"].append(compute_stoi(aud_b, aud_a, 24000))
        results_a["snr"].append(compute_snr(aud_a, aud_b, 24000))
        results_b["snr"].append(compute_snr(aud_b, aud_a, 24000))

    wins_a = {"mcd": 0, "stoi": 0, "snr": 0, "quality_score": 0, "rtf": 0}
    wins_b = {"mcd": 0, "stoi": 0, "snr": 0, "quality_score": 0, "rtf": 0}
    for m in ["mcd", "stoi", "snr", "quality_score", "rtf"]:
        if m == "mcd" or m == "rtf":
            better = "lower"
        else:
            better = "higher"
        for va, vb in zip(results_a[m], results_b[m]):
            if better == "lower":
                if va < vb:
                    wins_a[m] += 1
                elif vb < va:
                    wins_b[m] += 1
            else:
                if va > vb:
                    wins_a[m] += 1
                elif vb > va:
                    wins_b[m] += 1

    n = len(results_a["quality_score"])
    return {
        "n_pairs": n,
        "model_a": {"wins": wins_a, "mean": {k: float(np.mean(v)) if v else 0 for k, v in results_a.items()}},
        "model_b": {"wins": wins_b, "mean": {k: float(np.mean(v)) if v else 0 for k, v in results_b.items()}},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Latency benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_latency(
    model: "TTSBundle",
    test_texts: List[str],
    device: str = "cpu",
    n_runs: int = 3,
) -> Dict[str, Any]:
    """Measure time-to-first-token and time-to-first-audio. Report P50/P95/P99."""
    ttft_all = []
    ttfa_all = []
    rtf_all = []
    for _ in range(n_runs):
        for text in test_texts:
            _, lat = generate_audio(model, text)
            if lat:
                ttft_all.append(lat.get("ttft_lm_ms", 0))
                ttfa_all.append(lat.get("ttfa_total_ms", 0))
                rtf_all.append(lat.get("rtf", 0))
    ttft_all = np.array(ttft_all) if ttft_all else np.array([0])
    ttfa_all = np.array(ttfa_all) if ttfa_all else np.array([0])
    rtf_all = np.array(rtf_all) if rtf_all else np.array([0])
    return {
        "ttft_ms": {
            "p50": float(np.percentile(ttft_all, 50)),
            "p95": float(np.percentile(ttft_all, 95)),
            "p99": float(np.percentile(ttft_all, 99)),
            "mean": float(np.mean(ttft_all)),
        },
        "ttfa_ms": {
            "p50": float(np.percentile(ttfa_all, 50)),
            "p95": float(np.percentile(ttfa_all, 95)),
            "p99": float(np.percentile(ttfa_all, 99)),
            "mean": float(np.mean(ttfa_all)),
        },
        "rtf": {
            "p50": float(np.percentile(rtf_all, 50)),
            "p95": float(np.percentile(rtf_all, 95)),
            "mean": float(np.mean(rtf_all)),
        },
        "n_samples": len(ttft_all),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Round-trip intelligibility
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_stt(audio: np.ndarray, sr: int, original_text: str) -> str:
    """Mock STT: return original (for when no STT available)."""
    return original_text


def _load_stt(stt_ckpt: str, device: torch.device):
    """Load Sonata CTC STT if available."""
    try:
        from config import STTConfig
        from stt import SonataCTC

        ckpt = torch.load(stt_ckpt, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        stt_cfg = STTConfig(**{k: v for k, v in cfg.items() if k in STTConfig.__dataclass_fields__})
        model = SonataCTC(stt_cfg).to(device)
        model.load_state_dict(ckpt.get("ema", ckpt.get("model", {})), strict=False)
        model.eval()

        def transcribe(audio: np.ndarray, sr: int) -> str:
            t = torch.from_numpy(audio).float().unsqueeze(0).to(device)
            if sr != 24000:
                new_len = int(len(audio) * 24000 / sr)
                t = F.interpolate(
                    t.unsqueeze(0).unsqueeze(0),
                    size=new_len,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)
            with torch.no_grad():
                results = model.recognize(t)
            return results[0] if results else ""

        return transcribe
    except Exception as e:
        print(f"  [STT] Could not load: {e}")
        return None


def round_trip_test(
    tts_model: "TTSBundle",
    stt_model: Optional[Any],
    test_texts: List[str],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Text → TTS → audio → STT → text. Compute WER/CER."""
    wers = []
    cers = []
    for text in test_texts:
        audio, _ = generate_audio(tts_model, text)
        if audio is None:
            continue
        if stt_model:
            hyp = stt_model(audio, 24000)
        else:
            hyp = _mock_stt(audio, 24000, text)
        wers.append(compute_wer(text, hyp))
        cers.append(compute_cer(text, hyp))
    return {
        "wer_mean": float(np.mean(wers)) if wers else float("nan"),
        "cer_mean": float(np.mean(cers)) if cers else float("nan"),
        "n_samples": len(wers),
        "stt_used": stt_model is not None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Model architecture comparison
# ═══════════════════════════════════════════════════════════════════════════════

def model_architecture_comparison(
    model_a: "TTSBundle",
    model_b: "TTSBundle",
    test_texts: List[str],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compare two checkpoints: speed, quality, memory."""
    qual_a = evaluate_quality(model_a, test_texts, device=device)
    qual_b = evaluate_quality(model_b, test_texts, device=device)
    lat_a = benchmark_latency(model_a, test_texts, device=device, n_runs=2)
    lat_b = benchmark_latency(model_b, test_texts, device=device, n_runs=2)

    def _params(bundle):
        n = 0
        for m in [bundle.codec, bundle.lm, bundle.flow, bundle.soundstorm]:
            if m:
                n += sum(p.numel() for p in m.parameters())
        return n

    return {
        "model_a": {"quality": qual_a, "latency": lat_a, "params": _params(model_a)},
        "model_b": {"quality": qual_b, "latency": lat_b, "params": _params(model_b)},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Formatting and CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _print_table(title: str, rows: List[Tuple[str, str]], width: int = 60):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")
    for left, right in rows:
        print(f"  {left:<35} {right:>18}")
    print(f"{'='*width}")


def _resolve_model_path(base: str) -> Dict[str, Optional[str]]:
    """Resolve codec/lm/flow paths from directory or single file."""
    p = Path(base)
    if not p.exists():
        return {"codec": None, "lm": None, "flow": None}
    if p.is_file():
        name = p.name.lower()
        if "codec" in name:
            return {"codec": str(p), "lm": None, "flow": None}
        if "lm" in name or "semantic" in name:
            return {"codec": None, "lm": str(p), "flow": None}
        if "flow" in name or "storm" in name:
            return {"codec": None, "lm": None, "flow": str(p)}
        return {"codec": None, "lm": None, "flow": str(p)}
    files = list(p.glob("*.pt"))
    out = {"codec": None, "lm": None, "flow": None}
    for f in files:
        n = f.name.lower()
        if "codec" in n:
            out["codec"] = str(f)
        elif "lm" in n or "semantic" in n:
            out["lm"] = str(f)
        elif "storm" in n or "soundstorm" in n:
            out["lm"] = str(f)
        elif "flow" in n:
            out["flow"] = str(f)
    return out


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Sonata TTS evaluation")
    parser.add_argument("--mode", default="all", choices=["quality", "ab", "latency", "roundtrip", "all"])
    parser.add_argument("--model-a", default="", help="Path to model (dir or .pt)")
    parser.add_argument("--model-b", default="", help="Path to second model (for A/B)")
    parser.add_argument("--codec-ckpt", default="")
    parser.add_argument("--lm-ckpt", default="")
    parser.add_argument("--flow-ckpt", default="")
    parser.add_argument("--soundstorm-ckpt", default="")
    parser.add_argument("--stt-ckpt", default="", help="Optional STT for round-trip")
    parser.add_argument("--reference-wavs", default="", help="Comma-separated WAV paths for quality")
    parser.add_argument("--sentences", default="", help="Text file; one sentence per line")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-runs", type=int, default=3, help="Runs for latency benchmark")
    parser.add_argument("--output", default="", help="JSON output path")
    args = parser.parse_args()

    # Resolve model paths
    paths_a = _resolve_model_path(args.model_a) if args.model_a else {}
    paths_b = _resolve_model_path(args.model_b) if args.model_b else {}
    codec = args.codec_ckpt or paths_a.get("codec") or paths_b.get("codec")
    lm = args.lm_ckpt or paths_a.get("lm") or paths_b.get("lm")
    flow = args.flow_ckpt or paths_a.get("flow") or paths_b.get("flow")
    if not flow and args.model_a:
        flow = paths_a.get("flow", args.model_a)
    if not lm and args.model_a and not codec:
        lm = paths_a.get("lm")

    test_texts = TEST_SENTENCES
    if args.sentences and Path(args.sentences).exists():
        test_texts = [l.strip() for l in Path(args.sentences).read_text().splitlines() if l.strip()]
    ref_wavs = [p.strip() for p in args.reference_wavs.split(",") if p.strip()] if args.reference_wavs else None

    bundle_a = load_tts_bundle(codec, lm, flow, args.soundstorm_ckpt or paths_a.get("lm"), args.device)
    if not bundle_a and (args.mode in ["quality", "latency", "roundtrip", "all"]):
        print("ERROR: Could not load TTS model. Need --codec-ckpt, --lm-ckpt (or --soundstorm-ckpt), --flow-ckpt")
        sys.exit(1)

    bundle_b = None
    if args.model_b:
        pb = _resolve_model_path(args.model_b)
        bundle_b = load_tts_bundle(
            pb.get("codec") or codec,
            pb.get("lm") or lm,
            pb.get("flow") or flow,
            pb.get("lm"),
            args.device,
        )

    stt_fn = None
    if args.stt_ckpt and Path(args.stt_ckpt).exists():
        stt_fn = _load_stt(args.stt_ckpt, torch.device(args.device))

    output_data = {}

    # Quality
    if args.mode in ["quality", "all"] and bundle_a:
        print("\n  --- QUALITY EVALUATION ---")
        qual = evaluate_quality(bundle_a, test_texts, ref_wavs, args.device)
        output_data["quality"] = qual
        _print_table(
            "QUALITY METRICS",
            [
                ("MCD (lower=better)", f"{qual['mcd_mean']:.3f}" if not math.isnan(qual['mcd_mean']) else "N/A"),
                ("STOI (higher=better)", f"{qual['stoi_mean']:.3f}" if not math.isnan(qual['stoi_mean']) else "N/A"),
                ("SNR dB (higher=better)", f"{qual['snr_mean']:.3f}" if not math.isnan(qual['snr_mean']) else "N/A"),
                ("Quality score", f"{qual['quality_score_mean']:.3f}"),
                ("Speaker similarity", f"{qual['speaker_similarity_mean']:.3f}" if not math.isnan(qual['speaker_similarity_mean']) else "N/A"),
                ("Samples", str(qual["n_samples"])),
            ],
        )

    # A/B
    if args.mode in ["ab", "all"] and bundle_a and bundle_b:
        print("\n  --- A/B TEST ---")
        ab = ab_test(bundle_a, bundle_b, test_texts, args.device)
        output_data["ab_test"] = ab
        rows = [
            ("Metric", "Model A wins / Model B wins"),
            ("MCD", f"{ab['model_a']['wins']['mcd']} / {ab['model_b']['wins']['mcd']}"),
            ("STOI", f"{ab['model_a']['wins']['stoi']} / {ab['model_b']['wins']['stoi']}"),
            ("SNR", f"{ab['model_a']['wins']['snr']} / {ab['model_b']['wins']['snr']}"),
            ("Quality score", f"{ab['model_a']['wins']['quality_score']} / {ab['model_b']['wins']['quality_score']}"),
            ("RTF (lower wins)", f"{ab['model_a']['wins']['rtf']} / {ab['model_b']['wins']['rtf']}"),
        ]
        _print_table("A/B COMPARISON", rows)

    # Latency
    if args.mode in ["latency", "all"] and bundle_a:
        print("\n  --- LATENCY BENCHMARK ---")
        lat = benchmark_latency(bundle_a, test_texts, args.device, args.n_runs)
        output_data["latency"] = lat
        _print_table(
            "LATENCY (ms)",
            [
                ("TTFT P50", f"{lat['ttft_ms']['p50']:.1f}"),
                ("TTFT P95", f"{lat['ttft_ms']['p95']:.1f}"),
                ("TTFA P50", f"{lat['ttfa_ms']['p50']:.1f}"),
                ("TTFA P95", f"{lat['ttfa_ms']['p95']:.1f}"),
                ("RTF mean", f"{lat['rtf']['mean']:.3f}x"),
                ("Samples", str(lat["n_samples"])),
            ],
        )

    # Round-trip
    if args.mode in ["roundtrip", "all"] and bundle_a:
        print("\n  --- ROUND-TRIP INTELLIGIBILITY ---")
        rt = round_trip_test(bundle_a, stt_fn, test_texts, args.device)
        output_data["round_trip"] = rt
        _print_table(
            "ROUND-TRIP WER/CER",
            [
                ("WER", f"{rt['wer_mean']*100:.2f}%"),
                ("CER", f"{rt['cer_mean']*100:.2f}%"),
                ("STT", "real" if rt["stt_used"] else "mock"),
                ("Samples", str(rt["n_samples"])),
            ],
        )

    # Model comparison
    if args.mode == "all" and bundle_a and bundle_b:
        print("\n  --- MODEL ARCHITECTURE COMPARISON ---")
        comp = model_architecture_comparison(bundle_a, bundle_b, test_texts, args.device)
        output_data["comparison"] = comp
        _print_table(
            "MODEL COMPARISON",
            [
                ("Model A params", f"{comp['model_a']['params']/1e6:.1f}M"),
                ("Model B params", f"{comp['model_b']['params']/1e6:.1f}M"),
                ("A quality score", f"{comp['model_a']['quality']['quality_score_mean']:.3f}"),
                ("B quality score", f"{comp['model_b']['quality']['quality_score_mean']:.3f}"),
                ("A RTF", f"{comp['model_a']['latency']['rtf']['mean']:.3f}x"),
                ("B RTF", f"{comp['model_b']['latency']['rtf']['mean']:.3f}x"),
            ],
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
