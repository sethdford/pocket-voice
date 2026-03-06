#!/usr/bin/env python3
"""Comprehensive TTS evaluation against human-level benchmarks.

Measures audio quality via objective metrics (all in Python, pip-installable):
  - MOS proxy: Weighted combination of PESQ, STOI, MCD, F0 correlation (1–5 scale)
  - PESQ: Perceptual Evaluation of Speech Quality (pesq)
  - STOI: Short-Time Objective Intelligibility (pystoi)
  - WER: Word Error Rate via Whisper transcription (openai-whisper)
  - Speaker Similarity: ECAPA-TDNN cosine similarity (speechbrain), fallback MFCC
  - F0 RMSE + correlation: Fundamental frequency accuracy (autocorrelation)
  - RTF: Real-Time Factor (generation time / audio duration)
  - MCD: Mel Cepstral Distortion (librosa)
  - Spectral convergence: ‖S_ref - S_gen‖_F / ‖S_ref‖_F

Evaluation modes:
  --mode single:   Evaluate one WAV against reference
  --mode batch:   Evaluate directory of generated WAVs vs references
  --mode synthesize: Run full pipeline (text → Flow v3 → vocoder → WAV → metrics)

Usage:
  # Single file
  python eval_tts.py --mode single --generated out.wav --reference ref.wav --text "Hello world"

  # Batch directory
  python eval_tts.py --mode batch --generated-dir gen/ --ref-dir ref/ --ref-texts ref.txt

  # Synthesize mode (Flow v3 + vocoder)
  python eval_tts.py --mode synthesize \\
    --flow-checkpoint checkpoints/flow_v3_libritts/flow_v3_best.pt \\
    --vocoder-checkpoint checkpoints/vocoder_libritts/vocoder_best.pt \\
    --output-dir eval_output/ --device mps
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import resampy
    HAS_RESAMPY = True
except ImportError:
    HAS_RESAMPY = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

try:
    from pystoi import stoi
    HAS_PYSTOI = True
except ImportError:
    HAS_PYSTOI = False

try:
    from jiwer import cer, wer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

# UTMOS (optional, --utmos flag)
HAS_UTMOS = False
_utmos_model = None


def _init_utmos():
    """Lazy-init UTMOS. Try speechmos first, then transformers."""
    global HAS_UTMOS, _utmos_model
    if _utmos_model is not None:
        return True
    try:
        from speechmos import UTMOS
        _utmos_model = UTMOS()
        HAS_UTMOS = True
        return True
    except ImportError:
        pass
    try:
        from transformers import pipeline
        _utmos_model = pipeline("audio-classification", model="utmos/utmos22_strong", device=-1)
        HAS_UTMOS = True
        return True
    except Exception:
        pass
    return False


def compute_utmos_single(audio: np.ndarray, sr: int) -> float:
    """Compute UTMOS score for one audio (1-5 scale). Returns -1 if unavailable."""
    if not _init_utmos():
        return -1.0
    if sr != 16000:
        if HAS_RESAMPY:
            audio = resample_audio(audio, sr, 16000)
            sr = 16000
        else:
            return -1.0
    min_len = 16000  # 1 sec minimum
    if len(audio) < min_len:
        return -1.0
    try:
        if hasattr(_utmos_model, "predict"):
            return float(_utmos_model.predict(audio, sr))
        result = _utmos_model({"raw": audio.astype(np.float32), "sampling_rate": sr})
        return float(sum(r["score"] * float(r["label"]) for r in result))
    except Exception:
        return -1.0


# Human-level targets
TARGETS = {
    "mos_proxy": 4.0,
    "pesq": 3.5,
    "stoi": 0.90,
    "wer_pct": 5.0,
    "mcd_db": 4.0,
    "f0_corr": 0.85,
    "rtf": 0.2,
    "speaker_sim": 0.85,
    "utmos": 4.0,
}

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells sea shells by the sea shore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "The rain in Spain falls mainly on the plain.",
    "Peter Piper picked a peck of pickled peppers.",
    "To be or not to be, that is the question.",
    "I think therefore I am, but sometimes I wonder.",
    "The weather today is absolutely beautiful, isn't it?",
    "Could you please pass me the salt and pepper?",
    "We're running late for the meeting, let's hurry up!",
]


@dataclass
class EvalMetrics:
    """Per-file evaluation metrics."""

    file: str = ""
    text: str = ""
    duration_sec: float = 0.0

    # Objective metrics
    mos_proxy: float = -1.0
    pesq: float = -1.0
    stoi: float = -1.0
    wer_pct: float = -1.0
    mcd_db: float = -1.0
    f0_rmse: float = -1.0
    f0_corr: float = -1.0
    rtf: float = -1.0
    speaker_sim: float = -1.0
    spectral_convergence: float = -1.0
    utmos: float = -1.0

    # Generation (synthesize mode)
    gen_time_sec: float = -1.0


@dataclass
class EvalReport:
    """Aggregate report with scorecard."""

    mode: str = ""
    n_samples: int = 0
    mean_mos_proxy: float = 0.0
    mean_pesq: float = 0.0
    mean_stoi: float = 0.0
    mean_wer_pct: float = 0.0
    mean_mcd_db: float = 0.0
    mean_f0_rmse: float = 0.0
    mean_f0_corr: float = 0.0
    mean_rtf: float = 0.0
    mean_speaker_sim: float = 0.0
    mean_spectral_convergence: float = 0.0
    mean_utmos: float = 0.0
    grade: str = ""
    grade_score: float = 0.0
    results: List[dict] = field(default_factory=list)
    targets: dict = field(default_factory=lambda: dict(TARGETS))


# ─── Audio I/O ─────────────────────────────────────────────────────────────


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load WAV. Returns (float32 mono, sample_rate)."""
    if not HAS_SOUNDFILE:
        raise ImportError("soundfile required: pip install soundfile")
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr)


def save_audio(path: str, audio: np.ndarray, sr: int) -> None:
    """Save float32 audio to WAV."""
    if HAS_SOUNDFILE:
        sf.write(path, audio, sr)
        return
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
        w.writeframes(int16.tobytes())


def resample_audio(audio: np.ndarray, sr_orig: int, sr_target: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if sr_orig == sr_target:
        return audio
    if not HAS_RESAMPY:
        # Simple linear interpolation fallback
        n_orig = len(audio)
        n_target = int(n_orig * sr_target / sr_orig)
        x_orig = np.linspace(0, 1, n_orig)
        x_new = np.linspace(0, 1, n_target, endpoint=False)
        return np.interp(x_new, x_orig, audio).astype(np.float32)
    return resampy.resample(audio, sr_orig, sr_target).astype(np.float32)


# ─── Metrics ─────────────────────────────────────────────────────────────


def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """PESQ (requires 8k or 16k). Returns -1 if unavailable."""
    if not HAS_PESQ:
        return -1.0
    if sr not in (8000, 16000):
        ref_16k = resample_audio(ref, sr, 16000)
        deg_16k = resample_audio(deg, sr, 16000)
        ref, deg, sr = ref_16k, deg_16k, 16000
    min_len = min(len(ref), len(deg))
    if min_len < 64:
        return -1.0
    try:
        score = pesq(sr, ref[:min_len], deg[:min_len], "wb")
        return float(score)
    except Exception:
        return -1.0


def compute_stoi_score(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """STOI. pystoi resamples internally."""
    if not HAS_PYSTOI:
        return -1.0
    min_len = min(len(ref), len(deg))
    if min_len < 256:
        return -1.0
    try:
        score = stoi(ref[:min_len], deg[:min_len], sr, extended=False)
        return float(score)
    except Exception:
        return -1.0


def compute_mcd(ref: np.ndarray, deg: np.ndarray, sr: int, n_mfcc: int = 13) -> float:
    """Mel Cepstral Distortion in dB. Lower is better."""
    if not HAS_LIBROSA:
        return -1.0
    try:
        ref_mfcc = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=256)
        deg_mfcc = librosa.feature.mfcc(y=deg, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=256)
        T = min(ref_mfcc.shape[1], deg_mfcc.shape[1])
        if T == 0:
            return -1.0
        diff = ref_mfcc[:, :T] - deg_mfcc[:, :T]
        # MCD = (10/ln10) * sqrt(2 * sum_{k=1}^{K} (c_ref - c_gen)^2)
        frame_mcd = (10.0 / math.log(10.0)) * np.sqrt(2 * np.sum(diff[1:, :] ** 2, axis=0))
        return float(np.mean(frame_mcd))
    except Exception:
        return -1.0


def compute_spectral_convergence(ref: np.ndarray, deg: np.ndarray, n_fft: int = 1024) -> float:
    """‖S_ref - S_gen‖_F / ‖S_ref‖_F. Lower is better."""
    if not HAS_LIBROSA:
        return -1.0
    try:
        S_ref = np.abs(librosa.stft(ref, n_fft=n_fft))
        S_deg = np.abs(librosa.stft(deg[:len(ref)], n_fft=n_fft))
        T = min(S_ref.shape[1], S_deg.shape[1])
        S_ref_t = S_ref[:, :T]
        S_deg_t = S_deg[:, :T]
        num = np.linalg.norm(S_ref_t - S_deg_t, "fro")
        den = np.linalg.norm(S_ref_t, "fro")
        return float(num / (den + 1e-10))
    except Exception:
        return -1.0


def estimate_f0_autocorr(audio: np.ndarray, sr: int, hop_ms: int = 10) -> np.ndarray:
    """F0 via autocorrelation. Returns per-frame F0 (0 = unvoiced)."""
    hop = sr * hop_ms // 1000
    frame_len = hop * 2
    n_frames = max(1, len(audio) // hop - 1)
    f0 = np.zeros(n_frames)
    min_lag = sr // 500
    max_lag = sr // 50

    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(audio))
        frame = audio[start:end]
        if len(frame) < min_lag * 2:
            continue
        energy = np.mean(frame ** 2)
        if energy < 1e-8:
            continue
        corr = np.correlate(frame, frame, mode="full")
        mid = len(corr) // 2
        corr = corr[mid:]
        search_end = min(max_lag, len(corr))
        if search_end <= min_lag:
            continue
        search = corr[min_lag:search_end]
        if len(search) == 0:
            continue
        peak = int(np.argmax(search)) + min_lag
        if peak > 0 and corr[peak] > energy * 0.3:
            f0[i] = sr / peak
    return f0


def compute_f0_metrics(ref: np.ndarray, deg: np.ndarray, sr: int) -> Tuple[float, float]:
    """(F0 RMSE, F0 Pearson correlation). Aligns by minimum length."""
    f0_ref = estimate_f0_autocorr(ref, sr)
    f0_deg = estimate_f0_autocorr(deg, sr)
    n = min(len(f0_ref), len(f0_deg))
    if n < 10:
        return -1.0, -1.0
    f0_ref = f0_ref[:n]
    f0_deg = f0_deg[:n]
    voiced = (f0_ref > 0) & (f0_deg > 0)
    if voiced.sum() < 5:
        return -1.0, -1.0
    vr = f0_ref[voiced]
    vd = f0_deg[voiced]
    rmse = float(np.sqrt(np.mean((vr - vd) ** 2)))
    corr = float(np.corrcoef(vr, vd)[0, 1]) if np.std(vr) > 0 and np.std(vd) > 0 else 0.0
    return rmse, max(0.0, corr)


def transcribe_whisper(audio_path: str, sr: int = 16000) -> Optional[str]:
    """Transcribe via Whisper. Returns text or None."""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, fp16=False)
        return (result.get("text") or "").strip()
    except Exception:
        return None


def compute_wer_pct(ref_text: str, hyp_text: str) -> float:
    """WER as percentage [0, 100]."""
    ref_text = (ref_text or "").strip()
    hyp_text = (hyp_text or "").strip()
    if not ref_text:
        return 0.0 if not hyp_text else 100.0
    if HAS_JIWER:
        return float(wer(ref_text, hyp_text) * 100)
    # Manual WER via Levenshtein on words
    ref_w = ref_text.lower().split()
    hyp_w = hyp_text.lower().split()
    n = len(ref_w)
    if n == 0:
        return 0.0
    d = np.zeros((n + 1, len(hyp_w) + 1))
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(len(hyp_w) + 1)
    for i in range(1, n + 1):
        for j in range(1, len(hyp_w) + 1):
            cost = 0 if ref_w[i - 1] == hyp_w[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[n, len(hyp_w)] / n * 100)


def compute_speaker_similarity(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """ECAPA-TDNN cosine similarity, or MFCC fallback."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        if sr != 16000:
            ref = resample_audio(ref, sr, 16000)
            deg = resample_audio(deg, sr, 16000)
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        deg_trunc = deg[:len(ref)]
        emb_ref = classifier.encode_batch(
            torch.from_numpy(ref).float().unsqueeze(0)
        ).squeeze()
        emb_deg = classifier.encode_batch(
            torch.from_numpy(deg_trunc).float().unsqueeze(0)
        ).squeeze()
        cos_sim = float(
            (emb_ref * emb_deg).sum() / (emb_ref.norm() * emb_deg.norm() + 1e-8)
        )
        return (cos_sim + 1) / 2  # map [-1,1] to [0,1]
    except Exception:
        pass

    # MFCC fallback
    if HAS_LIBROSA:
        try:
            ref_mfcc = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=13).mean(axis=1)
            deg_mfcc = librosa.feature.mfcc(y=deg[:len(ref)], sr=sr, n_mfcc=13).mean(axis=1)
            cos = np.dot(ref_mfcc, deg_mfcc) / (np.linalg.norm(ref_mfcc) * np.linalg.norm(deg_mfcc) + 1e-8)
            return float((cos + 1) / 2)
        except Exception:
            pass
    return -1.0


def compute_mos_proxy(pesq_val: float, stoi_val: float, mcd_val: float, f0_corr_val: float) -> float:
    """MOS proxy (1-5) from sub-metrics. Higher is better."""
    p = 0.0
    w = 0.0
    if pesq_val >= 0:
        p += min(1.0, pesq_val / 4.5) * 0.35
        w += 0.35
    if stoi_val >= 0:
        p += stoi_val * 0.35
        w += 0.35
    if mcd_val >= 0:
        mcd_score = max(0, 1 - mcd_val / 6.0)
        p += mcd_score * 0.15
        w += 0.15
    if f0_corr_val >= 0:
        p += f0_corr_val * 0.15
        w += 0.15
    if w < 1e-6:
        return -1.0
    raw = p / w
    return 1.0 + 4.0 * raw  # map [0,1] to [1,5]


# ─── Grade ───────────────────────────────────────────────────────────────


def compute_grade(grade_score: float) -> str:
    """Grade from MOS/UTMOS blend: A (≥4.0), B (≥3.5), C (≥3.0), D (≥2.5), F (<2.5)."""
    if grade_score >= 4.0:
        return "A"
    if grade_score >= 3.5:
        return "B"
    if grade_score >= 3.0:
        return "C"
    if grade_score >= 2.5:
        return "D"
    return "F"


# ─── Evaluate single pair ───────────────────────────────────────────────


def evaluate_single_pair(
    gen_path: str,
    ref_path: str,
    ref_text: str,
    file_id: str = "",
    use_utmos: bool = False,
) -> EvalMetrics:
    """Compute all metrics for one generated vs reference pair."""
    m = EvalMetrics(file=file_id or Path(gen_path).name, text=ref_text or "")

    gen_audio, sr_gen = load_audio(gen_path)
    ref_audio, sr_ref = load_audio(ref_path)
    m.duration_sec = len(gen_audio) / sr_gen

    sr = max(sr_gen, sr_ref)
    if sr_gen != sr:
        gen_audio = resample_audio(gen_audio, sr_gen, sr)
    if sr_ref != sr:
        ref_audio = resample_audio(ref_audio, sr_ref, sr)
    min_len = min(len(gen_audio), len(ref_audio))
    gen_audio = gen_audio[:min_len]
    ref_audio = ref_audio[:min_len]

    m.pesq = compute_pesq(ref_audio, gen_audio, sr)
    m.stoi = compute_stoi_score(ref_audio, gen_audio, sr)
    m.mcd_db = compute_mcd(ref_audio, gen_audio, sr)
    m.spectral_convergence = compute_spectral_convergence(ref_audio, gen_audio)
    m.f0_rmse, m.f0_corr = compute_f0_metrics(ref_audio, gen_audio, sr)
    m.speaker_sim = compute_speaker_similarity(ref_audio, gen_audio, sr)

    if use_utmos:
        m.utmos = compute_utmos_single(gen_audio, sr)

    m.mos_proxy = compute_mos_proxy(m.pesq, m.stoi, m.mcd_db, m.f0_corr)

    if ref_text:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            save_audio(tmp_path, gen_audio, sr)
            hyp = transcribe_whisper(tmp_path, sr)
            if hyp is not None:
                m.wer_pct = compute_wer_pct(ref_text, hyp)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return m


# ─── Synthesize mode ─────────────────────────────────────────────────────


def load_flow_v3(ckpt_path: str, device: str, model_size: str = "base"):
    """Load Flow v3 from checkpoint."""
    from config import FlowV3Config, FlowV3LargeConfig
    from flow_v3 import SonataFlowV3

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg_cls = FlowV3LargeConfig if model_size == "large" else FlowV3Config
    cfg = cfg_cls(**{k: v for k, v in cfg_dict.items()
                     if k in cfg_cls.__dataclass_fields__})
    model = SonataFlowV3(cfg).to(device)
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def load_vocoder(ckpt_path: str, device: str):
    """Load vocoder from checkpoint."""
    from config import VocoderConfig
    from vocoder import SonataVocoder

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = VocoderConfig(**{k: v for k, v in cfg_dict.items()
                          if k in VocoderConfig.__dataclass_fields__})
    model = SonataVocoder(cfg).to(device)
    gen_state = ckpt.get("generator", None)
    if gen_state:
        model.generator.load_state_dict(gen_state, strict=False)
    else:
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    return model


def synthesize_sentence(
    flow, vocoder, text: str, device,
    n_steps: int = 8, cfg_scale: float = 2.0,
    use_heun: bool = False,
    g2p=None, char_vocab_size: int = 256,
) -> Tuple[np.ndarray, float]:
    """Synthesize one sentence. Returns (audio, gen_time_sec)."""

    if g2p is not None:
        char_ids = g2p.encode(text, add_bos=True, add_eos=True)
    else:
        char_ids = torch.tensor(
            [ord(c) % char_vocab_size for c in text], dtype=torch.long
        )
    if char_ids.dim() == 1:
        char_ids = char_ids.unsqueeze(0)
    char_ids = char_ids.to(device)

    speaker_ids = None
    if flow.cfg.n_speakers > 0:
        speaker_ids = torch.tensor([0], device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        mel = flow.sample(char_ids, n_frames=0, n_steps=n_steps,
                          speaker_ids=speaker_ids, cfg_scale=cfg_scale,
                          use_heun=use_heun)
        audio = vocoder.generate(mel)
    gen_time = time.perf_counter() - t0
    return audio.squeeze().cpu().numpy(), gen_time


# ─── Main entrypoints ───────────────────────────────────────────────────


def run_single_mode(args) -> EvalReport:
    """--mode single."""
    m = evaluate_single_pair(
        args.generated, args.reference, args.text or "", "single",
        use_utmos=getattr(args, "utmos", False),
    )
    results = [asdict(m)]
    report = aggregate_report(results)
    report.mode = "single"
    return report


def run_batch_mode(args) -> EvalReport:
    """--mode batch."""
    gen_dir = Path(args.generated_dir)
    ref_dir = Path(args.ref_dir) if args.ref_dir else None
    ref_texts = []
    if args.ref_texts:
        ref_texts = Path(args.ref_texts).read_text().strip().split("\n")

    wavs = sorted(gen_dir.glob("*.wav"))
    results = []
    for i, wav in enumerate(wavs):
        ref_wav = None
        if ref_dir:
            cand = ref_dir / wav.name
            if not cand.exists():
                refs = list(ref_dir.glob("*.wav"))
                cand = refs[i] if i < len(refs) else None
            ref_wav = cand
        ref_txt = ref_texts[i] if i < len(ref_texts) else ""
        if not ref_wav:
            continue
        m = evaluate_single_pair(
            str(wav), str(ref_wav), ref_txt, wav.name,
            use_utmos=getattr(args, "utmos", False),
        )
        results.append(asdict(m))

    report = aggregate_report(results)
    report.mode = "batch"
    return report


def run_synthesize_mode(args) -> EvalReport:
    """--mode synthesize: Flow v3 + vocoder, generate, then evaluate."""

    device = args.device
    g2p = None
    if args.phonemes:
        try:
            from g2p import PhonemeFrontend
            g2p = PhonemeFrontend()
        except ImportError:
            print("WARNING: --phonemes requires g2p; falling back to characters")

    print("Loading Flow v3...")
    flow, flow_cfg = load_flow_v3(args.flow_checkpoint, device, args.model_size)
    print("Loading Vocoder...")
    vocoder = load_vocoder(args.vocoder_checkpoint, device)
    char_vocab_size = getattr(flow_cfg, "char_vocab_size", 256)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences = TEST_SENTENCES
    if args.sentences_file:
        sentences = Path(args.sentences_file).read_text().strip().split("\n")
    ref_dir = Path(args.ref_dir) if args.ref_dir else None
    ref_texts = sentences

    results = []
    sr = 24000
    for i, text in enumerate(sentences):
        print(f"  [{i+1}/{len(sentences)}] Synthesizing...")
        audio, gen_time = synthesize_sentence(
            flow, vocoder, text, device,
            n_steps=args.n_steps, cfg_scale=args.cfg_scale,
            use_heun=args.heun,
            g2p=g2p, char_vocab_size=char_vocab_size,
        )
        duration = len(audio) / sr
        rtf = gen_time / max(duration, 0.01)

        out_path = out_dir / f"eval_{i:04d}.wav"
        save_audio(str(out_path), audio, sr)

        m = EvalMetrics(
            file=out_path.name,
            text=text,
            duration_sec=duration,
            gen_time_sec=gen_time,
            rtf=rtf,
        )

        if ref_dir:
            ref_wav = ref_dir / out_path.name
            if not ref_wav.exists():
                refs = list(ref_dir.glob("*.wav"))
                ref_wav = refs[i] if i < len(refs) else None
            if ref_wav:
                ref_audio, _ = load_audio(str(ref_wav))
                min_len = min(len(audio), len(ref_audio))
                m.pesq = compute_pesq(ref_audio[:min_len], audio[:min_len], sr)
                m.stoi = compute_stoi_score(ref_audio[:min_len], audio[:min_len], sr)
                m.mcd_db = compute_mcd(ref_audio[:min_len], audio[:min_len], sr)
                m.f0_rmse, m.f0_corr = compute_f0_metrics(
                    ref_audio[:min_len], audio[:min_len], sr
                )
                m.speaker_sim = compute_speaker_similarity(
                    ref_audio[:min_len], audio[:min_len], sr
                )
                m.spectral_convergence = compute_spectral_convergence(
                    ref_audio[:min_len], audio[:min_len]
                )
                m.mos_proxy = compute_mos_proxy(m.pesq, m.stoi, m.mcd_db, m.f0_corr)
        if getattr(args, "utmos", False):
            m.utmos = compute_utmos_single(audio, sr)

        if ref_texts and i < len(ref_texts):
            hyp = transcribe_whisper(str(out_path), sr)
            if hyp is not None:
                m.wer_pct = compute_wer_pct(ref_texts[i], hyp)

        results.append(asdict(m))

    report = aggregate_report(results)
    report.mode = "synthesize"
    return report


def aggregate_report(results: List[dict]) -> EvalReport:
    """Build aggregate report from per-file results."""
    report = EvalReport(n_samples=len(results), results=results)

    def mean_valid(key: str, default: float = 0.0) -> float:
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float)) and r[key] >= 0]
        return float(np.mean(vals)) if vals else default

    report.mean_mos_proxy = mean_valid("mos_proxy", -1.0)
    report.mean_pesq = mean_valid("pesq", -1.0)
    report.mean_stoi = mean_valid("stoi", -1.0)
    report.mean_wer_pct = mean_valid("wer_pct", -1.0)
    report.mean_mcd_db = mean_valid("mcd_db", -1.0)
    report.mean_f0_rmse = mean_valid("f0_rmse", -1.0)
    report.mean_f0_corr = mean_valid("f0_corr", -1.0)
    report.mean_rtf = mean_valid("rtf", -1.0)
    report.mean_speaker_sim = mean_valid("speaker_sim", -1.0)
    report.mean_spectral_convergence = mean_valid("spectral_convergence", -1.0)
    report.mean_utmos = mean_valid("utmos", -1.0)

    mos_for_grade = report.mean_mos_proxy
    if mos_for_grade < 0:
        mos_for_grade = compute_mos_proxy(
            report.mean_pesq, report.mean_stoi,
            report.mean_mcd_db, report.mean_f0_corr
        )
    # Blend MOS proxy with UTMOS when both available for grade
    if report.mean_utmos >= 0 and mos_for_grade >= 0:
        grade_score = 0.5 * mos_for_grade + 0.5 * report.mean_utmos
    else:
        grade_score = mos_for_grade if mos_for_grade >= 0 else report.mean_utmos
    report.grade_score = grade_score
    report.grade = compute_grade(grade_score)
    return report


def print_scorecard(report: EvalReport) -> None:
    """Print formatted scorecard table."""
    t = TARGETS
    r = report

    print("\n" + "=" * 70)
    print("  TTS EVALUATION SCORECARD")
    print("=" * 70)
    print(f"  Mode: {report.mode}  |  Samples: {report.n_samples}")
    print("-" * 70)
    print(f"  {'Metric':<24} {'Value':>10} {'Target':>10} {'Status':>8}")
    print("-" * 70)

    def row(name: str, val: float, target: float, higher_better: bool = True):
        if val < 0:
            status = "N/A"
        else:
            ok = (val >= target) if higher_better else (val <= target)
            status = "✓" if ok else "✗"
        print(f"  {name:<24} {val:>10.3f} {target:>10.1f} {status:>8}")

    if r.mean_mos_proxy >= 0:
        row("MOS proxy", r.mean_mos_proxy, t["mos_proxy"])
    if r.mean_pesq >= 0:
        row("PESQ", r.mean_pesq, t["pesq"])
    if r.mean_stoi >= 0:
        row("STOI", r.mean_stoi, t["stoi"])
    if r.mean_wer_pct >= 0:
        row("WER (%)", r.mean_wer_pct, t["wer_pct"], higher_better=False)
    if r.mean_mcd_db >= 0:
        row("MCD (dB)", r.mean_mcd_db, t["mcd_db"], higher_better=False)
    if r.mean_f0_corr >= 0:
        row("F0 correlation", r.mean_f0_corr, t["f0_corr"])
    if r.mean_rtf >= 0:
        row("RTF", r.mean_rtf, t["rtf"], higher_better=False)
    if r.mean_speaker_sim >= 0:
        row("Speaker similarity", r.mean_speaker_sim, t["speaker_sim"])
    if r.mean_utmos >= 0:
        row("UTMOS", r.mean_utmos, t["utmos"])

    print("-" * 70)
    print(f"  GRADE: {report.grade}  (score: {report.grade_score:.2f})")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(
        description="Comprehensive TTS evaluation against human-level benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--mode", choices=["single", "batch", "synthesize"], required=True)
    ap.add_argument("--generated", help="Generated WAV (single mode)")
    ap.add_argument("--reference", help="Reference WAV (single mode)")
    ap.add_argument("--text", default="", help="Reference text (single mode, for WER)")
    ap.add_argument("--generated-dir", help="Directory of generated WAVs (batch)")
    ap.add_argument("--ref-dir", help="Reference WAV directory (batch/synthesize)")
    ap.add_argument("--ref-texts", help="Reference texts file, one per line (batch)")
    ap.add_argument("--flow-checkpoint", help="Flow v3 checkpoint (synthesize)")
    ap.add_argument("--vocoder-checkpoint", help="Vocoder checkpoint (synthesize)")
    ap.add_argument("--output-dir", default="eval_output", help="Output dir (synthesize)")
    ap.add_argument("--sentences-file", help="Custom sentences file (synthesize)")
    ap.add_argument("--output", default="eval_report.json", help="JSON report path")
    ap.add_argument("--device", default="mps", help="Device (mps/cuda/cpu)")
    ap.add_argument("--n-steps", type=int, default=8, help="ODE steps (synthesize)")
    ap.add_argument("--cfg-scale", type=float, default=2.0, help="CFG scale (synthesize)")
    ap.add_argument("--phonemes", action="store_true",
                    help="Use PhonemeFrontend for text encoding (synthesize)")
    ap.add_argument("--heun", action="store_true",
                    help="Use Heun 2nd-order ODE solver (synthesize)")
    ap.add_argument("--model-size", choices=["base", "large"], default="base",
                    help="Flow config: base or large (synthesize)")
    ap.add_argument("--utmos", action="store_true",
                    help="Compute UTMOS (requires speechmos or transformers, disabled by default)")
    args = ap.parse_args()

    if not HAS_SOUNDFILE:
        print("ERROR: soundfile required. pip install soundfile")
        return 1

    report = None
    if args.mode == "single":
        if not args.generated or not args.reference:
            print("ERROR: single mode requires --generated and --reference")
            return 1
        report = run_single_mode(args)
    elif args.mode == "batch":
        if not args.generated_dir or not args.ref_dir:
            print("ERROR: batch mode requires --generated-dir and --ref-dir")
            return 1
        report = run_batch_mode(args)
    elif args.mode == "synthesize":
        if not args.flow_checkpoint or not args.vocoder_checkpoint:
            print("ERROR: synthesize mode requires --flow-checkpoint and --vocoder-checkpoint")
            return 1
        report = run_synthesize_mode(args)

    if report is None:
        return 1

    print_scorecard(report)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    print(f"\n  Report saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
