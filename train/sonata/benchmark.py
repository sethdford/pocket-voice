#!/usr/bin/env python3
"""Sonata Comprehensive Benchmark Suite.

Measures PESQ, STOI, MCD, RTF, and codec reconstruction quality.
Compares against reference audio to prove human-quality synthesis.

Metrics:
  - PESQ (Perceptual Evaluation of Speech Quality): ≥3.5 = toll quality
  - STOI (Short-Time Objective Intelligibility): ≥0.95 = excellent
  - MCD (Mel-Cepstral Distortion): ≤4.0 dB = high quality
  - RTF (Real-Time Factor): <0.2 = 5x realtime
  - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio): ≥15 dB = good

Usage:
  python train/sonata/benchmark.py \
    --codec-ckpt train/checkpoints/codec/sonata_codec_final.pt \
    --audio-dir train/data/LibriSpeech/dev-clean \
    --output bench_output/sonata_codec_bench.json
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np

from config import CodecConfig
from codec import SonataCodec


# ─── Audio Quality Metrics ────────────────────────────────────────────────────

def compute_si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (dB).
    ≥20 dB = excellent, ≥15 dB = good, ≥10 dB = acceptable.
    """
    ref = reference - np.mean(reference)
    est = estimate - np.mean(estimate)

    dot = np.dot(ref, est)
    s_ref_sq = np.dot(ref, ref)
    if s_ref_sq < 1e-10:
        return 0.0

    s_target = (dot / s_ref_sq) * ref
    e_noise = est - s_target
    si_sdr = 10 * np.log10(
        np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-10)
    )
    return float(si_sdr)


def compute_stoi(reference: np.ndarray, estimate: np.ndarray,
                 sr: int = 24000) -> float:
    """Simplified Short-Time Objective Intelligibility.
    ≥0.95 = excellent, ≥0.90 = good, ≥0.80 = acceptable.
    """
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)

    min_len = min(len(reference), len(estimate))
    ref = reference[:min_len]
    est = estimate[:min_len]

    n_frames = (min_len - frame_len) // hop
    if n_frames < 1:
        return 0.0

    correlations = []
    for i in range(n_frames):
        start = i * hop
        r = ref[start:start + frame_len]
        e = est[start:start + frame_len]

        r_mean = np.mean(r)
        e_mean = np.mean(e)
        r_std = np.std(r) + 1e-10
        e_std = np.std(e) + 1e-10

        corr = np.mean((r - r_mean) * (e - e_mean)) / (r_std * e_std)
        correlations.append(max(0, corr))

    return float(np.mean(correlations))


def compute_mcd(reference: np.ndarray, estimate: np.ndarray,
                sr: int = 24000, n_mfcc: int = 13) -> float:
    """Mel-Cepstral Distortion (dB).
    ≤4.0 dB = high quality, ≤6.0 dB = acceptable.
    """
    def mfcc(audio, sr, n_mfcc=13, n_fft=1024, hop=256, n_mels=80):
        spec = np.abs(np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(
                np.pad(audio, (n_fft//2, n_fft//2)),
                n_fft
            )[::hop] * np.hanning(n_fft)
        ))

        mel_fb = np.zeros((n_mels, n_fft // 2 + 1))
        low_mel = 0
        high_mel = 2595 * np.log10(1 + sr / 2 / 700)
        mel_pts = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
        bins = np.floor((hz_pts / sr) * n_fft).astype(int)

        for i in range(n_mels):
            lo, mid, hi = bins[i], bins[i+1], bins[i+2]
            for j in range(lo, mid):
                mel_fb[i, j] = (j - lo) / max(1, mid - lo)
            for j in range(mid, hi):
                mel_fb[i, j] = (hi - j) / max(1, hi - mid)

        mel_spec = np.log(np.dot(spec, mel_fb.T).clip(min=1e-7))
        from scipy.fft import dct
        return dct(mel_spec, type=2, n=n_mfcc, axis=-1)

    try:
        ref_mfcc = mfcc(reference, sr, n_mfcc)
        est_mfcc = mfcc(estimate, sr, n_mfcc)
    except Exception:
        return 99.0

    min_frames = min(ref_mfcc.shape[0], est_mfcc.shape[0])
    diff = ref_mfcc[:min_frames, 1:] - est_mfcc[:min_frames, 1:]
    mcd = (10.0 / np.log(10)) * np.sqrt(2) * np.mean(np.sqrt(np.sum(diff**2, axis=-1)))
    return float(mcd)


def compute_pesq_approx(reference: np.ndarray, estimate: np.ndarray,
                        sr: int = 24000) -> float:
    """Approximate PESQ score based on spectral and temporal similarity.
    Range: 1.0–4.5. ≥3.5 = toll quality, ≥4.0 = excellent.

    This is an approximation. For true PESQ, install pesq package.
    """
    try:
        from pesq import pesq
        if sr != 16000:
            ref_t = torch.from_numpy(reference).float().unsqueeze(0).unsqueeze(0)
            est_t = torch.from_numpy(estimate).float().unsqueeze(0).unsqueeze(0)
            new_len = int(ref_t.shape[-1] * 16000 / sr)
            ref_16k = F.interpolate(ref_t, size=new_len, mode='linear',
                                     align_corners=False).squeeze().numpy()
            est_16k = F.interpolate(est_t, size=new_len, mode='linear',
                                     align_corners=False).squeeze().numpy()
        else:
            ref_16k, est_16k = reference, estimate
        return pesq(16000, ref_16k, est_16k, 'wb')
    except ImportError:
        pass

    stoi = compute_stoi(reference, estimate, sr)
    si_sdr = compute_si_sdr(reference, estimate)
    pesq_approx = 1.0 + 3.5 * max(0, min(1, stoi)) * max(0, min(1, si_sdr / 30))
    return float(min(4.5, pesq_approx))


# ─── Benchmark Runner ─────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_codec(args):
    device = torch.device(args.device)

    ckpt = torch.load(args.codec_ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = CodecConfig(**{k: v for k, v in cfg_dict.items()
                         if k in CodecConfig.__dataclass_fields__})
    model = SonataCodec(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}")
    print(f"  SONATA CODEC BENCHMARK")
    print(f"{'='*70}")
    print(f"  Model: {n_params/1e6:.1f}M params")
    print(f"  Frame rate: {cfg.frame_rate} Hz")
    print(f"  Device: {device}")

    audio_dir = Path(args.audio_dir)
    audio_files = sorted(
        list(audio_dir.rglob("*.flac")) +
        list(audio_dir.rglob("*.wav"))
    )[:args.max_files]
    print(f"  Audio files: {len(audio_files)}")
    print()

    results = {
        "si_sdr": [], "stoi": [], "mcd": [], "pesq": [],
        "rtf": [], "n_frames": [], "duration_sec": [],
    }

    for i, audio_file in enumerate(audio_files):
        data, sr = sf.read(str(audio_file), dtype='float32')
        audio = torch.from_numpy(data)
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)
        if sr != cfg.sample_rate:
            ratio = cfg.sample_rate / sr
            new_len = int(audio.shape[0] * ratio)
            audio = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear',
                align_corners=False
            ).squeeze()

        if audio.shape[0] > cfg.sample_rate * args.max_duration:
            audio = audio[:cfg.sample_rate * int(args.max_duration)]

        duration = audio.shape[0] / cfg.sample_rate

        audio_gpu = audio.unsqueeze(0).to(device)

        t0 = time.perf_counter()
        reconstructed, tokens, acoustic = model(audio_gpu)
        if device.type == 'mps':
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0

        reconstructed = reconstructed.squeeze(0).cpu()
        min_len = min(reconstructed.shape[0], audio.shape[0])
        ref = audio[:min_len].numpy()
        est = reconstructed[:min_len].numpy()

        si_sdr = compute_si_sdr(ref, est)
        stoi = compute_stoi(ref, est, cfg.sample_rate)
        mcd = compute_mcd(ref, est, cfg.sample_rate)
        pesq_val = compute_pesq_approx(ref, est, cfg.sample_rate)
        rtf = elapsed / duration

        results["si_sdr"].append(si_sdr)
        results["stoi"].append(stoi)
        results["mcd"].append(mcd)
        results["pesq"].append(pesq_val)
        results["rtf"].append(rtf)
        results["n_frames"].append(tokens.shape[1])
        results["duration_sec"].append(duration)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:4d}/{len(audio_files)}] {audio_file.name[:30]:30s} "
                  f"SI-SDR={si_sdr:6.1f}dB  STOI={stoi:.3f}  "
                  f"MCD={mcd:5.2f}dB  PESQ≈{pesq_val:.2f}  "
                  f"RTF={rtf:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY ({len(audio_files)} files)")
    print(f"{'='*70}")

    summary = {}
    for metric in ["si_sdr", "stoi", "mcd", "pesq", "rtf"]:
        vals = results[metric]
        summary[metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p50": float(np.percentile(vals, 50)),
            "p95": float(np.percentile(vals, 95)),
        }

    targets = {
        "si_sdr": {"target": 15.0, "unit": "dB", "higher_better": True},
        "stoi": {"target": 0.95, "unit": "", "higher_better": True},
        "mcd": {"target": 4.0, "unit": "dB", "higher_better": False},
        "pesq": {"target": 3.5, "unit": "", "higher_better": True},
        "rtf": {"target": 0.2, "unit": "x", "higher_better": False},
    }

    for metric, info in targets.items():
        s = summary[metric]
        met = s["mean"]
        target = info["target"]
        if info["higher_better"]:
            grade = "PASS" if met >= target else "FAIL"
        else:
            grade = "PASS" if met <= target else "FAIL"

        print(f"  {metric:8s}: mean={met:7.3f} ± {s['std']:5.3f}"
              f"  (p50={s['p50']:7.3f}, p95={s['p95']:7.3f})"
              f"  target={target:5.2f}{info['unit']:2s}  [{grade}]")

    total_audio = sum(results["duration_sec"])
    total_time = sum(results["rtf"]) * sum(results["duration_sec"]) / len(results["rtf"])
    print(f"\n  Total audio: {total_audio:.1f}s")
    print(f"  Avg RTF: {np.mean(results['rtf']):.4f}x (={1/np.mean(results['rtf']):.0f}x realtime)")

    overall_pass = all(
        (summary[m]["mean"] >= t["target"] if t["higher_better"]
         else summary[m]["mean"] <= t["target"])
        for m, t in targets.items()
    )
    grade = "A" if overall_pass else "B" if sum(1 for m, t in targets.items()
        if (summary[m]["mean"] >= t["target"] if t["higher_better"]
            else summary[m]["mean"] <= t["target"])) >= 3 else "C"

    print(f"\n  OVERALL GRADE: {grade}")
    print(f"{'='*70}")

    output = {
        "model": "sonata_codec",
        "n_params": n_params,
        "n_files": len(audio_files),
        "summary": summary,
        "targets": targets,
        "grade": grade,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec-ckpt", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output", default="bench_output/sonata_codec_bench.json")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--max-duration", type=float, default=10.0)
    args = parser.parse_args()
    benchmark_codec(args)


if __name__ == "__main__":
    main()
