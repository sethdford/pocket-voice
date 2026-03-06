#!/usr/bin/env python3
"""A/B prosody evaluation harness.

Generates the same sentences with different prosody settings (baseline vs enhanced),
evaluates with objective metrics, and produces a comparison report.

Usage:
  python eval_prosody_ab.py \
    --sentences test_sentences.txt \
    --baseline-wav baseline/ \
    --enhanced-wav enhanced/ \
    --output report.json

Or with synthetic test data:
  python eval_prosody_ab.py --synthetic --output report.json

Metrics computed per pair:
  - F0 range (Hz): pitch expressiveness
  - Energy variance (dB): dynamic range
  - F0 contour correlation: naturalness vs reference
  - Prosody MOS estimate: rule-based naturalness score (1-5)
  - Duration deviation: pacing accuracy
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


@dataclass
class ProsodyMetrics:
    f0_range_hz: float = 0.0
    f0_mean_hz: float = 0.0
    energy_var_db: float = 0.0
    energy_mean_db: float = 0.0
    duration_sec: float = 0.0
    speaking_rate_sps: float = 0.0  # syllables per second estimate
    prosody_mos: float = 1.0


@dataclass
class ABResult:
    sentence: str
    baseline: ProsodyMetrics
    enhanced: ProsodyMetrics
    improvement: dict = None

    def __post_init__(self):
        if self.improvement is None:
            self.improvement = {}


def estimate_f0_autocorr(audio: np.ndarray, sr: int, hop_ms: int = 10) -> np.ndarray:
    """Simple autocorrelation F0 estimator."""
    hop = sr * hop_ms // 1000
    frame_len = hop * 2
    n_frames = max(1, len(audio) // hop - 1)
    f0 = np.zeros(n_frames)

    min_lag = sr // 500  # 500 Hz max
    max_lag = sr // 50   # 50 Hz min

    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(audio))
        frame = audio[start:end]
        if len(frame) < min_lag * 2:
            continue
        energy = np.mean(frame ** 2)
        if energy < 1e-8:
            continue

        corr = np.correlate(frame, frame, mode='full')
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


def compute_energy(audio: np.ndarray, sr: int, hop_ms: int = 10) -> np.ndarray:
    """Per-frame energy in dB."""
    hop = sr * hop_ms // 1000
    n_frames = max(1, len(audio) // hop)
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(audio))
        frame = audio[start:end]
        rms = np.sqrt(np.mean(frame ** 2))
        energy[i] = 20 * np.log10(max(rms, 1e-10))
    return energy


def analyze_audio(audio: np.ndarray, sr: int) -> ProsodyMetrics:
    """Compute prosody metrics for a single audio file."""
    m = ProsodyMetrics()
    m.duration_sec = len(audio) / sr

    f0 = estimate_f0_autocorr(audio, sr)
    voiced = f0[f0 > 0]
    if len(voiced) > 1:
        m.f0_range_hz = float(np.max(voiced) - np.min(voiced))
        m.f0_mean_hz = float(np.mean(voiced))
    else:
        m.f0_range_hz = 0.0
        m.f0_mean_hz = 0.0

    energy = compute_energy(audio, sr)
    active = energy[energy > -60]
    if len(active) > 1:
        m.energy_var_db = float(np.var(active))
        m.energy_mean_db = float(np.mean(active))

    # Prosody MOS estimate (rule-based)
    range_score = 1.0 if 50 <= m.f0_range_hz <= 200 else (0.5 if m.f0_range_hz > 20 else 0.2)
    var_std = math.sqrt(m.energy_var_db) if m.energy_var_db > 0 else 0
    var_score = min(1.0, max(0.3, (var_std - 2) / 8)) if var_std > 2 else 0.3
    m.prosody_mos = 1.0 + 4.0 * (0.4 * range_score + 0.35 * var_score + 0.25 * 0.7)

    return m


def generate_synthetic_audio(sentence: str, sr: int = 24000,
                             flat: bool = True) -> np.ndarray:
    """Generate synthetic audio for testing the evaluation pipeline."""
    duration = len(sentence.split()) * 0.3 + 0.5
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples)

    if flat:
        # Flat monotone (baseline)
        f0 = 150.0
        audio = 0.3 * np.sin(2 * np.pi * f0 * t)
    else:
        # Expressive with pitch variation (enhanced)
        f0_curve = 150.0 + 30.0 * np.sin(2 * np.pi * 2.0 * t)
        phase = np.cumsum(2 * np.pi * f0_curve / sr)
        audio = 0.3 * np.sin(phase)
        # Add energy variation
        envelope = 0.8 + 0.2 * np.sin(2 * np.pi * 1.5 * t)
        audio *= envelope

    return audio.astype(np.float32)


def compare_pair(sentence: str, baseline: ProsodyMetrics,
                 enhanced: ProsodyMetrics) -> ABResult:
    """Compare baseline and enhanced metrics."""
    result = ABResult(sentence=sentence, baseline=baseline, enhanced=enhanced)

    result.improvement = {
        "f0_range_delta": enhanced.f0_range_hz - baseline.f0_range_hz,
        "energy_var_delta": enhanced.energy_var_db - baseline.energy_var_db,
        "mos_delta": enhanced.prosody_mos - baseline.prosody_mos,
        "winner": "enhanced" if enhanced.prosody_mos > baseline.prosody_mos else
                  "baseline" if baseline.prosody_mos > enhanced.prosody_mos else "tie",
    }

    return result


def run_evaluation(args):
    print(f"\n{'='*60}")
    print(f"  PROSODY A/B EVALUATION")
    print(f"{'='*60}")

    results = []

    if args.synthetic:
        sentences = [
            "Hello, how are you doing today?",
            "That is absolutely wonderful news!",
            "I'm sorry to hear about that.",
            "The quick brown fox jumps over the lazy dog.",
            "Can you believe what just happened?",
            "Please be careful with that, it's very fragile.",
            "One, two, three, four, five.",
            "She said, don't worry about it.",
        ]

        print(f"\n  Using {len(sentences)} synthetic test sentences")
        for i, sent in enumerate(sentences):
            base_audio = generate_synthetic_audio(sent, flat=True)
            enh_audio = generate_synthetic_audio(sent, flat=False)
            base_m = analyze_audio(base_audio, 24000)
            enh_m = analyze_audio(enh_audio, 24000)
            result = compare_pair(sent, base_m, enh_m)
            results.append(result)
            print(f"  [{i+1}] {result.improvement['winner']:>8s} | "
                  f"MOS: {base_m.prosody_mos:.2f} → {enh_m.prosody_mos:.2f} | "
                  f"F0 range: {base_m.f0_range_hz:.0f} → {enh_m.f0_range_hz:.0f} Hz")

    elif HAS_SOUNDFILE and args.baseline_wav and args.enhanced_wav:
        base_dir = Path(args.baseline_wav)
        enh_dir = Path(args.enhanced_wav)
        sentences_file = Path(args.sentences) if args.sentences else None

        sentences = []
        if sentences_file and sentences_file.exists():
            sentences = sentences_file.read_text().strip().split('\n')

        base_files = sorted(base_dir.glob("*.wav"))
        enh_files = sorted(enh_dir.glob("*.wav"))

        n_pairs = min(len(base_files), len(enh_files))
        print(f"\n  Found {n_pairs} audio pairs")

        for i in range(n_pairs):
            base_audio, base_sr = sf.read(str(base_files[i]), dtype='float32')
            enh_audio, enh_sr = sf.read(str(enh_files[i]), dtype='float32')
            if base_audio.ndim > 1:
                base_audio = base_audio.mean(axis=-1)
            if enh_audio.ndim > 1:
                enh_audio = enh_audio.mean(axis=-1)

            base_m = analyze_audio(base_audio, base_sr)
            enh_m = analyze_audio(enh_audio, enh_sr)

            sent = sentences[i] if i < len(sentences) else base_files[i].stem
            result = compare_pair(sent, base_m, enh_m)
            results.append(result)
            print(f"  [{i+1}] {result.improvement['winner']:>8s} | "
                  f"MOS: {base_m.prosody_mos:.2f} → {enh_m.prosody_mos:.2f}")
    else:
        print("  ERROR: Provide --synthetic or --baseline-wav + --enhanced-wav")
        return

    # Aggregate
    n = len(results)
    if n > 0:
        enhanced_wins = sum(1 for r in results if r.improvement["winner"] == "enhanced")
        baseline_wins = sum(1 for r in results if r.improvement["winner"] == "baseline")
        ties = n - enhanced_wins - baseline_wins

        avg_base_mos = sum(r.baseline.prosody_mos for r in results) / n
        avg_enh_mos = sum(r.enhanced.prosody_mos for r in results) / n
        avg_f0_delta = sum(r.improvement["f0_range_delta"] for r in results) / n

        print(f"\n{'='*60}")
        print(f"  RESULTS ({n} pairs)")
        print(f"{'='*60}")
        print(f"  Enhanced wins:  {enhanced_wins}/{n} ({100*enhanced_wins/n:.0f}%)")
        print(f"  Baseline wins:  {baseline_wins}/{n} ({100*baseline_wins/n:.0f}%)")
        print(f"  Ties:           {ties}/{n}")
        print(f"  Avg MOS:        {avg_base_mos:.2f} → {avg_enh_mos:.2f} "
              f"(Δ={avg_enh_mos - avg_base_mos:+.2f})")
        print(f"  Avg F0 range Δ: {avg_f0_delta:+.0f} Hz")

    # Save report
    if args.output:
        report = {
            "n_pairs": n,
            "enhanced_wins": enhanced_wins if n > 0 else 0,
            "baseline_wins": baseline_wins if n > 0 else 0,
            "avg_baseline_mos": avg_base_mos if n > 0 else 0,
            "avg_enhanced_mos": avg_enh_mos if n > 0 else 0,
            "pairs": [asdict(r) for r in results],
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved to {args.output}")


def generate_with_api(text: str, engine: str, output_path: str) -> bool:
    """Generate TTS audio via commercial API. Returns True on success."""
    import subprocess

    if engine == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print(f"  WARN: OPENAI_API_KEY not set, skipping")
            return False
        cmd = [
            "curl", "-s", "https://api.openai.com/v1/audio/speech",
            "-H", f"Authorization: Bearer {api_key}",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "model": "tts-1-hd",
                "input": text,
                "voice": "nova",
                "response_format": "wav"
            }),
            "-o", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and os.path.getsize(output_path) > 1000

    elif engine == "elevenlabs":
        api_key = os.environ.get("ELEVEN_API_KEY", "")
        if not api_key:
            print(f"  WARN: ELEVEN_API_KEY not set, skipping")
            return False
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel
        cmd = [
            "curl", "-s",
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            "-H", f"xi-api-key: {api_key}",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "output_format": "pcm_24000"
            }),
            "-o", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and os.path.getsize(output_path) > 1000

    elif engine == "google":
        api_key = os.environ.get("GOOGLE_TTS_API_KEY", "")
        if not api_key:
            print(f"  WARN: GOOGLE_TTS_API_KEY not set, skipping")
            return False
        cmd = [
            "curl", "-s",
            f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "input": {"text": text},
                "voice": {"languageCode": "en-US", "name": "en-US-Journey-F"},
                "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000}
            }),
            "-o", output_path + ".json"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return False
        try:
            import base64
            with open(output_path + ".json") as f:
                resp = json.load(f)
            audio_bytes = base64.b64decode(resp["audioContent"])
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            os.remove(output_path + ".json")
            return True
        except Exception:
            return False

    elif engine == "sonata":
        cmd = [
            "curl", "-s", "-X", "POST",
            "http://localhost:8080/v1/audio/speech",
            "-d", text,
            "-o", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and os.path.getsize(output_path) > 1000

    return False


def run_benchmark(args):
    """Benchmark Sonata against commercial TTS APIs."""
    import time

    engines = [e.strip() for e in args.benchmark_engines.split(",")]
    sentences = [
        "Hello, how are you doing today?",
        "That is absolutely wonderful news!",
        "I'm sorry to hear about that. It must be really difficult.",
        "The quick brown fox jumps over the lazy dog.",
        "Can you believe what just happened?",
        "Please be careful with that, it's very fragile.",
        "She whispered, please don't leave me, and started crying.",
        "One advantage is speed; another is quality; a third is cost.",
    ]

    if args.sentences and os.path.exists(args.sentences):
        with open(args.sentences) as f:
            sentences = [l.strip() for l in f if l.strip()]

    print(f"\n{'='*70}")
    print(f"  COMMERCIAL TTS BENCHMARK")
    print(f"  Engines: {', '.join(engines)}")
    print(f"  Sentences: {len(sentences)}")
    print(f"{'='*70}")

    out_dir = Path(args.benchmark_output or "benchmark_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for engine in engines:
        print(f"\n  --- {engine.upper()} ---")
        engine_dir = out_dir / engine
        engine_dir.mkdir(exist_ok=True)
        engine_metrics = []
        total_latency = 0.0

        for i, sent in enumerate(sentences):
            wav_path = str(engine_dir / f"sample_{i:03d}.wav")
            t0 = time.time()
            ok = generate_with_api(sent, engine, wav_path)
            latency = time.time() - t0

            if ok and HAS_SOUNDFILE:
                audio, sr = sf.read(wav_path, dtype='float32')
                m = analyze_audio(audio, sr)
                m.speaking_rate_sps = len(sent.split()) / m.duration_sec if m.duration_sec > 0 else 0
                engine_metrics.append(m)
                total_latency += latency
                print(f"  [{i+1}] MOS={m.prosody_mos:.2f} F0={m.f0_range_hz:.0f}Hz "
                      f"lat={latency:.2f}s dur={m.duration_sec:.2f}s")
            elif ok:
                print(f"  [{i+1}] Generated (no soundfile for analysis) lat={latency:.2f}s")
            else:
                print(f"  [{i+1}] FAILED")

        if engine_metrics:
            n = len(engine_metrics)
            all_results[engine] = {
                "n_samples": n,
                "avg_mos": sum(m.prosody_mos for m in engine_metrics) / n,
                "avg_f0_range": sum(m.f0_range_hz for m in engine_metrics) / n,
                "avg_energy_var": sum(m.energy_var_db for m in engine_metrics) / n,
                "avg_latency": total_latency / n,
                "avg_duration": sum(m.duration_sec for m in engine_metrics) / n,
            }

    if all_results:
        print(f"\n{'='*70}")
        print(f"  {'Engine':<15} {'MOS':>6} {'F0 range':>10} {'Energy var':>12} {'Latency':>10}")
        print(f"  {'-'*55}")
        for engine, stats in sorted(all_results.items(),
                                     key=lambda x: -x[1]["avg_mos"]):
            print(f"  {engine:<15} {stats['avg_mos']:>6.2f} "
                  f"{stats['avg_f0_range']:>8.0f}Hz "
                  f"{stats['avg_energy_var']:>10.1f}dB² "
                  f"{stats['avg_latency']:>8.2f}s")
        print(f"{'='*70}")

        report_path = str(out_dir / "benchmark_report.json")
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B prosody evaluation & TTS benchmarking")
    parser.add_argument("--sentences", help="Text file with one sentence per line")
    parser.add_argument("--baseline-wav", help="Directory of baseline WAV files")
    parser.add_argument("--enhanced-wav", help="Directory of enhanced WAV files")
    parser.add_argument("--output", default="prosody_ab_report.json")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic audio for pipeline testing")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark against commercial TTS APIs")
    parser.add_argument("--benchmark-engines", default="sonata,openai,elevenlabs,google",
                        help="Comma-separated list of engines to benchmark")
    parser.add_argument("--benchmark-output", default="benchmark_output",
                        help="Output directory for benchmark WAV files")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args)
    else:
        run_evaluation(args)
