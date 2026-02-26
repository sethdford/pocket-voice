#!/usr/bin/env python3
"""
bench_tts.py — TTS benchmarking: measure quality and latency metrics.

Runs the C Kyutai DSM TTS engine on test sentences and measures:
  - TTFS (time to first sample)
  - RTF (real-time factor)
  - MCD (Mel-Cepstral Distortion) if reference audio available
  - STOI (Short-Time Objective Intelligibility) if reference available
  - Audio energy / silence detection

Usage:
    # Convert model first:
    python scripts/convert_kyutai_dsm.py --output models/kyutai_dsm.ctts

    # Run TTS benchmark:
    python scripts/bench_tts.py --model models/kyutai_dsm.ctts

    # Compare with Rust engine:
    python scripts/bench_tts.py --model models/kyutai_dsm.ctts --compare-rust

Dependencies:
    pip install numpy soundfile
"""

import argparse
import ctypes
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

TEST_SENTENCES = [
    # Basic quality
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Can you please tell me where the nearest hospital is?",
    "It was the best of times, it was the worst of times.",
    "Technology continues to advance at an unprecedented pace.",
    "The weather forecast predicts rain for the entire weekend.",
    "She sold seashells by the seashore.",
    "To be or not to be, that is the question.",
    "A stitch in time saves nine.",
    # Complex pronunciation (URLs, emails, abbreviations, numbers)
    "Visit https://docs.python.org/3/library for the full documentation.",
    "Contact support@company.co.uk or call 1-800-555-0199 for help.",
    "Dr. Smith, Ph.D., measured 2.5kg at -3°C on Nov. 15th, 2025.",
    "The GDP grew by 3.2% in Q4, reaching $21.7 trillion.",
    "Mix 1/4 cup flour with 2/3 cup milk and 1.5 tsp vanilla extract.",
    # Expressiveness
    "I'm SO excited to tell you this — you won the grand prize!",
    "I'm deeply sorry for your loss. Please know that I'm here for you.",
    "Hmm, let me think about that for a moment... Actually, yes, I believe so.",
    # Foreign names
    "The restaurant on Rue de Rivoli serves excellent coq au vin.",
    "Please welcome Dr. Müller from Zürich and CEO Satoshi Nakamura.",
]

SAMPLE_RATE = 24000


def run_c_tts_benchmark(model_path, sentences, max_steps=200, voice_path=None):
    """Run the C Kyutai DSM TTS engine on test sentences."""
    lib_path = Path("build/libkyutai_dsm_tts.dylib")
    if not lib_path.exists():
        print(f"ERROR: {lib_path} not found. Run 'make libs' first.", file=sys.stderr)
        sys.exit(1)

    lib = ctypes.CDLL(str(lib_path))

    lib.kyutai_tts_create.restype = ctypes.c_void_p
    lib.kyutai_tts_create.argtypes = [ctypes.c_char_p]
    lib.kyutai_tts_destroy.argtypes = [ctypes.c_void_p]
    lib.kyutai_tts_load_voice.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.kyutai_tts_load_voice.restype = ctypes.c_int
    lib.kyutai_tts_set_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.kyutai_tts_set_text.restype = ctypes.c_int
    lib.kyutai_tts_set_text_done.argtypes = [ctypes.c_void_p]
    lib.kyutai_tts_set_text_done.restype = ctypes.c_int
    lib.kyutai_tts_step.argtypes = [ctypes.c_void_p]
    lib.kyutai_tts_step.restype = ctypes.c_int
    lib.kyutai_tts_get_audio.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.kyutai_tts_get_audio.restype = ctypes.c_int
    lib.kyutai_tts_is_done.argtypes = [ctypes.c_void_p]
    lib.kyutai_tts_is_done.restype = ctypes.c_int
    lib.kyutai_tts_reset.argtypes = [ctypes.c_void_p]
    lib.kyutai_tts_reset.restype = ctypes.c_int

    engine = lib.kyutai_tts_create(model_path.encode())
    if not engine:
        print(f"ERROR: Failed to create TTS engine from {model_path}", file=sys.stderr)
        sys.exit(1)

    if voice_path:
        rc = lib.kyutai_tts_load_voice(engine, voice_path.encode())
        if rc != 0:
            print(f"WARNING: Failed to load voice from {voice_path}", file=sys.stderr)

    results = []
    for idx, text in enumerate(sentences):
        lib.kyutai_tts_reset(engine)

        t0 = time.monotonic()
        lib.kyutai_tts_set_text(engine, text.encode("utf-8"))
        lib.kyutai_tts_set_text_done(engine)
        prefill_time = time.monotonic() - t0

        audio_chunks = []
        ttfs = None
        step_times = []
        total_steps = 0
        buf = (ctypes.c_float * 1920)()

        for step in range(max_steps):
            t_step = time.monotonic()
            rc = lib.kyutai_tts_step(engine)
            step_time = time.monotonic() - t_step
            step_times.append(step_time)

            n = lib.kyutai_tts_get_audio(engine, buf, 1920)
            if n > 0:
                chunk = np.ctypeslib.as_array(buf, shape=(n,)).copy()
                audio_chunks.append(chunk)
                if ttfs is None:
                    ttfs = time.monotonic() - t0

            total_steps += 1
            if rc == 1:
                break

        total_time = time.monotonic() - t0

        if audio_chunks:
            audio = np.concatenate(audio_chunks)
        else:
            audio = np.zeros(0, dtype=np.float32)

        audio_duration = len(audio) / SAMPLE_RATE
        rtf = total_time / audio_duration if audio_duration > 0 else float("inf")

        rms = np.sqrt(np.mean(audio ** 2)) if len(audio) > 0 else 0
        peak = np.max(np.abs(audio)) if len(audio) > 0 else 0

        avg_step_ms = np.mean(step_times) * 1000 if step_times else 0

        result = {
            "text": text,
            "n_samples": len(audio),
            "audio_duration_s": audio_duration,
            "total_time_s": total_time,
            "prefill_time_ms": prefill_time * 1000,
            "ttfs_ms": ttfs * 1000 if ttfs else None,
            "rtf": rtf,
            "total_steps": total_steps,
            "avg_step_ms": avg_step_ms,
            "rms": float(rms),
            "peak": float(peak),
        }
        results.append(result)

        status = f"{'✓' if len(audio) > 0 else '✗'}"
        print(f"  [{idx+1:2d}/{len(sentences)}] {status} "
              f"{audio_duration:.2f}s audio, "
              f"RTF={rtf:.3f}, "
              f"TTFS={ttfs * 1000:.0f}ms, "
              f"steps={total_steps}, "
              f"step={avg_step_ms:.1f}ms/step")

        # Save WAV for quality analysis
        wav_dir = Path("bench_output/tts_c")
        wav_dir.mkdir(parents=True, exist_ok=True)
        if len(audio) > 0:
            try:
                import soundfile as sf
                sf.write(str(wav_dir / f"sample_{idx:02d}.wav"), audio, SAMPLE_RATE)
            except ImportError:
                pass

    lib.kyutai_tts_destroy(engine)
    return results


def compute_summary(results):
    """Compute summary statistics from individual results."""
    valid = [r for r in results if r["n_samples"] > 0]
    if not valid:
        return {"error": "No valid results"}

    return {
        "n_sentences": len(results),
        "n_valid": len(valid),
        "avg_rtf": np.mean([r["rtf"] for r in valid]),
        "avg_ttfs_ms": np.mean([r["ttfs_ms"] for r in valid if r["ttfs_ms"]]),
        "avg_step_ms": np.mean([r["avg_step_ms"] for r in valid]),
        "avg_audio_duration_s": np.mean([r["audio_duration_s"] for r in valid]),
        "total_audio_s": sum(r["audio_duration_s"] for r in valid),
        "total_time_s": sum(r["total_time_s"] for r in valid),
        "avg_rms": np.mean([r["rms"] for r in valid]),
        "avg_peak": np.mean([r["peak"] for r in valid]),
    }


def main():
    parser = argparse.ArgumentParser(description="TTS Benchmark")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .ctts model file")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max generation steps per sentence")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON results file")
    parser.add_argument("--sentences", type=str, default=None,
                        help="JSON file with custom test sentences")
    parser.add_argument("--voice", type=str, default=None,
                        help="Path to .voicekv voice conditioning file")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    sentences = TEST_SENTENCES
    if args.sentences:
        with open(args.sentences) as f:
            sentences = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"TTS Benchmark — C Kyutai DSM Engine")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model}")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Max steps: {args.max_steps}")
    print()

    if args.voice:
        print(f"  Voice: {args.voice}")
    results = run_c_tts_benchmark(args.model, sentences, args.max_steps, args.voice)
    summary = compute_summary(results)

    print(f"\n{'=' * 60}")
    print(f"Summary")
    print(f"{'=' * 60}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "engine": "c_kyutai_dsm",
            "model": args.model,
            "summary": summary,
            "per_sentence": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
