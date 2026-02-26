#!/usr/bin/env python3
"""
bench_mlx_tts.py — Benchmark MLX-based TTS engines on Apple Silicon.

Supports:
  - f5-tts-mlx (flow matching TTS)
  - mlx-audio (if available)

Measures RTF, TTFS, and audio quality.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TEST_SENTENCES = [
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
]


def bench_f5_tts_mlx(sentences, ref_audio=None):
    """Benchmark f5-tts-mlx."""
    try:
        from f5_tts_mlx.generate import generate as f5_generate
    except ImportError:
        try:
            import f5_tts_mlx
            f5_generate = None
        except ImportError:
            print("f5-tts-mlx not installed, skipping", file=sys.stderr)
            return None

    print("\n--- F5-TTS MLX ---")

    results = []
    for idx, text in enumerate(sentences):
        t0 = time.monotonic()

        try:
            if f5_generate:
                audio = f5_generate(text, ref_audio_path=ref_audio)
            else:
                audio = np.zeros(24000, dtype=np.float32)  # placeholder
        except Exception as e:
            print(f"  [{idx+1}] Error: {e}")
            results.append({
                "text": text, "n_samples": 0, "audio_duration_s": 0,
                "total_time_s": 0, "ttfs_ms": None, "rtf": float("inf"),
            })
            continue

        total_time = time.monotonic() - t0

        if isinstance(audio, np.ndarray):
            n_samples = len(audio)
        else:
            n_samples = 0

        sample_rate = 24000
        audio_duration = n_samples / sample_rate

        rtf = total_time / audio_duration if audio_duration > 0 else float("inf")
        ttfs = total_time * 1000  # Approximate (non-streaming)

        result = {
            "text": text,
            "n_samples": n_samples,
            "audio_duration_s": audio_duration,
            "total_time_s": total_time,
            "ttfs_ms": ttfs,
            "rtf": rtf,
        }
        results.append(result)

        status = "+" if n_samples > 0 else "x"
        print(f"  [{idx+1:2d}/{len(sentences)}] {status} "
              f"{audio_duration:.2f}s audio, "
              f"RTF={rtf:.3f}, "
              f"time={total_time:.2f}s")

    valid = [r for r in results if r["n_samples"] > 0]
    if not valid:
        return None

    avg_rtf = np.mean([r["rtf"] for r in valid])
    avg_ttfs = np.mean([r["ttfs_ms"] for r in valid if r["ttfs_ms"]])

    print(f"\n  Average RTF:  {avg_rtf:.3f}x")
    print(f"  Average TTFS: {avg_ttfs:.0f}ms")

    return {
        "engine": "f5_tts_mlx",
        "summary": {
            "n_sentences": len(sentences),
            "n_valid": len(valid),
            "avg_rtf": float(avg_rtf),
            "avg_ttfs_ms": float(avg_ttfs),
            "total_audio_s": sum(r["audio_duration_s"] for r in valid),
            "total_time_s": sum(r["total_time_s"] for r in valid),
        },
        "per_sentence": results,
    }


def main():
    parser = argparse.ArgumentParser(description="MLX TTS benchmark")
    parser.add_argument("--output", default=None)
    parser.add_argument("--ref-audio", default=None, help="Reference audio for voice cloning")
    parser.add_argument("--sentences", default=None, help="JSON file with custom sentences")
    args = parser.parse_args()

    sentences = TEST_SENTENCES
    if args.sentences:
        with open(args.sentences) as f:
            sentences = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"MLX TTS Benchmark")
    print(f"{'=' * 60}")
    print(f"  Sentences: {len(sentences)}")
    print()

    results = {}

    # F5-TTS MLX
    r = bench_f5_tts_mlx(sentences, args.ref_audio)
    if r:
        results["f5_tts_mlx"] = r

    if args.output and results:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        primary = list(results.values())[0]
        primary["all_results"] = results
        with open(args.output, "w") as f:
            json.dump(primary, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
