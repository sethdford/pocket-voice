#!/usr/bin/env python3
"""
bench_mlx_whisper.py — Benchmark MLX Whisper on LibriSpeech test-clean.

Requires: pip install mlx-whisper
Optional: pip install lightning-whisper-mlx

Measures RTF, WER, CER, and memory usage.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def load_librispeech(max_samples=None):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print("Loading LibriSpeech test-clean...")
    ds = load_dataset("librispeech_asr", "clean", split="test", trust_remote_code=True)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def bench_mlx_whisper(samples, model_name="mlx-community/whisper-large-v3-mlx"):
    """Benchmark mlx-whisper."""
    try:
        import mlx_whisper
    except ImportError:
        print("mlx-whisper not installed, skipping", file=sys.stderr)
        return None

    print(f"\n--- MLX Whisper ({model_name}) ---")

    # Cold start (first transcription loads the model)
    dummy = np.zeros(16000, dtype=np.float32)
    t0 = time.monotonic()
    mlx_whisper.transcribe(dummy, path_or_hf_repo=model_name)
    cold_start = time.monotonic() - t0
    print(f"  Cold start (model load + first inference): {cold_start:.2f}s")

    references = []
    hypotheses = []
    total_audio_sec = 0.0
    total_proc_sec = 0.0

    for i, sample in enumerate(samples):
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        ref = sample["text"].lower().strip()

        audio_sec = len(audio) / sr
        total_audio_sec += audio_sec

        t0 = time.monotonic()
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model_name,
            language="en",
        )
        proc_time = time.monotonic() - t0
        total_proc_sec += proc_time

        hyp = result.get("text", "").lower().strip()
        references.append(ref)
        hypotheses.append(hyp)

        if (i + 1) % 10 == 0:
            rtf = total_proc_sec / total_audio_sec
            print(f"  Processed {i + 1}/{len(samples)} (RTF: {rtf:.3f}x)")

    # WER
    try:
        from jiwer import wer, cer
        wer_val = wer(references, hypotheses)
        cer_val = cer(references, hypotheses)
    except ImportError:
        wer_val = -1.0
        cer_val = -1.0

    rtf = total_proc_sec / total_audio_sec if total_audio_sec > 0 else float("inf")

    print(f"\n{'=' * 50}")
    print(f"MLX Whisper Results ({model_name})")
    print(f"{'=' * 50}")
    print(f"  Samples:       {len(references)}")
    print(f"  Audio:         {total_audio_sec:.1f}s")
    print(f"  Processing:    {total_proc_sec:.1f}s")
    print(f"  RTF:           {rtf:.4f}x")
    print(f"  WER:           {wer_val * 100:.2f}%")
    print(f"  CER:           {cer_val * 100:.2f}%")
    print(f"  Cold start:    {cold_start:.2f}s")

    return {
        "engine": "mlx_whisper",
        "model": model_name,
        "n_samples": len(references),
        "audio_seconds": total_audio_sec,
        "processing_seconds": total_proc_sec,
        "rtf": rtf,
        "wer": wer_val,
        "cer": cer_val,
        "cold_start_seconds": cold_start,
    }


def bench_lightning_whisper(samples, model_size="large-v3"):
    """Benchmark lightning-whisper-mlx (faster batched inference)."""
    try:
        from lightning_whisper_mlx import LightningWhisperMLX
    except ImportError:
        print("lightning-whisper-mlx not installed, skipping", file=sys.stderr)
        return None

    print(f"\n--- Lightning Whisper MLX ({model_size}) ---")

    t0 = time.monotonic()
    whisper = LightningWhisperMLX(model=model_size, batch_size=12, quant=None)
    cold_start = time.monotonic() - t0
    print(f"  Cold start: {cold_start:.2f}s")

    references = []
    hypotheses = []
    total_audio_sec = 0.0
    total_proc_sec = 0.0

    for i, sample in enumerate(samples):
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        ref = sample["text"].lower().strip()

        audio_sec = len(audio) / sr
        total_audio_sec += audio_sec

        t0 = time.monotonic()
        result = whisper.transcribe(audio)
        proc_time = time.monotonic() - t0
        total_proc_sec += proc_time

        hyp = result.get("text", "").lower().strip()
        references.append(ref)
        hypotheses.append(hyp)

        if (i + 1) % 10 == 0:
            rtf = total_proc_sec / total_audio_sec
            print(f"  Processed {i + 1}/{len(samples)} (RTF: {rtf:.3f}x)")

    try:
        from jiwer import wer, cer
        wer_val = wer(references, hypotheses)
        cer_val = cer(references, hypotheses)
    except ImportError:
        wer_val = -1.0
        cer_val = -1.0

    rtf = total_proc_sec / total_audio_sec if total_audio_sec > 0 else float("inf")

    print(f"\n{'=' * 50}")
    print(f"Lightning Whisper MLX Results ({model_size})")
    print(f"{'=' * 50}")
    print(f"  RTF:           {rtf:.4f}x")
    print(f"  WER:           {wer_val * 100:.2f}%")

    return {
        "engine": "lightning_whisper_mlx",
        "model_size": model_size,
        "n_samples": len(references),
        "audio_seconds": total_audio_sec,
        "processing_seconds": total_proc_sec,
        "rtf": rtf,
        "wer": wer_val,
        "cer": cer_val,
        "cold_start_seconds": cold_start,
    }


def main():
    parser = argparse.ArgumentParser(description="MLX Whisper benchmark")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-mlx")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    samples = load_librispeech(args.max_samples)
    print(f"Loaded {len(samples)} test samples")

    results = {}

    # Standard mlx-whisper
    r = bench_mlx_whisper(samples, args.model)
    if r:
        results["mlx_whisper"] = r

    # Lightning whisper (if available)
    r = bench_lightning_whisper(samples)
    if r:
        results["lightning_whisper_mlx"] = r

    if args.output and results:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        # Save the first result as the primary
        primary = results.get("mlx_whisper") or list(results.values())[0]
        primary["all_results"] = results
        with open(args.output, "w") as f:
            json.dump(primary, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
