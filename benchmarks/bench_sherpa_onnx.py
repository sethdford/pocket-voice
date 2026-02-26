#!/usr/bin/env python3
"""
bench_sherpa_onnx.py — Benchmark sherpa-onnx STT on LibriSpeech test-clean.

Requires: pip install sherpa-onnx

Supports both streaming and offline modes.
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


def find_model(model_dir=None):
    """Find a sherpa-onnx model directory."""
    search_paths = [
        model_dir,
        "models/sherpa-onnx",
        Path.home() / "sherpa-onnx-models",
        Path.home() / ".cache/sherpa-onnx",
    ]

    for p in search_paths:
        if p and Path(p).exists():
            # Look for encoder/decoder files
            p = Path(p)
            if list(p.glob("*encoder*")) or list(p.glob("*.onnx")):
                return p
            for sub in p.iterdir():
                if sub.is_dir() and (list(sub.glob("*encoder*")) or list(sub.glob("*.onnx"))):
                    return sub

    return None


def bench_offline(samples, model_path):
    """Benchmark sherpa-onnx offline recognizer."""
    try:
        import sherpa_onnx
    except ImportError:
        print("ERROR: pip install sherpa-onnx", file=sys.stderr)
        return None

    print(f"\n--- sherpa-onnx offline ---")
    print(f"  Model: {model_path}")

    # Auto-detect model type
    model_dir = Path(model_path)
    encoder = list(model_dir.glob("*encoder*.onnx"))
    decoder = list(model_dir.glob("*decoder*.onnx"))
    joiner = list(model_dir.glob("*joiner*.onnx"))
    tokens = list(model_dir.glob("tokens.txt"))

    if not tokens:
        print("  ERROR: tokens.txt not found in model directory")
        return None

    config = None

    if encoder and decoder and joiner:
        # Transducer model
        config = sherpa_onnx.OfflineRecognizerConfig(
            model_config=sherpa_onnx.OfflineModelConfig(
                transducer=sherpa_onnx.OfflineTransducerModelConfig(
                    encoder_filename=str(encoder[0]),
                    decoder_filename=str(decoder[0]),
                    joiner_filename=str(joiner[0]),
                ),
                tokens=str(tokens[0]),
                num_threads=1,
            ),
        )
    else:
        # Try paraformer or other model types
        paraformer = list(model_dir.glob("*paraformer*.onnx")) or list(model_dir.glob("model.onnx"))
        if paraformer:
            config = sherpa_onnx.OfflineRecognizerConfig(
                model_config=sherpa_onnx.OfflineModelConfig(
                    paraformer=sherpa_onnx.OfflineParaformerModelConfig(
                        model=str(paraformer[0]),
                    ),
                    tokens=str(tokens[0]),
                    num_threads=1,
                ),
            )

    if config is None:
        print("  ERROR: Could not detect model type")
        return None

    t0 = time.monotonic()
    recognizer = sherpa_onnx.OfflineRecognizer(config)
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

        # Resample to 16kHz if needed
        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        audio_sec = len(audio) / 16000
        total_audio_sec += audio_sec

        t0 = time.monotonic()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        recognizer.decode_stream(stream)
        proc_time = time.monotonic() - t0
        total_proc_sec += proc_time

        hyp = stream.result.text.lower().strip()
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
    print(f"sherpa-onnx Results (offline)")
    print(f"{'=' * 50}")
    print(f"  Samples:       {len(references)}")
    print(f"  Audio:         {total_audio_sec:.1f}s")
    print(f"  Processing:    {total_proc_sec:.1f}s")
    print(f"  RTF:           {rtf:.4f}x")
    print(f"  WER:           {wer_val * 100:.2f}%")
    print(f"  Cold start:    {cold_start:.2f}s")

    return {
        "engine": "sherpa_onnx",
        "mode": "offline",
        "model": str(model_path),
        "n_samples": len(references),
        "audio_seconds": total_audio_sec,
        "processing_seconds": total_proc_sec,
        "rtf": rtf,
        "wer": wer_val,
        "cer": cer_val,
        "cold_start_seconds": cold_start,
    }


def main():
    parser = argparse.ArgumentParser(description="sherpa-onnx benchmark")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-dir", default=None, help="Path to sherpa-onnx model")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model_path = find_model(args.model_dir)
    if not model_path:
        print("ERROR: No sherpa-onnx model found.", file=sys.stderr)
        print("Download one from: https://github.com/k2-fsa/sherpa-onnx/releases", file=sys.stderr)
        sys.exit(1)

    samples = load_librispeech(args.max_samples)
    print(f"Loaded {len(samples)} test samples")

    results = bench_offline(samples, model_path)

    if results and args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
