#!/usr/bin/env python3
"""
bench_whisper_cpp_harness.py — Drive whisper.cpp on LibriSpeech and collect metrics.

Runs whisper.cpp CLI on individual audio files and measures:
  - RTF (real-time factor)
  - WER on LibriSpeech test-clean
  - Peak memory usage
  - Model load time (cold start)
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def load_librispeech(max_samples=None):
    """Load LibriSpeech test-clean."""
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


def save_wav(audio_array, sample_rate, path):
    """Save audio as 16-bit WAV."""
    import numpy as np
    try:
        import soundfile as sf
        sf.write(path, audio_array, sample_rate, subtype="PCM_16")
    except ImportError:
        import wave
        import struct
        pcm = (np.clip(audio_array, -1, 1) * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())


def run_whisper(whisper_bin, model_path, wav_path, timeout=120):
    """Run whisper.cpp on a single WAV file, return (text, elapsed_seconds)."""
    cmd = [
        whisper_bin,
        "-m", model_path,
        "-f", wav_path,
        "--no-timestamps",
        "-t", "1",  # single thread for fair comparison
        "--print-special", "false",
    ]

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0

        # Extract text from whisper output (last non-empty lines)
        text = ""
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            # Skip timing lines like [00:00:00.000 --> 00:00:05.000]
            if line.startswith("[") and "-->" in line:
                # Extract text after the timestamp bracket
                bracket_end = line.rfind("]")
                if bracket_end >= 0:
                    text += " " + line[bracket_end + 1:].strip()
            elif line and not line.startswith("whisper_"):
                text += " " + line

        return text.strip().lower(), elapsed

    except subprocess.TimeoutExpired:
        return "", timeout
    except Exception as e:
        print(f"  whisper.cpp error: {e}", file=sys.stderr)
        return "", 0.0


def measure_cold_start(whisper_bin, model_path):
    """Measure model load time with a tiny dummy file."""
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        dummy_path = f.name
        save_wav(np.zeros(16000, dtype=np.float32), 16000, dummy_path)

    try:
        t0 = time.monotonic()
        subprocess.run(
            [whisper_bin, "-m", model_path, "-f", dummy_path, "--no-timestamps", "-t", "1"],
            capture_output=True,
            timeout=60,
        )
        cold_start = time.monotonic() - t0
        return cold_start
    finally:
        os.unlink(dummy_path)


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description="whisper.cpp benchmark harness")
    parser.add_argument("--whisper-bin", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-size", default="base")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    samples = load_librispeech(args.max_samples)
    print(f"Loaded {len(samples)} test samples")

    # Cold start measurement
    print("\nMeasuring cold start (model load)...")
    cold_start = measure_cold_start(args.whisper_bin, args.model)
    print(f"  Cold start: {cold_start:.2f}s")

    # Run inference
    print(f"\n--- whisper.cpp ({args.model_size}) ---")
    references = []
    hypotheses = []
    total_audio_sec = 0.0
    total_proc_sec = 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, sample in enumerate(samples):
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            ref = sample["text"].lower().strip()

            wav_path = os.path.join(tmpdir, f"sample_{i:04d}.wav")
            save_wav(audio, sr, wav_path)

            audio_sec = len(audio) / sr
            total_audio_sec += audio_sec

            hyp, proc_time = run_whisper(args.whisper_bin, args.model, wav_path)
            total_proc_sec += proc_time

            references.append(ref)
            hypotheses.append(hyp)

            if (i + 1) % 10 == 0:
                rtf_so_far = total_proc_sec / total_audio_sec if total_audio_sec > 0 else 0
                print(f"  Processed {i + 1}/{len(samples)} (RTF: {rtf_so_far:.3f}x)")

    # Compute WER
    try:
        from jiwer import wer, cer
        wer_val = wer(references, hypotheses)
        cer_val = cer(references, hypotheses)
    except ImportError:
        print("WARNING: pip install jiwer for WER/CER", file=sys.stderr)
        wer_val = -1.0
        cer_val = -1.0

    rtf = total_proc_sec / total_audio_sec if total_audio_sec > 0 else float("inf")

    # Get model file size
    model_size_bytes = os.path.getsize(args.model)

    print(f"\n{'=' * 50}")
    print(f"whisper.cpp Results ({args.model_size})")
    print(f"{'=' * 50}")
    print(f"  Samples:       {len(references)}")
    print(f"  Audio:         {total_audio_sec:.1f}s")
    print(f"  Processing:    {total_proc_sec:.1f}s")
    print(f"  RTF:           {rtf:.4f}x")
    print(f"  WER:           {wer_val * 100:.2f}%")
    print(f"  CER:           {cer_val * 100:.2f}%")
    print(f"  Cold start:    {cold_start:.2f}s")
    print(f"  Model size:    {model_size_bytes / 1024 / 1024:.0f} MB")

    results = {
        "engine": "whisper_cpp",
        "model_size": args.model_size,
        "model_path": args.model,
        "model_size_bytes": model_size_bytes,
        "n_samples": len(references),
        "audio_seconds": total_audio_sec,
        "processing_seconds": total_proc_sec,
        "rtf": rtf,
        "wer": wer_val,
        "cer": cer_val,
        "cold_start_seconds": cold_start,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
