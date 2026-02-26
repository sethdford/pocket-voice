#!/usr/bin/env python3
"""Evaluate Sonata STT WER on LibriSpeech test sets.

Loads the exported .cstt_sonata model via ctypes and runs inference on
LibriSpeech test-clean/test-other audio, computing WER.

Usage:
  python scripts/eval_wer.py \
    --model models/sonata/sonata_stt.cstt_sonata \
    --test-dir train/data/LibriSpeech/test-clean \
    --max-samples 100
"""

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F


def load_sonata_stt(model_path: str, lib_dir: str = "build"):
    """Load the Sonata STT C library and create an engine."""
    lib_path = os.path.join(lib_dir, "libsonata_stt.dylib")
    mel_path = os.path.join(lib_dir, "libmel_spectrogram.dylib")

    ctypes.CDLL(mel_path)
    lib = ctypes.CDLL(lib_path)

    lib.sonata_stt_create.restype = ctypes.c_void_p
    lib.sonata_stt_create.argtypes = [ctypes.c_char_p]
    lib.sonata_stt_destroy.argtypes = [ctypes.c_void_p]
    lib.sonata_stt_reset.argtypes = [ctypes.c_void_p]
    lib.sonata_stt_process.restype = ctypes.c_int
    lib.sonata_stt_process.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int
    ]

    engine = lib.sonata_stt_create(model_path.encode())
    if not engine:
        raise RuntimeError(f"Failed to load model: {model_path}")

    return lib, engine


def transcribe(lib, engine, audio_24k: np.ndarray) -> str:
    """Run inference on 24kHz float32 audio."""
    buf = (ctypes.c_float * len(audio_24k))(*audio_24k)
    out = ctypes.create_string_buffer(4096)
    lib.sonata_stt_process(engine, buf, len(audio_24k), out, 4096)
    lib.sonata_stt_reset(engine)
    return out.value.decode("utf-8", errors="replace").strip()


def levenshtein(ref_words, hyp_words):
    """Word-level edit distance."""
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[n][m]


def normalize_text(text: str) -> str:
    """Normalize for WER: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = "".join(c if c.isalnum() or c == " " or c == "'" else " " for c in text)
    return " ".join(text.split())


def load_librispeech(test_dir: str, max_samples: int = 0):
    """Load LibriSpeech-format audio + transcripts."""
    pairs = []
    root = Path(test_dir)
    for trans_file in sorted(root.rglob("*.trans.txt")):
        for line in trans_file.read_text().strip().split("\n"):
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            uid, text = parts
            for ext in (".flac", ".wav"):
                audio_path = trans_file.parent / f"{uid}{ext}"
                if audio_path.exists():
                    pairs.append((str(audio_path), text.strip()))
                    break
        if max_samples and len(pairs) >= max_samples:
            break
    if max_samples:
        pairs = pairs[:max_samples]
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Sonata STT WER Evaluation")
    parser.add_argument("--model", required=True, help="Path to .cstt_sonata model")
    parser.add_argument("--test-dir", required=True, help="LibriSpeech test directory")
    parser.add_argument("--lib-dir", default="build", help="Directory with .dylib files")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples (0=all)")
    parser.add_argument("--target-sr", type=int, default=24000, help="Model sample rate")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SONATA STT — WER EVALUATION")
    print(f"{'='*60}")

    lib, engine = load_sonata_stt(args.model, args.lib_dir)
    pairs = load_librispeech(args.test_dir, args.max_samples)
    print(f"  Model: {args.model}")
    print(f"  Test set: {args.test_dir} ({len(pairs)} utterances)")
    print()

    total_words = 0
    total_errors = 0
    total_chars = 0
    total_char_errors = 0
    total_time = 0.0
    total_audio_sec = 0.0
    n_empty = 0

    for i, (audio_path, ref_text) in enumerate(pairs):
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        audio_sec = len(data) / sr
        total_audio_sec += audio_sec

        if sr != args.target_sr:
            ratio = args.target_sr / sr
            new_len = int(len(data) * ratio)
            tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
            resampled = F.interpolate(tensor, size=new_len, mode="linear", align_corners=False)
            data = resampled.squeeze().numpy()

        t0 = time.time()
        hyp_text = transcribe(lib, engine, data)
        elapsed = time.time() - t0
        total_time += elapsed

        ref_norm = normalize_text(ref_text)
        hyp_norm = normalize_text(hyp_text)

        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()

        if not hyp_words:
            n_empty += 1

        word_errors = levenshtein(ref_words, hyp_words)
        total_words += len(ref_words)
        total_errors += word_errors

        ref_chars = list(ref_norm.replace(" ", ""))
        hyp_chars = list(hyp_norm.replace(" ", ""))
        char_errors = levenshtein(ref_chars, hyp_chars)
        total_chars += len(ref_chars)
        total_char_errors += char_errors

        if args.verbose or (i < 5):
            wer_i = word_errors / max(1, len(ref_words)) * 100
            print(f"  [{i+1:4d}] WER={wer_i:5.1f}% | REF: {ref_norm[:60]}")
            print(f"         {' '*6}  | HYP: {hyp_norm[:60]}")

    wer = total_errors / max(1, total_words) * 100
    cer = total_char_errors / max(1, total_chars) * 100
    rtf = total_time / max(0.001, total_audio_sec)

    print(f"\n{'='*60}")
    print(f"  RESULTS ({len(pairs)} utterances, {total_audio_sec:.0f}s audio)")
    print(f"{'='*60}")
    print(f"  WER:  {wer:.1f}% ({total_errors} errors / {total_words} words)")
    print(f"  CER:  {cer:.1f}% ({total_char_errors} errors / {total_chars} chars)")
    print(f"  RTF:  {rtf:.4f} ({1/max(0.001,rtf):.0f}x realtime)")
    print(f"  Empty: {n_empty}/{len(pairs)} ({n_empty/max(1,len(pairs))*100:.0f}%)")
    print(f"  Latency: {total_time/max(1,len(pairs))*1000:.0f}ms avg per utterance")
    print(f"{'='*60}\n")

    lib.sonata_stt_destroy(engine)


if __name__ == "__main__":
    main()
