#!/usr/bin/env python3
"""
diagnose_stt.py — Pinpoint where the C Conformer STT diverges from NeMo reference.

Compares intermediate outputs (mel features, subsampled output, logits, decode)
between the C engine and NeMo Python reference to identify the WER=1.0 bug.

Usage:
    python scripts/diagnose_stt.py --model models/parakeet-ctc-0.6b.cstt
"""

import argparse
import ctypes
import struct
import sys
import os
from pathlib import Path

import numpy as np

def read_cstt_header(model_path):
    """Read and display the .cstt file header."""
    with open(model_path, "rb") as f:
        raw = f.read(96)
    fields = struct.unpack("<" + "I" * 24, raw)
    names = [
        "magic", "version", "n_layers", "d_model", "n_heads", "ff_mult",
        "conv_kernel", "vocab_size", "n_mels", "sample_rate", "hop_length",
        "win_length", "n_fft", "subsample_factor", "dtype", "flags",
        "sub_type", "n_sub_convs", "sub_feat_in", "sub_conv_kernel",
        "reserved0", "reserved1", "reserved2", "reserved3"
    ]
    header = dict(zip(names, fields))
    print("=== .cstt Header ===")
    for k, v in header.items():
        extra = ""
        if k == "magic":
            extra = f"  ({v:#010x}, expected 0x54545343)"
        if k == "dtype":
            extra = f"  ({'fp16' if v == 1 else 'fp32'})"
        if k == "flags":
            flag_names = []
            if v & 1: flag_names.append("HAS_BIAS")
            if v & 2: flag_names.append("SLANEY_NORM")
            if v & 4: flag_names.append("REL_PE")
            if v & 8: flag_names.append("HAS_EOU")
            if v & 16: flag_names.append("CACHE_AWARE")
            if v & 32: flag_names.append("XSCALING")
            if v & 64: flag_names.append("TDT")
            extra = f"  ({' | '.join(flag_names)})"
        if k == "sub_type":
            st = {0: "CONV1D", 1: "CONV2D", 2: "DW_STRIDING"}.get(v, f"UNKNOWN({v})")
            extra = f"  ({st})"
        print(f"  {k:20s} = {v}{extra}")
    return header


def read_vocab(vocab_path):
    """Read vocabulary file."""
    tokens = []
    with open(vocab_path) as f:
        for line in f:
            tokens.append(line.rstrip('\n'))
    return tokens


def load_c_engine(model_path):
    """Load the C STT engine via ctypes."""
    lib_path = Path("build/libconformer_stt.dylib")
    if not lib_path.exists():
        print(f"ERROR: {lib_path} not found. Run 'make libs' first.", file=sys.stderr)
        sys.exit(1)

    lib = ctypes.CDLL(str(lib_path))
    lib.conformer_stt_create.restype = ctypes.c_void_p
    lib.conformer_stt_create.argtypes = [ctypes.c_char_p]
    lib.conformer_stt_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.conformer_stt_process.restype = ctypes.c_int
    lib.conformer_stt_flush.argtypes = [ctypes.c_void_p]
    lib.conformer_stt_flush.restype = ctypes.c_int
    lib.conformer_stt_get_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.conformer_stt_get_text.restype = ctypes.c_int
    lib.conformer_stt_reset.argtypes = [ctypes.c_void_p]
    lib.conformer_stt_destroy.argtypes = [ctypes.c_void_p]

    engine = lib.conformer_stt_create(model_path.encode())
    if not engine:
        print(f"ERROR: Failed to create STT engine from {model_path}", file=sys.stderr)
        sys.exit(1)

    return lib, engine


def test_with_sine_wave(lib, engine, header):
    """Test with a known sine wave to check basic functionality."""
    print("\n=== Test 1: Sine wave (440Hz, 1 second) ===")
    sr = header["sample_rate"]
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    lib.conformer_stt_reset(engine)
    n_chars = lib.conformer_stt_process(engine, audio.ctypes.data, len(audio))
    lib.conformer_stt_flush(engine)

    buf = ctypes.create_string_buffer(4096)
    lib.conformer_stt_get_text(engine, buf, 4096)
    text = buf.value.decode("utf-8", errors="replace")
    print(f"  Output ({n_chars} chars): '{text}'")
    if not text.strip():
        print("  → Model outputs nothing for sine wave (expected: blank/silence)")
    else:
        print(f"  → Model outputs text for sine wave (may indicate issues)")
    return text


def test_with_speech(lib, engine, header):
    """Test with actual speech audio."""
    print("\n=== Test 2: Speech audio ===")

    try:
        from datasets import load_dataset
        ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
        sample = next(iter(ds))
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        ref = sample["text"].lower().strip()
        print(f"  Reference: '{ref}'")
        print(f"  Audio: {len(audio)} samples @ {sr}Hz ({len(audio)/sr:.1f}s)")
    except Exception as e:
        print(f"  Could not load LibriSpeech: {e}")
        print("  Generating synthetic speech-like noise instead...")
        sr = header["sample_rate"]
        audio = np.random.randn(sr * 3).astype(np.float32) * 0.1
        ref = "(synthetic noise)"

    target_sr = header["sample_rate"]
    if sr != target_sr:
        ratio = target_sr / sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        print(f"  Resampled to {target_sr}Hz ({len(audio)} samples)")

    lib.conformer_stt_reset(engine)
    n_chars = lib.conformer_stt_process(engine, audio.ctypes.data, len(audio))
    n_chars2 = lib.conformer_stt_flush(engine)

    buf = ctypes.create_string_buffer(4096)
    lib.conformer_stt_get_text(engine, buf, 4096)
    text = buf.value.decode("utf-8", errors="replace")
    print(f"  Hypothesis ({n_chars}+{n_chars2} chars): '{text}'")

    if not text.strip():
        print("  → PROBLEM: Model outputs NOTHING for speech audio!")
        print("    Possible causes: mel features are all-zero, subsampling produces 0 frames,")
        print("    or all logits point to <blank>.")
    else:
        print(f"  → Model produces output. Checking quality...")
        try:
            import jiwer
            transforms = jiwer.Compose([
                jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                jiwer.Strip(), jiwer.RemoveMultipleSpaces(),
            ])
            r = transforms(ref) or "empty"
            h = transforms(text) or "empty"
            wer = jiwer.wer(r, h)
            print(f"  WER: {wer*100:.1f}%")
            if wer > 0.5:
                print("  → HIGH WER: output is largely wrong")
        except ImportError:
            pass

    return text, ref


def compare_mel_features(header, model_path):
    """Compare C mel features against librosa/NeMo reference."""
    print("\n=== Test 3: Mel feature comparison (C vs Python) ===")

    sr = header["sample_rate"]
    n_fft = header["n_fft"]
    hop = header["hop_length"]
    win_len = header["win_length"]
    n_mels = header["n_mels"]

    t = np.linspace(0, 0.5, sr // 2, dtype=np.float32)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t) +
             0.2 * np.sin(2 * np.pi * 880 * t) +
             0.1 * np.sin(2 * np.pi * 1320 * t)).astype(np.float32)

    try:
        import librosa
    except ImportError:
        print("  librosa not available, skipping mel comparison")
        return

    preemph_audio = np.zeros_like(audio)
    preemph_audio[0] = audio[0]
    for i in range(1, len(audio)):
        preemph_audio[i] = audio[i] - 0.97 * audio[i - 1]

    py_mel = librosa.feature.melspectrogram(
        y=preemph_audio, sr=sr, n_fft=n_fft, hop_length=hop,
        win_length=win_len, n_mels=n_mels, fmin=0.0, fmax=sr/2,
        center=True, window='hann', power=2.0, norm='slaney', htk=False
    )
    py_log_mel = np.log(py_mel.T + 5.96046448e-8)  # [T, n_mels]

    T = py_log_mel.shape[0]
    mean = py_log_mel.mean(axis=0)
    std = py_log_mel.std(axis=0)
    std[std < 1e-5] = 1e-5
    py_norm = (py_log_mel - mean) / std

    print(f"  Python mel: shape={py_log_mel.shape}, range=[{py_log_mel.min():.2f}, {py_log_mel.max():.2f}]")
    print(f"  Python normalized: range=[{py_norm.min():.2f}, {py_norm.max():.2f}]")
    print(f"  Mean per-feature mean: {mean.mean():.4f}")
    print(f"  Mean per-feature std:  {std.mean():.4f}")

    mel_lib = ctypes.CDLL("build/libmel_spectrogram.dylib")

    class MelConfig(ctypes.Structure):
        _fields_ = [
            ("sample_rate", ctypes.c_int),
            ("n_fft", ctypes.c_int),
            ("hop_length", ctypes.c_int),
            ("win_length", ctypes.c_int),
            ("n_mels", ctypes.c_int),
            ("fmin", ctypes.c_float),
            ("fmax", ctypes.c_float),
            ("log_floor", ctypes.c_float),
            ("preemph", ctypes.c_float),
        ]

    mel_lib.mel_create.restype = ctypes.c_void_p
    mel_lib.mel_create.argtypes = [ctypes.POINTER(MelConfig)]
    mel_lib.mel_process.restype = ctypes.c_int
    mel_lib.mel_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                     ctypes.c_void_p, ctypes.c_int]
    mel_lib.mel_destroy.argtypes = [ctypes.c_void_p]
    mel_lib.mel_reset.argtypes = [ctypes.c_void_p]

    cfg = MelConfig()
    cfg.sample_rate = sr
    cfg.n_fft = n_fft
    cfg.hop_length = hop
    cfg.win_length = win_len
    cfg.n_mels = n_mels
    cfg.fmin = 0.0
    cfg.fmax = 0.0
    cfg.log_floor = 5.96046448e-8
    cfg.preemph = 0.97

    mel_handle = mel_lib.mel_create(ctypes.byref(cfg))
    if not mel_handle:
        print("  ERROR: Failed to create C mel extractor")
        return

    max_frames = len(audio) // hop + 10
    c_out = np.zeros(max_frames * n_mels, dtype=np.float32)
    n_frames = mel_lib.mel_process(mel_handle, audio.ctypes.data, len(audio),
                                    c_out.ctypes.data, max_frames)
    mel_lib.mel_destroy(mel_handle)

    if n_frames <= 0:
        print(f"  ERROR: C mel returned {n_frames} frames")
        return

    c_mel = c_out[:n_frames * n_mels].reshape(n_frames, n_mels)
    print(f"  C mel: shape=({n_frames}, {n_mels}), range=[{c_mel.min():.2f}, {c_mel.max():.2f}]")

    min_frames = min(T, n_frames)
    if min_frames > 0:
        diff = np.abs(c_mel[:min_frames] - py_log_mel[:min_frames])
        print(f"  Frame diff (first {min_frames} frames): mean={diff.mean():.4f}, max={diff.max():.4f}")
        if diff.mean() > 1.0:
            print("  → LARGE MEL DIFFERENCE: C and Python mel features diverge significantly!")
            print("    This is likely a major contributor to WER=1.0")
            print(f"  C frame 0[:5]:  {c_mel[0, :5]}")
            print(f"  Py frame 0[:5]: {py_log_mel[0, :5]}")
        else:
            print("  → Mel features are similar (good)")


def check_vocab(vocab_path, header):
    """Check vocabulary for common issues."""
    print(f"\n=== Test 4: Vocabulary check ({vocab_path}) ===")
    if not os.path.exists(vocab_path):
        print(f"  ERROR: Vocab file not found: {vocab_path}")
        return

    tokens = read_vocab(vocab_path)
    print(f"  Vocab size: {len(tokens)} (header says {header['vocab_size']})")
    if len(tokens) != header['vocab_size']:
        print("  → MISMATCH: vocab file size != header vocab_size!")

    blank_idx = -1
    for i, t in enumerate(tokens):
        if t in ("<blank>", "<blk>"):
            blank_idx = i
            break

    print(f"  Blank token: index {blank_idx} ('{tokens[blank_idx] if blank_idx >= 0 else 'NOT FOUND'}')")

    if blank_idx == 0:
        print("  → Blank at index 0 (typical for standard CTC)")
    elif blank_idx == len(tokens) - 1:
        print("  → Blank at last index (NeMo convention)")
    elif blank_idx >= 0:
        print(f"  → Blank at unusual index {blank_idx}")
    else:
        print("  → WARNING: No <blank> token found in vocabulary!")

    sp_chars = [i for i, t in enumerate(tokens) if t.startswith("▁") or t.startswith(" ")]
    print(f"  SentencePiece-style tokens (▁/space prefix): {len(sp_chars)}")

    print(f"  First 10 tokens: {tokens[:10]}")
    print(f"  Last 5 tokens: {tokens[-5:]}")

    has_alpha = sum(1 for t in tokens if any(c.isalpha() for c in t))
    print(f"  Tokens containing alphabetic chars: {has_alpha}")

    if has_alpha < 26:
        print("  → WARNING: Very few alphabetic tokens — vocab may be corrupted")


def check_logits_distribution(lib, engine, header):
    """Run a short audio through the engine and analyze the CTC output distribution."""
    print("\n=== Test 5: Logits distribution analysis ===")

    sr = header["sample_rate"]
    audio = np.random.randn(sr * 2).astype(np.float32) * 0.01  # quiet noise

    lib.conformer_stt_reset(engine)
    lib.conformer_stt_process(engine, audio.ctypes.data, len(audio))
    lib.conformer_stt_flush(engine)

    buf = ctypes.create_string_buffer(4096)
    lib.conformer_stt_get_text(engine, buf, 4096)
    text = buf.value.decode("utf-8", errors="replace")

    if text.strip():
        print(f"  For near-silence, model outputs: '{text[:200]}'")
        print("  → Model should output blank/nothing for silence. If it outputs text,")
        print("    the logits may be corrupted (pointing to wrong tokens).")
    else:
        print("  For near-silence, model outputs empty (correct)")


def main():
    parser = argparse.ArgumentParser(description="Diagnose STT issues")
    parser.add_argument("--model", type=str, default="models/parakeet-ctc-0.6b.cstt")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    header = read_cstt_header(args.model)

    vocab_path = args.model.rsplit(".", 1)[0] + ".vocab"
    check_vocab(vocab_path, header)

    print("\n--- Loading C STT engine ---")
    lib, engine = load_c_engine(args.model)

    test_with_sine_wave(lib, engine, header)
    check_logits_distribution(lib, engine, header)
    test_with_speech(lib, engine, header)
    compare_mel_features(header, args.model)

    lib.conformer_stt_destroy(engine)
    print("\n=== Diagnostics complete ===")


if __name__ == "__main__":
    main()
