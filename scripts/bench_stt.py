#!/usr/bin/env python3
"""
bench_stt.py — STT benchmarking: measure WER/CER/RTF of the C Conformer engine.

Downloads LibriSpeech test-clean, runs inference through the C STT engine,
and computes standard ASR metrics against reference transcriptions.

Usage:
    # Convert model first:
    python scripts/convert_nemo.py nvidia/parakeet-ctc-0.6b -o models/parakeet-ctc.cstt

    # Run benchmark:
    python scripts/bench_stt.py --model models/parakeet-ctc.cstt

    # Compare C vs Rust engines:
    python scripts/bench_stt.py --model models/parakeet-ctc.cstt --compare-rust

Dependencies:
    pip install jiwer datasets soundfile numpy
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


def load_librispeech_test_clean(max_samples=None):
    """Load LibriSpeech test-clean samples using the datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print("Loading LibriSpeech test-clean...")
    ds = load_dataset("librispeech_asr", "clean", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def compute_wer(references, hypotheses):
    """Compute Word Error Rate with standard ASR normalization."""
    try:
        import jiwer
        transforms = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.Strip(),
            jiwer.RemoveMultipleSpaces(),
        ])
        safe_refs = [transforms(r) if transforms(r).strip() else "empty"
                     for r in references]
        safe_hyps = [transforms(h) if transforms(h).strip() else "empty"
                     for h in hypotheses]
        w = jiwer.wer(safe_refs, safe_hyps)
        c = jiwer.cer(safe_refs, safe_hyps)
        return w, c
    except ImportError:
        print("WARNING: pip install jiwer for WER/CER computation", file=sys.stderr)
        return -1.0, -1.0


def run_c_stt_benchmark(model_path, samples, sample_rate=16000,
                        beam_size=0, lm_path=None, lm_weight=1.5, word_score=0.0,
                        streaming=False, chunk_frames=0):
    """Run the C Conformer STT engine on audio samples."""
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
    lib.conformer_stt_enable_beam_search.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_float, ctypes.c_float]
    lib.conformer_stt_enable_beam_search.restype = ctypes.c_int
    lib.conformer_stt_set_cache_aware.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.conformer_stt_set_chunk_frames.argtypes = [ctypes.c_void_p, ctypes.c_int]

    engine = lib.conformer_stt_create(model_path.encode())
    if not engine:
        print(f"ERROR: Failed to create STT engine from {model_path}", file=sys.stderr)
        sys.exit(1)

    if streaming:
        lib.conformer_stt_set_cache_aware(engine, 1)
        if chunk_frames > 0:
            lib.conformer_stt_set_chunk_frames(engine, chunk_frames)

    if beam_size > 0:
        lm_bytes = lm_path.encode() if lm_path else None
        rc = lib.conformer_stt_enable_beam_search(
            engine, lm_bytes, beam_size,
            ctypes.c_float(lm_weight), ctypes.c_float(word_score))
        if rc != 0:
            print("WARNING: Failed to enable beam search", file=sys.stderr)

    references = []
    hypotheses = []
    total_audio_sec = 0.0
    total_proc_sec = 0.0

    for i, sample in enumerate(samples):
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        ref = sample["text"].lower().strip()

        if sr != sample_rate:
            ratio = sample_rate / sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        audio_sec = len(audio) / sample_rate
        total_audio_sec += audio_sec

        lib.conformer_stt_reset(engine)

        t0 = time.monotonic()
        lib.conformer_stt_process(engine, audio.ctypes.data, len(audio))
        lib.conformer_stt_flush(engine)
        proc_time = time.monotonic() - t0
        total_proc_sec += proc_time

        buf = ctypes.create_string_buffer(4096)
        lib.conformer_stt_get_text(engine, buf, 4096)
        hyp = buf.value.decode("utf-8", errors="replace").lower().strip()

        references.append(ref)
        hypotheses.append(hyp)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples "
                  f"(RTF: {total_proc_sec / total_audio_sec:.3f}x)")

    lib.conformer_stt_destroy(engine)

    # Show worst samples for error analysis
    import jiwer
    transforms = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
    ])
    errors = []
    for i, (r, h) in enumerate(zip(references, hypotheses)):
        r_n = transforms(r) or "empty"
        h_n = transforms(h) or "empty"
        sample_wer = jiwer.wer(r_n, h_n)
        if sample_wer > 0:
            errors.append((sample_wer, i, r_n, h_n))
    errors.sort(reverse=True)
    print(f"\n  Top 20 errors:")
    for w, i, r, h in errors[:20]:
        print(f"    [{i:3d}] WER={w*100:5.1f}%  REF: {r[:80]}")
        print(f"          HYP: {h[:80]}")

    return references, hypotheses, total_audio_sec, total_proc_sec


def main():
    parser = argparse.ArgumentParser(description="STT Benchmark")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .cstt model file")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of test samples")
    parser.add_argument("--beam-size", type=int, default=0,
                        help="Beam width for CTC beam search (0 = greedy)")
    parser.add_argument("--lm-path", type=str, default=None,
                        help="Path to KenLM .bin language model")
    parser.add_argument("--lm-weight", type=float, default=1.5,
                        help="LM weight for beam search (default: 1.5)")
    parser.add_argument("--word-score", type=float, default=0.0,
                        help="Per-word insertion bonus for beam search")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON results file")
    parser.add_argument("--streaming", action="store_true",
                        help="Enable cache-aware streaming mode")
    parser.add_argument("--chunk-frames", type=int, default=0,
                        help="Mel frames per encoder chunk (0 = default)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    samples = load_librispeech_test_clean(args.max_samples)
    print(f"Loaded {len(samples)} test samples")

    # Detect TDT from model header flags
    is_tdt = False
    with open(args.model, "rb") as f:
        hdr = f.read(96)
        if len(hdr) >= 64:
            flags = struct.unpack_from("<I", hdr, 60)[0]
            is_tdt = bool(flags & (1 << 6))

    if is_tdt:
        decode_mode = "TDT transducer (greedy)"
    else:
        decode_mode = "greedy"
        if args.beam_size > 0:
            decode_mode = f"beam={args.beam_size}"
            if args.lm_path:
                decode_mode += f" + LM (w={args.lm_weight})"

    if args.streaming:
        decode_mode += " + streaming"
        if args.chunk_frames > 0:
            decode_mode += f" (chunk={args.chunk_frames})"

    print(f"\n--- C Conformer STT ({decode_mode}) ---")
    refs, hyps, audio_sec, proc_sec = run_c_stt_benchmark(
        args.model, samples,
        beam_size=args.beam_size,
        lm_path=args.lm_path,
        lm_weight=args.lm_weight,
        word_score=args.word_score,
        streaming=args.streaming,
        chunk_frames=args.chunk_frames)

    wer_val, cer_val = compute_wer(refs, hyps)
    rtf = proc_sec / audio_sec if audio_sec > 0 else float("inf")

    print(f"\n{'=' * 50}")
    print(f"STT Benchmark Results (C Conformer)")
    print(f"{'=' * 50}")
    print(f"  Samples:       {len(refs)}")
    print(f"  Audio:         {audio_sec:.1f}s")
    print(f"  Processing:    {proc_sec:.1f}s")
    print(f"  RTF:           {rtf:.4f}x")
    print(f"  WER:           {wer_val * 100:.2f}%")
    print(f"  CER:           {cer_val * 100:.2f}%")

    results = {
        "engine": "c_conformer",
        "model": args.model,
        "n_samples": len(refs),
        "audio_seconds": audio_sec,
        "processing_seconds": proc_sec,
        "rtf": rtf,
        "wer": wer_val,
        "cer": cer_val,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
