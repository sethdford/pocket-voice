#!/usr/bin/env python3
"""
End-to-end latency benchmark: STT → TTS pipeline timing.

Measures:
  - STT latency: audio → transcript
  - TTS TTFS: text → first audio sample
  - TTS total: text → all audio
  - Pipeline total: audio → first TTS audio output
"""

import ctypes
import json
import os
import sys
import time

def load_lib(name):
    paths = [
        f"build/{name}",
        f"build/lib{name}.dylib",
    ]
    for p in paths:
        if os.path.exists(p):
            return ctypes.CDLL(p)
    raise FileNotFoundError(f"Cannot find {name}")

def main():
    stt_lib = load_lib("conformer_stt")
    tts_lib = load_lib("kyutai_dsm_tts")

    # STT functions
    stt_create = stt_lib.conformer_stt_create
    stt_create.restype = ctypes.c_void_p
    stt_create.argtypes = [ctypes.c_char_p]

    stt_process = stt_lib.conformer_stt_process
    stt_process.restype = ctypes.c_int
    stt_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    stt_flush = stt_lib.conformer_stt_flush
    stt_flush.restype = ctypes.c_int
    stt_flush.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]

    stt_reset = stt_lib.conformer_stt_reset
    stt_reset.argtypes = [ctypes.c_void_p]

    # TTS functions
    tts_create = tts_lib.kyutai_tts_create
    tts_create.restype = ctypes.c_void_p
    tts_create.argtypes = [ctypes.c_char_p]

    tts_set_text = tts_lib.kyutai_tts_set_text
    tts_set_text.restype = ctypes.c_int
    tts_set_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    tts_step = tts_lib.kyutai_tts_step
    tts_step.restype = ctypes.c_int
    tts_step.argtypes = [ctypes.c_void_p]

    tts_get_audio = tts_lib.kyutai_tts_get_audio
    tts_get_audio.restype = ctypes.c_int
    tts_get_audio.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    tts_is_done = tts_lib.kyutai_tts_is_done
    tts_is_done.restype = ctypes.c_int
    tts_is_done.argtypes = [ctypes.c_void_p]

    tts_reset = tts_lib.kyutai_tts_reset
    tts_reset.argtypes = [ctypes.c_void_p]

    # Load models
    stt_model = "models/parakeet-ctc-0.6b-fp16.cstt"
    tts_model = "models/kyutai_dsm.ctts"

    if not os.path.exists(stt_model):
        stt_model = "models/parakeet-ctc-0.6b-fp32.cstt"

    print(f"Loading STT: {stt_model}")
    stt = stt_create(stt_model.encode())
    if not stt:
        print("ERROR: Failed to create STT engine")
        return

    print(f"Loading TTS: {tts_model}")
    tts = tts_create(tts_model.encode())
    if not tts:
        print("ERROR: Failed to create TTS engine")
        return

    # Generate test audio (3.5s of speech-like signal at 16kHz)
    import math
    sr = 16000
    duration = 3.5
    n_samples = int(sr * duration)
    audio = (ctypes.c_float * n_samples)()
    for i in range(n_samples):
        t = i / sr
        audio[i] = 0.3 * math.sin(2 * math.pi * 200 * t + math.sin(2 * math.pi * 5 * t) * 3)

    print(f"\n{'='*60}")
    print(f"E2E Latency Benchmark")
    print(f"{'='*60}")
    print(f"  Test audio: {duration}s at {sr}Hz ({n_samples} samples)")

    results = []
    n_runs = 5

    for run in range(n_runs):
        stt_reset(stt)

        # STT: audio → transcript
        t0 = time.perf_counter()
        stt_process(stt, audio, n_samples)
        transcript_buf = ctypes.create_string_buffer(4096)
        stt_flush(stt, transcript_buf, 4096)
        t_stt = time.perf_counter()
        stt_latency_ms = (t_stt - t0) * 1000

        transcript = transcript_buf.value.decode('utf-8', errors='replace').strip()
        if not transcript:
            transcript = "hello world this is a test of the voice pipeline"

        # TTS: text → first audio
        tts_reset(tts)
        t_tts_start = time.perf_counter()
        tts_set_text(tts, transcript.encode())

        ttfs_ms = 0
        total_samples = 0
        pcm_buf = (ctypes.c_float * 4800)()

        while not tts_is_done(tts):
            tts_step(tts)
            n = tts_get_audio(tts, pcm_buf, 4800)
            if n > 0:
                if total_samples == 0:
                    ttfs_ms = (time.perf_counter() - t_tts_start) * 1000
                total_samples += n
                if total_samples > 24000 * 5:  # cap at 5s
                    break

        t_tts_end = time.perf_counter()
        tts_total_ms = (t_tts_end - t_tts_start) * 1000
        pipeline_ms = stt_latency_ms + ttfs_ms

        result = {
            "stt_ms": round(stt_latency_ms, 1),
            "tts_ttfs_ms": round(ttfs_ms, 1),
            "tts_total_ms": round(tts_total_ms, 1),
            "pipeline_ms": round(pipeline_ms, 1),
            "tts_samples": total_samples,
        }
        results.append(result)

        print(f"  Run {run+1}: STT={stt_latency_ms:.1f}ms  TTFS={ttfs_ms:.1f}ms  "
              f"Pipeline={pipeline_ms:.1f}ms  TTS_total={tts_total_ms:.1f}ms")

    # Average
    avg = {k: round(sum(r[k] for r in results) / len(results), 1) for k in results[0]}
    print(f"\n  AVERAGE:")
    print(f"    STT latency:      {avg['stt_ms']:.1f} ms")
    print(f"    TTS TTFS:         {avg['tts_ttfs_ms']:.1f} ms")
    print(f"    Pipeline (STT→TTFS): {avg['pipeline_ms']:.1f} ms")
    print(f"    TTS total:        {avg['tts_total_ms']:.1f} ms")

    out = {"runs": results, "average": avg, "stt_model": stt_model, "tts_model": tts_model}
    os.makedirs("bench_output", exist_ok=True)
    with open("bench_output/e2e_latency.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to bench_output/e2e_latency.json")


if __name__ == "__main__":
    main()
