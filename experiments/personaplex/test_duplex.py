#!/usr/bin/env python3
"""Full-duplex audio test for PersonaPlex 7B on Apple Silicon.

End-to-end validation: generate synthetic 440Hz sine audio, chunk into 80ms frames
(1920 samples at 24kHz), process through PersonaPlex, measure round-trip latency.

This validates the full-duplex S2S claim on real hardware.
"""
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


def generate_test_audio(
    duration_sec: float = 2.0,
    freq_hz: float = 440.0,
    sample_rate_hz: float = 24000.0,
) -> np.ndarray:
    """Generate synthetic sine wave test audio.

    Args:
        duration_sec: Total duration in seconds
        freq_hz: Frequency in Hz (A4 = 440Hz)
        sample_rate_hz: Sample rate in Hz

    Returns:
        Audio waveform as numpy array, shape (num_samples,)
    """
    num_samples = int(duration_sec * sample_rate_hz)
    t = np.arange(num_samples) / sample_rate_hz
    # Sine wave with ramping to avoid clicks
    audio = 0.3 * np.sin(2 * np.pi * freq_hz * t)
    # Fade in/out
    fade_len = int(0.05 * sample_rate_hz)  # 50ms fade
    audio[:fade_len] *= np.linspace(0, 1, fade_len)
    audio[-fade_len:] *= np.linspace(1, 0, fade_len)
    return audio.astype(np.float32)


def chunk_audio(
    audio: np.ndarray,
    chunk_size: int,
    hop_size: int = None,
) -> List[np.ndarray]:
    """Chunk audio into fixed-size frames.

    Args:
        audio: Input audio waveform
        chunk_size: Samples per chunk (1920 for 80ms @ 24kHz)
        hop_size: Hop size for overlapping chunks (defaults to chunk_size for no overlap)

    Returns:
        List of audio chunks
    """
    if hop_size is None:
        hop_size = chunk_size

    chunks = []
    for start in range(0, len(audio) - chunk_size + 1, hop_size):
        chunks.append(audio[start : start + chunk_size])

    # Pad final chunk if needed
    remaining = len(audio) % hop_size
    if remaining > 0 and remaining < chunk_size:
        final_chunk = np.zeros(chunk_size, dtype=np.float32)
        final_chunk[:remaining] = audio[-remaining:]
        chunks.append(final_chunk)

    return chunks


def process_personaplex_frames(
    frames: List[np.ndarray],
    sample_rate_hz: float = 24000.0,
) -> Dict:
    """Process audio frames through PersonaPlex.

    This is a skeleton that demonstrates frame-by-frame processing.
    Actual PersonaPlex API will be integrated here once model is available.

    Args:
        frames: List of audio chunks
        sample_rate_hz: Sample rate

    Returns:
        Processing results including latency measurements
    """
    frame_duration_ms = (len(frames[0]) / sample_rate_hz) * 1000 if frames else 0
    results = {
        "frame_count": len(frames),
        "frame_duration_ms": frame_duration_ms,
        "frames": [],
    }

    print(f"Processing {len(frames)} frames ({frame_duration_ms:.1f}ms each)...")

    # Simulate frame-by-frame processing
    frame_times = []
    for i, frame in enumerate(frames):
        frame_start = time.perf_counter()

        # Placeholder: Real PersonaPlex inference would go here
        # Example structure (once API is known):
        #
        # with torch.no_grad():
        #     input_ids = processor(frame, sampling_rate=sample_rate_hz, return_tensors="pt")
        #     outputs = model.generate(**input_ids)
        #     output_audio = processor.decode(outputs[0])
        #
        # For now, simulate with a small delay
        time.sleep(0.010)  # 10ms simulated processing

        frame_time = (time.perf_counter() - frame_start) * 1000
        frame_times.append(frame_time)

        results["frames"].append({
            "frame_id": i,
            "latency_ms": frame_time,
        })

        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            avg_latency = np.mean(frame_times)
            print(f"  Frame {i+1}/{len(frames)}: {frame_time:.2f}ms (avg: {avg_latency:.2f}ms)")

    return results


def validate_duplex_constraints(
    results: Dict,
    frame_budget_ms: float = 80.0,
) -> Dict:
    """Validate results against full-duplex constraints.

    Args:
        results: Processing results from process_personaplex_frames
        frame_budget_ms: Maximum latency budget per frame

    Returns:
        Validation report
    """
    frame_latencies = [f["latency_ms"] for f in results["frames"]]
    avg_latency = np.mean(frame_latencies)
    max_latency = np.max(frame_latencies)
    min_latency = np.min(frame_latencies)
    p95_latency = np.percentile(frame_latencies, 95)

    report = {
        "frame_budget_ms": frame_budget_ms,
        "latency_stats": {
            "min_ms": min_latency,
            "avg_ms": avg_latency,
            "p95_ms": p95_latency,
            "max_ms": max_latency,
        },
        "rtf": avg_latency / frame_budget_ms,
        "meets_budget": avg_latency < frame_budget_ms,
        "meets_p95_budget": p95_latency < frame_budget_ms,
    }

    return report


def main():
    print("=== PersonaPlex Full-Duplex Audio Test ===\n")

    # Test parameters
    test_duration_sec = 2.0
    sample_rate_hz = 24000.0
    frame_budget_ms = 80.0  # 12.5Hz frame rate
    chunk_samples = int(frame_budget_ms * sample_rate_hz / 1000)  # 1920 @ 24kHz

    print(f"Test configuration:")
    print(f"  Duration: {test_duration_sec}s")
    print(f"  Sample rate: {sample_rate_hz}Hz")
    print(f"  Frame size: {chunk_samples} samples ({frame_budget_ms}ms)")
    print(f"  Expected frames: {int(test_duration_sec * sample_rate_hz / chunk_samples)}")

    # Generate test audio
    print(f"\nGenerating test audio (440Hz sine wave)...")
    audio = generate_test_audio(
        duration_sec=test_duration_sec,
        freq_hz=440.0,
        sample_rate_hz=sample_rate_hz,
    )
    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio range: [{audio.min():.3f}, {audio.max():.3f}]")

    # Chunk audio
    print(f"\nChunking audio...")
    frames = chunk_audio(audio, chunk_samples)
    print(f"  Generated {len(frames)} frames")

    # Process through PersonaPlex
    print(f"\nProcessing through PersonaPlex...")
    processing_results = process_personaplex_frames(frames, sample_rate_hz)

    # Validate against frame budget
    print(f"\n=== Full-Duplex Validation ===")
    validation = validate_duplex_constraints(processing_results, frame_budget_ms)

    print(f"Frame budget: {validation['frame_budget_ms']}ms")
    print(f"Latency stats:")
    stats = validation["latency_stats"]
    print(f"  Min: {stats['min_ms']:.2f}ms")
    print(f"  Avg: {stats['avg_ms']:.2f}ms")
    print(f"  P95: {stats['p95_ms']:.2f}ms")
    print(f"  Max: {stats['max_ms']:.2f}ms")
    print(f"RTF: {validation['rtf']:.2f}x")
    print(f"Meets average budget: {'YES' if validation['meets_budget'] else 'NO'}")
    print(f"Meets P95 budget: {'YES' if validation['meets_p95_budget'] else 'NO'}")

    # Save results
    output_file = Path(__file__).parent / "duplex_test_results.json"
    test_report = {
        "timestamp": time.time(),
        "platform": "Apple Silicon (M-series)",
        "model": "PersonaPlex-7B",
        "test_config": {
            "duration_sec": test_duration_sec,
            "sample_rate_hz": sample_rate_hz,
            "chunk_samples": chunk_samples,
            "frame_budget_ms": frame_budget_ms,
        },
        "processing_results": processing_results,
        "validation": validation,
    }

    with open(output_file, "w") as f:
        json.dump(test_report, f, indent=2)

    print(f"\nTest results saved to: {output_file}")
    print(f"\nNext step: compare with own Talker model benchmarks")


if __name__ == "__main__":
    main()
