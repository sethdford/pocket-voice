#!/usr/bin/env python3
"""Latency benchmark for PersonaPlex 7B on Apple Silicon (M-series).

Measures per-step inference latency to validate full-duplex S2S feasibility.
PersonaPlex claims 68ms/step on M-series. Our 500M Talker targets 100ms/step.

Frame budget: 80ms (12.5Hz frame rate).
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict


def benchmark_mlx_matmuls() -> Dict:
    """Benchmark MLX matmuls at PersonaPlex model dimensions.

    PersonaPlex 7B typical dims:
    - Hidden: 4096
    - FFN hidden: 11008
    - Heads: 32
    - Head dim: 128
    """
    try:
        import mlx.core as mx
    except ImportError:
        print("Warning: MLX not installed, skipping matmul benchmarks")
        return {}

    print("Benchmarking MLX matmuls at PersonaPlex 7B dimensions...")

    # PersonaPlex-like dimensions
    hidden_dim = 4096
    ffn_hidden = 11008
    batch_size = 1
    seq_len = 1  # Single token (streaming mode)

    results = {}

    try:
        # Benchmark attention projection: (batch, seq, hidden) @ (hidden, hidden)
        print(f"  Attention QKV projection: ({batch_size}, {seq_len}, {hidden_dim}) @ ({hidden_dim}, {hidden_dim*3})")
        x_attn = mx.zeros((batch_size, seq_len, hidden_dim))
        w_attn = mx.zeros((hidden_dim, hidden_dim * 3))

        times = []
        for _ in range(10):
            mx.eval([x_attn, w_attn])
            start = time.perf_counter()
            result = mx.matmul(x_attn, w_attn)
            mx.eval(result)
            times.append(time.perf_counter() - start)

        attn_time_ms = (sum(times) / len(times)) * 1000
        results["attention_qkv_ms"] = attn_time_ms
        print(f"    Time: {attn_time_ms:.2f} ms")

        # Benchmark FFN: (batch, seq, hidden) @ (hidden, ffn_hidden)
        print(f"  FFN up-proj: ({batch_size}, {seq_len}, {hidden_dim}) @ ({hidden_dim}, {ffn_hidden})")
        x_ffn = mx.zeros((batch_size, seq_len, hidden_dim))
        w_ffn_up = mx.zeros((hidden_dim, ffn_hidden))

        times = []
        for _ in range(10):
            mx.eval([x_ffn, w_ffn_up])
            start = time.perf_counter()
            result = mx.matmul(x_ffn, w_ffn_up)
            mx.eval(result)
            times.append(time.perf_counter() - start)

        ffn_time_ms = (sum(times) / len(times)) * 1000
        results["ffn_up_ms"] = ffn_time_ms
        print(f"    Time: {ffn_time_ms:.2f} ms")

        # Rough estimate: 32 layers * (attn + 2x FFN) per step
        estimated_step_ms = 32 * (attn_time_ms + 2 * ffn_time_ms)
        results["estimated_step_ms"] = estimated_step_ms
        print(f"\n  Rough estimate (32 layers): {estimated_step_ms:.0f} ms/step")

    except Exception as e:
        print(f"  Error during matmul benchmark: {e}")

    return results


def validate_frame_budget(step_time_ms: float, frame_budget_ms: float = 80) -> Dict:
    """Validate feasibility against frame budget.

    Returns validation report with margins and RTF.
    """
    # Assume 2 forward passes per audio frame (predict + refine, or similar)
    forward_passes_per_frame = 2
    total_time_needed = step_time_ms * forward_passes_per_frame

    margin_ms = frame_budget_ms - total_time_needed
    margin_pct = (margin_ms / frame_budget_ms) * 100 if frame_budget_ms > 0 else 0

    # RTF = model_time / wall_time_budget
    rtf = total_time_needed / frame_budget_ms

    return {
        "frame_budget_ms": frame_budget_ms,
        "step_time_ms": step_time_ms,
        "forward_passes_per_frame": forward_passes_per_frame,
        "total_time_needed_ms": total_time_needed,
        "margin_ms": margin_ms,
        "margin_pct": margin_pct,
        "rtf": rtf,
        "feasible": margin_ms > 0,
    }


def main():
    print("=== PersonaPlex M4 Latency Benchmark ===\n")

    results = {
        "platform": "Apple Silicon (M-series)",
        "model": "PersonaPlex-7B",
        "timestamp": time.time(),
    }

    # Benchmark MLX operations
    matmul_results = benchmark_mlx_matmuls()
    results["matmul_benchmarks"] = matmul_results

    # If we have estimates, validate against frame budget
    if "estimated_step_ms" in matmul_results:
        estimated_ms = matmul_results["estimated_step_ms"]
        validation = validate_frame_budget(estimated_ms)
        results["frame_validation"] = validation

        print(f"\n=== Frame Budget Validation ===")
        print(f"Target frame budget: {validation['frame_budget_ms']}ms (12.5Hz)")
        print(f"Estimated per-step latency: {validation['step_time_ms']:.0f}ms")
        print(f"Total needed (2 passes): {validation['total_time_needed_ms']:.0f}ms")
        print(f"RTF: {validation['rtf']:.2f}x")
        print(f"Margin: {validation['margin_ms']:.0f}ms ({validation['margin_pct']:.0f}%)")
        print(f"Feasible: {'YES' if validation['feasible'] else 'NO'}")

    # Real inference timing (requires actual model)
    print(f"\n=== Real Model Timing ===")
    print("To measure actual PersonaPlex inference latency:")
    print("  1. Run: python download.py")
    print("  2. Run: python quantize_mlx.py")
    print("  3. Implement PersonaPlex.forward() timing in test_duplex.py")
    print("  4. Results will be saved to benchmark_results.json")

    # Save results
    output_file = Path(__file__).parent / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark results saved to: {output_file}")
    print(f"\nNext step: python test_duplex.py")


if __name__ == "__main__":
    main()
