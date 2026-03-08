# PersonaPlex S2S Validation

Validate NVIDIA PersonaPlex-7B (full-duplex speech-to-speech model) on Apple Silicon M-series.

## Quick Start

```bash
cd experiments/personaplex

# 1. Download weights (PersonaPlex or fallback S2S model)
python download.py

# 2. Quantize to INT4 for M-series
python quantize_mlx.py

# 3. Benchmark latency
python benchmark.py

# 4. Full-duplex audio test
python test_duplex.py
```

## Scripts

### download.py

Downloads PersonaPlex-7B (or alternative S2S model) from HuggingFace.

- Primary: `nvidia/PersonaPlex-7B`
- Fallback 1: `kyutai/moshi-7b` (parent architecture)
- Fallback 2: `gpt2` (testing)

Saves to `weights/` directory.

### quantize_mlx.py

Quantizes downloaded weights to INT4 using MLX (Apple's ML framework).

- Group size: 64
- Output: `weights_int4/`
- Purpose: Optimize inference on M-series by reducing memory bandwidth

### benchmark.py

Measures per-step inference latency on M-series.

- Benchmarks MLX matmuls at PersonaPlex dimensions (hidden=4096, ffn=11008)
- Estimates step time from matrix operation timings
- Validates against 80ms frame budget (12.5Hz)
- Calculates RTF (Real-Time Factor)
- Output: `benchmark_results.json`

### test_duplex.py

End-to-end full-duplex audio validation.

- Generates synthetic 440Hz sine wave at 24kHz
- Chunks into 80ms frames (1920 samples)
- Processes through PersonaPlex (placeholder)
- Measures latency per frame
- Validates RTF for real-time constraints
- Output: `duplex_test_results.json`

## Requirements

```bash
pip install huggingface-hub mlx mlx-lm numpy
```

## Expected Results

| Metric                   | Target | Notes                              |
| ------------------------ | ------ | ---------------------------------- |
| PersonaPlex step latency | <68ms  | Claimed by NVIDIA on M-series      |
| RTF (Real-Time Factor)   | <1.0   | <1.0 = faster than realtime        |
| Frame budget headroom    | >0ms   | Available time for other ops       |
| Full-duplex feasibility  | YES    | Should run without dropping frames |

## Architecture

PersonaPlex is Moshi-derived:

- 7B parameters
- Full-duplex (simultaneous input+output)
- Int4 quantization reduces from ~14GB to ~3.5GB
- Target platform: Apple Silicon M4 (14-core GPU)

## Next Steps

1. Run download → quantize → benchmark → test_duplex
2. Compare PersonaPlex latency vs custom 500M Talker model
3. Identify quality gaps and optimization opportunities
4. Design custom Talker training (500M params, <100ms/step target)

## References

- PersonaPlex: NVIDIA research paper
- Moshi: Kyutai Speech-to-Speech foundation
- MLX: Apple ML framework (Metal GPU acceleration)
- Frame rate: 12.5Hz (80ms) = optimal for streaming speech
