"""PersonaPlex S2S Validation Experiments

This package contains tools to download, quantize, and benchmark NVIDIA's PersonaPlex-7B
(or alternative open-source speech-to-speech models) on Apple Silicon.

Scripts:
  - download.py: Download PersonaPlex weights from HuggingFace
  - quantize_mlx.py: Quantize to INT4 via MLX for M-series optimization
  - benchmark.py: Measure per-step latency and RTF
  - test_duplex.py: Full-duplex audio processing test

Reference:
  - PersonaPlex: NVIDIA research (Moshi-derived)
  - Frame budget: 80ms (12.5Hz for real-time)
  - Target: <200ms round-trip latency for full-duplex S2S
"""
