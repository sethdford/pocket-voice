#!/usr/bin/env python3
"""Quantize PersonaPlex 7B (or alternative S2S model) to INT4 via MLX for Apple Silicon inference.

Uses MLX's quantization tools to convert to INT4 with group size 64 for optimal M-series performance.
"""
import os
import sys
from pathlib import Path


def main():
    # Check for MLX installation
    try:
        import mlx.core as mx
    except ImportError:
        print("Error: MLX not installed")
        print("Install with: pip install mlx mlx-lm")
        sys.exit(1)

    weights_dir = Path(__file__).parent / "weights"
    output_dir = Path(__file__).parent / "weights_int4"

    # Verify input weights exist
    if not weights_dir.exists():
        print(f"Error: Weights directory not found: {weights_dir}")
        print("Run download.py first")
        sys.exit(1)

    # Check for model files
    model_files = list(weights_dir.glob("*.pt")) + list(weights_dir.glob("*.bin")) + list(weights_dir.glob("*.safetensors"))
    if not model_files:
        print(f"Warning: No model files found in {weights_dir}")
        print("Expected .pt, .bin, or .safetensors files")
        return

    print(f"Found {len(model_files)} model file(s)")
    print(f"Input directory: {weights_dir}")
    print(f"Output directory: {output_dir}")

    try:
        from mlx_lm import convert
    except ImportError:
        print("Error: mlx-lm not installed")
        print("Install with: pip install mlx-lm")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nQuantizing to INT4 (group_size=64)...")
    print("This may take several minutes...")

    try:
        # MLX quantization with INT4, group size 64
        # The exact API may vary; this is a common pattern
        quantize_config = {
            "quantize": True,
            "q_bits": 4,
            "q_group_size": 64,
        }

        print(f"Config: {quantize_config}")

        # Note: MLX's convert API signature may vary
        # This is a placeholder that documents the intended behavior
        convert(
            str(weights_dir),
            str(output_dir),
            **quantize_config
        )

        print(f"\nQuantization complete!")
        print(f"Quantized model saved to: {output_dir}")

        # Estimate size reduction
        input_size = sum(f.stat().st_size for f in weights_dir.glob("*") if f.is_file())
        if list(output_dir.glob("*")):
            output_size = sum(f.stat().st_size for f in output_dir.glob("*") if f.is_file())
            ratio = output_size / input_size if input_size > 0 else 0
            print(f"Input size: {input_size / 1024 / 1024:.1f} MB")
            print(f"Output size: {output_size / 1024 / 1024:.1f} MB")
            print(f"Compression ratio: {ratio:.2%}")

        print(f"\nNext step: python benchmark.py")

    except Exception as e:
        print(f"Error during quantization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
