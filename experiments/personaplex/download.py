#!/usr/bin/env python3
"""Download PersonaPlex 7B (or alternative S2S model) weights from HuggingFace.

PersonaPlex is an NVIDIA Moshi-derived full-duplex speech-to-speech model.
If the exact model ID is unavailable, falls back to alternative S2S models.
"""
import os
import sys
from pathlib import Path

# Models to try, in priority order
MODEL_CANDIDATES = [
    "nvidia/PersonaPlex-7B",  # Primary: NVIDIA PersonaPlex
    "kyutai/moshi-7b",         # Fallback 1: Kyutai Moshi (parent architecture)
    "gpt2",                    # Fallback 2: Simple model for testing infrastructure
]


def main():
    try:
        from huggingface_hub import snapshot_download, model_info
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Install with: pip install huggingface-hub")
        sys.exit(1)

    local_dir = Path(__file__).parent / "weights"
    local_dir.mkdir(parents=True, exist_ok=True)

    model_id = None
    for candidate in MODEL_CANDIDATES:
        try:
            print(f"Checking availability of {candidate}...")
            info = model_info(candidate)
            model_id = candidate
            print(f"Found: {candidate}")
            break
        except Exception as e:
            print(f"  Not available: {candidate} ({type(e).__name__})")
            continue

    if not model_id:
        print(f"\nError: No models available from candidates: {MODEL_CANDIDATES}")
        print("Check HuggingFace Hub for alternative speech-to-speech models.")
        sys.exit(1)

    print(f"\nDownloading {model_id}...")
    print(f"Target directory: {local_dir}")

    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            ignore_patterns=["*.md", "*.txt"],
            force_download=False,
        )
        print(f"\nSuccessfully downloaded to: {path}")

        # List downloaded files and their sizes
        total_size = 0
        print("\nDownloaded files:")
        for f in sorted(os.listdir(path)):
            fpath = os.path.join(path, f)
            if os.path.isfile(fpath):
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                total_size += size_mb
                print(f"  {f}: {size_mb:.1f} MB")

        print(f"\nTotal size: {total_size:.1f} MB")
        print(f"\nNext step: python quantize_mlx.py")

    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
