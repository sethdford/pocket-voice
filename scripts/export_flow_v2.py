#!/usr/bin/env python3
"""Export Sonata Flow v2 checkpoint from PyTorch to safetensors for Rust inference.

Flow v2 is a single-stage text-to-mel model. This script extracts the model
state dict and saves it as safetensors, with config as JSON.

Usage:
  python scripts/export_flow_v2.py \
    --ckpt train/checkpoints/flow_v2/flow_v2_final.pt \
    --output models/sonata/flow_v2.safetensors
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def dataclass_to_dict(obj):
    """Convert a dataclass to a JSON-serializable dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return obj


def _fuse_weight_norm(state_dict: dict) -> dict:
    """Fuse weight_g and weight_v into weight: w = g * v / ||v||.

    Vocoder/decoder layers may use nn.utils.parametrizations.weight_norm() which
    stores weight_g and weight_v instead of weight. Rust inference expects plain
    weight tensors.
    """
    fused = {}
    processed = set()
    for key in list(state_dict.keys()):
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            v_key = base + ".weight_v"
            if v_key in state_dict:
                g = state_dict[key]
                v = state_dict[v_key]
                # weight = g * v / ||v||; norm over parameter dims (1..ndim)
                dims = list(range(1, v.dim()))
                norm = v.norm(dim=dims, keepdim=True).clamp(min=1e-12)
                g_expand = g.view(g.shape[0], *([1] * (v.dim() - 1)))
                fused[base + ".weight"] = g_expand * v / norm
                processed.add(key)
                processed.add(v_key)
    # Copy non-weight-norm keys
    for key, val in state_dict.items():
        if key not in processed:
            fused[key] = val
    return fused


def main():
    parser = argparse.ArgumentParser(
        description="Export Flow v2 checkpoint to safetensors"
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for .safetensors (config saved alongside as .json)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.output)
    if out_path.suffix != ".safetensors":
        out_path = out_path.with_suffix(".safetensors")
    config_path = out_path.with_suffix(".json")

    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract state dict (handle different checkpoint formats)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            sd = ckpt["model"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
        config = ckpt.get("config", ckpt.get("cfg", {}))
    else:
        sd = ckpt
        config = {}

    # Convert config dataclass to dict if needed
    if hasattr(config, "__dataclass_fields__"):
        config = dataclass_to_dict(config)

    # Ensure we have FlowV2Config defaults if config is empty
    if not config:
        root = Path(__file__).resolve().parent.parent
        if (root / "train" / "sonata" / "config.py").exists():
            sys.path.insert(0, str(root))
            from train.sonata.config import FlowV2Config

            cfg = FlowV2Config()
            config = dataclass_to_dict(cfg)

    # Fuse weight_norm (weight_g + weight_v -> weight) for Rust compatibility
    if any(k.endswith(".weight_g") for k in sd.keys()):
        sd = _fuse_weight_norm(sd)
        print("  Fused weight_norm layers (weight_g/weight_v -> weight)")

    # Try safetensors first, fall back to numpy dict
    try:
        from safetensors.torch import save_file

        mapped = {}
        for key, tensor in sd.items():
            mapped[key] = tensor.contiguous().float()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(mapped, str(out_path))
        fsize = out_path.stat().st_size
        print(f"Exported {len(mapped)} tensors to {out_path} ({fsize / 1e6:.1f} MB)")

    except ImportError:
        print("safetensors not available, saving as .pt with numpy-compatible format")
        # Fallback: save as torch .pt (still loadable by Rust via PyTorch format
        # or we'd need to implement numpy loading - Rust candle uses safetensors)
        mapped = {k: v.contiguous().float().numpy() for k, v in sd.items()}
        import numpy as np

        np_path = out_path.with_suffix(".npz")
        np.savez_compressed(np_path, **mapped)
        fsize = np_path.stat().st_size
        print(f"Exported {len(mapped)} arrays to {np_path} ({fsize / 1e6:.1f} MB)")
        print("Note: Rust engine expects safetensors. Install: pip install safetensors")
        out_path = np_path  # for config path below

    # Print sample weight names
    for i, (name, t) in enumerate(list(sd.items())[:12]):
        shape = list(t.shape) if hasattr(t, "shape") else "?"
        print(f"  {name}: {shape}")
    if len(sd) > 12:
        print(f"  ... ({len(sd) - 12} more)")

    # Save config
    config_out = config_path
    if out_path.suffix == ".npz":
        config_out = out_path.with_suffix(".json")
    with open(config_out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_out}")


if __name__ == "__main__":
    main()
