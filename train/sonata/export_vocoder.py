#!/usr/bin/env python3
"""Export trained vocoder checkpoint to safetensors for Rust inference.

Handles weight normalization fusion: PyTorch stores parametrizations.weight.original0 (v)
and original1 (g), which get fused to: weight = g * (v / ||v||).

Usage:
  python train/sonata/export_vocoder.py \
    --checkpoint train/checkpoints/vocoder/vocoder_step50000.pt \
    --output-dir models/sonata
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def fuse_weight_norm(state: dict) -> dict:
    """Fuse parametrized weight normalization into plain weights.

    PyTorch weight_norm stores:
      key.parametrizations.weight.original0 = v (direction)
      key.parametrizations.weight.original1 = g (magnitude)
    Fused: weight = g * (v / ||v||)
    """
    fused = {}
    param_keys = set()

    # Find all parametrized weight pairs
    # PyTorch weight_norm parametrize convention:
    #   original0 = g (magnitude), shape (out_ch, 1, 1)
    #   original1 = v (direction), shape (out_ch, in_ch, kernel)
    for k in state:
        if ".parametrizations.weight.original0" in k:
            base = k.replace(".parametrizations.weight.original0", "")
            g_key = k                                    # original0 = g
            v_key = k.replace(".original0", ".original1")  # original1 = v
            if v_key in state:
                g = state[g_key]  # (out_ch, 1, 1)
                v = state[v_key]  # (out_ch, in_ch, kernel)
                # Fuse: w = g * v / ||v||
                v_norm = torch.linalg.norm(v.reshape(v.shape[0], -1), dim=1, keepdim=True)
                # g is already broadcastable (out_ch, 1, 1)
                v_normalized = v / v_norm.unsqueeze(-1).clamp(min=1e-12)
                w = g * v_normalized
                fused[base + ".weight"] = w.contiguous()
                param_keys.add(g_key)
                param_keys.add(v_key)

    # Copy non-parametrized keys
    for k, v in state.items():
        if k not in param_keys:
            fused[k] = v.contiguous()

    return fused


def export_vocoder(ckpt_path: str, output_dir: str):
    """Export vocoder generator weights with fused weight norm."""
    print(f"[export] Loading vocoder: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = ckpt.get("config", {})
    generator_state = ckpt.get("generator", ckpt.get("model", {}))

    # Fuse weight normalization
    print(f"[export] Fusing weight normalization...")
    n_before = len(generator_state)
    fused_state = fuse_weight_norm(generator_state)
    n_after = len(fused_state)
    print(f"[export] {n_before} tensors → {n_after} tensors (fused {n_before - n_after} param pairs)")

    # Wrap in "generator." prefix for Rust loader (expects vb.pp("generator"))
    wrapped = {}
    for k, v in fused_state.items():
        wrapped[f"generator.{k}"] = v

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_file(wrapped, str(out / "sonata_vocoder.safetensors"))

    vocoder_config = {
        "sample_rate": config.get("sample_rate", 24000),
        "n_mels": config.get("n_mels", 80),
        "hop_length": config.get("hop_length", 480),
        "upsample_initial_channel": config.get("upsample_initial_channel", 512),
        "upsample_rates": config.get("upsample_rates", [10, 6, 2, 2, 2]),
        "upsample_kernel_sizes": config.get("upsample_kernel_sizes", [20, 12, 4, 4, 4]),
        "resblock_kernel_sizes": config.get("resblock_kernel_sizes", [3, 7, 11]),
        "resblock_dilation_sizes": config.get("resblock_dilation_sizes",
                                               [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
    }
    with open(out / "sonata_vocoder_config.json", "w") as f:
        json.dump(vocoder_config, f, indent=2)

    n_params = sum(v.numel() for v in wrapped.values())
    print(f"[export] Vocoder: {n_params/1e6:.1f}M params → {out / 'sonata_vocoder.safetensors'}")
    print(f"[export] Config: {out / 'sonata_vocoder_config.json'}")

    # Verify
    from safetensors import safe_open
    with safe_open(str(out / "sonata_vocoder.safetensors"), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        sample = f.get_tensor(keys[0])
        print(f"[export] Verified: {len(keys)} tensors (sample: {keys[0]} {list(sample.shape)})")


def main():
    ap = argparse.ArgumentParser(description="Export vocoder to safetensors")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", default="models/sonata")
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return 1

    export_vocoder(args.checkpoint, args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
