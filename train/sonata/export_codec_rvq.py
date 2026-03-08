#!/usr/bin/env python3
"""Export trained RVQ codec checkpoint to safetensors format for Rust inference.

Extracts the encoder, RVQ codebooks, and decoder from a trained checkpoint
and exports weights in safetensors format for sonata_codec Rust inference engine.

The RVQ codec maps 24kHz audio to 12.5Hz quantized codes (8 codebooks x 2048 entries).

Usage:
  python train/sonata/export_codec_rvq.py \
    --checkpoint checkpoints/codec_rvq/codec_rvq_step200000.pt \
    --output-dir models/sonata_codec_rvq
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
from safetensors.torch import save_file


def export_codec_rvq(
    ckpt_path: str,
    output_dir: str,
) -> None:
    """
    Export RVQ codec weights to safetensors.

    Args:
        ckpt_path: Path to trained checkpoint (codec_rvq_step_*.pt or codec_rvq_latest.pt)
        output_dir: Output directory for safetensors + config.json
    """
    # Load checkpoint
    print(f"[export] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract model state dict and metadata
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        step = ckpt.get("step", 0)
        epoch = ckpt.get("epoch", 0)
    else:
        # Fallback: raw state dict
        state = ckpt
        step = 0
        epoch = 0

    print(f"  Step: {step}, Epoch: {epoch}")

    # Prepare output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Convert to contiguous for safetensors
    codec_state = {k: v.contiguous() for k, v in state.items()}

    # Save weights
    safetensors_path = out / "sonata_codec_rvq.safetensors"
    save_file(codec_state, str(safetensors_path))
    print(f"[export] Weights: {safetensors_path}")

    # Build config for Rust inference
    codec_config = {
        # Encoder/decoder architecture
        "enc_dim": 512,
        "n_strides": 5,
        "strides_enc": [4, 8, 5, 4, 3],  # 1920x downsample
        "strides_dec": [3, 4, 5, 8, 4],  # 1920x upsample
        # Transformer bottleneck
        "n_transformer_layers": 8,
        "n_transformer_heads": 8,
        "ff_mult": 4.0,
        # RVQ quantization
        "n_codebooks": 8,
        "codebook_size": 2048,
        "codebook_dim": 128,
        # Audio parameters
        "sample_rate": 24000,
        "hop_length": 1920,  # 24000 / 1920 = 12.5 Hz
        "target_frame_rate": 12.5,
        # Training metadata
        "trained_step": step,
        "trained_epoch": epoch,
    }

    config_path = out / "sonata_codec_rvq_config.json"
    with open(config_path, "w") as f:
        json.dump(codec_config, f, indent=2)
    print(f"[export] Config: {config_path}")

    # Calculate and report model statistics
    n_params = sum(v.numel() for v in codec_state.values())
    total_bytes = sum(v.numel() * v.element_size() for v in codec_state.values())
    file_size_mb = total_bytes / (1024 * 1024)

    print(f"\n[export] Model: {n_params/1e6:.1f}M params, {file_size_mb:.1f} MB")
    print(f"[export] Architecture:")
    print(f"          Encoder:       {' × '.join(map(str, codec_config['strides_enc']))} = 1920x downsample")
    print(f"          Bottleneck:    {codec_config['n_transformer_layers']}-layer Transformer ({codec_config['enc_dim']}D, {codec_config['n_transformer_heads']} heads)")
    print(f"          RVQ:           {codec_config['n_codebooks']} codebooks × {codec_config['codebook_size']} entries ({codec_config['codebook_dim']}D each)")
    print(f"          Decoder:       {' × '.join(map(str, codec_config['strides_dec']))} = 1920x upsample")
    print(f"[export] Frame rate:    24kHz → {codec_config['target_frame_rate']}Hz (hop={codec_config['hop_length']})")

    # Verify weights can be loaded back
    print(f"\n[export] Verifying weights can be loaded...")
    try:
        from safetensors import safe_open
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            keys = f.keys()
            n_keys = len(keys)
            sample_key = next(iter(keys))
            sample_tensor = f.get_tensor(sample_key)
            print(f"[export] ✓ Verification passed ({n_keys} tensors, sample: {sample_key} shape {sample_tensor.shape})")
    except Exception as e:
        print(f"[export] ✗ Verification failed: {e}")
        raise

    print(f"\n[export] Complete!")
    print(f"[export] Ready for Rust inference:")
    print(f"          weights:  {safetensors_path}")
    print(f"          config:   {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export trained RVQ codec weights to safetensors for Rust inference"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint (codec_rvq_step_*.pt or codec_rvq_latest.pt)"
    )
    parser.add_argument(
        "--output-dir",
        default="models/sonata_codec_rvq",
        help="Output directory for safetensors + config.json"
    )
    args = parser.parse_args()

    # Validate checkpoint exists
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return

    export_codec_rvq(
        str(ckpt_path),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
