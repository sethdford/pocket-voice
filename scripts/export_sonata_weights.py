#!/usr/bin/env python3
"""Export Sonata LM + Flow weights from PyTorch to safetensors for Rust/candle.

Usage:
  python scripts/export_sonata_weights.py \
      --lm-checkpoint train/checkpoints_sonata_lm/step_50000.pt \
      --lm-output models/sonata_lm.safetensors

  python scripts/export_sonata_weights.py \
      --codec-checkpoint train/checkpoints_sonata_codec/step_50000.pt \
      --codec-output models/sonata_codec.safetensors
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file


def export_lm(checkpoint_path: str, output_path: str, config_path: str):
    """Export Sonata Semantic LM weights."""
    print(f"Loading LM checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    sd = ckpt["model"]
    config = ckpt.get("config", {})

    mapped = OrderedDict()
    for key, tensor in sd.items():
        tensor = tensor.contiguous().float()
        mapped[key] = tensor

    save_file(mapped, output_path)
    fsize = Path(output_path).stat().st_size
    print(f"Exported {len(mapped)} tensors to {output_path} ({fsize/1e6:.1f} MB)")

    for name, t in list(mapped.items())[:15]:
        print(f"  {name}: {list(t.shape)}")
    if len(mapped) > 15:
        print(f"  ... ({len(mapped) - 15} more)")

    # Save config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


def export_codec(checkpoint_path: str, output_path: str, config_path: str):
    """Export Sonata Codec weights (encoder + FSQ + decoder)."""
    print(f"Loading Codec checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    sd = ckpt["model"]
    config = ckpt.get("config", {})

    # Split into encoder, decoder, FSQ
    encoder_weights = OrderedDict()
    decoder_weights = OrderedDict()

    for key, tensor in sd.items():
        tensor = tensor.contiguous().float()
        if key.startswith("encoder.") or key.startswith("mel."):
            encoder_weights[key] = tensor
        else:
            decoder_weights[key] = tensor

    # Save decoder (for C inference)
    save_file(decoder_weights, output_path)
    fsize = Path(output_path).stat().st_size
    print(f"Decoder: {len(decoder_weights)} tensors → {output_path} ({fsize/1e6:.1f} MB)")

    # Save encoder (for Python encoding pipeline)
    enc_path = output_path.replace(".safetensors", "_encoder.safetensors")
    save_file(encoder_weights, enc_path)
    fsize = Path(enc_path).stat().st_size
    print(f"Encoder: {len(encoder_weights)} tensors → {enc_path} ({fsize/1e6:.1f} MB)")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm-checkpoint", default="")
    parser.add_argument("--lm-output", default="models/sonata_lm.safetensors")
    parser.add_argument("--lm-config", default="models/sonata_lm.json")
    parser.add_argument("--codec-checkpoint", default="")
    parser.add_argument("--codec-output", default="models/sonata_codec.safetensors")
    parser.add_argument("--codec-config", default="models/sonata_codec.json")
    args = parser.parse_args()

    if args.lm_checkpoint:
        export_lm(args.lm_checkpoint, args.lm_output, args.lm_config)
    if args.codec_checkpoint:
        export_codec(args.codec_checkpoint, args.codec_output, args.codec_config)
    if not args.lm_checkpoint and not args.codec_checkpoint:
        print("Specify --lm-checkpoint and/or --codec-checkpoint")


if __name__ == "__main__":
    main()
