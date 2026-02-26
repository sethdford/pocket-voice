#!/usr/bin/env python3
"""Export Mimi-Lite LM weights from PyTorch checkpoint to safetensors for candle.

Maps PyTorch weight names to a flat namespace that the Rust candle model expects.

Usage:
  python scripts/export_mimi_lite_weights.py \
      --checkpoint train/checkpoints/step_5000.pt \
      --output models/mimi_lite_lm.safetensors
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file


def export(checkpoint_path: str, output_path: str, config_path: str):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    sd = ckpt["model"]
    config = ckpt.get("config", {})

    mapped = OrderedDict()

    for key, tensor in sd.items():
        tensor = tensor.contiguous().float()

        # Embeddings
        if key == "text_emb.weight":
            mapped["text_emb.weight"] = tensor
        elif key == "audio_emb.weight":
            mapped["audio_emb.weight"] = tensor
        elif key == "speaker_proj.weight":
            mapped["speaker_proj.weight"] = tensor

        # Transformer layers
        elif key.startswith("layers."):
            parts = key.split(".")
            layer_idx = parts[1]

            # Self-attention
            if ".attn." in key:
                sub = ".".join(parts[3:])
                mapped[f"layers.{layer_idx}.attn.{sub}"] = tensor
            elif ".attn_norm." in key:
                sub = ".".join(parts[3:])
                mapped[f"layers.{layer_idx}.attn_norm.{sub}"] = tensor
            # FFN
            elif ".ffn." in key:
                sub = ".".join(parts[3:])
                mapped[f"layers.{layer_idx}.ffn.{sub}"] = tensor
            elif ".ffn_norm." in key:
                sub = ".".join(parts[3:])
                mapped[f"layers.{layer_idx}.ffn_norm.{sub}"] = tensor
            # Cross-attention
            elif ".cross_attn." in key:
                sub = ".".join(parts[3:])
                mapped[f"layers.{layer_idx}.cross_attn.{sub}"] = tensor
            else:
                mapped[key] = tensor

        # Output
        elif key == "output_norm.weight":
            mapped["output_norm.weight"] = tensor
        elif key == "output_head.weight":
            mapped["output_head.weight"] = tensor
        else:
            mapped[key] = tensor

    save_file(mapped, output_path)
    file_size = Path(output_path).stat().st_size
    print(f"Exported {len(mapped)} tensors to {output_path} ({file_size/1e6:.1f} MB)")

    for name, t in list(mapped.items())[:20]:
        print(f"  {name}: {list(t.shape)}")
    if len(mapped) > 20:
        print(f"  ... ({len(mapped) - 20} more)")

    # Save config alongside
    config_out = Path(config_path)
    with open(config_out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="models/mimi_lite_lm.safetensors")
    parser.add_argument("--config", default="models/mimi_lite_lm.json")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.config)


if __name__ == "__main__":
    main()
