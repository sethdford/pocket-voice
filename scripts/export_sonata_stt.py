#!/usr/bin/env python3
"""Export trained Sonata CTC model to .cstt_sonata binary for C inference.

Weight layout matches sonata_stt.c expectations:
  Header (40 bytes) → input_proj → blocks × N → adapter → ctc_proj

Usage:
  python scripts/export_sonata_stt.py \
    --ckpt train/checkpoints/stt/ctc/sonata_ctc_final.pt \
    --output models/sonata/sonata_stt.cstt_sonata
"""

import argparse
import struct
import sys
from pathlib import Path

import torch
import numpy as np


MAGIC = 0x53545453  # "STTS"
VERSION = 1


def write_tensor(f, tensor):
    """Write a float32 tensor to file in row-major order."""
    data = tensor.detach().cpu().float().numpy()
    f.write(data.tobytes())
    return data.size


def export(ckpt_path: str, output_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    cfg = ckpt.get("config", {})

    enc_dim = cfg.get("enc_dim", 256)
    n_layers = cfg.get("enc_n_layers", 4)
    n_heads = cfg.get("enc_n_heads", 4)
    n_mels = cfg.get("n_mels", 80)
    conv_kernel = cfg.get("enc_conv_kernel", 31)
    text_vocab = cfg.get("text_vocab_size", 30)

    ff_dim = enc_dim * 4  # Conformer FF multiplier = 4

    total_weights = 0
    tensors = []

    def queue(key, expected_shape=None):
        nonlocal total_weights
        t = state[key]
        if expected_shape:
            assert t.shape == torch.Size(expected_shape), \
                f"{key}: expected {expected_shape}, got {list(t.shape)}"
        n = t.numel()
        total_weights += n
        tensors.append(t)
        return n

    # input_proj: Linear(n_mels, enc_dim)
    queue("encoder.input_proj.weight", [enc_dim, n_mels])
    queue("encoder.input_proj.bias", [enc_dim])

    use_rope = cfg.get("use_rope", True)

    # Conformer blocks
    for l in range(n_layers):
        prefix = f"encoder.blocks.{l}"

        # FF1: LayerNorm → Linear(D, 4D) → SiLU → Linear(4D, D)
        queue(f"{prefix}.ff1.net.0.weight", [enc_dim])
        queue(f"{prefix}.ff1.net.0.bias", [enc_dim])
        queue(f"{prefix}.ff1.net.1.weight", [ff_dim, enc_dim])
        queue(f"{prefix}.ff1.net.1.bias", [ff_dim])
        queue(f"{prefix}.ff1.net.4.weight", [enc_dim, ff_dim])
        queue(f"{prefix}.ff1.net.4.bias", [enc_dim])

        # MHSA — RoPE-based uses separate Wq/Wk/Wv; legacy uses fused in_proj
        if use_rope:
            attn_prefix = f"{prefix}.mhsa"
            queue(f"{attn_prefix}.norm.weight", [enc_dim])
            queue(f"{attn_prefix}.norm.bias", [enc_dim])

            wq = state[f"{attn_prefix}.wq.weight"]  # [D, D]
            wk = state[f"{attn_prefix}.wk.weight"]  # [D, D]
            wv = state[f"{attn_prefix}.wv.weight"]  # [D, D]
            fused_qkv = torch.cat([wq, wk, wv], dim=0)   # [3D, D]
            fused_bias = torch.zeros(3 * enc_dim)
            total_weights += fused_qkv.numel() + fused_bias.numel()
            tensors.extend([fused_qkv, fused_bias])

            queue(f"{attn_prefix}.wo.weight", [enc_dim, enc_dim])
            wo_bias = torch.zeros(enc_dim)
            total_weights += wo_bias.numel()
            tensors.append(wo_bias)
        else:
            queue(f"{prefix}.mhsa.norm.weight", [enc_dim])
            queue(f"{prefix}.mhsa.norm.bias", [enc_dim])
            queue(f"{prefix}.mhsa.attn.in_proj_weight", [3 * enc_dim, enc_dim])
            queue(f"{prefix}.mhsa.attn.in_proj_bias", [3 * enc_dim])
            queue(f"{prefix}.mhsa.attn.out_proj.weight", [enc_dim, enc_dim])
            queue(f"{prefix}.mhsa.attn.out_proj.bias", [enc_dim])

        # Conv module
        queue(f"{prefix}.conv.norm.weight", [enc_dim])
        queue(f"{prefix}.conv.norm.bias", [enc_dim])
        queue(f"{prefix}.conv.pointwise1.weight")  # [2D, D, 1] → flatten
        queue(f"{prefix}.conv.pointwise1.bias", [2 * enc_dim])
        queue(f"{prefix}.conv.depthwise.weight")   # [D, 1, K] → reshape [D, K]
        queue(f"{prefix}.conv.depthwise.bias", [enc_dim])
        queue(f"{prefix}.conv.batch_norm.weight", [enc_dim])
        queue(f"{prefix}.conv.batch_norm.bias", [enc_dim])
        queue(f"{prefix}.conv.batch_norm.running_mean", [enc_dim])
        queue(f"{prefix}.conv.batch_norm.running_var", [enc_dim])
        queue(f"{prefix}.conv.pointwise2.weight")  # [D, D, 1]
        queue(f"{prefix}.conv.pointwise2.bias", [enc_dim])

        # FF2
        queue(f"{prefix}.ff2.net.0.weight", [enc_dim])
        queue(f"{prefix}.ff2.net.0.bias", [enc_dim])
        queue(f"{prefix}.ff2.net.1.weight", [ff_dim, enc_dim])
        queue(f"{prefix}.ff2.net.1.bias", [ff_dim])
        queue(f"{prefix}.ff2.net.4.weight", [enc_dim, ff_dim])
        queue(f"{prefix}.ff2.net.4.bias", [enc_dim])

        # Final LayerNorm
        queue(f"{prefix}.norm.weight", [enc_dim])
        queue(f"{prefix}.norm.bias", [enc_dim])

    # Adapter: LayerNorm → Linear → SiLU (no dropout in inference)
    queue("adapter.0.weight", [enc_dim])
    queue("adapter.0.bias", [enc_dim])
    queue("adapter.1.weight", [enc_dim, enc_dim])
    queue("adapter.1.bias", [enc_dim])

    # CTC projection
    queue("ctc_proj.weight", [text_vocab, enc_dim])
    queue("ctc_proj.bias", [text_vocab])

    print(f"Exporting Sonata STT CTC model:")
    print(f"  enc_dim={enc_dim}, n_layers={n_layers}, n_heads={n_heads}")
    print(f"  n_mels={n_mels}, conv_kernel={conv_kernel}, vocab={text_vocab}")
    print(f"  Total weights: {total_weights} ({total_weights * 4 / 1e6:.1f} MB)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        # Header: 10 × uint32 = 40 bytes
        f.write(struct.pack("<10I",
                            MAGIC, VERSION, enc_dim, n_layers, n_heads,
                            n_mels, conv_kernel, text_vocab, total_weights, 0))

        written = 0
        for t in tensors:
            written += write_tensor(f, t)

        assert written == total_weights, f"Expected {total_weights}, wrote {written}"

    file_size = Path(output_path).stat().st_size
    print(f"  Written: {output_path} ({file_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Sonata CTC to .cstt_sonata")
    parser.add_argument("--ckpt", required=True, help="Path to sonata_ctc checkpoint")
    parser.add_argument("--output", required=True, help="Output .cstt_sonata path")
    args = parser.parse_args()
    export(args.ckpt, args.output)
