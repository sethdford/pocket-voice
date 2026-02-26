#!/usr/bin/env python3
"""Export trained Sonata STT Refiner model (semantic tokens → text) to .cref binary for C inference.

Weight layout: semantic_emb → encoder_pos → encoder_layers → encoder_norm
             → text_emb → decoder_layers → decoder_norm → output_proj

Usage:
  python scripts/export_sonata_refiner.py \
    --ckpt train/checkpoints/stt/refiner/sonata_refiner_final.pt \
    --output models/sonata/sonata_refiner.cref
"""

import argparse
import struct
import sys
from pathlib import Path

# Allow imports from train/sonata
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRAIN_SONATA = REPO_ROOT / "train" / "sonata"
sys.path.insert(0, str(TRAIN_SONATA))

import torch


MAGIC = b"CREF"
VERSION = 2  # v2: header includes enc_d_ff, dec_d_ff


def write_tensor(f, tensor):
    """Write a float32 tensor to file in row-major order."""
    data = tensor.detach().cpu().float().numpy()
    f.write(data.tobytes())
    return data.size


def export(ckpt_path: str, output_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    cfg = ckpt.get("config", {})

    # Config with defaults from RefinerConfig
    semantic_vocab_size = int(cfg.get("semantic_vocab_size", 32768))
    text_vocab_size = int(cfg.get("text_vocab_size", 4096))
    enc_d_model = int(cfg.get("enc_d_model", 512))
    enc_n_layers = int(cfg.get("enc_n_layers", 4))
    enc_n_heads = int(cfg.get("enc_n_heads", 8))
    dec_d_model = int(cfg.get("dec_d_model", 512))
    dec_n_layers = int(cfg.get("dec_n_layers", 4))
    dec_n_heads = int(cfg.get("dec_n_heads", 8))
    dec_n_kv_heads = int(cfg.get("dec_n_kv_heads", 4))
    max_audio_len = int(cfg.get("max_audio_len", 2048))

    enc_d_ff = int(enc_d_model * cfg.get("enc_ff_mult", 4.0))
    dec_d_ff = int(dec_d_model * cfg.get("dec_ff_mult", 4.0))
    dec_head_dim = dec_d_model // dec_n_heads

    sem_emb_size = semantic_vocab_size + 4
    text_emb_size = text_vocab_size + 4

    total_weights = 0
    tensors = []

    def queue(key: str, expected_shape=None):
        nonlocal total_weights
        t = state[key]
        if expected_shape:
            assert t.shape == torch.Size(expected_shape), (
                f"{key}: expected {expected_shape}, got {list(t.shape)}"
            )
        n = t.numel()
        total_weights += n
        tensors.append(t)
        return n

    # 1. Semantic embedding
    queue("sem_emb.weight", [sem_emb_size, enc_d_model])

    # 2. Encoder positional embedding
    queue("sem_pos.weight", [max_audio_len, enc_d_model])

    # 3. Encoder layers (attn_norm → attn → ffn_norm → ffn)
    for l in range(enc_n_layers):
        prefix = f"enc_layers.{l}"
        queue(f"{prefix}.attn_norm.weight", [enc_d_model])
        queue(f"{prefix}.attn.in_proj_weight", [3 * enc_d_model, enc_d_model])
        queue(f"{prefix}.attn.in_proj_bias", [3 * enc_d_model])
        queue(f"{prefix}.attn.out_proj.weight", [enc_d_model, enc_d_model])
        queue(f"{prefix}.attn.out_proj.bias", [enc_d_model])
        queue(f"{prefix}.ffn_norm.weight", [enc_d_model])
        queue(f"{prefix}.ffn.0.weight", [enc_d_ff, enc_d_model])
        queue(f"{prefix}.ffn.0.bias", [enc_d_ff])
        queue(f"{prefix}.ffn.2.weight", [enc_d_model, enc_d_ff])
        queue(f"{prefix}.ffn.2.bias", [enc_d_model])

    # 4. Encoder norm
    queue("enc_norm.weight", [enc_d_model])

    # 5. Text embedding (decoder input)
    queue("text_emb.weight", [text_emb_size, dec_d_model])

    # 6. Decoder layers (self_attn → cross_attn → ffn)
    for l in range(dec_n_layers):
        prefix = f"dec_layers.{l}"
        queue(f"{prefix}.self_attn_norm.weight", [dec_d_model])
        queue(f"{prefix}.wq.weight", [dec_n_heads * dec_head_dim, dec_d_model])
        queue(f"{prefix}.wk.weight", [dec_n_kv_heads * dec_head_dim, dec_d_model])
        queue(f"{prefix}.wv.weight", [dec_n_kv_heads * dec_head_dim, dec_d_model])
        queue(f"{prefix}.wo.weight", [dec_d_model, dec_n_heads * dec_head_dim])
        queue(f"{prefix}.cross_norm.weight", [dec_d_model])
        queue(f"{prefix}.cross_q.weight", [dec_n_heads * dec_head_dim, dec_d_model])
        queue(f"{prefix}.cross_k.weight", [dec_n_kv_heads * dec_head_dim, enc_d_model])
        queue(f"{prefix}.cross_v.weight", [dec_n_kv_heads * dec_head_dim, enc_d_model])
        queue(f"{prefix}.cross_o.weight", [dec_d_model, dec_n_heads * dec_head_dim])
        queue(f"{prefix}.ffn_norm.weight", [dec_d_model])
        queue(f"{prefix}.ffn.0.weight", [dec_d_ff, dec_d_model])
        queue(f"{prefix}.ffn.0.bias", [dec_d_ff])
        queue(f"{prefix}.ffn.2.weight", [dec_d_model, dec_d_ff])
        queue(f"{prefix}.ffn.2.bias", [dec_d_model])

    # 7. Decoder norm
    queue("dec_norm.weight", [dec_d_model])

    # 8. Text projection head (output_proj)
    queue("output_proj.weight", [text_emb_size, dec_d_model])

    print("Exporting Sonata STT Refiner model:")
    print(f"  semantic_vocab_size={semantic_vocab_size}, text_vocab_size={text_vocab_size}")
    print(f"  encoder: {enc_n_layers}L × d={enc_d_model}, n_heads={enc_n_heads}")
    print(f"  decoder: {dec_n_layers}L × d={dec_d_model}, n_heads={dec_n_heads}, n_kv_heads={dec_n_kv_heads}")
    print(f"  max_audio_len={max_audio_len}")
    print(f"  Total weights: {total_weights} ({total_weights * 4 / 1e6:.1f} MB)")
    print(f"  Weight count: {len(tensors)} tensors")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as f:
        # Header: magic (4) + version (4) + 10 × uint32 (40) + n_weights (8) + enc_d_ff (4) + dec_d_ff (4) = 64 bytes (v2)
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(
            struct.pack(
                "<10I",
                semantic_vocab_size,
                text_vocab_size,
                enc_d_model,
                enc_n_layers,
                enc_n_heads,
                dec_d_model,
                dec_n_layers,
                dec_n_heads,
                dec_n_kv_heads,
                max_audio_len,
            )
        )
        f.write(struct.pack("<Q", total_weights))
        f.write(struct.pack("<2I", enc_d_ff, dec_d_ff))

        written = 0
        for t in tensors:
            written += write_tensor(f, t)

        assert written == total_weights, f"Expected {total_weights}, wrote {written}"

    file_size = output_file.stat().st_size
    print(f"  Written: {output_path} ({file_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Sonata STT Refiner to .cref binary"
    )
    parser.add_argument(
        "--ckpt",
        default="train/checkpoints/stt/refiner/sonata_refiner_final.pt",
        help="Path to sonata_refiner checkpoint",
    )
    parser.add_argument(
        "--output",
        default="models/sonata/sonata_refiner.cref",
        help="Output .cref path",
    )
    args = parser.parse_args()
    export(args.ckpt, args.output)
