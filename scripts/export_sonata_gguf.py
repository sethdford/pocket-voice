#!/usr/bin/env python3
"""Export Sonata LM safetensors to GGUF format for quantized inference.

Converts the Sonata Semantic LM (241M Llama-style transformer) from FP16
safetensors to GGUF with optional quantization. The GGUF can be loaded by
candle's quantized inference path or any GGUF-compatible runtime.

Architecture mapping (Sonata → GGUF/Llama conventions):
  text_emb.weight           → token_embd.weight
  semantic_emb.weight       → token_embd_semantic.weight  (custom)
  layers.N.attn_norm.weight → blk.N.attn_norm.weight
  layers.N.attn.wq.weight   → blk.N.attn_q.weight
  layers.N.attn.wk.weight   → blk.N.attn_k.weight
  layers.N.attn.wv.weight   → blk.N.attn_v.weight
  layers.N.attn.wo.weight   → blk.N.attn_output.weight
  layers.N.ffn_norm.weight  → blk.N.ffn_norm.weight
  layers.N.ffn.gate.weight  → blk.N.ffn_gate.weight
  layers.N.ffn.up.weight    → blk.N.ffn_up.weight
  layers.N.ffn.down.weight  → blk.N.ffn_down.weight
  output_norm.weight        → output_norm.weight
  semantic_head.weight      → output.weight

Usage:
  python scripts/export_sonata_gguf.py \\
    --weights models/sonata/sonata_lm.safetensors \\
    --config  models/sonata/sonata_lm_config.json \\
    --output  models/sonata/sonata_lm.gguf \\
    --dtype   f16          # f32, f16, q8_0, q4_0
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors not installed. Run: pip install safetensors", file=sys.stderr)
    sys.exit(1)


# GGUF magic and version
GGUF_MAGIC = 0x46475547  # "GGUF" little-endian
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING  = 8
GGUF_TYPE_UINT64  = 10

# GGML tensor types
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8

GGML_TYPE_SIZES = {
    GGML_TYPE_F32:  4,
    GGML_TYPE_F16:  2,
    GGML_TYPE_Q4_0: 0.5 + 2/32,  # 4 bits + scale per block of 32
    GGML_TYPE_Q8_0: 1.0 + 2/32,  # 8 bits + scale per block of 32
}


class GGUFWriter:
    """Minimal GGUF file writer."""

    def __init__(self):
        self.kv_data = []
        self.tensors = []

    def add_string(self, key: str, value: str):
        self.kv_data.append((key, GGUF_TYPE_STRING, value))

    def add_uint32(self, key: str, value: int):
        self.kv_data.append((key, GGUF_TYPE_UINT32, value))

    def add_int32(self, key: str, value: int):
        self.kv_data.append((key, GGUF_TYPE_INT32, value))

    def add_float32(self, key: str, value: float):
        self.kv_data.append((key, GGUF_TYPE_FLOAT32, value))

    def add_uint64(self, key: str, value: int):
        self.kv_data.append((key, GGUF_TYPE_UINT64, value))

    def add_tensor(self, name: str, data: np.ndarray, ggml_type: int):
        self.tensors.append((name, data, ggml_type))

    def _write_string(self, f, s: str):
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

    def _write_kv(self, f, key: str, vtype: int, value):
        self._write_string(f, key)
        f.write(struct.pack("<I", vtype))
        if vtype == GGUF_TYPE_STRING:
            self._write_string(f, value)
        elif vtype == GGUF_TYPE_UINT32:
            f.write(struct.pack("<I", value))
        elif vtype == GGUF_TYPE_INT32:
            f.write(struct.pack("<i", value))
        elif vtype == GGUF_TYPE_FLOAT32:
            f.write(struct.pack("<f", value))
        elif vtype == GGUF_TYPE_UINT64:
            f.write(struct.pack("<Q", value))

    def write(self, path: str):
        with open(path, "wb") as f:
            # Header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.kv_data)))

            # KV pairs
            for key, vtype, value in self.kv_data:
                self._write_kv(f, key, vtype, value)

            # Tensor info (names, shapes, types, offsets)
            # We need to compute offsets; tensors are packed sequentially
            # after the header with 32-byte alignment
            tensor_data_parts = []
            for name, data, ggml_type in self.tensors:
                if ggml_type == GGML_TYPE_F32:
                    raw = data.astype(np.float32).tobytes()
                elif ggml_type == GGML_TYPE_F16:
                    raw = data.astype(np.float16).tobytes()
                elif ggml_type == GGML_TYPE_Q8_0:
                    raw = quantize_q8_0(data)
                elif ggml_type == GGML_TYPE_Q4_0:
                    raw = quantize_q4_0(data)
                else:
                    raw = data.astype(np.float32).tobytes()
                tensor_data_parts.append(raw)

            offset = 0
            for i, (name, data, ggml_type) in enumerate(self.tensors):
                # Align to 32 bytes
                pad = (32 - (offset % 32)) % 32
                offset += pad

                self._write_string(f, name)
                n_dims = len(data.shape)
                f.write(struct.pack("<I", n_dims))
                for dim in data.shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", ggml_type))
                f.write(struct.pack("<Q", offset))

                offset += len(tensor_data_parts[i])

            # Pad to 32-byte alignment before tensor data
            current = f.tell()
            pad = (32 - (current % 32)) % 32
            f.write(b"\x00" * pad)

            # Tensor data
            for i, raw in enumerate(tensor_data_parts):
                current = f.tell()
                pad = (32 - (current % 32)) % 32
                f.write(b"\x00" * pad)
                f.write(raw)


def quantize_q8_0(data: np.ndarray) -> bytes:
    """Quantize float32 array to Q8_0 format (block size 32)."""
    flat = data.astype(np.float32).flatten()
    n = len(flat)
    # Pad to multiple of 32
    if n % 32 != 0:
        flat = np.concatenate([flat, np.zeros(32 - (n % 32), dtype=np.float32)])
    n = len(flat)
    blocks = flat.reshape(-1, 32)
    result = bytearray()
    for block in blocks:
        amax = np.max(np.abs(block))
        scale = amax / 127.0 if amax > 0 else 0.0
        result += struct.pack("<e", np.float16(scale))  # f16 scale
        if scale > 0:
            quant = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
        else:
            quant = np.zeros(32, dtype=np.int8)
        result += quant.tobytes()
    return bytes(result)


def quantize_q4_0(data: np.ndarray) -> bytes:
    """Quantize float32 array to Q4_0 format (block size 32)."""
    flat = data.astype(np.float32).flatten()
    n = len(flat)
    if n % 32 != 0:
        flat = np.concatenate([flat, np.zeros(32 - (n % 32), dtype=np.float32)])
    blocks = flat.reshape(-1, 32)
    result = bytearray()
    for block in blocks:
        amax = np.max(np.abs(block))
        scale = amax / 7.0 if amax > 0 else 0.0
        result += struct.pack("<e", np.float16(scale))  # f16 scale
        if scale > 0:
            quant = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
        else:
            quant = np.zeros(32, dtype=np.int8)
        # Pack two 4-bit values per byte
        packed = bytearray(16)
        for j in range(16):
            lo = int(quant[j * 2]) & 0x0F
            hi = int(quant[j * 2 + 1]) & 0x0F
            packed[j] = (hi << 4) | lo
        result += bytes(packed)
    return bytes(result)


# Weight name mapping: safetensors → GGUF
WEIGHT_MAP = {
    "text_emb.weight": "token_embd.weight",
    "semantic_emb.weight": "token_embd_semantic.weight",
    "output_norm.weight": "output_norm.weight",
    "semantic_head.weight": "output.weight",
}

LAYER_WEIGHT_MAP = {
    "attn_norm.weight":  "attn_norm.weight",
    "attn.wq.weight":    "attn_q.weight",
    "attn.wk.weight":    "attn_k.weight",
    "attn.wv.weight":    "attn_v.weight",
    "attn.wo.weight":    "attn_output.weight",
    "ffn_norm.weight":   "ffn_norm.weight",
    "ffn.gate.weight":   "ffn_gate.weight",
    "ffn.up.weight":     "ffn_up.weight",
    "ffn.down.weight":   "ffn_down.weight",
}


def map_weight_name(name: str) -> str | None:
    if name in WEIGHT_MAP:
        return WEIGHT_MAP[name]

    # Layer weights: layers.N.xxx → blk.N.xxx
    if name.startswith("layers."):
        parts = name.split(".", 2)
        if len(parts) == 3:
            layer_idx = parts[1]
            suffix = parts[2]
            if suffix in LAYER_WEIGHT_MAP:
                return f"blk.{layer_idx}.{LAYER_WEIGHT_MAP[suffix]}"

    return None


def main():
    parser = argparse.ArgumentParser(description="Export Sonata LM to GGUF")
    parser.add_argument("--weights", required=True, help="Path to safetensors file")
    parser.add_argument("--config", help="Path to JSON config (optional)")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--dtype", default="f16", choices=["f32", "f16", "q8_0", "q4_0"],
                        help="Output tensor type (default: f16)")
    args = parser.parse_args()

    ggml_type = {
        "f32": GGML_TYPE_F32, "f16": GGML_TYPE_F16,
        "q8_0": GGML_TYPE_Q8_0, "q4_0": GGML_TYPE_Q4_0,
    }[args.dtype]

    # Load config
    config = {
        "d_model": 1024, "n_layers": 16, "n_heads": 16, "n_kv_heads": 4,
        "d_ff": 2560, "max_seq_len": 4096, "text_vocab_size": 32000,
        "semantic_vocab_size": 4096, "n_special_tokens": 4,
        "rope_theta": 10000.0, "norm_eps": 1e-5,
    }
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            user_cfg = json.load(f)
        config.update(user_cfg)
        if "ffn_mult" in user_cfg and "d_ff" not in user_cfg:
            raw = int(config["d_model"] * user_cfg["ffn_mult"])
            config["d_ff"] = raw - (raw % 256)

    print(f"Sonata LM → GGUF converter")
    print(f"  Weights: {args.weights}")
    print(f"  Config:  d={config['d_model']}, L={config['n_layers']}, "
          f"H={config['n_heads']}, KV={config['n_kv_heads']}, FF={config['d_ff']}")
    print(f"  Output:  {args.output} ({args.dtype})")

    # Build GGUF
    writer = GGUFWriter()

    # Metadata
    writer.add_string("general.architecture", "sonata_lm")
    writer.add_string("general.name", "Sonata Semantic LM")
    writer.add_uint32("sonata_lm.context_length", config["max_seq_len"])
    writer.add_uint32("sonata_lm.embedding_length", config["d_model"])
    writer.add_uint32("sonata_lm.block_count", config["n_layers"])
    writer.add_uint32("sonata_lm.attention.head_count", config["n_heads"])
    writer.add_uint32("sonata_lm.attention.head_count_kv", config["n_kv_heads"])
    writer.add_uint32("sonata_lm.feed_forward_length", config["d_ff"])
    writer.add_float32("sonata_lm.rope.freq_base", config["rope_theta"])
    writer.add_float32("sonata_lm.attention.layer_norm_rms_epsilon", config["norm_eps"])
    writer.add_uint32("sonata_lm.text_vocab_size", config["text_vocab_size"])
    writer.add_uint32("sonata_lm.semantic_vocab_size", config["semantic_vocab_size"])
    writer.add_uint32("sonata_lm.n_special_tokens", config["n_special_tokens"])

    # Load and convert tensors
    total_params = 0
    skipped = []
    with safe_open(args.weights, framework="numpy") as f:
        for name in f.keys():
            gguf_name = map_weight_name(name)
            if gguf_name is None:
                skipped.append(name)
                continue

            tensor = f.get_tensor(name)
            total_params += tensor.size
            tensor_f32 = tensor.astype(np.float32) if tensor.dtype != np.float32 else tensor

            # Norm weights and embeddings stay in f16/f32 (don't quantize)
            is_norm = "norm" in gguf_name
            is_embd = "embd" in gguf_name
            if (is_norm or is_embd) and ggml_type in (GGML_TYPE_Q4_0, GGML_TYPE_Q8_0):
                actual_type = GGML_TYPE_F16
            else:
                actual_type = ggml_type

            writer.add_tensor(gguf_name, tensor_f32, actual_type)
            print(f"  {name:45s} → {gguf_name:35s} {list(tensor.shape)} ({args.dtype})")

    if skipped:
        print(f"\n  Skipped {len(skipped)} non-Llama tensors: {', '.join(skipped[:5])}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Writing GGUF to {args.output}...")
    writer.write(args.output)

    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"  Done! {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
