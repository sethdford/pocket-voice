#!/usr/bin/env python3
"""
quantize_int8.py — Quantize a .cstt fp32 model to INT8 per-channel symmetric.

Output format: .cstt with dtype=2 (INT8), each weight tensor stored as:
  - int8 data: [N * K] bytes (per-channel symmetric quantization)
  - fp32 scales: [N] floats (one scale per output channel)

Per-channel symmetric quantization: W_int8[n,k] = round(W_fp32[n,k] / scale[n])
where scale[n] = max(|W_fp32[n,:]|) / 127

This reduces model size by ~4x vs fp32 (and ~2x vs fp16) with minimal accuracy loss
for the large GEMM weights. Biases, norms, and BN stats remain fp32.
"""

import argparse
import struct
import sys
import numpy as np
from pathlib import Path


# CSTTHeader format: 24 uint32 values
HEADER_FMT = "<24I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
CSTT_MAGIC = 0x54545343

DTYPE_FP32 = 0
DTYPE_FP16 = 1
DTYPE_INT8 = 2


def read_fp32_tensor(data, offset, count):
    end = offset + count * 4
    return np.frombuffer(data[offset:end], dtype=np.float32).copy(), end


def read_fp16_tensor(data, offset, count):
    end = offset + count * 2
    vals = np.frombuffer(data[offset:end], dtype=np.float16).astype(np.float32)
    return vals, end


def quantize_symmetric_per_channel(weights, out_channels, in_features):
    """Per-channel symmetric INT8 quantization.
    
    weights: [out_channels, in_features] fp32
    Returns: (int8_data, scales) where scales are [out_channels] fp32
    """
    w = weights.reshape(out_channels, in_features)
    abs_max = np.abs(w).max(axis=1)
    abs_max = np.maximum(abs_max, 1e-8)
    scales = abs_max / 127.0
    w_int8 = np.clip(np.round(w / scales[:, None]), -128, 127).astype(np.int8)
    return w_int8.flatten(), scales.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Quantize .cstt model to INT8")
    parser.add_argument("input", help="Input .cstt file (fp32 or fp16)")
    parser.add_argument("-o", "--output", help="Output .cstt file (default: input.int8.cstt)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    output_path = args.output or str(input_path).replace(".cstt", ".int8.cstt")

    with open(input_path, "rb") as f:
        data = f.read()

    if len(data) < HEADER_SIZE:
        print("ERROR: File too small for header")
        sys.exit(1)

    header_vals = list(struct.unpack(HEADER_FMT, data[:HEADER_SIZE]))
    magic = header_vals[0]
    if magic != CSTT_MAGIC:
        print(f"ERROR: Invalid magic 0x{magic:08X}")
        sys.exit(1)

    src_dtype = header_vals[14]
    n_layers = header_vals[2]
    d_model = header_vals[3]
    n_heads = header_vals[4]
    ff_mult = header_vals[5]
    conv_kernel = header_vals[6]
    vocab_size = header_vals[7]
    n_mels = header_vals[8]
    flags = header_vals[15]
    sub_type = header_vals[16]
    n_sub_convs = header_vals[17]
    sub_feat_in = header_vals[18]
    sub_conv_kernel = header_vals[19]

    ff_dim = d_model * ff_mult
    has_rel_pe = bool(flags & (1 << 2))
    has_tdt = bool(flags & (1 << 6))

    print(f"Input: {input_path}")
    print(f"  dtype={'fp32' if src_dtype == 0 else 'fp16'}, "
          f"{n_layers} layers, d={d_model}, heads={n_heads}, "
          f"ff_mult={ff_mult}, conv_k={conv_kernel}, vocab={vocab_size}")
    print(f"  rel_pe={has_rel_pe}, tdt={has_tdt}")

    if src_dtype == DTYPE_INT8:
        print("ERROR: Model is already INT8")
        sys.exit(1)

    def read_tensor(offset, count):
        if src_dtype == DTYPE_FP16:
            return read_fp16_tensor(data, offset, count)
        return read_fp32_tensor(data, offset, count)

    out_buf = bytearray()

    # Quantize: large GEMM weights become int8+scales, everything else stays fp32
    offset = HEADER_SIZE
    n_quant = 0
    n_kept = 0
    total_saved = 0

    def write_fp32(arr):
        out_buf.extend(arr.astype(np.float32).tobytes())

    def write_int8_quantized(arr, out_ch, in_feat):
        nonlocal n_quant, total_saved
        q_data, scales = quantize_symmetric_per_channel(arr, out_ch, in_feat)
        out_buf.extend(q_data.tobytes())
        out_buf.extend(scales.tobytes())
        n_quant += 1
        orig_bytes = out_ch * in_feat * (2 if src_dtype == DTYPE_FP16 else 4)
        new_bytes = out_ch * in_feat + out_ch * 4  # int8 data + fp32 scales
        total_saved += orig_bytes - new_bytes

    def write_kept(arr):
        nonlocal n_kept
        write_fp32(arr)
        n_kept += 1

    # Read and quantize subsampling weights
    if sub_type == 0 and n_sub_convs == 0:  # Legacy Conv1D
        K = sub_conv_kernel if sub_conv_kernel > 0 else 3
        # Conv1: [D, n_mels, K]
        t, offset = read_tensor(offset, d_model * n_mels * K)
        write_int8_quantized(t, d_model, n_mels * K)
        t, offset = read_tensor(offset, d_model)
        write_kept(t)
        # Conv2: [D, D, K]
        t, offset = read_tensor(offset, d_model * d_model * K)
        write_int8_quantized(t, d_model, d_model * K)
        t, offset = read_tensor(offset, d_model)
        write_kept(t)
        # Linear proj
        t, offset = read_tensor(offset, d_model * d_model)
        write_int8_quantized(t, d_model, d_model)
        t, offset = read_tensor(offset, d_model)
        write_kept(t)
    else:
        # General format: descriptors are raw uint32 metadata
        desc_bytes = n_sub_convs * 5 * 4
        out_buf.extend(data[offset:offset + desc_bytes])
        descs = []
        for i in range(n_sub_convs):
            base = offset + i * 20
            c_in, c_out, k, s, g = struct.unpack("<5I", data[base:base + 20])
            descs.append((c_in, c_out, k, s, g))
        offset += desc_bytes

        for c_in, c_out, k, s, g in descs:
            ci = c_in // g
            K2 = k * k
            wcount = c_out * ci * K2
            t, offset = read_tensor(offset, wcount)
            write_int8_quantized(t, c_out, ci * K2)
            t, offset = read_tensor(offset, c_out)
            write_kept(t)

        # Linear projection
        t, offset = read_tensor(offset, d_model * sub_feat_in)
        write_int8_quantized(t, d_model, sub_feat_in)
        t, offset = read_tensor(offset, d_model)
        write_kept(t)

    # Read and quantize block weights
    for layer in range(n_layers):
        # FFN1 norm (keep fp32)
        for _ in range(2):  # norm_w, norm_b
            t, offset = read_tensor(offset, d_model)
            write_kept(t)
        # FFN1 up: [ff_dim, D] — quantize
        t, offset = read_tensor(offset, d_model * ff_dim)
        write_int8_quantized(t, ff_dim, d_model)
        t, offset = read_tensor(offset, ff_dim)  # bias
        write_kept(t)
        # FFN1 down: [D, ff_dim] — quantize
        t, offset = read_tensor(offset, ff_dim * d_model)
        write_int8_quantized(t, d_model, ff_dim)
        t, offset = read_tensor(offset, d_model)  # bias
        write_kept(t)

        # Attention norm (keep fp32)
        for _ in range(2):
            t, offset = read_tensor(offset, d_model)
            write_kept(t)
        # Q,K,V,Out projections: each [D, D] — quantize
        for _ in range(4):
            t, offset = read_tensor(offset, d_model * d_model)
            write_int8_quantized(t, d_model, d_model)
            t, offset = read_tensor(offset, d_model)
            write_kept(t)

        if has_rel_pe:
            # linear_pos: [D, D]
            t, offset = read_tensor(offset, d_model * d_model)
            write_int8_quantized(t, d_model, d_model)
            # pos_bias_u, pos_bias_v: [D]
            t, offset = read_tensor(offset, d_model)
            write_kept(t)
            t, offset = read_tensor(offset, d_model)
            write_kept(t)

        # Conv module: norm
        for _ in range(2):
            t, offset = read_tensor(offset, d_model)
            write_kept(t)
        # pw1: [2D, D]
        t, offset = read_tensor(offset, 2 * d_model * d_model)
        write_int8_quantized(t, 2 * d_model, d_model)
        t, offset = read_tensor(offset, 2 * d_model)
        write_kept(t)
        # dw: [D, K] — keep fp32 (small, depthwise)
        t, offset = read_tensor(offset, d_model * conv_kernel)
        write_kept(t)
        t, offset = read_tensor(offset, d_model)
        write_kept(t)
        # BN: gamma, beta, mean, var — all fp32
        for _ in range(4):
            t, offset = read_tensor(offset, d_model)
            write_kept(t)
        # pw2: [D, D]
        t, offset = read_tensor(offset, d_model * d_model)
        write_int8_quantized(t, d_model, d_model)
        t, offset = read_tensor(offset, d_model)
        write_kept(t)

        # FFN2 norm
        for _ in range(2):
            t, offset = read_tensor(offset, d_model)
            write_kept(t)
        # FFN2 up: [ff_dim, D]
        t, offset = read_tensor(offset, d_model * ff_dim)
        write_int8_quantized(t, ff_dim, d_model)
        t, offset = read_tensor(offset, ff_dim)
        write_kept(t)
        # FFN2 down: [D, ff_dim]
        t, offset = read_tensor(offset, ff_dim * d_model)
        write_int8_quantized(t, d_model, ff_dim)
        t, offset = read_tensor(offset, d_model)
        write_kept(t)

        # Final norm
        for _ in range(2):
            t, offset = read_tensor(offset, d_model)
            write_kept(t)

    # CTC head: [vocab, D]
    t, offset = read_tensor(offset, d_model * vocab_size)
    write_int8_quantized(t, vocab_size, d_model)
    t, offset = read_tensor(offset, vocab_size)
    write_kept(t)

    # TDT weights: copy verbatim (no quantization for now)
    if has_tdt and offset < len(data):
        out_buf.extend(data[offset:])

    # Update header: dtype = 2 (INT8)
    header_vals[14] = DTYPE_INT8
    new_header = struct.pack(HEADER_FMT, *header_vals)

    with open(output_path, "wb") as f:
        f.write(new_header)
        f.write(out_buf)

    orig_size = len(data)
    new_size = HEADER_SIZE + len(out_buf)
    ratio = orig_size / new_size if new_size > 0 else 0

    print(f"\nQuantization complete:")
    print(f"  {n_quant} weight tensors quantized to INT8")
    print(f"  {n_kept} tensors kept as fp32 (norms, biases, BN)")
    print(f"  Original: {orig_size / 1024 / 1024:.1f} MB")
    print(f"  Quantized: {new_size / 1024 / 1024:.1f} MB")
    print(f"  Compression: {ratio:.1f}x")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
