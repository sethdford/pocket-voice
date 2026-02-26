#!/usr/bin/env python3
"""
Extract weights from Silero VAD v5 ONNX model into .nvad binary format
for the pure C native_vad engine.

Usage:
    python scripts/extract_silero_weights.py models/silero_vad.onnx models/silero_vad.nvad

Requires: pip install onnx numpy

The Silero VAD ONNX model uses an If control-flow node to branch on sample rate.
We extract from the 16kHz branch (then_branch when Equal(sr, 16000) is True).
"""

import sys
import struct
import numpy as np

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.onnx> <output.nvad>")
        sys.exit(1)

    onnx_path = sys.argv[1]
    nvad_path = sys.argv[2]

    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        print("Error: 'onnx' package required. Install with: pip install onnx")
        sys.exit(1)

    model = onnx.load(onnx_path)
    print(f"Loaded ONNX model: {onnx_path}")

    # ── Find the 16kHz branch inside the If node ────────────────────────
    if_node = None
    for node in model.graph.node:
        if node.op_type == "If":
            if_node = node
            break
    if not if_node:
        print("Error: No If node found in model graph")
        sys.exit(1)

    # The model does Equal(sr, 16000). then_branch=16kHz, else_branch=8kHz.
    target_graph = None
    for attr in if_node.attribute:
        if attr.name == "then_branch":
            target_graph = attr.g
            break
    if not target_graph:
        print("Error: Could not find then_branch (16kHz) in If node")
        sys.exit(1)

    # Extract all Constant tensors from the 16kHz subgraph
    weights = {}
    for node in target_graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    t = numpy_helper.to_array(a.t)
                    name = node.output[0]
                    # Strip the long prefix for readability
                    short = name.split("__")[-1] if "__" in name else name
                    weights[short] = t
                    if t.size > 10:
                        print(f"  {short:50s} {str(t.shape):20s} {t.dtype}")

    print(f"\nFound {len(weights)} constant tensors in 16kHz branch")

    # ── Extract weights by name ──────────────────────────────────────────

    def get(pattern, expected_shape=None):
        matches = [(k, v) for k, v in weights.items() if pattern in k]
        if not matches:
            raise KeyError(f"Weight '{pattern}' not found. Available: {list(weights.keys())[:10]}")
        name, w = matches[0]
        if expected_shape and list(w.shape) != expected_shape:
            raise ValueError(f"{name}: expected {expected_shape}, got {list(w.shape)}")
        return name, w

    # STFT basis [258, 1, 256]
    stft_name, stft_basis = get("stft.forward_basis_buffer", [258, 1, 256])
    stft_basis = stft_basis.reshape(258, 256).astype(np.float32)
    print(f"\n  STFT basis:     {stft_name} → [258, 256]")

    # Encoder conv layers
    enc_specs = [
        (129, 128, 3),  # layer 0
        (128,  64, 3),  # layer 1
        ( 64,  64, 3),  # layer 2
        ( 64, 128, 3),  # layer 3
    ]
    enc_weights = []
    enc_biases = []
    for i, (in_ch, out_ch, k) in enumerate(enc_specs):
        wn, w = get(f"encoder.{i}.reparam_conv.weight", [out_ch, in_ch, k])
        bn, b = get(f"encoder.{i}.reparam_conv.bias", [out_ch])
        w = w.reshape(out_ch, in_ch * k).astype(np.float32)
        b = b.astype(np.float32)
        enc_weights.append(w)
        enc_biases.append(b)
        print(f"  Encoder[{i}]:     w=[{out_ch}, {in_ch*k}]  b=[{out_ch}]")

    # LSTM weights
    _, lstm_wi = get("decoder.rnn.weight_ih", [512, 128])
    _, lstm_wh = get("decoder.rnn.weight_hh", [512, 128])
    _, lstm_bi = get("decoder.rnn.bias_ih", [512])
    _, lstm_bh = get("decoder.rnn.bias_hh", [512])
    lstm_wi = lstm_wi.astype(np.float32)
    lstm_wh = lstm_wh.astype(np.float32)
    lstm_bias = (lstm_bi + lstm_bh).astype(np.float32)
    print(f"  LSTM:           Wi=[512,128]  Wh=[512,128]  bias=[512] (ih+hh combined)")

    # Output projection
    _, out_w = get("decoder.decoder.2.weight")
    out_w = out_w.reshape(128).astype(np.float32)

    # Find the output bias — it's a scalar [1]
    out_b_val = np.float32(0.0)
    bias_matches = [(k, v) for k, v in weights.items()
                    if "decoder.decoder.2.bias" in k]
    if bias_matches:
        _, out_b_arr = bias_matches[0]
        out_b_val = out_b_arr.reshape(1).astype(np.float32)
        print(f"  Output:         w=[128]  b=[1]")
    else:
        # Some exports might not have an explicit bias — check for small [1] constants
        for k, v in weights.items():
            if list(v.shape) == [1] and v.dtype == np.float32 and abs(float(v)) < 10:
                if "decoder" in k and "bias" in k.lower():
                    out_b_val = v.reshape(1).astype(np.float32)
                    print(f"  Output:         w=[128]  b=[1] (found as {k})")
                    break
        else:
            print(f"  Output:         w=[128]  b=0.0 (no explicit bias found)")
            out_b_val = np.array([0.0], dtype=np.float32)

    # ── Write .nvad binary ───────────────────────────────────────────────

    NVAD_MAGIC = 0x4441564E  # "NVAD" LE

    with open(nvad_path, "wb") as f:
        # Header (36 bytes)
        f.write(struct.pack("<IIIIIIIII",
            NVAD_MAGIC,  # magic
            1,           # version
            16000,       # sample_rate
            256,         # filter_length
            128,         # hop_length
            129,         # n_freq_bins
            4,           # n_enc_layers
            128,         # lstm_hidden
            64,          # context_size
        ))

        # Encoder layer descriptors (4 × 12 bytes)
        enc_strides = [1, 2, 2, 1]
        for i, (in_ch, out_ch, _) in enumerate(enc_specs):
            f.write(struct.pack("<III", in_ch, out_ch, enc_strides[i]))

        # Weight data (raw float32)
        total_params = 0

        stft_basis.tofile(f)
        total_params += stft_basis.size

        for i in range(4):
            enc_weights[i].tofile(f)
            enc_biases[i].tofile(f)
            total_params += enc_weights[i].size + enc_biases[i].size

        lstm_wi.tofile(f)
        lstm_wh.tofile(f)
        lstm_bias.tofile(f)
        total_params += lstm_wi.size + lstm_wh.size + lstm_bias.size

        out_w.tofile(f)
        out_b_val.tofile(f)
        total_params += out_w.size + out_b_val.size

    import os
    file_size = os.path.getsize(nvad_path)
    print(f"\nWrote {nvad_path}: {total_params:,} params, {file_size:,} bytes ({file_size/1024:.1f} KB)")

if __name__ == "__main__":
    main()
