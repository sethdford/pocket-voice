#!/usr/bin/env python3
"""
convert_nemo_tdt.py — Convert NeMo Parakeet TDT models to .cstt format.

The TDT (Token Duration Transducer) model has the same FastConformer encoder
as CTC models, plus a small prediction network (LSTM) and joint network.
This script reuses convert_nemo.py for the encoder and appends TDT decoder
weights after the CTC-like section.

The .cstt file layout for TDT models:
  [CSTTHeader]           (96 bytes, with sub_type flags indicating TDT)
  [Encoder weights]      (same as CTC format)
  [TDT decoder weights]  (prediction net + joint net, appended)

Usage:
    python scripts/convert_nemo_tdt.py nvidia/parakeet-tdt-0.6b-v2 \
        -o models/parakeet-tdt-0.6b.cstt --fp16
"""

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from convert_nemo import (
    _check_deps, extract_nemo, download_nemo, parse_encoder_config,
    parse_preprocessor_config, detect_subsampling_convs, get_vocab_from_tokenizer,
    get_weight, squeeze_conv1d, tensor_to_bytes, write_vocab,
    CSTT_MAGIC, CSTT_VERSION, CSTT_SUB_DW_STRIDING, CSTT_SUB_CONV1D, CSTT_SUB_CONV2D,
    CSTT_FLAG_HAS_BIAS, CSTT_FLAG_SLANEY_NORM, CSTT_FLAG_REL_PE, CSTT_FLAG_XSCALING,
)

CSTT_FLAG_TDT = 1 << 6  # Flag indicating TDT decoder is appended


def build_tdt_header(enc_config, pre_config, vocab_size, sub_type,
                     n_sub_convs, sub_feat_in, sub_conv_kernel,
                     fp16=False, tdt_info=None):
    """Build header with TDT flag set."""
    flags = (CSTT_FLAG_HAS_BIAS | CSTT_FLAG_SLANEY_NORM |
             CSTT_FLAG_REL_PE | CSTT_FLAG_TDT)
    if enc_config.get("xscaling", True):
        flags |= CSTT_FLAG_XSCALING

    reserved = [0, 0, 0, 0]
    if tdt_info:
        reserved[0] = tdt_info.get("pred_hidden", 640)
        reserved[1] = tdt_info.get("pred_layers", 2)
        reserved[2] = tdt_info.get("n_durations", 5)
        reserved[3] = tdt_info.get("joint_dim", 640)

    return struct.pack("<" + "I" * 24,
        CSTT_MAGIC,
        CSTT_VERSION,
        enc_config["n_layers"],
        enc_config["d_model"],
        enc_config["n_heads"],
        enc_config["ff_mult"],
        enc_config["conv_kernel"],
        vocab_size,
        pre_config["n_mels"],
        pre_config["sample_rate"],
        pre_config["hop_length"],
        pre_config["win_length"],
        pre_config["n_fft"],
        enc_config["subsampling_factor"],
        1 if fp16 else 0,
        flags,
        sub_type,
        n_sub_convs,
        sub_feat_in,
        sub_conv_kernel,
        *reserved,
    )


def convert_tdt_model(model_path_str, output_path, fp16=False):
    """Convert a NeMo TDT model to .cstt + .vocab."""
    import torch

    model_path = Path(model_path_str)

    print(f"Loading TDT model from {model_path}...")
    if model_path.is_file() or model_path.is_dir():
        config, sd, tok_path = extract_nemo(model_path)
    else:
        print(f"Downloading from HuggingFace: {model_path}...")
        local_path = download_nemo(str(model_path))
        config, sd, tok_path = extract_nemo(local_path)

    enc_config = parse_encoder_config(config)
    pre_config = parse_preprocessor_config(config)

    D = enc_config["d_model"]
    ff_dim = D * enc_config["ff_mult"]
    K = enc_config["conv_kernel"]
    n_layers = enc_config["n_layers"]
    n_heads = enc_config["n_heads"]
    n_mels = pre_config["n_mels"]

    print(f"Encoder: d_model={D}, layers={n_layers}, heads={n_heads}, conv_k={K}")

    # --- TDT decoder analysis ---
    embed_w = get_weight(sd, "decoder.prediction.embed.weight")
    vocab_size = embed_w.shape[0]
    pred_hidden = embed_w.shape[1]

    # Count LSTM layers
    pred_layers = 0
    while f"decoder.prediction.dec_rnn.lstm.weight_ih_l{pred_layers}" in sd:
        pred_layers += 1

    # Joint network
    joint_enc_w = get_weight(sd, "joint.enc.weight")
    joint_out_w = get_weight(sd, "joint.joint_net.2.weight")
    joint_dim = joint_enc_w.shape[0]
    total_out = joint_out_w.shape[0]
    n_durations = total_out - vocab_size

    # Extract duration values from model config (check multiple NeMo config paths)
    duration_values = None
    for path_fn in [
        lambda c: c.get("decoding", {}).get("durations"),
        lambda c: c.get("model_defaults", {}).get("tdt_durations"),
        lambda c: c.get("loss", {}).get("tdt_kwargs", {}).get("durations"),
        lambda c: c.get("model", {}).get("decoding", {}).get("durations"),
        lambda c: c.get("decoder", {}).get("durations"),
    ]:
        try:
            val = path_fn(config)
            if val is not None:
                duration_values = list(val)
                break
        except (AttributeError, TypeError):
            continue
    if duration_values is None:
        duration_values = list(range(n_durations))
        print(f"  WARNING: Duration values not found in config, using sequential: {duration_values}")

    assert len(duration_values) == n_durations, \
        f"Duration values count {len(duration_values)} != n_durations {n_durations}"

    print(f"TDT Decoder:")
    print(f"  Prediction: {pred_layers}-layer LSTM, hidden={pred_hidden}")
    print(f"  Embedding: {vocab_size} × {pred_hidden}")
    print(f"  Joint dim: {joint_dim}")
    print(f"  Output: {total_out} = {vocab_size} tokens + {n_durations} durations")
    print(f"  Duration values: {duration_values}")

    tdt_info = {
        "pred_hidden": pred_hidden,
        "pred_layers": pred_layers,
        "n_durations": n_durations,
        "joint_dim": joint_dim,
        "duration_values": duration_values,
    }

    # --- Projection + subsampling ---
    proj_w = get_weight(sd, "encoder.pre_encode.out.weight", required=False)
    proj_b = get_weight(sd, "encoder.pre_encode.out.bias", required=False)
    if proj_w is None:
        proj_w = get_weight(sd, "encoder.pre_encode.out.0.weight")
        proj_b = get_weight(sd, "encoder.pre_encode.out.0.bias")
    sub_feat_in = proj_w.shape[1]

    sub_convs = detect_subsampling_convs(sd, sub_feat_in, n_mels)
    has_depthwise = any(sc["groups"] > 1 for sc in sub_convs)
    if has_depthwise:
        sub_type = CSTT_SUB_DW_STRIDING
    elif sub_convs and sub_convs[0]["w"].ndim == 4:
        sub_type = CSTT_SUB_CONV2D
    else:
        sub_type = CSTT_SUB_CONV1D
    sub_conv_kernel = sub_convs[0]["kernel"] if sub_convs else 3

    # --- Build weight chunks ---
    chunks = []
    bpv = 2 if fp16 else 4

    import numpy as np

    def wb(t):
        return tensor_to_bytes(t, fp16=fp16)

    def wb_or_zero(key, size):
        """Write weight bytes if key exists, else write zeros."""
        t = get_weight(sd, key, required=False)
        if t is not None:
            return wb(t)
        return b"\x00" * (size * bpv)

    # Subsampling descriptors
    for sc in sub_convs:
        chunks.append(struct.pack("<5I", sc["c_in"], sc["c_out"],
                                   sc["kernel"], sc["stride"], sc["groups"]))
    for sc in sub_convs:
        chunks.append(wb(sc["w"]))
        if sc["b"] is not None:
            chunks.append(wb(sc["b"]))
        else:
            chunks.append(b"\x00" * (sc["c_out"] * bpv))

    chunks.append(wb(proj_w))
    chunks.append(wb(proj_b))

    # --- Encoder conformer blocks ---
    has_linear1 = f"encoder.layers.0.feed_forward1.linear1.weight" in sd
    has_conv_module = f"encoder.layers.0.conv_module.pointwise_conv1.weight" in sd
    ff_up = "linear1" if has_linear1 else "0"
    ff_down = "linear2" if has_linear1 else "2"
    conv_pre = "conv_module" if has_conv_module else "conv"

    print(f"Converting {n_layers} encoder blocks...")
    for i in range(n_layers):
        pre = f"encoder.layers.{i}"
        # FFN1
        chunks.append(wb(get_weight(sd, f"{pre}.norm_feed_forward1.weight")))
        chunks.append(wb(get_weight(sd, f"{pre}.norm_feed_forward1.bias")))
        chunks.append(wb(get_weight(sd, f"{pre}.feed_forward1.{ff_up}.weight")))
        chunks.append(wb_or_zero(f"{pre}.feed_forward1.{ff_up}.bias", ff_dim))
        chunks.append(wb(get_weight(sd, f"{pre}.feed_forward1.{ff_down}.weight")))
        chunks.append(wb_or_zero(f"{pre}.feed_forward1.{ff_down}.bias", D))
        # MHSA
        chunks.append(wb(get_weight(sd, f"{pre}.norm_self_att.weight")))
        chunks.append(wb(get_weight(sd, f"{pre}.norm_self_att.bias")))
        chunks.append(wb(get_weight(sd, f"{pre}.self_attn.linear_q.weight")))
        chunks.append(wb_or_zero(f"{pre}.self_attn.linear_q.bias", D))
        chunks.append(wb(get_weight(sd, f"{pre}.self_attn.linear_k.weight")))
        chunks.append(wb_or_zero(f"{pre}.self_attn.linear_k.bias", D))
        chunks.append(wb(get_weight(sd, f"{pre}.self_attn.linear_v.weight")))
        chunks.append(wb_or_zero(f"{pre}.self_attn.linear_v.bias", D))
        chunks.append(wb(get_weight(sd, f"{pre}.self_attn.linear_out.weight")))
        chunks.append(wb_or_zero(f"{pre}.self_attn.linear_out.bias", D))
        # Rel PE
        chunks.append(wb(get_weight(sd, f"{pre}.self_attn.linear_pos.weight")))
        chunks.append(wb(get_weight(sd, f"{pre}.self_attn.pos_bias_u")))
        chunks.append(wb(get_weight(sd, f"{pre}.self_attn.pos_bias_v")))
        # Conv
        chunks.append(wb(get_weight(sd, f"{pre}.norm_conv.weight")))
        chunks.append(wb(get_weight(sd, f"{pre}.norm_conv.bias")))
        pw1_w = squeeze_conv1d(get_weight(sd, f"{pre}.{conv_pre}.pointwise_conv1.weight"))
        chunks.append(wb(pw1_w))
        chunks.append(wb_or_zero(f"{pre}.{conv_pre}.pointwise_conv1.bias", 2 * D))
        dw_w = get_weight(sd, f"{pre}.{conv_pre}.depthwise_conv.weight")
        if dw_w.ndim == 3: dw_w = dw_w.squeeze(1)
        chunks.append(wb(dw_w))
        chunks.append(wb_or_zero(f"{pre}.{conv_pre}.depthwise_conv.bias", D))
        chunks.append(wb(get_weight(sd, f"{pre}.{conv_pre}.batch_norm.weight")))
        chunks.append(wb(get_weight(sd, f"{pre}.{conv_pre}.batch_norm.bias")))
        chunks.append(wb(get_weight(sd, f"{pre}.{conv_pre}.batch_norm.running_mean")))
        chunks.append(wb(get_weight(sd, f"{pre}.{conv_pre}.batch_norm.running_var")))
        pw2_w = squeeze_conv1d(get_weight(sd, f"{pre}.{conv_pre}.pointwise_conv2.weight"))
        chunks.append(wb(pw2_w))
        chunks.append(wb_or_zero(f"{pre}.{conv_pre}.pointwise_conv2.bias", D))
        # FFN2
        chunks.append(wb(get_weight(sd, f"{pre}.norm_feed_forward2.weight")))
        chunks.append(wb(get_weight(sd, f"{pre}.norm_feed_forward2.bias")))
        chunks.append(wb(get_weight(sd, f"{pre}.feed_forward2.{ff_up}.weight")))
        chunks.append(wb_or_zero(f"{pre}.feed_forward2.{ff_up}.bias", ff_dim))
        chunks.append(wb(get_weight(sd, f"{pre}.feed_forward2.{ff_down}.weight")))
        chunks.append(wb_or_zero(f"{pre}.feed_forward2.{ff_down}.bias", D))
        # Final norm
        chunks.append(wb(get_weight(sd, f"{pre}.norm_out.weight")))
        chunks.append(wb(get_weight(sd, f"{pre}.norm_out.bias")))

    # --- CTC head placeholder (TDT uses joint net instead) ---
    chunks.append(b"\x00" * (D * vocab_size * bpv))
    chunks.append(b"\x00" * (vocab_size * bpv))

    # --- TDT Decoder Weights ---
    print("Writing TDT decoder weights...")

    # TDT header: [pred_hidden, pred_layers, n_durations, joint_dim] + duration_values as uint32s
    tdt_header = struct.pack("<4I", pred_hidden, pred_layers, n_durations, joint_dim)
    for dv in duration_values:
        tdt_header += struct.pack("<I", dv)
    chunks.append(tdt_header)

    # Embedding: [vocab_size, pred_hidden]
    chunks.append(wb(embed_w))

    # LSTM layers
    for l in range(pred_layers):
        chunks.append(wb(get_weight(sd, f"decoder.prediction.dec_rnn.lstm.weight_ih_l{l}")))
        chunks.append(wb(get_weight(sd, f"decoder.prediction.dec_rnn.lstm.bias_ih_l{l}")))
        chunks.append(wb(get_weight(sd, f"decoder.prediction.dec_rnn.lstm.weight_hh_l{l}")))
        chunks.append(wb(get_weight(sd, f"decoder.prediction.dec_rnn.lstm.bias_hh_l{l}")))

    # Joint network
    chunks.append(wb(get_weight(sd, "joint.enc.weight")))
    chunks.append(wb(get_weight(sd, "joint.enc.bias")))
    chunks.append(wb(get_weight(sd, "joint.pred.weight")))
    chunks.append(wb(get_weight(sd, "joint.pred.bias")))
    chunks.append(wb(get_weight(sd, "joint.joint_net.2.weight")))
    chunks.append(wb(get_weight(sd, "joint.joint_net.2.bias")))

    # --- Write files ---
    if fp16:
        print("Using fp16 weight storage")
    header = build_tdt_header(enc_config, pre_config, vocab_size,
                               sub_type, len(sub_convs), sub_feat_in,
                               sub_conv_kernel, fp16=fp16, tdt_info=tdt_info)

    import os
    with open(output_path, "wb") as f:
        f.write(header)
        for chunk in chunks:
            f.write(chunk)
    total = os.path.getsize(output_path)
    print(f"Wrote {output_path} ({total / 1024 / 1024:.1f} MB)")

    vocab = None
    if tok_path:
        vocab = get_vocab_from_tokenizer(tok_path, config)
    if vocab is None:
        vocab = ["<blank>"] + [f"tok_{i}" for i in range(1, vocab_size)]
    while len(vocab) < vocab_size:
        vocab.append("<blank>")
    write_vocab(output_path, vocab)

    total_params = sum(sd[k].numel() for k in sd
                       if not k.endswith("num_batches_tracked"))
    print(f"\nConversion complete!")
    print(f"  Parameters: {total_params:,}")
    print(f"  Encoder: {n_layers} layers, d={D}")
    print(f"  TDT decoder: {pred_layers}-layer LSTM(h={pred_hidden})")
    print(f"  Joint: {joint_dim}→{total_out} ({vocab_size} tokens + {n_durations} durations)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeMo TDT models to .cstt format")
    parser.add_argument("model",
                        help="HuggingFace model ID or path to .nemo file")
    parser.add_argument("-o", "--output", default="model.cstt",
                        help="Output .cstt file path")
    parser.add_argument("--fp16", action="store_true",
                        help="Store weights as float16")
    args = parser.parse_args()
    _check_deps()
    convert_tdt_model(args.model, args.output, fp16=args.fp16)


if __name__ == "__main__":
    main()
