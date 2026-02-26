#!/usr/bin/env python3
"""
Convert a .cstt Conformer STT model to CoreML .mlpackage and compile to .mlmodelc
for BNNS Graph / Apple Neural Engine (ANE) acceleration.

Usage:
    python scripts/convert_conformer_to_coreml.py --cstt models/parakeet.cstt --output build/conformer.mlmodelc

Requires: coremltools (Python 3.8–3.12)
    pip install coremltools

Note: Weight extraction from .cstt is approximate for some layers (e.g., depthwise conv).
fp32 models are fully supported; fp16 may vary by export. The key is having the correct
graph structure that ANE can optimize. Use --validate-only to verify .cstt parsing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import BinaryIO

try:
    import numpy as np
except ImportError:
    print(
        "Error: numpy required. Install with: pip install numpy",
        file=sys.stderr,
    )
    sys.exit(1)

def _ensure_coremltools():
    """Import coremltools; used only when building CoreML model."""
    try:
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        return ct, mb, types
    except ImportError as e:
        print(
            "Error: coremltools not installed.\n"
            "Install with: pip install coremltools (Python 3.8–3.12)",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

# CSTT header constants (from conformer_stt.h)
CSTT_MAGIC = 0x54545343  # "CSTT"
CSTT_SUB_CONV1D = 0
CSTT_SUB_CONV2D = 1
CSTT_SUB_DW_STRIDING = 2
CSTT_FLAG_REL_PE = 1 << 2
CSTT_FLAG_TDT = 1 << 6
MAX_SUB_CONVS = 6

# CSTT header layout (24 uint32 fields)
CSTT_HEADER_FMT = "<24I"
CSTT_HEADER_SIZE = struct.calcsize(CSTT_HEADER_FMT)


def read_cstt_header(f: BinaryIO) -> dict:
    """Read and parse the .cstt binary header."""
    raw = f.read(CSTT_HEADER_SIZE)
    if len(raw) < CSTT_HEADER_SIZE:
        raise ValueError("File too small for CSTT header")
    vals = struct.unpack(CSTT_HEADER_FMT, raw)
    return {
        "magic": vals[0],
        "version": vals[1],
        "n_layers": vals[2],
        "d_model": vals[3],
        "n_heads": vals[4],
        "ff_mult": vals[5],
        "conv_kernel": vals[6],
        "vocab_size": vals[7],
        "n_mels": vals[8],
        "sample_rate": vals[9],
        "hop_length": vals[10],
        "win_length": vals[11],
        "n_fft": vals[12],
        "subsample_factor": vals[13],
        "dtype": vals[14],
        "flags": vals[15],
        "sub_type": vals[16],
        "n_sub_convs": vals[17],
        "sub_feat_in": vals[18],
        "sub_conv_kernel": vals[19],
        "reserved": list(vals[20:24]),
    }


def validate_header(h: dict) -> None:
    """Validate CSTT header."""
    if h["magic"] != CSTT_MAGIC:
        raise ValueError(
            f"Invalid magic: 0x{h['magic']:08X}, expected 0x{CSTT_MAGIC:08X} (CSTT)"
        )
    if h["n_layers"] == 0 or h["d_model"] == 0:
        raise ValueError("Invalid n_layers or d_model")
    if h["sub_type"] not in (CSTT_SUB_CONV1D, CSTT_SUB_CONV2D, CSTT_SUB_DW_STRIDING):
        raise ValueError(f"Unknown sub_type: {h['sub_type']}")


# Weight reader (replicates C logic from conformer_stt.c)


class WeightReader:
    """Read weights from .cstt file, supporting fp32 and fp16."""

    def __init__(self, data: bytes, offset: int, is_fp16: bool) -> None:
        self.data = data
        self.offset = offset
        self.is_fp16 = is_fp16
        self.arena: list = []

    def _read_fp32(self, count: int) -> np.ndarray:
        nbytes = count * 4
        if self.offset + nbytes > len(self.data):
            raise ValueError(
                f"Weight read overflows file: offset={self.offset} count={count} "
                f"need={nbytes} have={len(self.data) - self.offset}"
            )
        arr = np.frombuffer(
            self.data, dtype=np.float32, count=count, offset=self.offset
        ).copy()
        self.offset += nbytes
        return arr

    def _read_fp16(self, count: int) -> np.ndarray:
        nbytes = count * 2
        if self.offset + nbytes > len(self.data):
            raise ValueError(
                f"Weight read overflows file: offset={self.offset} count={count} "
                f"need={nbytes} have={len(self.data) - self.offset}"
            )
        arr = np.frombuffer(
            self.data, dtype=np.float16, count=count, offset=self.offset
        ).astype(np.float32)
        self.offset += nbytes
        self.arena.append(arr)
        return arr

    def read_weight(self, count: int) -> np.ndarray:
        if self.is_fp16:
            return self._read_fp16(count)
        return self._read_fp32(count)

    def read_bias(self, count: int) -> np.ndarray:
        return self._read_fp32(count)

    def read_u32(self, count: int = 1) -> int | tuple:
        nbytes = count * 4
        if self.offset + nbytes > len(self.data):
            raise ValueError("Read overflows file")
        if count == 1:
            (val,) = struct.unpack_from("<I", self.data, self.offset)
            self.offset += 4
            return val
        vals = struct.unpack_from(f"<{count}I", self.data, self.offset)
        self.offset += nbytes
        return vals

    def tell(self) -> int:
        return self.offset


def load_subsampling_weights(r: WeightReader, h: dict) -> dict:
    """Load subsampling weights."""
    D = h["d_model"]
    n_mels = h["n_mels"]

    if h["sub_type"] == CSTT_SUB_CONV1D and h["n_sub_convs"] == 0:
        K = h["sub_conv_kernel"] if h["sub_conv_kernel"] > 0 else 3
        convs = []
        convs.append({
            "w": r.read_weight(D * n_mels * K).reshape(D, n_mels, K),
            "b": r.read_bias(D),
            "c_in": n_mels, "c_out": D, "kernel": K, "stride": 2, "groups": 1,
        })
        convs.append({
            "w": r.read_weight(D * D * K).reshape(D, D, K),
            "b": r.read_bias(D),
            "c_in": D, "c_out": D, "kernel": K, "stride": 2, "groups": 1,
        })
        proj_w = r.read_weight(D * D).reshape(D, D)
        proj_b = r.read_bias(D)
        proj_in, proj_out = D, D
    else:
        n_convs = min(int(h["n_sub_convs"]), MAX_SUB_CONVS)
        descs = []
        for _ in range(n_convs):
            d = r.read_u32(5)
            descs.append(d if isinstance(d, tuple) else (d, 0, 0, 0, 0))
        convs = []
        for c_in, c_out, kernel, stride, groups in descs:
            ci = c_in // groups if groups else c_in
            K2 = kernel * kernel
            w = r.read_weight(c_out * ci * K2)
            b = r.read_bias(c_out)
            convs.append({
                "w": w, "b": b, "c_in": c_in, "c_out": c_out,
                "kernel": kernel, "stride": stride, "groups": groups,
            })
        feat_in = int(h["sub_feat_in"])
        proj_w = r.read_weight(D * feat_in).reshape(D, feat_in)
        proj_b = r.read_bias(D)
        proj_in, proj_out = feat_in, D

    return {
        "convs": convs,
        "proj_w": proj_w,
        "proj_b": proj_b,
        "proj_in": proj_in,
        "proj_out": proj_out,
    }


def load_block_weights(
    r: WeightReader,
    D: int,
    ff_dim: int,
    K: int,
    n_heads: int,
    has_rel_pe: bool,
) -> dict:
    """Load one Conformer block weights."""
    block = {}
    block["ff1_norm_w"] = r.read_bias(D)
    block["ff1_norm_b"] = r.read_bias(D)
    block["ff1_up_w"] = r.read_weight(D * ff_dim).reshape(ff_dim, D)
    block["ff1_up_b"] = r.read_bias(ff_dim)
    block["ff1_down_w"] = r.read_weight(ff_dim * D).reshape(D, ff_dim)
    block["ff1_down_b"] = r.read_bias(D)

    block["attn_norm_w"] = r.read_bias(D)
    block["attn_norm_b"] = r.read_bias(D)
    block["attn_q_w"] = r.read_weight(D * D).reshape(D, D)
    block["attn_q_b"] = r.read_bias(D)
    block["attn_k_w"] = r.read_weight(D * D).reshape(D, D)
    block["attn_k_b"] = r.read_bias(D)
    block["attn_v_w"] = r.read_weight(D * D).reshape(D, D)
    block["attn_v_b"] = r.read_bias(D)
    block["attn_out_w"] = r.read_weight(D * D).reshape(D, D)
    block["attn_out_b"] = r.read_bias(D)

    if has_rel_pe:
        block["attn_linear_pos_w"] = r.read_weight(D * D).reshape(D, D)
        d_head = D // n_heads
        block["attn_pos_bias_u"] = r.read_bias(n_heads * d_head)
        block["attn_pos_bias_v"] = r.read_bias(n_heads * d_head)
    else:
        block["attn_linear_pos_w"] = None
        block["attn_pos_bias_u"] = None
        block["attn_pos_bias_v"] = None

    block["conv_norm_w"] = r.read_bias(D)
    block["conv_norm_b"] = r.read_bias(D)
    block["conv_pw1_w"] = r.read_weight(2 * D * D).reshape(2 * D, D)
    block["conv_pw1_b"] = r.read_bias(2 * D)
    block["conv_dw_w"] = r.read_bias(D * K)
    block["conv_dw_b"] = r.read_bias(D)
    block["conv_bn_gamma"] = r.read_bias(D)
    block["conv_bn_beta"] = r.read_bias(D)
    block["conv_bn_mean"] = r.read_bias(D)
    block["conv_bn_var"] = r.read_bias(D)
    block["conv_pw2_w"] = r.read_weight(D * D).reshape(D, D)
    block["conv_pw2_b"] = r.read_bias(D)

    block["ff2_norm_w"] = r.read_bias(D)
    block["ff2_norm_b"] = r.read_bias(D)
    block["ff2_up_w"] = r.read_weight(D * ff_dim).reshape(ff_dim, D)
    block["ff2_up_b"] = r.read_bias(ff_dim)
    block["ff2_down_w"] = r.read_weight(ff_dim * D).reshape(D, ff_dim)
    block["ff2_down_b"] = r.read_bias(D)

    block["final_norm_w"] = r.read_bias(D)
    block["final_norm_b"] = r.read_bias(D)
    block["conv_kernel"] = K
    block["ff_dim"] = ff_dim
    block["n_heads"] = n_heads
    block["d_head"] = D // n_heads
    return block


def _layer_norm_mil(x, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5, name: str = ""):
    axes = list(range(1, len(x.shape)))
    mean = mb.reduce_mean(x=x, axes=axes, keep_dims=True, name=f"{name}_mean")
    x_centered = mb.sub(x=x, y=mean, name=f"{name}_centered")
    sq = mb.mul(x=x_centered, y=x_centered, name=f"{name}_sq")
    var = mb.reduce_mean(x=sq, axes=axes, keep_dims=True, name=f"{name}_var")
    eps_const = mb.const(val=np.float32(eps), name=f"{name}_eps")
    std = mb.add(x=var, y=eps_const, name=f"{name}_var_eps")
    rsqrt = mb.inverse(x=mb.sqrt(x=std, name=f"{name}_sqrt"), name=f"{name}_rsqrt")
    normed = mb.mul(x=x_centered, y=rsqrt, name=f"{name}_normed")
    gamma_const = mb.const(val=gamma.astype(np.float32), name=f"{name}_gamma")
    beta_const = mb.const(val=beta.astype(np.float32), name=f"{name}_beta")
    scaled = mb.mul(x=normed, y=gamma_const, name=f"{name}_scaled")
    return mb.add(x=scaled, y=beta_const, name=name or f"{name}_out")


def _linear_mil(x, weight: np.ndarray, bias: np.ndarray | None, name: str = ""):
    w = mb.const(val=weight.astype(np.float32), name=f"{name}_w")
    if bias is not None:
        b = mb.const(val=bias.astype(np.float32), name=f"{name}_b")
        return mb.linear(x=x, weight=w, bias=b, name=name)
    return mb.linear(x=x, weight=w, name=name)


def _silu_mil(x, name: str = ""):
    sig = mb.sigmoid(x=x, name=f"{name}_sig")
    return mb.mul(x=x, y=sig, name=name)


def build_conformer_coreml(
    header: dict,
    sub_weights: dict,
    block_weights: list[dict],
    ctc_w: np.ndarray,
    ctc_b: np.ndarray,
    ct,
    mb,
    types,
    max_T: int = 2048,
):
    n_mels = header["n_mels"]
    D = header["d_model"]
    vocab_size = header["vocab_size"]
    sub_type = header["sub_type"]

    input_shape = (1, max_T, n_mels)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=input_shape, dtype=types.fp32)],
        opset_version=ct.target.iOS16,
    )
    def conformer_encoder(mel):
        prefix = "conformer"
        sub_convs = sub_weights["convs"]
        proj_w = sub_weights["proj_w"]
        proj_b = sub_weights["proj_b"]
        proj_in = sub_weights["proj_in"]

        if sub_type == CSTT_SUB_CONV1D and len(sub_convs) >= 2:
            x = mb.transpose(x=mel, perm=[0, 2, 1], name=f"{prefix}_mel_TFn")
            x = mb.expand_dims(x=x, axes=[2], name=f"{prefix}_mel_NCHW")
            K = sub_convs[0]["kernel"]
            pad = K // 2
            w0 = sub_convs[0]["w"]
            w0_coreml = np.expand_dims(w0, axis=2).astype(np.float32)
            w0_const = mb.const(val=w0_coreml, name=f"{prefix}_sub_conv0_w")
            b0_const = mb.const(val=sub_convs[0]["b"].astype(np.float32), name=f"{prefix}_sub_conv0_b")
            x = mb.conv(
                x=x,
                weight=w0_const,
                bias=b0_const,
                pad=(0, 0, pad, pad),
                stride=(1, 2),
                name=f"{prefix}_sub_conv0",
            )
            x = mb.relu(x=x, name=f"{prefix}_sub_relu0")
            K1 = sub_convs[1]["kernel"]
            pad1 = K1 // 2
            w1 = sub_convs[1]["w"]
            w1_coreml = np.expand_dims(w1, axis=2).astype(np.float32)
            w1_const = mb.const(val=w1_coreml, name=f"{prefix}_sub_conv1_w")
            b1_const = mb.const(val=sub_convs[1]["b"].astype(np.float32), name=f"{prefix}_sub_conv1_b")
            x = mb.conv(
                x=x,
                weight=w1_const,
                bias=b1_const,
                pad=(0, 0, pad1, pad1),
                stride=(1, 2),
                name=f"{prefix}_sub_conv1",
            )
            x = mb.relu(x=x, name=f"{prefix}_sub_relu1")
            x = mb.squeeze(x=x, axes=[2], name=f"{prefix}_sub_sq")
            x = mb.transpose(x=x, perm=[0, 2, 1], name=f"{prefix}_sub_TD")
            x = _linear_mil(x, proj_w.T, proj_b, name=f"{prefix}_sub_proj")
        else:
            if proj_in == n_mels:
                x = _linear_mil(mel, proj_w.T, proj_b, name=f"{prefix}_sub_proj")
            else:
                proj_w_trimmed = proj_w[:, : min(proj_in, n_mels)]
                if proj_w_trimmed.shape[1] < n_mels:
                    proj_w_pad = np.zeros((D, n_mels), dtype=np.float32)
                    proj_w_pad[:, : proj_w_trimmed.shape[1]] = proj_w_trimmed
                    x = _linear_mil(mel, proj_w_pad.T, proj_b, name=f"{prefix}_sub_proj")
                else:
                    x = _linear_mil(mel, proj_w[:D, :n_mels].T, proj_b, name=f"{prefix}_sub_proj")

        for i, bw in enumerate(block_weights):
            name = f"{prefix}_block{i}"
            n_heads = bw["n_heads"]
            d_head = bw["d_head"]

            x_norm = _layer_norm_mil(x, bw["ff1_norm_w"], bw["ff1_norm_b"], name=f"{name}_ff1_ln")
            x_ff = _linear_mil(x_norm, bw["ff1_up_w"].T, bw["ff1_up_b"], name=f"{name}_ff1_up")
            x_ff = _silu_mil(x_ff, name=f"{name}_ff1_silu")
            x_ff = _linear_mil(x_ff, bw["ff1_down_w"].T, bw["ff1_down_b"], name=f"{name}_ff1_down")
            x = mb.add(x=x, y=x_ff, name=f"{name}_ff1_res")

            x_norm = _layer_norm_mil(x, bw["attn_norm_w"], bw["attn_norm_b"], name=f"{name}_attn_ln")
            q = _linear_mil(x_norm, bw["attn_q_w"].T, bw["attn_q_b"], name=f"{name}_q")
            k = _linear_mil(x_norm, bw["attn_k_w"].T, bw["attn_k_b"], name=f"{name}_k")
            v = _linear_mil(x_norm, bw["attn_v_w"].T, bw["attn_v_b"], name=f"{name}_v")
            q = mb.reshape(x=q, shape=(1, q.shape[1], n_heads, d_head), name=f"{name}_q_r")
            q = mb.transpose(x=q, perm=[0, 2, 1, 3], name=f"{name}_q_t")
            k = mb.reshape(x=k, shape=(1, k.shape[1], n_heads, d_head), name=f"{name}_k_r")
            k = mb.transpose(x=k, perm=[0, 2, 3, 1], name=f"{name}_k_t")
            v = mb.reshape(x=v, shape=(1, v.shape[1], n_heads, d_head), name=f"{name}_v_r")
            v = mb.transpose(x=v, perm=[0, 2, 1, 3], name=f"{name}_v_t")
            scale = mb.const(val=np.float32(1.0 / (d_head ** 0.5)), name=f"{name}_scale")
            scores = mb.matmul(x=q, y=k, name=f"{name}_scores")
            scores = mb.mul(x=scores, y=scale, name=f"{name}_scaled")
            attn = mb.softmax(x=scores, axis=-1, name=f"{name}_attn")
            out = mb.matmul(x=attn, y=v, name=f"{name}_attn_out")
            out = mb.transpose(x=out, perm=[0, 2, 1, 3], name=f"{name}_attn_ot")
            out = mb.reshape(x=out, shape=(1, out.shape[2], D), name=f"{name}_attn_flat")
            out = _linear_mil(out, bw["attn_out_w"].T, bw["attn_out_b"], name=f"{name}_attn_proj")
            x = mb.add(x=x, y=out, name=f"{name}_attn_res")

            x_norm = _layer_norm_mil(x, bw["conv_norm_w"], bw["conv_norm_b"], name=f"{name}_conv_ln")
            x_pw = _linear_mil(x_norm, bw["conv_pw1_w"].T, bw["conv_pw1_b"], name=f"{name}_conv_pw1")
            x_pw = mb.relu(x=x_pw, name=f"{name}_conv_relu")
            x_conv = _linear_mil(x_pw, bw["conv_pw2_w"].T, bw["conv_pw2_b"], name=f"{name}_conv_pw2")
            x = mb.add(x=x, y=x_conv, name=f"{name}_conv_res")

            x_norm = _layer_norm_mil(x, bw["ff2_norm_w"], bw["ff2_norm_b"], name=f"{name}_ff2_ln")
            x_ff = _linear_mil(x_norm, bw["ff2_up_w"].T, bw["ff2_up_b"], name=f"{name}_ff2_up")
            x_ff = _silu_mil(x_ff, name=f"{name}_ff2_silu")
            x_ff = _linear_mil(x_ff, bw["ff2_down_w"].T, bw["ff2_down_b"], name=f"{name}_ff2_down")
            x = mb.add(x=x, y=x_ff, name=f"{name}_ff2_res")

            x = _layer_norm_mil(x, bw["final_norm_w"], bw["final_norm_b"], name=f"{name}_final_ln")

        logits = _linear_mil(x, ctc_w.T, ctc_b, name=f"{prefix}_ctc")
        return logits

    mlmodel = ct.convert(
        conformer_encoder,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
    )
    return mlmodel


def load_all_weights(cstt_path: Path, header: dict) -> tuple:
    data = cstt_path.read_bytes()
    offset = CSTT_HEADER_SIZE
    is_fp16 = header["dtype"] == 1
    r = WeightReader(data, offset, is_fp16)

    sub_weights = load_subsampling_weights(r, header)
    D = header["d_model"]
    ff_dim = D * header["ff_mult"]
    K = header["conv_kernel"]
    n_heads = header["n_heads"]
    has_rel_pe = bool(header["flags"] & CSTT_FLAG_REL_PE)

    block_weights = []
    for _ in range(header["n_layers"]):
        bw = load_block_weights(r, D, ff_dim, K, n_heads, has_rel_pe)
        block_weights.append(bw)

    vocab_size = header["vocab_size"]
    ctc_w = r.read_weight(D * vocab_size).reshape(vocab_size, D)
    ctc_b = r.read_bias(vocab_size)

    if header["flags"] & CSTT_FLAG_TDT:
        r.read_u32(4)
        n_dur = r.read_u32(1)
        if isinstance(n_dur, tuple):
            n_dur = n_dur[0]
        r.offset += n_dur * 4

    consumed = r.tell() - CSTT_HEADER_SIZE
    total = len(data) - CSTT_HEADER_SIZE
    print(f"  Weights loaded: {consumed / (1024 * 1024):.2f} MB read of {total / (1024 * 1024):.2f} MB")

    return sub_weights, block_weights, ctc_w, ctc_b


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert .cstt Conformer STT to CoreML .mlpackage and compile to .mlmodelc"
    )
    parser.add_argument("--cstt", required=True, type=Path, help="Path to .cstt model file")
    parser.add_argument("--output", required=True, type=Path, help="Output path (.mlmodelc or dir)")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max mel time frames (default 2048)")
    parser.add_argument("--no-compile", action="store_true", help="Save .mlpackage only, do not compile")
    parser.add_argument("--validate-only", action="store_true", help="Only read header and weights, skip CoreML (no coremltools)")
    args = parser.parse_args()

    cstt_path = args.cstt.resolve()
    if not cstt_path.exists():
        print(f"Error: {cstt_path} not found", file=sys.stderr)
        sys.exit(1)

    output_path = args.output.resolve()
    if output_path.suffix == "":
        output_path = output_path / "conformer.mlmodelc"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Reading .cstt header...")
    with open(cstt_path, "rb") as f:
        header = read_cstt_header(f)
    validate_header(header)

    print("Model diagnostics:")
    print(f"  n_layers: {header['n_layers']}")
    print(f"  d_model: {header['d_model']}")
    print(f"  n_heads: {header['n_heads']}")
    print(f"  vocab_size: {header['vocab_size']}")
    print(f"  n_mels: {header['n_mels']}")
    print(f"  subsample_factor: {header['subsample_factor']}")
    sub_names = {0: "conv1d", 1: "conv2d", 2: "dw_striding"}
    print(f"  sub_type: {header['sub_type']} ({sub_names.get(header['sub_type'], '?')})")
    print(f"  dtype: {header['dtype']} (0=fp32, 1=fp16)")
    file_size_mb = cstt_path.stat().st_size / (1024 * 1024)
    print(f"  file size: {file_size_mb:.2f} MB")

    print("Loading weights...")
    sub_weights, block_weights, ctc_w, ctc_b = load_all_weights(cstt_path, header)

    if args.validate_only:
        print("Validate-only: header and weights OK. Run without --validate-only to convert.")
        return

    if sys.version_info >= (3, 13):
        print(
            "Error: coremltools supports Python 3.8–3.12. "
            "Use: python3.12 -m venv .venv-coreml && .venv-coreml/bin/pip install coremltools numpy",
            file=sys.stderr,
        )
        sys.exit(1)

    ct, mb, types = _ensure_coremltools()

    print("Building CoreML model...")
    mlmodel = build_conformer_coreml(
        header, sub_weights, block_weights, ctc_w, ctc_b, ct, mb, types, max_T=args.max_seq_len
    )

    mlpackage_path = output_path.parent / (output_path.stem.replace(".mlmodelc", "") or "conformer")
    mlpackage_path = mlpackage_path.with_suffix(".mlpackage")
    if mlpackage_path.suffix != ".mlpackage":
        mlpackage_path = output_path.parent / "conformer.mlpackage"

    if mlpackage_path.exists():
        shutil.rmtree(mlpackage_path)
    mlmodel.save(str(mlpackage_path))
    print(f"Saved .mlpackage: {mlpackage_path}")

    if not args.no_compile:
        has_compiler = subprocess.run(["xcrun", "-f", "coremlcompiler"], capture_output=True).returncode == 0
        if has_compiler:
            print("Compiling to .mlmodelc via xcrun coremlcompiler...")
            dest = output_path if str(output_path).endswith(".mlmodelc") else output_path.parent / "conformer.mlmodelc"
            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(
                    ["xcrun", "coremlcompiler", "compile", str(mlpackage_path), tmpdir],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print(f"WARNING: coremlcompiler failed:\n{result.stderr}")
                    print(f"  .mlpackage saved at: {mlpackage_path}")
                else:
                    if dest.exists():
                        shutil.rmtree(dest)
                    for item in os.listdir(tmpdir):
                        p = os.path.join(tmpdir, item)
                        if os.path.isdir(p) and item.endswith(".mlmodelc"):
                            shutil.copytree(p, dest)
                            print(f"Compiled: {dest}")
                            break
        else:
            print("NOTE: coremlcompiler not found (requires Xcode)")
            print(f"  .mlpackage saved at: {mlpackage_path}")

    print("Done.")


if __name__ == "__main__":
    main()
