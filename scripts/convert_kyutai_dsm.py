#!/usr/bin/env python3
"""
convert_kyutai_dsm.py — Convert Kyutai DSM TTS weights to .ctts binary format.

Downloads safetensors from HuggingFace (or loads local files), extracts and
repacks all weights into a single flat binary for mmap-loading in C.

Usage:
    python scripts/convert_kyutai_dsm.py --output models/kyutai_dsm.ctts

    # With explicit paths:
    python scripts/convert_kyutai_dsm.py \
        --weights path/to/tts_b6369a24.safetensors \
        --tokenizer path/to/tokenizer.model \
        --output models/kyutai_dsm.ctts
"""

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from safetensors.torch import load_file as _load_safetensors_torch

    def load_safetensors(path):
        """Load via torch backend (handles bfloat16), convert to numpy float32."""
        sd = _load_safetensors_torch(path, device="cpu")
        return {k: v.float().numpy() for k, v in sd.items()}
except ImportError:
    from safetensors import safe_open

    def load_safetensors(path):
        tensors = {}
        with safe_open(path, framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors


try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

CTTS_MAGIC = 0x53545443  # "CTTS"
CTTS_VERSION = 1

# b6369a24 config
FLOWLM_D_MODEL = 1024
FLOWLM_N_LAYERS = 6
FLOWLM_N_HEADS = 16
FLOWLM_FFN_DIM = 4096
FLOWLM_HEAD_DIM = FLOWLM_D_MODEL // FLOWLM_N_HEADS  # 64
FLOWLM_MAX_PERIOD = 10000
FLOWLM_LDIM = 32  # quantizer dimension (latent)
FLOWLM_VOCAB_SIZE = 4000

FLOW_DIM = 512
FLOW_DEPTH = 6
FLOW_COND_DIM = 1024
FLOW_FREQ_EMBED_SIZE = 256
FLOW_NUM_TIME_CONDS = 2

MIMI_DIM = 512
MIMI_N_FILTERS = 64
MIMI_RATIOS = [6, 5, 4]
MIMI_KERNEL = 7
MIMI_LAST_KERNEL = 3
MIMI_RES_KERNEL = 3
MIMI_COMPRESS = 2
MIMI_N_RESIDUAL = 1
MIMI_XFMR_D_MODEL = 512
MIMI_XFMR_N_HEADS = 8
MIMI_XFMR_N_LAYERS = 2
MIMI_XFMR_FFN_DIM = 2048
MIMI_XFMR_CONTEXT = 250
MIMI_UPSAMPLE_STRIDE = 16
MIMI_UPSAMPLE_KERNEL = 32
MIMI_QUANT_DIM = 32
MIMI_QUANT_OUT_DIM = 512

DEFAULT_HF_REPO = "kyutai/pocket-tts-without-voice-cloning"
DEFAULT_WEIGHTS_FILE = "tts_b6369a24.safetensors"
DEFAULT_TOKENIZER_FILE = "tokenizer.model"


def download_weights(repo, filename, revision=None):
    if hf_hub_download is None:
        print("ERROR: huggingface_hub not installed. pip install huggingface_hub",
              file=sys.stderr)
        sys.exit(1)
    print(f"Downloading {filename} from {repo}...")
    return Path(hf_hub_download(repo, filename, revision=revision))


def transpose_conv1d(w):
    """PyTorch Conv1d (out, in, K) -> C row-major (out, K, in)."""
    if w.ndim == 3:
        return np.ascontiguousarray(np.swapaxes(w, 1, 2))
    return w


def transpose_convtr1d(w):
    """PyTorch ConvTranspose1d (in, out, K) -> C (out, K, in)."""
    if w.ndim == 3:
        if w.shape[1] == 1:
            return np.ascontiguousarray(np.swapaxes(w, 1, 2))
        return np.ascontiguousarray(np.transpose(w, (1, 2, 0)))
    return w


def get_tensor(sd, key, required=True):
    if key in sd:
        return sd[key].astype(np.float32).flatten()
    if required:
        raise KeyError(f"Missing required tensor: {key}")
    return None


def pack_header():
    """Pack .ctts header: magic, version, all hyperparameters."""
    fields = [
        CTTS_MAGIC, CTTS_VERSION,
        # FlowLM
        FLOWLM_D_MODEL, FLOWLM_N_LAYERS, FLOWLM_N_HEADS, FLOWLM_FFN_DIM,
        FLOWLM_LDIM, FLOWLM_VOCAB_SIZE, FLOWLM_MAX_PERIOD,
        # Flow network
        FLOW_DIM, FLOW_DEPTH, FLOW_COND_DIM, FLOW_FREQ_EMBED_SIZE,
        FLOW_NUM_TIME_CONDS,
        # Mimi
        MIMI_DIM, MIMI_XFMR_D_MODEL, MIMI_XFMR_N_HEADS, MIMI_XFMR_N_LAYERS,
        MIMI_XFMR_FFN_DIM, MIMI_XFMR_CONTEXT, MIMI_UPSAMPLE_STRIDE,
        MIMI_UPSAMPLE_KERNEL, MIMI_N_FILTERS, len(MIMI_RATIOS),
        MIMI_RATIOS[0], MIMI_RATIOS[1], MIMI_RATIOS[2],
        MIMI_KERNEL, MIMI_LAST_KERNEL, MIMI_RES_KERNEL, MIMI_COMPRESS,
        MIMI_QUANT_DIM, MIMI_QUANT_OUT_DIM,
    ]
    return struct.pack(f"<{len(fields)}I", *fields)


def remap_flowlm_keys(sd):
    """Rename state_dict keys from HF naming to canonical form."""
    remap = {}
    for k, v in sd.items():
        nk = k
        if k.startswith("flow_lm."):
            nk = k[len("flow_lm."):]
        if k.startswith("condition_provider.conditioners.transcript_in_segment.embed.weight"):
            nk = "conditioner.embed.weight"
        if k.startswith("condition_provider.conditioners.speaker_wavs.output_proj.weight"):
            nk = "speaker_proj_weight"
        if nk.startswith("flow.w_s_t.") or nk == "condition_provider.conditioners.transcript_in_segment.learnt_padding" or nk == "condition_provider.conditioners.speaker_wavs.learnt_padding":
            continue
        remap[nk] = v.astype(np.float32)
    return remap


def remap_mimi_keys(sd):
    """Rename Mimi state_dict keys, apply conv transpositions."""
    remap = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model."):]
        if nk.startswith("mimi."):
            nk = nk[len("mimi."):]
        if nk.startswith("quantizer.vq.") or nk == "quantizer.logvar_proj.weight":
            continue
        arr = v.astype(np.float32)
        if arr.ndim == 3:
            if any(nk.endswith(s) for s in (".conv.weight", ".output_proj.weight")):
                arr = transpose_conv1d(arr)
            elif nk.endswith(".convtr.weight"):
                arr = transpose_convtr1d(arr)
        remap[nk] = arr
    return remap


def pack_flowlm_weights(sd):
    """Pack FlowLM weights in C-expected order."""
    buf = []

    # conditioner.embed.weight: (4001, 1024)
    buf.append(get_tensor(sd, "conditioner.embed.weight"))

    # input_linear.weight: (1024, 32) -- nn.Linear weight is (out, in)
    buf.append(get_tensor(sd, "input_linear.weight"))

    # out_norm
    buf.append(get_tensor(sd, "out_norm.weight"))
    buf.append(get_tensor(sd, "out_norm.bias"))

    # out_eos: Linear(1024, 1)
    buf.append(get_tensor(sd, "out_eos.weight"))
    buf.append(get_tensor(sd, "out_eos.bias"))

    # emb_std, emb_mean, bos_emb: (32,) each
    buf.append(get_tensor(sd, "emb_std"))
    buf.append(get_tensor(sd, "emb_mean"))
    buf.append(get_tensor(sd, "bos_emb"))

    # Transformer layers x6
    for layer in range(FLOWLM_N_LAYERS):
        pfx = f"transformer.layers.{layer}"

        # Self-attention: in_proj (3*1024, 1024), out_proj (1024, 1024)
        buf.append(get_tensor(sd, f"{pfx}.self_attn.in_proj.weight"))
        buf.append(get_tensor(sd, f"{pfx}.self_attn.out_proj.weight"))

        # Layer norms
        buf.append(get_tensor(sd, f"{pfx}.norm1.weight"))
        buf.append(get_tensor(sd, f"{pfx}.norm1.bias"))
        buf.append(get_tensor(sd, f"{pfx}.norm2.weight"))
        buf.append(get_tensor(sd, f"{pfx}.norm2.bias"))

        # FFN
        # gating linear: (ffn_dim, d_model) and (ffn_dim, d_model) for gated
        # Or standard: linear1 (ffn_dim, d_model), linear2 (d_model, ffn_dim)
        key_l1 = f"{pfx}.linear1.weight"
        key_l2 = f"{pfx}.linear2.weight"
        if key_l1 in sd:
            buf.append(get_tensor(sd, key_l1))
            buf.append(get_tensor(sd, key_l2))
        else:
            key_gating = f"{pfx}.gating.linear1.weight"
            key_gating2 = f"{pfx}.gating.linear2.weight"
            buf.append(get_tensor(sd, key_gating))
            buf.append(get_tensor(sd, key_gating2))

    return np.concatenate(buf)


def pack_flow_weights(sd):
    """Pack LSD flow network weights per amx_flow_fused.c layout."""
    buf = []
    mc = FLOW_DIM
    ic = FLOWLM_LDIM
    cc = FLOW_COND_DIM
    fes = FLOW_FREQ_EMBED_SIZE

    # input_proj: Linear(ic, mc)
    buf.append(get_tensor(sd, "flow_net.input_proj.weight"))
    buf.append(get_tensor(sd, "flow_net.input_proj.bias"))

    # cond_embed: Linear(cc, mc)
    buf.append(get_tensor(sd, "flow_net.cond_embed.weight"))
    buf.append(get_tensor(sd, "flow_net.cond_embed.bias"))

    # Timestep embedders x2
    for t in range(FLOW_NUM_TIME_CONDS):
        pfx = f"flow_net.time_embed.{t}"
        buf.append(get_tensor(sd, f"{pfx}.freqs"))

        # mlp: [Linear(fes, mc), SiLU, Linear(mc, mc), RMSNorm(mc)]
        buf.append(get_tensor(sd, f"{pfx}.mlp.0.weight"))
        buf.append(get_tensor(sd, f"{pfx}.mlp.0.bias"))
        buf.append(get_tensor(sd, f"{pfx}.mlp.2.weight"))
        buf.append(get_tensor(sd, f"{pfx}.mlp.2.bias"))
        buf.append(get_tensor(sd, f"{pfx}.mlp.3.alpha"))

    # ResBlocks
    for b in range(FLOW_DEPTH):
        pfx = f"flow_net.res_blocks.{b}"
        buf.append(get_tensor(sd, f"{pfx}.in_ln.weight"))
        buf.append(get_tensor(sd, f"{pfx}.in_ln.bias"))

        # adaLN: [SiLU, Linear(mc, 3*mc)]
        buf.append(get_tensor(sd, f"{pfx}.adaLN_modulation.1.weight"))
        buf.append(get_tensor(sd, f"{pfx}.adaLN_modulation.1.bias"))

        # mlp: [Linear(mc, mc), SiLU, Linear(mc, mc)]
        buf.append(get_tensor(sd, f"{pfx}.mlp.0.weight"))
        buf.append(get_tensor(sd, f"{pfx}.mlp.0.bias"))
        buf.append(get_tensor(sd, f"{pfx}.mlp.2.weight"))
        buf.append(get_tensor(sd, f"{pfx}.mlp.2.bias"))

    # Final layer
    pfx = "flow_net.final_layer"
    buf.append(get_tensor(sd, f"{pfx}.adaLN_modulation.1.weight"))
    buf.append(get_tensor(sd, f"{pfx}.adaLN_modulation.1.bias"))

    norm_w = sd.get(f"{pfx}.norm_final.weight")
    norm_b = sd.get(f"{pfx}.norm_final.bias")
    if norm_w is not None:
        buf.append(norm_w.astype(np.float32).flatten())
        buf.append(norm_b.astype(np.float32).flatten())
    else:
        buf.append(np.zeros(mc, dtype=np.float32))
        buf.append(np.zeros(mc, dtype=np.float32))

    buf.append(get_tensor(sd, f"{pfx}.linear.weight"))
    buf.append(get_tensor(sd, f"{pfx}.linear.bias"))

    return np.concatenate(buf)


def pack_mimi_weights(sd):
    """Pack Mimi decoder weights per bnns_mimi_decoder.c layout."""
    buf = []

    # ConvTrUpsample1d: depthwise (512, K=32, 1) weight
    up_candidates = [
        "upsample.convtr.convtr.weight",
        "upsample.convtr.weight",
        "downsample.convtr.convtr.weight",
        "downsample.convtr.weight",
    ]
    found_up = False
    for key_up in up_candidates:
        if key_up in sd:
            buf.append(sd[key_up].astype(np.float32).flatten())
            found_up = True
            break
    if not found_up:
        raise KeyError(f"Missing upsample weight: tried {up_candidates}")

    # Decoder transformer layers x2
    for layer in range(MIMI_XFMR_N_LAYERS):
        pfx = f"decoder_transformer.transformer.layers.{layer}"

        buf.append(get_tensor(sd, f"{pfx}.norm1.weight"))
        buf.append(get_tensor(sd, f"{pfx}.norm1.bias"))
        buf.append(get_tensor(sd, f"{pfx}.self_attn.in_proj.weight"))
        buf.append(get_tensor(sd, f"{pfx}.self_attn.out_proj.weight"))

        ls1_key = f"{pfx}.layer_scale_1.scale"
        if ls1_key in sd:
            buf.append(get_tensor(sd, ls1_key))
        elif f"{pfx}.layer_scale_1" in sd:
            buf.append(get_tensor(sd, f"{pfx}.layer_scale_1"))
        else:
            buf.append(get_tensor(sd, f"{pfx}.self_attn_layer_scale"))

        buf.append(get_tensor(sd, f"{pfx}.norm2.weight"))
        buf.append(get_tensor(sd, f"{pfx}.norm2.bias"))

        # FFN
        key_l1 = f"{pfx}.linear1.weight"
        key_l2 = f"{pfx}.linear2.weight"
        if key_l1 in sd:
            buf.append(get_tensor(sd, key_l1))
            buf.append(get_tensor(sd, key_l2))
        else:
            buf.append(get_tensor(sd, f"{pfx}.gating.linear1.weight"))
            buf.append(get_tensor(sd, f"{pfx}.gating.linear2.weight"))

        ls2_key = f"{pfx}.layer_scale_2.scale"
        if ls2_key in sd:
            buf.append(get_tensor(sd, ls2_key))
        elif f"{pfx}.layer_scale_2" in sd:
            buf.append(get_tensor(sd, f"{pfx}.layer_scale_2"))
        else:
            buf.append(get_tensor(sd, f"{pfx}.ff_layer_scale"))

    # Decoder transformer input/output projections (may not exist in all checkpoints)
    pfx = "decoder_transformer"
    for key_name in ("input_proj.weight", "output_projs.0.weight"):
        full_key = f"{pfx}.{key_name}"
        if full_key in sd:
            arr = sd[full_key].astype(np.float32)
            if arr.ndim == 3:
                arr = transpose_conv1d(arr)
            buf.append(arr.flatten())
            print(f"  Found {full_key}: {sd[full_key].shape}")
        else:
            print(f"  (skipped {full_key} — not in checkpoint)")

    # SEANet Decoder
    dim = MIMI_N_FILTERS
    for i in range(len(MIMI_RATIOS) - 1, -1, -1):
        dim *= 2

    # Initial conv: (dim, K=7, 512) -> already transposed if needed
    conv0_w = sd.get("decoder.model.0.conv.weight")
    if conv0_w is not None:
        conv0_w = transpose_conv1d(conv0_w.astype(np.float32))
        buf.append(conv0_w.flatten())
    conv0_b = sd.get("decoder.model.0.conv.bias")
    if conv0_b is not None:
        buf.append(conv0_b.astype(np.float32).flatten())

    # Decoder blocks
    cur_dim = dim
    block_idx = 0
    for i, ratio in enumerate(MIMI_RATIOS):
        next_dim = cur_dim // 2
        model_offset = 1 + i * 3

        # ELU is index model_offset
        # ConvTranspose1d at model_offset+1
        convtr_key = f"decoder.model.{model_offset + 1}.convtr.weight"
        if convtr_key in sd:
            w = transpose_convtr1d(sd[convtr_key].astype(np.float32))
            buf.append(w.flatten())
        convtr_bias_key = f"decoder.model.{model_offset + 1}.convtr.bias"
        if convtr_bias_key in sd:
            buf.append(sd[convtr_bias_key].astype(np.float32).flatten())

        # ResBlock at model_offset+2
        hidden = next_dim // MIMI_COMPRESS
        res_pfx = f"decoder.model.{model_offset + 2}.block"
        for r in range(MIMI_N_RESIDUAL):
            # Each residual block: ELU->Conv1(K=3)->ELU->Conv2(K=1)
            c1w = sd.get(f"{res_pfx}.{r * 2 + 1}.conv.weight")
            if c1w is not None:
                buf.append(transpose_conv1d(c1w.astype(np.float32)).flatten())
            c1b = sd.get(f"{res_pfx}.{r * 2 + 1}.conv.bias")
            if c1b is not None:
                buf.append(c1b.astype(np.float32).flatten())
            c2w = sd.get(f"{res_pfx}.{r * 2 + 2}.conv.weight")  # K=1
            if c2w is None:
                c2w = sd.get(f"{res_pfx}.{r * 2 + 3}.conv.weight")
            if c2w is not None:
                buf.append(transpose_conv1d(c2w.astype(np.float32)).flatten())
            c2b = sd.get(f"{res_pfx}.{r * 2 + 2}.conv.bias")
            if c2b is None:
                c2b = sd.get(f"{res_pfx}.{r * 2 + 3}.conv.bias")
            if c2b is not None:
                buf.append(c2b.astype(np.float32).flatten())

        cur_dim = next_dim

    # Final conv: ELU + Conv1d(n_filters -> 1, K=3)
    n_blocks = len(MIMI_RATIOS)
    final_offset = 1 + n_blocks * 3
    final_w_key = f"decoder.model.{final_offset + 1}.conv.weight"
    if final_w_key in sd:
        buf.append(transpose_conv1d(sd[final_w_key].astype(np.float32)).flatten())
    final_b_key = f"decoder.model.{final_offset + 1}.conv.bias"
    if final_b_key in sd:
        buf.append(sd[final_b_key].astype(np.float32).flatten())

    # Quantizer: projection (32 -> 512)
    q_w_candidates = [
        "quantizer.output_proj.weight",
        "quantizer.input_proj.conv.weight",
        "quantizer.input_proj.weight",
    ]
    for qk in q_w_candidates:
        q_w = sd.get(qk)
        if q_w is not None:
            q_w = q_w.astype(np.float32)
            if q_w.ndim == 3:
                q_w = transpose_conv1d(q_w)
            buf.append(q_w.flatten())
            break
    q_b_candidates = [
        "quantizer.output_proj.bias",
        "quantizer.input_proj.conv.bias",
        "quantizer.input_proj.bias",
    ]
    for qk in q_b_candidates:
        q_b = sd.get(qk)
        if q_b is not None:
            buf.append(q_b.astype(np.float32).flatten())
            break

    return np.concatenate([b for b in buf if b is not None and len(b) > 0])


def main():
    parser = argparse.ArgumentParser(description="Convert Kyutai DSM TTS to .ctts")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to safetensors file (downloads from HF if not set)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to sentencepiece tokenizer.model")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .ctts file path")
    parser.add_argument("--repo", type=str, default=DEFAULT_HF_REPO,
                        help="HuggingFace repo ID")
    args = parser.parse_args()

    # Download or locate weights
    if args.weights is None:
        weights_path = download_weights(args.repo, DEFAULT_WEIGHTS_FILE)
    else:
        weights_path = Path(args.weights)

    if args.tokenizer is None:
        tokenizer_path = download_weights(args.repo, DEFAULT_TOKENIZER_FILE)
    else:
        tokenizer_path = Path(args.tokenizer)

    print(f"Loading weights from {weights_path}...")
    raw_sd = load_safetensors(str(weights_path))

    print(f"  Loaded {len(raw_sd)} tensors")
    for k in sorted(raw_sd.keys())[:10]:
        print(f"    {k}: {raw_sd[k].shape} {raw_sd[k].dtype}")
    if len(raw_sd) > 10:
        print(f"    ... and {len(raw_sd) - 10} more")

    # Split into FlowLM and Mimi state dicts
    flowlm_sd = {}
    mimi_sd = {}

    for k, v in raw_sd.items():
        if k.startswith("flow_lm.") or k.startswith("condition_provider."):
            nk = k
            if nk.startswith("flow_lm."):
                nk = nk[len("flow_lm."):]
            if "condition_provider.conditioners.transcript_in_segment.embed.weight" in k:
                nk = "conditioner.embed.weight"
            elif "condition_provider.conditioners.speaker_wavs.output_proj.weight" in k:
                nk = "speaker_proj_weight"
            elif k.startswith("condition_provider."):
                continue
            if nk.startswith("flow.w_s_t."):
                continue
            flowlm_sd[nk] = v.astype(np.float32)
        elif k.startswith("mimi.") or k.startswith("model."):
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model."):]
            if nk.startswith("mimi."):
                nk = nk[len("mimi."):]
            if nk.startswith("quantizer.vq.") or nk == "quantizer.logvar_proj.weight":
                continue
            mimi_sd[nk] = v.astype(np.float32)

    print(f"  FlowLM keys: {len(flowlm_sd)}, Mimi keys: {len(mimi_sd)}")

    # Read tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer_blob = tokenizer_path.read_bytes()
    print(f"  Tokenizer size: {len(tokenizer_blob)} bytes")

    # Pack everything
    print("Packing header...")
    header = pack_header()

    print("Packing tokenizer...")
    tok_header = struct.pack("<I", len(tokenizer_blob))

    print("Packing FlowLM weights...")
    flowlm_weights = pack_flowlm_weights(flowlm_sd)
    print(f"  FlowLM: {flowlm_weights.shape[0]} floats ({flowlm_weights.nbytes / 1e6:.1f} MB)")

    print("Packing LSD flow weights...")
    flow_weights = pack_flow_weights(flowlm_sd)
    print(f"  Flow: {flow_weights.shape[0]} floats ({flow_weights.nbytes / 1e6:.1f} MB)")

    print("Packing Mimi weights...")
    mimi_weights = pack_mimi_weights(mimi_sd)
    print(f"  Mimi: {mimi_weights.shape[0]} floats ({mimi_weights.nbytes / 1e6:.1f} MB)")

    # Section sizes
    flowlm_size = struct.pack("<I", flowlm_weights.shape[0])
    flow_size = struct.pack("<I", flow_weights.shape[0])
    mimi_size = struct.pack("<I", mimi_weights.shape[0])

    # Write .ctts file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {output_path}...")
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(tok_header)
        f.write(tokenizer_blob)
        # Align to 4 bytes
        pad = (4 - len(tokenizer_blob) % 4) % 4
        f.write(b"\x00" * pad)
        f.write(flowlm_size)
        f.write(flowlm_weights.tobytes())
        f.write(flow_size)
        f.write(flow_weights.tobytes())
        f.write(mimi_size)
        f.write(mimi_weights.tobytes())

    total_size = output_path.stat().st_size
    print(f"Done! {output_path}: {total_size / 1e6:.1f} MB")
    print(f"  Header: {len(header)} bytes")
    print(f"  Tokenizer: {len(tokenizer_blob)} bytes")
    print(f"  FlowLM: {flowlm_weights.nbytes / 1e6:.1f} MB")
    print(f"  Flow: {flow_weights.nbytes / 1e6:.1f} MB")
    print(f"  Mimi: {mimi_weights.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
