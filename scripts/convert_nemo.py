#!/usr/bin/env python3
"""
convert_nemo.py — Convert NeMo ASR models to .cstt format for pocket-voice.

Supports NVIDIA Parakeet TDT/CTC and other NeMo FastConformer models.
Downloads from HuggingFace, extracts weights, writes .cstt + .vocab files.

Usage:
    python scripts/convert_nemo.py nvidia/parakeet-tdt_ctc-110m -o model.cstt
    python scripts/convert_nemo.py path/to/model.nemo -o model.cstt

Dependencies:
    pip install torch pyyaml huggingface_hub sentencepiece
"""

import argparse
import io
import os
import struct
import sys
import tarfile
import tempfile
from pathlib import Path


def _check_deps():
    """Verify required Python packages are installed."""
    missing = []
    for pkg in ["torch", "yaml", "numpy", "huggingface_hub"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg if pkg != "yaml" else "pyyaml")
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}", file=sys.stderr)
        print(f"Install with: pip install -r scripts/requirements.txt", file=sys.stderr)
        sys.exit(1)


CSTT_MAGIC = 0x54545343
CSTT_VERSION = 1

CSTT_SUB_CONV1D = 0
CSTT_SUB_CONV2D = 1
CSTT_SUB_DW_STRIDING = 2

CSTT_FLAG_HAS_BIAS = 1 << 0
CSTT_FLAG_SLANEY_NORM = 1 << 1
CSTT_FLAG_REL_PE = 1 << 2


def download_nemo(model_id: str, cache_dir: str = None) -> Path:
    """Download a .nemo file from HuggingFace."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(model_id, filename=f"{model_id.split('/')[-1]}.nemo",
                               cache_dir=cache_dir)
        return Path(path)
    except Exception:
        pass

    for fname in ["model.nemo", f"{model_id.split('/')[-1]}.nemo"]:
        try:
            path = hf_hub_download(model_id, filename=fname, cache_dir=cache_dir)
            return Path(path)
        except Exception:
            continue

    tmpdir = tempfile.mkdtemp(prefix="nemo_")
    for fname in ["model_config.yaml", "model_weights.ckpt"]:
        try:
            hf_hub_download(model_id, filename=fname, local_dir=tmpdir)
        except Exception as e:
            print(f"Failed to download {fname}: {e}", file=sys.stderr)
            sys.exit(1)

    for tok_name in ["tokenizer.model", "tokenizer_spe_unigram_v1024.model",
                     "tokenizer_spe_bpe_v1024.model", "tokenizer_spe_bpe_v128.model"]:
        try:
            hf_hub_download(model_id, filename=tok_name, local_dir=tmpdir)
        except Exception:
            continue

    return Path(tmpdir)


def extract_nemo(nemo_path: Path) -> tuple:
    """Extract config, state_dict, and tokenizer from a .nemo file or directory."""
    import torch
    import yaml

    if nemo_path.is_dir():
        config_path = nemo_path / "model_config.yaml"
        weights_path = nemo_path / "model_weights.ckpt"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        tok_path = None
        for p in nemo_path.iterdir():
            if p.suffix == ".model" and "tokenizer" in p.name:
                tok_path = p
                break
        return config, state_dict, tok_path

    config = None
    state_dict = None
    tok_data = None

    with tarfile.open(nemo_path, "r:*") as tar:
        for member in tar.getmembers():
            name = os.path.basename(member.name)
            if name == "model_config.yaml":
                f = tar.extractfile(member)
                config = yaml.safe_load(f.read())
            elif name == "model_weights.ckpt":
                f = tar.extractfile(member)
                buf = io.BytesIO(f.read())
                state_dict = torch.load(buf, map_location="cpu", weights_only=False)
            elif name.endswith(".model") and "tokenizer" in name:
                f = tar.extractfile(member)
                tok_data = f.read()

    if config is None or state_dict is None:
        print("Error: Could not extract config and weights from .nemo file", file=sys.stderr)
        sys.exit(1)

    tok_path = None
    if tok_data:
        tok_path = Path(tempfile.mktemp(suffix=".model"))
        tok_path.write_bytes(tok_data)

    return config, state_dict, tok_path


def parse_encoder_config(config: dict) -> dict:
    """Extract encoder hyperparameters from NeMo config."""
    enc = config.get("encoder", config)
    result = {
        "d_model": enc.get("d_model", 512),
        "n_layers": enc.get("n_layers", 17),
        "n_heads": enc.get("n_heads", 8),
        "ff_mult": enc.get("ff_expansion_factor", 4),
        "conv_kernel": enc.get("conv_kernel_size", 9),
        "subsampling": enc.get("subsampling", "dw_striding"),
        "subsampling_factor": enc.get("subsampling_factor", 8),
        "subsampling_conv_channels": enc.get("subsampling_conv_channels", -1),
    }
    if result["subsampling_conv_channels"] <= 0:
        result["subsampling_conv_channels"] = result["d_model"]
    return result


def parse_preprocessor_config(config: dict) -> dict:
    """Extract audio preprocessor config."""
    pre = config.get("preprocessor", {})
    sr = pre.get("sample_rate", 16000)
    return {
        "sample_rate": sr,
        "n_mels": pre.get("features", pre.get("n_mels", 80)),
        "n_fft": pre.get("n_fft", 512),
        "hop_length": int(pre.get("hop_length", pre.get("window_stride", 0.01) * sr)),
        "win_length": int(pre.get("win_length", pre.get("window_size", 0.025) * sr)),
    }


def detect_subsampling_convs(state_dict: dict, proj_feat_in: int, n_mels: int) -> list:
    """Detect subsampling conv layers and infer strides from projection dim."""
    convs = []
    prefix = "encoder.pre_encode.conv"

    # Collect all conv layers (skip ReLU gaps in indices)
    prev_c_out = 1  # mel spectrogram has 1 channel
    i = 0
    while i < 20:
        wkey = f"{prefix}.{i}.weight"
        if wkey in state_dict:
            w = state_dict[wkey]
            b = state_dict.get(f"{prefix}.{i}.bias")
            c_out, c_in_per_g = w.shape[0], w.shape[1]
            kh = w.shape[2] if w.ndim >= 3 else 1

            # Depthwise: c_in_per_g == 1, spatial kernel, and c_out == prev_c_out
            # (i.e., same channel count in/out, each channel gets its own filter)
            is_depthwise = (c_in_per_g == 1 and c_out > 1 and kh > 1
                            and c_out == prev_c_out)
            groups = c_out if is_depthwise else 1
            c_in = c_in_per_g * groups

            convs.append({
                "idx": i, "w": w, "b": b,
                "c_in": c_in, "c_out": c_out,
                "kernel": kh, "stride": 1, "groups": groups,
            })
            prev_c_out = c_out
        i += 1

    # Infer strides: total freq reduction = n_mels / (proj_feat_in / C_final)
    if convs:
        import math
        C_final = convs[-1]["c_out"]
        F_final = proj_feat_in // C_final
        total_freq_reduction = n_mels // F_final if F_final > 0 else 1
        n_stride2 = int(math.log2(total_freq_reduction)) if total_freq_reduction > 1 else 0

        # Assign stride=2: depthwise convs first, then regular convs
        stride2_assigned = 0
        for c in convs:
            if stride2_assigned >= n_stride2:
                break
            if c["groups"] > 1 and c["kernel"] > 1:
                c["stride"] = 2
                stride2_assigned += 1

        if stride2_assigned < n_stride2:
            for c in convs:
                if stride2_assigned >= n_stride2:
                    break
                if c["groups"] == 1 and c["kernel"] > 1 and c["stride"] == 1:
                    c["stride"] = 2
                    stride2_assigned += 1

    return convs


def get_vocab_from_tokenizer(tok_path: Path, config: dict) -> list:
    """Extract vocabulary from SentencePiece tokenizer."""
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(str(tok_path))
        vocab = []
        for i in range(sp.GetPieceSize()):
            piece = sp.IdToPiece(i)
            piece = piece.replace("\u2581", " ")
            vocab.append(piece)
        return vocab
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}", file=sys.stderr)
        return None


def build_header(enc_config: dict, pre_config: dict, vocab_size: int,
                 sub_type: int, n_sub_convs: int, sub_feat_in: int,
                 sub_conv_kernel: int) -> bytes:
    """Build the CSTTHeader as raw bytes (24 x uint32 = 96 bytes)."""
    flags = CSTT_FLAG_HAS_BIAS | CSTT_FLAG_SLANEY_NORM | CSTT_FLAG_REL_PE

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
        0,  # dtype: fp32
        flags,
        sub_type,
        n_sub_convs,
        sub_feat_in,
        sub_conv_kernel,
        0, 0, 0, 0,
    )


def tensor_to_bytes(t) -> bytes:
    """Convert a tensor to fp32 little-endian bytes."""
    return t.detach().float().contiguous().cpu().numpy().tobytes()


def write_cstt(output_path: str, header: bytes, weight_chunks: list):
    """Write the .cstt binary file."""
    with open(output_path, "wb") as f:
        f.write(header)
        for chunk in weight_chunks:
            f.write(chunk)
    total = os.path.getsize(output_path)
    print(f"Wrote {output_path} ({total / 1024 / 1024:.1f} MB)")


def write_vocab(output_path: str, vocab: list):
    """Write the .vocab file (one token per line)."""
    vocab_path = output_path.rsplit(".", 1)[0] + ".vocab"
    with open(vocab_path, "w") as f:
        for token in vocab:
            f.write(token + "\n")
    print(f"Wrote {vocab_path} ({len(vocab)} tokens)")


def get_weight(sd: dict, key: str, required: bool = True):
    """Get a weight tensor from state dict."""
    if key in sd:
        return sd[key]
    if required:
        print(f"Error: Missing required weight: {key}", file=sys.stderr)
        prefix = ".".join(key.split(".")[:-1])
        for k in sorted(sd.keys()):
            if k.startswith(prefix):
                print(f"  {k}: {list(sd[k].shape)}", file=sys.stderr)
        sys.exit(1)
    return None


def squeeze_conv1d(t):
    """Squeeze trailing dim from Conv1d weights: [out, in, 1] → [out, in]."""
    if t.ndim == 3 and t.shape[-1] == 1:
        return t.squeeze(-1)
    return t


def convert_model(model_path: str, output_path: str):
    """Main conversion pipeline."""
    import torch

    model_path = Path(model_path)

    print(f"Loading model from {model_path}...")
    if model_path.is_file() and model_path.suffix == ".nemo":
        config, sd, tok_path = extract_nemo(model_path)
    elif model_path.is_dir():
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

    print(f"Architecture: d_model={D}, layers={n_layers}, heads={n_heads}, "
          f"ff_mult={enc_config['ff_mult']}, conv_k={K}")
    print(f"Subsampling: {enc_config['subsampling']}, factor={enc_config['subsampling_factor']}")
    print(f"Audio: {pre_config['sample_rate']}Hz, {n_mels} mels")

    # --- CTC decoder head ---
    ctc_w_key = None
    for pattern in ["ctc_decoder.decoder_layers.0.weight",
                    "decoder.decoder_layers.0.weight"]:
        if pattern in sd:
            ctc_w_key = pattern
            break
    if ctc_w_key is None:
        for k in sd:
            if "decoder_layers.0.weight" in k:
                ctc_w_key = k
                break
    if ctc_w_key is None:
        print("Error: Could not find CTC decoder weights", file=sys.stderr)
        for k in sorted(sd.keys()):
            if "decoder" in k.lower():
                print(f"  {k}: {list(sd[k].shape)}", file=sys.stderr)
        sys.exit(1)

    ctc_b_key = ctc_w_key.replace(".weight", ".bias")
    ctc_w = squeeze_conv1d(sd[ctc_w_key])
    ctc_b = sd[ctc_b_key]
    vocab_size = ctc_w.shape[0]
    print(f"CTC head: {ctc_w_key} [{list(sd[ctc_w_key].shape)}] → vocab_size={vocab_size}")

    # --- Projection layer (determines sub_feat_in) ---
    proj_w = get_weight(sd, "encoder.pre_encode.out.weight", required=False)
    proj_b = get_weight(sd, "encoder.pre_encode.out.bias", required=False)
    if proj_w is None:
        proj_w = get_weight(sd, "encoder.pre_encode.out.0.weight")
        proj_b = get_weight(sd, "encoder.pre_encode.out.0.bias")
    sub_feat_in = proj_w.shape[1]
    print(f"Projection: [{list(proj_w.shape)}], sub_feat_in={sub_feat_in}")

    # --- Subsampling convolutions ---
    sub_convs = detect_subsampling_convs(sd, sub_feat_in, n_mels)
    print(f"Subsampling: {len(sub_convs)} conv layers detected")
    for sc in sub_convs:
        w = sc["w"]
        print(f"  conv.{sc['idx']}: {list(w.shape)}, groups={sc['groups']}, stride={sc['stride']}")

    has_depthwise = any(sc["groups"] > 1 for sc in sub_convs)
    if has_depthwise:
        sub_type = CSTT_SUB_DW_STRIDING
    elif sub_convs and sub_convs[0]["w"].ndim == 4:
        sub_type = CSTT_SUB_CONV2D
    else:
        sub_type = CSTT_SUB_CONV1D

    sub_conv_kernel = sub_convs[0]["kernel"] if sub_convs else 3

    # --- Build weight data ---
    chunks = []

    # Subsampling: conv descriptors (5 uint32 each) then weights
    for sc in sub_convs:
        desc = struct.pack("<5I", sc["c_in"], sc["c_out"], sc["kernel"],
                           sc["stride"], sc["groups"])
        chunks.append(desc)

    for sc in sub_convs:
        chunks.append(tensor_to_bytes(sc["w"]))
        if sc["b"] is not None:
            chunks.append(tensor_to_bytes(sc["b"]))
        else:
            chunks.append(b"\x00" * (sc["c_out"] * 4))

    # Projection
    chunks.append(tensor_to_bytes(proj_w))
    chunks.append(tensor_to_bytes(proj_b))

    # --- Conformer blocks ---
    # Detect naming convention from first block
    has_linear1 = f"encoder.layers.0.feed_forward1.linear1.weight" in sd
    has_conv_module = f"encoder.layers.0.conv_module.pointwise_conv1.weight" in sd
    ff_up_suffix = "linear1" if has_linear1 else "0"
    ff_down_suffix = "linear2" if has_linear1 else "2"
    conv_prefix = "conv_module" if has_conv_module else "conv"

    print(f"Converting {n_layers} conformer blocks...")
    print(f"  FFN naming: feed_forward1.{ff_up_suffix}/{ff_down_suffix}")
    print(f"  Conv naming: {conv_prefix}.*")

    for i in range(n_layers):
        pre = f"encoder.layers.{i}"

        # FFN1
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.norm_feed_forward1.weight")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.norm_feed_forward1.bias")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.feed_forward1.{ff_up_suffix}.weight")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.feed_forward1.{ff_up_suffix}.bias")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.feed_forward1.{ff_down_suffix}.weight")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.feed_forward1.{ff_down_suffix}.bias")))

        # MHSA
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.norm_self_att.weight")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.norm_self_att.bias")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.self_attn.linear_q.weight")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.self_attn.linear_q.bias")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.self_attn.linear_k.weight")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.self_attn.linear_k.bias")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.self_attn.linear_v.weight")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.self_attn.linear_v.bias")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.self_attn.linear_out.weight")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.self_attn.linear_out.bias")))

        # Relative PE: linear_pos projection + biases
        pos_bias_u = get_weight(sd, f"{pre}.self_attn.pos_bias_u", required=False)
        pos_bias_v = get_weight(sd, f"{pre}.self_attn.pos_bias_v", required=False)
        linear_pos_w = get_weight(sd, f"{pre}.self_attn.linear_pos.weight", required=False)
        if pos_bias_u is not None:
            if linear_pos_w is not None:
                chunks.append(tensor_to_bytes(linear_pos_w))
            else:
                chunks.append(b"\x00" * (D * D * 4))
            chunks.append(tensor_to_bytes(pos_bias_u))
            chunks.append(tensor_to_bytes(pos_bias_v))
        else:
            d_head = D // n_heads
            chunks.append(b"\x00" * (D * D * 4))
            chunks.append(b"\x00" * (n_heads * d_head * 4))
            chunks.append(b"\x00" * (n_heads * d_head * 4))

        # Conv module
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.norm_conv.weight")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.norm_conv.bias")))

        pw1_w = squeeze_conv1d(get_weight(sd, f"{pre}.{conv_prefix}.pointwise_conv1.weight"))
        chunks.append(tensor_to_bytes(pw1_w))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.{conv_prefix}.pointwise_conv1.bias")))

        dw_w = get_weight(sd, f"{pre}.{conv_prefix}.depthwise_conv.weight")
        if dw_w.ndim == 3:
            dw_w = dw_w.squeeze(1)  # [D, 1, K] → [D, K]
        chunks.append(tensor_to_bytes(dw_w))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.{conv_prefix}.depthwise_conv.bias")))

        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.{conv_prefix}.batch_norm.weight")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.{conv_prefix}.batch_norm.bias")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.{conv_prefix}.batch_norm.running_mean")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.{conv_prefix}.batch_norm.running_var")))

        pw2_w = squeeze_conv1d(get_weight(sd, f"{pre}.{conv_prefix}.pointwise_conv2.weight"))
        chunks.append(tensor_to_bytes(pw2_w))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.{conv_prefix}.pointwise_conv2.bias")))

        # FFN2
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.norm_feed_forward2.weight")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.norm_feed_forward2.bias")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.feed_forward2.{ff_up_suffix}.weight")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.feed_forward2.{ff_up_suffix}.bias")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.feed_forward2.{ff_down_suffix}.weight")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.feed_forward2.{ff_down_suffix}.bias")))

        # Final norm
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.norm_out.weight")))
        chunks.append(tensor_to_bytes(get_weight(sd, f"{pre}.norm_out.bias")))

    # CTC head (squeeze Conv1d → Linear shape)
    chunks.append(tensor_to_bytes(ctc_w))
    chunks.append(tensor_to_bytes(ctc_b))

    # --- Write output files ---
    header = build_header(enc_config, pre_config, vocab_size,
                          sub_type, len(sub_convs), sub_feat_in, sub_conv_kernel)
    write_cstt(output_path, header, chunks)

    vocab = None
    if tok_path:
        vocab = get_vocab_from_tokenizer(tok_path, config)
    if vocab is None:
        print("Warning: No tokenizer found, generating numbered vocabulary")
        vocab = ["<blank>"] + [f"tok_{i}" for i in range(1, vocab_size)]

    # NeMo CTC models put <blank> at the end (index = vocab_size - 1).
    # SentencePiece gives us vocab_size - 1 tokens; append <blank>.
    while len(vocab) < vocab_size:
        vocab.append("<blank>")

    write_vocab(output_path, vocab)

    total_params = sum(sd[k].numel() for k in sd
                       if k.startswith("encoder.") or k.startswith("ctc_decoder."))
    print(f"\nConversion complete!")
    print(f"  Parameters: {total_params:,}")
    print(f"  Model: {output_path}")
    print(f"  Vocab: {output_path.rsplit('.', 1)[0]}.vocab")
    print(f"\nTo use with pocket-voice:")
    print(f"  ConformerSTT *stt = conformer_stt_create(\"{output_path}\");")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeMo ASR models to .cstt format for pocket-voice")
    parser.add_argument("model",
                        help="HuggingFace model ID (e.g. nvidia/parakeet-tdt_ctc-110m) "
                             "or path to .nemo file")
    parser.add_argument("-o", "--output", default="model.cstt",
                        help="Output .cstt file path (default: model.cstt)")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace cache directory")
    parser.add_argument("--list-keys", action="store_true",
                        help="List all state dict keys and exit")

    args = parser.parse_args()
    _check_deps()

    if args.list_keys:
        model_path = Path(args.model)
        if model_path.is_file() or model_path.is_dir():
            config, sd, tok_path = extract_nemo(model_path)
        else:
            local_path = download_nemo(args.model, args.cache_dir)
            config, sd, tok_path = extract_nemo(local_path)
        for k in sorted(sd.keys()):
            print(f"{k}: {list(sd[k].shape)} {sd[k].dtype}")
        return

    convert_model(args.model, args.output)


if __name__ == "__main__":
    main()
