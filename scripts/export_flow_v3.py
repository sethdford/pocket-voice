#!/usr/bin/env python3
"""Export Sonata Flow v3 + Vocoder checkpoints to safetensors for Rust/C inference.

This exports:
  1. Flow v3 model weights → flow_v3.safetensors + flow_v3_config.json
  2. Duration predictor weights (included in flow_v3.safetensors)
  3. Vocoder generator weights → vocoder.safetensors + vocoder_config.json
  4. Phoneme vocabulary → phoneme_vocab.json (when char_vocab_size=66)

The exported models are loaded by the Rust sonata_flow crate and C sonata_istft.c.

Usage:
  python scripts/export_flow_v3.py \
    --flow-ckpt train/checkpoints/flow_v3_libritts/flow_v3_best.pt \
    --vocoder-ckpt train/checkpoints/vocoder_libritts/vocoder_epoch50.pt \
    --output-dir models/sonata/v3/

  # Large model
  python scripts/export_flow_v3.py --model-size large \
    --flow-ckpt ... --vocoder-ckpt ... --output-dir models/sonata/v3_large/

  # Use model weights (not EMA)
  python scripts/export_flow_v3.py --no-ema --flow-ckpt ... --output-dir models/sonata/
"""

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    from safetensors.torch import save_file
except ImportError:
    print("ERROR: safetensors required. Install: pip install safetensors")
    sys.exit(1)

# Add train/sonata for config and g2p imports
_train_sonata = Path(__file__).resolve().parent.parent / "train" / "sonata"
sys.path.insert(0, str(_train_sonata))

try:
    from config import (
        FlowV3Config,
        FlowV3LargeConfig,
        VocoderConfig,
        VocoderLargeConfig,
    )
    from g2p import PHONEME_VOCAB, PHONE_TO_ID, ID_TO_PHONE
except ImportError as e:
    print(f"ERROR: Could not import train/sonata modules: {e}")
    print("Run from project root: python scripts/export_flow_v3.py ...")
    sys.exit(1)


def dataclass_to_dict(obj):
    """Convert dataclass to JSON-serializable dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return obj


def _flow_config_complete(cfg: dict, model_size: str) -> dict:
    """Ensure flow config has ALL fields needed for Rust inference.

    Required for reconstruction: d_model, n_layers, n_heads, mel_dim, char_vocab_size,
    window_size, chunk_size, n_speakers, n_fft, hop_length, sample_rate, ff_mult,
    head_dim (computed).
    """
    default = FlowV3LargeConfig() if model_size == "large" else FlowV3Config()
    default_dict = dataclass_to_dict(default)

    out = {**default_dict, **cfg}
    # Compute head_dim from d_model / n_heads
    out["head_dim"] = out["d_model"] // out["n_heads"]
    # Ensure char_vocab_size is present (auto-detected or from checkpoint)
    if "char_vocab_size" not in out:
        out["char_vocab_size"] = default_dict["char_vocab_size"]
    return out


def _fuse_weight_norm(state_dict: dict) -> dict:
    """Fuse weight_g and weight_v into weight: w = g * v / ||v||.

    Vocoder uses nn.utils.parametrizations.weight_norm() which stores weight_g
    and weight_v instead of weight. Rust vocoder expects plain weight tensors.
    """
    fused = {}
    processed = set()
    for key in list(state_dict.keys()):
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            v_key = base + ".weight_v"
            if v_key in state_dict:
                g = state_dict[key]
                v = state_dict[v_key]
                # weight = g * v / ||v||; norm over parameter dims (1..ndim)
                dims = list(range(1, v.dim()))
                norm = v.norm(dim=dims, keepdim=True).clamp(min=1e-12)
                g_expand = g.view(g.shape[0], *([1] * (v.dim() - 1)))
                fused[base + ".weight"] = g_expand * v / norm
                processed.add(key)
                processed.add(v_key)
    # Copy non-weight-norm keys
    for key, val in state_dict.items():
        if key not in processed:
            fused[key] = val
    return fused


def _vocoder_config_complete(cfg: dict, model_size: str) -> dict:
    """Ensure vocoder config has ALL fields for inference (Rust reconstruction).

    Includes: n_fft, hop_length, sample_rate, win_length, n_mels,
    upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
    resblock_kernel_sizes, resblock_dilation_sizes.
    """
    default = VocoderLargeConfig() if model_size == "large" else VocoderConfig()
    default_dict = dataclass_to_dict(default)

    out = {**default_dict, **cfg}
    # All vocoder params needed for inference (exclude discriminator/training-only)
    keys = [
        "sample_rate", "n_fft", "hop_length", "n_mels", "win_length",
        "upsample_initial_channel", "upsample_rates", "upsample_kernel_sizes",
        "resblock_kernel_sizes", "resblock_dilation_sizes",
    ]
    return {k: out[k] for k in keys if k in out}


def export_flow(
    ckpt_path: str,
    output_dir: Path,
    model_size: str = "base",
    use_ema: bool = True,
) -> dict:
    """Export Flow v3 model to safetensors."""
    print(f"Loading Flow v3 from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # EMA support: when checkpoint has "ema" key, use EMA weights unless --no-ema
    has_ema = "ema" in ckpt
    if use_ema and has_ema:
        state = ckpt["ema"]
        print("  Using EMA weights")
    elif "model" in ckpt:
        state = ckpt["model"]
        print("  Using model weights" + (" (--no-ema)" if not use_ema else " (no EMA in checkpoint)"))
    elif has_ema:
        state = ckpt["ema"]
        print("  WARNING: No 'model' key, using EMA weights" + (" (--no-ema ignored)" if not use_ema else ""))
    else:
        state = ckpt
        print("  Using checkpoint state (no model/ema key)")

    config = ckpt.get("config", {})

    # Auto-detect vocab: 66 = phoneme, 256 = character-based
    char_vocab_size = config.get("char_vocab_size")
    if char_vocab_size is None:
        # Infer from interleaved_enc.char_emb weight shape
        for k, v in state.items():
            if "char_emb.weight" in k or "interleaved_enc.char_emb.weight" in k:
                char_vocab_size = v.shape[0]
                break
        if char_vocab_size is None:
            char_vocab_size = FlowV3LargeConfig().char_vocab_size if model_size == "large" else FlowV3Config().char_vocab_size

    config["char_vocab_size"] = char_vocab_size
    vocab_type = "phoneme" if char_vocab_size == 66 else "character"
    config["vocab_type"] = vocab_type
    print(f"  Vocab: char_vocab_size={char_vocab_size} ({vocab_type})")

    tensors = {}
    for name, param in state.items():
        if not isinstance(param, torch.Tensor):
            continue
        tensors[name] = param.contiguous().half()

    if not tensors:
        raise ValueError("No tensors found in checkpoint state. Check checkpoint format.")

    # Complete config for Rust
    full_config = _flow_config_complete(config, model_size)

    weights_path = output_dir / "flow_v3.safetensors"
    save_file(tensors, str(weights_path))

    config_path = output_dir / "flow_v3_config.json"
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2, default=str)

    n_params = sum(t.numel() for t in tensors.values())
    size_mb = sum(t.numel() * 2 for t in tensors.values()) / 1e6
    print(f"  Flow v3: {n_params/1e6:.1f}M params → {weights_path} ({size_mb:.1f} MB)")
    print(f"  Config: {config_path}")

    dur_params = sum(t.numel() for n, t in tensors.items() if "duration_predictor" in n)
    if dur_params > 0:
        print(f"  Duration predictor: {dur_params/1e6:.3f}M params (included)")

    # Verification: weight tensor names and shapes
    print("\n  Flow weight tensors:")
    for name in sorted(tensors.keys()):
        t = tensors[name]
        print(f"    {name}: {list(t.shape)}")

    return full_config


def export_phoneme_vocab(output_dir: Path) -> None:
    """Export phoneme vocabulary to phoneme_vocab.json.

    Contains PHONEME_VOCAB list, phone_to_id and id_to_phone mappings,
    vocab_size, and special token indices (pad, bos, eos, word_boundary, silence).
    """
    vocab = {
        "vocab": PHONEME_VOCAB,
        "phone_to_id": PHONE_TO_ID,
        "id_to_phone": {str(k): v for k, v in ID_TO_PHONE.items()},
        "vocab_size": len(PHONEME_VOCAB),
        "special_tokens": {
            "pad": 0,
            "bos": 1,
            "eos": 2,
            "word_boundary": 3,
            "silence": 4,
        },
    }
    path = output_dir / "phoneme_vocab.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"  Phoneme vocab: {len(PHONEME_VOCAB)} tokens → {path}")


def export_vocoder(
    ckpt_path: str,
    output_dir: Path,
    model_size: str = "base",
) -> dict:
    """Export Vocoder generator to safetensors with complete config."""
    print(f"Loading Vocoder from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    gen_state = ckpt.get("generator", None)
    if gen_state is None:
        model = ckpt.get("model", ckpt)
        if isinstance(model, dict) and "generator" in model and not any(
            str(k).startswith("generator.") for k in model.keys()
        ):
            gen_state = model["generator"]
        else:
            full_state = model if isinstance(model, dict) else ckpt
            gen_state = {
                k.replace("generator.", ""): v
                for k, v in full_state.items()
                if str(k).startswith("generator.")
            }

    if not gen_state:
        print("  ERROR: No generator state found in checkpoint (expected 'generator' or 'model' with 'generator.*' keys)")
        raise ValueError("Vocoder checkpoint has no generator state")

    # Fuse weight_norm (weight_g + weight_v -> weight) for Rust compatibility
    if any(k.endswith(".weight_g") for k in gen_state.keys()):
        gen_state = _fuse_weight_norm(gen_state)
        print("  Fused weight_norm layers (weight_g/weight_v -> weight)")

    config = ckpt.get("config", {})
    full_config = _vocoder_config_complete(config, model_size)

    tensors = {}
    for name, param in gen_state.items():
        tensors[name] = param.contiguous().float()

    weights_path = output_dir / "vocoder.safetensors"
    save_file(tensors, str(weights_path))

    config_path = output_dir / "vocoder_config.json"
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2, default=str)

    n_params = sum(t.numel() for t in tensors.values())
    size_mb = sum(t.numel() * 4 for t in tensors.values()) / 1e6
    print(f"  Vocoder: {n_params/1e6:.1f}M params → {weights_path} ({size_mb:.1f} MB)")
    print(f"  Config: {config_path}")

    # Verification
    print("\n  Vocoder weight tensors:")
    for name in sorted(tensors.keys()):
        t = tensors[name]
        print(f"    {name}: {list(t.shape)}")

    return full_config


def main():
    parser = argparse.ArgumentParser(description="Export Sonata Flow v3 + Vocoder")
    parser.add_argument("--flow-ckpt", required=True, help="Flow v3 checkpoint")
    parser.add_argument("--vocoder-ckpt", default="", help="Vocoder checkpoint (optional)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--model-size",
        choices=["base", "large"],
        default="base",
        help="Model size for config defaults (base=55M, large=~150M)",
    )
    parser.add_argument("--no-ema", action="store_true", help="Don't use EMA weights")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    flow_cfg = export_flow(
        args.flow_ckpt,
        out,
        model_size=args.model_size,
        use_ema=not args.no_ema,
    )

    # Export phoneme vocab when char_vocab_size is 66
    if flow_cfg.get("char_vocab_size") == 66:
        export_phoneme_vocab(out)

    if args.vocoder_ckpt:
        export_vocoder(args.vocoder_ckpt, out, model_size=args.model_size)

    print(f"\nExport complete → {out}/")
    print("Load in Rust: sonata_flow_create(\"flow_v3.safetensors\", \"flow_v3_config.json\", ...)")


if __name__ == "__main__":
    main()
