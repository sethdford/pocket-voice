#!/usr/bin/env python3
"""Export Talker checkpoint to safetensors format for Rust inference.

Converts a PyTorch Talker model checkpoint (speech-to-speech acoustic codec)
to safetensors format for inference on Apple Silicon via the sonata_talker
Rust crate.

The Talker architecture includes:
- Temporal Transformer (12 layers, 768 dim) for semantic + acoustic prediction
- Depth Transformer (6 layers, 512 dim) for codebook generation
- Audio Embeddings (8 codebooks × 2048 vocab × 768 dim)
- Thinker Projection (4096 → 768 dim) for LLM conditioning

Usage:
  python3 train/sonata/export_talker.py \
    --checkpoint checkpoints/talker/talker_final.pt \
    --output-dir models/sonata_talker

  # Dry-run mode: show expected keys without checkpoint
  python3 train/sonata/export_talker.py --dry-run --output-dir models/sonata_talker
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
from safetensors.torch import save_file


def build_key_map_with_shapes() -> Dict[str, Tuple[str, List[int]]]:
    """Build mapping from PyTorch keys to (Rust path, expected shape).

    Expected shapes are based on the Rust config:
    - d_model=768, n_heads=12, n_kv_heads=4, d_ff=3072
    - depth_dim=512, depth_heads=8, depth_d_ff=2048
    - n_codebooks=8, codebook_size=2048
    - thinker_hidden_dim=4096

    Returns a dict mapping PyTorch key -> (rust_key, expected_shape)
    Shape is [out_dim, in_dim] for Linear layers, or [vocab, dim] for embeddings.
    """
    key_specs = {}

    # ─── Audio Embeddings (8 codebooks) ────────────────────────────────
    # Embeddings: (codebook_size, d_model) = (2048, 768)
    for i in range(8):
        pytorch_key = f"audio_emb.codebook_embs.{i}.weight"
        rust_key = f"audio_emb.book.{i}.weight"
        key_specs[pytorch_key] = (rust_key, [2048, 768])

    # ─── Thinker Projector (2-layer MLP) ──────────────────────────────
    # linear1: (d_model, thinker_hidden_dim) = (768, 4096)
    key_specs["thinker_proj.linear1.weight"] = ("thinker_proj.linear1.weight", [768, 4096])
    # linear2: (d_model, d_model) = (768, 768)
    key_specs["thinker_proj.linear2.weight"] = ("thinker_proj.linear2.weight", [768, 768])

    # ─── Temporal Transformer (12 layers) ──────────────────────────────
    # GQA: n_heads=12, n_kv_heads=4, head_dim=64
    # wq: (n_heads * head_dim, d_model) = (768, 768)
    # wk/wv: (n_kv_heads * head_dim, d_model) = (256, 768)
    # wo: (d_model, n_heads * head_dim) = (768, 768)
    # w_gate/w_up: (d_ff, d_model) = (3072, 768)
    # w_down: (d_model, d_ff) = (768, 3072)
    # RMSNorm: (d_model,) = (768,)
    for layer_idx in range(12):
        prefix = f"temporal.layer.{layer_idx}"
        rust_prefix = f"temporal.layer.{layer_idx}"

        # Attention norm
        key_specs[f"{prefix}.attn_norm.weight"] = (f"{rust_prefix}.attn_norm.weight", [768])

        # Attention weights
        key_specs[f"{prefix}.attn.wq.weight"] = (f"{rust_prefix}.attn.wq.weight", [768, 768])
        key_specs[f"{prefix}.attn.wk.weight"] = (f"{rust_prefix}.attn.wk.weight", [256, 768])
        key_specs[f"{prefix}.attn.wv.weight"] = (f"{rust_prefix}.attn.wv.weight", [256, 768])
        key_specs[f"{prefix}.attn.wo.weight"] = (f"{rust_prefix}.attn.wo.weight", [768, 768])

        # FFN norm
        key_specs[f"{prefix}.ffn_norm.weight"] = (f"{rust_prefix}.ffn_norm.weight", [768])

        # SwiGLU FFN
        key_specs[f"{prefix}.ffn.w_gate.weight"] = (f"{rust_prefix}.ffn.w_gate.weight", [3072, 768])
        key_specs[f"{prefix}.ffn.w_up.weight"] = (f"{rust_prefix}.ffn.w_up.weight", [3072, 768])
        key_specs[f"{prefix}.ffn.w_down.weight"] = (f"{rust_prefix}.ffn.w_down.weight", [768, 3072])

    # Final temporal norm
    key_specs["temporal.norm.weight"] = ("temporal.norm.weight", [768])

    # ─── Semantic Head ─────────────────────────────────────────────────
    # (codebook_size, d_model) = (2048, 768)
    key_specs["semantic_head.weight"] = ("semantic_head.weight", [2048, 768])

    # ─── Depth Transformer (6 layers) ──────────────────────────────────
    # Project in: (depth_dim, d_model) = (512, 768)
    key_specs["depth.project_in.weight"] = ("depth.project_in.weight", [512, 768])

    # Depth: n_heads=8, head_dim=64, d_ff=2048
    # wq/wk/wv/wo: (512, 512)
    # w_gate/w_up: (2048, 512)
    # w_down: (512, 2048)
    for layer_idx in range(6):
        prefix = f"depth.layer.{layer_idx}"
        rust_prefix = f"depth.layer.{layer_idx}"

        # Attention norm
        key_specs[f"{prefix}.attn_norm.weight"] = (f"{rust_prefix}.attn_norm.weight", [512])

        # Attention weights
        key_specs[f"{prefix}.attn.wq.weight"] = (f"{rust_prefix}.attn.wq.weight", [512, 512])
        key_specs[f"{prefix}.attn.wk.weight"] = (f"{rust_prefix}.attn.wk.weight", [512, 512])
        key_specs[f"{prefix}.attn.wv.weight"] = (f"{rust_prefix}.attn.wv.weight", [512, 512])
        key_specs[f"{prefix}.attn.wo.weight"] = (f"{rust_prefix}.attn.wo.weight", [512, 512])

        # FFN norm
        key_specs[f"{prefix}.ffn_norm.weight"] = (f"{rust_prefix}.ffn_norm.weight", [512])

        # SwiGLU FFN
        key_specs[f"{prefix}.ffn.w_gate.weight"] = (f"{rust_prefix}.ffn.w_gate.weight", [2048, 512])
        key_specs[f"{prefix}.ffn.w_up.weight"] = (f"{rust_prefix}.ffn.w_up.weight", [2048, 512])
        key_specs[f"{prefix}.ffn.w_down.weight"] = (f"{rust_prefix}.ffn.w_down.weight", [512, 2048])

    # Final depth norm
    key_specs["depth.norm.weight"] = ("depth.norm.weight", [512])

    # ─── Codebook Heads & Embeddings (7 acoustic codebooks) ────────────
    # head.i: (codebook_size, depth_dim) = (2048, 512)
    # emb.i: (codebook_size, depth_dim) = (2048, 512)
    for i in range(7):
        key_specs[f"depth.heads.{i}.weight"] = (f"depth.head.{i}.weight", [2048, 512])
        key_specs[f"depth.embs.{i}.weight"] = (f"depth.emb.{i}.weight", [2048, 512])

    return key_specs


def build_key_map() -> Dict[str, str]:
    """Build mapping from PyTorch keys to Rust VarBuilder paths (shape-agnostic)."""
    key_specs = build_key_map_with_shapes()
    return {pt_key: rust_key for pt_key, (rust_key, _) in key_specs.items()}


def check_tensor_validity(tensor: torch.Tensor, name: str) -> Tuple[bool, str]:
    """Validate a tensor for NaN/Inf and report issues.

    Returns (is_valid, message)
    """
    issues = []

    # Check for NaN
    if torch.isnan(tensor).any():
        issues.append(f"contains NaN")

    # Check for Inf
    if torch.isinf(tensor).any():
        issues.append(f"contains Inf")

    # Check for denormalized values (very small)
    if tensor.dtype in [torch.float32, torch.float16]:
        min_normal = torch.finfo(tensor.dtype).tiny
        if (tensor.abs() > 0).any() and (tensor.abs() < min_normal).any():
            issues.append(f"contains denormalized values")

    if issues:
        return False, f"{name}: " + ", ".join(issues)
    else:
        return True, f"{name}: OK"


def verify_weights(
    talker_state: Dict[str, torch.Tensor],
    key_specs: Dict[str, Tuple[str, List[int]]],
) -> Tuple[bool, List[str], List[str]]:
    """Verify exported weights match expected shapes and have no NaN/Inf.

    Returns (all_valid, errors, warnings)
    """
    errors = []
    warnings = []
    rust_keys_found = set()

    for pytorch_key, (rust_key, expected_shape) in key_specs.items():
        if rust_key not in talker_state:
            errors.append(f"Missing weight: {rust_key}")
            continue

        tensor = talker_state[rust_key]
        rust_keys_found.add(rust_key)

        # Check shape
        actual_shape = list(tensor.shape)
        if actual_shape != expected_shape:
            errors.append(
                f"Shape mismatch for {rust_key}: "
                f"expected {expected_shape}, got {actual_shape}"
            )
            continue

        # Check validity
        is_valid, msg = check_tensor_validity(tensor, rust_key)
        if not is_valid:
            errors.append(msg)

    # Check for unexpected keys
    for rust_key in talker_state.keys():
        if rust_key not in rust_keys_found:
            warnings.append(f"Unexpected key in export: {rust_key}")

    return len(errors) == 0, errors, warnings


def show_dry_run(output_dir: str) -> None:
    """Show expected keys and shapes without needing a checkpoint."""
    key_specs = build_key_map_with_shapes()
    print("\n" + "="*80)
    print("DRY RUN: Expected weight keys and shapes")
    print("="*80)
    print(f"\nTotal keys: {len(key_specs)}")
    print(f"Breakdown:")
    print(f"  - Audio embeddings: 8 codebooks")
    print(f"  - Thinker projector: 2 layers")
    print(f"  - Temporal transformer: 12 layers × 7 weights + norm")
    print(f"  - Semantic head: 1 layer")
    print(f"  - Depth transformer: 6 layers × 7 weights + project_in + norm")
    print(f"  - Codebook heads & embeddings: 7 × 2 weights")
    print(f"\nWeights by module:\n")

    sections = {
        "Audio Embeddings": [k for k in key_specs if "audio_emb" in k],
        "Thinker Projector": [k for k in key_specs if "thinker_proj" in k],
        "Temporal Transformer": [k for k in key_specs if "temporal" in k],
        "Semantic Head": [k for k in key_specs if "semantic_head" in k],
        "Depth Transformer": [k for k in key_specs if "depth" in k],
    }

    for section, keys in sections.items():
        print(f"{section} ({len(keys)} keys):")
        for key in sorted(keys)[:3]:
            rust_key, shape = key_specs[key]
            print(f"  {key:50s} -> {rust_key:45s} {shape}")
        if len(keys) > 3:
            print(f"  ... and {len(keys) - 3} more")
        print()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Write expected keys to file for reference
    keys_file = output / "EXPECTED_KEYS.txt"
    with open(keys_file, "w") as f:
        f.write("Expected safetensors keys for sonata_talker\n")
        f.write("=" * 80 + "\n\n")
        for section, keys in sections.items():
            f.write(f"{section} ({len(keys)} keys):\n")
            for key in sorted(keys):
                rust_key, shape = key_specs[key]
                f.write(f"  {rust_key:50s} {shape}\n")
            f.write("\n")

    print(f"Expected keys written to: {keys_file}")


def export_talker(
    ckpt_path: str,
    output_dir: str,
    config_override: str = "",
) -> bool:
    """
    Export Talker weights to safetensors format.

    Args:
        ckpt_path: Path to checkpoint (talker_final.pt or talker_step_*.pt)
        output_dir: Output directory for safetensors + config.json
        config_override: Optional external config.json to use instead of checkpoint config

    Returns:
        True if export succeeded, False otherwise
    """
    # Load checkpoint
    print(f"[export] Loading checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[export] ERROR: Failed to load checkpoint: {e}")
        return False

    # Extract state dict and config
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        config = ckpt.get("config", {})
    else:
        # Fallback: raw state dict
        state = ckpt
        config = {}

    print(f"[export] Checkpoint has {len(state)} keys")

    # Override config if provided
    if config_override:
        print(f"[export] Loading config override: {config_override}")
        try:
            with open(config_override) as f:
                config = json.load(f)
        except Exception as e:
            print(f"[export] ERROR: Failed to load config: {e}")
            return False

    # Prepare output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build key mapping with shape specs
    print("[export] Remapping PyTorch keys to Rust VarBuilder paths...")
    key_specs = build_key_map_with_shapes()
    key_map = build_key_map()
    talker_state = {}
    missing_keys = []
    unmapped_keys = []

    for pytorch_key in sorted(key_specs.keys()):
        if pytorch_key not in state:
            missing_keys.append(pytorch_key)
            continue

        value = state[pytorch_key]
        rust_key = key_map[pytorch_key]
        expected_shape = key_specs[pytorch_key][1]

        # Check shape before adding
        actual_shape = list(value.shape)
        if actual_shape != expected_shape:
            print(f"[export] ERROR: Shape mismatch for {pytorch_key}:")
            print(f"         Expected: {expected_shape}")
            print(f"         Got: {actual_shape}")
            return False

        talker_state[rust_key] = value.contiguous()

    # Report unmapped keys (optimizer state, etc.)
    for pytorch_key in state.keys():
        if pytorch_key not in key_specs:
            unmapped_keys.append(pytorch_key)

    if missing_keys:
        print(f"[export] ERROR: Missing {len(missing_keys)} keys from checkpoint:")
        for key in missing_keys[:5]:
            print(f"         - {key}")
        if len(missing_keys) > 5:
            print(f"         ... and {len(missing_keys) - 5} more")
        return False

    if unmapped_keys:
        print(f"[export] Note: {len(unmapped_keys)} unmapped keys (likely optimizer state):")
        for key in unmapped_keys[:3]:
            print(f"         - {key}")
        if len(unmapped_keys) > 3:
            print(f"         ... and {len(unmapped_keys) - 3} more")

    print(f"[export] ✓ All {len(talker_state)} required weights found")

    # Verify weight validity (NaN/Inf check)
    print("[export] Validating weights (checking for NaN/Inf)...")
    all_valid, errors, warnings = verify_weights(talker_state, key_specs)

    if errors:
        print(f"[export] ERROR: {len(errors)} validation errors:")
        for err in errors[:5]:
            print(f"         - {err}")
        if len(errors) > 5:
            print(f"         ... and {len(errors) - 5} more")
        return False

    if warnings:
        print(f"[export] Warning: {len(warnings)} unexpected keys (will be ignored)")
        for warn in warnings[:3]:
            print(f"         - {warn}")

    print(f"[export] ✓ All weights valid")

    # Save weights
    safetensors_path = out / "sonata_talker.safetensors"
    try:
        save_file(talker_state, str(safetensors_path))
        print(f"[export] Weights: {safetensors_path}")
    except Exception as e:
        print(f"[export] ERROR: Failed to save weights: {e}")
        return False

    # Build config for Rust inference
    talker_config = {
        # Temporal Transformer
        "d_model": config.get("d_model", 768),
        "n_temporal_layers": config.get("n_temporal_layers", 12),
        "n_heads": config.get("n_heads", 12),
        "n_kv_heads": config.get("n_kv_heads", 4),
        "d_ff": config.get("d_ff", 3072),

        # Depth Transformer
        "depth_dim": config.get("depth_dim", 512),
        "n_depth_layers": config.get("n_depth_layers", 6),
        "depth_heads": config.get("depth_heads", 8),
        "depth_d_ff": config.get("depth_d_ff", 2048),

        # Audio codec
        "n_codebooks": config.get("n_codebooks", 8),
        "codebook_size": config.get("codebook_size", 2048),
        "text_vocab_size": config.get("text_vocab_size", 32000),

        # Thinker (LLM) projection
        "thinker_hidden_dim": config.get("thinker_hidden_dim", 4096),

        # General
        "max_seq_len": config.get("max_seq_len", 4096),
        "rope_theta": config.get("rope_theta", 10000.0),
        "norm_eps": config.get("norm_eps", 1e-5),
        "frame_rate_hz": config.get("frame_rate_hz", 12.5),
        "acoustic_delay": config.get("acoustic_delay", 1),
    }

    config_path = out / "sonata_talker_config.json"
    try:
        with open(config_path, "w") as f:
            json.dump(talker_config, f, indent=2)
        print(f"[export] Config: {config_path}")
    except Exception as e:
        print(f"[export] ERROR: Failed to save config: {e}")
        return False

    # Calculate and report model statistics
    n_params = sum(v.numel() for v in talker_state.values())
    total_bytes = sum(v.numel() * v.element_size() for v in talker_state.values())
    file_size_mb = total_bytes / (1024 * 1024)

    print(f"\n[export] Model Statistics:")
    print(f"         Params: {n_params/1e6:.1f}M")
    print(f"         Size: {file_size_mb:.1f} MB")
    print(f"         Temporal: {talker_config['n_temporal_layers']} layers @ {talker_config['d_model']}D")
    print(f"         Depth: {talker_config['n_depth_layers']} layers @ {talker_config['depth_dim']}D")
    print(f"         Codebooks: {talker_config['n_codebooks']} × {talker_config['codebook_size']} vocab")

    # Verify weights can be loaded back
    print(f"\n[export] Verifying safetensors file...")
    try:
        from safetensors import safe_open
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            loaded_keys = set(f.keys())
            expected_keys = set(talker_state.keys())
            if loaded_keys == expected_keys:
                sample_key = next(iter(loaded_keys))
                sample_tensor = f.get_tensor(sample_key)
                print(f"[export] ✓ Safetensors file valid ({len(loaded_keys)} tensors)")
                print(f"         Sample: {sample_key} {list(sample_tensor.shape)}")
            else:
                missing = expected_keys - loaded_keys
                extra = loaded_keys - expected_keys
                print(f"[export] ERROR: Key mismatch in safetensors")
                if missing:
                    print(f"         Missing: {missing}")
                if extra:
                    print(f"         Extra: {extra}")
                return False
    except Exception as e:
        print(f"[export] ERROR: Failed to verify safetensors: {e}")
        return False

    print(f"\n[export] ✓ Export complete!")
    print(f"[export] Ready for Rust inference:")
    print(f"          weights:  {safetensors_path}")
    print(f"          config:   {config_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export Talker checkpoint to safetensors for Rust inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export a trained checkpoint
  python3 train/sonata/export_talker.py \\
    --checkpoint checkpoints/talker/talker_final.pt \\
    --output-dir models/sonata_talker

  # Use external config
  python3 train/sonata/export_talker.py \\
    --checkpoint checkpoints/talker/talker_final.pt \\
    --output-dir models/sonata_talker \\
    --config configs/talker_config.json

  # Dry-run: show expected keys without needing checkpoint
  python3 train/sonata/export_talker.py --dry-run --output-dir models/sonata_talker
        """
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Path to Talker checkpoint (talker_final.pt or talker_step_*.pt). Not required if --dry-run is set."
    )
    parser.add_argument(
        "--output-dir",
        default="models/sonata_talker",
        help="Output directory for safetensors + config.json"
    )
    parser.add_argument(
        "--config",
        default="",
        help="External config.json to use instead of checkpoint config"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show expected keys and shapes without requiring a checkpoint"
    )
    args = parser.parse_args()

    # Dry-run mode
    if args.dry_run:
        print("\n[dry-run] Generating expected weight spec...")
        show_dry_run(args.output_dir)
        return

    # Normal export mode: checkpoint required
    if not args.checkpoint:
        print("ERROR: --checkpoint is required (unless --dry-run is set)")
        parser.print_help()
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    success = export_talker(
        str(ckpt_path),
        args.output_dir,
        config_override=args.config,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
