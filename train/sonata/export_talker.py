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
  python train/sonata/export_talker.py \
    --checkpoint checkpoints/talker/talker_final.pt \
    --output-dir models/sonata_talker
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
from safetensors.torch import save_file


def build_key_map() -> Dict[str, str]:
    """Build mapping from PyTorch state dict keys to Rust VarBuilder paths.

    This maps the training checkpoint structure to the structure expected by
    the Rust crate's VarBuilder loading.
    """
    key_map = {}

    # ─── Audio Embeddings ────────────────────────────────────────────────
    # 8 codebook embeddings: audio_emb.book.{0-7}.weight (2048 x 768)
    for i in range(8):
        key_map[f"audio_emb.codebook_embs.{i}.weight"] = f"audio_emb.book.{i}.weight"

    # ─── Thinker Projector ───────────────────────────────────────────────
    # Two-layer MLP: 4096 -> 768 -> 768
    key_map["thinker_proj.linear1.weight"] = "thinker_proj.linear1.weight"
    key_map["thinker_proj.linear2.weight"] = "thinker_proj.linear2.weight"

    # ─── Temporal Transformer ────────────────────────────────────────────
    # 12 layers with grouped query attention + FFN
    for layer_idx in range(12):
        prefix = f"temporal.layer.{layer_idx}"
        rust_prefix = f"temporal.layer.{layer_idx}"

        # Attention normalization
        key_map[f"{prefix}.attn_norm.weight"] = f"{rust_prefix}.attn_norm.weight"

        # Grouped query attention: Q (768 x 768), KV heads (256 x 768 each)
        key_map[f"{prefix}.attn.wq.weight"] = f"{rust_prefix}.attn.wq.weight"
        key_map[f"{prefix}.attn.wk.weight"] = f"{rust_prefix}.attn.wk.weight"
        key_map[f"{prefix}.attn.wv.weight"] = f"{rust_prefix}.attn.wv.weight"
        key_map[f"{prefix}.attn.wo.weight"] = f"{rust_prefix}.attn.wo.weight"

        # FFN normalization
        key_map[f"{prefix}.ffn_norm.weight"] = f"{rust_prefix}.ffn_norm.weight"

        # SwiGLU FFN: gate + up + down projections
        key_map[f"{prefix}.ffn.w_gate.weight"] = f"{rust_prefix}.ffn.w_gate.weight"
        key_map[f"{prefix}.ffn.w_up.weight"] = f"{rust_prefix}.ffn.w_up.weight"
        key_map[f"{prefix}.ffn.w_down.weight"] = f"{rust_prefix}.ffn.w_down.weight"

    # Final temporal norm
    key_map["temporal.norm.weight"] = "temporal.norm.weight"

    # ─── Semantic Head ───────────────────────────────────────────────────
    # Predicts semantic codebook (2048 vocab from 768 dim)
    key_map["semantic_head.weight"] = "semantic_head.weight"

    # ─── Depth Transformer ───────────────────────────────────────────────
    # Project to depth dimension (512)
    key_map["depth.project_in.weight"] = "depth.project_in.weight"

    # 6 layers of depth attention + FFN
    for layer_idx in range(6):
        prefix = f"depth.layer.{layer_idx}"
        rust_prefix = f"depth.layer.{layer_idx}"

        # Attention normalization
        key_map[f"{prefix}.attn_norm.weight"] = f"{rust_prefix}.attn_norm.weight"

        # Full multi-head attention (no grouping at depth level)
        key_map[f"{prefix}.attn.wq.weight"] = f"{rust_prefix}.attn.wq.weight"
        key_map[f"{prefix}.attn.wk.weight"] = f"{rust_prefix}.attn.wk.weight"
        key_map[f"{prefix}.attn.wv.weight"] = f"{rust_prefix}.attn.wv.weight"
        key_map[f"{prefix}.attn.wo.weight"] = f"{rust_prefix}.attn.wo.weight"

        # FFN normalization
        key_map[f"{prefix}.ffn_norm.weight"] = f"{rust_prefix}.ffn_norm.weight"

        # SwiGLU FFN
        key_map[f"{prefix}.ffn.w_gate.weight"] = f"{rust_prefix}.ffn.w_gate.weight"
        key_map[f"{prefix}.ffn.w_up.weight"] = f"{rust_prefix}.ffn.w_up.weight"
        key_map[f"{prefix}.ffn.w_down.weight"] = f"{rust_prefix}.ffn.w_down.weight"

    # Final depth norm
    key_map["depth.norm.weight"] = "depth.norm.weight"

    # ─── Codebook Heads ──────────────────────────────────────────────────
    # 7 acoustic codebook heads (indices 1-7, since 0 is semantic)
    for i in range(7):
        key_map[f"depth.heads.{i}.weight"] = f"depth.head.{i}.weight"
        key_map[f"depth.embs.{i}.weight"] = f"depth.emb.{i}.weight"

    return key_map


def export_talker(
    ckpt_path: str,
    output_dir: str,
    config_override: str = "",
) -> None:
    """
    Export Talker weights to safetensors format.

    Args:
        ckpt_path: Path to checkpoint (talker_final.pt or talker_step_*.pt)
        output_dir: Output directory for safetensors + config.json
        config_override: Optional external config.json to use instead of checkpoint config
    """
    # Load checkpoint
    print(f"[export] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract state dict and config
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        config = ckpt.get("config", {})
    else:
        # Fallback: raw state dict
        state = ckpt
        config = {}

    # Override config if provided
    if config_override:
        print(f"[export] Loading config override: {config_override}")
        with open(config_override) as f:
            config = json.load(f)

    # Prepare output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build key mapping and remap state dict
    print("[export] Remapping PyTorch keys to Rust VarBuilder paths...")
    key_map = build_key_map()
    talker_state = {}
    unmapped_keys = []

    for pytorch_key, value in state.items():
        if pytorch_key in key_map:
            rust_key = key_map[pytorch_key]
            talker_state[rust_key] = value.contiguous()
        else:
            unmapped_keys.append(pytorch_key)

    # Warn about unmapped keys (may be optimizer state, etc.)
    if unmapped_keys:
        print(f"[export] Warning: {len(unmapped_keys)} unmapped keys (likely optimizer state):")
        for key in unmapped_keys[:5]:  # Show first 5
            print(f"         - {key}")
        if len(unmapped_keys) > 5:
            print(f"         ... and {len(unmapped_keys) - 5} more")

    # Save weights
    safetensors_path = out / "sonata_talker.safetensors"
    save_file(talker_state, str(safetensors_path))
    print(f"[export] Weights: {safetensors_path}")

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
    with open(config_path, "w") as f:
        json.dump(talker_config, f, indent=2)
    print(f"[export] Config: {config_path}")

    # Calculate and report model statistics
    n_params = sum(v.numel() for v in talker_state.values())
    total_bytes = sum(v.numel() * v.element_size() for v in talker_state.values())
    file_size_mb = total_bytes / (1024 * 1024)

    print(f"\n[export] Model: {n_params/1e6:.1f}M params, {file_size_mb:.1f} MB")
    print(f"[export] Temporal: {talker_config['n_temporal_layers']} layers @ {talker_config['d_model']}D")
    print(f"[export] Depth: {talker_config['n_depth_layers']} layers @ {talker_config['depth_dim']}D")
    print(f"[export] Codebooks: {talker_config['n_codebooks']} × {talker_config['codebook_size']} vocab")

    # Verify weights can be loaded back
    print(f"\n[export] Verifying weights can be loaded...")
    try:
        from safetensors import safe_open
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            keys = f.keys()
            sample_key = next(iter(keys))
            sample_tensor = f.get_tensor(sample_key)
            print(f"[export] ✓ Verification passed (sample key: {sample_key}, shape: {sample_tensor.shape})")
    except Exception as e:
        print(f"[export] ✗ Verification failed: {e}")
        raise

    print(f"\n[export] Complete!")
    print(f"[export] Ready for Rust inference:")
    print(f"          weights:  {safetensors_path}")
    print(f"          config:   {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Talker checkpoint to safetensors for Rust inference"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to Talker checkpoint (talker_final.pt or talker_step_*.pt)"
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
    args = parser.parse_args()

    # Validate checkpoint exists
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return

    export_talker(
        str(ckpt_path),
        args.output_dir,
        config_override=args.config,
    )


if __name__ == "__main__":
    main()
