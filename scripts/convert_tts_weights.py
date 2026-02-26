#!/usr/bin/env python3
"""Convert Kyutai TTS-1.6B safetensors to the format expected by the moshi Rust crate.

The TTS-1.6B model uses a 'weights-per-step' DepFormer architecture:
  - Shared transformer layers with multi-linear gating (11 weight steps)
  - Packed multi-linear self-attention weights
  - Per-codebook embeddings at the top level

The Rust moshi crate expects per-slice DepFormer:
  - 32 separate transformers, one per codebook
  - All weights nested under depformer.{slice_idx}.*

This script expands shared weights into per-slice copies and renames keys.
"""

import argparse
import json
import os
import glob
import torch
from safetensors.torch import load_file, save_file


def find_model_files(hf_repo="kyutai/tts-1.6b-en_fr"):
    """Find model files in HuggingFace cache."""
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    repo_dir = hf_repo.replace("/", "--")
    
    config_matches = glob.glob(
        f"{hf_cache}/models--{repo_dir}/**/config.json", recursive=True
    )
    config_path = next((m for m in config_matches if "snapshot" in m), None)
    if not config_path:
        raise FileNotFoundError(f"Config not found for {hf_repo}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    snapshot_dir = os.path.dirname(config_path)
    model_path = os.path.join(snapshot_dir, config["moshi_name"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return config, model_path


def convert_weights(config, model_path, output_path):
    """Convert TTS-1.6B weights to moshi Rust crate format."""
    print(f"Loading model from {model_path}...")
    tensors = load_file(model_path)
    print(f"Loaded {len(tensors)} tensors ({sum(t.numel() for t in tensors.values()) / 1e6:.1f}M params)")
    
    schedule = config["depformer_weights_per_step_schedule"]
    n_slices = len(schedule)  # 32
    n_weight_steps = max(schedule) + 1  # 11
    dep_layers = config["depformer_num_layers"]  # 4
    dep_dim = config["depformer_dim"]  # 1024
    main_dim = config["dim"]  # 2048
    
    print(f"Schedule: {n_slices} slices, {n_weight_steps} weight steps, {dep_layers} layers")
    
    output = {}
    
    # === MAIN TRANSFORMER (mostly pass-through, split cross-attention) ===
    for key, tensor in tensors.items():
        if not key.startswith("transformer."):
            continue
        
        if ".cross_attention.in_proj_weight" in key:
            # Split packed [3*dim, dim] → Q [dim, dim] + KV [2*dim, dim]
            q_weight = tensor[:main_dim, :]
            kv_weight = tensor[main_dim:, :]
            new_key = key.replace(".in_proj_weight", ".in_proj_weight_q")
            output[new_key] = q_weight
            new_key = key.replace(".in_proj_weight", ".in_proj_weight_kv")
            output[new_key] = kv_weight
        else:
            output[key] = tensor
    
    # === TEXT EMBEDDING (fuse out1 projection for demux_second_stream) ===
    # With demux_second_stream=True, forward is: out1(embedding(token % vocab))
    # We bake the out1 projection into the embedding weight so Rust's simple
    # embedding lookup produces the correct values. No sqrt(dim) scaling —
    # ScaledEmbedding's "scaled" refers to learning rate, not values.
    text_emb_weight = tensors["text_emb.weight"]
    if "text_emb.out1.weight" in tensors:
        out1 = tensors["text_emb.out1.weight"]  # [dim, dim]
        text_emb_weight = text_emb_weight @ out1.T
        print(f"Fused text_emb with out1 projection: {text_emb_weight.shape}")
    output["text_emb.weight"] = text_emb_weight
    
    # === TEXT LINEAR, OUT NORM ===
    output["text_linear.weight"] = tensors["text_linear.weight"]
    output["out_norm.alpha"] = tensors["out_norm.alpha"]
    
    # === AUDIO EMBEDDINGS (emb.{0..31}) — pass through, no scaling ===
    # ScaledEmbedding forward (non-demux) is just: F.embedding(input, weight)
    for i in range(n_slices):
        key = f"emb.{i}.weight"
        if key in tensors:
            output[key] = tensors[key]
    
    # === CONDITION PROVIDER ===
    for key, tensor in tensors.items():
        if key.startswith("condition_provider."):
            output[key] = tensor
    
    # === DEPFORMER: expand shared weights into per-slice ===
    print("Expanding depformer shared weights into per-slice format...")
    
    for slice_idx in range(n_slices):
        ws = schedule[slice_idx]  # weight step index
        
        # --- Embedding (no scaling — ScaledEmbedding "scale" is for LR, not values) ---
        if slice_idx == 0:
            # Text embedding (first codebook) — has out1 projection as low_rank
            output[f"depformer.{slice_idx}.emb.weight"] = tensors["depformer_text_emb.weight"]
            if "depformer_text_emb.out1.weight" in tensors:
                output[f"depformer.{slice_idx}.emb.low_rank.weight"] = tensors["depformer_text_emb.out1.weight"]
            elif "depformer_text_emb.low_rank.weight" in tensors:
                output[f"depformer.{slice_idx}.emb.low_rank.weight"] = tensors["depformer_text_emb.low_rank.weight"]
        else:
            # Audio embedding (codebook > 0)
            emb_idx = slice_idx - 1
            emb_key = f"depformer_emb.{emb_idx}.weight"
            lr_key = f"depformer_emb.{emb_idx}.low_rank.weight"
            if emb_key in tensors:
                output[f"depformer.{slice_idx}.emb.weight"] = tensors[emb_key]
            if lr_key in tensors:
                output[f"depformer.{slice_idx}.emb.low_rank.weight"] = tensors[lr_key]
        
        # --- Linear in (per weight step, clone for shared steps) ---
        lin_key = f"depformer_in.{ws}.weight"
        if lin_key in tensors:
            output[f"depformer.{slice_idx}.linear_in.weight"] = tensors[lin_key].clone().contiguous()
        
        # --- Linear out (per codebook, unique so no clone needed) ---
        out_key = f"linears.{slice_idx}.weight"
        if out_key in tensors:
            output[f"depformer.{slice_idx}.linear_out.weight"] = tensors[out_key]
        
        # --- Transformer layers ---
        for layer_idx in range(dep_layers):
            layer_prefix = f"depformer.{slice_idx}.transformer.layers.{layer_idx}"
            src_prefix = f"depformer.layers.{layer_idx}"
            
            # Self-attention: slice packed multi-linear weights
            # .clone().contiguous() required because shared weights cause safetensors save errors
            packed_in = tensors.get(f"{src_prefix}.self_attn.in_proj_weight")
            if packed_in is not None:
                chunk_size = 3 * dep_dim
                start = ws * chunk_size
                output[f"{layer_prefix}.self_attn.in_proj_weight"] = packed_in[start:start + chunk_size, :].clone().contiguous()
            
            packed_out = tensors.get(f"{src_prefix}.self_attn.out_proj.weight")
            if packed_out is not None:
                start = ws * dep_dim
                output[f"{layer_prefix}.self_attn.out_proj.weight"] = packed_out[start:start + dep_dim, :].clone().contiguous()
            
            # Gating: copy per-step weights (clone for shared steps)
            for suffix in ["linear_in.weight", "linear_out.weight"]:
                src_key = f"{src_prefix}.gating.{ws}.{suffix}"
                if src_key in tensors:
                    output[f"{layer_prefix}.gating.{suffix}"] = tensors[src_key].clone().contiguous()
            
            # Norms: shared across all steps (clone to avoid shared memory)
            for norm_name in ["norm1", "norm2"]:
                alpha_key = f"{src_prefix}.{norm_name}.alpha"
                if alpha_key in tensors:
                    output[f"{layer_prefix}.{norm_name}.alpha"] = tensors[alpha_key].clone().contiguous()
    
    # === Validation ===
    print(f"\nConverted: {len(output)} tensors")
    total_params = sum(t.numel() for t in output.values())
    print(f"Total parameters: {total_params / 1e6:.1f}M ({total_params * 4 / 1e9:.2f} GB in f32)")
    
    # Check for expected key patterns
    dep_keys = [k for k in output if k.startswith("depformer.")]
    trans_keys = [k for k in output if k.startswith("transformer.")]
    print(f"  Depformer keys: {len(dep_keys)}")
    print(f"  Transformer keys: {len(trans_keys)}")
    
    # Spot check a few key shapes
    checks = [
        (f"depformer.0.transformer.layers.0.self_attn.in_proj_weight", (3 * dep_dim, dep_dim)),
        (f"depformer.0.transformer.layers.0.gating.linear_in.weight", None),
        (f"depformer.0.emb.weight", None),
        (f"depformer.0.linear_in.weight", (dep_dim, main_dim)),
        (f"depformer.0.linear_out.weight", None),
    ]
    for key, expected_shape in checks:
        if key in output:
            shape = tuple(output[key].shape)
            status = "✓" if expected_shape is None or shape == expected_shape else f"✗ (expected {expected_shape})"
            print(f"  {key}: {shape} {status}")
        else:
            print(f"  {key}: MISSING!")
    
    # Save
    print(f"\nSaving to {output_path}...")
    save_file(output, output_path)
    file_size = os.path.getsize(output_path) / (1024 * 1024 * 1024)
    print(f"Done! Output: {file_size:.2f} GB")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert TTS-1.6B weights for moshi Rust crate")
    parser.add_argument("--repo", default="kyutai/tts-1.6b-en_fr", help="HuggingFace repo")
    parser.add_argument("--output", default=None, help="Output safetensors path")
    args = parser.parse_args()
    
    config, model_path = find_model_files(args.repo)
    
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(model_path),
            "converted_for_rust.safetensors"
        )
    
    convert_weights(config, model_path, args.output)
    print(f"\nTo use with pocket-voice TTS, set the model path to:\n  {args.output}")


if __name__ == "__main__":
    main()
