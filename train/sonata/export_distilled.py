#!/usr/bin/env python3
"""Export distilled Flow v3 checkpoint to safetensors format for Rust inference.

Extracts the student model from a distilled checkpoint (after consistency distillation)
and exports weights in safetensors format for sonata_flow Rust inference engine.

The distilled student is trained to produce high-quality output in 1-2 ODE steps
instead of the teacher's 8 steps, enabling real-time voice synthesis on Apple Silicon.

Usage:
  python train/sonata/export_distilled.py \
    --checkpoint checkpoints/flow_v3_distilled/flow_v3_distill_final.pt \
    --output-dir models/sonata_flow_distilled \
    --student-steps 1
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from safetensors.torch import save_file


def export_distilled_flow(
    ckpt_path: str,
    output_dir: str,
    student_steps: int = 1,
    config_override: str = "",
) -> None:
    """
    Export distilled Flow v3 student weights.

    Args:
        ckpt_path: Path to distilled checkpoint (flow_v3_distill_*.pt)
        output_dir: Output directory for safetensors + config.json
        student_steps: Number of ODE steps for inference (typically 1-2 for distilled)
        config_override: Optional external config.json to use instead of checkpoint config
    """
    # Load distilled checkpoint
    print(f"[export] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract student state dict and config
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        config = ckpt.get("config", {})
    else:
        # Fallback: raw state dict (shouldn't happen for distillation checkpoints)
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

    # Convert to contiguous and prepare for safetensors
    flow_state = {k: v.contiguous() for k, v in state.items()}

    # Detect optional conditioning features in the model
    has_speaker = any("speaker" in k for k in flow_state)
    has_emotion = any("emotion" in k for k in flow_state)
    has_prosody = any("prosody" in k for k in flow_state)
    has_ref_audio = any("ref_audio" in k for k in flow_state)

    # Save weights
    safetensors_path = out / "sonata_flow_distilled.safetensors"
    save_file(flow_state, str(safetensors_path))
    print(f"[export] Weights: {safetensors_path}")

    # Build config for Rust inference
    # Use student_steps instead of teacher's n_steps_inference
    flow_config = {
        "d_model": config.get("d_model", 768),
        "n_layers": config.get("n_layers", 16),
        "n_heads": config.get("n_heads", 12),
        "acoustic_dim": config.get("acoustic_dim", 256),
        "cond_dim": config.get("cond_dim", 768),
        "semantic_vocab_size": config.get("semantic_vocab_size", 32768),
        "n_steps_inference": student_steps,  # Override: use 1-2 steps for distilled model
        "sigma_min": config.get("sigma_min", 1e-4),
        "n_emotions": config.get("n_emotions", 0) if has_emotion else 0,
        "prosody_dim": config.get("prosody_dim", 3) if has_prosody else 0,
        "n_speakers": config.get("n_speakers", 0) if has_speaker else 0,
        "speaker_dim": config.get("speaker_dim", 256),
        "use_rope": config.get("use_rope", True),
        "use_ref_audio": config.get("use_ref_audio", False) or has_ref_audio,
        "ref_audio_dim": config.get("ref_audio_dim", 80) if has_ref_audio else 0,
        "distilled": True,  # Flag to indicate this is a distilled (fast) model
        "distillation_student_steps": student_steps,  # For verification/logging
    }

    config_path = out / "sonata_flow_distilled_config.json"
    with open(config_path, "w") as f:
        json.dump(flow_config, f, indent=2)
    print(f"[export] Config: {config_path}")

    # Calculate and report model statistics
    n_params = sum(v.numel() for v in flow_state.values())
    total_bytes = sum(v.numel() * v.element_size() for v in flow_state.values())

    # Estimate file size (safetensors adds minimal overhead)
    file_size_mb = total_bytes / (1024 * 1024)

    # Feature summary
    extras = []
    if has_emotion:
        extras.append(f"emotion({flow_config['n_emotions']})")
    if has_prosody:
        extras.append(f"prosody({flow_config['prosody_dim']}D)")
    if has_speaker:
        extras.append(f"speaker({flow_config['n_speakers']})")
    if has_ref_audio:
        extras.append("ref_audio_xattn")

    suffix = f" [{', '.join(extras)}]" if extras else ""
    print(f"\n[export] Model: {n_params/1e6:.1f}M params, {file_size_mb:.1f} MB{suffix}")
    print(f"[export] Inference: {student_steps} ODE step(s) at {student_steps/8:.1f}x speedup")

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
        description="Export distilled Flow v3 student weights to safetensors for Rust inference"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to distilled checkpoint (flow_v3_distill_final.pt or flow_v3_distill_step_*.pt)"
    )
    parser.add_argument(
        "--output-dir",
        default="models/sonata_flow_distilled",
        help="Output directory for safetensors + config.json"
    )
    parser.add_argument(
        "--student-steps",
        type=int,
        default=1,
        help="Number of ODE steps at inference (1=real-time, 2=higher quality)"
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

    # Validate student steps
    if args.student_steps < 1 or args.student_steps > 8:
        print(f"WARNING: student_steps={args.student_steps}. Typical range: 1-2 (was trained with 8-step teacher)")

    export_distilled_flow(
        str(ckpt_path),
        args.output_dir,
        student_steps=args.student_steps,
        config_override=args.config,
    )


if __name__ == "__main__":
    main()
