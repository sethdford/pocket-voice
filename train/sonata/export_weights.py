#!/usr/bin/env python3
"""Export trained Sonata model weights to safetensors format for Rust/C inference.

Exports:
  1. Codec decoder (ConvNeXt backbone + iSTFT head) → models/sonata_decoder.safetensors
  2. Semantic LM → models/sonata_lm.safetensors
  3. Flow → models/sonata_flow.safetensors
  4. Config JSON files for each

Usage:
  python train/sonata/export_weights.py \
    --codec-ckpt train/checkpoints/codec/sonata_codec_final.pt \
    --lm-ckpt train/checkpoints/lm/sonata_lm_final.pt \
    --flow-ckpt train/checkpoints/flow/sonata_flow_final.pt \
    --output-dir models/sonata
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def export_codec_decoder(ckpt_path: str, output_dir: str):
    """Export codec decoder weights (ConvNeXt + iSTFT head)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    config = ckpt["config"]

    decoder_state = {}
    for k, v in state.items():
        if k.startswith("decoder.") or k.startswith("fsq."):
            decoder_state[k] = v.contiguous()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_file(decoder_state, str(out / "sonata_decoder.safetensors"))
    with open(out / "sonata_decoder_config.json", "w") as f:
        json.dump({
            "n_fft": config.get("n_fft", 1024),
            "hop_length": config.get("hop_length", 480),
            "dec_dim": config.get("dec_dim", 384),
            "dec_n_layers": config.get("dec_n_layers", 8),
            "dec_conv_kernel": config.get("dec_conv_kernel", 7),
            "dec_ff_mult": config.get("dec_ff_mult", 4.0),
            "fsq_levels": config.get("fsq_levels", [8, 8, 8, 8]),
            "acoustic_dim": config.get("acoustic_dim", 256),
        }, f, indent=2)

    n_params = sum(v.numel() for v in decoder_state.values())
    print(f"[export] Codec decoder: {n_params/1e6:.1f}M params → {out / 'sonata_decoder.safetensors'}")


def export_lm(ckpt_path: str, output_dir: str):
    """Export Semantic LM weights (including prosody conditioning if present)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    config = ckpt["config"]

    lm_state = {k: v.contiguous() for k, v in state.items()}

    has_prosody = any(k.startswith("prosody_") for k in lm_state)
    has_duration = any("duration" in k for k in lm_state)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_file(lm_state, str(out / "sonata_lm.safetensors"))
    lm_config = {
        "d_model": config.get("d_model", 1024),
        "n_layers": config.get("n_layers", 16),
        "n_heads": config.get("n_heads", 16),
        "n_kv_heads": config.get("n_kv_heads", 4),
        "ffn_mult": config.get("ffn_mult", 2.667),
        "max_seq_len": config.get("max_seq_len", 4096),
        "rope_theta": config.get("rope_theta", 10000.0),
        "text_vocab_size": config.get("text_vocab_size", 32000),
        "semantic_vocab_size": config.get("semantic_vocab_size", 4096),
        "n_special_tokens": config.get("n_special_tokens", 4),
        "use_prosody": config.get("use_prosody", has_prosody),
        "prosody_dim": config.get("prosody_dim", 3),
    }
    if has_duration:
        lm_config["use_duration"] = True
    with open(out / "sonata_lm_config.json", "w") as f:
        json.dump(lm_config, f, indent=2)

    n_params = sum(v.numel() for v in lm_state.values())
    extras = []
    if has_prosody:
        extras.append("prosody")
    if has_duration:
        extras.append("duration")
    suffix = f" [{', '.join(extras)}]" if extras else ""
    print(f"[export] Semantic LM: {n_params/1e6:.1f}M params{suffix} → {out / 'sonata_lm.safetensors'}")


def export_flow(ckpt_path: str, output_dir: str, config_override: str = ""):
    """Export Flow weights (including emotion/prosody/speaker/ref_audio conditioning).

    Supports both full training checkpoints {"model": ..., "config": ...}
    and raw state_dict files (from distillation). When using raw state_dict,
    pass --flow-config to provide the config JSON separately.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        config = ckpt.get("config", {})
    else:
        state = ckpt
        config = {}

    if config_override:
        import json as json_mod
        with open(config_override) as f:
            config = json_mod.load(f)

    flow_state = {k: v.contiguous() for k, v in state.items()}

    has_emotion = any("emotion" in k for k in flow_state)
    has_prosody = any("prosody" in k for k in flow_state)
    has_speaker = any("speaker" in k for k in flow_state)
    has_ref_audio = any("ref_audio" in k for k in flow_state)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_file(flow_state, str(out / "sonata_flow.safetensors"))
    flow_config = {
        "d_model": config.get("d_model", 512),
        "n_layers": config.get("n_layers", 8),
        "n_heads": config.get("n_heads", 8),
        "acoustic_dim": config.get("acoustic_dim", 256),
        "cond_dim": config.get("cond_dim", 256),
        "semantic_vocab_size": config.get("semantic_vocab_size", 4096),
        "n_steps_inference": config.get("n_steps_inference", 8),
        "sigma_min": config.get("sigma_min", 1e-4),
        "n_emotions": config.get("n_emotions", 11) if has_emotion else 0,
        "prosody_dim": config.get("prosody_dim", 3) if has_prosody else 0,
        "n_speakers": config.get("n_speakers", 0) if has_speaker else 0,
        "speaker_dim": config.get("speaker_dim", 256),
        "use_rope": config.get("use_rope", False),
        "use_ref_audio": config.get("use_ref_audio", has_ref_audio),
        "ref_audio_dim": config.get("ref_audio_dim", 80) if has_ref_audio else 0,
    }
    with open(out / "sonata_flow_config.json", "w") as f:
        json.dump(flow_config, f, indent=2)

    n_params = sum(v.numel() for v in flow_state.values())
    extras = []
    if has_emotion:
        extras.append("emotion")
    if has_prosody:
        extras.append("prosody")
    if has_speaker:
        extras.append(f"speaker({config.get('n_speakers', '?')})")
    if has_ref_audio:
        extras.append("ref_audio_xattn")
    suffix = f" [{', '.join(extras)}]" if extras else ""
    print(f"[export] Flow: {n_params/1e6:.1f}M params{suffix} → {out / 'sonata_flow.safetensors'}")


def export_soundstorm(ckpt_path: str, output_dir: str, config_override: str = ""):
    """Export SoundStorm (MaskGIT parallel decoder) weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        config = ckpt.get("config", {})
    else:
        state = ckpt
        config = {}

    if config_override:
        with open(config_override) as f:
            config = json.load(f)

    storm_state = {k: v.contiguous() for k, v in state.items()}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_file(storm_state, str(out / "sonata_storm.safetensors"))
    storm_config = {
        "d_model": config.get("d_model", 1024),
        "n_layers": config.get("n_layers", 16),
        "n_heads": config.get("n_heads", 16),
        "n_kv_heads": config.get("n_kv_heads", 4),
        "ffn_mult": config.get("ffn_mult", 2.667),
        "max_seq_len": config.get("max_seq_len", 4096),
        "text_vocab_size": config.get("text_vocab_size", 32000),
        "semantic_vocab_size": config.get("semantic_vocab_size", 32768),
        "n_special_tokens": config.get("n_special_tokens", 4),
        "n_text_layers": config.get("n_text_layers", 4),
        "norm_eps": config.get("norm_eps", 1e-5),
    }
    with open(out / "sonata_storm_config.json", "w") as f:
        json.dump(storm_config, f, indent=2)

    n_params = sum(v.numel() for v in storm_state.values())
    print(f"[export] SoundStorm: {n_params/1e6:.1f}M params → {out / 'sonata_storm.safetensors'}")


def export_causal_flow(ckpt_path: str, output_dir: str, config_override: str = ""):
    """Export CausalFlow (streaming flow model) weights.

    Uses the same safetensors format as standard flow but config includes causal fields.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        config = ckpt.get("config", {})
    else:
        state = ckpt
        config = {}

    if config_override:
        with open(config_override) as f:
            config = json.load(f)

    flow_state = {k: v.contiguous() for k, v in state.items()}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_file(flow_state, str(out / "sonata_causal_flow.safetensors"))
    flow_config = {
        "d_model": config.get("d_model", 512),
        "n_layers": config.get("n_layers", 8),
        "n_heads": config.get("n_heads", 8),
        "acoustic_dim": config.get("acoustic_dim", 256),
        "cond_dim": config.get("cond_dim", 256),
        "semantic_vocab_size": config.get("semantic_vocab_size", 32768),
        "n_steps_inference": config.get("n_steps_inference", 8),
        "sigma_min": config.get("sigma_min", 1e-4),
        "n_speakers": config.get("n_speakers", 0),
        "speaker_dim": config.get("speaker_dim", 256),
        "use_rope": config.get("use_rope", False),
        "chunk_size": config.get("chunk_size", 25),
        "causal": True,
    }
    with open(out / "sonata_causal_flow_config.json", "w") as f:
        json.dump(flow_config, f, indent=2)

    n_params = sum(v.numel() for v in flow_state.values())
    print(f"[export] CausalFlow: {n_params/1e6:.1f}M params → {out / 'sonata_causal_flow.safetensors'}")


def export_moe_flow(ckpt_path: str, output_dir: str, config_override: str = ""):
    """Export MoE Flow (Mixture of Experts flow model) weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        config = ckpt.get("config", {})
    else:
        state = ckpt
        config = {}

    if config_override:
        with open(config_override) as f:
            config = json.load(f)

    flow_state = {k: v.contiguous() for k, v in state.items()}

    has_moe = any("moe_router" in k for k in flow_state)
    n_experts_detected = 0
    if has_moe:
        expert_keys = [k for k in flow_state if "experts." in k]
        if expert_keys:
            expert_indices = set()
            for k in expert_keys:
                parts = k.split(".")
                for i, p in enumerate(parts):
                    if p == "experts" and i + 1 < len(parts):
                        expert_indices.add(int(parts[i + 1]))
            n_experts_detected = max(expert_indices) + 1 if expert_indices else 0

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_file(flow_state, str(out / "sonata_moe_flow.safetensors"))
    flow_config = {
        "d_model": config.get("d_model", 512),
        "n_layers": config.get("n_layers", 8),
        "n_heads": config.get("n_heads", 8),
        "acoustic_dim": config.get("acoustic_dim", 256),
        "cond_dim": config.get("cond_dim", 256),
        "semantic_vocab_size": config.get("semantic_vocab_size", 32768),
        "n_steps_inference": config.get("n_steps_inference", 8),
        "sigma_min": config.get("sigma_min", 1e-4),
        "n_speakers": config.get("n_speakers", 0),
        "speaker_dim": config.get("speaker_dim", 256),
        "use_rope": config.get("use_rope", False),
        "n_experts": config.get("n_experts", n_experts_detected),
        "top_k_experts": config.get("top_k_experts", 2),
        "moe_every_n": config.get("moe_every_n", 2),
    }
    with open(out / "sonata_moe_flow_config.json", "w") as f:
        json.dump(flow_config, f, indent=2)

    n_params = sum(v.numel() for v in flow_state.values())
    extras = [f"moe({n_experts_detected} experts)" if has_moe else "no_moe"]
    print(f"[export] MoE Flow: {n_params/1e6:.1f}M params [{', '.join(extras)}] → {out / 'sonata_moe_flow.safetensors'}")


def export_voice_prompt(ckpt_path: str, output_dir: str, config_override: str = ""):
    """Export VoicePrompt reference encoder weights for zero-shot voice cloning."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        config = ckpt.get("config", {})
    else:
        state = ckpt
        config = {}

    if config_override:
        with open(config_override) as f:
            config = json.load(f)

    ref_state = {}
    for k, v in state.items():
        if k.startswith("ref_encoder."):
            ref_state[k] = v.contiguous()

    if not ref_state:
        ref_state = {k: v.contiguous() for k, v in state.items()}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_file(ref_state, str(out / "sonata_ref_encoder.safetensors"))
    ref_config = {
        "n_mels": config.get("n_mels", 80),
        "n_conv_layers": config.get("n_conv_layers", 6),
        "conv_channels": config.get("conv_channels", 512),
        "output_dim": config.get("output_dim", 512),
    }
    with open(out / "sonata_ref_encoder_config.json", "w") as f:
        json.dump(ref_config, f, indent=2)

    n_params = sum(v.numel() for v in ref_state.values())
    print(f"[export] VoicePrompt RefEncoder: {n_params/1e6:.1f}M params → {out / 'sonata_ref_encoder.safetensors'}")


def export_continuous_predictor(ckpt_path: str, output_dir: str, config_override: str = ""):
    """Export ContinuousPredictor (text → continuous semantic features) weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        config = ckpt.get("config", {})
    else:
        state = ckpt
        config = {}

    if config_override:
        with open(config_override) as f:
            config = json.load(f)

    pred_state = {k: v.contiguous() for k, v in state.items()}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_file(pred_state, str(out / "sonata_continuous_predictor.safetensors"))
    pred_config = {
        "d_model": config.get("d_model", 512),
        "n_layers": config.get("n_layers", 8),
        "n_heads": config.get("n_heads", 8),
        "ff_mult": config.get("ff_mult", 4.0),
        "text_vocab_size": config.get("text_vocab_size", 32000),
        "semantic_dim": config.get("semantic_dim", 256),
        "sigma_min": config.get("sigma_min", 1e-4),
    }
    with open(out / "sonata_continuous_predictor_config.json", "w") as f:
        json.dump(pred_config, f, indent=2)

    n_params = sum(v.numel() for v in pred_state.values())
    print(f"[export] ContinuousPredictor: {n_params/1e6:.1f}M params → {out / 'sonata_continuous_predictor.safetensors'}")


def main():
    parser = argparse.ArgumentParser(description="Export Sonata model weights to safetensors for Rust/C inference")
    parser.add_argument("--codec-ckpt", default="")
    parser.add_argument("--lm-ckpt", default="")
    parser.add_argument("--flow-ckpt", default="")
    parser.add_argument("--flow-config", default="", help="External config JSON for distilled flow")
    parser.add_argument("--storm-ckpt", default="", help="SoundStorm checkpoint")
    parser.add_argument("--storm-config", default="", help="SoundStorm config JSON override")
    parser.add_argument("--causal-flow-ckpt", default="", help="Causal streaming flow checkpoint")
    parser.add_argument("--causal-flow-config", default="", help="Causal flow config JSON override")
    parser.add_argument("--moe-flow-ckpt", default="", help="MoE flow checkpoint")
    parser.add_argument("--moe-flow-config", default="", help="MoE flow config JSON override")
    parser.add_argument("--voice-prompt-ckpt", default="", help="VoicePrompt reference encoder checkpoint")
    parser.add_argument("--voice-prompt-config", default="", help="VoicePrompt config JSON override")
    parser.add_argument("--continuous-ckpt", default="", help="ContinuousPredictor checkpoint")
    parser.add_argument("--continuous-config", default="", help="ContinuousPredictor config JSON override")
    parser.add_argument("--output-dir", default="models/sonata")
    args = parser.parse_args()

    exported = False

    if args.codec_ckpt:
        export_codec_decoder(args.codec_ckpt, args.output_dir)
        exported = True
    if args.lm_ckpt:
        export_lm(args.lm_ckpt, args.output_dir)
        exported = True
    if args.flow_ckpt:
        export_flow(args.flow_ckpt, args.output_dir, args.flow_config)
        exported = True
    if args.storm_ckpt:
        export_soundstorm(args.storm_ckpt, args.output_dir, args.storm_config)
        exported = True
    if args.causal_flow_ckpt:
        export_causal_flow(args.causal_flow_ckpt, args.output_dir, args.causal_flow_config)
        exported = True
    if args.moe_flow_ckpt:
        export_moe_flow(args.moe_flow_ckpt, args.output_dir, args.moe_flow_config)
        exported = True
    if args.voice_prompt_ckpt:
        export_voice_prompt(args.voice_prompt_ckpt, args.output_dir, args.voice_prompt_config)
        exported = True
    if args.continuous_ckpt:
        export_continuous_predictor(args.continuous_ckpt, args.output_dir, args.continuous_config)
        exported = True

    if not exported:
        print("No checkpoints specified. Available flags:")
        print("  --codec-ckpt, --lm-ckpt, --flow-ckpt, --storm-ckpt,")
        print("  --causal-flow-ckpt, --moe-flow-ckpt, --voice-prompt-ckpt, --continuous-ckpt")


if __name__ == "__main__":
    main()
