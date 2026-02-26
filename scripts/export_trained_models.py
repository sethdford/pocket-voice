#!/usr/bin/env python3
"""Export trained checkpoints to safetensors for Rust inference.

Uses the MATCHING set of models:
  - Codec: checkpoints/codec/sonata_codec_final.pt (dec_dim=384, step 15k)
  - LM:    checkpoints/lm/sonata_lm_step_20000.pt (d=1024, L=16, step 20k)
  - Flow:  checkpoints/flow_v3/sonata_flow_final.pt (d=256, L=4, step 5k)

All trained on encoded_dev-clean.pt which was encoded by codec_final.
"""

import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models", "sonata")


def export_model(ckpt_path, out_safetensors, config_out_path, key_filter=None, label=""):
    """Generic export: load .pt checkpoint → save safetensors + config JSON."""
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] Not found: {ckpt_path}")
        return False

    print(f"  Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    sd = ckpt["model"]
    cfg = ckpt.get("config", {})
    step = ckpt.get("step", "?")
    print(f"  Step: {step}, Config: {cfg}")

    if key_filter:
        filtered = {}
        skipped = 0
        for k, v in sd.items():
            if key_filter(k):
                filtered[k] = v
            else:
                skipped += 1
        print(f"  Filtered: {len(filtered)} kept, {skipped} skipped")
        sd = filtered

    n_params = sum(t.numel() for t in sd.values())
    print(f"  Tensors: {len(sd)}, Parameters: {n_params / 1e6:.1f}M")

    # Check if model is trained (vs random init)
    n_nonzero_bias = 0
    for name, t in sd.items():
        if "bias" in name and t.float().abs().max() > 1e-6:
            n_nonzero_bias += 1
    norm_keys = [k for k in sd.keys() if "norm" in k.lower()]
    for k in norm_keys[:2]:
        t = sd[k].float()
        print(f"    {k}: mean={t.mean():.6f} std={t.std():.6f}")

    # Save safetensors
    sd_f32 = {k: v.float().contiguous() for k, v in sd.items()}
    save_file(sd_f32, out_safetensors)
    size_mb = os.path.getsize(out_safetensors) / 1e6
    print(f"  Saved: {out_safetensors} ({size_mb:.1f} MB)")

    return cfg


def main():
    print(f"\n{'='*60}")
    print(f"  Export Matching Sonata Models to Safetensors")
    print(f"{'='*60}\n")

    # Backup
    backup_dir = os.path.join(MODELS, "backup_mismatched")
    os.makedirs(backup_dir, exist_ok=True)
    for name in ["sonata_lm.safetensors", "sonata_flow.safetensors",
                  "sonata_decoder.safetensors",
                  "sonata_lm_config.json", "sonata_flow_config.json",
                  "sonata_decoder_config.json"]:
        src = os.path.join(MODELS, name)
        dst = os.path.join(backup_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Backed up: {name}")

    results = {}

    # 1. LM — use newest checkpoint
    lm_ckpt = os.path.join(ROOT, "train/checkpoints/lm/sonata_lm_step_25000.pt")
    if not os.path.exists(lm_ckpt):
        lm_ckpt = os.path.join(ROOT, "train/checkpoints/lm/sonata_lm_step_20000.pt")
    print(f"\n--- LM ({os.path.basename(lm_ckpt)}) ---")
    lm_cfg = export_model(
        lm_ckpt,
        os.path.join(MODELS, "sonata_lm.safetensors"),
        os.path.join(MODELS, "sonata_lm_config.json"),
        label="LM",
    )
    if lm_cfg:
        results["lm"] = True
        lm_config_out = {
            "d_model": lm_cfg.get("d_model", 1024),
            "n_layers": lm_cfg.get("n_layers", 16),
            "n_heads": lm_cfg.get("n_heads", 16),
            "n_kv_heads": lm_cfg.get("n_kv_heads", 4),
            "ffn_mult": lm_cfg.get("ffn_mult", 2.667),
            "max_seq_len": lm_cfg.get("max_seq_len", 4096),
            "rope_theta": lm_cfg.get("rope_theta", 10000.0),
            "text_vocab_size": lm_cfg.get("text_vocab_size", 32000),
            "semantic_vocab_size": lm_cfg.get("semantic_vocab_size", 4096),
            "n_special_tokens": lm_cfg.get("n_special_tokens", 4),
            "use_prosody": False,
            "prosody_dim": 3,
        }
        with open(os.path.join(MODELS, "sonata_lm_config.json"), "w") as f:
            json.dump(lm_config_out, f, indent=2)
        print(f"  Config saved: sonata_lm_config.json")

    # 2. Flow
    print(f"\n--- Flow (checkpoints/flow_v3/final) ---")
    flow_cfg = export_model(
        os.path.join(ROOT, "train/checkpoints/flow_v3/sonata_flow_final.pt"),
        os.path.join(MODELS, "sonata_flow.safetensors"),
        os.path.join(MODELS, "sonata_flow_config.json"),
        label="Flow",
    )
    if flow_cfg:
        results["flow"] = True
        flow_config_out = {
            "d_model": flow_cfg.get("d_model", 256),
            "n_layers": flow_cfg.get("n_layers", 4),
            "n_heads": flow_cfg.get("n_heads", 4),
            "ff_mult": flow_cfg.get("ff_mult", 4.0),
            "norm_eps": flow_cfg.get("norm_eps", 1e-5),
            "semantic_vocab_size": flow_cfg.get("semantic_vocab_size", 4096),
            "acoustic_dim": flow_cfg.get("acoustic_dim", 256),
            "cond_dim": flow_cfg.get("cond_dim", 256),
            "sigma_min": flow_cfg.get("sigma_min", 0.0001),
            "n_steps_inference": flow_cfg.get("n_steps_inference", 8),
            "speaker_dim": flow_cfg.get("speaker_dim", 256),
            "n_speakers": flow_cfg.get("n_speakers", 0),
            "n_emotions": 0,
            "emotion_dim": 256,
        }
        with open(os.path.join(MODELS, "sonata_flow_config.json"), "w") as f:
            json.dump(flow_config_out, f, indent=2)
        print(f"  Config saved: sonata_flow_config.json")

    # 3. Decoder (extract from codec checkpoint)
    print(f"\n--- Decoder (from checkpoints/codec/final) ---")
    codec_ckpt_path = os.path.join(ROOT, "train/checkpoints/codec/sonata_codec_final.pt")
    dec_cfg = export_model(
        codec_ckpt_path,
        os.path.join(MODELS, "sonata_decoder.safetensors"),
        os.path.join(MODELS, "sonata_decoder_config.json"),
        key_filter=lambda k: k.startswith("decoder.") or k.startswith("fsq."),
        label="Decoder",
    )
    if dec_cfg:
        results["decoder"] = True
        dec_config_out = {
            "n_fft": dec_cfg.get("n_fft", 1024),
            "hop_length": dec_cfg.get("hop_length", 480),
            "dec_dim": dec_cfg.get("dec_dim", 384),
            "dec_n_layers": dec_cfg.get("dec_n_layers", 8),
            "dec_conv_kernel": dec_cfg.get("dec_conv_kernel", 7),
            "dec_ff_mult": dec_cfg.get("dec_ff_mult", 4.0),
            "fsq_levels": dec_cfg.get("fsq_levels", [8, 8, 8, 8]),
            "acoustic_dim": dec_cfg.get("acoustic_dim", 256),
        }
        with open(os.path.join(MODELS, "sonata_decoder_config.json"), "w") as f:
            json.dump(dec_config_out, f, indent=2)
        print(f"  Config saved: sonata_decoder_config.json")

    # Summary
    print(f"\n{'='*60}")
    print(f"  EXPORT SUMMARY")
    print(f"{'='*60}")
    for model in ["lm", "flow", "decoder"]:
        status = "EXPORTED" if model in results else "FAILED"
        print(f"  {model:8s}: {status}")

    if all(m in results for m in ["lm", "flow", "decoder"]):
        print(f"\n  All models exported from matching checkpoints!")
        print(f"  Next: 'make test-sonata-quality' or 'make bench-sonata'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
