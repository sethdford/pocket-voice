#!/usr/bin/env python3
"""Compute EmoSteer direction vectors from emotionally-labeled speech data.

Extracts per-layer activations from the Sonata Flow model on emotion-labeled
audio, computes mean activations per emotion, and saves direction vectors as
JSON for training-free activation steering at inference time.

The algorithm follows EmoSteer-TTS (2024):
  1. Run emotional audio through codec → semantic tokens (v1) or use mel+text (v3)
  2. Run conditioning + random noise through Flow network
  3. Capture hidden activations at each transformer layer
  4. Compute mean activation per emotion class
  5. Direction vector = mean(emotion) - mean(neutral)

Usage:
  # Flow v1 (default)
  python compute_emosteer.py \
    --data-dir data/emotional_pairs/ \
    --flow-weights models/sonata/flow.safetensors \
    --flow-config models/sonata/sonata_flow_config.json \
    --output models/sonata/emosteer_directions.json

  # Flow v3
  python compute_emosteer.py --model-version v3 \
    --flow-weights models/sonata/flow_v3.safetensors \
    --flow-config models/sonata/flow_v3_config.json \
    --output models/sonata/emosteer_v3_directions.json \
    [--phonemes] [--model-size large]

Data format:
  v1: .pt with { "semantic_tokens": LongTensor(T,), "emotion": str, "acoustic_latents": FloatTensor(T, 256) }
  v3: .pt with { "mel": FloatTensor(T, 80), "char_ids": LongTensor(T_text,) or "text": str, "emotion": str }
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import numpy as np

from config import FlowConfig, FlowV3Config, FlowV3LargeConfig


EMOTION_NAMES = [
    "happy", "excited", "sad", "angry", "fearful",
    "surprised", "warm", "serious", "calm", "confident",
    "whisper", "emphatic",
]


class ActivationCollector:
    """Hook-based activation collector for transformer layers.

    Works with both SonataFlow (v1) and SonataFlowV3. v3 FlowV3Block returns
    (x, new_cache), so we extract output[0]; v1 returns a tensor directly.
    """

    def __init__(self, model, target_layers: Optional[List[int]] = None):
        self.model = model
        self.activations: Dict[int, List[torch.Tensor]] = {}
        self.hooks = []

        n_layers = len(model.blocks)
        layers = target_layers or list(range(n_layers))

        for layer_idx in layers:
            self.activations[layer_idx] = []
            hook = model.blocks[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # v3 FlowV3Block returns (x, new_cache); v1 returns tensor
            act = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx].append(act.detach().cpu())
        return hook_fn

    def clear(self):
        for k in self.activations:
            self.activations[k] = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_mean_activations(self) -> Dict[int, torch.Tensor]:
        """Return mean activation per layer (averaged over all collected samples)."""
        result = {}
        for layer_idx, acts in self.activations.items():
            if acts:
                stacked = torch.cat(acts, dim=0)  # (N, T, D)
                result[layer_idx] = stacked.mean(dim=(0, 1))  # (D,)
        return result


def load_emotional_data(data_dir: str, manifest: Optional[str] = None) -> Dict[str, List[dict]]:
    """Load and group data by emotion label.

    Supports two modes:
      1. Directory of .pt files (each with 'emotion' key)
      2. JSONL manifest from prepare_prosody_data.py (--from-manifest)
    """
    by_emotion: Dict[str, List[dict]] = {e: [] for e in EMOTION_NAMES}
    by_emotion["neutral"] = []

    if manifest and os.path.exists(manifest):
        import json as _json
        with open(manifest) as f:
            for line in f:
                entry = _json.loads(line)
                pt_path = entry.get("pt_file", "")
                if pt_path and os.path.exists(pt_path):
                    data = torch.load(pt_path, weights_only=True)
                    emotion = data.get("emotion", entry.get("emotion", "neutral"))
                    if emotion not in by_emotion:
                        by_emotion[emotion] = []
                    by_emotion[emotion].append(data)
        print(f"  Loaded from manifest: {manifest}")
    else:
        files = sorted(Path(data_dir).glob("*.pt"))
        for f in files:
            data = torch.load(f, weights_only=True)
            emotion = data.get("emotion", "neutral")
            if emotion not in by_emotion:
                by_emotion[emotion] = []
            by_emotion[emotion].append(data)

    for emo, items in by_emotion.items():
        if items:
            print(f"  {emo}: {len(items)} utterances")

    return by_emotion


def generate_synthetic_data(n_per_emotion: int = 50, T: int = 100,
                            acoustic_dim: int = 256) -> Dict[str, List[dict]]:
    """Generate synthetic emotional data for testing the pipeline."""
    by_emotion = {}
    for emo in ["neutral"] + EMOTION_NAMES:
        items = []
        for _ in range(n_per_emotion):
            sem = torch.randint(0, 4096, (T,))
            aco = torch.randn(T, acoustic_dim)
            items.append({
                "semantic_tokens": sem,
                "acoustic_latents": aco,
                "emotion": emo,
            })
        by_emotion[emo] = items
    return by_emotion


def collect_activations(
    model: SonataFlow,
    data: List[dict],
    device: torch.device,
    max_samples: int = 100,
    target_layers: Optional[List[int]] = None,
) -> Dict[int, torch.Tensor]:
    """Run data through model and collect mean activations per layer."""
    collector = ActivationCollector(model, target_layers)

    n = min(len(data), max_samples)
    for i in range(n):
        item = data[i]
        sem = item["semantic_tokens"].unsqueeze(0).to(device)
        T = sem.shape[1]

        noise = torch.randn(1, T, model.cfg.acoustic_dim, device=device)
        t = torch.tensor([0.5], device=device)  # midpoint timestep

        with torch.no_grad():
            model(noise, t, sem)

    means = collector.get_mean_activations()
    collector.remove_hooks()
    return means


def compute_directions(
    model,
    by_emotion: Dict[str, List[dict]],
    device: torch.device,
    max_samples: int = 100,
    target_layers: Optional[List[int]] = None,
    model_version: str = "v1",
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Compute direction vectors: mean(emotion) - mean(neutral) per layer."""

    if not by_emotion.get("neutral"):
        print("  WARNING: No neutral data. Using random baseline.")
        neutral_means = {}
    else:
        print("  Computing neutral baseline...")
        neutral_means = collect_activations(
            model, by_emotion["neutral"], device, max_samples, target_layers, model_version
        )

    directions = {}
    for emo in EMOTION_NAMES:
        if not by_emotion.get(emo):
            continue
        print(f"  Computing {emo} direction...")
        emo_means = collect_activations(
            model, by_emotion[emo], device, max_samples, target_layers, model_version
        )

        emo_dirs = {}
        for layer_idx in emo_means:
            if layer_idx in neutral_means:
                direction = emo_means[layer_idx] - neutral_means[layer_idx]
            else:
                direction = emo_means[layer_idx]
            # L2 normalize
            norm = direction.norm()
            if norm > 1e-8:
                direction = direction / norm
            emo_dirs[layer_idx] = direction
        directions[emo] = emo_dirs

    return directions


def save_directions(
    directions: Dict[str, Dict[int, torch.Tensor]],
    output_path: str,
    d_model: int,
    layer_start: int = 0,
    layer_end: int = 7,
    default_scale: float = 0.5,
):
    """Save direction vectors as JSON for the C EmoSteer loader."""
    result = {
        "dim": d_model,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "scale": default_scale,
        "emotions": {},
    }

    for emo, layer_dirs in directions.items():
        # Average direction across target layers for a single vector
        all_dirs = [d for d in layer_dirs.values()]
        if not all_dirs:
            continue
        avg_dir = torch.stack(all_dirs).mean(dim=0)
        norm = avg_dir.norm()
        if norm > 1e-8:
            avg_dir = avg_dir / norm
        result["emotions"][emo] = avg_dir.tolist()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    n_emotions = len(result["emotions"])
    print(f"\n  Saved {n_emotions} emotion directions ({d_model}-dim) to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute EmoSteer direction vectors for training-free emotion control"
    )
    parser.add_argument("--model-version", choices=["v1", "v3"], default="v1",
                        help="Flow model version (default: v1 for backward compat)")
    parser.add_argument("--model-size", choices=["base", "large"], default="base",
                        help="Flow v3 config size (ignored for v1)")
    parser.add_argument("--phonemes", action="store_true",
                        help="Use phoneme conditioning for v3 (requires g2p)")
    parser.add_argument("--data-dir", default="data/emotional_pairs/",
                        help="Directory of .pt files with emotion labels")
    parser.add_argument("--from-manifest",
                        help="JSONL manifest from prepare_prosody_data.py (alternative to --data-dir)")
    parser.add_argument("--flow-weights", default="models/sonata/flow.safetensors")
    parser.add_argument("--flow-config", default="models/sonata/sonata_flow_config.json")
    parser.add_argument("--output", default="models/sonata/emosteer_directions.json")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Max samples per emotion for averaging")
    parser.add_argument("--layer-start", type=int, default=2)
    parser.add_argument("--layer-end", type=int, default=6)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing the pipeline")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  EMOSTEER DIRECTION COMPUTATION (Flow {args.model_version})")
    print(f"{'='*60}")

    # Load config and model
    with open(args.flow_config) as f:
        cfg_dict = json.load(f)

    if args.model_version == "v3":
        from flow_v3 import SonataFlowV3
        cfg_cls = FlowV3LargeConfig if args.model_size == "large" else FlowV3Config
        valid = {k: v for k, v in cfg_dict.items() if k in cfg_cls.__dataclass_fields__}
        cfg = cfg_cls(**valid)
        model = SonataFlowV3(cfg).to(device).eval()
    else:
        from flow import SonataFlow
        valid = {k: v for k, v in cfg_dict.items() if hasattr(FlowConfig, k)}
        cfg = FlowConfig(**valid)
        model = SonataFlow(cfg).to(device).eval()

    if os.path.exists(args.flow_weights):
        if args.flow_weights.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(args.flow_weights, device=str(device))
        else:
            ckpt = torch.load(args.flow_weights, map_location=device, weights_only=False)
            state = ckpt.get("ema", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
        if not isinstance(state, dict):
            state = dict(state.named_parameters())
        keys = list(state.keys())
        if keys and keys[0].startswith("model."):
            state = {k.replace("model.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"  Loaded flow weights from {args.flow_weights}")
    else:
        print(f"  WARNING: No weights at {args.flow_weights}, using random model")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.1f}M params, {cfg.n_layers} layers, d={cfg.d_model}")

    # Load or generate data
    target_layers = list(range(args.layer_start, args.layer_end + 1))

    if args.synthetic:
        print(f"\n  Using synthetic data (for pipeline testing)")
        by_emotion = generate_synthetic_data(
            model_version=args.model_version,
            mel_dim=getattr(cfg, "mel_dim", 80),
            char_vocab_size=getattr(cfg, "char_vocab_size", 256),
        )
    elif args.from_manifest:
        print(f"\n  Loading data from manifest: {args.from_manifest}")
        by_emotion = load_emotional_data(
            args.data_dir, manifest=args.from_manifest,
            model_version=args.model_version, use_phonemes=args.phonemes,
        )
    else:
        print(f"\n  Loading data from {args.data_dir}")
        by_emotion = load_emotional_data(
            args.data_dir,
            model_version=args.model_version, use_phonemes=args.phonemes,
        )

    # Compute directions
    print(f"\n  Computing directions (layers {args.layer_start}-{args.layer_end})...")
    directions = compute_directions(
        model, by_emotion, device,
        max_samples=args.max_samples,
        target_layers=target_layers,
        model_version=args.model_version,
    )

    # Save
    save_directions(
        directions, args.output,
        d_model=cfg.d_model,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        default_scale=args.scale,
    )

    print(f"\n{'='*60}")
    print(f"  Done. Use with: pocket-voice --emosteer {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
