"""Train Sonata Flow with emotion, prosody, and duration conditioning.

Extends the base flow matching training with:
  - Emotion labels (from dataset or automatic labeling)
  - Prosody features (pitch, energy, rate extracted from audio)
  - Duration predictor (learns to predict frame-level durations)

Usage:
  python train_flow.py \
    --data-dir data/sonata_pairs/ \
    --device mps \
    --use-emotion --use-prosody --use-duration

Data format: directory of .pt files, each containing:
  { "semantic_tokens": LongTensor(T,),
    "acoustic_latents": FloatTensor(T, 256),
    "emotion_id": int (optional),
    "prosody_features": FloatTensor(T, 3) (optional: log_pitch, energy, rate) }
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F_loss
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import FlowConfig
from flow import SonataFlow
from ema import EMA
from modules import cosine_lr
from quantize import quantize_model, compute_compression_stats


class LatentDiscriminator(nn.Module):
    """Lightweight discriminator on acoustic latent sequences.

    Judges whether predicted acoustic latents look real or generated.
    Sharper flow outputs — captures fine-grained temporal detail
    that MSE loss misses.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, n_layers: int = 4):
        super().__init__()
        layers = [nn.Conv1d(input_dim, hidden_dim, 7, padding=3), nn.LeakyReLU(0.1)]
        for _ in range(n_layers - 2):
            layers += [nn.Conv1d(hidden_dim, hidden_dim, 7, stride=2, padding=3), nn.LeakyReLU(0.1)]
        layers += [nn.Conv1d(hidden_dim, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).squeeze(1)


EMOTION_LABELS = [
    "neutral", "happy", "excited", "sad", "angry", "fearful",
    "surprised", "warm", "serious", "calm", "confident",
]


class FlowDataset(Dataset):
    """Loads (semantic, acoustic, speaker, emotion, prosody) tuples.

    Supports two formats:
    - Directory of individual .pt files (one per utterance)
    - Sharded .pt files (list of dicts per shard) via manifest
    """

    def __init__(self, data_dir: str, max_frames: int = 500,
                 use_emotion: bool = False, use_prosody: bool = False,
                 manifest: str = "", n_speakers: int = 0):
        self.max_frames = max_frames
        self.use_emotion = use_emotion
        self.use_prosody = use_prosody
        self.n_speakers = n_speakers
        self._use_shards = False
        self._speaker_to_id = {}

        if manifest and Path(manifest).exists():
            self._load_from_shards(manifest)
        else:
            self.files = sorted(Path(data_dir).glob("*.pt"))
            print(f"[flow-data] {len(self.files)} .pt files from {data_dir}")

    def _load_from_shards(self, manifest: str):
        """Load from sharded .pt files or encoded data."""
        self._use_shards = True
        self._data = []
        paths = []
        if manifest.endswith(".shards.txt"):
            with open(manifest) as f:
                paths = [l.strip() for l in f if l.strip()]
        elif "," in manifest:
            paths = [p.strip() for p in manifest.split(",") if p.strip()]
        else:
            paths = [manifest]
        for p in paths:
            chunk = torch.load(p, weights_only=False)
            self._data.extend(chunk)
            print(f"[flow-data] Loaded {len(chunk)} from {p}")

        if self.n_speakers > 0:
            speakers = sorted(set(d.get("speaker", "") for d in self._data if d.get("speaker")))
            self._speaker_to_id = {s: i + 1 for i, s in enumerate(speakers)}
            print(f"[flow-data] {len(speakers)} unique speakers mapped to IDs")

        print(f"[flow-data] Total: {len(self._data)} utterances from shards")

    def __len__(self) -> int:
        if self._use_shards:
            return len(self._data)
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple:
        if self._use_shards:
            data = self._data[idx]
        else:
            data = torch.load(self.files[idx], weights_only=True)

        T = min(data["semantic_tokens"].shape[0], self.max_frames)
        sem = data["semantic_tokens"][:T]

        aco_key = "acoustic_latents" if "acoustic_latents" in data else "acoustic_latent"
        aco = data[aco_key][:T]

        speaker_id = 0
        if self.n_speakers > 0 and "speaker" in data:
            speaker_id = self._speaker_to_id.get(data["speaker"], 0)

        emotion_id = data.get("emotion_id", 0) if self.use_emotion else 0

        if self.use_prosody and "prosody_features" in data:
            prosody = data["prosody_features"][:T]
        else:
            prosody = torch.zeros(T, 3)

        return sem, aco, speaker_id, emotion_id, prosody


def collate_fn(batch: list) -> tuple:
    sem_list, aco_list, spk_list, emo_list, pros_list = zip(*batch)
    max_t = max(s.shape[0] for s in sem_list)
    B = len(sem_list)
    acoustic_dim = aco_list[0].shape[-1]

    sem = torch.zeros(B, max_t, dtype=torch.long)
    aco = torch.zeros(B, max_t, acoustic_dim)
    mask = torch.zeros(B, max_t)
    spk = torch.tensor(spk_list, dtype=torch.long)
    emo = torch.tensor(emo_list, dtype=torch.long)
    pros = torch.zeros(B, max_t, 3)

    for i, (s, a, _, _, p) in enumerate(zip(sem_list, aco_list, spk_list, emo_list, pros_list)):
        T = s.shape[0]
        sem[i, :T] = s
        aco[i, :T] = a
        mask[i, :T] = 1.0
        pros[i, :T] = p

    return sem, aco, spk, emo, pros, mask


@torch.no_grad()
def validate_flow(model, val_loader, device, use_emotion, use_prosody,
                  use_speakers, max_batches=50):
    model.eval()
    total_loss = 0.0
    n = 0
    for sem, aco, spk, emo, pros, mask in val_loader:
        if n >= max_batches:
            break
        sem = sem.to(device)
        aco = aco.to(device)
        mask = mask.to(device)
        spk = spk.to(device) if use_speakers else None
        emo = emo.to(device) if use_emotion else None
        pros = pros.to(device) if use_prosody else None
        loss = model.compute_loss(aco, sem, mask=mask, speaker_ids=spk,
                                  emotion_ids=emo, prosody_features=pros)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def _detect_flow_data_props(manifest: str) -> dict:
    """Peek at encoded data to detect semantic vocab size, acoustic dim, n_speakers."""
    paths = []
    if manifest.endswith(".shards.txt"):
        with open(manifest) as f:
            paths = [l.strip() for l in f if l.strip()]
    elif "," in manifest:
        paths = [p.strip() for p in manifest.split(",") if p.strip()]
    elif Path(manifest).exists():
        paths = [manifest]
    if not paths:
        return {}
    chunk = torch.load(paths[0], weights_only=False)
    if not chunk:
        return {}
    max_tok = 0
    acoustic_dim = 0
    speakers = set()
    for entry in chunk[:500]:
        if "semantic_tokens" in entry:
            max_tok = max(max_tok, entry["semantic_tokens"].max().item())
        aco_key = "acoustic_latents" if "acoustic_latents" in entry else "acoustic_latent"
        if aco_key in entry:
            acoustic_dim = entry[aco_key].shape[-1]
        if "speaker" in entry:
            speakers.add(entry["speaker"])
    del chunk
    vocab = 32768
    for size in [4096, 8192, 16384, 32768, 65536]:
        if max_tok < size:
            vocab = size
            break
    return {"semantic_vocab_size": vocab, "acoustic_dim": acoustic_dim,
            "n_speakers": len(speakers)}


def train(args):
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  SONATA FLOW — TRAINING")
    print(f"{'='*60}")
    print(f"  Gradient accumulation: {args.grad_accum}")

    if args.resume:
        _ckpt_peek = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg_dict = _ckpt_peek.get("config", {})
        cfg = FlowConfig(**FlowConfig._normalize_loaded_dict(cfg_dict))
        print(f"  Config loaded from checkpoint: {args.resume}")
        del _ckpt_peek
    else:
        data_props = {}
        if args.manifest and Path(args.manifest).exists():
            data_props = _detect_flow_data_props(args.manifest)
            if data_props:
                print(f"  Auto-detected from data: vocab={data_props.get('semantic_vocab_size')}, "
                      f"acoustic_dim={data_props.get('acoustic_dim')}, "
                      f"speakers={data_props.get('n_speakers')}")

        n_speakers = args.n_speakers
        if n_speakers == 0 and data_props.get("n_speakers", 0) > 1:
            n_speakers = data_props["n_speakers"] + 10  # headroom

        if args.flow_large:
            from config import FlowLargeConfig
            cfg = FlowLargeConfig(
                n_emotions=len(EMOTION_LABELS) if args.use_emotion else 0,
                prosody_dim=3 if args.use_prosody else 0,
                use_energy_predictor=args.use_duration,
                semantic_vocab_size=data_props.get("semantic_vocab_size", 32768),
                acoustic_dim=data_props.get("acoustic_dim", 256),
                n_speakers=n_speakers,
            )
            print(f"  Using FlowLargeConfig (768d, 16L, 12H, RoPE)")
        else:
            cfg = FlowConfig(
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                n_emotions=len(EMOTION_LABELS) if args.use_emotion else 0,
                prosody_dim=3 if args.use_prosody else 0,
                use_energy_predictor=args.use_duration,
                use_rope=args.use_rope,
                use_ref_audio=args.use_ref_audio,
                ref_audio_dim=args.ref_audio_dim,
                semantic_vocab_size=data_props.get("semantic_vocab_size", 32768),
                acoustic_dim=data_props.get("acoustic_dim", 256),
                n_speakers=n_speakers,
            )

    model = SonataFlow(cfg, cfg_dropout_prob=args.cfg_dropout).to(device)

    if args.qat:
        exclude = [p.strip() for p in args.qat_exclude.split(",") if p.strip()]
        quantize_model(model, exclude_patterns=exclude)
        stats = compute_compression_stats(model)
        print(f"  QAT: quantized Linear layers (exclude: {exclude or 'none'})")
        print(f"  Compression: {stats['original_mb']:.1f}MB → {stats['quantized_mb']:.1f}MB ({stats['compression_ratio']:.1f}x)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.1f}M params")
    print(f"  Semantic vocab: {cfg.semantic_vocab_size}, Acoustic dim: {cfg.acoustic_dim}")
    print(f"  Speakers: {cfg.n_speakers}, Emotion: {cfg.n_emotions > 0}, "
          f"Prosody: {cfg.prosody_dim > 0}, Energy: {cfg.use_energy_predictor}")

    dataset = FlowDataset(
        args.data_dir, max_frames=args.max_frames,
        use_emotion=args.use_emotion, use_prosody=args.use_prosody,
        manifest=args.manifest, n_speakers=cfg.n_speakers,
    )
    if len(dataset) == 0:
        print("  WARNING: No data found. Using synthetic data.")
        dataset = SyntheticFlowDataset(1000, args.max_frames, cfg.acoustic_dim, cfg.n_speakers)

    # Train/val split
    val_size = min(max(int(len(dataset) * 0.02), 50), 1000)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )

    # Adversarial training (optional)
    flow_disc = None
    opt_disc = None
    if args.adversarial:
        flow_disc = LatentDiscriminator(cfg.acoustic_dim).to(device)
        opt_disc = torch.optim.AdamW(flow_disc.parameters(), lr=args.lr, betas=(0.8, 0.99))
        disc_n = sum(p.numel() for p in flow_disc.parameters())
        print(f"  Adversarial: LatentDiscriminator ({disc_n/1e6:.1f}M params)")
        print(f"  Adversarial starts at step: {args.adv_start_step}")

    ema = EMA(model, decay=args.ema_decay)
    print(f"  EMA: decay={args.ema_decay}")

    os.makedirs(args.output_dir, exist_ok=True)

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError):
            print(f"  WARNING: Could not load optimizer state")
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
            print(f"  EMA weights restored from checkpoint")
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed from step {start_step}")

    step = start_step
    running_loss = 0.0
    t0 = time.time()

    eff_batch = args.batch_size * args.grad_accum
    print(f"\n  Training: step {start_step} → {args.steps}")
    print(f"  Batch: {args.batch_size} × {args.grad_accum} accum = {eff_batch} effective")
    print(f"  Train/val: {train_size}/{val_size}")

    use_speakers = cfg.n_speakers > 0
    model.train()
    optimizer.zero_grad()
    while step < args.steps:
        for sem, aco, spk, emo, pros, mask in loader:
            if step >= args.steps:
                break

            sem = sem.to(device)
            aco = aco.to(device)
            mask = mask.to(device)
            spk = spk.to(device) if use_speakers else None
            emo = emo.to(device) if args.use_emotion else None
            pros = pros.to(device) if args.use_prosody else None

            lr = cosine_lr(step, args.warmup, args.lr, args.lr * 0.01, args.steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            loss = model.compute_loss(
                aco, sem,
                mask=mask,
                speaker_ids=spk,
                emotion_ids=emo,
                prosody_features=pros,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARN] step {step}: NaN/Inf loss, skipping")
                optimizer.zero_grad()
                continue

            # Adversarial training on flow outputs
            if flow_disc is not None and step >= args.adv_start_step:
                with torch.no_grad():
                    noise = torch.randn_like(aco)
                    t = torch.rand(aco.shape[0], device=device).clamp(0.8, 1.0)
                    t_expand = t[:, None, None]
                    x_t = (1 - t_expand) * noise + t_expand * aco

                v_pred = model(x_t, t, sem, spk, emo, pros)
                fake_latent = x_t + (1 - t_expand) * v_pred

                # Generator adversarial loss
                fake_score = flow_disc(fake_latent)
                adv_loss = F_loss.relu(1 - fake_score).mean()
                loss = loss + args.adv_weight * adv_loss

                # Discriminator step
                with torch.no_grad():
                    fake_det = fake_latent.detach()
                real_score = flow_disc(aco)
                fake_score_d = flow_disc(fake_det)
                d_loss = F_loss.relu(1 - real_score).mean() + F_loss.relu(1 + fake_score_d).mean()
                opt_disc.zero_grad()
                d_loss.backward()
                opt_disc.step()

            loss = loss / args.grad_accum
            loss.backward()

            running_loss += loss.item() * args.grad_accum
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                ema.update()

            if step % args.log_every == 0:
                n = args.log_every
                elapsed = time.time() - t0
                print(f"  step {step:6d} | loss={running_loss/n:.4f} | "
                      f"lr={lr:.2e} | {n/elapsed:.1f} steps/s")
                running_loss = 0.0
                t0 = time.time()

            if step % args.save_every == 0:
                path = os.path.join(args.output_dir, f"flow_step_{step}.pt")
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "ema": ema.state_dict(),
                            "step": step, "config": vars(cfg),
                            "best_val_loss": best_val_loss}, path)
                print(f"  [ckpt] {path}")

            # Validation (use EMA weights)
            if step % args.val_every == 0 and len(val_ds) > 0:
                ema.apply_shadow()
                val_loss = validate_flow(model, val_loader, device,
                                         args.use_emotion, args.use_prosody,
                                         use_speakers)
                ema.restore()
                is_best = val_loss < best_val_loss
                print(f"  [val] step {step}: loss={val_loss:.4f}"
                      f" {'(best!)' if is_best else ''}")
                if is_best:
                    best_val_loss = val_loss
                    path = os.path.join(args.output_dir, "flow_best.pt")
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "ema": ema.state_dict(),
                                "step": step, "config": vars(cfg),
                                "best_val_loss": best_val_loss}, path)
                    print(f"  [best] {path}")

    path = os.path.join(args.output_dir, "flow_final.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict(),
                "step": step, "config": vars(cfg),
                "best_val_loss": best_val_loss}, path)
    print(f"\n  Final: {path}")
    print(f"  Best val loss: {best_val_loss:.4f}")

    cfg_path = os.path.join(args.output_dir, "flow_config.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(cfg), f, indent=2)
    print(f"  Config: {cfg_path}")


class SyntheticFlowDataset(Dataset):
    def __init__(self, n: int, max_frames: int, acoustic_dim: int, n_speakers: int = 0):
        self.n = n
        self.max_frames = max_frames
        self.acoustic_dim = acoustic_dim
        self.n_speakers = n_speakers

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        T = torch.randint(32, self.max_frames, (1,)).item()
        sem = torch.randint(0, 4096, (T,))
        aco = torch.randn(T, self.acoustic_dim)
        spk = torch.randint(0, max(1, self.n_speakers), (1,)).item()
        emo = torch.randint(0, 11, (1,)).item()
        pros = torch.randn(T, 3) * 0.1
        return sem, aco, spk, emo, pros


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/sonata_pairs/")
    parser.add_argument("--manifest", default="",
                        help="Sharded data: .shards.txt or .pt or comma-separated")
    parser.add_argument("--output-dir", default="train/checkpoints/flow")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--use-rope", action="store_true",
                        help="Use RoPE positional encoding in flow attention")
    parser.add_argument("--flow-large", action="store_true",
                        help="Use large flow config (768d, 16L, 12H, RoPE, ~150M params)")
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
    parser.add_argument("--n-speakers", type=int, default=0,
                        help="Number of speakers (0=auto-detect from data, >0=explicit)")
    parser.add_argument("--use-emotion", action="store_true")
    parser.add_argument("--use-prosody", action="store_true")
    parser.add_argument("--use-duration", action="store_true")
    parser.add_argument("--resume", default="", help="Resume from checkpoint")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate for model weights")
    parser.add_argument("--use-ref-audio", action="store_true",
                        help="Enable reference audio cross-attention for voice cloning")
    parser.add_argument("--ref-audio-dim", type=int, default=80,
                        help="Dimension of reference audio features (mel bins)")
    parser.add_argument("--adversarial", action="store_true",
                        help="Add latent-space discriminator on flow outputs")
    parser.add_argument("--adv-weight", type=float, default=0.1,
                        help="Adversarial loss weight")
    parser.add_argument("--adv-start-step", type=int, default=10000,
                        help="Step to start adversarial training")
    parser.add_argument("--qat", action="store_true",
                        help="Quantization-aware training (BitTTS-style ternary)")
    parser.add_argument("--qat-exclude", default="semantic_emb,output_proj",
                        help="Comma-separated name patterns to exclude from QAT")
    args = parser.parse_args()
    train(args)
