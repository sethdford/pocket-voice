"""Training loop for Full Mimi TTS (Main Transformer + DepFormer).

Trains on Mimi-encoded audio: (text_tokens, audio_codes[n_q][T]) pairs.
Uses PyTorch with MPS backend for Apple Silicon GPU training.

Loss = text_ce_loss + audio_ce_loss (sum over codebooks).

Usage:
  python train/train_full_mimi.py --data-dir train/data/mimi_encoded --steps 100000
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from full_mimi_model import FullMimiConfig, FullMimiTTS


class MimiCodecDataset(Dataset):
    """Dataset of (text, audio_codes) pairs from Mimi-encoded .pt files."""

    def __init__(self, data_dir: str, max_seq_len: int = 512,
                 text_vocab_size: int = 32000, text_pad: int = 0):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.pt"))
        self.max_seq_len = max_seq_len
        self.text_vocab_size = text_vocab_size
        self.text_pad = text_pad
        print(f"[dataset] Found {len(self.files)} samples in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        text = sample["text"]
        audio_codes = sample["audio_codes"]  # (n_q, T_frames)

        n_q, T = audio_codes.shape
        T = min(T, self.max_seq_len)
        audio_codes = audio_codes[:, :T]

        # Simple character-level text tokenization (will be replaced by SentencePiece)
        text_ids = [min(ord(c), self.text_vocab_size - 1) for c in text]

        # Pad/truncate text to match audio length
        if len(text_ids) < T:
            text_ids = text_ids + [self.text_pad] * (T - len(text_ids))
        else:
            text_ids = text_ids[:T]

        text_tensor = torch.tensor(text_ids, dtype=torch.long)
        return text_tensor, audio_codes.long()


def collate_fn(batch):
    """Collate with padding to max length in batch."""
    texts, audios = zip(*batch)
    max_len = max(t.shape[0] for t in texts)
    n_q = audios[0].shape[0]
    B = len(batch)

    text_padded = torch.zeros(B, max_len, dtype=torch.long)
    audio_padded = torch.zeros(B, n_q, max_len, dtype=torch.long)

    for i, (t, a) in enumerate(zip(texts, audios)):
        text_padded[i, :t.shape[0]] = t
        audio_padded[i, :, :a.shape[1]] = a

    return text_padded, audio_padded


def get_lr(step: int, warmup: int, max_lr: float, min_lr: float, total_steps: int) -> float:
    if step < warmup:
        return max_lr * step / warmup
    decay_ratio = (step - warmup) / max(1, total_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(args):
    device = torch.device(args.device)
    print(f"[train] Device: {device}")

    cfg = FullMimiConfig(
        n_codebooks=args.n_codebooks,
        n_layers=args.n_layers,
        d_model=args.d_model,
        dep_n_layers=args.dep_n_layers,
        dep_d_model=args.dep_d_model,
    )

    model = FullMimiTTS(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model: {n_params/1e6:.1f}M params")
    print(f"[train]   Main: {cfg.n_layers}L × d={cfg.d_model}")
    print(f"[train]   DepFormer: {cfg.dep_n_layers}L × d={cfg.dep_d_model}")
    print(f"[train]   Codebooks: {cfg.n_codebooks} × {cfg.audio_vocab_size}")

    dataset = MimiCodecDataset(args.data_dir, max_seq_len=args.max_seq_len,
                               text_vocab_size=cfg.text_vocab_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt.get("step", 0)
            print(f"[train] Resumed from step {start_step}")

    model.train()
    step = start_step
    running_loss = 0
    running_text_loss = 0
    running_audio_loss = 0
    t0 = time.time()

    print(f"[train] Starting training from step {step}")
    print(f"[train] Target: {args.steps} steps, batch_size={args.batch_size}")

    while step < args.steps:
        for text_batch, audio_batch in loader:
            if step >= args.steps:
                break

            text_batch = text_batch.to(device)
            audio_batch = audio_batch.to(device)

            # Shift: input is [0:T-1], target is [1:T]
            T = text_batch.shape[1]
            if T < 2:
                continue

            input_text = text_batch[:, :-1]
            input_audio = audio_batch[:, :, :-1]
            target_text = text_batch[:, 1:]
            target_audio = audio_batch[:, :, 1:]

            lr = get_lr(step, args.warmup, args.lr, args.lr * 0.1, args.steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            text_logits, audio_logits, losses = model(
                input_text, input_audio,
                target_text=target_text,
                target_audio=target_audio,
            )

            loss = losses.get("text", 0) + losses.get("audio", 0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            running_text_loss += losses.get("text", torch.tensor(0)).item()
            running_audio_loss += losses.get("audio", torch.tensor(0)).item()
            step += 1

            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                avg_text = running_text_loss / args.log_every
                avg_audio = running_audio_loss / args.log_every
                elapsed = time.time() - t0
                steps_per_sec = args.log_every / elapsed

                print(f"  step {step:6d} | loss {avg_loss:.4f} "
                      f"(text {avg_text:.4f}, audio {avg_audio:.4f}) | "
                      f"lr {lr:.2e} | {steps_per_sec:.1f} steps/s")

                running_loss = 0
                running_text_loss = 0
                running_audio_loss = 0
                t0 = time.time()

            if step % args.save_every == 0:
                ckpt_path = ckpt_dir / f"full_mimi_step_{step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": {
                        "d_model": cfg.d_model,
                        "n_layers": cfg.n_layers,
                        "n_heads": cfg.n_heads,
                        "n_kv_heads": cfg.n_kv_heads,
                        "ffn_mult": cfg.ffn_mult,
                        "max_seq_len": cfg.max_seq_len,
                        "text_vocab_size": cfg.text_vocab_size,
                        "audio_vocab_size": cfg.audio_vocab_size,
                        "n_codebooks": cfg.n_codebooks,
                        "n_special_tokens": cfg.n_special_tokens,
                        "dep_d_model": cfg.dep_d_model,
                        "dep_n_layers": cfg.dep_n_layers,
                        "dep_n_heads": cfg.dep_n_heads,
                        "dep_n_kv_heads": cfg.dep_n_kv_heads,
                        "rope_theta": cfg.rope_theta,
                        "norm_eps": cfg.norm_eps,
                    },
                    "loss": avg_loss if step % args.log_every == 0 else None,
                }, ckpt_path)
                print(f"  [ckpt] Saved {ckpt_path}")

    print(f"\n[train] Done: {step} steps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="train/data/mimi_encoded")
    parser.add_argument("--checkpoint-dir", default="train/checkpoints_full_mimi")
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--n-codebooks", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--dep-n-layers", type=int, default=6)
    parser.add_argument("--dep-d-model", type=int, default=512)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
