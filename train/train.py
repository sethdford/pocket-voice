"""
Mimi-Lite TTS training loop.

Trains the LM to predict NeuCodec audio tokens from text, with speaker conditioning.
Runs on Apple Silicon via PyTorch MPS backend.

Usage:
  python train/train.py --batch-size 4 --lr 3e-4 --max-steps 100000
  python train/train.py --resume train/checkpoints/step_10000.pt
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from model import MimiLiteConfig, create_model
from data import DataConfig, create_dataloader


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cosine_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(args):
    device = get_device()
    print(f"Device: {device}")

    # Create model
    model_cfg = MimiLiteConfig()
    model = create_model(model_cfg)
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )

    # Loss function (ignore padding target = -100)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Create data loader
    data_cfg = DataConfig(tokenizer_path=args.tokenizer)
    loader = create_dataloader(
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        cfg=data_cfg,
    )

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    # Training loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = Path(args.checkpoint_dir) / "train_log.jsonl"

    model.train()
    step = start_step
    total_loss = 0.0
    total_tokens = 0
    log_interval = args.log_every
    t0 = time.time()

    try:
        import wandb
        if args.wandb:
            wandb.init(project="mimi-lite-tts", config=vars(args))
    except ImportError:
        args.wandb = False

    print(f"\nTraining: max_steps={args.max_steps}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Grad accumulation: {args.grad_accum} → effective batch={args.batch_size * args.grad_accum}")

    data_iter = iter(loader)

    while step < args.max_steps:
        # Get batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        text_tokens = batch["text_tokens"].to(device)
        audio_input = batch["audio_input"].to(device)
        audio_target = batch["audio_target"].to(device)

        # Forward
        logits, _ = model(text_tokens, audio_input)

        # Loss: flatten for cross-entropy
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            audio_target.reshape(-1),
        )

        loss_val = loss.item()

        # Backward with gradient accumulation
        loss = loss / args.grad_accum
        loss.backward()

        n_tokens = (audio_target != -100).sum().item()
        total_loss += loss_val * n_tokens
        total_tokens += n_tokens

        if (step + 1) % args.grad_accum == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # LR schedule
            lr = cosine_lr(step, args.warmup_steps, args.max_steps,
                          args.lr, args.lr * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

        step += 1

        # Logging
        if step % log_interval == 0:
            avg_loss = total_loss / max(total_tokens, 1)
            elapsed = time.time() - t0
            tok_per_sec = total_tokens / elapsed
            ppl = math.exp(min(avg_loss, 20))

            log_entry = {
                "step": step,
                "loss": round(avg_loss, 4),
                "ppl": round(ppl, 1),
                "lr": round(lr, 7),
                "tok_s": round(tok_per_sec),
                "elapsed_s": round(elapsed, 1),
            }
            print(f"[{step:>7d}] loss={avg_loss:.4f} ppl={ppl:.1f} "
                  f"lr={lr:.2e} tok/s={tok_per_sec:.0f}")

            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if args.wandb:
                wandb.log(log_entry, step=step)

            total_loss = 0.0
            total_tokens = 0
            t0 = time.time()

        # Checkpoint
        if step % args.save_every == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"step_{step}.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vars(model_cfg),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    final_path = Path(args.checkpoint_dir) / "final.pt"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vars(model_cfg),
    }, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Mimi-Lite TTS LM")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset size for debugging")
    parser.add_argument("--tokenizer", default="models/tokenizer.model")
    parser.add_argument("--checkpoint-dir", default="train/checkpoints")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
