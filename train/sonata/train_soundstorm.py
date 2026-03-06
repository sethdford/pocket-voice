"""Train SonataStorm — MaskGIT-style parallel semantic token predictor.

10-50x faster than autoregressive LM at inference. Uses the same encoded data
format as train_lm.py (semantic tokens + text tokens from encoded datasets).

Usage:
  python train_soundstorm.py --data train/data/encoded_all.shards.txt --steps 100000
  python train_soundstorm.py --synthetic --steps 5000  # Quick test
"""

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import SemanticLMConfig
from soundstorm import SonataStorm
from ema import EMA
from modules import cosine_lr


def _resolve_data_paths(data_arg: str):
    p = Path(data_arg)
    if p.suffix == ".txt" and p.exists():
        paths = []
        for line in p.read_text().strip().split("\n"):
            line = line.strip()
            if line and Path(line).exists():
                paths.append(Path(line))
        return paths
    elif "," in data_arg:
        return [Path(x.strip()) for x in data_arg.split(",") if Path(x.strip()).exists()]
    else:
        return [p] if p.exists() else []


class StormDataset(Dataset):
    def __init__(self, data_path: str, max_seq_len: int = 1024,
                 text_vocab_size: int = 32000, semantic_vocab_size: int = 32768,
                 synthetic: bool = False):
        self.max_seq_len = max_seq_len
        self.text_vocab_size = text_vocab_size
        self.semantic_vocab_size = semantic_vocab_size
        self.synthetic = synthetic
        self.data = []

        if synthetic:
            self.data = [None] * 2000
        else:
            paths = _resolve_data_paths(data_path)
            for p in paths:
                shard = torch.load(str(p), map_location="cpu", weights_only=False)
                if isinstance(shard, list):
                    self.data.extend(shard)
                elif isinstance(shard, dict) and "entries" in shard:
                    self.data.extend(shard["entries"])
            print(f"[storm-data] {len(self.data)} entries from {len(paths)} files")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.synthetic:
            text_len = random.randint(5, 50)
            sem_len = random.randint(20, self.max_seq_len)
            text = torch.randint(4, self.text_vocab_size, (text_len,))
            semantic = torch.randint(4, self.semantic_vocab_size, (sem_len,))
            return text, semantic

        entry = self.data[idx]
        text = entry.get("text_tokens", entry.get("text_ids", torch.zeros(1, dtype=torch.long)))
        semantic = entry.get("semantic_tokens", torch.zeros(1, dtype=torch.long))

        if isinstance(text, list):
            text = torch.tensor(text, dtype=torch.long)
        if isinstance(semantic, list):
            semantic = torch.tensor(semantic, dtype=torch.long)

        semantic = semantic[:self.max_seq_len]
        return text, semantic


def collate_fn(batch):
    text_list, sem_list = zip(*batch)
    max_text = max(t.shape[0] for t in text_list)
    max_sem = max(s.shape[0] for s in sem_list)
    B = len(batch)

    text_padded = torch.zeros(B, max_text, dtype=torch.long)
    sem_padded = torch.zeros(B, max_sem, dtype=torch.long)
    for i, (t, s) in enumerate(zip(text_list, sem_list)):
        text_padded[i, :t.shape[0]] = t
        sem_padded[i, :s.shape[0]] = s

    return text_padded, sem_padded


@torch.no_grad()
def validate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for text, semantic in val_loader:
        if n >= max_batches:
            break
        text = text.to(device)
        semantic = semantic.to(device)
        loss, info = model.compute_loss(text, semantic)
        total_loss += loss.item()
        total_acc += info["acc"]
        n += 1
    model.train()
    return total_loss / max(n, 1), total_acc / max(n, 1)


def train(args):
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  SONATA STORM — PARALLEL MASKED PREDICTION")
    print(f"{'='*60}")

    cfg = SemanticLMConfig(
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, n_kv_heads=max(1, args.n_heads // 4),
        semantic_vocab_size=args.semantic_vocab_size,
    )

    model = SonataStorm(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.1f}M params")
    print(f"  {cfg.n_layers}L × d={cfg.d_model}, H={cfg.n_heads}")

    dataset = StormDataset(
        args.data, max_seq_len=args.max_seq_len,
        semantic_vocab_size=cfg.semantic_vocab_size,
        synthetic=args.synthetic,
    )

    val_size = min(max(int(len(dataset) * 0.02), 50), 1000)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    ema = EMA(model, decay=0.999)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_opt_step = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_opt_step = ckpt.get("step", 0)

    opt_step = start_opt_step
    micro_step = 0
    running_loss = 0.0
    running_acc = 0.0
    t0 = time.time()

    print(f"  Training: {start_opt_step} → {args.steps} optimizer steps")
    print(f"  Batch: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print(f"  Train/val: {train_size}/{val_size}\n")

    model.train()
    optimizer.zero_grad()
    while opt_step < args.steps:
        for text, semantic in loader:
            if opt_step >= args.steps:
                break

            text = text.to(device)
            semantic = semantic.to(device)

            lr = cosine_lr(opt_step, args.warmup, args.lr, args.lr * 0.01, args.steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            loss, info = model.compute_loss(text, semantic)
            loss = loss / args.grad_accum
            loss.backward()

            running_loss += loss.item() * args.grad_accum
            running_acc += info["acc"]
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                ema.update()
                opt_step += 1

                if opt_step % args.log_every == 0:
                    n = args.log_every
                    elapsed = time.time() - t0
                    print(f"  step {opt_step:6d} | loss={running_loss/n:.4f} acc={running_acc/n:.2%}"
                          f" | lr={lr:.2e} | {n/elapsed:.1f} steps/s")
                    running_loss = 0.0
                    running_acc = 0.0
                    t0 = time.time()

                if opt_step % args.save_every == 0:
                    path = ckpt_dir / f"storm_step_{opt_step}.pt"
                    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                                "step": opt_step, "config": vars(cfg)}, path)
                    print(f"  [ckpt] {path}")

                if opt_step % args.val_every == 0:
                    ema.apply_shadow()
                    val_loss, val_acc = validate(model, val_loader, device)
                    ema.restore()
                    is_best = val_loss < best_val_loss
                    print(f"  [val] step {opt_step}: loss={val_loss:.4f} acc={val_acc:.2%}"
                          f" {'(best!)' if is_best else ''}")
                    if is_best:
                        best_val_loss = val_loss
                        path = ckpt_dir / "storm_best.pt"
                        torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                                    "step": opt_step, "config": vars(cfg)}, path)

    path = ckpt_dir / "storm_final.pt"
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "step": opt_step, "config": vars(cfg)}, path)
    print(f"\n  Final: {path}, best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train/data/encoded_dev-clean.pt")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--checkpoint-dir", default="train/checkpoints/storm")
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--semantic-vocab-size", type=int, default=32768)
    args = parser.parse_args()
    train(args)
