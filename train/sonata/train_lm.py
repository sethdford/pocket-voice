"""Train Sonata Semantic LM on codec-encoded speech data.

Learns to predict semantic tokens (4096 vocab @ 50Hz) conditioned on text.
Uses teacher-forced cross-entropy with text + semantic token interleaving.

Training data format (from encode_dataset.py / data_pipeline.py):
  List of dicts with {text, semantic_tokens, acoustic_latent, ...}

Supports sharded data loading for 10K+ hour datasets:
  --data data/encoded.shards.txt  (file containing shard paths)
  --data data/shard0000.pt,data/shard0001.pt  (comma-separated)
  --data data/encoded.pt  (single file)

Usage:
  python train/sonata/train_lm.py \
    --data train/data/encoded_dev-clean.pt \
    --steps 50000

  python train/sonata/train_lm.py \
    --data train/data/encoded.shards.txt \
    --steps 200000 --grad-accum 8
"""

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset

from config import SemanticLMConfig
from semantic_lm import SonataSemanticLM
from ema import EMA
from modules import TrainingLog
from quantize import quantize_model, compute_compression_stats


def _resolve_data_paths(data_path: str) -> list:
    """Resolve data paths from shards.txt, comma-separated list, or single file."""
    data_path = data_path.strip()
    if data_path.endswith(".shards.txt"):
        with open(data_path) as f:
            return [line.strip() for line in f if line.strip()]
    if "," in data_path:
        return [p.strip() for p in data_path.split(",") if p.strip()]
    return [data_path]


class SemanticDataset(Dataset):
    """Dataset of (text_token_ids, semantic_token_ids) pairs from encoded audio.

    Supports:
    - Sharded loading via .shards.txt index file
    - Lazy shard loading to avoid OOM on large datasets
    - Variable-length text and audio (for cross-attention architecture)
    """

    def __init__(self, data_path: str, max_seq_len: int = 512,
                 text_vocab_size: int = 32000, semantic_vocab_size: int = 32768,
                 synthetic: bool = False, n_synthetic: int = 5000,
                 use_g2p: bool = False, use_prosody: bool = False,
                 lazy_load: bool = False):
        self.max_seq_len = max_seq_len
        self.text_vocab_size = text_vocab_size
        self.semantic_vocab_size = semantic_vocab_size
        self.synthetic = synthetic
        self.use_g2p = use_g2p
        self.use_prosody = use_prosody
        self.lazy_load = lazy_load

        if synthetic:
            self.length = n_synthetic
            self.data = None
            self._shard_index = None
        else:
            paths = _resolve_data_paths(data_path)
            self.data = []
            self._shard_index = None

            if lazy_load and len(paths) > 1:
                self._shard_paths = paths
                self._shard_sizes = []
                self._shard_offsets = []
                total = 0
                for dp in paths:
                    chunk = torch.load(dp, weights_only=False)
                    self._shard_offsets.append(total)
                    self._shard_sizes.append(len(chunk))
                    total += len(chunk)
                    del chunk
                self.length = total
                self._loaded_shard = -1
                self._loaded_data = None
                self._length_cache = None
                print(f"[dataset] Lazy loading: {len(paths)} shards, {total} utterances")
            else:
                for dp in paths:
                    chunk = torch.load(dp, weights_only=False)
                    self.data.extend(chunk)
                    print(f"[dataset] Loaded {len(chunk)} utterances from {dp}")
                self.length = len(self.data)

            self._prepare_tokenizer()
            first_few = self.data[:5] if self.data else self._peek_first(5)
            has_prosody = any("prosody_features" in d for d in first_few)
            if use_prosody and not has_prosody:
                print("[dataset] WARNING: --use-prosody set but data lacks prosody_features. "
                      "Re-encode with updated encode_dataset.py.")
                self.use_prosody = False
            print(f"[dataset] Total: {self.length} utterances"
                  f"{' (with prosody)' if self.use_prosody else ''}")

    def _build_length_cache(self):
        """Build per-item length cache from shard files (for lazy_load bucketing)."""
        if self._length_cache is not None:
            return
        if not getattr(self, "_shard_paths", None):
            return
        self._length_cache = [0] * self.length
        idx = 0
        for path, size in zip(self._shard_paths, self._shard_sizes):
            chunk = torch.load(path, weights_only=False)
            for entry in chunk:
                self._length_cache[idx] = entry["semantic_tokens"].shape[0]
                idx += 1
            del chunk
        assert idx == self.length
        print(f"[dataset] Built length cache for {self.length} entries")

    def _peek_first(self, n):
        """Peek at first n items without loading all shards."""
        if hasattr(self, '_shard_paths') and self._shard_paths:
            chunk = torch.load(self._shard_paths[0], weights_only=False)
            return chunk[:n]
        return []

    def _get_item_lazy(self, idx):
        for i, (offset, size) in enumerate(zip(self._shard_offsets, self._shard_sizes)):
            if idx < offset + size:
                if self._loaded_shard != i:
                    self._loaded_data = torch.load(self._shard_paths[i], weights_only=False)
                    self._loaded_shard = i
                return self._loaded_data[idx - offset]
        raise IndexError(f"Index {idx} out of range")

    def _prepare_tokenizer(self):
        if self.use_g2p:
            from g2p import PhonemeFrontend
            self.g2p = PhonemeFrontend()
            self.text_vocab_size = self.g2p.vocab_size
            print(f"[dataset] Using G2P phoneme frontend: {self.g2p.vocab_size} tokens")
            return

        try:
            import sentencepiece as spm
            model_path = "models/parakeet-ctc-1.1b-fp16.vocab"
            if Path(model_path).exists():
                self.sp = spm.SentencePieceProcessor()
                self.sp.Load(model_path)
                print(f"[dataset] Loaded SentencePiece tokenizer: {self.sp.GetPieceSize()} tokens")
                return
        except (ImportError, Exception):
            pass
        self.g2p = None
        self.sp = None
        print("[dataset] No tokenizer — using character-level encoding")

    def _tokenize_text(self, text: str):
        if self.use_g2p and hasattr(self, 'g2p') and self.g2p is not None:
            return self.g2p.encode(text, add_bos=False, add_eos=False)
        if hasattr(self, 'sp') and self.sp is not None:
            return torch.tensor(self.sp.EncodeAsIds(text), dtype=torch.long)
        return torch.tensor(
            [ord(c) % self.text_vocab_size for c in text],
            dtype=torch.long,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        EOS_ID = 2

        if self.synthetic:
            T_text = torch.randint(8, 64, (1,)).item()
            T_audio = torch.randint(32, self.max_seq_len - 1, (1,)).item()
            text_ids = torch.randint(3, self.text_vocab_size, (T_text,))
            sem_ids = torch.randint(3, self.semantic_vocab_size, (T_audio,))
            sem_ids = torch.cat([sem_ids, torch.tensor([EOS_ID])])
            prosody = torch.zeros(sem_ids.shape[0], 3)
            return text_ids, sem_ids, prosody

        if self.lazy_load and self._loaded_shard >= 0:
            entry = self._get_item_lazy(idx)
        elif self.lazy_load:
            entry = self._get_item_lazy(idx)
        else:
            entry = self.data[idx]

        text_ids = self._tokenize_text(entry["text"])
        sem_ids = entry["semantic_tokens"]

        T_audio = min(sem_ids.shape[0], self.max_seq_len - 1)
        sem_ids = sem_ids[:T_audio]
        sem_ids = torch.cat([sem_ids, torch.tensor([EOS_ID])])
        text_ids = text_ids[:self.max_seq_len]

        prosody = torch.zeros(sem_ids.shape[0], 3)
        if self.use_prosody and "prosody_features" in entry:
            pf = entry["prosody_features"]
            pT = min(pf.shape[0], T_audio)
            prosody[:pT] = pf[:pT]

        return text_ids, sem_ids, prosody


def collate_fn(batch):
    """Pad text and audio to their respective max lengths (they can differ).
    Semantic sequences padded with -1 (FSQ uses valid indices [0, codebook_size-1])."""
    text_ids, sem_ids, prosody = zip(*batch)
    max_text = max(t.shape[0] for t in text_ids)
    max_audio = max(s.shape[0] for s in sem_ids)

    text_padded = torch.zeros(len(batch), max_text, dtype=torch.long)
    sem_padded = torch.full((len(batch), max_audio), -1, dtype=torch.long)
    pros_padded = torch.zeros(len(batch), max_audio, 3)

    for i, (t, s, p) in enumerate(zip(text_ids, sem_ids, prosody)):
        text_padded[i, :t.shape[0]] = t
        sem_padded[i, :s.shape[0]] = s
        pros_padded[i, :p.shape[0]] = p

    return text_padded, sem_padded, pros_padded


class BucketSampler(Sampler):
    """Length-based bucketing: sorts by semantic token length within buckets for less padding waste."""

    def __init__(self, subset, batch_size, parent_dataset=None, bucket_size=1000):
        self.subset = subset
        self.parent = parent_dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size

    def _get_length(self, local_idx):
        ds = self.parent or self.subset
        if hasattr(self.subset, "indices"):
            real_idx = self.subset.indices[local_idx]
        else:
            real_idx = local_idx

        if isinstance(ds, ConcatDataset):
            cum = 0
            for d in ds.datasets:
                if real_idx < cum + len(d):
                    inner_idx = real_idx - cum
                    ds = d
                    break
                cum += len(d)
            else:
                return 0
            real_idx = inner_idx

        if hasattr(ds, "data") and ds.data and real_idx < len(ds.data):
            return ds.data[real_idx]["semantic_tokens"].shape[0]

        if getattr(ds, "lazy_load", False) and getattr(ds, "_shard_paths", None):
            ds._build_length_cache()
        if hasattr(ds, "_length_cache") and ds._length_cache is not None:
            return ds._length_cache.get(real_idx, 0) if isinstance(
                ds._length_cache, dict
            ) else (ds._length_cache[real_idx] if real_idx < len(ds._length_cache) else 0)

        return 0

    def __iter__(self):
        indices = list(range(len(self.subset)))
        random.shuffle(indices)
        buckets = [indices[i:i + self.bucket_size]
                   for i in range(0, len(indices), self.bucket_size)]
        for bucket in buckets:
            bucket.sort(key=self._get_length)
            yield from bucket

    def __len__(self):
        return len(self.subset)


def get_lr(step, warmup, max_lr, min_lr, total_steps):
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    ratio = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))


@torch.no_grad()
def validate_lm(model, val_loader, device, use_prosody=False, label_smoothing=0.0):
    """Run validation and return average loss + accuracy."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for text_ids, sem_ids, prosody in val_loader:
        text_ids = text_ids.to(device)
        sem_ids = sem_ids.to(device)
        prosody = prosody.to(device) if use_prosody else None
        sem_input = F.pad(sem_ids[:, :-1], (1, 0), value=1)
        sem_input = sem_input.clamp(min=0)  # -1 padding -> PAD(0) for embedding
        target = sem_ids
        logits, losses = model(text_ids, sem_input, target_semantic=target,
                               prosody_features=prosody)
        total_loss += losses["semantic"].item()
        preds = logits.argmax(dim=-1)
        mask = target != -1
        total_acc += (preds[mask] == target[mask]).float().mean().item()
        n += 1
        if n >= 50:
            break
    model.train()
    return total_loss / max(n, 1), total_acc / max(n, 1)


def _detect_vocab_from_data(data_path: str) -> int:
    """Peek at encoded data to detect semantic vocab size."""
    paths = _resolve_data_paths(data_path)
    chunk = torch.load(paths[0], weights_only=False)
    if not chunk:
        return 0
    max_tok = 0
    for entry in chunk[:200]:
        if "semantic_tokens" in entry:
            max_tok = max(max_tok, entry["semantic_tokens"].max().item())
    del chunk
    # Round up to nearest power of 2 or known FSQ size
    for size in [4096, 8192, 16384, 32768, 65536]:
        if max_tok < size:
            return size
    return 32768


def train(args):
    device = torch.device(args.device)
    print(f"\n{'='*70}")
    print(f"  SONATA SEMANTIC LM — TRAINING")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Gradient accumulation: {args.grad_accum} steps")
    print(f"  Label smoothing: {args.label_smoothing}")

    if args.resume:
        _ckpt_peek = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg_dict = _ckpt_peek.get("config", {})
        cfg = SemanticLMConfig(**{k: v for k, v in cfg_dict.items()
                                   if k in SemanticLMConfig.__dataclass_fields__})
        print(f"  Config loaded from checkpoint: {args.resume}")
        del _ckpt_peek
    else:
        vocab_size = args.semantic_vocab_size
        if not args.synthetic and vocab_size == 32768:
            detected = _detect_vocab_from_data(args.data)
            if detected and detected != vocab_size:
                print(f"  Auto-detected semantic_vocab_size={detected} from data")
                vocab_size = detected
        cfg = SemanticLMConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            semantic_vocab_size=vocab_size,
        )

    use_cross_attn = not args.no_cross_attention
    model = SonataSemanticLM(cfg, use_prosody=args.use_prosody,
                             use_cross_attention=use_cross_attn).to(device)

    if args.qat:
        exclude = [p.strip() for p in args.qat_exclude.split(",") if p.strip()]
        quantize_model(model, exclude_patterns=exclude)
        stats = compute_compression_stats(model)
        print(f"  QAT: quantized Linear layers (exclude: {exclude or 'none'})")
        print(f"  Compression: {stats['original_mb']:.1f}MB → {stats['quantized_mb']:.1f}MB ({stats['compression_ratio']:.1f}x)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.1f}M params")
    print(f"  {cfg.n_layers}L × d={cfg.d_model}, H={cfg.n_heads}, KV={cfg.n_kv_heads}")
    print(f"  Semantic vocab: {cfg.semantic_vocab_size}")
    print(f"  FF dim: {cfg.d_ff}")

    # Draft model for speculative decoding (trained alongside main model)
    draft_model = None
    draft_optimizer = None
    draft_ema = None
    if args.draft_layers > 0:
        draft_cfg = SemanticLMConfig(
            d_model=cfg.d_model, n_layers=args.draft_layers,
            n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
            semantic_vocab_size=cfg.semantic_vocab_size,
            text_vocab_size=cfg.text_vocab_size,
        )
        draft_model = SonataSemanticLM(draft_cfg, use_prosody=args.use_prosody,
                                        use_cross_attention=use_cross_attn).to(device)
        draft_n = sum(p.numel() for p in draft_model.parameters())
        print(f"  Draft: {draft_n/1e6:.1f}M params ({args.draft_layers}L)")
        draft_optimizer = torch.optim.AdamW(
            draft_model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
        draft_ema = EMA(draft_model, decay=args.ema_decay)

    if args.use_g2p:
        from g2p import PhonemeFrontend
        g2p = PhonemeFrontend()
        cfg.text_vocab_size = g2p.vocab_size
        print(f"  G2P: {g2p.vocab_size} phoneme tokens (replaces {32000} text tokens)")

    lazy = args.lazy_load or _resolve_data_paths(args.data).__len__() > 5
    dataset = SemanticDataset(
        args.data,
        max_seq_len=args.max_seq_len,
        text_vocab_size=cfg.text_vocab_size,
        semantic_vocab_size=cfg.semantic_vocab_size,
        synthetic=args.synthetic,
        use_g2p=args.use_g2p,
        use_prosody=args.use_prosody,
        lazy_load=lazy,
    )

    # Train/val split
    val_size = min(max(int(len(dataset) * 0.02), 100), 2000)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    use_bucket = not args.synthetic and (
        (not lazy and dataset.data)
        or (lazy and getattr(dataset, "_shard_paths", None))
    )
    sampler = BucketSampler(train_ds, args.batch_size, parent_dataset=dataset) if use_bucket else None
    loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, drop_last=False, collate_fn=collate_fn,
    )

    # Per-parameter weight decay: exclude biases and LayerNorm
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))

    ema = EMA(model, decay=args.ema_decay)
    print(f"  EMA: decay={args.ema_decay}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tlog = TrainingLog(str(ckpt_dir / "losses.jsonl"))

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys (new layers will be randomly initialized)")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys in checkpoint")
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError):
            print(f"  WARNING: Could not load optimizer state (architecture changed)")
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
            print(f"  EMA weights restored from checkpoint")
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed from step {start_step}")

    model.train()
    step = start_step
    running_loss = 0.0
    running_acc = 0.0
    running_eos_acc = 0.0
    t0 = time.time()

    eff_batch = args.batch_size * args.grad_accum
    print(f"\n  Training: step {step} → {args.steps}")
    print(f"  Batch size: {args.batch_size} × {args.grad_accum} accum = {eff_batch} effective")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Train/val split: {train_size}/{val_size}")
    print()

    optimizer.zero_grad()
    while step < args.steps:
        for text_ids, sem_ids, prosody in loader:
            if step >= args.steps:
                break

            text_ids = text_ids.to(device)
            sem_ids = sem_ids.to(device)
            prosody = prosody.to(device) if args.use_prosody else None

            sem_input = F.pad(sem_ids[:, :-1], (1, 0), value=1)
            sem_input = sem_input.clamp(min=0)  # -1 padding -> PAD(0) for embedding
            target = sem_ids

            lr = get_lr(step, args.warmup, args.lr, args.lr * 0.01, args.steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            logits, losses = model(text_ids, sem_input, target_semantic=target,
                                   prosody_features=prosody)
            loss = losses["semantic"]

            if args.label_smoothing > 0:
                n_classes = logits.shape[-1]
                log_probs = F.log_softmax(logits, dim=-1)
                smooth_loss = -log_probs.mean(dim=-1)
                mask = (target != -1).float()
                nll_loss = F.nll_loss(log_probs.transpose(1, 2), target, reduction="none", ignore_index=-1)
                loss = ((1 - args.label_smoothing) * nll_loss + args.label_smoothing * smooth_loss)
                loss = (loss * mask).sum() / mask.sum().clamp(min=1)

            loss = loss / args.grad_accum
            loss.backward()

            # Train draft model on same data
            if draft_model is not None:
                draft_logits, draft_losses = draft_model(
                    text_ids, sem_input, target_semantic=target, prosody_features=prosody)
                draft_loss = draft_losses["semantic"] / args.grad_accum
                draft_loss.backward()
                if step % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(draft_model.parameters(), 1.0)
                    draft_optimizer.step()
                    draft_optimizer.zero_grad()
                    draft_ema.update()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = target != -1
                acc = (preds[mask] == target[mask]).float().mean().item()
                eos_mask = target == 2
                eos_acc = (preds[eos_mask] == 2).float().mean().item() if eos_mask.any() else 0.0

            running_loss += loss.item() * args.grad_accum
            running_acc += acc
            running_eos_acc += eos_acc
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                ema.update()

            if step % args.log_every == 0:
                n = args.log_every
                elapsed = time.time() - t0
                sps = n / elapsed
                print(f"  step {step:6d} | loss={running_loss/n:.4f}"
                      f" acc={running_acc/n:.2%}"
                      f" eos={running_eos_acc/n:.2%} |"
                      f" lr={lr:.2e} | {sps:.1f} steps/s")
                tlog.log(step=step, loss=running_loss/n, acc=running_acc/n,
                         eos_acc=running_eos_acc/n, lr=lr, steps_per_sec=sps)
                running_loss = 0.0
                running_acc = 0.0
                running_eos_acc = 0.0
                t0 = time.time()

            if step % args.save_every == 0:
                path = ckpt_dir / f"sonata_lm_step_{step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema": ema.state_dict(),
                    "step": step,
                    "config": vars(cfg),
                    "best_val_loss": best_val_loss,
                }, path)
                print(f"  [ckpt] Saved {path}")

            # Validation (use EMA weights)
            if step % args.val_every == 0 and len(val_ds) > 0:
                ema.apply_shadow()
                val_loss, val_acc = validate_lm(model, val_loader, device, args.use_prosody)
                ema.restore()
                is_best = val_loss < best_val_loss
                print(f"  [val] step {step}: loss={val_loss:.4f} acc={val_acc:.2%}"
                      f" {'(best!)' if is_best else ''}")
                if is_best:
                    best_val_loss = val_loss
                    path = ckpt_dir / "sonata_lm_best.pt"
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "ema": ema.state_dict(),
                        "step": step,
                        "config": vars(cfg),
                        "best_val_loss": best_val_loss,
                    }, path)
                    print(f"  [best] Saved {path}")

    # Final save
    path = ckpt_dir / "sonata_lm_final.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "step": step,
        "config": vars(cfg),
        "best_val_loss": best_val_loss,
    }, path)
    print(f"\n  [ckpt] Final: {path}")

    if draft_model is not None:
        draft_path = ckpt_dir / "sonata_lm_draft.pt"
        draft_ema.apply_shadow()
        torch.save({
            "model": draft_model.state_dict(),
            "config": vars(SemanticLMConfig(
                d_model=cfg.d_model, n_layers=args.draft_layers,
                n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
                semantic_vocab_size=cfg.semantic_vocab_size,
                text_vocab_size=cfg.text_vocab_size,
            )),
            "step": step,
        }, draft_path)
        draft_ema.restore()
        print(f"  [draft] Saved {draft_path}")

    print(f"  Training complete: {step} steps, best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train/data/encoded_dev-clean.pt",
                        help="Data path: .pt file, comma-separated list, or .shards.txt index")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--checkpoint-dir", default="train/checkpoints/lm")
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--semantic-vocab-size", type=int, default=32768,
                        help="Semantic token vocabulary size (must match codec FSQ: 8^5=32768)")
    parser.add_argument("--no-cross-attention", action="store_true",
                        help="Use decoder-only mode (text concatenated, no cross-attention)")
    parser.add_argument("--use-g2p", action="store_true",
                        help="Use espeak-ng G2P phoneme input instead of character/SentencePiece")
    parser.add_argument("--use-prosody", action="store_true",
                        help="Condition on prosody features (pitch, energy, rate) from encoded data")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing for CE loss (0=none, 0.1=default)")
    parser.add_argument("--lazy-load", action="store_true",
                        help="Lazy shard loading to reduce memory for large datasets")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate for model weights")
    parser.add_argument("--draft-layers", type=int, default=0,
                        help="Train a draft model with N layers for speculative decoding (0=disable)")
    parser.add_argument("--qat", action="store_true",
                        help="Quantization-aware training (BitTTS-style ternary)")
    parser.add_argument("--qat-exclude", default="embedding,semantic_head",
                        help="Comma-separated name patterns to exclude from QAT")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
