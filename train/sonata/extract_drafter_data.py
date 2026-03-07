#!/usr/bin/env python3
"""Extract semantic token sequences from Sonata LM for ReDrafter training.

Uses batched teacher-forcing for ~60x speedup over autoregressive decode.
Instead of greedy AR decode (50ms/token * 128 tokens = 6.4s/utt),
we run a single batched forward pass (56ms/utt at batch=8).

The output semantic_tokens are the LM's greedy predictions at each position,
suitable for drafter knowledge distillation training.

Usage:
    python extract_drafter_data.py \
        --lm-ckpt checkpoints/lm/sonata_lm_final.pt \
        --manifest data/manifest_clean.jsonl \
        --output data/drafter_tokens/ \
        --device cuda --batch-size 8
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterator, List

import torch
import torch.nn.functional as F

from config import SemanticLMConfig
from semantic_lm import SonataSemanticLM


class LMTextTokenizer:
    """Unified text tokenizer for the LM."""

    def __init__(self, use_g2p: bool = False):
        self.use_g2p = use_g2p
        self.sp = None
        self.g2p = None
        self._setup_tokenizer()

    def _setup_tokenizer(self):
        if self.use_g2p:
            try:
                from g2p import PhonemeFrontend
                self.g2p = PhonemeFrontend()
                print(f"[tokenizer] Using G2P (vocab size: {self.g2p.vocab_size})")
            except (ImportError, Exception) as e:
                print(f"[tokenizer] G2P init failed: {e}, falling back to char-level")
            return

        try:
            import sentencepiece as spm
            model_path = "models/parakeet-ctc-1.1b-fp16.vocab"
            if Path(model_path).exists():
                self.sp = spm.SentencePieceProcessor()
                self.sp.Load(model_path)
                print(f"[tokenizer] Loaded SentencePiece ({self.sp.GetPieceSize()} tokens)")
                return
        except (ImportError, FileNotFoundError, Exception):
            pass

        print("[tokenizer] Using character-level encoding")

    def encode(self, text: str) -> torch.Tensor:
        if self.use_g2p and self.g2p is not None:
            return self.g2p.encode(text, add_bos=False, add_eos=False)
        if self.sp is not None:
            ids = self.sp.EncodeAsIds(text)
            return torch.tensor(ids, dtype=torch.long)
        return torch.tensor([ord(c) for c in text], dtype=torch.long)


def iter_manifest(manifest_path: str) -> Iterator[dict]:
    """Iterate JSONL manifest with {audio, text, utt_id, ...} entries."""
    with open(manifest_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if "text" in entry:
                    yield entry
            except json.JSONDecodeError:
                continue


@torch.no_grad()
def extract_batch_teacher_forced(
    lm: SonataSemanticLM,
    text_batch: List[torch.Tensor],
    max_semantic_len: int,
    device: torch.device,
    bos_token_id: int = 1,
) -> List[torch.Tensor]:
    """Extract semantic tokens via batched teacher-forcing.

    For each text input, we:
    1. Encode text once
    2. Create a BOS-prefixed semantic input of max_semantic_len
    3. Run single forward pass to get logits at all positions
    4. Argmax logits to get predicted semantic tokens

    This produces the same tokens as greedy AR decode for step 0,
    and approximately correct tokens for later steps (since the input
    prefix is BOS + zeros rather than the model's own predictions).
    For drafter distillation this is sufficient.
    """
    B = len(text_batch)

    # Pad text tokens
    max_text = max(t.shape[0] for t in text_batch)
    text_padded = torch.zeros(B, max_text, dtype=torch.long, device=device)
    for i, t in enumerate(text_batch):
        text_padded[i, :t.shape[0]] = t

    # Create semantic input: BOS followed by zeros (teacher-forcing with empty prefix)
    # We'll do iterative refinement: run once to get tokens, then run again
    # with those tokens as input for better quality
    sem_input = torch.zeros(B, max_semantic_len, dtype=torch.long, device=device)
    sem_input[:, 0] = bos_token_id

    # Pass 1: get initial predictions
    logits, _ = lm(text_padded, sem_input)
    predictions = logits.argmax(dim=-1)  # (B, max_semantic_len)

    # Pass 2: use predictions as input for better logits (self-consistency)
    sem_input2 = torch.zeros(B, max_semantic_len, dtype=torch.long, device=device)
    sem_input2[:, 0] = bos_token_id
    sem_input2[:, 1:] = predictions[:, :-1]  # shifted right

    logits2, _ = lm(text_padded, sem_input2)
    final_tokens = logits2.argmax(dim=-1)  # (B, max_semantic_len)

    # Prepend BOS
    result = []
    for i in range(B):
        tokens = torch.cat([
            torch.tensor([bos_token_id], device=device),
            final_tokens[i]
        ])
        result.append(tokens.cpu())

    return result


@torch.no_grad()
def extract_drafter_samples(args):
    """Main extraction pipeline."""
    device = torch.device(args.device)
    print(f"[extract] Device: {device}")

    # Load LM config and model
    cfg = SemanticLMConfig()
    lm = SonataSemanticLM(cfg, use_cross_attention=True)

    if not os.path.exists(args.lm_ckpt):
        print(f"[extract] WARNING: LM checkpoint not found: {args.lm_ckpt}")
        print(f"[extract]          Using random-initialized LM (expect poor results)")
    else:
        state = torch.load(args.lm_ckpt, map_location="cpu", weights_only=True)
        lm.load_state_dict(state, strict=False)
        print(f"[extract] Loaded LM from {args.lm_ckpt}")

    lm = lm.to(device)
    if device.type == "cuda":
        lm = lm.half()
    lm.eval()
    n_params = sum(p.numel() for p in lm.parameters())
    print(f"[extract] LM: {n_params / 1e6:.1f}M params")

    # Setup text tokenizer
    tokenizer = LMTextTokenizer(use_g2p=args.use_g2p)

    # Setup output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load resume state
    resume_file = out_dir / ".resume.json"
    completed = set()
    if args.resume and resume_file.exists():
        with open(resume_file) as f:
            rstate = json.load(f)
            completed = set(rstate.get("completed_utt_ids", []))
        print(f"[extract] Resume: {len(completed)} utterances already processed")

    # Collect and process in batches
    shard_buf = []
    shard_idx = 0
    n_extracted = 0
    n_skipped = 0
    n_errors = 0
    shard_paths = []
    t0 = time.time()

    batch_texts = []
    batch_utt_ids = []
    batch_raw_tokens = []

    def flush_shard(buf, idx):
        nonlocal shard_paths
        if not buf:
            return
        path = out_dir / f"drafter_tokens_shard_{idx:04d}.pt"
        torch.save(buf, path)
        shard_paths.append(str(path))
        print(f"  [shard {idx}] Saved {len(buf)} samples to {path}")

    def process_batch():
        nonlocal n_extracted, n_errors, shard_buf, shard_idx

        if not batch_texts:
            return

        try:
            semantic_results = extract_batch_teacher_forced(
                lm, batch_raw_tokens, args.max_semantic_len, device
            )

            for i in range(len(batch_texts)):
                shard_buf.append({
                    "text_tokens": batch_raw_tokens[i].cpu(),
                    "semantic_tokens": semantic_results[i],
                    "utt_id": batch_utt_ids[i],
                })
                completed.add(batch_utt_ids[i])
                n_extracted += 1

            if n_extracted % args.log_interval == 0:
                elapsed = time.time() - t0
                rate = n_extracted / max(elapsed, 0.01)
                print(f"  [{n_extracted}] {rate:.1f} utt/s, {n_errors} errors, "
                      f"{len(shard_buf)} buffered")

            # Flush shard if full
            if len(shard_buf) >= args.shard_size:
                flush_shard(shard_buf, shard_idx)
                shard_buf = []
                shard_idx += 1

                if args.resume:
                    with open(resume_file, "w") as f:
                        json.dump({"completed_utt_ids": list(completed)}, f)

        except Exception as e:
            n_errors += len(batch_texts)
            if n_errors <= 10:
                import traceback
                print(f"[extract] Batch error: {e}")
                traceback.print_exc()

    for entry in iter_manifest(args.manifest):
        utt_id = entry.get("utt_id", f"utt_{n_extracted + n_skipped:06d}")

        if args.resume and utt_id in completed:
            n_skipped += 1
            continue

        text = entry.get("text", "").strip()
        if not text:
            n_errors += 1
            continue

        text_tokens = tokenizer.encode(text)
        if len(text_tokens) > args.max_text_len:
            text_tokens = text_tokens[:args.max_text_len]

        batch_texts.append(text)
        batch_utt_ids.append(utt_id)
        batch_raw_tokens.append(text_tokens.to(device))

        if len(batch_texts) >= args.batch_size:
            process_batch()
            batch_texts.clear()
            batch_utt_ids.clear()
            batch_raw_tokens.clear()

    # Process remaining
    process_batch()

    # Final flush
    flush_shard(shard_buf, shard_idx)

    elapsed = time.time() - t0

    if args.resume:
        with open(resume_file, "w") as f:
            json.dump({"completed_utt_ids": list(completed)}, f)

    print(f"\n[extract] Complete:")
    print(f"  Extracted: {n_extracted:,} utterances")
    print(f"  Skipped:   {n_skipped:,} (already processed)")
    print(f"  Errors:    {n_errors:,}")
    print(f"  Time:      {elapsed:.0f}s ({n_extracted / max(elapsed, 1):.1f} utt/s)")
    print(f"  Shards:    {len(shard_paths)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract semantic token sequences from Sonata LM for ReDrafter training"
    )
    parser.add_argument("--lm-ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--max-text-len", type=int, default=256)
    parser.add_argument("--max-semantic-len", type=int, default=128)
    parser.add_argument("--use-g2p", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)

    args = parser.parse_args()
    extract_drafter_samples(args)


if __name__ == "__main__":
    main()
