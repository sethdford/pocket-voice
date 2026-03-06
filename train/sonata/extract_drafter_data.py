#\!/usr/bin/env python3
"""Extract semantic token sequences from Sonata LM for ReDrafter training.

The drafter model learns to predict what the Sonata LM would generate.
This script loads a trained LM checkpoint, runs inference on text inputs,
and saves (text_tokens, semantic_tokens) pairs suitable for training.

The extraction process:
  1. Load frozen Sonata LM checkpoint
  2. For each text utterance in the manifest:
     a. Tokenize text (via SentencePiece or G2P)
     b. Run greedy decoding on the LM to get semantic token sequence
     c. Cache the LM's hidden states and logits
  3. Save as sharded .pt files compatible with train_drafter.py

Usage:
    python extract_drafter_data.py \\
        --lm-ckpt checkpoints/lm/sonata_lm_final.pt \\
        --manifest data/manifest_clean.jsonl \\
        --output data/drafter_tokens/ \\
        --batch-size 8 --device mps

Output format:
    Each sample dict contains:
      - text_tokens: (T_text,) token IDs from text
      - semantic_tokens: (T_audio,) predicted semantic tokens from LM
      - utt_id: utterance ID for resume/debugging
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F

from config import SemanticLMConfig
from semantic_lm import SonataSemanticLM


class LMTextTokenizer:
    """Unified text tokenizer for the LM — handles SentencePiece or G2P."""

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

        # Try SentencePiece
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
        """Tokenize text to LM input IDs."""
        if self.use_g2p and self.g2p is not None:
            return self.g2p.encode(text, add_bos=False, add_eos=False)
        if self.sp is not None:
            ids = self.sp.EncodeAsIds(text)
            return torch.tensor(ids, dtype=torch.long)
        # Character-level fallback
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
def greedy_decode_semantic_tokens(
    lm: SonataSemanticLM,
    text_tokens: torch.Tensor,
    max_len: int = 512,
    pad_token_id: int = 0,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
) -> tuple:
    """Greedily decode semantic tokens from LM given text tokens.

    Args:
        lm: Frozen Sonata LM
        text_tokens: (T_text,) text token IDs
        max_len: Maximum semantic token sequence length
        pad_token_id: Padding token ID
        bos_token_id: Beginning-of-sequence token
        eos_token_id: End-of-sequence token

    Returns:
        (semantic_tokens, logits)
          - semantic_tokens: (T_semantic,) predicted tokens
          - logits: (T_semantic, vocab_size) LM logits for each step
    """
    text_tokens = text_tokens.unsqueeze(0)  # (1, T_text)
    B = 1
    device = text_tokens.device

    # Pre-compute text encoding (if using cross-attention)
    text_enc = lm.encode_text(text_tokens) if lm.use_cross_attention else None

    # Start with BOS token
    semantic_tokens = [bos_token_id]
    all_logits = []

    for step in range(max_len):
        # Current semantic token sequence: (1, T_semantic)
        sem_ids = torch.tensor(semantic_tokens, dtype=torch.long, device=device).unsqueeze(0)

        # Forward pass through LM
        logits, _ = lm(text_tokens, sem_ids, text_encoded=text_enc)
        # logits: (B, T_semantic, V)

        # Get logits at the last position (most recent token)
        last_logits = logits[:, -1, :]  # (B, V)
        all_logits.append(last_logits)

        # Greedy: take argmax
        next_token = last_logits.argmax(dim=-1).item()  # scalar
        semantic_tokens.append(next_token)

        # Stop on EOS
        if next_token == eos_token_id:
            break

    semantic_tokens = torch.tensor(semantic_tokens, dtype=torch.long, device=device)
    logits = torch.cat(all_logits, dim=0)  # (T_semantic, V)

    return semantic_tokens, logits


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

    lm = lm.to(device).eval()
    n_params = sum(p.numel() for p in lm.parameters())
    print(f"[extract] LM: {n_params / 1e6:.1f}M params")

    # Setup text tokenizer
    tokenizer = LMTextTokenizer(use_g2p=args.use_g2p)

    # Setup output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load resume state if available
    resume_file = out_dir / ".resume.json"
    completed = set()
    if args.resume and resume_file.exists():
        with open(resume_file) as f:
            state = json.load(f)
            completed = set(state.get("completed_utt_ids", []))
        print(f"[extract] Resume: {len(completed)} utterances already processed")

    # Iterate manifest and extract
    shard_buf = []
    shard_idx = 0
    n_extracted = 0
    n_skipped = 0
    n_errors = 0
    shard_paths = []
    t0 = time.time()

    def flush_shard(buf, idx):
        nonlocal shard_paths
        if not buf:
            return
        path = out_dir / f"drafter_tokens_shard_{idx:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(buf, path)
        shard_paths.append(str(path))
        print(f"  [shard {idx}] Saved {len(buf)} samples to {path}")

    for entry in iter_manifest(args.manifest):
        utt_id = entry.get("utt_id", f"utt_{n_extracted:06d}")

        # Skip if already processed
        if args.resume and utt_id in completed:
            n_skipped += 1
            continue

        text = entry.get("text", "").strip()
        if not text:
            n_errors += 1
            continue

        try:
            # Tokenize text
            text_tokens = tokenizer.encode(text)

            # Truncate to max length
            if len(text_tokens) > args.max_text_len:
                text_tokens = text_tokens[: args.max_text_len]

            # Greedy decode semantic tokens
            semantic_tokens, logits = greedy_decode_semantic_tokens(
                lm,
                text_tokens.to(device),
                max_len=args.max_semantic_len,
                bos_token_id=1,
                eos_token_id=2,
            )

            # Clip to max length (excluding EOS)
            if len(semantic_tokens) > args.max_semantic_len:
                semantic_tokens = semantic_tokens[: args.max_semantic_len]

            shard_buf.append({
                "text_tokens": text_tokens.cpu(),
                "semantic_tokens": semantic_tokens.cpu(),
                "utt_id": utt_id,
            })
            n_extracted += 1

            if n_extracted % args.log_interval == 0:
                elapsed = time.time() - t0
                rate = n_extracted / elapsed
                print(f"  [{n_extracted}] {rate:.1f} utt/s, {n_errors} errors, "
                      f"{len(shard_buf)} buffered")

            # Flush shard if full
            if len(shard_buf) >= args.shard_size:
                flush_shard(shard_buf, shard_idx)
                shard_buf = []
                shard_idx += 1

        except Exception as e:
            n_errors += 1
            if n_errors <= 10:
                print(f"[extract] Error on '{utt_id}': {e}")

    # Final flush
    flush_shard(shard_buf, shard_idx)

    elapsed = time.time() - t0

    # Save resume state
    if args.resume:
        completed_list = list(completed)
        with open(resume_file, "w") as f:
            json.dump({"completed_utt_ids": completed_list}, f, indent=2)

    # Summary
    print(f"\n[extract] Complete:")
    print(f"  Extracted: {n_extracted:,} utterances")
    print(f"  Skipped:   {n_skipped:,} (already processed)")
    print(f"  Errors:    {n_errors:,}")
    print(f"  Time:      {elapsed:.0f}s ({n_extracted / max(elapsed, 1):.1f} utt/s)")
    print(f"  Shards:    {len(shard_paths)}")

    if len(shard_paths) > 1:
        index_path = out_dir / ".shards.txt"
        with open(index_path, "w") as f:
            for p in shard_paths:
                f.write(p + "\n")
        print(f"  Index:     {index_path}")
        print(f"\n  To train drafter:")
        print(f"    python train_drafter.py --base_model {args.lm_ckpt} \\")
        print(f"                            --data {out_dir} \\")
        print(f"                            --output models/sonata/rnn_drafter.safetensors")
    elif shard_paths:
        print(f"  Output:    {shard_paths[0]}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract semantic token sequences from Sonata LM for ReDrafter training"
    )
    parser.add_argument(
        "--lm-ckpt",
        required=True,
        help="Path to Sonata LM checkpoint (frozen during extraction)"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSONL manifest with {text, utt_id, ...} entries"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for sharded token data"
    )
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device (mps, cpu, cuda)"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="Utterances per shard file"
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=512,
        help="Max text tokens per utterance"
    )
    parser.add_argument(
        "--max-semantic-len",
        type=int,
        default=512,
        help="Max semantic tokens to generate"
    )
    parser.add_argument(
        "--use-g2p",
        action="store_true",
        help="Use G2P (phoneme) tokenizer instead of SentencePiece"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed utterance"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print stats every N utterances"
    )

    args = parser.parse_args()
    extract_drafter_samples(args)


if __name__ == "__main__":
    main()
