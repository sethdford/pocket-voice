#!/usr/bin/env python3
"""
train_semantic_eou.py — Train byte-level LSTM for sentence completion detection.

Architecture (must match semantic_eou.c):
  Byte embedding[256x32] -> LayerNorm(32) -> LSTM(input=32, hidden=64) -> ReLU -> Linear(1) -> sigmoid

~33K parameters, <1ms inference on Apple Silicon.

Training data:
  - LibriSpeech transcripts split at natural sentence boundaries (complete)
  - Same transcripts split mid-sentence (incomplete)
  - Binary classification: 1 = complete, 0 = incomplete

Usage:
  # Install dependencies
  pip install torch datasets

  # Train
  python train_semantic_eou.py --epochs 20 --batch-size 64

  # Export weights for C inference
  python train_semantic_eou.py --export --checkpoint checkpoints/semantic_eou_best.pt \
    --output models/semantic_eou.seou
"""

import argparse
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Architecture constants (must match semantic_eou.h) ──────────────────────

VOCAB_SIZE = 256      # byte-level tokenization
EMBED_DIM = 32
HIDDEN_DIM = 64
MAX_SEQ_LEN = 128

# ── Binary format constants (must match semantic_eou.c) ─────────────────────

SEOU_MAGIC = 0x554F4553     # "SEOU" little-endian
SEOU_VERSION = 1

# ── Model ────────────────────────────────────────────────────────────────────


class SemanticEOUModel(nn.Module):
    """Byte-level LSTM for sentence completion detection."""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0)
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        # Forget gate bias = 1.0 (only on bias_ih_l0, will be added with bias_hh_l0 in export)
        gate_size = self.hidden_dim
        self.lstm.bias_ih_l0.data[gate_size:2 * gate_size] = 1.0
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        """x: [batch, seq_len] of byte tokens (0-255). Returns logits [batch, 1]."""
        emb = self.embedding(x)           # [B, T, E]
        emb = self.norm(emb)              # [B, T, E]
        _, (h_n, _) = self.lstm(emb)      # h_n: [1, B, H]
        h = h_n.squeeze(0)               # [B, H]
        h = torch.relu(h)                # ReLU before output
        return self.output(h).squeeze(-1)  # [B]


# ── Dataset ──────────────────────────────────────────────────────────────────

# Sentence-ending punctuation patterns
SENTENCE_END_RE = re.compile(r'[.!?]+\s*$')
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def text_to_tokens(text, max_len=MAX_SEQ_LEN):
    """Convert text to byte tokens, taking last max_len bytes."""
    raw = text.encode("utf-8", errors="replace")
    if len(raw) > max_len:
        raw = raw[-max_len:]
    tokens = list(raw)
    # Pad to max_len
    if len(tokens) < max_len:
        tokens = [0] * (max_len - len(tokens)) + tokens
    return tokens


class SentenceCompletionDataset(Dataset):
    """Binary classification: is this text fragment a complete sentence?"""

    def __init__(self, transcripts, max_len=MAX_SEQ_LEN):
        self.samples = []
        self.max_len = max_len
        self._build_samples(transcripts)

    def _build_samples(self, transcripts):
        for text in transcripts:
            text = text.strip()
            if not text:
                continue

            # Split into sentences
            sentences = SENTENCE_SPLIT_RE.split(text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

            for sent in sentences:
                # Complete sentence (label=1)
                self.samples.append((sent, 1))

                # Incomplete fragments (label=0) — cut at various word positions
                words = sent.split()
                if len(words) >= 4:
                    # Cut at ~25%, ~50%, and ~75% word boundaries
                    for frac in [0.25, 0.5, 0.75]:
                        cut = max(2, int(len(words) * frac))
                        fragment = " ".join(words[:cut])
                        self.samples.append((fragment, 0))

                # Also create fragments without final punctuation
                stripped = sent.rstrip(".!?;:,")
                if stripped != sent and len(stripped) > 3:
                    self.samples.append((stripped, 0))

                # Hard negatives: mid-word truncation (character-level cuts)
                if len(sent) >= 10:
                    # Cut at ~60%, ~70%, ~80% of characters (mid-word)
                    for frac in [0.6, 0.7, 0.8]:
                        cut = max(5, int(len(sent) * frac))
                        # Only add if this doesn't end at a space (true mid-word)
                        if cut < len(sent) - 1 and sent[cut] != ' ':
                            fragment = sent[:cut]
                            self.samples.append((fragment, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        tokens = text_to_tokens(text, self.max_len)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


def load_librispeech_transcripts():
    """Load transcripts from LibriSpeech via HuggingFace datasets."""
    try:
        from datasets import load_dataset
        ds = load_dataset("librispeech_asr", "clean", split="train.100")
        transcripts = [item["text"] for item in ds if item.get("text")]
        print(f"Loaded {len(transcripts)} transcripts from LibriSpeech clean-100")
        return transcripts
    except Exception as e:
        print(f"Could not load LibriSpeech: {e}")
        print("Falling back to synthetic training data")
        return generate_synthetic_transcripts()


def generate_synthetic_transcripts():
    """Generate synthetic training data for testing."""
    complete = [
        "The weather is beautiful today.",
        "I think we should go to the park.",
        "Can you tell me where the library is?",
        "She finished reading the book yesterday.",
        "The meeting starts at three o'clock.",
        "We need to buy some groceries.",
        "He decided to take the train instead.",
        "The movie was really entertaining.",
        "I would like to order a coffee please.",
        "They arrived at the airport on time.",
        "The project deadline is next Friday.",
        "She always takes her dog for a walk.",
        "The restaurant serves excellent pasta.",
        "We should schedule a meeting for tomorrow.",
        "He completed the assignment before midnight.",
        "The children are playing in the garden.",
        "I need to update my phone software.",
        "The concert was absolutely amazing!",
        "Can you help me with this problem?",
        "She graduated from university last year.",
    ]

    # Repeat with variations for more training data
    transcripts = []
    for sent in complete:
        transcripts.append(sent)
        # Add concatenated versions
        for other in complete[:5]:
            transcripts.append(f"{sent} {other}")

    print(f"Generated {len(transcripts)} synthetic transcripts")
    return transcripts


# ── Training ─────────────────────────────────────────────────────────────────


def train(args):
    """Train the semantic EOU model."""
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    transcripts = load_librispeech_transcripts()
    dataset = SentenceCompletionDataset(transcripts)
    print(f"Dataset size: {len(dataset)} samples")

    # Split train/val
    val_size = min(len(dataset) // 10, 2000)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SemanticEOUModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params * 4 / 1024:.1f} KB)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_tp = train_fp = train_fn = 0  # For precision/recall/F1

        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            logits = model(tokens)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * tokens.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += tokens.size(0)

            # Metrics for F1
            train_tp += ((preds == 1) & (labels == 1)).sum().item()
            train_fp += ((preds == 1) & (labels == 0)).sum().item()
            train_fn += ((preds == 0) & (labels == 1)).sum().item()

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = val_fp = val_fn = 0

        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens, labels = tokens.to(device), labels.to(device)
                logits = model(tokens)
                loss = criterion(logits, labels)

                val_loss += loss.item() * tokens.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += tokens.size(0)

                # Metrics for F1
                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        # Compute precision, recall, F1
        train_prec = train_tp / max(train_tp + train_fp, 1)
        train_rec = train_tp / max(train_tp + train_fn, 1)
        train_f1 = 2 * train_prec * train_rec / max(train_prec + train_rec, 1e-6)

        val_prec = val_tp / max(val_tp + val_fp, 1)
        val_rec = val_tp / max(val_tp + val_fn, 1)
        val_f1 = 2 * val_prec * val_rec / max(val_prec + val_rec, 1e-6)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss={train_loss / max(train_total, 1):.4f} acc={train_acc:.3f} "
              f"prec={train_prec:.3f} rec={train_rec:.3f} f1={train_f1:.3f} | "
              f"Val loss={val_loss / max(val_total, 1):.4f} acc={val_acc:.3f} "
              f"prec={val_prec:.3f} rec={val_rec:.3f} f1={val_f1:.3f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.checkpoint_dir, "semantic_eou_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  -> Saved best model (val_acc={val_acc:.3f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    return best_val_acc


# ── Export ────────────────────────────────────────────────────────────────────


def export_seou(args):
    """Export trained model to .seou binary weight file for C inference."""
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = SemanticEOUModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    state = model.state_dict()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "wb") as f:
        # Header: magic(4) version(4) vocab(4) embed(4) hidden(4) seq_len(4) = 24 bytes
        f.write(struct.pack("<IIIIII",
                            SEOU_MAGIC, SEOU_VERSION,
                            VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, MAX_SEQ_LEN))

        # Embedding table [V x E]
        emb = state["embedding.weight"].numpy().astype(np.float32)
        assert emb.shape == (VOCAB_SIZE, EMBED_DIM)
        f.write(emb.tobytes())

        # LayerNorm: weight [E] + bias [E]
        norm_w = state["norm.weight"].numpy().astype(np.float32)
        norm_b = state["norm.bias"].numpy().astype(np.float32)
        f.write(norm_w.tobytes())
        f.write(norm_b.tobytes())

        # LSTM weights
        # PyTorch LSTM weight_ih: [4H, input_size] — matches C layout
        # PyTorch LSTM weight_hh: [4H, hidden_size] — matches C layout
        # PyTorch gate order: i, f, g, o (same as our C code)
        wi = state["lstm.weight_ih_l0"].numpy().astype(np.float32)
        wh = state["lstm.weight_hh_l0"].numpy().astype(np.float32)
        assert wi.shape == (4 * HIDDEN_DIM, EMBED_DIM)
        assert wh.shape == (4 * HIDDEN_DIM, HIDDEN_DIM)
        f.write(wi.tobytes())
        f.write(wh.tobytes())

        # LSTM bias: combine ih and hh biases
        bi = state["lstm.bias_ih_l0"].numpy().astype(np.float32)
        bh = state["lstm.bias_hh_l0"].numpy().astype(np.float32)
        bias = (bi + bh).astype(np.float32)
        assert bias.shape == (4 * HIDDEN_DIM,)
        f.write(bias.tobytes())

        # Output: weight [1, H] -> flatten to [H], bias [1] -> scalar
        out_w = state["output.weight"].numpy().astype(np.float32).flatten()
        out_b = state["output.bias"].numpy().astype(np.float32).flatten()
        assert out_w.shape == (HIDDEN_DIM,)
        assert out_b.shape == (1,)
        f.write(out_w.tobytes())
        f.write(out_b.tobytes())

    # Verify
    expected_params = (VOCAB_SIZE * EMBED_DIM +     # embedding
                       EMBED_DIM + EMBED_DIM +       # layer norm
                       4 * HIDDEN_DIM * EMBED_DIM +  # lstm_wi
                       4 * HIDDEN_DIM * HIDDEN_DIM + # lstm_wh
                       4 * HIDDEN_DIM +              # lstm_bias
                       HIDDEN_DIM +                  # out_w
                       1)                            # out_b
    file_size = os.path.getsize(args.output)
    expected_size = 24 + expected_params * 4  # header + floats
    print(f"Exported {args.output}: {expected_params:,} params "
          f"({file_size / 1024:.1f} KB, expected {expected_size / 1024:.1f} KB)")
    assert file_size == expected_size, f"Size mismatch: {file_size} != {expected_size}"
    print("Export verified OK")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train semantic EOU model")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--export", action="store_true", help="Export weights only")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to export from")
    parser.add_argument("--output", type=str, default="models/semantic_eou.seou",
                        help="Output .seou file path")
    args = parser.parse_args()

    if args.export:
        if not args.checkpoint:
            print("Error: --checkpoint required for export")
            sys.exit(1)
        export_seou(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
