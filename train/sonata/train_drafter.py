"""Train RNN (GRU) draft model for ReDrafter-style tree speculative decoding.

The draft model is a small GRU conditioned on the frozen Sonata LM's hidden states.
It learns to predict the next token distribution via knowledge distillation from
the main model's logits.

Architecture:
  - hidden_proj: Linear(d_model → gru_hidden)  [projects LM hidden state]
  - token_emb:   Embedding(vocab_size, emb_dim) [draft-specific embeddings]
  - gru:         2-layer GRU(emb_dim, gru_hidden)
  - output_head: Linear(gru_hidden, vocab_size)

Training:
  - Base LM is frozen (no gradient)
  - Loss: KL divergence from LM logits (knowledge distillation)
  - Optional: cross-entropy on ground-truth tokens (weighted blend)
  - Trains in ~10% of the compute of training the full LM

Usage:
    python train_drafter.py --base_model models/sonata/sonata_lm.pt \\
                            --data data/semantic_tokens/ \\
                            --output models/sonata/rnn_drafter.safetensors
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import SemanticLMConfig
from semantic_lm import SonataSemanticLM


class GruDrafter(nn.Module):
    """GRU-based draft model for ReDrafter speculative decoding.

    Conditioned on the main LM's hidden state at position t, predicts tokens
    at positions t+1, t+2, ..., t+D. Trained via knowledge distillation.

    Params (~3.5M with defaults):
      hidden_proj: 1024 * 512            = 524K
      token_emb:   4096 * 256            = 1049K
      gru (2 layers): 2 * 3 * (256*512 + 512*512) = 3932K  (approx with bias)
      output_head: 512 * 4096            = 2097K
      Total: ~3.5M params (vs 241M main model = 1.4%)
    """

    def __init__(
        self,
        d_model: int = 1024,
        vocab_size: int = 4096,
        gru_hidden: int = 512,
        gru_layers: int = 2,
        emb_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.emb_dim = emb_dim

        # Project LM hidden state to initial GRU state
        self.hidden_proj = nn.Linear(d_model, gru_hidden, bias=False)

        # Draft-specific token embeddings (smaller than LM embeddings)
        self.token_emb = nn.Embedding(vocab_size, emb_dim)

        # Multi-layer GRU: maps token embedding → hidden state sequence
        # Note: we use manual GRU cells to match the Rust implementation exactly
        self.gru_cells = nn.ModuleList()
        self.gru_cells.append(GruCellModule(emb_dim, gru_hidden))
        for _ in range(1, gru_layers):
            self.gru_cells.append(GruCellModule(gru_hidden, gru_hidden))

        # Project hidden state to vocabulary logits
        self.output_head = nn.Linear(gru_hidden, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.hidden_proj.weight, std=0.02)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.output_head.weight, std=0.02)
        for cell in self.gru_cells:
            for name, p in cell.named_parameters():
                nn.init.normal_(p, std=0.02)

    def forward(
        self,
        lm_hidden: torch.Tensor,
        token_ids: torch.Tensor,
        max_steps: int = 3,
    ) -> torch.Tensor:
        """Training forward: predict next tokens from LM hidden states.

        Args:
            lm_hidden: (B, T, d_model) — hidden states from frozen LM
            token_ids: (B, T) — ground-truth token IDs at each position
            max_steps: number of future positions to predict per position

        Returns:
            all_logits: (B, T, max_steps, vocab_size) — predicted logits
        """
        B, T, D = lm_hidden.shape

        # Project all hidden states to GRU initial state: (B, T, gru_hidden)
        h0 = self.hidden_proj(lm_hidden)

        all_logits = []
        for step in range(max_steps):
            # Token input: for step 0, use the token at position t
            # For step s > 0, use the token at position t+s (teacher forcing)
            if step == 0:
                tok_input = token_ids  # (B, T)
            else:
                # Shift tokens: position t needs token at t+step
                tok_input = torch.zeros_like(token_ids)
                if T > step:
                    tok_input[:, :T-step] = token_ids[:, step:]

            emb = self.token_emb(tok_input)  # (B, T, emb_dim)

            # Run GRU: process each position with the same GRU but different init states
            # Flatten batch and time for efficient processing
            emb_flat = emb.reshape(B * T, self.emb_dim)  # (B*T, emb_dim)
            h_flat = h0.reshape(B * T, self.gru_hidden)   # (B*T, gru_hidden)

            # Initialize layer states
            layer_h = [h_flat if i == 0 else torch.zeros_like(h_flat)
                       for i in range(self.gru_layers)]

            x = emb_flat
            for li, cell in enumerate(self.gru_cells):
                layer_h[li] = cell(x, layer_h[li])
                x = layer_h[li]

            logits = self.output_head(x)  # (B*T, vocab_size)
            logits = logits.reshape(B, T, self.vocab_size)
            all_logits.append(logits)

            # Update h0 for next step: the GRU state after this step
            h0 = x.reshape(B, T, self.gru_hidden)

        return torch.stack(all_logits, dim=2)  # (B, T, max_steps, vocab_size)


class GruCellModule(nn.Module):
    """Manual GRU cell matching the Rust implementation (no bias, explicit gates)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.w_z = nn.Linear(input_dim, hidden_dim, bias=False)
        self.u_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_r = nn.Linear(input_dim, hidden_dim, bias=False)
        self.u_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_h = nn.Linear(input_dim, hidden_dim, bias=False)
        self.u_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.w_z(x) + self.u_z(h))
        r = torch.sigmoid(self.w_r(x) + self.u_r(h))
        h_cand = torch.tanh(self.w_h(x) + self.u_h(r * h))
        return (1 - z) * h + z * h_cand


class SemanticTokenDataset(Dataset):
    """Dataset of (text_tokens, semantic_tokens) pairs for drafter training."""

    def __init__(self, data_dir: str, max_len: int = 512):
        self.samples = []
        data_path = Path(data_dir)
        for f in sorted(data_path.glob("*.pt")):
            data = torch.load(f, weights_only=True)
            # Handle both shard format (list of dicts) and single-sample format
            if isinstance(data, list):
                for sample in data:
                    if isinstance(sample, dict) and "text_tokens" in sample and "semantic_tokens" in sample:
                        self.samples.append(sample)
            elif isinstance(data, dict) and "text_tokens" in data and "semantic_tokens" in data:
                self.samples.append(data)
        for f in sorted(data_path.glob("*.json")):
            with open(f) as fp:
                sample = json.load(fp)
            if "text_tokens" in sample and "semantic_tokens" in sample:
                self.samples.append({
                    "text_tokens": torch.tensor(sample["text_tokens"], dtype=torch.long),
                    "semantic_tokens": torch.tensor(sample["semantic_tokens"], dtype=torch.long),
                })
        self.max_len = max_len
        print(f"[train_drafter] Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = s["text_tokens"][:self.max_len]
        sem = s["semantic_tokens"][:self.max_len]
        return text, sem


def collate_fn(batch):
    texts, sems = zip(*batch)
    max_t = max(t.shape[0] for t in texts)
    max_s = max(s.shape[0] for s in sems)
    text_padded = torch.zeros(len(texts), max_t, dtype=torch.long)
    sem_padded = torch.zeros(len(sems), max_s, dtype=torch.long)
    for i, (t, s) in enumerate(zip(texts, sems)):
        text_padded[i, :t.shape[0]] = t
        sem_padded[i, :s.shape[0]] = s
    return text_padded, sem_padded


def extract_hidden_states(
    base_model: SonataSemanticLM,
    text_tokens: torch.Tensor,
    semantic_tokens: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run base model and extract hidden states + logits (no gradient)."""
    base_model.eval()
    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
        B, T_audio = semantic_tokens.shape

        if base_model.use_cross_attention:
            text_enc = base_model.encode_text(text_tokens)
            x = base_model.semantic_emb(semantic_tokens)
            T_total = T_audio
        else:
            text_x = base_model.text_emb(text_tokens)
            sem_x = base_model.semantic_emb(semantic_tokens)
            x = torch.cat([text_x, sem_x], dim=1)
            T_total = x.shape[1]

        freqs = base_model.rope_freqs[:T_total]
        causal_mask = torch.tril(
            torch.ones(T_total, T_total, device=device, dtype=torch.bool)
        )
        text_enc_for_layers = text_enc if base_model.use_cross_attention else None
        for layer in base_model.layers:
            x = layer(x, text_enc_for_layers, freqs, mask=causal_mask)

        if not base_model.use_cross_attention:
            T_text = text_tokens.shape[1]
            x = x[:, T_text:]

        hidden = base_model.output_norm(x)
        logits = base_model.semantic_head(hidden)

    return hidden, logits


def train_drafter(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[train_drafter] Device: {device}")

    # Load base model (frozen)
    cfg = SemanticLMConfig()
    base_model = SonataSemanticLM(cfg)
    if os.path.exists(args.base_model):
        state = torch.load(args.base_model, map_location="cpu", weights_only=True)
        base_model.load_state_dict(state, strict=False)
        print(f"[train_drafter] Base model loaded from {args.base_model}")
    base_model = base_model.to(device).eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    total_vocab = cfg.semantic_vocab_size + cfg.n_special_tokens

    # Create drafter
    drafter = GruDrafter(
        d_model=cfg.d_model,
        vocab_size=cfg.semantic_vocab_size,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        emb_dim=args.emb_dim,
    ).to(device)

    n_params = sum(p.numel() for p in drafter.parameters())
    print(f"[train_drafter] GRU drafter: {n_params/1e6:.2f}M params")

    # Data
    dataset = SemanticTokenDataset(args.data, max_len=args.max_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        drafter.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader),
    )

    # Training loop
    max_steps = args.tree_depth
    kd_weight = args.kd_weight  # knowledge distillation weight
    ce_weight = 1.0 - kd_weight
    kd_temp = args.kd_temperature

    best_loss = float("inf")

    for epoch in range(args.epochs):
        drafter.train()
        total_loss = 0.0
        n_batches = 0

        for text_tokens, sem_tokens in loader:
            text_tokens = text_tokens.to(device)
            sem_tokens = sem_tokens.to(device)

            # Get hidden states from frozen base model
            hidden, lm_logits = extract_hidden_states(
                base_model, text_tokens, sem_tokens, device,
            )

            # Drafter forward with AMP for memory efficiency
            use_amp = device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                draft_logits = drafter(hidden, sem_tokens, max_steps=max_steps)
                # draft_logits: (B, T, max_steps, V)

                B, T, _, V = draft_logits.shape
                loss = torch.tensor(0.0, device=device)

                for step in range(max_steps):
                    # Target: token at position t + step + 1
                    shift = step + 1
                    if T <= shift:
                        continue

                    step_logits = draft_logits[:, :T-shift, step, :]  # (B, T-shift, V)
                    target_tokens = sem_tokens[:, shift:T]             # (B, T-shift)

                    # Cross-entropy loss on ground truth
                    ce_loss = F.cross_entropy(
                        step_logits.reshape(-1, V),
                        target_tokens.reshape(-1),
                        ignore_index=0,
                    )

                    # Knowledge distillation: KL div from base model logits
                    if shift <= 1:
                        lm_step_logits = lm_logits[:, :T-shift, :]
                    else:
                        lm_step_logits = lm_logits[:, shift-1:T-1, :]

                    kd_loss = F.kl_div(
                        F.log_softmax(step_logits / kd_temp, dim=-1),
                        F.softmax(lm_step_logits / kd_temp, dim=-1),
                        reduction="batchmean",
                    ) * (kd_temp ** 2)

                    decay = 0.8 ** step
                    step_loss = decay * (ce_weight * ce_loss + kd_weight * kd_loss)
                    loss = loss + step_loss

            # Free large tensors before backward
            del draft_logits, hidden, lm_logits

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if n_batches <= 3 or n_batches % 100 == 0:
                print(f"  step {n_batches}: loss={loss.item():.4f}", flush=True)

        avg_loss = total_loss / max(n_batches, 1)
        print(f"[train_drafter] Epoch {epoch+1}/{args.epochs} — loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_drafter(drafter, args.output, args)
            print(f"[train_drafter] Saved best model (loss={best_loss:.4f})")

    print(f"[train_drafter] Training complete. Best loss: {best_loss:.4f}")
    print(f"[train_drafter] Model saved to: {args.output}")


def save_drafter(model: GruDrafter, output_path: str, args):
    """Save drafter weights in safetensors format for Rust loading.

    Weight name mapping (PyTorch → Rust/candle):
      hidden_proj.weight     → hidden_proj.weight
      token_emb.weight       → token_emb.weight
      gru_cells.{i}.w_z.weight → gru.{i}.w_z.weight
      gru_cells.{i}.u_z.weight → gru.{i}.u_z.weight
      (same for w_r, u_r, w_h, u_h)
      output_head.weight     → output_head.weight
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("[train_drafter] safetensors not installed, saving as .pt")
        torch.save(model.state_dict(), output_path.replace(".safetensors", ".pt"))
        return

    state = {}
    for name, param in model.named_parameters():
        # Rename gru_cells.{i}.* → gru.{i}.*
        key = name.replace("gru_cells.", "gru.")
        state[key] = param.data.half()  # save as FP16 for Metal

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file(state, output_path)

    # Save config alongside weights
    config_path = output_path.replace(".safetensors", "_config.json")
    config = {
        "gru_hidden": args.gru_hidden,
        "gru_layers": args.gru_layers,
        "tree_width": args.tree_width,
        "tree_depth": args.tree_depth,
        "emb_dim": args.emb_dim,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[train_drafter] Config saved to: {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU drafter for speculative decoding")
    parser.add_argument("--base_model", required=True, help="Path to frozen Sonata LM weights")
    parser.add_argument("--data", required=True, help="Directory of semantic token data")
    parser.add_argument("--output", default="models/sonata/rnn_drafter.safetensors",
                        help="Output path for drafter weights")
    parser.add_argument("--gru_hidden", type=int, default=512)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--tree_width", type=int, default=4)
    parser.add_argument("--tree_depth", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--kd_weight", type=float, default=0.7,
                        help="Knowledge distillation weight (vs cross-entropy)")
    parser.add_argument("--kd_temperature", type=float, default=2.0,
                        help="Temperature for KD softmax")
    args = parser.parse_args()
    train_drafter(args)
