# S2S Codec RVQ Upgrade — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade codec_12hz training from FSQ single-codebook to 8-codebook RVQ (2048 entries each) with WavLM semantic distillation on the first codebook. This codec is the foundation for the Talker S2S model.

**Architecture:** Mimi-Sonata hybrid codec: ConvNet encoder + 8-layer Transformer bottleneck + 8-level RVQ + WavLM distillation + ConvNet decoder. 12.5Hz frame rate, 24kHz audio.

**Tech Stack:** Python, PyTorch, WavLM (HuggingFace), existing train infrastructure on GCE

---

### Task 1: Add RVQ module to codec training

**Files:**

- Create: `train/sonata/rvq_module.py`
- Test: `train/sonata/test_rvq_module.py`

**Step 1: Write test**

```python
# test_rvq_module.py
import torch
from rvq_module import ResidualVQ

def test_rvq_output_shape():
    rvq = ResidualVQ(input_dim=512, n_codebooks=8, codebook_size=2048, codebook_dim=128)
    z = torch.randn(2, 512, 50)  # (batch, channels, time)
    codes, quantized, commit_loss = rvq(z)
    assert codes.shape == (2, 8, 50), f"Expected (2,8,50), got {codes.shape}"
    assert quantized.shape == z.shape
    assert commit_loss.item() >= 0

def test_rvq_codebook_usage():
    rvq = ResidualVQ(input_dim=512, n_codebooks=8, codebook_size=2048, codebook_dim=128)
    z = torch.randn(4, 512, 100)
    codes, _, _ = rvq(z)
    # All codes should be valid indices
    assert (codes >= 0).all() and (codes < 2048).all()

def test_rvq_decode():
    rvq = ResidualVQ(input_dim=512, n_codebooks=8, codebook_size=2048, codebook_dim=128)
    z = torch.randn(1, 512, 10)
    codes, _, _ = rvq(z)
    z_hat = rvq.decode(codes)
    assert z_hat.shape == z.shape

if __name__ == "__main__":
    test_rvq_output_shape()
    test_rvq_codebook_usage()
    test_rvq_decode()
    print("All RVQ tests passed!")
```

**Step 2: Implement ResidualVQ**

```python
# rvq_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        self.codebook_size = codebook_size

    def forward(self, z):
        # z: (batch, dim, time) -> (batch, time, dim)
        z_t = z.permute(0, 2, 1)
        # L2 distance: ||z - e||^2
        d = (z_t.pow(2).sum(-1, keepdim=True)
             + self.codebook.weight.pow(2).sum(-1)
             - 2 * z_t @ self.codebook.weight.t())
        codes = d.argmin(-1)  # (batch, time)
        quantized = self.codebook(codes).permute(0, 2, 1)  # (batch, dim, time)
        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()
        commit_loss = F.mse_loss(quantized.detach(), z)
        return codes, quantized_st, commit_loss

    def decode(self, codes):
        return self.codebook(codes).permute(0, 2, 1)


class ResidualVQ(nn.Module):
    def __init__(self, input_dim: int, n_codebooks: int = 8,
                 codebook_size: int = 2048, codebook_dim: int = 128):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.project_in = nn.Linear(input_dim, codebook_dim) if input_dim != codebook_dim else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, input_dim) if input_dim != codebook_dim else nn.Identity()
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, codebook_dim)
            for _ in range(n_codebooks)
        ])

    def forward(self, z):
        # z: (batch, channels, time)
        # Project to codebook dim
        z_t = z.permute(0, 2, 1)  # (batch, time, channels)
        z_proj = self.project_in(z_t).permute(0, 2, 1)  # (batch, codebook_dim, time)

        residual = z_proj
        all_codes = []
        total_commit = 0.0
        quantized_sum = torch.zeros_like(z_proj)

        for vq in self.quantizers:
            codes, quantized, commit = vq(residual)
            all_codes.append(codes.unsqueeze(1))
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized.detach()
            total_commit = total_commit + commit

        codes = torch.cat(all_codes, dim=1)  # (batch, n_codebooks, time)

        # Project back to input dim
        out_t = quantized_sum.permute(0, 2, 1)  # (batch, time, codebook_dim)
        out = self.project_out(out_t).permute(0, 2, 1)  # (batch, channels, time)

        return codes, out, total_commit / self.n_codebooks

    def decode(self, codes):
        # codes: (batch, n_codebooks, time)
        quantized_sum = None
        for i, vq in enumerate(self.quantizers):
            book_codes = codes[:, i, :]  # (batch, time)
            q = vq.decode(book_codes)
            quantized_sum = q if quantized_sum is None else quantized_sum + q
        out_t = quantized_sum.permute(0, 2, 1)
        return self.project_out(out_t).permute(0, 2, 1)
```

**Step 3: Run tests**

Run: `cd train/sonata && python test_rvq_module.py`
Expected: "All RVQ tests passed!"

**Step 4: Commit**

```bash
git add train/sonata/rvq_module.py train/sonata/test_rvq_module.py
git commit -m "feat: RVQ module with 8 codebooks x 2048 for S2S codec"
```

---

### Task 2: WavLM semantic distillation loss

**Files:**

- Create: `train/sonata/wavlm_distill.py`

**Step 1: Implement WavLM feature extractor + distillation loss**

```python
# wavlm_distill.py
"""WavLM semantic distillation for first RVQ codebook.

The first codebook should capture semantic content (like Mimi codec).
We distill from WavLM-Large hidden states using contrastive + MSE loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WavLMDistillLoss(nn.Module):
    """Distillation loss between codec first-codebook embeddings and WavLM features.

    Uses a projection head to map codebook embeddings to WavLM feature space,
    then applies MSE + contrastive loss.
    """
    def __init__(self, codebook_dim: int = 128, wavlm_dim: int = 1024,
                 temperature: float = 0.07, mse_weight: float = 1.0,
                 contrastive_weight: float = 0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(codebook_dim, wavlm_dim),
            nn.GELU(),
            nn.Linear(wavlm_dim, wavlm_dim),
        )
        self.temperature = temperature
        self.mse_weight = mse_weight
        self.contrastive_weight = contrastive_weight

    def forward(self, codec_emb, wavlm_features):
        """
        Args:
            codec_emb: (batch, codebook_dim, time) — first codebook embeddings
            wavlm_features: (batch, wavlm_dim, time) — WavLM hidden states (downsampled to codec rate)
        Returns:
            loss: scalar
        """
        # Project codec embeddings to WavLM space
        codec_proj = self.proj(codec_emb.permute(0, 2, 1))  # (batch, time, wavlm_dim)
        target = wavlm_features.permute(0, 2, 1)  # (batch, time, wavlm_dim)

        # MSE loss
        mse = F.mse_loss(codec_proj, target.detach())

        # Contrastive loss (InfoNCE)
        # Normalize
        codec_norm = F.normalize(codec_proj.reshape(-1, codec_proj.shape[-1]), dim=-1)
        target_norm = F.normalize(target.reshape(-1, target.shape[-1]), dim=-1)

        # Similarity matrix
        logits = codec_norm @ target_norm.t() / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        contrastive = F.cross_entropy(logits, labels)

        return self.mse_weight * mse + self.contrastive_weight * contrastive
```

**Step 2: Commit**

```bash
git add train/sonata/wavlm_distill.py
git commit -m "feat: WavLM distillation loss for semantic codebook (MSE + InfoNCE)"
```

---

### Task 3: Modify codec training script

**Files:**

- Modify: `train/sonata/codec_12hz.py`
- Create: `train/sonata/train_codec_rvq.py`

Create a new training script `train_codec_rvq.py` that extends codec_12hz with:

1. 8-level RVQ (replacing FSQ)
2. WavLM distillation on first codebook
3. Transformer bottleneck (8 layers, 512 dim)
4. Same ConvNet encoder/decoder structure

This is a training script — needs WavLM model loading, RVQ integration, combined loss function (reconstruction + commitment + WavLM distillation + adversarial).

**The training script should support:**

- `--wavlm_model` path to WavLM-Large weights
- `--n_codebooks 8`
- `--codebook_size 2048`
- `--distill_weight 0.1` for WavLM distillation loss
- Resume from checkpoint
- GCS checkpoint sync (following existing patterns in train_wrapper.sh)

**Step 1: Write the training script skeleton**

Follow the patterns from `train_vocoder.py` and `codec_12hz.py` for:

- Argument parsing
- DataLoader setup
- Training loop with AMP
- Checkpoint saving + resume
- Logging

**Step 2: Commit**

```bash
git add train/sonata/train_codec_rvq.py
git commit -m "feat: codec RVQ training with WavLM semantic distillation"
```

---

### Task 4: Update train_wrapper.sh for codec_rvq

**Files:**

- Modify: `train/gce/train_wrapper.sh`

Add a `codec_rvq` job type that:

1. Downloads WavLM-Large from HuggingFace
2. Runs `train_codec_rvq.py` with appropriate args
3. Syncs checkpoints to GCS

**Step 1: Add codec_rvq case to wrapper**

```bash
codec_rvq)
    echo "[train_wrapper] Starting Codec RVQ training..."
    # Download WavLM-Large if not present
    if [ ! -f "$CKPT_DIR/wavlm_large.pt" ]; then
        python3 -c "
from transformers import WavLMModel
model = WavLMModel.from_pretrained('microsoft/wavlm-large')
import torch; torch.save(model.state_dict(), '$CKPT_DIR/wavlm_large.pt')
"
    fi
    python3 -u train_codec_rvq.py \
        --manifest "$DATA_DIR/libritts_r_full_manifest.jsonl" \
        --wavlm_model "$CKPT_DIR/wavlm_large.pt" \
        --output-dir "$CKPT_DIR/codec_rvq" \
        --n_codebooks 8 --codebook_size 2048 \
        --distill_weight 0.1 \
        --device cuda --epochs 100 \
        --batch-size 8 --grad-accum 4 --amp \
        --save-steps 5000 --log-interval 50
    ;;
```

**Step 2: Commit**

```bash
git add train/gce/train_wrapper.sh
git commit -m "feat: add codec_rvq job type to train_wrapper.sh"
```

---

## Summary

| Task | Component          | Files                | Notes              |
| ---- | ------------------ | -------------------- | ------------------ |
| 1    | RVQ module         | rvq_module.py + test | 8 codebooks x 2048 |
| 2    | WavLM distillation | wavlm_distill.py     | MSE + InfoNCE loss |
| 3    | Training script    | train_codec_rvq.py   | Full training loop |
| 4    | GCE wrapper        | train_wrapper.sh     | Deploy to GPU      |

Depends on: WavLM-Large weights from HuggingFace, LibriTTS-R dataset.
