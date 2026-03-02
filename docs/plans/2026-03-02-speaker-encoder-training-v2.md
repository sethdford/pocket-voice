# Speaker Encoder Training v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace collapsed GE2E speaker encoder training with AAM-Softmax, add augmentation + EER validation, deploy to GCE, and produce a production ECAPA-TDNN model.

**Architecture:** ECAPA-TDNN (24.6M params, 256-dim embeddings) trained with AAM-Softmax loss on LibriTTS-R (~2300 speakers). Validated by EER on test-clean. Exports to safetensors + ONNX for Rust/C inference in pocket-voice.

**Tech Stack:** Python 3.11, PyTorch 2.x, torchaudio, safetensors, ONNX. GCE g2-standard-8 (L4 GPU).

**Repo:** `/Users/sethford/Documents/pocket-voice` (https://github.com/sethdford/pocket-voice)

**GCE Instance:** `sonata-train-speaker-encoder` (us-central1-a, project johnb-2025)

---

## Task 1: Stop broken training + kill zombie processes on GCE

**Context:** The current training has collapsed (loss ~0, epoch 30/100). Four extra training processes were also spawned. We need a clean slate.

**Step 1: Kill all training processes**

```bash
gcloud compute ssh sonata-train-speaker-encoder --zone=us-central1-a --project=johnb-2025 \
  --command="pkill -f train_speaker_encoder.py; sleep 2; ps aux | grep train_speaker | grep -v grep || echo 'All killed'"
```

Expected: "All killed"

**Step 2: Back up the best checkpoint (epoch 8) to GCS**

```bash
gcloud compute ssh sonata-train-speaker-encoder --zone=us-central1-a --project=johnb-2025 \
  --command="gsutil cp /opt/sonata/train/checkpoints/speaker_encoder/speaker_encoder_epoch8.pt \
    gs://sonata-training-johnb-2025/checkpoints/speaker_encoder/v1_epoch8_ge2e.pt"
```

Expected: file uploaded to GCS

**Step 3: Commit**

No code changes — this is an operational step.

---

## Task 2: Create `train/requirements.txt`

**Files:**
- Create: `train/requirements.txt`

**Step 1: Write requirements**

```
torch>=2.1.0
torchaudio>=2.1.0
safetensors>=0.4.0
numpy>=1.24.0
tqdm>=4.65.0
onnx>=1.15.0
onnxruntime>=1.17.0
```

**Step 2: Commit**

```bash
cd /Users/sethford/Documents/pocket-voice
git add train/requirements.txt
git commit -m "feat(train): add training requirements"
```

---

## Task 3: Write `train/train_speaker_encoder.py` — Model + Loss

This is the core file. We reuse the ECAPA-TDNN architecture from the broken script and replace GE2E with AAM-Softmax.

**Files:**
- Create: `train/train_speaker_encoder.py`

**Step 1: Write the full training script**

The script has these sections (in order):

1. **Imports + CLI args** — argparse with all hyperparameters
2. **MelSpectrogramExtractor** — identical to v1 (80-mel, 512 FFT, 160 hop)
3. **Data Augmentation** — SpecAugment, speed perturbation, Gaussian noise
4. **ECAPA-TDNN** — identical architecture to v1 (BatchNorm1d, SEBlock, Res2NetBlock, SERes2NetBlock, AttentiveStatisticsPooling, EcapaTdnn)
5. **AAMSoftmax** — replaces GE2E. L2-normalize both weights and embeddings, add angular margin `m` to true class cosine, scale by `s`, cross-entropy
6. **LibriTTSRDataset** — same data loading but with augmentation toggle
7. **EER computation** — extract embeddings, compute cosine similarity pairs, sweep threshold
8. **train_epoch()** — standard training loop with gradient clipping
9. **validate_eer()** — compute EER on test set every N epochs
10. **main()** — orchestrates everything: dataset, model, optimizer, training loop, checkpointing, early stopping, export

Key differences from v1:
- `AAMSoftmax` class with `forward(embeddings, labels)` → cross-entropy loss
- `SpecAugment` class that masks freq/time bands on mel spectrograms
- `speed_perturb()` function using `torchaudio.functional.speed`
- `compute_eer()` function for validation
- Checkpoint saves `classifier.state_dict()` alongside model
- Logging at `:.6f` precision with EER, LR, grad norm
- Early stopping with patience counter
- ONNX export function at end of training

**AAMSoftmax implementation:**

```python
class AAMSoftmax(nn.Module):
    def __init__(self, embedding_dim: int, num_speakers: int, scale: float = 30.0, margin: float = 0.2):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_speakers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # L2 normalize both
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight)  # [B, num_speakers]
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Add angular margin to target class
        theta = torch.acos(cosine)
        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
        target_cosine = torch.cos(theta + self.margin * one_hot)

        # Scale and cross-entropy
        logits = self.scale * target_cosine
        return F.cross_entropy(logits, labels)
```

**SpecAugment implementation:**

```python
class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param: int = 10, time_mask_param: int = 50,
                 n_freq_masks: int = 2, n_time_masks: int = 2):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_freq_masks):
            mel = self.freq_mask(mel)
        for _ in range(self.n_time_masks):
            mel = self.time_mask(mel)
        return mel
```

**EER computation:**

```python
def compute_eer(embeddings: np.ndarray, labels: np.ndarray, n_pairs: int = 50000) -> float:
    """Compute Equal Error Rate from embeddings."""
    unique_labels = np.unique(labels)
    label_to_indices = {l: np.where(labels == l)[0] for l in unique_labels}

    scores, targets = [], []
    for _ in range(n_pairs):
        # Positive pair
        spk = np.random.choice(unique_labels)
        if len(label_to_indices[spk]) >= 2:
            i, j = np.random.choice(label_to_indices[spk], 2, replace=False)
            scores.append(np.dot(embeddings[i], embeddings[j]))
            targets.append(1)

        # Negative pair
        spk1, spk2 = np.random.choice(unique_labels, 2, replace=False)
        i = np.random.choice(label_to_indices[spk1])
        j = np.random.choice(label_to_indices[spk2])
        scores.append(np.dot(embeddings[i], embeddings[j]))
        targets.append(0)

    scores = np.array(scores)
    targets = np.array(targets)

    # Sweep thresholds
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    best_eer = 1.0
    for t in thresholds:
        far = np.mean(scores[targets == 0] >= t)
        frr = np.mean(scores[targets == 1] < t)
        eer = (far + frr) / 2
        if abs(far - frr) < abs(best_eer - 0.5):
            best_eer = eer
        if far <= frr:
            best_eer = (far + frr) / 2
            break

    return best_eer
```

**Step 2: Verify it parses without errors**

```bash
cd /Users/sethford/Documents/pocket-voice
python3 -c "import ast; ast.parse(open('train/train_speaker_encoder.py').read()); print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add train/train_speaker_encoder.py
git commit -m "feat(train): speaker encoder v2 with AAM-Softmax

Replace collapsed GE2E training with AAM-Softmax loss.
Add SpecAugment, speed perturbation, noise injection.
Add EER validation, proper checkpointing, early stopping.
Add ONNX export for pocket-voice C inference."
```

---

## Task 4: Write `scripts/export_speaker_encoder.py` — ONNX + safetensors export

**Files:**
- Create: `scripts/export_speaker_encoder.py`

**Step 1: Write the export script**

This script:
1. Loads a `.pt` checkpoint (AAM-Softmax training format)
2. Strips the AAM-Softmax classifier head
3. Exports the ECAPA-TDNN encoder to:
   - `.safetensors` (for Rust inference in voiceai)
   - `.onnx` (for pocket-voice C inference via ONNX Runtime)
4. Validates the ONNX model produces correct-shape output

**Key function — ONNX export:**

```python
def export_to_onnx(model, mel_extractor, output_path, embedding_dim=256):
    """Export encoder to ONNX with fbank input [1, n_frames, 80]."""
    model.eval()
    # ONNX expects fbank input (what the C code computes), not raw audio
    dummy_fbank = torch.randn(1, 100, 80)  # [batch, frames, mels]
    # Transpose to [batch, mels, frames] for ECAPA-TDNN
    dummy_input = dummy_fbank.transpose(1, 2)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['fbank'],
        output_names=['embedding'],
        dynamic_axes={'fbank': {2: 'time'}, 'embedding': {}},
        opset_version=17,
    )
```

**Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('scripts/export_speaker_encoder.py').read()); print('OK')"
```

**Step 3: Commit**

```bash
git add scripts/export_speaker_encoder.py
git commit -m "feat(scripts): add speaker encoder ONNX + safetensors export"
```

---

## Task 5: Write `train/README.md`

**Files:**
- Create: `train/README.md`

**Step 1: Write the README**

Content covers:
- Prerequisites (Python 3.11+, CUDA GPU)
- Installing requirements
- Dataset preparation (download LibriTTS-R, expected directory layout)
- Training command with recommended flags
- Monitoring (what to look for in logs)
- Export command after training completes
- Hyperparameter reference table

**Step 2: Commit**

```bash
git add train/README.md
git commit -m "docs(train): add speaker encoder training README"
```

---

## Task 6: Push to GitHub and deploy to GCE

**Step 1: Push all commits**

```bash
cd /Users/sethford/Documents/pocket-voice
git push origin main
```

**Step 2: Clone pocket-voice on GCE**

```bash
gcloud compute ssh sonata-train-speaker-encoder --zone=us-central1-a --project=johnb-2025 \
  --command="cd /opt/sonata/train && git clone https://github.com/sethdford/pocket-voice.git pocket-voice 2>&1 || (cd pocket-voice && git pull)"
```

**Step 3: Install training requirements on GCE**

```bash
gcloud compute ssh sonata-train-speaker-encoder --zone=us-central1-a --project=johnb-2025 \
  --command="pip3 install -r /opt/sonata/train/pocket-voice/train/requirements.txt"
```

**Step 4: Download test data (LibriTTS-R test-clean)**

LibriTTS-R test splits are publicly available from OpenSLR:

```bash
gcloud compute ssh sonata-train-speaker-encoder --zone=us-central1-a --project=johnb-2025 \
  --command="cd /opt/sonata/train/data-local/libritts-r && \
    wget -q https://www.openslr.org/resources/141/test_clean.tar.gz && \
    tar xzf test_clean.tar.gz && rm test_clean.tar.gz && \
    echo 'test-clean downloaded' && \
    ls LibriTTS_R/test-clean/ | wc -l"
```

Expected: `test-clean downloaded` + number of speaker dirs

**Step 5: Commit** — no code change, operational step

---

## Task 7: Start training on GCE and verify it's working

**Step 1: Launch training in background**

```bash
gcloud compute ssh sonata-train-speaker-encoder --zone=us-central1-a --project=johnb-2025 \
  --command="cd /opt/sonata/train/pocket-voice/train && \
    nohup python3 -u train_speaker_encoder.py \
      --data-dir /opt/sonata/train/data-local/libritts-r/LibriTTS_R \
      --output-dir /opt/sonata/train/checkpoints/speaker_encoder_v2 \
      --device cuda \
      --batch-size 64 \
      --n-epochs 40 \
      --lr 0.001 \
      --scale 30.0 \
      --margin 0.2 \
      --warmup-epochs 2 \
      --val-every 2 \
      --patience 10 \
      > /var/log/sonata-training-v2.log 2>&1 &
    echo 'Training started'
    sleep 5
    tail -20 /var/log/sonata-training-v2.log"
```

Expected: Log output showing dataset indexing, model creation, epoch 1 starting with non-zero loss

**Step 2: Verify GPU is being used**

```bash
gcloud compute ssh sonata-train-speaker-encoder --zone=us-central1-a --project=johnb-2025 \
  --command="nvidia-smi | grep python3"
```

Expected: Python process using GPU memory

**Step 3: Check first epoch completes with healthy metrics**

Wait ~40 minutes for epoch 1, then:

```bash
gcloud compute ssh sonata-train-speaker-encoder --zone=us-central1-a --project=johnb-2025 \
  --command="grep -E '(Epoch|Train loss|Val|EER|LR)' /var/log/sonata-training-v2.log | tail -20"
```

Expected:
- Train loss between 1.0 and 10.0 (healthy for AAM-Softmax epoch 1)
- LR shows warmup progression
- No NaN or Inf values

---

## Task 8: Set up GCS checkpoint sync watchdog

**Step 1: Create a watchdog script on GCE**

```bash
gcloud compute ssh sonata-train-speaker-encoder --zone=us-central1-a --project=johnb-2025 \
  --command="cat > /opt/sonata/train/sync_checkpoints_v2.sh << 'WATCHDOG'
#!/bin/bash
# Sync speaker encoder v2 checkpoints to GCS every 30 minutes
CKPT_DIR=/opt/sonata/train/checkpoints/speaker_encoder_v2
GCS_DIR=gs://sonata-training-johnb-2025/checkpoints/speaker_encoder_v2

while true; do
    if ls \$CKPT_DIR/*.pt 1>/dev/null 2>&1; then
        gsutil -m -q rsync \$CKPT_DIR/ \$GCS_DIR/
        echo \"[\$(date)] Synced checkpoints to GCS\"
    fi
    sleep 1800
done
WATCHDOG
chmod +x /opt/sonata/train/sync_checkpoints_v2.sh
nohup /opt/sonata/train/sync_checkpoints_v2.sh > /var/log/sync-ckpts-v2.log 2>&1 &
echo 'Watchdog started'"
```

---

## Summary

| Task | What | Depends On |
|------|------|-----------|
| 1 | Stop broken training, back up epoch 8 | — |
| 2 | Create `train/requirements.txt` | — |
| 3 | Write `train/train_speaker_encoder.py` (core) | — |
| 4 | Write `scripts/export_speaker_encoder.py` | 3 |
| 5 | Write `train/README.md` | 3 |
| 6 | Push to GitHub, deploy to GCE, sync test data | 1, 2, 3, 4, 5 |
| 7 | Start training, verify healthy metrics | 6 |
| 8 | Set up checkpoint sync watchdog | 7 |

**Parallelizable:** Tasks 1-5 can all run in parallel (1 is GCE ops, 2-5 are local code).
Task 6 waits for all. Tasks 7-8 are sequential after 6.

**Estimated training time:** ~40 epochs × ~40 min/epoch = ~27 hours (~$19 on g2-standard-8).

**Success criteria:** Validation EER < 3% on test-clean. Successful ONNX export that loads in pocket-voice C inference.
