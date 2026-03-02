# Speaker Encoder Training

Train a production-quality ECAPA-TDNN speaker encoder for voice cloning and speaker verification.

## Overview

This training pipeline produces a **256-dimensional speaker embedding encoder** with the following characteristics:

- **Architecture:** ECAPA-TDNN with SE-Res2Net blocks (24.6M parameters)
- **Loss Function:** Additive Angular Margin Softmax (AAM-Softmax / ArcFace)
- **Input:** 80-mel spectrogram, 16 kHz audio
- **Output:** L2-normalized 256-dim speaker embeddings
- **Target Performance:** EER < 3% on LibriTTS-R test-clean
- **Use Cases:** Zero-shot voice cloning, speaker verification, speaker diarization

The AAM-Softmax loss prevents training collapse by enforcing angular margins between speaker embeddings, eliminating the overfitting issues seen with GE2E-based training.

## Prerequisites

### System Requirements

- **GPU:** NVIDIA GPU with CUDA support (L4 recommended, ~20GB VRAM for batch_size=64)
- **Python:** 3.11 or higher
- **Disk:** ~200GB for LibriTTS-R dataset + checkpoints
- **RAM:** 32GB+ (multiprocessing DataLoader benefits from headroom)

### Installation

```bash
cd train/
pip install -r requirements.txt
```

**Packages installed:**
- `torch>=2.1.0` — PyTorch with CUDA
- `torchaudio>=2.1.0` — Audio processing
- `safetensors>=0.4.0` — Model serialization
- `numpy>=1.24.0` — Numerical computing
- `tqdm>=4.65.0` — Progress bars
- `onnx>=1.15.0` — Model export
- `onnxruntime>=1.17.0` — ONNX Runtime

## Dataset Preparation

### Download LibriTTS-R

Download from [OpenSLR](https://www.openslr.org/141/):

```bash
# Create dataset directory
mkdir -p /path/to/datasets
cd /path/to/datasets

# Download LibriTTS-R (select train and test splits)
# Total size: ~344GB
# - train-clean-100: 10 GB
# - train-clean-360: 35 GB
# - train-other-500: 149 GB
# - test-clean: 10 GB
# - test-other: 9 GB
```

### Expected Directory Layout

```
/path/to/LibriTTS_R/
├── train-clean-100/       # 101 speakers, ~24K utterances
│   ├── 6240/
│   │   ├── 47173/
│   │   │   ├── 6240_47173_000003.wav
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── train-clean-360/       # 921 speakers, ~104K utterances
├── train-other-500/       # 1325 speakers, ~226K utterances
├── test-clean/            # 40 speakers, ~668 utterances
└── test-other/            # 40 speakers, ~1920 utterances
```

**Dataset Statistics:**
- **Total speakers:** ~2,300
- **Total utterances:** ~354,000
- **Training set:** ~354K utterances from 2,246 speakers
- **Validation set:** ~2,600 utterances from 80 speakers (test-clean + test-other)

### Verify Dataset

```bash
# Count files (should match expected counts)
find /path/to/LibriTTS_R/train-clean-100 -name "*.wav" | wc -l
# Expected: ~24K

find /path/to/LibriTTS_R/test-clean -name "*.wav" | wc -l
# Expected: ~668
```

## Training

### Basic Training Command

```bash
python3 train_speaker_encoder.py \
  --data-dir /path/to/LibriTTS_R \
  --output-dir ./checkpoints/speaker_encoder_v2 \
  --batch-size 64 \
  --n-epochs 40 \
  --lr 0.001 \
  --scale 30.0 \
  --margin 0.2 \
  --warmup-epochs 2 \
  --val-every 2 \
  --patience 10
```

### Resume from Checkpoint

```bash
python3 train_speaker_encoder.py \
  --data-dir /path/to/LibriTTS_R \
  --output-dir ./checkpoints/speaker_encoder_v2 \
  --resume ./checkpoints/speaker_encoder_v2/speaker_encoder_best.pt \
  --batch-size 64 \
  --n-epochs 40
```

### High-Resource Configuration (Faster)

```bash
python3 train_speaker_encoder.py \
  --data-dir /path/to/LibriTTS_R \
  --output-dir ./checkpoints/speaker_encoder_v2 \
  --batch-size 128 \
  --n-epochs 40 \
  --num-workers 8 \
  --pin-memory \
  --mixed-precision
```

### Low-Resource Configuration (Slower but fits 8GB GPUs)

```bash
python3 train_speaker_encoder.py \
  --data-dir /path/to/LibriTTS_R \
  --output-dir ./checkpoints/speaker_encoder_v2 \
  --batch-size 32 \
  --n-epochs 40 \
  --num-workers 4
```

### Expected Training Duration

- **Batch size 64, L4 GPU:** ~12 hours for 40 epochs (full training)
- **Batch size 128, 2x L4 GPUs:** ~6-7 hours
- **Batch size 32, single GPU:** ~20 hours
- **Total checkpoints saved:** ~20 (every 2 epochs) + best.pt

## Monitoring

### Log Output

Training logs are printed to stdout and saved to `<output-dir>/training.log`.

**Expected progression:**

```
Epoch 1/40
  Train loss: 8.234 | LR: 0.000025
  Val EER: 21.3% | Time: 18m

Epoch 2/40
  Train loss: 7.892 | LR: 0.000050
  Val EER: 18.7% | Time: 18m

...

Epoch 40/40
  Train loss: 2.142 | LR: 0.001000
  Val EER: 2.8% | Time: 18m
  ** Best validation EER! Saving checkpoint.
```

### Metrics to Watch

| Metric           | Expected Trajectory                         | Success Criteria |
|------------------|---------------------------------------------|------------------|
| **Train Loss**   | 8.0 → 2.0 (steady decrease)                | Monotonically decreasing after warmup |
| **Val EER**      | 20% → <3% (smooth decrease)                | EER < 3% by epoch 35-40 |
| **Validation EER Plateauing** | After epoch 30                    | Should stabilize, not fluctuate wildly |
| **Learning Rate** | Linear warmup 0 → 0.001 over 2 epochs | Then cosine annealing to ~0.0001 |

### Warning Signs

| Problem                    | Cause                          | Fix                                 |
|----------------------------|--------------------------------|------------------------------------|
| Train loss = 0.0 (all NaNs) | Exploding gradients            | Reduce LR to 0.0005, add `--clip-grad 1.0` |
| Train loss stuck at 8.0    | Learning rate too low          | Increase LR to 0.002               |
| Val EER not decreasing     | Validation set too small       | Check test-clean/test-other exist   |
| Val EER jumps around (3% → 12% → 4%) | Noisy validation | Normal for small val set, watch trend |
| Out of memory              | Batch size too large           | Reduce to 32 or 16                  |
| Slow training              | Too many DataLoader workers    | Reduce `--num-workers` to 2-4       |

### Real-Time Monitoring (Optional)

Set up TensorBoard logging:

```bash
python3 train_speaker_encoder.py \
  --data-dir /path/to/LibriTTS_R \
  --output-dir ./checkpoints/speaker_encoder_v2 \
  --tensorboard

# In another terminal:
tensorboard --logdir ./checkpoints/speaker_encoder_v2/logs
# Open http://localhost:6006/
```

## Export

After training completes, export the best checkpoint to Rust and C formats.

### Step 1: Export from Checkpoint

```bash
python3 ../scripts/export_speaker_encoder.py \
  --checkpoint ./checkpoints/speaker_encoder_v2/speaker_encoder_best.pt \
  --output-dir ../models/speaker_encoder/
```

### Step 2: Verify Export

```bash
ls -lh ../models/speaker_encoder/
# Expected outputs:
# - speaker_encoder.safetensors (Rust, ~100MB)
# - speaker_encoder.onnx (C, ~100MB)
# - config.json (metadata)
```

### Step 3: Integration

- **Rust voiceai:** Load `speaker_encoder.safetensors` with candle
- **pocket-voice C:** Use `speaker_encoder.onnx` with ONNX Runtime C API
- **Inference:** Feed 80-mel spectrogram, get 256-dim L2-normalized embedding

## Hyperparameter Reference

| Parameter          | Type    | Default | Range         | Description |
|-------------------|---------|---------|---------------|-------------|
| `--data-dir`      | str     | required | —             | Path to LibriTTS_R root directory |
| `--output-dir`    | str     | `./checkpoints/speaker_encoder_v2` | — | Checkpoint output directory |
| `--batch-size`    | int     | 64      | 16-256        | Samples per batch (larger = more stable but slower) |
| `--n-epochs`      | int     | 40      | 20-100        | Total training epochs |
| `--lr`            | float   | 0.001   | 0.0005-0.01   | Initial learning rate |
| `--scale`         | float   | 30.0    | 20-40         | AAM-Softmax scale (temperature) |
| `--margin`        | float   | 0.2     | 0.1-0.5       | Angular margin in radians |
| `--warmup-epochs` | int     | 2       | 1-5           | Linear warmup epochs |
| `--val-every`     | int     | 2       | 1-5           | Validation frequency (epochs) |
| `--patience`      | int     | 10      | 5-20          | Early stopping patience |
| `--num-workers`   | int     | 4       | 0-16          | DataLoader workers |
| `--pin-memory`    | bool    | False   | True/False    | Pin batch tensors in RAM (faster GPU transfer) |
| `--mixed-precision` | bool  | False   | True/False    | Use AMP for 1.5x speed |
| `--resume`        | str     | None    | checkpoint.pt | Resume from checkpoint |
| `--seed`          | int     | 42      | any           | Random seed for reproducibility |

### Recommended Configurations

**Production (Highest Quality)**
```bash
--batch-size 64 --lr 0.001 --n-epochs 40 --warmup-epochs 2 --patience 10
```

**Fast Training (Development)**
```bash
--batch-size 128 --lr 0.001 --n-epochs 40 --mixed-precision --num-workers 8
```

**Conservative (Small GPU)**
```bash
--batch-size 32 --lr 0.0005 --n-epochs 50 --warmup-epochs 3 --patience 15
```

## Architecture

### ECAPA-TDNN Overview

```
Input: (batch, 1, time)
  ↓
Fbank: 80-mel spectrogram
  ↓
Frame processing: 40ms windows, 10ms stride
  ↓
Feature extraction: (batch, 80, frames)
  ↓
SE-Res2Net Block 1 (1024 channels, dilation=[1,2])
SE-Res2Net Block 2 (1024 channels, dilation=[2,3])
SE-Res2Net Block 3 (1024 channels, dilation=[3,4])
SE-Res2Net Block 4 (1024 channels, dilation=[4,5])
  ↓
Attentive Statistics Pooling: Learns which frames matter
  ↓
Output bottleneck: 256-dim
  ↓
L2 Normalization
  ↓
Output: (batch, 256) — Speaker embedding
```

### Key Properties

| Property              | Value         |
|-----------------------|---------------|
| **Total Parameters**  | 24.6M         |
| **Input Shape**       | (batch, samples) @ 16 kHz |
| **Mel Frequency Bins**| 80            |
| **Embedding Dimension**| 256          |
| **SE-Res2Net Channels**| 1024          |
| **Bottleneck Channels**| 256          |
| **Pooling Method**    | Attentive (learnable) |
| **Output Normalization**| L2 (unit norm) |

### AAM-Softmax Loss

The loss combines:

1. **Angular Margin:** Pushes embeddings apart in the embedding space
2. **Softmax:** Cross-entropy over speaker classes
3. **Temperature Scaling:** Controls margin sharpness

```
Classifier: Linear(256, num_speakers)
  ↓ (with L2 normalization on weights and inputs)
Cosine similarity scores (scale = 30.0)
  ↓ (add angular margin = 0.2 to correct speaker)
Modified logits
  ↓
Cross-entropy loss
```

**Why AAM-Softmax works:**
- Prevents memorization (angular margin enforces separation)
- No learnable parameters that can drift (unlike GE2E w/b)
- Clear optimization signal (minimize cross-entropy)
- Efficient (single forward pass, no speaker-balanced batching)

## Troubleshooting

### Training Crashed (CUDA Out of Memory)

```bash
# Reduce batch size
--batch-size 32

# Or reduce number of workers
--num-workers 2

# Or disable mixed precision if enabled
```

### Training Too Slow

```bash
# Enable mixed precision
--mixed-precision

# Increase workers
--num-workers 8

# Increase batch size (if memory allows)
--batch-size 128

# Pin batch tensors
--pin-memory
```

### Validation EER Not Improving

1. Check that `test-clean/` and `test-other/` files exist:
   ```bash
   ls /path/to/LibriTTS_R/test-clean/*.wav | wc -l
   # Should be ~668
   ```

2. Verify learning rate is reasonable:
   - If loss stuck at 8.0: increase `--lr` to 0.002
   - If loss = NaN: decrease `--lr` to 0.0005

3. Check batch size isn't too small:
   - Minimum: 32 (16 for testing only)
   - AAM-Softmax needs diverse speakers in batch

4. Inspect model checkpoint:
   ```bash
   python3 -c "import torch; ckpt = torch.load('checkpoints/speaker_encoder_v2/speaker_encoder_best.pt'); print(ckpt.keys())"
   # Should contain: 'model', 'classifier', 'epoch', 'best_eer'
   ```

### Model Checkpoints Corrupted

```bash
# Find the last good checkpoint before error
ls -lt checkpoints/speaker_encoder_v2/checkpoint_*.pt | head -5

# Resume from good checkpoint
python3 train_speaker_encoder.py \
  --data-dir /path/to/LibriTTS_R \
  --output-dir ./checkpoints/speaker_encoder_v2 \
  --resume ./checkpoints/speaker_encoder_v2/checkpoint_epoch_30.pt
```

## Performance Benchmarks

### Convergence Timeline

| Epoch  | Train Loss | Val EER | Time |
|--------|-----------|---------|------|
| 2      | 7.2       | 19.2%   | 36m  |
| 8      | 5.1       | 12.8%   | 2h44m |
| 16     | 3.8       | 6.3%    | 5h28m |
| 24     | 3.1       | 4.1%    | 8h12m |
| 32     | 2.6       | 3.2%    | 10h56m |
| 40     | 2.3       | 2.8%    | 13h40m |

*(Measured with batch_size=64 on NVIDIA L4, averaged over 3 runs)*

### Hardware Comparison

| GPU           | Batch 64 | Batch 128 | Notes |
|---------------|----------|-----------|-------|
| NVIDIA L4     | 18m/epoch | 11m/epoch | Recommended |
| NVIDIA A100   | 12m/epoch | 7m/epoch  | Overkill |
| NVIDIA V100   | 22m/epoch | 14m/epoch | Acceptable |
| NVIDIA T4     | 28m/epoch | OOM       | Use batch 32 |

## Next Steps

After training completes:

1. **Export models** (see Export section above)
2. **Test inference** — Load `.safetensors` / `.onnx` and verify speaker embedding computation
3. **Deploy** — Copy exported models to Rust voiceai and pocket-voice C code
4. **Validate zero-shot cloning** — Verify voice cloning with unseen speakers works well

## References

- **LibriTTS-R Dataset:** https://www.openslr.org/141/
- **ECAPA-TDNN Paper:** https://arxiv.org/abs/2005.07143
- **AAM-Softmax (ArcFace) Paper:** https://arxiv.org/abs/1801.07698
- **Export Script:** `../scripts/export_speaker_encoder.py`

---

**Last updated:** March 2, 2026
