# Speaker Encoder Training v2 — AAM-Softmax

**Date:** 2026-03-02
**Status:** Approved
**Goal:** Fix collapsed speaker encoder training and produce a production-quality ECAPA-TDNN model for zero-shot voice cloning and speaker verification.

## Problem

The current training run on GCE (`sonata-train-speaker-encoder`) has collapsed:

- **Train loss hit 3.8e-08 by epoch 8** — model memorized the training set
- **No validation set** — test-clean/test-other never synced to VM
- **GE2E criterion state not saved** in checkpoints (learnable w/b missing)
- **No data augmentation** — model overfits trivially
- **Wasting GPU** — epochs 9-100 produce zero learning

Root causes: GE2E loss with random batches (not speaker-balanced), oversized model for dataset, no regularization.

## Design

### Loss: AAM-Softmax (ArcFace)

Replace GE2E with Additive Angular Margin Softmax:

```
embedding (256-dim, L2-norm) -> W (256 x num_speakers, L2-norm columns)
-> cos(theta) = embedding . W_j
-> cos(theta_yi + m) for true speaker yi (margin m=0.2, scale s=30)
-> cross_entropy(s * modified_cosine, labels)
```

Advantages over GE2E:
- No learnable w/b that can drift
- No speaker-balanced batching required
- Clear training signal from angular margin (prevents collapse)
- State-of-the-art on VoxCeleb benchmarks

### Architecture (Unchanged)

ECAPA-TDNN: 80-mel input, 4 SE-Res2Net blocks (1024 channels), attentive statistics pooling, 256-dim L2-normalized embeddings.

### Data Pipeline

| Component | Details |
|-----------|---------|
| Training data | LibriTTS-R: train-clean-100 + train-clean-360 + train-other-500 (~2300 speakers) |
| Validation data | LibriTTS-R: test-clean + test-other (sync to GCE) |
| SpecAugment | F=10, 2 freq masks; T=50, 2 time masks |
| Speed perturbation | 0.9x, 1.0x, 1.1x (triples effective data) |
| Noise injection | Gaussian, SNR 15-40dB, 20% probability |

### Training Config

| Parameter | Value |
|-----------|-------|
| Loss | AAM-Softmax (s=30, m=0.2) |
| LR | 0.001 with linear warmup (2 epochs) |
| Scheduler | CosineAnnealingLR |
| Batch size | 64 |
| Epochs | 40 |
| Validation | EER on test-clean every 2 epochs |
| Checkpointing | Best validation EER |
| Early stopping | Patience 10 epochs on val EER |
| Target EER | < 3% on test-clean |

### Checkpoint Format

```python
torch.save({
    'model': model.state_dict(),
    'classifier': classifier.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'epoch': epoch,
    'best_eer': best_eer,
    'num_speakers': num_speakers,
    'config': model_config,
})
```

### EER Computation

1. Extract embeddings for all test utterances
2. Cosine similarity for positive pairs (same speaker) and sampled negative pairs
3. Sweep threshold to find EER (FAR = FRR)

### Export Pipeline

After training:
1. `.pt` -> `.safetensors` (Rust inference in voiceai)
2. `.pt` -> `.onnx` (pocket-voice C inference)
3. Strip AAM-Softmax head — only export encoder

### File Layout

```
pocket-voice/
  train/
    train_speaker_encoder.py   # Fixed training script
    requirements.txt           # torch, torchaudio, safetensors
    README.md                  # How to run training
  scripts/
    export_trained_models.py   # Existing export script
  src/
    speaker_encoder.c          # Existing C inference
```

### Deployment Steps

1. Stop current broken training on GCE
2. Add `train/` to pocket-voice repo, push to GitHub
3. Clone pocket-voice on GCE
4. Sync test-clean + test-other from GCS
5. Start training with new script
6. Monitor until EER < 3%
7. Export to ONNX + safetensors
