# VAP & Intent Router Training — Production Upgrade Design

## Overview

Upgrade `train_vap.py` and `train_intent_router.py` from minimal training scripts to production quality, matching the conventions of `train_flow_v3.py` and `train_codec.py`. Add a new `prepare_vap_data.py` for Fisher/Switchboard/CallHome corpus ingestion.

## Files

| File                                  | Action                                                            |
| ------------------------------------- | ----------------------------------------------------------------- |
| `train/sonata/train_vap.py`           | Rewrite training loop, add val, LR schedule, resume, augmentation |
| `train/sonata/train_intent_router.py` | Rewrite training loop, expand synthetic data, add val, resume     |
| `train/sonata/prepare_vap_data.py`    | New — download + convert corpora to vap_manifest.jsonl            |

## VAP Training (`train_vap.py`)

### Architecture (unchanged)

- Input [T, 160] → Linear → SinPE → 4-layer causal Transformer → 4 sigmoid heads
- Matches C inference (vap_model.c) exactly
- Binary export to .vap format (unchanged)

### Training Loop (upgraded)

- Train/val split: 90/10
- Loss: BCEWithLogitsLoss with per-head class weights (backchannel ~5% of frames)
- LR: cosine decay 1e-3 → 1e-5, 500 step warmup
- Gradient clipping: norm 1.0
- TrainingLog → losses.jsonl
- Checkpoint: save every N steps, best by mean val AUC
- Resume: --resume loads model + optimizer + step
- Val metrics: AUC-ROC, accuracy, F1 per head

### Data Augmentation

- Speed perturbation (0.9-1.1x)
- Additive noise (SNR 5-20dB)
- Channel swap (10% probability)
- Random segment cropping

## Data Pipeline (`prepare_vap_data.py`)

### Supported Corpora

- Fisher Corpus: .stp timestamps + SPH audio → stereo WAV + JSONL
- Switchboard: ISIP/MS-State transcripts → JSONL
- CallHome: .cha CHAT format → JSONL

### Output Format

Unified vap_manifest.jsonl:

```json
{
  "audio_path": "conv001.wav",
  "channels": 2,
  "annotations": [
    { "start": 0.0, "end": 2.5, "speaker": "A", "type": "speech" },
    { "start": 1.8, "end": 2.0, "speaker": "B", "type": "backchannel" },
    { "start": 2.8, "end": 5.1, "speaker": "B", "type": "speech" }
  ]
}
```

## Intent Router Training (`train_intent_router.py`)

### Architecture (unchanged)

- Input(20) → Hidden(128, ReLU) → Hidden(64, ReLU) → Output(4, softmax)
- Matches C implementation exactly
- Binary export to .router format (unchanged)

### Training Loop (upgraded)

- Train/val split: 80/20
- LR: cosine decay, 100 step warmup
- TrainingLog → losses.jsonl
- Checkpoint: save per epoch, best by val accuracy
- Resume: --resume
- Val metrics: accuracy, precision, recall, F1 per route + confusion matrix

### Synthetic Data (expanded)

- 200+ templates (up from 17) covering edge cases per route
- Augmentation: word dropout, case variation, synonym replacement
- Class-balanced sampling
