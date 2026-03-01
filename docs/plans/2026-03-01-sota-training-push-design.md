# SOTA Training Push — Design

**Date**: 2026-03-01
**Goal**: Train 3 additional models to close SOTA gaps: speaker encoder (zero-shot voice cloning), 12.5Hz codec (4x token reduction), and flow distillation (1-2 step generation).

## Compute Plan

Two L4 GPUs running in parallel on GCE (project: johnb-2025, zone: us-central1-a).

```
VM1 (spot, ~$0.22/hr) — already running:
  Flow v3 (step 80K→200K) ──────────────┐ ~4.6 days remaining
                                         ▼
                                    Flow Distillation (~1 day)
                                         ▼
                                    VM1 teardown

VM2 (on-demand, ~$0.70/hr) — new:
  Speaker Encoder (~12h) → 12.5Hz Codec (~3-4 days)
                                         ▼
                                    VM2 teardown
```

**Estimated total cost**: ~$96

| VM              | Job               | Duration  | Rate     | Cost |
| --------------- | ----------------- | --------- | -------- | ---- |
| VM1 (spot)      | Flow v3 remaining | ~4.6 days | $0.22/hr | ~$24 |
| VM1 (spot)      | Flow distillation | ~1 day    | $0.22/hr | ~$5  |
| VM2 (on-demand) | Speaker encoder   | ~12h      | $0.70/hr | ~$8  |
| VM2 (on-demand) | 12.5Hz codec      | ~3.5 days | $0.70/hr | ~$59 |

## Workstream 1: Speaker Encoder (VM2)

**Model**: ECAPA-TDNN, ~6M params, 256-dim d-vector output
**Loss**: GE2E (Generalized End-to-End) contrastive
**Data**: LibriTTS-R (already in GCS at gs://sonata-training-johnb-2025/data/)
**Script**: `train/sonata/train_speaker_encoder.py` (complete)
**Output**: `speaker_encoder_best.safetensors` + config JSON

Training config:

- 100 epochs, batch 32, lr 0.001
- CosineAnnealingLR scheduler
- 3-second random segments per utterance
- ~12 hours on L4

Integration: Speaker embedding feeds into flow v3 via learned projection layer for zero-shot voice cloning. Rust inference via `sonata_speaker` crate (complete). C FFI via `speaker_encoder_native.h` (complete).

## Workstream 2: 12.5Hz Codec (VM2, after speaker encoder)

**Model**: Conformer encoder + TemporalContextModule + FSQ 4096 + ConvNeXt decoder, ~135M params
**Data**: LibriTTS-R audio (same GCS bucket)
**Script**: `train/sonata/train_codec.py` (needs `--codec-version 12hz` flag)
**Architecture**: `train/sonata/codec_12hz.py` (complete)
**Output**: `codec_12hz_best.pt`

3-stage training (200K steps total):

1. Reconstruction-only (50K steps): STFT + Mel loss, lr 3e-4
2. Add perceptual (50K steps): + WavLM loss (weight 0.5), lr 1e-4
3. Adversarial fine-tuning (100K steps): + MPD/MSD discriminators, lr 1e-4

Key specs: 12.5 Hz frame rate (vs 50 Hz baseline), 150 bps (vs 750), 1920 hop length, 160 mel bins. 4x token reduction enables faster LM inference.

## Workstream 3: Flow Distillation (VM1, after flow v3 finishes)

**Model**: Student SonataFlowV3 initialized from teacher, with EMA target model
**Approach**: Consistency distillation (Song et al. 2023)
**Teacher**: Flow v3 best checkpoint from current run
**Script**: `train/sonata/train_distill_v3.py` (complete)
**Output**: `flow_v3_distilled.pt`

Algorithm:

- Sample t uniformly, interpolate x_t = (1-t)*noise + t*x_0
- Teacher takes one Heun ODE step (frozen)
- Student predicts x_0 from x_t
- EMA target predicts x*0 from x*{t+dt}
- Loss: MSE(student_x0, detach(ema_x0))
- EMA decay ramps 0.95 → 0.999

50K steps, ~1 day on L4. Validation every 5K steps compares 1-step student vs 8-step teacher.

## Implementation Changes Required

1. **Generalize launch.sh**: Support job types `speaker_encoder`, `codec_12hz`, `distill_v3` in addition to `flow_v3` and `vocoder`
2. **Generalize train_wrapper.sh**: Add CLI args and checkpoint patterns for new jobs
3. **train_codec.py**: Add `--codec-version 12hz` flag to select `Codec12HzConfig`
4. **Manifest path fix**: Ensure speaker encoder training script resolves Mac paths via symlink (same fix as flow v3)

## Data Scaling Decision

LibriLight (60K hours) deferred. All training uses LibriTTS-R (585 hours) already uploaded to GCS. Data scaling can be revisited after these models are trained and integrated.

## Post-Training Integration

1. Speaker encoder → safetensors export → Rust `sonata_speaker` loads weights → zero-shot voice cloning via flow v3 speaker conditioning
2. Distilled flow v3 → replaces teacher model → 1-2 step generation (4-8x faster TTS)
3. 12.5Hz codec → re-encode training data → 4x fewer tokens for future LM training
4. Combined: streaming zero-shot TTS with ~80ms generation latency on Apple Silicon
