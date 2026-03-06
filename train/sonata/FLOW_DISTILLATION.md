# Sonata Flow Distillation Guide

Compress the multi-step Sonata Flow model into a 1-step model. Two modes:

- **Progressive distillation**: 8→4→2→1 steps (Salimans & Ho, 2022)
- **Consistency training**: Direct 1-step via endpoint matching + velocity alignment (DSFlow-style)

## Prerequisites

```bash
cd pocket-voice
source .venv/bin/activate
pip install torch safetensors
```

## Step 1: Prepare Training Data

The distillation script supports three data formats:

1. **Directory of `.pt` files** — one per utterance
2. **Single shard `.pt` file** — list of dicts from `data_pipeline.py`
3. **Shard index `.shards.txt`** — points to multiple shard files

### Using encoded data from the pipeline

If you've already run `data_pipeline.py`, use the shard index directly:

```bash
python train_flow_distill.py \
    --data-dir train/data/encoded_all.shards.txt \
    ...
```

### Extracting pairs from audio (standalone)

```python
"""Extract (semantic_tokens, acoustic_latents) pairs for distillation."""
import torch, os
from codec import SonataCodec
from config import CodecConfig

codec = SonataCodec(CodecConfig()).to("mps").eval()
# Load checkpoint...

output_dir = "data/sonata_pairs"
os.makedirs(output_dir, exist_ok=True)

for i, wav_path in enumerate(audio_files):
    audio = load_audio(wav_path)  # 24kHz mono tensor
    with torch.no_grad():
        out = codec.encode(audio.unsqueeze(0).to("mps"))
        sem = out["semantic_tokens"].squeeze(0).cpu()
        aco = out["acoustic_latents"].squeeze(0).cpu()
    torch.save({"semantic_tokens": sem, "acoustic_latents": aco},
               f"{output_dir}/pair_{i:05d}.pt")
```

## Step 2: Run Distillation

### Option A: Progressive Distillation (8→4→2→1)

```bash
cd train/sonata

python train_flow_distill.py \
    --teacher checkpoints/flow/flow_best.pt \
    --teacher-config checkpoints/flow/flow_config.json \
    --data-dir ../../data/sonata_pairs/ \
    --output-dir ../../checkpoints/flow_distilled/ \
    --device mps \
    --phase-epochs 50 \
    --batch-size 8 \
    --lr 1e-4
```

Three phases run automatically:

| Phase | Teacher | Student | What happens                                       |
| ----- | ------- | ------- | -------------------------------------------------- |
| 1     | 8 steps | 4 steps | Student matches teacher's 8-step output in 4 steps |
| 2     | 4 steps | 2 steps | Promoted student matches previous student          |
| 3     | 2 steps | 1 steps | Final 1-step model                                 |

EMA is applied throughout. Final checkpoint uses EMA weights.

### Option B: Consistency Training (direct 1-step)

```bash
python train_flow_distill.py \
    --teacher checkpoints/flow/flow_best.pt \
    --teacher-config checkpoints/flow/flow_config.json \
    --data-dir ../../data/sonata_pairs/ \
    --output-dir ../../checkpoints/flow_consistency/ \
    --consistency \
    --consistency-steps 50000 \
    --device mps \
    --batch-size 8 \
    --lr 1e-4
```

Consistency training uses three loss terms:

1. **Endpoint matching**: From any x_t, one step should reach x_1
2. **Velocity alignment**: Predicted velocity matches true mean velocity
3. **Self-consistency**: Two points on same trajectory agree on endpoint

This often produces better 1-step quality than progressive distillation.

## Step 3: Deploy

```bash
# Copy distilled weights
cp checkpoints/flow_distilled/flow_distilled_1step.pt models/sonata/

# Or export to safetensors for Rust inference
python export_weights.py \
    --flow-ckpt checkpoints/flow_distilled/flow_distilled_1step.pt \
    --flow-config checkpoints/flow/flow_config.json \
    --output-dir models/sonata

# Run with 1-step inference
./pocket-voice --tts-engine sonata \
    --sonata-flow-weights models/sonata/sonata_flow.safetensors \
    --flow-steps 1
```

## CLI Reference

| Flag                  | Default                      | Description                                     |
| --------------------- | ---------------------------- | ----------------------------------------------- |
| `--teacher`           | required                     | Path to teacher flow weights                    |
| `--teacher-config`    | required                     | Path to flow config JSON                        |
| `--data-dir`          | required                     | Data path (directory, .pt file, or .shards.txt) |
| `--output-dir`        | `checkpoints/flow_distilled` | Output directory                                |
| `--device`            | `mps`                        | Device (mps/cuda/cpu)                           |
| `--batch-size`        | 16                           | Batch size                                      |
| `--lr`                | 1e-4                         | Learning rate                                   |
| `--phase-epochs`      | 10                           | Epochs per phase (progressive mode)             |
| `--consistency`       | false                        | Use consistency training instead of progressive |
| `--consistency-steps` | 50000                        | Total steps for consistency mode                |
| `--aux-loss-weight`   | 0.1                          | Weight for auxiliary OT-CFM loss                |

## Expected Results

| Model       | Steps | Quality | Speed | Training |
| ----------- | ----- | ------- | ----- | -------- |
| Original    | 8     | Best    | 1x    | —        |
| Progressive | 4     | ~Same   | 2x    | Phase 1  |
| Progressive | 2     | Good    | 4x    | Phase 2  |
| Progressive | 1     | Good    | 8x    | Phase 3  |
| Consistency | 1     | Good+   | 8x    | Direct   |

## Troubleshooting

- **OOM on MPS**: Reduce `--batch-size` to 4 or 2
- **Diverging loss**: Reduce `--lr` to 5e-5
- **Poor quality at 1 step**: Try consistency mode, or more epochs
- **Missing config**: Use `--flow-config` to provide config separately for raw state_dict checkpoints
