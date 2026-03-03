# Sonata v2 Training Pipeline

Complete PyTorch training pipeline for Sonata v2 neural audio components.

## Quick Start

### 1. Audio Codec (Encoder + RVQ + Decoder)

Train a neural audio codec that compresses 24kHz audio into 8 quantized codes:

```bash
python train/train_codec.py \
  --data_dir /path/to/audio/files \
  --batch_size 16 \
  --lr 3e-4 \
  --epochs 100

# Resume training
python train/train_codec.py \
  --data_dir /path/to/audio/files \
  --resume train/checkpoints/codec/codec_best.pt \
  --epochs 200
```

**Model size**: ~25M parameters
**Input**: 24kHz mono audio (variable length)
**Output**: 8 quantized codes per ~20ms frame (8 codebooks, 1024 codes each)
**Loss**: L1 reconstruction + VQ commitment + codebook losses

### 2. Speaker Encoder (CAM++)

Already provided in `train_speaker_encoder.py`. Trains ECAPA-TDNN speaker encoder with ArcFace loss:

```bash
python train/train_speaker_encoder.py \
  --data-dir /path/to/LibriTTS_R \
  --batch-size 64 \
  --n-epochs 40 \
  --lr 0.001 \
  --scale 30.0 \
  --margin 0.2

# With augmentation
python train/train_speaker_encoder.py \
  --data-dir /path/to/LibriTTS_R \
  --musan-dir /path/to/MUSAN \
  --rir-dir /path/to/RIRS_NOISES \
  --crop-duration 4.0
```

**Model size**: ~24.6M parameters (ECAPA-TDNN)
**Input**: 16kHz mono speech
**Output**: 256-dim speaker embedding (L2 normalized)
**Loss**: AAM-Softmax with optional sub-centers

### 3. STT (Streaming Conformer CTC)

Train speech-to-text with CTC loss for alignment-free training:

```bash
python train/train_stt.py \
  --data_dir /path/to/audio \
  --text_dir /path/to/transcripts \
  --batch_size 32 \
  --lr 5e-4 \
  --epochs 100

# With custom Conformer depth
python train/train_stt.py \
  --data_dir /path/to/audio \
  --text_dir /path/to/transcripts \
  --num_layers 12 \
  --model_dim 256
```

**Model size**: ~100M parameters
**Input**: 16kHz mono speech (variable length)
**Output**: Character-level tokens (CTC decoder)
**Loss**: CTC loss (character-level, no alignment needed)
**Tokenizer**: Built-in character tokenizer (a-z, 0-9, space)

### 4. TTS (Text-to-Speech with Emotion)

Train text-to-speech with AdaIN speaker + emotion conditioning:

```bash
python train/train_tts.py \
  --text_file texts.txt \
  --speaker_file speakers.txt \
  --emotion_file emotions.txt \
  --batch_size 32 \
  --lr 1e-3 \
  --epochs 100
```

**Model size**: ~100M parameters
**Input**: Text (character IDs) + speaker ID + emotion ID
**Output**: 8 quantized codec codes per frame + nonverbal actions
**Loss**: Cross-entropy (codec tokens) + emotion loss + nonverbal loss
**Features**:
- AdaIN conditioning (speaker + emotion embeddings)
- Dual output heads (codec prediction + nonverbal actions)
- Character-level text encoding

## Test Mode (Synthetic Data)

All training scripts support synthetic data for quick testing:

```bash
# Test codec training (2 epochs, synthetic data)
python train/train_codec.py \
  --data_dir /tmp/dummy \
  --synthetic \
  --batch_size 4 \
  --epochs 2

# Test TTS training
python train/train_tts.py \
  --synthetic \
  --batch_size 4 \
  --epochs 2
```

## Export to Rust/Candle

Export trained PyTorch models to safetensors format for Rust inference:

```bash
# Export codec
python train/export_candle.py \
  --model codec \
  --checkpoint train/checkpoints/codec/codec_best.pt \
  --output models/sonata-codec.safetensors

# Export speaker encoder
python train/export_candle.py \
  --model cam \
  --checkpoint train/checkpoints/speaker_encoder_best.pt \
  --output models/cam-plus-plus.safetensors

# Export STT
python train/export_candle.py \
  --model stt \
  --checkpoint train/checkpoints/stt/stt_best.pt \
  --output models/sonata-stt.safetensors

# Export TTS
python train/export_candle.py \
  --model tts \
  --checkpoint train/checkpoints/tts/tts_best.pt \
  --output models/sonata-tts.safetensors

# Export with validation
python train/export_candle.py \
  --model codec \
  --checkpoint train/checkpoints/codec/codec_best.pt \
  --output models/sonata-codec.safetensors \
  --validate
```

Output files are stored in `models/` directory and ready for Rust candle inference.

## Dataset Format

### Codec Dataset
```
audio_dir/
├── audio1.wav
├── audio2.mp3
├── subdir/audio3.flac
└── ...
```

Any directory structure. Supports: `.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`

### STT Dataset
```
audio_dir/
├── sample1.wav
├── sample2.wav
└── ...

text_dir/
├── sample1.txt  (matching audio files)
├── sample2.txt
└── ...
```

Text files contain one line of transcript per audio file.

### TTS Dataset
```
texts.txt
speakers.txt
emotions.txt
```

Each line corresponds to one training example:
- `texts.txt`: Text string per line
- `speakers.txt`: Speaker ID (0-based integer) per line
- `emotions.txt`: Emotion ID (0-based integer) per line

Example:
```
# texts.txt
hello world
how are you today

# speakers.txt
0
1

# emotions.txt
5
10
```

## Training Hyperparameters

### Codec
| Parameter | Default | Notes |
|-----------|---------|-------|
| batch_size | 16 | Increase for better convergence |
| lr | 3e-4 | AdamW learning rate |
| segment_length | 24000 | 1 second at 24kHz |
| save_every | 10 | Checkpoint interval |
| val_split | 0.1 | 10% validation set |

### Speaker Encoder
| Parameter | Default | Notes |
|-----------|---------|-------|
| batch_size | 64 | Recommend 64+ for stable ArcFace |
| lr | 0.001 | Initial learning rate |
| scale | 30.0 | ArcFace scale parameter |
| margin | 0.2 | ArcFace margin (radians) |
| crop_duration | 3.0 | Seconds of audio per sample |
| warmup_epochs | 2 | Linear LR warmup |
| patience | 10 | Early stopping patience |

### STT
| Parameter | Default | Notes |
|-----------|---------|-------|
| batch_size | 32 | CTC prefers larger batches |
| lr | 5e-4 | Conformer learning rate |
| num_layers | 12 | Transformer depth |
| model_dim | 256 | Embedding dimension |

### TTS
| Parameter | Default | Notes |
|-----------|---------|-------|
| batch_size | 32 | AdaIN benefits from variety |
| lr | 1e-3 | Slightly higher for AdaIN |
| num_speakers | 100 | Vocabulary size |
| num_emotions | 64 | Emotion style tokens |

## Model Architecture Summary

### Codec
- **Encoder**: 4 strided convs (8×5×4×3 = 480× downsampling) → [B, 512, 50]
- **RVQ**: 8 quantizers with straight-through estimator
- **Decoder**: 4 transposed convs with matching upsample rates
- **Bottleneck**: 8 × 1024 = 8,192 unique codes per frame

### Speaker Encoder (CAM++)
- **Feature extraction**: 80-bin mel-spectrogram
- **Backbone**: SE-Res2Net blocks with multi-branch dilated convolutions
- **Pooling**: Attentive Statistics Pooling (channel-wise attention)
- **Output**: 256-dim L2-normalized embedding

### STT
- **Input**: Mel-spectrograms (80 bins)
- **Backbone**: 12 Conformer blocks (self-attention + depthwise conv)
- **Decoder**: CTC (Connectionist Temporal Classification)
- **Output**: Character-level tokens (blank + [a-z0-9 ])

### TTS
- **Text encoder**: Character embedding + duration prediction
- **Decoder**: 6 transformer blocks with AdaIN conditioning
- **Style tokens**: Speaker (100×256) + Emotion (64×256) embeddings
- **Output heads**:
  - Codec prediction (8 codes × 1024 vocabulary)
  - Nonverbal action prediction (10 classes)

## Checkpoints

### Saving
Automatic saving at:
- **Best**: `train/checkpoints/{model}/model_best.pt` (based on lowest loss)
- **Periodic**: `train/checkpoints/{model}/model_epoch{N}.pt` (every `save_every` epochs)

Checkpoint format:
```python
{
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),  # if applicable
    'epoch': int,
    'best_loss': float,
}
```

### Resuming
```bash
python train/train_codec.py \
  --data_dir /path/to/audio \
  --resume train/checkpoints/codec/codec_epoch50.pt \
  --epochs 100  # Trains for 100 total epochs
```

## GPU Memory Requirements

Approximate GPU memory usage:

| Model | Batch 16 | Batch 32 | Batch 64 |
|-------|----------|----------|----------|
| Codec | ~4GB | ~7GB | ~14GB |
| Codec (training) | ~5GB | ~9GB | ~18GB |
| STT | ~6GB | ~12GB | ~24GB |
| TTS | ~5GB | ~10GB | ~20GB |
| Speaker Enc. | ~3GB | ~6GB | ~12GB |

Use `--batch_size 4` on limited GPU memory (<8GB).

## Performance Tips

1. **Larger batches** (32+) improve convergence stability, especially for codec and speaker encoder
2. **Data augmentation** helps generalization:
   - Speed perturbation (0.9x, 1.0x, 1.1x)
   - MUSAN noise/music/speech augmentation (speaker encoder)
   - RIR reverb augmentation
3. **Mixed precision** training (not implemented) can save 30-40% GPU memory
4. **Gradient accumulation** enables larger effective batches on limited memory
5. **Learning rate scheduling**: Cosine annealing works well for all models

## Logging

All scripts use Python logging at INFO level:
- Model parameters at startup
- Per-epoch metrics
- Checkpoint save paths
- Training time per epoch

Enable debug logging:
```bash
export PYTHONUNBUFFERED=1
python train/train_codec.py --data_dir ... 2>&1 | tee training.log
```

## Files

```
train/
├── train_codec.py              # Audio codec training (25M params)
├── train_stt.py                # STT training (100M params, CTC loss)
├── train_tts.py                # TTS training (100M params, AdaIN)
├── train_speaker_encoder.py    # Speaker encoder training (24.6M params)
├── export_candle.py            # Export to safetensors for Rust
├── data/
│   ├── __init__.py
│   ├── codec_dataset.py        # Audio codec dataset + synthetic
│   └── (STT/TTS datasets built-in to train scripts)
├── checkpoints/
│   ├── codec/                  # Codec checkpoints
│   ├── stt/                    # STT checkpoints
│   ├── tts/                    # TTS checkpoints
│   └── speaker_encoder_v2/     # Speaker encoder checkpoints
└── TRAINING_README.md          # This file
```

## Next Steps

1. **Prepare data** in the required format (see Dataset Format)
2. **Train codec** first (audio tokens needed for TTS)
3. **Train speaker encoder** (speaker embeddings needed for TTS)
4. **Train STT** and **TTS** in parallel (independent training)
5. **Export models** using `export_candle.py`
6. **Integrate** exported models into Rust inference code

---

## References

- Codec architecture: Inspired by Encodec (Meta) and Descript's neural codec
- Speaker encoder: ECAPA-TDNN with ArcFace loss (Wei et al., 2020)
- STT: Conformer with CTC (Gulati et al., 2021)
- TTS: AdaIN normalization (Ulyanov et al., 2016) for speaker/emotion control

See individual training scripts for detailed architecture documentation.
