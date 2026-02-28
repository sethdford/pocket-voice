# Sonata Codec 12.5Hz — Architecture Design

## Overview

Replace the 50Hz codec with a 12.5Hz variant, reducing token count 4x while maintaining speech quality. This compounds with speculative decoding for ~9x effective LM speedup.

| Parameter       | 50Hz (current) | 12.5Hz (new)      | Ratio                  |
| --------------- | -------------- | ----------------- | ---------------------- |
| Frame rate      | 50 Hz          | 12.5 Hz           | 4x fewer tokens        |
| hop_length      | 480            | 1920              | 4x                     |
| n_fft           | 1024           | 4096              | 4x                     |
| n_mels          | 80             | 160               | 2x richer              |
| enc_dim         | 256            | 512               | 2x capacity            |
| acoustic_dim    | 256            | 512               | 2x capacity            |
| FSQ codebook    | 32768 (8^5)    | 4096 (8^4)        | Smaller but sufficient |
| Encoder strides | [8,5,4,3]=480x | [4,8,5,4,3]=1920x | +1 stage               |
| Decoder strides | [8,5,4,3]=480x | [3,4,5,8,4]=1920x | +1 stage               |
| Bits/sec        | 750 bps        | 150 bps           | 5x reduction           |

## Motivation

The Sonata LM generates semantic tokens autoregressively. At 50Hz, a 10-second utterance requires 500 tokens. At 12.5Hz, the same utterance needs only 125 tokens — 4x fewer autoregressive steps.

Combined with ReDrafter speculative decoding (2.3x speedup from Task #1), the effective throughput improvement is:

- **4x (fewer tokens) x 2.3x (spec decoding) = 9.2x faster generation**

This is critical for real-time TTS on Apple Silicon where LM inference is the latency bottleneck.

## Architecture

### Semantic Encoder Path

```
Audio (24kHz)
  → Mel Spectrogram (n_fft=4096, hop=1920, n_mels=160)  → 12.5Hz, 160-dim
  → Linear projection (160 → 512)
  → TemporalContextModule (multi-scale dilated conv)      → context aggregation
  → Conformer (6 layers, dim=512, 8 heads)                → sequence modeling
  → Linear projection (512 → 4)
  → FSQ ([8,8,8,8] = 4096 entries)                        → discrete tokens
```

**TemporalContextModule**: Parallel dilated convolutions at rates [1, 2, 4, 8] fused via 1x1 conv. With kernel_size=7, the effective receptive field spans 7 + 14 + 28 + 56 = 105 frames at 12.5Hz (8.4 seconds). This compensates for the 4x lower temporal resolution by giving the Conformer pre-aggregated multi-scale context.

### Acoustic Encoder Path

```
Audio (24kHz)
  → Conv1d (1 → 24, k=7)
  → 5x ConvEncoderBlock:
      stride 4:  24 →  48 channels
      stride 8:  48 →  96 channels
      stride 5:  96 → 192 channels
      stride 4: 192 → 384 channels
      stride 3: 384 → 768 channels
  → Conv1d (768 → 512, k=1)                               → 12.5Hz, 512-dim
```

Product of strides: 4 x 8 x 5 x 4 x 3 = 1920x downsample.

Each ConvEncoderBlock: strided conv + 3 dilated residual units (d=1,3,9).

### Decoder (ConvTranspose)

```
[FSQ codes (4-dim) + Acoustic latent (512-dim)]  → 516-dim input
  → Conv1d (516 → 768, k=7)
  → ConvNeXt backbone (10 layers, dim=768)                 → feature refinement
  → 5x UpsampleBlock:
      stride 3: 768 → 768 channels
      stride 4: 768 → 384 channels
      stride 5: 384 → 192 channels
      stride 8: 192 →  96 channels
      stride 4:  96 →  48 channels
  → Conv1d (48 → 1, k=7) + Tanh                           → 24kHz audio
```

Product of strides: 3 x 4 x 5 x 8 x 4 = 1920x upsample.

### Alternative: iSTFT Decoder

For lower latency at the cost of some quality, an iSTFT decoder variant is supported:

- ConvNeXt backbone predicts magnitude (2049 bins) and instantaneous frequency
- iSTFT reconstructs audio from predicted STFT
- n_fft=4096, hop=1920 → each frame produces 1920 output samples (80ms)

## Trade-off Analysis

### Quality Impact

At 12.5Hz, each token must encode 4x more temporal information. Expected impacts:

1. **Consonant clarity**: Plosives (p, t, k) and fricatives (s, sh) have durations of 50-100ms. At 50Hz, these span 2-5 frames. At 12.5Hz, they span 0.6-1.25 frames. **Mitigation**: Larger acoustic latent (512-dim) and deeper encoder capture sub-frame detail.

2. **Pitch tracking**: F0 for speech ranges 80-300Hz. At 12.5Hz, the Nyquist limit for pitch modulation is 6.25Hz — sufficient for normal prosody (~3-5Hz modulation). Fast pitch vibrato (>6Hz) may be smoothed.

3. **Background texture**: Subtle room tone and noise patterns that change faster than 80ms will be averaged. This is acceptable for TTS (generated speech has no room tone).

4. **Reconstruction SNR**: Expected -2 to -4 dB vs 50Hz codec initially. Gap closes with adversarial training (discriminator forces high-frequency detail).

### Comparison with Reference Architectures

| System            | Frame Rate  | Codebooks   | Bitrate     | PESQ    | Approach                |
| ----------------- | ----------- | ----------- | ----------- | ------- | ----------------------- |
| Sonata 50Hz       | 50 Hz       | 1 (FSQ)     | 750 bps     | TBD     | Conformer + ConvDec     |
| **Sonata 12.5Hz** | **12.5 Hz** | **1 (FSQ)** | **150 bps** | **TBD** | **Conformer + ConvDec** |
| Mimi              | 12.5 Hz     | 8 (RVQ)     | 1100 bps    | 3.9     | SeaNet + causal         |
| DualCodec         | 12.5 Hz     | 1 + multi   | ~300 bps    | 3.7     | Dual-track FSQ          |
| WavTokenizer      | 40 Hz       | 1           | 480 bps     | 3.8     | Attention decoder       |
| Encodec           | 75 Hz       | 8 (RVQ)     | 6000 bps    | 3.5     | Conv + LSTM             |

Our 150 bps is lower than Mimi's 1100 bps because we use a single FSQ codebook while Mimi uses 8 RVQ codebooks. Our acoustic latent (512-dim continuous) carries the fine detail that Mimi's extra codebooks provide. The Flow network predicts this latent at inference time.

## FSQ Design

We use FSQ [8, 8, 8, 8] = 4096 codebook entries (4 dimensions, 8 levels each).

**Why 4096 instead of 32768?**

- At 12.5Hz, each token represents 4x more audio → needs to be a coarser "semantic" token
- The acoustic detail is carried by the 512-dim continuous latent (predicted by Flow)
- 4096 entries at 12.5Hz = 150 bps, which is sufficient for semantic content
- Matches Mimi's semantic codebook (first RVQ level, ~100 bps)
- Proven no-collapse behavior with FSQ (no codebook utilization issues)

**Why not larger?**

- Larger codebook = harder LM modeling task = slower convergence
- The LM prediction quality matters more than codebook granularity
- 4096 = 2^12, efficient for embedding tables and GPU memory alignment

## Latent Dimension Analysis

The acoustic latent dimension is increased from 256 to 512:

- **Information density**: Each 12.5Hz frame encodes 1920 audio samples (vs 480 at 50Hz). The 4x increase in temporal information requires proportionally more latent capacity.
- **512-dim at 12.5Hz**: 512 \* 12.5 = 6400 floats/sec = 25.6 KB/sec (32-bit)
- **256-dim at 50Hz**: 256 \* 50 = 12800 floats/sec = 51.2 KB/sec (32-bit)
- Net effect: 2x reduction in acoustic bandwidth (6400 vs 12800 floats/sec), which the Flow model compensates for through richer per-frame generation.

## Training Configuration

### Stage 1: Reconstruction-only (50K steps)

```
lr: 3e-4 (cosine decay to 1e-5)
batch_size: 32 (8 per GPU × 4 GPUs) or 16 on single M-series
warmup: 2000 steps
optimizer: AdamW (β1=0.8, β2=0.99, weight_decay=0.01)
audio_length: 3.0 seconds (72000 samples)

Loss weights:
  multi_scale_stft: 1.0
  mel_reconstruction: 1.0
  fsq_entropy: 0.1
  wavlm_perceptual: 0.0 (added in stage 2)
```

### Stage 2: Add perceptual loss (50K steps)

```
lr: 1e-4 (cosine decay to 1e-5)
+ wavlm_perceptual: 0.5
```

### Stage 3: Adversarial fine-tuning (100K steps)

```
lr_generator: 1e-4
lr_discriminator: 2e-4
discriminator: SonataDiscriminator(use_mrd=True)

Loss weights:
  multi_scale_stft: 1.0
  mel_reconstruction: 1.0
  fsq_entropy: 0.05
  wavlm_perceptual: 0.5
  adversarial: 1.0
  feature_matching: 2.0
  r1_penalty: 10.0 (every 16 steps)
```

### Metrics to Track

- PESQ (target: > 3.5)
- STOI (target: > 0.92)
- Mel cepstral distortion (target: < 4.0 dB)
- FSQ codebook utilization (target: > 90%)
- Reconstruction SNR (target: > 15 dB)
- WER of reconstructed speech via STT (target: < 5%)

## Migration Plan

### 1. Retrain Sonata LM on 12.5Hz tokens

**Changes needed:**

- `semantic_vocab_size`: 32768 → 4096 (smaller embedding table)
- `max_seq_len`: 4096 → 1024 (4x fewer tokens per utterance)
- Re-encode training data with 12.5Hz codec (data_pipeline.py)
- Adjust positional encoding scale (each position = 80ms vs 20ms)
- Training time: ~60% of original (fewer tokens per sample)

**Config changes:**

```python
SemanticLMConfig(
    semantic_vocab_size=4096,  # was 32768
    max_seq_len=1024,          # was 4096 (4x fewer tokens)
)
```

### 2. Retrain Flow network on 12.5Hz acoustic latents

**Changes needed:**

- `acoustic_dim`: 256 → 512 (larger latent to predict)
- `semantic_vocab_size`: 32768 → 4096
- Adjust conditioning: each flow step generates 80ms of audio context
- Re-encode training data with 12.5Hz codec
- Flow v3 chunk_size: adjust from 25 frames (500ms at 50Hz) to 6-7 frames (500ms at 12.5Hz)

**Config changes:**

```python
FlowConfig(
    acoustic_dim=512,           # was 256
    semantic_vocab_size=4096,   # was 32768
)
FlowV3Config(
    chunk_size=7,               # was 25 (to maintain ~500ms chunks)
)
```

### 3. Update iSTFT decoder C code

**Changes to `src/sonata_istft.c`:**

- `n_fft`: 1024 → 4096 (must be power of 2 — validated by existing check)
- `hop_length`: 480 → 1920
- Buffer sizes scale with n_fft: `real_buf`, `imag_buf` = 2048 floats (was 512)
- `frame_buf`, `overlap_buf` = 4096 floats (was 1024)
- Per-frame output: 1920 samples (80ms) vs 480 samples (20ms)
- Ring buffer arithmetic unchanged (indexes by n_fft modular)
- vDSP FFT setup: `log2n` = 12 (was 10)

**No algorithmic changes needed** — the iSTFT implementation is already parameterized by n_fft and hop_length. Only the creation call changes:

```c
// Before:
SonataISTFT *dec = sonata_istft_create(1024, 480);
// After:
SonataISTFT *dec = sonata_istft_create(4096, 1920);
```

### 4. Update data pipeline

Re-encode all training data:

```bash
python data_pipeline.py --source manifest \
  --manifest data/manifest_clean.jsonl \
  --codec-ckpt checkpoints/codec/sonata_codec_12hz.pt \
  --output data/encoded_12hz.pt --shard-size 10000
```

The data pipeline (data_pipeline.py) loads codec config from the checkpoint, so it will automatically use the 12.5Hz parameters.

### 5. Streaming latency impact

At 12.5Hz, the minimum latency increases from 20ms to 80ms per frame. However, the LM generates tokens 4x faster (fewer tokens), so the net effect on end-to-end latency is:

- **First-token latency**: Slightly higher (+60ms for first codec frame)
- **Throughput latency**: 4x lower (4x fewer LM steps per utterance)
- **Net round-trip**: Estimated ~280ms (was ~320ms) due to LM speedup dominating

## File Summary

| File                            | Purpose                                   |
| ------------------------------- | ----------------------------------------- |
| `train/sonata/codec_12hz.py`    | 12.5Hz codec implementation               |
| `train/sonata/config.py`        | `Codec12HzConfig` dataclass               |
| `docs/codec_12hz_design.md`     | This design document                      |
| `src/sonata_istft.c`            | No changes needed (parameterized)         |
| `train/sonata/data_pipeline.py` | No changes needed (loads from checkpoint) |
