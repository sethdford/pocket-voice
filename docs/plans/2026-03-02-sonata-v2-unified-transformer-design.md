# Sonata v2: Unified Multimodal Transformer Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified on-device AI voice agent: Sonata v2 (SOTA voice pipeline) + SeaClaw (C11 agent runtime) + LLM provider (reasoning). One binary. All 4 voice pillars (STT, TTS, Full-Duplex, STS). Native C/Rust. On-device. Incorporating the best innovations from Kokoro, F5-TTS, CosyVoice, Dia, Chatterbox, Moshi, CAM++, and PersonaPlex.

**Architecture:** Sonata voice pipeline (~260M params) compiled as static library into SeaClaw agent runtime (349KB). SeaClaw provides LLM routing (50+ providers), 47 tools, memory, security. Sonata provides SOTA STT/TTS/full-duplex with AdaIN voice conditioning, emotion control, nonverbal vocabulary, CAM++ speaker encoder, and CFM decoder.

**Tech Stack:** C11 (SeaClaw agent runtime), Rust (candle for ML, Metal GPU), C (vDSP/Accelerate for DSP)

**Author:** Seth Ford + Claude
**Date:** March 2, 2026
**Updated:** March 2, 2026 (added SeaClaw integration)
**Status:** Approved for implementation planning
**Supersedes:** SONATA_V2_VISION.md Option B (Hybrid) — this design goes Full SOTA Rewrite
**Integrates with:** [SeaClaw](https://github.com/sethdford/seaclaw) — autonomous AI agent runtime in C11

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Sonata Codec (Audio Tokenizer)](#3-sonata-codec-audio-tokenizer)
4. [CAM++ Speaker Encoder](#4-cam-speaker-encoder)
5. [Sonata Core Transformer](#5-sonata-core-transformer)
6. [CFM Decoder](#6-cfm-decoder-conditional-flow-matching)
7. [Full-Duplex Architecture](#7-full-duplex-architecture)
8. [Speech-to-Speech (STS)](#8-speech-to-speech-sts-direct-path)
9. [Training Strategy](#9-training-strategy)
10. [SeaClaw Integration (Unified Binary)](#10-seaclaw-integration-unified-binary)
11. [Implementation Language Strategy](#11-implementation-language-strategy)
12. [Comparison Matrix](#12-comparison-matrix)
13. [Migration from v1](#13-migration-from-v1)
14. [Risk Assessment](#14-risk-assessment)

---

## 1. Executive Summary

### Why Full SOTA Rewrite (Not Hybrid)

The existing `SONATA_V2_VISION.md` recommended Option B (Hybrid Dual-Stream, 4-6 months). After analyzing 10 SOTA voice models (Kokoro, F5-TTS, CosyVoice, Dia, Chatterbox, Moshi, CAM++, PersonaPlex, Fish Speech, Orpheus TTS), we identified innovations that push beyond hybrid compromise. Sonata v2 should BE the best — not a wrapper or halfway measure.

### SOTA Innovations Incorporated

| Source Model | Innovation | Where in Sonata v2 |
|-------------|-----------|-------------------|
| **Moshi** (Kyutai) | Dual-stream full-duplex, Depth Transformer, Mimi codec | Core Transformer dual-stream, Depth Transformer, Codec architecture |
| **Kokoro** (Hexgrad) | AdaIN voice conditioning at every layer, 82M params | AdaIN in Core Transformer + CFM Decoder |
| **F5-TTS** (SWivid) | DiT + Conditional Flow Matching, 335M params | CFM Decoder architecture |
| **CosyVoice** (Alibaba) | Causal flow matching for streaming, LLM-based TTS | Causal CFM for streaming TTS |
| **Dia** (Nari Labs) | 24 nonverbal tags, multi-speaker dialogue, 1.6B | Nonverbal token vocabulary |
| **Chatterbox** (Resemble AI) | Emotion exaggeration scalar, paralinguistic tags | Emotion style tokens + exaggeration scalar |
| **CAM++** (WeSpeaker) | Context-aware masking, 0.56% EER at 7.18M params | Speaker encoder (replaces ECAPA-TDNN) |
| **PersonaPlex** (NVIDIA) | Full-duplex voice agent, MIT license | Full-duplex conversation training |
| **Piper** (Rhasspy) | 20-80MB on-device, C++/ONNX | On-device optimization patterns |

### Design Decision

**Chosen approach:** Unified Binary — Sonata voice pipeline + SeaClaw agent runtime + LLM provider
**Key evolution:** The original "unified transformer" design evolved to separate voice (Sonata ~260M) from reasoning (SeaClaw + LLM). Sonata focuses on being the BEST voice pipeline. SeaClaw provides the agent brain. LLM providers handle reasoning. This is more powerful than a single model trying to do everything.
**Rejected alternatives:**
- Option B Hybrid (SONATA_V2_VISION.md) — compromise, doesn't achieve SOTA across all pillars
- Modular Best-of-Breed — cascade latency, full-duplex is bolted on
- Codec-Unified — codec quality is single point of failure, too novel/risky
- Sonata as separate process (IPC bridge) — adds latency, complexity
- Sonata wrapping other providers — user wants Sonata to BE the best

---

## 2. System Architecture Overview

```
+------------------------------------------------------------------+
|                    SONATA v2 SYSTEM                                |
|                                                                    |
|  +------------+    +-----------------------+    +--------------+  |
|  | Sonata     |    |  Sonata Core          |    | Sonata       |  |
|  | Codec      |--->|  Unified Transformer  |--->| CFM Decoder  |  |
|  | Encoder    |    |  (400M params)        |    | + iSTFT      |  |
|  +------------+    |                       |    +--------------+  |
|       ^            |  +--Dual-Stream-----+ |          |           |
|       |            |  | User + System    | |     24kHz Audio      |
|  24kHz Audio       |  +--AdaIN Speaker---+ |                      |
|  from mic          |  +--Emotion Style---+ |                      |
|                    |  +--Text Sideband---+ |                      |
|  +------------+    |  +--Nonverbal Tags--+ |                      |
|  | CAM++      |--->|                       |                      |
|  | Speaker    |    +-----------------------+                      |
|  | Encoder    |                                                   |
|  | (7.18M)    |                                                   |
|  +------------+                                                   |
+------------------------------------------------------------------+
```

### Five Native Components

| Component | Params | Language | Hardware | Purpose |
|-----------|--------|----------|----------|---------|
| **Sonata Codec** | ~25M | Rust (candle) | Metal GPU | Audio <-> discrete tokens |
| **CAM++ Speaker Encoder** | 7.18M | Rust (candle) | Metal GPU | Voice identity -> AdaIN embedding |
| **Sonata Core Transformer** | ~400M | Rust (candle) | Metal GPU | Unified intelligence (all 4 pillars) |
| **CFM Decoder** | ~35M | Rust (candle) | Metal GPU | Audio tokens -> mel spectrogram |
| **iSTFT Vocoder** | ~0 (DSP) | C (vDSP) | AMX/ANE | Mel -> 24kHz waveform (5000x RT) |

### Data Flow Per Pillar

| Pillar | Path | Latency |
|--------|------|---------|
| **STT** | Audio -> Codec -> Core (CTC head) -> Text | ~80ms |
| **TTS** | Text -> Core (audio head) -> CFM -> iSTFT -> Audio | ~100ms |
| **Full-Duplex** | User Audio + System Audio -> Core (dual-stream) -> System Audio | ~160ms |
| **STS** | Audio -> Codec -> Core -> Codec -> CFM -> iSTFT -> Audio | ~160ms |

### Latency Breakdown (Full-Duplex STS)

```
Codec Encoder:      5ms   (audio -> tokens)
Core Transformer: 120ms   (token generation)
Depth Transformer:  5ms   (acoustic detail, parallel)
CFM Decoder:       25ms   (tokens -> mel, 4-step)
iSTFT Vocoder:    < 1ms   (mel -> audio, vDSP)
Pipeline overhead:  4ms
                  --------
Total:           ~160ms
```

---

## 3. Sonata Codec (Audio Tokenizer)

The codec is the **foundation** — everything depends on high-quality audio tokenization.

### Architecture

Based on Moshi's Mimi codec + DualCodec's semantic-acoustic unification:

```
24kHz Audio (1 channel)
    |
+------------------------------+
|  Encoder (1D ConvNet)         |
|  Downsample: 24000 -> 50 Hz  |   <-- 480x downsampling
|  Strides: [8, 5, 4, 3]       |   <-- Same as current Code2Wav
|  Channels: 64->128->256->512 |
|  Snake activation (learned)   |
+------------------------------+
    |
+------------------------------+
|  Semantic Conditioning        |
|  Inject text/phoneme context  |   <-- DualCodec innovation
|  via cross-attention          |   <-- Unifies semantic+acoustic
+------------------------------+
    |
+------------------------------+
|  Residual Vector Quantizer    |
|  8 codebooks x 1024 entries   |   <-- 8192 effective vocab
|  Codebook dim: 128            |
|  Split quantization:          |
|    Books 1-2: Semantic        |   <-- Content, phonemes
|    Books 3-8: Acoustic        |   <-- Timbre, prosody, quality
+------------------------------+
    |
  50 Hz discrete tokens (8 streams)
    |
+------------------------------+
|  Decoder (1D TransConvNet)    |
|  Upsample: 50 Hz -> 24000    |
|  Strides: [3, 4, 5, 8]       |
|  Snake activation             |
|  Multi-band decomposition     |
+------------------------------+
    |
24kHz Audio reconstruction
```

### Key Design Decisions

1. **50Hz frame rate** (not 12.5Hz): Higher rate = better quality. Moshi uses 12.5Hz but struggles with quality. 50Hz gives excellent reconstruction at the cost of more tokens per second. At 50Hz with 8 codebooks, that's 400 tokens/second — manageable for modern transformers.

2. **Split RVQ (semantic + acoustic)**: First 2 codebooks capture content (what is said), remaining 6 capture quality (how it sounds). The Core transformer primarily generates books 1-2 (semantic); the Depth Transformer fills in books 3-8 (acoustic detail) in parallel.

3. **Same downsampling strides as current Code2Wav** `[8,5,4,3]`: Existing C/Rust infrastructure supports this exact architecture. The current `code2wav` in `src/sonata_flow/` uses identical ConvTranspose1d upsampling.

4. **Snake activation**: Learned periodic activation function that adapts to audio frequency content. Better than ELU/ReLU for audio synthesis.

### Training

- **Data**: LibriTTS-R + VCTK + CommonVoice (10K+ hours)
- **Loss**: Reconstruction (L1 + multi-scale STFT) + commitment loss + adversarial (multi-scale discriminator)
- **Hardware**: 4x A100 (or equivalent)
- **Duration**: 2-3 weeks
- **Target quality**: PESQ > 4.0, STOI > 0.95, ViSQOL > 4.0

### Codec File Structure

```
sonata-codec/
  src/
    lib.rs          # Public API
    encoder.rs      # 1D ConvNet encoder
    decoder.rs      # 1D TransConvNet decoder
    quantizer.rs    # Residual Vector Quantizer
    snake.rs        # Snake activation function
    semantic.rs     # Semantic conditioning layer
  Cargo.toml
```

---

## 4. CAM++ Speaker Encoder

Replace ECAPA-TDNN (23M params, 0.86% EER) with CAM++ (7.18M params, **0.56% EER**).

### Architecture

From WeSpeaker research:

```
80-bin Mel Spectrogram (16kHz input)
    |
+-------------------------------------+
|  Multi-Scale Aggregation Frontend    |
|  Conv2D stack: 3x3 kernels          |
|  Channel progression: 64->128->256  |
|  Frequency-aware pooling             |
+-------------------------------------+
    |
+-------------------------------------+
|  Context-Aware Masking (CAM) Blocks  |
|  6 blocks, each:                     |
|    1. Multi-head self-attention       |
|    2. Context-dependent masking       |   <-- KEY INNOVATION
|    3. Segment-level pooling           |   <-- Captures local+global
|    4. Feed-forward + residual         |
|  Heads: 8, dim: 512                  |
+-------------------------------------+
    |
+-------------------------------------+
|  Attentive Statistics Pooling        |
|  Weighted mean + std aggregation     |
|  -> 192-dim speaker embedding        |
+-------------------------------------+
    |
  192-dim embedding -> AdaIN injection into Core + CFM
```

### What Makes CAM++ Special

- **Context-Aware Masking**: Each attention block generates a binary mask that selectively attends to speaker-informative frames and ignores noise/silence. This is why it achieves 0.56% EER — it automatically focuses on the parts of speech that carry identity.
- **Segment Pooling**: Unlike ECAPA-TDNN's global statistics pooling, CAM++ pools over segments first, then aggregates. Captures both local (phoneme-level) and global (utterance-level) speaker characteristics.

### Training

- **Same pipeline** as current v3 speaker encoder training on GCE
- **Same data**: LibriTTS-R (2311 speakers) + augmentation (MUSAN + RIR)
- **Same loss**: AAM-Softmax + sub-center ArcFace K=2
- **New architecture**: CAM++ blocks instead of ECAPA-TDNN SE-Res2Net blocks
- **Target**: EER < 0.6% on VoxCeleb1-O

### Output Usage

The 192-dim speaker embedding is injected via AdaIN into:
1. Every layer of the Core Transformer (voice identity in generation)
2. Every layer of the CFM Decoder (voice identity in synthesis)
3. Speaker verification (voice enrollment, recognition)

### File Structure

```
sonata-cam/
  src/
    lib.rs          # Public API
    frontend.rs     # Multi-scale aggregation
    cam_block.rs    # Context-aware masking block
    pooling.rs      # Attentive statistics pooling
    adain.rs        # Adaptive Instance Normalization
  Cargo.toml
```

---

## 5. Sonata Core Transformer

The unified intelligence — one transformer handling all 4 pillars simultaneously.

### Architecture

Inspired by Moshi's Helium + Kokoro's AdaIN + Dia's nonverbal tokens:

```
+----------------------------------------------------------+
|                   SONATA CORE (400M)                      |
|                                                           |
|  Input Embedding Layer:                                   |
|  +---------------+----------------+----------------+     |
|  | Audio Tokens   | Text Tokens    | Style Tokens   |     |
|  | (codec bk 1-2) | (32K vocab)    | (emotion/NV)   |     |
|  | 2048 vocab     |                | 64 vocab       |     |
|  +-------+--------+--------+-------+--------+-------+     |
|          |                 |                |              |
|          +-----------------+-----------+----+              |
|                            |                               |
|  +-------------------------------------------+            |
|  |  Dual-Stream Interleaver                  |            |
|  |  +------------------+------------------+  |            |
|  |  | User Stream      | System Stream    |  |            |
|  |  | (listen)         | (speak)          |  |            |
|  |  | Audio tokens     | Audio tokens     |  |            |
|  |  | from codec       | to generate      |  |            |
|  |  +------------------+------------------+  |            |
|  +-------------------------------------------+            |
|                            |                               |
|  +-------------------------------------------+  +------+ |
|  |  Outer Transformer (24 layers)            |<-| AdaIN| |
|  |  d_model: 1024                            |  | from | |
|  |  Heads: 16 query + 4 KV (GQA)            |  | CAM++| |
|  |  FFN: 4096 (SwiGLU)                      |  | spkr | |
|  |  RoPE positional encoding                 |  | embed| |
|  |  Sliding window attention (4096 tokens)   |  +------+ |
|  |                                           |            |
|  |  Every layer:                             |            |
|  |    1. Self-attention (causal)             |            |
|  |    2. AdaIN(speaker_embed)       <Kokoro> |            |
|  |    3. Cross-attn(user<->system)  <Moshi>  |            |
|  |    4. SwiGLU FFN                          |            |
|  |    5. AdaIN(emotion_style)  <Chatterbox>  |            |
|  +-------------------------------------------+            |
|                            |                               |
|  +-------------------------------------------+            |
|  |  Depth Transformer (6 layers, small)      |            |
|  |  d_model: 256                             |            |
|  |  Fills in codec books 3-8 from books 1-2  |            |
|  |  (Acoustic detail, runs in parallel)      |            |
|  +-------------------------------------------+  <Moshi>  |
|                            |                               |
|  Output Heads:                                            |
|  +----------------+----------------+----------------+     |
|  | Audio Head      | Text Head      | Style Head     |     |
|  | Codec tokens    | Text tokens    | Nonverbal tags |     |
|  | (for TTS/STS)   | (for tools)    |                |     |
|  +----------------+----------------+----------------+     |
+----------------------------------------------------------+
```

### Parameter Budget

| Component | Params | Notes |
|-----------|--------|-------|
| Token embeddings | ~35M | Audio (2048x1024) + Text (32Kx1024) + Style (64x1024) |
| Outer Transformer | ~300M | 24 layers x 1024 dim x 16 heads |
| Depth Transformer | ~25M | 6 layers x 256 dim (acoustic detail) |
| AdaIN layers | ~10M | Speaker + emotion conditioning at each layer |
| Output heads | ~30M | Audio + text + style projections |
| **Total** | **~400M** | Fits on Apple M-series with Metal |

### Key Innovations

1. **Dual-Stream (Moshi)**: User and system audio tokens are interleaved in the same sequence. At each 20ms frame, the model consumes 1 user audio token and generates 1 system audio token. This IS full-duplex — no EOU detection needed.

2. **AdaIN Speaker Conditioning (Kokoro)**: After every self-attention layer, apply Adaptive Instance Normalization using the 192-dim CAM++ speaker embedding:
   ```
   AdaIN(x, speaker_embed) = gamma(speaker_embed) * normalize(x) + beta(speaker_embed)
   ```
   This conditions EVERY layer with voice identity — the model always "knows" whose voice to generate.

3. **Emotion Style Tokens (Chatterbox)**: 64 learned style embeddings for emotion states (happy, sad, excited, calm, concerned, surprised, thoughtful, playful, etc.). Plus an **exaggeration scalar** (0.0-2.0) that scales the style contribution:
   ```
   style_contribution = emotion_embed * exaggeration_scalar
   ```
   At 0.0 = neutral. At 1.0 = natural emotion. At 2.0 = dramatically exaggerated.

4. **Nonverbal Vocabulary (Dia)**: Special tokens in the audio vocabulary:
   - `[laugh]`, `[chuckle]`, `[giggle]` — different laugh types
   - `[sigh]`, `[breath]`, `[gasp]` — breathing
   - `[hmm]`, `[uh-huh]`, `[oh]`, `[right]` — backchannels
   - `[pause_short]`, `[pause_long]` — intentional pauses
   - `[whisper]`, `[emphasis]` — speaking style modifiers

5. **Text Sideband**: The text head enables tool calling, reasoning, and API access. Audio-primary but text-capable when needed:
   - During STS mode: text head is inactive (pure audio)
   - During complex reasoning: text head activates
   - During tool calling: text head generates structured output

### Attention Pattern

```
For each timestep t:

Outer Transformer sees:
  [..., u_{t-2}, s_{t-2}, u_{t-1}, s_{t-1}, u_t, s_t]

  u_i = user audio token at frame i
  s_i = system audio token at frame i

  Causal mask: can see all past tokens, not future
  Cross-attention: user stream attends to system stream (bidirectional within window)
  Sliding window: 4096 tokens = ~40 seconds of audio

Depth Transformer sees:
  [semantic_codes_1_2] -> generates [acoustic_codes_3_8]
  Runs in parallel with Outer Transformer output
  6 small layers, d_model=256
```

### File Structure

```
sonata-core/
  src/
    lib.rs              # Public API + pipeline
    transformer.rs      # Outer transformer (24 layers)
    depth.rs            # Depth transformer (6 layers)
    attention.rs        # GQA + sliding window + cross-attention
    adain.rs            # AdaIN conditioning layers
    dual_stream.rs      # User/system stream interleaving
    embeddings.rs       # Audio + text + style token embeddings
    heads.rs            # Audio, text, and style output heads
    rope.rs             # Rotary positional encoding
    swiglu.rs           # SwiGLU feed-forward
  Cargo.toml
```

---

## 6. CFM Decoder (Conditional Flow Matching)

Replace the current 8-step ODE flow with CFM — 10-50x faster than diffusion, proven by F5-TTS and CosyVoice.

### Architecture

```
Codec Tokens (from Core, books 1-8)
    |
+--------------------------------------+
|  Token -> Continuous Embedding        |
|  Lookup: 8 codebooks -> 8x128 = 1024 |
|  Project to mel-space: 1024 -> 80    |
+--------------------------------------+
    |
+--------------------------------------+
|  CFM Transformer (DiT-style)          |
|  12 layers, d_model: 512             |   <-- F5-TTS architecture
|  Heads: 8                            |
|  AdaIN conditioning:                 |
|    - Speaker embedding (CAM++)       |
|    - Emotion style embedding         |
|  Timestep embedding (sinusoidal)     |
|                                      |
|  ODE Solver: Euler (4-8 steps)       |   <-- CosyVoice causal CFM
|  Causal masking for streaming        |
|  Classifier-free guidance (optional) |
+--------------------------------------+
    |
  80-bin Mel Spectrogram (50 Hz)
    |
+--------------------------------------+
|  iSTFT Vocoder (existing, vDSP)      |
|  Mel -> 24kHz waveform               |
|  5000x realtime (pure DSP)           |
+--------------------------------------+
    |
  24kHz Audio Output
```

### Why CFM Over Current Flow

- Current Sonata Flow uses Euler/Heun ODE solver with 4-8 steps. This works but isn't optimized.
- CFM uses a **straight-line ODE path** from noise to target — simpler, faster, more stable.
- Can be distilled to **1-2 steps** with consistency distillation (F5-TTS achieves 1-step with minimal quality loss).
- CosyVoice's **causal CFM** enables streaming — each mel frame can be generated independently without seeing future context.

### Streaming TTS Latency

```
Core Transformer: 1 audio token   ->   ~2ms per token
CFM Decoder:      4-step ODE       ->   ~10ms per frame
iSTFT:            mel -> audio     ->   ~0.2ms per frame
                                   ---------------------
                  Total per frame:     ~12ms (80x realtime)
```

### Distillation Path

| Steps | Quality (PESQ) | Latency | Notes |
|-------|---------------|---------|-------|
| 8 | 4.0+ | ~20ms/frame | Full quality |
| 4 | 3.9 | ~10ms/frame | Default target |
| 2 | 3.8 | ~5ms/frame | Distilled |
| 1 | 3.6-3.7 | ~3ms/frame | Consistency distilled, slight quality loss |

### File Structure

```
sonata-cfm/
  src/
    lib.rs          # Public API
    dit.rs          # DiT transformer blocks
    flow.rs         # CFM ODE solver (Euler, Heun)
    conditioning.rs # AdaIN + timestep conditioning
    streaming.rs    # Causal masking for streaming
  Cargo.toml
```

---

## 7. Full-Duplex Architecture

Native full-duplex via Moshi-style dual-stream tokens. No EOU detection needed.

### How It Works

```
Time ->  t1    t2    t3    t4    t5    t6    t7    t8
         -----------------------------------------------
User:   [u1]  [u2]  [u3]  [u4]  [u5]  [u6]  [u7]  [u8]
System: [s1]  [s2]  [s3]  [s4]  [s5]  [s6]  [s7]  [s8]

At each 20ms frame:
1. Encode user audio -> user token u_t (Sonata Codec)
2. Feed [u_t, s_{t-1}] into Core Transformer
3. Generate s_t (system audio token) + optional text token
4. Decode s_t -> system audio (CFM + iSTFT)
5. Play system audio to speaker

Both streams processed simultaneously.
Model learns when to speak, when to listen, when to backchannel.
No EOU detection. No turn-taking logic. It's all learned.
```

### Learned Behaviors

| User State | System Behavior | Token Pattern |
|-----------|----------------|---------------|
| Speaking | Silence + backchannels | `[silence]`, `[hmm]`, `[right]` |
| Pausing (thinking) | Wait patiently | `[silence]` with occasional `[hmm]` |
| Finished speaking | Begin response | Transition from `[silence]` to content tokens |
| Interrupting | Stop generating | Fade current tokens, switch to listening |
| Emotional (crying) | Empathetic sounds | `[oh]`, gentle `[hmm]`, softer prosody |

### Enhancement Over Moshi

Our dual-stream adds **style-conditioned backchanneling**. The emotion style tokens let the model generate contextually appropriate backchannels — not just `[hmm]` but the RIGHT kind of `[hmm]`:
- Encouraging `[hmm]` when user shares good news
- Thoughtful `[hmm]` when user is working through a problem
- Concerned `[oh]` when user mentions difficulty
- Excited `[oh!]` when user shares exciting news

### Backward Compatibility

For clients that don't support full-duplex:
- Disable user stream input (feed silence tokens)
- System generates as half-duplex (standard turn-taking)
- Use existing v1 EOU logic as fallback
- Gradual migration: full-duplex for supported clients, half-duplex for others

---

## 8. Speech-to-Speech (STS) Direct Path

Direct audio-to-audio without text intermediary:

```
User speaks "I'm feeling stressed today"
    |
Sonata Codec -> audio tokens [u1...un]
    |
Core Transformer processes audio tokens directly
    - Understands meaning from semantic codebooks (books 1-2)
    - Detects emotion from acoustic codebooks (books 3-8)
    - No text conversion needed for empathetic response
    |
Generates response audio tokens [s1...sm]
    - With matched emotion (concern, warmth)
    - With appropriate prosody (slower, softer)
    - With nonverbal tokens ([breath], gentle [hmm])
    |
CFM Decoder + iSTFT -> warm, empathetic audio response
```

### When Text Sideband Activates

| Scenario | Text Sideband | Why |
|----------|--------------|-----|
| Emotional conversation | OFF | Prosody carries the meaning |
| Backchanneling | OFF | No text needed |
| Simple responses | OFF | "Yes", "No", "Tell me more" |
| Tool calling | ON | "What's the weather?" needs structured API call |
| Complex reasoning | ON | "Explain quantum computing" benefits from text grounding |
| Memory retrieval | ON | "What did I say last Tuesday?" needs text query |
| Music/ambient | OFF | No text needed |

### Quality Advantage of STS

When text is removed as intermediary:
- **Prosody preservation**: 95% (vs 40% through text)
- **Emotion fidelity**: Near-perfect (acoustic codebooks carry full emotion)
- **Speaker characteristics**: Maintained (semantic codebooks preserve identity cues)
- **Latency**: ~160ms total (vs 630ms through text cascade)

---

## 9. Training Strategy

### Phase 1: Sonata Codec (2-3 weeks)

| Aspect | Detail |
|--------|--------|
| **Data** | LibriTTS-R (585h) + VCTK (44h) + CommonVoice English (2500h+) + LibriLight-small (600h) |
| **Architecture** | Encoder-RVQ-Decoder as described in Section 3 |
| **Loss** | L1 reconstruction + multi-scale STFT + commitment + adversarial (multi-scale discriminator) |
| **Hardware** | 4x A100 80GB (or equivalent) |
| **Duration** | 200K steps, batch size 32, ~2-3 weeks |
| **Target** | PESQ > 4.0, STOI > 0.95 |
| **Output** | Codec encoder + decoder + quantizer weights |

### Phase 2: STT Model Training (2-3 weeks)

| Aspect | Detail |
|--------|--------|
| **Data** | LibriSpeech (960h) + CommonVoice (2500h+) + GigaSpeech (10K hours) |
| **Architecture** | Enhanced Conformer (~100M) with streaming CTC head |
| **Task** | CTC loss on codec token -> text mapping |
| **Hardware** | 4x A100 80GB (or equivalent) |
| **Duration** | 100K steps, 2-3 weeks |
| **Target** | WER < 3% on LibriSpeech test-clean |
| **Output** | Streaming STT model weights |

### Phase 3: CAM++ Speaker Encoder (2-3 weeks)

| Aspect | Detail |
|--------|--------|
| **Data** | LibriTTS-R (2311 speakers) + VoxCeleb1+2 (7000+ speakers) |
| **Architecture** | CAM++ (7.18M params) with context-aware masking |
| **Loss** | AAM-Softmax + sub-center ArcFace K=2 |
| **Augmentation** | MUSAN + RIR noise injection, speed perturbation |
| **Hardware** | 1-2x A100 (smaller model) |
| **Duration** | 2-3 weeks (v3 training already in progress on GCE!) |
| **Target** | EER < 0.6% on VoxCeleb1-O |

### Phase 4: TTS Model Training (3-4 weeks)

| Aspect | Detail |
|--------|--------|
| **Data** | LibriTTS-R (585h) + VCTK (44h) + emotional speech (RAVDESS, CREMA-D, IEMOCAP) |
| **Architecture** | Transformer TTS (~100M) with AdaIN speaker conditioning, emotion style tokens, nonverbal vocabulary |
| **AdaIN** | Freeze CAM++, inject 192-dim embedding at every layer |
| **Emotion** | 64 learned style embeddings + exaggeration scalar (0-2) |
| **Nonverbal** | 24 special tokens ([laugh], [sigh], [breath], [hmm], etc.) |
| **Hardware** | 4x A100 80GB |
| **Duration** | 150K steps, 3-4 weeks |
| **Target** | PESQ > 3.8, emotion accuracy > 80%, natural nonverbal generation |

### Phase 5: Full-Duplex Controller (1-2 weeks)

| Aspect | Detail |
|--------|--------|
| **Data** | Fisher Corpus (2000h conversations), Switchboard (300h), CALLHOME (120h) |
| **Synthetic data** | Generate backchanneling data from dialogue transcripts using v1 TTS |
| **Architecture** | Lightweight controller (~10M) that decides: backchannel vs silence vs response |
| **Integration** | Bridges Sonata STT streaming output with SeaClaw agent + Sonata TTS |
| **Duration** | 1-2 weeks |
| **Target** | Natural turn-taking, contextual backchanneling while waiting for LLM |

### Phase 5: CFM Decoder Training (1-2 weeks)

| Aspect | Detail |
|--------|--------|
| **Data** | Mel spectrograms generated from codec outputs |
| **Architecture** | DiT with AdaIN conditioning |
| **Distillation schedule** | 8-step -> 4-step (consistency distillation) -> 2-step -> 1-step |
| **Duration** | 1-2 weeks total |
| **Target** | PESQ > 3.8 at 4-step, > 3.6 at 1-step |

### Phase 6: Native C/Rust Implementation (4-6 weeks)

| Aspect | Detail |
|--------|--------|
| **Port models** | PyTorch training weights -> candle Rust inference |
| **Optimize** | Metal GPU kernels, NEON SIMD, memory mapping |
| **Streaming** | Implement streaming pipeline with ring buffers |
| **NAPI** | Node-API bindings for TypeScript integration |
| **Testing** | Unit tests, integration tests, latency benchmarks |
| **Duration** | 4-6 weeks |

### Total Timeline: ~6-9 months

```
Month 1:    Codec training + CAM++ training (parallel, CAM++ v3 already started!)
Month 2:    STT model training + CFM decoder training (parallel)
Month 3-4:  TTS model training (depends on codec + CAM++)
Month 4:    Full-duplex controller (depends on STT + TTS)
Month 5-7:  Native C/Rust implementation + SeaClaw integration
Month 7-8:  Optimization + testing + benchmarking
Month 8-9:  Integration testing + release
```

### Compute Requirements

| Phase | Hardware | Duration | Estimated Cost |
|-------|----------|----------|---------------|
| Codec | 4x A100 | 3 weeks | ~$3K-5K |
| STT | 4x A100 | 3 weeks | ~$3K-5K |
| CAM++ | 1-2x A100 | 3 weeks | ~$1K-2K |
| TTS | 4x A100 | 4 weeks | ~$5K-8K |
| CFM | 4x A100 | 2 weeks | ~$2K-3K |
| Full-duplex | 2x A100 | 2 weeks | ~$1K-2K |
| **Total training** | | | **~$15K-25K** |

**Note:** Compute cost is lower than original design because we're training separate specialized models (~100M each) instead of one 400M unified transformer. Each model trains faster and can be trained in parallel.

---

## 10. SeaClaw Integration (Unified Binary)

### Overview

Sonata v2 integrates directly into [SeaClaw](https://github.com/sethdford/seaclaw), Seth's C11 autonomous AI agent runtime. The result is a **single binary** that is a complete on-device AI voice agent:

- **Sonata** provides SOTA voice (STT, TTS, full-duplex, emotion, nonverbal)
- **SeaClaw** provides agent intelligence (LLM routing, tools, memory, security)
- **LLM Provider** provides reasoning (Claude, GPT-4, Gemini, Ollama, llama.cpp)

### Unified Binary Architecture

```
+===================================================================+
|            UNIFIED BINARY (~50MB with models)                      |
|                                                                    |
|  +-- SEACLAW AGENT RUNTIME (C11, 349KB core) --+                  |
|  |                                              |                  |
|  |  Agent Loop ---------> LLM Provider ------+  |                  |
|  |  (planner, dispatcher)  (50+ options)     |  |                  |
|  |       |                   - Claude API    |  |                  |
|  |       |                   - GPT-4 API     |  |                  |
|  |       |                   - Gemini API    |  |                  |
|  |       |                   - Ollama (local)|  |                  |
|  |       |                   - llama.cpp     |  |                  |
|  |       v                                   |  |                  |
|  |  47 Tools    Memory      Security         |  |                  |
|  |  (shell,     (SQLite,    (pairing,        |  |                  |
|  |   web,       vectors,    sandbox,         |  |                  |
|  |   file,      FTS5,       ChaCha20,        |  |                  |
|  |   git,       hybrid)     Landlock)        |  |                  |
|  |   hw...)                                  |  |                  |
|  |                                           |  |                  |
|  +-------------------------------------------+  |                  |
|       ^                    |                     |                  |
|       | text               | text                |                  |
|       |                    v                     |                  |
|  +-- SONATA VOICE PIPELINE (Rust/C, ~260M) -+   |                  |
|  |                                           |   |                  |
|  |  sc_channel_voice_t (new voice channel)   |   |                  |
|  |       ^                    |               |   |                  |
|  |       | user text          | response text |   |                  |
|  |       |                    v               |   |                  |
|  |  Sonata STT           Sonata TTS          |   |                  |
|  |  (Conformer CTC)      (AdaIN + Emotion    |   |                  |
|  |       ^                + Nonverbal)        |   |                  |
|  |       |                    |               |   |                  |
|  |  Sonata Codec          CFM Decoder         |   |                  |
|  |  (audio->tokens)       (tokens->mel)       |   |                  |
|  |       ^                    |               |   |                  |
|  |       |                    v               |   |                  |
|  |  CAM++ Speaker         iSTFT Vocoder       |   |                  |
|  |  (voice identity)      (mel->audio)        |   |                  |
|  |       ^                    |               |   |                  |
|  |       |                    v               |   |                  |
|  +------|--------------------|-----------+   |                  |
|         |                    |               |                  |
|    Microphone            Speaker             |                  |
|    (24kHz in)            (24kHz out)         |                  |
+===================================================================+
```

### How the LLM Fits In the Middle

The data flow for a voice conversation:

```
1. User speaks into microphone
   |
2. Sonata Codec encodes audio -> tokens (5ms)
   |
3. Sonata STT converts tokens -> text via CTC (80ms)
   |
4. sc_channel_voice_t delivers text to SeaClaw agent loop
   |
5. SeaClaw agent dispatches to LLM provider:
   |   - Cloud: Claude API, GPT-4 API, Gemini API (~200-500ms)
   |   - Local: llama.cpp, Ollama (~100-300ms on M-series)
   |   LLM has access to SeaClaw's 47 tools, memory, context
   |
6. LLM response text flows back through voice channel
   |
7. Sonata TTS converts text -> audio tokens with:
   |   - AdaIN speaker conditioning (from CAM++)
   |   - Emotion style tokens (from Chatterbox)
   |   - Nonverbal tags (from Dia)
   |
8. CFM Decoder converts tokens -> mel spectrogram (25ms)
   |
9. iSTFT converts mel -> 24kHz audio (<1ms)
   |
10. Audio plays through speaker
```

### Full-Duplex with LLM in the Middle

Full-duplex requires Sonata to handle audio streaming independently of the LLM's response time:

```
SONATA LAYER (continuous, 20ms frames):
  User audio -> Codec -> STT (streaming partial results)
  While waiting for LLM:
    Generate backchannels: [hmm], [right], [uh-huh]
    Generate silence tokens
    Mirror emotional state
  When LLM response arrives:
    Smoothly transition from backchannel to response
    Apply AdaIN + emotion + nonverbal as generating

SEACLAW LAYER (request-response with streaming):
  Receive completed user utterance text
  Route to LLM provider
  Stream response tokens back to Sonata TTS
  Handle tool calls inline (pause TTS, execute, resume)

STREAMING BRIDGE (new):
  sc_stream_callback_t from LLM provider
  -> Feeds tokens to Sonata TTS in real-time
  -> TTS generates audio as tokens arrive (no wait for full response)
```

### Three Integration Points in SeaClaw

#### 1. `sc_channel_voice_t` — Voice as a Channel

Sonata becomes a new channel type (like Telegram, Discord, but for voice):

```c
/* include/seaclaw/channel_voice.h */

typedef struct sc_channel_voice_config {
    /* Sonata model paths */
    const char *codec_model_path;
    const char *stt_model_path;
    const char *tts_model_path;
    const char *cam_model_path;
    const char *cfm_model_path;

    /* Voice settings */
    const char *speaker_id;         /* Voice identity for TTS */
    float emotion_exaggeration;     /* 0.0-2.0 (Chatterbox-style) */
    const char *default_emotion;    /* "neutral", "warm", "excited" */

    /* Audio I/O */
    uint32_t sample_rate;           /* 24000 */
    uint32_t channels;              /* 1 (mono) */
    bool enable_full_duplex;        /* Enable dual-stream */
    bool enable_backchanneling;     /* Generate hmm/right while waiting */

    /* Hardware */
    bool use_metal;                 /* Metal GPU acceleration */
    bool use_ane;                   /* Apple Neural Engine */
} sc_channel_voice_config_t;

/* Channel vtable implementation */
static const sc_channel_vtable_t voice_channel_vtable = {
    .init     = sc_channel_voice_init,
    .send     = sc_channel_voice_send,     /* TTS: text -> audio -> speaker */
    .receive  = sc_channel_voice_receive,  /* STT: mic -> audio -> text */
    .stream   = sc_channel_voice_stream,   /* Streaming TTS as LLM generates */
    .deinit   = sc_channel_voice_deinit,
    .get_name = sc_channel_voice_name,     /* "voice" */
};
```

#### 2. Replace `sc_voice_stt/tts` — Native Voice Functions

Replace SeaClaw's current API-based voice functions with Sonata native:

```c
/* Current SeaClaw voice (API-based, ~500ms latency): */
sc_voice_stt(alloc, &config, audio, len, &text, &text_len);  /* Groq Whisper API */
sc_voice_tts(alloc, &config, text, len, &audio, &audio_len);  /* OpenAI TTS API */

/* New Sonata voice (native, ~80ms STT, ~100ms TTS): */
sonata_stt(pipeline, audio, len, &text, &text_len);  /* On-device Conformer CTC */
sonata_tts(pipeline, text, len, speaker, emotion, &audio, &audio_len);  /* On-device CFM */
```

#### 3. Streaming Provider Callback — Real-Time TTS

Connect SeaClaw's LLM streaming to Sonata's streaming TTS:

```c
/* SeaClaw streams LLM tokens -> Sonata generates audio in real-time */

void on_llm_token(const char *token, void *ctx) {
    sonata_tts_stream_token(ctx, token);  /* Feed token to Sonata TTS */
    /* Sonata generates audio for this token immediately */
    /* Audio plays while next token is still being generated */
}

sc_chat_request_t req = {
    .messages = messages,
    .model = "claude-sonnet-4-20250514",
    .stream_callback = on_llm_token,
    .stream_ctx = sonata_pipeline,
};
provider->vtable->stream_chat(provider, alloc, &req, &response);
```

### Sonata's Revised Role (With SeaClaw)

With SeaClaw providing the agent brain, Sonata v2 focuses purely on voice excellence:

| Component | Params | Purpose | Status |
|-----------|--------|---------|--------|
| **Sonata Codec** | ~25M | Audio <-> discrete tokens | New (train) |
| **CAM++ Speaker Encoder** | 7.18M | Voice identity -> AdaIN embedding | New (train, v3 in progress) |
| **Sonata STT** | ~100M | Streaming Conformer CTC (text output) | Enhanced from v1 |
| **Sonata TTS** | ~100M | Text -> audio tokens (AdaIN + emotion + nonverbal) | New (train) |
| **CFM Decoder** | ~35M | Audio tokens -> mel spectrogram | New (train) |
| **iSTFT Vocoder** | ~0 | Mel -> 24kHz waveform (vDSP) | Keep from v1 |
| **Full-Duplex Controller** | ~0 | Streaming orchestration, backchanneling | New (code) |
| **Total Sonata** | **~260M** | Complete voice pipeline | |

**Note:** The 400M "Core Transformer" from the original design is replaced by the combination of:
- Sonata STT (~100M) for speech understanding
- SeaClaw + LLM for reasoning
- Sonata TTS (~100M) for speech generation

This is more efficient — each component is optimized for its task.

### Dual-Path: Fast (STS) and Rich (Text)

```
FAST PATH (STS, ~160ms):
  Audio -> Codec -> Lightweight response model -> Codec -> CFM -> Audio
  For: backchannels, simple responses, emotional mirroring
  No SeaClaw/LLM involved. Pure Sonata.

RICH PATH (Text, ~300-600ms):
  Audio -> Codec -> STT -> Text -> SeaClaw -> LLM -> Text -> TTS -> Audio
  For: complex reasoning, tool use, factual answers, memory retrieval
  Full SeaClaw agent intelligence.

DECISION LOGIC:
  If user utterance is short (<3 words) -> Fast path (backchannel)
  If user utterance needs tools -> Rich path
  If user utterance is emotional -> Fast path (emotional mirror)
  If user utterance is a question -> Rich path
  If user is interrupting -> Fast path (acknowledge)
  Default -> Rich path
```

### Build Integration

```cmake
# SeaClaw CMakeLists.txt additions

# Link Sonata as static library
option(SC_ENABLE_SONATA "Enable native Sonata voice pipeline" ON)

if(SC_ENABLE_SONATA)
    # Sonata Rust crates compiled via cargo
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/libsonata.a
        COMMAND cargo build --release --manifest-path ${SONATA_DIR}/Cargo.toml
        COMMENT "Building Sonata voice pipeline"
    )

    add_library(sonata STATIC IMPORTED)
    set_target_properties(sonata PROPERTIES
        IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/libsonata.a
    )

    # Link Sonata + Apple frameworks
    target_link_libraries(seaclaw PRIVATE
        sonata
        "-framework Accelerate"
        "-framework Metal"
        "-framework CoreAudio"
    )

    target_compile_definitions(seaclaw PRIVATE SC_HAS_SONATA=1)
endif()
```

### Binary Size Budget

| Component | Size (stripped, LTO) |
|-----------|---------------------|
| SeaClaw core | ~349 KB |
| Sonata voice code | ~2 MB |
| Sonata models (quantized 4-bit) | ~40-50 MB |
| **Total binary** | **~50 MB** |
| **Total with llama.cpp** | **~55 MB** |

For comparison: a Python venv with PyTorch + transformers is ~2 GB.

### Fully On-Device Scenario

With llama.cpp as the LLM provider, the entire system runs on-device with zero cloud:

```
Hardware: Apple M1 Pro (or better)
Memory:  ~4 GB (Sonata models + 7B LLM quantized)

User speaks -> Sonata STT (on-device, 80ms)
           -> SeaClaw agent (on-device, <1ms routing)
           -> llama.cpp (on-device, 100-300ms for 7B model)
           -> Sonata TTS (on-device, 100ms)
           -> Audio out

Total latency: ~300-500ms (fully on-device, no cloud)
Privacy: 100% — no data leaves the device
```

---

## 11. Implementation Language Strategy

### Target Structure

```
sonata-v2/
  crates/
    sonata-codec/      # Codec encoder/decoder (Rust, candle)
    sonata-cam/        # CAM++ speaker encoder (Rust, candle)
    sonata-core/       # Unified transformer (Rust, candle)
    sonata-cfm/        # CFM decoder (Rust, candle)
    sonata-pipeline/   # Orchestration + streaming (Rust)
    sonata-napi/       # Node-API bindings (Rust, napi-rs)

  c_dsp/
    sonata_istft.c     # iSTFT vocoder (vDSP, 5000x RT)
    mel_spectrogram.c  # 80-bin log-mel (vDSP)
    audio_converter.c  # Sample rate conversion (vDSP)

  train/
    train_codec.py     # Codec training (PyTorch)
    train_core.py      # Core Transformer training (PyTorch)
    train_cam.py       # CAM++ training (PyTorch)
    train_cfm.py       # CFM training (PyTorch)
    export_onnx.py     # Export to ONNX (intermediate)
    convert_candle.py  # Convert ONNX -> candle safetensors

  models/
    codec/             # Trained codec weights
    core/              # Trained core transformer weights
    cam/               # Trained CAM++ weights
    cfm/               # Trained CFM weights

  Cargo.toml           # Workspace manifest
  Makefile             # Build orchestration
```

### Language Rationale

**ALL ML in Rust (candle):**
- Current codebase already has 200K+ LOC Rust via candle
- candle provides Metal GPU, NEON SIMD, custom kernel support
- Type safety eliminates 93 unsafe blocks from C FFI
- Single build system (cargo) instead of 1039-line Makefile + cargo

**Keep C for DSP only (3 files):**
- `sonata_istft.c` — Apple's Accelerate/vDSP has the best FFT implementation
- `mel_spectrogram.c` — vDSP FFT for mel spectrogram computation
- `audio_converter.c` — vDSP for sample rate conversion
- These are thin wrappers around Apple frameworks, not complex logic

**FFI boundary: 3 functions only:**
```rust
extern "C" {
    fn istft_process(mel: *const f32, len: usize, output: *mut f32) -> i32;
    fn mel_compute(audio: *const f32, len: usize, mel: *mut f32) -> i32;
    fn audio_convert(input: *const f32, in_rate: u32, out_rate: u32, ...) -> i32;
}
```

**Training in Python (PyTorch):**
- Training is done on GPU clusters (A100s) using PyTorch
- Weights are exported to safetensors format
- Loaded into Rust candle for inference
- No Python in production inference path

---

## 12. Comparison Matrix

| Metric | Sonata v1 | **Sonata v2** | Moshi | GPT-4o RT | CosyVoice | Kokoro |
|--------|-----------|---------------|-------|-----------|-----------|--------|
| **Latency** | 320ms | **160ms** | 160ms | ~200ms | 150ms | N/A |
| **Full-Duplex** | Via EOU | **Native** | Native | Native | No | No |
| **STS** | No | **Yes** | Yes | Yes | No | No |
| **Voice Clone** | 3s ECAPA | **3s CAM++ (0.56% EER)** | No | No | 3s | No |
| **Emotion Control** | No | **Yes (scalar 0-2)** | No | No | No | No |
| **Nonverbal Tags** | No | **24 tags** | Limited | Yes | No | No |
| **On-Device** | Yes | **Yes** | Partial | No (cloud) | No (cloud) | Yes |
| **Native C/Rust** | Yes | **Yes** | Python | Proprietary | Python | Python |
| **Params** | ~280M | **~470M** | 1B | Unknown | 500M | 82M |
| **AdaIN Voice** | No | **Every layer** | No | Unknown | No | Yes |
| **Streaming** | Yes | **Yes (causal CFM)** | Yes | Yes | Yes | No |
| **Open Source** | No | No | Yes | No | Yes | Yes |

### Unique Advantages of Sonata v2

1. **Only system with ALL of**: full-duplex + emotion control + nonverbal tags + on-device + native C/Rust
2. **AdaIN at every layer**: Better voice identity preservation than any competitor
3. **Emotion exaggeration scalar**: Unique control not found in any other system
4. **CAM++ speaker encoder**: Best speaker verification accuracy (0.56% EER) with fewest params (7.18M)
5. **On-device with Metal GPU**: No cloud dependency, privacy-preserving

---

## 13. Migration from v1

### What Gets Preserved

| v1 Component | v2 Status | Notes |
|-------------|-----------|-------|
| iSTFT Vocoder (vDSP) | **Keep** | Still 5000x RT, unchanged |
| Mel spectrogram (vDSP) | **Keep** | Same 80-bin computation |
| Audio converter (vDSP) | **Keep** | Same sample rate conversion |
| NAPI bindings pattern | **Keep** | Same napi-rs approach |
| Test infrastructure | **Keep** | Same test patterns |
| Speculative prefill logic | **Adapt** | Keep concept, integrate into Core |
| Build system (Makefile) | **Replace** | Move to pure Cargo workspace |
| Conformer STT | **Replace** | Absorbed into Core Transformer |
| Sonata LM | **Replace** | Absorbed into Core Transformer |
| Sonata Flow | **Replace** | Replaced by CFM Decoder |
| ECAPA-TDNN speaker encoder | **Replace** | Replaced by CAM++ |
| Fused EOU (5-signal) | **Deprecate** | Keep as fallback, model learns turn-taking |
| Pipeline state machine (C) | **Replace** | New Rust pipeline orchestration |
| FFI layer (100 exports) | **Reduce** | Down to 3 C DSP functions |

### Migration Strategy

1. **Build v2 alongside v1** — don't delete v1 until v2 is proven
2. **Share vDSP code** — iSTFT, mel, audio converter work for both
3. **Incremental cutover** — start with TTS (easiest), then STT, then full-duplex
4. **A/B testing** — run v1 and v2 in parallel on test calls
5. **v1 maintenance mode** — bug fixes only, no new features

---

## 14. Risk Assessment

### Risk 1: Training Compute Cost

**Likelihood**: Medium
**Impact**: Could delay timeline by months
**Mitigation**:
- Start with smaller model (200M) to validate architecture
- Use mixed precision (bf16) training throughout
- Leverage spot instances for pre-training (60-70% cost reduction)
- Phase training — don't train everything at once

### Risk 2: Codec Quality Insufficient

**Likelihood**: Low-Medium
**Impact**: Everything downstream degrades
**Mitigation**:
- Start codec training first (it blocks everything)
- Set hard quality gates: PESQ > 4.0 or iterate
- Fall back to proven architectures (SoundStream, Encodec) if custom codec struggles
- 50Hz frame rate gives quality headroom vs 12.5Hz

### Risk 3: Full-Duplex Training Data Scarcity

**Likelihood**: Medium
**Impact**: Full-duplex quality suffers
**Mitigation**:
- Fisher Corpus has 2000h of real conversations
- Synthesize dual-stream data from dialogue transcripts using v1 TTS
- Start with half-duplex, add full-duplex incrementally
- Keep v1 EOU as fallback path

### Risk 4: On-Device Performance (Apple M-series)

**Likelihood**: Low
**Impact**: Latency target missed
**Mitigation**:
- 400M params fits comfortably in M1 Pro+ unified memory
- Current v1 runs 280M params on Metal with room to spare
- Aggressive quantization (4-bit) can reduce to ~200MB memory
- KV-cache optimization with GQA (4 KV heads, not 16)

### Risk 5: Paradigm Shifts During Development

**Likelihood**: Medium-High
**Impact**: Architecture becomes outdated
**Mitigation**:
- Modular crate structure allows component replacement
- Codec is standard — compatible with future models
- Dual-stream architecture is the current frontier (2025-2026)
- Design for upgradability, not permanence
- 6-9 month timeline limits exposure

---

## Appendix A: SOTA Model Repositories (Reference)

All cloned to `/Users/sethford/Documents/pocket-voice/research/`:

| Repo | Model | Key Innovation to Study |
|------|-------|----------------------|
| `kokoro/` | Kokoro TTS 82M | AdaIN conditioning, iSTFTNet decoder |
| `F5-TTS/` | F5-TTS 335M | DiT + CFM flow matching |
| `CosyVoice/` | CosyVoice 500M | Causal CFM, LLM-based TTS, streaming |
| `dia/` | Dia 1.6B | 24 nonverbal tags, multi-speaker dialogue |
| `chatterbox/` | Chatterbox ~350M | Emotion exaggeration scalar |
| `piper/` | Piper 20-80MB | C++ ONNX on-device patterns |
| `sherpa-onnx/` | sherpa-onnx | C/C++ framework patterns (DO NOT USE as dependency) |
| `wespeaker/` | CAM++ 7.18M | Context-aware masking, 0.56% EER |
| `fish-speech/` | Fish Speech | SOTA open source TTS patterns |
| `Orpheus-TTS/` | Orpheus TTS | Human-sounding speech patterns |

---

## Appendix B: Key Metrics to Track

| Metric | Target | How to Measure |
|--------|--------|---------------|
| E2E latency (STS) | < 160ms | Time from last user audio frame to first system audio frame |
| E2E latency (TTS) | < 100ms | Time from text input to first audio frame |
| Speaker EER | < 0.6% | VoxCeleb1-O evaluation set |
| TTS PESQ | > 3.8 | PESQ-WB on LibriTTS-R test set |
| Codec PESQ | > 4.0 | PESQ-WB on reconstruction quality |
| Emotion accuracy | > 80% | Emotion classification on IEMOCAP |
| Full-duplex naturalness | > 4.0 MOS | Human evaluation of turn-taking quality |
| Memory (inference) | < 2GB | M1 Pro with 4-bit quantization |
| Model size (disk) | < 1GB | Quantized weights + codec |

---

*Document created: March 2, 2026*
*Authors: Seth Ford + Claude*
*Status: Approved — ready for implementation planning*
*Next step: Invoke writing-plans skill to create detailed implementation plan*
