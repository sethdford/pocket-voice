# Sonata S2S — Unified Speech-to-Speech Model Design

**Date**: 2026-03-05
**Status**: DESIGN (no implementation)
**Scope**: Complete migration path from STT→LM→TTS pipeline to unified speech-to-speech model

---

## Executive Summary

Moshi (Kyutai Labs) demonstrated that speech-to-speech models can rival pipelined systems while enabling full-duplex dialogue with natural interruptions and overlaps. Sonata currently achieves 320ms latency via a sequential pipeline (STT → LLM → TTS). A unified Sonata S2S model would:

1. **Reduce latency**: Eliminate stage boundaries, enable joint acoustic-semantic planning
2. **Enable full-duplex**: Model speaker/user streams in parallel (no explicit turn boundaries)
3. **Improve coherence**: Semantic-to-acoustic generation learned jointly rather than cascaded
4. **Simplify deployment**: One model instead of three (STT + LM + TTS)

**Trade-off**: Unified models are harder to audit/interpret and more computationally expensive during training. This design proposes **phased migration** rather than immediate replacement — S2S coexists with the pipeline for 6+ months, reducing risk.

---

## Moshi Architecture Analysis

### Overview

Moshi is a 7B speech-text foundation model combining:

- **Helium**: 7B text LLM (32 layers, 4096 d_model)
- **Depth Transformer**: Hierarchical audio codebook modeling (6 layers, 1024 d_model)
- **Temporal Transformer**: Dual-stream audio-text modeling (32 layers, 4096 d_model)
- **Mimi codec**: 12.5Hz neural audio encoder with 8 quantizers + semantic distillation

### Key Innovation: Dual-Stream Processing

Unlike Moshi-as-agent (Moshi-1B), which processes text-to-speech unidirectionally, Sonata S2S should adopt Moshi's **multi-stream autoregressive architecture**:

```
User Audio Stream (semantic + 7 acoustic codes)
        ↓
    [Temporal Transformer]
        ↓
Text + Semantic + Acoustic Codes (interleaved)
        ↓
    [Depth Transformer]
        ↓
Assistant Audio Stream (semantic + 7 acoustic codes)
```

**Key design elements:**

1. **Interleaved token streams** at audio framerate (12.5 Hz):
   - Text tokens (sparse, ~5-10 Hz)
   - User audio: 1 semantic + 7 acoustic quantizer levels
   - Assistant audio: 1 semantic + 7 acoustic quantizer levels
   - Total: K=17 parallel streams per timestep

2. **Acoustic delay (τ)**:
   - Acoustic tokens conditioned on semantic predictions 1-2 frames prior
   - Stabilizes generation, reduces entropy of acoustic predictions

3. **Inner Monologue** (optional, not critical):
   - Text tokens generated as prefix to audio tokens
   - Improves linguistic quality but adds latency
   - Can be omitted in Sonata v1 for speed

4. **Causal attention throughout**:
   - Streaming-compatible, no lookahead
   - FlashAttention for efficiency

### Codec: Mimi vs Sonata

| Component       | Mimi              | Sonata (12.5Hz)        | Notes                                 |
| --------------- | ----------------- | ---------------------- | ------------------------------------- |
| Frame rate      | 12.5 Hz           | 12.5 Hz                | Sonata's upcoming codec_12hz.py       |
| Codebooks       | 8 RVQ             | 1 FSQ + acoustic laten | Sonata uses Flow for detail, Mimi RVQ |
| Semantic tokens | Distilled (WavLM) | Conformer CTC (50Hz)   | Need to retrain at 12.5Hz             |
| Bitrate         | 1100 bps          | 150 bps (semantic)     | Sonata lower bitrate                  |
| Encoder latency | 80ms              | 80ms                   | Same frame delay                      |

---

## Current Sonata Architecture

### State Machine

```
Listening → Recording → Processing → Streaming → Speaking → Listening
                │
                └───────────  Barge-in → Listening
```

### Data Flow

```
┌─ User Audio (48kHz) ─────────────────────────────────────────┐
│                                                               │
▼                                                               │
Resample (48→16kHz)                                             │
│                                                               │
▼                                                               │
Noise Gate + Preemphasis                                        │
│                                                               │
▼                                                               │
[STT: Conformer CTC] ────────→ Transcription + confidence       │
│                                                               │
▼                                                               │
[VAD + EOU Detection]                                           │
│                                                               ▼
├──→ [LLM Client: Claude/Gemini/Local Llama]                  Barge-in
│         ↓                                                  detection
│    LLM Response Tokens (streaming)
│         │
│         ▼
│    [Sonata LM] ──→ Semantic Tokens (50 Hz)
│         │
│         ▼
│    [Sonata Flow] ──→ Acoustic Latents
│         │
│         ▼
│    [iSTFT Decoder] ──→ Audio (24kHz)
│         │
└──────→  Resample (24→48kHz)
           │
           ▼
      [Post-processing: pitch, LUFS, breath, watermark, spatial]
           │
           ▼
      Speaker (48kHz)
```

### Latency Breakdown (Current)

| Stage                     | Latency    | Notes                                 |
| ------------------------- | ---------- | ------------------------------------- |
| **Input**                 | 80ms       | 48→16kHz resample + noise gate        |
| **STT (Conformer)**       | 100ms      | Streaming, EOU detection adds +40ms   |
| **LLM (Cloud/Local)**     | 500-2000ms | Claude/Gemini/Llama, first token time |
| **Sonata LM (50 Hz)**     | 100ms      | 5 semantic tokens at 50 Hz            |
| **Sonata Flow (4 steps)** | 80ms       | ODE solver on GPU                     |
| **iSTFT + Post-proc**     | 60ms       | Pitch shift, LUFS, breath             |
| **Output resample**       | 20ms       | 24→48kHz                              |
| **Speaker buffering**     | 80ms       | Audio queue depth                     |
| **Total (pipeline)**      | 320-1000ms | Dominated by LLM                      |
| **Total (full-duplex)**   | 160-320ms  | Overlapped stages in parallel         |

### Key Strengths

1. **Modularity**: Each stage independently tunable
2. **Interpretability**: Transcription visible, LLM response auditable
3. **Flexibility**: Can swap STT (Conformer/Kyutai), LLM (Claude/Gemini/local), TTS (Piper/Sonata)
4. **Proven latency**: 320ms end-to-end with parallelization

### Key Weaknesses

1. **Error cascading**: STT → LLM → TTS creates cumulative errors
2. **Semantic gap**: LLM never sees acoustic features (emotion, rate, prosody)
3. **Redundant processing**: User audio tokenized twice (STT + semantic encoder)
4. **Barge-in latency**: ~500ms delay before TTS stops (LLM generation in flight)
5. **No natural interruptions**: Model trained on turn-taking, not overlaps

---

## Gap Analysis: Moshi vs Sonata

### What Moshi Does That Sonata Doesn't

| Capability                | Moshi                             | Sonata Current   | Sonata Need      |
| ------------------------- | --------------------------------- | ---------------- | ---------------- |
| **Full-duplex**           | Simultaneous user/assistant audio | No (turn-taking) | New model        |
| **Direct audio→audio**    | User audio → assistant audio      | Via text proxy   | End-to-end train |
| **Acoustic features**     | Emotion, rate, prosody preserved  | Lost to LLM      | Joint model      |
| **Overlap handling**      | Models interruption naturally     | Hard cutoff      | Semantic fusion  |
| **Streaming inference**   | Causal, no buffering              | Partial (flow)   | Full causal      |
| **Semantic distillation** | WavLM→Mimi distilled              | Conformer CTC    | Contrastive loss |

### What Sonata Does Better

| Capability           | Sonata                          | Moshi              | Advantage         |
| -------------------- | ------------------------------- | ------------------ | ----------------- |
| **Codec efficiency** | 150 bps (Flow-predicted latent) | 1100 bps (8 RVQ)   | 7.3x fewer tokens |
| **Inference speed**  | 13x realtime (STT)              | 5x realtime        | Faster on mobile  |
| **LLM quality**      | Claude 3.5 (via API)            | 7B Helium          | Better reasoning  |
| **On-device TTS**    | Fully local Sonata Flow         | Moshi TTS remote   | Privacy + latency |
| **Speaker encoder**  | Trained in-house (Wave 2)       | Included (generic) | Better voice id   |

---

## Proposed Architecture: Sonata S2S

### Design Goals

1. **Reuse existing components** where possible (codec_12hz, speaker encoder, MEL processing)
2. **Maintain modularity** — can fall back to pipeline if S2S fails
3. **Streaming-first** — causal, no lookahead
4. **Metal-optimized** — same acceleration strategy as current models
5. **Incremental training** — reuse Sonata LM/Flow weights as initialization

### Model Architecture

#### High-Level Data Flow

```
User Audio (24kHz) ─────────→ [codec_12hz encoder] ─────→ Semantic + Acoustic Codes
                                                              │
            ┌─────────────────────────────────────────────────┤
            │                                                  │
            ▼                                                  ▼
    [Optional: Text Buffer] ──→ Text Tokenizer ──→ [Causal] Self-Attention
            │                                          Block
            │                 ┌───────────────────────────┘
            │                 │
            ▼                 ▼
    [Audio Embedding] ────────[Sonata S2S Temporal Transformer]
            │                 │ (32 layers, 4096 d_model)
            │                 │
            └─────────────────┤
                              │
                              ▼
                    [Context Vector]
                              │
                              ▼
                    [Depth Transformer]
                    (6 layers, 1024 d_model)
                              │
                              ▼
                    [Audio RVQ / Acoustic Codes]
                              │
                              ▼
                    [codec_12hz decoder]
                              │
                              ▼
                    Assistant Audio (24kHz)
```

#### Layer Dimensions (Sonata S2S-3B)

To keep Apple Silicon friendly while improving over Sonata LM (241M):

| Component               | Dimension | Layers | Heads | FFN | Params   | Notes                               |
| ----------------------- | --------- | ------ | ----- | --- | -------- | ----------------------------------- |
| **Text Tokenizer**      | input     | —      | —     | —   | 32M      | SentencePiece, 32K vocab            |
| **Audio Embedder**      | 1024      | —      | —     | —   | 256M     | (semantic + 7\*acoustic) → 1024-dim |
| **Temporal Trans**      | 1024      | 16     | 8     | 4K  | 1.2B     | Causal, GQA (8 groups)              |
| **Depth Transformer**   | 512       | 6      | 8     | 2K  | 180M     | Per-quantizer, depthwise            |
| **Codebook Embeddings** | 512       | —      | —     | —   | 1M       | 8 codebooks × 2048 vocab ea         |
| **LM Head (text)**      | 32K       | —      | —     | —   | 32M      | If inner monologue enabled          |
| **TOTAL**               | —         | —      | —     | —   | **3.0B** | (vs Moshi 7B, vs Sonata LM 241M)    |

**Why 3B instead of 7B?**

- Apple Silicon (M4 Max) has 8GB on-device; 3B = 6GB weights + 1GB intermediate
- 7B would require GPU offload (slower on mobile)
- Sonata already delegates reasoning to Claude API; S2S only needs dialogue flow modeling
- Trade: Lower zero-shot quality, but fine-tuning on target domain recovers it

**Why GQA with 8 groups?**

- Reduces KV cache from 4096×2 to 4096×0.5 → 75% smaller
- 3000-token context = 6MB (vs 24MB with MHA)

#### Sequence Format

At each timestep, process 17-token streams (interleaved):

```
[text_t] [user_sem_t] [user_a0_t] [user_a1_t] ... [user_a7_t]
         [asst_sem_t] [asst_a0_t] [asst_a1_t] ... [asst_a7_t]
```

**Text tokens**: Sparse, generated on-demand when LLM has new ideas (Optional for v1)
**Semantic**: Distilled from user audio via contrastive loss (WavLM-based)
**Acoustic**: Residual quantizer codes, 0-7 (fine→coarse detail)
**Causal masking**: User tokens can attend to prior user audio; assistant audio can attend to all user + prior assistant

#### Depth Transformer Design

Unlike Moshi's single Depth transformer, use **depthwise parametrization**:

```
For each of 8 quantizer levels i:
    DecoderBlock_i (1024→1024):
        LayerNorm → MultiHeadAttn(heads=8, context=32 frames)
                 → FFN(1024→4096→1024)

Share: embeddings, attention structure
Separate: weights for each i
```

This reduces parameters vs 8 separate decoders (1.2B → 180M) while learning quantizer-specific distributions.

#### Acoustic Delay (τ)

Sonata S2S should use **τ=2 frames (~160ms)** delay between semantic and acoustic tokens:

```
[Gen step t]:
    Input: user_sem_t, user_acoustic_{t-2..t}, asst_sem_{t-1}
    Output: asst_sem_t, asst_acoustic_{t}

[Inference]:
    Buffer user acoustic codes 160ms before decoding
    Reduces output entropy by conditioning on more stable semantic context
```

#### Streaming Inference

**Phase 1 (Encoder):**

- Process user audio chunks (160ms at 12.5Hz = 2 frames)
- Encoder runs offline (80ms latency)
- Cache hidden states in ring buffer

**Phase 2 (Temporal Transformer):**

- Causal attention: new frame attends to [all prior user] + [all prior assistant up to τ frames ago]
- Generate semantic token (deterministic top-1 or low-temp sampling)
- Enqueue for Depth Transformer

**Phase 3 (Depth Transformer):**

- Per-quantizer decoding from coarse (q=0) to fine (q=7)
- Codebook lookup for each level
- Each level recombined at decoder

**Phase 4 (Decoder):**

- codec_12hz decoder outputs 1920 PCM samples per frame (80ms)
- Apply post-processing (pitch, LUFS, spatial) → playback

**Total latency target**: 200ms (matching Moshi's practical latency)

```
Encoder:       80ms (offline)
Temporal:      40ms (1-3 forward passes with caching)
Depth:         30ms (8 parallel codebook decoders)
Decoder:       20ms (iSTFT-like)
Buffering:    30ms
─────────────────
Total:        200ms
```

---

## Training Plan

### Data Requirements

| Dataset            | Hours     | Purpose                                 | Source            |
| ------------------ | --------- | --------------------------------------- | ----------------- |
| Fisher + CallHome  | 4,000     | Multi-speaker conversational (base)     | LDC license       |
| Spoken Wikipedia   | 2,000     | Diverse domains, clear speech           | Public download   |
| Expresso           | 500       | Code-switching, social media (optional) | LDC (paid)        |
| Synthetic (Claude) | 10,000    | Prompt: [topic] → generate + TTS        | Use Sonata LM+TTS |
| In-domain custom   | 1,000     | Customer service calls, target domain   | Collect/license   |
| **Total**          | **17.5K** |                                         |                   |

**Why 17.5K hours?**

- Moshi trained on 7M hours but included 20K hours synthetic
- Sonata S2S-3B is smaller (3B vs 7B), needs less data
- 17.5K is achievable in 3 months with licensing + synthesis

### Training Stages

#### Stage 1: Codec Pre-training (Existing: codec_12hz.py)

- Reconstruct 24kHz audio from semantic + acoustic codes
- Already in progress; assume checkpoint by 2026-03-15
- **GPU hours**: 1,000 (H100)
- **Duration**: 2 weeks

#### Stage 2: Semantic Distillation

Train semantic encoder (Conformer) at 12.5Hz to match WavLM embeddings:

```python
# Target: semantic_t ≈ WavLM(audio[t:t+2048])[pooled]
# Loss: contrastive (SimCLR-style) + MSE

Conformer(audio @ 24kHz)
    → avg pool to 12.5Hz
    → linear projection to 1024-dim
    → match WavLM(audio).mean()
```

- **Duration**: 3 weeks (40K steps on 4× H100)
- **GPU hours**: 2,000
- **Checkpoint**: Save every 10K steps for ablation

#### Stage 3: Pre-training Temporal Transformer (unsupervised)

Causal language modeling on audio codes:

```
Input:  [user_audio_codes (prior) + text (sparse)]
Target: [user_audio_codes (current) + asst_audio_codes (current)]
Loss:   Cross-entropy on each code stream
```

Run on 17.5K hours of conversational audio:

- **Batch size**: 256 (32 × 8 GPUs)
- **Learning rate**: 6e-4 (cosine decay)
- **Steps**: 500K (18 days of training, ~100 H100 GPU-hours per 1K steps)
- **Total GPU hours**: 50,000
- **Checkpoints**: Every 50K steps (10 per training)
- **Duration**: 3 weeks real time (parallelized)

#### Stage 4: Supervised Fine-tuning (Phoneme Alignment)

Align audio codes to phoneme boundaries using Montreal Forced Aligner:

```
Input:  user_audio_codes
Target: asst_audio_codes (at aligned phoneme boundaries)
Loss:   Supervised cross-entropy + KL divergence penalty
```

- **Data**: 1,000 hours manually transcribed (phoneme-aligned)
- **Duration**: 1 week (50K steps)
- **GPU hours**: 5,000
- **Learning rate**: 1e-4 (fine-tuning)

#### Stage 5: SFT on Dialogue Pairs

Fine-tune on curated {user_audio, asst_audio_text, asst_audio} triplets:

```
Input:  user_audio_codes + asst_text (for context/reasoning)
Target: asst_audio_codes
Loss:   Supervised CE + style matching (pitch, rate, emotion)
```

Use a mix of:

- **Real conversations**: 500 hours (Fisher subset, manually reviewed)
- **Synthetic pairs**: 10K hours (Claude responses + Sonata TTS)
- **Domain-specific**: 1K hours (custom customer service calls)

- **Duration**: 2 weeks (100K steps)
- **GPU hours**: 10,000
- **Learning rate**: 5e-5 (conservative)
- **Warmup**: 2K steps

#### Stage 6: Full-Duplex Fine-tuning (Optional, Phase 2)

If enabling true full-duplex (user + assistant speaking simultaneously):

```
Input:  [user_audio_codes, asst_audio_codes_prior, text_partial]
Target: asst_audio_codes (with user interruptions)
Loss:   Cross-entropy + overlap tolerance (let model learn interruption recovery)
```

- **Data**: Synthetic overlaps (generate, then layer recordings)
- **Duration**: 2 weeks
- **GPU hours**: 10,000

### Total Training Cost (Stages 1-5)

| Stage                   | GPU Hours  | Duration     | Hardware |
| ----------------------- | ---------- | ------------ | -------- |
| 1. Codec                | 1,000      | 2 weeks      | H100 × 4 |
| 2. Semantic distill     | 2,000      | 3 weeks      | H100 × 4 |
| 3. Pre-training         | 50,000     | 3 weeks      | H100 × 8 |
| 4. Supervised fine-tune | 5,000      | 1 week       | H100 × 4 |
| 5. Dialogue SFT         | 10,000     | 2 weeks      | H100 × 4 |
| **TOTAL**               | **68,000** | **11 weeks** | —        |

**Cost estimate** (H100 at $2.50/hr on cloud):

- **$170,000** for training
- **$30,000** for data licensing (Fisher, Expresso, custom)
- **Total: $200,000 over 3 months**

**Alternative: In-house (if using existing cluster):**

- 68K H100-hours ÷ 8 GPUs = 8,500 hours = 19 days of fully utilized cluster
- Highly parallelizable; can overlap stages

---

## Migration Path

### Phase 1: Research & Parallel Development (Weeks 1-6)

**Goal**: Build Sonata S2S alongside existing pipeline, zero disruption.

#### Activities

1. **Train codec_12hz** (already in progress)
   - Complete by 2026-03-15
   - Verify reconstruction PESQ ≥ 3.5

2. **Train semantic encoder** (Conformer @ 12.5Hz)
   - Target: match WavLM embeddings
   - Output: 512M checkpoint

3. **Data preparation**
   - License Fisher dataset (1,000 hrs)
   - Download Spoken Wikipedia (2,000 hrs)
   - Generate synthetic via existing pipeline (10K hrs)
   - Align with forced aligner (phonemes)

4. **Infrastructure**
   - Set up training cluster (GCE or internal)
   - Implement codec_12hz loader in PyTorch
   - Write Temporal Transformer (torch + candle)
   - Write Depth Transformer (depthwise)

#### Deployment Status

- **Sonata Pipeline**: LIVE (unchanged)
- **Sonata S2S**: In development (research branch)
- **User experience**: No impact

---

### Phase 2: Alpha Testing (Weeks 7-12)

**Goal**: Validate S2S quality and latency on single-user test device.

#### Activities

1. **Complete training** (Stages 1-5)
   - Semantic encoder done
   - Temporal pre-training done
   - Supervised fine-tuning done

2. **Inference optimization**
   - Export to Metal via candle
   - Optimize codec_12hz decoder (cuDNN → vDSP)
   - KV cache pruning for GQA
   - Batch processing for Depth Transformer

3. **A/B testing setup**
   - Model switch flag: `use_s2s=true`
   - Latency profiling: log each stage (encoder, temporal, depth, decoder)
   - Quality metrics: PESQ, speaker similarity (cosine with speaker encoder)
   - Barge-in latency: time from user speech start to TTS cutoff

4. **Benchmarks** (on M4 Max, 8GB)
   - Latency: target ≤ 200ms end-to-end
   - Memory: target ≤ 7GB (6GB weights + 1GB runtime)
   - Throughput: 15+ tokens/sec (semantic)

#### Testing Scenarios

- **Office chatter**: Real conversation, 5 min
- **Information retrieval**: Questions → answers (no LLM, rules-based)
- **Narrative**: Long-form story generation
- **Interruption**: User barge-in during assistant speaking
- **Accent variation**: Regional accents, non-native speakers

#### Deployment Status

- **Sonata Pipeline**: LIVE (production)
- **Sonata S2S**: Alpha on test device
- **User experience**: No impact

---

### Phase 3: Beta Rollout (Weeks 13-24)

**Goal**: Gradual rollout to 10% of users, measure production metrics.

#### Activities

1. **Full-duplex fine-tuning** (optional)
   - If alpha showed strong interruption handling, continue training
   - If not needed, skip and proceed with turn-based S2S

2. **Mobile optimization**
   - Quantization: INT4 weight-only (NEON dequant)
   - Memory pooling: shared attention buffers
   - Batch decode: process 2-4 users' audio streams concurrently

3. **Fallback logic**
   - If S2S confidence < 0.8, fall back to pipeline
   - If S2S latency > 500ms (timeout), fall back
   - If memory pressure, fall back to lighter model variant

4. **Monitoring**
   - Latency percentiles (p50, p95, p99)
   - Quality (PESQ, user ratings)
   - Fallback frequency
   - Inference cost (GPU utilization)

#### Deployment

- **Canary**: 10% of user base
- **Metrics dashboard**: Real-time monitoring
- **Rollback plan**: Feature flag to disable S2S instantly
- **User experience**: "Faster response mode (beta)" with opt-out

---

### Phase 4: Sundown Pipeline (Weeks 25-36)

**Goal**: Fully replace STT→LM→TTS with S2S once proven.

#### Activities

1. **Performance tuning**
   - Batch size optimization for multi-user
   - Codec streaming mode (output chunks as generated, not waiting)
   - KV cache eviction policies for long contexts

2. **Remove legacy code**
   - Deprecate `conformer_stt.c`, `sonata_lm.c`, `sonata_flow.c` from production
   - Keep as fallback (compressed, archived)
   - Version checkpoint in Git

3. **Training continuation**
   - Collect production conversations (with user consent)
   - Online learning: periodic re-train on new dialogues
   - Dataset versioning (S2S-v1.0, v1.1, v1.2, ...)

4. **Documentation & handoff**
   - Training reproducibility guide
   - Inference optimization techniques
   - Common failure modes and mitigations

#### Deployment

- **Sunset**: STT→LM→TTS becomes legacy option
- **Default**: Sonata S2S
- **User experience**: Faster, more natural, full-duplex ready

---

## Risk Analysis

### Risk 1: Semantic Drift (Probability: Medium, Impact: High)

**What**: S2S generates plausible-sounding but factually wrong responses (hallucination worse than Claude).

**Why**:

- Text stream is optional/sparse; model less grounded in language
- No explicit reasoning stage (LM abstracted away)
- Smaller model (3B vs Claude 3.5)

**Mitigation**:

1. **Constraint decoding**: Sample only from allowed utterance templates
2. **Fact checking**: Quick lookup (entity database) before generation
3. **Hybrid**: Keep Claude LLM for reasoning, S2S only for TTS generation
4. **Monitoring**: Flag low-confidence outputs for human review

**Acceptance**: Acceptable if S2S acts as voice interface to LLM (text stream preserved).

---

### Risk 2: Quality Regression in Audio (Probability: Low, Impact: Medium)

**What**: S2S audio quality < Sonata Flow (worse intelligibility, prosody).

**Why**:

- Codec_12hz not yet validated (PESQ target: 3.5)
- Joint training may not separate semantic/acoustic well
- Acoustic delay (τ=2) may cause artifacts

**Mitigation**:

1. **Codec validation first**: Ensure codec_12hz PESQ ≥ 3.5 before S2S training
2. **Ablation**: Train with τ=0, 1, 2 and compare
3. **Reference baseline**: Always keep Sonata Flow checkpoint; A/B test with users
4. **Perceptual loss**: Add style consistency term (speaker similarity)

**Acceptance**: Acceptable if within 0.2 PESQ of Flow (3.5 → 3.3).

---

### Risk 3: Latency Overrun (Probability: Medium, Impact: High)

**What**: S2S inference ≥ 300ms on average device (breaks real-time requirement).

**Why**:

- Temporal Transformer (32 layers, 4096 dim) is expensive
- GQA still requires large KV cache (3000 tokens)
- Depth Transformer per-quantizer overhead

**Mitigation**:

1. **Streaming micro-batching**: Process 4 frames at once (not frame-by-frame)
2. **Token pruning**: Drop KV entries for frames > 5 seconds old
3. **Conditional depth**: Use simpler Depth Transformer for confident (high-prob) semantics
4. **Quantization**: INT4 weight-only + low-precision intermediate activations
5. **Fallback**: At 200ms latency target, fall back to pipeline

**Acceptance**: Acceptable up to 250ms (still better than pipeline + LLM).

---

### Risk 4: Full-Duplex Complexity (Probability: High, Impact: Medium)

**What**: Overlapping audio (user + assistant) is hard to model; training doesn't converge well.

**Why**:

- No clear token boundaries in overlapping regions
- Loss function must handle ambiguous alignment
- Inference needs buffering strategy

**Mitigation**:

1. **Phase 1 (v1.0)**: Launch with turn-taking only (explicit speaker boundaries)
2. **Synthetic overlaps**: Train on artificially layered audio for robustness
3. **Overlap loss**: Only weight overlapping frames 0.5x vs solo 1.0x
4. **Phase 2 (v1.1)**: Natural overlap training after turn-taking validated
5. **User opt-in**: Let early adopters enable full-duplex, measure quality

**Acceptance**: Defer full-duplex to v1.1; v1.0 focuses on turn-taking quality.

---

### Risk 5: Data Privacy (Probability: Medium, Impact: High)

**What**: Training data contains PII; users don't consent to inclusion.

**Why**:

- Fisher dataset includes personal calls (home, family)
- Synthetic data via Claude includes echoes of training corpus
- On-device learning collects user conversations

**Mitigation**:

1. **Data audit**: GDPR/CCPA review before licensing Fisher
2. **Anonymization**: Remove speaker IDs, sensitive utterances
3. **Consent**: Opt-in for on-device learning; users can request data deletion
4. **Encryption**: Store raw audio encrypted; train only on anonymized codes
5. **Legal review**: Contract amendments for commercial use

**Acceptance**: Require legal sign-off before production training.

---

### Risk 6: Cold-Start Hallucination (Probability: Low, Impact: Low)

**What**: Model converges to pathological attractor (e.g., repeating "hello" infinitely).

**Why**:

- Autoregressive decoding can amplify errors via feedback loop
- Depth Transformer may collapse to mode (same codebooks repeatedly)

**Mitigation**:

1. **Curriculum learning**: Start with high-quality synthetic data, gradually mix real conversations
2. **Entropy regularization**: Penalty on low-entropy quantizer distributions
3. **Top-k sampling**: Force diversity in codebook selection
4. **Supervised initialization**: Warm-start Temporal Transformer with Sonata LM weights
5. **Early stopping**: Monitor PESQ on held-out set; stop if degrading

**Acceptance**: Acceptable if addressed in Stage 5 SFT.

---

## Resource Estimates

### Compute

| Phase        | GPU-Hours  | Hardware | Real Time    | Cost ($2.50/hr) |
| ------------ | ---------- | -------- | ------------ | --------------- |
| 1. Codec     | 1,000      | H100 × 4 | 2 weeks      | $2,500          |
| 2. Semantic  | 2,000      | H100 × 4 | 3 weeks      | $5,000          |
| 3. Pre-train | 50,000     | H100 × 8 | 3 weeks      | $125,000        |
| 4. SFT       | 15,000     | H100 × 4 | 3 weeks      | $37,500         |
| 5. Fine-tune | —          | H100 × 4 | ongoing      | TBD             |
| **Total**    | **68,000** | —        | **11 weeks** | **$170,000**    |

### Data

| Category         | Cost        | Notes                        |
| ---------------- | ----------- | ---------------------------- |
| Fisher License   | $15,000     | LDC membership (1-year)      |
| Custom recording | $10,000     | 1K hours @ $10/hr annotation |
| Synthetic gen    | Free        | Use existing Sonata pipeline |
| **Total**        | **$25,000** | —                            |

### Personnel (11-week project)

| Role             | FTE | Weeks | Cost      |
| ---------------- | --- | ----- | --------- |
| ML Engineer      | 1.0 | 11    | $150K     |
| Infra / DevOps   | 0.5 | 11    | $40K      |
| Audio QA/Testing | 0.5 | 6     | $20K      |
| **Total**        | 2.0 | —     | **$210K** |

### Total Project Cost

```
Training:        $170,000
Data:            $25,000
Personnel:       $210,000
─────────────────────────
Subtotal:        $405,000

Contingency (20%): $81,000
─────────────────────────
TOTAL:           $486,000
```

**Timeline**: 11 weeks (12-15 calendar weeks with overhead)

---

## Apple Silicon Optimization Strategy

### Memory Layout

Sonata S2S on M4 Max (8GB shared DRAM):

```
Model Weights (FP16):     6.0 GB
├─ Temporal Trans:        2.4 GB (1.2B params)
├─ Depth Trans:           360 MB
├─ Codec:                 80 MB
├─ Embeddings:            1.6 GB

Runtime Activations:      1.0 GB
├─ KV cache (GQA):        600 MB (3000 tokens × 8 heads × 128 dim)
├─ Intermediate:          300 MB (max at bottleneck)
├─ Ring buffer:           100 MB (audio)

Reserved (OS, etc.):      1.0 GB
─────────────────────────
Total:                    8.0 GB
```

### Hardware Dispatch

| Compute Unit  | Workload                | Framework     |
| ------------- | ----------------------- | ------------- |
| **Metal GPU** | Temporal/Depth Trans    | Candle+Metal  |
| **AMX**       | Codec (convolutions)    | Accelerate    |
| **NEON**      | Ring buffers, iSTFT     | Intrinsics    |
| **ANE**       | Optional: codec decoder | BNNS (future) |

### Kernel Optimization

1. **Fused attention**: FlashAttention v2 in Metal (already done for Sonata LM)
2. **GQA-optimized**: Interleaved KV layout for grouped heads
3. **Causal masking**: Precomputed masks per-batch to avoid runtime branching
4. **Codec streaming**: Output decoded audio before Frame N fully complete (pipelined)

### Inference Pipeline Parallelization

```
Frame t-1:  Encoder   │ Temporal  │ Depth  │ Decoder │ ────→ Audio
Frame t  :  Encoder   │           ↓        ↓         │ ────→ Audio
Frame t+1:  ────────→ Temporal   │        ↓         │ ────→ Audio
            ─────────────────→   Depth  Decoder    │
                               ──────────→
```

Each stage is pipeline-parallel; while Temporal processes Frame t, Encoder fetches Frame t+1.

---

## Fallback & Graceful Degradation

### S2S → Pipeline Fallback Triggers

| Trigger                         | Condition              | Action                     |
| ------------------------------- | ---------------------- | -------------------------- |
| **High latency**                | > 250ms per frame      | Log; fall back to pipeline |
| **OOM**                         | Alloc failure          | Free S2S; use lite model   |
| **Low confidence**              | Semantic entropy < 0.5 | Re-run with Temporal-only  |
| **Audio quality**               | PESQ < 3.0 (detected)  | Switch codec mode          |
| **User barge-in (no response)** | S2S generates silence  | Fall back to pipeline TTS  |

### Lite Model Option (Optional)

If 3B is too large for some devices, train a 1.2B variant:

| Component  | Full (3.0B) | Lite (1.2B) | Notes                   |
| ---------- | ----------- | ----------- | ----------------------- |
| Temporal   | 16L × 1024d | 12L × 768d  | Drop 4 layers, dim      |
| Depth      | 6L × 1024d  | 6L × 512d   | Keep same depth         |
| **Memory** | 6.0 GB      | 3.0 GB      | Fits all Apple devices  |
| **Speed**  | 40 tok/s    | 80 tok/s    | 2x faster (50% quality) |

Deploy both; auto-select based on device at runtime.

---

## Key Milestones & Success Criteria

| Milestone                       | Target Date | Success Criteria                           |
| ------------------------------- | ----------- | ------------------------------------------ |
| **Codec_12hz validation**       | 2026-03-15  | PESQ ≥ 3.5 on LibriSpeech test             |
| **Semantic encoder trained**    | 2026-04-01  | WavLM match MSE < 0.05 on test set         |
| **Pre-training complete**       | 2026-04-15  | Val loss plateau; no divergence            |
| **Alpha inference on M4**       | 2026-04-20  | ≤ 200ms latency; ≥ 3.2 PESQ                |
| **Beta rollout (10% users)**    | 2026-05-01  | < 5% fallback rate; positive user feedback |
| **Production (100% users)**     | 2026-06-01  | < 1% fallback; user satisfaction ≥ 4.0/5   |
| **Full-duplex v1.1 (optional)** | 2026-08-01  | Handles natural overlaps; no mode collapse |

---

## Open Questions & Future Work

1. **Inner Monologue**: Worth the extra latency? Measure text quality with/without.
2. **Streaming codec**: Can codec_12hz decoder operate in real-time (vs batch)?
3. **Multi-speaker**: Can S2S handle 3+ speakers (group calls)?
4. **Emotion transfer**: Preserve user emotion in assistant response?
5. **Language mixing**: Code-switching (English-Spanish) — how well?
6. **Speaker cloning**: Can reference audio condition speaker identity without training?

---

## Conclusion

Sonata S2S represents an 18-month roadmap to unified speech-to-speech dialogue, with:

- **Reuse**: Codec_12hz, speaker encoder, MEL pipeline carry forward
- **Modularity**: Pipeline coexists with S2S for 6+ months; fallback safety net
- **Performance**: 3B model optimized for Apple Silicon; 200ms target latency
- **Quality**: Joint training improves semantic-acoustic coherence vs cascaded approach
- **Pragmatism**: Defer full-duplex to v1.1; v1.0 focuses on turn-taking quality

**Next step**: Approve data licensing and compute allocation to begin Stage 1 (codec validation) by 2026-03-15.

---

## References & Related Work

- **Moshi** (Kyutai Labs): [GitHub](https://github.com/kyutai-labs/moshi), [Paper](https://arxiv.org/abs/2410.00037)
- **Sonata Codec 12.5Hz Design**: `/docs/codec_12hz_design.md`
- **Sonata Architecture**: `/docs/architecture.md`
- **Training Scripts**: `/train/sonata/{flow.py, semantic_lm.py, train_flow_v2.py}`
- **Inference**: `/src/sonata_flow/src/lib.rs`, `/src/pocket_voice_pipeline.c`

---

**Document prepared**: 2026-03-05
**Author**: Architecture Team
**Status**: DESIGN PHASE (READY FOR REVIEW)
