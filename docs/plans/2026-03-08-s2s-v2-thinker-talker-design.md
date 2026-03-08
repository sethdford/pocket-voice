# Sonata S2S v2 — Hybrid Thinker-Talker Architecture

**Date**: 2026-03-08
**Status**: DESIGN (supersedes 2026-03-05-moshi-s2s-design.md)
**Scope**: Updated speech-to-speech design incorporating Jan-Mar 2026 SOTA

---

## What Changed Since v1 Design (March 5)

Three major releases in Jan 2026 shift the architecture:

1. **Qwen2.5-Omni Thinker-Talker** — separates reasoning (Thinker) from speech generation (Talker), proving that a lightweight Talker conditioned on LLM hidden states matches or beats monolithic S2S models
2. **NVIDIA PersonaPlex 7B** — Moshi-derived full-duplex S2S running on Apple Silicon via MLX at 68ms/step (RTF 0.87), validating on-device feasibility
3. **Qwen3-TTS** — 97ms streaming latency with 12Hz multi-codebook tokenizer, demonstrating that dual-track streaming architectures work in production

**Key insight**: Sonata already delegates reasoning to Claude/Gemini. Building a 3B monolithic S2S model duplicates this capability at lower quality. Instead, build a lightweight Talker that converts LLM reasoning into speech.

---

## Architecture: Hybrid Thinker-Talker

### Design Principles

1. **Keep reasoning SOTA** — Claude/Gemini as Thinker, not a 3B model
2. **Talker is small** — 500M params, fits any M-series chip in INT4
3. **Full-duplex via dual streams** — user + assistant audio modeled in parallel
4. **Reuse existing components** — codec_12hz, speaker encoder, flow distillation
5. **Graceful fallback** — if Talker fails, existing TTS pipeline takes over

### High-Level Data Flow

```
User Audio (24kHz)
    |
    v
[Codec Encoder (Mimi-style, 12.5Hz)]
    |
    +---> Audio Tokens (semantic + 7 acoustic RVQ codes)
    |         |
    |         v
    |    [User Audio Stream] ----+
    |                            |
    +---> [STT Head] ---+        |
              |         |        |
              v         |        |
         Text Tokens    |        |
              |         |        |
              v         |        |
    [Thinker: Claude API]|       |
         |              |        |
         | hidden states|        |
         | + text tokens|        |
         v              v        v
    [Talker: Dual-Track AR Transformer, 500M]
         |
         +---> Text tokens (inner monologue, optional)
         +---> Audio codec tokens (12.5 Hz, 8 codebooks)
                   |
                   v
            [Codec Decoder (Mimi-style)]
                   |
                   v
            Assistant Audio (24kHz)
```

### Talker Architecture (500M params)

| Component            | Dim  | Layers | Heads | FFN | Params    | Notes                              |
| -------------------- | ---- | ------ | ----- | --- | --------- | ---------------------------------- |
| Audio Embedder       | 768  | --     | --    | --  | 48M       | (sem + 7 acoustic) x 2048 codebook |
| Text Embedder        | 768  | --     | --    | --  | 24M       | 32K vocab SentencePiece            |
| Thinker Projector    | 768  | 2      | --    | 3K  | 12M       | Projects LLM hidden dim -> 768     |
| Temporal Transformer | 768  | 12     | 12    | 3K  | 340M      | GQA (4 groups), RoPE, causal       |
| Depth Transformer    | 512  | 6      | 8     | 2K  | 72M       | Per-codebook, depthwise            |
| LM Heads             | 2048 | --     | --    | --  | 16M       | 8 codebook heads                   |
| **TOTAL**            | --   | --     | --    | --  | **~512M** | 1GB FP16, 256MB INT4               |

**Why 500M instead of 3B?**

- The Thinker (Claude) handles reasoning — Talker only needs acoustic generation
- Qwen2.5-Omni's Talker is ~200M and produces high-quality speech
- 500M INT4 = 256MB weights — fits alongside Thinker on any M-series
- Faster inference: ~80 tok/s on M4 (vs ~15 tok/s for 3B)

### Dual-Track Streaming (from CosyVoice 2 / Qwen3-TTS)

Single model supports both streaming and non-streaming:

**Streaming mode** (real-time conversation):

```
Frame t: Encode user audio -> Temporal(attend to t-1..0) -> Depth -> Decode -> Play
         Latency: 80ms encode + 30ms temporal + 20ms depth + 20ms decode = 150ms
```

**Non-streaming mode** (pre-generated responses):

```
Full text -> All temporal steps -> All depth steps -> Full decode
Higher quality, used for canned responses or pre-buffered content
```

### Thinker Integration Modes

**Mode A: Cloud LLM (Claude/Gemini)**

```
User text (from STT head) -> Claude API -> response text + hidden states
                                           -> Talker generates speech
Latency: +200-500ms for API round-trip
Quality: Best reasoning, worst latency
```

**Mode B: Local LLM (Llama 3B quantized)**

```
User text -> Local Llama -> response text + hidden states -> Talker
Latency: +100ms for local inference
Quality: Good reasoning, good latency
```

**Mode C: Direct S2S (no explicit Thinker)**

```
User audio tokens -> Temporal Transformer (acts as both) -> Talker
Latency: ~150ms total (Moshi-like)
Quality: Limited reasoning, best latency
```

Start with Mode A, add Mode C for full-duplex later.

### Full-Duplex Design

Following Moshi/PersonaPlex dual-stream approach:

```
At each 12.5Hz timestep, process interleaved tokens:

[user_sem_t] [user_a0_t] ... [user_a7_t]  <- listening stream
[asst_sem_t] [asst_a0_t] ... [asst_a7_t]  <- speaking stream
[text_t]                                    <- inner monologue (optional)

Total: 17 tokens per timestep
Acoustic delay: tau=2 frames (160ms) for stability
```

**Barge-in handling**:

- User audio stream updates continuously (never paused)
- When user energy > threshold during assistant speech:
  - Talker receives user tokens in real-time
  - Learns to stop/pause naturally (trained on interruption data)
  - No explicit state machine — model handles it

---

## Codec: Mimi-Sonata Hybrid

### Architecture

Build on our codec_12hz (training at step 64K) + Mimi's semantic distillation:

| Component       | Spec                          | Notes                                              |
| --------------- | ----------------------------- | -------------------------------------------------- |
| Sample rate     | 24 kHz                        | Same as current Sonata                             |
| Frame rate      | 12.5 Hz                       | 80ms frames (matches Mimi)                         |
| Codebooks       | 8 RVQ                         | First = semantic (WavLM-distilled), 2-8 = acoustic |
| Codebook size   | 2048                          | Same as Mimi                                       |
| Bandwidth       | 1.1 kbps                      | Matching Mimi                                      |
| Encoder         | ConvNet + 8-layer Transformer | Streaming, 80ms latency                            |
| Decoder         | 8-layer Transformer + ConvNet | Streaming                                          |
| Semantic target | WavLM (distilled)             | First codebook matches WavLM embeddings            |

### Training Delta (from current codec_12hz)

Current codec_12hz uses FSQ + single codebook. Need to:

1. Switch to 8-level RVQ with 2048 entries each
2. Add WavLM distillation loss on first codebook
3. Add Transformer in bottleneck (8 layers, 512 dim)
4. Train for 4M steps (vs current 200K target)

**Estimated cost**: 2,000 H100-hours (~$5,000)

---

## Training Plan (Revised)

### Stage 1: Codec with Semantic Distillation (3 weeks)

Modify codec_12hz training:

- Add RVQ with 8 codebooks x 2048 entries
- Add WavLM distillation loss on first codebook (contrastive + MSE)
- Add Transformer bottleneck (8 layers, 512 dim, RoPE)
- Train on LibriTTS-R + internal data (~2000 hours)
- **GPU hours**: 2,000 (H100 x 4)

### Stage 2: Talker Pre-training (4 weeks)

Causal language modeling on audio codes:

- Input: interleaved user + assistant audio codes + sparse text
- Target: next-token prediction on all streams
- Data: 10K hours conversational + 10K hours synthetic (Sonata TTS)
- **Batch size**: 128 (on 8x H100)
- **Steps**: 300K
- **GPU hours**: 30,000

### Stage 3: Thinker Alignment (2 weeks)

Fine-tune Talker to condition on Thinker hidden states:

- Freeze Temporal Transformer, train Thinker Projector + LM heads
- Input: Claude/Gemini hidden states for 5K dialogue pairs
- Loss: next-token CE + speaker similarity + prosody matching
- **GPU hours**: 5,000

### Stage 4: Full-Duplex Fine-tuning (2 weeks, optional)

- Train on overlapping speech data
- Synthetic: layer user + assistant audio with natural overlaps
- Real: Fisher corpus subset (4K hours)
- **GPU hours**: 5,000

### Total Cost (Revised)

| Stage          | GPU Hours  | Duration     | Cost ($2.50/hr) |
| -------------- | ---------- | ------------ | --------------- |
| 1. Codec       | 2,000      | 3 weeks      | $5,000          |
| 2. Pre-train   | 30,000     | 4 weeks      | $75,000         |
| 3. Alignment   | 5,000      | 2 weeks      | $12,500         |
| 4. Full-duplex | 5,000      | 2 weeks      | $12,500         |
| **TOTAL**      | **42,000** | **11 weeks** | **$105,000**    |

**vs v1 design**: 38% cheaper ($105K vs $170K training), same timeline.
**vs v1 total**: 57% cheaper ($130K vs $486K including personnel savings from simpler model).

---

## Fast Prototype: PersonaPlex Fine-tune (2 weeks)

Before full Talker training, validate the concept by fine-tuning PersonaPlex:

1. Download PersonaPlex 7B weights (CC-BY-4.0)
2. Fine-tune on Sonata voice domain (100 hours, LoRA)
3. Quantize to INT4 for Apple Silicon
4. Test full-duplex latency on M4

**Expected outcomes**:

- Validate full-duplex works on-device
- Measure real latency (target <200ms)
- Identify quality gaps to address in custom Talker
- **Cost**: <$1,000 (LoRA on single H100, 2 days)

---

## Apple Silicon Optimization

### Memory Layout (500M Talker, INT4)

```
Model Weights (INT4):       256 MB
Codec Encoder:               40 MB
Codec Decoder:               40 MB
KV Cache (GQA, 3K tokens):  200 MB
Ring Buffers (audio):        100 MB
Intermediates:               200 MB
OS + Framework:              500 MB
-----------------------------------
Total:                      ~1.3 GB  (vs 8GB for v1 3B model)
```

Fits on ANY M-series Mac or iPhone with Neural Engine.

### Hardware Dispatch

| Compute Unit | Workload                           | Framework      |
| ------------ | ---------------------------------- | -------------- |
| Metal GPU    | Temporal + Depth Transformers      | Candle + Metal |
| ANE          | Codec encoder/decoder (conv-heavy) | CoreML/BNNS    |
| AMX          | Embedding lookups, softmax         | Accelerate     |
| NEON         | Ring buffers, iSTFT, resampling    | Intrinsics     |

### Latency Target

| Component            | Target    | Notes                           |
| -------------------- | --------- | ------------------------------- |
| Codec encoder        | 20ms      | Conv + 8-layer Transformer      |
| Temporal Transformer | 30ms      | 12 layers, GQA, cached KV       |
| Depth Transformer    | 15ms      | 6 layers, parallel codebooks    |
| Codec decoder        | 15ms      | Conv + 8-layer Transformer      |
| Buffering            | 20ms      | Double-buffer output            |
| **Total**            | **100ms** | vs 200ms Moshi, 150ms CosyVoice |

---

## Migration Path

### Phase 1: Prototype (Weeks 1-2)

- Fine-tune PersonaPlex 7B on Sonata domain
- Test full-duplex on Apple Silicon
- Measure latency, quality, barge-in handling

### Phase 2: Codec Training (Weeks 3-5)

- Modify codec_12hz for RVQ + WavLM distillation
- Train 2K H100-hours
- Validate: PESQ >= 3.5, semantic token accuracy >= 90%

### Phase 3: Talker Training (Weeks 6-9)

- Pre-train 500M Talker on 20K hours audio
- Align with Claude/Gemini hidden states
- Validate: latency <= 100ms, MOS >= 3.8

### Phase 4: Integration (Weeks 10-11)

- Wire into pocket_voice_pipeline.c
- Add runtime mode switch (pipeline vs S2S)
- A/B test against existing TTS pipeline
- Validate: user preference >= 60% for S2S

### Phase 5: Full-Duplex (Weeks 12-14, optional)

- Fine-tune on overlapping speech
- Test natural interruptions, barge-in
- Ship as opt-in "conversation mode"

---

## Key Differences from v1 Design

| Aspect                | v1 (March 5)       | v2 (This doc)         |
| --------------------- | ------------------ | --------------------- |
| Architecture          | Monolithic 3B S2S  | Hybrid Thinker-Talker |
| Model size            | 3.0B (6GB FP16)    | 500M (256MB INT4)     |
| Reasoning             | Built-in (limited) | Claude/Gemini (SOTA)  |
| Training cost         | $170K compute      | $105K compute         |
| Total cost            | $486K              | ~$130K                |
| Memory on device      | 8GB                | 1.3GB                 |
| Latency target        | 200ms              | 100ms                 |
| PersonaPlex reference | Not considered     | Fast prototype path   |
| Qwen Thinker-Talker   | Not available      | Core architecture     |

---

## References

- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215) — Thinker-Talker architecture
- [NVIDIA PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/) — Full-duplex S2S on Apple Silicon
- [Moshi (Kyutai)](https://github.com/kyutai-labs/moshi) — Original dual-stream S2S
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — 97ms streaming, 12Hz tokenizer
- [CosyVoice 2](https://curateclick.com/blog/2025-cosyvoice-complete-guide) — Unified streaming/non-streaming
- [F5-TTS](https://arxiv.org/abs/2410.06885) — Flow-matching + DiT, sway sampling
- [PersonaPlex on MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23) — 68ms/step on Apple Silicon
- [Mimi Codec Explainer](https://kyutai.org/codec-explainer) — RVQ + WavLM semantic distillation

---

**Document prepared**: 2026-03-08
**Author**: Architecture Team
**Status**: READY FOR REVIEW
