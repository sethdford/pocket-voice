# TTS SOTA Quick Reference (2025-2026)

## One-Page Comparison: Sonata vs Leading SOTA Models

| Dimension                      | Sonata Current          | SOTA Model               | SOTA Value                | Gap                         |
| ------------------------------ | ----------------------- | ------------------------ | ------------------------- | --------------------------- |
| **Flow Matching Architecture** | Supervised regression   | F5-TTS                   | FM loss + OT path         | Missing importance sampling |
| **Inference Steps (NFE)**      | 16-32 (estimated)       | F5-TTS EPSS              | 7 steps (4× faster)       | -4.6× speedup potential     |
| **RTF (Real-Time Factor)**     | 0.015 (Piper legacy)    | F5-TTS @ 7 steps         | 0.03                      | Flow RTF not measured       |
| **MOS (Naturalness)**          | Not measured            | F5-TTS / CosyVoice 3     | 4.3-4.5                   | Unknown deficit             |
| **WER (Quality)**              | Not measured            | F5-TTS                   | 0.9%                      | Unknown                     |
| **Speaker Similarity (SIM-O)** | No metric               | MaskGCT                  | 0.687 (human parity)      | No measurement              |
| **Speaker Encoder**            | Basic speaker_encoder.c | ECAPA-TDNN / WavLM       | Pretrained baseline       | Untested                    |
| **Streaming FPL / TTFA**       | 320 ms round-trip       | VoXtream                 | 102 ms                    | 3.1× higher                 |
| **Codec Token Rate**           | FSQ 50 tokens/s         | 12.5 Hz codec (training) | 12.5 tokens/s             | 4× reduction pending        |
| **Speculative Decode**         | ReDrafter defined only  | Full pipeline            | 2.3× speedup              | Training not started        |
| **RL Fine-Tuning**             | None                    | F5R-TTS / CosyVoice 3    | +4.6% SIM (relative)      | +4-5% quality potential     |
| **Multilingual Support**       | Not tested              | CosyVoice 3              | 9 languages + 18 dialects | Feature gap                 |
| **Codec Grouping**             | No                      | VALL-E 2                 | 2-8× token reduction      | Applicable to FSQ           |

---

## SOTA Leaders by Category (2025-2026)

### 1. Flow Matching (Few-Step NAR)

- **SOTA**: CosyVoice 3.0 (Dec 2025)
- **Metrics**: MOS ~4.5, CER 0.81% (Chinese), 1.5B params, 1M hours training
- **Key Innovation**: RL fine-tuning with supervised ASR semantic tokens
- **Sonata Gap**: No RL, 241M params vs 1.5B, missing ASR-supervised tokens

### 2. Zero-Shot Speaker Cloning

- **SOTA**: MaskGCT (Sep 2024)
- **Metrics**: SIM-O 0.687 (LibriSpeech), human parity (+0.017 above human baseline)
- **Key Innovation**: Masked generative codec transformer + accent imitation (SIM-Accent)
- **Sonata Gap**: No SIM-O measurement, no accent preservation

### 3. Streaming / Low-Latency

- **SOTA**: VoXtream (Sep 2025)
- **Metrics**: 102 ms FPL on GPU, 3.24% WER (LibriSpeech-long), begins speaking after 1st word
- **Key Innovation**: Monotonic alignment + dynamic lookahead for streaming
- **Sonata Gap**: Architecture requires full text upfront, 3.1× higher latency

### 4. Codec Efficiency

- **SOTA**: WavTokenizer (2024)
- **Token Rate**: 40-75 tokens/s (vs EnCodec 50)
- **Bitrate**: 0.5-0.9 kbps (ultra-low)
- **Sonata Gap**: 12.5 Hz codec training in progress (4× reduction when ready)

### 5. Speculative Decoding (Emerging)

- **SOTA**: Collective & Adaptive Spec Decode (SC 2025)
- **Speed Improvement**: 2-3× on LLM inference
- **Sonata Readiness**: ReDrafter architecture defined, training not started
- **Gap**: No measurement infrastructure or draft training schedule

---

## Key Paper Summary Table

| Paper                  | Authors      | Date     | Key Metric                        | Relevance to Sonata                |
| ---------------------- | ------------ | -------- | --------------------------------- | ---------------------------------- |
| F5-TTS                 | Liu et al.   | Oct 2024 | 4× speedup @ 7 NFE, MOS 4.3       | Flow matching + step optimization  |
| EPSS                   | Zheng et al. | Mar 2025 | 4× speedup on E2-TTS / F5-TTS     | Directly applicable to Sonata flow |
| MaskGCT                | Wang et al.  | Sep 2024 | SIM-O 0.687, human parity         | Zero-shot voice cloning benchmark  |
| CosyVoice 3            | Deng et al.  | Dec 2025 | MOS 4.5, 1M hours, 9 languages    | RL fine-tuning + scaling           |
| VoXtream               | Xue et al.   | Sep 2025 | 102 ms FPL, streaming SOTA        | Streaming latency reference        |
| VALL-E 2               | Wang et al.  | Jun 2024 | Grouped codec modeling            | Token reduction technique          |
| WavTokenizer           | Zhang et al. | 2024     | 40-75 tokens/s, ultra-low bitrate | Codec efficiency benchmark         |
| Collective Spec Decode | Xin et al.   | SC 2025  | 2-3× LLM speedup                  | Speculative decoding pattern       |

---

## Immediate Actionable Tasks (P0/P1)

### P0 Blockers

1. **12.5 Hz Codec Training Status**
   - **Current**: GCE Step ~200/200K (as of 2026-03-06)
   - **ETA**: ~42 hours remaining
   - **Impact**: Unlocks 4× token reduction, enables 9.2× compound speedup (4× codec × 2.3× spec decode)
   - **Next Step**: Monitor `/opt/sonata/auto_shutdown.sh` watchdog, export checkpoint

2. **ReDrafter Training**
   - **Current**: train_drafter.py script ready, not scheduled
   - **Model**: 3.5M GRU draft model for semantic tokens
   - **Impact**: ~2.3× semantic LM speedup
   - **Next Step**: Schedule training on available GPU, estimate time

### P1 Quality Improvements

1. **Speaker Encoder Evaluation**
   - **Task**: Baseline SIM-O measurement against ECAPA-TDNN / WavLM
   - **Estimated Impact**: +1-2% speaker similarity if delta found
   - **Effort**: 2-3 days (training + eval)

2. **Flow Matching Upgrade**
   - **Task**: Replace supervised regression with FM loss + OT path
   - **Technique**: Apply Sway Sampling for variable-step inference
   - **Estimated Impact**: +2-3% WER improvement
   - **Reference**: F5-TTS paper + existing codebase

3. **RL Fine-Tuning Pipeline**
   - **Task**: GRPO reward model training
   - **Estimated Impact**: +4.6% relative SIM improvement (per F5R-TTS)
   - **Infrastructure**: GCE setup already operational
   - **Effort**: 1-2 weeks including reward model training

### P2 Long-Term (Streaming Optimization)

1. **Monotonic Alignment** — VoXtream-style for true streaming
2. **Dynamic Lookahead** — Reduce TTFA from 320 ms to <150 ms
3. **Input-Time Speculation** — Overlap TTS with speech buffering

---

## Quality Metrics Cheat Sheet

| Metric       | Definition                                             | SOTA 2025           | Sonata Current       |
| ------------ | ------------------------------------------------------ | ------------------- | -------------------- |
| **MOS**      | Mean Opinion Score (1-5 scale, subjective naturalness) | 4.3-4.5             | Not measured         |
| **WER**      | Word Error Rate (via ASR readback, lower better)       | 0.009 (0.9%)        | Not measured         |
| **SIM-O**    | Speaker similarity (cosine, 0-1, higher better)        | 0.687 (MaskGCT)     | Not measured         |
| **RTF**      | Real-Time Factor (1.0 = realtime, lower better)        | 0.03 @ 7 steps (F5) | 0.015 (legacy Piper) |
| **FPL/TTFA** | First Packet / First Audio Latency (ms, lower better)  | 102 ms (VoXtream)   | ~320 ms round-trip   |
| **CER**      | Character Error Rate (Chinese, lower better)           | 0.81% (CosyVoice 3) | Not measured         |

---

## Architecture Comparison: Key Innovations

### F5-TTS (Oct 2024) — Flow Matching SOTA

- **Sway Sampling**: Importance sampling during inference for adaptive steps
- **DiT (Diffusion Transformer)**: Replaces CNN diffusion encoder
- **Result**: 4× speedup @ 7 steps, MOS 4.3, open source

### MaskGCT (Sep 2024) — Zero-Shot SOTA

- **Masked generation**: Codec tokens + text tokens jointly masked during training
- **ECAPA speaker encoder**: Pre-trained speaker similarity extraction
- **Accent preservation**: SIM-Accent metric for accent fidelity
- **Result**: Human-parity SIM-O (0.687), 100K hours in-the-wild training

### CosyVoice 3 (Dec 2025) — Production SOTA

- **Supervised semantic tokens**: From ASR (not self-supervised)
- **Reinforcement learning**: GRPO fine-tuning with reward model
- **Scaling**: 1M hours, 9 languages, 18 Chinese dialects, 1.5B params
- **Result**: MOS 4.5, 150 ms streaming latency, multilingual zero-shot

### VoXtream (Sep 2025) — Streaming SOTA

- **Monotonic alignment**: Maps phoneme stream to audio tokens in real-time
- **Dynamic lookahead**: Variable context window (1-3 words) for latency/quality tradeoff
- **Full-stream inference**: Begins speaking after first word
- **Result**: 102 ms FPL, 3.24% WER on streaming LibriSpeech-long

---

## Codec Landscape 2025-2026

| Codec              | Frame Rate | Token Rate      | Bitrate      | Qualities    | Notes                               |
| ------------------ | ---------- | --------------- | ------------ | ------------ | ----------------------------------- |
| **EnCodec**        | 50 Hz      | 50 tokens/s     | 6 kbps       | Good         | VALL-E 2 baseline                   |
| **DAC**            | 50-100 Hz  | 900 t/s @ 1kbps | 1-6 kbps     | High quality | Over-parameterized for TTS          |
| **WavTokenizer**   | Variable   | 40-75 tokens/s  | 0.5-0.9 kbps | Good         | Ultra-efficient, newer              |
| **Sonata FSQ**     | 50 Hz      | 50 tokens/s     | Unknown      | Training     | Current, not yet optimized          |
| **Sonata 12.5 Hz** | 12.5 Hz    | 12.5 tokens/s   | Unknown      | Training     | Under GCE training (Step ~200/200K) |

**Token reduction via grouping** (VALL-E 2): Pack codec frames into groups (size 1, 2, 4, 8) — applicable to any codec, directly reduces sequence length by 2-8×.

---

## Research Timeline

- **2023**: Matcha-TTS (4-16 steps), baseline diffusion
- **2024 Q2**: VALL-E 2, SoundStorm, Mega-TTS 2, XTTS v2
- **2024 Q3**: MaskGCT (human-parity speaker cloning)
- **2024 Q4**: F5-TTS (flow matching breakthrough), CosyVoice 2 (streaming)
- **2025 Q1**: EPSS (4× speedup on flow), Collective Spec Decode
- **2025 Q2-Q3**: VoXtream (102 ms FPL streaming), SpeakStream, CosyVoice 3
- **2025 Q4**: New codec papers (DualCodec, FreeCodec), MLX integration emerging

---

## Sonata Roadmap Alignment

### Next 2 Weeks

- [ ] Complete 12.5 Hz codec training (monitor GCE)
- [ ] Schedule ReDrafter training
- [ ] Measure SIM-O baseline

### 1-2 Months

- [ ] FM loss upgrade (replace supervised regression)
- [ ] Speaker encoder swap (ECAPA or WavLM)
- [ ] RL fine-tuning infrastructure setup

### 2-6 Months

- [ ] Streaming optimization (monotonic alignment)
- [ ] Full speculative pipeline integration
- [ ] Multilingual expansion
