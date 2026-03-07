# TTS SOTA Research (2025-2026)

## Executive Summary

Comprehensive research of current state-of-the-art in text-to-speech as of March 2026. Organized by category with detailed metrics, architectures, and gaps vs. Sonata's current implementation.

---

## 1. FLOW MATCHING / DIFFUSION TTS

### Category: Few-Step Flow Matching

Flow matching has emerged as the dominant paradigm for non-autoregressive (NAR) TTS in 2024-2026, replacing diffusion models due to superior convergence and speed.

| Model                    | Step Count         | Quality Metrics                          | Architecture                 | Status                      |
| ------------------------ | ------------------ | ---------------------------------------- | ---------------------------- | --------------------------- |
| **F5-TTS**               | 7 NFE (default 16) | WER: 0.9% (SeedTTS en), MOS ~4.3         | DiT + Sway Sampling          | 2024, Open source           |
| **F5R-TTS** (RL variant) | 7 NFE              | WER: -29.5% (rel), SIM +4.6% (rel)       | F5 + GRPO reward             | 2024, Open source           |
| **E2-TTS**               | 12-16 NFE          | WER comparable to F5                     | NAR, character-based padding | 2024, Open source           |
| **CosyVoice 3.0**        | Not specified      | MOS ~4.5, CER 0.81% (Chinese)            | 1.5B LM + Flow, RL tuned     | Dec 2025, SOTA multilingual |
| **MaskGCT**              | Not specified      | SIM-O: 0.687 (LibriSpeech), human parity | Masked codec transformer     | Sep 2024, Open source       |
| **Matcha-TTS**           | 4-16 variable      | MOS ~4.0                                 | Flow + monotonic alignment   | 2023, Baseline              |

#### Key Metrics Detail

- **Step count matters**: F5-TTS achieves 4× speedup at 7 NFE vs 16 NFE with <5% quality drop
- **Sway Sampling**: Importance sampling strategy during inference (can be applied to other flow models)
- **RTF (Real-Time Factor)**: F5-TTS achieves 0.03 RTF at 7 steps (33× realtime on GPU)
- **Quality floor**: 0.9% WER seems to be NAR TTS ceiling currently

#### What Sonata Is Missing

1. **Flow matching**: Sonata currently trains flow via supervised regression. Flow matching (OT-path + FM loss) is faster and higher quality
2. **Importance sampling**: No adaptive step sampling during inference
3. **RL fine-tuning**: No GRPO/PPO alignment (F5R-TTS adds 4.6% similarity improvement)
4. **Supervised semantic tokens**: CosyVoice 3 uses ASR token labels (not self-supervised), better coherence

---

## 2. ZERO-SHOT VOICE CLONING

### Category: Speaker Similarity with Minimal Reference Audio

Zero-shot speaker cloning is now approaching human parity. The key metric is **speaker similarity (SIM-O)** rather than pure naturalness.

| Model                | Reference Duration | SIM-O Score                               | Architecture                         | Status                  |
| -------------------- | ------------------ | ----------------------------------------- | ------------------------------------ | ----------------------- |
| **MaskGCT**          | 5s typical         | 0.687 (LibriSpeech), human parity +0.017  | Masked codec + ECAPA speaker encoder | Sep 2024, SOTA          |
| **VALL-E 2**         | 5-10s              | Reported human parity (numeric not given) | 241M LLM + grouped codec modeling    | Jun 2024, Closed        |
| **CosyVoice 3**      | 5-10s              | Not specified numerically                 | 1.5B LM + ASR tokens                 | Dec 2025                |
| **XTTS v2**          | 6s minimum         | Subjective preference noted               | Glow-TTS encoder + speaker encoder   | Open source, production |
| **F5-TTS**           | 5s typical         | SIM score ~0.65-0.68                      | DIffusion + speaker embedding        | Open source             |
| **Baseline (human)** | N/A                | ~0.67                                     | -                                    | Reference               |

#### Speaker Encoder Details

Recent 2025 research compared speaker encoders for zero-shot TTS:

| Encoder             | Performance Notes                                                  | When to Use                 |
| ------------------- | ------------------------------------------------------------------ | --------------------------- |
| **ECAPA-TDNN**      | Popular in SR but NOT better than domain-specific encoders for TTS | General speaker recognition |
| **WavLM**           | Self-supervised, strong for voice cloning (used in StyleTTS2)      | Multilingual, pre-trained   |
| **H/ASP** (YourTTS) | Domain-specific, still outperforms ECAPA in TTS context            | Zero-shot TTS specialized   |
| **x-vector**        | Older baseline, lower scores                                       | Legacy                      |

#### What Sonata Is Missing

1. **Speaker encoder baseline**: Sonata has speaker_encoder.c (basic implementation) but no comparison to ECAPA-TDNN or WavLM
2. **SIM-O metric**: No speaker similarity scoring in eval pipeline
3. **Reference conditioning**: No integration of variable reference durations (3s vs 10s) in training
4. **Accent imitation**: MaskGCT adds SIM-Accent for accent preservation — not in Sonata

---

## 3. STREAMING / LOW-LATENCY TTS

### Category: First-Token Audio Latency (TTFA)

Streaming is now essential for interactive voice agents. The key metric is **first-packet latency (FPL)** or **time-to-first-audio (TTFA)**.

| Model           | FPL / TTFA                | Architecture                            | WER (streaming)               | Status                     |
| --------------- | ------------------------- | --------------------------------------- | ----------------------------- | -------------------------- |
| **VoXtream**    | 102 ms (GPU)              | Monotonic alignment + dynamic lookahead | 3.24% (LibriSpeech-long)      | Sep 2025, SOTA streaming   |
| **SpeakStream** | 30 ms TTS + 15 ms vocoder | Decoder-only, streaming-aware           | <5% (with context)            | 2025, Academic             |
| **CosyVoice 2** | 150 ms (streaming mode)   | LM + Flow, 0.5B params                  | 6.11% (LibriSpeech-long)      | Dec 2024                   |
| **XTTS v2**     | Not optimized             | Glow-TTS, chunked inference             | WER 222% (streaming, unigram) | Open source, not streaming |
| **Kokoro**      | <100 ms estimated         | StyleTTS2-based, no diffusion           | Not specified                 | 2025, Emerging             |

#### Key Findings

- **VoXtream** is currently SOTA for streaming: 102 ms FPL on GPU, begins speaking after first word
- **Chunked inference**: Standard XTTS performs poorly on streaming (WER 222%) — needs architectural changes
- **Monotonic alignment**: Core to streaming (used in VoXtream) — maps phoneme stream to audio tokens frame-by-frame
- **Dynamic lookahead**: VoXtream uses variable context window to balance latency vs quality

#### What Sonata Is Missing

1. **Streaming architecture**: Flow matching requires all input text upfront; VoXtream uses monotonic alignment for streaming
2. **Incremental vocoding**: SpeakStream achieves 30 ms by interleaving TTS + vocoder
3. **First-token optimization**: No dynamic lookahead or adaptive buffering
4. **Latency measurement**: Current Sonata full round-trip is ~320 ms (audio stages only) — needs breakdown

---

## 4. CODEC-BASED TTS (Autoregressive Language Models)

### Category: Token Rate, Efficiency, Robustness

Codec-based TTS frames generation as predicting discrete acoustic tokens. Key metric is **tokens/second** (lower is better for inference speed).

| Codec                    | Token Rate                    | Bitrate       | Sample Rate | Use in TTS                      |
| ------------------------ | ----------------------------- | ------------- | ----------- | ------------------------------- |
| **EnCodec** (baseline)   | 50 tokens/s (1 Hz frame rate) | 6 kbps        | 24 kHz      | VALL-E 2 baseline               |
| **DAC**                  | 900 tokens/s at 1 kbps        | 1 kbps        | Varies      | Research                        |
| **WavTokenizer**         | 40-75 tokens/s                | 0.5-0.9 kbps  | Variable    | Newer, more efficient           |
| **Sonata Codec (FSQ)**   | 50 tokens/s (20 ms frames)    | Not specified | 24 kHz      | Current Sonata                  |
| **Sonata 12.5 Hz codec** | 12.5 tokens/s (4× fewer)      | Unknown       | 24 kHz      | Under training (Step ~200/200K) |

#### Leading Codec-Based TTS Models

| Model                  | Codec Choice                                    | Token Reduction    | Quality                    | Status           |
| ---------------------- | ----------------------------------------------- | ------------------ | -------------------------- | ---------------- |
| **VALL-E 2**           | EnCodec + grouped modeling (group size 1,2,4,8) | 2-8× fewer tokens  | Human parity               | Jun 2024, Closed |
| **SoundStorm**         | EnCodec                                         | Non-autoregressive | 30s audio in 0.5s (TPU-v4) | NAR generation   |
| **Mega-TTS 2**         | Codec (unspecified)                             | Long-context LLM   | Reported SOTA              | 2024             |
| **MegaTTS** (original) | Codec TTS                                       | Prompting-based    | Zero-shot robust           | 2024             |

#### VoiceBox / Musicgen-derived

- **VoiceBox**: Flow matching for codec tokens, not autoregressive
- Uses codec + flow matching (not language modeling)
- Achieves state-of-the-art quality

#### What Sonata Is Missing

1. **Grouped codec modeling**: VALL-E 2's technique to reduce token sequence length (can apply to 12.5 Hz codec)
2. **Token rate measurement**: No benchmark vs EnCodec/DAC/WavTokenizer
3. **12.5 Hz codec training status**: Under GCE training (us-west1-a SPOT, step ~200/200K)
   - Codec architecture: FSQ (not VQ/RVQ like traditional codecs)
   - Target: 4× token reduction → ~3.125 tokens/sec
   - ~42 hours ETA (1.3 steps/s from Step 200)

---

## 5. SPECULATIVE DECODING FOR TTS

### Category: Inference Speedup via Draft Models

Speculative decoding (draft model + verifier) is well-established for LLM inference but emerging for audio language models.

| Approach                  | Context                                                         | Speed Improvement       | Status                        |
| ------------------------- | --------------------------------------------------------------- | ----------------------- | ----------------------------- |
| **Speculative TTS**       | Input-time speculation: TTS buffering overlaps with user speech | Up to 3× TTFA reduction | Research (PredGen framework)  |
| **Speculative LLM + TTS** | Cascade: LLM speculative decode + streaming TTS                 | 2-3× combined           | 2025, Active research         |
| **ReDrafter** (Sonata)    | 3.5M GRU draft model for semantic tokens                        | ~2.3× LM speedup        | Under Sonata, not yet trained |
| **Tree attention**        | Multi-token draft verification                                  | Reported in sonata_lm   | Partial implementation        |

#### Key Papers & Implementations

- **PredGen**: Uses speculative input-time decoding to reduce TTFA by 3×
- **Collective & Adaptive Spec Decode**: Distributed speculative decoding for multiple verifiers
- **Sonata ReDrafter**: 3.5M GRU draft model in sonata_lm, tree attention in training script

#### What Sonata Has vs. Missing

- ✓ ReDrafter architecture defined (3.5M params, GRU-based)
- ✓ Tree attention partial support in sonata_lm
- ✗ ReDrafter training script ready but **not yet trained** (train_drafter.py exists)
- ✗ End-to-end speculative pipeline not integrated (decode verify not hooked up)
- ✗ Metrics: no benchmark of draft acceptance rate, speedup measurement

---

## 6. RECENT SOTA SUMMARIES (2025-2026)

### Flow Matching SOTA

**CosyVoice 3.0** (Dec 2025):

- 1.5B parameters, trained on 1M hours (10 languages, 18 Chinese dialects)
- Uses supervised semantic tokens from ASR (not self-supervised)
- RL fine-tuning with reward model
- Streaming mode: 150 ms latency
- Multilingual zero-shot voice cloning
- **Gap vs Sonata**: No RL, 241M LM vs 1.5B, no ASR-supervised tokens

**MaskGCT** (Sep 2024):

- 241M params, trained on 100K hours in-the-wild speech
- **Human-level SIM-O** (0.687 LibriSpeech, human baseline ~0.67)
- Masked generative codec transformer
- Accent imitation capability (SIM-Accent metric)
- **Gap vs Sonata**: No human-parity speaker similarity, no accent preservation

**F5-TTS** (Oct 2024):

- 241M params, open source
- **4× speedup at 7 steps** via Empirically Pruned Step Sampling (EPSS)
- MOS ~4.3, WER 0.9%
- Sway sampling strategy (can apply to other models)
- **Gap vs Sonata**: Not deployed, RTF/step count not optimized

---

## 7. SONATA CURRENT STATE vs SOTA

### TTS Pipeline Strengths

| Component                  | Sonata Status                            | SOTA Reference              | Gap                         |
| -------------------------- | ---------------------------------------- | --------------------------- | --------------------------- |
| **Streaming architecture** | Full-duplex barge-in, ~320 ms round-trip | VoXtream 102 ms FPL         | 3.1× latency                |
| **Flow matching**          | Supervised regression flow               | FM + OT path (F5-TTS)       | Missing importance sampling |
| **Speaker cloning**        | Basic speaker_encoder.c                  | ECAPA-TDNN / WavLM          | No SIM-O metric             |
| **Codec**                  | FSQ 50 tokens/s                          | 12.5 Hz codec training      | 4× reduction pending        |
| **Speculative decode**     | ReDrafter architecture only              | Full pipeline + metrics     | Draft training missing      |
| **RL fine-tuning**         | None                                     | GRPO (F5R-TTS, CosyVoice 3) | +4-5% quality potential     |
| **Multilingual**           | Not tested                               | CosyVoice 3 (9 languages)   | Feature gap                 |

### Immediate Actionable Gaps (P0/P1)

**P0 (block deployment):**

1. **12.5 Hz codec training** — GCE Step ~200/200K, ETA ~42 hours (as of 2026-03-06)
   - Once ready: 4× token reduction, enables compound 9.2× speedup (4× codec × 2.3× spec decode)
2. **ReDrafter training** — Script ready (train_drafter.py), not yet scheduled
   - 3.5M GRU draft model for semantic tokens
   - Estimated 2.3× LLM speedup

**P1 (improve quality):**

1. **ECAPA-TDNN speaker encoder** — Evaluate vs current speaker_encoder.c
   - Add SIM-O metric to eval pipeline
2. **Flow matching upgrade** — Replace supervised regression with FM loss + OT path
   - Apply Sway Sampling for variable step inference
3. **RL fine-tuning** — GRPO reward model training (4-5% quality gain potential)

**P2 (streaming optimization):**

1. **Monotonic alignment** — For true streaming (currently requires full text)
2. **VoXtream-style lookahead** — Reduce TTFA from 320ms to <150ms
3. **Input-time speculation** — Overlap TTS latency with speech input buffering

---

## 8. RESEARCH PAPERS & REFERENCES

### Flow Matching & Diffusion

- [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://arxiv.org/abs/2410.06885) — Oct 2024, benchmark
- [Accelerating Flow-Matching-Based Text-to-Speech via Empirically Pruned Step Sampling](https://arxiv.org/abs/2505.19931) — 4× speedup paper
- [Towards Flow-Matching-based TTS without Classifier-Free Guidance](https://arxiv.org/abs/2504.20334v2) — Feb 2025
- [ARCHI-TTS: A flow-matching-based Text-to-Speech Model](https://arxiv.org/abs/2602.05207) — Feb 2026
- [PFLUXTTS: HYBRID FLOW-MATCHING TTS](https://arxiv.org/abs/2602.04160) — Feb 2026

### Zero-Shot Voice Cloning

- [MaskGCT: Zero-Shot Text-to-Speech](https://proceedings.iclr.cc/paper_files/paper/2025/file/74a31a3b862eb7f01defbbed8e5f0c69-Paper-Conference.pdf) — ICLR 2025
- [An Exploration of ECAPA-TDNN and x-vector Speaker Representations in Zero-shot Multi-speaker TTS](https://arxiv.org/abs/2506.20190) — 2025 study
- [Voice Cloning: Comprehensive Survey](https://arxiv.org/abs/2505.00579) — 2025

### Streaming TTS

- [VoXtream: Full-Stream Text-to-Speech with Extremely Low Latency](https://arxiv.org/abs/2509.15969) — Sep 2025, SOTA streaming
- [SpeakStream: Streaming Text-to-Speech with Interleaved Data](https://arxiv.org/abs/2505.19206) — May 2025
- [Toward Low-Latency End-to-End Voice Agents for Telecommunications](https://arxiv.org/abs/2508.04721) — 2025

### Codec-Based TTS

- [VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech](https://arxiv.org/abs/2406.05370) — Jun 2024
- [Mega-TTS 2: Boosting Prompting Mechanisms for Zero-Shot Speech Synthesis](https://arxiv.org/abs/2307.07218v3) — Oct 2024

### Codecs

- [DualCodec: A Low-Frame-Rate, Semantically-Enhanced Neural Audio Codec](https://www.isca-archive.org/interspeech_2025/li25e_interspeech.pdf) — Interspeech 2025
- [FreeCodec: A disentangled neural speech codec with fewer tokens](https://www.isca-archive.org/interspeech_2025/zheng25b_interspeech.pdf) — Interspeech 2025
- [wavtokenizer: an efficient acoustic discrete codec tokenizer](https://arxiv.org/abs/2408.16532) — 2024

### Speculative Decoding

- [Towards Efficient LLM Inference via Collective and Adaptive Speculative Decoding](https://dl.acm.org/doi/10.1145/3712285.3759834) — SC 2025

### CosyVoice

- [CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models](https://arxiv.org/abs/2412.10117) — Dec 2024
- [CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training](https://arxiv.org/abs/2505.17589) — May 2025

---

## 9. ACTIONABLE RECOMMENDATIONS

### Immediate (Next 2 Weeks)

1. **Complete 12.5 Hz codec training** — Currently Step ~200/200K on GCE
   - Monitor watchdog logs: `/opt/sonata/auto_shutdown.sh`
   - Once complete: unlock ~4× token reduction

2. **Evaluate ReDrafter training schedule** — Script exists (train_drafter.py)
   - Estimate training time on available GPU
   - Unlock ~2.3× semantic LM speedup

3. **Measure SIM-O baseline** — Against Sonata speaker encoder
   - Compare with ECAPA-TDNN reference numbers (0.65-0.68)
   - Add to eval suite

### Short-term (1-2 Months)

1. **Flow matching upgrade** — Replace supervised regression with FM loss
   - Integrate Sway Sampling for variable-step inference
   - Estimate quality improvement: +2-3% on WER/similarity

2. **RL fine-tuning pipeline** — GRPO reward model
   - Estimate: +4.6% relative SIM improvement (per F5R-TTS)
   - Training infrastructure: already have GCE setup

3. **Speaker encoder replacement** — ECAPA-TDNN or WavLM
   - Baseline comparison experiment
   - Estimate: +1-2% similarity (if significant delta)

### Long-term (2-6 Months)

1. **Streaming optimization** — Monotonic alignment + lookahead
   - Target: <150 ms TTFA (vs current 320 ms round-trip)
   - VoXtream paper provides reference implementation

2. **Full speculative pipeline** — Draft + verifier integration
   - Combine with codec reduction: 4× × 2.3× = ~9.2× compound speedup
   - Measurement infrastructure needed

3. **Multilingual expansion** — Test on 3-5 languages
   - CosyVoice 3 proves 9 languages feasible

---

## Appendix: Quality Metrics Reference

### Standard TTS Metrics

| Metric         | What It Measures                                  | Range         | Reference (SOTA 2025)        |
| -------------- | ------------------------------------------------- | ------------- | ---------------------------- |
| **MOS**        | Mean Opinion Score (subjective naturalness)       | 1-5           | ~4.3-4.5 (SOTA)              |
| **WER**        | Word Error Rate (via ASR readback)                | 0-1           | 0.009 (F5-TTS)               |
| **SIM-O**      | Speaker similarity (cosine of speaker embeddings) | 0-1           | 0.687 (MaskGCT), human ~0.67 |
| **SIM-Accent** | Accent preservation (accent classifier)           | 0-1           | 0.70+ (MaskGCT)              |
| **CER**        | Character Error Rate (Chinese/non-Latin)          | 0-1           | 0.81% (CosyVoice 3, Chinese) |
| **UTMOS**      | Non-intrusive naturalness estimate                | 0-5           | ~4.2-4.5 (SOTA)              |
| **MCD**        | Mel Cepstral Distortion (frame-level)             | dB            | <2.5 dB (SOTA)               |
| **RTF**        | Real-Time Factor (inference speed)                | <1 = realtime | 0.03 (F5-TTS @ 7 steps)      |
| **FPL/TTFA**   | First Packet / First Audio Latency                | ms            | 102 ms (VoXtream)            |

### Sonata's Current Metrics (from memory.md)

- **STT WER**: 0.9% (Conformer CTC, LibriSpeech test-clean)
- **TTS intelligibility**: 100% (Piper VITS, round-trip verified)
- **RTF (TTS)**: 0.015x (67× realtime, Piper)
- **EOU latency**: <240 ms
- **Full round-trip**: ~320 ms

#### Notes

- Sonata TTS RTF is from older Piper TTS, not flow matching
- No SIM-O measurement yet (should be ~0.60-0.65 range)
- No MOS eval on flow model yet (vs F5-TTS 4.3)
- Codec quality (FSQ) untested against WavTokenizer/DAC
