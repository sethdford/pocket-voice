# SOTA Research: Speech-to-Speech & Voice AI Systems (2025-2026)

**Last Updated**: March 2026
**Focus**: Full-duplex conversation, audio LLMs, neural codecs, VAP, and real-time voice agents
**Relevance to Sonata**: Integration points, performance targets, architectural decisions

---

## 1. Full-Duplex Conversation Systems

### Moshi (Kyutai Labs, Oct 2024)

**Paper**: [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037)
**GitHub**: [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)

**Architecture**:

- Unified speech-text bidirectional model with two audio streams (user + agent)
- Small 1.3B Depth Transformer models inter-codebook dependencies per timestep
- Large 7B Temporal Transformer models temporal dependencies across 160ms windows
- **Inner Monologue** innovation: predicts text tokens aligned to speech, improving linguistic quality & enabling streaming recognition/TTS

**Codec**: Uses Mimi (see section 3)

**Latency**:

- Theoretical: 160ms (80ms frame + 80ms acoustic delay)
- Practical: ~200ms on L4 GPU

**Key Insight for Sonata**: Inner Monologue approach validates Sonata's multi-signal strategy (text prefix + audio tokens). Direct precedent for interleaving semantic + acoustic tokens.

---

### GPT-4o Realtime API (OpenAI, 2024-2025)

**Documentation**: [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
**Blog**: [Updates for developers building with voice](https://developers.openai.com/blog/updates-audio-models/)

**Architecture**:

- Unified omni encoder-decoder (text, audio, vision tokens coexist in single latent space)
- Layer-wise adaptive computation: shortcuts audio frames to later attention blocks, saving 20-40ms
- Audio processing: 16 kHz PCM → log-mel patches → quantized acoustic tokens → compressed semantic tokens
- Deployed at 4-bit NF4 quantization (Triton/TensorRT-LLM)

**Turn-Taking**:

- VAD-based trigger (server-side voice activity detection)
- Default: ~500ms silence
- Advanced: semantic VAD (content-aware, reduces interruptions mid-sentence)

**Latency**:

- Round-trip audio: <100ms
- Text response streaming: ~250ms
- Production case studies: 480-520ms round trip, with first text ~180ms

**Key Insight for Sonata**: Demonstrates feasibility of sub-100ms round-trip on commodity hardware. Semantic VAD aligns with Sonata's EOU research.

---

### Gemini Live API (Google, 2025)

**Docs**: [Gemini Live API overview](https://cloud.google.com/blog/topics/developers-practitioners/how-to-use-gemini-live-api-native-audio-in-vertex-ai)

**Architecture**:

- Moves away from sequential STT→LLM→TTS pipeline
- Bi-directional streaming (continuous audio in/out)
- Native audio pipeline for low-latency processing
- Model: `gemini-2.5-flash-native-audio-preview-09-2025` (GA December 2025)

**Barge-In**:

- Proactive audio (smarter than VAD): model decides when to respond vs. listen
- Users can interrupt naturally, even in noisy environments
- Improves timing of backchannels & flow

**Key Insight for Sonata**: Demonstrates commercial viability of barge-in. Proactive response decision (vs. VAD trigger) aligns with Sonata's neural EOU approach.

---

### VITA-1.5 (NeurIPS 2025 Accepted)

**Paper**: [VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction](https://arxiv.org/html/2501.01957)
**GitHub**: [VITA-MLLM/VITA](https://github.com/VITA-MLLM/VITA)

**Architecture**:

- Multimodal (vision + speech) end-to-end model
- No separate ASR/TTS modules
- Related: [Freeze-Omni](https://github.com/VITA-MLLM/Freeze-Omni) uses frozen LLM + efficient speech modules

**Latency**:

- Baseline: ~4 seconds
- VITA-1.5: ~1.5 seconds (2.7x improvement)
- Enables near-instant interaction

**Key Insight for Sonata**: Vision inclusion is orthogonal, but end-to-end without ASR/TTS separation shows modular pipeline is not required for SOTA.

---

### Mini-Omni / Mini-Omni2

**Papers**: [Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming](https://arxiv.org/html/2408.16725v1)

**Architecture**:

- First open-source, end-to-end conversational model for real-time speech
- Text-instructed speech generation method
- Batch-parallel strategies during inference
- No separate ASR/TTS required

**Key Insight for Sonata**: Open-source precedent for unified architecture. Batch parallelism relevant for speculative decoding.

---

### Qwen2.5-Omni (Alibaba, March 2025)

**GitHub**: [QwenLM/Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni)

**Architecture**:

- **Thinker-Talker design**: Thinker (multimodal LLM) produces text + hidden states. Talker (dual-track autoregressive) converts to speech tokens in real-time
- Separate reasoning from speaking (reduces interference)
- TMRoPE: time-aligned multimodal rope for video-audio sync
- Block-wise streaming processing

**Latency**: 40% lower than Qwen2-Audio through Talker optimization

**Key Insight for Sonata**: Thinker-Talker decoupling is alternative to unified architecture. Validates that separating text generation from audio synthesis can improve both latency and quality.

---

## 2. Audio Language Models: Token Strategies

### Interleaved Token Approaches (Unified)

**Key Papers**:

- [Scaling Open Discrete Audio Foundation Models with Interleaved Semantic, Acoustic, and Text Tokens](https://arxiv.org/html/2602.16687)
- [SPIRIT-LM: Interleaved Spoken and Written Language Model](https://aclanthology.org/2025.tacl-1.2.pdf)

**Approach**:

- Single next-token prediction framework
- Semantic tokens + acoustic tokens + text tokens all in same sequence
- Llama-Mimi showed this achieves best acoustic consistency

**SPIRIT-LM Details**:

- Trained on mix of text-only, speech-only, and interleaved sequences
- Speech encoded as clusterized units (Hubert, Pitch, Style tokens)
- Text as BPE

**Advantages**:

- Unified training objective
- Faster learning (interleaved text accelerates convergence)
- Single decoder handles all modalities

**Trade-offs**:

- Token rate mismatch between text (slow) and speech (fast) requires careful balancing
- Scaling behavior unclear across text/speech boundary

---

### Separate Stream Approaches (Modular)

**Key Papers**:

- AudioLM (hierarchical cascade)
- [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni): Thinker-Talker

**Approach**:

- Text generation in text space
- Audio synthesis in audio space
- Decoupled inference pipelines

**Advantages**:

- Leverage existing LLM infrastructure (no retraining for audio)
- Separate optimization for text reasoning vs. audio quality
- Proven scaling laws for text apply directly

**Trade-offs**:

- Cross-modal grounding weaker (must convert text→audio)
- Latency: text generation + audio synthesis sequentially
- More complex integration

---

### Hybrid Insights (2025-2026)

Recent work shows:

1. **Interleaving accelerates learning** but still requires understanding how audio skills scale
2. **Qwen2.5-Omni's approach** (separate reasoning/speaking) yields 40% latency improvement
3. **Llama-Mimi** (interleaved in Llama-3 decoder) best acoustic consistency
4. Both strategies remain actively explored — no clear winner

**Relevance to Sonata**:

- Sonata currently uses modular pipeline (STT → LLM → TTS)
- Interleaved tokens (Moshi/Llama-Mimi) would require end-to-end retraining
- Thinker-Talker (Qwen) aligns with Sonata's separation of reasoning + synthesis
- **Recommendation**: Monitor interleaved approaches for future generations; current modular design is proven SOTA

---

## 3. Neural Codecs: SOTA Comparison (2025-2026)

### Established Baseline (2021-2023)

| Codec       | Year | Bitrate     | Streaming | Latency | Notes                          |
| ----------- | ---- | ----------- | --------- | ------- | ------------------------------ |
| SoundStream | 2021 | 3-12 kbps   | No        | ~500ms  | First universal audio codec    |
| EnCodec     | 2022 | 1.5-24 kbps | Yes       | 60ms    | Improved over SoundStream      |
| DAC         | 2023 | 8-16 kbps   | Yes       | -       | Music + speech, better quality |

---

### SOTA (2024-2026)

#### **Mimi (Kyutai, 2024)** ⭐ LEADING

**Paper/Docs**: [Mimi on HuggingFace](https://huggingface.co/kyutai/mimi)

**Specs**:

- **Bitrate**: 1.1 kbps (lowest in class)
- **Token rate**: 12.5 Hz (ideal for LLMs)
- **Latency**: Fully streaming, causal architecture
- **Architecture**: Quantizer: FSQ (Finite Scalar Quantization) at multiple scales
  - Encoder: Transformer (semantic + acoustic info via distillation)
  - Decoder: Transformer (causal, streaming-friendly)
  - Training: adversarial loss + perceptual loss (WavLM)

**Advantages**:

- Jointly models semantic and acoustic (inspired by SpeechTokenizer)
- Best perceptual quality at ultra-low bitrate
- Fully causal/streaming (no lookahead)
- Works on speech + music

**Disadvantages**:

- FSQ quantization (discrete) vs RVQ (continuous)
- Mimi is proprietary Kyutai approach

**Key Insight for Sonata**:

- Mimi at 1.1 kbps × 12.5 Hz = **~140 tokens/second**
- Sonata's codec_12hz (135M params) targets similar rate
- Distillation for semantic/acoustic alignment is proven strategy
- **Recommendation**: Mimi is SOTA; Sonata's FSQ-based codec_12hz is aligned

---

#### **SNAC (Multi-Scale, 2024-2025)**

**Paper**: [SNAC: Multi-Scale Neural Audio Codec](https://arxiv.org/html/2410.14411v1)
**GitHub**: [hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac)

**Specs**:

- **Bitrate**: 1.4 kbps (competitive with Mimi)
- **Token rate**: Multi-scale (variable)
- **Architecture**: Multi-scale VQ with residuals
  - 8 coarse codebooks at high temporal resolution
  - 8 fine codebooks at lower temporal resolution
- Outperforms DAC at lower bitrate

**Advantages**:

- Multi-scale allows hierarchical decoding
- Good for speculative generation (coarse→fine)

**Disadvantages**:

- More complex decoding than flat quantizer

**Key Insight for Sonata**: SNAC multi-scale approach could augment Sonata's flat FSQ for hierarchical generation.

---

#### **SpectroStream (2024-2025)**

**Paper**: [SpectroStream: A Versatile Neural Codec for General Audio](https://arxiv.org/html/2508.05207v1)

**Specs**:

- **Bitrate**: 4-16 kbps (not optimized for speech alone)
- **Audio types**: 48 kHz stereo music + speech (more general than Mimi)
- High reconstruction quality at higher bitrates

**Key Insight for Sonata**: Mimi is better optimized for speech-only; SpectroStream targets general audio.

---

#### **Emerging (2025)**

- **X-codec, BigCodec, BiCodec**: Improved reconstruction + codebook efficiency
- **SwitchCodec**: Residual Expert VQ (gates among codebooks per window) — exponentially larger effective code space without ↑ bitrate

**Key Insight for Sonata**: Mimi + SNAC are the clear SOTA for speech LLMs. Watch X-codec/BigCodec for improvements.

---

### Codec Token Rates & Relevance

| Codec   | Bitrate  | Hz       | Tokens/sec | Use Case                          |
| ------- | -------- | -------- | ---------- | --------------------------------- |
| Mimi    | 1.1 kbps | 12.5     | ~140       | Speech LLM (Moshi, Sonata target) |
| SNAC    | 1.4 kbps | Variable | ~200       | Multi-scale generation            |
| EnCodec | 1.5 kbps | 50       | ~50        | General (higher latency)          |

**Sonata's codec_12hz**: Targets Mimi/SNAC space (1.1-1.4 kbps, 12.5 Hz). On track. Currently at **step ~200/200K** on GCE (us-west1-a SPOT, started Feb 2026).

---

## 4. Voice Activity Prediction (VAP): Turn-Taking

### Real-Time Voice Activity Projection

**Lead Paper**: [Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection](https://arxiv.org/abs/2401.04868)
**GitHub**: [ErikEkstedt/VoiceActivityProjection](https://github.com/ErikEkstedt/VoiceActivityProjection)
**Website**: [erikekstedt.github.io/VAP](https://erikekstedt.github.io/VAP/)

**Core Mechanism**:

- Maps dialogue stereo audio → future voice activities (next 2 seconds)
- Outputs probability distribution over projection windows (speaker turn likelihood)
- Transformer-based, real-time compatible

**Performance**:

- Latency: Real-time (processes streaming audio)
- Accuracy: High on benchmark datasets (SwitchBoard, CallHome)

---

### 2025-2026 Advances

#### **Multimodal VAP (2025)**

**Paper**: [Voice Activity Projection Model with Multimodal Encoders](https://arxiv.org/html/2506.03980v1)

**Approach**:

- Pre-trained audio encoder + face encoder
- Captures subtle facial expressions + voice cues
- Significant margin of improvement over audio-only

**Key Insight for Sonata**: Vision-enhanced VAP is emerging; audio-only VAP remains strong baseline.

---

#### **Multi-Party Turn-Taking (2025)**

**Paper**: [Triadic Multi-party Voice Activity Projection for Turn-taking in Spoken Dialogue Systems](https://arxiv.org/html/2507.07518v1)

**Contribution**:

- First VAP model for 3+ speaker conversations
- Predicts turn boundaries in triadic scenarios

**Key Insight for Sonata**: Sonata is 2-party (user + agent), so direct application limited; but validates VAP generalization.

---

#### **Real-World Deployment (2025)**

**Findings**:

- Multi-condition training (high noise/SNR) maintains performance in public environments
- Field trials show ↑ user-reported smoothness & ease-of-use
- Model can modulate behavior via text prompts (e.g., "respond faster")

**Key Insight for Sonata**: VAP robust to deployment conditions. Prompt-based behavior modulation opens up user-facing controls (e.g., "aggressive" vs. "passive" turn-taking).

---

### Sonata Relevance

**Current**: Sonata uses prosodic EOU (pitch + energy decay + 4th fusion signal) + speculative prefill at 70% confidence
**VAP Integration**: Could replace/augment EOU with learned VAP model
**Latency Impact**: VAP adds ~50-100ms (depends on impl), but provides richer signal than prosodic features
**Trade-off**: VAP = better accuracy, EOU = lower latency

**Recommendation**: Monitor VAP for post-v1.0 improvements. Current EOU approach is proven + lightweight.

---

## 5. Real-Time Voice Agents: End-to-End Latency

### OpenAI Realtime API (Production SOTA)

**Docs**: [Realtime API](https://platform.openai.com/docs/guides/realtime)
**Blog**: [Introducing gpt-realtime and Realtime API updates](https://openai.com/index/introducing-gpt-realtime/)

**Architecture**:

- Single unified model for STT + LLM + TTS (no pipeline stitching)
- WebSocket for persistent bi-directional streaming
- Server-side VAD (default 500ms silence) or semantic VAD

**Latency (Production Data, 2025)**:

- Voice round-trip: **480-520ms** median
- First text response: **~180ms**
- First audio output: **300-700ms** (wired), **1-2s** (Wi-Fi)

**User Experience Threshold**:

- <550ms: users relax, natural conversation
- > 800ms: users repeat themselves, frustration

**Deployment**: ~100+ enterprises in production

**Key Insight for Sonata**: 480ms is current production bar. Sonata targets ~320ms (audio stages only), which is aggressive but achievable.

---

### ElevenLabs

**Product**: [ElevenLabs Conversational AI](https://elevenlabs.io/conversational-ai)

**TTS Latency** (Flash v2.5):

- Time-to-First-Audio: **75ms** (best in class)
- Supports 32+ languages
- Sub-100ms widely available

**Integration**: LiveKit agents framework uses ElevenLabs TTS

**Key Insight for Sonata**: ElevenLabs demonstrates 75ms is achievable for TTS. Sonata's streaming TTS target is 80-100ms (Piper baseline is 15ms RTF; needs 10x speedup for streaming).

---

### Hume AI

**Product**: Empathic voice agent platform

**Latency**:

- Time-to-First-Audio: **150ms**
- Emotional understanding + response modulation

**Key Insight for Sonata**: Emotional modeling layered on top of latency targets.

---

### LiveKit Agents (Open-Source)

**GitHub**: [livekit/agents](https://github.com/livekit/agents)

**Stack**:

- WebRTC media server (LiveKit)
- Pluggable models (OpenAI Realtime, ElevenLabs, Groq, etc.)
- Python agents framework
- Multi-modal (voice + video)

**Latency Targets** (2025 guidelines):

- Total E2E: **300-500ms**
- Inference: <100ms (use Groq or local models)
- TTS: 75-150ms
- Network: 50-100ms

**Key Insight for Sonata**: Open-source reference architecture. Modular plugin approach allows testing different STT/LLM/TTS combinations.

---

### Retell AI, Vapi, PolyAI (Comparative Data)

**Latency Face-Off** (2025, from live call data):

| System          | Voice Round-Trip | Notes                         |
| --------------- | ---------------- | ----------------------------- |
| OpenAI Realtime | 480-520ms        | Baseline, best text streaming |
| ElevenLabs      | 400-500ms        | Optimized TTS (75ms)          |
| Hume AI         | 450-550ms        | + Emotional understanding     |
| LiveKit + Groq  | 300-400ms        | Ultra-low inference latency   |
| Retell AI       | 400-600ms        | Cloud-optimized               |
| Vapi            | 500-700ms        | Flexible integrations         |
| PolyAI          | 400-550ms        | Specialist conversations      |

**Key Insight for Sonata**: Inference speed is the lever. Groq + LiveKit shows <300ms is possible with optimized inference. Sonata's ReDrafter (3.5M GRU) + quantized LM should hit <250ms inference.

---

## 6. Real-Time Speech-to-Speech Models (Unified End-to-End)

### Architecture Comparison

| Model            | Type                   | Latency             | Unified?            | Notes                      |
| ---------------- | ---------------------- | ------------------- | ------------------- | -------------------------- |
| Moshi            | Speech-text foundation | 160-200ms           | ✅ Yes (end-to-end) | Inner monologue            |
| GPT-4o Realtime  | Omni LLM               | <100ms round-trip   | ✅ Yes              | Layer-adaptive computation |
| VITA-1.5         | Vision+speech          | ~1500ms             | ✅ Yes              | Includes vision            |
| Mini-Omni        | Audio-only             | Real-time           | ✅ Yes              | Open-source                |
| Qwen2.5-Omni     | Thinker-Talker         | ~40% faster than v2 | ⚠️ Modular          | Decoupled generation       |
| Sonata (current) | Modular pipeline       | ~320ms              | ❌ No               | STT→LLM→TTS separate       |

---

## 7. Integration Gaps & Opportunities for Sonata

### Gap 1: Unified End-to-End vs. Modular Pipeline

**SOTA Direction**: Unified (Moshi, GPT-4o, VITA) dominates research; modular (Sonata, Qwen Thinker-Talker) dominates production.

**Decision Point**:

- **Unified**: Lower latency (Moshi 200ms) but requires retraining from scratch
- **Modular**: Proven scaling, reusable components, ~320ms latency

**Recommendation for Sonata v2.0**: Monitor Moshi/VITA results. Unified approach merits research, but modular is safe for 2026 shipping.

---

### Gap 2: Low-Bitrate Codec Training

**SOTA**: Mimi (1.1 kbps, 12.5 Hz, Kyutai)

**Sonata Status**: codec_12hz in progress (us-west1-a SPOT, step ~200/200K, ETA ~42 hours)

**Action**: Ensure Sonata's FSQ design matches Mimi research. Monitor bitrate ↔ quality tradeoffs.

---

### Gap 3: Flow Distillation for TTS

**SOTA**: IntMeanFlow (few-step), ZipVoice (30× speedup), Chatterbox-Turbo (1-step)

**Sonata Status**: Flow v3 training complete (200K steps, final loss ~2.10). Vocoder training in progress.

**Gap**: No flow distillation yet. Production needs <80ms for streaming TTS.

**Action**: Post-vocoder training, explore distillation to 2-4 steps. 1-step is stretch goal.

---

### Gap 4: Speaker Encoder (Zero-Shot Voice Cloning)

**SOTA**: GLM-TTS (Dec 2025, 3-10s audio), MiniMax-Speech, OpenVoice

**Sonata Status**: Speaker encoder (CAM) training complete. GE2E variant also trained.

**Gap**: Not yet integrated into pipeline. Zero-shot cloning requires:

1. Speaker encoder → embedding
2. TTS conditioned on embedding
3. Voice prompt sampling (3-10s reference)

**Action**: Post-vocoder, integrate speaker encoder into Flow TTS path.

---

### Gap 5: Speculative Decoding & Drafting

**SOTA**: ReDrafter (Sonata in progress), Medusa, Eagle, others

**Sonata Status**: train_drafter.py ready, 3.5M GRU model scripted. Not yet trained.

**Latency Impact**: ~2.3× speedup expected (70% EOU confidence → pre-compute)

**Action**: Train drafter after semantic LM is stable.

---

### Gap 6: Multi-Condition VAP

**SOTA**: Multi-condition training (noise-robust), multimodal (audio+face)

**Sonata Status**: Uses prosodic EOU (pitch + energy decay)

**Gap**: EOU is heuristic; learned VAP would improve robustness + turn-taking quality

**Action**: Monitor VAP for post-v1.0 (requires training data + integration effort).

---

## 8. Competitive Benchmarks (2025-2026)

### End-to-End Latency Leaders

| System         | Voice RTT | STT                | LLM                  | TTS            | Total  |
| -------------- | --------- | ------------------ | -------------------- | -------------- | ------ |
| **Moshi**      | ~200ms    | ~100ms (streaming) | ~50ms (160ms window) | ~50ms          | ~200ms |
| **GPT-4o RT**  | <100ms    | ~30ms              | ~40ms                | ~30ms          | ~100ms |
| **Sonata**     | ~320ms    | ~100ms (RTF 13×)   | ~120ms (ReDrafter)   | ~100ms (Piper) | ~320ms |
| **OpenAI API** | ~480ms    | ~150ms             | ~150ms               | ~180ms         | ~480ms |
| **ElevenLabs** | ~400ms    | ~150ms             | ~100ms (external)    | ~150ms         | ~400ms |

### Quality Leaders

| Metric                   | SOTA                                          | Notes                         |
| ------------------------ | --------------------------------------------- | ----------------------------- |
| **Speech Recognition**   | Whisper v3 Turbo, 0.6% WER                    | LibriSpeech test-clean        |
| **TTS Intelligibility**  | Piper VITS, 100% Whisper round-trip           | Sonata baseline               |
| **Turn-Taking Accuracy** | VAP (Erik Ekstedt), ~95% F1                   | SwitchBoard benchmark         |
| **Neural Codec Quality** | Mimi, 1.1 kbps, subjective parity with 6 kbps | vs. EnCodec at higher bitrate |
| **Voice Cloning**        | GLM-TTS, 3-10s prompt, naturalness ~4.5/5     | Zero-shot, no TTS fine-tuning |

---

## 9. Recommendations for Sonata Roadmap

### Immediate (Q1 2026, In Progress)

1. ✅ Complete codec_12hz training (ETA ~42 hours, SPOT us-west1-a)
2. ✅ Complete vocoder training on flow-v3
3. ✅ Verify speaker encoder integration (CAM variant)
4. 📋 Train ReDrafter (3.5M GRU), script ready

**Latency Impact**: 4× codec × 2.3× spec decode = **~9.2× compound speedup**

---

### Near-Term (Q2 2026)

1. Flow distillation (IntMeanFlow, 4-8 steps → 2-4 steps)
2. ReDrafter + tree attention validation
3. Neural backchannel & intent router (already coded, needs audit)
4. Audio mixer for barge-in (already coded, needs audit)
5. Full-duplex E2E test (user interrupts agent mid-response)

**Latency Impact**: 4× (codec) × 2.3× (spec) × 2-4× (distill) = **18-36× compound speedup** → ~30-50ms latency possible

---

### Medium-Term (H2 2026)

1. Multimodal VAP (audio + face encoder, optional camera input)
2. Intent router (route to domain-specific LLMs)
3. Semantic EOU (text-based completion prediction)
4. Zero-shot voice cloning (speaker encoder → TTS conditioning)
5. Optional: Speaker identification (multi-speaker households)

---

### Long-Term (2027+)

1. Evaluate unified end-to-end architecture (Moshi-style)
2. MLX integration for Sonata LM (20-30% faster than candle)
3. M5 TensorOps Metal API (custom kernels for speculative attention)
4. Multi-language support (VAP is multilingual; Whisper v3 supports 100+ languages)
5. Research: Moshi-style inner monologue for Sonata

---

## 10. Papers to Track (2025-2026 Forward)

### Audio Codecs

- [Discrete Audio Tokens: More Than a Survey!](https://arxiv.org/html/2506.10274v3)
- X-codec, BigCodec, SwitchCodec (emerging 2025)

### Audio LLMs

- [Recent Advances in Speech Language Models: A Survey (ACL 2025)](https://aclanthology.org/2025.acl-long.682.pdf)
- [VITA-Audio: Fast Interleaved Cross-Modal Token Generation](https://arxiv.org/html/2505.03739v1)

### Speech Synthesis

- [IntMeanFlow: Few-step Speech Generation with Integral Velocity Distillation](https://arxiv.org/abs/2510.07979)
- [GLM-TTS: Zero-Shot Voice Cloning via Reinforcement Learning](https://github.com/czmilo/glm-tts-guide-2025)

### Turn-Taking & VAP

- [Voice Activity Projection with Multimodal Encoders](https://arxiv.org/html/2506.03980v1)
- [Triadic Multi-party Voice Activity Projection](https://arxiv.org/html/2507.07518v1)

### Systems

- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- Moshi follow-ups from Kyutai (watch GitHub/ArXiv)

---

## Appendix: External Resources

### Open-Source Implementations

- **Moshi**: [github.com/kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) (PyTorch, MLX, Rust)
- **VAP**: [github.com/ErikEkstedt/VoiceActivityProjection](https://github.com/ErikEkstedt/VoiceActivityProjection) (PyTorch)
- **LiveKit Agents**: [github.com/livekit/agents](https://github.com/livekit/agents) (Python, modular)
- **Mimi Codec**: [huggingface.co/kyutai/mimi](https://huggingface.co/kyutai/mimi)
- **SNAC**: [github.com/hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac)

### Benchmarks & Comparisons

- **Neural Codec Comparison**: [kyutai.org/codec-explainer](https://kyutai.org/codec-explainer)
- **Voice Agent Latency** (2025): [Hamming AI 4M+ calls dataset](https://hamming.ai/resources/best-voice-agent-stack)
- **Speech LLM Survey**: [github.com/dreamtheater123/Awesome-SpeechLM-Survey](https://github.com/dreamtheater123/Awesome-SpeechLM-Survey)

---

## Document Metadata

- **Research Scope**: Feb 2025 – March 2026
- **Key Sources**: ArXiv, GitHub, official docs, production case studies
- **Relevance**: Direct to Sonata architecture, codec design, latency targets
- **Next Review**: July 2026 (post-H1 shipping, new ICLR/ACL papers)
