# Architecture

Sonata is a zero-Python, real-time voice pipeline for Apple Silicon. Every component — from speech recognition to text-to-speech — runs natively in C and Rust, using all four Apple Silicon compute units concurrently.

## Pipeline Overview

```
Mic (48kHz) → STT → LLM → TTS → Speaker (48kHz)
```

The pipeline processes voice input through five stages, orchestrated by a state machine in `pocket_voice_pipeline.c`.

## State Machine

```
Listening → Recording → Processing → Streaming → Speaking → Listening
                │           │            ↑
                │           │  (speculative prefill at 70% EOU)
                │           │
                └───────────┘  Barge-in (any state) → Listening
```

| State          | Description                                               |
| -------------- | --------------------------------------------------------- |
| **IDLE**       | Pipeline initialized, waiting for activation              |
| **LISTENING**  | Mic active, VAD monitoring for speech onset               |
| **RECORDING**  | Speech detected, audio being captured and streamed to STT |
| **PROCESSING** | End-of-utterance detected, transcript sent to LLM         |
| **STREAMING**  | LLM tokens arriving, being fed to TTS                     |
| **SPEAKING**   | TTS audio playing through speaker                         |

Barge-in (speaking over the system) triggers an immediate return to LISTENING from any state.

## Audio Flow

### Input Path

```
Mic (48kHz) → Resample (16kHz) → Noise Gate → STT (Conformer/Sonata)
```

- **CoreAudio VoiceProcessingIO** provides hardware AEC (acoustic echo cancellation)
- Dual ring buffers (capture + playback) with 128-byte cache-line aligned atomics
- Noise gate reduces stationary background noise via spectral gating (512-point FFT)

### Output Path

```
TTS (24kHz) → Pitch Shift → Formant EQ → Volume → Soft Limit
  → LUFS Normalize → Breath Insert → Resample (48kHz) → Spatial Audio → Speaker
```

Each post-processing stage is optional and controlled via CLI flags or config.

## End-of-Utterance Detection

Sonata uses a 3-signal fused EOU system to determine when the user has finished speaking. This is critical for conversational latency — triggering too early cuts off the user, too late adds perceptible delay.

### Signal 1: Energy VAD

- `vDSP_rmsqv` on capture frames
- Hysteresis-based speech/silence classification
- Near-zero latency but prone to false positives during natural pauses

### Signal 2: Mimi Endpointer (LSTM)

- Single-layer LSTM on 80-band mel-energy features
- 4-class softmax: silence, speech, ending, end-of-turn
- Fed at ~12.5 Hz from capture audio
- AMX-accelerated via `cblas_sgemv`

### Signal 3: ASR-Inline EOU Token

- Conformer CTC decoder detects `<eou>` tokens in the recognition output
- Per-frame softmax probability, smoothed over recent frames
- Most semantically meaningful signal — the model understands sentence completion

### Fusion

The three signals are combined in `fused_eou.c`:

- **Weighted average**: 0.4 energy + 0.3 Mimi + 0.3 STT
- **Solo thresholds**: Any signal > 0.95 triggers immediately
- **EMA smoothing** on fused probability (alpha = 0.3)
- **Consecutive frame requirement**: 2 frames (160ms) to prevent false triggers
- **Speech gate**: Won't trigger on initial silence

### Speculative Prefill

At 70% fused EOU confidence, Sonata speculatively sends the current transcript to the LLM. If the user keeps speaking (confidence drops below 30%), the request is cancelled. If EOU confirms, the LLM response is already in-flight — saving 100–300ms of perceived latency.

## Sonata TTS Pipeline

The flagship TTS system is a 3-stage neural pipeline:

### Stage 1: Sonata LM (241.7M params)

A Llama-style transformer that converts text tokens into semantic audio tokens at 50 Hz.

- 16 layers, 1024 d_model, GQA (16 heads, 4 KV groups)
- RoPE positional encoding, SwiGLU FFN
- FP16 inference on Metal GPU (~45 tok/s)
- Top-k + top-p sampling with repetition penalty
- Supports streaming text append for sub-sentence generation

### Stage 2: Sonata Flow (35.7M params)

Conditional flow matching that transforms semantic tokens into magnitude + instantaneous frequency spectrograms.

- Euler/Heun ODE solver (configurable steps: 4–16)
- Classifier-free guidance (CFG) for quality control
- Speaker conditioning via embedding table or external embedding (voice cloning)
- Emotion conditioning via embedding table or EmoSteer activation steering
- Prosody conditioning (pitch, energy, rate)
- Cumulative phase for frame-to-frame continuity

### Stage 3: iSTFT Decoder

Converts magnitude + phase spectrograms back to audio waveform.

- Pure vDSP/AMX implementation, no neural network
- 5,000x+ faster than realtime
- Alternative: ConvNeXt neural decoder or ConvDecoder (HiFi-GAN style)

### Streaming Chunked Generation

Sonata generates audio in adaptive chunks rather than waiting for full sentences:

1. **First chunk**: 12 semantic tokens (~240ms of audio) for low time-to-first-audio
2. **Subsequent chunks**: 20–80 tokens with prosody-aware boundary detection
3. **Crossfade**: Smooth transitions between chunks
4. **Pipeline-parallel**: Flow+decoder runs on GPU while LM generates the next chunk concurrently

### Sub-Sentence Streaming

An eager 4-word flush feeds text to Sonata LM before the full sentence arrives from the LLM. The LM's `append_text()` API extends the text buffer mid-generation without resetting the KV cache.

## Hardware Dispatch

All three Apple Silicon compute units run concurrently:

| Unit                | What Runs                                     | How                                           |
| ------------------- | --------------------------------------------- | --------------------------------------------- |
| **Metal GPU**       | STT + TTS inference (candle)                  | Transformer attention, FFN, codec             |
| **AMX Coprocessor** | Prosody, FFT, LUFS, LSTM, mel spectrogram     | `cblas_sgemv`, `vDSP_fft_zrip`, `vDSP_biquad` |
| **ARM NEON**        | PCM conversion, crossfade, ring buffer copies | 8-wide float32 SIMD intrinsics                |

Additionally, the **Apple Neural Engine (ANE)** can be used via BNNS/CoreML for:

- BNNS ConvNeXt Decoder (offloads decoder from GPU, freeing it for flow inference)
- CoreML EP for ONNX models (speaker encoder)

## Memory Architecture

### Arena Allocator

Per-turn bump-pointer arena (`arena.h`) provides zero-free memory management. All allocations within a conversational turn are freed in one operation at turn end.

### Ring Buffers

- **SPSC ring**: Lock-free single-producer single-consumer for audio I/O between CoreAudio thread and pipeline
- **SPMC ring**: Single-producer multi-consumer for fan-out from post-processor to speaker, Opus encoder, WebSocket, etc.
- **VM-mirrored ring** (`vm_ring.c`): Uses `mach_vm_remap` for zero-copy wraparound reads

### Zero-Allocation Hot Path

The CoreAudio callback (real-time thread) performs no allocations, locks, or syscalls. All audio buffer management uses pre-allocated ring buffers with cache-line aligned atomics.

### KV Cache Layout

Interleaved `[H][T][2][D]` layout places K and V for the same (head, time) position in adjacent cache lines, halving L2 misses during attention computation.

## LLM Integration

Sonata supports three LLM backends via the `LLMClient` function-pointer interface:

| Backend         | Connection                                   | Notes                                 |
| --------------- | -------------------------------------------- | ------------------------------------- |
| **Claude**      | Anthropic Messages API via libcurl SSE       | Default, requires `ANTHROPIC_API_KEY` |
| **Gemini**      | Google streamGenerateContent via libcurl SSE | Requires `GEMINI_API_KEY`             |
| **Local Llama** | On-device Llama-3.2-3B via candle + Metal    | No API key needed, ~43 tok/s          |

All backends stream responses token-by-token for immediate TTS feeding.

## Prosody System

Sonata includes a multi-layer prosody system:

1. **SSML Parser**: Handles `<prosody>`, `<break>`, `<emphasis>`, `<emotion>`, `<voice>` tags
2. **Auto-Intonation**: Punctuation-based rules (questions +8% pitch, exclamations +6%, etc.)
3. **Emphasis Prediction**: Linguistics-based insertion at contrast markers, intensifiers, negations
4. **Emotion Detection**: Text-based emotion keywords and punctuation patterns
5. **Conversational Adaptation**: EMA-tracked user speech characteristics, partially mirrored
6. **Model-Aware Prosody**: Parameters passed to both Sonata LM and Flow for conditioned generation
7. **EmoSteer**: Training-free emotion control via activation steering in the Flow network

## Key Source Files

| File                      | Lines | Role                                          |
| ------------------------- | ----- | --------------------------------------------- |
| `pocket_voice_pipeline.c` | ~6400 | Main orchestrator, state machine, LLM clients |
| `pocket_voice.c`          | —     | CoreAudio engine, ring buffers, VAD           |
| `fused_eou.c`             | —     | 3-signal EOU fusion                           |
| `sonata_istft.c`          | —     | iSTFT decoder (vDSP)                          |
| `conformer_stt.c`         | —     | Pure C FastConformer CTC                      |
| `src/sonata_lm/`          | —     | Rust: Sonata LM (candle + Metal)              |
| `src/sonata_flow/`        | —     | Rust: Sonata Flow + decoder (candle + Metal)  |
| `vdsp_prosody.c`          | —     | AMX audio effects (pitch, EQ, volume)         |
| `http_api.c`              | —     | REST API server                               |
| `websocket.c`             | —     | WebSocket protocol                            |
