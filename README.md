# Sonata

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)]()
[![Language](https://img.shields.io/badge/C%20%2B%20Rust-native-orange)]()

Real-time voice intelligence for Apple Silicon. Native C + Rust. Full-duplex conversation.

## Quick Start

```bash
brew install curl opus onnxruntime espeak-ng
make
ANTHROPIC_API_KEY=sk-ant-... ./sonata
```

## Why Sonata?

Most voice pipelines are Python glue connecting cloud APIs. Sonata is different — every audio processing component from speech recognition to text-to-speech runs natively on Apple Silicon, using all four hardware compute units simultaneously (GPU + AMX + ANE + NEON). The result: low-latency full-duplex conversation with barge-in support, entirely on your Mac.

```
Mic → STT → LLM → TTS → Speaker
```

> **Note:** The default LLM backend is Claude (cloud API). Use `--llm local` for fully on-device inference with Llama 3.2.

## Key Features

- **100% native audio pipeline**: C + Rust. Single `make` builds 43 shared libraries, 5 Rust crates, custom Metal kernels, and one binary.
- **Full-duplex barge-in**: CoreAudio VoiceProcessingIO with hardware AEC. Speak while the assistant is talking — playback interrupts immediately.
- **Fused 3-signal end-of-utterance**: Energy VAD + LSTM endpointer + ASR-inline EOU token, weighted and fused for <240ms turn detection.
- **Speculative prefill**: LLM request fires at 70% EOU confidence. Saves 100-300ms when the prediction is correct.
- **Multi-engine STT**: Conformer CTC (0.9% WER on LibriSpeech), Kyutai Rust 1B, BNNS/ANE-accelerated. Switch with `--stt-engine`.
- **Apple Silicon optimized**: Every stage runs on the right compute unit — Metal GPU for transformers, AMX for DSP, ANE for power-efficient inference, NEON for SIMD.

## TTS Status

Sonata currently ships with two **production-ready** TTS engines and one **in-development** custom TTS:

| Engine                           | Quality                                             | Speed                       | Status               |
| -------------------------------- | --------------------------------------------------- | --------------------------- | -------------------- |
| **Piper VITS** (60M, ONNX)       | 100% intelligibility (Whisper-verified)             | 67x realtime, 51ms/sentence | **Production ready** |
| **Supertonic-2** (250M, ONNX)    | 70-90% (voice-dependent), 10 voices                 | 48x realtime                | **Production ready** |
| **Sonata TTS** (294M, Metal GPU) | In training — not yet producing intelligible speech | Pipeline functional         | **In development**   |

The default pipeline uses Piper for TTS. Sonata TTS is a custom from-scratch system designed for Apple Silicon — the inference pipeline is complete and mechanically correct, but the models require training on paired speech data before producing quality output.

### Sonata TTS Architecture (In Development)

```
Text → Semantic LM (241M, Metal GPU) → Semantic Tokens (50 Hz)
     → Flow Matching (36M, Metal/ANE) → Acoustic Latents
     → iSTFT Decoder (5M, vDSP/AMX)  → Waveform (24kHz)
```

| Component                       | Params     | Hardware  | Notes                         |
| ------------------------------- | ---------- | --------- | ----------------------------- |
| Sonata Codec (encode/decode)    | 16.8M      | AMX/vDSP  | 5,373x realtime (iSTFT stage) |
| Sonata LM (text→semantic)       | 241.7M     | Metal GPU | 43 tok/s (target: 50 Hz)      |
| Sonata Flow (semantic→acoustic) | 35.7M      | Metal/ANE | Parallel (non-autoregressive) |
| **Total**                       | **294.2M** |           |                               |

**Design innovations:**

- **Single LM pass per frame** — no sequential codebook prediction (unlike Mimi's DepFormer)
- **FSQ quantization** — 4096-entry codebook with zero codebook collapse
- **Conditional Flow Matching** — continuous acoustic latents, zero quantization error
- **iSTFT decoder** — 100x faster than ConvTranspose, AMX-native

## Performance (Measured)

| Metric                           | Value                     | Engine             | Source                               |
| -------------------------------- | ------------------------- | ------------------ | ------------------------------------ |
| STT WER (LibriSpeech test-clean) | 0.9%                      | Conformer CTC 0.6B | bench_output/TTS_BENCHMARK_REPORT.md |
| STT Real-Time Factor             | 0.075x (13x realtime)     | Conformer CTC 0.6B | bench_output/TTS_BENCHMARK_REPORT.md |
| TTS Intelligibility              | 100% (Whisper round-trip) | Piper VITS         | bench_output/TTS_BENCHMARK_REPORT.md |
| TTS Real-Time Factor             | 0.015x (67x realtime)     | Piper VITS         | bench_output/TTS_BENCHMARK_REPORT.md |
| TTS→STT Round-Trip WER           | 1.62%                     | Piper → Conformer  | bench_output/BENCHMARK_REPORT.md     |
| Full Round-Trip Latency          | ~320ms                    | Piper + Conformer  | bench_output/TTS_BENCHMARK_REPORT.md |
| EOU Turn Detection               | <240ms                    | Fused 3-signal     | README (design target)               |

> **Note:** Latency numbers are for audio processing stages. End-to-end latency including LLM response time depends on backend (Claude API ~300ms TTFT, on-device Llama ~100ms TTFT).

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              sonata                                       │
│                                                                           │
│  ┌──────────────┐   ┌────────────┐   ┌────────────┐   ┌───────────────┐ │
│  │ CoreAudio     │   │ STT Engine │   │ LLM Engine │   │ TTS Engine    │ │
│  │ VoiceProc IO  │──▶│ Rust/C/ANE│──▶│Claude/Gemini──▶│ Piper/Sonata │ │
│  │ (48kHz, AEC)  │   │ fp32/16/8 │   │  /Local    │   │              │ │
│  └──────┬────────┘   └─────┬──────┘   └────────────┘   └──────┬────────┘ │
│         │                  │                                    │         │
│         │           ┌──────▼─────────────────────────┐         │         │
│         │           │  Speech Detector (unified)      │         │         │
│         │           │  Native VAD + LSTM + ASR Token   │         │         │
│         │           │  → Speculative Prefill (70%)     │         │         │
│         │           └────────────────────────────────┘         │         │
│         │                                                       │         │
│         │           Audio Post-Processing (AMX/vDSP)            │         │
│         │    ┌──────────────────────────────────────────┐       │         │
│         │    │ vDSP Prosody (pitch/volume/EQ/limiter)   │       │         │
│         │    │ Noise Gate (spectral, vDSP FFT)          │◀──────┘         │
│         │    │ LUFS Loudness Normalization (BS.1770)     │                 │
│         │    │ AudioConverter HW Resampler (24→48kHz)   │                 │
│         │    │ Breath Synthesis (Voss-McCartney pink)    │                 │
│         │    │ Spatial Audio HRTF (optional 3D)          │                 │
│         ◀────│ SPMC Ring → Speaker + Opus Encoder        │                 │
│              └──────────────────────────────────────────┘                 │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │  Conversation Intelligence                                         │   │
│  │  Backchannel · Emotion · Diarizer · Memory · Prosody Prediction   │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │  Network APIs: HTTP REST · WebSocket · Web Remote                  │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────┘
```

### Pipeline State Machine

```
Listening → Recording → Processing → Streaming → Speaking → Listening
                 │           │            ↑
                 │           │  (speculative prefill at 70% EOU)
                 │           │
                 └───────────┘  Barge-in (any state) ──→ Listening
```

| State          | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| **Listening**  | Energy VAD monitors mic for speech onset                                    |
| **Recording**  | Captures audio, feeds STT frame-by-frame, runs fused EOU detection          |
| **Processing** | Sends transcript to LLM (or skips if speculative prefill already in-flight) |
| **Streaming**  | Receives LLM tokens via SSE, feeds sentence buffer → TTS incrementally      |
| **Speaking**   | Drains remaining TTS audio to speaker                                       |
| **Barge-in**   | User speaks during playback → immediate interrupt, back to Listening        |

## Native Libraries (43 C shared libraries + 5 Rust cdylibs + 1 metallib)

### Core Audio Engine

| Library          | Purpose                                                                 | Hardware              |
| ---------------- | ----------------------------------------------------------------------- | --------------------- |
| `pocket_voice.c` | CoreAudio VoiceProcessingIO, lock-free SPSC rings, energy VAD, barge-in | CoreAudio RT thread   |
| `neon_audio.h`   | ARM NEON SIMD: float32↔int16 PCM, vectorized copy, crossfade            | NEON (8 floats/cycle) |

### Speech Recognition

| Library                | Purpose                                                                                      | Hardware            |
| ---------------------- | -------------------------------------------------------------------------------------------- | ------------------- |
| `pocket_stt` (Rust)    | Kyutai STT 1B inference                                                                      | candle + Metal GPU  |
| `conformer_stt.c`      | Pure C FastConformer CTC engine, `.cstt` format (fp32/fp16/int8), EOU, cache-aware streaming | AMX + NEON          |
| `sonata_stt.c`         | CTC streaming ASR with RoPE conformer, beam search, inline EOU detection                     | AMX (cblas_sgemm)   |
| `sonata_refiner.c`     | Semantic token → text encoder-decoder transformer (GQA, RoPE, RMSNorm)                       | AMX (cblas_sgemm)   |
| `bnns_conformer.c`     | BNNS Graph accelerated Conformer encoder for ANE dispatch (macOS 15+)                        | Apple Neural Engine |
| `mel_spectrogram.c`    | Streaming 80-bin log-mel extraction using vDSP FFT                                           | AMX (vDSP_fft_zrip) |
| `ctc_beam_decoder.cpp` | CTC prefix beam search with optional KenLM n-gram rescoring                                  | CPU                 |
| `tdt_decoder.c`        | Token Duration Transducer decoder (LSTM prediction + joint network)                          | AMX (Accelerate)    |
| `spm_tokenizer.c`      | SentencePiece unigram tokenizer (pure C, Viterbi decode)                                     | CPU                 |

### Speech Synthesis

| Library                   | Purpose                                                                              | Hardware            |
| ------------------------- | ------------------------------------------------------------------------------------ | ------------------- |
| `sonata_lm` (Rust)        | 241M semantic language model (text → semantic tokens at 50 Hz)                       | candle + Metal GPU  |
| `sonata_flow` (Rust)      | 35.7M conditional flow matching (semantic → acoustic latents)                        | candle + Metal GPU  |
| `sonata_storm` (Rust)     | Parallel TTS — non-autoregressive batch synthesis                                    | candle + Metal GPU  |
| `sonata_istft.c`          | iSTFT decoder: magnitude+phase → waveform via vDSP (~100x faster than ConvTranspose) | AMX (vDSP)          |
| `bnns_convnext_decoder.c` | ANE-accelerated ConvNeXt decoder (frees GPU for flow network)                        | Apple Neural Engine |
| `phonemizer.c`            | espeak-ng IPA phonemizer for TTS text preprocessing                                  | libespeak-ng        |

### LLM Backends

| Library              | Purpose                                                      | Hardware           |
| -------------------- | ------------------------------------------------------------ | ------------------ |
| Claude SSE (libcurl) | Anthropic Claude API with streaming SSE                      | Network            |
| Gemini SSE (libcurl) | Google Gemini API with streaming SSE                         | Network            |
| `pocket_llm` (Rust)  | On-device Llama 3.2 (1B-3B) with top-p sampling and KV cache | candle + Metal GPU |

### Voice Activity & End-of-Utterance Detection

| Library             | Purpose                                                                                       | Hardware          |
| ------------------- | --------------------------------------------------------------------------------------------- | ----------------- |
| `native_vad.c`      | Pure C VAD: STFT → Conv → LSTM (weights from Silero VAD ONNX)                                 | AMX (Accelerate)  |
| `speech_detector.c` | Unified VAD+EOU wrapper: manages resampling, chunking, and signal fusion                      | CPU               |
| `mimi_endpointer.c` | LSTM-based endpointer on mel-energy features from capture audio                               | AMX (cblas_sgemv) |
| `fused_eou.c`       | 3-signal weighted fusion: energy + LSTM + ASR token, EMA smoothing, speculative prefill logic | CPU               |

### Audio Post-Processing

| Library              | Purpose                                                                          | Hardware             |
| -------------------- | -------------------------------------------------------------------------------- | -------------------- |
| `vdsp_prosody.c`     | Phase vocoder pitch shift, WSOLA time stretch, biquad EQ, soft-knee limiter      | AMX (vDSP, vForce)   |
| `audio_converter.c`  | Apple AudioConverter HW sample rate conversion (24↔48kHz)                        | Apple AudioConverter |
| `spatial_audio.c`    | HRTF binaural 3D audio positioning (azimuth/elevation)                           | vDSP convolution     |
| `breath_synthesis.c` | Voss-McCartney pink noise with Butterworth bandpass, ADSR envelopes              | Accelerate + NEON    |
| `lufs.c`             | ITU-R BS.1770 loudness meter and normalization with K-weighting                  | vDSP biquad          |
| `noise_gate.c`       | Spectral noise gate for STT preprocessing (adaptive noise floor, per-bin gating) | AMX (vDSP FFT)       |
| `opus_codec.c`       | Real-time Opus encoding/decoding                                                 | libopus              |

### Prosody & Expression

| Library              | Purpose                                                                               | Hardware |
| -------------------- | ------------------------------------------------------------------------------------- | -------- |
| `prosody_predict.c`  | Text-based prosody prediction (syllable duration, emotion, conversational adaptation) | CPU      |
| `prosody_log.c`      | JSONL prosody logging for visualization dashboard                                     | CPU      |
| `emphasis_predict.c` | Linguistics-based emphasis prediction (contrast, intensifiers, negation, enumeration) | CPU      |

### Text Processing

| Library             | Purpose                                                                              |
| ------------------- | ------------------------------------------------------------------------------------ |
| `text_normalize.c`  | Number, date, currency, phone number expansion for STT/TTS                           |
| `ssml_parser.c`     | SSML parsing: `<prosody>`, `<break>`, `<say-as>`, `<emphasis>`                       |
| `sentence_buffer.c` | Streaming LLM token accumulation, sentence boundary detection, predictive length EMA |

### Conversation Intelligence

| Library                 | Purpose                                                                    | Hardware     |
| ----------------------- | -------------------------------------------------------------------------- | ------------ |
| `audio_emotion.c`       | Real-time emotion detection from mel-energy features (valence, arousal)    | AMX (vDSP)   |
| `speaker_encoder.c`     | ONNX-based speaker embedding extraction (ECAPA-TDNN, WavLM)                | ONNX Runtime |
| `speaker_diarizer.c`    | Speaker diarization via cosine similarity on running centroid embeddings   | Accelerate   |
| `conversation_memory.c` | Conversation context persistence (JSONL history, token-aware truncation)   | CPU          |
| `backchannel.c`         | Active listening backchannel generation ("mhm", "yeah") from acoustic cues | Accelerate   |
| `voice_onboard.c`       | Real-time voice onboarding: speaker embedding + prosody profile extraction | Accelerate   |

### Network & API

| Library        | Purpose                                                                    | Hardware |
| -------------- | -------------------------------------------------------------------------- | -------- |
| `http_api.c`   | REST API server with TTS endpoints (PCM, WAV, mu-law, A-law encoding)      | CPU      |
| `websocket.c`  | RFC 6455 WebSocket implementation (text, binary, ping/pong, close)         | CPU      |
| `web_remote.c` | WebSocket audio server: phone browser mic → pipeline → audio back to phone | CPU      |

### Quality & Profiling

| Library              | Purpose                                                                                      | Hardware               |
| -------------------- | -------------------------------------------------------------------------------------------- | ---------------------- |
| `voice_quality.c`    | PESQ-lite, STOI-lite, Log-Spectral Distance, MOS prediction                                  | Accelerate (vDSP FFT)  |
| `latency_profiler.c` | Per-stage nanosecond latency with P50/P95/P99 stats                                          | mach_absolute_time     |
| `apple_perf.c`       | Apple Silicon perf: RT thread scheduling, huge pages, IOSurface zero-copy, NEON softmax/GELU | IOSurface + Foundation |
| `metal_loader.c`     | Runtime .metallib loader for custom GPU kernels                                              | Metal (Objective-C)    |

### Infrastructure

| Library           | Purpose                                                                         |
| ----------------- | ------------------------------------------------------------------------------- |
| `vm_ring.c`       | VM-mirrored ring buffer via `mach_vm_remap` for zero-copy wraparound            |
| `spmc_ring.h`     | Single-producer multi-consumer lock-free ring (speaker + Opus encoder)          |
| `kv_cache.h`      | Cache-oblivious interleaved KV cache `[H][T][2][D]` — halves L2 misses          |
| `triple_buffer.h` | Lock-free triple buffer for GPU→CPU→CoreAudio                                   |
| `arena.h`         | Bump-pointer arena allocator, checkpoint/restore, zero per-turn malloc overhead |
| `lstm_ops.h`      | LSTM cell operations for native VAD and endpointer                              |

## Optimizations

### Hardware Utilization

Sonata runs on **all four Apple Silicon compute units simultaneously**:

| Unit                | What Runs                                                             | Why                                |
| ------------------- | --------------------------------------------------------------------- | ---------------------------------- |
| **Metal GPU**       | Sonata LM + Flow + Storm inference, custom kernels (.metallib)        | Massively parallel transformer ops |
| **AMX Coprocessor** | Prosody, FFT, LUFS, LSTM endpointer, Conformer STT (Accelerate)       | Matrix-vector products, vDSP       |
| **Neural Engine**   | BNNS Graph Conformer encoder + ConvNeXt decoder (macOS 15+)           | Power-efficient inference          |
| **ARM NEON**        | PCM conversion, INT8 dequantize, crossfade, ring copies, softmax/GELU | 8-wide SIMD on integer/float       |

### Key Performance Techniques

- **Lock-free everywhere**: SPSC and SPMC ring buffers with 128-byte cache-line aligned atomics. Zero locks in the audio hot path.
- **Zero allocation in hot loops**: Arena allocator provides per-turn memory. CoreAudio callback does zero malloc/free.
- **INT8 quantization**: Per-channel symmetric quantization with NEON-vectorized dequantize → cblas_sgemm. ~4x memory reduction.
- **Fused operations**: vDSP pitch→EQ→volume→limit chain. Interleaved KV cache eliminates separate K/V fetches.
- **VM-mirrored ring buffers**: `mach_vm_remap` creates a virtual memory mirror — reads/writes never need to handle wraparound.
- **Hardware resampling**: Apple AudioConverter for sample rate conversion is higher quality and faster than FIR for non-integer ratios.
- **Speculative prefill**: LLM API request starts at 70% EOU confidence, saving 100-300ms when the prediction is correct.
- **Conformer cache-aware streaming**: Per-layer K/V projection and convolution state caching for true frame-by-frame inference without recomputation.
- **Custom Metal kernels**: Flash Attention v2, fused SiLU+gate, layer norm — loaded at runtime from compiled .metallib.
- **Concurrent compute unit dispatch**: GPU runs Sonata LM while ANE runs ConvNeXt decoder while AMX runs iSTFT — all three stages overlap.
- **Apple perf primitives**: RT thread scheduling, huge page model loading, IOSurface zero-copy GPU↔CPU sharing.

## Quality Assurance Framework

Sonata includes a comprehensive native C quality benchmark suite for proving STT and TTS quality.

### Metrics Implemented

| Category            | Metric                                      | Golden Signal      |
| ------------------- | ------------------------------------------- | ------------------ |
| **Intelligibility** | WER (Word Error Rate)                       | < 5% (human-level) |
| **Intelligibility** | CER (Character Error Rate)                  | < 2%               |
| **Intelligibility** | STOI (Short-Time Objective Intelligibility) | > 0.9              |
| **Naturalness**     | MCD (Mel-Cepstral Distortion)               | < 4.0 dB           |
| **Naturalness**     | F0 RMSE + Correlation                       | < 15 Hz, r > 0.85  |
| **Voice Quality**   | Segmental SNR                               | > 25 dB            |
| **Voice Quality**   | Speaker Similarity                          | > 0.90             |
| **Latency**         | RTF (Real-Time Factor)                      | < 0.2x             |
| **Latency**         | First Chunk Latency                         | < 200ms            |

### Round-Trip Testing

The ultimate quality proof: `text → TTS → audio → STT → transcript → WER(original, transcript)`. If the round-trip WER is low, both TTS and STT are working correctly.

```bash
make bench-quality    # Run benchmark self-tests
make test-quality     # Run all quality metric tests
make test-roundtrip   # Round-trip with mock TTS/STT callbacks
```

## Test Suite

```bash
make test    # Run all 30 test suites
```

| Suite                      | What It Covers                                                                |
| -------------------------- | ----------------------------------------------------------------------------- |
| `bench-quality`            | Self-tests for all quality metric implementations                             |
| `test-quality`             | WER, CER, MCD, STOI, SNR, F0, speaker similarity, latency harness, grading    |
| `test-eou`                 | Mimi endpointer LSTM, fused EOU, speculative prefill, Conformer EOU flags/API |
| `test-roundtrip`           | Round-trip framework with mock TTS/STT callbacks, NULL handling               |
| `test-pipeline`            | Text normalization, sentence buffer, SSML parser                              |
| `test-new-modules`         | Breath synthesis, LUFS, arena, VM ring, triple buffer, SPMC ring, KV cache    |
| `test-new-engines`         | Phonemizer and speaker encoder API tests                                      |
| `test-bugfixes`            | SPMC mirror copy, KV cache overflow, arena accounting, LUFS non-48kHz         |
| `test-conformer`           | Mel spectrogram extraction, Conformer STT API                                 |
| `test-llm-prosody`         | LLM prosody integration with SSML and text normalization                      |
| `test-optimizations`       | INT8 round-trip, BNNS API, FP16 NEON, latency profiler, voice quality metrics |
| `test-beam-search`         | CTC beam decoder with and without KenLM                                       |
| `test-sonata`              | Sonata iSTFT, SentencePiece tokenizer, ConvNeXt decoder, LM + Flow FFI        |
| `test-sonata-v3`           | Sonata Flow v3 API tests                                                      |
| `test-real-models`         | Phonemizer and speaker encoder with real model files                          |
| `test-prosody-predict`     | Text-based prosody prediction (syllables, emotion, adaptation)                |
| `test-prosody-log`         | JSONL prosody logging and replay                                              |
| `test-emphasis`            | Linguistics-based emphasis prediction rules                                   |
| `test-prosody-integration` | End-to-end prosody pipeline (emphasis + SSML + prediction)                    |
| `test-voice-onboard`       | Voice onboarding: embedding extraction + prosody profiling                    |
| `test-conversation-memory` | Conversation context persistence and token-aware truncation                   |
| `test-diarizer`            | Speaker diarization with cosine similarity tracking                           |
| `test-vdsp-prosody`        | vDSP prosody processing: pitch shift, time stretch, EQ, limiter               |
| `test-http-api`            | REST API server endpoints and WebSocket integration                           |
| `test-sonata-storm`        | Sonata Storm parallel TTS FFI                                                 |
| `test-audio-emotion`       | Audio emotion detection (valence, arousal)                                    |
| `test-sonata-flow-ffi`     | Sonata Flow Rust FFI boundary tests                                           |
| `test-sonata-lm-ffi`       | Sonata LM Rust FFI boundary tests                                             |
| `test-pipeline-threading`  | Pipeline threading, lock-free ring buffer concurrency                         |
| `test-phase2-regressions`  | Phase 2 regression tests across breath, mel, sentence, conformer modules      |

Additional targets (not in `make test`):

```bash
make test-sonata-quality     # End-to-end TTS→STT quality with Sonata
make test-sonata-stt         # Sonata STT + refiner API tests
make test-native-vad         # Pure C VAD unit tests
make test-speech-detector    # Unified speech detector tests
make test-apple-perf         # Apple Silicon perf primitives
make test-quality-improvements  # Noise gate + LUFS + voice quality
make bench-vad               # VAD throughput benchmark
make bench-sonata            # Sonata TTS throughput benchmark
make bench-live              # Live model benchmark with real audio
make bench-industry          # Industry-standard quality benchmarks
```

## Additional Features

- **INT8 quantization**: Per-channel symmetric INT8 weights with NEON-vectorized dequantization. ~4x smaller models with <1% accuracy loss.
- **Apple Neural Engine**: BNNS Graph-accelerated Conformer encoder and ConvNeXt decoder dispatch to the ANE on macOS 15+ for power-efficient inference.
- **Custom Metal kernels**: Flash Attention v2, fused SiLU+gate, layer norm, and fp16 GEMM compiled as .metallib and loaded at runtime.
- **Per-turn latency profiling**: `--profiler` flag enables nanosecond-precision breakdown of every pipeline stage (STT, LLM TTFT, TTS TTFS, E2E) with P50/P95/P99 statistics.
- **Voice quality metrics**: PESQ-lite, STOI-lite, Log-Spectral Distance, and MOS prediction for automated TTS quality evaluation.
- **Speaker diarization**: ONNX-based speaker embedding extraction with cosine similarity tracking for multi-speaker conversations.
- **Active listening**: Backchannel generation ("mhm", "yeah") from acoustic cues at ~50ms latency — no LLM round-trip required.
- **Web remote**: Phone browser captures mic via Web Audio API → streams PCM over WebSocket → pipeline processes and streams audio back.
- **REST API**: HTTP server with WebSocket streaming for integration with external applications.
- **Multi-LLM backend**: Supports Claude (SSE), Gemini, and on-device Llama 3.2 (1B-3B). Switch with `--llm claude|gemini|local`.
- **Multi-STT backend**: Kyutai Rust (1B), pure C Conformer CTC (fp32/fp16/int8), Sonata CTC (RoPE conformer), and BNNS/ANE-accelerated Conformer. Switch with `--stt-engine rust|conformer|bnns`.

## Requirements

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)
- Rust (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Homebrew dependencies: `brew install curl opus onnxruntime espeak-ng`
- Anthropic API key (`ANTHROPIC_API_KEY`) for Claude backend (or use `--llm local` for on-device)

## Build

```bash
make          # Build everything: 43 C shared libraries + 5 Rust cdylibs + 1 metallib + binary
make libs     # Build just the C shared libraries + metallib
make clean    # Remove all build artifacts
```

Build output:

1. `build/*.dylib` — 43 C shared libraries
2. `build/tensor_ops.metallib` — Custom Metal kernels (Flash Attention v2, fused SiLU+gate, layer norm)
3. `src/stt/target/release/libpocket_stt.dylib` — Rust STT cdylib (Kyutai 1B, candle + Metal)
4. `src/llm/target/release/libpocket_llm.dylib` — Rust local LLM cdylib (Llama 3.2, candle + Metal)
5. `src/sonata_lm/target/release/libsonata_lm.dylib` — Rust Sonata LM cdylib (241M semantic model)
6. `src/sonata_flow/target/release/libsonata_flow.dylib` — Rust Sonata Flow cdylib (35.7M flow matching)
7. `src/sonata_storm/target/release/libsonata_storm.dylib` — Rust Sonata Storm cdylib (parallel TTS)
8. `sonata` — Pipeline binary linking everything

## Run

```bash
ANTHROPIC_API_KEY=sk-ant-... ./sonata
```

Speak naturally. The pipeline detects speech onset, transcribes in real-time, sends to the LLM, and streams TTS audio back — all with full-duplex barge-in support.

### Options

```
sonata [OPTIONS]

Voice & Model:
  --voice PATH       Voice .wav or .safetensors path
  --stt-repo REPO    STT HuggingFace repo (default: kyutai/stt-1b-en_fr-candle)
  --tts-repo REPO    TTS HuggingFace repo (default: kyutai/tts-1.6b-en_fr)
  --n-q N            Audio codebooks for TTS (default: 24)

STT Engine:
  --stt-engine E     STT engine: rust (default), conformer, or bnns
  --cstt-model PATH  Conformer STT .cstt model file path
  --bnns-model PATH  BNNS .mlmodelc path (ANE accelerated)

TTS Engine:
  --tts-engine E     TTS engine: rust (default), c (Kyutai C), or pocket (100M)

LLM Backend:
  --llm ENGINE       LLM backend: claude (default), gemini, or local
  --llm-model M      LLM model name (auto-detected per engine)
  --system PROMPT    System prompt

VAD:
  --no-vad           Disable semantic VAD (use energy VAD only)
  --vad-threshold F  Semantic VAD threshold (default: 0.7)

Audio Post-Processing:
  --pitch F          Pitch multiplier (1.0 = normal, 1.2 = higher)
  --volume F         Volume in dB (0.0 = normal, 6.0 = louder)
  --no-hw-resample   Disable AudioConverter (use FIR fallback)
  --spatial AZ       Enable 3D spatial audio at azimuth AZ degrees

Advanced:
  --metallib PATH    Custom .metallib path for GPU kernels
  --profiler         Enable per-turn latency profiling
  --prosody          Enable SSML-aware system prompt
```

### Examples

```bash
# Default — conversational voice assistant
ANTHROPIC_API_KEY=sk-... ./sonata

# Custom voice with pitched-up output
./sonata --voice /path/to/voice.wav --pitch 1.15

# Spatial audio — voice positioned 30° to the right
./sonata --spatial 30

# On-device LLM (no API key needed)
./sonata --llm local --llm-model meta-llama/Llama-3.2-1B-Instruct

# Conformer STT with INT8 model
./sonata --stt-engine conformer --cstt-model models/parakeet_0.6b_int8.cstt

# BNNS/ANE accelerated STT (macOS 15+)
./sonata --stt-engine bnns --cstt-model models/parakeet_0.6b.cstt \
  --bnns-model models/conformer_ctc_0.6b.mlmodelc

# Latency profiling
./sonata --profiler

# Custom Metal kernels
./sonata --metallib build/tensor_ops.metallib

# Custom Claude model with system prompt
./sonata --llm-model claude-sonnet-4-20250514 \
  --system "You are a helpful coding assistant."
```

## Project Structure

```
sonata/
├── Makefile              # Build system (C libs + Rust + binary + tests)
├── README.md             # This file
├── CONTRIBUTING.md       # Contributing guide
├── AGENTS.md             # AI agent guidance
├── LICENSE               # MIT
├── src/                  # 43 C source files + 5 Rust crates + Metal kernels
├── include/              # C header files
├── tests/                # 30+ test suites
├── scripts/              # Benchmarking, conversion, and export scripts
├── web/                  # Web dashboard and remote UI
├── docs/                 # API reference, architecture, troubleshooting
├── examples/             # curl, WebSocket, and integration examples
├── models/               # Downloaded/compiled models (gitignored)
└── build/                # Compiled output (gitignored)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and pull request process.

## Companion Project

**[pocket-tts](https://github.com/sethdford/pocket-tts)** — Python/MLX text-to-speech library with custom Metal kernels, Apple AMX/Accelerate DSP, speculative decoding, and OpenAI-compatible API. Sonata uses Rust inference crates (candle + Metal) instead of MLX, but shares the same philosophy of deep Apple Silicon optimization.

## License

MIT — Copyright (c) 2026 Seth Ford
