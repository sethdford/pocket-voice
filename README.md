# Sonata

Real-time voice intelligence for Apple Silicon. Fully native C and Rust — zero Python, zero compromise.

```
Mic → STT → LLM → TTS → Speaker
```

Sonata is a conversational AI voice pipeline that runs entirely in C and Rust on Apple Silicon. It captures audio from the microphone, transcribes speech in real-time, sends the transcript to an LLM (Claude, Gemini, or on-device Llama), synthesizes the response with state-of-the-art TTS, and plays it back through the speaker — all with sub-200ms latency and full-duplex barge-in support.

## Sonata TTS Architecture

Sonata includes a novel from-scratch TTS system designed for phenomenal quality AND speed:

```
Text → Semantic LM (241M, Metal GPU) → Semantic Tokens (50 Hz)
     → Flow Matching (36M, Metal/ANE) → Acoustic Latents
     → iSTFT Decoder (5M, vDSP/AMX)  → Waveform (24kHz)
```

| Component                       | Params     | Hardware  | Speed             |
| ------------------------------- | ---------- | --------- | ----------------- |
| Sonata Codec (encode/decode)    | 16.8M      | AMX/vDSP  | 5,373x realtime   |
| Sonata LM (text→semantic)       | 241.7M     | Metal GPU | Streaming 50 Hz   |
| Sonata Flow (semantic→acoustic) | 35.7M      | Metal/ANE | Parallel (non-AR) |
| **Total**                       | **294.2M** |           |                   |

**Key innovations:**

- **Single LM pass per frame** — no sequential codebook prediction (unlike Mimi's DepFormer)
- **FSQ quantization** — 4096-entry codebook with zero codebook collapse
- **Conditional Flow Matching** — continuous acoustic latents, zero quantization error
- **iSTFT decoder** — 100x faster than ConvTranspose, AMX-native

## Key Features

- **100% native**: C + Rust. No Python, no Node, no JVM. Single `make` builds everything.
- **Full-duplex audio**: CoreAudio VoiceProcessingIO with hardware AEC (Acoustic Echo Cancellation). Speak while the assistant is talking — barge-in immediately interrupts playback.
- **Fused 3-signal end-of-utterance**: Energy VAD + LSTM endpointer + ASR-inline EOU token detection, weighted and fused for <240ms turn detection.
- **Speculative prefill**: Sends the LLM API request at 70% EOU confidence. If the user resumes speaking, it cancels. If they stop, response starts streaming before the final silence timeout — shaving 100-300ms off perceived latency.
- **Multi-LLM backend**: Supports Claude (SSE), Gemini, and on-device Llama 3.2 (1B-3B) via a pluggable LLM interface. Switch with `--llm claude|gemini|local`.
- **Multi-STT backend**: Kyutai Rust (1B), pure C Conformer CTC (fp32/fp16/int8), Sonata CTC (RoPE conformer), and BNNS/ANE-accelerated Conformer. Switch with `--stt-engine rust|conformer|bnns`.
- **INT8 quantization**: Per-channel symmetric INT8 weights with NEON-vectorized dequantization. ~4x smaller models with <1% accuracy loss.
- **Apple Neural Engine**: BNNS Graph-accelerated Conformer encoder and ConvNeXt decoder dispatch to the ANE on macOS 15+ for power-efficient inference.
- **Custom Metal kernels**: Flash Attention v2, fused SiLU+gate, layer norm, and fp16 GEMM compiled as .metallib and loaded at runtime.
- **Per-turn latency profiling**: `--profiler` flag enables nanosecond-precision breakdown of every pipeline stage (STT, LLM TTFT, TTS TTFS, E2E) with P50/P95/P99 statistics.
- **Voice quality metrics**: PESQ-lite, STOI-lite, Log-Spectral Distance, and MOS prediction for automated TTS quality evaluation.
- **Speaker diarization**: ONNX-based speaker embedding extraction with cosine similarity tracking for multi-speaker conversations.
- **Active listening**: Backchannel generation ("mhm", "yeah") from acoustic cues at ~50ms latency — no LLM round-trip required.
- **Web remote**: Phone browser captures mic via Web Audio API → streams PCM over WebSocket → pipeline processes and streams audio back.
- **REST API**: HTTP server with WebSocket streaming for integration with external applications.
- **Apple Silicon optimized**: Every audio processing stage runs on the AMX coprocessor (via Apple Accelerate), while ML inference runs on the Metal GPU. ARM NEON SIMD for PCM conversion. All four hardware units (GPU + AMX + ANE + NEON) run concurrently without contention.
- **30 automated test suites**: Comprehensive coverage across quality metrics, EOU detection, STT, TTS, prosody, diarization, threading, and regression tests.

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          pocket-voice binary                              │
│                                                                           │
│  ┌──────────────┐   ┌────────────┐   ┌────────────┐   ┌───────────────┐ │
│  │ CoreAudio     │   │ STT Engine │   │ LLM Engine │   │ TTS Engine    │ │
│  │ VoiceProc IO  │──▶│ Rust/C/ANE│──▶│Claude/Gemini──▶│ Sonata/Storm │ │
│  │ (48kHz, AEC)  │   │ fp32/16/8 │   │  /Local    │   │ 294M params  │ │
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

| State          | Description                                                                        |
| -------------- | ---------------------------------------------------------------------------------- |
| **Listening**  | Energy VAD monitors mic for speech onset                                           |
| **Recording**  | Captures audio, feeds STT frame-by-frame, runs fused EOU detection                 |
| **Processing** | Sends transcript to Claude API (or skips if speculative prefill already in-flight) |
| **Streaming**  | Receives Claude tokens via SSE, feeds sentence buffer → TTS incrementally          |
| **Speaking**   | Drains remaining TTS audio to speaker                                              |
| **Barge-in**   | User speaks during playback → immediate interrupt, back to Listening               |

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

### Speech Synthesis (Sonata TTS)

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

pocket-voice runs on **all four Apple Silicon compute units simultaneously**:

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

pocket-voice includes a comprehensive native C quality benchmark suite for proving STT and TTS quality.

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

### Latency Measurement

Uses `mach_absolute_time` for nanosecond-precision hardware timestamps. Tracks P50/P95/P99 percentiles via quickselect.

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

## Requirements

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)
- Rust (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Homebrew dependencies: `brew install curl opus onnxruntime espeak-ng`
- Anthropic API key (`ANTHROPIC_API_KEY`) for Claude backend

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
8. `pocket-voice` — Pipeline binary linking everything

## Run

```bash
ANTHROPIC_API_KEY=sk-ant-... ./pocket-voice
```

Speak naturally. The pipeline detects speech onset, transcribes in real-time, sends to Claude, and streams TTS audio back — all with full-duplex barge-in support.

### Options

```
pocket-voice [OPTIONS]

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
ANTHROPIC_API_KEY=sk-... ./pocket-voice

# Custom voice with pitched-up output
./pocket-voice --voice /path/to/voice.wav --pitch 1.15

# Spatial audio — voice positioned 30° to the right
./pocket-voice --spatial 30

# On-device LLM (no API key needed)
./pocket-voice --llm local --llm-model meta-llama/Llama-3.2-1B-Instruct

# Conformer STT with INT8 model
./pocket-voice --stt-engine conformer --cstt-model models/parakeet_0.6b_int8.cstt

# BNNS/ANE accelerated STT (macOS 15+)
./pocket-voice --stt-engine bnns --cstt-model models/parakeet_0.6b.cstt \
  --bnns-model models/conformer_ctc_0.6b.mlmodelc

# Latency profiling
./pocket-voice --profiler

# Custom Metal kernels
./pocket-voice --metallib build/tensor_ops.metallib

# Custom Claude model with system prompt
./pocket-voice --llm-model claude-sonnet-4-20250514 \
  --system "You are a helpful coding assistant."
```

## Project Structure

```
pocket-voice/
├── Makefile                        # Build system (C libs + Rust + binary + tests)
├── README.md                       # This file
├── AGENTS.md                       # AI agent guidance
├── LICENSE                         # MIT
├── src/
│   ├── pocket_voice_pipeline.c     # Main orchestrator: CLI, state machine, LLM SSE
│   ├── pocket_voice.c              # CoreAudio VoiceProcessingIO engine
│   │
│   │── # ─── Speech Recognition ───
│   ├── conformer_stt.c             # Pure C Conformer CTC STT (fp32/fp16/int8)
│   ├── sonata_stt.c                # CTC streaming ASR with RoPE conformer
│   ├── sonata_refiner.c            # Semantic → text encoder-decoder transformer
│   ├── bnns_conformer.c            # BNNS Graph Conformer for ANE (macOS 15+)
│   ├── mel_spectrogram.c           # Streaming log-mel spectrogram (vDSP FFT)
│   ├── ctc_beam_decoder.cpp        # CTC beam search with optional KenLM
│   ├── tdt_decoder.c               # Token Duration Transducer decoder
│   ├── spm_tokenizer.c             # SentencePiece unigram tokenizer (pure C)
│   │
│   │── # ─── Speech Synthesis ───
│   ├── sonata_istft.c              # iSTFT decoder (mag+phase → waveform, vDSP)
│   ├── bnns_convnext_decoder.c     # ANE-accelerated ConvNeXt decoder
│   ├── phonemizer.c                # espeak-ng IPA phonemizer
│   │
│   │── # ─── VAD & EOU Detection ───
│   ├── native_vad.c                # Pure C VAD (STFT+Conv+LSTM from Silero)
│   ├── speech_detector.c           # Unified VAD+EOU wrapper
│   ├── mimi_endpointer.c           # LSTM end-of-utterance on mel features
│   ├── fused_eou.c                 # 3-signal fused EOU detector
│   │
│   │── # ─── Audio Post-Processing ───
│   ├── vdsp_prosody.c              # AMX prosody: pitch, volume, EQ, limiter
│   ├── audio_converter.c           # Apple AudioConverter resampling
│   ├── spatial_audio.c             # HRTF binaural 3D audio
│   ├── breath_synthesis.c          # Pink noise breath synthesis (ADSR)
│   ├── lufs.c                      # ITU-R BS.1770 LUFS loudness normalization
│   ├── noise_gate.c                # Spectral noise gate (vDSP FFT)
│   ├── opus_codec.c                # Real-time Opus encode/decode
│   │
│   │── # ─── Prosody & Expression ───
│   ├── prosody_predict.c           # Text-based prosody prediction
│   ├── prosody_log.c               # JSONL prosody logging for dashboard
│   ├── emphasis_predict.c          # Linguistics-based emphasis prediction
│   │
│   │── # ─── Text Processing ───
│   ├── text_normalize.c            # Number/date/currency text expansion
│   ├── ssml_parser.c               # SSML parsing and prosody extraction
│   ├── sentence_buffer.c           # LLM token → sentence boundary detection
│   │
│   │── # ─── Conversation Intelligence ───
│   ├── audio_emotion.c             # Real-time emotion detection from audio
│   ├── speaker_encoder.c           # ONNX speaker embedding extraction
│   ├── speaker_diarizer.c          # Speaker diarization via embeddings
│   ├── conversation_memory.c       # Conversation context persistence
│   ├── backchannel.c               # Active listening backchannel generation
│   ├── voice_onboard.c             # Voice onboarding + prosody profiling
│   │
│   │── # ─── Network & API ───
│   ├── http_api.c                  # REST API server
│   ├── websocket.c                 # RFC 6455 WebSocket
│   ├── web_remote.c                # Web remote mic/speaker via WebSocket
│   │
│   │── # ─── Quality & Profiling ───
│   ├── voice_quality.c             # PESQ-lite, STOI-lite, LSD, MOS prediction
│   ├── latency_profiler.c          # Per-stage nanosecond latency measurement
│   ├── apple_perf.c                # Apple Silicon performance primitives
│   ├── metal_loader.c              # Runtime .metallib GPU kernel loader
│   │
│   │── # ─── Infrastructure ───
│   ├── vm_ring.c                   # VM-mirrored ring buffer (mach_vm)
│   ├── cJSON.c                     # Vendored JSON parser (MIT)
│   │
│   │── # ─── Header-Only Libraries ───
│   ├── neon_audio.h                # ARM NEON SIMD PCM conversion
│   ├── spmc_ring.h                 # Lock-free SPMC ring buffer
│   ├── kv_cache.h                  # Interleaved KV cache [H][T][2][D]
│   ├── triple_buffer.h             # Lock-free triple buffer
│   ├── arena.h                     # Bump-pointer arena allocator
│   ├── lstm_ops.h                  # LSTM cell operations
│   │
│   │── # ─── Metal GPU Kernels ───
│   ├── tensor_ops.metal            # Flash Attention v2, fused SiLU+gate, layer norm
│   │
│   │── # ─── Rust Crates ───
│   ├── stt/                        # Kyutai STT 1B (candle + Metal)
│   ├── llm/                        # On-device Llama 3.2 (candle + Metal)
│   ├── sonata_lm/                  # Sonata 241M semantic language model
│   ├── sonata_flow/                # Sonata 35.7M conditional flow matching
│   ├── sonata_storm/               # Sonata parallel (non-AR) TTS
│   └── quality/                    # Quality benchmark suite (WER, MCD, STOI, etc.)
├── scripts/
│   ├── benchmark_sweep.sh          # fp32 vs fp16 vs int8 comparison
│   ├── quantize_int8.py            # Convert fp32/fp16 .cstt to INT8
│   ├── convert_nemo_coreml.py      # NeMo → CoreML conversion for BNNS
│   └── ...                         # Model export, validation, and benchmarking scripts
├── tests/                          # 30 test suites + additional benchmarks
├── web/                            # Web dashboard and remote UI
├── models/                         # Downloaded/compiled models (gitignored)
└── build/                          # Compiled output (gitignored)
```

## Companion Project

**[pocket-tts](https://github.com/sethdford/pocket-tts)** — Python/MLX text-to-speech library with custom Metal kernels, Apple AMX/Accelerate DSP, speculative decoding, and OpenAI-compatible API. pocket-voice uses Kyutai's Rust inference crates instead of MLX, but shares the same philosophy of deep Apple Silicon optimization.

## License

MIT — Copyright (c) 2026 Seth Ford
