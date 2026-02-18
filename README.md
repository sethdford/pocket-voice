# pocket-voice

Zero-Python real-time voice pipeline for Apple Silicon. Fully native C and Rust — no interpreter, no GIL, no compromises.

```
Mic → STT → Claude → TTS → Speaker
```

pocket-voice is a conversational AI voice assistant that runs entirely in C and Rust on Apple Silicon. It captures audio from the microphone, transcribes speech in real-time, sends the transcript to Claude via streaming SSE, synthesizes the response with TTS, and plays it back through the speaker — all with sub-200ms latency and full-duplex barge-in support.

## Key Features

- **100% native**: C + Rust. No Python, no Node, no JVM. Single `make` builds everything.
- **Full-duplex audio**: CoreAudio VoiceProcessingIO with hardware AEC (Acoustic Echo Cancellation). Speak while the assistant is talking — barge-in immediately interrupts playback.
- **Fused 3-signal end-of-utterance**: Energy VAD + LSTM endpointer + ASR-inline EOU token detection, weighted and fused for <240ms turn detection.
- **Speculative prefill**: Sends the Claude API request at 70% EOU confidence. If the user resumes speaking, it cancels. If they stop, response starts streaming before the final silence timeout — shaving 100-300ms off perceived latency.
- **Apple Silicon optimized**: Every audio processing stage runs on the AMX coprocessor (via Apple Accelerate), while ML inference runs on the Metal GPU. ARM NEON SIMD for PCM conversion. All three hardware units (GPU + AMX + NEON) run concurrently without contention.
- **157 automated tests**: 8 test suites covering quality metrics, EOU detection, round-trip intelligibility, conformer STT, pipeline modules, and regression tests.

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          pocket-voice binary                              │
│                                                                           │
│  ┌──────────────┐   ┌────────────┐   ┌────────────┐   ┌───────────────┐ │
│  │ CoreAudio     │   │ Kyutai     │   │ Claude     │   │ Kyutai DSM    │ │
│  │ VoiceProc IO  │──▶│ STT 1B    │──▶│ API (SSE)  │──▶│ TTS 1.6B     │ │
│  │ (48kHz, AEC)  │   │ (Rust)    │   │ (libcurl)  │   │ (Rust)        │ │
│  └──────┬────────┘   └─────┬──────┘   └────────────┘   └──────┬────────┘ │
│         │                  │                                    │         │
│         │           ┌──────▼─────────────────────────┐         │         │
│         │           │  Fused 3-Signal EOU Detector    │         │         │
│         │           │  Energy VAD + LSTM + ASR Token   │         │         │
│         │           │  → Speculative Prefill (70%)     │         │         │
│         │           └────────────────────────────────┘         │         │
│         │                                                       │         │
│         │           Audio Post-Processing (AMX/vDSP)            │         │
│         │    ┌──────────────────────────────────────────┐       │         │
│         │    │ vDSP Prosody (pitch/volume/EQ/limiter)   │       │         │
│         │    │ LUFS Loudness Normalization (BS.1770)     │◀──────┘         │
│         │    │ AudioConverter HW Resampler (24→48kHz)   │                 │
│         │    │ Breath Synthesis (Voss-McCartney pink)    │                 │
│         │    │ Spatial Audio HRTF (optional 3D)          │                 │
│         ◀────│ SPMC Ring → Speaker + Opus Encoder        │                 │
│              └──────────────────────────────────────────┘                 │
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

## Native Libraries (18 modules)

### Core Audio Engine

| Library          | Purpose                                                                 | Hardware              |
| ---------------- | ----------------------------------------------------------------------- | --------------------- |
| `pocket_voice.c` | CoreAudio VoiceProcessingIO, lock-free SPSC rings, energy VAD, barge-in | CoreAudio RT thread   |
| `neon_audio.h`   | ARM NEON SIMD: float32↔int16 PCM, vectorized copy, crossfade            | NEON (8 floats/cycle) |

### Speech Recognition

| Library             | Purpose                                                                                           | Hardware                 |
| ------------------- | ------------------------------------------------------------------------------------------------- | ------------------------ |
| `pocket_stt` (Rust) | Kyutai STT 1B inference                                                                           | candle + Metal GPU       |
| `conformer_stt.c`   | Pure C FastConformer CTC engine, `.cstt` model format, EOU token detection, cache-aware streaming | AMX (cblas_sgemv) + NEON |
| `mel_spectrogram.c` | Streaming 80-bin log-mel extraction using vDSP FFT                                                | AMX (vDSP_fft_zrip)      |

### Speech Synthesis

| Library                | Purpose                                        | Hardware            |
| ---------------------- | ---------------------------------------------- | ------------------- |
| `pocket_tts_rs` (Rust) | Kyutai DSM TTS 1.6B inference                  | candle + Metal GPU  |
| `bnns_mimi_decoder.c`  | Mimi neural audio codec decoder via BNNS Graph | Apple Neural Engine |

### End-of-Utterance Detection

| Library             | Purpose                                                                                       | Hardware          |
| ------------------- | --------------------------------------------------------------------------------------------- | ----------------- |
| `mimi_endpointer.c` | LSTM-based endpointer on mel-energy features from capture audio                               | AMX (cblas_sgemv) |
| `fused_eou.c`       | 3-signal weighted fusion: energy + LSTM + ASR token, EMA smoothing, speculative prefill logic | CPU               |

### Audio Post-Processing

| Library              | Purpose                                                                     | Hardware             |
| -------------------- | --------------------------------------------------------------------------- | -------------------- |
| `vdsp_prosody.c`     | Phase vocoder pitch shift, WSOLA time stretch, biquad EQ, soft-knee limiter | AMX (vDSP, vForce)   |
| `audio_converter.c`  | Apple AudioConverter HW sample rate conversion (24↔48kHz)                   | Apple AudioConverter |
| `spatial_audio.c`    | HRTF binaural 3D audio positioning (azimuth/elevation)                      | vDSP convolution     |
| `breath_synthesis.c` | Voss-McCartney pink noise with Butterworth bandpass, ADSR envelopes         | Accelerate + NEON    |
| `lufs.c`             | ITU-R BS.1770 loudness meter and normalization with K-weighting             | vDSP biquad          |
| `opus_codec.c`       | Real-time Opus encoding/decoding                                            | libopus              |

### Text Processing

| Library             | Purpose                                                                              |
| ------------------- | ------------------------------------------------------------------------------------ |
| `text_normalize.c`  | Number, date, currency, phone number expansion for STT/TTS                           |
| `ssml_parser.c`     | SSML parsing: `<prosody>`, `<break>`, `<say-as>`, `<emphasis>`                       |
| `sentence_buffer.c` | Streaming LLM token accumulation, sentence boundary detection, predictive length EMA |

### Infrastructure

| Library           | Purpose                                                                         |
| ----------------- | ------------------------------------------------------------------------------- |
| `vm_ring.c`       | VM-mirrored ring buffer via `mach_vm_remap` for zero-copy wraparound            |
| `spmc_ring.h`     | Single-producer multi-consumer lock-free ring (speaker + Opus encoder)          |
| `kv_cache.h`      | Cache-oblivious interleaved KV cache `[H][T][2][D]` — halves L2 misses          |
| `triple_buffer.h` | Lock-free triple buffer for GPU→CPU→CoreAudio                                   |
| `arena.h`         | Bump-pointer arena allocator, checkpoint/restore, zero per-turn malloc overhead |

## Optimizations

### Hardware Utilization

pocket-voice runs on **all three Apple Silicon compute units simultaneously**:

| Unit                | What Runs                                        | Why                                |
| ------------------- | ------------------------------------------------ | ---------------------------------- |
| **Metal GPU**       | STT + TTS inference (candle)                     | Massively parallel transformer ops |
| **AMX Coprocessor** | Prosody, FFT, LUFS, LSTM endpointer (Accelerate) | Matrix-vector products, vDSP       |
| **ARM NEON**        | PCM conversion, crossfade, ring buffer copies    | 8-wide SIMD on integer/float       |

### Key Performance Techniques

- **Lock-free everywhere**: SPSC and SPMC ring buffers with 128-byte cache-line aligned atomics. Zero locks in the audio hot path.
- **Zero allocation in hot loops**: Arena allocator provides per-turn memory. CoreAudio callback does zero malloc/free.
- **Fused operations**: vDSP pitch→EQ→volume→limit chain. Interleaved KV cache eliminates separate K/V fetches.
- **VM-mirrored ring buffers**: `mach_vm_remap` creates a virtual memory mirror — reads/writes never need to handle wraparound.
- **Hardware resampling**: Apple AudioConverter for sample rate conversion is higher quality and faster than FIR for non-integer ratios.
- **Speculative prefill**: Claude API request starts at 70% EOU confidence, saving 100-300ms when the prediction is correct.
- **Conformer cache-aware streaming**: Per-layer K/V projection and convolution state caching for true frame-by-frame inference without recomputation.

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
make test    # Run everything (157 tests across 8 suites)
```

| Suite              | Tests | What It Covers                                                                |
| ------------------ | ----- | ----------------------------------------------------------------------------- |
| `test-quality`     | 35    | WER, CER, MCD, STOI, SNR, F0, speaker similarity, latency harness, grading    |
| `test-eou`         | 33    | Mimi endpointer LSTM, fused EOU, speculative prefill, Conformer EOU flags/API |
| `test-pipeline`    | 35    | Text normalization, sentence buffer, SSML parser                              |
| `test-new-modules` | 22    | Breath synthesis, LUFS, arena, VM ring, triple buffer, SPMC ring, KV cache    |
| `test-roundtrip`   | 7     | Round-trip framework with mock TTS/STT callbacks, NULL handling               |
| `test-conformer`   | 8     | Mel spectrogram extraction, Conformer STT API                                 |
| `test-bugfixes`    | 6     | SPMC mirror copy, KV cache overflow, arena accounting, LUFS non-48kHz         |
| `bench-quality`    | 11    | Self-tests for all quality metric implementations                             |

Individual suites:

```bash
make test-eou          # EOU detection + speculative prefill
make test-conformer    # Mel spectrogram + Conformer STT
make test-pipeline     # Text normalization + SSML
make test-new-modules  # All infrastructure modules
make test-bugfixes     # Regression tests
make test-roundtrip    # Round-trip intelligibility
make test-quality      # Quality metrics
make bench-quality     # Benchmark self-tests
```

## Requirements

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)
- Rust (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Homebrew curl (`brew install curl`)
- Anthropic API key (`ANTHROPIC_API_KEY`)

## Build

```bash
make          # Build everything: 18 C shared libraries + 2 Rust cdylibs + binary
make libs     # Build just the C shared libraries
make clean    # Remove all build artifacts
```

Build output:

1. `build/*.dylib` — 18 C shared libraries
2. `src/stt/target/release/libpocket_stt.dylib` — Rust STT cdylib (candle + Metal)
3. `src/tts/target/release/libpocket_tts_rs.dylib` — Rust TTS cdylib (candle + Metal)
4. `pocket-voice` — Pipeline binary linking everything

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
  --claude-model M   Claude model (default: claude-sonnet-4-20250514)
  --system PROMPT    System prompt for Claude
  --n-q N            Audio codebooks for TTS (default: 24)

VAD:
  --no-vad           Disable semantic VAD (use energy VAD only)
  --vad-threshold F  Semantic VAD threshold (default: 0.7)

Audio Post-Processing:
  --pitch F          Pitch multiplier (1.0 = normal, 1.2 = higher)
  --volume F         Volume in dB (0.0 = normal, 6.0 = louder)
  --no-hw-resample   Disable AudioConverter (use FIR fallback)
  --spatial AZ       Enable 3D spatial audio at azimuth AZ degrees
```

### Examples

```bash
# Default — conversational voice assistant
ANTHROPIC_API_KEY=sk-... ./pocket-voice

# Custom voice with pitched-up output
./pocket-voice --voice /path/to/voice.wav --pitch 1.15

# Spatial audio — voice positioned 30° to the right
./pocket-voice --spatial 30

# Custom Claude model with system prompt
./pocket-voice --claude-model claude-sonnet-4-20250514 \
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
│   ├── pocket_voice_pipeline.c     # Main orchestrator: CLI, state machine, Claude SSE
│   ├── pocket_voice.c              # CoreAudio VoiceProcessingIO engine
│   ├── vdsp_prosody.c              # AMX prosody: pitch, volume, EQ, limiter
│   ├── audio_converter.c           # Apple AudioConverter resampling
│   ├── spatial_audio.c             # HRTF binaural 3D audio
│   ├── conformer_stt.c             # Pure C Conformer CTC STT engine
│   ├── mel_spectrogram.c           # Streaming log-mel spectrogram (vDSP FFT)
│   ├── mimi_endpointer.c           # LSTM end-of-utterance on mel features
│   ├── fused_eou.c                 # 3-signal fused EOU detector
│   ├── bnns_mimi_decoder.c         # Mimi codec decoder (BNNS/ANE)
│   ├── sentence_buffer.c           # LLM token → sentence boundary detection
│   ├── ssml_parser.c               # SSML parsing and prosody extraction
│   ├── text_normalize.c            # Number/date/currency text expansion
│   ├── breath_synthesis.c          # Pink noise breath synthesis (ADSR)
│   ├── lufs.c                      # ITU-R BS.1770 LUFS loudness normalization
│   ├── opus_codec.c                # Real-time Opus encode/decode
│   ├── vm_ring.c                   # VM-mirrored ring buffer (mach_vm)
│   ├── cJSON.c                     # Vendored JSON parser (MIT)
│   ├── neon_audio.h                # ARM NEON SIMD intrinsics (header-only)
│   ├── spmc_ring.h                 # SPMC lock-free ring buffer (header-only)
│   ├── kv_cache.h                  # Interleaved KV cache (header-only)
│   ├── triple_buffer.h             # Lock-free triple buffer (header-only)
│   ├── arena.h                     # Bump-pointer arena allocator (header-only)
│   ├── quality/
│   │   ├── wer.c                   # Word/Character Error Rate (Levenshtein)
│   │   ├── audio_quality.c         # MCD, STOI, SNR, F0, speaker similarity
│   │   ├── latency_harness.c       # mach_absolute_time latency measurement
│   │   ├── roundtrip.c             # TTS→STT round-trip testing framework
│   │   └── bench_quality.c         # Quality benchmark driver
│   ├── stt/                        # Rust: Kyutai STT 1B (candle + Metal)
│   └── tts/                        # Rust: Kyutai DSM TTS 1.6B (candle + Metal)
├── tests/
│   ├── test_quality.c              # 35 quality metric tests
│   ├── test_eou.c                  # 33 EOU detection + speculative prefill tests
│   ├── test_pipeline.c             # 35 text normalize + sentence buffer + SSML tests
│   ├── test_new_modules.c          # 22 infrastructure module tests
│   ├── test_conformer_stt.c        # 8 mel spectrogram + Conformer tests
│   ├── test_roundtrip.c            # 7 round-trip framework tests
│   ├── test_bugfixes.c             # 6 regression tests
│   ├── test_load_model.c           # Conformer model load smoke test
│   └── ...
└── build/                          # Compiled output (gitignored)
```

## Companion Project

**[pocket-tts](https://github.com/sethdford/pocket-tts)** — Python/MLX text-to-speech library with custom Metal kernels, Apple AMX/Accelerate DSP, speculative decoding, and OpenAI-compatible API. pocket-voice uses Kyutai's Rust inference crates instead of MLX, but shares the same philosophy of deep Apple Silicon optimization.

## License

MIT — Copyright (c) 2026 Seth Ford
