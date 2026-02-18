# AGENTS.md

Guidance for AI agents working in the pocket-voice repository.

## Project Overview

pocket-voice is a zero-Python, real-time voice pipeline for Apple Silicon. It runs entirely in C and Rust: Mic -> STT -> Claude -> TTS -> Speaker. No Python, no GIL, no interpreter overhead.

**Companion project:** [pocket-tts](https://github.com/sethdford/pocket-tts) (Python/MLX TTS library). pocket-voice uses Kyutai's Rust inference crates (`candle`-based) instead of MLX.

## Code Structure

```
pocket-voice/
├── Makefile                        # Top-level build (C libs + Rust cdylibs + binary + tests)
├── README.md                       # User-facing documentation
├── AGENTS.md                       # This file — AI agent guidance
├── LICENSE                         # MIT
├── src/
│   ├── pocket_voice_pipeline.c     # Main orchestrator: CLI, Claude SSE, state machine, EOU
│   ├── pocket_voice.c              # CoreAudio VoiceProcessingIO: mic/speaker, ring buffers, VAD
│   ├── vdsp_prosody.c              # AMX-backed: pitch shift (STFT), volume, biquad EQ, limiter
│   ├── audio_converter.c           # Apple AudioConverter: HW sample-rate conversion
│   ├── spatial_audio.c             # HRTF binaural 3D audio positioning
│   ├── opus_codec.c                # libopus: real-time audio compression
│   ├── bnns_mimi_decoder.c         # Mimi codec decoder on ANE via BNNS Graph
│   ├── conformer_stt.c             # Pure C FastConformer CTC STT engine
│   ├── mel_spectrogram.c           # Streaming 80-bin log-mel spectrogram via vDSP FFT
│   ├── mimi_endpointer.c           # LSTM-based end-of-utterance detector
│   ├── fused_eou.c                 # 3-signal fused EOU: energy + LSTM + ASR token
│   ├── sentence_buffer.c           # LLM token accumulation, sentence boundary detection
│   ├── ssml_parser.c               # SSML parsing: prosody, break, say-as, emphasis
│   ├── text_normalize.c            # Number/date/currency/phone text expansion
│   ├── breath_synthesis.c          # Voss-McCartney pink noise breath synthesis
│   ├── lufs.c                      # ITU-R BS.1770 LUFS loudness normalization
│   ├── vm_ring.c                   # VM-mirrored ring buffer via mach_vm_remap
│   ├── cJSON.c / cJSON.h           # Lightweight JSON parser (vendored, MIT)
│   ├── neon_audio.h                # ARM NEON SIMD: float↔int16, copy, crossfade (header-only)
│   ├── arena.h                     # Bump-pointer arena allocator (header-only)
│   ├── spmc_ring.h                 # Single-producer multi-consumer ring buffer (header-only)
│   ├── kv_cache.h                  # Interleaved KV cache for attention (header-only)
│   ├── triple_buffer.h             # Lock-free triple buffer (header-only)
│   ├── quality/
│   │   ├── wer.c / wer.h           # Word/Character Error Rate (Levenshtein)
│   │   ├── audio_quality.c / .h    # MCD, STOI, SNR, F0, speaker similarity metrics
│   │   ├── latency_harness.c / .h  # mach_absolute_time nanosecond latency measurement
│   │   ├── roundtrip.c / .h        # TTS→STT round-trip testing framework
│   │   └── bench_quality.c         # Quality benchmark driver with golden audio
│   ├── stt/                        # Rust cdylib: Kyutai STT 1B via candle + Metal
│   │   ├── Cargo.toml
│   │   ├── build.rs
│   │   └── src/lib.rs
│   └── tts/                        # Rust cdylib: Kyutai DSM TTS 1.6B via candle + Metal
│       ├── Cargo.toml
│       ├── build.rs
│       └── src/lib.rs
├── tests/
│   ├── test_quality.c              # 35 tests: WER, CER, MCD, STOI, SNR, F0, latency, grading
│   ├── test_eou.c                  # 33 tests: Mimi EP, fused EOU, speculative prefill, Conformer EOU
│   ├── test_pipeline.c             # 35 tests: text_normalize, sentence_buffer, ssml_parser
│   ├── test_new_modules.c          # 22 tests: breath, lufs, arena, vm_ring, spmc, kv_cache
│   ├── test_conformer_stt.c        # 8 tests: mel spectrogram, Conformer STT API
│   ├── test_roundtrip.c            # 7 tests: round-trip with mock TTS/STT, NULL handling
│   ├── test_bugfixes.c             # 6 tests: SPMC mirror, KV overflow, arena, LUFS, sentbuf
│   ├── test_load_model.c           # Smoke test: load real .cstt model + inference
│   ├── test_full_forward.c         # Full Conformer forward pass on PCM
│   ├── test_debug_forward.c        # Conformer debug stats
│   └── test_validate.c             # C STT validation vs reference transcript
├── validation/
│   └── ref_transcript.txt          # Reference transcript for validation tests
├── build/                          # Compiled output (gitignored)
└── .venv/                          # Python virtualenv for tooling (torch, etc.)
```

## Common Commands

```bash
# Build everything (C shared libs + Rust cdylibs + binary)
make

# Build just the binary
make pocket-voice

# Build just the C shared libraries (no Rust)
make libs

# Clean all artifacts
make clean

# Run
ANTHROPIC_API_KEY=sk-ant-... ./pocket-voice

# Run with options
./pocket-voice --pitch 1.15 --volume 3.0 --spatial 30

# Run ALL tests (157 tests across 8 suites)
make test

# Run individual test suites
make test-eou          # EOU detection + speculative prefill (33 tests)
make test-quality      # Quality metrics (35 tests)
make test-pipeline     # Text normalize + SSML (35 tests)
make test-new-modules  # Infrastructure modules (22 tests)
make test-conformer    # Mel spectrogram + Conformer (8 tests)
make test-roundtrip    # Round-trip intelligibility (7 tests)
make test-bugfixes     # Regression tests (6 tests)
make bench-quality     # Benchmark self-tests (11 tests)
```

## Architecture

### Pipeline State Machine

```
Listening → Recording → Processing → Streaming → Speaking → Listening
                 │           │            ↑
                 │           │  (speculative prefill at 70% EOU)
                 │           │
                 └───────────┘  Barge-in (any state) ──→ Listening
```

### Key Design Decisions

1. **Zero Python**: Entire pipeline runs in C + Rust. No CPython interpreter, no GIL contention.
2. **CoreAudio VoiceProcessingIO**: Apple's built-in AEC (acoustic echo cancellation) for full-duplex audio.
3. **Lock-free ring buffers**: SPSC ring buffers with 128-byte cache-line aligned atomics for zero-copy audio transfer between CoreAudio real-time thread and pipeline thread.
4. **Rust cdylibs**: STT and TTS use Kyutai's `candle`-based crates compiled as C-ABI dynamic libraries. Loaded via `dlopen` at runtime.
5. **Claude SSE**: Streams Claude responses via libcurl + Server-Sent Events for token-by-token TTS feeding.
6. **AMX/vDSP**: All audio post-processing (pitch, volume, EQ, limiting, loudness, endpointer) runs on the AMX coprocessor via Apple Accelerate, concurrent with Metal GPU inference.
7. **Fused 3-signal EOU**: Energy VAD + LSTM endpointer + ASR-inline EOU token, weighted and smoothed via EMA, with speculative prefill at 70% confidence.

### Audio Post-Processing Chain

```
TTS output (24kHz) → Pitch Shift → Formant EQ → Volume → Soft Limit
    → LUFS Normalize → Breath Insert → Resample (48kHz) → Spatial Audio → Speaker
```

Each stage is optional and controlled by CLI flags. Processing functions are declared as FFI in `pocket_voice_pipeline.c` and linked at build time.

### Hardware Utilization

All three Apple Silicon compute units run concurrently:

| Unit                | What Runs                                                           | How                                     |
| ------------------- | ------------------------------------------------------------------- | --------------------------------------- |
| **Metal GPU**       | STT + TTS inference (candle)                                        | Transformer attention, FFN, codec       |
| **AMX Coprocessor** | Prosody, FFT, LUFS, LSTM endpointer, mel spectrogram, Conformer STT | cblas_sgemv, vDSP_fft_zrip, vDSP_biquad |
| **ARM NEON**        | PCM conversion, crossfade, ring buffer copies, breath synthesis     | 8-wide float32 SIMD                     |

## Source File Details

### `pocket_voice_pipeline.c` (Main)

The orchestrator. Handles:

- CLI argument parsing
- Claude API connection (libcurl SSE)
- State machine transitions
- Coordinator between STT, TTS, and audio subsystems
- `AudioPostProcessor` struct managing all post-processing state
- Fused 3-signal EOU detection and speculative prefill logic
- Mimi endpointer feeding from capture audio mel-energy features

Key structs:

- `PipelineConfig`: All CLI-configurable options
- `AudioPostProcessor`: Holds resampler, formant EQ, spatial engine, LUFS meter, SPMC ring, breath synth, fused EOU, Mimi endpointer, mel-feature bridge
- `PipelineState` enum: `IDLE`, `LISTENING`, `RECORDING`, `PROCESSING`, `STREAMING`, `SPEAKING`
- `TurnMetrics`: Per-turn latency tracking (speech_start, stt_done, claude_sent, first_audio, etc.)
- `SttAccum`: Frame accumulation buffer for STT

### `pocket_voice.c` (Audio Engine)

CoreAudio VoiceProcessingIO with:

- Dual ring buffers (capture + playback) with `__attribute__((aligned(64)))` atomics
- Energy-based VAD using `vDSP_rmsqv` with hysteresis
- 48kHz↔24kHz resampling via `vDSP_desamp` with 31-tap FIR
- Barge-in detection (speech during playback triggers state reset)

### `conformer_stt.c` (Pure C STT Engine)

FastConformer CTC engine:

- Loads `.cstt` binary model format (header + vocab + weights)
- Multi-head self-attention with RoPE or relative positional encoding
- Conformer blocks: MHSA + depthwise conv + feed-forward
- CTC greedy decoding with blank collapsing
- **EOU token detection**: Recognizes `<eou>`, `<|endofutterance|>`, `<eos>` tokens
- **Cache-aware streaming**: Per-layer K/V projection and convolution state caching via `LayerCache`
- All linear projections via `cblas_sgemv` (AMX-accelerated)

### `mel_spectrogram.c` (Feature Extraction)

Streaming mel spectrogram:

- 80-bin log-mel with Hann window
- vDSP FFT (`vDSP_fft_zrip`) for each frame
- Mel filterbank multiply via `cblas_sgemv`
- Natural log with configurable floor value
- Overlap buffer for frame-by-frame streaming

### `mimi_endpointer.c` (LSTM Endpointer)

Lightweight LSTM on mel-energy features from capture audio:

- Single-layer LSTM with 4-class softmax (silence, speech, ending, end-of-turn)
- All matrix-vector products via `cblas_sgemv` (AMX)
- Layer normalization on input features
- Configurable threshold and consecutive frame requirements
- `feed_endpointer()` in pipeline: splits 80ms audio frames into 80 bands, computes RMS per band

### `fused_eou.c` (3-Signal Fusion)

Combines three EOU signals:

1. **Energy VAD**: Silence detection from `pocket_voice.c`
2. **Mimi endpointer**: LSTM P(end-of-turn) from mel features
3. **ASR-inline EOU**: Token probability from Conformer/Rust STT

Features:

- Configurable per-signal weights (default: 0.4 energy, 0.3 Mimi, 0.3 STT)
- Solo thresholds for immediate trigger (e.g., energy > 0.95 → instant trigger)
- EMA smoothing on fused probability
- Consecutive frame requirement to prevent false triggers
- Requires prior speech detection (won't trigger on initial silence)

### `vdsp_prosody.c` (Audio Effects)

AMX-accelerated via Apple Accelerate:

- `prosody_pitch_shift()`: STFT-based phase vocoder pitch shifting
- `prosody_time_stretch()`: Overlap-add time stretching
- `prosody_create_formant_eq()` / `prosody_apply_biquad()`: Biquad cascade for formant correction
- `prosody_volume()`: dB-scale volume with fade-in
- `prosody_soft_limit()`: Soft-knee limiter using `vvtanhf`

### `audio_converter.c` (Resampling)

Wraps Apple's `AudioConverterNew` for hardware-accelerated sample rate conversion. Higher quality than FIR for non-integer ratios.

### `spatial_audio.c` (3D Audio)

HRTF-based binaural rendering:

- Pre-computed azimuth/elevation → ITD/ILD tables
- Per-source positioning via `spatial_set_position()`
- Stereo output downmixed to mono with phase differences preserved

### `breath_synthesis.c` (Breath Noise)

Natural breath sound synthesis:

- Voss-McCartney pink noise (16-octave 1/f spectrum)
- 2nd-order Butterworth bandpass (200-2000Hz) for vocal tract
- ADSR-shaped envelopes for natural attack-decay
- vDSP vectorized amplitude application
- NEON crossfade at segment boundaries

### `lufs.c` (Loudness)

ITU-R BS.1770-4 LUFS loudness:

- K-weighting filter chain (pre-filter + RLB high-shelf)
- Per-block gated loudness measurement
- Automatic gain normalization to target LUFS level
- Multi-sample-rate support (24kHz, 44.1kHz, 48kHz)

### `sentence_buffer.c` (Streaming Text)

Buffers streaming LLM tokens:

- Sentence boundary detection (., ?, !, semicolons)
- Predictive sentence length via EMA
- Adaptive warmup threshold
- Code block stripping (skip `blocks`)
- Speculative flush for low-latency first sentence

### `ssml_parser.c` (SSML)

SSML segment parser:

- `<prosody rate/pitch/volume>` → multiplier extraction
- `<break time>` → millisecond pause
- `<say-as interpret-as>` → text normalization dispatch
- `<emphasis level>` → rate/volume adjustment
- Returns array of `SSMLSegment` structs with per-segment parameters

### Quality Framework (`src/quality/`)

Native C quality metrics for proving STT+TTS quality:

- **WER/CER** (`wer.c`): Levenshtein-based with text normalization (lowercase, strip punctuation, collapse whitespace)
- **MCD** (`audio_quality.c`): 13-coefficient Mel-Cepstral Distortion
- **STOI** (`audio_quality.c`): Short-Time Objective Intelligibility
- **Segmental SNR** (`audio_quality.c`): Per-frame signal-to-noise ratio
- **F0 Analysis** (`audio_quality.c`): Fundamental frequency RMSE + correlation + voicing accuracy
- **Speaker Similarity** (`audio_quality.c`): MFCC-based cosine similarity
- **Latency Harness** (`latency_harness.c`): `mach_absolute_time` with P50/P95/P99 via quickselect
- **Round-Trip** (`roundtrip.c`): Full TTS→STT loop with built-in test sentences
- **Quality Scorecard**: Weighted composite score with A/B/C/D/F grading

### Infrastructure Headers

- **`neon_audio.h`**: ARM NEON intrinsics — `neon_f32_to_s16`, `neon_s16_to_f32`, `neon_copy_f32`, `neon_mix_f32`, `neon_scale_f32`, `neon_zero_stuff_2x`. 8 floats/cycle.
- **`arena.h`**: Bump-pointer arena — `arena_alloc`, `arena_checkpoint/restore`, `arena_strdup`. Cache-line aligned. Per-turn memory with zero free calls.
- **`spmc_ring.h`**: SPMC ring — 1 producer (post-processor), up to 4 consumers (speaker, Opus, WebSocket, etc.). Uses 2x software mirroring for zero-copy peek.
- **`kv_cache.h`**: Interleaved `[H][T][2][D]` layout — K and V for same (h,t) in adjacent cache lines. Halves L2 misses during attention.
- **`triple_buffer.h`**: Lock-free triple buffer — writer→processor→reader flow without blocking.

### Rust Crates (`src/stt/`, `src/tts/`)

Both expose C-ABI functions (`pocket_stt_*`, `pocket_tts_rs_*`) via `#[no_mangle] extern "C"`. Use Kyutai's `moshi` Rust crate with `candle` for Metal GPU inference.

## Build System

The `Makefile` builds in three stages:

1. **C shared libraries** → `build/*.dylib` (18 libraries, each `.c` compiles independently)
2. **Rust cdylibs** → `src/{stt,tts}/target/release/*.dylib` (via `cargo build --release`)
3. **Pipeline binary** → `pocket-voice` (links all of the above)

Frameworks linked: Accelerate, CoreAudio, AudioToolbox, libcurl.

### Adding a New C Library

1. Create `src/my_lib.c` with C-ABI functions
2. Add a build target in `Makefile`:
   ```makefile
   $(BUILD)/libmy_lib.dylib: src/my_lib.c | $(BUILD)
       $(CC) $(CFLAGS) -shared -fPIC -framework Accelerate \
         -install_name @rpath/libmy_lib.dylib -o $@ $<
   ```
3. Add to `libs:` dependencies
4. Add FFI declarations in `pocket_voice_pipeline.c`
5. Link via `-lmy_lib` in the `pocket-voice` target

### Adding Tests

1. Create `tests/test_my_feature.c`
2. Add a Make target:
   ```makefile
   test-my-feature: tests/test_my_feature.c $(BUILD)/libmy_lib.dylib | $(BUILD)
       $(CC) $(CFLAGS) -Isrc -L$(BUILD) -lmy_lib \
         -Wl,-rpath,@executable_path -o $(BUILD)/test-my-feature tests/test_my_feature.c
       ./$(BUILD)/test-my-feature
   ```
3. Add to `.PHONY` and to the `test:` dependencies

### Test Binary rpath

Test binaries are compiled into `build/`, so rpath is `@executable_path` (not `@executable_path/build`). The main `pocket-voice` binary lives in the project root, so its rpath is `@executable_path/build`.

## Common Gotchas

1. **Homebrew curl**: macOS system curl may lack features. The Makefile auto-detects `brew --prefix curl`.
2. **ARM64 only**: All compilation targets `-arch arm64`. No x86 support.
3. **Rust crate versions**: The STT/TTS Rust crates pin `candle-core`, `candle-nn`, `candle-transformers` to specific Git revisions. Updating requires testing Metal kernel compatibility.
4. **CoreAudio real-time thread**: The `audioCallback` in `pocket_voice.c` runs on a real-time thread. No allocations, no locks, no syscalls allowed.
5. **AMX concurrency**: AMX coprocessor and Metal GPU are separate hardware. Post-processing (AMX) runs concurrently with inference (GPU) without contention.
6. **cJSON is vendored**: The JSON parser is a copy of [cJSON](https://github.com/DaveGamble/cJSON) (MIT license). Don't modify it; update from upstream if needed.
7. **ACCELERATE_NEW_LAPACK**: All files that include `<Accelerate/Accelerate.h>` must define `ACCELERATE_NEW_LAPACK` to suppress deprecation warnings. Guard with `#ifndef` to avoid redefinition when also passed via `-DACCELERATE_NEW_LAPACK` CFLAG.
8. **Test binary rpath**: Test binaries go in `build/` so use `-Wl,-rpath,@executable_path` (they're co-located with the dylibs). The main binary uses `-Wl,-rpath,@executable_path/build`.
9. **Mimi endpointer input**: The LSTM takes 80-dimensional features (mel-energy bands, not raw Mimi latents). The pipeline's `feed_endpointer()` splits 1920 samples (80ms @ 24kHz) into 80 bands and computes per-band RMS.
10. **Speculative prefill thresholds**: Sends at fused_prob >= 0.70, cancels at fused_prob < 0.30. These are tuned for conversational latency; adjust for different use cases.

## End-of-Utterance Detection System

The EOU system is inspired by NVIDIA's Parakeet Realtime EOU 120M and combines three independent signals:

### Signal 1: Energy VAD (`pocket_voice.c`)

- `vDSP_rmsqv` on capture frames
- Hysteresis-based speech/silence classification
- Near-zero latency but prone to false positives (pauses within sentences)

### Signal 2: Mimi Endpointer LSTM (`mimi_endpointer.c`)

- Fed by `feed_endpointer()` in the pipeline at ~12.5Hz
- Capture audio (24kHz) → 80-band RMS features → LSTM → 4-class softmax
- Classes: silence, speech, ending, end-of-turn
- `cblas_sgemv` for AMX-accelerated matrix-vector products

### Signal 3: ASR-Inline EOU Token (`conformer_stt.c`)

- Conformer CTC decoder detects `<eou>` / `<|endofutterance|>` / `<eos>` tokens
- Per-frame softmax probability for EOU token
- Trailing average over recent frames for smoothing
- Most semantically meaningful signal (model understands sentence completion)

### Fusion (`fused_eou.c`)

- Weighted combination: `0.4 * energy + 0.3 * mimi + 0.3 * stt`
- Solo thresholds: any signal > 0.95 triggers immediately
- EMA smoothing on fused probability (α = 0.3)
- Consecutive frame requirement (default: 2 frames = 160ms)
- Requires prior speech detection to avoid false triggers on initial silence

### Speculative Prefill (`pocket_voice_pipeline.c`)

- At fused_prob >= 0.70: sends Claude API request with current transcript
- At fused_prob < 0.30: cancels in-flight request (user resumed speaking)
- If EOU triggers with speculative request in-flight: skips `STATE_PROCESSING` → direct to `STATE_STREAMING`
- Saves 100-300ms of perceived latency on correct predictions

## Quality Assurance

### Metrics (all in native C)

| Metric      | Implementation                                      | Golden Signal |
| ----------- | --------------------------------------------------- | ------------- |
| WER         | Levenshtein edit distance with text normalization   | < 5%          |
| CER         | Character-level edit distance                       | < 2%          |
| STOI        | Short-time energy correlation per 1/3-octave band   | > 0.9         |
| MCD         | 13-coefficient mel-cepstral distortion              | < 4.0 dB      |
| Seg-SNR     | Per-frame signal-to-noise ratio                     | > 25 dB       |
| F0 RMSE     | Fundamental frequency error                         | < 15 Hz       |
| F0 Corr     | F0 Pearson correlation                              | > 0.85        |
| Speaker Sim | MFCC cosine similarity                              | > 0.90        |
| RTF         | Real-time factor (generation time / audio duration) | < 0.2x        |
| First Chunk | Time to first audio chunk                           | < 200ms       |
| P50/P95/P99 | Latency percentiles via quickselect                 | Tracked       |

### Quality Scorecard

Weighted composite score (0-100) with letter grading:

- **A** (90+): Best-in-class
- **B** (80+): Excellent
- **C** (70+): Good
- **D** (60+): Needs improvement
- **F** (<60): Failing

### Round-Trip Testing

Ultimate proof: `text → TTS → audio → STT → transcript → WER(original, transcript)`. If WER < 5%, both systems are human-level intelligible.

## Performance Optimization Architecture

### Current Optimizations

**Kernel-Level:**

- ARM NEON SIMD for PCM conversion (8 samples/cycle via `vld1q_f32`/`vmovn_s32`)
- vDSP FFT for mel spectrogram and pitch shift (AMX-accelerated)
- `cblas_sgemv` for all Conformer and LSTM linear projections (AMX)
- Fused prosody chain (pitch→EQ→volume→limit in minimal passes)
- Interleaved KV cache `[H][T][2][D]` for halved L2 misses

**Data Structure:**

- VM-mirrored ring buffers for zero-copy wraparound
- Lock-free SPSC (audio) and SPMC (post-processing → speaker + Opus)
- Bump-pointer arena for zero per-turn allocation overhead
- 128-byte cache-line aligned atomics (M-series DMA line size)

**Algorithmic:**

- Speculative prefill at 70% EOU confidence
- Cache-aware Conformer streaming (per-layer K/V + conv state)
- EMA-smoothed fused EOU with consecutive frame requirement
- Sentence buffer predictive length for optimal flush timing
- Adaptive warmup threshold in sentence boundary detection

**System-Level:**

- CoreAudio real-time thread for audio I/O (no malloc, no locks)
- Metal GPU for inference concurrent with AMX for post-processing
- Hardware AudioConverter for sample rate conversion
- BNNS Graph for Mimi decoder → ANE (when available)

### Future Opportunities

1. **Conformer on ANE**: Export trained Conformer to CoreML/BNNS for Neural Engine inference
2. **Mimi encoder on capture**: Run full Mimi encoder on mic audio for richer endpointer features
3. **Learned EOU weights**: Train the fused EOU weights on real conversation data
4. **Opus streaming**: Integrate Opus encoder consumer for network streaming
5. **Multi-turn context**: Cache Conformer encoder states across turns for faster re-engagement
