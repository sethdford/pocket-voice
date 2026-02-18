# AGENTS.md

Guidance for AI agents working in the pocket-voice repository.

## Project Overview

pocket-voice is a zero-Python, real-time voice pipeline for Apple Silicon. It runs entirely in C and Rust: Mic -> STT -> Claude -> TTS -> Speaker. No Python, no GIL, no interpreter overhead.

**Companion project:** [pocket-tts](https://github.com/sethdford/pocket-tts) (Python/MLX TTS library). pocket-voice uses Kyutai's Rust inference crates (`candle`-based) instead of MLX.

## Code Structure

```
pocket-voice/
├── Makefile                    # Top-level build (C libs + Rust cdylibs + binary)
├── src/
│   ├── pocket_voice_pipeline.c # Main orchestrator: CLI, Claude SSE, state machine
│   ├── pocket_voice.c          # CoreAudio VoiceProcessingIO: mic/speaker, ring buffers, VAD
│   ├── vdsp_prosody.c          # AMX-backed: pitch shift (STFT), volume, biquad EQ, limiter
│   ├── audio_converter.c       # Apple AudioConverter: HW sample-rate conversion
│   ├── spatial_audio.c         # HRTF binaural 3D audio positioning
│   ├── opus_codec.c            # libopus: real-time audio compression
│   ├── bnns_mimi_decoder.c     # Mimi codec on ANE (stub — decode step not implemented)
│   ├── cJSON.c / cJSON.h       # Lightweight JSON parser (vendored, MIT)
│   ├── stt/                    # Rust cdylib: Kyutai STT 1B via candle + Metal
│   │   ├── Cargo.toml
│   │   ├── build.rs
│   │   └── src/lib.rs
│   └── tts/                    # Rust cdylib: Kyutai DSM TTS 1.6B via candle + Metal
│       ├── Cargo.toml
│       ├── build.rs
│       └── src/lib.rs
├── build/                      # Compiled .dylib output (gitignored)
├── LICENSE
└── README.md
```

## Common Commands

```bash
# Build everything (C shared libs + Rust cdylibs + binary)
make

# Build just the binary
make pocket-voice

# Clean all artifacts
make clean

# Run
ANTHROPIC_API_KEY=sk-ant-... ./pocket-voice

# Run with options
./pocket-voice --pitch 1.15 --volume 3.0 --spatial 30
```

## Architecture

### Pipeline State Machine

```
Listening → Recording → Processing → Streaming → Speaking → Listening
                                        ↑
                    Barge-in (any speaking state) ──→ Listening
```

### Key Design Decisions

1. **Zero Python**: Entire pipeline runs in C + Rust. No CPython interpreter, no GIL contention.
2. **CoreAudio VoiceProcessingIO**: Apple's built-in AEC (acoustic echo cancellation) for full-duplex audio.
3. **Lock-free ring buffers**: SPSC ring buffers with cache-line aligned atomics for zero-copy audio transfer between CoreAudio real-time thread and pipeline thread.
4. **Rust cdylibs**: STT and TTS use Kyutai's `candle`-based crates compiled as C-ABI dynamic libraries. Loaded via `dlopen` at runtime.
5. **Claude SSE**: Streams Claude responses via libcurl + Server-Sent Events for token-by-token TTS feeding.
6. **AMX/vDSP**: All audio post-processing (pitch, volume, EQ, limiting) runs on the AMX coprocessor via Apple Accelerate, concurrent with Metal GPU inference.

### Audio Post-Processing Chain

```
TTS output (24kHz) → Pitch Shift → Formant EQ → Volume → Soft Limit → Resample (48kHz) → Spatial Audio → Speaker
```

Each stage is optional and controlled by CLI flags. Processing functions are declared as FFI in `pocket_voice_pipeline.c` and linked at build time.

## Source File Details

### `pocket_voice_pipeline.c` (Main)

The orchestrator. Handles:

- CLI argument parsing
- Claude API connection (libcurl SSE)
- State machine transitions
- Coordinator between STT, TTS, and audio subsystems
- `AudioPostProcessor` struct managing all post-processing state

Key structs:

- `PipelineConfig`: All CLI-configurable options
- `AudioPostProcessor`: Holds resampler, formant EQ, spatial engine state
- `PipelineState` enum: `IDLE`, `LISTENING`, `RECORDING`, `PROCESSING`, `STREAMING`, `SPEAKING`

### `pocket_voice.c` (Audio Engine)

CoreAudio VoiceProcessingIO with:

- Dual ring buffers (capture + playback) with `__attribute__((aligned(64)))` atomics
- Energy-based VAD using `vDSP_rmsqv` with hysteresis
- 48kHz↔24kHz resampling via `vDSP_desamp` with 31-tap FIR
- Barge-in detection (speech during playback triggers state reset)

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

### `opus_codec.c` (Compression)

Wraps libopus for streaming audio compression/decompression. Not currently linked into the pipeline binary but available as a library.

### Rust Crates (`src/stt/`, `src/tts/`)

Both expose C-ABI functions (`pocket_stt_*`, `pocket_tts_rs_*`) via `#[no_mangle] extern "C"`. Use Kyutai's `moshi` Rust crate with `candle` for Metal GPU inference.

## Build System

The `Makefile` builds in three stages:

1. **C shared libraries** → `build/*.dylib` (each `.c` file compiles independently)
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

## Common Gotchas

1. **Homebrew curl**: macOS system curl may lack features. The Makefile auto-detects `brew --prefix curl`.
2. **ARM64 only**: All compilation targets `-arch arm64`. No x86 support.
3. **Rust crate versions**: The STT/TTS Rust crates pin `candle-core`, `candle-nn`, `candle-transformers` to specific Git revisions. Updating requires testing Metal kernel compatibility.
4. **CoreAudio real-time thread**: The `audioCallback` in `pocket_voice.c` runs on a real-time thread. No allocations, no locks, no syscalls allowed.
5. **AMX concurrency**: AMX coprocessor and Metal GPU are separate hardware. Post-processing (AMX) runs concurrently with inference (GPU) without contention.
6. **cJSON is vendored**: The JSON parser is a copy of [cJSON](https://github.com/DaveGamble/cJSON) (MIT license). Don't modify it; update from upstream if needed.
7. **opus_codec.c**: Requires libopus headers at compile time. Not currently built into the default `libs` target — add if needed.
8. **bnns_mimi_decoder.c**: Stub only. The `mimi_decode_step()` function is not implemented. Would require extracting and converting Mimi weights.
