# pocket-voice

Zero-Python real-time voice pipeline for Apple Silicon.

```
Mic → STT → Claude → TTS → Speaker
```

Everything runs in C and Rust — no Python, no GIL, no interpreter overhead. Uses CoreAudio VoiceProcessingIO for ultra-low-latency audio with built-in AEC (Acoustic Echo Cancellation).

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        pocket-voice binary                           │
│                                                                      │
│  ┌─────────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐ │
│  │ CoreAudio    │   │ Kyutai   │   │ Claude    │   │ Kyutai DSM   │ │
│  │ VoiceProc IO │──▶│ STT 1B   │──▶│ API (SSE) │──▶│ TTS 1.6B    │ │
│  │ (48kHz)      │   │ (Rust)   │   │ (libcurl) │   │ (Rust)       │ │
│  └──────┬───────┘   └──────────┘   └───────────┘   └──────┬───────┘ │
│         │                                                   │        │
│         │              Audio Post-Processing                │        │
│         │    ┌──────────────────────────────────────┐       │        │
│         │    │ vDSP Prosody (pitch/volume/limiter)  │       │        │
│         │    │ AudioConverter HW Resampler (24→48k) │◀──────┘        │
│         │    │ Spatial Audio HRTF (optional 3D)     │                │
│         ◀────│ Opus Codec (optional streaming)      │                │
│              └──────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
```

### Native Libraries

| Library                | Purpose                          | Hardware              |
| ---------------------- | -------------------------------- | --------------------- |
| `pocket_voice.c`       | CoreAudio I/O, ring buffers, VAD | CoreAudio RT thread   |
| `vdsp_prosody.c`       | Pitch shift, volume, EQ, limiter | AMX via Accelerate    |
| `audio_converter.c`    | HW sample rate conversion        | Apple AudioConverter  |
| `spatial_audio.c`      | HRTF binaural 3D positioning     | vDSP convolution      |
| `opus_codec.c`         | Real-time audio compression      | libopus               |
| `simd_audio.c`         | PCM conversion                   | ARM NEON SIMD         |
| `amx_flow_fused.c`     | Fused LSD decode on AMX          | Apple AMX coprocessor |
| `bnns_mimi_decoder.c`  | Mimi codec on ANE (stub)         | Apple Neural Engine   |
| `pocket_stt` (Rust)    | Kyutai STT 1B inference          | candle + Metal GPU    |
| `pocket_tts_rs` (Rust) | Kyutai DSM TTS 1.6B inference    | candle + Metal GPU    |

## Requirements

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)
- Rust (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Homebrew curl (`brew install curl`)
- Anthropic API key

## Build

```bash
make
```

This builds:

1. All C shared libraries (`build/`)
2. Rust STT cdylib (candle + Metal)
3. Rust TTS cdylib (candle + Metal)
4. `pocket-voice` binary linking everything together

## Run

```bash
ANTHROPIC_API_KEY=sk-ant-... ./pocket-voice
```

Speak naturally — the pipeline detects speech onset via energy VAD, transcribes with Kyutai STT, sends to Claude, and streams TTS audio back through the speaker.

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

# Use a specific Claude model with custom system prompt
./pocket-voice --claude-model claude-sonnet-4-20250514 \
  --system "You are a pirate. Respond in character."
```

## Pipeline State Machine

```
Listening → Recording → Processing → Streaming → Speaking → Listening
                                        ↑
                    Barge-in (any speaking state) ──→ Listening
```

- **Listening**: Energy VAD watches for speech onset
- **Recording**: Captures audio, feeds STT frame-by-frame
- **Processing**: Sends transcript to Claude API
- **Streaming**: Receives Claude tokens via SSE, feeds TTS incrementally
- **Speaking**: Drains remaining TTS audio to speaker
- **Barge-in**: User speaks during playback → immediate interrupt

## License

MIT
