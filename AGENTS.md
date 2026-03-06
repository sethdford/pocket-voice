# AGENTS.md

Guidance for AI agents working in the Sonata repository.

## Project Overview

Sonata (formerly pocket-voice) is a zero-Python, real-time voice pipeline for Apple Silicon. It runs entirely in C and Rust: Mic -> STT -> LLM -> TTS -> Speaker. No Python, no GIL, no interpreter overhead.

The flagship **Sonata TTS** is a from-scratch SOTA speech synthesis system:

- **Sonata Codec** (16.8M): Conformer encoder + FSQ quantizer + Vocos-style iSTFT decoder
- **Sonata LM** (241.7M): Llama-style transformer predicting semantic tokens at 50 Hz
- **Sonata Flow** (35.7M): Conditional Flow Matching with CFG, speaker conditioning, cumulative phase
- **iSTFT Decoder**: vDSP/AMX-accelerated, 5000x+ faster than realtime
- **Streaming Chunked Generation**: Adaptive chunk sizes (12 first / 20-80 subsequent tokens) with prosody-aware boundary detection and crossfade
- **Sub-sentence Streaming**: Eager 4-word flush + LM text append — starts audio before full sentence arrives

Supports Claude, Gemini, Gemini Live, OpenAI Realtime, and on-device Llama 3.2 as LLM backends.

## Code Structure

```
sonata/
├── Makefile                        # Top-level build (C libs + Rust cdylibs + binary + tests)
├── README.md                       # User-facing documentation
├── AGENTS.md                       # This file — AI agent guidance
├── LICENSE                         # MIT
├── config.example.json             # Example JSON config file
├── scripts/benchmark.sh            # Comprehensive benchmark suite
├── scripts/extract_silero_weights.py  # Extract Silero ONNX → .nvad for native_vad
├── scripts/eval_all.sh            # Unified evaluation pipeline
├── scripts/export_flow_v3.py      # Export Flow v3 to safetensors
├── scripts/export_phoneme_map.py  # Export phoneme vocab for C/Rust
├── scripts/normalize_text.py      # Python text normalizer
├── src/
│   ├── pocket_voice_pipeline.c     # Main orchestrator: CLI, LLM SSE (Claude/Gemini), state machine, EOU
│   ├── pocket_voice.c              # CoreAudio VoiceProcessingIO: mic/speaker, ring buffers, VAD
│   ├── vdsp_prosody.c              # AMX-backed: pitch shift (STFT), volume, biquad EQ, limiter
│   ├── audio_converter.c           # Apple AudioConverter: HW sample-rate conversion
│   ├── audio_mixer.c / .h          # 4-channel lock-free mixer with ducking, crossfade, soft limiter
│   ├── spatial_audio.c             # HRTF binaural 3D audio positioning
│   ├── opus_codec.c                # libopus: real-time audio compression
│   ├── conformer_stt.c / .h        # Pure C FastConformer CTC STT engine
│   ├── mel_spectrogram.c / .h      # Streaming 80-bin log-mel spectrogram via vDSP FFT
│   ├── mimi_endpointer.c / .h      # LSTM-based end-of-utterance detector
│   ├── fused_eou.c / .h            # 3-signal fused EOU: energy + LSTM + ASR token
│   ├── sentence_buffer.c / .h      # LLM token accumulation, sentence boundary detection
│   ├── ssml_parser.c / .h          # SSML parsing: prosody, break, say-as, emphasis, emotion
│   ├── text_normalize.c / .h       # Number/date/currency/phone/URL/email/abbreviation text expansion
│   ├── breath_synthesis.c / .h     # Voss-McCartney pink noise breath synthesis
│   ├── lufs.c / .h                 # ITU-R BS.1770 LUFS loudness normalization
│   ├── vm_ring.c / .h              # VM-mirrored ring buffer via mach_vm_remap
│   ├── http_api.c / .h             # REST API server: STT/TTS/chat endpoints over HTTP
│   ├── sonata_istft.c / .h         # Sonata iSTFT decoder: mag+phase → audio via vDSP (5000x RT)
│   ├── sonata_stt.c / .h           # Sonata STT: CTC streaming ASR (RoPE conformer, beam, EOU, timestamps)
│   ├── sonata_refiner.c / .h       # Sonata Refiner: semantic → text encoder-decoder (GQA, KV cache)
│   ├── speculative_gen.c / .h     # Multi-draft speculative LLM generation with prefix validation
│   ├── spm_tokenizer.c / .h        # SentencePiece tokenizer (pure C, greedy unigram)
│   ├── prosody_predict.c / .h      # Text-based prosody: syllables, emotion, adaptation, EmoSteer
│   ├── prosody_log.c / .h          # Real-time JSONL prosody logging for dashboard visualization
│   ├── emphasis_predict.c / .h     # Linguistics-based emphasis + quoted speech detection
│   ├── voice_onboard.c / .h        # Real-time voice onboarding for prosody transfer
│   ├── phonemizer.c / .h           # espeak-ng IPA phonemizer for TTS text preprocessing
│   ├── speaker_encoder.c / .h      # ONNX speaker encoder for zero-shot voice cloning
│   ├── speaker_diarizer.c / .h     # ONNX speaker diarization (multi-speaker segmentation)
│   ├── noise_gate.c / .h           # Spectral noise gate for STT preprocessing (vDSP FFT)
│   ├── native_vad.c / .h           # Pure C VAD: learned STFT + Conv + LSTM, AMX-accelerated
│   ├── intent_router.c / .h        # Neural MLP response routing: fast/medium/full/backchannel
│   ├── neural_backchannel.c / .h   # TTS-synthesized voice-matched backchannels (9 types)
│   ├── response_cache.c / .h       # Pre-synthesized audio cache for instant fast-path responses
│   ├── speech_detector.c / .h      # Unified VAD + EOU: wraps native_vad + mimi_ep + fused_eou
│   ├── bnns_conformer.c / .h       # BNNS Conformer: ANE-offloaded STT encoder
│   ├── bnns_convnext_decoder.c / .h # BNNS ConvNeXt decoder: ANE-offloaded Sonata decoder
│   ├── tdt_decoder.c / .h          # Token-and-Duration Transducer decoder
│   ├── ctc_beam_decoder.cpp / .h   # CTC prefix beam search with optional KenLM LM
│   ├── vap_model.c / .h            # Voice Activity Projection: transformer turn-taking at 50Hz
│   ├── voice_quality.c / .h        # Voice quality measurement (Accelerate-based)
│   ├── latency_profiler.c / .h     # mach_absolute_time nanosecond latency profiling
│   ├── metal_loader.c / .h         # Metal shader loader (Objective-C bridge)
│   ├── apple_perf.c / .h           # Apple Silicon perf: RT threads, AMX, huge pages, IOSurface zero-copy
│   ├── audio_emotion.c / .h        # Real-time audio emotion: valence, arousal, rate, pitch (vDSP)
│   ├── backchannel.c / .h          # Active listening backchannel ("mhm", "yeah") from mel features
│   ├── conversation_memory.c / .h  # Multi-turn conversation memory with JSONL persistence
│   ├── web_remote.c / .h           # WebSocket remote mic: phone browser → PCM → pipeline
│   ├── websocket.c / .h            # RFC 6455 WebSocket: upgrade, frames, PING/PONG
│   ├── tensor_ops.metal            # Metal compute shaders for GPU tensor operations
│   ├── silero_vad.h                # (deprecated) Silero ONNX VAD header — use native_vad instead
│   ├── sonata_storm.h              # Sonata Storm FFI header (Rust cdylib interface)
│   ├── streaming_llm.c / .h        # Gemini Live + OpenAI Realtime WebSocket audio LLM backends
│   ├── streaming_tts.c / .h        # Continuous speculative TTS with token-level rollback
│   ├── cJSON.c / cJSON.h           # Lightweight JSON parser (vendored, MIT)
│   ├── neon_audio.h                # ARM NEON SIMD: float↔int16, copy, crossfade (header-only)
│   ├── arena.h                     # Bump-pointer arena allocator (header-only)
│   ├── spmc_ring.h                 # Single-producer multi-consumer ring buffer (header-only)
│   ├── kv_cache.h                  # Interleaved KV cache for attention (header-only)
│   ├── triple_buffer.h             # Lock-free triple buffer (header-only)
│   ├── lstm_ops.h                  # Shared LSTM step for native_vad + mimi_endpointer (header-only)
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
│   ├── llm/                        # Rust cdylib: On-device LLM (Llama-3.2-3B) via candle + Metal
│   │   ├── Cargo.toml
│   │   ├── build.rs
│   │   └── src/lib.rs
│   ├── local_llm/                  # Rust cdylib: Local LLM alternative (candle + Metal)
│   │   ├── Cargo.toml
│   │   ├── build.rs
│   │   └── src/lib.rs
│   ├── sonata_lm/                  # Rust cdylib: Sonata Semantic LM (241M) via candle + Metal
│   │   ├── Cargo.toml             #   Top-k + top-p sampling, repetition penalty, temperature
│   │   ├── build.rs
│   │   └── src/lib.rs
│   ├── sonata_flow/                # Rust cdylib: Sonata Flow (35.7M) + ConvNeXt Decoder
│   │   ├── Cargo.toml             #   CFG, speaker embedding, cumulative phase, configurable ODE steps
│   │   ├── build.rs
│   │   └── src/lib.rs
│   ├── sonata_storm/               # Rust cdylib: Sonata Storm via candle + Metal
│   │   ├── Cargo.toml
│   │   ├── build.rs
│   │   └── src/lib.rs
│   └── metal/                      # Metal shader sources
│       └── tensor_ops.metal        # GPU tensor operations (also compiled at top level)
├── tests/
│   ├── test_quality.c              # 35 tests: WER, CER, MCD, STOI, SNR, F0, latency, grading
│   ├── test_eou.c                  # 33 tests: Mimi EP, fused EOU, speculative prefill, Conformer EOU
│   ├── test_pipeline.c             # 35 tests: text_normalize, sentence_buffer, ssml_parser
│   ├── test_llm_prosody.c          # 68 tests: Gemini SSE, emotion tags, prosody flow, URLs/emails
│   ├── test_new_modules.c          # 22 tests: breath, lufs, arena, vm_ring, spmc, kv_cache
│   ├── test_conformer_stt.c        # 8 tests: mel spectrogram, Conformer STT API
│   ├── test_roundtrip.c            # 7 tests: round-trip with mock TTS/STT, NULL handling
│   ├── test_bugfixes.c             # 6 tests: SPMC mirror, KV overflow, arena, LUFS, sentbuf
│   ├── test_sonata.c               # Sonata iSTFT, SPM, LM, flow API, chunking, phase, crossfade
│   ├── test_sonata_stt.c           # Sonata STT + Refiner: CTC, streaming, EOU, timestamps, beam
│   ├── test_sonata_v3.c            # Flow v3 + Vocoder FFI tests
│   ├── test_sonata_quality.c       # Sonata audio quality: WAV gen, stats, round-trip WER
│   ├── test_sonata_flow_ffi.c      # Sonata Flow Rust FFI tests
│   ├── test_sonata_lm_ffi.c        # Sonata LM Rust FFI tests
│   ├── test_sonata_storm.c         # Sonata Storm Rust FFI tests
│   ├── test_native_vad.c           # Native C VAD: API + model + audio tests (20 tests)
│   ├── test_bench_vad.c            # Native vs ONNX VAD benchmark + validation (7 tests)
│   ├── test_speech_detector.c      # Unified SpeechDetector: API + lifecycle + speech + EOU (25 tests)
│   ├── test_prosody_predict.c      # Prosody prediction: syllables, emotion, adaptation (52 tests)
│   ├── test_prosody_log.c          # Prosody logging, pause frames, chunk boundaries (31 tests)
│   ├── test_prosody_integration.c  # Full pipeline: emphasis → SSML → prosody → duration (40 tests)
│   ├── test_emphasis.c             # Emphasis prediction, quoted speech, prosody feedback (40 tests)
│   ├── test_real_models.c          # Integration tests with real ONNX models (20 tests)
│   ├── test_new_engines.c          # Phonemizer, speaker encoder tests
│   ├── test_beam_search.c          # CTC beam search + KenLM (17 tests)
│   ├── test_websocket.c            # WebSocket protocol unit tests
│   ├── test_voice_onboard.c        # Voice onboarding: capture, F0, prosody profile (21 tests)
│   ├── test_optimizations.c        # BNNS conformer, text normalize, latency profiler tests
│   ├── test_quality_improvements.c # Noise gate + LUFS + voice quality tests
│   ├── test_conversation_memory.c  # Conversation memory unit tests
│   ├── test_diarizer.c             # Speaker diarizer unit tests
│   ├── test_vdsp_prosody.c         # vDSP prosody (pitch shift, EQ, volume) tests
│   ├── test_http_api.c             # HTTP API server unit tests
│   ├── test_audio_emotion.c        # Audio emotion detection tests
│   ├── test_apple_perf.c           # Apple performance counter tests
│   ├── test_phonemizer_v3.c        # Phonemizer v3 tests
│   ├── test_pipeline_threading.c   # Pipeline threading / vm_ring tests
│   ├── test_phase2_regressions.c   # Phase 2 regression tests
│   ├── test_audio_mixer.c          # Audio mixer: ducking, crossfade, soft limiter (19 tests)
│   ├── test_full_duplex.c          # Full-duplex integration: all modules working together (46 tests)
│   ├── test_intent_router.c        # Intent router: routing paths, confidence, heuristic (26 tests)
│   ├── test_neural_backchannel.c   # Neural backchannel: cache, generate, speaker match (27 tests)
│   ├── test_response_cache.c       # Response cache: warm, save/load, variants (55 tests)
│   ├── test_speculative_gen.c      # Speculative gen: drafts, commit, cancel, prefix (38 tests)
│   ├── test_streaming_llm.c        # Streaming LLM: Gemini/OpenAI protocol, base64 (23 tests)
│   ├── test_streaming_tts.c       # Streaming TTS: feed, rollback, crossfade (39 tests)
│   ├── test_vap.c                 # VAP model: KV cache, smoothing, streaming (23 tests)
│   ├── bench_sonata.c              # Sonata TTS benchmark: LM, Flow+Decoder, iSTFT, E2E
│   ├── bench_live.c                # Live STT benchmark with real audio
│   └── bench_industry.c            # Industry benchmark comparison
├── train/
│   ├── scripts/
│   │   ├── precompute_phonemes.py     # Precompute phoneme IDs for manifests
│   │   └── build_unified_manifest.py  # Merge multi-dataset manifests
│   ├── requirements-train.txt        # Python training dependencies
│   └── sonata/                       # Sonata model training (PyTorch/MPS)
│       ├── config.py                 # FlowV3Config, FlowV3LargeConfig, VocoderLargeConfig; shared config
│       ├── codec.py                  # Sonata Codec: Conformer + FSQ + iSTFT decoder
│       ├── semantic_lm.py            # Sonata LM: Llama-style semantic token predictor
│       ├── flow.py                   # Sonata Flow: Conditional Flow Matching
│       ├── flow_v3.py                # Sonata Flow v3: Interleaved streaming CFM (text → mel)
│       ├── vocoder.py                # Sonata Vocoder: BigVGAN-lite (mel → waveform)
│       ├── train_codec.py            # Codec training script
│       ├── train_lm.py               # Semantic LM training script
│       ├── train_flow.py             # Flow training with emotion/prosody/duration conditioning
│       ├── train_flow_v3.py          # Flow v3 training script
│       ├── train_vocoder.py          # Vocoder training script
│       ├── train_joint_v3.py         # Joint Flow v3 + Vocoder fine-tuning
│       ├── train_flow_distill.py     # Flow consistency distillation (8→1 step)
│       ├── train_distill_v3.py       # Flow v3 consistency distillation (8→1 step)
│       ├── compute_emosteer.py       # EmoSteer direction vector computation from labeled audio
│       ├── eval_prosody_ab.py        # A/B prosody evaluation harness with objective metrics
│       ├── eval_tts.py               # Comprehensive TTS evaluation (PESQ, STOI, WER, UTMOS)
│       ├── eval_quality.py           # Audio quality evaluation (codec mode + UTMOS)
│       ├── prepare_prosody_data.py   # Dataset prep for EmoV-DB/RAVDESS/VCTK with prosody features
│       ├── perceptual_loss.py        # PerceptualMelLoss + WavLMPerceptualLoss
│       ├── g2p.py                    # Grapheme-to-Phoneme frontend (66-token IPA vocab)
│       ├── synthesize.py            # End-to-end TTS inference script
│       ├── generate_test_audio.py    # Generate test audio from checkpoints
│       ├── visualize_alignment.py    # Alignment + duration visualization
│       ├── acquire_data.py           # Dataset download and manifest building
│       ├── stt.py                    # Sonata STT: SonataCTC + SonataRefiner + SubwordCTCTokenizer
│       ├── train_stt.py              # STT training: CTC + refiner with SpecAugment, speed perturb
│       ├── train_vap.py              # VAP model training (Fisher/Switchboard/synthetic)
│       ├── train_intent_router.py     # Intent router MLP training from conversation logs
│       └── train_all.sh              # Unified training pipeline (codec→LM→flow→emosteer)
├── tools/
│   ├── gen_wav.c                    # WAV file generation utility
│   ├── prosody_dashboard.html       # Web-based prosody visualization dashboard
│   └── training_dashboard.html      # Web-based training loss visualization
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
make sonata

# Build just the C shared libraries (no Rust)
make libs

# Clean all artifacts
make clean

# Run with Claude (default)
ANTHROPIC_API_KEY=sk-ant-... ./sonata

# Run with Gemini
GEMINI_API_KEY=... ./sonata --llm gemini

# Run with Gemini Flash Lite + prosody enhancement
GEMINI_API_KEY=... ./sonata --llm gemini --llm-model gemini-2.5-flash-lite --prosody

# Run with Claude + prosody-aware SSML annotations
ANTHROPIC_API_KEY=sk-ant-... ./sonata --prosody

# Run with options
./sonata --pitch 1.15 --volume 3.0 --spatial 30

# Run with local LLM (Llama-3.2-3B on Metal)
./sonata --llm local

# Run ALL tests (30 suites via `make test`)
make test

# Run individual test suites (included in `make test`)
make bench-quality             # Benchmark self-tests (11 tests)
make test-quality              # Quality metrics (35 tests)
make test-eou                  # EOU detection + speculative prefill (33 tests)
make test-roundtrip            # Round-trip intelligibility (7 tests)
make test-pipeline             # Text normalize + SSML (35 tests)
make test-new-modules          # Infrastructure modules (22 tests)
make test-new-engines          # Phonemizer, speaker encoder tests
make test-bugfixes             # Regression tests (6 tests)
make test-conformer            # Mel spectrogram + Conformer (8 tests)
make test-llm-prosody          # Gemini SSE, emotion tags, prosody flow (68 tests)
make test-optimizations        # BNNS conformer, text normalize, latency profiler
make test-beam-search          # CTC beam search + KenLM (17 tests)
make test-sonata               # Sonata iSTFT, SPM, LM, flow, chunking, phase
make test-sonata-v3            # Flow v3 + Vocoder FFI tests
make test-real-models          # Integration tests with real ONNX models (20 tests)
make test-prosody-predict      # Prosody prediction: syllables, emotion, adaptation (52 tests)
make test-prosody-log          # Prosody logging, pause frames, chunk boundaries (31 tests)
make test-emphasis             # Emphasis prediction, quoted speech, prosody feedback (40 tests)
make test-prosody-integration  # Full pipeline: emphasis → SSML → prosody → duration (40 tests)
make test-voice-onboard        # Voice onboarding: capture, F0, prosody profile (21 tests)
make test-conversation-memory  # Conversation memory unit tests
make test-diarizer             # Speaker diarizer unit tests
make test-vdsp-prosody         # vDSP prosody effects tests
make test-http-api             # HTTP API server unit tests
make test-sonata-storm         # Sonata Storm Rust FFI tests
make test-audio-emotion        # Audio emotion detection tests
make test-sonata-flow-ffi      # Sonata Flow Rust FFI tests
make test-sonata-lm-ffi        # Sonata LM Rust FFI tests
make test-pipeline-threading   # Pipeline threading / vm_ring tests
make test-phase2-regressions   # Phase 2 regression tests

# Full-duplex / SOTA tests
make test-vap                 # VAP turn-taking model (23 tests)
make test-audio-mixer          # Audio mixer ducking/crossfade (19 tests)
make test-neural-backchannel   # Neural backchannel synthesis (27 tests)
make test-intent-router        # Intent router paths (26 tests)
make test-speculative-gen      # Speculative generation drafts (38 tests)
make test-full-duplex          # Full-duplex integration (46 tests)
make test-response-cache       # Response cache warm/save/load (55 tests)
make test-streaming-llm        # Streaming LLM protocols (23 tests)
make test-streaming-tts        # Streaming TTS rollback (39 tests)

# Additional test targets (not in `make test` aggregate)
make test-sonata-quality       # Sonata audio quality: WAV gen, stats, round-trip WER
make test-sonata-stt           # Sonata STT: CTC, streaming, EOU, beam search
make test-native-vad           # Native C VAD: API + model + audio tests (20 tests)
make bench-vad                 # Native vs ONNX VAD benchmark (7 tests)
make test-speech-detector      # Unified SpeechDetector: API + lifecycle + speech + EOU (25 tests)
make test-quality-improvements # Noise gate + LUFS + voice quality tests
make test-websocket            # WebSocket protocol unit tests
make test-apple-perf           # Apple performance counter tests

# Benchmarks
make bench-sonata              # Sonata TTS benchmark: LM tok/s, Flow+Decoder RTF, iSTFT, E2E latency
make bench-live                # Live STT benchmark with real audio
make bench-industry            # Industry benchmark comparison
make bench                     # Run comprehensive benchmark suite (scripts/benchmark.sh)

# Run with JSON config file
./sonata --config config.json

# Run as HTTP API server
./sonata --server --server-port 8080

# HTTP API endpoints (when in server mode)
curl http://localhost:8080/health
curl -X POST http://localhost:8080/v1/audio/transcriptions -d @audio.wav
curl -X POST http://localhost:8080/v1/audio/speech -d "Hello world" -o output.wav
curl -X POST http://localhost:8080/v1/chat -d "Tell me a joke"

# OpenAI-compatible TTS request
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello world","voice":"alloy","response_format":"opus"}' \
  -o output.opus

# TTS with word timestamps
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","word_timestamps":true}'

# Voice cloning
curl -X POST http://localhost:8080/v1/voices -d @voice.wav
curl http://localhost:8080/v1/voices
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello","voice":"voice_0"}' -o output.wav

# API key authentication
SONATA_API_KEY=my-secret ./sonata --server
curl -H "Authorization: Bearer my-secret" http://localhost:8080/v1/voices
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
4. **Rust cdylibs**: STT uses Kyutai's `candle`-based crates; Sonata LM/Flow use `candle` for Metal inference. All compiled as C-ABI dynamic libraries. Linked at build time via `-l` flags (dyld resolves at launch, not `dlopen`).
5. **LLM abstraction**: `LLMClient` interface with function pointers enables swappable LLM backends (Claude, Gemini). Streams responses via libcurl + SSE for token-by-token TTS feeding.
6. **AMX/vDSP**: All audio post-processing (pitch, volume, EQ, limiting, loudness, endpointer) runs on the AMX coprocessor via Apple Accelerate, concurrent with Metal GPU inference.
7. **Fused 3-signal EOU**: Energy VAD + LSTM endpointer + ASR-inline EOU token, weighted and smoothed via EMA, with speculative prefill at 70% confidence.
8. **Prosody annotation**: Optional SSML-aware system prompt (`--prosody`) instructs the LLM to annotate responses with `<emphasis>`, `<break>`, `<prosody>`, and `<emotion>` tags. The existing SSML parser + prosody chain renders these as pitch/volume/timing changes — analogous to "strong prompting" in TTS benchmarks.
9. **Auto-intonation rules**: Context-aware prosody when no explicit SSML pitch is set:
   - `?` → +8% pitch, 5% slower (interrogative contour)
   - `!` → +6% pitch, +1.5 dB (exclamatory energy)
   - `,` → +3% pitch (continuation rise)
   - `;` / `:` → -3% pitch, 5% slower (moderate boundary)
   - Em-dash / `--` → -4% pitch, 7% slower (parenthetical aside)
   - Quoted text → +4% pitch (reported speech voice shift)
10. **Emotion-conditioned prosody**: The `<emotion type="...">` SSML tag maps 12 emotion labels (happy, excited, sad, angry, fearful, surprised, warm, serious, calm, confident, whisper, emphatic) to rate/pitch/volume. SSML parser is the single source of truth — no double application.
11. **Model-aware prosody**: Pipeline passes prosody parameters (log_pitch, energy, rate) to both Sonata LM (`sonata_lm_set_prosody`) and Sonata Flow (`sonata_flow_set_prosody`) so the models generate tokens and acoustic features conditioned on the desired expressiveness, not just post-hoc audio effects.
12. **Pause tokens**: SSML `<break>` tags inject silence frames directly into the LM semantic stream (PAD token=0, each frame=20ms at 50Hz), letting the Flow model generate natural silence rather than cutting/splicing audio.
13. **Prosody-aware chunking**: Sonata chunk boundaries prefer "natural" split points (3+ consecutive identical tokens = sustained phoneme) over fixed thresholds, reducing mid-phoneme artifacts.
14. **Speaker interpolation**: `sonata_flow_interpolate_speakers()` blends two speaker embeddings with L2 normalization for voice style transfer (alpha=0→A, alpha=1→B).
15. **Prosody logging**: Optional JSONL logging of per-segment prosody parameters and per-turn metrics for real-time visualization via the web dashboard (`tools/prosody_dashboard.html`).
16. **Emphasis prediction**: Linguistics-based `emphasis_predict()` inserts `<emphasis>` tags at contrast markers, intensifiers, negations, sentence-final content words, and list-final items. Capped at 3 per sentence. Passthrough when SSML already present.
17. **Multi-voice quoted speech**: `emphasis_detect_quotes()` wraps quoted text in `<voice name="quoted">` tags; pipeline applies +8% pitch shift for audible voice differentiation.
18. **Adaptive LLM prompt**: Every 3 turns, the system prompt suffix adjusts based on conversational prosody state (user pace/energy) and TTS quality feedback.
19. **Prosody feedback loop**: After each turn, F0 range and energy variance of TTS output are analyzed. If below targets, `prosody_boost` increases (up to 1.3x), widening pitch/volume deviations for the next turn.
20. **One-stop voice cloning**: `--clone-voice voice.wav` auto-detects speaker encoder model from standard paths, extracts embedding, and sets on Flow — no separate `--speaker-encoder` flag needed.
21. **Emotion-aware barge-in**: During TTS playback, barge-in sensitivity adapts to emotional content. Empathetic/calm content (sad, warm, calm) raises the energy threshold by 40%, making it harder to interrupt comforting responses. Excited/angry content lowers it by 20%.
22. **Voice onboarding**: `voice_onboard.c` captures a short speech sample, extracts F0/energy/rate prosody profile and optionally speaker embedding for real-time voice cloning.
23. **Quote-before-emphasis ordering**: Quote detection runs before emphasis prediction to prevent false-positive quote detection from `"` chars in SSML attributes.
24. **Commercial TTS benchmarking**: `eval_prosody_ab.py --benchmark` compares Sonata against OpenAI, ElevenLabs, and Google TTS APIs on prosody metrics (MOS, F0 range, energy variance, latency).
25. **Flow v3 interleaved encoding**: Flow v3 uses interleaved text-mel encoding — no separate encoder → duration → decoder; phoneme and mel frames are jointly encoded and generated in one stream.
26. **BigVGAN-lite SnakeAlpha**: Sonata Vocoder uses SnakeAlpha activation for improved phase modeling and anti-aliased multi-periodicity (AMPBlock).
27. **Dragon-FM streaming**: Flow v3 streaming provides bidirectional attention within chunks (Dragon-FM) for higher quality streaming TTS while maintaining causal constraints across chunks.
28. **Voice Activity Projection (VAP)**: Single transformer model replaces three rule-based systems (barge-in detection, backchannel timing, EOU weights). Predicts 4 turn-taking signals at 50Hz from dual-speaker mel features. Causal self-attention with KV cache for streaming.
29. **Full-duplex pipeline**: Speech detector, VAP model, and backchannel generator run continuously during ALL pipeline states, including STREAMING and SPEAKING. Capture audio is always processed, enabling neural barge-in and backchannel during playback.
30. **4-channel audio mixer**: TTS, backchannel, pre-synthesized cache, and cloud audio LLM all route through a priority mixer with ducking. Backchannels reduce main TTS volume by 0.4x during overlap. Lock-free SPSC rings per channel.
31. **Neural intent routing**: Learned MLP classifies user utterances into fast/medium/full/backchannel response paths. Fast path serves pre-synthesized audio in <50ms. Heuristic fallback when no trained weights.
32. **Multi-draft speculation**: Instead of single-shot speculative prefill at 70% EOU, maintains 2 concurrent LLM drafts with transcript prefix validation. VAP's `p_eou` and `p_system_turn` trigger speculation earlier and more accurately.
33. **Response cache persistence**: Pre-synthesized audio for 7 response types × 3 variants, saved to disk for cross-launch reuse. Warmed at startup via TTS callback. Voice-matched when speaker embedding is set.
34. **Streaming audio LLM backends**: Gemini Live and OpenAI Realtime integrated via WebSocket. Audio sent/received bidirectionally while local STT+TTS runs in parallel for post-processing and quality control.
35. **Token-level TTS rollback**: LLM tokens feed directly to TTS without sentence buffering. Audio segments track token-to-sample mapping. On barge-in or prediction error, uncommitted audio rolls back with crossfade at splice point.

### Audio Post-Processing Chain

```
Mic (48kHz) → Resample (16kHz) → Noise Gate → STT (Conformer)

TTS output (24kHz) → Pitch Shift → Formant EQ → Volume → Soft Limit
    → LUFS Normalize → Breath Insert → Resample (48kHz) → Spatial Audio → Speaker
```

The noise gate on the STT input path reduces stationary background noise. LUFS normalization is enabled by default (-16 LUFS target). Each stage is optional and controlled by CLI flags.

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
- LLM API connection via `LLMClient` abstraction (Claude, Gemini, Gemini Live, OpenAI Realtime via libcurl SSE or WebSocket)
- State machine transitions
- Coordinator between STT, TTS, and audio subsystems
- `AudioPostProcessor` struct managing all post-processing state
- Fused 3-signal EOU detection and speculative prefill logic
- Mimi endpointer feeding from capture audio mel-energy features
- Auto-intonation rules (question, exclamation, comma, semicolon, em-dash, quoted speech)
- **Full-duplex operation**: Continuous speech detection during all states (including STREAMING and SPEAKING)
- **VAP-based neural barge-in and backchannel timing** when `--vap-model` and `--neural-barge-in` enabled
- **Audio mixer** for concurrent output sources (TTS, backchannel, cache, cloud)
- **Intent routing** for fast-path pre-synthesized responses
- **Multi-draft speculative generation** with transcript prefix validation
- **Streaming TTS** with token-level rollback on barge-in or prediction error
- **Streaming audio LLM backends** (Gemini Live, OpenAI Realtime)
- **Response cache** for pre-synthesized audio

Key structs:

- `LLMClient`: Function-pointer interface for swappable LLM backends (init, send, poll, peek_tokens, cancel, commit_turn)
- `LLMEngineType`: `LLM_ENGINE_CLAUDE`, `LLM_ENGINE_GEMINI`, `LLM_ENGINE_LOCAL`, `LLM_ENGINE_GEMINI_LIVE`, `LLM_ENGINE_OPENAI_REALTIME`
- `ClaudeClient`: Anthropic Messages API SSE implementation
- `GeminiClient`: Google Gemini streamGenerateContent SSE implementation
- `PipelineConfig`: All CLI-configurable options (including `llm_engine`, `prosody_prompt`)
- `AudioPostProcessor`: Holds resampler, formant EQ, spatial engine, LUFS meter, SPMC ring, breath synth, `SpeechDetector` (unified VAD + EOU)
- `PipelineState` enum: `IDLE`, `LISTENING`, `RECORDING`, `PROCESSING`, `STREAMING`, `SPEAKING`
- `TurnMetrics`: Per-turn latency tracking (speech_start, stt_done, llm_sent, first_audio, etc.)
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

### `noise_gate.c` (STT Preprocessing)

Spectral noise gate using vDSP FFT for reducing stationary background noise before STT:

- 512-point FFT with Hann analysis window, 50% overlap (256-sample hop)
- Adaptive noise floor estimation from initial silence frames
- Per-bin spectral gating with soft-knee attenuation curve
- Smoothed gain transitions (5ms attack, 50ms release) to prevent musical noise
- Automatically enabled in the pipeline's `feed_stt()` path on 16kHz audio
- Configurable threshold (default: 6 dB above noise floor)

### Native VAD (`src/native_vad.c`) — Recommended

Pure C reimplementation of Silero VAD v5, AMX-accelerated, no ONNX dependency:

- Architecture: Learned STFT → 4× Conv1d+ReLU encoder → 1-layer LSTM → Linear → Sigmoid
- 309K parameters (~1.2MB `.nvad` binary), extracted from Silero ONNX via `scripts/extract_silero_weights.py`
- 512-sample chunks (32ms at 16kHz) with 64-sample context overlap for STFT boundary accuracy
- All matrix ops via `cblas_sgemm`/`cblas_sgemv` (AMX-accelerated on Apple Silicon)
- No ONNX Runtime dependency — pure C with only Accelerate framework
- Zero allocations in the hot path (all working memory pre-allocated in struct)
- Real-time safe: can run on CoreAudio thread if needed
- Drop-in replacement API matching `silero_vad.h` interface
- CLI: `--vad models/silero_vad.nvad`
- Config: `"audio": { "vad": "models/silero_vad.nvad" }`
- Weight extraction: `python scripts/extract_silero_weights.py models/silero_vad.onnx models/silero_vad.nvad`

### Silero VAD (`src/silero_vad.h`) — Deprecated

Neural voice activity detection via ONNX Runtime. **Deprecated**: No longer built or linked; use `native_vad` instead. Only the header file (`silero_vad.h`) remains for API reference; the implementation has been removed.

- Silero VAD v5/v6 ONNX model (2.2MB) with LSTM state
- 512-sample chunks (32ms at 16kHz) → speech probability [0,1]
- Stateful: LSTM state persists between chunks for temporal context

### Speech Detector (`src/speech_detector.c`) — Unified VAD + EOU

Consolidates native_vad, mimi_endpointer, and fused_eou into a single module:

- Owns all buffer management (16kHz VAD accumulation, 24kHz endpointer framing)
- Internal 24→16kHz resampling for the VAD path
- Mel-energy feature extraction (80-band RMS via vDSP) for the endpointer
- 3-signal fusion (energy + neural VAD + Mimi EOU + STT)
- Clean lifecycle: `speech_detector_create()` → `feed()` → `speech_active()` / `eou()` → `reset()` → `destroy()`
- Supports both 24kHz pipeline feed and direct 16kHz feed paths
- `SpeechDetectorConfig` struct for all tunable parameters
- **Fully integrated into pipeline**: `AudioPostProcessor` holds a single `SpeechDetector*` replacing ~80 lines of scattered buffer management, resampling, and signal assembly

### WebSocket Streaming (`src/websocket.c`)

RFC 6455 WebSocket protocol implementation:

- WebSocket upgrade from HTTP (SHA-1 via CommonCrypto)
- Binary and text frame encoding/decoding
- Client frame unmasking per spec
- Auto PING/PONG handling
- Integrated into HTTP API at `GET /v1/stream`
- Protocol: binary audio in → JSON events + binary audio out
- Events: `{"type":"listening"}`, `{"type":"transcript","text":"..."}`, `{"type":"llm_token","text":"..."}`, `{"type":"speaking"}`

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
- `<emotion type>` → emotion-to-prosody mapping (10 emotions: happy, excited, sad, angry, fearful, surprised, warm, serious, calm, confident)
- `<voice name>` → per-segment voice selection
- Returns array of `SSMLSegment` structs with per-segment parameters (including `emotion[32]` field)

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

### CTC Beam Search Decoder (`src/ctc_beam_decoder.cpp`)

Prefix beam search with optional KenLM n-gram language model:

- Hannun (2017) prefix beam search: maintains (p_blank, p_nonblank) per hypothesis
- LM scoring at word boundaries (SentencePiece `▁` or space delimiter)
- Configurable beam size (default 16), LM weight (default 1.5), word insertion bonus
- `blank_skip_thresh` optimization: skip time steps where P(blank) dominates
- Compiled with `-DUSE_KENLM` when `third_party/kenlm/libkenlm.a` exists
- Pipeline integration: `--beam-size N --lm-path models/3-gram.pruned.1e-7.bin --lm-weight 1.5`

### Infrastructure Headers

- **`neon_audio.h`**: ARM NEON intrinsics — `neon_f32_to_s16`, `neon_s16_to_f32`, `neon_copy_f32`, `neon_mix_f32`, `neon_scale_f32`, `neon_zero_stuff_2x`. 8 floats/cycle.
- **`arena.h`**: Bump-pointer arena — `arena_alloc`, `arena_checkpoint/restore`, `arena_strdup`. Cache-line aligned. Per-turn memory with zero free calls.
- **`spmc_ring.h`**: SPMC ring — 1 producer (post-processor), up to 4 consumers (speaker, Opus, WebSocket, etc.). Uses 2x software mirroring for zero-copy peek.
- **`kv_cache.h`**: Interleaved `[H][T][2][D]` layout — K and V for same (h,t) in adjacent cache lines. Halves L2 misses during attention.
- **`triple_buffer.h`**: Lock-free triple buffer — writer→processor→reader flow without blocking.
- **`lstm_ops.h`**: Shared AMX-accelerated LSTM step function — used by `native_vad.c` and `mimi_endpointer.c`. `cblas_sgemv` for Wi/Wh, `vDSP_vadd` for bias, sigmoid/tanh gate activations. Supports strided input for column-wise access.

### Prosody Predictor (`src/prosody_predict.c`)

Text-based prosody prediction for TTS without neural models:

- **Syllable counting**: English vowel-cluster heuristic with silent-e detection
- **Duration estimation**: `prosody_estimate_durations()` distributes 50Hz frames proportional to syllable count per word
- **Multi-scale prosody**: `prosody_analyze_text()` returns `MultiScaleProsody` struct with:
  - Utterance contour: declarative / interrogative / exclamatory / imperative / continuation / list
  - Per-word prosody hints: ALL CAPS → emphasis (pitch +10%, rate -10%, energy +2dB), function words de-emphasized
  - List pattern detection (comma-separated items get rising pitch)
- **Emotion detection**: `prosody_detect_emotion()` scans for punctuation patterns (!!!, ..., ??) and 22 emotion keywords
  - Returns `EmotionDetection` with emotion ID, confidence, and suggested prosody hint
  - 11 emotions: neutral, happy, excited, sad, angry, surprised, warm, serious, calm, confident, fearful
- **Conversational adaptation**: `ConversationProsodyState` tracks user speech across turns via EMA
  - `prosody_conversation_update()` updates from (duration, words, energy, pitch)
  - `prosody_conversation_adapt()` returns `ProsodyHint` matching user's pace/energy/pitch
  - Partial mirror: 40% of user's pace deviation, 20% of energy/pitch deviation, with safety clamps
- **EmoSteer loading**: `emosteer_load()` reads emotion direction vectors from JSON file
  - Used for training-free activation steering in Sonata Flow
  - Format: `{ "dim": 512, "emotions": { "happy": [...], "sad": [...] } }`
- Pipeline integration: multi-scale prosody, duration estimation, and emotion detection all feed into `process_segment()` automatically
- CLI: `--emosteer path/to/directions.json` loads emotion steering vectors

### Prosody Log (`src/prosody_log.c`)

Real-time JSONL logging of prosody parameters for dashboard visualization:

- `prosody_log_open(path)` / `prosody_log_close(log)` — session lifecycle
- `prosody_log_segment()` — logs pitch, rate, volume, emotion, contour, duration per segment
- `prosody_log_turn()` — logs per-turn aggregate stats (VRL, TTS RTF, user rate, response prosody)
- `prosody_log_contour()` — logs raw F0/energy curves with automatic downsampling (max 200 points)
- Output format: one JSON object per line (JSONL), compatible with dashboard viewer
- Integrated into pipeline via `AudioPostProcessor.prosody_log`
- CLI: `--prosody-log prosody.jsonl` enables logging
- Config: `"audio": { "prosody_log": "prosody.jsonl" }`
- Viewer: `tools/prosody_dashboard.html` — drag-and-drop JSONL file for visual analysis

### Emphasis Predictor (`src/emphasis_predict.c`)

Lightweight emphasis and quoted-speech prediction for TTS without SSML from the LLM:

- **Emphasis rules** (linguistics-informed):
  - Contrast markers: word after "but", "however", "actually" → `<emphasis level="strong">`
  - Intensifiers: word after "very", "really", "extremely" → `<emphasis level="moderate">`
  - Negation stress: "not", "never" and next content word → emphasis
  - Sentence-final content word → mild emphasis (skips function words)
  - List-final item: "X, Y, and Z" → Z gets emphasis
  - Cap at 3 emphasis points per sentence to avoid over-marking
- **Quoted speech**: `emphasis_detect_quotes()` wraps text between quotes in `<voice name="quoted">` tags
  - Supports both straight quotes (`"..."`) and smart quotes (UTF-8 `"..."`)
  - Only wraps substantial text (>3 chars)
  - Pipeline applies +8% pitch shift for quoted voice differentiation
- **SSML passthrough**: if input already contains `<emphasis>` tags, no additional tags added
- **Ordering**: Quote detection MUST run BEFORE emphasis prediction, since emphasis inserts `"` chars in SSML attributes that would cause false-positive quote detection
- Integrated into sentence processing: LLM output → quote detection → emphasis prediction → SSML parse → process_segment

### Voice Onboarding (`src/voice_onboard.c`)

Real-time voice onboarding for prosody transfer and zero-shot voice cloning:

- Captures mic audio into a buffer (configurable duration, default 5s)
- Extracts prosody profile: F0 mean/range, energy, speaking rate
- F0 estimation via autocorrelation (vDSP-accelerated dot product)
- Speaking rate estimation from energy contour transitions
- Optional integration with ONNX speaker encoder for embedding extraction
- Progress tracking API for UI feedback
- Pipeline integration: `voice_onboard_create()` → `voice_onboard_feed()` → `voice_onboard_finalize()`

### Emotion-Aware Barge-In (`pocket_voice_pipeline.c`)

Context-sensitive barge-in sensitivity based on TTS emotional content:

- Tracks `last_tts_emotion` and `barge_in_energy_scale` in AudioPostProcessor
- Empathetic content (sad, warm, calm) → 1.4x energy threshold (harder to interrupt)
- Serious/fearful content → 1.3x threshold
- Excited/angry content → 0.8x threshold (easier to interrupt)
- Neutral content → 1.0x (default)
- Temporarily adjusts VAD thresholds via `voice_engine_set_vad_thresholds()` during barge-in check
- Resets emotion state after barge-in to prevent lingering threshold changes

### Phonemizer (`src/phonemizer.c`)

espeak-ng phonemizer for TTS text preprocessing:

- Converts text → IPA phoneme string via `espeak_TextToPhonemes()`
- Loads phoneme-to-ID mapping from JSON (greedy longest-match)
- Configurable language (en-us, en-gb, de, fr, etc.)
- **Not thread-safe** (espeak-ng is process-global)
- Pipeline flag: `--phonemize` enables IPA phonemization before TTS
- Replaces SentencePiece tokenization for improved pronunciation of heteronyms, proper nouns

### Speaker Encoder (`src/speaker_encoder.c`)

ONNX-based speaker embedding extraction for zero-shot voice cloning:

- Loads any ONNX speaker encoder (ECAPA-TDNN 192-dim, WavLM 256-dim, etc.)
- Extracts L2-normalized embeddings from reference audio
- WAV file loading with automatic resampling to 16kHz
- CoreML EP for ANE acceleration (falls back to CPU)
- Pipeline: `--speaker-encoder model.onnx --ref-wav voice.wav` → `sonata_flow_set_speaker_embedding()`
- vDSP-accelerated L2 normalization

### Moonshine STT — Removed

> **Note**: Source files (`moonshine_stt.c/.h`) have been removed from the codebase. Documentation retained for reference.

Lightweight encoder-decoder ASR via ONNX Runtime:

- Moonshine Tiny (27M) or Base (60M) — 5x faster than Whisper Tiny
- Variable-length audio (no 30s zero-padding like Whisper)
- 4-model pipeline: preprocess → encode → uncached_decode → cached_decode
- KV cache for autoregressive decoding
- HuggingFace tokenizer.json loading for detokenization
- CoreML EP for ANE acceleration
- Pipeline: `--stt-engine moonshine --moonshine-model models/moonshine/`

**Streaming API** (removed):

- `moonshine_stt_stream_start()` — start session, allocate 30s audio buffer
- `moonshine_stt_stream_feed()` — append audio frames
- `moonshine_stt_stream_flush()` — transcribe accumulated audio (growing window)
- `moonshine_stt_stream_end()` — cleanup
- Re-encodes full window on each flush (Moonshine is fast enough at 27M params)

### Sonata STT (`src/sonata_stt.c`, `src/sonata_refiner.c`)

Two-pass ASR system built on the Sonata Codec Conformer encoder:

**Pass 1 — Streaming CTC** (`sonata_stt.c`, 6.2M params):

- Reuses Sonata Codec Conformer Encoder (4L d=256) + CTC head
- RoPE positional encoding, SpecAugment, speed perturbation, noise augmentation
- 30-token character vocab: blank(0) + space(1) + a-z(2-27) + apostrophe(28) + eou(29)
- Streaming API: `sonata_stt_stream_start/feed/flush/end`
- Per-word timestamps via CTC alignment: `sonata_stt_get_words()`
- EOU detection: `sonata_stt_eou_peak()` / `sonata_stt_eou_probs()`
- Mel spectrogram: 24kHz, n_fft=1024, hop=480, 80 mels (no Slaney norm, periodic Hann)
- Binary format: `.cstt_sonata` (~24.8MB)

**Pass 2 — Refiner** (`sonata_refiner.c`, 49.3M params):

- Encoder-decoder transformer: semantic tokens → text
- Encoder: 4L self-attention (d=512, 8 heads)
- Decoder: 4L with self-attention + cross-attention + GQA (8/4 heads), KV cache
- SiLU FFN, RMSNorm, RoPE
- 95.7% token accuracy on dev-clean
- Binary format: `.cref` (~188MB)

**Pipeline integration**:

- `--stt-engine sonata --sonata-stt-model PATH --sonata-refiner PATH`
- Config: `"stt": { "engine": "sonata", "sonata_model": "...", "sonata_refiner": "..." }`
- 106x realtime CTC inference speed (0.0095 RTF)

### Piper TTS — Removed

> **Note**: Source files (`piper_tts.c/.h`) have been removed from the codebase. Documentation retained for reference.

VITS-based neural TTS via ONNX Runtime and espeak-ng:

- Phonemization via `espeak_TextToPhonemes()` (IPA output)
- Phoneme → ID mapping from Piper JSON config (Unicode codepoint table)
- Single ONNX model inference: phoneme IDs + scales → PCM audio
- Configurable: noise_scale, length_scale, noise_w
- Sample rate from model config (typically 22050 Hz)
- **CoreML EP**: Automatically uses Apple Neural Engine when available (falls back to CPU)
- Pipeline-compatible: `peek_audio` / `advance_audio` / `is_done` API
- **0% WER** on round-trip test (Piper→resample→Conformer STT+KenLM)

### Supertonic-2 TTS — Removed

> **Note**: Source files (`supertonic_tts.c/.h`) have been removed from the codebase. Documentation retained for reference.

Flow-matching neural TTS via 4 ONNX models:

- Unicode tokenization via JSON indexer (`unicode_indexer.json`)
- Voice style loading from JSON embeddings (`voice_styles/*.json`)
- 4-stage inference: text_encoder → duration_predictor → vector_estimator (N flow steps) → vocoder
- Box-Muller Gaussian noise generation for flow-matching latent
- Configurable flow steps (2-20, default 5) — quality vs speed tradeoff
- **Streaming sentence synthesis**: Splits multi-sentence text, synthesizes first sentence immediately, queues rest for lazy synthesis in `get_audio()`. Reduces time-to-first-audio to ~37ms.
- **Crossfade**: 256-sample crossfade between sentence chunks for seamless audio
- **CoreML EP**: Automatically uses Apple Neural Engine when available (falls back to CPU)
- Sample rate: 44100 Hz
- Pipeline-compatible: `peek_audio` / `advance_audio` / `is_done` API

### Kyutai DSM TTS — Removed

Kyutai's Pocket TTS model in pure C. **Removed**: Source file has been deleted from the codebase.

### Rust Crates (`src/stt/`, `src/llm/`, `src/sonata_lm/`, `src/sonata_flow/`, `src/sonata_storm/`)

STT exposes C-ABI functions (`pocket_stt_*`) via `#[no_mangle] extern "C"`. Uses Kyutai's `moshi` Rust crate with `candle` for Metal GPU inference. Sonata LM, Flow, and Storm use `candle` for Metal inference.

**On-device LLM** (`src/llm/`): Llama-architecture models (default: Llama-3.2-3B-Instruct) via candle + Metal. Exposes C-ABI:

- `pocket_llm_create(repo_id, model_file) → *engine` — downloads from HuggingFace Hub
- `pocket_llm_set_prompt(engine, system, user)` — auto-detects chat template (Llama-3 or ChatML), includes multi-turn context
- `pocket_llm_step(engine) → 1=token, 0=done, -1=error` — with repetition penalty + top-p sampling
- `pocket_llm_get_token(engine, buf, size) → len` — get decoded text
- `pocket_llm_set_temperature(engine, temp)`
- `pocket_llm_clear_context(engine)` — clear conversation history
- `pocket_llm_reset(engine)` / `pocket_llm_destroy(engine)`

Key implementation details:

- **Auto-detecting chat template**: Detects Llama-3 (`<|start_header_id|>`) vs ChatML (`<|im_start|>`) from tokenizer vocabulary
- **Multi-turn conversation context**: Stores last 8 turns in `Vec<(String, String)>`, included in prompt formatting
- **Repetition penalty**: 1.15x penalty on last 64 tokens to prevent loops
- **Top-p sampling**: Temperature 0.6 + nucleus sampling (p=0.9) for coherent but varied output

**Sonata LM** (`src/sonata_lm/`): 241M Llama-style transformer predicting semantic tokens at 50 Hz. Exposes C-ABI:

- `sonata_lm_create(weights, config) → *engine` — loads FP16 safetensors onto Metal GPU (~2x throughput)
- `sonata_lm_set_text(engine, token_ids, n)` — set input text tokens
- `sonata_lm_append_text(engine, token_ids, n)` — extend text buffer during streaming (no KV reset)
- `sonata_lm_finish_text(engine)` — signal text complete (re-encodes for cross-attention models)
- `sonata_lm_step(engine, &token) → 0=more, 1=done, -1=error` — one autoregressive step
- `sonata_lm_set_params(engine, temp, top_k, top_p, rep_penalty)` — sampling parameters
- `sonata_lm_load_draft(engine, weights, config)` — load draft model for speculative decoding
- `sonata_lm_speculate_step(engine, out_tokens, max, &count) → 0=more, 1=done` — batch-verified speculative decoding
- `sonata_lm_set_speculate_k(engine, k)` — set speculation depth (default: 5)
- `sonata_lm_set_coarse_grained(engine, enable)` — enable coarse-grained speculative decoding (acoustic similarity groups)
- `sonata_lm_prosody_token_base(engine) → base_id` — returns base token ID for prosody tokens
- `sonata_lm_num_prosody_tokens() → 12` — number of prosody token types
- `sonata_lm_inject_prosody_token(engine, offset)` — inject a prosody control token (STRESS/BREAK/PITCH/RATE/EMPHASIS)
- `sonata_lm_inject_pause(engine, n_frames)` — inject N silence frames (PAD token=0, each frame=20ms) for natural pauses
- `sonata_lm_ms_to_frames(ms) → frames` — convert milliseconds to 50Hz frame count
- `sonata_lm_reset/destroy(engine)`

Key implementation details:

- **FP16 inference**: All weights, KV caches, and RoPE in float16 for ~2x Metal throughput. Logits cast to F32 for sampling.
- **Top-k + top-p sampling**: Combined nucleus sampling (default: k=50, p=0.92, temp=0.8). Uses `select_nth_unstable` partial sort for O(n) top-k selection instead of full O(n log n) sort.
- **Repetition penalty**: 1.15x on last 64 tokens, applied before temperature
- **Batch forward pass**: `forward_seq()` processes multiple tokens at once with causal mask (enables speculative decoding)
- **Speculative decoding**: Draft model generates K tokens → main model verifies in one batch forward → accept/reject with KV cache rollback. Pipeline auto-enables via `sonata_lm_speculate_step` when `--sonata-draft` model is provided, processing multiple tokens per `sonata_step` call.
- **Streaming text append**: `append_text()` extends the text buffer mid-generation for sub-sentence streaming. The pipeline's eager 4-word flush feeds Sonata LM before the full sentence arrives, cutting ~150-300ms off time-to-first-audio (inspired by Liquid AI's interleaved generation). `finish_text()` re-encodes for cross-attention models.
- **Auto-download**: If weights path is a HuggingFace repo ID, auto-downloads via `hf-hub`
- **Architecture**: 16 layers, 1024 d_model, GQA (16 heads, 4 KV), RoPE, SwiGLU FFN

**Sonata Flow** (`src/sonata_flow/`): 35.7M conditional flow-matching network + ConvNeXt decoder. Exposes C-ABI:

- `sonata_flow_create(flow_weights, flow_config, dec_weights, dec_config) → *engine`
- `sonata_flow_generate(engine, semantic_tokens, n_frames, out_mag, out_phase) → n_bins` — with cumulative phase
- `sonata_flow_set_speaker(engine, speaker_id)` — multi-voice speaker conditioning via embedding table
- `sonata_flow_set_speaker_embedding(engine, float*, dim)` — zero-shot voice cloning from raw embedding vector
- `sonata_flow_clear_speaker_embedding(engine)` — clear voice cloning override
- `sonata_flow_set_cfg_scale(engine, scale)` — classifier-free guidance (1.0=off, 1.5=default, 2.0+=strong)
- `sonata_flow_set_n_steps(engine, steps)` — ODE steps (4=fast, 8=default, 16=quality)
- `sonata_flow_set_solver(engine, use_heun)` — 0=Euler (1st-order), 1=Heun (2nd-order, better quality at same step count)
- `sonata_flow_reset_phase(engine)` — reset cumulative phase (call between utterances)
- `sonata_flow_set_emotion(engine, id)` — emotion ID conditioning via embedding table (requires trained weights)
- `sonata_flow_set_emotion_steering(engine, direction, dim, layer_start, layer_end, scale)` — EmoSteer: training-free activation steering for emotion control
- `sonata_flow_clear_emotion_steering(engine)` — clear emotion steering
- `sonata_flow_set_prosody(engine, features, n)` — prosody conditioning (log_pitch, energy, rate) broadcast to flow
- `sonata_flow_set_durations(engine, durations, n_frames)` — per-frame duration conditioning
- `sonata_flow_set_prosody_embedding(engine, embedding, dim)` — reference audio prosody transfer
- `sonata_flow_clear_prosody_embedding(engine)` — clear prosody transfer embedding
- `sonata_flow_interpolate_speakers(engine, emb_a, emb_b, dim, alpha)` — blend two speaker embeddings for style transfer (L2-normalized after interpolation, alpha=0→A, alpha=1→B)
- `sonata_flow_generate_audio(engine, tokens, n, out_audio, max_samples) → n_samples` — direct waveform generation when ConvDecoder is available (bypasses iSTFT)
- `sonata_flow_decoder_type(engine) → int` — 0=ConvNeXt mag/phase (needs iSTFT), 1=ConvDecoder (direct audio)
- `sonata_flow_samples_per_frame(engine) → int` — samples per semantic frame (varies by decoder type)

Key implementation details:

- **FP16 inference**: All weights and intermediate tensors in float16. Timestep frequencies cached at load time (avoiding per-call recomputation). Output cast to F32 for phase accumulation.
- **Heun's 2nd-order ODE solver**: `x_{n+1} = x_n + dt/2 * (v1 + v2)` where v2 is evaluated at the Euler-predicted point. Same quality as 2x Euler steps. Falls back to Euler for last step.
- **Voice cloning**: `set_speaker_embedding()` injects an external speaker embedding vector (from any speaker encoder) instead of the embedding table lookup. Projected through speaker_proj before additive conditioning injection.
- **Classifier-free guidance (CFG)**: Dual forward pass — `v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)`
- **Speaker conditioning**: Embedding table or raw vector → projection → additive injection into conditioning
- **Emotion conditioning**: Parallel to speaker — embedding table or EmoSteer activation steering. Additive injection into conditioning.
- **EmoSteer activation steering**: Training-free emotion control via direction vectors added to transformer activations at specified layers. Supports emotion conversion, interpolation, and composition without retraining.
- **Prosody conditioning**: (log_pitch, energy, rate) → linear projection → additive injection. Pipeline automatically passes SSML prosody to both LM and Flow.
- **Duration conditioning**: Per-frame log-duration → linear projection → additive frame-level conditioning.
- **Reference audio prosody transfer**: External prosody embedding → projection → conditioning injection for "speak it like this" capability. **Note**: `prosody_embedding_override` is stored but not yet wired into `predict_velocity()`/`sample()` — requires adding a `prosody_embedding_proj` layer to the flow model. The API is reserved for when trained weights with this projection become available.
- **Cumulative phase**: inst_freq from decoder accumulated frame-by-frame for phase continuity
- **Streaming chunks**: Pipeline generates in adaptive chunks (12 tokens first, 20-80 subsequent with prosody boundary detection) with crossfade
- **Pipeline-parallel generation**: FlowWorker runs on dedicated pthread — flow+decoder on GPU thread while LM generates next chunk's tokens concurrently. Non-blocking `try_collect` enables the LM to pick up Flow results opportunistically without stalling.
- **Auto-download**: If weights path is a HuggingFace repo ID, auto-downloads via `hf-hub`
- **ConvDecoder (HiFi-GAN style)**: Alternative decoder that generates raw audio waveform directly via ConvTranspose1d upsampling + dilated residual blocks, bypassing iSTFT entirely. Loaded from `sonata_decoder.safetensors`. All weights pre-converted to F32 at load time (`convert_to_f32()`) since Metal conv1d requires F32 — eliminates ~30 per-call `to_dtype` conversions for 2.2x speedup. Metal shader warmup at load time eliminates 137ms first-inference penalty. Pipeline auto-selects via `sonata_flow_decoder_type()`.
- **Architecture**: Euler/Heun ODE, AdaLayerNorm, FlowAttention, ConvNeXt or ConvDecoder

**BNNS ConvNeXt Decoder** (`src/bnns_convnext_decoder.c`): ANE-offloaded alternative to the Rust ConvNeXt decoder. Runs on Apple Neural Engine via BNNS/Accelerate, freeing the GPU for flow network inference. All three compute units run concurrently: GPU (flow) → ANE (decoder) → AMX (iSTFT).

- `bnns_convnext_create(n_layers, dec_dim, conv_kernel, ff_mult, input_dim, n_fft) → *decoder`
- `bnns_convnext_forward(dec, semantic, acoustic, n_frames, out_mag, out_freq) → n_bins`
- `bnns_convnext_load_mlmodelc(dec, path)` — load compiled CoreML model for ANE execution
- All convolutions and linear layers use `cblas_sgemv` (AMX-accelerated), with `vvexpf` for magnitude head

### Sonata Flow v3 and Vocoder Rust FFI

**Flow v3** (`sonata_flow_v3_*`):

- `sonata_flow_v3_create(weights, config) → *engine` / `sonata_flow_v3_destroy(engine)`
- `sonata_flow_v3_generate(engine, phoneme_ids, n_phonemes, out_mel, max_frames) → n_frames`
- `sonata_flow_v3_set_cfg_scale(engine, scale)` / `sonata_flow_v3_set_n_steps(engine, steps)` / `sonata_flow_v3_set_solver(engine, use_heun)`
- `sonata_flow_v3_set_speaker(engine, id)` / `sonata_flow_v3_set_emotion(engine, id)`
- `sonata_flow_v3_stream_start(engine)` / `sonata_flow_v3_stream_chunk(engine, phoneme_ids, n, out_mel, max) → n_out` / `sonata_flow_v3_stream_end(engine)`
- `sonata_flow_v3_set_streaming_mode(engine, mode)` — causal vs Dragon-FM
- `sonata_flow_v3_set_reference(engine, mel, n_frames)` / `sonata_flow_v3_clear_reference(engine)` — zero-shot voice cloning
- `sonata_flow_v3_get_durations(engine, out_durations, max) → n` — predicted phoneme durations

**Vocoder** (`sonata_vocoder_*`):

- `sonata_vocoder_create(weights, config) → *engine` / `sonata_vocoder_destroy(engine)`
- `sonata_vocoder_generate(engine, mel, n_frames, out_audio, max_samples) → n_samples`

### Sonata Flow v3 (text → mel)

Interleaved streaming conditional flow matching for text-to-mel synthesis. Replaces the semantic-token-based pipeline with direct phoneme-conditioned mel generation.

- **Architecture**: InterleavedEncoder (phoneme + mel interleaving) + CausalSlidingWindowAttention + DurationPredictor + CFM blocks
- **Configs**: FlowV3Config (base, 35M) and FlowV3LargeConfig (162M)
- **Conditioning**: Phoneme conditioning (66-token IPA vocab from `g2p.py`), speaker embedding, emotion (TokenLevelEmoSteer)
- **Duration**: Learned duration prediction with MAS (Monotonic Alignment Search) alignment
- **Streaming**: Causal streaming + Dragon-FM (bidirectional attention within chunk for higher quality)
- **Timestep schedule**: EPSS (Exponential Probability Staircase Schedule) for better quality at low step counts
- **Reference mel prompting**: Zero-shot voice cloning via reference mel injection
- **Rust inference**: `sonata_flow_v3_*` FFI functions in `sonata_flow_v3` crate

### Sonata Vocoder (mel → waveform)

BigVGAN-lite neural vocoder converting mel spectrograms to waveform audio.

- **Architecture**: BigVGAN-lite with SnakeAlpha activation + AMPBlock (Anti-Aliased Multi-periodicity)
- **Configs**: VocoderConfig (base) and VocoderLargeConfig
- **Training**: Multi-discriminator — MPD (Multi-Period Discriminator) + MSD (Multi-Scale Discriminator) + MultiResolutionSTFT
- **Losses**: Perceptual mel loss + WavLM perceptual loss (from `perceptual_loss.py`)
- **Rust inference**: `sonata_vocoder_*` FFI functions

### Sonata STT (`src/sonata_stt.c`, `train/sonata/stt.py`)

Pure C CTC speech recognition engine symmetric to Sonata TTS. Two-pass architecture:

**Pass 1 — SonataCTC**: Audio → Mel → SpecAugment → Conformer (RoPE) → CTC → Text

- Base encoder (4L d=256, 6.2M params) — reuses Sonata Codec weights
- Large encoder (12L d=512, 73M params) — standalone STT-optimized
- RoPE positional encoding in self-attention (replaces nn.MultiheadAttention)
- SpecAugment: 2 freq masks (F=15) + 2 time masks (T=50)
- Speed perturbation: 0.9x/1.0x/1.1x random resampling
- CTC vocab: blank(0) + space(1) + a-z(2-27) + '(28) + `<eou>`(29) = 30 tokens
- Optional SentencePiece BPE subword vocab (1024-4096 tokens)

**Pass 2 — SonataRefiner**: Semantic tokens → Transformer encoder → Cross-attention decoder → Text

- 4L encoder (d=512) + 4L decoder (d=512, GQA), 49.3M params
- RoPE in decoder self-attention, cross-attention to encoder output
- Character-level or subword tokenization

**C Inference Engine** (`src/sonata_stt.c`):

- `sonata_stt_create(weights_path) → *engine` — mmap'd weight loading
- `sonata_stt_process(engine, pcm, n, out, max) → chars` — batch transcription
- `sonata_stt_stream_start/feed/flush/end()` — growing-window streaming
- `sonata_stt_set_beam_decoder(engine, beam)` — pluggable CTC beam search + KenLM
- `sonata_stt_process_beam(engine, pcm, n, out, max)` — beam search transcription
- `sonata_stt_eou_peak(engine, window)` — inline EOU probability from CTC logits
- `sonata_stt_eou_probs(engine, out, max)` — per-frame EOU probability extraction
- `sonata_stt_enable_fp16(engine)` — FP16 weight storage for 2x memory bandwidth
- Weak-links to `ctc_beam_decoder` (falls back to greedy if not available)
- Pipeline: `--stt-engine sonata --sonata-stt-model models/sonata/sonata_stt.cstt_sonata`

**Training** (`train/sonata/train_stt.py`):

- `--mode ctc --audio-dir <LibriSpeech>` — train on raw audio with SpecAugment + speed perturb
- `--mode refiner --data <encoded.pt>` — train on semantic tokens
- `--encoder-size large` — use 12L d=512 encoder (73M params)
- Resume support: `--resume <checkpoint.pt>`

**Round-trip verification** (`stt.py:semantic_roundtrip_score()`):

- text → Sonata LM → semantic tokens → Sonata Refiner → text'
- Tests symbolic pipeline without audio, using shared FSQ codebook

**Export**: `scripts/export_sonata_stt.py` — CTC PyTorch → `.cstt_sonata` binary
**Export**: `scripts/export_sonata_refiner.py` — Refiner PyTorch → `.cref` binary
**Eval**: `scripts/eval_wer.py` — WER/CER benchmark on LibriSpeech via C engine (ctypes)

- Handles both RoPE (separate Wq/Wk/Wv) and legacy (fused in_proj) weight layouts

### Voice Activity Projection (`src/vap_model.c`)

Transformer-based turn-taking predictor that replaces rule-based EOU and backchannel timing:

- Architecture: Input projection (160→d_model) → N causal transformer layers → 4 linear heads → sigmoid
- Default config: d_model=128, n_layers=4, n_heads=4, ff_dim=256 (~2M params)
- Input: concatenated user mel [80] + system mel [80] = [160] at 50Hz (20ms per frame)
- Output: 4 predictions per frame:
  - `p_user_speaking` — voice activity projection for user
  - `p_system_turn` — when the system should take a turn
  - `p_backchannel` — backchannel timing appropriateness
  - `p_eou` — end of user utterance
- KV cache for streaming (single frame at a time, ring buffer at context_len)
- RMSNorm + SiLU activation (consistent with Sonata LM/Refiner)
- All matrix ops via `cblas_sgemm`/`cblas_sgemv` (AMX-accelerated)
- EMA smoothing on predictions (configurable alpha, default 0.3)
- Binary weight format: `.vap` (header + float32 weights)
- Pipeline integration: replaces energy-based barge-in, rule-based backchannel timing, and fixed EOU weights
- CLI: `--vap-model PATH`, `--neural-barge-in`

### Audio Mixer (`src/audio_mixer.c`)

4-channel lock-free audio mixer for concurrent output sources:

- Channels: MAIN (TTS), BACKCHANNEL, PRESYNTHESIZED (fast cache), CLOUD_AUDIO (streaming LLM)
- Per-channel SPSC ring buffers (48k samples = 2s at 24kHz)
- Priority-based ducking: lower-priority channels attenuated when higher-priority plays
- Crossfade: per-channel fade state (0→1) for smooth activation/deactivation
- Soft limiter: cubic soft clip, NEON-optimized
- vDSP for gain and summing operations
- Lock-free with `__atomic` head/tail per channel
- Default priorities: MAIN=10, CLOUD=9, PRESYNTHESIZED=8, BACKCHANNEL=5

### Neural Backchannel (`src/neural_backchannel.c`)

TTS-synthesized backchannels that match the current speaker voice and emotion:

- 9 backchannel types: MHM, YEAH, RIGHT, OKAY, UH_HUH, I_SEE, SURE, HMHM, LAUGH
- Text-to-speech synthesis via weak-linked TTS engine callback
- Pre-generation cache: `nbc_warm_cache()` synthesizes all types at startup
- Speaker embedding: voice-matched backchannels via `nbc_set_speaker()`
- Emotion conditioning: emotion-appropriate responses via `nbc_set_emotion()`
- Fallback: pink-noise breath-style synthesis when no TTS engine available
- WAV file override: `nbc_load_wav()` for custom recordings

### Intent Router (`src/intent_router.c`)

Learned response path classifier for minimizing latency:

- Routes: FAST (<50ms), MEDIUM (local LLM), FULL (cloud LLM), BACKCHANNEL
- Neural path: MLP (20→128→64→4) with cblas_sgemv, softmax output
- 20 input features: word count, avg length, question mark, greeting/thanks/bye detection, question words, optional audio/VAP features
- Heuristic fallback: pattern-matching rules when no neural weights loaded
- Fast responses: 7 types (greeting, acknowledge, thinking, yes, no, thanks, goodbye)
- Binary weight format: `.router` (header + float32)
- CLI: `--intent-router PATH`, `--intent-router-default`

### Speculative Generation (`src/speculative_gen.c`)

Multi-draft speculative LLM generation starting before EOU:

- Manages 2 concurrent draft LLM responses
- VAP-informed: uses `p_eou` and `p_system_turn` to decide when to speculate
- Transcript prefix validation: drafts remain valid as long as user speech extends, not diverges
- Commit: best valid draft selected when EOU confirmed (most tokens, highest EOU)
- Cancel: all drafts cancelled when EOU drops below threshold
- Thread-safe: pthread_mutex for cross-thread token feeding
- Pipeline integration: replaces single-shot speculative prefill
- CLI: `--speculative`, `--no-speculative`

### Response Cache (`src/response_cache.c`)

Pre-synthesized audio for instant fast-path responses:

- 7 response types × 3 text variants each (greetings, acknowledgments, etc.)
- Warm at startup via TTS synthesis callback
- Persistent: save/load binary format for cross-launch reuse
- Variant rotation: incrementing counter for natural variety
- Speaker embedding support: voice-matched cache via `response_cache_set_speaker()`
- WAV file addition: `response_cache_add_wav()` for custom recordings
- CLI: `--response-cache PATH`

### Streaming LLM (`src/streaming_llm.c`)

Gemini Live and OpenAI Realtime WebSocket audio backends:

- Gemini Live: `wss://generativelanguage.googleapis.com` with BidiGenerateContent
- OpenAI Realtime: `wss://api.openai.com/v1/realtime`
- Text mode: LLMClient-compatible (send text, peek/consume tokens)
- Audio mode: send PCM audio in, receive PCM audio out (bidirectional streaming)
- Base64: self-contained encode/decode (no external dependency)
- Audio encoding: float32 PCM for Gemini, int16 PCM for OpenAI
- Lock-free audio receive ring buffer
- Server VAD: configurable (use API's VAD or our own VAP/EOU)
- LLMClient adapter: `streaming_llm_as_llm_client()` for drop-in use
- CLI: `--llm gemini-live`, `--llm openai-rt`

### Streaming TTS (`src/streaming_tts.c`)

Continuous speculative TTS with token-level rollback:

- Token-by-token: LLM tokens feed directly to TTS without sentence buffering
- First chunk: eager synthesis after min_tokens_to_start (default: 4 tokens)
- Lookahead buffer: keeps 2 tokens un-synthesized to avoid word boundary issues
- Audio segment tracking: maps token ranges to audio sample ranges
- Rollback: invalidate uncommitted audio from any token index forward
- Crossfade: vDSP-accelerated fade at rollback splice points
- Commit: mark audio as sent-to-speaker (non-rollbackable)
- Stats: latency, rollback count, samples generated/committed/rolled back
- Thread-safe: pthread_mutex for concurrent feed/get
- CLI: `--streaming-tts`, `--no-streaming-tts`

## Build System

The `Makefile` builds in three stages:

1. **C shared libraries** → `build/*.dylib` (44 dylibs + 1 metallib, each `.c` compiles independently)
2. **Rust cdylibs** → `src/{stt,llm,local_llm,sonata_lm,sonata_flow,sonata_storm}/target/release/*.dylib` (via `cargo build --release`)
3. **Pipeline binary** → `sonata` (links all of the above)

Frameworks linked: Accelerate, CoreAudio, AudioToolbox, Security, Metal, Foundation, IOSurface, libcurl. Homebrew deps: curl, opus, onnxruntime, espeak-ng.

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
5. Link via `-lmy_lib` in the `sonata` target

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

Test binaries are compiled into `build/`, so rpath is `@executable_path` (not `@executable_path/build`). The main `sonata` binary lives in the project root, so its rpath is `@executable_path/build`.

## Common Gotchas

1. **Homebrew curl**: macOS system curl may lack features. The Makefile auto-detects `brew --prefix curl`.
2. **ARM64 only**: All compilation targets `-arch arm64`. No x86 support.
3. **Rust crate versions**: The STT Rust crate pins `candle-core`, `candle-nn`, `candle-transformers` to specific Git revisions. Updating requires testing Metal kernel compatibility.
4. **CoreAudio real-time thread**: The `audioCallback` in `pocket_voice.c` runs on a real-time thread. No allocations, no locks, no syscalls allowed.
5. **AMX concurrency**: AMX coprocessor and Metal GPU are separate hardware. Post-processing (AMX) runs concurrently with inference (GPU) without contention.
6. **cJSON is vendored**: The JSON parser is a copy of [cJSON](https://github.com/DaveGamble/cJSON) (MIT license). Don't modify it; update from upstream if needed.
7. **ACCELERATE_NEW_LAPACK**: All files that include `<Accelerate/Accelerate.h>` must define `ACCELERATE_NEW_LAPACK` to suppress deprecation warnings. Guard with `#ifndef` to avoid redefinition when also passed via `-DACCELERATE_NEW_LAPACK` CFLAG.
8. **Test binary rpath**: Test binaries go in `build/` so use `-Wl,-rpath,@executable_path` (they're co-located with the dylibs). The main binary uses `-Wl,-rpath,@executable_path/build`.
9. **Mimi endpointer input**: The LSTM takes 80-dimensional features (mel-energy bands, not raw Mimi latents). The pipeline's `feed_endpointer()` splits 1920 samples (80ms @ 24kHz) into 80 bands and computes per-band RMS.
10. **Speculative prefill thresholds**: Sends at fused_prob >= 0.70, cancels at fused_prob < 0.30. These are tuned for conversational latency; adjust for different use cases.
11. **ONNX Runtime headers**: Homebrew installs to `/opt/homebrew/include/onnxruntime/`. Use `#include <onnxruntime/onnxruntime_c_api.h>` and `-I/opt/homebrew/include` in CFLAGS.
12. **CoreML EP**: Makefile auto-detects Python ONNX Runtime with CoreML EP for ANE acceleration. Falls back to Homebrew ORT (CPU-only) if Python ORT not found. All ONNX modules benefit when CoreML EP is available.
13. **espeak-ng**: Required for Piper TTS phonemization. Install via `brew install espeak-ng`. Headers at `/opt/homebrew/include/espeak-ng/`, lib at `/opt/homebrew/lib/`.
14. **Piper model format**: Expects `.onnx` model + matching `.onnx.json` config containing phoneme_id_map, sample_rate, and inference scales.
15. **LLM Rust edition**: The `src/llm` crate uses Rust 2024 edition which requires `#[unsafe(no_mangle)]` instead of `#[no_mangle]`.

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

**Proven results** (historical — test target removed):

- Piper TTS (22kHz) → resample → Conformer STT 1.1B + KenLM 3-gram
- **0% WER** across all 5 test sentences (hello, conversational, pangram, counting, weather)
- Total pipeline: TTS ~50ms + STT ~600ms per sentence

**Full loop** (historical — test target removed):

- STT → LLM → TTS → STT: 13/13 tests passing
- LLM generates at 43.5 tok/s on Metal
- End-to-end round-trip WER: 18.8% (includes LLM response variation)

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
- Eager sub-sentence flush (4 words) for Sonata streaming text append

**System-Level:**

- CoreAudio real-time thread for audio I/O (no malloc, no locks)
- Metal GPU for inference concurrent with AMX for post-processing
- Hardware AudioConverter for sample rate conversion
- **CoreML EP**: Makefile auto-detects Python ONNX Runtime with CoreML EP; falls back to Homebrew ORT (CPU-only) if not found. ONNX modules (Speaker Encoder) benefit. CoreML accelerates graph nodes on Apple Neural Engine.
- **Streaming chunked generation**: Sonata generates in adaptive chunks (12 first / 20-80 subsequent tokens) with crossfade
- **CTC beam search + KenLM**: 3-gram language model rescoring for STT (beam_size=16, lm_weight=1.5)

### On-Device TTS Engines (Historical)

> **Note**: Piper and Supertonic source files have been removed from the codebase. Sonata is the active TTS engine.

**Piper VITS** (removed):

- 100% Whisper-verified intelligibility, 90% on-device Conformer STT
- 0.015x RTF (67x faster than real-time), ~51ms per sentence
- 22.05kHz output, en_US-amy-medium voice (60MB ONNX)
- Phonemization via espeak-ng → VITS neural vocoder
- **Status: Production ready**

**Supertonic-2 Flow-Matching** (removed):

- 90% Whisper-verified (M2/M4 voices), 80% Conformer STT
- 0.021x RTF, ~80ms per sentence
- 44.1kHz studio-quality output, 10 voice styles (5M/5F)
- Pipeline: text_encoder → duration_predictor → vector_estimator (5 flow steps) → vocoder
- **Critical**: noise_scale must be 1.0 (standard Gaussian), not reduced
- **Status: Excellent for M2/M4 voices, variable for others**

**On-Device LLM** (`src/llm/`):

- Llama-3.2-3B-Instruct via candle + Metal GPU (default)
- Auto-detecting chat template (Llama-3 or ChatML)
- BF16 inference, multi-turn conversation context (8 turns)
- Repetition penalty (1.15) + top-p sampling (0.9) for quality output
- **Status: Production ready. Full STT→LLM→TTS→STT loop verified.**

### TTS Performance Benchmarks

All measured on Apple Silicon M-series (single core, no GPU):

| Engine                            | RTF   | Speed        | Audio Quality               |
| --------------------------------- | ----- | ------------ | --------------------------- |
| **Piper VITS** (en_US-amy-medium) | 0.029 | 34x realtime | High (VITS, 22kHz)          |
| **Supertonic-2** (5 flow steps)   | 0.028 | 36x realtime | High (flow-matching, 44kHz) |
| Supertonic-2 (2 flow steps)       | 0.020 | 50x realtime | Good                        |
| Supertonic-2 (10 flow steps)      | 0.046 | 22x realtime | Excellent                   |

Run benchmarks: `make bench-sonata`

### Native VAD Benchmark Results

Native C VAD vs Silero ONNX (CoreML EP), 1000 × 32ms chunks:

| Engine                 | Total   | Per Chunk | RTF     | Speed         |
| ---------------------- | ------- | --------- | ------- | ------------- |
| **Native C (AMX)**     | 36.1 ms | 36.1 µs   | 0.00113 | 885x realtime |
| **Silero ONNX+CoreML** | 60.1 ms | 60.1 µs   | 0.00188 | 532x realtime |

- **1.7x speedup** (native over ONNX with CoreML EP)
- **MAE 0.0012** between native and ONNX probability outputs on real audio
- Both engines agree on silence vs speech classification
- No ONNX Runtime dependency, no session overhead, real-time thread safe

Run: `make bench-vad`

### Moonshine STT Benchmark Results (Historical)

> **Note**: Moonshine STT source files have been removed. Benchmark data retained for reference.

Moonshine Tiny (27M) with CoreML EP:

- hello_piper.wav (9.6s): 497ms latency, 0.052 RTF (19x realtime)
- cute_piper.wav (13.9s): 777ms latency, 0.056 RTF (18x realtime)
- Streaming: 93ms time-to-first-result, 405ms per-flush

### Sonata TTS Benchmark Results

Measured with `make bench-sonata` on Apple Silicon:

| Component                     | Metric                  | Value                               |
| ----------------------------- | ----------------------- | ----------------------------------- |
| **Sonata LM** (241M F16)      | Throughput              | 43-46 tok/s                         |
|                               | TTFT                    | 7-26 ms                             |
|                               | RTF                     | ~1.1x (needs 50 tok/s for realtime) |
| **Sonata Flow + ConvDecoder** | 12 frames (first chunk) | 135ms, RTF 0.27 (4x RT)             |
|                               | 50 frames (cached)      | 130ms, RTF 0.13 (8x RT)             |
|                               | 100 frames              | 275ms, RTF 0.14 (7x RT)             |
| **iSTFT**                     | 1000 frames (20s)       | 1.9ms, RTF 0.0001 (10,000x RT)      |
| **End-to-End**                | First chunk (12 tokens) | **543-600ms**                       |
|                               | Overall RTF             | 1.3-1.6x                            |

Key optimizations applied:

- **ConvDecoder F32 weight caching**: 2.2x Flow speedup (weights pre-converted at load)
- **ConvDecoder Metal warmup**: 137ms shader compilation moved to load time
- **Smaller first chunk**: 12 tokens (240ms audio) for lower TTFA
- **Non-blocking FlowWorker**: LM continues generating while Flow processes previous chunk
- **Self-speculative decoding TESTED**: 4-layer draft on same GPU = **2x slower** (22 tok/s) — not recommended on single Metal GPU

Run: `make bench-sonata`

### Sonata Audio Quality Status

**Current**: Models are NOT trained — audio is noise-like, not speech. Both Python and
Rust inference produce identical noise characteristics, confirming the inference pipeline
is correct but the weights need training.

Quality validation tests:

- `make test-sonata-quality` — Generates WAVs, audio stats, round-trip WER via Conformer STT
- `python scripts/cross_validate_sonata.py` — Python vs Rust numerical comparison
- Full report: `bench_output/SONATA_QUALITY_REPORT.md`

Quality targets (when models are trained):

- Round-trip WER: < 5% (human-level intelligibility)
- Zero-crossing rate: 0.01-0.30 (speech range)
- LM EOS generation: variable duration per sentence (not fixed max tokens)
- Prosody MOS prediction: > 3.0 (natural-sounding)

### CLI Quick Reference (TTS + STT)

```bash
# Full-duplex with VAP turn-taking
./sonata --vap-model models/vap.vap --neural-barge-in

# Intent routing with response cache
./sonata --intent-router-default --response-cache models/response_cache.bin

# Streaming TTS (token-level, no sentence buffering)
./sonata --streaming-tts

# Gemini Live audio streaming
GEMINI_API_KEY=... ./sonata --llm gemini-live

# OpenAI Realtime audio streaming
OPENAI_API_KEY=... ./sonata --llm openai-rt

# Full SOTA pipeline (all features)
./sonata --vap-model models/vap.vap --intent-router-default \
  --response-cache models/response_cache.bin --streaming-tts --speculative

# Train VAP model
python train/sonata/train_vap.py --synthetic 500 --epochs 10 --export --output models/vap.vap

# Train intent router
python train/sonata/train_intent_router.py --synthetic --output models/intent.router

# Native C VAD (recommended — no ONNX dependency, AMX-accelerated)
./sonata --vad models/silero_vad.nvad

# Extract native VAD weights from Silero ONNX model (one-time)
python scripts/extract_silero_weights.py models/silero_vad.onnx models/silero_vad.nvad

# WebSocket streaming (start server, connect via ws://host:port/v1/stream)
./sonata --server --server-port 8080
# Then: wscat -c ws://localhost:8080/v1/stream

# Sonata TTS (default)
./sonata --tts-engine sonata

# Flow v3 TTS (text → mel → waveform)
./sonata --tts-engine sonata-v3 --flow-v3-weights models/sonata/flow_v3.safetensors \
  --flow-v3-config models/sonata/flow_v3_config.json \
  --vocoder-weights models/sonata/vocoder.safetensors \
  --vocoder-config models/sonata/vocoder_config.json

# Flow v3 with phonemes
./sonata --tts-engine sonata-v3 --phonemize --phoneme-map models/sonata/phoneme_map.json

# Piper TTS (removed — source files no longer in codebase)
# ./sonata --tts-engine piper --piper-model models/piper/en_US-amy-medium.onnx

# Supertonic-2 TTS (removed — source files no longer in codebase)
# ./sonata --tts-engine supertonic --flow-steps 10 --supertonic-voice models/supertonic-2/voice_styles/F1.json

# Conformer STT with beam search + KenLM
./sonata --stt-engine conformer --cstt-model models/parakeet-ctc-1.1b-fp16.cstt \
  --beam-size 16 --lm-path models/3-gram.pruned.1e-7.bin --lm-weight 1.5

# Local LLM (no API key needed)
./sonata --llm-engine local --llm-model meta-llama/Llama-3.2-3B-Instruct

# Moonshine STT (removed — source files no longer in codebase)
# ./sonata --stt-engine moonshine --moonshine-model models/moonshine/

# Voice cloning from reference audio
./sonata --speaker-encoder models/ecapa_tdnn.onnx --ref-wav voice.wav

# Phonemization for improved pronunciation
./sonata --phonemize --phoneme-map models/sonata/phoneme_map.json

# EmoSteer emotion direction vectors (training-free emotion control)
./sonata --emosteer models/sonata/emosteer_directions.json

# Prosody logging for dashboard visualization
./sonata --prosody-log prosody.jsonl
# Then open tools/prosody_dashboard.html and drop the JSONL file

# Compute EmoSteer direction vectors from labeled data
cd train/sonata && python compute_emosteer.py \
  --flow-weights models/sonata/flow.safetensors \
  --flow-config models/sonata/sonata_flow_config.json \
  --data-dir data/emotional_pairs/ \
  --output models/sonata/emosteer_directions.json

# A/B prosody evaluation (synthetic test data)
cd train/sonata && python eval_prosody_ab.py --synthetic --output report.json

# A/B prosody evaluation (real audio)
cd train/sonata && python eval_prosody_ab.py \
  --baseline-wav baseline/ --enhanced-wav enhanced/ \
  --sentences sentences.txt --output report.json

# One-stop voice cloning (auto-detects speaker encoder)
./sonata --clone-voice reference_voice.wav

# Benchmark against commercial TTS APIs (requires API keys in env)
python train/sonata/eval_prosody_ab.py --benchmark \
  --benchmark-engines sonata,openai,elevenlabs,google

# Download training data automatically
bash train/sonata/train_all.sh --download-librispeech --download-emov

# Full training pipeline with auto-export
bash train/sonata/train_all.sh \
  --data-dir data/librispeech-dev/ --emotion-dir data/emov-db/ \
  --distill --export --device mps

# Prepare prosody data from EmoV-DB
cd train/sonata && python prepare_prosody_data.py \
  --dataset emov-db --data-dir ~/data/EmoV-DB/ --output data/prosody/

# Full training pipeline (codec → encode → LM → flow → emosteer)
cd train/sonata && bash train_all.sh \
  --data-dir data/librispeech/ --emotion-dir data/EmoV-DB/ --device mps

# Flow training with emotion/prosody/duration conditioning
cd train/sonata && python train_flow.py \
  --data-dir data/sonata_pairs/ \
  --use-emotion --use-prosody --use-duration --device mps

# Flow distillation training (8→1 step)
cd train/sonata && python train_flow_distill.py \
  --teacher models/sonata/flow.safetensors \
  --teacher-config models/sonata/sonata_flow_config.json \
  --data-dir data/sonata_pairs/ --device mps

# Flow v3 training
cd train/sonata && python train_flow_v3.py --manifest ../data/libritts_r_full_manifest_phonemes.jsonl --phonemes --model-size large

# Vocoder training
cd train/sonata && python train_vocoder.py --data-dir ../data/libritts-r

# Flow v3 + Vocoder joint fine-tuning
cd train/sonata && python train_joint_v3.py --flow-checkpoint ckpt.pt --vocoder voc.pt --manifest manifest.jsonl

# Flow v3 distillation (8→1 step)
cd train/sonata && python train_distill_v3.py --teacher ckpt.pt --data-dir ../data/libritts-r

# Synthesize from checkpoints
cd train/sonata && python synthesize.py --checkpoint ckpt.pt --vocoder voc.pt --phonemes --text "Hello world"

# TTS evaluation (PESQ, STOI, WER, UTMOS)
cd train/sonata && python eval_tts.py --mode synthesize --checkpoint ckpt.pt --vocoder voc.pt --utmos

# Unified evaluation pipeline
./scripts/eval_all.sh --checkpoint ckpt.pt --vocoder voc.pt --output report.md

# Export Sonata LM to GGUF (quantized inference, broader deployment)
python scripts/export_sonata_gguf.py \
  --weights models/sonata/sonata_lm.safetensors \
  --config models/sonata/sonata_lm_config.json \
  --output models/sonata/sonata_lm-q8_0.gguf --dtype q8_0
```

### `http_api.c` (REST API Server)

Multi-threaded HTTP server for programmatic access:

- **Thread pool**: 4 worker threads + 1 accept thread, mutex+condvar connection queue for concurrent request handling
- **API key auth**: `SONATA_API_KEY` env var or `http_api_set_api_key()`. Bearer token checked on all endpoints except `/health`. If unset, open access.
- `POST /v1/audio/transcriptions`: Send WAV audio → get JSON transcript
- `POST /v1/audio/speech`: Send text/JSON → audio (WAV/raw/MP3/Opus)
- `POST /v1/voices`: Voice cloning — send WAV → get `voice_id` for TTS
- `GET /v1/voices`: List cloned voices
- `POST /v1/chat`: Send message → get JSON response
- `GET /v1/stream`: WebSocket upgrade — binary audio in → JSON events + binary audio out
- `GET /health`: Health check endpoint
- **OpenAI TTS API compatibility**: Accepts `{model, input, voice, response_format, speed}` format as drop-in replacement for OpenAI's `/v1/audio/speech`
- **Output containers**: WAV, raw PCM, MP3 (via LAME), Opus (via libpocket_opus)
- **Output encodings**: pcm_s16le, pcm_f32le, pcm_mulaw (G.711), pcm_alaw (G.711)
- **Word-level timestamps**: `"word_timestamps": true` returns JSON with estimated word boundaries
- **Streaming TTS**: `"stream": true` sends audio via HTTP chunked Transfer-Encoding as it's generated
- **Voice cloning**: In-memory voice registry (32 slots). `POST /v1/voices` with WAV body → speaker encoder → embedding stored → `voice_id` returned. Use `voice_id` in TTS `voice` field.
- **Pronunciation overrides**: Inline per-request `pronunciation_overrides` array in JSON body
- CORS headers for browser access
- WAV parsing/generation for 16-bit and 32-bit float
- 30s request timeout, 16 MB max request body

### JSON Config File Support

`--config config.json` loads a JSON configuration file using cJSON. All sections are optional:

```json
{
  "stt": { "engine": "conformer", "beam_size": 5, "lm_path": "..." },
  "tts": { "engine": "sonata" },
  "sonata": { "lm_weights": "...", "heun": true, "speculate_k": 5 },
  "llm": { "engine": "claude", "model": "claude-sonnet-4-20250514" },
  "audio": {
    "pitch": 1.0,
    "spatial": false,
    "vad": "models/silero_vad.nvad"
  },
  "server": { "enabled": true, "port": 8080 },
  "profiler": true
}
```

CLI arguments always override config file values (two-pass loading).

### Turn Latency Instrumentation

Every turn prints a detailed latency breakdown:

```
┌─── Turn Latency Breakdown ───────────────────┐
│ Speech duration:     1200.0 ms               │
│ STT inference:        180.0 ms (12 frames)   │
│ LLM TTFT:             250.0 ms               │
│ TTS first audio:      120.0 ms               │
│ ═══════════════════════════════════════════ │
│ Voice Response Lat:   450.0 ms  ◄── KEY      │
│ TTS RTF:              0.150 (6x realtime)    │
│ LLM throughput:       85.0 tok/s             │
└───────────────────────────────────────────────┘
```

The **Voice Response Latency** (VRL) is the key metric: time from end-of-speech to first TTS audio output. Target: < 500ms.

### Metal GPU Warmup

Both Sonata LM and Sonata Flow run a dummy forward pass at load time to compile Metal shader kernels. This eliminates the ~200-500ms first-inference penalty that would otherwise hit the first user utterance.

### Future Opportunities

1. **Conformer on ANE**: Export trained Conformer to CoreML/BNNS for Neural Engine inference
2. **Learned EOU weights**: Train the fused EOU weights on real conversation data
3. **Multi-turn context**: Cache Conformer encoder states across turns for faster re-engagement
4. **Voice pool for emotion**: Pre-record WAV voice prompts for each emotion (warm, serious, excited, etc.). Pipeline maps `SSMLSegment.emotion` to voice selection via Rust TTS voice cloning. Requires engine pool or fast destroy/recreate.
