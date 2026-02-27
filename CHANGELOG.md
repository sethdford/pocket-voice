# Changelog

All notable changes to Sonata are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-02-26

Initial public release of Sonata.

### Added

- **Sonata TTS pipeline**: 3-stage neural TTS (Sonata LM 241M + Flow 35.7M + iSTFT decoder)
  - Streaming chunked generation with adaptive chunk sizes
  - Sub-sentence streaming via eager 4-word flush
  - ConvNeXt and ConvDecoder (HiFi-GAN style) decoder options
  - FP16 Metal GPU inference
  - Speculative decoding support
- **25+ native C modules**: conformer_stt, mel_spectrogram, fused_eou, vdsp_prosody, breath_synthesis, lufs, noise_gate, native_vad, speech_detector, ssml_parser, text_normalize, sentence_buffer, prosody_predict, emphasis_predict, voice_onboard, phonemizer, speaker_encoder, speaker_diarizer, audio_emotion, backchannel, conversation_memory, spatial_audio, opus_codec, vm_ring, and more
- **3-signal fused EOU detection**: Energy VAD + LSTM endpointer + ASR-inline EOU token
- **Pure C STT engines**: FastConformer CTC with RoPE, beam search + KenLM, Sonata STT + Refiner
- **Rust cdylib crates**: pocket_stt (Kyutai/candle), pocket_llm (Llama-3.2/candle), sonata_lm, sonata_flow, sonata_storm
- **HTTP API server**: OpenAI-compatible REST endpoints for STT, TTS, chat, and voice cloning
- **WebSocket streaming**: Real-time bidirectional audio via RFC 6455
- **Multi-LLM support**: Claude (Anthropic), Gemini (Google), and on-device Llama-3.2-3B
- **Prosody system**: SSML parser, auto-intonation, emphasis prediction, emotion detection, conversational adaptation, EmoSteer activation steering
- **Voice cloning**: Zero-shot via ONNX speaker encoder + Flow conditioning
- **Audio post-processing**: Pitch shift, formant EQ, LUFS normalization, spatial audio, soft limiting, breath synthesis
- **Native VAD**: Pure C Silero VAD reimplementation (AMX-accelerated, 885x realtime, no ONNX dependency)
- **Quality framework**: Native C WER/CER, MCD, STOI, SNR, F0, speaker similarity, latency harness
- **Apple Silicon optimization**: Metal GPU, AMX coprocessor, ARM NEON SIMD, ANE via BNNS/CoreML
- **Full test suite**: 30+ test targets, 800+ tests
- **GitHub Pages site**: Apple-inspired documentation site
- **Training pipeline**: PyTorch/MPS training for Codec, LM, Flow, Flow v3, Vocoder, STT with distillation support
- **Comprehensive benchmarking**: STT WER/RTF, TTS quality, VAD comparison, industry benchmarks

### Architecture

- Pipeline state machine: Listening → Recording → Processing → Streaming → Speaking
- CoreAudio VoiceProcessingIO with hardware AEC for full-duplex audio
- Lock-free ring buffers (SPSC, SPMC, VM-mirrored) with cache-line aligned atomics
- Bump-pointer arena allocator for zero per-turn allocation overhead
- All four Apple Silicon compute units (GPU + AMX + NEON + ANE) running concurrently
