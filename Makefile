# Makefile — Build Sonata: real-time voice pipeline for Apple Silicon.
#
# Targets:
#   all              Build everything
#   sonata           Standalone pipeline binary (mic → STT → Claude → TTS → speaker)
#   libs             Build all shared libraries
#   clean            Remove build artifacts
#
# Usage:
#   make                 # Build all
#   make sonata          # Build just the binary
#   make clean           # Clean everything
#   ANTHROPIC_API_KEY=sk-... ./sonata  # Run

CC      := cc
CFLAGS  := -O3 -flto -mcpu=native -arch arm64 -Wall -Wextra -Wno-unused-parameter \
            -D_FORTIFY_SOURCE=2 -fstack-protector-strong -Wformat-security
LDFLAGS := -flto -arch arm64

FRAMEWORKS := -framework Accelerate -framework CoreAudio -framework AudioToolbox

# Detect Homebrew curl (macOS system curl may lack features)
CURL_PREFIX := $(shell brew --prefix curl 2>/dev/null || echo "")
ifneq ($(CURL_PREFIX),)
    CURL_CFLAGS := -I$(CURL_PREFIX)/include
    CURL_LDFLAGS := -L$(CURL_PREFIX)/lib -lcurl
else
    CURL_CFLAGS :=
    CURL_LDFLAGS := -lcurl
endif

# Piper TTS: ONNX Runtime + espeak-ng (brew install onnxruntime espeak-ng)
# Phonemizer: espeak-ng only (brew install espeak-ng)
HOMEBREW_PREFIX := $(shell brew --prefix 2>/dev/null || echo "/opt/homebrew")

# Use Python ORT which has CoreML EP (homebrew ORT is CPU-only)
ORT_LIB := $(shell ls .venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.*.dylib 2>/dev/null | head -1)
ORT_DIR := $(shell dirname $(ORT_LIB) 2>/dev/null)
# When ORT_DIR set, add -Wl,-rpath, for Python ORT; else empty (avoid comma in $(if) - use var)
ORT_RPATH_VAL := -Wl,-rpath,$(ORT_DIR)
ORT_RPATH := $(if $(ORT_DIR),$(ORT_RPATH_VAL))
ifneq ($(ORT_LIB),)
  # Link against versioned dylib by path (libonnxruntime.1.x.dylib has no libonnxruntime.dylib symlink)
  ONNX_LDFLAGS = $(ORT_LIB) -Wl,-rpath,$(ORT_DIR)
  ONNX_CFLAGS = -I$(HOMEBREW_PREFIX)/include
else
  ONNX_LDFLAGS = -L$(HOMEBREW_PREFIX)/lib -lonnxruntime -Wl,-rpath,$(HOMEBREW_PREFIX)/lib
  ONNX_CFLAGS = -I$(HOMEBREW_PREFIX)/include
endif

PIPER_CFLAGS := -I$(HOMEBREW_PREFIX)/include
PIPER_LDFLAGS := $(ONNX_LDFLAGS) -L$(HOMEBREW_PREFIX)/lib -lespeak-ng -Wl,-rpath,$(HOMEBREW_PREFIX)/lib
PHONEMIZER_LDFLAGS := -L$(HOMEBREW_PREFIX)/lib -lespeak-ng

# Detect Homebrew opus for optional Opus codec
OPUS_PREFIX := $(shell brew --prefix opus 2>/dev/null || echo "")
ifneq ($(OPUS_PREFIX),)
    OPUS_CFLAGS := -I$(OPUS_PREFIX)/include
    OPUS_LDFLAGS := -L$(OPUS_PREFIX)/lib
else
    OPUS_CFLAGS :=
    OPUS_LDFLAGS :=
endif

# Output directories
BUILD := build

# Rust release output directories
STT_DYLIB   := target/release/libpocket_stt.dylib

.PHONY: all clean libs sonata pocket-voice

all: libs sonata

# ─── Shared Libraries ──────────────────────────────────────────────────────

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/libpocket_voice.dylib: src/pocket_voice.c src/neon_audio.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC $(FRAMEWORKS) \
	  -install_name @rpath/libpocket_voice.dylib -o $@ src/pocket_voice.c

$(BUILD)/libvdsp_prosody.dylib: src/vdsp_prosody.c | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -framework Accelerate \
	  -install_name @rpath/libvdsp_prosody.dylib -o $@ $<

$(BUILD)/libaudio_converter.dylib: src/audio_converter.c | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -framework AudioToolbox -framework CoreFoundation \
	  -install_name @rpath/libaudio_converter.dylib -o $@ $<

$(BUILD)/libspatial_audio.dylib: src/spatial_audio.c | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -framework AudioToolbox -framework Accelerate \
	  -install_name @rpath/libspatial_audio.dylib -o $@ $<

$(BUILD)/libtext_normalize.dylib: src/text_normalize.c src/text_normalize.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC \
	  -install_name @rpath/libtext_normalize.dylib -o $@ src/text_normalize.c

$(BUILD)/libsentence_buffer.dylib: src/sentence_buffer.c src/sentence_buffer.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC \
	  -install_name @rpath/libsentence_buffer.dylib -o $@ src/sentence_buffer.c

$(BUILD)/libprosody_predict.dylib: src/prosody_predict.c src/prosody_predict.h $(BUILD)/cJSON.o | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC \
	  -install_name @rpath/libprosody_predict.dylib -o $@ src/prosody_predict.c $(BUILD)/cJSON.o

$(BUILD)/libprosody_log.dylib: src/prosody_log.c src/prosody_log.h $(BUILD)/cJSON.o | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC \
	  -install_name @rpath/libprosody_log.dylib -o $@ src/prosody_log.c $(BUILD)/cJSON.o

$(BUILD)/libemphasis_predict.dylib: src/emphasis_predict.c src/emphasis_predict.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC \
	  -install_name @rpath/libemphasis_predict.dylib -o $@ src/emphasis_predict.c

$(BUILD)/libvoice_onboard.dylib: src/voice_onboard.c src/voice_onboard.h src/speaker_encoder.h \
                                 $(BUILD)/libspeaker_encoder.dylib | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -framework Accelerate \
	  -Lbuild -lspeaker_encoder -Wl,-rpath,@loader_path \
	  -install_name @rpath/libvoice_onboard.dylib -o $@ src/voice_onboard.c

$(BUILD)/libssml_parser.dylib: src/ssml_parser.c src/ssml_parser.h src/text_normalize.h \
                               $(BUILD)/libtext_normalize.dylib | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC \
	  -L$(BUILD) -ltext_normalize \
	  -Wl,-rpath,@loader_path \
	  -install_name @rpath/libssml_parser.dylib -o $@ src/ssml_parser.c

$(BUILD)/libpocket_opus.dylib: src/opus_codec.c | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC $(OPUS_CFLAGS) $(OPUS_LDFLAGS) -lopus \
	  -install_name @rpath/libpocket_opus.dylib -o $@ $<

$(BUILD)/libbreath_synthesis.dylib: src/breath_synthesis.c src/breath_synthesis.h src/neon_audio.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -framework Accelerate \
	  -install_name @rpath/libbreath_synthesis.dylib -o $@ src/breath_synthesis.c

$(BUILD)/liblufs.dylib: src/lufs.c src/lufs.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -framework Accelerate \
	  -install_name @rpath/liblufs.dylib -o $@ src/lufs.c

$(BUILD)/libvm_ring.dylib: src/vm_ring.c src/vm_ring.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC \
	  -install_name @rpath/libvm_ring.dylib -o $@ src/vm_ring.c

$(BUILD)/libmel_spectrogram.dylib: src/mel_spectrogram.c src/mel_spectrogram.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libmel_spectrogram.dylib -o $@ src/mel_spectrogram.c

$(BUILD)/libctc_beam_decoder.dylib: src/ctc_beam_decoder.cpp src/ctc_beam_decoder.h | $(BUILD)
	@if [ -f third_party/kenlm/libkenlm.a ]; then \
	  echo "Building CTC beam decoder with KenLM support"; \
	  c++ -std=c++17 -O3 -arch arm64 -Wall -Wextra -Wno-unused-parameter \
	    -shared -fPIC -DUSE_KENLM -DKENLM_MAX_ORDER=6 \
	    -I src -I third_party/kenlm \
	    -install_name @rpath/libctc_beam_decoder.dylib \
	    -o $@ src/ctc_beam_decoder.cpp \
	    third_party/kenlm/libkenlm.a; \
	else \
	  echo "Building CTC beam decoder without KenLM"; \
	  c++ -std=c++17 -O3 -arch arm64 -Wall -Wextra -Wno-unused-parameter \
	    -shared -fPIC -I src \
	    -install_name @rpath/libctc_beam_decoder.dylib \
	    -o $@ src/ctc_beam_decoder.cpp; \
	fi

$(BUILD)/libtdt_decoder.dylib: src/tdt_decoder.c src/tdt_decoder.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libtdt_decoder.dylib -o $@ src/tdt_decoder.c

$(BUILD)/libvoice_quality.dylib: src/voice_quality.c src/voice_quality.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libvoice_quality.dylib -o $@ src/voice_quality.c

$(BUILD)/liblatency_profiler.dylib: src/latency_profiler.c src/latency_profiler.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC \
	  -install_name @rpath/liblatency_profiler.dylib -o $@ src/latency_profiler.c

$(BUILD)/libbnns_conformer.dylib: src/bnns_conformer.c src/bnns_conformer.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libbnns_conformer.dylib -o $@ src/bnns_conformer.c

$(BUILD)/libbnns_convnext_decoder.dylib: src/bnns_convnext_decoder.c src/bnns_convnext_decoder.h $(BUILD)/cJSON.o | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -install_name @rpath/libbnns_convnext_decoder.dylib -o $@ src/bnns_convnext_decoder.c $(BUILD)/cJSON.o

$(BUILD)/libwebsocket.dylib: src/websocket.c src/websocket.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -framework Security \
	  -install_name @rpath/libwebsocket.dylib -o $@ src/websocket.c

$(BUILD)/libhttp_api.dylib: src/http_api.c src/http_api.h src/cJSON.h \
                            $(BUILD)/libwebsocket.dylib $(BUILD)/libpocket_opus.dylib \
                            $(BUILD)/libaudio_converter.dylib $(BUILD)/cJSON.o | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -L$(BUILD) -lwebsocket -lpocket_opus -laudio_converter \
	  $(BUILD)/cJSON.o \
	  -Wl,-rpath,@loader_path \
	  -install_name @rpath/libhttp_api.dylib -o $@ src/http_api.c

$(BUILD)/libmetal_loader.dylib: src/metal_loader.c src/metal_loader.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -x objective-c -framework Metal -framework Foundation \
	  -install_name @rpath/libmetal_loader.dylib -o $@ src/metal_loader.c

$(BUILD)/libmetal_dispatch.dylib: src/metal_dispatch.c src/metal_dispatch.h \
                                 $(BUILD)/libmetal_loader.dylib $(BUILD)/libapple_perf.dylib | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -x objective-c -framework Metal -framework Foundation -framework Accelerate \
	  -L$(BUILD) -lmetal_loader -lapple_perf -Wl,-rpath,@loader_path \
	  -install_name @rpath/libmetal_dispatch.dylib -o $@ src/metal_dispatch.c

$(BUILD)/libconformer_stt.dylib: src/conformer_stt.c src/conformer_stt.h \
                                  $(BUILD)/libmel_spectrogram.dylib \
                                  $(BUILD)/libctc_beam_decoder.dylib \
                                  $(BUILD)/libtdt_decoder.dylib | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -L$(BUILD) -lmel_spectrogram -lctc_beam_decoder -ltdt_decoder \
	  -Wl,-rpath,@loader_path \
	  -install_name @rpath/libconformer_stt.dylib -o $@ src/conformer_stt.c

$(BUILD)/libspm_tokenizer.dylib: src/spm_tokenizer.c src/spm_tokenizer.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC \
	  -install_name @rpath/libspm_tokenizer.dylib -o $@ src/spm_tokenizer.c


$(BUILD)/libphonemizer.dylib: src/phonemizer.c src/phonemizer.h src/cJSON.c src/cJSON.h | $(BUILD)
	$(CC) $(CFLAGS) $(PIPER_CFLAGS) -Isrc -shared -fPIC \
	  $(PHONEMIZER_LDFLAGS) -Wl,-rpath,$(HOMEBREW_PREFIX)/lib \
	  -install_name @rpath/libphonemizer.dylib \
	  -o $@ src/phonemizer.c src/cJSON.c



$(BUILD)/libspeaker_encoder.dylib: src/speaker_encoder.c src/speaker_encoder.h $(SONATA_SPEAKER_DYLIB) | $(BUILD)
	cp -f $(SONATA_SPEAKER_DYLIB) $(BUILD)/libsonata_speaker.dylib
	$(CC) $(CFLAGS) -Isrc -shared -fPIC -framework Accelerate \
	  -L$(BUILD) -lsonata_speaker \
	  -Wl,-rpath,@loader_path \
	  -install_name @rpath/libspeaker_encoder.dylib \
	  -o $@ src/speaker_encoder.c

$(BUILD)/libspeaker_diarizer.dylib: src/speaker_diarizer.c src/speaker_diarizer.h \
                                     $(BUILD)/libspeaker_encoder.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -shared -fPIC -framework Accelerate \
	  -L$(BUILD) -lspeaker_encoder -Wl,-rpath,@loader_path \
	  -install_name @rpath/libspeaker_diarizer.dylib \
	  -o $@ src/speaker_diarizer.c

$(BUILD)/libnative_vad.dylib: src/native_vad.c src/native_vad.h src/lstm_ops.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libnative_vad.dylib -o $@ src/native_vad.c

$(BUILD)/libspeech_detector.dylib: src/speech_detector.c src/speech_detector.h \
                                    $(BUILD)/libnative_vad.dylib \
                                    $(BUILD)/libmimi_endpointer.dylib \
                                    $(BUILD)/libfused_eou.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -L$(BUILD) -lnative_vad -lmimi_endpointer -lfused_eou \
	  -install_name @rpath/libspeech_detector.dylib -o $@ src/speech_detector.c


$(BUILD)/libsonata_stt.dylib: src/sonata_stt.c src/sonata_stt.h $(BUILD)/libmel_spectrogram.dylib | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lmel_spectrogram \
	  -install_name @rpath/libsonata_stt.dylib -o $@ src/sonata_stt.c

$(BUILD)/libsonata_refiner.dylib: src/sonata_refiner.c src/sonata_refiner.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -install_name @rpath/libsonata_refiner.dylib -o $@ src/sonata_refiner.c

$(BUILD)/libvap_model.dylib: src/vap_model.c src/vap_model.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -install_name @rpath/libvap_model.dylib -o $@ src/vap_model.c

$(BUILD)/libsonata_istft.dylib: src/sonata_istft.c src/sonata_istft.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libsonata_istft.dylib -o $@ src/sonata_istft.c

$(BUILD)/libcodec_12hz.dylib: src/codec_12hz.c src/codec_12hz.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libcodec_12hz.dylib -o $@ src/codec_12hz.c

$(BUILD)/libnoise_gate.dylib: src/noise_gate.c src/noise_gate.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libnoise_gate.dylib -o $@ src/noise_gate.c

$(BUILD)/libdeep_filter.dylib: src/deep_filter.c src/deep_filter.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libdeep_filter.dylib -o $@ src/deep_filter.c

$(BUILD)/libaudio_watermark.dylib: src/audio_watermark.c src/audio_watermark.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libaudio_watermark.dylib -o $@ src/audio_watermark.c

$(BUILD)/libweb_remote.dylib: src/web_remote.c src/web_remote.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC \
	  -install_name @rpath/libweb_remote.dylib -o $@ src/web_remote.c

$(BUILD)/libapple_perf.dylib: src/apple_perf.c src/apple_perf.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -x objective-c \
	  -framework IOSurface -framework Foundation \
	  -install_name @rpath/libapple_perf.dylib -o $@ src/apple_perf.c

$(BUILD)/libconversation_memory.dylib: src/conversation_memory.c src/conversation_memory.h \
                                       $(BUILD)/cJSON.o | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -shared -fPIC \
	  -install_name @rpath/libconversation_memory.dylib \
	  -o $@ src/conversation_memory.c $(BUILD)/cJSON.o

$(BUILD)/libbackchannel.dylib: src/backchannel.c src/backchannel.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -shared -fPIC -framework Accelerate \
	  -install_name @rpath/libbackchannel.dylib -o $@ src/backchannel.c

$(BUILD)/libneural_backchannel.dylib: src/neural_backchannel.c src/neural_backchannel.h \
                                      $(BUILD)/libbreath_synthesis.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -shared -fPIC -framework Accelerate \
	  -L$(BUILD) -lbreath_synthesis -Wl,-rpath,@loader_path \
	  -install_name @rpath/libneural_backchannel.dylib -o $@ src/neural_backchannel.c

$(BUILD)/libintent_router.dylib: src/intent_router.c src/intent_router.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libintent_router.dylib -o $@ src/intent_router.c

$(BUILD)/libresponse_cache.dylib: src/response_cache.c src/response_cache.h src/intent_router.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -shared -fPIC \
	  -install_name @rpath/libresponse_cache.dylib -o $@ src/response_cache.c

$(BUILD)/libaudio_emotion.dylib: src/audio_emotion.c src/audio_emotion.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -shared -fPIC -framework Accelerate \
	  -install_name @rpath/libaudio_emotion.dylib -o $@ src/audio_emotion.c

$(BUILD)/libaudio_mixer.dylib: src/audio_mixer.c src/audio_mixer.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -shared -fPIC -framework Accelerate \
	  -install_name @rpath/libaudio_mixer.dylib -o $@ src/audio_mixer.c

$(BUILD)/libspeculative_gen.dylib: src/speculative_gen.c src/speculative_gen.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -shared -fPIC -lpthread \
	  -install_name @rpath/libspeculative_gen.dylib -o $@ src/speculative_gen.c

$(BUILD)/libstreaming_tts.dylib: src/streaming_tts.c src/streaming_tts.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -shared -fPIC -framework Accelerate -lpthread \
	  -install_name @rpath/libstreaming_tts.dylib -o $@ src/streaming_tts.c

$(BUILD)/libstreaming_llm.dylib: src/streaming_llm.c src/streaming_llm.h $(BUILD)/cJSON.o | $(BUILD)
	$(CC) $(CFLAGS) $(CURL_CFLAGS) -Isrc -shared -fPIC \
	  -L$(HOMEBREW_PREFIX)/lib -lcurl \
	  -install_name @rpath/libstreaming_llm.dylib -o $@ src/streaming_llm.c $(BUILD)/cJSON.o

$(BUILD)/tensor_ops.metallib: src/tensor_ops.metal | $(BUILD)
	xcrun -sdk macosx metal -c -O3 -o $(BUILD)/tensor_ops.air src/tensor_ops.metal
	xcrun -sdk macosx metallib -o $@ $(BUILD)/tensor_ops.air

libs: $(BUILD)/libpocket_voice.dylib $(BUILD)/libvdsp_prosody.dylib \
      $(BUILD)/libaudio_converter.dylib $(BUILD)/libspatial_audio.dylib \
      $(BUILD)/libtext_normalize.dylib $(BUILD)/libsentence_buffer.dylib \
      $(BUILD)/libssml_parser.dylib $(BUILD)/libpocket_opus.dylib \
      $(BUILD)/libbreath_synthesis.dylib $(BUILD)/liblufs.dylib \
      $(BUILD)/libvm_ring.dylib \
      $(BUILD)/libmimi_endpointer.dylib $(BUILD)/libfused_eou.dylib \
      $(BUILD)/libmel_spectrogram.dylib $(BUILD)/libconformer_stt.dylib \
      $(BUILD)/libctc_beam_decoder.dylib $(BUILD)/libtdt_decoder.dylib \
      $(BUILD)/libbnns_conformer.dylib \
      $(BUILD)/libbnns_convnext_decoder.dylib \
      $(BUILD)/libwebsocket.dylib \
      $(BUILD)/libhttp_api.dylib \
      $(BUILD)/libvoice_quality.dylib $(BUILD)/liblatency_profiler.dylib \
      $(BUILD)/libmetal_loader.dylib \
      $(BUILD)/libspm_tokenizer.dylib \
      $(BUILD)/libphonemizer.dylib \
      $(BUILD)/libspeaker_encoder.dylib \
      $(BUILD)/libspeaker_diarizer.dylib \
      $(BUILD)/libnative_vad.dylib \
      $(BUILD)/libspeech_detector.dylib \
      $(BUILD)/libsemantic_eou.dylib \
      $(BUILD)/libprosody_predict.dylib \
      $(BUILD)/libprosody_log.dylib \
      $(BUILD)/libemphasis_predict.dylib \
      $(BUILD)/libvoice_onboard.dylib \
      $(BUILD)/libnoise_gate.dylib \
      $(BUILD)/libdeep_filter.dylib \
      $(BUILD)/libaudio_watermark.dylib \
      $(BUILD)/libweb_remote.dylib \
      $(BUILD)/libapple_perf.dylib \
      $(BUILD)/libconversation_memory.dylib \
      $(BUILD)/libbackchannel.dylib \
      $(BUILD)/libneural_backchannel.dylib \
      $(BUILD)/libintent_router.dylib \
      $(BUILD)/libresponse_cache.dylib \
      $(BUILD)/libaudio_emotion.dylib \
      $(BUILD)/libaudio_mixer.dylib \
      $(BUILD)/libspeculative_gen.dylib \
      $(BUILD)/libstreaming_tts.dylib \
      $(BUILD)/libstreaming_llm.dylib \
      $(BUILD)/libsonata_istft.dylib \
      $(BUILD)/libcodec_12hz.dylib \
      $(BUILD)/libsonata_stt.dylib \
      $(BUILD)/libsonata_refiner.dylib \
      $(BUILD)/libvap_model.dylib \
      $(BUILD)/tensor_ops.metallib

# ─── Rust cdylibs ─────────────────────────────────────────────────────────

$(STT_DYLIB): src/stt/src/lib.rs src/stt/Cargo.toml src/stt/build.rs
	cargo build --release -p pocket-stt

LLM_DYLIB := target/release/libpocket_llm.dylib
$(LLM_DYLIB): src/llm/src/lib.rs src/llm/Cargo.toml src/llm/build.rs
	cargo build --release -p pocket-llm

SONATA_LM_DYLIB := target/release/libsonata_lm.dylib
$(SONATA_LM_DYLIB): src/sonata_lm/src/lib.rs src/sonata_lm/Cargo.toml src/sonata_lm/build.rs
	cargo build --release -p sonata-lm

SONATA_FLOW_DYLIB := target/release/libsonata_flow.dylib
$(SONATA_FLOW_DYLIB): src/sonata_flow/src/lib.rs src/sonata_flow/Cargo.toml src/sonata_flow/build.rs
	cargo build --release -p sonata-flow

SONATA_STORM_DYLIB := target/release/libsonata_storm.dylib
$(SONATA_STORM_DYLIB): src/sonata_storm/src/lib.rs src/sonata_storm/Cargo.toml src/sonata_storm/build.rs
	RUSTFLAGS="-C target-cpu=native" cargo build --release -p sonata_storm

SONATA_SPEAKER_DYLIB := src/sonata_speaker/target/release/libsonata_speaker.dylib
$(SONATA_SPEAKER_DYLIB): src/sonata_speaker/src/lib.rs src/sonata_speaker/Cargo.toml src/sonata_speaker/build.rs
	cd src/sonata_speaker && RUSTFLAGS="-C target-cpu=native" cargo build --release

# ─── cJSON object ─────────────────────────────────────────────────────────

$(BUILD)/cJSON.o: src/cJSON.c src/cJSON.h | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ src/cJSON.c

# ─── Main pipeline binary ─────────────────────────────────────────────────

sonata: src/pocket_voice_pipeline.c libs $(STT_DYLIB) $(LLM_DYLIB) $(SONATA_LM_DYLIB) $(SONATA_FLOW_DYLIB) $(SONATA_STORM_DYLIB)
	$(CC) $(CFLAGS) $(CURL_CFLAGS) \
	  -DACCELERATE_NEW_LAPACK \
	  -Isrc \
	  $(FRAMEWORKS) $(CURL_LDFLAGS) \
	  -L$(BUILD) -lpocket_voice -lvdsp_prosody -laudio_converter -lspatial_audio \
	  -ltext_normalize -lsentence_buffer -lssml_parser -lpocket_opus \
	  -lbreath_synthesis -llufs -lvm_ring \
	  -lmimi_endpointer -lfused_eou \
	  -lmel_spectrogram -lconformer_stt -lctc_beam_decoder -ltdt_decoder \
	  -lbnns_conformer -lbnns_convnext_decoder -lvoice_quality -llatency_profiler -lmetal_loader \
	  -lspm_tokenizer -lnoise_gate -ldeep_filter -laudio_watermark \
	  -lspeaker_encoder -lnative_vad -lspeech_detector -lsemantic_eou -lphonemizer -lprosody_predict \
	  -lprosody_log -lemphasis_predict -lvoice_onboard -lsonata_istft -lsonata_stt -lsonata_refiner -lweb_remote -lwebsocket -lhttp_api -lapple_perf \
	  -lbackchannel -laudio_emotion -lconversation_memory -lspeaker_diarizer -lvap_model \
	  -lneural_backchannel -lintent_router -lresponse_cache -lstreaming_tts -laudio_mixer -lspeculative_gen -lstreaming_llm \
	  $(OPUS_LDFLAGS) -Ltarget/release -lpocket_llm \
	  $(STT_DYLIB) $(LLM_DYLIB) $(SONATA_LM_DYLIB) $(SONATA_FLOW_DYLIB) $(SONATA_STORM_DYLIB) \
	  -Wl,-rpath,@executable_path/$(BUILD) \
	  -Wl,-rpath,@executable_path/target/release \
	  -Wl,-rpath,$(HOMEBREW_PREFIX)/lib \
	  $(ORT_RPATH) \
	  -o $@ src/pocket_voice_pipeline.c

# Backward-compat alias
pocket-voice: sonata
	ln -sf sonata pocket-voice

# ─── Quality Benchmark Suite ───────────────────────────────────────────────

QUALITY_SRC = src/quality/wer.c src/quality/audio_quality.c \
              src/quality/latency_harness.c src/quality/roundtrip.c

bench-quality: src/quality/bench_quality.c $(QUALITY_SRC) | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -framework Accelerate \
	  $(QUALITY_SRC) src/quality/bench_quality.c \
	  -o $(BUILD)/bench-quality

test-quality: tests/test_quality.c $(QUALITY_SRC) | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -framework Accelerate \
	  $(QUALITY_SRC) tests/test_quality.c \
	  -o $(BUILD)/test-quality
	./$(BUILD)/test-quality

# ─── EOU Detection Suite ──────────────────────────────────────────────────

EOU_SRC = src/mimi_endpointer.c src/fused_eou.c

$(BUILD)/libmimi_endpointer.dylib: src/mimi_endpointer.c src/mimi_endpointer.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libmimi_endpointer.dylib -o $@ src/mimi_endpointer.c

$(BUILD)/libfused_eou.dylib: src/fused_eou.c src/fused_eou.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC \
	  -install_name @rpath/libfused_eou.dylib -o $@ src/fused_eou.c

$(BUILD)/libsemantic_eou.dylib: src/semantic_eou.c src/semantic_eou.h src/lstm_ops.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libsemantic_eou.dylib -o $@ src/semantic_eou.c

test-eou: tests/test_eou.c $(EOU_SRC) $(BUILD)/libmel_spectrogram.dylib $(BUILD)/libconformer_stt.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  $(EOU_SRC) tests/test_eou.c \
	  -L$(BUILD) -lmel_spectrogram -lconformer_stt \
	  -Wl,-rpath,@executable_path \
	  -o $(BUILD)/test-eou -lm
	./$(BUILD)/test-eou

test-semantic-eou: tests/test_semantic_eou.c $(BUILD)/libsemantic_eou.dylib $(BUILD)/libfused_eou.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -L$(BUILD) -lsemantic_eou -lfused_eou \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -lm -o $(BUILD)/test-semantic-eou tests/test_semantic_eou.c
	./$(BUILD)/test-semantic-eou

# ─── Unit Test Build Targets ──────────────────────────────────────────────

test-pipeline: tests/test_pipeline.c $(BUILD)/libtext_normalize.dylib \
               $(BUILD)/libsentence_buffer.dylib $(BUILD)/libssml_parser.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -L$(BUILD) \
	  -ltext_normalize -lsentence_buffer -lssml_parser \
	  -Wl,-rpath,@executable_path \
	  -o $(BUILD)/test-pipeline tests/test_pipeline.c
	./$(BUILD)/test-pipeline

test-new-modules: tests/test_new_modules.c \
                  $(BUILD)/libbreath_synthesis.dylib $(BUILD)/liblufs.dylib \
                  $(BUILD)/libvm_ring.dylib $(BUILD)/libsentence_buffer.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lbreath_synthesis -llufs -lvm_ring -lsentence_buffer \
	  -Wl,-rpath,@executable_path \
	  -o $(BUILD)/test-new-modules tests/test_new_modules.c
	./$(BUILD)/test-new-modules

test-bugfixes: tests/test_bugfixes.c \
               $(BUILD)/liblufs.dylib $(BUILD)/libvm_ring.dylib \
               $(BUILD)/libsentence_buffer.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -framework Accelerate \
	  -L$(BUILD) -llufs -lvm_ring -lsentence_buffer \
	  -Wl,-rpath,@executable_path \
	  -o $(BUILD)/test-bugfixes tests/test_bugfixes.c
	./$(BUILD)/test-bugfixes

test-conformer: tests/test_conformer_stt.c \
                $(BUILD)/libmel_spectrogram.dylib $(BUILD)/libconformer_stt.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lmel_spectrogram -lconformer_stt \
	  -Wl,-rpath,@executable_path \
	  -o $(BUILD)/test-conformer tests/test_conformer_stt.c
	./$(BUILD)/test-conformer

test-roundtrip: tests/test_roundtrip.c $(QUALITY_SRC) | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -framework Accelerate \
	  $(QUALITY_SRC) tests/test_roundtrip.c \
	  -o $(BUILD)/test-roundtrip
	./$(BUILD)/test-roundtrip

test-llm-prosody: tests/test_llm_prosody.c $(BUILD)/cJSON.o \
                  $(BUILD)/libtext_normalize.dylib \
                  $(BUILD)/libsentence_buffer.dylib $(BUILD)/libssml_parser.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -L$(BUILD) \
	  -ltext_normalize -lsentence_buffer -lssml_parser \
	  -Wl,-rpath,@executable_path \
	  -o $(BUILD)/test-llm-prosody tests/test_llm_prosody.c $(BUILD)/cJSON.o
	@bash -c 'DYLD_LIBRARY_PATH=$(BUILD) ./$(BUILD)/test-llm-prosody'

test-websocket: tests/test_websocket.c $(BUILD)/libwebsocket.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -L$(BUILD) -lwebsocket -framework Security \
	  -Wl,-rpath,@executable_path \
	  -o $(BUILD)/test-websocket tests/test_websocket.c
	./$(BUILD)/test-websocket

test-conversation-memory: tests/test_conversation_memory.c \
                          $(BUILD)/libconversation_memory.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -L$(BUILD) -lconversation_memory \
	  -Wl,-rpath,@executable_path \
	  -o $(BUILD)/test-conversation-memory tests/test_conversation_memory.c
	./$(BUILD)/test-conversation-memory

# ─── Run All Tests ────────────────────────────────────────────────────────

test-optimizations: tests/test_optimizations.c \
                    $(BUILD)/libbnns_conformer.dylib \
                    $(BUILD)/libtext_normalize.dylib $(BUILD)/libsentence_buffer.dylib \
                    $(BUILD)/liblatency_profiler.dylib $(BUILD)/libvoice_quality.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lbnns_conformer -ltext_normalize -lsentence_buffer \
	  -llatency_profiler -lvoice_quality \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -o $(BUILD)/test-optimizations tests/test_optimizations.c
	./$(BUILD)/test-optimizations

test-new-engines: tests/test_new_engines.c \
                  $(BUILD)/libphonemizer.dylib $(BUILD)/libspeaker_encoder.dylib | $(BUILD)
	$(CC) $(CFLAGS) $(PIPER_CFLAGS) -Isrc \
	  -L$(BUILD) -lphonemizer -lspeaker_encoder \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -Wl,-rpath,$(HOMEBREW_PREFIX)/lib \
	  $(ORT_RPATH) \
	  -o $(BUILD)/test-new-engines tests/test_new_engines.c
	./$(BUILD)/test-new-engines

test-diarizer: tests/test_diarizer.c $(BUILD)/libspeaker_diarizer.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lspeaker_diarizer -lspeaker_encoder \
	  -Wl,-rpath,@executable_path -lm \
	  -o $(BUILD)/test-diarizer tests/test_diarizer.c
	./$(BUILD)/test-diarizer

test-real-models: tests/test_real_models.c \
                  $(BUILD)/libphonemizer.dylib $(BUILD)/libspeaker_encoder.dylib | $(BUILD)
	$(CC) $(CFLAGS) $(PIPER_CFLAGS) -Isrc \
	  -L$(BUILD) -lphonemizer -lspeaker_encoder \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -Wl,-rpath,$(HOMEBREW_PREFIX)/lib \
	  $(ORT_RPATH) \
	  -lm -o $(BUILD)/test-real-models tests/test_real_models.c
	./$(BUILD)/test-real-models

test-native-vad: tests/test_native_vad.c $(BUILD)/libnative_vad.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lnative_vad \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -lm -framework Accelerate \
	  -o $(BUILD)/test-native-vad tests/test_native_vad.c
	./$(BUILD)/test-native-vad

test-vap: tests/test_vap.c $(BUILD)/libvap_model.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lvap_model -Wl,-rpath,@executable_path \
	  -o $(BUILD)/test-vap tests/test_vap.c
	./$(BUILD)/test-vap

test-audio-mixer: tests/test_audio_mixer.c $(BUILD)/libaudio_mixer.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -framework Accelerate \
	  -L$(BUILD) -laudio_mixer -Wl,-rpath,@executable_path \
	  -o $(BUILD)/test-audio-mixer tests/test_audio_mixer.c
	./$(BUILD)/test-audio-mixer

test-speech-detector: tests/test_speech_detector.c $(BUILD)/libspeech_detector.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lspeech_detector -lnative_vad -lmimi_endpointer -lfused_eou \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -lm -framework Accelerate \
	  -o $(BUILD)/test-speech-detector tests/test_speech_detector.c
	./$(BUILD)/test-speech-detector

test-fused-eou-parallel: tests/test_fused_eou_parallel.c $(BUILD)/libfused_eou.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lfused_eou \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -lm -framework Accelerate \
	  -o $(BUILD)/test-fused-eou-parallel tests/test_fused_eou_parallel.c
	./$(BUILD)/test-fused-eou-parallel

bench-vad: tests/test_bench_vad.c $(BUILD)/libnative_vad.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lnative_vad \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -lm -framework Accelerate \
	  -o $(BUILD)/bench-vad tests/test_bench_vad.c
	./$(BUILD)/bench-vad

test-prosody-predict: tests/test_prosody_predict.c $(BUILD)/libprosody_predict.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lprosody_predict \
	  -Wl,-rpath,@executable_path \
	  -lm -o $(BUILD)/test-prosody-predict tests/test_prosody_predict.c
	./$(BUILD)/test-prosody-predict

test-prosody-log: tests/test_prosody_log.c $(BUILD)/libprosody_log.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lprosody_log \
	  -Wl,-rpath,@executable_path \
	  -lm -o $(BUILD)/test-prosody-log tests/test_prosody_log.c
	./$(BUILD)/test-prosody-log

test-emphasis: tests/test_emphasis.c $(BUILD)/libemphasis_predict.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lemphasis_predict \
	  -Wl,-rpath,@executable_path \
	  -lm -o $(BUILD)/test-emphasis tests/test_emphasis.c
	./$(BUILD)/test-emphasis

test-prosody-integration: tests/test_prosody_integration.c \
                          $(BUILD)/libemphasis_predict.dylib \
                          $(BUILD)/libssml_parser.dylib \
                          $(BUILD)/libprosody_predict.dylib \
                          $(BUILD)/libsentence_buffer.dylib \
                          $(BUILD)/libtext_normalize.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lemphasis_predict -lssml_parser -lprosody_predict -lsentence_buffer -ltext_normalize \
	  -Wl,-rpath,@executable_path \
	  -lm -o $(BUILD)/test-prosody-integration tests/test_prosody_integration.c
	./$(BUILD)/test-prosody-integration

test-voice-onboard: tests/test_voice_onboard.c $(BUILD)/libvoice_onboard.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lvoice_onboard \
	  -Wl,-rpath,@executable_path \
	  -lm -framework Accelerate \
	  -o $(BUILD)/test-voice-onboard tests/test_voice_onboard.c
	./$(BUILD)/test-voice-onboard

test-beam-search: tests/test_beam_search.c $(BUILD)/libctc_beam_decoder.dylib | $(BUILD)
	c++ -std=c++17 -O3 -arch arm64 -Wall -Wextra -Wno-unused-parameter -Isrc \
	  -L$(BUILD) -lctc_beam_decoder \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-beam-search tests/test_beam_search.c
	./$(BUILD)/test-beam-search

test-sonata-stt: tests/test_sonata_stt.c $(BUILD)/libsonata_stt.dylib $(BUILD)/libsonata_refiner.dylib $(BUILD)/libmel_spectrogram.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lsonata_stt -lsonata_refiner -lmel_spectrogram \
	  -Wl,-rpath,@executable_path -lm \
	  -o $(BUILD)/test-sonata-stt tests/test_sonata_stt.c
	./$(BUILD)/test-sonata-stt

test-sonata: tests/test_sonata.c $(BUILD)/libsonata_istft.dylib $(BUILD)/libspm_tokenizer.dylib $(BUILD)/libbnns_convnext_decoder.dylib $(SONATA_LM_DYLIB) $(SONATA_FLOW_DYLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lsonata_istft -lspm_tokenizer -lbnns_convnext_decoder \
	  -Ltarget/release \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -Wl,-rpath,$(CURDIR)/target/release \
	  -lsonata_lm -lsonata_flow -lm \
	  -o $(BUILD)/test-sonata tests/test_sonata.c
	./$(BUILD)/test-sonata

test-sonata-quality: tests/test_sonata_quality.c \
                     $(BUILD)/libspm_tokenizer.dylib $(BUILD)/libconformer_stt.dylib \
                     $(BUILD)/libctc_beam_decoder.dylib $(BUILD)/libtdt_decoder.dylib \
                     $(BUILD)/libmel_spectrogram.dylib \
                     $(SONATA_LM_DYLIB) $(SONATA_FLOW_DYLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lspm_tokenizer -lconformer_stt -lctc_beam_decoder -lmel_spectrogram -ltdt_decoder \
	  -Ltarget/release \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -Wl,-rpath,$(CURDIR)/target/release \
	  -lsonata_lm -lsonata_flow -lm \
	  src/quality/wer.c src/quality/audio_quality.c \
	  -o $(BUILD)/test-sonata-quality tests/test_sonata_quality.c
	./$(BUILD)/test-sonata-quality

eval-generate: tests/eval_sonata_baseline.c \
               $(BUILD)/libconformer_stt.dylib \
               $(BUILD)/libctc_beam_decoder.dylib $(BUILD)/libtdt_decoder.dylib \
               $(BUILD)/libmel_spectrogram.dylib \
               $(SONATA_FLOW_DYLIB) | $(BUILD)
	@mkdir -p eval/generated eval/reports
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lconformer_stt -lctc_beam_decoder -lmel_spectrogram -ltdt_decoder \
	  -Ltarget/release \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -Wl,-rpath,$(CURDIR)/target/release \
	  -lsonata_flow -lm \
	  src/quality/wer.c src/quality/audio_quality.c \
	  -o $(BUILD)/eval-sonata-baseline tests/eval_sonata_baseline.c
	$(BUILD)/eval-sonata-baseline

eval-python: eval/run_eval.py
	@mkdir -p eval/reports
	python3 eval/run_eval.py --wav-dir eval/generated \
	  --c-report eval/reports/c_eval_report.json \
	  --output eval/reports/eval_report.json

eval: eval-generate eval-python
	@echo ""
	@echo "=== Full eval complete: eval/reports/eval_report.json ==="

eval-full: eval-generate
	python3 eval/run_eval.py --wav-dir eval/generated \
	  --c-report eval/reports/c_eval_report.json \
	  --utmos --output eval/reports/eval_report.json

QUALITY_SRC_FULL = src/quality/wer.c src/quality/audio_quality.c \
                   src/quality/latency_harness.c src/quality/roundtrip.c

bench-live: tests/bench_live.c $(QUALITY_SRC_FULL) \
            $(BUILD)/libconformer_stt.dylib \
            $(BUILD)/libvoice_quality.dylib $(BUILD)/libapple_perf.dylib \
            $(BUILD)/liblatency_profiler.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  $(QUALITY_SRC_FULL) tests/bench_live.c \
	  -L$(BUILD) -lconformer_stt -lmel_spectrogram \
	  -lctc_beam_decoder -ltdt_decoder -lspm_tokenizer \
	  -lvoice_quality -lapple_perf -llatency_profiler \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/bench-live

bench-industry: tests/bench_industry.c $(QUALITY_SRC_FULL) \
                $(BUILD)/libvoice_quality.dylib $(BUILD)/libapple_perf.dylib \
                $(BUILD)/liblatency_profiler.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -framework Accelerate \
	  $(QUALITY_SRC_FULL) tests/bench_industry.c \
	  -L$(BUILD) -lvoice_quality -lapple_perf -llatency_profiler \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/bench-industry
	./$(BUILD)/bench-industry

test-apple-perf: tests/test_apple_perf.c $(BUILD)/libapple_perf.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -x objective-c \
	  -framework IOSurface -framework Foundation \
	  -L$(BUILD) -lapple_perf \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-apple-perf tests/test_apple_perf.c
	./$(BUILD)/test-apple-perf

test-quality-improvements: tests/test_quality_improvements.c \
                           $(BUILD)/libnoise_gate.dylib $(BUILD)/liblufs.dylib \
                           $(BUILD)/libvoice_quality.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lnoise_gate -llufs -lvoice_quality \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-quality-improvements tests/test_quality_improvements.c
	./$(BUILD)/test-quality-improvements

test-deep-filter: tests/test_deep_filter.c $(BUILD)/libdeep_filter.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -ldeep_filter \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-deep-filter tests/test_deep_filter.c
	./$(BUILD)/test-deep-filter

test-codec-12hz: tests/test_codec_12hz.c $(BUILD)/libcodec_12hz.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lcodec_12hz \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-codec-12hz tests/test_codec_12hz.c
	./$(BUILD)/test-codec-12hz

test-vdsp-prosody: tests/test_vdsp_prosody.c $(BUILD)/libvdsp_prosody.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -framework Accelerate \
	  -L$(BUILD) -lvdsp_prosody \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-vdsp-prosody tests/test_vdsp_prosody.c
	./$(BUILD)/test-vdsp-prosody

test-http-api: tests/test_http_api.c $(BUILD)/libhttp_api.dylib $(BUILD)/libwebsocket.dylib $(BUILD)/libpocket_opus.dylib $(BUILD)/cJSON.o | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lhttp_api -lwebsocket -lpocket_opus $(BUILD)/cJSON.o \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -o $(BUILD)/test-http-api tests/test_http_api.c
	./$(BUILD)/test-http-api

bench-sonata: tests/bench_sonata.c $(BUILD)/libsonata_istft.dylib $(BUILD)/libspm_tokenizer.dylib $(SONATA_LM_DYLIB) $(SONATA_FLOW_DYLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lsonata_istft -lspm_tokenizer \
	  -Ltarget/release \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -Wl,-rpath,$(CURDIR)/target/release \
	  -lsonata_lm -lsonata_flow -lm \
	  -o $(BUILD)/bench-sonata tests/bench_sonata.c
	./$(BUILD)/bench-sonata

bench-e2e-latency: tools/bench_e2e_latency.c $(BUILD)/libsonata_istft.dylib $(BUILD)/libspm_tokenizer.dylib $(BUILD)/libconformer_stt.dylib $(SONATA_LM_DYLIB) $(SONATA_FLOW_DYLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lsonata_istft -lspm_tokenizer -lconformer_stt \
	  -Ltarget/release \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -Wl,-rpath,$(CURDIR)/target/release \
	  -lsonata_lm -lsonata_flow -lm \
	  -o $(BUILD)/bench-e2e-latency tools/bench_e2e_latency.c
	@echo "E2E latency benchmark built: ./$(BUILD)/bench-e2e-latency"
	@echo "Usage: ./$(BUILD)/bench-e2e-latency [optionally set model environment variables]"

bench-speculative: tools/bench_speculative.c $(SONATA_LM_DYLIB) | $(BUILD)
	mkdir -p bench_output
	$(CC) $(CFLAGS) -Isrc \
	  -Ltarget/release \
	  -Wl,-rpath,$(CURDIR)/target/release \
	  -lsonata_lm -lm \
	  -o $(BUILD)/bench-speculative tools/bench_speculative.c
	@echo "Speculative decoding benchmark built: ./$(BUILD)/bench-speculative"
	@echo "Usage: ./$(BUILD)/bench-speculative [model_path] [drafter_path] [num_iters]"
	@echo "Example: ./$(BUILD)/bench-speculative models/sonata_v3/sonata_lm.safetensors rnn_drafter.safetensors 10"

# ─── Test Coverage Campaign Targets ───────────────────────────────────────

test-sonata-storm: tests/test_sonata_storm.c $(SONATA_STORM_DYLIB) | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -Ltarget/release \
	  -Wl,-rpath,$(CURDIR)/target/release \
	  -lsonata_storm -lm \
	  -o $(BUILD)/test-sonata-storm tests/test_sonata_storm.c
	./$(BUILD)/test-sonata-storm

test-audio-emotion: tests/test_audio_emotion.c $(BUILD)/libaudio_emotion.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -framework Accelerate \
	  -L$(BUILD) -laudio_emotion \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-audio-emotion tests/test_audio_emotion.c
	./$(BUILD)/test-audio-emotion

test-sonata-flow-ffi: tests/test_sonata_flow_ffi.c $(SONATA_FLOW_DYLIB) | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -Ltarget/release \
	  -Wl,-rpath,$(CURDIR)/target/release \
	  -lsonata_flow -lm \
	  -o $(BUILD)/test-sonata-flow-ffi tests/test_sonata_flow_ffi.c
	./$(BUILD)/test-sonata-flow-ffi

test-flow-quality-modes: tests/test_flow_quality_modes.c $(SONATA_FLOW_DYLIB) | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -Ltarget/release \
	  -Wl,-rpath,$(CURDIR)/target/release \
	  -lsonata_flow -lm \
	  -o $(BUILD)/test-flow-quality-modes tests/test_flow_quality_modes.c
	./$(BUILD)/test-flow-quality-modes

test-sonata-flow-distilled: $(SONATA_FLOW_DYLIB)
	cd src/sonata_flow && cargo test --lib --release distilled

test-sonata-v3: tests/test_sonata_v3.c $(SONATA_FLOW_DYLIB) | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -Ltarget/release \
	  -Wl,-rpath,$(CURDIR)/target/release \
	  -lsonata_flow -lm \
	  -o $(BUILD)/test-sonata-v3 tests/test_sonata_v3.c
	./$(BUILD)/test-sonata-v3

test-sonata-lm-ffi: tests/test_sonata_lm_ffi.c $(SONATA_LM_DYLIB) | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -Ltarget/release \
	  -Wl,-rpath,$(CURDIR)/target/release \
	  -lsonata_lm -lm \
	  -o $(BUILD)/test-sonata-lm-ffi tests/test_sonata_lm_ffi.c
	./$(BUILD)/test-sonata-lm-ffi

test-sonata-lm-dual-head: tests/test_sonata_lm_dual_head.c $(SONATA_LM_DYLIB) | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -Ltarget/release \
	  -Wl,-rpath,$(CURDIR)/target/release \
	  -lsonata_lm -lm \
	  -o $(BUILD)/test-sonata-lm-dual-head tests/test_sonata_lm_dual_head.c
	./$(BUILD)/test-sonata-lm-dual-head

test-speaker-encoder-unit: tests/test_speaker_encoder_unit.c $(BUILD)/libspeaker_encoder.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lspeaker_encoder \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -o $(BUILD)/test-speaker-encoder-unit tests/test_speaker_encoder_unit.c -lm
	./$(BUILD)/test-speaker-encoder-unit

test-sonata-lm-prosody-edge: tests/test_sonata_lm_prosody_edge.c $(SONATA_LM_DYLIB) | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -Ltarget/release \
	  -Wl,-rpath,$(CURDIR)/target/release \
	  -lsonata_lm -lm \
	  -o $(BUILD)/test-sonata-lm-prosody-edge tests/test_sonata_lm_prosody_edge.c
	./$(BUILD)/test-sonata-lm-prosody-edge

test-pipeline-threading: tests/test_pipeline_threading.c $(BUILD)/libvm_ring.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lvm_ring \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm -lpthread \
	  -o $(BUILD)/test-pipeline-threading tests/test_pipeline_threading.c
	./$(BUILD)/test-pipeline-threading

test-phonemizer-v3: tests/test_phonemizer_v3.c $(BUILD)/libphonemizer.dylib | $(BUILD)
	$(CC) $(CFLAGS) $(PIPER_CFLAGS) -Isrc \
	  -L$(BUILD) -lphonemizer \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -Wl,-rpath,$(HOMEBREW_PREFIX)/lib \
	  -o $(BUILD)/test-phonemizer-v3 tests/test_phonemizer_v3.c
	./$(BUILD)/test-phonemizer-v3

test-phoneme-word-mapping: tests/test_phoneme_word_mapping.c | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -lm -o $(BUILD)/test-phoneme-word-mapping tests/test_phoneme_word_mapping.c
	./$(BUILD)/test-phoneme-word-mapping

test-phase2-regressions: tests/test_phase2_regressions.c \
                          $(BUILD)/libbreath_synthesis.dylib \
                          $(BUILD)/libmel_spectrogram.dylib \
                          $(BUILD)/libsentence_buffer.dylib \
                          $(BUILD)/libtext_normalize.dylib \
                          $(BUILD)/libconformer_stt.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lbreath_synthesis -lmel_spectrogram \
	  -lsentence_buffer -ltext_normalize -lconformer_stt \
	  -lctc_beam_decoder -ltdt_decoder \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-phase2-regressions tests/test_phase2_regressions.c
	./$(BUILD)/test-phase2-regressions

test-backchannel: tests/test_backchannel.c $(BUILD)/libbackchannel.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -framework Accelerate \
	  -L$(BUILD) -lbackchannel \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-backchannel tests/test_backchannel.c
	./$(BUILD)/test-backchannel

test-neural-backchannel: tests/test_neural_backchannel.c $(BUILD)/libneural_backchannel.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -framework Accelerate \
	  -L$(BUILD) -lneural_backchannel \
	  -Wl,-rpath,@executable_path -lm \
	  -o $(BUILD)/test-neural-backchannel tests/test_neural_backchannel.c
	./$(BUILD)/test-neural-backchannel

test-intent-router: tests/test_intent_router.c $(BUILD)/libintent_router.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -L$(BUILD) -lintent_router \
	  -Wl,-rpath,@executable_path -lm \
	  -o $(BUILD)/test-intent-router tests/test_intent_router.c
	./$(BUILD)/test-intent-router

test-response-cache: tests/test_response_cache.c $(BUILD)/libresponse_cache.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -L$(BUILD) -lresponse_cache \
	  -Wl,-rpath,@executable_path -o $(BUILD)/test-response-cache tests/test_response_cache.c
	./$(BUILD)/test-response-cache

test-speculative-gen: tests/test_speculative_gen.c $(BUILD)/libspeculative_gen.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -L$(BUILD) -lspeculative_gen -lpthread \
	  -Wl,-rpath,@executable_path -o $(BUILD)/test-speculative-gen tests/test_speculative_gen.c
	./$(BUILD)/test-speculative-gen

test-streaming-tts: tests/test_streaming_tts.c $(BUILD)/libstreaming_tts.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -L$(BUILD) -lstreaming_tts -lpthread \
	  -framework Accelerate -Wl,-rpath,@executable_path -o $(BUILD)/test-streaming-tts tests/test_streaming_tts.c
	./$(BUILD)/test-streaming-tts

test-streaming-llm: tests/test_streaming_llm.c $(BUILD)/libstreaming_llm.dylib | $(BUILD)
	$(CC) $(CFLAGS) $(CURL_CFLAGS) -Isrc \
	  -L$(BUILD) -lstreaming_llm $(CURL_LDFLAGS) \
	  -Wl,-rpath,@executable_path -o $(BUILD)/test-streaming-llm tests/test_streaming_llm.c
	./$(BUILD)/test-streaming-llm

test-full-duplex: tests/test_full_duplex.c $(BUILD)/libvap_model.dylib $(BUILD)/libaudio_mixer.dylib \
	$(BUILD)/libneural_backchannel.dylib $(BUILD)/libintent_router.dylib \
	$(BUILD)/libspeculative_gen.dylib $(BUILD)/libspeech_detector.dylib $(BUILD)/libbackchannel.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lvap_model -laudio_mixer -lneural_backchannel -lbreath_synthesis -lintent_router \
	  -lspeculative_gen -lspeech_detector -lnative_vad -lmimi_endpointer -lfused_eou -lbackchannel \
	  -Wl,-rpath,@executable_path -lm \
	  -o $(BUILD)/test-full-duplex tests/test_full_duplex.c
	./$(BUILD)/test-full-duplex

test-sonata-refiner: tests/test_sonata_refiner.c $(BUILD)/libsonata_refiner.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lsonata_refiner \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-sonata-refiner tests/test_sonata_refiner.c
	./$(BUILD)/test-sonata-refiner

test-tdt-decoder: tests/test_tdt_decoder.c $(BUILD)/libtdt_decoder.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -ltdt_decoder \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-tdt-decoder tests/test_tdt_decoder.c
	./$(BUILD)/test-tdt-decoder

test-web-remote: tests/test_web_remote.c $(BUILD)/libweb_remote.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lweb_remote \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-web-remote tests/test_web_remote.c
	./$(BUILD)/test-web-remote

test-security-audit: tests/test_security_audit.c | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -o $(BUILD)/test-security-audit tests/test_security_audit.c
	./$(BUILD)/test-security-audit

test-assumptions: tests/test_assumptions.c | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-assumptions tests/test_assumptions.c
	./$(BUILD)/test-assumptions

test-opus-codec: tests/test_opus_codec.c $(BUILD)/libpocket_opus.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc $(OPUS_CFLAGS) \
	  -L$(BUILD) -lpocket_opus $(OPUS_LDFLAGS) -lopus \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-opus-codec tests/test_opus_codec.c
	./$(BUILD)/test-opus-codec

test-audio-converter: tests/test_audio_converter.c $(BUILD)/libaudio_converter.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -framework AudioToolbox -framework CoreFoundation \
	  -L$(BUILD) -laudio_converter \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-audio-converter tests/test_audio_converter.c
	./$(BUILD)/test-audio-converter

test-spatial-audio: tests/test_spatial_audio.c $(BUILD)/libspatial_audio.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -framework AudioToolbox -framework Accelerate \
	  -L$(BUILD) -lspatial_audio \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-spatial-audio tests/test_spatial_audio.c
	./$(BUILD)/test-spatial-audio

test-metal-loader: tests/test_metal_loader.c $(BUILD)/libmetal_loader.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -x objective-c \
	  -framework Metal -framework Foundation \
	  -L$(BUILD) -lmetal_loader \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-metal-loader tests/test_metal_loader.c
	./$(BUILD)/test-metal-loader

test-metal-dispatch: tests/test_metal_dispatch.c $(BUILD)/libmetal_dispatch.dylib $(BUILD)/libmetal_loader.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -x objective-c \
	  -framework Metal -framework Foundation -framework Accelerate \
	  -L$(BUILD) -lmetal_dispatch -lmetal_loader \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-metal-dispatch tests/test_metal_dispatch.c
	./$(BUILD)/test-metal-dispatch

test-bnns-convnext: tests/test_bnns_convnext.c $(BUILD)/libbnns_convnext_decoder.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lbnns_convnext_decoder \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-bnns-convnext tests/test_bnns_convnext.c
	./$(BUILD)/test-bnns-convnext

test-coverage-gaps: tests/test_coverage_gaps.c \
                    $(BUILD)/libbreath_synthesis.dylib \
                    $(BUILD)/libnoise_gate.dylib \
                    $(BUILD)/liblufs.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -framework Accelerate \
	  -L$(BUILD) -lbreath_synthesis -lnoise_gate -llufs \
	  -Wl,-rpath,@executable_path \
	  -lm -o $(BUILD)/test-coverage-gaps tests/test_coverage_gaps.c
	./$(BUILD)/test-coverage-gaps

test-correctness-audit: tests/test_correctness_audit.c $(BUILD)/libmel_spectrogram.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lmel_spectrogram \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-correctness-audit tests/test_correctness_audit.c
	./$(BUILD)/test-correctness-audit

test-integration-audit: tests/test_integration_audit.c \
                        $(BUILD)/libsonata_istft.dylib $(BUILD)/libmel_spectrogram.dylib \
                        $(BUILD)/libtext_normalize.dylib $(BUILD)/libsentence_buffer.dylib \
                        $(BUILD)/libssml_parser.dylib $(BUILD)/libsonata_stt.dylib \
                        $(BUILD)/libsonata_refiner.dylib $(BUILD)/libconformer_stt.dylib \
                        $(BUILD)/libctc_beam_decoder.dylib $(BUILD)/libtdt_decoder.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lsonata_istft -lmel_spectrogram \
	  -ltext_normalize -lsentence_buffer -lssml_parser \
	  -lsonata_stt -lsonata_refiner -lconformer_stt \
	  -lctc_beam_decoder -ltdt_decoder \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-integration-audit tests/test_integration_audit.c
	./$(BUILD)/test-integration-audit

bench-audit: tests/bench_audit.c \
             $(BUILD)/libmel_spectrogram.dylib $(BUILD)/libsonata_istft.dylib \
             $(BUILD)/libmetal_dispatch.dylib $(BUILD)/libmetal_loader.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc \
	  -framework Accelerate -framework Metal -framework Foundation \
	  -L$(BUILD) -lmel_spectrogram -lsonata_istft -lmetal_dispatch -lmetal_loader \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/bench-audit tests/bench_audit.c
	./$(BUILD)/bench-audit

# ─── Research Implementation Tests ────────────────────────────────────────

test-research-stt: tests/test_research_stt.c \
                   $(BUILD)/libconformer_stt.dylib $(BUILD)/libmel_spectrogram.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lconformer_stt -lmel_spectrogram \
	  -lctc_beam_decoder -ltdt_decoder \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-research-stt tests/test_research_stt.c
	./$(BUILD)/test-research-stt

test-research-eou: tests/test_research_eou.c $(BUILD)/libfused_eou.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lfused_eou \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-research-eou tests/test_research_eou.c
	./$(BUILD)/test-research-eou

test-research-istft: tests/test_research_istft.c \
                     $(BUILD)/libsonata_istft.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lsonata_istft \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-research-istft tests/test_research_istft.c
	./$(BUILD)/test-research-istft

test-research-metal: tests/test_research_metal.c \
                     $(BUILD)/libmetal_dispatch.dylib $(BUILD)/libapple_perf.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc \
	  -framework Metal -framework Foundation -framework Accelerate \
	  -L$(BUILD) -lmetal_dispatch -lapple_perf \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-research-metal tests/test_research_metal.c
	./$(BUILD)/test-research-metal

test-audio-watermark: tests/test_audio_watermark.c \
                      $(BUILD)/libaudio_watermark.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc \
	  -framework Accelerate \
	  -L$(BUILD) -laudio_watermark \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) -lm \
	  -o $(BUILD)/test-audio-watermark tests/test_audio_watermark.c
	./$(BUILD)/test-audio-watermark

test-gru-drafter: tests/test_gru_drafter.c $(SONATA_LM_DYLIB) | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -ldl \
	  -o $(BUILD)/test-gru-drafter tests/test_gru_drafter.c
	./$(BUILD)/test-gru-drafter

# ─── E2E Validation Tests (Untested Features) ──────────────────────────

test-voice-cloning-e2e: tests/test_voice_cloning_e2e.c $(BUILD)/libmel_spectrogram.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -framework Accelerate \
	  -L$(BUILD) -lmel_spectrogram \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -o $(BUILD)/test-voice-cloning-e2e tests/test_voice_cloning_e2e.c -lm
	./$(BUILD)/test-voice-cloning-e2e

test-flow-streaming: tests/test_flow_streaming.c $(SONATA_FLOW_DYLIB) | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -Ltarget/release -lsonata_flow \
	  -Wl,-rpath,$(CURDIR)/target/release \
	  -o $(BUILD)/test-flow-streaming tests/test_flow_streaming.c -lm
	./$(BUILD)/test-flow-streaming

test-flow-distilled-loading: tests/test_flow_distilled.c | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -o $(BUILD)/test-flow-distilled-loading tests/test_flow_distilled.c -lm
	./$(BUILD)/test-flow-distilled-loading

test-speaker-encoder-integration: tests/test_speaker_encoder.c $(BUILD)/libmel_spectrogram.dylib | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -framework Accelerate \
	  -L$(BUILD) -lmel_spectrogram \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -o $(BUILD)/test-speaker-encoder-integration tests/test_speaker_encoder.c -lm
	./$(BUILD)/test-speaker-encoder-integration

test-speaker-encoder: tests/test_speaker_encoder.c $(BUILD)/libspeaker_encoder.dylib $(BUILD)/libmel_spectrogram.dylib $(SONATA_SPEAKER_DYLIB) | $(BUILD)
	$(CC) $(CFLAGS) -Isrc \
	  -L$(BUILD) -lspeaker_encoder -lmel_spectrogram \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -Wl,-rpath,$(CURDIR)/src/sonata_speaker/target/release \
	  -o $(BUILD)/test-speaker-encoder tests/test_speaker_encoder.c
	./$(BUILD)/test-speaker-encoder

.PHONY: test test-eou test-semantic-eou test-pipeline test-new-modules test-new-engines test-bugfixes test-conformer test-roundtrip test-llm-prosody test-websocket test-optimizations test-sonata test-sonata-quality test-sonata-stt test-sonata-v3 test-beam-search bench-sonata bench-e2e-latency bench-speculative bench-quality bench-live bench-industry test-apple-perf test-quality-improvements test-real-models test-native-vad bench-vad test-speech-detector test-fused-eou-parallel test-prosody-predict test-prosody-log test-emphasis test-prosody-integration test-voice-onboard test-conversation-memory test-diarizer test-vdsp-prosody test-http-api test-sonata-storm test-audio-emotion test-audio-mixer test-sonata-flow-ffi test-flow-quality-modes test-sonata-flow-distilled test-sonata-lm-ffi test-sonata-lm-dual-head test-speaker-encoder-unit test-sonata-lm-prosody-edge test-pipeline-threading test-phase2-regressions test-phonemizer-v3 test-backchannel test-neural-backchannel test-intent-router test-response-cache test-speculative-gen test-streaming-tts test-streaming-llm test-full-duplex test-sonata-refiner test-tdt-decoder test-web-remote test-opus-codec test-audio-converter test-spatial-audio test-metal-loader test-metal-dispatch test-bnns-convnext test-coverage-gaps test-correctness-audit test-integration-audit test-security-audit test-assumptions bench-audit bench test-research-stt test-research-eou test-research-istft test-research-metal test-audio-watermark test-deep-filter test-codec-12hz test-gru-drafter test-speaker-encoder test-voice-cloning-e2e test-flow-streaming test-flow-distilled-loading test-speaker-encoder-integration eval eval-python eval-full eval-generate

bench: libs sonata
	@bash scripts/benchmark.sh --all
test: bench-quality test-quality test-eou test-semantic-eou test-roundtrip test-pipeline test-new-modules test-new-engines test-bugfixes test-conformer test-llm-prosody test-optimizations test-beam-search test-sonata test-sonata-v3 test-sonata-quality test-sonata-stt test-real-models test-websocket test-native-vad test-speech-detector test-fused-eou-parallel test-prosody-predict test-prosody-log test-emphasis test-prosody-integration test-voice-onboard test-conversation-memory test-diarizer test-vdsp-prosody test-http-api test-quality-improvements test-sonata-storm test-audio-emotion test-audio-mixer test-sonata-flow-ffi test-flow-quality-modes test-sonata-flow-distilled test-sonata-lm-ffi test-sonata-lm-dual-head test-speaker-encoder-unit test-sonata-lm-prosody-edge test-pipeline-threading test-phase2-regressions test-phonemizer-v3 test-backchannel test-neural-backchannel test-intent-router test-response-cache test-speculative-gen test-streaming-tts test-streaming-llm test-full-duplex test-sonata-refiner test-tdt-decoder test-web-remote test-opus-codec test-audio-converter test-spatial-audio test-metal-loader test-metal-dispatch test-bnns-convnext test-coverage-gaps test-integration-audit test-correctness-audit test-security-audit test-assumptions test-research-stt test-research-eou test-research-istft test-research-metal test-audio-watermark test-deep-filter test-speaker-encoder test-voice-cloning-e2e test-flow-streaming test-flow-distilled-loading test-speaker-encoder-integration
	@echo ""
	@echo "═══ Quality Benchmark Self-Tests ═══"
	./$(BUILD)/bench-quality
	@echo ""
	@echo "═══ ALL TESTS COMPLETE ═══"

# ─── Clean ─────────────────────────────────────────────────────────────────

clean:
	rm -rf $(BUILD) sonata pocket-voice
	cd src/stt && cargo clean
	cd src/llm && cargo clean 2>/dev/null || true
	cd src/sonata_lm && cargo clean 2>/dev/null || true
	cd src/sonata_flow && cargo clean 2>/dev/null || true
	cd src/sonata_storm && cargo clean 2>/dev/null || true
