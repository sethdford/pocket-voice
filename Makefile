# Makefile — Build pocket-voice: zero-Python voice pipeline for Apple Silicon.
#
# Targets:
#   all              Build everything
#   pocket-voice     Standalone pipeline binary (mic → STT → Claude → TTS → speaker)
#   libs             Build all shared libraries
#   clean            Remove build artifacts
#
# Usage:
#   make                 # Build all
#   make pocket-voice    # Build just the binary
#   make clean           # Clean everything
#   ANTHROPIC_API_KEY=sk-... ./pocket-voice  # Run

CC      := cc
CFLAGS  := -O3 -arch arm64 -Wall -Wextra -Wno-unused-parameter
LDFLAGS := -arch arm64

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
STT_DYLIB   := src/stt/target/release/libpocket_stt.dylib
TTS_DYLIB   := src/tts/target/release/libpocket_tts_rs.dylib

.PHONY: all clean libs pocket-voice

all: libs pocket-voice

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

$(BUILD)/libssml_parser.dylib: src/ssml_parser.c src/ssml_parser.h src/text_normalize.h \
                               $(BUILD)/libtext_normalize.dylib | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC \
	  -L$(BUILD) -ltext_normalize \
	  -Wl,-rpath,@loader_path \
	  -install_name @rpath/libssml_parser.dylib -o $@ src/ssml_parser.c

$(BUILD)/libpocket_opus.dylib: src/opus_codec.c | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC $(OPUS_CFLAGS) \
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

$(BUILD)/libbnns_mimi.dylib: src/bnns_mimi_decoder.c | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libbnns_mimi.dylib -o $@ src/bnns_mimi_decoder.c

$(BUILD)/libmel_spectrogram.dylib: src/mel_spectrogram.c src/mel_spectrogram.h | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -install_name @rpath/libmel_spectrogram.dylib -o $@ src/mel_spectrogram.c

$(BUILD)/libconformer_stt.dylib: src/conformer_stt.c src/conformer_stt.h \
                                  $(BUILD)/libmel_spectrogram.dylib | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -DACCELERATE_NEW_LAPACK -framework Accelerate \
	  -L$(BUILD) -lmel_spectrogram \
	  -Wl,-rpath,@loader_path \
	  -install_name @rpath/libconformer_stt.dylib -o $@ src/conformer_stt.c

libs: $(BUILD)/libpocket_voice.dylib $(BUILD)/libvdsp_prosody.dylib \
      $(BUILD)/libaudio_converter.dylib $(BUILD)/libspatial_audio.dylib \
      $(BUILD)/libtext_normalize.dylib $(BUILD)/libsentence_buffer.dylib \
      $(BUILD)/libssml_parser.dylib $(BUILD)/libpocket_opus.dylib \
      $(BUILD)/libbreath_synthesis.dylib $(BUILD)/liblufs.dylib \
      $(BUILD)/libvm_ring.dylib $(BUILD)/libbnns_mimi.dylib \
      $(BUILD)/libmimi_endpointer.dylib $(BUILD)/libfused_eou.dylib \
      $(BUILD)/libmel_spectrogram.dylib $(BUILD)/libconformer_stt.dylib

# ─── Rust cdylibs ─────────────────────────────────────────────────────────

$(STT_DYLIB): src/stt/src/lib.rs src/stt/Cargo.toml src/stt/build.rs
	cd src/stt && cargo build --release

$(TTS_DYLIB): src/tts/src/lib.rs src/tts/Cargo.toml src/tts/build.rs
	cd src/tts && cargo build --release

# ─── cJSON object ─────────────────────────────────────────────────────────

$(BUILD)/cJSON.o: src/cJSON.c src/cJSON.h | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ src/cJSON.c

# ─── Main pipeline binary ─────────────────────────────────────────────────

pocket-voice: src/pocket_voice_pipeline.c $(BUILD)/cJSON.o libs $(STT_DYLIB) $(TTS_DYLIB)
	$(CC) $(CFLAGS) $(CURL_CFLAGS) \
	  -DACCELERATE_NEW_LAPACK \
	  -Isrc \
	  $(FRAMEWORKS) $(CURL_LDFLAGS) \
	  -L$(BUILD) -lpocket_voice -lvdsp_prosody -laudio_converter -lspatial_audio \
	  -ltext_normalize -lsentence_buffer -lssml_parser -lpocket_opus \
	  -lbreath_synthesis -llufs -lvm_ring -lbnns_mimi \
	  -lmimi_endpointer -lfused_eou \
	  -lmel_spectrogram -lconformer_stt \
	  $(OPUS_LDFLAGS) \
	  $(STT_DYLIB) $(TTS_DYLIB) \
	  -Wl,-rpath,@executable_path/$(BUILD) \
	  -Wl,-rpath,@executable_path/src/stt/target/release \
	  -Wl,-rpath,@executable_path/src/tts/target/release \
	  -o $@ src/pocket_voice_pipeline.c $(BUILD)/cJSON.o

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

test-eou: tests/test_eou.c $(EOU_SRC) $(BUILD)/libmel_spectrogram.dylib $(BUILD)/libconformer_stt.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  $(EOU_SRC) tests/test_eou.c \
	  -L$(BUILD) -lmel_spectrogram -lconformer_stt \
	  -Wl,-rpath,@executable_path \
	  -o $(BUILD)/test-eou -lm
	./$(BUILD)/test-eou

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
                  $(BUILD)/libvm_ring.dylib $(BUILD)/libsentence_buffer.dylib \
                  $(BUILD)/libbnns_mimi.dylib | $(BUILD)
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lbreath_synthesis -llufs -lvm_ring -lsentence_buffer -lbnns_mimi \
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

# ─── Run All Tests ────────────────────────────────────────────────────────

.PHONY: test test-eou test-pipeline test-new-modules test-bugfixes test-conformer test-roundtrip
test: bench-quality test-quality test-eou test-roundtrip test-pipeline test-new-modules test-bugfixes test-conformer
	@echo ""
	@echo "═══ Quality Benchmark Self-Tests ═══"
	./$(BUILD)/bench-quality
	@echo ""
	@echo "═══ ALL TESTS COMPLETE ═══"

# ─── Clean ─────────────────────────────────────────────────────────────────

clean:
	rm -rf $(BUILD) pocket-voice
	cd src/stt && cargo clean
	cd src/tts && cargo clean
