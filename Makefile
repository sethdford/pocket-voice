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

$(BUILD)/libpocket_voice.dylib: src/pocket_voice.c | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC $(FRAMEWORKS) \
	  -install_name @rpath/libpocket_voice.dylib -o $@ $<

$(BUILD)/libvdsp_prosody.dylib: src/vdsp_prosody.c | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -framework Accelerate \
	  -install_name @rpath/libvdsp_prosody.dylib -o $@ $<

$(BUILD)/libaudio_converter.dylib: src/audio_converter.c | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -framework AudioToolbox -framework CoreFoundation \
	  -install_name @rpath/libaudio_converter.dylib -o $@ $<

$(BUILD)/libspatial_audio.dylib: src/spatial_audio.c | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC -framework AudioToolbox -framework Accelerate \
	  -install_name @rpath/libspatial_audio.dylib -o $@ $<

libs: $(BUILD)/libpocket_voice.dylib $(BUILD)/libvdsp_prosody.dylib \
      $(BUILD)/libaudio_converter.dylib $(BUILD)/libspatial_audio.dylib

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
	  $(FRAMEWORKS) $(CURL_LDFLAGS) \
	  -L$(BUILD) -lpocket_voice -lvdsp_prosody -laudio_converter -lspatial_audio \
	  $(STT_DYLIB) $(TTS_DYLIB) \
	  -Wl,-rpath,@executable_path/$(BUILD) \
	  -Wl,-rpath,@executable_path/src/stt/target/release \
	  -Wl,-rpath,@executable_path/src/tts/target/release \
	  -o $@ src/pocket_voice_pipeline.c $(BUILD)/cJSON.o

# ─── Clean ─────────────────────────────────────────────────────────────────

clean:
	rm -rf $(BUILD) pocket-voice
	cd src/stt && cargo clean
	cd src/tts && cargo clean
