// whisper_turbo.h — C FFI for Whisper v3 Turbo multilingual ASR on Apple Silicon
//
// Streaming ASR pipeline:
//   1. Create engine: handle = whisper_turbo_create(model_dir)
//   2. Feed audio in chunks: whisper_turbo_feed(handle, pcm, n_samples)
//   3. Flush and transcribe: whisper_turbo_flush(handle, out_text, max_len, out_lang, lang_max)
//   4. Destroy: whisper_turbo_destroy(handle)
//
// Blocking single-call API:
//   whisper_turbo_process(handle, pcm, n_samples, lang_hint, out_text, max_len)
//
// Returns: bytes written to out_text (>= 0) or -1 on error

#ifndef WHISPER_TURBO_H
#define WHISPER_TURBO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* WhisperTurboHandle;

/// Create a Whisper Turbo ASR engine.
/// @param model_dir Path to model directory (config.json, tokenizer.json, model.safetensors)
/// @return Handle to engine, or NULL on error
WhisperTurboHandle whisper_turbo_create(const char* model_dir);

/// Destroy a Whisper Turbo engine and free resources.
/// @param handle Engine handle
void whisper_turbo_destroy(WhisperTurboHandle handle);

/// Reset the audio buffer and internal state.
/// @param handle Engine handle
/// @return 0 on success, -1 on error
int whisper_turbo_reset(WhisperTurboHandle handle);

/// Feed audio samples into the streaming buffer.
/// Audio accumulates until flush is called. Buffer is capped at 15 seconds.
/// @param handle Engine handle
/// @param pcm PCM audio samples (f32, 16kHz)
/// @param n_samples Number of samples
/// @return 0 on success, -1 on error
int whisper_turbo_feed(WhisperTurboHandle handle, const float* pcm, int32_t n_samples);

/// Transcribe buffered audio and detect language.
/// @param handle Engine handle
/// @param out_text Output buffer for transcribed text (null-terminated)
/// @param max_len Size of out_text buffer
/// @param out_lang Output buffer for detected language code (e.g., "en", "zh"), or NULL
/// @param lang_max Size of out_lang buffer (if not NULL)
/// @return Bytes written to out_text (>= 0) or -1 on error
int32_t whisper_turbo_flush(
    WhisperTurboHandle handle,
    char* out_text,
    int32_t max_len,
    char* out_lang,
    int32_t lang_max
);

/// Single-call blocking transcription (feed + flush).
/// @param handle Engine handle
/// @param pcm PCM audio samples (f32, 16kHz)
/// @param n_samples Number of samples
/// @param lang Language hint (unused for now; use NULL)
/// @param out_text Output buffer for transcribed text (null-terminated)
/// @param max_len Size of out_text buffer
/// @return Bytes written to out_text (>= 0) or -1 on error
int32_t whisper_turbo_process(
    WhisperTurboHandle handle,
    const float* pcm,
    int32_t n_samples,
    const char* lang,
    char* out_text,
    int32_t max_len
);

#ifdef __cplusplus
}
#endif

#endif // WHISPER_TURBO_H
