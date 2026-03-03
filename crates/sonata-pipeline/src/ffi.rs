//! C-compatible FFI exports for the Sonata pipeline.
//!
//! These functions provide a C ABI for initializing, running, and shutting down
//! the Sonata voice pipeline from C code (e.g., SeaClaw integration).
//!
//! # Error Codes
//!
//! - `0`: Success
//! - `-1`: Pipeline not initialized
//! - `-2`: Invalid argument (null pointer, invalid size, etc.)
//! - `-3`: Internal error (model failure, memory allocation, etc.)
//! - `-4`: Not implemented
//!
//! # Safety
//!
//! All functions are marked `unsafe` from the Rust side since they cross the FFI boundary.
//! C callers must ensure:
//! - Valid non-null pointers (except where documented as nullable)
//! - Buffer sizes are correct
//! - Buffers don't overlap
//! - Thread safety: Initialize and deinit only once per process

use std::sync::Mutex;

use crate::orchestrator::PipelineOrchestrator;
use candle_core::Device;

/// Error code constants for FFI return values
pub const SC_OK: i32 = 0;
pub const SC_ERR_NOT_INITIALIZED: i32 = -1;
pub const SC_ERR_INVALID_ARGUMENT: i32 = -2;
pub const SC_ERR_INTERNAL: i32 = -3;
pub const SC_ERR_NOT_IMPLEMENTED: i32 = -4;

/// Global pipeline instance (lazy-initialized).
static PIPELINE: Mutex<Option<PipelineOrchestrator>> = Mutex::new(None);

/// Initialize the Sonata pipeline.
///
/// # Arguments
///
/// * `config_json` - Pointer to JSON configuration (currently unused, reserved for future)
/// * `config_len` - Length of JSON configuration in bytes
///
/// # Returns
///
/// `SC_OK` (0) on success, negative error code on failure.
///
/// # Safety
///
/// This function is unsafe because it accepts raw pointers. The caller must ensure:
/// - `config_json` is either NULL or points to valid UTF-8 JSON
/// - If non-NULL, `config_json` has at least `config_len` bytes
///
/// # Example (C)
///
/// ```c
/// int ret = sonata_pipeline_init(NULL, 0);
/// if (ret != 0) {
///     // Handle initialization failure
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn sonata_pipeline_init(
    _config_json: *const u8,
    _config_len: usize,
) -> i32 {
    // Initialize on CPU device for now (Metal requires setup)
    let dev = Device::Cpu;
    match PipelineOrchestrator::new(&dev) {
        Ok(orchestrator) => {
            let mut pipeline = PIPELINE.lock().expect("Failed to acquire pipeline lock");
            *pipeline = Some(orchestrator);
            SC_OK
        }
        Err(_) => SC_ERR_INTERNAL,
    }
}

/// Run speech-to-text on raw audio.
///
/// # Arguments
///
/// * `audio` - Pointer to f32 samples (24kHz mono)
/// * `samples` - Number of samples
/// * `text` - Output buffer for transcribed text (UTF-8)
/// * `text_len` - In: buffer capacity, Out: bytes written
///
/// # Returns
///
/// `SC_OK` (0) on success, negative error code on failure.
///
/// # Safety
///
/// This function is unsafe because it accepts raw pointers. The caller must ensure:
/// - `audio` points to valid f32 samples
/// - `samples` is correct
/// - `text` points to a valid buffer of at least `*text_len` bytes
/// - `text_len` pointer is valid and dereferenceable
///
/// # Note
///
/// Currently returns `SC_ERR_NOT_IMPLEMENTED` as STT requires mel computation
/// infrastructure not yet wired.
///
/// # Example (C)
///
/// ```c
/// float audio[48000]; // 2 seconds at 24kHz
/// char text[1024];
/// size_t text_len = sizeof(text);
/// int ret = sonata_stt(audio, 48000, (uint8_t*)text, &text_len);
/// if (ret == 0) {
///     printf("Transcribed: %.*s\n", (int)text_len, text);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn sonata_stt(
    _audio: *const f32,
    _samples: usize,
    _text: *mut u8,
    _text_len: *mut usize,
) -> i32 {
    // Stub: STT requires mel computation infrastructure not yet wired
    SC_ERR_NOT_IMPLEMENTED
}

/// Run text-to-speech.
///
/// # Arguments
///
/// * `text` - Input text (UTF-8)
/// * `text_len` - Length of text in bytes
/// * `speaker_id` - Speaker identifier (NULL-safe, currently unused)
/// * `emotion_exag` - Emotion exaggeration factor (0.0-2.0)
/// * `audio` - Output buffer for f32 samples (24kHz mono)
/// * `audio_len` - In: buffer capacity (samples), Out: samples written
///
/// # Returns
///
/// `SC_OK` (0) on success, negative error code on failure.
///
/// # Safety
///
/// This function is unsafe because it accepts raw pointers. The caller must ensure:
/// - `text` points to valid UTF-8 data
/// - `text_len` is correct
/// - `speaker_id` is NULL or points to valid UTF-8
/// - `audio` points to a valid f32 buffer
/// - `audio_len` pointer is valid and dereferenceable
/// - Output buffer is large enough (typically 3-5 seconds → 72k-120k samples at 24kHz)
///
/// # Note
///
/// Currently returns `SC_ERR_NOT_IMPLEMENTED` as TTS flow requires mel computation
/// and other infrastructure not yet wired.
///
/// # Example (C)
///
/// ```c
/// const char* text = "Hello, world!";
/// float audio[120000]; // ~5 seconds at 24kHz
/// size_t audio_len = 120000;
/// int ret = sonata_tts(
///     (const uint8_t*)text, strlen(text),
///     NULL, 1.0,
///     audio, &audio_len
/// );
/// if (ret == 0) {
///     printf("Generated %zu samples\n", audio_len);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn sonata_tts(
    _text: *const u8,
    _text_len: usize,
    _speaker_id: *const u8,
    _emotion_exag: f32,
    _audio: *mut f32,
    _audio_len: *mut usize,
) -> i32 {
    // Stub: TTS requires mel computation and other infrastructure not yet wired
    SC_ERR_NOT_IMPLEMENTED
}

/// Encode speaker embedding from audio.
///
/// # Arguments
///
/// * `audio` - Pointer to f32 samples (16kHz mono for mel computation)
/// * `samples` - Number of samples
/// * `embedding` - Output buffer for f32[192] speaker embedding
///
/// # Returns
///
/// `SC_OK` (0) on success, negative error code on failure.
///
/// # Safety
///
/// This function is unsafe because it accepts raw pointers. The caller must ensure:
/// - `audio` points to valid f32 samples
/// - `samples` is correct
/// - `embedding` points to a valid buffer of at least 192 * sizeof(f32) bytes
///
/// # Note
///
/// Currently returns `SC_ERR_NOT_IMPLEMENTED` as speaker encoding requires
/// mel computation infrastructure not yet wired.
///
/// # Example (C)
///
/// ```c
/// float audio[16000]; // 1 second at 16kHz
/// float embedding[192];
/// int ret = sonata_speaker_encode(audio, 16000, embedding);
/// if (ret == 0) {
///     // Use embedding for speaker-conditioned TTS
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn sonata_speaker_encode(
    _audio: *const f32,
    _samples: usize,
    _embedding: *mut f32,
) -> i32 {
    // Stub: Speaker encoding requires mel computation infrastructure not yet wired
    SC_ERR_NOT_IMPLEMENTED
}

/// Shut down the Sonata pipeline and release resources.
///
/// # Safety
///
/// This function is safe to call multiple times. Each call after the first
/// is a no-op.
///
/// # Example (C)
///
/// ```c
/// sonata_pipeline_deinit();
/// ```
#[no_mangle]
pub extern "C" fn sonata_pipeline_deinit() {
    if let Ok(mut pipeline) = PIPELINE.lock() {
        *pipeline = None;
    }
}

/// Streaming TTS callback for use with SeaClaw's streaming LLM.
///
/// # Arguments
///
/// * `ctx` - Opaque context pointer (typically a StreamingBridge)
/// * `text_delta` - Text chunk from LLM (UTF-8)
/// * `text_len` - Length of text chunk in bytes
/// * `is_final` - True if this is the last chunk from the LLM
///
/// # Returns
///
/// `SC_OK` (0) on success, negative error code on failure.
///
/// # Safety
///
/// This function is unsafe because it accepts raw pointers. The caller must ensure:
/// - `ctx` is a valid pointer to a StreamingBridge (or NULL)
/// - `text_delta` points to valid UTF-8 data
/// - `text_len` is correct
///
/// # Note
///
/// Currently returns `SC_ERR_NOT_IMPLEMENTED`. Full streaming integration
/// requires SeaClaw LLM bridge not yet wired.
///
/// # Example (C)
///
/// ```c
/// // In a streaming LLM callback:
/// void llm_callback(const char* token) {
///     size_t len = strlen(token);
///     sonata_streaming_tts_callback(NULL, (uint8_t*)token, len, false);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn sonata_streaming_tts_callback(
    _ctx: *mut std::ffi::c_void,
    _text_delta: *const u8,
    _text_len: usize,
    _is_final: bool,
) -> i32 {
    // Stub: Streaming integration requires SeaClaw LLM bridge not yet wired
    SC_ERR_NOT_IMPLEMENTED
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_init_and_deinit() {
        unsafe {
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);

            sonata_pipeline_deinit();
            // After deinit, pipeline should be None
            let pipeline = PIPELINE.lock().unwrap();
            assert!(pipeline.is_none());
        }
    }

    #[test]
    fn test_ffi_double_init() {
        unsafe {
            let ret1 = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret1, SC_OK);

            // Second init should succeed (replaces previous)
            let ret2 = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret2, SC_OK);

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_deinit_without_init() {
        // Deinit without init should be safe (no crash)
        sonata_pipeline_deinit();
        let pipeline = PIPELINE.lock().unwrap();
        assert!(pipeline.is_none());
    }

    #[test]
    fn test_ffi_stt_not_initialized() {
        // STT without init should return not implemented (stub)
        unsafe {
            sonata_pipeline_deinit(); // Ensure deinitialized
            let ret = sonata_stt(std::ptr::null(), 0, std::ptr::null_mut(), std::ptr::null_mut());
            assert_eq!(ret, SC_ERR_NOT_IMPLEMENTED);
        }
    }

    #[test]
    fn test_ffi_null_pointer_safety() {
        unsafe {
            // These should not crash even with null pointers
            let ret1 = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret1, SC_OK);

            let ret2 = sonata_stt(std::ptr::null(), 100, std::ptr::null_mut(), std::ptr::null_mut());
            assert_eq!(ret2, SC_ERR_NOT_IMPLEMENTED);

            let ret3 =
                sonata_tts(std::ptr::null(), 0, std::ptr::null(), 1.0, std::ptr::null_mut(), std::ptr::null_mut());
            assert_eq!(ret3, SC_ERR_NOT_IMPLEMENTED);

            let ret4 = sonata_speaker_encode(std::ptr::null(), 0, std::ptr::null_mut());
            assert_eq!(ret4, SC_ERR_NOT_IMPLEMENTED);

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_error_codes() {
        assert_eq!(SC_OK, 0);
        assert_eq!(SC_ERR_NOT_INITIALIZED, -1);
        assert_eq!(SC_ERR_INVALID_ARGUMENT, -2);
        assert_eq!(SC_ERR_INTERNAL, -3);
        assert_eq!(SC_ERR_NOT_IMPLEMENTED, -4);
    }

    #[test]
    fn test_ffi_streaming_callback_stub() {
        unsafe {
            let ret = sonata_streaming_tts_callback(std::ptr::null_mut(), std::ptr::null(), 0, false);
            assert_eq!(ret, SC_ERR_NOT_IMPLEMENTED);
        }
    }
}
