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
//!
//! # Error Diagnostics
//!
//! Call `sonata_last_error()` after any non-zero return to get a human-readable
//! error message describing what went wrong.

use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
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

thread_local! {
    static LAST_ERROR: RefCell<String> = RefCell::new(String::new());
}

/// Store an error message for later retrieval via `sonata_last_error`.
fn set_last_error(msg: impl Into<String>) {
    let msg = msg.into();
    eprintln!("[sonata-ffi] error: {}", &msg);
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = msg;
    });
}

/// Retrieve the last error message.
///
/// Copies the most recent error string into the caller-provided buffer.
/// Returns the number of bytes written (excluding any NUL), or
/// `SC_ERR_INVALID_ARGUMENT` if `buf` is null.
///
/// If the buffer is too small, the message is truncated (no NUL terminator).
/// If no error has occurred, writes 0 bytes and returns 0.
///
/// # Safety
///
/// `buf` must point to a writable buffer of at least `buf_len` bytes, or be null.
#[no_mangle]
pub unsafe extern "C" fn sonata_last_error(buf: *mut u8, buf_len: usize) -> i32 {
    if buf.is_null() {
        return SC_ERR_INVALID_ARGUMENT;
    }
    LAST_ERROR.with(|e| {
        let msg = e.borrow();
        let bytes = msg.as_bytes();
        let copy_len = bytes.len().min(buf_len);
        if copy_len > 0 {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, copy_len);
        }
        copy_len as i32
    })
}

/// Initialize the Sonata pipeline.
///
/// # Arguments
///
/// * `config_json` - Pointer to JSON configuration (nullable). If non-null,
///   parsed for `"device":"metal"` or `"device":"gpu"` to select Metal acceleration.
///   Defaults to CPU if null or unrecognized.
/// * `config_len` - Length of JSON configuration in bytes
///
/// # Returns
///
/// `SC_OK` (0) on success, negative error code on failure.
///
/// # Safety
///
/// - `config_json` is either NULL or points to valid UTF-8 JSON of `config_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn sonata_pipeline_init(
    config_json: *const u8,
    config_len: usize,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Parse device preference from config JSON (simple string matching)
        let dev = parse_device_from_config(config_json, config_len);

        let orchestrator = match PipelineOrchestrator::new(&dev) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(format!("pipeline init failed: {e}"));
                return SC_ERR_INTERNAL;
            }
        };

        let mut guard = match PIPELINE.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                // Recover from poisoned mutex — take the inner guard
                set_last_error("pipeline mutex was poisoned, recovering");
                poisoned.into_inner()
            }
        };
        *guard = Some(orchestrator);
        SC_OK
    }));
    result.unwrap_or_else(|_| {
        set_last_error("panic during sonata_pipeline_init");
        SC_ERR_INTERNAL
    })
}

/// Parse device preference from a JSON config blob (simple string matching).
///
/// Returns `Device::Cpu` if config is null, empty, not valid UTF-8, or doesn't
/// contain a recognized device string.
unsafe fn parse_device_from_config(config_json: *const u8, config_len: usize) -> Device {
    if config_json.is_null() || config_len == 0 {
        return Device::Cpu;
    }
    let slice = std::slice::from_raw_parts(config_json, config_len);
    let Ok(json_str) = std::str::from_utf8(slice) else {
        return Device::Cpu;
    };
    let lower = json_str.to_lowercase();
    if lower.contains("\"metal\"") || lower.contains("\"gpu\"") {
        // Attempt Metal device; fall back to CPU on failure
        #[cfg(feature = "metal")]
        {
            match Device::new_metal(0) {
                Ok(d) => return d,
                Err(_) => return Device::Cpu,
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            return Device::Cpu;
        }
    }
    Device::Cpu
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
/// - `audio` points to valid f32 samples of length `samples`
/// - `text` points to a valid buffer of at least `*text_len` bytes
/// - `text_len` pointer is valid and dereferenceable
///
/// Currently returns `SC_ERR_NOT_IMPLEMENTED`.
#[no_mangle]
pub unsafe extern "C" fn sonata_stt(
    audio: *const f32,
    samples: usize,
    text: *mut u8,
    text_len: *mut usize,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate pointers
        if audio.is_null() && samples > 0 {
            set_last_error("sonata_stt: audio pointer is null with non-zero samples");
            return SC_ERR_INVALID_ARGUMENT;
        }
        if text.is_null() {
            set_last_error("sonata_stt: text output buffer is null");
            return SC_ERR_INVALID_ARGUMENT;
        }
        if text_len.is_null() {
            set_last_error("sonata_stt: text_len pointer is null");
            return SC_ERR_INVALID_ARGUMENT;
        }

        // Check pipeline is initialized
        let guard = match PIPELINE.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                set_last_error("sonata_stt: pipeline mutex poisoned");
                poisoned.into_inner()
            }
        };
        if guard.is_none() {
            set_last_error("sonata_stt: pipeline not initialized");
            return SC_ERR_NOT_INITIALIZED;
        }

        // Stub: STT requires mel computation infrastructure not yet wired
        set_last_error("sonata_stt: not implemented");
        SC_ERR_NOT_IMPLEMENTED
    }));
    result.unwrap_or_else(|_| {
        set_last_error("panic during sonata_stt");
        SC_ERR_INTERNAL
    })
}

/// Run text-to-speech.
///
/// # Arguments
///
/// * `text` - Input text (UTF-8)
/// * `text_len` - Length of text in bytes
/// * `speaker_id` - Speaker identifier (nullable)
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
/// - `text` points to valid UTF-8 data of `text_len` bytes
/// - `speaker_id` is NULL or points to valid UTF-8
/// - `audio` points to a valid f32 buffer of at least `*audio_len` samples
/// - `audio_len` pointer is valid and dereferenceable
///
/// Currently returns `SC_ERR_NOT_IMPLEMENTED`.
#[no_mangle]
pub unsafe extern "C" fn sonata_tts(
    text: *const u8,
    text_len: usize,
    _speaker_id: *const u8,
    _emotion_exag: f32,
    audio: *mut f32,
    audio_len: *mut usize,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate pointers
        if text.is_null() && text_len > 0 {
            set_last_error("sonata_tts: text pointer is null with non-zero length");
            return SC_ERR_INVALID_ARGUMENT;
        }
        if audio.is_null() {
            set_last_error("sonata_tts: audio output buffer is null");
            return SC_ERR_INVALID_ARGUMENT;
        }
        if audio_len.is_null() {
            set_last_error("sonata_tts: audio_len pointer is null");
            return SC_ERR_INVALID_ARGUMENT;
        }

        // Check pipeline is initialized
        let guard = match PIPELINE.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                set_last_error("sonata_tts: pipeline mutex poisoned");
                poisoned.into_inner()
            }
        };
        if guard.is_none() {
            set_last_error("sonata_tts: pipeline not initialized");
            return SC_ERR_NOT_INITIALIZED;
        }

        // Stub: TTS requires mel computation infrastructure not yet wired
        set_last_error("sonata_tts: not implemented");
        SC_ERR_NOT_IMPLEMENTED
    }));
    result.unwrap_or_else(|_| {
        set_last_error("panic during sonata_tts");
        SC_ERR_INTERNAL
    })
}

/// Encode speaker embedding from audio.
///
/// # Arguments
///
/// * `audio` - Pointer to f32 samples (16kHz mono)
/// * `samples` - Number of samples
/// * `embedding` - Output buffer for f32[192] speaker embedding
///
/// # Returns
///
/// `SC_OK` (0) on success, negative error code on failure.
///
/// # Safety
///
/// - `audio` points to valid f32 samples of length `samples`
/// - `embedding` points to a buffer of at least 192 f32 values
///
/// Currently returns `SC_ERR_NOT_IMPLEMENTED`.
#[no_mangle]
pub unsafe extern "C" fn sonata_speaker_encode(
    audio: *const f32,
    samples: usize,
    embedding: *mut f32,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate pointers
        if audio.is_null() && samples > 0 {
            set_last_error("sonata_speaker_encode: audio pointer is null with non-zero samples");
            return SC_ERR_INVALID_ARGUMENT;
        }
        if embedding.is_null() {
            set_last_error("sonata_speaker_encode: embedding output buffer is null");
            return SC_ERR_INVALID_ARGUMENT;
        }

        // Check pipeline is initialized
        let guard = match PIPELINE.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                set_last_error("sonata_speaker_encode: pipeline mutex poisoned");
                poisoned.into_inner()
            }
        };
        if guard.is_none() {
            set_last_error("sonata_speaker_encode: pipeline not initialized");
            return SC_ERR_NOT_INITIALIZED;
        }

        // Stub: Speaker encoding requires mel computation infrastructure not yet wired
        set_last_error("sonata_speaker_encode: not implemented");
        SC_ERR_NOT_IMPLEMENTED
    }));
    result.unwrap_or_else(|_| {
        set_last_error("panic during sonata_speaker_encode");
        SC_ERR_INTERNAL
    })
}

/// Shut down the Sonata pipeline and release resources.
///
/// Safe to call multiple times. Each call after the first is a no-op.
/// Recovers gracefully from a poisoned mutex.
#[no_mangle]
pub extern "C" fn sonata_pipeline_deinit() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut guard = match PIPELINE.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                set_last_error("sonata_pipeline_deinit: pipeline mutex poisoned, recovering");
                poisoned.into_inner()
            }
        };
        *guard = None;
    }));
    if result.is_err() {
        set_last_error("panic during sonata_pipeline_deinit");
    }
}

/// Streaming TTS callback for use with SeaClaw's streaming LLM.
///
/// # Arguments
///
/// * `ctx` - Opaque context pointer (nullable)
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
/// - `ctx` is a valid pointer or NULL
/// - `text_delta` points to valid UTF-8 data of `text_len` bytes (or is NULL if `text_len` is 0)
///
/// Currently returns `SC_ERR_NOT_IMPLEMENTED`.
#[no_mangle]
pub unsafe extern "C" fn sonata_streaming_tts_callback(
    _ctx: *mut std::ffi::c_void,
    text_delta: *const u8,
    text_len: usize,
    _is_final: bool,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if text_delta.is_null() && text_len > 0 {
            set_last_error("sonata_streaming_tts_callback: text_delta is null with non-zero length");
            return SC_ERR_INVALID_ARGUMENT;
        }

        // Stub: Streaming integration requires SeaClaw LLM bridge not yet wired
        SC_ERR_NOT_IMPLEMENTED
    }));
    result.unwrap_or_else(|_| {
        set_last_error("panic during sonata_streaming_tts_callback");
        SC_ERR_INTERNAL
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_error_codes() {
        assert_eq!(SC_OK, 0);
        assert_eq!(SC_ERR_NOT_INITIALIZED, -1);
        assert_eq!(SC_ERR_INVALID_ARGUMENT, -2);
        assert_eq!(SC_ERR_INTERNAL, -3);
        assert_eq!(SC_ERR_NOT_IMPLEMENTED, -4);
    }

    #[test]
    fn test_ffi_init_and_deinit() {
        unsafe {
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);

            sonata_pipeline_deinit();
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
    fn test_ffi_last_error() {
        set_last_error("test error message");
        let mut buf = [0u8; 256];
        unsafe {
            let len = sonata_last_error(buf.as_mut_ptr(), buf.len());
            assert!(len > 0);
            let msg = std::str::from_utf8(&buf[..len as usize]).unwrap();
            assert_eq!(msg, "test error message");
        }
    }

    #[test]
    fn test_ffi_last_error_null_buf() {
        unsafe {
            let ret = sonata_last_error(std::ptr::null_mut(), 100);
            assert_eq!(ret, SC_ERR_INVALID_ARGUMENT);
        }
    }

    #[test]
    fn test_ffi_last_error_truncation() {
        set_last_error("a]long error message that should be truncated");
        let mut buf = [0u8; 5];
        unsafe {
            let len = sonata_last_error(buf.as_mut_ptr(), buf.len());
            assert_eq!(len, 5);
            assert_eq!(&buf, b"a]lon");
        }
    }

    #[test]
    fn test_ffi_stt_null_text_output() {
        unsafe {
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);

            // null text buffer → invalid argument
            let mut tl: usize = 64;
            let ret = sonata_stt(std::ptr::null(), 0, std::ptr::null_mut(), &mut tl);
            assert_eq!(ret, SC_ERR_INVALID_ARGUMENT);

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_stt_null_text_len() {
        unsafe {
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);

            let mut buf = [0u8; 64];
            let ret = sonata_stt(std::ptr::null(), 0, buf.as_mut_ptr(), std::ptr::null_mut());
            assert_eq!(ret, SC_ERR_INVALID_ARGUMENT);

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_stt_not_initialized() {
        unsafe {
            sonata_pipeline_deinit(); // Ensure deinitialized
            let mut buf = [0u8; 64];
            let mut tl: usize = 64;
            let ret = sonata_stt(std::ptr::null(), 0, buf.as_mut_ptr(), &mut tl);
            assert_eq!(ret, SC_ERR_NOT_INITIALIZED);
        }
    }

    #[test]
    fn test_ffi_stt_audio_null_nonzero_samples() {
        unsafe {
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);

            let mut buf = [0u8; 64];
            let mut tl: usize = 64;
            let ret = sonata_stt(std::ptr::null(), 100, buf.as_mut_ptr(), &mut tl);
            assert_eq!(ret, SC_ERR_INVALID_ARGUMENT);

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_tts_null_audio_output() {
        unsafe {
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);

            let mut al: usize = 1000;
            let ret = sonata_tts(
                b"hello".as_ptr(), 5,
                std::ptr::null(), 1.0,
                std::ptr::null_mut(), &mut al,
            );
            assert_eq!(ret, SC_ERR_INVALID_ARGUMENT);

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_tts_null_audio_len() {
        unsafe {
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);

            let mut audio = [0f32; 100];
            let ret = sonata_tts(
                b"hello".as_ptr(), 5,
                std::ptr::null(), 1.0,
                audio.as_mut_ptr(), std::ptr::null_mut(),
            );
            assert_eq!(ret, SC_ERR_INVALID_ARGUMENT);

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_tts_not_initialized() {
        unsafe {
            sonata_pipeline_deinit();
            let mut audio = [0f32; 100];
            let mut al: usize = 100;
            let ret = sonata_tts(
                b"hello".as_ptr(), 5,
                std::ptr::null(), 1.0,
                audio.as_mut_ptr(), &mut al,
            );
            assert_eq!(ret, SC_ERR_NOT_INITIALIZED);
        }
    }

    #[test]
    fn test_ffi_speaker_encode_null_embedding() {
        unsafe {
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);

            let ret = sonata_speaker_encode(std::ptr::null(), 0, std::ptr::null_mut());
            assert_eq!(ret, SC_ERR_INVALID_ARGUMENT);

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_speaker_encode_not_initialized() {
        unsafe {
            sonata_pipeline_deinit();
            let mut emb = [0f32; 192];
            let ret = sonata_speaker_encode(std::ptr::null(), 0, emb.as_mut_ptr());
            assert_eq!(ret, SC_ERR_NOT_INITIALIZED);
        }
    }

    #[test]
    fn test_ffi_streaming_callback_null_text_nonzero() {
        unsafe {
            let ret = sonata_streaming_tts_callback(std::ptr::null_mut(), std::ptr::null(), 10, false);
            assert_eq!(ret, SC_ERR_INVALID_ARGUMENT);
        }
    }

    #[test]
    fn test_ffi_streaming_callback_stub() {
        unsafe {
            let ret = sonata_streaming_tts_callback(std::ptr::null_mut(), std::ptr::null(), 0, false);
            assert_eq!(ret, SC_ERR_NOT_IMPLEMENTED);
        }
    }

    #[test]
    fn test_ffi_null_pointer_safety() {
        unsafe {
            // Init with null config is fine
            let ret1 = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret1, SC_OK);

            // STT with null audio + zero samples → still returns not-implemented (init'd)
            let mut buf = [0u8; 64];
            let mut tl: usize = 64;
            let ret2 = sonata_stt(std::ptr::null(), 0, buf.as_mut_ptr(), &mut tl);
            assert_eq!(ret2, SC_ERR_NOT_IMPLEMENTED);

            // TTS with null text + zero len → still returns not-implemented (init'd)
            let mut audio = [0f32; 100];
            let mut al: usize = 100;
            let ret3 = sonata_tts(
                std::ptr::null(), 0,
                std::ptr::null(), 1.0,
                audio.as_mut_ptr(), &mut al,
            );
            assert_eq!(ret3, SC_ERR_NOT_IMPLEMENTED);

            // Speaker encode with null audio + zero samples → invalid (embedding null)
            let mut emb = [0f32; 192];
            let ret4 = sonata_speaker_encode(std::ptr::null(), 0, emb.as_mut_ptr());
            assert_eq!(ret4, SC_ERR_NOT_IMPLEMENTED);

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_config_json_cpu_default() {
        unsafe {
            // Null config → CPU (default)
            let dev = parse_device_from_config(std::ptr::null(), 0);
            assert!(matches!(dev, Device::Cpu));

            // Empty config → CPU
            let dev = parse_device_from_config(b"".as_ptr(), 0);
            assert!(matches!(dev, Device::Cpu));
        }
    }

    #[test]
    fn test_ffi_config_json_invalid_utf8() {
        unsafe {
            // Invalid UTF-8 → CPU fallback
            let bad = [0xFF, 0xFE, 0xFD];
            let dev = parse_device_from_config(bad.as_ptr(), bad.len());
            assert!(matches!(dev, Device::Cpu));
        }
    }

    #[test]
    fn test_ffi_config_json_cpu_explicit() {
        unsafe {
            let cfg = b"{\"device\":\"cpu\"}";
            let dev = parse_device_from_config(cfg.as_ptr(), cfg.len());
            assert!(matches!(dev, Device::Cpu));
        }
    }

    #[test]
    fn test_ffi_operations_after_deinit() {
        unsafe {
            // Init then deinit
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);
            sonata_pipeline_deinit();

            // All ops should return NOT_INITIALIZED
            let mut buf = [0u8; 64];
            let mut tl: usize = 64;
            let ret = sonata_stt(std::ptr::null(), 0, buf.as_mut_ptr(), &mut tl);
            assert_eq!(ret, SC_ERR_NOT_INITIALIZED);

            let mut audio = [0f32; 100];
            let mut al: usize = 100;
            let ret = sonata_tts(
                b"hi".as_ptr(), 2,
                std::ptr::null(), 1.0,
                audio.as_mut_ptr(), &mut al,
            );
            assert_eq!(ret, SC_ERR_NOT_INITIALIZED);

            let mut emb = [0f32; 192];
            let ret = sonata_speaker_encode(std::ptr::null(), 0, emb.as_mut_ptr());
            assert_eq!(ret, SC_ERR_NOT_INITIALIZED);
        }
    }
}
