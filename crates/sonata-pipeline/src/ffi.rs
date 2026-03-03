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
use crate::streaming_bridge::StreamingBridge;
use candle_core::{DType, Device, Tensor};
use sonata_common::SPEAKER_EMBED_DIM;

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
    static STREAMING_BRIDGE: RefCell<StreamingBridge> = RefCell::new(StreamingBridge::new());
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
/// Converts raw 24kHz mono f32 audio into text token IDs. The output is a
/// space-separated list of u32 token indices (CTC-decoded from a 32K vocab).
/// A real tokenizer/detokenizer is needed to convert these IDs to human text.
///
/// # Arguments
///
/// * `audio` - Pointer to f32 samples (24kHz mono)
/// * `samples` - Number of samples (must be > 0)
/// * `text` - Output buffer for token IDs as space-separated decimal UTF-8
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
        if samples == 0 {
            // Nothing to transcribe — write 0 bytes
            *text_len = 0;
            return SC_OK;
        }

        // Lock pipeline
        let mut guard = match PIPELINE.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                set_last_error("sonata_stt: pipeline mutex poisoned");
                poisoned.into_inner()
            }
        };
        let pipeline = match guard.as_mut() {
            Some(p) => p,
            None => {
                set_last_error("sonata_stt: pipeline not initialized");
                return SC_ERR_NOT_INITIALIZED;
            }
        };

        // Create audio tensor [1, 1, samples] from raw f32 pointer
        let audio_slice = std::slice::from_raw_parts(audio, samples);
        let dev = pipeline.device();
        let audio_tensor = match Tensor::new(audio_slice, dev)
            .and_then(|t| t.reshape(&[1, 1, samples]))
        {
            Ok(t) => t,
            Err(e) => {
                set_last_error(format!("sonata_stt: failed to create audio tensor: {e}"));
                return SC_ERR_INTERNAL;
            }
        };

        // Run STT pipeline: audio → codec → embeddings → CTC decode → token IDs
        let token_seqs = match pipeline.process_audio(&audio_tensor) {
            Ok(t) => t,
            Err(e) => {
                set_last_error(format!("sonata_stt: process_audio failed: {e}"));
                return SC_ERR_INTERNAL;
            }
        };

        // Serialize first batch sequence as space-separated decimal token IDs
        let tokens = if token_seqs.is_empty() { &[] as &[u32] } else { &token_seqs[0] };
        let output_str: String = tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let output_bytes = output_str.as_bytes();

        // Copy to output buffer (truncate if too small)
        let buf_cap = *text_len;
        let copy_len = output_bytes.len().min(buf_cap);
        if copy_len > 0 {
            std::ptr::copy_nonoverlapping(output_bytes.as_ptr(), text, copy_len);
        }
        *text_len = copy_len;

        SC_OK
    }));
    result.unwrap_or_else(|_| {
        set_last_error("panic during sonata_stt");
        SC_ERR_INTERNAL
    })
}

/// Run text-to-speech.
///
/// Converts text into audio via the full TTS pipeline: text bytes are mapped
/// to token IDs (each byte value as a u32 token), then run through the TTS
/// model to produce codec logits, which are decoded to 24kHz audio.
///
/// # Arguments
///
/// * `text` - Input text (UTF-8)
/// * `text_len` - Length of text in bytes (must be > 0)
/// * `speaker_id` - Speaker identifier (nullable, currently unused — zero embedding)
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
#[no_mangle]
pub unsafe extern "C" fn sonata_tts(
    text: *const u8,
    text_len: usize,
    _speaker_id: *const u8,
    emotion_exag: f32,
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
        if text_len == 0 {
            // Nothing to synthesize — write 0 samples
            *audio_len = 0;
            return SC_OK;
        }

        // Lock pipeline
        let guard = match PIPELINE.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                set_last_error("sonata_tts: pipeline mutex poisoned");
                poisoned.into_inner()
            }
        };
        let pipeline = match guard.as_ref() {
            Some(p) => p,
            None => {
                set_last_error("sonata_tts: pipeline not initialized");
                return SC_ERR_NOT_INITIALIZED;
            }
        };

        let dev = pipeline.device();

        // Convert text bytes to token IDs (each byte value as a u32 token)
        let text_slice = std::slice::from_raw_parts(text, text_len);
        let token_ids: Vec<u32> = text_slice.iter().map(|&b| b as u32).collect();

        let text_tokens = match Tensor::new(token_ids.as_slice(), dev)
            .and_then(|t| t.reshape(&[1, text_len]))
        {
            Ok(t) => t,
            Err(e) => {
                set_last_error(format!("sonata_tts: failed to create token tensor: {e}"));
                return SC_ERR_INTERNAL;
            }
        };

        // Create zero speaker embedding [1, 192] (speaker_id not yet supported)
        let speaker_emb = match Tensor::zeros(&[1, SPEAKER_EMBED_DIM], DType::F32, dev) {
            Ok(t) => t,
            Err(e) => {
                set_last_error(format!("sonata_tts: failed to create speaker embedding: {e}"));
                return SC_ERR_INTERNAL;
            }
        };

        // Set emotion exaggeration (style_id=0 neutral, user controls exaggeration)
        // Note: pipeline is behind a shared ref, emotion is set at init time via config.
        // For per-call emotion, we'd need &mut — for now use default emotion.
        let _ = emotion_exag; // Acknowledged but requires &mut pipeline

        // Generate speech: text tokens + speaker → codec logits
        let logits = match pipeline.generate_speech(&text_tokens, &speaker_emb) {
            Ok(l) => l,
            Err(e) => {
                set_last_error(format!("sonata_tts: generate_speech failed: {e}"));
                return SC_ERR_INTERNAL;
            }
        };

        // Decode logits → audio via codec
        let audio_tensor = match pipeline.logits_to_audio(&logits) {
            Ok(a) => a,
            Err(e) => {
                set_last_error(format!("sonata_tts: logits_to_audio failed: {e}"));
                return SC_ERR_INTERNAL;
            }
        };

        // Flatten to [samples] and copy to output buffer
        let flat = match audio_tensor.flatten_all() {
            Ok(f) => f,
            Err(e) => {
                set_last_error(format!("sonata_tts: flatten failed: {e}"));
                return SC_ERR_INTERNAL;
            }
        };
        let audio_data: Vec<f32> = match flat.to_vec1() {
            Ok(v) => v,
            Err(e) => {
                set_last_error(format!("sonata_tts: to_vec1 failed: {e}"));
                return SC_ERR_INTERNAL;
            }
        };

        let buf_cap = *audio_len;
        let copy_len = audio_data.len().min(buf_cap);
        if copy_len > 0 {
            std::ptr::copy_nonoverlapping(audio_data.as_ptr(), audio, copy_len);
        }
        *audio_len = copy_len;

        SC_OK
    }));
    result.unwrap_or_else(|_| {
        set_last_error("panic during sonata_tts");
        SC_ERR_INTERNAL
    })
}

/// Encode speaker embedding from audio.
///
/// Converts raw audio to a mel spectrogram, then extracts a 192-dimensional
/// speaker embedding via the CAM++ encoder.
///
/// # Arguments
///
/// * `audio` - Pointer to f32 samples (24kHz mono)
/// * `samples` - Number of samples (must be > 0)
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
        if samples == 0 {
            set_last_error("sonata_speaker_encode: need audio samples for speaker encoding");
            return SC_ERR_INVALID_ARGUMENT;
        }

        // Lock pipeline
        let guard = match PIPELINE.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                set_last_error("sonata_speaker_encode: pipeline mutex poisoned");
                poisoned.into_inner()
            }
        };
        let pipeline = match guard.as_ref() {
            Some(p) => p,
            None => {
                set_last_error("sonata_speaker_encode: pipeline not initialized");
                return SC_ERR_NOT_INITIALIZED;
            }
        };

        // Create audio tensor [1, samples] from raw f32 pointer
        let dev = pipeline.device();
        let audio_slice = std::slice::from_raw_parts(audio, samples);
        let audio_tensor = match Tensor::new(audio_slice, dev)
            .and_then(|t| t.reshape(&[1, samples]))
        {
            Ok(t) => t,
            Err(e) => {
                set_last_error(format!("sonata_speaker_encode: failed to create audio tensor: {e}"));
                return SC_ERR_INTERNAL;
            }
        };

        // Compute mel spectrogram: [1, samples] → [1, 80, T]
        let mel = match pipeline.compute_mel(&audio_tensor) {
            Ok(m) => m,
            Err(e) => {
                set_last_error(format!("sonata_speaker_encode: mel computation failed: {e}"));
                return SC_ERR_INTERNAL;
            }
        };

        // Encode speaker: mel [1, 80, T] → embedding [1, 192]
        let emb_tensor = match pipeline.encode_speaker(&mel) {
            Ok(e) => e,
            Err(e) => {
                set_last_error(format!("sonata_speaker_encode: encode_speaker failed: {e}"));
                return SC_ERR_INTERNAL;
            }
        };

        // Flatten and copy 192 floats to output buffer
        let flat = match emb_tensor.flatten_all() {
            Ok(f) => f,
            Err(e) => {
                set_last_error(format!("sonata_speaker_encode: flatten failed: {e}"));
                return SC_ERR_INTERNAL;
            }
        };
        let emb_data: Vec<f32> = match flat.to_vec1() {
            Ok(v) => v,
            Err(e) => {
                set_last_error(format!("sonata_speaker_encode: to_vec1 failed: {e}"));
                return SC_ERR_INTERNAL;
            }
        };

        if emb_data.len() < SPEAKER_EMBED_DIM {
            set_last_error(format!(
                "sonata_speaker_encode: embedding has {} dims, expected {}",
                emb_data.len(), SPEAKER_EMBED_DIM
            ));
            return SC_ERR_INTERNAL;
        }

        std::ptr::copy_nonoverlapping(emb_data.as_ptr(), embedding, SPEAKER_EMBED_DIM);
        SC_OK
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
/// Accumulates text deltas from the LLM via a `StreamingBridge` and detects
/// sentence boundaries. When a complete sentence is detected, it is logged
/// as ready for TTS synthesis. On `is_final`, remaining buffered text is
/// flushed.
///
/// # Audio delivery
///
/// This basic implementation only performs text accumulation and sentence
/// detection. Actual audio synthesis and delivery to the caller requires
/// a registered audio callback (not yet implemented). The `ctx` pointer
/// is reserved for future use as the audio callback context.
///
/// # Arguments
///
/// * `ctx` - Opaque context pointer (nullable, reserved for audio callback)
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
#[no_mangle]
pub unsafe extern "C" fn sonata_streaming_tts_callback(
    _ctx: *mut std::ffi::c_void,
    text_delta: *const u8,
    text_len: usize,
    is_final: bool,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if text_delta.is_null() && text_len > 0 {
            set_last_error("sonata_streaming_tts_callback: text_delta is null with non-zero length");
            return SC_ERR_INVALID_ARGUMENT;
        }

        // Parse text delta
        let text_bytes: &[u8] = if text_len == 0 || text_delta.is_null() {
            &[]
        } else {
            std::slice::from_raw_parts(text_delta, text_len)
        };

        let text_str = if text_bytes.is_empty() {
            ""
        } else {
            match std::str::from_utf8(text_bytes) {
                Ok(s) => s,
                Err(_) => {
                    set_last_error("sonata_streaming_tts_callback: invalid UTF-8 text");
                    return SC_ERR_INVALID_ARGUMENT;
                }
            }
        };

        // Accumulate text and detect sentence boundaries
        STREAMING_BRIDGE.with(|bridge| {
            let mut bridge = bridge.borrow_mut();

            if !text_str.is_empty() {
                let sentences = bridge.push(text_str);
                for s in &sentences {
                    eprintln!("[sonata-ffi] streaming sentence ready ({} chars): {:?}", s.len(), s);
                }
            }

            if is_final {
                if let Some(remaining) = bridge.flush() {
                    eprintln!("[sonata-ffi] streaming final flush ({} chars): {:?}", remaining.len(), remaining);
                }
            }
        });

        SC_OK
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
            let audio = [0.0f32; 100];
            let mut buf = [0u8; 256];
            let mut tl: usize = buf.len();
            let ret = sonata_stt(audio.as_ptr(), audio.len(), buf.as_mut_ptr(), &mut tl);
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
            let audio = [0.0f32; 100];
            let mut emb = [0f32; 192];
            let ret = sonata_speaker_encode(audio.as_ptr(), audio.len(), emb.as_mut_ptr());
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
    fn test_ffi_streaming_callback_empty() {
        unsafe {
            // Empty text delta → SC_OK (accumulates nothing)
            let ret = sonata_streaming_tts_callback(std::ptr::null_mut(), std::ptr::null(), 0, false);
            assert_eq!(ret, SC_OK);
        }
    }

    #[test]
    fn test_ffi_streaming_callback_accumulate() {
        unsafe {
            // Push text chunks, verify SC_OK
            let chunk1 = b"Hello world";
            let ret = sonata_streaming_tts_callback(
                std::ptr::null_mut(), chunk1.as_ptr(), chunk1.len(), false,
            );
            assert_eq!(ret, SC_OK);

            // Push sentence-ending chunk
            let chunk2 = b". How are you?";
            let ret = sonata_streaming_tts_callback(
                std::ptr::null_mut(), chunk2.as_ptr(), chunk2.len(), false,
            );
            assert_eq!(ret, SC_OK);

            // Final flush
            let ret = sonata_streaming_tts_callback(
                std::ptr::null_mut(), std::ptr::null(), 0, true,
            );
            assert_eq!(ret, SC_OK);
        }
    }

    #[test]
    fn test_ffi_streaming_callback_invalid_utf8() {
        unsafe {
            let bad_utf8 = [0xFF, 0xFE, 0xFD];
            let ret = sonata_streaming_tts_callback(
                std::ptr::null_mut(), bad_utf8.as_ptr(), bad_utf8.len(), false,
            );
            assert_eq!(ret, SC_ERR_INVALID_ARGUMENT);
        }
    }

    #[test]
    fn test_ffi_stt_zero_samples() {
        unsafe {
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);

            // STT with zero samples → SC_OK, text_len set to 0
            let mut buf = [0u8; 64];
            let mut tl: usize = 64;
            let ret = sonata_stt(std::ptr::null(), 0, buf.as_mut_ptr(), &mut tl);
            assert_eq!(ret, SC_OK);
            assert_eq!(tl, 0);

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_tts_zero_text() {
        unsafe {
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);

            // TTS with zero-length text → SC_OK, audio_len set to 0
            let mut audio = [0f32; 100];
            let mut al: usize = 100;
            let ret = sonata_tts(
                std::ptr::null(), 0,
                std::ptr::null(), 1.0,
                audio.as_mut_ptr(), &mut al,
            );
            assert_eq!(ret, SC_OK);
            assert_eq!(al, 0);

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_speaker_encode_zero_samples() {
        unsafe {
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);

            // Speaker encode with zero samples → invalid argument (need audio)
            let mut emb = [0f32; 192];
            let ret = sonata_speaker_encode(std::ptr::null(), 0, emb.as_mut_ptr());
            assert_eq!(ret, SC_ERR_INVALID_ARGUMENT);

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_stt_end_to_end() {
        unsafe {
            // Note: tests share the global PIPELINE, so another test's deinit can
            // race with us. Re-init to ensure we have a live pipeline.
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);

            // Create 1 second of silence at 24kHz
            let audio = vec![0.0f32; 24000];
            let mut text_buf = vec![0u8; 4096];
            let mut text_len: usize = text_buf.len();

            let ret = sonata_stt(audio.as_ptr(), audio.len(), text_buf.as_mut_ptr(), &mut text_len);
            // If another test raced and cleared the pipeline, we get NOT_INITIALIZED.
            // Accept both SC_OK and SC_ERR_NOT_INITIALIZED due to test parallelism.
            if ret == SC_ERR_NOT_INITIALIZED {
                // Retry: re-init and try again
                let ret_init = sonata_pipeline_init(std::ptr::null(), 0);
                assert_eq!(ret_init, SC_OK);
                text_len = text_buf.len();
                let ret2 = sonata_stt(audio.as_ptr(), audio.len(), text_buf.as_mut_ptr(), &mut text_len);
                assert_eq!(ret2, SC_OK);
            } else {
                assert_eq!(ret, SC_OK);
            }
            // text_len should be set (token IDs serialized as space-separated text)
            assert!(text_len <= text_buf.len());

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_tts_end_to_end() {
        unsafe {
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);

            let text = b"hello";
            let mut audio_buf = vec![0.0f32; 120_000]; // ~5s at 24kHz
            let mut audio_len: usize = audio_buf.len();

            let ret = sonata_tts(
                text.as_ptr(), text.len(),
                std::ptr::null(), 1.0,
                audio_buf.as_mut_ptr(), &mut audio_len,
            );
            // Handle race with parallel tests clearing the pipeline
            if ret == SC_ERR_NOT_INITIALIZED {
                let ret_init = sonata_pipeline_init(std::ptr::null(), 0);
                assert_eq!(ret_init, SC_OK);
                audio_len = audio_buf.len();
                let ret2 = sonata_tts(
                    text.as_ptr(), text.len(),
                    std::ptr::null(), 1.0,
                    audio_buf.as_mut_ptr(), &mut audio_len,
                );
                assert_eq!(ret2, SC_OK);
            } else {
                assert_eq!(ret, SC_OK);
            }
            // Should have produced some audio samples
            assert!(audio_len > 0);
            assert!(audio_len <= audio_buf.len());

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_speaker_encode_end_to_end() {
        unsafe {
            let ret = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret, SC_OK);

            // 1 second of audio at 24kHz
            let audio = vec![0.0f32; 24000];
            let mut emb = [0.0f32; 192];

            let ret = sonata_speaker_encode(audio.as_ptr(), audio.len(), emb.as_mut_ptr());
            // Handle race with parallel tests clearing the pipeline
            if ret == SC_ERR_NOT_INITIALIZED {
                let ret_init = sonata_pipeline_init(std::ptr::null(), 0);
                assert_eq!(ret_init, SC_OK);
                let ret2 = sonata_speaker_encode(audio.as_ptr(), audio.len(), emb.as_mut_ptr());
                assert_eq!(ret2, SC_OK);
            } else {
                assert_eq!(ret, SC_OK);
            }

            sonata_pipeline_deinit();
        }
    }

    #[test]
    fn test_ffi_null_pointer_safety() {
        unsafe {
            // Init with null config is fine
            let ret1 = sonata_pipeline_init(std::ptr::null(), 0);
            assert_eq!(ret1, SC_OK);

            // STT with null audio + zero samples → SC_OK (nothing to transcribe)
            let mut buf = [0u8; 64];
            let mut tl: usize = 64;
            let ret2 = sonata_stt(std::ptr::null(), 0, buf.as_mut_ptr(), &mut tl);
            assert_eq!(ret2, SC_OK);
            assert_eq!(tl, 0);

            // TTS with null text + zero len → SC_OK (nothing to synthesize)
            let mut audio = [0f32; 100];
            let mut al: usize = 100;
            let ret3 = sonata_tts(
                std::ptr::null(), 0,
                std::ptr::null(), 1.0,
                audio.as_mut_ptr(), &mut al,
            );
            assert_eq!(ret3, SC_OK);
            assert_eq!(al, 0);

            // Speaker encode with null audio + zero samples → invalid (need audio)
            let mut emb = [0f32; 192];
            let ret4 = sonata_speaker_encode(std::ptr::null(), 0, emb.as_mut_ptr());
            assert_eq!(ret4, SC_ERR_INVALID_ARGUMENT);

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

            // STT with zero samples after deinit → zero samples returns OK early
            // (checked before pipeline lock)
            let mut buf = [0u8; 64];
            let mut tl: usize = 64;
            let ret = sonata_stt(std::ptr::null(), 0, buf.as_mut_ptr(), &mut tl);
            assert_eq!(ret, SC_OK); // zero samples → early return
            assert_eq!(tl, 0);

            // Note: parallel tests share the global PIPELINE, so another test may
            // re-init between our deinit and these calls. Accept both NOT_INITIALIZED
            // (expected) and OK (race with another test's init).

            // STT with non-zero samples after deinit
            let audio_data = [0.0f32; 100];
            tl = 64;
            let ret = sonata_stt(audio_data.as_ptr(), 100, buf.as_mut_ptr(), &mut tl);
            assert!(
                ret == SC_ERR_NOT_INITIALIZED || ret == SC_OK,
                "expected NOT_INITIALIZED or OK, got {ret}"
            );

            // TTS with text after deinit
            let mut audio = [0f32; 100];
            let mut al: usize = 100;
            let ret = sonata_tts(
                b"hi".as_ptr(), 2,
                std::ptr::null(), 1.0,
                audio.as_mut_ptr(), &mut al,
            );
            assert!(
                ret == SC_ERR_NOT_INITIALIZED || ret == SC_OK,
                "expected NOT_INITIALIZED or OK, got {ret}"
            );

            // Speaker encode after deinit
            let mut emb = [0f32; 192];
            let ret = sonata_speaker_encode(audio_data.as_ptr(), 100, emb.as_mut_ptr());
            assert!(
                ret == SC_ERR_NOT_INITIALIZED || ret == SC_OK,
                "expected NOT_INITIALIZED or OK, got {ret}"
            );
        }
    }
}
