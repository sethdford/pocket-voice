// Copyright (c) pocket-voice contributors, all rights reserved.
// Rust FFI wrapper around Kyutai's moshi STT, compiled as a cdylib.
//
// Provides C-callable functions for streaming speech-to-text on Apple Silicon
// via candle + Metal. Zero Python in the inference path.
//
// Architecture:
//   CoreAudio (C) → capture_ring → pocket_stt_process_frame()
//   → candle Metal GPU inference → text out
//
// Build: cargo build --release (metal feature is default)
// Output: target/release/libpocket_stt.dylib

use candle::{DType, Device, Tensor};
use std::ffi::{CStr, c_char, c_double, c_float, c_int, c_void};
use std::ptr;

// ─── Model Config ──────────────────────────────────────────────────────────────
// Matches the config.json format from HuggingFace kyutai/stt-*-candle repos.

#[derive(Debug, serde::Deserialize)]
struct SttConfig {
    audio_silence_prefix_seconds: f64,
    audio_delay_seconds: f64,
}

#[derive(Debug, serde::Deserialize)]
struct Config {
    mimi_name: String,
    tokenizer_name: String,
    card: usize,
    text_card: usize,
    dim: usize,
    n_q: usize,
    context: usize,
    max_period: f64,
    num_heads: usize,
    num_layers: usize,
    causal: bool,
    stt_config: SttConfig,
}

impl Config {
    fn model_config(&self, vad: bool) -> moshi::lm::Config {
        let lm_cfg = moshi::transformer::Config {
            d_model: self.dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            dim_feedforward: self.dim * 4,
            causal: self.causal,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: self.context,
            max_period: self.max_period as usize,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: moshi::NormType::RmsNorm,
            positional_embedding: moshi::transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096 * 4,
            shared_cross_attn: false,
        };
        let extra_heads = if vad {
            Some(moshi::lm::ExtraHeadsConfig {
                num_heads: 4,
                dim: 6,
            })
        } else {
            None
        };
        moshi::lm::Config {
            transformer: lm_cfg,
            depformer: None,
            audio_vocab_size: self.card + 1,
            text_in_vocab_size: self.text_card + 1,
            text_out_vocab_size: self.text_card,
            audio_codebooks: self.n_q,
            conditioners: Default::default(),
            extra_heads,
        }
    }
}

// ─── STT Engine ────────────────────────────────────────────────────────────────

struct WordInfo {
    text: String,
    start_time: f64,
    end_time: f64,
}

struct SttEngine {
    state: moshi::asr::State,
    text_tokenizer: sentencepiece::SentencePieceProcessor,
    words: Vec<WordInfo>,
    vad_probs: Vec<Vec<f32>>,
    pending_word: Option<(String, f64)>,
    has_vad: bool,
    config: Config,
}

impl SttEngine {
    fn load(
        hf_repo: &str,
        model_path: &str,
        enable_vad: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = if candle::utils::metal_is_available() {
            Device::new_metal(0)?
        } else {
            eprintln!("[pocket_stt] Metal not available, falling back to CPU");
            Device::Cpu
        };

        eprintln!("[pocket_stt] Device: {:?}", device);
        eprintln!("[pocket_stt] Loading model: {} ({})", hf_repo, model_path);

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(hf_repo.to_string());

        let config_file = repo.get("config.json")?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;

        let tokenizer_file = repo.get(&config.tokenizer_name)?;
        let model_file = repo.get(model_path)?;
        let mimi_file = repo.get(&config.mimi_name)?;

        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&tokenizer_file)?;

        let is_quantized = model_file
            .to_str()
            .map_or(false, |s| s.ends_with(".gguf"));

        eprintln!(
            "[pocket_stt] Loading LM (quantized={}, vad={})...",
            is_quantized, enable_vad
        );

        let lm = if is_quantized {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &model_file,
                &device,
            )?;
            moshi::lm::LmModel::new(
                &config.model_config(enable_vad),
                moshi::nn::MaybeQuantizedVarBuilder::Quantized(vb),
            )?
        } else {
            let dtype = device.bf16_default_to_f32();
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, &device)?
            };
            moshi::lm::LmModel::new(
                &config.model_config(enable_vad),
                moshi::nn::MaybeQuantizedVarBuilder::Real(vb),
            )?
        };

        eprintln!("[pocket_stt] Loading Mimi audio tokenizer...");
        let mimi_path = mimi_file
            .to_str()
            .ok_or("Mimi file path is not valid UTF-8")?;
        let audio_tokenizer = moshi::mimi::load(mimi_path, Some(32), &device)?;

        let asr_delay = (config.stt_config.audio_delay_seconds * 12.5) as usize;
        let mut state = moshi::asr::State::new(1, asr_delay, 0., audio_tokenizer, lm)?;

        // Warmup: compile Metal shaders and pre-allocate buffers
        eprintln!("[pocket_stt] Warming up (4 frames)...");
        for _ in 0..4 {
            let pcm = Tensor::zeros((1, 1, 1920), DType::F32, &device)?;
            let _ = state.step_pcm(pcm, None, &().into(), |_, _, _| ());
        }
        state.reset()?;
        eprintln!("[pocket_stt] Ready.");

        Ok(SttEngine {
            state,
            text_tokenizer,
            words: Vec::new(),
            vad_probs: Vec::new(),
            pending_word: None,
            has_vad: enable_vad,
            config,
        })
    }

    fn process_msgs(&mut self, msgs: Vec<moshi::asr::AsrMsg>) {
        for msg in msgs {
            match msg {
                moshi::asr::AsrMsg::Step { prs, .. } => {
                    self.vad_probs = prs;
                }
                moshi::asr::AsrMsg::Word {
                    tokens,
                    start_time,
                    ..
                } => {
                    let word = self
                        .text_tokenizer
                        .decode_piece_ids(&tokens)
                        .unwrap_or_default()
                        .replace('\u{2581}', " ");

                    if let Some((prev_word, prev_start)) = self.pending_word.take() {
                        self.words.push(WordInfo {
                            text: prev_word,
                            start_time: prev_start,
                            end_time: start_time,
                        });
                    }
                    self.pending_word = Some((word, start_time));
                }
                moshi::asr::AsrMsg::EndWord { stop_time, .. } => {
                    if let Some((word, start)) = self.pending_word.take() {
                        self.words.push(WordInfo {
                            text: word,
                            start_time: start,
                            end_time: stop_time,
                        });
                    }
                }
            }
        }
    }

    fn process_frame(&mut self, pcm: &[f32]) -> Result<usize, Box<dyn std::error::Error>> {
        self.words.clear();
        self.vad_probs.clear();

        let device = self.state.device().clone();
        let pcm_tensor = Tensor::from_slice(pcm, (1, 1, pcm.len()), &device)?;

        let msgs = self
            .state
            .step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;

        self.process_msgs(msgs);
        Ok(self.words.len())
    }

    fn flush_tail(&mut self) -> Result<usize, Box<dyn std::error::Error>> {
        self.words.clear();
        self.vad_probs.clear();

        let suffix_len =
            (self.config.stt_config.audio_delay_seconds * 24000.0) as usize + 24000;
        let device = self.state.device().clone();
        let frame_size = 1920;
        let mut remaining = suffix_len;

        while remaining > 0 {
            let n = std::cmp::min(remaining, frame_size);
            let pcm = Tensor::zeros((1, 1, n), DType::F32, &device)?;
            let msgs = self
                .state
                .step_pcm(pcm, None, &().into(), |_, _, _| ())?;
            self.process_msgs(msgs);
            remaining -= n;
        }

        // Emit any pending word
        if let Some((word, start)) = self.pending_word.take() {
            self.words.push(WordInfo {
                text: word,
                start_time: start,
                end_time: -1.0,
            });
        }

        Ok(self.words.len())
    }

    fn reset_streaming(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.state.reset()?;
        self.words.clear();
        self.vad_probs.clear();
        self.pending_word = None;

        // Feed silence prefix so the model is in the right state
        let silence_len =
            (self.config.stt_config.audio_silence_prefix_seconds * 24000.0) as usize;
        if silence_len > 0 {
            let device = self.state.device().clone();
            let frame_size = 1920;
            let mut remaining = silence_len;
            while remaining > 0 {
                let n = std::cmp::min(remaining, frame_size);
                let pcm = Tensor::zeros((1, 1, n), DType::F32, &device)?;
                let _ = self
                    .state
                    .step_pcm(pcm, None, &().into(), |_, _, _| ())?;
                remaining -= n;
            }
        }

        Ok(())
    }
}

// ─── C FFI ─────────────────────────────────────────────────────────────────────
// All functions use catch_unwind to prevent panics from crossing the FFI boundary.

/// Truncate byte slice to a valid UTF-8 boundary.
fn utf8_safe_len(bytes: &[u8], max: usize) -> usize {
    let limit = std::cmp::min(bytes.len(), max);
    // Walk backwards from limit to find a char boundary
    let s = unsafe { std::str::from_utf8_unchecked(&bytes[..bytes.len()]) };
    let truncated = &s[..s.floor_char_boundary(limit)];
    truncated.len()
}

/// Copy a Rust string into a C buffer with null termination and UTF-8 safety.
unsafe fn copy_str_to_c(text: &str, buf: *mut c_char, buf_size: c_int) -> c_int {
    let bytes = text.as_bytes();
    let copy_len = utf8_safe_len(bytes, (buf_size - 1) as usize);
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, copy_len);
        *buf.add(copy_len) = 0;
    }
    copy_len as c_int
}

/// Create a new STT engine. Downloads model from HuggingFace on first call.
/// Returns an opaque pointer, or NULL on failure.
///
/// hf_repo: e.g. "kyutai/stt-1b-en_fr-candle"
/// model_path: e.g. "model.safetensors" or "model.gguf"
/// enable_vad: 1 to enable semantic VAD (only for 1B model), 0 to disable
#[unsafe(no_mangle)]
pub extern "C" fn pocket_stt_create(
    hf_repo: *const c_char,
    model_path: *const c_char,
    enable_vad: c_int,
) -> *mut c_void {
    let result = std::panic::catch_unwind(|| {
        let repo = if hf_repo.is_null() {
            "kyutai/stt-1b-en_fr-candle"
        } else {
            unsafe { CStr::from_ptr(hf_repo) }
                .to_str()
                .unwrap_or("kyutai/stt-1b-en_fr-candle")
        };

        let path = if model_path.is_null() {
            "model.safetensors"
        } else {
            unsafe { CStr::from_ptr(model_path) }
                .to_str()
                .unwrap_or("model.safetensors")
        };

        match SttEngine::load(repo, path, enable_vad != 0) {
            Ok(engine) => Box::into_raw(Box::new(engine)) as *mut c_void,
            Err(e) => {
                eprintln!("[pocket_stt] Failed to create engine: {}", e);
                ptr::null_mut()
            }
        }
    });

    result.unwrap_or(ptr::null_mut())
}

/// Destroy a STT engine, freeing all resources.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_stt_destroy(engine: *mut c_void) {
    if engine.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(|| {
        unsafe {
            drop(Box::from_raw(engine as *mut SttEngine));
        }
    });
}

/// Process one frame of PCM audio (float32, 24kHz, mono).
/// Recommended frame size: 1920 samples (80ms).
/// Returns the number of recognized words, or -1 on error.
/// Words can be retrieved with pocket_stt_get_word / pocket_stt_get_all_text.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_stt_process_frame(
    engine: *mut c_void,
    pcm: *const c_float,
    num_samples: c_int,
) -> c_int {
    if engine.is_null() || pcm.is_null() || num_samples <= 0 || num_samples > 192000 {
        return -1;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &mut *(engine as *mut SttEngine) };
        let pcm = unsafe { std::slice::from_raw_parts(pcm, num_samples as usize) };

        match engine.process_frame(pcm) {
            Ok(n) => n as c_int,
            Err(e) => {
                eprintln!("[pocket_stt] process_frame error: {}", e);
                -1
            }
        }
    }));

    result.unwrap_or(-1)
}

/// Feed silence to flush remaining text after speech ends.
/// The model has an inherent delay; this processes enough silence to
/// extract all pending text. Returns number of words, or -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_stt_flush(engine: *mut c_void) -> c_int {
    if engine.is_null() {
        return -1;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &mut *(engine as *mut SttEngine) };

        match engine.flush_tail() {
            Ok(n) => n as c_int,
            Err(e) => {
                eprintln!("[pocket_stt] flush error: {}", e);
                -1
            }
        }
    }));

    result.unwrap_or(-1)
}

/// Get the i-th word from the last process_frame/flush call.
/// Copies the word text into buf (null-terminated, UTF-8 safe).
/// Writes start/end timestamps (seconds) if pointers are non-null.
/// end_time is -1.0 if the word boundary hasn't been seen yet.
/// Returns the byte length of the word, or -1 if index is out of range.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_stt_get_word(
    engine: *mut c_void,
    index: c_int,
    buf: *mut c_char,
    buf_size: c_int,
    start_time: *mut c_double,
    end_time: *mut c_double,
) -> c_int {
    if engine.is_null() || buf.is_null() || index < 0 || buf_size <= 0 {
        return -1;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &*(engine as *const SttEngine) };

        if index as usize >= engine.words.len() {
            return -1;
        }

        let word = &engine.words[index as usize];

        unsafe {
            if !start_time.is_null() {
                *start_time = word.start_time;
            }
            if !end_time.is_null() {
                *end_time = word.end_time;
            }
            copy_str_to_c(&word.text, buf, buf_size)
        }
    }));

    result.unwrap_or(-1)
}

/// Get all recognized text from the last process_frame/flush call
/// as a single concatenated string. Returns byte length, or -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_stt_get_all_text(
    engine: *mut c_void,
    buf: *mut c_char,
    buf_size: c_int,
) -> c_int {
    if engine.is_null() || buf.is_null() || buf_size <= 0 {
        return -1;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &*(engine as *const SttEngine) };
        let text: String = engine.words.iter().map(|w| w.text.as_str()).collect();
        unsafe { copy_str_to_c(&text, buf, buf_size) }
    }));

    result.unwrap_or(-1)
}

/// Get semantic VAD probability for the given time horizon.
/// Horizons for the 1B model: 0=0.5s, 1=1.0s, 2=2.0s, 3=3.0s
/// Returns probability of NO voice activity (higher = more likely silent).
/// Returns -1.0 if VAD is not enabled or horizon is out of range.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_stt_get_vad_prob(
    engine: *mut c_void,
    horizon: c_int,
) -> c_float {
    if engine.is_null() || horizon < 0 {
        return -1.0;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &*(engine as *const SttEngine) };

        if engine.vad_probs.is_empty() || horizon as usize >= engine.vad_probs.len() {
            return -1.0;
        }

        engine.vad_probs[horizon as usize]
            .first()
            .copied()
            .unwrap_or(-1.0)
    }));

    result.unwrap_or(-1.0)
}

/// Returns 1 if the engine was created with VAD enabled, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_stt_has_vad(engine: *mut c_void) -> c_int {
    if engine.is_null() {
        return 0;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &*(engine as *const SttEngine) };
        engine.has_vad as c_int
    }));
    result.unwrap_or(0)
}

/// Reset streaming state for a new utterance.
/// Processes the silence prefix internally.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_stt_reset(engine: *mut c_void) {
    if engine.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &mut *(engine as *mut SttEngine) };
        if let Err(e) = engine.reset_streaming() {
            eprintln!("[pocket_stt] reset error: {}", e);
        }
    }));
}

/// Returns the expected frame size in samples (1920 = 80ms at 24kHz).
#[unsafe(no_mangle)]
pub extern "C" fn pocket_stt_frame_size() -> c_int {
    1920
}

/// Returns the expected sample rate in Hz (24000).
#[unsafe(no_mangle)]
pub extern "C" fn pocket_stt_sample_rate() -> c_int {
    24000
}

/// Returns the model's audio delay in seconds.
/// Text output lags behind audio input by this amount.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_stt_audio_delay(engine: *mut c_void) -> c_double {
    if engine.is_null() {
        return 0.0;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &*(engine as *const SttEngine) };
        engine.config.stt_config.audio_delay_seconds
    }));
    result.unwrap_or(0.0)
}
