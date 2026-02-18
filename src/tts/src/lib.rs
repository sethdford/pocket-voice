// pocket_tts_rs — Rust FFI wrapper for DSM TTS 1.6B on Apple Silicon.
//
// Compiled as a cdylib, callable from the C orchestrator (pocket_voice_pipeline.c).
// Uses candle + Metal for GPU inference, moshi::tts_streaming for the DSM model,
// and moshi::mimi for streaming audio decode.
//
// Architecture:
//   C orchestrator → pocket_tts_rs_set_text() → tokenize + queue
//   C orchestrator → pocket_tts_rs_step()     → LM step + Mimi decode
//   C orchestrator → pocket_tts_rs_get_audio() → read PCM from output buffer
//
// Build: cargo build --release (metal feature is default)
// Output: target/release/libpocket_tts_rs.dylib

use candle::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use moshi::tts_streaming::{AllowedTokens, Config as TtsConfig, State as TtsState};
use moshi::StreamTensor;
use std::collections::VecDeque;
use std::ffi::{CStr, c_char, c_float, c_int, c_void};
use std::ptr;

const FRAME_SAMPLES: usize = 1920; // 80ms at 24kHz
const SAMPLE_RATE: i32 = 24000;
const MAX_STEP_IDX: usize = 2000; // ~160s at 12.5 Hz
const DEFAULT_N_Q: usize = 24;

// ─── Config Parsing ──────────────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct RawConfig {
    mimi_name: String,
    tokenizer_name: String,
    #[serde(default = "default_moshi_name")]
    moshi_name: String,

    // LM architecture
    #[serde(default = "default_dim")]
    dim: usize,
    #[serde(default = "default_num_heads")]
    num_heads: usize,
    #[serde(default = "default_num_layers")]
    num_layers: usize,
    #[serde(default = "default_card")]
    card: usize,
    #[serde(default = "default_text_card")]
    text_card: usize,
    #[serde(default = "default_n_q")]
    n_q: usize,
    #[serde(default = "default_context")]
    context: usize,
    #[serde(default = "default_max_period")]
    max_period: f64,
    #[serde(default)]
    causal: Option<bool>,

    // Depformer
    #[serde(default)]
    depformer_dim: Option<usize>,
    #[serde(default)]
    depformer_num_heads: Option<usize>,
    #[serde(default)]
    depformer_num_layers: Option<usize>,
    #[serde(default)]
    depformer_context: Option<usize>,

    // Conditioners
    #[serde(default)]
    speaker_cond: Option<bool>,
}

fn default_moshi_name() -> String {
    "model.safetensors".into()
}
fn default_dim() -> usize {
    2048
}
fn default_num_heads() -> usize {
    32
}
fn default_num_layers() -> usize {
    24
}
fn default_card() -> usize {
    2048
}
fn default_text_card() -> usize {
    8002
}
fn default_n_q() -> usize {
    32
}
fn default_context() -> usize {
    4096
}
fn default_max_period() -> f64 {
    10000.0
}

impl RawConfig {
    fn build_lm_config(&self, n_q: usize) -> moshi::lm::Config {
        let transformer_cfg = moshi::transformer::Config {
            d_model: self.dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            dim_feedforward: self.dim * 4,
            causal: self.causal.unwrap_or(true),
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
            max_seq_len: self.context,
            shared_cross_attn: false,
        };

        let depformer = self.depformer_dim.map(|dim| {
            let dep_d = dim;
            let dep_h = self.depformer_num_heads.unwrap_or(16);
            let dep_l = self.depformer_num_layers.unwrap_or(6);
            let dep_ctx = self.depformer_context.unwrap_or(8);
            moshi::lm::DepFormerConfig {
                transformer: moshi::transformer::Config {
                    d_model: dep_d,
                    num_heads: dep_h,
                    num_layers: dep_l,
                    dim_feedforward: dep_d * 4,
                    causal: true,
                    norm_first: true,
                    bias_ff: false,
                    bias_attn: false,
                    layer_scale: None,
                    context: dep_ctx,
                    max_period: 10000,
                    use_conv_block: false,
                    use_conv_bias: true,
                    cross_attention: None,
                    gating: Some(candle_nn::Activation::Silu),
                    norm: moshi::NormType::RmsNorm,
                    positional_embedding: moshi::transformer::PositionalEmbedding::Rope,
                    conv_layout: false,
                    conv_kernel_size: 3,
                    kv_repeat: 1,
                    max_seq_len: dep_ctx,
                    shared_cross_attn: false,
                },
                num_slices: n_q,
                low_rank_embeddings: None,
            }
        });

        let conditioners = if self.speaker_cond.unwrap_or(false) {
            // Multi-speaker model with voice conditioning
            Default::default()
        } else {
            Default::default()
        };

        moshi::lm::Config {
            transformer: transformer_cfg,
            depformer,
            audio_vocab_size: self.card + 1,
            text_in_vocab_size: self.text_card + 1,
            text_out_vocab_size: self.text_card,
            audio_codebooks: n_q,
            conditioners,
            extra_heads: None,
        }
    }
}

// ─── TTS Engine ──────────────────────────────────────────────────────────────

struct TtsEngine {
    state: TtsState,
    mimi: moshi::mimi::Mimi,
    tokenizer: sentencepiece::SentencePieceProcessor,
    device: Device,

    // Text word queue: tokenized words waiting to be fed to the model
    word_queue: VecDeque<Vec<u32>>,
    current_word: Option<Vec<u32>>,
    current_word_idx: usize,

    // TTS streaming config
    tts_config: TtsConfig,

    // Audio output buffer (PCM samples from Mimi decode)
    audio_buf: Vec<f32>,
    audio_read_pos: usize,

    // Generation state
    text_done: bool,
    gen_done: bool,
    extra_steps_remaining: usize,
    prev_text_token: u32,

    // Stored for re-creating State on reset (State has no reset() method)
    lm_config: moshi::lm::Config,
    model_path: std::path::PathBuf,
    #[allow(dead_code)]
    n_q: usize,
}

impl TtsEngine {
    fn load(
        hf_repo: &str,
        voice_path: Option<&str>,
        n_q: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = if candle::utils::metal_is_available() {
            Device::new_metal(0)?
        } else {
            eprintln!("[pocket_tts_rs] Metal not available, falling back to CPU");
            Device::Cpu
        };

        eprintln!("[pocket_tts_rs] Device: {:?}", device);
        eprintln!("[pocket_tts_rs] Loading TTS model: {}", hf_repo);

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(hf_repo.to_string());

        // Download and parse config
        let config_file = repo.get("config.json")?;
        let raw_config: RawConfig =
            serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;

        let tokenizer_file = repo.get(&raw_config.tokenizer_name)?;
        let model_file = repo.get(&raw_config.moshi_name)?;
        let mimi_file = repo.get(&raw_config.mimi_name)?;

        // Load text tokenizer
        eprintln!("[pocket_tts_rs] Loading tokenizer...");
        let tokenizer = sentencepiece::SentencePieceProcessor::open(&tokenizer_file)?;

        // Build LM config
        let lm_config = raw_config.build_lm_config(n_q);
        let dtype = device.bf16_default_to_f32();
        eprintln!(
            "[pocket_tts_rs] Loading LM (dim={}, layers={}, n_q={})...",
            raw_config.dim, raw_config.num_layers, n_q
        );

        // Load Mimi audio codec
        eprintln!("[pocket_tts_rs] Loading Mimi audio codec...");
        let mimi_path = mimi_file
            .to_str()
            .ok_or("Mimi file path is not valid UTF-8")?;
        let mut mimi = moshi::mimi::load(mimi_path, Some(n_q), &device)?;

        // Load voice conditioning if voice path provided
        let ca_src = if let Some(_voice) = voice_path {
            // Voice conditioning would load speaker embeddings here.
            // For now, use None (unconditioned generation).
            // Full implementation would use SpeakerEncoder to encode voice audio.
            None
        } else {
            None
        };

        // Create TTS streaming state
        let tts_config = TtsConfig::v202501();
        let state = Self::create_tts_state(
            &lm_config, &model_file, &device, dtype, ca_src, &tts_config,
        )?;

        // Warmup: run a few Mimi decode steps to compile Metal shaders
        eprintln!("[pocket_tts_rs] Warming up Metal shaders...");
        mimi.reset_state();
        for _ in 0..4 {
            let zeros = Tensor::zeros((1, n_q, 1), candle::DType::U32, &device);
            if let Ok(codes) = zeros {
                let st = StreamTensor::from_tensor(codes);
                let mask = moshi::StreamMask::from(());
                let _ = mimi.decode_step(&st, &mask);
            }
        }
        mimi.reset_state();

        let bos_token = tts_config.text_bos_token;
        eprintln!("[pocket_tts_rs] Ready (n_q={}).", n_q);

        Ok(TtsEngine {
            state,
            mimi,
            tokenizer,
            device,
            word_queue: VecDeque::new(),
            current_word: None,
            current_word_idx: 0,
            tts_config,
            audio_buf: Vec::with_capacity(FRAME_SAMPLES * 100),
            audio_read_pos: 0,
            text_done: false,
            gen_done: false,
            extra_steps_remaining: 0,
            prev_text_token: bos_token,
            lm_config,
            model_path: model_file,
            n_q,
        })
    }

    fn create_tts_state(
        lm_config: &moshi::lm::Config,
        model_path: &std::path::Path,
        device: &Device,
        dtype: candle::DType,
        ca_src: Option<moshi::transformer::CaSrc>,
        tts_config: &TtsConfig,
    ) -> Result<TtsState, Box<dyn std::error::Error>> {
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[model_path], dtype, device)?
        };
        let model = moshi::lm::LmModel::new(
            lm_config,
            moshi::nn::MaybeQuantizedVarBuilder::Real(vb),
        )?;

        let audio_lp = LogitsProcessor::from_sampling(
            299792458,
            Sampling::TopK {
                k: 250,
                temperature: 0.8,
            },
        );
        let text_lp = LogitsProcessor::from_sampling(
            299792458,
            Sampling::TopK {
                k: 50,
                temperature: 0.0,
            },
        );

        Ok(TtsState::new(
            model,
            ca_src,
            MAX_STEP_IDX,
            audio_lp,
            text_lp,
            None,
            tts_config.clone(),
        ))
    }

    fn set_text(&mut self, text: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Tokenize each word separately (DSM interleaver pattern)
        for word in text.split_whitespace() {
            let token_ids = self.tokenizer.encode(word)?;
            let ids: Vec<u32> = token_ids.iter().map(|p| p.id as u32).collect();
            if !ids.is_empty() {
                self.word_queue.push_back(ids);
            }
        }
        Ok(())
    }

    fn set_text_done(&mut self) {
        self.text_done = true;
    }

    fn step(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        if self.gen_done {
            return Ok(true);
        }

        // Determine what text token to feed
        let (text_token, allowed) = self.next_text_token();

        // Run one LM step
        let _out_token = self.state.step(text_token, allowed, None)?;
        self.prev_text_token = text_token;

        // Check for audio output
        if let Some(audio_tokens) = self.state.last_audio_tokens() {
            // Decode audio tokens through Mimi
            let n_codebooks = audio_tokens.len();
            let codes =
                Tensor::from_vec(audio_tokens, (1, n_codebooks, 1), &self.device)?;
            let stream_codes = StreamTensor::from_tensor(codes);
            let mask = moshi::StreamMask::from(());
            let pcm = self.mimi.decode_step(&stream_codes, &mask)?;

            if let Some(pcm_tensor) = pcm.as_option() {
                let pcm_data: Vec<f32> = pcm_tensor
                    .flatten_all()?
                    .to_vec1()?;
                // Clamp to [-1, 1]
                for &s in &pcm_data {
                    self.audio_buf.push(s.clamp(-1.0, 1.0));
                }
            }
        }

        // Check if we're done
        if self.text_done && self.word_queue.is_empty() && self.current_word.is_none() {
            if self.extra_steps_remaining == 0 {
                self.extra_steps_remaining = self.tts_config.extra_steps
                    + self.tts_config.acoustic_delay
                    + self.tts_config.text_audio_delay_in_tokens;
            }
            self.extra_steps_remaining = self.extra_steps_remaining.saturating_sub(1);
            if self.extra_steps_remaining == 0 {
                self.gen_done = true;
            }
        }

        Ok(self.gen_done)
    }

    fn next_text_token(&mut self) -> (u32, AllowedTokens) {
        // If we have a current word being fed token by token
        if let Some(ref word_tokens) = self.current_word {
            if self.current_word_idx < word_tokens.len() {
                let token = word_tokens[self.current_word_idx];
                self.current_word_idx += 1;
                return (token, AllowedTokens::Text(token));
            }
            // Word finished, emit EOP
            self.current_word = None;
            self.current_word_idx = 0;
            return (
                self.tts_config.text_eop_token,
                AllowedTokens::Text(self.tts_config.text_eop_token),
            );
        }

        // Try to get next word from queue
        if let Some(word_tokens) = self.word_queue.pop_front() {
            if word_tokens.len() == 1 {
                let token = word_tokens[0];
                return (token, AllowedTokens::Text(token));
            }
            let token = word_tokens[0];
            self.current_word = Some(word_tokens);
            self.current_word_idx = 1;
            return (token, AllowedTokens::Text(token));
        }

        // No text available — emit padding
        (self.tts_config.text_pad_token, AllowedTokens::PadOrEpad)
    }

    fn get_audio(&mut self, buf: &mut [f32]) -> usize {
        let available = self.audio_buf.len() - self.audio_read_pos;
        let to_copy = available.min(buf.len());
        if to_copy > 0 {
            buf[..to_copy]
                .copy_from_slice(&self.audio_buf[self.audio_read_pos..self.audio_read_pos + to_copy]);
            self.audio_read_pos += to_copy;
        }

        // Compact: if we've consumed most of the buffer, shift remaining data to front
        if self.audio_read_pos > FRAME_SAMPLES * 50 {
            let remaining = self.audio_buf.len() - self.audio_read_pos;
            if remaining > 0 {
                self.audio_buf.copy_within(self.audio_read_pos.., 0);
            }
            self.audio_buf.truncate(remaining);
            self.audio_read_pos = 0;
        }

        to_copy
    }

    fn is_done(&self) -> bool {
        self.gen_done
    }

    fn reset(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.word_queue.clear();
        self.current_word = None;
        self.current_word_idx = 0;
        self.audio_buf.clear();
        self.audio_read_pos = 0;
        self.text_done = false;
        self.gen_done = false;
        self.extra_steps_remaining = 0;
        self.prev_text_token = self.tts_config.text_bos_token;
        self.mimi.reset_state();

        // Re-create TtsState to reset the LM's KV cache and step counter.
        // The model weights are mmap'd so this is fast (~100ms, no download).
        let dtype = self.device.bf16_default_to_f32();
        self.state = Self::create_tts_state(
            &self.lm_config,
            &self.model_path,
            &self.device,
            dtype,
            None,
            &self.tts_config,
        )?;

        Ok(())
    }
}

// ─── C FFI ─────────────────────────────────────────────────────────────────────

/// Create a TTS engine. Downloads model from HuggingFace on first call.
/// Returns an opaque pointer, or NULL on failure.
///
/// hf_repo: e.g. "kyutai/tts-1.6b-en_fr"
/// voice_path: path to voice .safetensors or .wav, or NULL for default
/// n_q: number of audio codebooks (8-32, recommended 24)
#[unsafe(no_mangle)]
pub extern "C" fn pocket_tts_rs_create(
    hf_repo: *const c_char,
    voice_path: *const c_char,
    n_q: c_int,
) -> *mut c_void {
    let result = std::panic::catch_unwind(|| {
        let repo = if hf_repo.is_null() {
            "kyutai/tts-1.6b-en_fr"
        } else {
            unsafe { CStr::from_ptr(hf_repo) }
                .to_str()
                .unwrap_or("kyutai/tts-1.6b-en_fr")
        };

        let voice = if voice_path.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(voice_path) }
                .to_str()
                .ok()
        };

        let nq = if n_q <= 0 { DEFAULT_N_Q } else { n_q as usize };

        match TtsEngine::load(repo, voice, nq) {
            Ok(engine) => Box::into_raw(Box::new(engine)) as *mut c_void,
            Err(e) => {
                eprintln!("[pocket_tts_rs] Failed to create engine: {}", e);
                ptr::null_mut()
            }
        }
    });

    result.unwrap_or(ptr::null_mut())
}

/// Destroy a TTS engine, freeing all resources.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_tts_rs_destroy(engine: *mut c_void) {
    if engine.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(|| {
        unsafe {
            drop(Box::from_raw(engine as *mut TtsEngine));
        }
    });
}

/// Feed text for synthesis. Can be called multiple times as LLM tokens arrive.
/// Text is tokenized and queued internally.
/// Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_tts_rs_set_text(
    engine: *mut c_void,
    text: *const c_char,
) -> c_int {
    if engine.is_null() || text.is_null() {
        return -1;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &mut *(engine as *mut TtsEngine) };
        let text = unsafe { CStr::from_ptr(text) }
            .to_str()
            .unwrap_or("");

        if text.is_empty() {
            return 0;
        }

        match engine.set_text(text) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[pocket_tts_rs] set_text error: {}", e);
                -1
            }
        }
    }));

    result.unwrap_or(-1)
}

/// Signal that all text has been provided. The engine will generate
/// remaining audio and drain to completion.
/// Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_tts_rs_set_text_done(engine: *mut c_void) -> c_int {
    if engine.is_null() {
        return -1;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &mut *(engine as *mut TtsEngine) };
        engine.set_text_done();
        0
    }));

    result.unwrap_or(-1)
}

/// Run one generation step. Produces ~80ms of audio per step (at 12.5 Hz).
/// Returns: 1 if generation is complete, 0 if more steps needed, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_tts_rs_step(engine: *mut c_void) -> c_int {
    if engine.is_null() {
        return -1;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &mut *(engine as *mut TtsEngine) };

        match engine.step() {
            Ok(done) => {
                if done {
                    1
                } else {
                    0
                }
            }
            Err(e) => {
                eprintln!("[pocket_tts_rs] step error: {}", e);
                -1
            }
        }
    }));

    result.unwrap_or(-1)
}

/// Read decoded PCM audio from the output buffer.
/// Copies up to max_samples float32 samples into pcm_buf.
/// Returns the number of samples copied, or -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_tts_rs_get_audio(
    engine: *mut c_void,
    pcm_buf: *mut c_float,
    max_samples: c_int,
) -> c_int {
    if engine.is_null() || pcm_buf.is_null() || max_samples <= 0 {
        return -1;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &mut *(engine as *mut TtsEngine) };
        let buf = unsafe { std::slice::from_raw_parts_mut(pcm_buf, max_samples as usize) };
        engine.get_audio(buf) as c_int
    }));

    result.unwrap_or(-1)
}

/// Returns 1 if generation is complete, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_tts_rs_is_done(engine: *mut c_void) -> c_int {
    if engine.is_null() {
        return 1;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &*(engine as *const TtsEngine) };
        engine.is_done() as c_int
    }));

    result.unwrap_or(1)
}

/// Reset the engine for a new utterance. Clears all queued text and audio,
/// and re-creates the LM state (KV cache, step counter). Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn pocket_tts_rs_reset(engine: *mut c_void) -> c_int {
    if engine.is_null() {
        return -1;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &mut *(engine as *mut TtsEngine) };
        match engine.reset() {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[pocket_tts_rs] reset error: {}", e);
                -1
            }
        }
    }));
    result.unwrap_or(-1)
}

/// Returns the output sample rate in Hz (24000).
#[unsafe(no_mangle)]
pub extern "C" fn pocket_tts_rs_sample_rate() -> c_int {
    SAMPLE_RATE
}

/// Returns the frame size in samples (1920 = 80ms at 24kHz).
#[unsafe(no_mangle)]
pub extern "C" fn pocket_tts_rs_frame_size() -> c_int {
    FRAME_SAMPLES as c_int
}
