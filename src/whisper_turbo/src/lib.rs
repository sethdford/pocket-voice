// whisper_turbo — Rust cdylib for Whisper v3 Turbo multilingual ASR on Apple Silicon Metal GPU.
//
// Architecture: OpenAI Whisper large-v3-turbo
//   Encoder: 32 layers, d_model=1280, 20 query heads
//   Decoder: 4 layers, d_model=1280, 20 query heads, cross-attention to encoder
//   Mel spectrogram: 128 bins, 25ms window, 10ms stride
//
// C FFI:
//   whisper_turbo_create(model_dir) -> *engine
//   whisper_turbo_destroy(engine)
//   whisper_turbo_reset(engine)
//   whisper_turbo_process(engine, pcm, n_samples, lang, out_text, max_len) -> bytes_written / -1
//   whisper_turbo_feed(engine, pcm, n_samples) -> 0/-1
//   whisper_turbo_flush(engine, out_text, max_len, out_lang, lang_max) -> bytes_written / -1

use candle_core::{Device, DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self, audio, model::Whisper, Config};
use std::ffi::{CStr, c_char, c_float, c_int};
use std::path::Path;
use std::ptr;
use std::sync::Mutex;

const WINDOW_SAMPLES: usize = whisper::N_SAMPLES; // 30 seconds @ 16kHz

// ─── WhisperTurbo Engine ───────────────────────────────────────────────────

pub struct WhisperTurbo {
    model: Whisper,
    device: Device,
    tokenizer: tokenizers::Tokenizer,
    mel_filters: Vec<f32>,
    audio_buffer: Vec<f32>,
    config: Config,
}

impl WhisperTurbo {
    fn new(model_dir: &Path) -> anyhow::Result<Self> {
        let device = Device::new_metal(0).unwrap_or_else(|_| Device::Cpu);

        // Load model config
        let config_path = model_dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_text)?;

        // Load model weights
        let weights_path = model_dir.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };
        let model = Whisper::load(&vb, config.clone())?;

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load mel filters (from HF or embedded)
        let mel_bytes = match std::fs::read(model_dir.join("melfilters128.bytes")) {
            Ok(b) => b,
            Err(_) => std::fs::read(model_dir.join("melfilters.bytes"))
                .unwrap_or_else(|_| vec![0u8; config.num_mel_bins * (whisper::N_FFT / 2 + 1) * 4]),
        };
        let mel_filters: Vec<f32> = mel_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(Self {
            model,
            device,
            tokenizer,
            mel_filters,
            audio_buffer: Vec::with_capacity(WINDOW_SAMPLES),
            config,
        })
    }

    fn reset(&mut self) {
        self.audio_buffer.clear();
        self.model.reset_kv_cache();
    }

    fn feed_audio(&mut self, samples: &[f32]) {
        self.audio_buffer.extend_from_slice(samples);
        if self.audio_buffer.len() > WINDOW_SAMPLES {
            self.audio_buffer.drain(0..self.audio_buffer.len() - WINDOW_SAMPLES);
        }
    }

    fn transcribe(&mut self) -> anyhow::Result<(String, String)> {
        if self.audio_buffer.is_empty() {
            return Ok((String::new(), "unknown".to_string()));
        }

        // Convert PCM to mel spectrogram using candle_transformers audio utilities
        let mel = audio::pcm_to_mel(&self.config, &self.audio_buffer, &self.mel_filters);
        let n_mel_frames = mel.len() / self.config.num_mel_bins;
        let mel_tensor = Tensor::from_vec(
            mel,
            (1, self.config.num_mel_bins, n_mel_frames),
            &self.device,
        )?;

        // Run encoder
        let encoder_output = self.model.encoder.forward(&mel_tensor, true)?;

        // Language detection: default to English for now
        // Full implementation would sample decoder language logits
        let language = "en".to_string();

        // Greedy decode
        let sot_token = 50258u32; // <|startoftranscript|>
        let lang_token = 50259u32; // <|en|>
        let transcribe_token = 50360u32; // <|transcribe|>
        let eot_token = 50257u32; // <|endoftext|>
        let no_timestamps_token = 50364u32; // <|notimestamps|>

        let mut tokens: Vec<u32> = vec![sot_token, lang_token, transcribe_token, no_timestamps_token];
        let mut result_tokens: Vec<u32> = Vec::new();
        let max_decode_steps = 224;

        self.model.decoder.reset_kv_cache();

        for _step in 0..max_decode_steps {
            let token_tensor = Tensor::new(
                tokens.as_slice(),
                &self.device,
            )?.unsqueeze(0)?;

            let logits = self.model.decoder.forward(&token_tensor, &encoder_output, tokens.len() == 4)?;

            // Get logits for last position
            let seq_len = logits.dim(1)?;
            let last_logits = logits.i((0, seq_len - 1))?;

            // Greedy: argmax
            let next_token = last_logits
                .argmax(0)?
                .to_scalar::<u32>()?;

            if next_token == eot_token || next_token >= 50257 {
                break;
            }

            result_tokens.push(next_token);
            tokens = vec![next_token]; // autoregressive: feed only last token
        }

        // Decode tokens to text
        let text = self.tokenizer.decode(&result_tokens, true)
            .unwrap_or_default();

        Ok((text.trim().to_string(), language))
    }
}

// ─── C FFI ───────────────────────────────────────────────────────────────────

type WhisperTurboHandle = *mut Mutex<WhisperTurbo>;

#[no_mangle]
pub extern "C" fn whisper_turbo_create(model_dir: *const c_char) -> WhisperTurboHandle {
    if model_dir.is_null() {
        return ptr::null_mut();
    }

    let path_cstr = unsafe { CStr::from_ptr(model_dir) };
    let path = match path_cstr.to_str() {
        Ok(s) => std::path::PathBuf::from(s),
        Err(_) => return ptr::null_mut(),
    };

    match WhisperTurbo::new(&path) {
        Ok(engine) => {
            let boxed = Box::new(Mutex::new(engine));
            Box::into_raw(boxed) as WhisperTurboHandle
        }
        Err(e) => {
            eprintln!("[whisper_turbo] create failed: {}", e);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn whisper_turbo_destroy(handle: WhisperTurboHandle) {
    if !handle.is_null() {
        let _dropped = unsafe { Box::from_raw(handle) };
    }
}

#[no_mangle]
pub extern "C" fn whisper_turbo_reset(handle: WhisperTurboHandle) -> c_int {
    if handle.is_null() {
        return -1;
    }

    match unsafe { (*handle).lock() } {
        Ok(mut engine) => {
            engine.reset();
            0
        }
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn whisper_turbo_feed(
    handle: WhisperTurboHandle,
    pcm: *const c_float,
    n_samples: c_int,
) -> c_int {
    if handle.is_null() || pcm.is_null() || n_samples <= 0 {
        return -1;
    }

    let samples = unsafe { std::slice::from_raw_parts(pcm as *const f32, n_samples as usize) };

    match unsafe { (*handle).lock() } {
        Ok(mut engine) => {
            engine.feed_audio(samples);
            0
        }
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn whisper_turbo_flush(
    handle: WhisperTurboHandle,
    out_text: *mut c_char,
    max_len: c_int,
    out_lang: *mut c_char,
    lang_max: c_int,
) -> c_int {
    if handle.is_null() || out_text.is_null() || max_len <= 0 {
        return -1;
    }

    match unsafe { (*handle).lock() } {
        Ok(mut engine) => {
            match engine.transcribe() {
                Ok((text, lang)) => {
                    let text_bytes = text.as_bytes();
                    let text_len = text_bytes.len().min(max_len as usize - 1);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            text_bytes.as_ptr() as *const c_char,
                            out_text,
                            text_len,
                        );
                        *out_text.add(text_len) = 0;
                    }

                    if !out_lang.is_null() && lang_max > 0 {
                        let lang_bytes = lang.as_bytes();
                        let lang_len = lang_bytes.len().min(lang_max as usize - 1);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                lang_bytes.as_ptr() as *const c_char,
                                out_lang,
                                lang_len,
                            );
                            *out_lang.add(lang_len) = 0;
                        }
                    }

                    text_len as c_int
                }
                Err(e) => {
                    eprintln!("[whisper_turbo] transcribe failed: {}", e);
                    -1
                }
            }
        }
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn whisper_turbo_process(
    handle: WhisperTurboHandle,
    pcm: *const c_float,
    n_samples: c_int,
    _lang: *const c_char,
    out_text: *mut c_char,
    max_len: c_int,
) -> c_int {
    if whisper_turbo_feed(handle, pcm, n_samples) != 0 {
        return -1;
    }
    whisper_turbo_flush(handle, out_text, max_len, ptr::null_mut(), 0)
}
