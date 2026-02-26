//! pocket-llm: On-device LLM inference via candle + Metal.
//!
//! Supports Llama-3, SmolLM2, and other HuggingFace Llama-architecture models.
//! Includes multi-turn conversation context, repetition penalty, and
//! auto-detecting chat templates.
//!
//! FFI contract:
//!   pocket_llm_create(repo, model_file) -> *mut c_void
//!   pocket_llm_destroy(engine)
//!   pocket_llm_set_prompt(engine, system, user) -> c_int
//!   pocket_llm_step(engine) -> c_int  (1=token, 0=done, -1=error)
//!   pocket_llm_get_token(engine, buf, size) -> c_int
//!   pocket_llm_is_done(engine) -> c_int
//!   pocket_llm_set_temperature(engine, temp) -> c_int
//!   pocket_llm_reset(engine) -> c_int

use std::ffi::{c_char, c_int, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as model;

use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

const DEFAULT_REPO: &str = "meta-llama/Llama-3.2-3B-Instruct";
const MAX_NEW_TOKENS: usize = 256;
const MAX_CONTEXT_TURNS: usize = 8;

// ─── Chat Template Detection ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum ChatTemplate {
    Llama3,   // <|begin_of_text|><|start_header_id|>...<|eot_id|>
    ChatML,   // <|im_start|>system\n...<|im_end|>
}

fn detect_template(tokenizer: &Tokenizer) -> ChatTemplate {
    if tokenizer.token_to_id("<|start_header_id|>").is_some() {
        ChatTemplate::Llama3
    } else {
        ChatTemplate::ChatML
    }
}

fn detect_eos(tokenizer: &Tokenizer, template: ChatTemplate) -> u32 {
    match template {
        ChatTemplate::Llama3 => {
            tokenizer.token_to_id("<|eot_id|>")
                .or_else(|| tokenizer.token_to_id("<|end_of_text|>"))
                .unwrap_or(128009)
        }
        ChatTemplate::ChatML => {
            tokenizer.token_to_id("<|im_end|>")
                .or_else(|| tokenizer.token_to_id("</s>"))
                .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
                .unwrap_or(2)
        }
    }
}

fn format_prompt(
    template: ChatTemplate,
    system: &str,
    turns: &[(String, String)],
    user: &str,
) -> String {
    match template {
        ChatTemplate::Llama3 => {
            let mut prompt = String::from("<|begin_of_text|>");
            if !system.is_empty() {
                prompt.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                prompt.push_str(system);
                prompt.push_str("<|eot_id|>");
            }
            for (u, a) in turns {
                prompt.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
                prompt.push_str(u);
                prompt.push_str("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
                prompt.push_str(a);
                prompt.push_str("<|eot_id|>");
            }
            prompt.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
            prompt.push_str(user);
            prompt.push_str("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
            prompt
        }
        ChatTemplate::ChatML => {
            let mut prompt = String::new();
            if !system.is_empty() {
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str(system);
                prompt.push_str("<|im_end|>\n");
            }
            for (u, a) in turns {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(u);
                prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
                prompt.push_str(a);
                prompt.push_str("<|im_end|>\n");
            }
            prompt.push_str("<|im_start|>user\n");
            prompt.push_str(user);
            prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
            prompt
        }
    }
}

// ─── Repetition Penalty + Top-p Sampling ────────────────────────────────────

fn apply_repetition_penalty(logits: &mut Vec<f32>, tokens: &[u32], penalty: f32) {
    for &tok in tokens {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

fn sample_top_p(logits_vec: &[f32], temperature: f64, top_p: f64) -> u32 {
    let max_val = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut probs: Vec<(usize, f32)> = logits_vec.iter().enumerate()
        .map(|(i, &v)| {
            let scaled = ((v - max_val) as f64 / temperature) as f32;
            (i, scaled.exp())
        })
        .collect();

    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum > 0.0 {
        for p in probs.iter_mut() { p.1 /= sum; }
    }

    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cum = 0.0f64;
    let mut candidates = Vec::new();
    for &(idx, prob) in &probs {
        cum += prob as f64;
        candidates.push((idx, prob));
        if cum >= top_p { break; }
    }

    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    let r = rand_f32() * total;
    let mut acc = 0.0f32;
    for &(idx, prob) in &candidates {
        acc += prob;
        if acc >= r { return idx as u32; }
    }
    candidates.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

fn rand_f32() -> f32 {
    use std::time::SystemTime;
    let t = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let x = t ^ (t >> 16);
    let x = x.wrapping_mul(0x45d9f3b).wrapping_add(0x12345);
    (x & 0xFFFFFF) as f32 / 16777215.0
}

// ─── Engine ─────────────────────────────────────────────────────────────────

struct LlmEngine {
    llm_model: model::Llama,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    config: model::LlamaConfig,
    template: ChatTemplate,
    eos_token: u32,

    system_prompt: String,
    conversation: Vec<(String, String)>,
    generation_state: Option<GenerationState>,
}

struct GenerationState {
    cache: model::Cache,
    tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    generated_count: usize,
    done: bool,
    last_decoded_len: usize,
    last_step_text: String,
    temperature: f64,
    top_p: f64,
    repetition_penalty: f32,
    current_user_text: String,
}

fn get_device() -> Device {
    #[cfg(feature = "metal")]
    { Device::new_metal(0).unwrap_or(Device::Cpu) }
    #[cfg(not(feature = "metal"))]
    { Device::Cpu }
}

impl LlmEngine {
    fn new(repo_id: &str, _model_file: Option<&str>) -> Result<Self, Box<dyn std::error::Error>> {
        let device = get_device();
        let dtype = if device.is_metal() { DType::BF16 } else { DType::F32 };

        eprintln!("[pocket_llm] Loading {} on {:?} ({:?})", repo_id, device, dtype);

        let api = Api::new()?;
        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

        let config_path = repo.get("config.json")?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: model::LlamaConfig = serde_json::from_str(&config_str)?;

        let tokenizer_path = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("tokenizer: {}", e))?;

        let template = detect_template(&tokenizer);
        let eos_token = detect_eos(&tokenizer, template);
        eprintln!("[pocket_llm] Template: {:?}, EOS token: {}", template, eos_token);

        let filenames = {
            if let Ok(f) = repo.get("model.safetensors") {
                vec![f]
            } else {
                let index_path = repo.get("model.safetensors.index.json")?;
                let index_str = std::fs::read_to_string(&index_path)?;
                let index: serde_json::Value = serde_json::from_str(&index_str)?;
                let weight_map = index["weight_map"].as_object()
                    .ok_or("no weight_map in index")?;
                let mut files: Vec<String> = weight_map.values()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                files.sort();
                files.dedup();
                files.iter().map(|f| repo.get(f)).collect::<Result<Vec<_>, _>>()?
            }
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)?
        };

        eprintln!("[pocket_llm] Building model ({} layers, d={}, vocab={})",
                  config.num_hidden_layers, config.hidden_size, config.vocab_size);

        let llama_config = config.clone().into_config(false);
        let llm_model = model::Llama::load(vb, &llama_config)?;

        eprintln!("[pocket_llm] Ready");

        Ok(LlmEngine {
            llm_model,
            tokenizer,
            device,
            dtype,
            config,
            template,
            eos_token,
            system_prompt: "You are a helpful voice assistant. Keep responses concise and conversational.".to_string(),
            conversation: Vec::new(),
            generation_state: None,
        })
    }

    fn set_system_prompt(&mut self, prompt: &str) {
        self.system_prompt = prompt.to_string();
    }

    fn generate_start(&mut self, user_text: &str) -> Result<(), Box<dyn std::error::Error>> {
        let prompt = format_prompt(
            self.template,
            &self.system_prompt,
            &self.conversation,
            user_text,
        );

        let encoding = self.tokenizer.encode(prompt.as_str(), true)
            .map_err(|e| format!("encode: {}", e))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

        eprintln!("[pocket_llm] Prompt: {} tokens ({} turns + new)",
                  prompt_tokens.len(), self.conversation.len());

        let cache = model::Cache::new(
            false,
            self.dtype,
            &self.config.clone().into_config(false),
            &self.device,
        )?;

        self.generation_state = Some(GenerationState {
            cache,
            tokens: prompt_tokens,
            generated_tokens: Vec::new(),
            generated_count: 0,
            done: false,
            last_decoded_len: 0,
            last_step_text: String::new(),
            temperature: 0.6,
            top_p: 0.9,
            repetition_penalty: 1.15,
            current_user_text: user_text.to_string(),
        });

        Ok(())
    }

    fn step_one(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        let state = self.generation_state.as_mut().ok_or("no active generation")?;
        if state.done { return Ok(false); }

        let (input, index_pos) = if state.generated_count == 0 {
            (Tensor::new(state.tokens.as_slice(), &self.device)?.unsqueeze(0)?, 0)
        } else {
            let last_token = *state.tokens.last().ok_or("empty tokens")?;
            (Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?, state.tokens.len() - 1)
        };

        let logits = self.llm_model.forward(&input, index_pos, &mut state.cache)?;

        let logits_1d = if logits.rank() == 3 {
            let s = logits.squeeze(0)?;
            s.get(s.dim(0)? - 1)?.to_dtype(DType::F32)?
        } else if logits.rank() == 2 {
            logits.get(logits.dim(0)? - 1)?.to_dtype(DType::F32)?
        } else {
            logits.flatten_all()?.to_dtype(DType::F32)?
        };

        let mut logits_vec: Vec<f32> = logits_1d.to_vec1()?;

        // Apply repetition penalty on recent context (last 64 tokens)
        let penalty_window: Vec<u32> = state.tokens.iter()
            .rev().take(64).copied().collect();
        apply_repetition_penalty(&mut logits_vec, &penalty_window, state.repetition_penalty);

        let next_token = sample_top_p(&logits_vec, state.temperature, state.top_p);

        if next_token == self.eos_token || state.generated_count >= MAX_NEW_TOKENS {
            state.done = true;
            state.last_step_text.clear();

            // Save this turn for multi-turn context
            let full_response = self.tokenizer
                .decode(&state.generated_tokens, true)
                .unwrap_or_default();
            if !full_response.is_empty() {
                let user_text = state.current_user_text.clone();
                self.conversation.push((user_text, full_response));
                while self.conversation.len() > MAX_CONTEXT_TURNS {
                    self.conversation.remove(0);
                }
            }

            return Ok(false);
        }

        state.tokens.push(next_token);
        state.generated_tokens.push(next_token);
        state.generated_count += 1;

        let decoded = self.tokenizer
            .decode(&state.generated_tokens, true)
            .unwrap_or_default();

        state.last_step_text.clear();
        if decoded.len() > state.last_decoded_len {
            state.last_step_text = decoded[state.last_decoded_len..].to_string();
            state.last_decoded_len = decoded.len();
        }

        Ok(true)
    }

    fn get_last_token_text(&self) -> &str {
        self.generation_state.as_ref()
            .map(|s| s.last_step_text.as_str())
            .unwrap_or("")
    }

    fn is_done(&self) -> bool {
        self.generation_state.as_ref().map(|s| s.done).unwrap_or(true)
    }

    fn reset_generation(&mut self) {
        self.generation_state = None;
    }

    fn clear_context(&mut self) {
        self.conversation.clear();
    }
}

// ── FFI ─────────────────────────────────────────────────────────────────────

struct SafeEngine(Mutex<LlmEngine>);

fn with_engine<F, T>(engine: *mut c_void, default: T, f: F) -> T
where F: FnOnce(&mut LlmEngine) -> T, T: Copy,
{
    if engine.is_null() { return default; }
    catch_unwind(AssertUnwindSafe(|| {
        let safe = unsafe { &*(engine as *const SafeEngine) };
        if let Ok(mut eng) = safe.0.lock() { f(&mut eng) } else { default }
    })).unwrap_or(default)
}

#[unsafe(no_mangle)]
pub extern "C" fn pocket_llm_create(
    model_id: *const c_char,
    _tokenizer_path: *const c_char,
) -> *mut c_void {
    let result = catch_unwind(|| {
        let repo_str = if model_id.is_null() {
            DEFAULT_REPO
        } else {
            unsafe { CStr::from_ptr(model_id) }.to_str().unwrap_or(DEFAULT_REPO)
        };
        match LlmEngine::new(repo_str, None) {
            Ok(engine) => {
                let safe = Box::new(SafeEngine(Mutex::new(engine)));
                Box::into_raw(safe) as *mut c_void
            }
            Err(e) => {
                eprintln!("[pocket_llm] Create failed: {}", e);
                std::ptr::null_mut()
            }
        }
    });
    result.unwrap_or(std::ptr::null_mut())
}

#[unsafe(no_mangle)]
pub extern "C" fn pocket_llm_destroy(engine: *mut c_void) {
    if engine.is_null() { return; }
    let _ = catch_unwind(|| {
        unsafe { drop(Box::from_raw(engine as *mut SafeEngine)) };
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn pocket_llm_set_prompt(
    engine: *mut c_void, system: *const c_char, user: *const c_char,
) -> c_int {
    if engine.is_null() || user.is_null() { return -1; }
    let result = catch_unwind(|| {
        let safe = unsafe { &*(engine as *const SafeEngine) };
        let user_str = unsafe { CStr::from_ptr(user) }.to_str().unwrap_or("");
        if let Ok(mut eng) = safe.0.lock() {
            if !system.is_null() {
                if let Ok(s) = unsafe { CStr::from_ptr(system) }.to_str() {
                    eng.set_system_prompt(s);
                }
            }
            match eng.generate_start(user_str) {
                Ok(()) => 0,
                Err(e) => { eprintln!("[pocket_llm] set_prompt: {}", e); -1 }
            }
        } else { -1 }
    });
    result.unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub extern "C" fn pocket_llm_step(engine: *mut c_void) -> c_int {
    with_engine(engine, -1, |eng| {
        match eng.step_one() {
            Ok(true) => 1,
            Ok(false) => 0,
            Err(e) => { eprintln!("[pocket_llm] step: {}", e); -1 }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn pocket_llm_get_token(
    engine: *mut c_void, buf: *mut c_char, buf_size: c_int,
) -> c_int {
    if engine.is_null() || buf.is_null() || buf_size <= 0 { return 0; }
    let result = catch_unwind(|| {
        let safe = unsafe { &*(engine as *const SafeEngine) };
        if let Ok(eng) = safe.0.lock() {
            let text = eng.get_last_token_text();
            let n = text.len().min((buf_size as usize).saturating_sub(1));
            if n > 0 {
                let out = unsafe { std::slice::from_raw_parts_mut(buf as *mut u8, buf_size as usize) };
                out[..n].copy_from_slice(&text.as_bytes()[..n]);
                out[n] = 0;
            } else {
                unsafe { *buf = 0 };
            }
            n as c_int
        } else { 0 }
    });
    result.unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn pocket_llm_is_done(engine: *mut c_void) -> c_int {
    with_engine(engine, 1, |eng| if eng.is_done() { 1 } else { 0 })
}

#[unsafe(no_mangle)]
pub extern "C" fn pocket_llm_set_temperature(engine: *mut c_void, temp: f32) -> c_int {
    with_engine(engine, -1, |eng| {
        if let Some(ref mut state) = eng.generation_state {
            state.temperature = temp as f64;
        }
        0
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn pocket_llm_reset(engine: *mut c_void) -> c_int {
    with_engine(engine, -1, |eng| { eng.reset_generation(); 0 })
}

/// Clear conversation history (new conversation).
#[unsafe(no_mangle)]
pub extern "C" fn pocket_llm_clear_context(engine: *mut c_void) -> c_int {
    with_engine(engine, -1, |eng| { eng.clear_context(); eng.reset_generation(); 0 })
}
