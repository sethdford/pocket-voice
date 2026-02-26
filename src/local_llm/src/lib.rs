// pocket_llm — On-device LLM inference via candle-transformers Llama.
//
// Loads a quantized GGUF or safetensors Llama model (1B-3B params)
// and exposes a C-ABI streaming token generation interface.
//
// This enables fully offline operation: Mic → STT → Local LLM → TTS → Speaker
// with zero network dependency.
//
// C-ABI:
//   pocket_llm_create(model_path, tokenizer_path) → engine
//   pocket_llm_set_prompt(engine, system, user) → 0
//   pocket_llm_step(engine) → token_id (0 = EOS, -1 = error)
//   pocket_llm_get_token(engine, buf, buf_size) → n_bytes
//   pocket_llm_destroy(engine)

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as llama_model;
use std::ffi::{CStr, c_char, c_int, c_void};

const MAX_SEQ_LEN: usize = 4096;
const DEFAULT_TEMP: f64 = 0.7;
const DEFAULT_TOP_P: f64 = 0.9;

struct LlamaEngine {
    model: llama_model::Llama,
    cache: llama_model::Cache,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    tokens: Vec<u32>,
    pos: usize,
    last_token_text: String,
    temperature: f64,
    top_p: f64,
    done: bool,
    eos_token_id: u32,
}

fn top_p_sample(logits: &Tensor, temperature: f64, top_p: f64) -> candle_core::Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let logits = if temperature > 0.0 {
        logits.affine(1.0 / temperature, 0.0)?
    } else {
        logits
    };

    let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
    let mut probs_vec: Vec<(usize, f32)> = probs
        .to_vec1::<f32>()?
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    probs_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cum_prob = 0.0f64;
    let mut candidates = Vec::new();
    for (idx, prob) in &probs_vec {
        cum_prob += *prob as f64;
        candidates.push((*idx, *prob));
        if cum_prob >= top_p {
            break;
        }
    }

    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    let r: f32 = rand_f32() * total;
    let mut acc = 0.0f32;
    for (idx, prob) in &candidates {
        acc += prob;
        if acc >= r {
            return Ok(*idx as u32);
        }
    }
    Ok(candidates.last().map(|(i, _)| *i as u32).unwrap_or(0))
}

fn rand_f32() -> f32 {
    use std::time::SystemTime;
    let t = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    ((t ^ (t >> 16)) & 0xFFFF) as f32 / 65535.0
}

impl LlamaEngine {
    fn new(model_id: &str, tokenizer_path: Option<&str>) -> anyhow::Result<Self> {
        eprintln!("[pocket_llm] Loading model: {}", model_id);

        #[cfg(feature = "metal")]
        let device = Device::new_metal(0).unwrap_or_else(|e| {
            eprintln!("[pocket_llm] Metal unavailable ({}), using CPU", e);
            Device::Cpu
        });
        #[cfg(not(feature = "metal"))]
        let device = Device::Cpu;
        eprintln!("[pocket_llm] Device: {:?}", device);

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_id.to_string());

        let config_path = repo.get("config.json")?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: llama_model::LlamaConfig = serde_json::from_str(&config_str)?;
        let config = config.into_config(false);

        let tokenizer_file = if let Some(tp) = tokenizer_path {
            std::path::PathBuf::from(tp)
        } else {
            repo.get("tokenizer.json")?
        };
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;

        let weight_files: Vec<std::path::PathBuf> = {
            let mut files = Vec::new();
            if let Ok(f) = repo.get("model.safetensors") {
                files.push(f);
            } else {
                for i in 1..=4 {
                    let name = format!("model-{:05}-of-{:05}.safetensors", i, 4);
                    match repo.get(&name) {
                        Ok(f) => files.push(f),
                        Err(_) => break,
                    }
                }
                if files.is_empty() {
                    anyhow::bail!("No safetensors weight files found");
                }
            }
            files
        };

        let dtype = if device.is_metal() { DType::BF16 } else { DType::F32 };
        let weight_refs: Vec<&std::path::Path> = weight_files.iter().map(|p| p.as_path()).collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_refs, dtype, &device)? };

        let cache = llama_model::Cache::new(true, dtype, &config, &device)?;
        let model = llama_model::Llama::load(vb, &config)?;
        eprintln!("[pocket_llm] Model loaded successfully");

        let eos_id = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|end_of_text|>"))
            .or_else(|| tokenizer.token_to_id("<|eot_id|>"))
            .unwrap_or(2);

        Ok(LlamaEngine {
            model,
            cache,
            tokenizer,
            device,
            tokens: Vec::new(),
            pos: 0,
            last_token_text: String::new(),
            temperature: DEFAULT_TEMP,
            top_p: DEFAULT_TOP_P,
            done: true,
            eos_token_id: eos_id,
        })
    }

    fn set_prompt(&mut self, system: &str, user: &str) -> anyhow::Result<()> {
        self.done = false;
        self.pos = 0;
        self.tokens.clear();
        self.last_token_text.clear();

        let prompt = if system.is_empty() {
            format!("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", user)
        } else {
            format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                system, user
            )
        };

        let encoding = self.tokenizer.encode(prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenize: {}", e))?;
        self.tokens = encoding.get_ids().to_vec();
        eprintln!("[pocket_llm] Prompt: {} tokens", self.tokens.len());

        let input = Tensor::new(self.tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let _logits = self.model.forward(&input, 0, &mut self.cache)?;
        self.pos = self.tokens.len();

        Ok(())
    }

    fn step(&mut self) -> i32 {
        if self.done { return 0; }
        if self.pos >= MAX_SEQ_LEN {
            self.done = true;
            return 0;
        }

        let last_tok = *self.tokens.last().unwrap_or(&0);
        let input = match Tensor::new(&[last_tok], &self.device) {
            Ok(t) => match t.unsqueeze(0) {
                Ok(t) => t,
                Err(_) => { self.done = true; return -1; }
            },
            Err(_) => { self.done = true; return -1; }
        };

        let logits = match self.model.forward(&input, self.pos, &mut self.cache) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[pocket_llm] Forward error: {}", e);
                self.done = true;
                return -1;
            }
        };

        let logits_last = match logits.i((.., logits.dim(1).unwrap_or(1) - 1, ..)) {
            Ok(l) => l,
            Err(_) => { self.done = true; return -1; }
        };

        let token_id = match top_p_sample(&logits_last, self.temperature, self.top_p) {
            Ok(t) => t,
            Err(_) => { self.done = true; return -1; }
        };

        if token_id == self.eos_token_id {
            self.done = true;
            return 0;
        }

        self.tokens.push(token_id);
        self.pos += 1;

        self.last_token_text = self.tokenizer
            .decode(&[token_id], false)
            .unwrap_or_default();

        token_id as i32
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// C-ABI FFI
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn pocket_llm_create(
    model_id: *const c_char,
    tokenizer_path: *const c_char,
) -> *mut c_void {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let model = if model_id.is_null() {
            "meta-llama/Llama-3.2-1B-Instruct"
        } else {
            match unsafe { CStr::from_ptr(model_id) }.to_str() {
                Ok(s) if !s.is_empty() => s,
                _ => "meta-llama/Llama-3.2-1B-Instruct",
            }
        };

        let tok_path = if tokenizer_path.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(tokenizer_path) }.to_str().ok()
        };

        match LlamaEngine::new(model, tok_path) {
            Ok(engine) => Box::into_raw(Box::new(engine)) as *mut c_void,
            Err(e) => {
                eprintln!("[pocket_llm] Failed to create: {}", e);
                std::ptr::null_mut()
            }
        }
    })).unwrap_or_else(|_| {
        eprintln!("[pocket_llm] Create panicked");
        std::ptr::null_mut()
    })
}

#[no_mangle]
pub extern "C" fn pocket_llm_destroy(engine: *mut c_void) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !engine.is_null() {
            unsafe { drop(Box::from_raw(engine as *mut LlamaEngine)) };
        }
    }));
}

#[no_mangle]
pub extern "C" fn pocket_llm_set_prompt(
    engine: *mut c_void,
    system: *const c_char,
    user: *const c_char,
) -> c_int {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if engine.is_null() || user.is_null() { return -1; }
        let eng = unsafe { &mut *(engine as *mut LlamaEngine) };

        let sys_str = if system.is_null() {
            ""
        } else {
            unsafe { CStr::from_ptr(system) }.to_str().unwrap_or("")
        };
        let user_str = match unsafe { CStr::from_ptr(user) }.to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        };

        match eng.set_prompt(sys_str, user_str) {
            Ok(()) => 0,
            Err(e) => { eprintln!("[pocket_llm] set_prompt error: {}", e); -1 }
        }
    })).unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn pocket_llm_step(engine: *mut c_void) -> c_int {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if engine.is_null() { return -1; }
        let eng = unsafe { &mut *(engine as *mut LlamaEngine) };
        eng.step()
    })).unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn pocket_llm_get_token(
    engine: *mut c_void,
    buf: *mut c_char,
    buf_size: c_int,
) -> c_int {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if engine.is_null() || buf.is_null() || buf_size <= 0 { return 0; }
        let eng = unsafe { &*(engine as *const LlamaEngine) };
        let bytes = eng.last_token_text.as_bytes();
        let copy = bytes.len().min((buf_size - 1) as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, copy);
            *buf.add(copy) = 0;
        }
        copy as c_int
    })).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn pocket_llm_is_done(engine: *mut c_void) -> c_int {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if engine.is_null() { return 1; }
        let eng = unsafe { &*(engine as *const LlamaEngine) };
        if eng.done { 1 } else { 0 }
    })).unwrap_or(1)
}

#[no_mangle]
pub extern "C" fn pocket_llm_set_temperature(engine: *mut c_void, temp: f32) -> c_int {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if engine.is_null() { return -1; }
        let eng = unsafe { &mut *(engine as *mut LlamaEngine) };
        eng.temperature = temp as f64;
        0
    })).unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn pocket_llm_reset(engine: *mut c_void) -> c_int {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if engine.is_null() { return -1; }
        let eng = unsafe { &mut *(engine as *mut LlamaEngine) };
        eng.tokens.clear();
        eng.pos = 0;
        eng.done = true;
        eng.last_token_text.clear();
        0
    })).unwrap_or(-1)
}
