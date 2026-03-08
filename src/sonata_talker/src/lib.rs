// sonata_talker — 500M Talker for on-device Speech-to-Speech.
//
// Architecture (Thinker-Talker, from Qwen2.5-Omni):
//   Temporal Transformer: 12 layers, 768 dim, 12 heads (4 KV groups), RoPE
//   Depth Transformer: 6 layers, 512 dim, 8 heads
//   Audio Embedder: 8 codebooks x 2048 entries x 768 dim
//   Text Embedder: 32K vocab x 768 dim
//   Thinker Projector: LLM hidden -> 768
//   LM Heads: 8 codebook heads x 2048 vocab
//
// Total: ~512M params (256MB INT4)
//
// C FFI:
//   sonata_talker_create(weights, config) -> *engine
//   sonata_talker_step(engine, user_codes, *out_codes) -> 0/1/-1
//   sonata_talker_reset(engine)
//   sonata_talker_destroy(engine)

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, VarBuilder};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;

pub mod temporal;
pub mod depth;
pub mod embedder;
pub mod stream;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TalkerConfig {
    // Temporal Transformer
    #[serde(default = "default_d_model")]      pub d_model: usize,        // 768
    #[serde(default = "default_n_temp_layers")] pub n_temporal_layers: usize, // 12
    #[serde(default = "default_n_heads")]       pub n_heads: usize,        // 12
    #[serde(default = "default_n_kv")]          pub n_kv_heads: usize,     // 4
    #[serde(default = "default_d_ff")]          pub d_ff: usize,           // 3072

    // Depth Transformer
    #[serde(default = "default_depth_dim")]     pub depth_dim: usize,      // 512
    #[serde(default = "default_n_depth")]       pub n_depth_layers: usize, // 6
    #[serde(default = "default_depth_heads")]   pub depth_heads: usize,    // 8
    #[serde(default = "default_depth_ff")]      pub depth_d_ff: usize,     // 2048

    // Codec
    #[serde(default = "default_n_codebooks")]   pub n_codebooks: usize,    // 8
    #[serde(default = "default_codebook_size")] pub codebook_size: usize,  // 2048
    #[serde(default = "default_text_vocab")]    pub text_vocab_size: usize,// 32000

    // Thinker projection
    #[serde(default = "default_thinker_dim")]   pub thinker_hidden_dim: usize, // 4096 (Claude hidden)

    // General
    #[serde(default = "default_max_seq")]       pub max_seq_len: usize,    // 4096
    #[serde(default = "default_theta")]         pub rope_theta: f64,       // 10000.0
    #[serde(default = "default_eps")]           pub norm_eps: f64,         // 1e-5
    #[serde(default = "default_frame_rate")]    pub frame_rate_hz: f32,    // 12.5
    #[serde(default = "default_tau")]           pub acoustic_delay: usize, // 1 frame (80ms)
}

fn default_d_model() -> usize { 768 }
fn default_n_temp_layers() -> usize { 12 }
fn default_n_heads() -> usize { 12 }
fn default_n_kv() -> usize { 4 }
fn default_d_ff() -> usize { 3072 }
fn default_depth_dim() -> usize { 512 }
fn default_n_depth() -> usize { 6 }
fn default_depth_heads() -> usize { 8 }
fn default_depth_ff() -> usize { 2048 }
fn default_n_codebooks() -> usize { 8 }
fn default_codebook_size() -> usize { 2048 }
fn default_text_vocab() -> usize { 32000 }
fn default_thinker_dim() -> usize { 4096 }
fn default_max_seq() -> usize { 4096 }
fn default_theta() -> f64 { 10000.0 }
fn default_eps() -> f64 { 1e-5 }
fn default_frame_rate() -> f32 { 12.5 }
fn default_tau() -> usize { 1 }

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            d_model: 768, n_temporal_layers: 12, n_heads: 12, n_kv_heads: 4,
            d_ff: 3072, depth_dim: 512, n_depth_layers: 6, depth_heads: 8,
            depth_d_ff: 2048, n_codebooks: 8, codebook_size: 2048,
            text_vocab_size: 32000, thinker_hidden_dim: 4096, max_seq_len: 4096,
            rope_theta: 10000.0, norm_eps: 1e-5, frame_rate_hz: 12.5,
            acoustic_delay: 1,
        }
    }
}

impl TalkerConfig {
    pub fn head_dim(&self) -> usize { self.d_model / self.n_heads }
    pub fn n_rep(&self) -> usize { self.n_heads / self.n_kv_heads }
    pub fn depth_head_dim(&self) -> usize { self.depth_dim / self.depth_heads }
    // Tokens per timestep: 1 semantic + 7 acoustic per stream x 2 streams + 1 text = 17
    pub fn tokens_per_step(&self) -> usize { self.n_codebooks * 2 + 1 }
}

// ─── TalkerEngine ───────────────────────────────────────────────────────────

pub struct TalkerEngine {
    cfg: TalkerConfig,
    audio_emb: embedder::AudioEmbedder,
    thinker_proj: embedder::ThinkerProjector,
    temporal: temporal::TemporalTransformer,
    depth: depth::DepthTransformer,
    semantic_head: Linear,  // d_model -> codebook_size (semantic code prediction)
    rope_cos: Tensor,
    rope_sin: Tensor,
    k_caches: Vec<Tensor>,
    v_caches: Vec<Tensor>,
    pos: usize,
    device: Device,
}

impl TalkerEngine {
    /// Create with zero weights (for testing).
    pub fn new_zeros(cfg: &TalkerConfig, dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);
        Self::from_vb(cfg, vb, dev)
    }

    /// Create from a VarBuilder (safetensors or zeros).
    pub fn from_vb(cfg: &TalkerConfig, vb: VarBuilder, dev: &Device) -> Result<Self> {
        let (rope_cos, rope_sin) = temporal::precompute_rope_cache(
            cfg.head_dim(), cfg.max_seq_len, cfg.rope_theta, dev,
        )?;
        let mut k_caches = Vec::new();
        let mut v_caches = Vec::new();
        for _ in 0..cfg.n_temporal_layers {
            k_caches.push(Tensor::zeros((1, cfg.n_kv_heads, 0, cfg.head_dim()), DType::F32, dev)?);
            v_caches.push(Tensor::zeros((1, cfg.n_kv_heads, 0, cfg.head_dim()), DType::F32, dev)?);
        }

        Ok(Self {
            audio_emb: embedder::AudioEmbedder::load(cfg, vb.pp("audio_emb"))?,
            thinker_proj: embedder::ThinkerProjector::load(cfg, vb.pp("thinker_proj"))?,
            temporal: temporal::TemporalTransformer::load(cfg, vb.pp("temporal"))?,
            depth: depth::DepthTransformer::load(cfg, vb.pp("depth"))?,
            semantic_head: linear_no_bias(cfg.d_model, cfg.codebook_size, vb.pp("semantic_head"))?,
            rope_cos, rope_sin, k_caches, v_caches,
            pos: 0,
            device: dev.clone(),
            cfg: cfg.clone(),
        })
    }

    /// Load from safetensors weights file.
    pub fn load(weights_path: &str, config_path: &str) -> Result<Self> {
        let dev = Device::new_metal(0).unwrap_or(Device::Cpu);
        let cfg: TalkerConfig = serde_json::from_str(
            &std::fs::read_to_string(config_path)
                .map_err(|e| candle_core::Error::Msg(format!("config read: {}", e)))?,
        ).map_err(|e| candle_core::Error::Msg(format!("config parse: {}", e)))?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &dev)?
        };
        Self::from_vb(&cfg, vb, &dev)
    }

    /// Single inference step:
    /// - Takes user audio codes (8 codebook indices)
    /// - Optionally takes thinker hidden state
    /// - Returns assistant audio codes (8 codebook indices)
    pub fn step(
        &mut self, user_codes: &[u32], thinker_hidden: Option<&Tensor>,
    ) -> Result<Vec<u32>> {
        let codes_t = Tensor::from_vec(user_codes.to_vec(), (1, self.cfg.n_codebooks), &self.device)?;
        let mut input = self.audio_emb.forward(&codes_t)?; // (1, 1, d_model)

        // Add thinker conditioning if available
        if let Some(hidden) = thinker_hidden {
            let proj = self.thinker_proj.forward(hidden)?;
            input = (input + proj)?;
        }

        // Temporal transformer step
        let temporal_out = self.temporal.forward(
            &input, &self.rope_cos, &self.rope_sin, self.pos,
            &mut self.k_caches, &mut self.v_caches,
        )?;

        // Predict semantic code
        let sem_logits = self.semantic_head.forward(&temporal_out)?;
        let sem_code = sem_logits.argmax(D::Minus1)?.squeeze(0)?.squeeze(0)?.to_scalar::<u32>()?;

        // Project to depth dim and generate acoustic codes
        let depth_input = self.depth.project_in.forward(&temporal_out)?;
        let acoustic_codes = self.depth.generate(&depth_input, self.cfg.n_codebooks - 1)?;

        self.pos += 1;

        // Combine: [semantic_code, acoustic_code_0, ..., acoustic_code_6]
        let mut out = vec![sem_code];
        out.extend(acoustic_codes);
        Ok(out)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        for i in 0..self.cfg.n_temporal_layers {
            self.k_caches[i] = Tensor::zeros(
                (1, self.cfg.n_kv_heads, 0, self.cfg.head_dim()),
                DType::F32, &self.device,
            )?;
            self.v_caches[i] = Tensor::zeros(
                (1, self.cfg.n_kv_heads, 0, self.cfg.head_dim()),
                DType::F32, &self.device,
            )?;
        }
        Ok(())
    }
}

// ─── C FFI ──────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn sonata_talker_create(
    weights_path: *const c_char, config_path: *const c_char,
) -> *mut c_void {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let wp = unsafe { CStr::from_ptr(weights_path) }.to_str().ok()?;
        let cp = unsafe { CStr::from_ptr(config_path) }.to_str().ok()?;
        TalkerEngine::load(wp, cp).ok()
    }));
    match result {
        Ok(Some(engine)) => Box::into_raw(Box::new(engine)) as *mut c_void,
        _ => { eprintln!("sonata_talker_create: failed"); ptr::null_mut() }
    }
}

#[no_mangle]
pub extern "C" fn sonata_talker_step(
    handle: *mut c_void, user_codes: *const u32, n_codes: c_int,
    out_codes: *mut u32, out_n: *mut c_int,
) -> c_int {
    if handle.is_null() || user_codes.is_null() || out_codes.is_null() { return -1; }
    let engine = unsafe { &mut *(handle as *mut TalkerEngine) };
    let codes = unsafe { std::slice::from_raw_parts(user_codes, n_codes as usize) };
    match engine.step(codes, None) {
        Ok(out) => {
            let n = out.len().min(8);
            unsafe {
                std::ptr::copy_nonoverlapping(out.as_ptr(), out_codes, n);
                if !out_n.is_null() { *out_n = n as c_int; }
            }
            0
        }
        Err(e) => { eprintln!("talker_step error: {}", e); -1 }
    }
}

#[no_mangle]
pub extern "C" fn sonata_talker_reset(handle: *mut c_void) -> c_int {
    if handle.is_null() { return -1; }
    let engine = unsafe { &mut *(handle as *mut TalkerEngine) };
    match engine.reset() { Ok(_) => 0, Err(_) => -1 }
}

#[no_mangle]
pub extern "C" fn sonata_talker_destroy(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle as *mut TalkerEngine)); }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_talker_engine_create() {
        let dev = &Device::Cpu;
        let cfg = TalkerConfig::default();
        let engine = TalkerEngine::new_zeros(&cfg, dev).unwrap();
        assert_eq!(engine.pos, 0);
    }

    #[test]
    fn test_talker_step_output_shape() {
        let dev = &Device::Cpu;
        let cfg = TalkerConfig::default();
        let mut engine = TalkerEngine::new_zeros(&cfg, dev).unwrap();
        let user_codes = vec![0u32; 8]; // 8 codebook indices
        let out = engine.step(&user_codes, None).unwrap();
        assert_eq!(out.len(), 8); // 8 output codebook indices (1 semantic + 7 acoustic)
    }

    #[test]
    fn test_talker_reset_clears_state() {
        let dev = &Device::Cpu;
        let cfg = TalkerConfig::default();
        let mut engine = TalkerEngine::new_zeros(&cfg, dev).unwrap();
        // Do a few steps
        let _ = engine.step(&[0u32; 8], None);
        let _ = engine.step(&[1u32; 8], None);
        assert!(engine.pos > 0);
        // Reset
        engine.reset().unwrap();
        assert_eq!(engine.pos, 0);
    }

    #[test]
    fn test_talker_with_thinker_hidden() {
        let dev = &Device::Cpu;
        let cfg = TalkerConfig::default();
        let mut engine = TalkerEngine::new_zeros(&cfg, dev).unwrap();
        let user_codes = vec![0u32; 8];
        let thinker_hidden = Tensor::zeros((1, 1, cfg.thinker_hidden_dim), DType::F32, dev).unwrap();
        let out = engine.step(&user_codes, Some(&thinker_hidden)).unwrap();
        assert_eq!(out.len(), 8); // Should produce 8 codes with thinker conditioning
    }
}
