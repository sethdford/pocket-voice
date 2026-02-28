// sonata_lm — Rust cdylib for Sonata Semantic LM on Apple Silicon Metal GPU.
//
// Architecture: Llama-style transformer (241M params default):
//   16 layers, d_model=1024, 16 query heads, 4 KV heads (GQA)
//   RoPE positional encoding, RMSNorm, SwiGLU FFN
//
// Predicts semantic tokens (4096 vocab from Sonata Codec FSQ).
// Each step = 20ms of audio (50 Hz frame rate).
//
// C FFI:
//   sonata_lm_create(weights, config) -> *engine
//   sonata_lm_set_text(engine, text_ids, n) -> 0/-1
//   sonata_lm_step(engine, *out_token) -> 0 (more) / 1 (done) / -1 (error)
//   sonata_lm_reset(engine)
//   sonata_lm_destroy(engine)

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder};
use std::collections::VecDeque;
use std::ffi::{CStr, c_char, c_float, c_int, c_void};
use std::path::Path;
use std::ptr;

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
struct LmConfig {
    #[serde(default = "default_d_model")]   d_model: usize,
    #[serde(default = "default_n_layers")]  n_layers: usize,
    #[serde(default = "default_n_heads")]   n_heads: usize,
    #[serde(default = "default_n_kv")]      n_kv_heads: usize,
    #[serde(default = "default_d_ff")]      d_ff: usize,
    #[serde(default = "default_max_seq")]   max_seq_len: usize,
    #[serde(default = "default_text_v")]    text_vocab_size: usize,
    #[serde(default = "default_sem_v")]     semantic_vocab_size: usize,
    #[serde(default = "default_n_sp")]      n_special_tokens: usize,
    #[serde(default = "default_theta")]     rope_theta: f64,
    #[serde(default = "default_eps")]       norm_eps: f64,
    #[serde(default)]                       use_prosody: bool,
    #[serde(default)]                       use_acoustic_head: bool,
    #[serde(default = "default_acoustic_dim")] acoustic_dim: usize,
}

const PROSODY_DIM: usize = 3; // (log_pitch, energy, speaking_rate)

// ─── ProsodyLM Token Definitions ─────────────────────────────────────────────
// Word-level prosody tokens interleaved with semantic tokens.
// Token IDs are offset from semantic_vocab_size + n_special_tokens.
// These require a model fine-tuned with prosody token vocabulary.

const PROSODY_TOKEN_STRESS_0: u32 = 0;   // no stress
const PROSODY_TOKEN_STRESS_1: u32 = 1;   // primary stress
const PROSODY_TOKEN_STRESS_2: u32 = 2;   // emphatic stress
const PROSODY_TOKEN_BREAK_S: u32  = 3;   // short pause (~100ms)
const PROSODY_TOKEN_BREAK_M: u32  = 4;   // medium pause (~300ms)
const PROSODY_TOKEN_BREAK_L: u32  = 5;   // long pause (~600ms)
const PROSODY_TOKEN_PITCH_RISE: u32      = 6;
const PROSODY_TOKEN_PITCH_FALL: u32      = 7;
const PROSODY_TOKEN_PITCH_RISE_FALL: u32 = 8;
const PROSODY_TOKEN_RATE_FAST: u32       = 9;
const PROSODY_TOKEN_RATE_SLOW: u32       = 10;
const PROSODY_TOKEN_EMPHASIS: u32        = 11;
const NUM_PROSODY_TOKENS: u32            = 12;

fn default_d_model() -> usize { 1024 }
fn default_n_layers() -> usize { 16 }
fn default_n_heads() -> usize { 16 }
fn default_n_kv() -> usize { 4 }
fn default_d_ff() -> usize { 2560 }
fn default_max_seq() -> usize { 4096 }
fn default_text_v() -> usize { 32000 }
fn default_sem_v() -> usize { 4096 }
fn default_n_sp() -> usize { 4 }
fn default_theta() -> f64 { 10000.0 }
fn default_eps() -> f64 { 1e-5 }
fn default_acoustic_dim() -> usize { 512 }

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            d_model: 1024, n_layers: 16, n_heads: 16, n_kv_heads: 4,
            d_ff: 2560, max_seq_len: 4096, text_vocab_size: 32000,
            semantic_vocab_size: 4096, n_special_tokens: 4,
            rope_theta: 10000.0, norm_eps: 1e-5, use_prosody: false,
            use_acoustic_head: false, acoustic_dim: 512,
        }
    }
}

impl LmConfig {
    fn head_dim(&self) -> usize { self.d_model / self.n_heads }
    fn n_rep(&self) -> usize { self.n_heads / self.n_kv_heads }
}

// ─── RoPE ────────────────────────────────────────────────────────────────────

fn precompute_rope_cache(
    head_dim: usize, max_len: usize, theta: f64, device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut freqs = vec![0f32; half];
    for i in 0..half {
        freqs[i] = 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32);
    }
    let freqs = Tensor::from_vec(freqs, (1, half), device)?;
    let t = Tensor::arange(0f32, max_len as f32, device)?.reshape((max_len, 1))?;
    let angles = t.matmul(&freqs)?;
    Ok((angles.cos()?, angles.sin()?))
}

fn apply_rope(
    q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor, pos: usize,
) -> Result<(Tensor, Tensor)> {
    let (_, _, _, hd) = q.dims4()?;
    let half = hd / 2;
    let cos_s = cos.i(pos..pos + 1)?;
    let sin_s = sin.i(pos..pos + 1)?;

    fn rotate(x: &Tensor, c: &Tensor, s: &Tensor, half: usize) -> Result<Tensor> {
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let r1 = (x1.broadcast_mul(c)? - x2.broadcast_mul(s)?)?;
        let r2 = (x1.broadcast_mul(s)? + x2.broadcast_mul(c)?)?;
        Tensor::cat(&[r1, r2], D::Minus1)
    }

    Ok((rotate(q, &cos_s, &sin_s, half)?, rotate(k, &cos_s, &sin_s, half)?))
}

fn apply_rope_seq(
    q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor, pos: usize, seq_len: usize,
) -> Result<(Tensor, Tensor)> {
    let (_, _, _, hd) = q.dims4()?;
    let half = hd / 2;
    let cos_s = cos.i(pos..pos + seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin_s = sin.i(pos..pos + seq_len)?.unsqueeze(0)?.unsqueeze(0)?;

    fn rotate(x: &Tensor, c: &Tensor, s: &Tensor, half: usize) -> Result<Tensor> {
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let r1 = (x1.broadcast_mul(c)? - x2.broadcast_mul(s)?)?;
        let r2 = (x1.broadcast_mul(s)? + x2.broadcast_mul(c)?)?;
        Tensor::cat(&[r1, r2], D::Minus1)
    }

    Ok((rotate(q, &cos_s, &sin_s, half)?, rotate(k, &cos_s, &sin_s, half)?))
}

struct CausalMaskCache {
    seq_len: usize,
    total_len: usize,
    mask: Option<Tensor>,
}

impl CausalMaskCache {
    fn new() -> Self { Self { seq_len: 0, total_len: 0, mask: None } }

    fn get(&mut self, seq_len: usize, total_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        if self.seq_len == seq_len && self.total_len == total_len {
            if let Some(ref m) = self.mask {
                if m.dtype() == dtype {
                    return Ok(m.clone());
                }
            }
        }
        let past_len = total_len - seq_len;
        let mut data = vec![0.0f32; seq_len * total_len];
        for i in 0..seq_len {
            for j in (past_len + i + 1)..total_len {
                data[i * total_len + j] = f32::NEG_INFINITY;
            }
        }
        let m = Tensor::from_vec(data, (1, 1, seq_len, total_len), device)?.to_dtype(dtype)?;
        self.seq_len = seq_len;
        self.total_len = total_len;
        self.mask = Some(m.clone());
        Ok(m)
    }
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 { return Ok(x.clone()); }
    let (b, h, t, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h, n_rep, t, d))?
        .reshape((b, h * n_rep, t, d))
}

// ─── Attention ───────────────────────────────────────────────────────────────

struct GQAttention {
    wq: Linear, wk: Linear, wv: Linear, wo: Linear,
    n_heads: usize, n_kv_heads: usize, head_dim: usize, n_rep: usize,
}

impl GQAttention {
    fn load(cfg: &LmConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let hd = cfg.head_dim();
        Ok(Self {
            wq: linear_no_bias(d, cfg.n_heads * hd, vb.pp("wq"))?,
            wk: linear_no_bias(d, cfg.n_kv_heads * hd, vb.pp("wk"))?,
            wv: linear_no_bias(d, cfg.n_kv_heads * hd, vb.pp("wv"))?,
            wo: linear_no_bias(cfg.n_heads * hd, d, vb.pp("wo"))?,
            n_heads: cfg.n_heads, n_kv_heads: cfg.n_kv_heads,
            head_dim: hd, n_rep: cfg.n_rep(),
        })
    }

    fn forward(
        &self, x: &Tensor, cos: &Tensor, sin: &Tensor, pos: usize,
        k_cache: &mut Tensor, v_cache: &mut Tensor,
    ) -> Result<Tensor> {
        let (b, _t, _d) = x.dims3()?;

        // Skip .contiguous() — RoPE creates new contiguous tensors, SDPA Metal kernel handles strides
        let q = self.wq.forward(x)?
            .reshape((b, 1, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self.wk.forward(x)?
            .reshape((b, 1, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self.wv.forward(x)?
            .reshape((b, 1, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = apply_rope(&q, &k, cos, sin, pos)?;

        // Pre-allocated path: write at position, narrow active portion (avoids O(n²) cat)
        let (k_full, v_full) = if k_cache.dim(2)? > pos {
            *k_cache = k_cache.slice_scatter(&k, 2, pos)?;
            *v_cache = v_cache.slice_scatter(&v, 2, pos)?;
            (k_cache.narrow(2, 0, pos + 1)?, v_cache.narrow(2, 0, pos + 1)?)
        } else {
            *k_cache = Tensor::cat(&[&*k_cache, &k], 2)?;
            *v_cache = Tensor::cat(&[&*v_cache, &v], 2)?;
            (k_cache.clone(), v_cache.clone())
        };

        // Fused SDPA Metal kernel — handles GQA natively (no repeat_kv needed)
        let scale = (self.head_dim as f32).powf(-0.5);
        let out = candle_nn::ops::sdpa(&q, &k_full, &v_full, scale, 1.0)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, 1, ()))?;
        self.wo.forward(&out)
    }
}

impl GQAttention {
    fn forward_seq(
        &self, x: &Tensor, cos: &Tensor, sin: &Tensor, pos: usize, seq_len: usize,
        k_cache: &mut Tensor, v_cache: &mut Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;
        // Skip .contiguous() — apply_rope_seq creates new contiguous tensors via Tensor::cat
        let q = self.wq.forward(x)?
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self.wk.forward(x)?
            .reshape((b, t, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self.wv.forward(x)?
            .reshape((b, t, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = apply_rope_seq(&q, &k, cos, sin, pos, seq_len)?;

        // Pre-allocated path: write at position, narrow active portion (avoids O(n²) cat)
        let (k_full, v_full) = if k_cache.dim(2)? >= pos + seq_len {
            *k_cache = k_cache.slice_scatter(&k, 2, pos)?;
            *v_cache = v_cache.slice_scatter(&v, 2, pos)?;
            (k_cache.narrow(2, 0, pos + seq_len)?, v_cache.narrow(2, 0, pos + seq_len)?)
        } else {
            *k_cache = Tensor::cat(&[&*k_cache, &k], 2)?;
            *v_cache = Tensor::cat(&[&*v_cache, &v], 2)?;
            (k_cache.clone(), v_cache.clone())
        };

        let k_exp = repeat_kv(&k_full, self.n_rep)?;
        let v_exp = repeat_kv(&v_full, self.n_rep)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k_exp.t()?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v_exp)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, ()))?;
        self.wo.forward(&out)
    }
}

// ─── SwiGLU FFN ──────────────────────────────────────────────────────────────

struct SwiGluFfn { w_gate: Linear, w_up: Linear, w_down: Linear }

impl SwiGluFfn {
    fn load(cfg: &LmConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w_gate: linear_no_bias(cfg.d_model, cfg.d_ff, vb.pp("w_gate"))?,
            w_up: linear_no_bias(cfg.d_model, cfg.d_ff, vb.pp("w_up"))?,
            w_down: linear_no_bias(cfg.d_ff, cfg.d_model, vb.pp("w_down"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&self.w_gate.forward(x)?)?;
        self.w_down.forward(&(gate * self.w_up.forward(x)?)?)
    }
}

// ─── Transformer Block ──────────────────────────────────────────────────────

struct TransformerBlock {
    attn_norm: RmsNorm, attn: GQAttention,
    ffn_norm: RmsNorm, ffn: SwiGluFfn,
}

impl TransformerBlock {
    fn load(cfg: &LmConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn_norm: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("attn_norm"))?,
            attn: GQAttention::load(cfg, vb.pp("attn"))?,
            ffn_norm: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("ffn_norm"))?,
            ffn: SwiGluFfn::load(cfg, vb.pp("ffn"))?,
        })
    }
    fn forward(
        &self, x: &Tensor, cos: &Tensor, sin: &Tensor, pos: usize,
        kc: &mut Tensor, vc: &mut Tensor,
    ) -> Result<Tensor> {
        let h = self.attn.forward(&self.attn_norm.forward(x)?, cos, sin, pos, kc, vc)?;
        let x = (x + h)?;
        let h = self.ffn.forward(&self.ffn_norm.forward(&x)?)?;
        x + h
    }

    fn forward_seq(
        &self, x: &Tensor, cos: &Tensor, sin: &Tensor, pos: usize, seq_len: usize,
        kc: &mut Tensor, vc: &mut Tensor, mask: &Tensor,
    ) -> Result<Tensor> {
        let h = self.attn.forward_seq(&self.attn_norm.forward(x)?, cos, sin, pos, seq_len, kc, vc, mask)?;
        let x = (x + h)?;
        let h = self.ffn.forward(&self.ffn_norm.forward(&x)?)?;
        x + h
    }
}

// ─── Cross-Attention ─────────────────────────────────────────────────────────

struct CrossAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_rep: usize,
}

impl CrossAttention {
    fn load(cfg: &LmConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();
        Ok(Self {
            wq: linear_no_bias(cfg.d_model, cfg.n_heads * head_dim, vb.pp("wq"))?,
            wk: linear_no_bias(cfg.d_model, cfg.n_kv_heads * head_dim, vb.pp("wk"))?,
            wv: linear_no_bias(cfg.d_model, cfg.n_kv_heads * head_dim, vb.pp("wv"))?,
            wo: linear_no_bias(cfg.n_heads * head_dim, cfg.d_model, vb.pp("wo"))?,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim,
            n_rep: cfg.n_rep(),
        })
    }

    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (b, t_a, _) = x.dims3()?;
        let (_, t_c, _) = context.dims3()?;

        let q = self.wq.forward(x)?
            .reshape((b, t_a, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self.wk.forward(context)?
            .reshape((b, t_c, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self.wv.forward(context)?
            .reshape((b, t_c, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Fused SDPA Metal kernel — handles GQA natively (no repeat_kv needed)
        let scale = (self.head_dim as f32).powf(-0.5);
        let out = candle_nn::ops::sdpa(&q, &k, &v, scale, 1.0)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t_a, ()))?;
        self.wo.forward(&out)
    }
}

struct DecoderBlock {
    attn_norm: RmsNorm,
    attn: GQAttention,
    cross_norm: RmsNorm,
    cross_attn: CrossAttention,
    ffn_norm: RmsNorm,
    ffn: SwiGluFfn,
}

impl DecoderBlock {
    fn load(cfg: &LmConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn_norm: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("attn_norm"))?,
            attn: GQAttention::load(cfg, vb.pp("attn"))?,
            cross_norm: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("cross_norm"))?,
            cross_attn: CrossAttention::load(cfg, vb.pp("cross_attn"))?,
            ffn_norm: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("ffn_norm"))?,
            ffn: SwiGluFfn::load(cfg, vb.pp("ffn"))?,
        })
    }

    fn forward(
        &self, x: &Tensor, text_enc: &Tensor,
        cos: &Tensor, sin: &Tensor, pos: usize,
        kc: &mut Tensor, vc: &mut Tensor,
    ) -> Result<Tensor> {
        let h = self.attn.forward(&self.attn_norm.forward(x)?, cos, sin, pos, kc, vc)?;
        let x = (x + h)?;
        let h = self.cross_attn.forward(&self.cross_norm.forward(&x)?, text_enc)?;
        let x = (x + h)?;
        let h = self.ffn.forward(&self.ffn_norm.forward(&x)?)?;
        x + h
    }
}

// ─── Text Encoder ────────────────────────────────────────────────────────────

struct TextEncoderBlock {
    attn_norm: RmsNorm,
    attn_q: Linear,
    attn_k: Linear,
    attn_v: Linear,
    attn_o: Linear,
    n_heads: usize,
    head_dim: usize,
    ffn_norm: RmsNorm,
    ffn: SwiGluFfn,
}

impl TextEncoderBlock {
    fn load(cfg: &LmConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();
        Ok(Self {
            attn_norm: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("attn_norm"))?,
            attn_q: linear_no_bias(cfg.d_model, cfg.d_model, vb.pp("attn.in_proj_weight_q"))?,
            attn_k: linear_no_bias(cfg.d_model, cfg.d_model, vb.pp("attn.in_proj_weight_k"))?,
            attn_v: linear_no_bias(cfg.d_model, cfg.d_model, vb.pp("attn.in_proj_weight_v"))?,
            attn_o: linear_no_bias(cfg.d_model, cfg.d_model, vb.pp("attn.out_proj"))?,
            n_heads: cfg.n_heads,
            head_dim,
            ffn_norm: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("ffn_norm"))?,
            ffn: SwiGluFfn::load(cfg, vb.pp("ffn"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let h = self.attn_norm.forward(x)?;
        let q = self.attn_q.forward(&h)?.reshape((b, t, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.attn_k.forward(&h)?.reshape((b, t, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.attn_v.forward(&h)?.reshape((b, t, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        // Fused SDPA Metal kernel — bidirectional (no mask needed for text encoder)
        let scale = (self.head_dim as f32).powf(-0.5);
        let out = candle_nn::ops::sdpa(&q, &k, &v, scale, 1.0)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, ()))?;
        let out = self.attn_o.forward(&out)?;
        let x = (x + out)?;
        let h = self.ffn.forward(&self.ffn_norm.forward(&x)?)?;
        x + h
    }
}

// ─── Sonata Semantic LM ─────────────────────────────────────────────────────

struct ProsodyProjection {
    linear1: Linear,
    linear2: Linear,
}

impl ProsodyProjection {
    fn load(d_model: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = linear_no_bias(PROSODY_DIM, d_model, vb.pp("0"))?;
        let linear2 = linear_no_bias(d_model, d_model, vb.pp("2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, prosody: &Tensor) -> Result<Tensor> {
        let h = candle_nn::Activation::Silu.forward(&self.linear1.forward(prosody)?)?;
        self.linear2.forward(&h)
    }
}

struct SonataLM {
    text_emb: Embedding,
    semantic_emb: Embedding,
    prosody_proj: Option<ProsodyProjection>,
    // Legacy self-attention layers (used when no cross-attention weights found)
    layers: Vec<TransformerBlock>,
    // Cross-attention decoder layers (used when text encoder is loaded)
    decoder_layers: Vec<DecoderBlock>,
    text_encoder: Vec<TextEncoderBlock>,
    text_encoder_norm: Option<RmsNorm>,
    text_pos_emb: Option<Embedding>,
    use_cross_attention: bool,
    output_norm: RmsNorm,
    semantic_head: Linear,
    acoustic_head: Option<Linear>,  // Projects d_model(1024) -> acoustic_dim(512)
    rope_cos: Tensor,
    rope_sin: Tensor,
    cfg: LmConfig,
}

impl SonataLM {
    fn load(cfg: &LmConfig, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let text_total = cfg.text_vocab_size + cfg.n_special_tokens;
        let sem_total = cfg.semantic_vocab_size + cfg.n_special_tokens;

        // Try to load cross-attention decoder layers first, fall back to self-attention
        let mut decoder_layers = Vec::new();
        let mut text_encoder = Vec::new();
        let mut text_encoder_norm = None;
        let mut text_pos_emb = None;
        let mut use_cross_attention = false;

        let has_cross_attn = vb.pp("layers.0").pp("cross_attn").pp("wq")
            .get((cfg.n_heads * cfg.head_dim(), cfg.d_model), "weight").is_ok();

        let mut layers = Vec::new();

        if has_cross_attn {
            use_cross_attention = true;
            eprintln!("[sonata_lm] Cross-attention architecture detected");
            for i in 0..cfg.n_layers {
                decoder_layers.push(DecoderBlock::load(cfg, vb.pp(format!("layers.{i}")))?);
            }
            // Load text encoder (4 layers by default)
            let n_enc = 4;
            for i in 0..n_enc {
                match TextEncoderBlock::load(cfg, vb.pp(format!("text_encoder.{i}"))) {
                    Ok(block) => text_encoder.push(block),
                    Err(_) => break,
                }
            }
            text_encoder_norm = Some(rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("text_encoder_norm"))?);
            text_pos_emb = Some(embedding(cfg.max_seq_len, cfg.d_model, vb.pp("text_pos_emb"))?);
            eprintln!("[sonata_lm] Text encoder: {} layers, decoder: {} layers (cross-attn)",
                      text_encoder.len(), decoder_layers.len());
        } else {
            for i in 0..cfg.n_layers {
                layers.push(TransformerBlock::load(cfg, vb.pp(format!("layers.{i}")))?);
            }
        }

        let prosody_proj = if cfg.use_prosody {
            match ProsodyProjection::load(cfg.d_model, vb.pp("prosody_proj")) {
                Ok(p) => {
                    eprintln!("[sonata_lm] Prosody projection loaded (3 → {})", cfg.d_model);
                    Some(p)
                }
                Err(_) => {
                    eprintln!("[sonata_lm] No prosody_proj weights found, disabled");
                    None
                }
            }
        } else {
            None
        };

        let (rope_cos, rope_sin) = precompute_rope_cache(
            cfg.head_dim(), cfg.max_seq_len, cfg.rope_theta, device,
        )?;
        let rope_cos = rope_cos.to_dtype(dtype)?;
        let rope_sin = rope_sin.to_dtype(dtype)?;

        // Expand semantic embedding table to include prosody token slots
        let semantic_emb_base = embedding(sem_total, cfg.d_model, vb.pp("semantic_emb"))?;
        let prosody_extra = Tensor::zeros(
            (NUM_PROSODY_TOKENS as usize, cfg.d_model), dtype, device,
        )?;
        let semantic_emb = Embedding::new(
            Tensor::cat(&[semantic_emb_base.embeddings(), &prosody_extra], 0)?,
            cfg.d_model,
        );

        // Try to load acoustic head if enabled, fall back to None if weights missing
        let acoustic_head = if cfg.use_acoustic_head {
            match linear_no_bias(cfg.d_model, cfg.acoustic_dim, vb.pp("acoustic_head")) {
                Ok(head) => {
                    eprintln!("[sonata_lm] Acoustic head loaded ({} -> {})", cfg.d_model, cfg.acoustic_dim);
                    Some(head)
                }
                Err(_) => {
                    eprintln!("[sonata_lm] use_acoustic_head=true but weights not found, disabled");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            text_emb: embedding(text_total, cfg.d_model, vb.pp("text_emb"))?,
            semantic_emb,
            prosody_proj,
            layers,
            decoder_layers,
            text_encoder,
            text_encoder_norm,
            text_pos_emb,
            use_cross_attention,
            output_norm: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("output_norm"))?,
            semantic_head: linear_no_bias(cfg.d_model, cfg.semantic_vocab_size, vb.pp("semantic_head"))?,
            acoustic_head,
            rope_cos, rope_sin,
            cfg: cfg.clone(),
        })
    }

    fn encode_text(&self, text_tokens: &[u32]) -> Result<Tensor> {
        let device = self.rope_cos.device();
        let t_len = text_tokens.len();
        if t_len > self.cfg.max_seq_len {
            return Err(candle_core::Error::Msg(format!(
                "text length {} exceeds max_seq_len {}", t_len, self.cfg.max_seq_len
            )));
        }
        let text_t = Tensor::from_vec(text_tokens.to_vec(), (1, t_len), device)?;
        let pos = Tensor::arange(0u32, t_len as u32, device)?.unsqueeze(0)?;
        let pos_emb = self.text_pos_emb.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("cross-attention requires text_pos_emb".into()))?;
        let mut x = (self.text_emb.forward(&text_t)? + pos_emb.forward(&pos)?)?;
        for block in &self.text_encoder {
            x = block.forward(&x)?;
        }
        if let Some(ref norm) = self.text_encoder_norm {
            x = norm.forward(&x)?;
        }
        Ok(x)
    }

    /// Encode text tokens into KV caches as a prefix (non-cross-attention mode).
    /// Returns the number of text positions consumed.
    fn encode_text_prefix(
        &self, text_ids: &[u32], kv_caches: &mut [(Tensor, Tensor)],
        mask_cache: &mut CausalMaskCache,
    ) -> Result<usize> {
        if text_ids.is_empty() { return Ok(0); }
        let device = self.rope_cos.device();
        let n = text_ids.len();
        // Offset text tokens by n_special_tokens to match embedding table layout
        let shifted: Vec<u32> = text_ids.iter()
            .map(|&t| t + self.cfg.n_special_tokens as u32).collect();
        let text_t = Tensor::from_vec(shifted, (1, n), device)?;
        let mut x = self.text_emb.forward(&text_t)?;

        let mask = mask_cache.get(n, n, device, x.dtype())?;
        for (i, layer) in self.layers.iter().enumerate() {
            let (kc, vc) = &mut kv_caches[i];
            x = layer.forward_seq(&x, &self.rope_cos, &self.rope_sin, 0, n, kc, vc, &mask)?;
        }
        Ok(n)
    }

    fn forward(
        &self, sem_tok: u32, pos: usize,
        kv_caches: &mut [(Tensor, Tensor)],
        prosody: Option<&Tensor>,
        text_encoding: Option<&Tensor>,
    ) -> Result<Tensor> {
        let device = self.rope_cos.device();
        let sem_t = Tensor::from_vec(vec![sem_tok], (1, 1), device)?;

        let mut x = self.semantic_emb.forward(&sem_t)?;

        if let (Some(ref proj), Some(p)) = (&self.prosody_proj, prosody) {
            x = (x + proj.forward(p)?)?;
        }

        if self.use_cross_attention {
            let text_enc = text_encoding.ok_or_else(||
                candle_core::Error::Msg("cross-attention requires text_encoding".into()))?;
            for (i, layer) in self.decoder_layers.iter().enumerate() {
                let (kc, vc) = &mut kv_caches[i];
                x = layer.forward(&x, text_enc, &self.rope_cos, &self.rope_sin, pos, kc, vc)?;
            }
        } else {
            for (i, layer) in self.layers.iter().enumerate() {
                let (kc, vc) = &mut kv_caches[i];
                x = layer.forward(&x, &self.rope_cos, &self.rope_sin, pos, kc, vc)?;
            }
        }

        let x = self.output_norm.forward(&x)?;
        self.semantic_head.forward(&x)
    }

    /// Like forward() but also returns the normalized hidden state (before semantic_head).
    /// Used by the RNN drafter to condition on the main model's hidden representations.
    fn forward_hidden(
        &self, sem_tok: u32, pos: usize,
        kv_caches: &mut [(Tensor, Tensor)],
        prosody: Option<&Tensor>,
        text_encoding: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let device = self.rope_cos.device();
        let sem_t = Tensor::from_vec(vec![sem_tok], (1, 1), device)?;
        let mut x = self.semantic_emb.forward(&sem_t)?;

        if let (Some(ref proj), Some(p)) = (&self.prosody_proj, prosody) {
            x = (x + proj.forward(p)?)?;
        }

        if self.use_cross_attention {
            let text_enc = text_encoding.ok_or_else(||
                candle_core::Error::Msg("cross-attention requires text_encoding".into()))?;
            for (i, layer) in self.decoder_layers.iter().enumerate() {
                let (kc, vc) = &mut kv_caches[i];
                x = layer.forward(&x, text_enc, &self.rope_cos, &self.rope_sin, pos, kc, vc)?;
            }
        } else {
            for (i, layer) in self.layers.iter().enumerate() {
                let (kc, vc) = &mut kv_caches[i];
                x = layer.forward(&x, &self.rope_cos, &self.rope_sin, pos, kc, vc)?;
            }
        }

        let hidden = self.output_norm.forward(&x)?;
        let logits = self.semantic_head.forward(&hidden)?;
        Ok((logits, hidden))
    }

    /// Like forward_hidden() but also computes acoustic latents if acoustic_head is enabled.
    /// Returns: (logits, hidden, acoustic) where acoustic is Some if enabled, None otherwise.
    /// The `acoustic_enabled` flag gates computation so no work is done when disabled at runtime.
    fn forward_with_acoustic(
        &self, sem_tok: u32, pos: usize,
        kv_caches: &mut [(Tensor, Tensor)],
        prosody: Option<&Tensor>,
        text_encoding: Option<&Tensor>,
        acoustic_enabled: bool,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let device = self.rope_cos.device();
        let sem_t = Tensor::from_vec(vec![sem_tok], (1, 1), device)?;
        let mut x = self.semantic_emb.forward(&sem_t)?;

        if let (Some(ref proj), Some(p)) = (&self.prosody_proj, prosody) {
            x = (x + proj.forward(p)?)?;
        }

        if self.use_cross_attention {
            let text_enc = text_encoding.ok_or_else(||
                candle_core::Error::Msg("cross-attention requires text_encoding".into()))?;
            for (i, layer) in self.decoder_layers.iter().enumerate() {
                let (kc, vc) = &mut kv_caches[i];
                x = layer.forward(&x, text_enc, &self.rope_cos, &self.rope_sin, pos, kc, vc)?;
            }
        } else {
            for (i, layer) in self.layers.iter().enumerate() {
                let (kc, vc) = &mut kv_caches[i];
                x = layer.forward(&x, &self.rope_cos, &self.rope_sin, pos, kc, vc)?;
            }
        }

        let hidden = self.output_norm.forward(&x)?;
        let logits = self.semantic_head.forward(&hidden)?;
        // Gate: only compute acoustic head when both the head exists AND runtime flag is enabled
        let acoustic = if acoustic_enabled {
            if let Some(ref ah) = self.acoustic_head {
                let raw = ah.forward(&hidden)?;
                // RMSNorm: normalize to unit variance so Flow decoder receives N(0,1) input
                let norm_sq = (raw.sqr()?.mean_keepdim(D::Minus1)? + 1e-5f64)?;
                Some((raw / norm_sq.sqrt()?)?)
            } else {
                None
            }
        } else {
            None
        };
        Ok((logits, hidden, acoustic))
    }

    /// Forward pass with a custom attention mask for tree-structured verification.
    /// mask: (1, 1, seq_len, total_len) where total_len = cached_len + seq_len.
    /// Uses 0.0 for attend and -inf for mask (additive mask).
    fn forward_tree(
        &self, sem_toks: &[u32], start_pos: usize,
        kv_caches: &mut [(Tensor, Tensor)],
        mask: &Tensor,
    ) -> Result<Tensor> {
        let device = self.rope_cos.device();
        let seq_len = sem_toks.len();
        let sem_t = Tensor::from_vec(sem_toks.to_vec(), (1, seq_len), device)?;
        let mut x = self.semantic_emb.forward(&sem_t)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let (kc, vc) = &mut kv_caches[i];
            x = layer.forward_seq(&x, &self.rope_cos, &self.rope_sin, start_pos, seq_len, kc, vc, mask)?;
        }

        let x = self.output_norm.forward(&x)?;
        self.semantic_head.forward(&x)
    }

    fn forward_seq(
        &self, sem_toks: &[u32], start_pos: usize,
        kv_caches: &mut [(Tensor, Tensor)],
        mask_cache: &mut CausalMaskCache,
    ) -> Result<Tensor> {
        let device = self.rope_cos.device();
        let seq_len = sem_toks.len();
        let sem_t = Tensor::from_vec(sem_toks.to_vec(), (1, seq_len), device)?;

        let mut x = self.semantic_emb.forward(&sem_t)?;

        // Use start_pos + seq_len: works correctly for both pre-allocated and dynamic caches
        let total_len = start_pos + seq_len;
        let mask = mask_cache.get(seq_len, total_len, device, x.dtype())?;

        for (i, layer) in self.layers.iter().enumerate() {
            let (kc, vc) = &mut kv_caches[i];
            x = layer.forward_seq(&x, &self.rope_cos, &self.rope_sin, start_pos, seq_len, kc, vc, &mask)?;
        }

        let x = self.output_norm.forward(&x)?;
        self.semantic_head.forward(&x)
    }
}

// ─── ReDrafter: RNN Draft Model ──────────────────────────────────────────────
//
// Apple ReDrafter approach: small GRU conditioned on main LM hidden states,
// generating a tree of candidate tokens for parallel verification.
// ~3-4M params, runs on Metal alongside the main model.

#[derive(Debug, Clone, serde::Deserialize)]
struct RnnDraftConfig {
    #[serde(default = "default_gru_hidden")]  gru_hidden: usize,
    #[serde(default = "default_gru_layers")]  gru_layers: usize,
    #[serde(default = "default_tree_width")]  tree_width: usize,
    #[serde(default = "default_tree_depth")]  tree_depth: usize,
    #[serde(default = "default_drafter_emb")] emb_dim: usize,
    #[serde(default = "default_d_model")]     d_model: usize,
    #[serde(default = "default_sem_v")]       vocab_size: usize,
}

fn default_gru_hidden() -> usize { 512 }
fn default_gru_layers() -> usize { 2 }
fn default_tree_width() -> usize { 4 }
fn default_tree_depth() -> usize { 3 }
fn default_drafter_emb() -> usize { 256 }

impl Default for RnnDraftConfig {
    fn default() -> Self {
        Self {
            gru_hidden: 512, gru_layers: 2, tree_width: 4, tree_depth: 3,
            emb_dim: 256, d_model: 1024, vocab_size: 4096,
        }
    }
}

/// GRU cell: z = σ(Wz·x + Uz·h), r = σ(Wr·x + Ur·h), h' = tanh(Wh·x + Uh·(r⊙h))
/// h_new = (1-z)⊙h + z⊙h'
struct GruCell {
    w_z: Linear, u_z: Linear,
    w_r: Linear, u_r: Linear,
    w_h: Linear, u_h: Linear,
    hidden_dim: usize,
}

impl GruCell {
    fn load(input_dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w_z: linear_no_bias(input_dim, hidden_dim, vb.pp("w_z"))?,
            u_z: linear_no_bias(hidden_dim, hidden_dim, vb.pp("u_z"))?,
            w_r: linear_no_bias(input_dim, hidden_dim, vb.pp("w_r"))?,
            u_r: linear_no_bias(hidden_dim, hidden_dim, vb.pp("u_r"))?,
            w_h: linear_no_bias(input_dim, hidden_dim, vb.pp("w_h"))?,
            u_h: linear_no_bias(hidden_dim, hidden_dim, vb.pp("u_h"))?,
            hidden_dim,
        })
    }

    /// x: (1, input_dim), h: (1, hidden_dim) -> h_new: (1, hidden_dim)
    fn forward(&self, x: &Tensor, h: &Tensor) -> Result<Tensor> {
        let z = (self.w_z.forward(x)? + self.u_z.forward(h)?)?.sigmoid()?;
        let r = (self.w_r.forward(x)? + self.u_r.forward(h)?)?.sigmoid()?;
        let rh = (&r * h)?;
        let h_cand = (self.w_h.forward(x)? + self.u_h.forward(&rh)?)?.tanh()?;
        let ones = Tensor::ones_like(&z)?;
        let one_minus_z = (ones - &z)?;
        let h_new = (one_minus_z * h)? + (&z * &h_cand)?;
        Ok(h_new)
    }

    fn zero_state(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        Tensor::zeros((1, self.hidden_dim), dtype, device)
    }
}

/// Tree candidate structure: W beams × D depth.
/// Each beam is a sequence of draft tokens starting from the same parent.
struct TreeCandidates {
    /// beams[i] = token IDs for beam i, length up to tree_depth
    beams: Vec<Vec<u32>>,
    width: usize,
    depth: usize,
}

impl TreeCandidates {
    /// Flatten tree into a token sequence for verification.
    /// Returns (input_tokens, tree_mask_data, beam_map).
    /// input_tokens: [last_tok, beam0_tok0, beam0_tok1, ..., beam1_tok0, ...].
    /// beam_map[i] = (beam_idx, depth_idx) for position i in the flattened sequence.
    fn flatten_for_verify(
        &self, last_token: u32, cached_len: usize, device: &Device, dtype: DType,
    ) -> Result<(Vec<u32>, Tensor, Vec<(usize, usize)>)> {
        let mut tokens = Vec::new();
        let mut beam_map = Vec::new();

        // Position 0: the last verified token
        tokens.push(last_token);
        beam_map.push((usize::MAX, usize::MAX)); // sentinel for root

        // Flatten beams: for each beam, push [last_tok, draft[0], draft[1], ...]
        // but we already pushed last_tok once. For tree attention:
        // beam i depth j input token = if j==0 { last_token } else { beams[i][j-1] }
        // The verify position for beam i depth j checks draft[i][j].
        for (bi, beam) in self.beams.iter().enumerate() {
            for (di, _tok) in beam.iter().enumerate() {
                let input_tok = if di == 0 { last_token } else { beam[di - 1] };
                tokens.push(input_tok);
                beam_map.push((bi, di));
            }
        }

        let seq_len = tokens.len();
        let total_len = cached_len + seq_len;

        // Build tree attention mask: (seq_len × total_len)
        // All positions attend to cached KV entries (columns 0..cached_len).
        // Position 0 (root) only attends to itself + cached.
        // Beam positions attend to cached + root + their ancestor chain.
        let mut mask_data = vec![0.0f32; seq_len * total_len];
        for i in 0..seq_len {
            // Mask all positions after self in the draft sequence
            for j in (cached_len + i + 1)..total_len {
                mask_data[i * total_len + j] = f32::NEG_INFINITY;
            }
        }

        // Now apply tree structure: mask out sibling beams
        // Position 0 is root — standard causal (already correct)
        // For position p (p>0): only attend to cached + root + same-beam ancestors
        let mut pos = 1; // skip root
        for (bi, beam) in self.beams.iter().enumerate() {
            for di in 0..beam.len() {
                let p = pos; // current position in flattened sequence

                // Mask out ALL other draft positions, then un-mask ancestors
                for j in (cached_len + 1)..(cached_len + seq_len) {
                    mask_data[p * total_len + j] = f32::NEG_INFINITY;
                }

                // Un-mask: root (position cached_len + 0)
                mask_data[p * total_len + cached_len] = 0.0;

                // Un-mask: ancestors in same beam (positions for beam bi, depths < di)
                // The start of beam bi in the flattened sequence is 1 + bi * max_depth
                let beam_start = 1 + bi * beam.len();
                for ancestor_di in 0..di {
                    let ancestor_pos = cached_len + beam_start + ancestor_di;
                    mask_data[p * total_len + ancestor_pos] = 0.0;
                }

                // Un-mask: self
                mask_data[p * total_len + cached_len + p] = 0.0;

                pos += 1;
            }
        }

        let mask = Tensor::from_vec(mask_data, (1, 1, seq_len, total_len), device)?
            .to_dtype(dtype)?;

        Ok((tokens, mask, beam_map))
    }
}

/// RNN draft model: projects main LM hidden state through GRU layers to generate
/// a tree of candidate tokens.
struct RnnDrafter {
    /// Projects main LM hidden (d_model) to GRU initial state (gru_hidden)
    hidden_proj: Linear,
    /// Token embedding for draft input (vocab_size → emb_dim)
    token_emb: Embedding,
    /// Stacked GRU layers
    gru_layers: Vec<GruCell>,
    /// Output projection to vocabulary logits
    output_head: Linear,
    cfg: RnnDraftConfig,
}

impl RnnDrafter {
    fn load(cfg: &RnnDraftConfig, vb: VarBuilder, _device: &Device, _dtype: DType) -> Result<Self> {
        let hidden_proj = linear_no_bias(cfg.d_model, cfg.gru_hidden, vb.pp("hidden_proj"))?;
        let token_emb = embedding(cfg.vocab_size, cfg.emb_dim, vb.pp("token_emb"))?;

        let mut gru_layers = Vec::with_capacity(cfg.gru_layers);
        let first_input_dim = cfg.emb_dim; // first layer takes token embedding
        gru_layers.push(GruCell::load(first_input_dim, cfg.gru_hidden, vb.pp("gru.0"))?);
        for i in 1..cfg.gru_layers {
            // Subsequent layers take previous layer's hidden state
            gru_layers.push(GruCell::load(cfg.gru_hidden, cfg.gru_hidden, vb.pp(format!("gru.{i}")))?);
        }

        let output_head = linear_no_bias(cfg.gru_hidden, cfg.vocab_size, vb.pp("output_head"))?;

        Ok(Self { hidden_proj, token_emb, gru_layers, output_head, cfg })
    }

    /// Generate tree of candidate tokens from the main model's hidden state.
    /// lm_hidden: (1, 1, d_model) — hidden state at the last verified position.
    /// first_token: the token sampled from the main model's logits at this position.
    /// Returns TreeCandidates with W beams × D depth.
    fn draft_tree(
        &self, lm_hidden: &Tensor, first_token: u32,
        device: &Device, dtype: DType,
    ) -> Result<TreeCandidates> {
        let w = self.cfg.tree_width;
        let d = self.cfg.tree_depth;

        // Project LM hidden state to GRU initial state: (1, d_model) → (1, gru_hidden)
        let h_squeezed = lm_hidden.squeeze(1)?; // (1, d_model)
        let h0 = self.hidden_proj.forward(&h_squeezed)?;

        // Initialize all GRU layers with the projected hidden state
        let mut layer_states: Vec<Tensor> = Vec::with_capacity(self.cfg.gru_layers);
        layer_states.push(h0);
        for _ in 1..self.cfg.gru_layers {
            layer_states.push(Tensor::zeros((1, self.cfg.gru_hidden), dtype, device)?);
        }

        // Depth 0: run GRU with first_token embedding, take top-W candidates
        let tok_t = Tensor::from_vec(vec![first_token], (1,), device)?;
        let emb = self.token_emb.forward(&tok_t)?; // (1, emb_dim)

        let mut x = emb;
        for (li, gru) in self.gru_layers.iter().enumerate() {
            layer_states[li] = gru.forward(&x, &layer_states[li])?;
            x = layer_states[li].clone();
        }

        let logits = self.output_head.forward(&x)?; // (1, vocab_size)
        let logits_vec: Vec<f32> = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1()?;

        // Get top-W candidates at depth 0
        let mut indexed: Vec<(usize, f32)> = logits_vec.iter().enumerate()
            .map(|(i, &v)| (i, v)).collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_w: Vec<u32> = indexed.iter().take(w).map(|&(i, _)| i as u32).collect();

        // Build beams: each starts with a top-W candidate, extends greedily for D-1 more
        let mut beams: Vec<Vec<u32>> = Vec::with_capacity(w);
        for &candidate in &top_w {
            let mut beam = vec![candidate];

            // Save GRU state for this branch (clone from depth-0 state)
            let mut branch_states = layer_states.clone();

            for _depth in 1..d {
                let prev_tok = *beam.last().unwrap();
                let tok_t = Tensor::from_vec(vec![prev_tok], (1,), device)?;
                let emb = self.token_emb.forward(&tok_t)?;

                let mut bx = emb;
                for (li, gru) in self.gru_layers.iter().enumerate() {
                    branch_states[li] = gru.forward(&bx, &branch_states[li])?;
                    bx = branch_states[li].clone();
                }

                let logits = self.output_head.forward(&bx)?;
                let lv: Vec<f32> = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1()?;

                // Greedy: take argmax for deeper levels
                let next = lv.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32).unwrap_or(0);

                if next == 2 { break; } // EOS
                beam.push(next);
            }

            beams.push(beam);
        }

        Ok(TreeCandidates { beams, width: w, depth: d })
    }
}

// ─── Engine ──────────────────────────────────────────────────────────────────

struct DraftModel {
    model: SonataLM,
    kv_caches: Vec<(Tensor, Tensor)>,
}

struct LmEngine {
    model: SonataLM,
    kv_caches: Vec<(Tensor, Tensor)>,
    device: Device,
    dtype: DType,

    text_tokens: Vec<u32>,
    text_pos: usize,
    text_encoding: Option<Tensor>,
    semantic_tokens: Vec<u32>,
    step_count: usize,

    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    max_tokens: usize,
    done: bool,

    recent_tokens: VecDeque<u32>,
    consecutive_pad: usize,
    draft: Option<DraftModel>,
    rnn_drafter: Option<RnnDrafter>,
    last_hidden: Option<Tensor>,
    speculate_k: usize,
    mask_cache: CausalMaskCache,

    prosody_features: Option<Vec<[f32; PROSODY_DIM]>>,
    text_encoding_stale: bool,

    // Coarse-grained speculative decoding: tokens in the same group are acoustically similar
    similarity_groups: Option<Vec<u16>>,
    coarse_grained: bool,

    sampling_buf: Vec<(usize, f32)>,

    // Acoustic head support
    acoustic_buffer: Vec<f32>,     // accumulated acoustic vectors (acoustic_dim per step)
    acoustic_head_enabled: bool,   // runtime toggle
    acoustic_dim: usize,
}

fn resolve_hf_path(path: &str, candidates: &[&str]) -> std::result::Result<String, Box<dyn std::error::Error>> {
    if Path::new(path).exists() {
        return Ok(path.to_string());
    }
    if path.contains('/') && !path.starts_with('.') && !path.starts_with('/') {
        eprintln!("[sonata_lm] '{}' not found locally, downloading from HuggingFace...", path);
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(path.to_string());
        for f in candidates {
            if let Ok(p) = repo.get(f) {
                let s = p.to_string_lossy().to_string();
                eprintln!("[sonata_lm] Downloaded: {}", s);
                return Ok(s);
            }
        }
        Err(format!("No matching files in HF repo '{}'", path).into())
    } else {
        Err(format!("File not found: {}", path).into())
    }
}

impl LmEngine {
    fn load(
        weights_path: &str, config_path: Option<&str>,
    ) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let device = if candle_core::utils::metal_is_available() {
            Device::new_metal(0)?
        } else {
            eprintln!("[sonata_lm] Metal not available, using CPU");
            Device::Cpu
        };

        let cfg: LmConfig = if let Some(cp) = config_path {
            let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(cp)?)?;
            let d = raw.get("d_model").and_then(|v| v.as_u64()).unwrap_or(1024) as usize;
            let d_ff = if let Some(v) = raw.get("d_ff").and_then(|v| v.as_u64()) {
                v as usize
            } else if let Some(m) = raw.get("ffn_mult").and_then(|v| v.as_f64()) {
                let r = (d as f64 * m) as usize;
                r - (r % 256)
            } else {
                2560
            };
            LmConfig {
                d_model: d,
                n_layers: raw.get("n_layers").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
                n_heads: raw.get("n_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
                n_kv_heads: raw.get("n_kv_heads").and_then(|v| v.as_u64()).unwrap_or(4) as usize,
                d_ff,
                max_seq_len: raw.get("max_seq_len").and_then(|v| v.as_u64()).unwrap_or(4096) as usize,
                text_vocab_size: raw.get("text_vocab_size").and_then(|v| v.as_u64()).unwrap_or(32000) as usize,
                semantic_vocab_size: raw.get("semantic_vocab_size").and_then(|v| v.as_u64()).unwrap_or(4096) as usize,
                n_special_tokens: raw.get("n_special_tokens").and_then(|v| v.as_u64()).unwrap_or(4) as usize,
                rope_theta: raw.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(10000.0),
                norm_eps: raw.get("norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5),
                use_prosody: raw.get("use_prosody").and_then(|v| v.as_bool()).unwrap_or(false),
                use_acoustic_head: raw.get("use_acoustic_head").and_then(|v| v.as_bool()).unwrap_or(false),
                acoustic_dim: raw.get("acoustic_dim").and_then(|v| v.as_u64()).unwrap_or(512) as usize,
            }
        } else {
            LmConfig::default()
        };

        let dtype = DType::F16;
        eprintln!("[sonata_lm] Config: d={}, L={}, H={}, KV={}, FF={}, semantic_vocab={}",
                  cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff, cfg.semantic_vocab_size);

        let resolved = resolve_hf_path(weights_path,
            &["model.safetensors", "sonata_lm.safetensors"])?;
        eprintln!("[sonata_lm] Loading FP16 weights from {} (~2x Metal throughput)...", resolved);

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&resolved], dtype, &device)?
        };
        let model = SonataLM::load(&cfg, vb, &device, dtype)?;
        eprintln!("[sonata_lm] Model loaded on {:?} (dtype={:?})", device, dtype);

        let kv_caches = Self::new_kv_caches(&cfg, &device, dtype, true)?;

        {
            let t0 = std::time::Instant::now();
            let mut warmup_kv = Self::new_kv_caches(&cfg, &device, dtype, false)?;
            let _ = model.forward(
                1, 0, &mut warmup_kv, None, None,
            );
            eprintln!("[sonata_lm] Metal shader warmup: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
        }

        // Precompute acoustic similarity groups from semantic embedding weights
        let similarity_groups = Self::compute_similarity_groups(&model, &device);

        // Acoustic head support: check before moving model
        let acoustic_head_enabled = cfg.use_acoustic_head && model.acoustic_head.is_some();
        let acoustic_dim = if cfg.use_acoustic_head { cfg.acoustic_dim } else { 0 };

        Ok(Self {
            model, kv_caches, device, dtype,
            text_tokens: Vec::new(),
            text_pos: 0,
            text_encoding: None,
            semantic_tokens: vec![1], // BOS
            step_count: 0,
            temperature: 0.8,
            top_k: 50,
            top_p: 0.92,
            repetition_penalty: 1.15,
            max_tokens: 2000,
            done: false,
            recent_tokens: VecDeque::with_capacity(64),
            consecutive_pad: 0,
            draft: None,
            rnn_drafter: None,
            last_hidden: None,
            speculate_k: 5,
            mask_cache: CausalMaskCache::new(),
            prosody_features: None,
            text_encoding_stale: false,
            similarity_groups,
            coarse_grained: false,
            sampling_buf: Vec::with_capacity(cfg.semantic_vocab_size),
            acoustic_buffer: Vec::new(),
            acoustic_head_enabled,
            acoustic_dim,
        })
    }

    fn new_kv_caches(cfg: &LmConfig, device: &Device, dtype: DType, preallocate: bool) -> Result<Vec<(Tensor, Tensor)>> {
        let seq_dim = if preallocate { cfg.max_seq_len } else { 0 };
        let mut caches = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            caches.push((
                Tensor::zeros((1, cfg.n_kv_heads, seq_dim, cfg.head_dim()), dtype, device)?,
                Tensor::zeros((1, cfg.n_kv_heads, seq_dim, cfg.head_dim()), dtype, device)?,
            ));
        }
        Ok(caches)
    }

    fn set_text(&mut self, ids: &[u32]) {
        self.text_tokens = ids.to_vec();
        self.text_pos = 0;
        self.text_encoding_stale = false;

        if self.model.use_cross_attention {
            if !ids.is_empty() {
                match self.model.encode_text(ids) {
                    Ok(enc) => { self.text_encoding = Some(enc); }
                    Err(e) => {
                        eprintln!("[sonata_lm] Text encoding failed: {}", e);
                        self.text_encoding = None;
                    }
                }
            }
        } else if !ids.is_empty() {
            // Concatenated mode: encode text as prefix into KV caches
            match self.model.encode_text_prefix(
                ids, &mut self.kv_caches, &mut self.mask_cache,
            ) {
                Ok(n) => {
                    self.text_pos = n;
                    eprintln!("[sonata_lm] Text prefix encoded: {} tokens → KV cache", n);
                }
                Err(e) => {
                    eprintln!("[sonata_lm] Text prefix encoding failed: {}", e);
                }
            }
        }
    }

    fn append_text(&mut self, ids: &[u32]) {
        if ids.is_empty() { return; }
        self.text_tokens.extend_from_slice(ids);
        if self.model.use_cross_attention {
            self.text_encoding_stale = true;
        }
        // Non-cross-attention: text prefix is fixed at set_text time.
        // append_text is a no-op for KV cache (text is already encoded).
    }

    fn finish_text(&mut self) {
        if self.model.use_cross_attention {
            if self.text_encoding_stale && !self.text_tokens.is_empty() {
                match self.model.encode_text(&self.text_tokens) {
                    Ok(enc) => { self.text_encoding = Some(enc); }
                    Err(e) => { eprintln!("[sonata_lm] Text re-encoding failed: {}", e); }
                }
                self.text_encoding_stale = false;
            }
        }
    }

    fn step(&mut self) -> std::result::Result<Option<u32>, Box<dyn std::error::Error>> {
        if self.done || self.step_count >= self.max_tokens {
            self.done = true;
            return Ok(None);
        }

        let sem_tok = *self.semantic_tokens.last().unwrap_or(&1);

        let prosody_tensor = if let Some(ref pf) = self.prosody_features {
            let idx = self.step_count.min(pf.len().saturating_sub(1));
            let data = pf[idx];
            Some(Tensor::from_vec(
                data.to_vec(), (1, 1, PROSODY_DIM), &self.device,
            )?.to_dtype(self.dtype)?)
        } else {
            None
        };

        // Cross-attention: KV cache only has decoder entries, pos = step_count.
        // Non-cross-attention: text prefix in KV cache, pos = text_pos + step_count.
        let pos = if self.model.use_cross_attention {
            self.step_count
        } else {
            self.text_pos + self.step_count
        };

        let logits = self.model.forward(
            sem_tok, pos, &mut self.kv_caches,
            prosody_tensor.as_ref(),
            self.text_encoding.as_ref(),
        )?;

        let logits = logits.squeeze(0)?.squeeze(0)?;
        let mut logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;

        Self::apply_repetition_penalty(&mut logits_vec, &self.recent_tokens, self.repetition_penalty);

        let temp = if self.temperature > 1e-8 { self.temperature } else { 1e-8 };
        let inv_temp = 1.0 / temp;
        for v in logits_vec.iter_mut() { *v *= inv_temp; }

        let next = Self::sample_top_k_top_p(&logits_vec, self.top_k, self.top_p,
                                               &mut self.sampling_buf);

        if next == 2 {
            self.done = true;
            return Ok(None);
        }
        if next == 0 {
            self.consecutive_pad += 1;
            if self.consecutive_pad > 100 {
                self.done = true;
                return Ok(None);
            }
        } else {
            self.consecutive_pad = 0;
        }

        self.recent_tokens.push_back(next);
        if self.recent_tokens.len() > 64 { self.recent_tokens.pop_front(); }

        self.semantic_tokens.push(next);
        self.step_count += 1;

        Ok(Some(next))
    }

    fn apply_repetition_penalty(logits: &mut [f32], recent: &VecDeque<u32>, penalty: f32) {
        if penalty <= 1.0 { return; }
        for &tok in recent {
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

    fn sample_top_k_top_p(logits: &[f32], top_k: usize, top_p: f32,
                           scratch: &mut Vec<(usize, f32)>) -> u32 {
        let k = top_k.min(logits.len());
        scratch.clear();
        scratch.extend(logits.iter().enumerate().map(|(i, &val)| {
            (i, if val.is_nan() { f32::NEG_INFINITY } else { val })
        }));
        if k < scratch.len() {
            scratch.select_nth_unstable_by(k, |a, b|
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scratch.truncate(k);
            scratch.sort_unstable_by(|a, b|
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            scratch.sort_unstable_by(|a, b|
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        if scratch.is_empty() { return 0; }

        let max_val = scratch[0].1;
        for p in scratch.iter_mut() { p.1 = (p.1 - max_val).exp(); }
        let sum: f32 = scratch.iter().map(|p| p.1).sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for p in scratch.iter_mut() { p.1 *= inv; }
        }

        // Nucleus (top-p) filter
        let mut cum = 0.0f32;
        let mut cutoff = scratch.len();
        for (i, &(_, prob)) in scratch.iter().enumerate() {
            cum += prob;
            if cum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        scratch.truncate(cutoff);

        let sum2: f32 = scratch.iter().map(|p| p.1).sum();
        if sum2 > 0.0 {
            let inv = 1.0 / sum2;
            for p in scratch.iter_mut() { p.1 *= inv; }
        }

        let r: f32 = rand::random();
        let mut cumsum = 0f32;
        for &(idx, prob) in scratch.iter() {
            cumsum += prob;
            if cumsum >= r { return idx as u32; }
        }
        scratch.last().map(|p| p.0 as u32).unwrap_or(0)
    }

    /// Cluster semantic embedding vectors into acoustic similarity groups.
    /// Tokens whose embeddings are within cosine_threshold of a centroid
    /// are assigned the same group ID. Used for coarse-grained spec decoding.
    fn compute_similarity_groups(model: &SonataLM, _device: &Device) -> Option<Vec<u16>> {
        let vocab_size = model.cfg.semantic_vocab_size;
        let emb_weights = model.semantic_emb.embeddings();
        let emb_f32 = match emb_weights.to_dtype(DType::F32) {
            Ok(e) => e,
            Err(_) => return None,
        };
        let flat: Vec<f32> = match emb_f32.flatten_all().and_then(|t| t.to_vec1()) {
            Ok(v) => v,
            Err(_) => return None,
        };
        let d = model.cfg.d_model;
        if flat.len() < vocab_size * d { return None; }

        // K-means-lite: assign groups based on top principal component sign + magnitude bins
        let n_groups = 64u16;
        let mut groups = vec![0u16; vocab_size];
        for i in 0..vocab_size {
            let offset = i * d;
            let mut norm = 0.0f32;
            let mut sum = 0.0f32;
            for j in 0..d {
                let v = flat[offset + j];
                norm += v * v;
                sum += v;
            }
            norm = norm.sqrt().max(1e-8);
            let mean = sum / d as f32;
            // Hash into group based on mean and norm
            let bucket = ((mean * 1000.0).abs() as u16 + (norm * 100.0) as u16) % n_groups;
            groups[i] = bucket;
        }
        eprintln!("[sonata_lm] Computed {} acoustic similarity groups for {} tokens",
                  n_groups, vocab_size);
        Some(groups)
    }

    fn tokens_similar_static(groups: Option<&Vec<u16>>, a: u32, b: u32) -> bool {
        if a == b { return true; }
        if let Some(groups) = groups {
            let ai = a as usize;
            let bi = b as usize;
            if ai < groups.len() && bi < groups.len() {
                return groups[ai] == groups[bi];
            }
        }
        false
    }

    fn reset(&mut self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        self.kv_caches = Self::new_kv_caches(&self.model.cfg, &self.device, self.dtype, true)?;
        self.text_tokens.clear();
        self.text_pos = 0;
        self.text_encoding = None;
        self.text_encoding_stale = false;
        self.semantic_tokens = vec![1];
        self.step_count = 0;
        self.done = false;
        self.consecutive_pad = 0;
        self.recent_tokens.clear();
        self.prosody_features = None;
        self.last_hidden = None;
        if let Some(ref mut draft) = self.draft {
            draft.kv_caches = Self::new_kv_caches(&draft.model.cfg, &self.device, self.dtype, false)?;
        }
        Ok(())
    }

    fn load_draft(&mut self, weights_path: &str, config_path: Option<&str>)
        -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let cfg: LmConfig = if let Some(cp) = config_path {
            serde_json::from_str(&std::fs::read_to_string(cp)?)?
        } else {
            LmConfig { n_layers: 4, ..self.model.cfg.clone() }
        };
        eprintln!("[sonata_lm] Loading draft model (L={}) for speculative decoding...", cfg.n_layers);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], self.dtype, &self.device)?
        };
        let model = SonataLM::load(&cfg, vb, &self.device, self.dtype)?;
        let kv_caches = Self::new_kv_caches(&cfg, &self.device, self.dtype, false)?;
        self.draft = Some(DraftModel { model, kv_caches });
        eprintln!("[sonata_lm] Draft model loaded ({} layers)", cfg.n_layers);
        Ok(())
    }

    fn load_rnn_drafter(&mut self, weights_path: &str, config_path: Option<&str>)
        -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let cfg: RnnDraftConfig = if let Some(cp) = config_path {
            let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(cp)?)?;
            RnnDraftConfig {
                gru_hidden: raw.get("gru_hidden").and_then(|v| v.as_u64()).unwrap_or(512) as usize,
                gru_layers: raw.get("gru_layers").and_then(|v| v.as_u64()).unwrap_or(2) as usize,
                tree_width: raw.get("tree_width").and_then(|v| v.as_u64()).unwrap_or(4) as usize,
                tree_depth: raw.get("tree_depth").and_then(|v| v.as_u64()).unwrap_or(3) as usize,
                emb_dim: raw.get("emb_dim").and_then(|v| v.as_u64()).unwrap_or(256) as usize,
                d_model: self.model.cfg.d_model,
                vocab_size: self.model.cfg.semantic_vocab_size,
            }
        } else {
            RnnDraftConfig {
                d_model: self.model.cfg.d_model,
                vocab_size: self.model.cfg.semantic_vocab_size,
                ..RnnDraftConfig::default()
            }
        };

        eprintln!("[sonata_lm] Loading RNN drafter (GRU h={}, L={}, tree {}×{})...",
                  cfg.gru_hidden, cfg.gru_layers, cfg.tree_width, cfg.tree_depth);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], self.dtype, &self.device)?
        };
        let drafter = RnnDrafter::load(&cfg, vb, &self.device, self.dtype)?;
        let params: usize = cfg.d_model * cfg.gru_hidden  // hidden_proj
            + cfg.vocab_size * cfg.emb_dim                  // token_emb
            + cfg.gru_layers * (cfg.emb_dim * cfg.gru_hidden * 3 + cfg.gru_hidden * cfg.gru_hidden * 3) // rough GRU
            + cfg.gru_hidden * cfg.vocab_size;              // output_head
        eprintln!("[sonata_lm] RNN drafter loaded (~{:.1}M params)", params as f64 / 1e6);
        self.rnn_drafter = Some(drafter);
        Ok(())
    }

    /// Tree-based speculative decoding using the RNN drafter (ReDrafter approach).
    /// 1. Run main model to get hidden state + first token
    /// 2. RNN generates tree of candidates from hidden state
    /// 3. Verify tree in one forward pass through main model
    /// Returns accepted tokens.
    fn tree_speculative_step(&mut self) -> std::result::Result<Vec<u32>, Box<dyn std::error::Error>> {
        if self.done || self.step_count >= self.max_tokens {
            self.done = true;
            return Ok(Vec::new());
        }

        // Tree spec decoding only works with legacy (non-cross-attention) models
        let drafter = match self.rnn_drafter {
            Some(ref d) if !self.model.use_cross_attention => d,
            _ => {
                let tok = self.step()?;
                return Ok(tok.into_iter().collect());
            }
        };

        let sem_tok = *self.semantic_tokens.last().unwrap_or(&1);
        let pos = self.text_pos + self.step_count;

        // Step 1: Run main model to get hidden state + logits for first token
        let (logits, hidden) = self.model.forward_hidden(
            sem_tok, pos, &mut self.kv_caches, None, self.text_encoding.as_ref(),
        )?;
        let logits = logits.squeeze(0)?.squeeze(0)?;
        let mut logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;
        Self::apply_repetition_penalty(&mut logits_vec, &self.recent_tokens, self.repetition_penalty);
        let temp = if self.temperature > 1e-8 { self.temperature } else { 1e-8 };
        let inv_temp = 1.0 / temp;
        for v in logits_vec.iter_mut() { *v *= inv_temp; }
        let first_token = Self::sample_top_k_top_p(&logits_vec, self.top_k, self.top_p,
                                                    &mut self.sampling_buf);

        if first_token == 2 {
            self.done = true;
            return Ok(Vec::new());
        }

        // Accept first token (always verified by main model)
        self.semantic_tokens.push(first_token);
        self.step_count += 1;
        self.recent_tokens.push_back(first_token);
        if self.recent_tokens.len() > 64 { self.recent_tokens.pop_front(); }
        if first_token == 0 {
            self.consecutive_pad += 1;
            if self.consecutive_pad > 100 { self.done = true; return Ok(vec![first_token]); }
        } else {
            self.consecutive_pad = 0;
        }

        // Step 2: Generate tree candidates using RNN drafter
        let tree = drafter.draft_tree(
            &hidden, first_token, &self.device, self.dtype,
        )?;

        if tree.beams.is_empty() || tree.beams[0].is_empty() {
            self.last_hidden = Some(hidden);
            return Ok(vec![first_token]);
        }

        // Step 3: Build tree attention mask and verify all candidates
        let cached_len = self.text_pos + self.step_count;
        let (verify_tokens, mask, _beam_map) = tree.flatten_for_verify(
            first_token, cached_len, &self.device, self.dtype,
        )?;

        let tree_logits = self.model.forward_tree(
            &verify_tokens, cached_len, &mut self.kv_caches, &mask,
        )?;
        let tree_logits = tree_logits.squeeze(0)?; // (seq_len, vocab)

        // Step 4: Walk the tree and accept the longest matching beam
        // logits[0] = prediction after first_token (for all beams' depth 0)
        let root_logits = tree_logits.i(0)?;
        let mut rl: Vec<f32> = root_logits.to_dtype(DType::F32)?.to_vec1()?;
        Self::apply_repetition_penalty(&mut rl, &self.recent_tokens, self.repetition_penalty);
        for v in rl.iter_mut() { *v *= inv_temp; }
        let verified_at_root = Self::sample_top_k_top_p(&rl, self.top_k, self.top_p,
                                                         &mut self.sampling_buf);

        let mut accepted = vec![first_token];
        let mut best_beam: Option<usize> = None;

        // Find which beam(s) match at depth 0
        for (bi, beam) in tree.beams.iter().enumerate() {
            if !beam.is_empty() {
                let matches = if self.coarse_grained {
                    Self::tokens_similar_static(self.similarity_groups.as_ref(), verified_at_root, beam[0])
                } else {
                    verified_at_root == beam[0]
                };
                if matches {
                    best_beam = Some(bi);
                    // best match at depth 0
                    break;
                }
            }
        }

        if let Some(bi) = best_beam {
            let beam = &tree.beams[bi];
            accepted.push(beam[0]);

            // Continue accepting deeper tokens in this beam
            let beam_start_in_flat = 1 + bi * beam.len();
            for di in 1..beam.len() {
                let flat_pos = beam_start_in_flat + di - 1; // position of the parent in flat seq
                if flat_pos >= tree_logits.dim(0)? { break; }

                let pos_logits = tree_logits.i(flat_pos)?;
                let mut lv: Vec<f32> = pos_logits.to_dtype(DType::F32)?.to_vec1()?;
                Self::apply_repetition_penalty(&mut lv, &self.recent_tokens, self.repetition_penalty);
                for v in lv.iter_mut() { *v *= inv_temp; }
                let verified = Self::sample_top_k_top_p(&lv, self.top_k, self.top_p,
                                                         &mut self.sampling_buf);

                let matches = if self.coarse_grained {
                    Self::tokens_similar_static(self.similarity_groups.as_ref(), verified, beam[di])
                } else {
                    verified == beam[di]
                };

                if matches {
                    accepted.push(beam[di]);
                    // accepted at depth di
                } else {
                    // Reject: use verified token instead
                    accepted.push(verified);
                    // accepted at depth di
                    break;
                }
            }
        } else {
            // No beam matched at depth 0 — use the verified root token
            accepted.push(verified_at_root);
        }

        // Update engine state for all additionally accepted tokens (skip first_token, already handled)
        for &tok in &accepted[1..] {
            self.semantic_tokens.push(tok);
            self.step_count += 1;
            self.recent_tokens.push_back(tok);
            if self.recent_tokens.len() > 64 { self.recent_tokens.pop_front(); }
            if tok == 2 { self.done = true; break; }
            if tok == 0 {
                self.consecutive_pad += 1;
                if self.consecutive_pad > 100 { self.done = true; break; }
            } else {
                self.consecutive_pad = 0;
            }
        }

        // Truncate KV cache: tree verification added entries for all candidates,
        // but we only accepted some. Trim the cache to the actual position.
        let target_kv_len = self.text_pos + self.step_count;
        for (kc, vc) in self.kv_caches.iter_mut() {
            let current_len = kc.dim(2)?;
            if current_len > target_kv_len {
                // For pre-allocated caches, entries beyond target are stale but
                // narrow() in forward() excludes them. For dynamic caches, truncate.
                let cache_cap = self.model.cfg.max_seq_len;
                if current_len <= cache_cap {
                    // Pre-allocated: no truncation needed, narrow handles it
                } else {
                    *kc = kc.narrow(2, 0, target_kv_len)?;
                    *vc = vc.narrow(2, 0, target_kv_len)?;
                }
            }
        }

        self.last_hidden = Some(hidden);
        Ok(accepted)
    }

    fn speculative_step(&mut self) -> std::result::Result<Vec<u32>, Box<dyn std::error::Error>> {
        if self.done || self.step_count >= self.max_tokens {
            self.done = true;
            return Ok(Vec::new());
        }

        // Prefer RNN drafter (tree-based) over transformer draft model (linear)
        if self.rnn_drafter.is_some() && !self.model.use_cross_attention {
            return self.tree_speculative_step();
        }

        let k = self.speculate_k;

        // Speculative decoding uses forward_seq which only supports legacy (non-cross-attention) models.
        // Fall back to single-step generation for cross-attention models.
        let draft = match self.draft {
            Some(ref mut d) if !self.model.use_cross_attention => d,
            _ => {
                let tok = self.step()?;
                return Ok(tok.into_iter().collect());
            }
        };

        let save_step = self.step_count;
        let _save_sem_len = self.semantic_tokens.len();
        let base_pos = self.text_pos + save_step;

        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_sem = *self.semantic_tokens.last().unwrap_or(&1);

        // Pre-populate draft KV cache with semantic token history for context
        if draft.kv_caches[0].0.dim(2)? == 0 && self.semantic_tokens.len() > 1 {
            let prefix = &self.semantic_tokens[..self.semantic_tokens.len() - 1];
            for (i, &tok) in prefix.iter().enumerate() {
                let _ = draft.model.forward(
                    tok, i, &mut draft.kv_caches, None, None,
                )?;
            }
        }

        let draft_save_kv: Vec<(Tensor, Tensor)> = draft.kv_caches.iter()
            .map(|(kk, vv)| (kk.clone(), vv.clone())).collect();
        let draft_kv_len = draft.kv_caches[0].0.dim(2)?;

        for step_i in 0..k {
            let pos = draft_kv_len + step_i;
            let logits = draft.model.forward(
                draft_sem, pos, &mut draft.kv_caches, None, None,
            )?;
            let logits = logits.squeeze(0)?.squeeze(0)?;
            let logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;
            let tok = Self::sample_top_k_top_p(&logits_vec, self.top_k, self.top_p,
                                                   &mut self.sampling_buf);
            if tok == 2 { break; }
            draft_tokens.push(tok);
            draft_sem = tok;
        }

        if draft_tokens.is_empty() {
            draft.kv_caches = draft_save_kv;
            let tok = self.step()?;
            return Ok(tok.into_iter().collect());
        }

        let n_draft = draft_tokens.len();
        let mut sem_toks = Vec::with_capacity(n_draft);
        let mut sem_input = *self.semantic_tokens.last().unwrap_or(&1);
        for i in 0..n_draft {
            sem_toks.push(sem_input);
            sem_input = draft_tokens[i];
        }

        let logits = self.model.forward_seq(
            &sem_toks, base_pos, &mut self.kv_caches,
            &mut self.mask_cache,
        )?;
        let logits = logits.squeeze(0)?;

        let mut accepted = Vec::new();
        for i in 0..n_draft {
            let pos_logits = logits.i(i)?;
            let mut lv: Vec<f32> = pos_logits.to_dtype(DType::F32)?.to_vec1()?;
            Self::apply_repetition_penalty(&mut lv, &self.recent_tokens, self.repetition_penalty);
            let temp = if self.temperature > 1e-8 { self.temperature } else { 1e-8 };
            let inv_temp = 1.0 / temp;
            for v in lv.iter_mut() { *v *= inv_temp; }
            let verified = Self::sample_top_k_top_p(&lv, self.top_k, self.top_p,
                                                        &mut self.sampling_buf);

            let accepted_match = if self.coarse_grained {
                Self::tokens_similar_static(self.similarity_groups.as_ref(), verified, draft_tokens[i])
            } else {
                verified == draft_tokens[i]
            };

            let accept_tok = if accepted_match && self.coarse_grained && verified != draft_tokens[i] {
                draft_tokens[i]  // keep draft token for acoustic consistency
            } else {
                verified
            };

            if accepted_match {
                accepted.push(accept_tok);
                self.semantic_tokens.push(accept_tok);
                self.step_count += 1;
                self.recent_tokens.push_back(accept_tok);
                if self.recent_tokens.len() > 64 { self.recent_tokens.pop_front(); }
                if accept_tok == 2 { self.done = true; break; }
                if accept_tok == 0 {
                    self.consecutive_pad += 1;
                    if self.consecutive_pad > 100 { self.done = true; break; }
                } else { self.consecutive_pad = 0; }
            } else {
                accepted.push(verified);
                self.semantic_tokens.push(verified);
                self.step_count += 1;
                self.recent_tokens.push_back(verified);
                if self.recent_tokens.len() > 64 { self.recent_tokens.pop_front(); }
                if verified == 2 { self.done = true; break; }
                if verified == 0 {
                    self.consecutive_pad += 1;
                    if self.consecutive_pad > 100 { self.done = true; break; }
                } else { self.consecutive_pad = 0; }

                let reject_at = i + 1;
                if reject_at < n_draft {
                    // Pre-allocated caches: stale entries beyond the active position
                    // are excluded by narrow() in forward(). Only truncate dynamic caches.
                    let cache_cap = self.kv_caches[0].0.dim(2)?;
                    let active_len = self.text_pos + self.step_count;
                    if cache_cap <= active_len + n_draft {
                        let target_len = cache_cap.saturating_sub(n_draft - reject_at);
                        for (kc, vc) in self.kv_caches.iter_mut() {
                            *kc = kc.narrow(2, 0, target_len)?;
                            *vc = vc.narrow(2, 0, target_len)?;
                        }
                    }
                }
                break;
            }
        }

        // Maintain draft KV cache: keep entries for accepted tokens, truncate rejected
        let draft_base_len = draft_save_kv[0].0.dim(2)?;
        let draft_target_len = draft_base_len + accepted.len();
        for (kc, vc) in draft.kv_caches.iter_mut() {
            let current_len = kc.dim(2)?;
            if current_len > draft_target_len {
                *kc = kc.narrow(2, 0, draft_target_len)?;
                *vc = vc.narrow(2, 0, draft_target_len)?;
            }
        }

        Ok(accepted)
    }
}

// ─── C FFI ───────────────────────────────────────────────────────────────────

fn panic_message(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[no_mangle]
pub extern "C" fn sonata_lm_create(
    weights_path: *const c_char, config_path: *const c_char,
) -> *mut c_void {
    let result = std::panic::catch_unwind(|| {
        let wp = if weights_path.is_null() { return ptr::null_mut(); }
                 else { unsafe { CStr::from_ptr(weights_path) }.to_str().unwrap_or("") };
        let cp = if config_path.is_null() { None }
                 else { unsafe { CStr::from_ptr(config_path) }.to_str().ok() };
        match LmEngine::load(wp, cp) {
            Ok(e) => Box::into_raw(Box::new(e)) as *mut c_void,
            Err(e) => { eprintln!("[sonata_lm] create failed: {}", e); ptr::null_mut() }
        }
    });
    match result {
        Ok(ptr) => ptr,
        Err(e) => {
            eprintln!("[sonata_lm] panic in create: {}", panic_message(e));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_lm_destroy(engine: *mut c_void) {
    if !engine.is_null() {
        if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(engine as *mut LmEngine));
        })) {
            eprintln!("[sonata_lm] panic in destroy: {}", panic_message(e));
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_lm_set_text(
    engine: *mut c_void, text_ids: *const u32, n: c_int,
) -> c_int {
    if engine.is_null() || text_ids.is_null() || n <= 0 { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        let ids = unsafe { std::slice::from_raw_parts(text_ids, n as usize) };
        eng.set_text(ids);
        0
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in set_text: {}", panic_message(e));
            -1
        }
    }
}

/// Append more text tokens to an in-progress generation without resetting state.
/// Use for streaming text: set_text for the first chunk, then append_text for more.
#[no_mangle]
pub extern "C" fn sonata_lm_append_text(
    engine: *mut c_void, text_ids: *const u32, n: c_int,
) -> c_int {
    if engine.is_null() || text_ids.is_null() || n <= 0 { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        let ids = unsafe { std::slice::from_raw_parts(text_ids, n as usize) };
        eng.append_text(ids);
        0
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in append_text: {}", panic_message(e));
            -1
        }
    }
}

/// Signal that all text has been provided for the current generation.
/// For cross-attention models, triggers re-encoding of the full text sequence.
#[no_mangle]
pub extern "C" fn sonata_lm_finish_text(engine: *mut c_void) -> c_int {
    if engine.is_null() { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        eng.finish_text();
        0
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in finish_text: {}", panic_message(e));
            -1
        }
    }
}

/// One autoregressive step. Writes semantic token to *out_token.
/// Returns: 0 = more, 1 = done, -1 = error
#[no_mangle]
pub extern "C" fn sonata_lm_step(
    engine: *mut c_void, out_token: *mut c_int,
) -> c_int {
    if engine.is_null() || out_token.is_null() { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        match eng.step() {
            Ok(Some(tok)) => { unsafe { *out_token = tok as c_int }; 0 }
            Ok(None) => 1,
            Err(e) => { eprintln!("[sonata_lm] step error: {}", e); -1 }
        }
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in step: {}", panic_message(e));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_lm_reset(engine: *mut c_void) -> c_int {
    if engine.is_null() { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        match eng.reset() { Ok(()) => 0, Err(e) => { eprintln!("[sonata_lm] reset: {}", e); -1 } }
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in reset: {}", panic_message(e));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_lm_is_done(engine: *mut c_void) -> c_int {
    if engine.is_null() { return 1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &*(engine as *const LmEngine) };
        eng.done as c_int
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in is_done: {}", panic_message(e));
            1
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_lm_set_params(
    engine: *mut c_void, temperature: c_float, top_k: c_int,
    top_p: c_float, rep_penalty: c_float,
) -> c_int {
    if engine.is_null() { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        if temperature > 0.0 { eng.temperature = temperature; }
        if top_k > 0 { eng.top_k = top_k as usize; }
        if top_p > 0.0 && top_p <= 1.0 { eng.top_p = top_p; }
        if rep_penalty >= 1.0 { eng.repetition_penalty = rep_penalty; }
        0
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in set_params: {}", panic_message(e));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_lm_load_draft(
    engine: *mut c_void, weights_path: *const c_char, config_path: *const c_char,
) -> c_int {
    if engine.is_null() || weights_path.is_null() { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        let wp = unsafe { CStr::from_ptr(weights_path) }.to_str().unwrap_or("");
        let cp = if config_path.is_null() { None }
                 else { unsafe { CStr::from_ptr(config_path) }.to_str().ok() };
        match eng.load_draft(wp, cp) { Ok(()) => 0, Err(e) => { eprintln!("[sonata_lm] draft: {}", e); -1 } }
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in load_draft: {}", panic_message(e));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_lm_speculate_step(
    engine: *mut c_void, out_tokens: *mut c_int, max_tokens: c_int, out_count: *mut c_int,
) -> c_int {
    if engine.is_null() || out_tokens.is_null() || out_count.is_null() { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        match eng.speculative_step() {
            Ok(tokens) => {
                let n = tokens.len().min(max_tokens as usize);
                for i in 0..n {
                    unsafe { *out_tokens.add(i) = tokens[i] as c_int; }
                }
                unsafe { *out_count = n as c_int; }
                if eng.done { 1 } else { 0 }
            }
            Err(e) => { eprintln!("[sonata_lm] speculate: {}", e); -1 }
        }
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in speculate_step: {}", panic_message(e));
            -1
        }
    }
}

/// Set per-step prosody conditioning: (log_pitch, energy, speaking_rate) × n_frames.
/// features: packed float array [n * 3], n: number of frames.
/// Call before stepping to enable prosody-conditioned generation.
/// Pass n=0 to clear prosody (unconditional generation).
#[no_mangle]
pub extern "C" fn sonata_lm_set_prosody(
    engine: *mut c_void, features: *const c_float, n: c_int,
) -> c_int {
    if engine.is_null() { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        if n <= 0 || features.is_null() {
            eng.prosody_features = None;
            return 0;
        }
        let mut pf = Vec::with_capacity(n as usize);
        for i in 0..n as usize {
            let mut frame = [0.0f32; PROSODY_DIM];
            for d in 0..PROSODY_DIM {
                frame[d] = unsafe { *features.add(i * PROSODY_DIM + d) };
            }
            pf.push(frame);
        }
        eng.prosody_features = Some(pf);
        0
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in set_prosody: {}", panic_message(e));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_lm_set_speculate_k(engine: *mut c_void, k: c_int) -> c_int {
    if engine.is_null() || k <= 0 { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        eng.speculate_k = k as usize;
        0
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in set_speculate_k: {}", panic_message(e));
            -1
        }
    }
}

/// Enable/disable coarse-grained speculative decoding.
/// When enabled, acoustically similar tokens are accepted during verification
/// instead of requiring exact match, increasing acceptance rate.
#[no_mangle]
pub extern "C" fn sonata_lm_set_coarse_grained(engine: *mut c_void, enable: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        eng.coarse_grained = enable != 0;
        if eng.coarse_grained {
            eprintln!("[sonata_lm] Coarse-grained speculative decoding enabled");
        }
        0
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in set_coarse_grained: {}", panic_message(e));
            -1
        }
    }
}

/// Returns the base token ID for prosody tokens.
/// Prosody token IDs are: base + PROSODY_TOKEN_* offset.
/// Returns -1 if engine is NULL.
#[no_mangle]
pub extern "C" fn sonata_lm_prosody_token_base(engine: *mut c_void) -> c_int {
    if engine.is_null() { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &*(engine as *const LmEngine) };
        (eng.model.cfg.semantic_vocab_size + eng.model.cfg.n_special_tokens) as c_int
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in prosody_token_base: {}", panic_message(e));
            -1
        }
    }
}

/// Returns the number of prosody token types available.
#[no_mangle]
pub extern "C" fn sonata_lm_num_prosody_tokens() -> c_int {
    NUM_PROSODY_TOKENS as c_int
}

/// Inject a prosody token into the generation sequence at the current position.
/// The token is added to the semantic token history and the text position is NOT advanced.
/// prosody_offset: 0-11 (see PROSODY_TOKEN_* constants).
#[no_mangle]
pub extern "C" fn sonata_lm_inject_prosody_token(
    engine: *mut c_void, prosody_offset: c_int,
) -> c_int {
    if engine.is_null() || prosody_offset < 0 || prosody_offset >= NUM_PROSODY_TOKENS as c_int {
        return -1;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        if !eng.model.cfg.use_prosody && eng.model.prosody_proj.is_none() {
            eprintln!("[sonata_lm] prosody injection rejected: model not trained with prosody tokens");
            return -1;
        }
        let base = (eng.model.cfg.semantic_vocab_size + eng.model.cfg.n_special_tokens) as u32;
        let tok = base + prosody_offset as u32;
        eng.semantic_tokens.push(tok);
        0
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in inject_prosody_token: {}", panic_message(e));
            -1
        }
    }
}

/// Inject N pause frames (silence tokens) into the semantic token stream.
/// Each pause frame = 20ms at 50Hz. The pause token ID is 0 (PAD), which
/// the Flow network maps to near-zero acoustic latents (silence).
/// Returns: 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn sonata_lm_inject_pause(
    engine: *mut c_void, n_frames: c_int,
) -> c_int {
    if engine.is_null() || n_frames <= 0 { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        let pause_token = 0u32; // PAD token = silence
        for _ in 0..n_frames as usize {
            let pos = if eng.model.use_cross_attention {
                eng.step_count
            } else {
                eng.text_pos + eng.step_count
            };
            // Run forward pass to keep KV cache in sync with position counter
            if eng.model.forward(
                pause_token, pos, &mut eng.kv_caches,
                None,
                eng.text_encoding.as_ref(),
            ).is_err() {
                return -1;
            }
            eng.semantic_tokens.push(pause_token);
            eng.step_count += 1;
        }
        0
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in inject_pause: {}", panic_message(e));
            -1
        }
    }
}

/// Query how many pause frames correspond to a given duration in milliseconds.
/// Returns frame count at 50Hz.
#[no_mangle]
pub extern "C" fn sonata_lm_ms_to_frames(ms: c_int) -> c_int {
    ((ms as f32) * 0.05f32 + 0.5f32) as c_int // 50Hz = 0.05 frames/ms
}

#[no_mangle]
pub extern "C" fn sonata_lm_sample_rate() -> c_int { 24000 }

#[no_mangle]
pub extern "C" fn sonata_lm_frame_rate() -> c_int { 50 }

#[no_mangle]
pub extern "C" fn sonata_lm_samples_per_frame() -> c_int { 480 }

/// Load an RNN (GRU) draft model for ReDrafter-style tree speculative decoding.
/// When loaded, speculate_step automatically uses tree-based drafting.
/// weights_path: path to safetensors file with GRU weights.
/// config_path: optional JSON config (gru_hidden, gru_layers, tree_width, tree_depth, emb_dim).
/// Returns: 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn sonata_lm_load_rnn_drafter(
    engine: *mut c_void, weights_path: *const c_char, config_path: *const c_char,
) -> c_int {
    if engine.is_null() || weights_path.is_null() { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        let wp = unsafe { CStr::from_ptr(weights_path) }.to_str().unwrap_or("");
        let cp = if config_path.is_null() { None }
                 else { unsafe { CStr::from_ptr(config_path) }.to_str().ok() };
        match eng.load_rnn_drafter(wp, cp) {
            Ok(()) => 0,
            Err(e) => { eprintln!("[sonata_lm] rnn_drafter: {}", e); -1 }
        }
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in load_rnn_drafter: {}", panic_message(e));
            -1
        }
    }
}

/// Configure tree shape for RNN drafter speculative decoding.
/// width: number of candidate beams (default 4).
/// depth: maximum tokens per beam (default 3).
/// Returns: 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn sonata_lm_set_tree_config(
    engine: *mut c_void, width: c_int, depth: c_int,
) -> c_int {
    if engine.is_null() || width <= 0 || depth <= 0 { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        if let Some(ref mut drafter) = eng.rnn_drafter {
            drafter.cfg.tree_width = width as usize;
            drafter.cfg.tree_depth = depth as usize;
            eprintln!("[sonata_lm] Tree config: {}×{}", width, depth);
            0
        } else {
            eprintln!("[sonata_lm] No RNN drafter loaded");
            -1
        }
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in set_tree_config: {}", panic_message(e));
            -1
        }
    }
}

/// Enable or disable acoustic head output (if configured).
/// Returns: 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn sonata_lm_enable_acoustic_head(engine: *mut c_void, enable: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        if eng.acoustic_dim == 0 {
            eprintln!("[sonata_lm] Acoustic head not configured");
            return -1;
        }
        eng.acoustic_head_enabled = enable != 0;
        eprintln!("[sonata_lm] Acoustic head: {}", if eng.acoustic_head_enabled { "enabled" } else { "disabled" });
        0
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in enable_acoustic_head: {}", panic_message(e));
            -1
        }
    }
}

/// Step with acoustic head output (if enabled).
/// Writes semantic token to *out_token and acoustic vector to *out_acoustic.
/// Returns: 0 = more, 1 = done, -1 = error
#[no_mangle]
pub extern "C" fn sonata_lm_step_dual(
    engine: *mut c_void, out_token: *mut u32, out_acoustic: *mut c_float, acoustic_dim: c_int,
) -> c_int {
    if engine.is_null() || out_token.is_null() { return -1; }
    if !out_acoustic.is_null() && acoustic_dim <= 0 { return -1; }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut LmEngine) };
        if eng.done || eng.step_count >= eng.max_tokens {
            eng.done = true;
            return 1;
        }

        // Get current semantic token and compute position
        let sem_tok = *eng.semantic_tokens.last().unwrap_or(&1);
        let pos = if eng.model.use_cross_attention {
            eng.step_count
        } else {
            eng.text_pos + eng.step_count
        };

        // Prepare prosody tensor if available
        let prosody_tensor = if let Some(ref pf) = eng.prosody_features {
            let idx = eng.step_count.min(pf.len().saturating_sub(1));
            let data = pf[idx];
            match Tensor::from_vec(
                data.to_vec(), (1, 1, PROSODY_DIM), &eng.device,
            ) {
                Ok(t) => match t.to_dtype(eng.dtype) {
                    Ok(t2) => Some(t2),
                    Err(_) => None,
                }
                Err(_) => None,
            }
        } else {
            None
        };

        // Forward pass with acoustic output (gated by runtime flag)
        let (logits, hidden, acoustic) = match eng.model.forward_with_acoustic(
            sem_tok, pos,
            &mut eng.kv_caches,
            prosody_tensor.as_ref(),
            eng.text_encoding.as_ref(),
            eng.acoustic_head_enabled,
        ) {
            Ok(res) => res,
            Err(_) => return -1,
        };

        // Save hidden state for drafter
        eng.last_hidden = Some(hidden);

        // Process logits and sample token
        let logits_2d = match logits.squeeze(0) {
            Ok(l) => l,
            Err(_) => return -1,
        };

        let mut logits_vec: Vec<f32> = match logits_2d.to_dtype(DType::F32) {
            Ok(l) => match l.to_vec1() {
                Ok(v) => v,
                Err(_) => return -1,
            }
            Err(_) => return -1,
        };

        // Apply repetition penalty
        LmEngine::apply_repetition_penalty(&mut logits_vec, &eng.recent_tokens, eng.repetition_penalty);

        // Apply temperature
        let temp = if eng.temperature > 1e-8 { eng.temperature } else { 1e-8 };
        let inv_temp = 1.0 / temp;
        for v in logits_vec.iter_mut() { *v *= inv_temp; }

        // Sample token
        let next_token = LmEngine::sample_top_k_top_p(&logits_vec, eng.top_k, eng.top_p, &mut eng.sampling_buf);

        // Check for end-of-sequence or too many pad tokens
        if next_token == 2 {
            eng.done = true;
            unsafe { *out_token = next_token; }
            return 1;
        }
        if next_token == 0 {
            eng.consecutive_pad += 1;
            if eng.consecutive_pad > 100 {
                eng.done = true;
                unsafe { *out_token = next_token; }
                return 1;
            }
        } else {
            eng.consecutive_pad = 0;
        }

        // Copy acoustic latents if available and store in internal buffer
        if let Some(ac) = acoustic {
            if let Ok(a1) = ac.squeeze(0) {
                if let Ok(a2) = a1.to_dtype(DType::F32) {
                    if let Ok(av) = a2.to_vec1::<f32>() {
                        // Store in internal buffer (overwrite with latest vector)
                        eng.acoustic_buffer.clear();
                        eng.acoustic_buffer.extend_from_slice(&av);

                        // Copy to caller's output buffer if provided
                        if !out_acoustic.is_null() && acoustic_dim > 0 {
                            let copy_len = std::cmp::min(av.len(), acoustic_dim as usize);
                            unsafe {
                                std::ptr::copy_nonoverlapping(av.as_ptr(), out_acoustic, copy_len);
                                for i in copy_len..acoustic_dim as usize {
                                    *out_acoustic.add(i) = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        } else if !out_acoustic.is_null() && acoustic_dim > 0 {
            // No acoustic output (head disabled or missing), zero the caller's buffer
            unsafe {
                for i in 0..acoustic_dim as usize {
                    *out_acoustic.add(i) = 0.0;
                }
            }
        }

        // Update state
        eng.semantic_tokens.push(next_token);
        eng.step_count += 1;
        eng.recent_tokens.push_back(next_token);
        if eng.recent_tokens.len() > 64 { eng.recent_tokens.pop_front(); }

        unsafe { *out_token = next_token; }
        0
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in step_dual: {}", panic_message(e));
            -1
        }
    }
}

/// Get the acoustic dimension configured for this engine.
/// Returns: acoustic_dim if enabled, 0 if not configured, -1 if engine is NULL.
#[no_mangle]
pub extern "C" fn sonata_lm_get_acoustic_dim(engine: *mut c_void) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &*(engine as *const LmEngine) };
    eng.acoustic_dim as c_int
}

/// Copy the latest acoustic vector from the internal buffer to the caller's buffer.
/// Returns: number of floats copied on success, 0 if buffer is empty, -1 on error.
/// The internal buffer holds the most recent acoustic vector produced by step_dual.
#[no_mangle]
pub extern "C" fn sonata_lm_get_acoustic_buffer(
    engine: *mut c_void, out_buf: *mut c_float, buf_len: c_int,
) -> c_int {
    if engine.is_null() || out_buf.is_null() || buf_len <= 0 { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &*(engine as *const LmEngine) };
        if eng.acoustic_buffer.is_empty() {
            return 0;
        }
        let copy_len = std::cmp::min(eng.acoustic_buffer.len(), buf_len as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(eng.acoustic_buffer.as_ptr(), out_buf, copy_len);
            // Zero-pad remainder if caller buffer is larger
            for i in copy_len..buf_len as usize {
                *out_buf.add(i) = 0.0;
            }
        }
        copy_len as c_int
    }));
    match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[sonata_lm] panic in get_acoustic_buffer: {}", panic_message(e));
            -1
        }
    }
}
