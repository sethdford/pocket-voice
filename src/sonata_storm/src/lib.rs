//! Sonata Storm — SoundStorm MaskGIT-style parallel semantic token predictor.
//!
//! Predicts ALL semantic tokens simultaneously via iterative refinement.
//! 10-50x faster than autoregressive sonata_lm for inference.
//!
//! Architecture: Text encoder + bidirectional predictor blocks with cross-attention.
//! Generation: Start all MASK → predict → keep confident → re-mask uncertain → repeat.
//!
//! C FFI:
//!   sonata_storm_create(weights, config) -> *engine
//!   sonata_storm_set_text(engine, text_ids, n) -> 0/-1
//!   sonata_storm_generate(engine, out_tokens, max_tokens, out_count) -> 0/-1
//!   sonata_storm_set_params(engine, temperature, n_rounds) -> 0/-1
//!   sonata_storm_reset(engine) -> 0/-1
//!   sonata_storm_sample_rate() -> 24000
//!   sonata_storm_frame_rate() -> 50

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{embedding, linear, linear_no_bias, rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder};
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use std::ffi::{c_char, c_float, c_int, c_void, CStr};
use std::path::Path;
use std::ptr;

const MASK_TOKEN: u32 = 3;

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
struct StormConfig {
    #[serde(default = "default_d_model")]
    d_model: usize,
    #[serde(default = "default_n_layers")]
    n_layers: usize,
    #[serde(default = "default_n_heads")]
    n_heads: usize,
    #[serde(default = "default_n_kv_heads")]
    n_kv_heads: usize,
    #[serde(default = "default_d_ff")]
    d_ff: usize,
    #[serde(default = "default_text_vocab")]
    text_vocab_size: usize,
    #[serde(default = "default_semantic_vocab")]
    semantic_vocab_size: usize,
    #[serde(default = "default_n_special")]
    n_special_tokens: usize,
    #[serde(default = "default_max_seq")]
    max_seq_len: usize,
    #[serde(default = "default_n_text_layers")]
    n_text_layers: usize,
    #[serde(default = "default_norm_eps")]
    norm_eps: f64,
}

fn default_d_model() -> usize { 1024 }
fn default_n_layers() -> usize { 16 }
fn default_n_heads() -> usize { 16 }
fn default_n_kv_heads() -> usize { 4 }
fn default_d_ff() -> usize { 2560 }
fn default_text_vocab() -> usize { 32000 }
fn default_semantic_vocab() -> usize { 32768 }
fn default_n_special() -> usize { 4 }
fn default_max_seq() -> usize { 4096 }
fn default_n_text_layers() -> usize { 4 }
fn default_norm_eps() -> f64 { 1e-5 }

impl StormConfig {
    fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
    fn n_rep(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
    fn total_semantic(&self) -> usize {
        self.semantic_vocab_size + self.n_special_tokens
    }
}

// ─── Bidirectional Attention (no causal mask) ──────────────────────────────────

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, h, t, d) = x.dims4()?;
    // unsqueeze(2) → expand → reshape: correctly repeats each KV head n_rep times
    x.unsqueeze(2)?
        .expand((b, h, n_rep, t, d))?
        .reshape((b, h * n_rep, t, d))
}

// Python (modules.BidirectionalAttention): fused qkv (Linear d_model→3*d_model) + out.
struct BidirectionalAttention {
    qkv: Linear,
    out: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl BidirectionalAttention {
    fn load(cfg: &StormConfig, vb: VarBuilder) -> Result<Self> {
        let hd = cfg.head_dim();
        Ok(Self {
            qkv: linear_no_bias(cfg.d_model, 3 * cfg.d_model, vb.pp("qkv"))?,
            out: linear_no_bias(cfg.d_model, cfg.d_model, vb.pp("out"))?,
            n_heads: cfg.n_heads,
            head_dim: hd,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape((b, t, 3, self.n_heads, self.head_dim))?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?;
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.t()?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, ()))?;
        self.out.forward(&out)
    }
}

// ─── Cross Attention ─────────────────────────────────────────────────────────

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
    fn load(cfg: &StormConfig, vb: VarBuilder) -> Result<Self> {
        let hd = cfg.head_dim();
        Ok(Self {
            wq: linear_no_bias(cfg.d_model, cfg.n_heads * hd, vb.pp("wq"))?,
            wk: linear_no_bias(cfg.d_model, cfg.n_kv_heads * hd, vb.pp("wk"))?,
            wv: linear_no_bias(cfg.d_model, cfg.n_kv_heads * hd, vb.pp("wv"))?,
            wo: linear_no_bias(cfg.n_heads * hd, cfg.d_model, vb.pp("wo"))?,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: hd,
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

        let k = repeat_kv(&k, self.n_rep)?;
        let v = repeat_kv(&v, self.n_rep)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.t()?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t_a, ()))?;
        self.wo.forward(&out)
    }
}

// ─── SwiGLU FFN ──────────────────────────────────────────────────────────────

struct SwiGluFfn {
    w_gate: Linear,
    w_up: Linear,
    w_down: Linear,
}

impl SwiGluFfn {
    fn load(cfg: &StormConfig, vb: VarBuilder) -> Result<Self> {
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

// ─── Text Encoder Block ──────────────────────────────────────────────────────
// Python: norm1, attn (BidirectionalAttention with qkv+out), norm2, ffn

struct TextEncoderBlock {
    attn_norm: RmsNorm,
    attn: BidirectionalAttention,
    ffn_norm: RmsNorm,
    ffn: SwiGluFfn,
}

impl TextEncoderBlock {
    fn load(cfg: &StormConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn_norm: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("norm1"))?,
            attn: BidirectionalAttention::load(cfg, vb.pp("attn"))?,
            ffn_norm: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("norm2"))?,
            ffn: SwiGluFfn::load(cfg, vb.pp("ffn"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.attn.forward(&self.attn_norm.forward(x)?)?;
        let x = (x + h)?;
        let h = self.ffn.forward(&self.ffn_norm.forward(&x)?)?;
        x + h
    }
}

// ─── SoundStorm Block ───────────────────────────────────────────────────────
// Python: norm1, self_attn (BidirectionalAttention qkv+out), norm2 (pre-cross-attn),
// cross_attn (CrossAttention wq/wk/wv/wo), norm3 (pre-FFN), ffn

struct SoundStormBlock {
    norm1: RmsNorm,
    self_attn: BidirectionalAttention,
    cross_norm: RmsNorm,
    cross_attn: CrossAttention,
    norm3: RmsNorm,
    ffn: SwiGluFfn,
}

impl SoundStormBlock {
    fn load(cfg: &StormConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("norm1"))?,
            self_attn: BidirectionalAttention::load(cfg, vb.pp("self_attn"))?,
            cross_norm: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("norm2"))?,
            cross_attn: CrossAttention::load(cfg, vb.pp("cross_attn"))?,
            norm3: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("norm3"))?,
            ffn: SwiGluFfn::load(cfg, vb.pp("ffn"))?,
        })
    }

    fn forward(&self, x: &Tensor, text_enc: &Tensor, step_emb: &Tensor) -> Result<Tensor> {
        // x + step_emb + self_attn(norm1(x))
        let h = self.self_attn.forward(&self.norm1.forward(x)?)?;
        let x = (x + step_emb + h)?;
        // x + cross_attn(cross_norm(x), text_enc)
        let h = self.cross_attn.forward(&self.cross_norm.forward(&x)?, text_enc)?;
        let x = (x + h)?;
        // x + ffn(norm3(x))
        let h = self.ffn.forward(&self.norm3.forward(&x)?)?;
        x + h
    }
}

// ─── Step Embedding ─────────────────────────────────────────────────────────

struct StepEmbedding {
    linear1: Linear,
    linear2: Linear,
}

impl StepEmbedding {
    fn load(cfg: &StormConfig, vb: VarBuilder) -> Result<Self> {
        // step_emb.0: Linear(1, d_model), step_emb.2: Linear(d_model, d_model) — with bias
        let linear1 = linear(1, cfg.d_model, vb.pp("0"))?;
        let linear2 = linear(cfg.d_model, cfg.d_model, vb.pp("2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, step_ratio: f32, device: &Device, dtype: DType) -> Result<Tensor> {
        let t = Tensor::from_vec(vec![step_ratio], (1, 1), device)?.to_dtype(dtype)?;
        let h = candle_nn::Activation::Silu.forward(&self.linear1.forward(&t)?)?;
        self.linear2.forward(&h)
    }
}

// ─── Sonata Storm Model ──────────────────────────────────────────────────────

struct SonataStorm {
    text_emb: Embedding,
    text_pos: Embedding,
    text_encoder: Vec<TextEncoderBlock>,
    text_norm: RmsNorm,

    semantic_emb: Embedding,
    pos_emb: Embedding,
    step_emb: StepEmbedding,
    blocks: Vec<SoundStormBlock>,
    output_norm: RmsNorm,
    head: Linear,

    cfg: StormConfig,
}

impl SonataStorm {
    fn load(cfg: &StormConfig, vb: VarBuilder, _device: &Device, _dtype: DType) -> Result<Self> {
        let text_total = cfg.text_vocab_size + cfg.n_special_tokens;
        let total_semantic = cfg.total_semantic();

        let text_emb = embedding(text_total, cfg.d_model, vb.pp("text_emb"))?;
        let text_pos = embedding(cfg.max_seq_len, cfg.d_model, vb.pp("text_pos"))?;

        let mut text_encoder = Vec::with_capacity(cfg.n_text_layers);
        for i in 0..cfg.n_text_layers {
            text_encoder.push(TextEncoderBlock::load(cfg, vb.pp(format!("text_encoder.{i}")))?);
        }
        let text_norm = rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("text_norm"))?;

        let semantic_emb = embedding(total_semantic, cfg.d_model, vb.pp("semantic_emb"))?;
        let pos_emb = embedding(cfg.max_seq_len, cfg.d_model, vb.pp("pos_emb"))?;
        let step_emb = StepEmbedding::load(cfg, vb.pp("step_emb"))?;

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            blocks.push(SoundStormBlock::load(cfg, vb.pp(format!("blocks.{i}")))?);
        }

        let output_norm = rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("output_norm"))?;
        let head = linear_no_bias(cfg.d_model, cfg.semantic_vocab_size, vb.pp("head"))?;

        Ok(Self {
            text_emb,
            text_pos,
            text_encoder,
            text_norm,
            semantic_emb,
            pos_emb,
            step_emb,
            blocks,
            output_norm,
            head,
            cfg: cfg.clone(),
        })
    }

    fn encode_text(&self, text_tokens: &[u32], device: &Device) -> Result<Tensor> {
        let t_len = text_tokens.len().max(1);
        let text_t = if text_tokens.is_empty() {
            Tensor::zeros((1, 1), DType::U32, device)?
        } else {
            // Text token ID offset: The C pipeline passes raw IDs from either (1) SPM tokenizer
            // (spm_encode → 0..vocab_size-1) or (2) phonemizer (phonemizer_text_to_ids →
            // 0..phoneme_map_size-1). Neither pre-offsets for special tokens. The Python model
            // reserves embedding indices 0..n_special_tokens-1 for PAD/BOS/EOS/MASK; content
            // tokens occupy n_special_tokens..text_vocab_size+n_special_tokens-1. We add
            // n_special_tokens so raw vocab ID 0 maps to embedding row 4, etc.
            let shifted: Vec<u32> = text_tokens
                .iter()
                .map(|&t| t + self.cfg.n_special_tokens as u32)
                .collect();
            Tensor::from_vec(shifted, (1, t_len), device)?
        };

        let pos = Tensor::arange(0u32, t_len as u32, device)?.unsqueeze(0)?;
        let mut x = (self.text_emb.forward(&text_t)? + self.text_pos.forward(&pos)?)?;

        for block in &self.text_encoder {
            x = block.forward(&x)?;
        }
        self.text_norm.forward(&x)
    }

    fn forward(
        &self,
        masked_tokens: &Tensor,
        text_enc: &Tensor,
        step_ratio: f32,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let (_, seq_len) = masked_tokens.dims2()?;
        let pos = Tensor::arange(0u32, seq_len as u32, device)?.unsqueeze(0)?;
        let x = (self.semantic_emb.forward(masked_tokens)?
            + self.pos_emb.forward(&pos)?)?;

        let step_emb = self.step_emb.forward(step_ratio, device, dtype)?;
        let step_emb = step_emb.unsqueeze(1)?; // [1, 1, d_model] for broadcast

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x, text_enc, &step_emb)?;
        }

        self.head.forward(&self.output_norm.forward(&x)?)
    }
}

// ─── Engine ──────────────────────────────────────────────────────────────────

struct StormEngine {
    model: SonataStorm,
    device: Device,
    dtype: DType,

    text_tokens: Vec<u32>,
    text_encoding: Option<Tensor>,

    temperature: f32,
    n_rounds: i32,
}

fn resolve_weights_path(
    path: &str,
    candidates: &[&str],
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    if Path::new(path).exists() {
        return Ok(path.to_string());
    }
    if path.contains('/')
        && !path.starts_with('.')
        && !path.starts_with('/')
    {
        eprintln!(
            "[sonata_storm] '{}' not found locally, downloading from HuggingFace...",
            path
        );
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(path.to_string());
        for f in candidates {
            if let Ok(p) = repo.get(f) {
                let s = p.to_string_lossy().to_string();
                eprintln!("[sonata_storm] Downloaded: {}", s);
                return Ok(s);
            }
        }
        Err(format!("No matching files in HF repo '{}'", path).into())
    } else {
        Err(format!("File not found: {}", path).into())
    }
}

impl StormEngine {
    fn load(
        weights_path: &str,
        config_path: Option<&str>,
    ) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let device = if candle_core::utils::metal_is_available() {
            Device::new_metal(0)?
        } else {
            eprintln!("[sonata_storm] Metal not available, using CPU");
            Device::Cpu
        };

        let cfg: StormConfig = if let Some(cp) = config_path {
            let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(cp)?)?;
            let d_model = raw.get("d_model").and_then(|v| v.as_u64()).unwrap_or(1024) as usize;
            let d_ff = raw.get("d_ff")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or_else(|| {
                    let ffn_mult = raw.get("ffn_mult")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(4.0);
                    (d_model as f64 * ffn_mult) as usize
                });
            StormConfig {
                d_model,
                n_layers: raw.get("n_layers").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
                n_heads: raw.get("n_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
                n_kv_heads: raw.get("n_kv_heads").and_then(|v| v.as_u64()).unwrap_or(4) as usize,
                d_ff,
                text_vocab_size: raw.get("text_vocab_size").and_then(|v| v.as_u64()).unwrap_or(32000) as usize,
                semantic_vocab_size: raw.get("semantic_vocab_size").and_then(|v| v.as_u64()).unwrap_or(32768) as usize,
                n_special_tokens: raw.get("n_special_tokens").and_then(|v| v.as_u64()).unwrap_or(4) as usize,
                max_seq_len: raw.get("max_seq_len").and_then(|v| v.as_u64()).unwrap_or(4096) as usize,
                n_text_layers: raw.get("n_text_layers").and_then(|v| v.as_u64()).unwrap_or(4) as usize,
                norm_eps: raw.get("norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5),
            }
        } else {
            StormConfig {
                d_model: 1024,
                n_layers: 16,
                n_heads: 16,
                n_kv_heads: 4,
                d_ff: 2560,
                text_vocab_size: 32000,
                semantic_vocab_size: 32768,
                n_special_tokens: 4,
                max_seq_len: 4096,
                n_text_layers: 4,
                norm_eps: 1e-5,
            }
        };

        let dtype = DType::F16;
        eprintln!(
            "[sonata_storm] Config: d={}, L={}, H={}, KV={}, FF={}, semantic_vocab={}, n_text_layers={}",
            cfg.d_model,
            cfg.n_layers,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.d_ff,
            cfg.semantic_vocab_size,
            cfg.n_text_layers
        );

        let resolved = resolve_weights_path(weights_path, &["model.safetensors", "sonata_storm.safetensors"])?;
        eprintln!(
            "[sonata_storm] Loading FP16 weights from {} (~2x Metal throughput)...",
            resolved
        );

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&resolved], dtype, &device)? };
        let model = SonataStorm::load(&cfg, vb, &device, dtype)?;
        eprintln!(
            "[sonata_storm] Model loaded on {:?} (dtype={:?})",
            device,
            dtype
        );

        // Metal warmup
        {
            let t0 = std::time::Instant::now();
            let text_enc = model.encode_text(&[0u32], &device)?;
            let tokens = Tensor::full(MASK_TOKEN as i64, (1, 4), &device)?;
            let _ = model.forward(&tokens, &text_enc, 0.5f32, &device, dtype);
            eprintln!(
                "[sonata_storm] Metal shader warmup: {:.1}ms",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }

        Ok(Self {
            model,
            device,
            dtype,
            text_tokens: Vec::new(),
            text_encoding: None,
            temperature: 1.0,
            n_rounds: 8,
        })
    }

    fn set_text(&mut self, ids: &[u32]) -> Result<()> {
        self.text_tokens = ids.to_vec();
        self.text_encoding = None;
        if ids.is_empty() {
            return Ok(());
        }
        let enc = self.model.encode_text(ids, &self.device)?;
        self.text_encoding = Some(enc);
        Ok(())
    }

    fn generate(&mut self, seq_len: usize, out_tokens: &mut [i32]) -> Result<usize> {
        let n_rounds = self.n_rounds.max(1) as usize;
        let temperature = self.temperature.max(0.0);

        // Text encoding: use cached or create minimal for empty text
        let text_enc = if let Some(ref enc) = self.text_encoding {
            enc.clone()
        } else {
            self.model.encode_text(&[0u32], &self.device)?
        };

        // Start with all MASK
        let mut tokens: Vec<u32> = vec![MASK_TOKEN; seq_len];
        let mut rng = rand::rngs::StdRng::from_entropy();

        for step in 0..n_rounds {
            let mask: Vec<bool> = tokens.iter().map(|&t| t == MASK_TOKEN).collect();
            let n_masked: usize = mask.iter().filter(|&&b| b).count();

            if n_masked == 0 {
                break;
            }

            // Linear masking schedule (matches Python): mask_ratio = 1.0 - step / n_steps
            let mask_ratio = 1.0 - (step as f64) / (n_rounds as f64);
            let step_ratio = mask_ratio as f32;

            let tokens_t = Tensor::from_vec(
                tokens.iter().map(|&t| t as i64).collect::<Vec<_>>(),
                (1, seq_len),
                &self.device,
            )?;

            let logits = self.model.forward(
                &tokens_t,
                &text_enc,
                step_ratio,
                &self.device,
                self.dtype,
            )?;

            let logits_f32 = logits.to_dtype(DType::F32)?.squeeze(0)?;
            let logits_2d = logits_f32.to_vec2::<f32>()?;

            let inv_temp = if temperature > 0.0 { 1.0 / temperature } else { 1e10 };
            let mut sampled: Vec<u32> = vec![0; seq_len];
            let mut confidence: Vec<f32> = vec![0.0; seq_len];

            for pos in 0..seq_len {
                let row = &logits_2d[pos];
                let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_vals: Vec<f32> = row
                    .iter()
                    .map(|&v| ((v - max_logit) * inv_temp).exp())
                    .collect();
                let sum: f32 = exp_vals.iter().sum();
                if sum > 0.0 {
                    for v in exp_vals.iter_mut() {
                        *v /= sum;
                    }
                }

                let max_prob = exp_vals.iter().cloned().fold(0.0f32, f32::max);
                confidence[pos] = if mask[pos] { max_prob } else { 1e9 };

                let tok = if temperature > 0.0 {
                    let r: f32 = rng.gen();
                    let mut cum = 0.0f32;
                    let mut chosen = (exp_vals.len() - 1) as u32;
                    for (i, &p) in exp_vals.iter().enumerate() {
                        cum += p;
                        if cum >= r {
                            chosen = i as u32;
                            break;
                        }
                    }
                    chosen
                } else {
                    exp_vals
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)
                };
                sampled[pos] = tok;
            }

            // Update masked positions only
            for pos in 0..seq_len {
                if mask[pos] {
                    tokens[pos] = sampled[pos];
                }
            }

            // Re-mask least confident for next round (linear schedule, matches Python)
            let next_mask_ratio = 1.0 - ((step + 1) as f64) / (n_rounds as f64);
            let n_remask = (next_mask_ratio * seq_len as f64).floor().max(0.0) as usize;
            if n_remask > 0 && step < n_rounds - 1 {
                let mut idx_conf: Vec<(usize, f32)> = (0..seq_len)
                    .map(|i| (i, confidence[i]))
                    .collect();
                idx_conf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                for i in 0..n_remask.min(idx_conf.len()) {
                    tokens[idx_conf[i].0] = MASK_TOKEN;
                }
            }
        }

        for (i, &t) in tokens.iter().enumerate().take(out_tokens.len()) {
            out_tokens[i] = t as i32;
        }
        Ok(seq_len)
    }

    fn reset(&mut self) -> Result<()> {
        self.text_tokens.clear();
        self.text_encoding = None;
        Ok(())
    }
}

// ─── C FFI ───────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn sonata_storm_create(
    weights_path: *const c_char,
    config_path: *const c_char,
) -> *mut c_void {
    let result = std::panic::catch_unwind(|| {
        let wp = if weights_path.is_null() {
            return ptr::null_mut();
        } else {
            unsafe { CStr::from_ptr(weights_path) }
                .to_str()
                .unwrap_or("")
        };
        let cp = if config_path.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(config_path) }.to_str().ok()
        };
        match StormEngine::load(wp, cp) {
            Ok(e) => Box::into_raw(Box::new(e)) as *mut c_void,
            Err(e) => {
                eprintln!("[sonata_storm] create failed: {}", e);
                ptr::null_mut()
            }
        }
    });
    result.unwrap_or(ptr::null_mut())
}

#[no_mangle]
pub extern "C" fn sonata_storm_destroy(engine: *mut c_void) {
    if !engine.is_null() {
        let _ = std::panic::catch_unwind(|| unsafe {
            drop(Box::from_raw(engine as *mut StormEngine));
        });
    }
}

#[no_mangle]
pub extern "C" fn sonata_storm_set_text(
    engine: *mut c_void,
    text_ids: *const u32,
    n: c_int,
) -> c_int {
    if engine.is_null() {
        return -1;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut StormEngine) };
        let ids = if text_ids.is_null() || n < 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(text_ids, n as usize) }
        };
        match eng.set_text(ids) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[sonata_storm] set_text: {}", e);
                -1
            }
        }
    }));
    result.unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn sonata_storm_generate(
    engine: *mut c_void,
    out_tokens: *mut i32,
    max_tokens: c_int,
    out_count: *mut c_int,
) -> c_int {
    if engine.is_null() || out_tokens.is_null() || out_count.is_null() || max_tokens <= 0 {
        return -1;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut StormEngine) };
        let seq_len = max_tokens as usize;
        let buf = unsafe { std::slice::from_raw_parts_mut(out_tokens, seq_len) };
        match eng.generate(seq_len, buf) {
            Ok(n) => {
                unsafe { *out_count = n as c_int };
                0
            }
            Err(e) => {
                eprintln!("[sonata_storm] generate: {}", e);
                -1
            }
        }
    }));
    result.unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn sonata_storm_set_params(
    engine: *mut c_void,
    temperature: c_float,
    n_rounds: c_int,
) -> c_int {
    if engine.is_null() {
        return -1;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut StormEngine) };
        if temperature >= 0.0 {
            eng.temperature = temperature;
        }
        if n_rounds > 0 {
            eng.n_rounds = n_rounds;
        }
        0
    }));
    result.unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn sonata_storm_reset(engine: *mut c_void) -> c_int {
    if engine.is_null() {
        return -1;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut StormEngine) };
        match eng.reset() {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[sonata_storm] reset: {}", e);
                -1
            }
        }
    }));
    result.unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn sonata_storm_sample_rate() -> c_int {
    24000
}

#[no_mangle]
pub extern "C" fn sonata_storm_frame_rate() -> c_int {
    50
}
