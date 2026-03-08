//! Sonata Flow + Decoder — Rust/Candle inference on Metal GPU.
//!
//! Flow: semantic tokens → acoustic latents (Euler ODE with optional CFG)
//! Decoder: (semantic_codes, acoustic_latents) → waveform
//!
//! Supports two decoder types:
//!   - VocosDecoder: ConvNeXt backbone + mag/phase heads (trained model) → iSTFT in C
//!   - ConvDecoder: ConvNeXt backbone + ConvTranspose upsample → direct audio
//!
//! Features:
//!   - Speaker conditioning via embedding table + additive injection
//!   - Classifier-free guidance (CFG) for improved conditioning adherence
//!   - Configurable ODE steps (4 for real-time, 8-16 for quality)

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::ffi::{c_char, c_float, c_int, c_void, CStr};
use std::path::Path;
use safetensors::SafeTensors;

// ─── Quality Mode Constants ──────────────────────────────────────────────────
const FLOW_QUALITY_FAST: i32 = 0;      // 4 steps, Euler only
const FLOW_QUALITY_BALANCED: i32 = 1;  // 6 steps, Euler only
const FLOW_QUALITY_HIGH: i32 = 2;      // 8 steps, Heun

fn panic_message(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct FlowConfig {
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    acoustic_dim: usize,
    cond_dim: usize,
    semantic_vocab_size: usize,
    n_steps_inference: usize,
    #[serde(default = "default_sigma_min")]
    sigma_min: f64,
    #[serde(default)]
    n_speakers: usize,
    #[serde(default = "default_speaker_dim")]
    speaker_dim: usize,
    #[serde(default)]
    n_emotions: usize,
    #[serde(default = "default_speaker_dim")]
    emotion_dim: usize,
    #[serde(default = "default_prosody_dim")]
    prosody_dim: usize,
    #[serde(default = "default_prosody_embedding_dim")]
    prosody_embedding_dim: usize,
    #[serde(default)]
    n_experts: usize,
    #[serde(default = "default_top_k")]
    top_k_experts: usize,
    #[serde(default = "default_moe_every_n")]
    moe_every_n: usize,
}

fn default_prosody_dim() -> usize { 3 }
fn default_prosody_embedding_dim() -> usize { 256 }
fn default_top_k() -> usize { 2 }
fn default_moe_every_n() -> usize { 2 }

fn default_sigma_min() -> f64 { 1e-4 }
fn default_speaker_dim() -> usize { 256 }
fn default_emotion_dim() -> usize { 64 }

#[derive(Debug, Deserialize)]
struct DecoderConfig {
    n_fft: usize,
    hop_length: usize,
    dec_dim: usize,
    dec_n_layers: usize,
    dec_conv_kernel: usize,
    dec_ff_mult: f64,
    fsq_levels: Vec<usize>,
    acoustic_dim: usize,
}

// ─── Timestep Embedding ──────────────────────────────────────────────────────

struct TimestepEmbedding {
    mlp_0: Linear,
    mlp_1: Linear,
    cached_freqs: Tensor,
}

impl TimestepEmbedding {
    fn load(dim: usize, device: &Device, vb: VarBuilder) -> Result<Self> {
        let mlp_0 = linear(dim, dim * 4, vb.pp("mlp.0"))?;
        let mlp_1 = linear(dim * 4, dim, vb.pp("mlp.2"))?;
        let half = dim / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(10000.0_f64.ln()) * i as f64 / half as f64).exp() as f32)
            .collect();
        let cached_freqs = Tensor::from_vec(freqs, half, device)?;
        Ok(Self { mlp_0, mlp_1, cached_freqs })
    }

    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let t_f32 = t.to_dtype(DType::F32)?;
        let args = t_f32.unsqueeze(1)?.broadcast_mul(&self.cached_freqs.unsqueeze(0)?)?;
        let cos_part = args.cos()?;
        let sin_part = args.sin()?;
        let emb = Tensor::cat(&[&cos_part, &sin_part], D::Minus1)?;
        let emb = emb.to_dtype(self.mlp_0.weight().dtype())?;

        let x = self.mlp_0.forward(&emb)?;
        let x = x.silu()?;
        self.mlp_1.forward(&x)
    }
}

// ─── Adaptive Layer Norm ─────────────────────────────────────────────────────

struct AdaLayerNorm {
    proj: Linear,
    eps: f64,
}

impl AdaLayerNorm {
    fn load(dim: usize, cond_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let proj = linear(cond_dim, 2 * dim, vb.pp("proj"))?;
        Ok(Self { proj, eps })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_sub(&mean)?.broadcast_div(
            &(var + self.eps)?.sqrt()?
        )?;

        let shift_scale = self.proj.forward(cond)?;
        let chunks = shift_scale.chunk(2, D::Minus1)?;
        let shift = &chunks[0];
        let scale = &chunks[1];

        let ones = Tensor::ones_like(scale)?;
        normed.broadcast_mul(&(ones + scale)?)?.broadcast_add(shift)
    }
}

// ─── MoE Layer ───────────────────────────────────────────────────────────────

struct MoELayer {
    router: Linear,
    experts: Vec<(Linear, Linear)>,
    top_k: usize,
}

impl MoELayer {
    fn load(dim: usize, n_experts: usize, top_k: usize, ff_mult: f64, vb: VarBuilder) -> Result<Self> {
        let ff_dim = (dim as f64 * ff_mult) as usize;
        let router = linear(dim, n_experts, vb.pp("moe_router"))?;
        let mut experts = Vec::new();
        for j in 0..n_experts {
            let p = vb.pp(format!("experts.{}", j));
            let e0 = linear(dim, ff_dim, p.pp("0"))?;
            let e1 = linear(ff_dim, dim, p.pp("2"))?;
            experts.push((e0, e1));
        }
        Ok(Self { router, experts, top_k })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();
        let n_experts = self.experts.len();
        let k = self.top_k.min(n_experts);

        let router_logits = self.router.forward(x)?;
        let gate = candle_nn::ops::softmax(&router_logits, D::Minus1)?;

        // Sparse top-k gating: keep only top-k experts per token, renormalize
        let gate_f32 = gate.to_dtype(DType::F32)?;
        let gate_cpu = gate_f32.to_vec3::<f32>()?;
        let mut masked = vec![0.0f32; b * t * n_experts];
        for bi in 0..b {
            for ti in 0..t {
                let base = (bi * t + ti) * n_experts;
                let mut indices: Vec<usize> = (0..n_experts).collect();
                indices.sort_by(|&a, &b_idx| gate_cpu[bi][ti][b_idx]
                    .partial_cmp(&gate_cpu[bi][ti][a])
                    .unwrap_or(std::cmp::Ordering::Equal));
                let mut sum = 0.0f32;
                for &idx in indices.iter().take(k) {
                    sum += gate_cpu[bi][ti][idx];
                }
                let renorm = if sum > 1e-8 { sum } else { 1.0 };
                for &idx in indices.iter().take(k) {
                    masked[base + idx] = gate_cpu[bi][ti][idx] / renorm;
                }
            }
        }
        let gate_sparse = Tensor::from_vec(masked, (b, t, n_experts), device)?.to_dtype(dtype)?;

        // Only run experts that have non-zero weight (sparse skip)
        let mut out = Tensor::zeros((b, t, d), dtype, device)?;
        for (j, (e0, e1)) in self.experts.iter().enumerate() {
            let w_col = gate_sparse.narrow(2, j, 1)?;
            let w_sum = w_col.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
            if w_sum.abs() < 1e-8 { continue; }
            let h = e0.forward(x)?;
            let h = h.gelu_erf()?;
            let h = e1.forward(&h)?;
            let w = w_col.expand((b, t, d))?;
            out = (out + (&h * &w))?;
        }
        Ok(out)
    }
}

// ─── Causal mask helper ──────────────────────────────────────────────────────

fn build_causal_mask(t: usize, total_len: usize, prefix_len: usize,
                     device: &Device, dtype: DType) -> Result<Tensor> {
    let mut data = vec![0.0f32; t * total_len];
    for i in 0..t {
        for j in (i + prefix_len + 1)..total_len {
            data[i * total_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (1, 1, t, total_len), device)?.to_dtype(dtype)
}

// ─── Flow Attention ──────────────────────────────────────────────────────────

struct FlowAttention {
    qkv: Linear,
    out_proj: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl FlowAttention {
    fn load(dim: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let qkv = linear(dim, 3 * dim, vb.pp("qkv"))?;
        let out_proj = linear(dim, dim, vb.pp("out"))?;
        let head_dim = dim / n_heads;
        Ok(Self { qkv, out_proj, n_heads, head_dim })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_impl(x, false, None)
    }

    /// Forward with optional causal mask and KV cache for streaming.
    /// When causal=true, applies lower-triangular mask. When cache is Some, concatenates
    /// cached K,V with new and updates cache in place.
    fn forward_impl(
        &self,
        x: &Tensor,
        causal: bool,
        mut kv_cache: Option<&mut Option<(Tensor, Tensor)>>,
    ) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();

        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape((b, t, 3, self.n_heads, self.head_dim))?;

        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
        let mut k = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
        let mut v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;

        let prefix_len = kv_cache.as_ref().and_then(|c| c.as_ref())
            .map(|(ck, _)| ck.dim(2).unwrap_or(0))
            .unwrap_or(0);

        if prefix_len > 0 {
            if let Some(ref mut cache) = kv_cache {
                if let Some((ck, cv)) = cache.take() {
                    k = Tensor::cat(&[&ck, &k], 2)?;
                    v = Tensor::cat(&[&cv, &v], 2)?;
                }
            }
        }

        let scale = (self.head_dim as f64).sqrt();
        let mut scores = q.matmul(&k.t()?)?.affine(1.0 / scale, 0.0)?;

        if causal {
            let total_len = k.dim(2)?;
            scores = scores.broadcast_add(
                &build_causal_mask(t, total_len, prefix_len, device, dtype)?
            )?;
        }

        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, t, self.n_heads * self.head_dim))?;
        let out = self.out_proj.forward(&out)?;

        if causal {
            if let Some(cache) = kv_cache {
                *cache = Some((k, v));
            }
        }

        Ok(out)
    }
}

// ─── Flow Transformer Block ─────────────────────────────────────────────────

enum FlowBlockFfn {
    Dense { mlp_0: Linear, mlp_1: Linear },
    MoE(MoELayer),
}

struct FlowBlock {
    norm1: AdaLayerNorm,
    attn: FlowAttention,
    norm2: AdaLayerNorm,
    ffn: FlowBlockFfn,
}

impl FlowBlock {
    fn load(dim: usize, n_heads: usize, cond_dim: usize,
            ff_mult: f64, eps: f64, moe_config: Option<(usize, usize)>, vb: VarBuilder) -> Result<Self> {
        let norm1 = AdaLayerNorm::load(dim, cond_dim, eps, vb.pp("norm1"))?;
        let attn = FlowAttention::load(dim, n_heads, vb.pp("attn"))?;
        let norm2 = AdaLayerNorm::load(dim, cond_dim, eps, vb.pp("norm2"))?;
        let ffn = if let Some((n_experts, top_k)) = moe_config {
            if vb.pp("moe_router").get((n_experts, dim), "weight").is_ok() {
                FlowBlockFfn::MoE(MoELayer::load(dim, n_experts, top_k.min(n_experts), ff_mult, vb)?)
            } else {
                let ff_dim = (dim as f64 * ff_mult) as usize;
                let mlp_0 = linear(dim, ff_dim, vb.pp("mlp.0"))?;
                let mlp_1 = linear(ff_dim, dim, vb.pp("mlp.2"))?;
                FlowBlockFfn::Dense { mlp_0, mlp_1 }
            }
        } else {
            let ff_dim = (dim as f64 * ff_mult) as usize;
            let mlp_0 = linear(dim, ff_dim, vb.pp("mlp.0"))?;
            let mlp_1 = linear(ff_dim, dim, vb.pp("mlp.2"))?;
            FlowBlockFfn::Dense { mlp_0, mlp_1 }
        };
        Ok(Self { norm1, attn, norm2, ffn })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        self.forward_impl(x, cond, false, None)
    }

    fn forward_impl(
        &self,
        x: &Tensor,
        cond: &Tensor,
        causal: bool,
        kv_cache: Option<&mut Option<(Tensor, Tensor)>>,
    ) -> Result<Tensor> {
        let h = self.norm1.forward(x, cond)?;
        let h = self.attn.forward_impl(&h, causal, kv_cache)?;
        let x = (x + h)?;

        let h = self.norm2.forward(&x, cond)?;
        let h = match &self.ffn {
            FlowBlockFfn::Dense { mlp_0, mlp_1 } => {
                let h = mlp_0.forward(&h)?;
                let h = h.gelu_erf()?;
                mlp_1.forward(&h)?
            }
            FlowBlockFfn::MoE(moe) => moe.forward(&h)?,
        };
        (x + h)
    }
}

// ─── Sonata Flow ─────────────────────────────────────────────────────────────

// ─── Emotion Steering (EmoSteer-TTS) ─────────────────────────────────────────
// Training-free emotion control via activation steering.
// A direction vector is added to transformer activations at inference time.

struct EmotionSteering {
    direction: Tensor,   // (d_model,) direction vector
    layer_mask: Vec<bool>, // which layers to steer
    scale: f32,
}

struct SonataFlow {
    semantic_emb: Embedding,
    time_emb: TimestepEmbedding,
    cond_proj: Linear,
    input_proj: Linear,
    blocks: Vec<FlowBlock>,
    output_norm_weight: Option<Tensor>,
    output_norm_bias: Option<Tensor>,
    output_norm_eps: f64,
    output_proj: Linear,
    speaker_emb: Option<Embedding>,
    speaker_proj: Option<Linear>,
    emotion_emb: Option<Embedding>,
    emotion_proj: Option<Linear>,
    prosody_proj: Option<Linear>,
    prosody_embedding_proj: Option<Linear>,
    duration_proj: Option<Linear>,
    config: FlowConfig,
}

impl SonataFlow {
    fn load(config: FlowConfig, vb: VarBuilder) -> Result<Self> {
        let sem_size = config.semantic_vocab_size + 4;
        let semantic_emb = embedding(sem_size, config.cond_dim, vb.pp("semantic_emb"))?;
        let time_emb = TimestepEmbedding::load(config.cond_dim, vb.device(), vb.pp("time_emb"))?;
        let cond_proj = linear(config.cond_dim * 2, config.d_model, vb.pp("cond_proj"))?;
        let input_proj = linear(config.acoustic_dim, config.d_model, vb.pp("input_proj"))?;

        let mut blocks = Vec::new();
        for i in 0..config.n_layers {
            let moe_config = if config.n_experts > 0 && (i % config.moe_every_n) == 0 {
                Some((config.n_experts, config.top_k_experts))
            } else {
                None
            };
            blocks.push(FlowBlock::load(
                config.d_model, config.n_heads, config.d_model,
                4.0, 1e-5, moe_config, vb.pp(format!("blocks.{}", i)),
            )?);
        }

        // Output LayerNorm (optional — present in trained v3 models)
        let (output_norm_weight, output_norm_bias) = match (
            vb.get(config.d_model, "output_norm.weight"),
            vb.get(config.d_model, "output_norm.bias"),
        ) {
            (Ok(w), Ok(b)) => {
                eprintln!("[sonata_flow] output_norm loaded");
                (Some(w), Some(b))
            }
            _ => (None, None),
        };
        let output_norm_eps = 1e-5;

        let output_proj = linear(config.d_model, config.acoustic_dim, vb.pp("output_proj"))?;

        let (speaker_emb, speaker_proj) = if config.n_speakers > 0 {
            let se = embedding(config.n_speakers, config.speaker_dim, vb.pp("speaker_emb"))?;
            let sp = linear(config.speaker_dim, config.d_model, vb.pp("speaker_proj"))?;
            (Some(se), Some(sp))
        } else {
            (None, None)
        };

        let (emotion_emb, emotion_proj) = if config.n_emotions > 0 {
            let ee = embedding(config.n_emotions, config.emotion_dim, vb.pp("emotion_emb"))?;
            let ep = linear(config.emotion_dim, config.d_model, vb.pp("emotion_proj"))?;
            eprintln!("[sonata_flow] Emotion conditioning loaded ({} emotions)", config.n_emotions);
            (Some(ee), Some(ep))
        } else {
            match (
                linear(config.emotion_dim, config.d_model, vb.pp("emotion_proj")),
                embedding(16, config.emotion_dim, vb.pp("emotion_emb")),
            ) {
                (Ok(ep), Ok(ee)) => {
                    eprintln!("[sonata_flow] Emotion conditioning found in weights");
                    (Some(ee), Some(ep))
                }
                _ => (None, None),
            }
        };

        let prosody_proj = match linear(config.prosody_dim, config.d_model, vb.pp("prosody_proj")) {
            Ok(p) => {
                eprintln!("[sonata_flow] Prosody projection loaded ({}→{})", config.prosody_dim, config.d_model);
                Some(p)
            }
            Err(_) => None,
        };

        let duration_proj = match linear(1, config.d_model, vb.pp("duration_proj")) {
            Ok(p) => {
                eprintln!("[sonata_flow] Duration projection loaded");
                Some(p)
            }
            Err(_) => None,
        };

        let prosody_embedding_proj = match linear(
            config.prosody_embedding_dim,
            config.d_model,
            vb.pp("prosody_embedding_proj"),
        ) {
            Ok(p) => {
                eprintln!("[sonata_flow] Prosody embedding projection loaded ({}→{})",
                    config.prosody_embedding_dim, config.d_model);
                Some(p)
            }
            Err(_) => None,
        };

        Ok(Self {
            semantic_emb, time_emb, cond_proj, input_proj,
            blocks, output_norm_weight, output_norm_bias, output_norm_eps,
            output_proj, speaker_emb, speaker_proj,
            emotion_emb, emotion_proj, prosody_proj, prosody_embedding_proj,
            duration_proj, config,
        })
    }

    fn predict_velocity(&self, x_t: &Tensor, t: &Tensor,
                        semantic_tokens: &Tensor,
                        speaker_id: Option<u32>,
                        speaker_emb_override: Option<&Tensor>,
                        prosody_embedding_override: Option<&Tensor>,
                        emotion_id: Option<u32>,
                        prosody_features: Option<&Tensor>,
                        duration_features: Option<&Tensor>,
                        emotion_steering: Option<&EmotionSteering>,
                        causal: bool,
                        mut kv_caches: Option<&mut [Option<(Tensor, Tensor)>]>,
                        ) -> Result<Tensor> {
        let sem_cond = self.semantic_emb.forward(semantic_tokens)?;
        let time_cond = self.time_emb.forward(t)?;
        let time_cond = time_cond.unsqueeze(1)?.broadcast_as(sem_cond.shape())?;
        let cond_input = Tensor::cat(&[&sem_cond, &time_cond], D::Minus1)?;
        let mut cond = self.cond_proj.forward(&cond_input)?;

        // Speaker conditioning (additive injection)
        if let Some(ref proj) = self.speaker_proj {
            let spk_emb = if let Some(override_emb) = speaker_emb_override {
                Some(proj.forward(override_emb)?)
            } else if let (Some(ref emb), Some(spk_id)) = (&self.speaker_emb, speaker_id) {
                let spk_t = Tensor::from_vec(vec![spk_id], 1, x_t.device())?;
                Some(proj.forward(&emb.forward(&spk_t)?)?)
            } else {
                None
            };
            if let Some(spk_emb) = spk_emb {
                let cond_shape = cond.shape().clone();
                cond = (cond + spk_emb.unsqueeze(1)?.broadcast_as(&cond_shape)?)?;
            }
        }

        // Emotion conditioning (additive injection, parallel to speaker)
        if let (Some(ref proj), Some(ref emb)) = (&self.emotion_proj, &self.emotion_emb) {
            if let Some(emo_id) = emotion_id {
                let emo_t = Tensor::from_vec(vec![emo_id], 1, x_t.device())?;
                let emo_emb = proj.forward(&emb.forward(&emo_t)?)?;
                let cond_shape = cond.shape().clone();
                cond = (cond + emo_emb.unsqueeze(1)?.broadcast_as(&cond_shape)?)?;
            }
        }

        // Prosody conditioning: (log_pitch, energy, rate) → d_model, broadcast across frames
        if let (Some(ref proj), Some(pf)) = (&self.prosody_proj, prosody_features) {
            let p_emb = proj.forward(pf)?;
            let cond_shape = cond.shape().clone();
            cond = (cond + p_emb.broadcast_as(&cond_shape)?)?;
        }

        // Prosody embedding: reference audio encoder output → d_model (prosody transfer)
        if let (Some(ref proj), Some(emb)) = (&self.prosody_embedding_proj, prosody_embedding_override) {
            let emb_dim = emb.dim(1)?;
            if emb_dim == self.config.prosody_embedding_dim {
                let p_emb = proj.forward(emb)?;
                let cond_shape = cond.shape().clone();
                cond = (cond + p_emb.unsqueeze(1)?.broadcast_as(&cond_shape)?)?;
            }
        }

        // Duration conditioning: per-frame duration → d_model
        if let (Some(ref proj), Some(df)) = (&self.duration_proj, duration_features) {
            let d_emb = proj.forward(df)?;
            cond = (cond + d_emb)?;
        }

        let mut x = self.input_proj.forward(x_t)?;
        for (i, block) in self.blocks.iter().enumerate() {
            let cache = kv_caches.as_mut().and_then(|c| c.get_mut(i));
            x = block.forward_impl(&x, &cond, causal, cache)?;

            // EmoSteer: add scaled direction vector after specified layers
            if let Some(ref steer) = emotion_steering {
                if i < steer.layer_mask.len() && steer.layer_mask[i] {
                    let dir = steer.direction.unsqueeze(0)?.unsqueeze(0)?
                        .broadcast_as(x.shape())?
                        .to_dtype(x.dtype())?;
                    x = (x + dir.affine(steer.scale as f64, 0.0)?)?;
                }
            }
        }

        // Apply output LayerNorm if present (trained v3+ models)
        if let (Some(ref w), Some(ref b)) = (&self.output_norm_weight, &self.output_norm_bias) {
            let mean = x.mean_keepdim(D::Minus1)?;
            let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
            let x_norm = x.broadcast_sub(&mean)?.broadcast_div(
                &(var + self.output_norm_eps)?.sqrt()?
            )?;
            let w_b = w.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(x_norm.shape())?;
            let b_b = b.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(x_norm.shape())?;
            x = (x_norm.broadcast_mul(&w_b)? + b_b)?;
        }

        self.output_proj.forward(&x)
    }

    fn compute_velocity_cfg(
        &self, x: &Tensor, t_val: f32, b: usize,
        semantic_tokens: &Tensor, null_tokens: Option<&Tensor>,
        speaker_id: Option<u32>, speaker_emb_override: Option<&Tensor>,
        prosody_embedding_override: Option<&Tensor>,
        emotion_id: Option<u32>, prosody_features: Option<&Tensor>,
        duration_features: Option<&Tensor>, emotion_steering: Option<&EmotionSteering>,
        cfg_scale: f32, dtype: DType,
        causal: bool,
        kv_caches: Option<&mut [Option<(Tensor, Tensor)>]>,
    ) -> Result<Tensor> {
        let device = x.device();
        let t_tensor = Tensor::from_vec(vec![t_val; b], b, device)?.to_dtype(dtype)?;
        if let Some(null_toks) = null_tokens {
            // Batched CFG: stack cond+uncond along batch dim for single forward pass
            let x_batch = Tensor::cat(&[x, x], 0)?;
            let t_batch = Tensor::cat(&[&t_tensor, &t_tensor], 0)?;
            let tok_batch = Tensor::cat(&[semantic_tokens, null_toks], 0)?;

            let v_batch = self.predict_velocity(
                &x_batch, &t_batch, &tok_batch, speaker_id, speaker_emb_override,
                prosody_embedding_override, emotion_id, prosody_features,
                duration_features, emotion_steering, causal, kv_caches,
            )?;

            let v_cond = v_batch.narrow(0, 0, b)?;
            let v_uncond = v_batch.narrow(0, b, b)?;
            let v_diff = (&v_cond - &v_uncond)?;
            (&v_uncond + v_diff.affine(cfg_scale as f64, 0.0)?)
        } else {
            self.predict_velocity(
                x, &t_tensor, semantic_tokens, speaker_id, speaker_emb_override,
                prosody_embedding_override, emotion_id, prosody_features,
                duration_features, emotion_steering, causal, kv_caches,
            )
        }
    }

    fn sample(&self, semantic_tokens: &Tensor, speaker_id: Option<u32>,
              cfg_scale: f32, n_steps: usize, use_heun: bool, dtype: DType,
              speaker_emb_override: Option<&Tensor>,
              prosody_embedding_override: Option<&Tensor>,
              emotion_id: Option<u32>, prosody_features: Option<&Tensor>,
              duration_features: Option<&Tensor>,
              emotion_steering: Option<&EmotionSteering>,
              ) -> Result<Tensor> {
        let (b, t) = semantic_tokens.dims2()?;
        let device = semantic_tokens.device();
        let steps = if n_steps > 0 { n_steps } else { self.config.n_steps_inference };

        let sigma_min = self.config.sigma_min as f32;
        let mut x = Tensor::randn(0.0_f32, 1.0, (b, t, self.config.acoustic_dim), device)?
            .to_dtype(dtype)?;

        let use_cfg = cfg_scale > 1.0;
        let null_tokens = if use_cfg {
            Some(Tensor::zeros_like(semantic_tokens)?)
        } else {
            None
        };

        for i in 0..steps {
            let t0 = sigma_min + i as f32 * (1.0 - sigma_min) / steps as f32;
            let t1 = sigma_min + (i + 1) as f32 * (1.0 - sigma_min) / steps as f32;
            let dt = (t1 - t0) as f64;

            let v1 = self.compute_velocity_cfg(
                &x, t0, b, semantic_tokens, null_tokens.as_ref(),
                speaker_id, speaker_emb_override, prosody_embedding_override,
                emotion_id, prosody_features, duration_features, emotion_steering,
                cfg_scale, dtype, false, None,
            )?;

            if use_heun && i + 1 < steps {
                let x_euler = (&x + v1.affine(dt, 0.0)?)?;
                let v2 = self.compute_velocity_cfg(
                    &x_euler, t1, b, semantic_tokens, null_tokens.as_ref(),
                    speaker_id, speaker_emb_override, prosody_embedding_override,
                    emotion_id, prosody_features, duration_features, emotion_steering,
                    cfg_scale, dtype, false, None,
                )?;
                x = (&x + (&v1 + &v2)?.affine(dt / 2.0, 0.0)?)?;
            } else {
                x = (x + v1.affine(dt, 0.0)?)?;
            }
        }

        Ok(x)
    }

    /// Streaming chunk generation with causal attention. semantic_tokens should have
    /// length chunk_offset + n_frames (full sequence so far). Processes one chunk,
    /// using prefix_x_at_step for previous chunks' x at each ODE step.
    /// Fills chunk_x_at_step with this chunk's x at each step (for next chunk's prefix).
    fn sample_streaming_chunk(
        &self,
        semantic_tokens: &Tensor,
        chunk_offset: usize,
        n_frames: usize,
        n_steps: usize,
        use_heun: bool,
        cfg_scale: f32,
        dtype: DType,
        speaker_id: Option<u32>,
        speaker_emb_override: Option<&Tensor>,
        prosody_embedding_override: Option<&Tensor>,
        emotion_id: Option<u32>,
        prosody_features: Option<&Tensor>,
        duration_features: Option<&Tensor>,
        emotion_steering: Option<&EmotionSteering>,
        prefix_x_at_step: &mut [Tensor],
        chunk_x_at_step: &mut [Tensor],
        kv_caches: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        let (b, _total) = semantic_tokens.dims2()?;
        let device = semantic_tokens.device();
        let steps = if n_steps > 0 { n_steps } else { self.config.n_steps_inference };
        let sigma_min = self.config.sigma_min as f32;
        let use_cfg = cfg_scale > 1.0;

        let chunk_sem = semantic_tokens.narrow(1, chunk_offset, n_frames)?;
        let null_tokens = if use_cfg { Some(Tensor::zeros_like(&chunk_sem)?) } else { None };

        let mut x = Tensor::randn(0.0_f32, 1.0, (b, n_frames, self.config.acoustic_dim), device)?
            .to_dtype(dtype)?;

        for i in 0..steps {
            // Reset KV caches at the start of each ODE step to prevent stale state accumulation
            for c in kv_caches.iter_mut() { *c = None; }

            let t0 = sigma_min + i as f32 * (1.0 - sigma_min) / steps as f32;
            let t1 = sigma_min + (i + 1) as f32 * (1.0 - sigma_min) / steps as f32;
            let dt = (t1 - t0) as f64;

            if chunk_offset > 0 {
                let prefix_x = &prefix_x_at_step[i];
                let prefix_sem = semantic_tokens.narrow(1, 0, chunk_offset)?;
                self.compute_velocity_cfg(
                    prefix_x, t0, b, &prefix_sem, None,
                    speaker_id, speaker_emb_override, prosody_embedding_override,
                    emotion_id, prosody_features, duration_features, emotion_steering,
                    1.0, dtype, true, Some(kv_caches),
                )?;
            } else {
                for c in kv_caches.iter_mut() { *c = None; }
            }

            let v1 = self.compute_velocity_cfg(
                &x, t0, b, &chunk_sem, null_tokens.as_ref(),
                speaker_id, speaker_emb_override, prosody_embedding_override,
                emotion_id, prosody_features, duration_features, emotion_steering,
                cfg_scale, dtype, true, Some(kv_caches),
            )?;

            if use_heun && i + 1 < steps {
                let x_euler = (&x + v1.affine(dt, 0.0)?)?;
                if chunk_offset > 0 {
                    let prefix_x = &prefix_x_at_step[i + 1];
                    let prefix_sem = semantic_tokens.narrow(1, 0, chunk_offset)?;
                    self.compute_velocity_cfg(
                        prefix_x, t1, b, &prefix_sem, None,
                        speaker_id, speaker_emb_override, prosody_embedding_override,
                        emotion_id, prosody_features, duration_features, emotion_steering,
                        1.0, dtype, true, Some(kv_caches),
                    )?;
                } else {
                    for c in kv_caches.iter_mut() { *c = None; }
                }
                let v2 = self.compute_velocity_cfg(
                    &x_euler, t1, b, &chunk_sem, null_tokens.as_ref(),
                    speaker_id, speaker_emb_override, prosody_embedding_override,
                    emotion_id, prosody_features, duration_features, emotion_steering,
                    cfg_scale, dtype, true, Some(kv_caches),
                )?;
                x = (&x + (&v1 + &v2)?.affine(dt / 2.0, 0.0)?)?;
            } else {
                x = (x + v1.affine(dt, 0.0)?)?;
            }
            if i < chunk_x_at_step.len() {
                chunk_x_at_step[i] = x.clone();
            }
        }

        Ok(x)
    }
}

// ─── ConvNeXt Decoder ────────────────────────────────────────────────────────

struct ConvNeXtBlock {
    dwconv_weight: Tensor,
    dwconv_bias: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
    dim: usize,
    kernel_size: usize,
}

impl ConvNeXtBlock {
    fn load(dim: usize, kernel_size: usize, mult: f64, vb: VarBuilder) -> Result<Self> {
        let inner = (dim as f64 * mult) as usize;
        let dwconv_weight = vb.get((dim, 1, kernel_size), "dwconv.weight")?;
        let dwconv_bias = vb.get(dim, "dwconv.bias")?;
        let norm_weight = vb.get(dim, "norm.weight")?;
        let norm_bias = vb.get(dim, "norm.bias")?;
        let pwconv1 = linear(dim, inner, vb.pp("pwconv1"))?;
        let pwconv2 = linear(inner, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self { dwconv_weight, dwconv_bias, norm_weight, norm_bias,
                   pwconv1, pwconv2, gamma, dim, kernel_size })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let pad = self.kernel_size / 2;
        let h = x.pad_with_zeros(2, pad, pad)?;
        let h = h.conv1d(&self.dwconv_weight, 0, 1, 1, self.dim)?;
        let h = h.broadcast_add(&self.dwconv_bias.reshape((1, self.dim, 1))?)?;

        let h = h.transpose(1, 2)?;
        let mean = h.mean_keepdim(D::Minus1)?;
        let var = h.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
        let h = h.broadcast_sub(&mean)?.broadcast_div(&(var + 1e-5)?.sqrt()?)?;
        let h = h.broadcast_mul(&self.norm_weight)?.broadcast_add(&self.norm_bias)?;
        let h = self.pwconv1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.pwconv2.forward(&h)?;
        let h = h.broadcast_mul(&self.gamma)?;
        let h = h.transpose(1, 2)?;

        x + &h
    }

    fn forward_f32(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

// ─── ConvDecoder: HiFi-GAN style vocoder ─────────────────────────────────────

struct ResidualUnit {
    conv1_weight: Tensor,
    conv1_bias: Tensor,
    conv2_weight: Tensor,
    conv2_bias: Tensor,
    dilation: usize,
    dim: usize,
}

impl ResidualUnit {
    fn load(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let conv1_weight = vb.get((dim, dim, 7), "net.1.weight")?;
        let conv1_bias = vb.get(dim, "net.1.bias")?;
        let conv2_weight = vb.get((dim, dim, 1), "net.3.weight")?;
        let conv2_bias = vb.get(dim, "net.3.bias")?;
        Ok(Self { conv1_weight, conv1_bias, conv2_weight, conv2_bias, dilation, dim })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.maximum(&(x * 0.1)?)?;
        let pad = 3 * self.dilation;
        let h = h.pad_with_zeros(2, pad, pad)?;
        let h = h.conv1d(&self.conv1_weight, 0, 1, self.dilation, 1)?;
        let h = h.broadcast_add(&self.conv1_bias.reshape((1, self.dim, 1))?)?;
        let h = h.maximum(&(&h * 0.1)?)?;
        let h = h.conv1d(&self.conv2_weight, 0, 1, 1, 1)?;
        let h = h.broadcast_add(&self.conv2_bias.reshape((1, self.dim, 1))?)?;
        x + &h
    }

    fn forward_f32(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

struct UpsampleBlock {
    upsample_weight: Tensor,
    upsample_bias: Tensor,
    residuals: Vec<ResidualUnit>,
    stride: usize,
    out_ch: usize,
}

impl UpsampleBlock {
    fn load(in_ch: usize, out_ch: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let kernel = stride * 2;
        let upsample_weight = vb.get((in_ch, out_ch, kernel), "upsample.weight")?;
        let upsample_bias = vb.get(out_ch, "upsample.bias")?;
        let dilations = [1, 3, 9];
        let mut residuals = Vec::new();
        for (j, &d) in dilations.iter().enumerate() {
            residuals.push(ResidualUnit::load(
                out_ch, d, vb.pp(format!("residuals.{}", j)),
            )?);
        }
        Ok(Self { upsample_weight, upsample_bias, residuals, stride, out_ch })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let pad = self.stride / 2;
        let mut x = x.conv_transpose1d(&self.upsample_weight, pad, 0, self.stride, 1, 1)?;
        x = x.broadcast_add(&self.upsample_bias.reshape((1, self.out_ch, 1))?)?;
        for res in &self.residuals {
            x = res.forward(&x)?;
        }
        Ok(x)
    }

    fn forward_f32(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

struct ConvDecoder {
    input_proj_weight: Tensor,
    input_proj_bias: Tensor,
    blocks: Vec<ConvNeXtBlock>,
    upsample: Vec<UpsampleBlock>,
    output_weight: Tensor,
    output_bias: Tensor,
    config: DecoderConfig,
}

impl ConvDecoder {
    fn load(config: DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let fsq_dim = config.fsq_levels.len();
        let input_dim = fsq_dim + config.acoustic_dim;
        let d = config.dec_dim;

        let input_proj_weight = vb.get(
            (d, input_dim, 7), "decoder.input_proj.0.weight"
        )?;
        let input_proj_bias = vb.get(d, "decoder.input_proj.0.bias")?;

        let mut blocks = Vec::new();
        for i in 0..config.dec_n_layers {
            blocks.push(ConvNeXtBlock::load(
                d, config.dec_conv_kernel, config.dec_ff_mult,
                vb.pp(format!("decoder.backbone.{}", i)),
            )?);
        }

        let strides = [8, 5, 4, 3];
        let channels = [d, d / 2, d / 4, d / 8, d / 16];
        let mut upsample = Vec::new();
        for (i, &s) in strides.iter().enumerate() {
            upsample.push(UpsampleBlock::load(
                channels[i], channels[i + 1], s,
                vb.pp(format!("decoder.upsample.{}", i)),
            )?);
        }

        let out_ch = channels[4];
        let output_weight = vb.get((1, out_ch, 7), "decoder.output.1.weight")?;
        let output_bias = vb.get(1, "decoder.output.1.bias")?;

        Ok(Self {
            input_proj_weight, input_proj_bias,
            blocks, upsample, output_weight, output_bias, config,
        })
    }

    fn forward(&self, semantic_codes: &Tensor, acoustic_latent: &Tensor) -> Result<Tensor> {
        let x = Tensor::cat(&[semantic_codes, acoustic_latent], D::Minus1)?
            .to_dtype(DType::F32)?;
        let x = x.transpose(1, 2)?;

        let pad = 3;
        let x = x.pad_with_zeros(2, pad, pad)?;
        let x = x.conv1d(&self.input_proj_weight, 0, 1, 1, 1)?;
        let x = x.broadcast_add(
            &self.input_proj_bias.reshape((1, self.config.dec_dim, 1))?
        )?;
        let mut x = x.maximum(&(&x * 0.1)?)?;

        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        for up in &self.upsample {
            x = up.forward(&x)?;
        }

        let x = x.maximum(&(&x * 0.1)?)?;
        let x = x.pad_with_zeros(2, 3, 3)?;
        let x = x.conv1d(&self.output_weight, 0, 1, 1, 1)?;
        let x = x.broadcast_add(
            &self.output_bias.reshape((1, 1, 1))?
        )?;
        x.tanh()
    }

    fn samples_per_frame(&self) -> usize {
        self.config.hop_length
    }

    fn convert_to_f32(&mut self) -> Result<()> {
        self.input_proj_weight = self.input_proj_weight.to_dtype(DType::F32)?;
        self.input_proj_bias = self.input_proj_bias.to_dtype(DType::F32)?;

        for block in &mut self.blocks {
            block.dwconv_weight = block.dwconv_weight.to_dtype(DType::F32)?;
            block.dwconv_bias = block.dwconv_bias.to_dtype(DType::F32)?;
            block.norm_weight = block.norm_weight.to_dtype(DType::F32)?;
            block.norm_bias = block.norm_bias.to_dtype(DType::F32)?;
            block.pwconv1 = Linear::new(
                block.pwconv1.weight().to_dtype(DType::F32)?,
                block.pwconv1.bias().map(|b| b.to_dtype(DType::F32)).transpose()?,
            );
            block.pwconv2 = Linear::new(
                block.pwconv2.weight().to_dtype(DType::F32)?,
                block.pwconv2.bias().map(|b| b.to_dtype(DType::F32)).transpose()?,
            );
            block.gamma = block.gamma.to_dtype(DType::F32)?;
        }

        for up in &mut self.upsample {
            up.upsample_weight = up.upsample_weight.to_dtype(DType::F32)?;
            up.upsample_bias = up.upsample_bias.to_dtype(DType::F32)?;
            for res in &mut up.residuals {
                res.conv1_weight = res.conv1_weight.to_dtype(DType::F32)?;
                res.conv1_bias = res.conv1_bias.to_dtype(DType::F32)?;
                res.conv2_weight = res.conv2_weight.to_dtype(DType::F32)?;
                res.conv2_bias = res.conv2_bias.to_dtype(DType::F32)?;
            }
        }

        self.output_weight = self.output_weight.to_dtype(DType::F32)?;
        self.output_bias = self.output_bias.to_dtype(DType::F32)?;
        Ok(())
    }
}

// ─── Vocos Decoder (iSTFT) ───────────────────────────────────────────────────
// ConvNeXt backbone → mag_proj + phase_proj → magnitude and instantaneous phase.
// The actual iSTFT is done in C via sonata_istft.c for AMX acceleration.

struct VocosDecoder {
    input_proj_weight: Tensor,
    input_proj_bias: Tensor,
    blocks: Vec<ConvNeXtBlock>,
    mag_proj: Linear,
    phase_proj: Linear,
    config: DecoderConfig,
}

impl VocosDecoder {
    fn load(config: DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let fsq_dim = config.fsq_levels.len();
        let input_dim = fsq_dim + config.acoustic_dim;
        let d = config.dec_dim;
        let n_bins = config.n_fft / 2 + 1;

        let input_proj_weight = vb.get(
            (d, input_dim, 7), "decoder.input_proj.weight"
        )?;
        let input_proj_bias = vb.get(d, "decoder.input_proj.bias")?;

        let mut blocks = Vec::new();
        for i in 0..config.dec_n_layers {
            blocks.push(ConvNeXtBlock::load(
                d, config.dec_conv_kernel, config.dec_ff_mult,
                vb.pp(format!("decoder.backbone.{}", i)),
            )?);
        }

        let mag_proj = linear(d, n_bins, vb.pp("decoder.head.mag_proj"))?;
        let phase_proj = linear(d, n_bins, vb.pp("decoder.head.phase_proj"))?;

        Ok(Self {
            input_proj_weight, input_proj_bias,
            blocks, mag_proj, phase_proj, config,
        })
    }

    fn forward(&self, semantic_codes: &Tensor, acoustic_latent: &Tensor)
        -> Result<(Tensor, Tensor)>
    {
        let x = Tensor::cat(&[semantic_codes, acoustic_latent], D::Minus1)?
            .to_dtype(DType::F32)?;
        let x = x.transpose(1, 2)?; // (B, C_in, T)

        let pad = 3;
        let x = x.pad_with_zeros(2, pad, pad)?;
        let x = x.conv1d(&self.input_proj_weight, 0, 1, 1, 1)?;
        let x = x.broadcast_add(
            &self.input_proj_bias.reshape((1, self.config.dec_dim, 1))?
        )?;
        let mut x = x.maximum(&(&x * 0.1)?)?; // LeakyReLU

        for block in &self.blocks {
            x = block.forward_f32(&x)?;
        }

        // (B, D, T) → (B, T, D)
        let x = x.transpose(1, 2)?;

        let mag = self.mag_proj.forward(&x)?.exp()?; // log-magnitude → linear magnitude
        let phase = self.phase_proj.forward(&x)?;

        Ok((mag, phase))
    }

    fn n_bins(&self) -> usize {
        self.config.n_fft / 2 + 1
    }

    fn convert_to_f32(&mut self) -> Result<()> {
        self.input_proj_weight = self.input_proj_weight.to_dtype(DType::F32)?;
        self.input_proj_bias = self.input_proj_bias.to_dtype(DType::F32)?;

        for block in &mut self.blocks {
            block.dwconv_weight = block.dwconv_weight.to_dtype(DType::F32)?;
            block.dwconv_bias = block.dwconv_bias.to_dtype(DType::F32)?;
            block.norm_weight = block.norm_weight.to_dtype(DType::F32)?;
            block.norm_bias = block.norm_bias.to_dtype(DType::F32)?;
            block.pwconv1 = Linear::new(
                block.pwconv1.weight().to_dtype(DType::F32)?,
                block.pwconv1.bias().map(|b| b.to_dtype(DType::F32)).transpose()?,
            );
            block.pwconv2 = Linear::new(
                block.pwconv2.weight().to_dtype(DType::F32)?,
                block.pwconv2.bias().map(|b| b.to_dtype(DType::F32)).transpose()?,
            );
            block.gamma = block.gamma.to_dtype(DType::F32)?;
        }

        self.mag_proj = Linear::new(
            self.mag_proj.weight().to_dtype(DType::F32)?,
            self.mag_proj.bias().map(|b| b.to_dtype(DType::F32)).transpose()?,
        );
        self.phase_proj = Linear::new(
            self.phase_proj.weight().to_dtype(DType::F32)?,
            self.phase_proj.bias().map(|b| b.to_dtype(DType::F32)).transpose()?,
        );
        Ok(())
    }
}

// ─── Combined Engine ─────────────────────────────────────────────────────────

enum DecoderVariant {
    Conv(ConvDecoder),
    Vocos(VocosDecoder),
}

struct SonataFlowEngine {
    flow: SonataFlow,
    decoder: Option<DecoderVariant>,
    device: Device,
    dtype: DType,
    speaker_id: Option<u32>,
    cfg_scale: f32,
    n_steps: usize,
    use_heun: bool,
    quality_mode: i32,
    first_chunk_steps: Option<usize>,
    is_first_chunk: bool,
    last_phase: Vec<f32>,
    speaker_embedding_override: Option<Tensor>,
    emotion_id: Option<u32>,
    emotion_steering: Option<EmotionSteering>,
    prosody_features: Option<Tensor>,
    duration_features: Option<Tensor>,
    prosody_embedding_override: Option<Tensor>,
    causal: bool,
    streaming_kv_caches: Vec<Option<(Tensor, Tensor)>>,
    streaming_prefix_x_at_step: Vec<Tensor>,
}

fn resolve_hf_flow(path: &str, candidates: &[&str]) -> std::result::Result<String, Box<dyn std::error::Error>> {
    if Path::new(path).exists() {
        return Ok(path.to_string());
    }
    if path.contains('/') && !path.starts_with('.') && !path.starts_with('/') {
        eprintln!("[sonata_flow] '{}' not found locally, downloading from HuggingFace...", path);
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(path.to_string());
        for f in candidates {
            if let Ok(p) = repo.get(f) {
                let s = p.to_string_lossy().to_string();
                eprintln!("[sonata_flow] Downloaded: {}", s);
                return Ok(s);
            }
        }
        Err(format!("No matching files in HF repo '{}'", path).into())
    } else {
        Err(format!("File not found: {}", path).into())
    }
}

// ─── C FFI ───────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn sonata_flow_create(
    flow_weights: *const c_char,
    flow_config: *const c_char,
    decoder_weights: *const c_char,
    decoder_config: *const c_char,
) -> *mut c_void {
    if flow_weights.is_null() || flow_config.is_null() {
        eprintln!("[sonata_flow] Create error: NULL weights or config path");
        return std::ptr::null_mut();
    }
    let flow_weights = unsafe { CStr::from_ptr(flow_weights).to_str().unwrap_or("") };
    // Reject path traversal
    if flow_weights.contains("..") {
        eprintln!("[sonata_flow] Create error: path traversal in weights path");
        return std::ptr::null_mut();
    }
    let flow_config_path = unsafe { CStr::from_ptr(flow_config).to_str().unwrap_or("") };

    let has_decoder = !decoder_weights.is_null() && !decoder_config.is_null();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (|| -> Result<SonataFlowEngine> {
        #[cfg(feature = "metal")]
        let device = Device::new_metal(0)?;
        #[cfg(not(feature = "metal"))]
        let device = Device::Cpu;

        let dtype = DType::F16;

        let config_str = std::fs::read_to_string(flow_config_path)
            .map_err(|e| candle_core::Error::Msg(format!("Config read: {}", e)))?;
        let flow_cfg: FlowConfig = serde_json::from_str(&config_str)
            .map_err(|e| candle_core::Error::Msg(format!("Config parse: {}", e)))?;

        let n_steps = flow_cfg.n_steps_inference;
        let n_speakers = flow_cfg.n_speakers;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[flow_weights], dtype, &device,
            )?
        };
        let flow = SonataFlow::load(flow_cfg, vb)?;

        let decoder = if has_decoder {
            let dec_config_path = unsafe { CStr::from_ptr(decoder_config).to_str().unwrap_or("") };
            let dec_weights_path = unsafe { CStr::from_ptr(decoder_weights).to_str().unwrap_or("") };

            let dec_config_str = std::fs::read_to_string(dec_config_path)
                .map_err(|e| candle_core::Error::Msg(format!("Dec config: {}", e)))?;
            let dec_cfg: DecoderConfig = serde_json::from_str(&dec_config_str)
                .map_err(|e| candle_core::Error::Msg(format!("Dec parse: {}", e)))?;

            let dec_vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[dec_weights_path], dtype, &device,
                )?
            };

            // Detect decoder type: try VocosDecoder first (has mag_proj),
            // fall back to ConvDecoder (has upsample blocks)
            let n_bins = dec_cfg.n_fft / 2 + 1;
            let is_vocos = dec_vb.pp("decoder.head.mag_proj")
                .get((n_bins, dec_cfg.dec_dim), "weight").is_ok();

            if is_vocos {
                let mut dec = VocosDecoder::load(dec_cfg, dec_vb)?;
                dec.convert_to_f32()?;
                eprintln!("[sonata_flow] VocosDecoder loaded (iSTFT, {} bins)", dec.n_bins());
                Some(DecoderVariant::Vocos(dec))
            } else {
                let mut dec = ConvDecoder::load(dec_cfg, dec_vb)?;
                dec.convert_to_f32()?;
                eprintln!("[sonata_flow] ConvDecoder loaded (ConvTranspose)");
                Some(DecoderVariant::Conv(dec))
            }
        } else {
            None
        };

        {
            let t0 = std::time::Instant::now();
            let warmup_sem = Tensor::from_vec(vec![0u32; 2], (1, 2), &device)?;
            let acoustic = flow.sample(&warmup_sem, None, 1.0, 1, false, dtype, None, None, None, None, None, None)?;
            eprintln!("[sonata_flow] Flow warmup: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

            match decoder {
                Some(DecoderVariant::Conv(ref dec)) => {
                    let t1 = std::time::Instant::now();
                    let fsq_dim = dec.config.fsq_levels.len();
                    let dummy_fsq = Tensor::zeros((1, 2, fsq_dim), DType::F32, &device)?;
                    let acoustic_f32 = acoustic.to_dtype(DType::F32)?;
                    let _ = dec.forward(&dummy_fsq, &acoustic_f32);
                    eprintln!("[sonata_flow] ConvDecoder warmup: {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);
                }
                Some(DecoderVariant::Vocos(ref dec)) => {
                    let t1 = std::time::Instant::now();
                    let fsq_dim = dec.config.fsq_levels.len();
                    let dummy_fsq = Tensor::zeros((1, 2, fsq_dim), DType::F32, &device)?;
                    let acoustic_f32 = acoustic.to_dtype(DType::F32)?;
                    let _ = dec.forward(&dummy_fsq, &acoustic_f32);
                    eprintln!("[sonata_flow] VocosDecoder warmup: {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);
                }
                None => {}
            }
        }

        eprintln!("[sonata_flow] Loaded on {:?} (dtype={:?}, speakers={}, steps={})",
                  device, dtype, n_speakers, n_steps);
        let n_layers = flow.blocks.len();
        Ok(SonataFlowEngine {
            flow, decoder, device, dtype,
            speaker_id: None,
            cfg_scale: 1.0,
            n_steps,
            use_heun: false,
            quality_mode: FLOW_QUALITY_BALANCED,
            first_chunk_steps: None,
            is_first_chunk: true,
            last_phase: Vec::new(),
            speaker_embedding_override: None,
            emotion_id: None,
            emotion_steering: None,
            prosody_features: None,
            duration_features: None,
            prosody_embedding_override: None,
            causal: false,
            streaming_kv_caches: vec![None; n_layers],
            streaming_prefix_x_at_step: Vec::new(),
        })
        })()
    }));

    match result {
        Ok(Ok(engine)) => Box::into_raw(Box::new(engine)) as *mut c_void,
        Ok(Err(e)) => {
            eprintln!("[sonata_flow] Create error: {}", e);
            std::ptr::null_mut()
        }
        Err(e) => {
            eprintln!("[sonata_flow] panic in create: {}", panic_message(e));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_flow_destroy(engine: *mut c_void) {
    if !engine.is_null() {
        if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            unsafe { drop(Box::from_raw(engine as *mut SonataFlowEngine)) };
        })) {
            eprintln!("[sonata_flow] panic in destroy: {}", panic_message(e));
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_flow_set_speaker(engine: *mut c_void, speaker_id: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    if speaker_id < 0 {
        eng.speaker_id = None;
    } else {
        eng.speaker_id = Some(speaker_id as u32);
    }
    0
}

#[no_mangle]
pub extern "C" fn sonata_flow_set_cfg_scale(engine: *mut c_void, scale: c_float) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    eng.cfg_scale = scale.max(0.0);
    0
}

#[no_mangle]
pub extern "C" fn sonata_flow_set_n_steps(engine: *mut c_void, n_steps: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    if n_steps > 0 && n_steps <= 64 {
        eng.n_steps = n_steps as usize;
    }
    0
}

/// Set step count for first chunk only (TTFA optimization). Use 3 for ~25% faster first chunk.
/// Pass 0 to disable. Resets with sonata_flow_reset_phase (between utterances).
#[no_mangle]
pub extern "C" fn sonata_flow_set_first_chunk_steps(engine: *mut c_void, steps: c_int) {
    if engine.is_null() { return; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    eng.first_chunk_steps = if steps > 0 && steps <= 64 {
        Some(steps as usize)
    } else {
        None
    };
}

/// Reset first-chunk flag so next generate() uses first_chunk_steps.
/// Also called automatically by sonata_flow_reset_phase.
#[no_mangle]
pub extern "C" fn sonata_flow_reset_first_chunk(engine: *mut c_void) {
    if engine.is_null() { return; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    eng.is_first_chunk = true;
}

/// Set quality mode (FAST=0, BALANCED=1, HIGH=2).
/// Automatically configures n_steps and use_heun based on mode.
/// Manual set_n_steps() calls override the quality mode.
#[no_mangle]
pub extern "C" fn sonata_flow_set_quality_mode(engine: *mut c_void, mode: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    match mode {
        0 => { // FAST
            eng.quality_mode = FLOW_QUALITY_FAST;
            eng.n_steps = 4;
            eng.use_heun = false;
            0
        }
        1 => { // BALANCED
            eng.quality_mode = FLOW_QUALITY_BALANCED;
            eng.n_steps = 6;
            eng.use_heun = false;
            0
        }
        2 => { // HIGH
            eng.quality_mode = FLOW_QUALITY_HIGH;
            eng.n_steps = 8;
            eng.use_heun = true;
            0
        }
        _ => -1, // Invalid mode
    }
}

#[no_mangle]
pub extern "C" fn sonata_flow_reset_phase(engine: *mut c_void) {
    if engine.is_null() { return; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    eng.last_phase.clear();
    eng.is_first_chunk = true;
}

/// Enable or disable causal (left-to-right) attention for streaming TTS.
#[no_mangle]
pub extern "C" fn sonata_flow_set_causal(engine: *mut c_void, enable: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    eng.causal = enable != 0;
    0
}

/// Reset streaming state: clears KV caches and prefix x. Call before first chunk of a new utterance.
#[no_mangle]
pub extern "C" fn sonata_flow_reset_streaming(engine: *mut c_void) {
    if engine.is_null() { return; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    for c in eng.streaming_kv_caches.iter_mut() {
        *c = None;
    }
    eng.streaming_prefix_x_at_step.clear();
    eng.is_first_chunk = true;
}

/// Generate one streaming chunk with causal attention. semantic_tokens must have length
/// chunk_offset + n_frames (full sequence so far). Returns n_bins on success, 0 on error.
#[no_mangle]
pub extern "C" fn sonata_flow_generate_streaming_chunk(
    engine: *mut c_void,
    semantic_tokens: *const c_int,
    n_frames: c_int,
    chunk_offset: c_int,
    out_magnitude: *mut c_float,
    out_phase: *mut c_float,
) -> c_int {
    const MAX_FRAMES: c_int = 16384; // ~5 min at 50Hz
    if engine.is_null() || semantic_tokens.is_null() || n_frames <= 0
        || n_frames > MAX_FRAMES || chunk_offset < 0
        || out_magnitude.is_null() || out_phase.is_null() { return 0; }
    // Guard against integer overflow: offset + n_frames must fit in usize
    let total_len_check = (chunk_offset as i64) + (n_frames as i64);
    if total_len_check > MAX_FRAMES as i64 { return 0; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (|| -> Result<usize> {
            let n_f = n_frames as usize;
            let offset = chunk_offset as usize;
            let total_len = offset + n_f;

            let tokens: Vec<u32> = (0..total_len)
                .map(|i| unsafe { *semantic_tokens.add(i) as u32 })
                .collect();
            let sem = Tensor::from_vec(tokens.clone(), (1, total_len), &eng.device)?;

            let steps = if offset == 0 && eng.is_first_chunk {
                eng.is_first_chunk = false;
                eng.first_chunk_steps.unwrap_or(eng.n_steps)
            } else {
                eng.n_steps
            };
            let n_layers = eng.flow.blocks.len();

            // Extend prefix if first chunk used fewer steps (pad with last element)
            if offset > 0 && !eng.streaming_prefix_x_at_step.is_empty()
                && eng.streaming_prefix_x_at_step.len() < steps
            {
                let last = eng.streaming_prefix_x_at_step.last().cloned();
                if let Some(expand) = last {
                    while eng.streaming_prefix_x_at_step.len() < steps {
                        eng.streaming_prefix_x_at_step.push(expand.clone());
                    }
                }
            }

            let prefix = if offset > 0 && eng.streaming_prefix_x_at_step.len() >= steps {
                eng.streaming_prefix_x_at_step.as_mut_slice()
            } else {
                &mut []
            };
            let prefix = if prefix.len() > steps { &mut prefix[..steps] } else { prefix };

            let mut chunk_x: Vec<Tensor> = (0..steps)
                .map(|_| Tensor::zeros((1, n_f, eng.flow.config.acoustic_dim), DType::F32, &eng.device))
                .collect::<Result<_>>()?;

            let mut kv = std::mem::take(&mut eng.streaming_kv_caches);
            kv.resize(n_layers, None);

            let acoustic = eng.flow.sample_streaming_chunk(
                &sem, offset, n_f, steps, eng.use_heun, eng.cfg_scale, eng.dtype,
                eng.speaker_id, eng.speaker_embedding_override.as_ref(),
                eng.prosody_embedding_override.as_ref(),
                eng.emotion_id, eng.prosody_features.as_ref(),
                eng.duration_features.as_ref(), eng.emotion_steering.as_ref(),
                prefix, &mut chunk_x, &mut kv,
            )?;

            eng.streaming_kv_caches = kv;

            if eng.streaming_prefix_x_at_step.len() != steps {
                eng.streaming_prefix_x_at_step = chunk_x;
            } else {
                for (s, prev) in eng.streaming_prefix_x_at_step.iter_mut().enumerate() {
                    if s < chunk_x.len() {
                        *prev = Tensor::cat(&[&*prev, &chunk_x[s]], 1)?;
                    }
                }
            }

            match eng.decoder.as_ref() {
                Some(DecoderVariant::Vocos(dec)) => {
                    let fsq_dim = dec.config.fsq_levels.len();
                    let levels = &dec.config.fsq_levels;
                    let token_slice: Vec<u32> = (offset..total_len).map(|i| tokens[i]).collect();
                    let mut code_data = vec![0.0f32; n_f * fsq_dim];
                    for t in 0..n_f {
                        let mut idx = token_slice[t] as usize;
                        for d in (0..fsq_dim).rev() {
                            let level = levels[d];
                            let code_val = (idx % level) as f32;
                            let half = (level as f32 - 1.0) / 2.0;
                            code_data[t * fsq_dim + d] = code_val - half;
                            idx /= level;
                        }
                    }
                    let fsq_codes = Tensor::from_vec(code_data, (1, n_f, fsq_dim), &eng.device)?;
                    let acoustic_f32 = acoustic.to_dtype(DType::F32)?;
                    let (mag_t, phase_t) = dec.forward(&fsq_codes, &acoustic_f32)?;
                    let n_bins = dec.n_bins();
                    let mag_flat = mag_t.squeeze(0)?.contiguous()?.to_vec1::<f32>()?;
                    let phase_flat = phase_t.squeeze(0)?.contiguous()?.to_vec1::<f32>()?;
                    let n_copy = (n_f * n_bins).min(mag_flat.len()).min(phase_flat.len());
                    unsafe {
                        std::ptr::copy_nonoverlapping(mag_flat.as_ptr(), out_magnitude, n_copy);
                        std::ptr::copy_nonoverlapping(phase_flat.as_ptr(), out_phase, n_copy);
                    }
                    Ok(n_bins)
                }
                _ => {
                    let n_bins = 513;
                    let acoustic_data = acoustic.squeeze(0)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                    if eng.last_phase.len() != n_bins {
                        eng.last_phase = vec![0.0f32; n_bins];
                    }
                    for t in 0..n_f {
                        for b in 0..n_bins {
                            let idx = b % acoustic_data[t].len();
                            let val = acoustic_data[t][idx];
                            let mag = (val.abs() * 0.5).exp();
                            eng.last_phase[b] += val * 0.1;
                            unsafe {
                                *out_magnitude.add(t * n_bins + b) = mag;
                                *out_phase.add(t * n_bins + b) = eng.last_phase[b];
                            }
                        }
                    }
                    Ok(n_bins)
                }
            }
        })()
    }));

    match result {
        Ok(Ok(n_bins)) => n_bins as c_int,
        Ok(Err(e)) => {
            eprintln!("[sonata_flow] generate_streaming_chunk error: {}", e);
            0
        }
        Err(e) => {
            eprintln!("[sonata_flow] panic in generate_streaming_chunk: {}", panic_message(e));
            0
        }
    }
}

/// Generate magnitude + phase from semantic tokens.
/// Phase is accumulated for iSTFT continuity. Call reset_phase between utterances.
/// Returns n_bins on success, 0 on error.
#[no_mangle]
pub extern "C" fn sonata_flow_generate(
    engine: *mut c_void,
    semantic_tokens: *const c_int,
    n_frames: c_int,
    out_magnitude: *mut c_float,
    out_phase: *mut c_float,
) -> c_int {
    const MAX_FRAMES: c_int = 16384;
    if engine.is_null() || semantic_tokens.is_null() || n_frames <= 0
        || n_frames > MAX_FRAMES
        || out_magnitude.is_null() || out_phase.is_null() { return 0; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<usize> {
        let tokens: Vec<u32> = (0..n_frames as usize)
            .map(|i| unsafe { *semantic_tokens.add(i) as u32 })
            .collect();

        let sem = Tensor::from_vec(tokens.clone(), (1, n_frames as usize), &eng.device)?;
        let spk_override = eng.speaker_embedding_override.as_ref();
        let steps = if eng.is_first_chunk {
            eng.is_first_chunk = false;
            eng.first_chunk_steps.unwrap_or(eng.n_steps)
        } else {
            eng.n_steps
        };
        let acoustic = eng.flow.sample(
            &sem, eng.speaker_id, eng.cfg_scale, steps,
            eng.use_heun, eng.dtype, spk_override,
            eng.prosody_embedding_override.as_ref(),
            eng.emotion_id,
            eng.prosody_features.as_ref(),
            eng.duration_features.as_ref(),
            eng.emotion_steering.as_ref(),
        )?;

        match eng.decoder.as_ref() {
            Some(DecoderVariant::Vocos(dec)) => {
                let fsq_dim = dec.config.fsq_levels.len();
                let levels = &dec.config.fsq_levels;
                let mut code_data = vec![0.0f32; n_frames as usize * fsq_dim];
                for t in 0..n_frames as usize {
                    let mut idx = tokens[t] as usize;
                    for d in (0..fsq_dim).rev() {
                        let level = levels[d];
                        let code_val = (idx % level) as f32;
                        let half = (level as f32 - 1.0) / 2.0;
                        code_data[t * fsq_dim + d] = code_val - half;
                        idx /= level;
                    }
                }
                let fsq_codes = Tensor::from_vec(
                    code_data, (1, n_frames as usize, fsq_dim), &eng.device
                )?;
                let acoustic_f32 = acoustic.to_dtype(DType::F32)?;
                let (mag_t, phase_t) = dec.forward(&fsq_codes, &acoustic_f32)?;
                let n_bins = dec.n_bins();
                let mag_flat = mag_t.squeeze(0)?.contiguous()?.to_vec1::<f32>()?;
                let phase_flat = phase_t.squeeze(0)?.contiguous()?.to_vec1::<f32>()?;
                let n_copy = (n_frames as usize * n_bins).min(mag_flat.len()).min(phase_flat.len());
                unsafe {
                    std::ptr::copy_nonoverlapping(mag_flat.as_ptr(), out_magnitude, n_copy);
                    std::ptr::copy_nonoverlapping(phase_flat.as_ptr(), out_phase, n_copy);
                }
                Ok(n_bins)
            }
            Some(DecoderVariant::Conv(_)) => {
                let n_bins = 513; // ConvDecoder doesn't produce mag/phase
                Ok(n_bins)
            }
            None => {
                // Fallback: raw acoustic latents to heuristic mag/phase
                let n_bins = 513;
                let acoustic_data = acoustic.squeeze(0)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                if eng.last_phase.len() != n_bins {
                    eng.last_phase = vec![0.0f32; n_bins];
                }

                for t in 0..n_frames as usize {
                    for b in 0..n_bins {
                        let idx = b % acoustic_data[t].len();
                        let val = acoustic_data[t][idx];
                        let mag = (val.abs() * 0.5).exp();
                        eng.last_phase[b] += val * 0.1;
                        unsafe {
                            *out_magnitude.add(t * n_bins + b) = mag;
                            *out_phase.add(t * n_bins + b) = eng.last_phase[b];
                        }
                    }
                }
                Ok(n_bins)
            }
        }
    }));

    match result {
        Ok(Ok(n_bins)) => n_bins as c_int,
        Ok(Err(e)) => {
            eprintln!("[sonata_flow] Generate error: {}", e);
            0
        }
        Err(e) => {
            eprintln!("[sonata_flow] panic in generate: {}", panic_message(e));
            0
        }
    }
}

/// Returns decoder type: 0=none, 1=conv (direct audio), 2=vocos (mag+phase).
#[no_mangle]
pub extern "C" fn sonata_flow_decoder_type(engine: *mut c_void) -> c_int {
    if engine.is_null() { return 0; }
    let eng = unsafe { &*(engine as *const SonataFlowEngine) };
    match eng.decoder.as_ref() {
        Some(DecoderVariant::Conv(_)) => 1,
        Some(DecoderVariant::Vocos(_)) => 2,
        None => 0,
    }
}

/// In-place radix-2 Cooley-Tukey FFT. Length must be a power of 2.
/// Falls back to zero-padded power-of-2 if needed.
fn fft_radix2(re: &mut Vec<f32>, im: &mut Vec<f32>, n: usize) {
    // Pad to next power of 2 if needed
    let len = if n.is_power_of_two() { n } else { n.next_power_of_two() };
    re.resize(len, 0.0);
    im.resize(len, 0.0);

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 0..len {
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
        let mut m = len >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Butterfly stages
    let mut size = 2;
    while size <= len {
        let half = size / 2;
        let angle_step = 2.0 * std::f32::consts::PI / size as f32;
        for k in (0..len).step_by(size) {
            for j_idx in 0..half {
                let angle = angle_step * j_idx as f32;
                let wr = angle.cos();
                let wi = -angle.sin();
                let a = k + j_idx;
                let b = a + half;
                let tr = wr * re[b] - wi * im[b];
                let ti = wr * im[b] + wi * re[b];
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        size <<= 1;
    }

    // Truncate back to original length
    re.truncate(n);
    im.truncate(n);
}

/// Generate audio directly from semantic tokens via Flow + ConvDecoder.
/// Returns number of audio samples written, or 0 on error.
/// Requires a ConvDecoder (decoder_type == 1).
#[no_mangle]
pub extern "C" fn sonata_flow_generate_audio(
    engine: *mut c_void,
    semantic_tokens: *const c_int,
    n_frames: c_int,
    out_audio: *mut c_float,
    max_samples: c_int,
) -> c_int {
    const MAX_FRAMES: c_int = 16384;
    if engine.is_null() || semantic_tokens.is_null() || n_frames <= 0
        || n_frames > MAX_FRAMES
        || out_audio.is_null() || max_samples <= 0 { return 0; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };

    if eng.decoder.is_none() {
        eprintln!("[sonata_flow] generate_audio: no decoder loaded");
        return 0;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<usize> {
        let tokens: Vec<u32> = (0..n_frames as usize)
            .map(|i| unsafe { *semantic_tokens.add(i) as u32 })
            .collect();

        let sem = Tensor::from_vec(tokens.clone(), (1, n_frames as usize), &eng.device)?;
        let spk_override = eng.speaker_embedding_override.as_ref();
        let steps = if eng.is_first_chunk {
            eng.is_first_chunk = false;
            eng.first_chunk_steps.unwrap_or(eng.n_steps)
        } else {
            eng.n_steps
        };
        let acoustic = eng.flow.sample(
            &sem, eng.speaker_id, eng.cfg_scale, steps,
            eng.use_heun, eng.dtype, spk_override,
            eng.prosody_embedding_override.as_ref(),
            eng.emotion_id,
            eng.prosody_features.as_ref(),
            eng.duration_features.as_ref(),
            eng.emotion_steering.as_ref(),
        )?;

        let (fsq_dim, levels) = match eng.decoder.as_ref() {
            Some(DecoderVariant::Conv(d)) => (d.config.fsq_levels.len(), d.config.fsq_levels.clone()),
            Some(DecoderVariant::Vocos(d)) => (d.config.fsq_levels.len(), d.config.fsq_levels.clone()),
            None => return Err(candle_core::Error::Msg("no decoder".into())),
        };
        let mut code_data = vec![0.0f32; n_frames as usize * fsq_dim];
        for t in 0..n_frames as usize {
            let mut idx = tokens[t] as usize;
            for d in (0..fsq_dim).rev() {
                let level = levels[d];
                let code_val = (idx % level) as f32;
                let half = (level as f32 - 1.0) / 2.0;
                code_data[t * fsq_dim + d] = code_val - half;
                idx /= level;
            }
        }
        let fsq_codes = Tensor::from_vec(
            code_data, (1, n_frames as usize, fsq_dim), &eng.device
        )?;

        match eng.decoder.as_ref() {
            None => return Err(candle_core::Error::Msg("no decoder".into())),
            Some(DecoderVariant::Conv(decoder)) => {
                let fsq_f16 = fsq_codes.to_dtype(DType::F32)?;
                let acoustic_f32 = acoustic.to_dtype(DType::F32)?;
                let audio_tensor = decoder.forward(&fsq_f16, &acoustic_f32)?;
                let audio = audio_tensor.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                let n_copy = audio.len().min(max_samples as usize);
                unsafe {
                    std::ptr::copy_nonoverlapping(audio.as_ptr(), out_audio, n_copy);
                }
                Ok(n_copy)
            }
            Some(DecoderVariant::Vocos(decoder)) => {
                let acoustic_f32 = acoustic.to_dtype(DType::F32)?;
                let (mag_t, phase_t) = decoder.forward(&fsq_codes, &acoustic_f32)?;
                let n_fft = decoder.config.n_fft;
                let hop = decoder.config.hop_length;
                let n_bins = n_fft / 2 + 1;
                let mag_flat = mag_t.squeeze(0)?.contiguous()?.to_vec1::<f32>()?;
                let phase_flat = phase_t.squeeze(0)?.contiguous()?.to_vec1::<f32>()?;

                let total_samples = n_frames as usize * hop;
                let mut audio = vec![0.0f32; total_samples + n_fft];
                let mut window = vec![0.0f32; n_fft];
                for i in 0..n_fft {
                    window[i] = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos());
                }

                for t in 0..n_frames as usize {
                    let mut real = vec![0.0f32; n_fft];
                    let mut imag = vec![0.0f32; n_fft];
                    for b in 0..n_bins {
                        let m = mag_flat[t * n_bins + b];
                        let p = phase_flat[t * n_bins + b];
                        real[b] = m * p.cos();
                        imag[b] = m * p.sin();
                        if b > 0 && b < n_bins - 1 {
                            real[n_fft - b] = real[b];
                            imag[n_fft - b] = -imag[b];
                        }
                    }
                    // IDFT via radix-2 Cooley-Tukey (inverse FFT)
                    let frame = {
                        // For IDFT: swap real/imag, apply forward FFT, swap back, scale by 1/N
                        // This converts IDFT into a forward FFT problem
                        let mut re = imag.clone(); // swap: feed imag as real
                        let mut im = real.clone(); // swap: feed real as imag
                        fft_radix2(&mut re, &mut im, n_fft);
                        // Result = (im / N) for real part after swap-back
                        let inv = 1.0 / n_fft as f32;
                        let mut f = vec![0.0f32; n_fft];
                        for n in 0..n_fft {
                            f[n] = im[n] * inv * window[n];
                        }
                        f
                    };
                    let offset = t * hop;
                    for n in 0..n_fft {
                        if offset + n < audio.len() {
                            audio[offset + n] += frame[n];
                        }
                    }
                }

                let n_copy = total_samples.min(max_samples as usize);
                unsafe {
                    std::ptr::copy_nonoverlapping(audio.as_ptr(), out_audio, n_copy);
                }
                Ok(n_copy)
            }
        }
    }));

    match result {
        Ok(Ok(n)) => n as c_int,
        Ok(Err(e)) => {
            eprintln!("[sonata_flow] generate_audio error: {}", e);
            0
        }
        Err(e) => {
            eprintln!("[sonata_flow] panic in generate_audio: {}", panic_message(e));
            0
        }
    }
}

/// Returns samples per frame for the ConvDecoder (hop_length), or 0 if no decoder.
#[no_mangle]
pub extern "C" fn sonata_flow_samples_per_frame(engine: *mut c_void) -> c_int {
    if engine.is_null() { return 0; }
    let eng = unsafe { &*(engine as *const SonataFlowEngine) };
    match eng.decoder.as_ref() {
        Some(DecoderVariant::Conv(d)) => d.samples_per_frame() as c_int,
        Some(DecoderVariant::Vocos(d)) => d.config.hop_length as c_int,
        None => 0,
    }
}

#[no_mangle]
pub extern "C" fn sonata_flow_set_solver(engine: *mut c_void, use_heun: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    eng.use_heun = use_heun != 0;
    eprintln!("[sonata_flow] Solver: {}", if eng.use_heun { "Heun (2nd-order)" } else { "Euler (1st-order)" });
    0
}

#[no_mangle]
pub extern "C" fn sonata_flow_set_speaker_embedding(
    engine: *mut c_void, embedding: *const c_float, dim: c_int,
) -> c_int {
    const MAX_EMB_DIM: c_int = 4096;
    if engine.is_null() || embedding.is_null() || dim <= 0 || dim > MAX_EMB_DIM { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
        let data: Vec<f32> = (0..dim as usize)
            .map(|i| unsafe { *embedding.add(i) })
            .collect();
        match Tensor::from_vec(data, (1, dim as usize), &eng.device)
            .and_then(|t| t.to_dtype(eng.dtype))
        {
            Ok(t) => { eng.speaker_embedding_override = Some(t); 0 }
            Err(e) => { eprintln!("[sonata_flow] set_speaker_embedding: {}", e); -1 }
        }
    }));
    result.unwrap_or_else(|e| {
        eprintln!("[sonata_flow] panic in set_speaker_embedding: {}", panic_message(e));
        -1
    })
}

#[no_mangle]
pub extern "C" fn sonata_flow_clear_speaker_embedding(engine: *mut c_void) {
    if engine.is_null() { return; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    eng.speaker_embedding_override = None;
}

/// Interpolate two speaker embeddings: result = (1-alpha) * emb_a + alpha * emb_b.
/// Enables style transfer by blending voice characteristics.
/// alpha=0.0 → pure speaker A, alpha=1.0 → pure speaker B.
#[no_mangle]
pub extern "C" fn sonata_flow_interpolate_speakers(
    engine: *mut c_void,
    emb_a: *const c_float, emb_b: *const c_float,
    dim: c_int, alpha: c_float,
) -> c_int {
    const MAX_EMB_DIM: c_int = 4096;
    if engine.is_null() || emb_a.is_null() || emb_b.is_null() || dim <= 0 || dim > MAX_EMB_DIM { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
        let d = dim as usize;
        let a_clamped = alpha.clamp(0.0, 1.0);
        let data: Vec<f32> = (0..d).map(|i| {
            let a = unsafe { *emb_a.add(i) };
            let b = unsafe { *emb_b.add(i) };
            (1.0 - a_clamped) * a + a_clamped * b
        }).collect();
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = if norm > 1e-8 {
            data.iter().map(|x| x / norm).collect()
        } else {
            data
        };
        match Tensor::from_vec(normalized, (1, d), &eng.device)
            .and_then(|t| t.to_dtype(eng.dtype))
        {
            Ok(t) => {
                eng.speaker_embedding_override = Some(t);
                eprintln!("[sonata_flow] Speaker interpolation: alpha={:.2} ({}-dim)", a_clamped, d);
                0
            }
            Err(e) => { eprintln!("[sonata_flow] interpolate_speakers: {}", e); -1 }
        }
    }));
    result.unwrap_or_else(|e| {
        eprintln!("[sonata_flow] panic in interpolate_speakers: {}", panic_message(e));
        -1
    })
}

// ─── Emotion Conditioning FFI ────────────────────────────────────────────────

/// Set emotion ID for embedding-based conditioning (requires trained emotion embeddings).
/// emotion_id < 0 clears.
#[no_mangle]
pub extern "C" fn sonata_flow_set_emotion(engine: *mut c_void, emotion_id: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    eng.emotion_id = if emotion_id < 0 { None } else { Some(emotion_id as u32) };
    0
}

/// EmoSteer: Set an activation steering direction vector for training-free emotion control.
/// direction: float[dim] direction vector, dim: vector dimensionality (must match d_model),
/// layer_start/layer_end: which layers to steer (inclusive), scale: steering strength.
#[no_mangle]
pub extern "C" fn sonata_flow_set_emotion_steering(
    engine: *mut c_void,
    direction: *const c_float,
    dim: c_int,
    layer_start: c_int,
    layer_end: c_int,
    scale: c_float,
) -> c_int {
    const MAX_EMB_DIM: c_int = 4096;
    if engine.is_null() || direction.is_null() || dim <= 0 || dim > MAX_EMB_DIM { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
        let data: Vec<f32> = (0..dim as usize).map(|i| unsafe { *direction.add(i) }).collect();
        let n_layers = eng.flow.blocks.len();
        if n_layers == 0 { return 0; }
        let ls = (layer_start as usize).min(n_layers - 1);
        let le = (layer_end as usize).min(n_layers - 1);
        let mut layer_mask = vec![false; n_layers];
        for i in ls..=le {
            layer_mask[i] = true;
        }
        match Tensor::from_vec(data, dim as usize, &eng.device) {
            Ok(t) => {
                eng.emotion_steering = Some(EmotionSteering {
                    direction: t,
                    layer_mask,
                    scale,
                });
                eprintln!("[sonata_flow] Emotion steering set (layers {}-{}, scale={:.2})",
                          ls, le, scale);
                0
            }
            Err(e) => { eprintln!("[sonata_flow] set_emotion_steering: {}", e); -1 }
        }
    }));
    result.unwrap_or_else(|e| {
        eprintln!("[sonata_flow] panic in set_emotion_steering: {}", panic_message(e));
        -1
    })
}

#[no_mangle]
pub extern "C" fn sonata_flow_clear_emotion_steering(engine: *mut c_void) {
    if engine.is_null() { return; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    eng.emotion_steering = None;
}

// ─── Prosody Conditioning FFI ────────────────────────────────────────────────

/// Set per-generation prosody features: (log_pitch, energy, speaking_rate).
/// features: float[3], broadcast across all frames.  n=0 to clear.
#[no_mangle]
pub extern "C" fn sonata_flow_set_prosody(
    engine: *mut c_void, features: *const c_float, n: c_int,
) -> c_int {
    if engine.is_null() { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
        if n <= 0 || features.is_null() {
            eng.prosody_features = None;
            return 0;
        }
        let dim = n as usize;
        let data: Vec<f32> = (0..dim).map(|i| unsafe { *features.add(i) }).collect();
        match Tensor::from_vec(data, (1, 1, dim), &eng.device)
            .and_then(|t| t.to_dtype(eng.dtype))
        {
            Ok(t) => { eng.prosody_features = Some(t); 0 }
            Err(e) => { eprintln!("[sonata_flow] set_prosody: {}", e); -1 }
        }
    }));
    result.unwrap_or_else(|e| {
        eprintln!("[sonata_flow] panic in set_prosody: {}", panic_message(e));
        -1
    })
}

/// Set per-frame duration features for duration conditioning.
/// durations: float[n_frames] (log-duration per frame), n_frames: count.
#[no_mangle]
pub extern "C" fn sonata_flow_set_durations(
    engine: *mut c_void, durations: *const c_float, n_frames: c_int,
) -> c_int {
    if engine.is_null() || durations.is_null() || n_frames <= 0 { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
        let nf = n_frames as usize;
        let data: Vec<f32> = (0..nf).map(|i| unsafe { *durations.add(i) }).collect();
        match Tensor::from_vec(data, (1, nf, 1), &eng.device)
            .and_then(|t| t.to_dtype(eng.dtype))
        {
            Ok(t) => { eng.duration_features = Some(t); 0 }
            Err(e) => { eprintln!("[sonata_flow] set_durations: {}", e); -1 }
        }
    }));
    result.unwrap_or_else(|e| {
        eprintln!("[sonata_flow] panic in set_durations: {}", panic_message(e));
        -1
    })
}

/// Set a prosody embedding from a reference audio encoder for prosody transfer.
/// embedding: float[dim], dim: embedding dimensionality.
#[no_mangle]
pub extern "C" fn sonata_flow_set_prosody_embedding(
    engine: *mut c_void, embedding: *const c_float, dim: c_int,
) -> c_int {
    const MAX_EMB_DIM: c_int = 4096;
    if engine.is_null() || embedding.is_null() || dim <= 0 || dim > MAX_EMB_DIM { return -1; }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
        let data: Vec<f32> = (0..dim as usize).map(|i| unsafe { *embedding.add(i) }).collect();
        match Tensor::from_vec(data, (1, dim as usize), &eng.device)
            .and_then(|t| t.to_dtype(eng.dtype))
        {
            Ok(t) => {
                eng.prosody_embedding_override = Some(t);
                eprintln!("[sonata_flow] Prosody embedding set ({}-dim)", dim);
                0
            }
            Err(e) => { eprintln!("[sonata_flow] set_prosody_embedding: {}", e); -1 }
        }
    }));
    result.unwrap_or_else(|e| {
        eprintln!("[sonata_flow] panic in set_prosody_embedding: {}", panic_message(e));
        -1
    })
}

#[no_mangle]
pub extern "C" fn sonata_flow_clear_prosody_embedding(engine: *mut c_void) {
    if engine.is_null() { return; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    eng.prosody_embedding_override = None;
}

#[no_mangle]
pub extern "C" fn sonata_flow_n_steps() -> c_int { 8 }

#[no_mangle]
pub extern "C" fn sonata_flow_acoustic_dim() -> c_int { 256 }

// ═══════════════════════════════════════════════════════════════════════════
// Sonata Flow v2 — Single-stage text → mel
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
struct FlowV2Config {
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    mel_dim: usize,
    cond_dim: usize,
    #[serde(default = "default_ff_mult")]
    ff_mult: f64,
    #[serde(default = "default_norm_eps")]
    norm_eps: f64,
    text_encoder_layers: usize,
    text_encoder_dim: usize,
    text_encoder_kernel: usize,
    #[serde(default = "default_char_vocab")]
    char_vocab_size: usize,
    #[serde(default)]
    filler_token_id: u32,
    #[serde(default = "default_n_steps")]
    n_steps_inference: usize,
    #[serde(default)]
    n_speakers: usize,
    #[serde(default = "default_speaker_dim")]
    speaker_dim: usize,
    #[serde(default)]
    n_emotions: usize,
    #[serde(default = "default_emotion_dim")]
    emotion_dim: usize,
    #[serde(default = "default_prosody_dim_v2")]
    prosody_dim: usize,
    #[serde(default = "default_sway")]
    sway_coefficient: f64,
}

fn default_ff_mult() -> f64 { 4.0 }
fn default_norm_eps() -> f64 { 1e-5 }
fn default_char_vocab() -> usize { 256 }
fn default_n_steps() -> usize { 6 }
fn default_sway() -> f64 { -1.0 }
fn default_prosody_dim_v2() -> usize { 3 }

struct FlowV2ConvNeXtBlock {
    dwconv_weight: Tensor,
    dwconv_bias: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    pw1: Linear,
    pw2: Linear,
    dim: usize,
    kernel_size: usize,
}

impl FlowV2ConvNeXtBlock {
    fn load(dim: usize, kernel_size: usize, ff_mult: f64, vb: VarBuilder) -> Result<Self> {
        let ff_dim = (dim as f64 * ff_mult) as usize;
        // Conv1d groups=dim: weight (dim, 1, kernel)
        let dwconv_weight = vb.get((dim, 1, kernel_size), "dwconv.weight")?;
        let dwconv_bias = vb.get(dim, "dwconv.bias")?;
        let norm_weight = vb.get(dim, "norm.weight")?;
        let norm_bias = vb.get(dim, "norm.bias")?;
        let pw1 = linear(dim, ff_dim, vb.pp("pw1"))?;
        let pw2 = linear(ff_dim, dim, vb.pp("pw2"))?;
        Ok(Self { dwconv_weight, dwconv_bias, norm_weight, norm_bias, pw1, pw2, dim, kernel_size })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, T, C) -> (B, C, T)
        let x_ct = x.transpose(1, 2)?;
        let pad = self.kernel_size / 2;
        let h = x_ct.pad_with_zeros(2, pad, pad)?;
        let h = h.conv1d(&self.dwconv_weight, 0, 1, 1, self.dim)?;
        let h = h.broadcast_add(&self.dwconv_bias.reshape((1, self.dim, 1))?)?;
        let h = h.transpose(1, 2)?; // (B, T, C)
        let mean = h.mean_keepdim(D::Minus1)?;
        let var = h.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
        let h = h.broadcast_sub(&mean)?.broadcast_div(&(var + 1e-5)?.sqrt()?)?;
        let h = h.broadcast_mul(&self.norm_weight)?.broadcast_add(&self.norm_bias)?;
        let h = self.pw1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.pw2.forward(&h)?;
        (x + &h)
    }
}

struct FlowV2TextEncoder {
    char_emb: Embedding,
    blocks: Vec<FlowV2ConvNeXtBlock>,
    proj: Option<Linear>,
    config: FlowV2Config,
}

impl FlowV2TextEncoder {
    fn load(config: FlowV2Config, vb: VarBuilder) -> Result<Self> {
        let char_emb = embedding(
            config.char_vocab_size,
            config.text_encoder_dim,
            vb.pp("char_emb"),
        )?;
        let mut blocks = Vec::new();
        for i in 0..config.text_encoder_layers {
            blocks.push(FlowV2ConvNeXtBlock::load(
                config.text_encoder_dim,
                config.text_encoder_kernel,
                4.0,
                vb.pp(format!("blocks.{}", i)),
            )?);
        }
        let proj = if config.text_encoder_dim != config.cond_dim {
            Some(linear(config.text_encoder_dim, config.cond_dim, vb.pp("proj"))?)
        } else {
            None
        };
        Ok(Self { char_emb, blocks, proj, config })
    }

    fn forward(&self, char_ids: &Tensor, mel_length: usize) -> Result<Tensor> {
        let (b, t_text) = char_ids.dims2()?;
        let device = char_ids.device();

        let mut x = self.char_emb.forward(char_ids)?;

        if t_text < mel_length {
            let filler_ids: Vec<u32> = (0..(mel_length - t_text))
                .map(|_| self.config.filler_token_id)
                .collect();
            let filler = Tensor::from_vec(filler_ids, (b, mel_length - t_text), device)?;
            let filler_emb = self.char_emb.forward(&filler)?;
            x = Tensor::cat(&[&x, &filler_emb], 1)?;
        } else if t_text > mel_length {
            x = x.narrow(1, 0, mel_length)?;
        }

        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        if let Some(ref p) = self.proj {
            x = p.forward(&x)?;
        }
        Ok(x)
    }
}

struct SonataFlowV2Model {
    text_encoder: FlowV2TextEncoder,
    time_emb: TimestepEmbedding,
    cond_proj: Linear,
    input_proj: Linear,
    blocks: Vec<FlowBlock>,
    output_norm_weight: Tensor,
    output_norm_bias: Tensor,
    output_norm_eps: f64,
    output_proj: Linear,
    speaker_emb: Option<Embedding>,
    speaker_proj: Option<Linear>,
    emotion_emb: Option<Embedding>,
    emotion_proj: Option<Linear>,
    prosody_proj: Option<ProsodyProjLayer>,
    config: FlowV2Config,
}

impl SonataFlowV2Model {
    fn load(config: FlowV2Config, vb: VarBuilder) -> Result<Self> {
        let text_encoder = FlowV2TextEncoder::load(config.clone(), vb.pp("text_encoder"))?;
        let time_emb = TimestepEmbedding::load(config.cond_dim, vb.device(), vb.pp("time_emb"))?;

        let mut cond_input_dim = config.cond_dim * 2;
        if config.n_speakers > 0 {
            cond_input_dim += config.cond_dim;
        }
        if config.n_emotions > 0 {
            cond_input_dim += config.cond_dim;
        }
        if config.prosody_dim > 0 {
            cond_input_dim += config.cond_dim;
        }

        let cond_proj = linear(cond_input_dim, config.d_model, vb.pp("cond_proj"))?;
        let input_proj = linear(config.mel_dim * 2, config.d_model, vb.pp("input_proj"))?;

        let mut blocks = Vec::new();
        for i in 0..config.n_layers {
            blocks.push(FlowBlock::load(
                config.d_model,
                config.n_heads,
                config.d_model,
                config.ff_mult,
                config.norm_eps,
                None,
                vb.pp(format!("blocks.{}", i)),
            )?);
        }

        let output_norm_weight = vb.get(config.d_model, "output_norm.weight")?;
        let output_norm_bias = vb.get(config.d_model, "output_norm.bias")?;
        let output_proj = linear(config.d_model, config.mel_dim, vb.pp("output_proj"))?;

        let (speaker_emb, speaker_proj) = if config.n_speakers > 0 {
            let se = embedding(config.n_speakers, config.speaker_dim, vb.pp("speaker_emb"))?;
            let sp = linear(config.speaker_dim, config.cond_dim, vb.pp("speaker_proj"))?;
            (Some(se), Some(sp))
        } else {
            (None, None)
        };

        let (emotion_emb, emotion_proj) = if config.n_emotions > 0 {
            let ee = embedding(config.n_emotions, config.emotion_dim, vb.pp("emotion_emb"))?;
            let ep = linear(config.emotion_dim, config.cond_dim, vb.pp("emotion_proj"))?;
            (Some(ee), Some(ep))
        } else {
            (None, None)
        };

        let prosody_proj = if config.prosody_dim > 0 {
            match (linear(config.prosody_dim, config.cond_dim, vb.pp("prosody_proj.0")),
                  linear(config.cond_dim, config.cond_dim, vb.pp("prosody_proj.2"))) {
                (Ok(p0), Ok(p2)) => Some(ProsodyProjLayer { p0, p2 }),
                _ => None,
            }
        } else {
            None
        };

        Ok(Self {
            text_encoder,
            time_emb,
            cond_proj,
            input_proj,
            blocks,
            output_norm_weight,
            output_norm_bias,
            output_norm_eps: 1e-5,
            output_proj,
            speaker_emb,
            speaker_proj,
            emotion_emb,
            emotion_proj,
            prosody_proj,
            config,
        })
    }
}

struct ProsodyProjLayer {
    p0: Linear,
    p2: Linear,
}

impl ProsodyProjLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.p0.forward(x)?;
        let h = h.silu()?;
        self.p2.forward(&h)
    }
}

// FlowV2Config needs Clone for load
impl Clone for FlowV2Config {
    fn clone(&self) -> Self {
        FlowV2Config {
            d_model: self.d_model,
            n_layers: self.n_layers,
            n_heads: self.n_heads,
            mel_dim: self.mel_dim,
            cond_dim: self.cond_dim,
            ff_mult: self.ff_mult,
            norm_eps: self.norm_eps,
            text_encoder_layers: self.text_encoder_layers,
            text_encoder_dim: self.text_encoder_dim,
            text_encoder_kernel: self.text_encoder_kernel,
            char_vocab_size: self.char_vocab_size,
            filler_token_id: self.filler_token_id,
            n_steps_inference: self.n_steps_inference,
            n_speakers: self.n_speakers,
            speaker_dim: self.speaker_dim,
            n_emotions: self.n_emotions,
            emotion_dim: self.emotion_dim,
            prosody_dim: self.prosody_dim,
            sway_coefficient: self.sway_coefficient,
        }
    }
}

fn sway_timestep(t: f64, s: f64) -> f64 {
    if s.abs() < 1e-6 {
        return t;
    }
    t + s * ((std::f64::consts::PI / 2.0 * t).cos() - 1.0 + t)
}

impl SonataFlowV2Model {
    fn predict_velocity(
        &self,
        x_t: &Tensor,
        t: &Tensor,
        char_ids: &Tensor,
        ref_mel: &Tensor,
        speaker_id: Option<u32>,
        emotion_id: Option<u32>,
        prosody_features: Option<&Tensor>,
        force_uncond: bool,
    ) -> Result<Tensor> {
        let (b, t_mel, _) = x_t.dims3()?;
        let device = x_t.device();
        let dtype = x_t.dtype();

        let text_cond = self.text_encoder.forward(char_ids, t_mel)?;
        let text_cond = if force_uncond {
            Tensor::zeros((b, t_mel, self.config.cond_dim), dtype, device)?
        } else {
            text_cond
        };

        let time_cond = self.time_emb.forward(t)?;
        let time_cond = time_cond.unsqueeze(1)?.broadcast_as(text_cond.shape())?;
        let mut cond_parts: Vec<Tensor> = vec![text_cond, time_cond];

        if let (Some(ref emb), Some(ref proj)) = (&self.speaker_emb, &self.speaker_proj) {
            let spk = if force_uncond {
                Tensor::zeros((b, self.config.cond_dim), dtype, device)?
            } else if let Some(sid) = speaker_id {
                let sid_t = Tensor::from_vec(vec![sid], 1, device)?;
                proj.forward(&emb.forward(&sid_t)?)?
            } else {
                Tensor::zeros((b, self.config.cond_dim), dtype, device)?
            };
            let spk = spk.unsqueeze(1)?.broadcast_as((b, t_mel, self.config.cond_dim))?;
            cond_parts.push(spk);
        }

        if let (Some(ref emb), Some(ref proj)) = (&self.emotion_emb, &self.emotion_proj) {
            let emo = if force_uncond {
                Tensor::zeros((b, t_mel, self.config.cond_dim), dtype, device)?
            } else if let Some(eid) = emotion_id {
                let eid_t = Tensor::from_vec(vec![eid], 1, device)?;
                let emo = proj.forward(&emb.forward(&eid_t)?)?;
                emo.unsqueeze(1)?.broadcast_as((b, t_mel, self.config.cond_dim))?
            } else {
                Tensor::zeros((b, t_mel, self.config.cond_dim), dtype, device)?
            };
            cond_parts.push(emo);
        }

        if let Some(ref proj) = &self.prosody_proj {
            let pros = if force_uncond {
                Tensor::zeros((b, t_mel, self.config.cond_dim), dtype, device)?
            } else if let Some(pf) = prosody_features {
                let p_emb = proj.forward(pf)?;
                p_emb.broadcast_as((b, t_mel, self.config.cond_dim))?
            } else {
                Tensor::zeros((b, t_mel, self.config.cond_dim), dtype, device)?
            };
            cond_parts.push(pros);
        }

        let cond = self.cond_proj.forward(&Tensor::cat(&cond_parts, D::Minus1)?)?;
        let x_input = self.input_proj.forward(&Tensor::cat(&[x_t, ref_mel], D::Minus1)?)?;

        let mut x = x_input;
        for block in &self.blocks {
            x = block.forward(&x, &cond)?;
        }

        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
        let x_norm = x.broadcast_sub(&mean)?.broadcast_div(
            &(var + self.output_norm_eps)?.sqrt()?
        )?;
        let w_b = self.output_norm_weight.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(x_norm.shape())?;
        let b_b = self.output_norm_bias.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(x_norm.shape())?;
        let x_out = (x_norm.broadcast_mul(&w_b)? + b_b)?;
        self.output_proj.forward(&x_out)
    }

    fn sample(
        &self,
        char_ids: &Tensor,
        mel_length: usize,
        ref_mel: Option<&Tensor>,
        n_steps: usize,
        cfg_scale: f32,
        use_heun: bool,
        sway_s: f64,
        dtype: DType,
        speaker_id: Option<u32>,
        emotion_id: Option<u32>,
        prosody_features: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, _) = char_ids.dims2()?;
        let device = char_ids.device();

        let ref_mel = match ref_mel {
            Some(m) => m.clone(),
            None => Tensor::zeros((b, mel_length, self.config.mel_dim), DType::F32, device)?,
        };

        let mut x = Tensor::randn(0.0_f32, 1.0, (b, mel_length, self.config.mel_dim), device)?
            .to_dtype(dtype)?;

        let use_cfg = cfg_scale > 1.0;

        let dt = 1.0 / n_steps as f64;

        for i in 0..n_steps {
            let mut t_val = i as f64 * dt;
            if sway_s.abs() > 1e-6 {
                t_val = sway_timestep(t_val, sway_s);
            }
            let t = Tensor::from_vec(vec![t_val as f32; b], (b, 1), device)?.to_dtype(dtype)?;

            let v1 = if use_cfg {
                let v_cond = self.predict_velocity(&x, &t, char_ids, &ref_mel, speaker_id, emotion_id, prosody_features, false)?;
                let v_uncond = self.predict_velocity(&x, &t, char_ids, &ref_mel, None, None, None, true)?;
                let v_diff = (&v_cond - &v_uncond)?;
                (&v_uncond + v_diff.affine(cfg_scale as f64, 0.0)?)?
            } else {
                self.predict_velocity(&x, &t, char_ids, &ref_mel, speaker_id, emotion_id, prosody_features, false)?
            };

            if use_heun && i + 1 < n_steps {
                let mut t2_val = (i + 1) as f64 * dt;
                if sway_s.abs() > 1e-6 {
                    t2_val = sway_timestep(t2_val, sway_s);
                }
                let x_euler = (&x + v1.affine(dt, 0.0)?)?;
                let t2 = Tensor::from_vec(vec![t2_val as f32; b], (b, 1), device)?.to_dtype(dtype)?;
                let v2 = if use_cfg {
                    let v_cond = self.predict_velocity(&x_euler, &t2, char_ids, &ref_mel, speaker_id, emotion_id, prosody_features, false)?;
                    let v_uncond = self.predict_velocity(&x_euler, &t2, char_ids, &ref_mel, None, None, None, true)?;
                    let v_diff = (&v_cond - &v_uncond)?;
                    (&v_uncond + v_diff.affine(cfg_scale as f64, 0.0)?)?
                } else {
                    self.predict_velocity(&x_euler, &t2, char_ids, &ref_mel, speaker_id, emotion_id, prosody_features, false)?
                };
                x = (&x + (&v1 + &v2)?.affine(dt / 2.0, 0.0)?)?;
            } else {
                x = (x + v1.affine(dt, 0.0)?)?;
            }
        }

        Ok(x)
    }
}

struct SonataFlowV2Engine {
    model: SonataFlowV2Model,
    device: Device,
    dtype: DType,
    speaker_id: Option<u32>,
    cfg_scale: f32,
    n_steps: usize,
    use_heun: bool,
    quality_mode: i32,
    prosody_features: Option<Tensor>,
}

#[no_mangle]
pub extern "C" fn sonata_flow_v2_create(
    weights_path: *const c_char,
    config_path: *const c_char,
) -> *mut c_void {
    if weights_path.is_null() || config_path.is_null() {
        eprintln!("[sonata_flow_v2] Create error: NULL weights or config path");
        return std::ptr::null_mut();
    }
    let weights_str = unsafe { CStr::from_ptr(weights_path).to_str().unwrap_or("") };
    let config_str = unsafe { CStr::from_ptr(config_path).to_str().unwrap_or("") };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (|| -> Result<SonataFlowV2Engine> {
            #[cfg(feature = "metal")]
            let device = Device::new_metal(0)?;
            #[cfg(not(feature = "metal"))]
            let device = Device::Cpu;

            let dtype = DType::F16;
            let config_content = std::fs::read_to_string(config_str)
                .map_err(|e| candle_core::Error::Msg(format!("Config: {}", e)))?;
            let config: FlowV2Config = serde_json::from_str(&config_content)
                .map_err(|e| candle_core::Error::Msg(format!("Config parse: {}", e)))?;

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_str], dtype, &device)?
            };
            let model = SonataFlowV2Model::load(config.clone(), vb)?;

            let t0 = std::time::Instant::now();
            let warmup_chars = Tensor::from_vec(vec![0u32; 5], (1, 5), &device)?;
            let _ = model.sample(&warmup_chars, 25, None, 2, 1.0, false, 0.0, dtype, None, None, None)?;
            eprintln!("[sonata_flow_v2] Warmup: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

            eprintln!("[sonata_flow_v2] Loaded ({}L, mel_dim={}, steps={})",
                      config.n_layers, config.mel_dim, config.n_steps_inference);

            Ok(SonataFlowV2Engine {
                model,
                device,
                dtype,
                speaker_id: None,
                cfg_scale: 2.0f32,
                n_steps: config.n_steps_inference,
                use_heun: false,
                quality_mode: FLOW_QUALITY_BALANCED,
                prosody_features: None,
            })
        })()
    }));

    match result {
        Ok(Ok(engine)) => Box::into_raw(Box::new(engine)) as *mut c_void,
        Ok(Err(e)) => {
            eprintln!("[sonata_flow_v2] Create error: {}", e);
            std::ptr::null_mut()
        }
        Err(e) => {
            eprintln!("[sonata_flow_v2] panic in create: {}", panic_message(e));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_flow_v2_destroy(engine: *mut c_void) {
    if !engine.is_null() {
        unsafe { drop(Box::from_raw(engine as *mut SonataFlowV2Engine)) };
    }
}

#[no_mangle]
pub extern "C" fn sonata_flow_v2_set_cfg_scale(engine: *mut c_void, scale: c_float) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV2Engine) };
    eng.cfg_scale = scale.max(0.0);
    0
}

#[no_mangle]
pub extern "C" fn sonata_flow_v2_set_n_steps(engine: *mut c_void, steps: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV2Engine) };
    if steps > 0 && steps <= 64 {
        eng.n_steps = steps as usize;
    }
    0
}

/// Set quality mode for Flow v2 (FAST=0, BALANCED=1, HIGH=2).
#[no_mangle]
pub extern "C" fn sonata_flow_v2_set_quality_mode(engine: *mut c_void, mode: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV2Engine) };
    match mode {
        0 => { // FAST
            eng.quality_mode = FLOW_QUALITY_FAST;
            eng.n_steps = 4;
            eng.use_heun = false;
            0
        }
        1 => { // BALANCED
            eng.quality_mode = FLOW_QUALITY_BALANCED;
            eng.n_steps = 6;
            eng.use_heun = false;
            0
        }
        2 => { // HIGH
            eng.quality_mode = FLOW_QUALITY_HIGH;
            eng.n_steps = 8;
            eng.use_heun = true;
            0
        }
        _ => -1,
    }
}

#[no_mangle]
pub extern "C" fn sonata_flow_v2_set_speaker(engine: *mut c_void, speaker_id: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV2Engine) };
    eng.speaker_id = if speaker_id < 0 { None } else { Some(speaker_id as u32) };
    0
}

/// Generate mel from text. Returns number of mel frames, or -1 on error.
/// ref_mel can be NULL; target_frames=0 uses auto heuristic from text length.
#[no_mangle]
pub extern "C" fn sonata_flow_v2_generate(
    engine: *mut c_void,
    text: *const c_char,
    ref_mel: *const c_float,
    ref_mel_frames: c_int,
    target_frames: c_int,
    out_mel: *mut c_float,
    max_frames: c_int,
) -> c_int {
    if engine.is_null() || text.is_null() || out_mel.is_null() || max_frames <= 0 {
        return -1;
    }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV2Engine) };
    let text_str = unsafe { CStr::from_ptr(text).to_str().unwrap_or("") };

    let result = (|| -> Result<i32> {
        let mel_dim = eng.model.config.mel_dim;

        let mut char_ids: Vec<u32> = text_str.bytes().map(|b| b as u32).collect();
        if char_ids.is_empty() {
            char_ids.push(eng.model.config.filler_token_id);
        }

        let max_f = max_frames as usize;
        let target_frames = if target_frames > 0 {
            (target_frames as usize).min(max_f)
        } else {
            (char_ids.len() * 4).max(20).min(400).min(max_f)
        };

        let char_tensor = Tensor::from_vec(
            char_ids.clone(),
            (1, char_ids.len()),
            &eng.device,
        )?;

        let ref_mel_opt = if !ref_mel.is_null() && ref_mel_frames > 0 {
            let n = (ref_mel_frames as usize * mel_dim).min(max_frames as usize * mel_dim);
            let data: Vec<f32> = (0..n).map(|i| unsafe { *ref_mel.add(i) }).collect();
            let frames = ref_mel_frames.min(max_frames);
            Some(Tensor::from_vec(data, (1, frames as usize, mel_dim), &eng.device)?)
        } else {
            None
        };

        let mel = eng.model.sample(
            &char_tensor,
            target_frames,
            ref_mel_opt.as_ref(),
            eng.n_steps,
            eng.cfg_scale,
            eng.use_heun,
            eng.model.config.sway_coefficient,
            eng.dtype,
            eng.speaker_id,
            None,
            eng.prosody_features.as_ref(),
        )?;

        let mel_f32 = mel.to_dtype(DType::F32)?.squeeze(0)?;
        let data_2d = mel_f32.to_vec2::<f32>()?;
        let mut data: Vec<f32> = Vec::with_capacity(target_frames * mel_dim);
        for row in &data_2d {
            data.extend_from_slice(row);
        }
        let n_copy = data.len().min(max_frames as usize * mel_dim);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                out_mel as *mut f32,
                n_copy,
            );
        }
        Ok(target_frames as i32)
    })();

    match result {
        Ok(n) => n,
        Err(e) => {
            eprintln!("[sonata_flow_v2] Generate error: {}", e);
            -1
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sonata Flow v3 — Interleaved streaming text → mel with causal attention
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
struct FlowV3Config {
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    #[serde(default = "default_ff_mult_v3")]
    ff_mult: f64,
    #[serde(default = "default_norm_eps")]
    norm_eps: f64,
    mel_dim: usize,
    cond_dim: usize,
    #[serde(default = "default_char_vocab")]
    char_vocab_size: usize,
    #[serde(default = "default_window_size")]
    window_size: usize,
    #[serde(default = "default_chunk_size")]
    chunk_size: usize,
    #[serde(default)]
    n_speakers: usize,
    #[serde(default = "default_speaker_dim_v3")]
    speaker_dim: usize,
    #[serde(default = "default_n_steps")]
    n_steps_inference: usize,
    #[serde(default = "default_sway")]
    sway_coefficient: f64,
    #[serde(default = "default_sample_rate")]
    sample_rate: usize,
    #[serde(default = "default_n_fft")]
    n_fft: usize,
    #[serde(default = "default_hop_length")]
    hop_length: usize,
    #[serde(default)]
    n_emotions: usize,
}

fn default_ff_mult_v3() -> f64 { 4.0 }
fn default_speaker_dim_v3() -> usize { 384 }
fn default_window_size() -> usize { 512 }
fn default_chunk_size() -> usize { 50 }
fn default_sample_rate() -> usize { 24000 }
fn default_n_fft() -> usize { 1024 }
fn default_hop_length() -> usize { 480 }

/// EPSS (empirical) timestep schedule for flow matching. t goes 0.0 → 1.0 (noise at t=0, signal at t=1).
/// Returns schedule with n_steps+1 elements, or empty vec to use linspace fallback.
/// Must match Python train/sonata/modules.py EMPIRICAL_SCHEDULES exactly.
fn epss_schedule(n_steps: usize) -> Vec<f64> {
    static SCHEDULES: &[(usize, &[f64])] = &[
        (4, &[0.0, 0.30, 0.58, 0.82, 1.0]),
        (5, &[0.0, 0.22, 0.44, 0.66, 0.85, 1.0]),
        (6, &[0.0, 0.16, 0.34, 0.52, 0.70, 0.86, 1.0]),
        (7, &[0.0, 0.12, 0.26, 0.42, 0.58, 0.72, 0.86, 1.0]),
        (8, &[0.0, 0.12, 0.26, 0.42, 0.5, 0.58, 0.72, 0.86, 1.0]),
    ];
    for &(k, s) in SCHEDULES {
        if n_steps == k {
            return s.to_vec();
        }
    }
    vec![] // fallback to linspace
}

impl Clone for FlowV3Config {
    fn clone(&self) -> Self {
        FlowV3Config {
            d_model: self.d_model,
            n_layers: self.n_layers,
            n_heads: self.n_heads,
            ff_mult: self.ff_mult,
            norm_eps: self.norm_eps,
            mel_dim: self.mel_dim,
            cond_dim: self.cond_dim,
            char_vocab_size: self.char_vocab_size,
            window_size: self.window_size,
            chunk_size: self.chunk_size,
            n_speakers: self.n_speakers,
            speaker_dim: self.speaker_dim,
            n_steps_inference: self.n_steps_inference,
            sway_coefficient: self.sway_coefficient,
            sample_rate: self.sample_rate,
            n_fft: self.n_fft,
            hop_length: self.hop_length,
            n_emotions: self.n_emotions,
        }
    }
}

/// Token-level emotion steering (TokenLevelEmoSteer). Applied after transformer blocks.
/// Uses directions (n_emotions, d_model) + scorer to weight emotion injection per token.
struct FlowV3EmoSteer {
    directions: Tensor,
    scorer_0: Linear,
    scorer_1: Linear,
}

impl FlowV3EmoSteer {
    fn load(d_model: usize, n_emotions: usize, vb: VarBuilder) -> Result<Self> {
        let vb_te = vb.pp("token_emosteer");
        let directions = vb_te.get((n_emotions, d_model), "directions")?;
        let scorer_0 = linear(d_model, d_model / 4, vb_te.pp("scorer.0"))?;
        let scorer_1 = linear(d_model / 4, 1, vb_te.pp("scorer.2"))?;
        Ok(Self { directions, scorer_0, scorer_1 })
    }

    fn forward(&self, hidden: &Tensor, emotion_id: i32, scale: f32) -> Result<Tensor> {
        let (b, t, d) = hidden.dims3()?;
        let device = hidden.device();
        let dtype = hidden.dtype();

        let h_scored = self.scorer_0.forward(hidden)?;
        let h_scored = h_scored.silu()?;
        let scores = self.scorer_1.forward(&h_scored)?; // (B, T, 1)
        let scores = scores.squeeze(D::Minus1)?; // (B, T)
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?; // (B, T)

        let emo_idx = emotion_id.max(0) as usize;
        let emo_idx = emo_idx.min(self.directions.dim(0)? - 1);
        let direction = self.directions.get(emo_idx)?; // (D,)
        let direction = direction.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, D)
        let direction = direction.broadcast_as((b, t, d))?;

        let weights = weights.unsqueeze(D::Minus1)?; // (B, T, 1)
        let steering = weights.broadcast_mul(&direction)?;
        let steering = steering.affine(scale as f64, 0.0)?;
        (hidden + steering)
    }
}

struct FlowV3InterleavedEncoder {
    char_emb: Embedding,
    mel_proj: Linear,
    type_emb: Embedding,
}

impl FlowV3InterleavedEncoder {
    fn load(char_vocab_size: usize, mel_dim: usize, d_model: usize, vb: VarBuilder) -> Result<Self> {
        let char_emb = embedding(char_vocab_size, d_model, vb.pp("char_emb"))?;
        let mel_proj = linear(mel_dim, d_model, vb.pp("mel_proj"))?;
        let type_emb = embedding(2, d_model, vb.pp("type_emb"))?;
        Ok(Self { char_emb, mel_proj, type_emb })
    }

    fn encode_text(&self, char_ids: &Tensor) -> Result<Tensor> {
        let emb = self.char_emb.forward(char_ids)?;
        let (b, t, _) = emb.dims3()?;
        let device = emb.device();
        let type_ids = Tensor::zeros((b, t), DType::U32, device)?;
        let type_emb = self.type_emb.forward(&type_ids)?;
        (emb + type_emb)
    }

    fn encode_mel(&self, mel: &Tensor) -> Result<Tensor> {
        let emb = self.mel_proj.forward(mel)?;
        let (b, t, _) = emb.dims3()?;
        let device = emb.device();
        let type_ones = Tensor::ones((b, t), DType::U32, device)?;
        let type_emb = self.type_emb.forward(&type_ones)?;
        (emb + type_emb)
    }
}

struct FlowV3CausalSlidingWindowAttention {
    qkv: Linear,
    out: Linear,
    n_heads: usize,
    head_dim: usize,
    window_size: usize,
    rope_cos: Tensor,
    rope_sin: Tensor,
}

impl FlowV3CausalSlidingWindowAttention {
    fn load(dim: usize, n_heads: usize, window_size: usize, max_len: usize, vb: VarBuilder) -> Result<Self> {
        let qkv = linear(dim, 3 * dim, vb.pp("qkv"))?;
        let out = linear(dim, dim, vb.pp("out"))?;
        let head_dim = dim / n_heads;
        let device = vb.device();

        let half = head_dim / 2;
        let mut freqs = vec![0f32; half];
        for i in 0..half {
            freqs[i] = 1.0 / 10000.0_f32.powf(2.0 * i as f32 / head_dim as f32);
        }
        let freqs = Tensor::from_vec(freqs, (1, half), device)?;
        let t = Tensor::arange(0f32, max_len as f32, device)?.reshape((max_len, 1))?;
        let angles = t.matmul(&freqs)?;
        let rope_cos = angles.cos()?;
        let rope_sin = angles.sin()?;

        Ok(Self {
            qkv,
            out,
            n_heads,
            head_dim,
            window_size,
            rope_cos,
            rope_sin,
        })
    }

    fn apply_rope(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, t, _) = x.dims3()?;
        let half = self.head_dim / 2;
        let cos_s = self.rope_cos.i(offset..offset + t)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin_s = self.rope_sin.i(offset..offset + t)?.unsqueeze(0)?.unsqueeze(0)?;

        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let r1 = (x1.broadcast_mul(&cos_s)? - x2.broadcast_mul(&sin_s)?)?;
        let r2 = (x1.broadcast_mul(&sin_s)? + x2.broadcast_mul(&cos_s)?)?;
        Tensor::cat(&[r1, r2], D::Minus1)
    }

    fn forward_causal(&self, x: &Tensor, kv_cache: Option<&(Tensor, Tensor)>, offset: usize,
                      causal: bool) -> Result<(Tensor, (Tensor, Tensor))> {
        let (b, t, _) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();

        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape((b, t, 3, self.n_heads, self.head_dim))?;
        // (b, t, n_heads, head_dim) → (b, n_heads, t, head_dim) for attention
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
        let mut k = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
        let mut v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;

        // apply_rope expects 3D (b*n_heads, t, head_dim)
        let q = self.apply_rope(
            &q.reshape((b * self.n_heads, t, self.head_dim))?, offset)?
            .reshape((b, self.n_heads, t, self.head_dim))?;
        k = self.apply_rope(
            &k.reshape((b * self.n_heads, t, self.head_dim))?, offset)?
            .reshape((b, self.n_heads, t, self.head_dim))?;

        // KV cache — concat on dim 2 (time)
        let prefix_len = kv_cache.as_ref().map(|(ck, _)| ck.dim(2).unwrap_or(0)).unwrap_or(0);
        if prefix_len > 0 {
            if let Some((ck, cv)) = kv_cache {
                k = Tensor::cat(&[ck, &k], 2)?;
                v = Tensor::cat(&[cv, &v], 2)?;
            }
        }

        let t_kv = k.dim(2)?;
        let t_trim = if t_kv > self.window_size { self.window_size } else { t_kv };
        let k = if t_trim < t_kv { k.narrow(2, t_kv - t_trim, t_trim)? } else { k };
        let v = if t_trim < t_kv { v.narrow(2, t_kv - t_trim, t_trim)? } else { v };

        let new_cache = (k.clone(), v.clone());

        // q/k/v are (b, n_heads, t, head_dim) — matmul directly
        let scale = (self.head_dim as f64).sqrt();
        let mut scores = q.contiguous()?.matmul(&k.contiguous()?.transpose(2, 3)?)?;
        scores = scores.affine(1.0 / scale, 0.0)?;

        let t_trim_actual = k.dim(2)?;
        if causal && t > 1 {
            let mask = build_causal_mask(t, t_trim_actual, prefix_len, device, dtype)?;
            scores = scores.broadcast_add(&mask)?;
        }

        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v.contiguous()?)?;
        // (b, n_heads, t, head_dim) → (b, t, n_heads * head_dim)
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, self.n_heads * self.head_dim))?;
        let out = self.out.forward(&out)?;
        Ok((out, new_cache))
    }

    fn forward(&self, x: &Tensor, kv_cache: Option<&(Tensor, Tensor)>, offset: usize) -> Result<(Tensor, (Tensor, Tensor))> {
        self.forward_causal(x, kv_cache, offset, true)
    }
}

struct FlowV3DurationPredictor {
    conv1_weight: Tensor,
    conv1_bias: Tensor,
    norm1_weight: Tensor,
    norm1_bias: Tensor,
    conv2_weight: Tensor,
    conv2_bias: Tensor,
    norm2_weight: Tensor,
    norm2_bias: Tensor,
    proj: Linear,
    d_model: usize,
}

impl FlowV3DurationPredictor {
    fn load(d_model: usize, vb: VarBuilder) -> Result<Self> {
        let conv1_weight = vb.get((d_model, d_model, 3), "conv1.weight")?;
        let conv1_bias = vb.get(d_model, "conv1.bias")?;
        let norm1_weight = vb.get(d_model, "norm1.weight")?;
        let norm1_bias = vb.get(d_model, "norm1.bias")?;
        let conv2_weight = vb.get((d_model, d_model, 3), "conv2.weight")?;
        let conv2_bias = vb.get(d_model, "conv2.bias")?;
        let norm2_weight = vb.get(d_model, "norm2.weight")?;
        let norm2_bias = vb.get(d_model, "norm2.bias")?;
        let proj = linear(d_model, 1, vb.pp("proj"))?;
        Ok(Self {
            conv1_weight, conv1_bias, norm1_weight, norm1_bias,
            conv2_weight, conv2_bias, norm2_weight, norm2_bias,
            proj, d_model,
        })
    }

    fn forward(&self, text_enc: &Tensor) -> Result<Tensor> {
        // PyTorch order: conv → transpose → norm → relu
        let x = text_enc.transpose(1, 2)?;
        let x = x.conv1d(&self.conv1_weight, 1, 1, 1, 1)?;
        let x = x.broadcast_add(&self.conv1_bias.reshape((1, self.d_model, 1))?)?;
        let x = x.transpose(1, 2)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_sub(&mean)?.broadcast_div(&(var + 1e-5)?.sqrt()?)?;
        let x = x.broadcast_mul(&self.norm1_weight)?.broadcast_add(&self.norm1_bias)?;
        let x = x.relu()?;

        let x = x.transpose(1, 2)?;
        let x = x.conv1d(&self.conv2_weight, 1, 1, 1, 1)?;
        let x = x.broadcast_add(&self.conv2_bias.reshape((1, self.d_model, 1))?)?;
        let x = x.transpose(1, 2)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_sub(&mean)?.broadcast_div(&(var + 1e-5)?.sqrt()?)?;
        let x = x.broadcast_mul(&self.norm2_weight)?.broadcast_add(&self.norm2_bias)?;
        let x = x.relu()?;

        let out = self.proj.forward(&x)?;
        Ok(out.squeeze(D::Minus1)?)
    }

    fn expand_encodings(text_enc: &Tensor, durations: &[Vec<u32>], target_len: usize) -> Result<Tensor> {
        let (b, _, d) = text_enc.dims3()?;
        let device = text_enc.device();
        let dtype = text_enc.dtype();
        let text_s = text_enc.to_dtype(DType::F32)?.to_vec3::<f32>()?;

        let mut data = vec![0.0f32; b * target_len * d];
        for bi in 0..b {
            let mut pos = 0usize;
            for ti in 0..durations[bi].len().min(text_s[bi].len()) {
                let dur = durations[bi][ti] as usize;
                if dur == 0 { continue; }
                let end = (pos + dur).min(target_len);
                for p in pos..end {
                    for di in 0..d {
                        data[bi * target_len * d + p * d + di] = text_s[bi][ti][di];
                    }
                }
                pos = end;
                if pos >= target_len { break; }
            }
        }
        Tensor::from_vec(data, (b, target_len, d), device)?.to_dtype(dtype)
    }
}

struct FlowV3Block {
    norm1: AdaLayerNorm,
    attn: FlowV3CausalSlidingWindowAttention,
    norm2: AdaLayerNorm,
    mlp_0: Linear,
    mlp_1: Linear,
}

impl FlowV3Block {
    fn load(dim: usize, n_heads: usize, cond_dim: usize, ff_mult: f64, eps: f64,
            window_size: usize, max_len: usize, vb: VarBuilder) -> Result<Self> {
        let norm1 = AdaLayerNorm::load(dim, cond_dim, eps, vb.pp("norm1"))?;
        let attn = FlowV3CausalSlidingWindowAttention::load(
            dim, n_heads, window_size, max_len, vb.pp("attn"),
        )?;
        let norm2 = AdaLayerNorm::load(dim, cond_dim, eps, vb.pp("norm2"))?;
        let ff_dim = (dim as f64 * ff_mult) as usize;
        let mlp_0 = linear(dim, ff_dim, vb.pp("mlp.0"))?;
        let mlp_1 = linear(ff_dim, dim, vb.pp("mlp.2"))?;
        Ok(Self { norm1, attn, norm2, mlp_0, mlp_1 })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor, kv_cache: Option<&(Tensor, Tensor)>, offset: usize)
        -> Result<(Tensor, (Tensor, Tensor))> {
        self.forward_with_causal(x, cond, kv_cache, offset, true)
    }

    fn forward_with_causal(&self, x: &Tensor, cond: &Tensor, kv_cache: Option<&(Tensor, Tensor)>,
                           offset: usize, causal: bool) -> Result<(Tensor, (Tensor, Tensor))> {
        let h_norm = self.norm1.forward(x, cond)?;
        let (h_attn, new_cache) = self.attn.forward_causal(&h_norm, kv_cache, offset, causal)?;
        let x = (x + &h_attn)?;
        let h_mlp = self.norm2.forward(&x, cond)?;
        let h_mlp = self.mlp_0.forward(&h_mlp)?;
        let h_mlp = h_mlp.gelu_erf()?;
        let h_mlp = self.mlp_1.forward(&h_mlp)?;
        let x = (x + h_mlp)?;
        Ok((x, new_cache))
    }
}

struct SonataFlowV3Model {
    interleaved_enc: FlowV3InterleavedEncoder,
    time_emb: TimestepEmbedding,
    speaker_emb: Option<Embedding>,
    speaker_proj: Option<Linear>,
    cond_merge: Linear,
    input_proj: Linear,
    blocks: Vec<FlowV3Block>,
    token_emosteer: Option<FlowV3EmoSteer>,
    duration_predictor: FlowV3DurationPredictor,
    output_norm_weight: Tensor,
    output_norm_bias: Tensor,
    output_norm_eps: f64,
    output_proj: Linear,
    config: FlowV3Config,
}

impl SonataFlowV3Model {
    fn load(config: FlowV3Config, vb: VarBuilder) -> Result<Self> {
        let interleaved_enc = FlowV3InterleavedEncoder::load(
            config.char_vocab_size,
            config.mel_dim,
            config.d_model,
            vb.pp("interleaved_enc"),
        )?;
        let time_emb = TimestepEmbedding::load(config.cond_dim, vb.device(), vb.pp("time_emb"))?;

        let (speaker_emb, speaker_proj) = if config.n_speakers > 0 {
            let se = embedding(config.n_speakers, config.speaker_dim, vb.pp("speaker_emb"))?;
            let sp = linear(config.speaker_dim, config.cond_dim, vb.pp("speaker_proj"))?;
            (Some(se), Some(sp))
        } else {
            (None, None)
        };

        let cond_input_dim = config.cond_dim + if config.n_speakers > 0 { config.cond_dim } else { 0 };
        let cond_merge = linear(cond_input_dim, config.d_model, vb.pp("cond_merge"))?;
        let input_proj = linear(config.mel_dim, config.d_model, vb.pp("input_proj"))?;

        let max_len = 8192usize;
        let mut blocks = Vec::new();
        for i in 0..config.n_layers {
            blocks.push(FlowV3Block::load(
                config.d_model,
                config.n_heads,
                config.d_model,
                config.ff_mult,
                config.norm_eps,
                config.window_size,
                max_len,
                vb.pp(format!("blocks.{}", i)),
            )?);
        }

        let token_emosteer = if config.n_emotions > 0 {
            match FlowV3EmoSteer::load(config.d_model, config.n_emotions, vb.pp("token_emosteer")) {
                Ok(em) => {
                    eprintln!("[sonata_flow_v3] TokenLevelEmoSteer loaded ({} emotions)", config.n_emotions);
                    Some(em)
                }
                Err(e) => {
                    eprintln!("[sonata_flow_v3] TokenLevelEmoSteer load skipped: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let duration_predictor = FlowV3DurationPredictor::load(config.d_model, vb.pp("duration_predictor"))?;
        let output_norm_weight = vb.get(config.d_model, "output_norm.weight")?;
        let output_norm_bias = vb.get(config.d_model, "output_norm.bias")?;
        let output_proj = linear(config.d_model, config.mel_dim, vb.pp("output_proj"))?;

        Ok(Self {
            interleaved_enc,
            time_emb,
            speaker_emb,
            speaker_proj,
            cond_merge,
            input_proj,
            blocks,
            token_emosteer,
            duration_predictor,
            output_norm_weight,
            output_norm_bias,
            output_norm_eps: 1e-5,
            output_proj,
            config,
        })
    }

    fn build_conditioning(&self, t: &Tensor, t_seq: usize, speaker_id: Option<u32>, force_uncond: bool) -> Result<Tensor> {
        // t may be (b,) or (b, 1) — flatten to (b,) for time_emb
        let t_flat = if t.rank() == 2 { t.squeeze(1)? } else { t.clone() };
        let b = t_flat.dims1()?;
        let device = t.device();
        let dtype = t.dtype();

        let time_cond = self.time_emb.forward(&t_flat)?;
        let target = Tensor::zeros((b, t_seq, self.config.cond_dim), dtype, device)?;
        let time_cond = time_cond.unsqueeze(1)?.broadcast_as(target.shape())?;
        let mut parts = vec![time_cond];

        if let (Some(ref emb), Some(ref proj)) = (&self.speaker_emb, &self.speaker_proj) {
            let spk = if force_uncond {
                Tensor::zeros((b, self.config.cond_dim), dtype, device)?
            } else if let Some(sid) = speaker_id {
                let sid_t = Tensor::from_vec(vec![sid], 1, device)?;
                proj.forward(&emb.forward(&sid_t)?)?
            } else {
                Tensor::zeros((b, self.config.cond_dim), dtype, device)?
            };
            let spk = spk.unsqueeze(1)?.broadcast_as(target.shape())?;
            parts.push(spk);
        }

        let cond = self.cond_merge.forward(&Tensor::cat(&parts, D::Minus1)?)?;
        Ok(cond)
    }

    fn predict_velocity(&self, x_t: &Tensor, t: &Tensor, text_cond: &Tensor,
                       speaker_id: Option<u32>, force_uncond: bool,
                       ref_mel: Option<&Tensor>, emotion_id: Option<i32>) -> Result<Tensor> {
        let (b, t_mel, _) = x_t.dims3()?;
        let device = x_t.device();
        let dtype = x_t.dtype();

        let (ref_frames, total_len) = if let Some(ref r) = ref_mel {
            let (_, rf, _) = r.dims3()?;
            (rf, t_mel + rf)
        } else {
            (0, t_mel)
        };

        let text_cond = if force_uncond {
            Tensor::zeros((b, t_mel, self.config.d_model), dtype, device)?
        } else {
            text_cond.clone()
        };

        let text_cond_full = if ref_frames > 0 {
            let ref_cond = text_cond.narrow(1, 0, 1)?
                .expand((b, ref_frames, self.config.d_model))?;
            Tensor::cat(&[&ref_cond, &text_cond], 1)?
        } else {
            text_cond
        };

        let cond = self.build_conditioning(t, total_len, speaker_id, force_uncond)?;
        let mut x = self.input_proj.forward(x_t)?;
        if let Some(ref r) = ref_mel {
            let ref_enc = self.interleaved_enc.encode_mel(r)?;
            x = Tensor::cat(&[&ref_enc, &x], 1)?;
        }
        x = (x + text_cond_full)?;

        for block in &self.blocks {
            let (x_out, _cache) = block.forward(&x, &cond, None, 0)?;
            x = x_out;
        }

        if let (Some(ref emosteer), Some(eid)) = (&self.token_emosteer, emotion_id) {
            if eid >= 0 && !force_uncond {
                x = emosteer.forward(&x, eid, 1.0)?;
            }
        }

        if ref_frames > 0 {
            x = x.narrow(1, ref_frames, t_mel)?;
        }

        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_sub(&mean)?.broadcast_div(&(var + self.output_norm_eps)?.sqrt()?)?;
        let w_b = self.output_norm_weight.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(x.shape())?;
        let b_b = self.output_norm_bias.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(x.shape())?;
        let x = (x.broadcast_mul(&w_b)? + b_b)?;
        self.output_proj.forward(&x)
    }

    fn predict_velocity_streaming(&self, x_t: &Tensor, t: &Tensor, text_cond: &Tensor,
                                  speaker_id: Option<u32>, force_uncond: bool,
                                  kv_caches: Option<&[(Tensor, Tensor)]>,
                                  offset: usize, causal: bool)
        -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let (b, t_mel, _) = x_t.dims3()?;
        let device = x_t.device();
        let dtype = x_t.dtype();

        let text_cond = if force_uncond {
            Tensor::zeros((b, t_mel, self.config.d_model), dtype, device)?
        } else {
            text_cond.clone()
        };

        let cond = self.build_conditioning(t, t_mel, speaker_id, force_uncond)?;
        let mut x = self.input_proj.forward(x_t)?;
        x = (x + text_cond)?;

        let mut new_caches = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let cache = kv_caches.and_then(|c| c.get(i));
            let (x_out, nc) = block.forward_with_causal(&x, &cond, cache, offset, causal)?;
            x = x_out;
            new_caches.push(nc);
        }

        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_sub(&mean)?.broadcast_div(&(var + self.output_norm_eps)?.sqrt()?)?;
        let w_b = self.output_norm_weight.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(x.shape())?;
        let b_b = self.output_norm_bias.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(x.shape())?;
        let x = (x.broadcast_mul(&w_b)? + b_b)?;
        let v = self.output_proj.forward(&x)?;
        Ok((v, new_caches))
    }

    fn sample_stream_chunk(&self, chunk_cond: &Tensor, n_steps: usize, cfg_scale: f32,
                           use_heun: bool, dtype: DType, speaker_id: Option<u32>,
                           kv_caches: Option<Vec<(Tensor, Tensor)>>,
                           offset: usize, causal: bool) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let (b, chunk_len, _) = chunk_cond.dims3()?;
        let device = chunk_cond.device();

        let mut x = Tensor::randn(0.0_f32, 1.0, (b, chunk_len, self.config.mel_dim), device)?;
        x = x.to_dtype(dtype)?;

        let t_schedule: Vec<f64> = match epss_schedule(n_steps).as_slice() {
            [] => (0..=n_steps).map(|i| i as f64 / n_steps as f64).collect(),
            s => s.to_vec(),
        };

        let use_cfg = cfg_scale > 1.0;
        let mut caches = kv_caches;

        for i in 0..n_steps {
            let t_val = t_schedule[i];
            let dt = t_schedule[i + 1] - t_schedule[i]; // positive: 0→1 (noise→signal)
            let t = Tensor::from_vec(vec![t_val as f32; b], (b, 1), device)?.to_dtype(dtype)?;

            let cache_ref = caches.as_deref();
            let (v1, new_caches) = if use_cfg {
                let (v_cond, nc) = self.predict_velocity_streaming(
                    &x, &t, chunk_cond, speaker_id, false, cache_ref, offset, causal)?;
                let (v_uncond, _) = self.predict_velocity_streaming(
                    &x, &t, chunk_cond, None, true, cache_ref, offset, causal)?;
                let v_diff = (&v_cond - &v_uncond)?;
                let v = (&v_uncond + v_diff.affine(cfg_scale as f64, 0.0)?)?;
                (v, nc)
            } else {
                self.predict_velocity_streaming(&x, &t, chunk_cond, speaker_id, false,
                                               cache_ref, offset, causal)?
            };
            caches = Some(new_caches.clone());

            if use_heun && i + 1 < n_steps {
                let t2_val = t_schedule[i + 1];
                let x_euler = (&x + v1.affine(dt, 0.0)?)?;
                let t2 = Tensor::from_vec(vec![t2_val as f32; b], (b, 1), device)?.to_dtype(dtype)?;
                let cache_ref = caches.as_ref().map(|c| c.as_slice());
                let (v2, nc2) = if use_cfg {
                    let (v_cond, nc) = self.predict_velocity_streaming(
                        &x_euler, &t2, chunk_cond, speaker_id, false, cache_ref, offset, causal)?;
                    let (v_uncond, _) = self.predict_velocity_streaming(
                        &x_euler, &t2, chunk_cond, None, true, cache_ref, offset, causal)?;
                    let v_diff = (&v_cond - &v_uncond)?;
                    let v = (&v_uncond + v_diff.affine(cfg_scale as f64, 0.0)?)?;
                    (v, nc)
                } else {
                    self.predict_velocity_streaming(&x_euler, &t2, chunk_cond, speaker_id, false,
                                                    cache_ref, offset, causal)?
                };
                caches = Some(nc2);
                x = (&x + (&v1 + &v2)?.affine(dt / 2.0, 0.0)?)?;
            } else {
                x = (x + v1.affine(dt, 0.0)?)?;
            }
        }
        Ok((x, caches.unwrap_or_default()))
    }

    fn sample(&self, char_ids: &Tensor, n_frames: usize, n_steps: usize, cfg_scale: f32,
              use_heun: bool, sway_s: f64, dtype: DType, speaker_id: Option<u32>,
              ref_mel: Option<&Tensor>, emotion_id: Option<i32>)
        -> Result<(Tensor, Vec<f32>)> {
        let (b, t_text) = char_ids.dims2()?;
        let device = char_ids.device();
        let profile = std::env::var("SONATA_PROFILE").ok().as_deref() == Some("1");
        let debug = std::env::var("SONATA_DEBUG").ok().as_deref() == Some("1");

        let (text_enc, text_enc_ms) = if profile {
            let t0 = std::time::Instant::now();
            let enc = self.interleaved_enc.encode_text(char_ids)?;
            (enc, t0.elapsed().as_secs_f64() * 1000.0)
        } else {
            (self.interleaved_enc.encode_text(char_ids)?, 0.0)
        };

        if debug {
            let te_f32 = text_enc.to_dtype(DType::F32)?;
            if let Ok(v) = te_f32.i((0, 0..3, 0..5))?.to_vec2::<f32>() {
                eprintln!("[DBG] text_enc[0,:3,:5]: {:?}", v);
            }
        }

        let (log_dur, dur_pred_ms) = if profile {
            let t0 = std::time::Instant::now();
            let ld = self.duration_predictor.forward(&text_enc)?;
            (ld, t0.elapsed().as_secs_f64() * 1000.0)
        } else {
            (self.duration_predictor.forward(&text_enc)?, 0.0)
        };
        let mut dur = log_dur.exp()?.clamp(1.0e-6, 1e6)?.to_dtype(DType::F32)?;
        let nonpad = char_ids.ne(0u32)?.to_dtype(DType::F32)?;
        dur = dur.broadcast_mul(&nonpad)?;
        if debug {
            if let Ok(v) = dur.to_vec2::<f32>() { eprintln!("[DBG] durations: {:?}", v[0]); }
        }
        let mut dur_int: Vec<Vec<u32>> = Vec::with_capacity(b);
        for bi in 0..b {
            let mut row = Vec::with_capacity(t_text);
            for ti in 0..t_text {
                let v = dur.get(bi).and_then(|r| r.get(ti)).and_then(|t| t.to_scalar::<f32>()).unwrap_or(0.0);
                row.push(v.round().max(1.0) as u32);
            }
            dur_int.push(row);
        }
        let mut dur_sums: Vec<usize> = dur_int.iter().map(|d| d.iter().map(|&x| x as usize).sum()).collect();
        let mut dur_int = dur_int;
        for bi in 0..b {
            let diff = n_frames as i64 - dur_sums[bi] as i64;
            if diff != 0 {
                let mut max_idx = 0usize;
                let mut max_val = 0u32;
                for ti in 0..t_text {
                    if dur_int[bi][ti] > max_val {
                        max_val = dur_int[bi][ti];
                        max_idx = ti;
                    }
                }
                let new_val = (dur_int[bi][max_idx] as i64 + diff).max(1) as u32;
                dur_sums[bi] = dur_sums[bi] - dur_int[bi][max_idx] as usize + new_val as usize;
                dur_int[bi][max_idx] = new_val;
            }
        }

        let text_cond = FlowV3DurationPredictor::expand_encodings(&text_enc, &dur_int, n_frames)?;
        let text_cond = text_cond.to_dtype(dtype)?;

        let mut x = Tensor::randn(0.0_f32, 1.0, (b, n_frames, self.config.mel_dim), device)?;
        x = x.to_dtype(dtype)?;

        let use_cfg = cfg_scale > 1.0;
        let t_schedule: Vec<f64> = match epss_schedule(n_steps).as_slice() {
            [] => (0..=n_steps).map(|i| i as f64 / n_steps as f64).collect(),
            s => s.to_vec(),
        };

        let ode_start = if profile { Some(std::time::Instant::now()) } else { None };

        for i in 0..n_steps {
            let t_val = if sway_s.abs() > 1e-6 {
                sway_timestep(t_schedule[i], sway_s)
            } else {
                t_schedule[i]
            };
            let dt = t_schedule[i + 1] - t_schedule[i]; // positive: 0→1 (noise→signal)
            let t = Tensor::from_vec(vec![t_val as f32; b], (b, 1), device)?.to_dtype(dtype)?;

            let v1 = if use_cfg {
                let v_cond = self.predict_velocity(&x, &t, &text_cond, speaker_id, false, ref_mel, emotion_id)?;
                let v_uncond = self.predict_velocity(&x, &t, &text_cond, None, true, None, None)?;
                let v_diff = (&v_cond - &v_uncond)?;
                (&v_uncond + v_diff.affine(cfg_scale as f64, 0.0)?)?
            } else {
                self.predict_velocity(&x, &t, &text_cond, speaker_id, false, ref_mel, emotion_id)?
            };

            if use_heun && i + 1 < n_steps {
                let t2_val = if sway_s.abs() > 1e-6 {
                    sway_timestep(t_schedule[i + 1], sway_s)
                } else {
                    t_schedule[i + 1]
                };
                let x_euler = (&x + v1.affine(dt, 0.0)?)?;
                let t2 = Tensor::from_vec(vec![t2_val as f32; b], (b, 1), device)?.to_dtype(dtype)?;
                let v2 = if use_cfg {
                    let v_cond = self.predict_velocity(&x_euler, &t2, &text_cond, speaker_id, false,
                                                      ref_mel, emotion_id)?;
                    let v_uncond = self.predict_velocity(&x_euler, &t2, &text_cond, None, true, None, None)?;
                    let v_diff = (&v_cond - &v_uncond)?;
                    (&v_uncond + v_diff.affine(cfg_scale as f64, 0.0)?)?
                } else {
                    self.predict_velocity(&x_euler, &t2, &text_cond, speaker_id, false,
                                          ref_mel, emotion_id)?
                };
                x = (&x + (&v1 + &v2)?.affine(dt / 2.0, 0.0)?)?;
            } else {
                x = (x + v1.affine(dt, 0.0)?)?;
            }
        }

        if profile {
            let ode_ms = ode_start.map(|s| s.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
            let total_ms = text_enc_ms + dur_pred_ms + ode_ms;
            eprintln!("[sonata_flow_v3] Profiling: text_enc={:.1}ms dur_pred={:.1}ms ode={:.1}ms total={:.1}ms",
                      text_enc_ms, dur_pred_ms, ode_ms, total_ms);
        }
        if debug {
            let x_f32 = x.to_dtype(DType::F32)?;
            if let Ok(mean) = x_f32.mean_all()?.to_scalar::<f32>() {
                let min = x_f32.flatten_all()?.min(0)?.to_scalar::<f32>().unwrap_or(0.0);
                let max = x_f32.flatten_all()?.max(0)?.to_scalar::<f32>().unwrap_or(0.0);
                eprintln!("[DBG] mel output: mean={:.3}, min={:.3}, max={:.3}, shape={:?}", mean, min, max, x.shape());
            }
        }
        // Extract per-phoneme durations (use adjusted dur_int for consistency with expand)
        let durations: Vec<f32> = (0..t_text)
            .map(|ti| dur_int[0][ti] as f32)
            .collect();
        Ok((x, durations))
    }
}

struct FlowV3StreamingState {
    text_cond: Tensor,
    generated_frames: usize,
    total_frames: usize,
    kv_caches: Vec<(Tensor, Tensor)>,
    chunk_idx: usize,
    overlap_frames: usize,
    prev_overlap: Option<Tensor>,
}

/// Check if a safetensors checkpoint is a distilled model by examining metadata.
/// Distilled checkpoints have a 'distilled: true' flag in the header.
fn is_distilled_checkpoint(weights_path: &str) -> bool {
    // Only read the header (first 8 bytes for size + header JSON), not the entire file
    let file = match std::fs::File::open(weights_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[sonata_flow_v3] Warning: could not open weights for distilled check: {}", e);
            return false;
        }
    };
    let file_size = match file.metadata() {
        Ok(m) => m.len(),
        Err(_) => return false,
    };
    // Safetensors header is typically <100KB. Cap read at 1MB to prevent OOM.
    const MAX_HEADER_READ: u64 = 1_048_576;
    if file_size > MAX_HEADER_READ {
        // For very large files, read just enough for the header
        use std::io::Read;
        let mut reader = std::io::BufReader::new(file);
        let mut header_size_buf = [0u8; 8];
        if std::io::Read::read_exact(&mut reader, &mut header_size_buf).is_err() {
            return false;
        }
        let header_size = u64::from_le_bytes(header_size_buf);
        if header_size > MAX_HEADER_READ {
            return false; // Header too large, skip
        }
        let mut header_buf = vec![0u8; header_size as usize];
        if std::io::Read::read_exact(&mut reader, &mut header_buf).is_err() {
            return false;
        }
        // Parse header JSON for "distilled" key
        let header_str = match std::str::from_utf8(&header_buf) {
            Ok(s) => s,
            Err(_) => return false,
        };
        return header_str.contains("\"distilled\":true") ||
               header_str.contains("\"distilled\": true") ||
               header_str.contains("\"distilled\":\"true\"") ||
               header_str.contains("\"distilled\": \"true\"");
    }
    // Small file: safe to read entirely, but enforce absolute size limit
    const MAX_FILE_SIZE: u64 = 1_000_000_000;  // 1GB limit
    if file_size > MAX_FILE_SIZE {
        eprintln!("[is_distilled] File too large: {} bytes", file_size);
        return false;
    }
    let data = match std::fs::read(weights_path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    match SafeTensors::read_metadata(&data) {
        Ok((_header_size, metadata)) => {
            if let Some(meta_map) = metadata.metadata() {
                if let Some(val) = meta_map.get("distilled") {
                    return val.to_lowercase() == "true";
                }
            }
            false
        }
        Err(_) => false,
    }
}

struct SonataFlowV3Engine {
    model: SonataFlowV3Model,
    device: Device,
    dtype: DType,
    n_steps: usize,
    cfg_scale: f32,
    use_heun: bool,
    quality_mode: i32,
    speaker_id: Option<u32>,
    ref_mel: Option<Tensor>,
    current_emotion_id: Option<i32>,
    streaming_mode: u32, // 0 = causal, 1 = dragon
    streaming: Option<FlowV3StreamingState>,
    /// Per-phoneme/char durations from last generate() call, in mel frames.
    /// Frame duration = hop_length/sample_rate (typically 20ms at 24kHz).
    last_durations: Vec<f32>,
    /// True if this is a distilled checkpoint (trained for 1-step generation)
    is_distilled: bool,
}

#[no_mangle]
pub extern "C" fn sonata_flow_v3_create(
    weights_path: *const c_char,
    config_path: *const c_char,
) -> *mut c_void {
    if weights_path.is_null() || config_path.is_null() {
        eprintln!("[sonata_flow_v3] Create error: NULL weights or config path");
        return std::ptr::null_mut();
    }
    let weights_str = unsafe { CStr::from_ptr(weights_path).to_str().unwrap_or("") };
    let config_str = unsafe { CStr::from_ptr(config_path).to_str().unwrap_or("") };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (|| -> Result<SonataFlowV3Engine> {
            #[cfg(feature = "metal")]
            let device = Device::new_metal(0)?;
            #[cfg(not(feature = "metal"))]
            let device = Device::Cpu;

            // Use F32 for correctness — F16 dtype mixing causes issues in ODE sampling
            let dtype = DType::F32;
            let config_content = std::fs::read_to_string(config_str)
                .map_err(|e| candle_core::Error::Msg(format!("Config: {}", e)))?;
            let config: FlowV3Config = serde_json::from_str(&config_content)
                .map_err(|e| candle_core::Error::Msg(format!("Config parse: {}", e)))?;

            // Check if this is a distilled checkpoint
            let is_distilled = is_distilled_checkpoint(weights_str);

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_str], dtype, &device)?
            };
            let model = SonataFlowV3Model::load(config.clone(), vb)?;

            {
                let t0 = std::time::Instant::now();
                let warmup_ids = Tensor::from_vec(vec![0u32; 4], (1, 4), &device)?;
                let _ = model.sample(&warmup_ids, 4, 2, 1.0, false, 0.0, dtype, None, None, None).map(|(m, _)| m)?;
                eprintln!("[sonata_flow_v3] Metal warmup: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
            }

            // Auto-configure n_steps and solver for distilled models
            let (n_steps, use_heun) = if is_distilled {
                eprintln!("[sonata_flow_v3] Distilled model detected: forcing n_steps=1, solver=Euler");
                (1, false)
            } else {
                (config.n_steps_inference, false)
            };

            eprintln!("[sonata_flow_v3] Loaded ({}L, mel_dim={}, steps={}, distilled={})",
                      config.n_layers, config.mel_dim, n_steps, is_distilled);

            Ok(SonataFlowV3Engine {
                model,
                device,
                dtype,
                n_steps,
                cfg_scale: 2.0f32,
                use_heun,
                quality_mode: FLOW_QUALITY_BALANCED,
                speaker_id: None,
                ref_mel: None,
                current_emotion_id: None,
                streaming_mode: 0,
                streaming: None,
                last_durations: Vec::new(),
                is_distilled,
            })
        })()
    }));

    match result {
        Ok(Ok(engine)) => Box::into_raw(Box::new(engine)) as *mut c_void,
        Ok(Err(e)) => {
            eprintln!("[sonata_flow_v3] Create error: {}", e);
            std::ptr::null_mut()
        }
        Err(e) => {
            eprintln!("[sonata_flow_v3] panic in create: {}", panic_message(e));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_flow_v3_destroy(engine: *mut c_void) {
    if !engine.is_null() {
        unsafe { drop(Box::from_raw(engine as *mut SonataFlowV3Engine)) };
    }
}

#[no_mangle]
pub extern "C" fn sonata_flow_v3_set_cfg_scale(engine: *mut c_void, scale: c_float) -> c_int {
    if engine.is_null() { return -1; }
    if scale < 0.0 {
        return -1;
    }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV3Engine) };
    eng.cfg_scale = scale;
    0
}

#[no_mangle]
pub extern "C" fn sonata_flow_v3_set_n_steps(engine: *mut c_void, steps: c_int) -> c_int {
    if engine.is_null() { return -1; }
    if steps <= 0 || steps > 64 {
        return -1;
    }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV3Engine) };
    if eng.is_distilled {
        eprintln!("[sonata_flow_v3] Warning: ignoring set_n_steps on distilled model (locked to 1)");
        return 0;
    }
    eng.n_steps = steps as usize;
    0
}

/// Set quality mode for Flow v3 (FAST=0, BALANCED=1, HIGH=2).
#[no_mangle]
pub extern "C" fn sonata_flow_v3_set_quality_mode(engine: *mut c_void, mode: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV3Engine) };
    if eng.is_distilled {
        eprintln!("[sonata_flow_v3] Warning: ignoring set_quality_mode on distilled model (locked to 1-step Euler)");
        return 0;
    }
    match mode {
        0 => { // FAST
            eng.quality_mode = FLOW_QUALITY_FAST;
            eng.n_steps = 4;
            eng.use_heun = false;
            0
        }
        1 => { // BALANCED
            eng.quality_mode = FLOW_QUALITY_BALANCED;
            eng.n_steps = 6;
            eng.use_heun = false;
            0
        }
        2 => { // HIGH
            eng.quality_mode = FLOW_QUALITY_HIGH;
            eng.n_steps = 8;
            eng.use_heun = true;
            0
        }
        _ => -1,
    }
}

#[no_mangle]
pub extern "C" fn sonata_flow_v3_set_speaker(engine: *mut c_void, speaker_id: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV3Engine) };
    eng.speaker_id = if speaker_id < 0 { None } else { Some(speaker_id as u32) };
    0
}

#[no_mangle]
pub extern "C" fn sonata_flow_v3_set_solver(engine: *mut c_void, use_heun: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV3Engine) };
    eng.use_heun = use_heun != 0;
    0
}

/// Query if the loaded model is a distilled checkpoint (trained for 1-step generation).
/// Returns 1 if distilled, 0 if not distilled, -1 on error.
#[no_mangle]
pub extern "C" fn sonata_flow_v3_is_distilled(engine: *mut c_void) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &*(engine as *mut SonataFlowV3Engine) };
    if eng.is_distilled { 1 } else { 0 }
}

#[no_mangle]
pub extern "C" fn sonata_flow_v3_set_streaming_mode(engine: *mut c_void, mode: c_int) -> c_int {
    if engine.is_null() { return -1; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV3Engine) };
    eng.streaming_mode = if mode == 1 { 1 } else { 0 };
    0
}

#[no_mangle]
pub extern "C" fn sonata_flow_v3_stream_start(
    engine: *mut c_void,
    phoneme_ids: *const c_int,
    n_ids: c_int,
) -> c_int {
    const MAX_IDS: c_int = 16384;
    if engine.is_null() || phoneme_ids.is_null() || n_ids <= 0 || n_ids > MAX_IDS {
        return -1;
    }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV3Engine) };
    let result = (|| -> Result<()> {
        let ids: Vec<u32> = (0..n_ids as usize)
            .map(|i| unsafe { *phoneme_ids.add(i) as u32 })
            .collect();
        let char_tensor = Tensor::from_vec(ids.clone(), (1, ids.len()), &eng.device)?;
        let text_enc = eng.model.interleaved_enc.encode_text(&char_tensor)?;
        let log_dur = eng.model.duration_predictor.forward(&text_enc)?;
        let mut dur = log_dur.exp()?.clamp(1.0e-6, 1e6)?.to_dtype(DType::F32)?;
        let nonpad = char_tensor.ne(0u32)?.to_dtype(DType::F32)?;
        dur = dur.broadcast_mul(&nonpad)?;
        let t_text = ids.len();
        let mut dur_int: Vec<Vec<u32>> = Vec::with_capacity(1);
        dur_int.push((0..t_text).map(|ti| {
            dur.get(0).and_then(|r| r.get(ti)).and_then(|t| t.to_scalar::<f32>())
                .unwrap_or(0.0).round().max(1.0) as u32
        }).collect());
        let dur_sum: usize = dur_int[0].iter().map(|&x| x as usize).sum();
        let n_frames = dur_sum.max(25).min(800);
        let diff = n_frames as i64 - dur_sum as i64;
        if diff != 0 {
            let mut max_idx = 0usize;
            let mut max_val = 0u32;
            for ti in 0..t_text {
                if dur_int[0][ti] > max_val {
                    max_val = dur_int[0][ti];
                    max_idx = ti;
                }
            }
            dur_int[0][max_idx] = (dur_int[0][max_idx] as i64 + diff).max(1) as u32;
        }
        let text_cond = FlowV3DurationPredictor::expand_encodings(&text_enc, &dur_int, n_frames)?;
        let text_cond = text_cond.to_dtype(eng.dtype)?;
        let overlap = 5usize;
        eng.streaming = Some(FlowV3StreamingState {
            text_cond,
            generated_frames: 0,
            total_frames: n_frames,
            kv_caches: Vec::new(),
            chunk_idx: 0,
            overlap_frames: overlap,
            prev_overlap: None,
        });
        Ok(())
    })();
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("[sonata_flow_v3] stream_start error: {}", e);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_flow_v3_stream_chunk(
    engine: *mut c_void,
    out_mel: *mut c_float,
    max_frames: c_int,
) -> c_int {
    if engine.is_null() || out_mel.is_null() || max_frames <= 0 {
        return -1;
    }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV3Engine) };
    let Some(ref mut st) = eng.streaming else {
        eprintln!("[sonata_flow_v3] stream_chunk: no active stream (call stream_start first)");
        return -1;
    };
    let chunk_size = eng.model.config.chunk_size;
    let stride = (chunk_size - st.overlap_frames).max(1);
    let chunk_start = st.chunk_idx * stride;
    if chunk_start >= st.total_frames {
        return 0;
    }
    let chunk_end = (chunk_start + chunk_size).min(st.total_frames);
    let chunk_len = chunk_end - chunk_start;

    let result = (|| -> Result<i32> {
        let chunk_cond = st.text_cond.narrow(1, chunk_start, chunk_len)?;
        let offset = st.generated_frames;
        let causal = eng.streaming_mode == 0;
        let kv = if st.kv_caches.is_empty() {
            None
        } else {
            Some(st.kv_caches.clone())
        };
        let (mel, new_kv) = eng.model.sample_stream_chunk(
            &chunk_cond, eng.n_steps, eng.cfg_scale, eng.use_heun, eng.dtype,
            eng.speaker_id, kv, offset, causal)?;
        st.kv_caches = new_kv;

        let out_tensor = mel.to_dtype(DType::F32)?.squeeze(0)?;
        let actual_overlap = if let Some(ref prev) = st.prev_overlap {
            st.overlap_frames.min(out_tensor.dim(1)?).min(prev.dim(1)?)
        } else {
            0
        };
        let out_slice = if actual_overlap > 0 {
            out_tensor.narrow(1, actual_overlap, out_tensor.dim(1)? - actual_overlap)?
        } else {
            out_tensor
        };
        if st.overlap_frames > 0 && chunk_end < st.total_frames {
            st.prev_overlap = Some(mel.narrow(1, chunk_len - st.overlap_frames, st.overlap_frames)?.clone());
        } else {
            st.prev_overlap = None;
        }

        let n_out = out_slice.dim(1)?;
        st.generated_frames += n_out;
        st.chunk_idx += 1;

        let mel_dim = eng.model.config.mel_dim;
        let data_2d = out_slice.to_vec2::<f32>()?;
        let mut data: Vec<f32> = Vec::with_capacity(n_out * mel_dim);
        for row in &data_2d {
            data.extend_from_slice(row);
        }
        let n_copy = data.len().min(max_frames as usize * mel_dim);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), out_mel as *mut f32, n_copy);
        }
        Ok(n_out as i32)
    })();

    match result {
        Ok(n) => n,
        Err(e) => {
            eprintln!("[sonata_flow_v3] stream_chunk error: {}", e);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_flow_v3_stream_end(engine: *mut c_void) {
    if engine.is_null() { return; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV3Engine) };
    eng.streaming = None;
}

/// Return per-phoneme/char durations from the last generate() call.
/// Durations are in mel frames (frame_duration = hop_length/sample_rate, typically 20ms).
/// Returns number of durations written, or 0 if none available.
#[no_mangle]
pub extern "C" fn sonata_flow_v3_get_durations(
    engine: *mut c_void,
    out_durations: *mut f32,
    max_n: c_int,
) -> c_int {
    if engine.is_null() || out_durations.is_null() || max_n <= 0 {
        return 0;
    }
    let eng = unsafe { &*(engine as *const SonataFlowV3Engine) };
    let n = eng.last_durations.len().min(max_n as usize);
    if n == 0 {
        return 0;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(eng.last_durations.as_ptr(), out_durations as *mut f32, n);
    }
    n as c_int
}

/// Generate mel from text or phoneme IDs.
/// text_ptr + text_len: character mode (ord(c) % char_vocab_size) when phoneme_ids_ptr is NULL.
/// phoneme_ids_ptr + phoneme_len: phoneme mode — use IDs directly when not NULL.
/// target_frames=0 uses heuristic. Returns number of mel frames, or -1 on error.
#[no_mangle]
pub extern "C" fn sonata_flow_v3_generate(
    engine: *mut c_void,
    text_ptr: *const c_char,
    text_len: c_int,
    phoneme_ids_ptr: *const c_int,
    phoneme_len: c_int,
    target_frames: c_int,
    out_mel: *mut c_float,
    max_frames: c_int,
) -> c_int {
    const MAX_TEXT: c_int = 16384;
    const MAX_MEL: c_int = 32768;
    if engine.is_null() || out_mel.is_null() || max_frames <= 0 || max_frames > MAX_MEL {
        return -1;
    }
    if text_len > MAX_TEXT || phoneme_len > MAX_TEXT {
        return -1;
    }
    let eng = unsafe { &mut *(engine as *mut SonataFlowV3Engine) };
    let char_vocab = eng.model.config.char_vocab_size;

    let result = (|| -> Result<i32> {
        let char_ids: Vec<u32> = if !phoneme_ids_ptr.is_null() && phoneme_len > 0 {
            (0..phoneme_len as usize)
                .map(|i| unsafe { *phoneme_ids_ptr.add(i) as u32 })
                .collect()
        } else {
            let text_str = if text_ptr.is_null() || text_len <= 0 {
                ""
            } else {
                unsafe {
                    let ptr = text_ptr as *const u8;
                    let len = text_len as usize;
                    std::str::from_utf8(std::slice::from_raw_parts(ptr, len))
                        .unwrap_or("")
                }
            };
            text_str.bytes().map(|b| (b as u32) % char_vocab as u32).collect()
        };

        if char_ids.is_empty() {
            return Err(candle_core::Error::Msg("empty input".into()));
        }

        let mel_dim = eng.model.config.mel_dim;
        let max_f = max_frames as usize;
        let n_frames = if target_frames > 0 {
            (target_frames as usize).min(max_f)
        } else {
            let t_text = char_ids.len();
            (t_text * 5).max(25).min(800).min(max_f)
        };

        let char_tensor = Tensor::from_vec(
            char_ids.clone(),
            (1, char_ids.len()),
            &eng.device,
        )?;

        let (mel, durations) = eng.model.sample(
            &char_tensor,
            n_frames,
            eng.n_steps,
            eng.cfg_scale,
            eng.use_heun,
            eng.model.config.sway_coefficient,
            eng.dtype,
            eng.speaker_id,
            eng.ref_mel.as_ref(),
            eng.current_emotion_id,
        )?;
        eng.last_durations = durations;

        let mel_f32 = mel.to_dtype(DType::F32)?.squeeze(0)?;
        let data_2d = mel_f32.to_vec2::<f32>()?;
        let mut data: Vec<f32> = Vec::with_capacity(n_frames * mel_dim);
        for row in &data_2d {
            data.extend_from_slice(row);
        }
        let n_copy = data.len().min(max_frames as usize * mel_dim);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                out_mel as *mut f32,
                n_copy,
            );
        }
        Ok(n_frames as i32)
    })();

    match result {
        Ok(n) => n,
        Err(e) => {
            eprintln!("[sonata_flow_v3] Generate error: {}", e);
            -1
        }
    }
}

// ─── BigVGAN-lite Vocoder (mel → waveform) ────────────────────────────────────
// Matches Python vocoder.py: SnakeAlpha + AMPBlock architecture.
// Weight layout: generator.input_conv, generator.upsamples, generator.amp_blocks,
// generator.output_act.alpha, generator.output_conv.

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct VocoderConfig {
    sample_rate: usize,
    n_mels: usize,
    hop_length: usize,
    upsample_initial_channel: usize,
    upsample_rates: Vec<usize>,
    upsample_kernel_sizes: Vec<usize>,
    resblock_kernel_sizes: Vec<usize>,
    resblock_dilation_sizes: Vec<Vec<usize>>,
}

fn vocoder_get_padding(kernel: usize, dilation: usize) -> usize {
    (kernel * dilation - dilation) / 2
}

/// SnakeAlpha: x + (1/(alpha+eps)) * sin(alpha*x)^2.
/// log_alpha shape (1, channels, 1) — alpha = exp(log_alpha) for stable training.
fn snake_alpha_forward(x: &Tensor, log_alpha: &Tensor) -> Result<Tensor> {
    let alpha = log_alpha.exp()?;
    let eps = 1e-9f32;
    let eps_t = Tensor::new(eps, x.device())?.to_dtype(alpha.dtype())?.broadcast_as(alpha.shape())?;
    let alpha_safe = (&alpha + &eps_t)?;
    let inv_alpha = alpha_safe.recip()?;
    let ax = x.broadcast_mul(&alpha)?;
    let sin_ax = ax.sin()?;
    let sin_sq = (&sin_ax * &sin_ax)?;
    let term = sin_sq.broadcast_mul(&inv_alpha)?;
    (x + &term)
}

/// One layer in AMPBlock: (weight, bias, dilation) for a Conv1d.
struct AMPConvLayer {
    weight: Tensor,
    bias: Tensor,
    kernel: usize,
    dilation: usize,
}

impl AMPConvLayer {
    fn convert_to_f32(&mut self) -> Result<()> {
        self.weight = self.weight.to_dtype(DType::F32)?;
        self.bias = self.bias.to_dtype(DType::F32)?;
        Ok(())
    }
}

/// AMPBlock: Anti-aliased multi-periodicity block. For each kernel size, chain act->conv,
/// sum outputs, divide by n_blocks. Matches Python amp_blocks[i].convs[j][k], acts[j][k].
struct AMPBlock {
    /// For each kernel index j: list of (conv_layer, act_alpha) for dilations in that kernel.
    layers: Vec<Vec<(AMPConvLayer, Tensor)>>,
    ch: usize,
    n_blocks: usize,
}

impl AMPBlock {
    fn convert_to_f32(&mut self) -> Result<()> {
        for branch in &mut self.layers {
            for (conv, alpha) in branch {
                conv.convert_to_f32()?;
                *alpha = alpha.to_dtype(DType::F32)?;
            }
        }
        Ok(())
    }

    fn load(
        ch: usize,
        kernel_sizes: &[usize],
        dilation_sizes: &[Vec<usize>],
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        for (j, (&k, ds)) in kernel_sizes.iter().zip(dilation_sizes).enumerate() {
            let mut branch = Vec::new();
            for (d_idx, &d) in ds.iter().enumerate() {
                let conv_vb = vb.pp(format!("convs.{}.{}", j, d_idx));
                let w = conv_vb.get((ch, ch, k), "weight")?;
                let b = conv_vb.get(ch, "bias")?;
                let conv_layer = AMPConvLayer {
                    weight: w,
                    bias: b,
                    kernel: k,
                    dilation: d,
                };

                let act_vb = vb.pp(format!("acts.{}.{}", j, d_idx));
                let alpha = act_vb.get((1, ch, 1), "log_alpha")?;
                branch.push((conv_layer, alpha));
            }
            layers.push(branch);
        }
        Ok(Self {
            layers,
            ch,
            n_blocks: kernel_sizes.len(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.zeros_like()?;
        for branch in &self.layers {
            let mut h = x.clone();
            for (conv, alpha) in branch {
                h = snake_alpha_forward(&h, alpha)?;
                let pad = vocoder_get_padding(conv.kernel, conv.dilation);
                h = h.pad_with_zeros(2, pad, pad)?;
                h = h.conv1d(&conv.weight, 0, 1, conv.dilation, 1)?;
                h = h.broadcast_add(&conv.bias.reshape((1, self.ch, 1))?)?;
            }
            out = (out + h)?;
        }
        // Residual connection: prevents variance explosion through stacked blocks
        (x + out.affine(1.0 / self.n_blocks as f64, 0.0)?)
    }
}

/// One upsample stage: SnakeAlpha activation → ConvTranspose1d → AMPBlock.
struct VocoderUpsampleStage {
    up_act_log_alpha: Tensor,
    upsample_weight: Tensor,
    upsample_bias: Tensor,
    amp_block: AMPBlock,
    stride: usize,
    kernel: usize,
    in_ch: usize,
    out_ch: usize,
}

impl VocoderUpsampleStage {
    fn load(
        in_ch: usize,
        out_ch: usize,
        stride: usize,
        kernel: usize,
        resblock_kernels: &[usize],
        resblock_dilations: &[Vec<usize>],
        stage_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let up_act_vb = vb.pp(format!("up_acts.{}", stage_idx));
        let up_act_log_alpha = up_act_vb.get((1, in_ch, 1), "log_alpha")?;

        let ups_vb = vb.pp(format!("upsamples.{}", stage_idx));
        let upsample_weight = ups_vb.get((in_ch, out_ch, kernel), "weight")?;
        let upsample_bias = ups_vb.get(out_ch, "bias")?;

        let amp_vb = vb.pp(format!("amp_blocks.{}", stage_idx));
        let amp_block = AMPBlock::load(out_ch, resblock_kernels, resblock_dilations, amp_vb)?;

        Ok(Self {
            up_act_log_alpha,
            upsample_weight,
            upsample_bias,
            amp_block,
            stride,
            kernel,
            in_ch,
            out_ch,
        })
    }

    fn convert_to_f32(&mut self) -> Result<()> {
        self.up_act_log_alpha = self.up_act_log_alpha.to_dtype(DType::F32)?;
        self.upsample_weight = self.upsample_weight.to_dtype(DType::F32)?;
        self.upsample_bias = self.upsample_bias.to_dtype(DType::F32)?;
        self.amp_block.convert_to_f32()?;
        Ok(())
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = snake_alpha_forward(x, &self.up_act_log_alpha)?;
        let pad = (self.kernel as i64 - self.stride as i64).max(0) as usize / 2;
        let mut x = x.conv_transpose1d(&self.upsample_weight, pad, 0, self.stride, 1, 1)?;
        x = x.broadcast_add(&self.upsample_bias.reshape((1, self.out_ch, 1))?)?;
        self.amp_block.forward(&x)
    }
}

struct VocoderGenerator {
    input_conv_weight: Tensor,
    input_conv_bias: Tensor,
    stages: Vec<VocoderUpsampleStage>,
    output_act_alpha: Tensor,
    output_conv_weight: Tensor,
    output_conv_bias: Tensor,
    config: VocoderConfig,
}

impl VocoderGenerator {
    fn load(config: VocoderConfig, vb: VarBuilder) -> Result<Self> {
        let g_vb = vb.pp("generator");
        let ch = config.upsample_initial_channel;

        let input_conv_weight = g_vb.get((ch, config.n_mels, 7), "input_conv.weight")?;
        let input_conv_bias = g_vb.get(ch, "input_conv.bias")?;

        let n_stages = config.upsample_rates.len();
        let mut stages = Vec::new();
        for i in 0..n_stages {
            let in_ch = ch / (1 << i);
            let out_ch = ch / (1 << (i + 1));
            let stride = config.upsample_rates[i];
            let kernel = config.upsample_kernel_sizes[i];
            stages.push(VocoderUpsampleStage::load(
                in_ch,
                out_ch,
                stride,
                kernel,
                &config.resblock_kernel_sizes,
                &config.resblock_dilation_sizes,
                i,
                g_vb.clone(),
            )?);
        }

        let final_ch = ch / (1 << n_stages);
        let output_act_alpha = g_vb.get((1, final_ch, 1), "output_act.log_alpha")?;
        let output_conv_weight = g_vb.get((1, final_ch, 7), "output_conv.weight")?;
        let output_conv_bias = g_vb.get(1, "output_conv.bias")?;

        Ok(Self {
            input_conv_weight,
            input_conv_bias,
            stages,
            output_act_alpha,
            output_conv_weight,
            output_conv_bias,
            config,
        })
    }

    fn convert_to_f32(&mut self) -> Result<()> {
        self.input_conv_weight = self.input_conv_weight.to_dtype(DType::F32)?;
        self.input_conv_bias = self.input_conv_bias.to_dtype(DType::F32)?;
        for stage in &mut self.stages {
            stage.convert_to_f32()?;
        }
        self.output_act_alpha = self.output_act_alpha.to_dtype(DType::F32)?;
        self.output_conv_weight = self.output_conv_weight.to_dtype(DType::F32)?;
        self.output_conv_bias = self.output_conv_bias.to_dtype(DType::F32)?;
        Ok(())
    }

    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let (_b, _t, _m) = mel.dims3()?;
        let mel_t = mel.transpose(1, 2)?;

        let mut x = mel_t.pad_with_zeros(2, 3, 3)?;
        x = x.conv1d(&self.input_conv_weight, 0, 1, 1, 1)?;
        x = x.broadcast_add(&self.input_conv_bias.reshape((1, self.config.upsample_initial_channel, 1))?)?;

        for stage in &self.stages {
            x = stage.forward(&x)?;
        }

        x = snake_alpha_forward(&x, &self.output_act_alpha)?;
        x = x.pad_with_zeros(2, 3, 3)?;
        x = x.conv1d(&self.output_conv_weight, 0, 1, 1, 1)?;
        x = x.broadcast_add(&self.output_conv_bias.reshape((1, 1, 1))?)?;
        x.tanh()
    }
}

struct SonataVocoderEngine {
    generator: VocoderGenerator,
    device: Device,
}

// ─── Vocoder C FFI ───────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn sonata_vocoder_create(weights_path: *const c_char, config_path: *const c_char) -> *mut c_void {
    if weights_path.is_null() || config_path.is_null() {
        eprintln!("[sonata_vocoder] Create error: NULL weights or config path");
        return std::ptr::null_mut();
    }
    let weights_str = unsafe { CStr::from_ptr(weights_path).to_str().unwrap_or("") };
    let config_str = unsafe { CStr::from_ptr(config_path).to_str().unwrap_or("") };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (|| -> Result<SonataVocoderEngine> {
        #[cfg(feature = "metal")]
        let device = Device::new_metal(0)?;
        #[cfg(not(feature = "metal"))]
        let device = Device::Cpu;

        let config_content = std::fs::read_to_string(config_str)
            .map_err(|e| candle_core::Error::Msg(format!("Config read: {}", e)))?;
        let config: VocoderConfig = serde_json::from_str(&config_content)
            .map_err(|e| candle_core::Error::Msg(format!("Config parse: {}", e)))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_str], DType::F32, &device,
            )?
        };
        let mut generator = VocoderGenerator::load(config, vb)?;
        generator.convert_to_f32()?;

        {
            let t0 = std::time::Instant::now();
            let mel_dim = generator.config.n_mels;
            let warmup_mel = Tensor::randn(0.0_f32, 1.0, (1, 4, mel_dim), &device)?;
            let _ = generator.forward(&warmup_mel)?;
            eprintln!("[sonata_vocoder] Metal warmup: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
        }

        eprintln!("[sonata_vocoder] Loaded on {:?}", device);
        Ok(SonataVocoderEngine { generator, device })
        })()
    }));

    match result {
        Ok(Ok(engine)) => Box::into_raw(Box::new(engine)) as *mut c_void,
        Ok(Err(e)) => {
            eprintln!("[sonata_vocoder] Create error: {}", e);
            std::ptr::null_mut()
        }
        Err(e) => {
            eprintln!("[sonata_vocoder] panic in create: {}", panic_message(e));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_vocoder_destroy(engine: *mut c_void) {
    if !engine.is_null() {
        if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            unsafe { drop(Box::from_raw(engine as *mut SonataVocoderEngine)) };
        })) {
            eprintln!("[sonata_vocoder] panic in destroy: {}", panic_message(e));
        }
    }
}

#[no_mangle]
pub extern "C" fn sonata_vocoder_generate(
    engine: *mut c_void,
    mel_data: *const c_float,
    n_frames: c_int,
    mel_dim: c_int,
    out_audio: *mut c_float,
    max_samples: c_int,
) -> c_int {
    const MAX_MEL_FRAMES: c_int = 32768;
    const MAX_MEL_DIM: c_int = 1024;
    if engine.is_null() || mel_data.is_null() || out_audio.is_null()
        || max_samples <= 0 || n_frames <= 0 || n_frames > MAX_MEL_FRAMES
        || mel_dim <= 0 || mel_dim > MAX_MEL_DIM
    {
        return -1;
    }
    let eng = unsafe { &*(engine as *const SonataVocoderEngine) };
    let n_f = n_frames as usize;
    let m = mel_dim as usize;

    let result = (|| -> Result<i32> {
        let data: Vec<f32> = (0..n_f * m).map(|i| unsafe { *mel_data.add(i) }).collect();
        let mel = Tensor::from_vec(data, (1, n_f, m), &eng.device)?;

        let out = if std::env::var("SONATA_PROFILE").ok().as_deref() == Some("1") {
            let t0 = std::time::Instant::now();
            let o = eng.generator.forward(&mel)?;
            eprintln!("[sonata_vocoder] Profiling: vocoder={:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
            o
        } else {
            eng.generator.forward(&mel)?
        };
        let out = out.squeeze(0)?.squeeze(0)?;
        let len = out.dim(0)?;
        let samples = out.to_vec1::<f32>()?;

        let n_copy = len.min(max_samples as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(samples.as_ptr(), out_audio as *mut f32, n_copy);
        }
        Ok(n_copy as i32)
    })();

    match result {
        Ok(n) => n,
        Err(e) => {
            eprintln!("[sonata_vocoder] Generate error: {}", e);
            -1
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Test detection of distilled checkpoint flag in safetensors metadata.
    /// Creates a minimal safetensors file with the 'distilled: true' metadata flag.
    #[test]
    fn test_distilled_checkpoint_detection() {
        // Create a minimal safetensors file with distilled metadata
        // Format: 8-byte header (size), followed by JSON metadata, then tensors
        // The JSON header can optionally contain file-level metadata.
        let header_with_meta = r#"{"__metadata__":{"distilled":"true"}}"#;
        let size = header_with_meta.len() as u64;

        let mut file = NamedTempFile::new().expect("Failed to create temp file");

        // Write header (little-endian u64)
        file.write_all(&size.to_le_bytes())
            .expect("Failed to write size");
        file.write_all(header_with_meta.as_bytes())
            .expect("Failed to write header");
        file.flush().expect("Failed to flush file");

        let path = file.path().to_str().unwrap();
        let is_distilled = is_distilled_checkpoint(path);

        assert!(is_distilled, "Expected to detect distilled=true in metadata");
    }

    /// Test that non-distilled checkpoints return false.
    #[test]
    fn test_non_distilled_checkpoint() {
        // Create a minimal safetensors file WITHOUT distilled metadata
        let metadata_json = r#"{"some_other_field": "value"}"#;
        let header_size = metadata_json.len() as u64;

        let mut file = NamedTempFile::new().expect("Failed to create temp file");

        // Write header (little-endian u64)
        file.write_all(&header_size.to_le_bytes())
            .expect("Failed to write header");

        // Write metadata JSON
        file.write_all(metadata_json.as_bytes())
            .expect("Failed to write metadata");

        file.flush().expect("Failed to flush file");

        let path = file.path().to_str().unwrap();
        let is_distilled = is_distilled_checkpoint(path);

        assert!(!is_distilled, "Expected to NOT detect distilled flag");
    }

    /// Test graceful handling of invalid/missing files.
    #[test]
    fn test_distilled_detection_missing_file() {
        let is_distilled = is_distilled_checkpoint("/nonexistent/path/to/weights.safetensors");
        assert!(!is_distilled, "Expected false for missing file (graceful fallback)");
    }
}
