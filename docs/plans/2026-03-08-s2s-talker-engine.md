# S2S Talker Inference Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the 500M-param Talker inference engine in Rust/Candle+Metal as a cdylib crate, following existing sonata_lm patterns.

**Architecture:** Temporal Transformer (12 layers, 768d, GQA) + Depth Transformer (6 layers, 512d) + Audio/Text embedders + LM heads. Follows the Thinker-Talker design from `docs/plans/2026-03-08-s2s-v2-thinker-talker-design.md`.

**Tech Stack:** Rust, candle-core 0.9, candle-nn 0.9, Metal GPU, safetensors, C FFI

---

## Workstream 1: Talker Crate Scaffold

### Task 1: Create crate skeleton

**Files:**

- Create: `src/sonata_talker/Cargo.toml`
- Create: `src/sonata_talker/src/lib.rs`
- Modify: `Cargo.toml` (workspace members)

**Step 1: Create Cargo.toml**

```toml
[package]
name = "sonata-talker"
version = "0.1.0"
edition = "2021"
autobenches = false

[lib]
crate-type = ["cdylib"]
name = "sonata_talker"

[dependencies]
candle-core = { workspace = true }
candle-nn = { workspace = true }
safetensors = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
rand = { workspace = true }

[features]
default = ["metal"]
metal = ["candle-core/metal", "candle-nn/metal"]
```

**Step 2: Create lib.rs with config**

```rust
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

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder};
use std::ffi::{CStr, c_char, c_float, c_int, c_void};

mod temporal;
mod depth;
mod embedder;

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
    #[serde(default = "default_tau")]           pub acoustic_delay: usize, // 2 frames
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
fn default_tau() -> usize { 2 }

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            d_model: 768, n_temporal_layers: 12, n_heads: 12, n_kv_heads: 4,
            d_ff: 3072, depth_dim: 512, n_depth_layers: 6, depth_heads: 8,
            depth_d_ff: 2048, n_codebooks: 8, codebook_size: 2048,
            text_vocab_size: 32000, thinker_hidden_dim: 4096, max_seq_len: 4096,
            rope_theta: 10000.0, norm_eps: 1e-5, frame_rate_hz: 12.5,
            acoustic_delay: 2,
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
```

**Step 3: Add workspace member**

Add `"src/sonata_talker"` to workspace members in root `Cargo.toml`.

**Step 4: Verify it compiles**

Run: `cargo check -p sonata-talker`
Expected: PASS (empty crate with config)

**Step 5: Commit**

```bash
git add src/sonata_talker/ Cargo.toml
git commit -m "feat: scaffold sonata-talker crate with TalkerConfig"
```

---

## Workstream 2: Temporal Transformer

### Task 2: RoPE + GQAttention (reuse from sonata_lm patterns)

**Files:**

- Create: `src/sonata_talker/src/temporal.rs`

**Step 1: Write test for RoPE cache**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rope_cache_shape() {
        let dev = &Device::Cpu;
        let (cos, sin) = precompute_rope_cache(64, 128, 10000.0, dev).unwrap();
        assert_eq!(cos.dims(), &[128, 32]); // max_len x half_dim
        assert_eq!(sin.dims(), &[128, 32]);
    }

    #[test]
    fn test_temporal_attention_output_shape() {
        let dev = &Device::Cpu;
        let cfg = TalkerConfig::default();
        let vb = VarBuilder::zeros(DType::F32, dev);
        let attn = TemporalAttention::load(&cfg, vb.pp("attn")).unwrap();
        let x = Tensor::zeros((1, 1, 768), DType::F32, dev).unwrap();
        let (cos, sin) = precompute_rope_cache(64, 128, 10000.0, dev).unwrap();
        let mut kc = Tensor::zeros((1, 4, 0, 64), DType::F32, dev).unwrap();
        let mut vc = Tensor::zeros((1, 4, 0, 64), DType::F32, dev).unwrap();
        let out = attn.forward(&x, &cos, &sin, 0, &mut kc, &mut vc).unwrap();
        assert_eq!(out.dims(), &[1, 1, 768]);
    }
}
```

**Step 2: Implement TemporalAttention (GQA with KV cache)**

Follow exact pattern from `sonata_lm/src/lib.rs:241-347` but with 768 dim, 12 heads, 4 KV heads:

```rust
use crate::TalkerConfig;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Linear, Module, RmsNorm, VarBuilder};

pub fn precompute_rope_cache(
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

pub struct TemporalAttention {
    wq: Linear, wk: Linear, wv: Linear, wo: Linear,
    n_heads: usize, n_kv_heads: usize, head_dim: usize,
}

impl TemporalAttention {
    pub fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let hd = cfg.head_dim();
        Ok(Self {
            wq: linear_no_bias(d, cfg.n_heads * hd, vb.pp("wq"))?,
            wk: linear_no_bias(d, cfg.n_kv_heads * hd, vb.pp("wk"))?,
            wv: linear_no_bias(d, cfg.n_kv_heads * hd, vb.pp("wv"))?,
            wo: linear_no_bias(cfg.n_heads * hd, d, vb.pp("wo"))?,
            n_heads: cfg.n_heads, n_kv_heads: cfg.n_kv_heads,
            head_dim: hd,
        })
    }

    pub fn forward(
        &self, x: &Tensor, cos: &Tensor, sin: &Tensor, pos: usize,
        k_cache: &mut Tensor, v_cache: &mut Tensor,
    ) -> Result<Tensor> {
        let (b, _t, _d) = x.dims3()?;
        let q = self.wq.forward(x)?
            .reshape((b, 1, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.wk.forward(x)?
            .reshape((b, 1, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.wv.forward(x)?
            .reshape((b, 1, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;

        let (q, k) = apply_rope(&q, &k, cos, sin, pos)?;

        let (k_full, v_full) = if k_cache.dim(2)? > pos {
            *k_cache = k_cache.slice_scatter(&k, 2, pos)?;
            *v_cache = v_cache.slice_scatter(&v, 2, pos)?;
            (k_cache.narrow(2, 0, pos + 1)?, v_cache.narrow(2, 0, pos + 1)?)
        } else {
            *k_cache = Tensor::cat(&[&*k_cache, &k], 2)?;
            *v_cache = Tensor::cat(&[&*v_cache, &v], 2)?;
            (k_cache.clone(), v_cache.clone())
        };

        let scale = (self.head_dim as f32).powf(-0.5);
        let out = candle_nn::ops::sdpa(&q, &k_full, &v_full, None, false, scale, 1.0)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, 1, ()))?;
        self.wo.forward(&out)
    }
}

pub struct SwiGluFfn { w_gate: Linear, w_up: Linear, w_down: Linear }

impl SwiGluFfn {
    pub fn load(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w_gate: linear_no_bias(d_model, d_ff, vb.pp("w_gate"))?,
            w_up: linear_no_bias(d_model, d_ff, vb.pp("w_up"))?,
            w_down: linear_no_bias(d_ff, d_model, vb.pp("w_down"))?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&self.w_gate.forward(x)?)?;
        self.w_down.forward(&(gate * self.w_up.forward(x)?)?)
    }
}

pub struct TemporalBlock {
    attn_norm: RmsNorm, attn: TemporalAttention,
    ffn_norm: RmsNorm, ffn: SwiGluFfn,
}

impl TemporalBlock {
    pub fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn_norm: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("attn_norm"))?,
            attn: TemporalAttention::load(cfg, vb.pp("attn"))?,
            ffn_norm: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("ffn_norm"))?,
            ffn: SwiGluFfn::load(cfg.d_model, cfg.d_ff, vb.pp("ffn"))?,
        })
    }

    pub fn forward(
        &self, x: &Tensor, cos: &Tensor, sin: &Tensor, pos: usize,
        kc: &mut Tensor, vc: &mut Tensor,
    ) -> Result<Tensor> {
        let h = self.attn.forward(&self.attn_norm.forward(x)?, cos, sin, pos, kc, vc)?;
        let x = (x + h)?;
        let h = self.ffn.forward(&self.ffn_norm.forward(&x)?)?;
        x + h
    }
}

pub struct TemporalTransformer {
    blocks: Vec<TemporalBlock>,
    norm: RmsNorm,
}

impl TemporalTransformer {
    pub fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..cfg.n_temporal_layers {
            blocks.push(TemporalBlock::load(cfg, vb.pp(format!("layer.{}", i)))?);
        }
        Ok(Self {
            blocks,
            norm: rms_norm(cfg.d_model, cfg.norm_eps, vb.pp("norm"))?,
        })
    }

    pub fn forward(
        &self, x: &Tensor, cos: &Tensor, sin: &Tensor, pos: usize,
        k_caches: &mut [Tensor], v_caches: &mut [Tensor],
    ) -> Result<Tensor> {
        let mut h = x.clone();
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, cos, sin, pos, &mut k_caches[i], &mut v_caches[i])?;
        }
        self.norm.forward(&h)
    }
}
```

**Step 3: Run tests**

Run: `cargo test -p sonata-talker -- temporal`
Expected: 2 tests PASS

**Step 4: Commit**

```bash
git add src/sonata_talker/src/temporal.rs
git commit -m "feat(talker): temporal transformer with GQA + RoPE + KV cache"
```

---

### Task 3: Depth Transformer

**Files:**

- Create: `src/sonata_talker/src/depth.rs`

The Depth Transformer processes codebooks sequentially within a single timestep. It's smaller (512d, 6 layers) and generates the 7 acoustic codes given the semantic code.

**Step 1: Write test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_depth_transformer_output_shape() {
        let dev = &Device::Cpu;
        let cfg = TalkerConfig::default();
        let vb = VarBuilder::zeros(DType::F32, dev);
        let depth = DepthTransformer::load(&cfg, vb.pp("depth")).unwrap();
        // Input: semantic code embedding projected to depth_dim
        let x = Tensor::zeros((1, 1, 512), DType::F32, dev).unwrap();
        let out = depth.forward(&x).unwrap();
        // Output: (1, 1, 512) — then projected to codebook logits externally
        assert_eq!(out.dims(), &[1, 1, 512]);
    }

    #[test]
    fn test_depth_generate_acoustic_codes() {
        let dev = &Device::Cpu;
        let cfg = TalkerConfig::default();
        let vb = VarBuilder::zeros(DType::F32, dev);
        let depth = DepthTransformer::load(&cfg, vb.pp("depth")).unwrap();
        let semantic_emb = Tensor::zeros((1, 1, 512), DType::F32, dev).unwrap();
        // Generate 7 acoustic codes autoregressively
        let codes = depth.generate(&semantic_emb, 7).unwrap();
        assert_eq!(codes.len(), 7);
    }
}
```

**Step 2: Implement DepthTransformer**

```rust
use crate::TalkerConfig;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Linear, Module, RmsNorm, VarBuilder};

struct DepthAttention {
    wq: Linear, wk: Linear, wv: Linear, wo: Linear,
    n_heads: usize, head_dim: usize,
}

impl DepthAttention {
    fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.depth_dim;
        let hd = cfg.depth_head_dim();
        Ok(Self {
            wq: linear_no_bias(d, cfg.depth_heads * hd, vb.pp("wq"))?,
            wk: linear_no_bias(d, cfg.depth_heads * hd, vb.pp("wk"))?,
            wv: linear_no_bias(d, cfg.depth_heads * hd, vb.pp("wv"))?,
            wo: linear_no_bias(cfg.depth_heads * hd, d, vb.pp("wo"))?,
            n_heads: cfg.depth_heads, head_dim: hd,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;
        let q = self.wq.forward(x)?.reshape((b, t, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.wk.forward(x)?.reshape((b, t, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.wv.forward(x)?.reshape((b, t, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let scale = (self.head_dim as f32).powf(-0.5);
        // Causal attention within depth sequence (codebook order)
        let out = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale, 1.0)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, ()))?;
        self.wo.forward(&out)
    }
}

struct DepthBlock {
    attn_norm: RmsNorm, attn: DepthAttention,
    ffn_norm: RmsNorm, ffn: super::temporal::SwiGluFfn,
}

impl DepthBlock {
    fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn_norm: rms_norm(cfg.depth_dim, cfg.norm_eps, vb.pp("attn_norm"))?,
            attn: DepthAttention::load(cfg, vb.pp("attn"))?,
            ffn_norm: rms_norm(cfg.depth_dim, cfg.norm_eps, vb.pp("ffn_norm"))?,
            ffn: super::temporal::SwiGluFfn::load(cfg.depth_dim, cfg.depth_d_ff, vb.pp("ffn"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.attn.forward(&self.attn_norm.forward(x)?)?;
        let x = (x + h)?;
        let h = self.ffn.forward(&self.ffn_norm.forward(&x)?)?;
        x + h
    }
}

pub struct DepthTransformer {
    blocks: Vec<DepthBlock>,
    norm: RmsNorm,
    codebook_heads: Vec<Linear>,  // one per acoustic codebook
    codebook_embs: Vec<candle_nn::Embedding>,  // embeddings for depth AR
    project_in: Linear,  // d_model -> depth_dim
}

impl DepthTransformer {
    pub fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..cfg.n_depth_layers {
            blocks.push(DepthBlock::load(cfg, vb.pp(format!("layer.{}", i)))?);
        }
        let mut codebook_heads = Vec::new();
        let mut codebook_embs = Vec::new();
        let n_acoustic = cfg.n_codebooks - 1; // 7 acoustic codebooks
        for i in 0..n_acoustic {
            codebook_heads.push(linear_no_bias(cfg.depth_dim, cfg.codebook_size, vb.pp(format!("head.{}", i)))?);
            codebook_embs.push(candle_nn::embedding(cfg.codebook_size, cfg.depth_dim, vb.pp(format!("emb.{}", i)))?);
        }
        Ok(Self {
            blocks,
            norm: rms_norm(cfg.depth_dim, cfg.norm_eps, vb.pp("norm"))?,
            codebook_heads,
            codebook_embs,
            project_in: linear_no_bias(cfg.d_model, cfg.depth_dim, vb.pp("project_in"))?,
        })
    }

    /// Forward pass through depth transformer blocks.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        self.norm.forward(&h)
    }

    /// Generate acoustic codes autoregressively given semantic embedding.
    /// Input: temporal_hidden projected to depth_dim (1, 1, depth_dim)
    /// Output: Vec of code indices, one per acoustic codebook
    pub fn generate(&self, semantic_emb: &Tensor, n_acoustic: usize) -> Result<Vec<u32>> {
        let mut h = semantic_emb.clone();
        let mut codes = Vec::new();
        for i in 0..n_acoustic {
            let out = self.forward(&h)?;
            let logits = self.codebook_heads[i].forward(&out)?;
            let code = logits.argmax(D::Minus1)?.squeeze(0)?.squeeze(0)?.to_scalar::<u32>()?;
            codes.push(code);
            if i < n_acoustic - 1 {
                let code_tensor = Tensor::new(&[code], semantic_emb.device())?;
                let emb = self.codebook_embs[i].forward(&code_tensor)?.unsqueeze(0)?;
                h = (h + emb)?;  // residual conditioning
            }
        }
        Ok(codes)
    }
}
```

**Step 3: Run tests**

Run: `cargo test -p sonata-talker -- depth`
Expected: 2 tests PASS

**Step 4: Commit**

```bash
git add src/sonata_talker/src/depth.rs
git commit -m "feat(talker): depth transformer for acoustic codebook generation"
```

---

### Task 4: Audio + Text Embedders

**Files:**

- Create: `src/sonata_talker/src/embedder.rs`

**Step 1: Write test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_audio_embedder_shape() {
        let dev = &Device::Cpu;
        let cfg = TalkerConfig::default();
        let vb = VarBuilder::zeros(DType::F32, dev);
        let emb = AudioEmbedder::load(&cfg, vb.pp("audio_emb")).unwrap();
        // 8 codebook indices for one timestep
        let codes = Tensor::zeros((1, 8), DType::U32, dev).unwrap();
        let out = emb.forward(&codes).unwrap();
        assert_eq!(out.dims(), &[1, 1, 768]); // (batch, 1, d_model)
    }

    #[test]
    fn test_thinker_projector_shape() {
        let dev = &Device::Cpu;
        let cfg = TalkerConfig::default();
        let vb = VarBuilder::zeros(DType::F32, dev);
        let proj = ThinkerProjector::load(&cfg, vb.pp("thinker_proj")).unwrap();
        let hidden = Tensor::zeros((1, 1, 4096), DType::F32, dev).unwrap();
        let out = proj.forward(&hidden).unwrap();
        assert_eq!(out.dims(), &[1, 1, 768]);
    }
}
```

**Step 2: Implement Embedders**

```rust
use crate::TalkerConfig;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, VarBuilder};

/// Embeds 8 codebook indices into a single d_model vector.
/// Each codebook has its own embedding table (2048 x 768).
/// Output = sum of all 8 codebook embeddings.
pub struct AudioEmbedder {
    codebook_embs: Vec<Embedding>,
}

impl AudioEmbedder {
    pub fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let mut codebook_embs = Vec::new();
        for i in 0..cfg.n_codebooks {
            codebook_embs.push(embedding(cfg.codebook_size, cfg.d_model, vb.pp(format!("book.{}", i)))?);
        }
        Ok(Self { codebook_embs })
    }

    /// codes: (batch, n_codebooks) — one set of codes per timestep
    /// Returns: (batch, 1, d_model) — sum of all codebook embeddings
    pub fn forward(&self, codes: &Tensor) -> Result<Tensor> {
        let mut sum = None;
        for (i, emb) in self.codebook_embs.iter().enumerate() {
            let c = codes.narrow(1, i, 1)?.squeeze(1)?; // (batch,)
            let e = emb.forward(&c)?.unsqueeze(1)?;      // (batch, 1, d_model)
            sum = Some(match sum {
                Some(s) => (s + e)?,
                None => e,
            });
        }
        sum.ok_or_else(|| candle_core::Error::Msg("No codebooks".into()))
    }
}

pub struct TextEmbedder {
    emb: Embedding,
}

impl TextEmbedder {
    pub fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            emb: embedding(cfg.text_vocab_size, cfg.d_model, vb.pp("text_emb"))?,
        })
    }

    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.emb.forward(token_ids)
    }
}

/// Projects LLM hidden states (e.g., Claude 4096-dim) to Talker d_model (768).
/// Two-layer MLP with GELU activation.
pub struct ThinkerProjector {
    linear1: Linear,
    linear2: Linear,
}

impl ThinkerProjector {
    pub fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let mid = cfg.d_model; // intermediate = d_model
        Ok(Self {
            linear1: linear_no_bias(cfg.thinker_hidden_dim, mid, vb.pp("linear1"))?,
            linear2: linear_no_bias(mid, cfg.d_model, vb.pp("linear2"))?,
        })
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let h = candle_nn::Activation::Gelu.forward(&self.linear1.forward(hidden)?)?;
        self.linear2.forward(&h)
    }
}
```

**Step 3: Run tests**

Run: `cargo test -p sonata-talker -- embedder`
Expected: 2 tests PASS

**Step 4: Commit**

```bash
git add src/sonata_talker/src/embedder.rs
git commit -m "feat(talker): audio/text embedders + thinker projector"
```

---

### Task 5: Full Talker Engine + C FFI

**Files:**

- Modify: `src/sonata_talker/src/lib.rs`

**Step 1: Write integration test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

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
        assert_eq!(out.len(), 8); // 8 output codebook indices
    }
}
```

**Step 2: Implement TalkerEngine**

Wire together all components: AudioEmbedder, TextEmbedder, ThinkerProjector, TemporalTransformer, DepthTransformer.

```rust
pub struct TalkerEngine {
    cfg: TalkerConfig,
    audio_emb: embedder::AudioEmbedder,
    text_emb: embedder::TextEmbedder,
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
            text_emb: embedder::TextEmbedder::load(cfg, vb.pp("text_emb"))?,
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
```

**Step 3: Add C FFI exports**

```rust
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
```

**Step 4: Run all tests**

Run: `cargo test -p sonata-talker`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/sonata_talker/src/lib.rs
git commit -m "feat(talker): full engine with temporal+depth+embedders+FFI"
```

---

## Workstream 3: Dual-Stream Token Engine

### Task 6: Dual-stream interleaver

**Files:**

- Create: `src/sonata_talker/src/stream.rs`

Implements the Moshi-style dual-stream token interleaving at 12.5Hz:

```
[user_sem_t] [user_a0_t] ... [user_a7_t]  ← listening stream
[asst_sem_t] [asst_a0_t] ... [asst_a7_t]  ← speaking stream
[text_t]                                    ← inner monologue
```

**Step 1: Write test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interleave_tokens() {
        let user_codes = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let asst_codes = [10u32, 20, 30, 40, 50, 60, 70, 80];
        let text_token = Some(100u32);
        let interleaved = DualStream::interleave(&user_codes, &asst_codes, text_token);
        assert_eq!(interleaved.len(), 17); // 8 + 8 + 1
        assert_eq!(interleaved[0], 1);   // user semantic
        assert_eq!(interleaved[8], 10);  // asst semantic
        assert_eq!(interleaved[16], 100); // text token
    }

    #[test]
    fn test_ring_buffer_push_pop() {
        let mut ring = AudioRingBuffer::new(4);
        ring.push(&[1, 2, 3, 4, 5, 6, 7, 8]);
        ring.push(&[10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(ring.len(), 2);
        let oldest = ring.pop().unwrap();
        assert_eq!(oldest, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_acoustic_delay() {
        let mut stream = DualStream::new(2); // tau=2
        // First 2 frames: no output (filling delay buffer)
        let out0 = stream.process_frame(&[0; 8], None);
        assert!(out0.is_none()); // delay not filled
        let out1 = stream.process_frame(&[1; 8], None);
        assert!(out1.is_none()); // delay not filled
        let out2 = stream.process_frame(&[2; 8], None);
        assert!(out2.is_some()); // delay filled, output frame 0
    }
}
```

**Step 2: Implement DualStream**

```rust
/// Ring buffer for audio code frames (pre-allocated, zero-alloc in hot path).
pub struct AudioRingBuffer {
    buf: Vec<[u32; 8]>,
    head: usize,
    tail: usize,
    capacity: usize,
    count: usize,
}

impl AudioRingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: vec![[0u32; 8]; capacity],
            head: 0, tail: 0, capacity, count: 0,
        }
    }

    pub fn push(&mut self, codes: &[u32; 8]) {
        self.buf[self.tail] = *codes;
        self.tail = (self.tail + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        } else {
            self.head = (self.head + 1) % self.capacity;
        }
    }

    pub fn pop(&mut self) -> Option<[u32; 8]> {
        if self.count == 0 { return None; }
        let frame = self.buf[self.head];
        self.head = (self.head + 1) % self.capacity;
        self.count -= 1;
        Some(frame)
    }

    pub fn len(&self) -> usize { self.count }
    pub fn is_full(&self) -> bool { self.count == self.capacity }
}

/// Dual-stream token engine for full-duplex S2S.
/// Handles acoustic delay (tau) and user/assistant stream interleaving.
pub struct DualStream {
    tau: usize,                        // acoustic delay in frames
    user_buffer: AudioRingBuffer,      // delayed user codes
    asst_prev: [u32; 8],              // previous assistant output
    frame_count: usize,
}

impl DualStream {
    pub fn new(tau: usize) -> Self {
        Self {
            tau,
            user_buffer: AudioRingBuffer::new(tau + 1),
            asst_prev: [0u32; 8],
            frame_count: 0,
        }
    }

    /// Interleave user + assistant codes + optional text token.
    pub fn interleave(
        user_codes: &[u32; 8], asst_codes: &[u32; 8], text_token: Option<u32>,
    ) -> Vec<u32> {
        let mut out = Vec::with_capacity(17);
        out.extend_from_slice(user_codes);
        out.extend_from_slice(asst_codes);
        out.push(text_token.unwrap_or(0));
        out
    }

    /// Process one 12.5Hz frame:
    /// - Buffers user codes for acoustic delay
    /// - Returns interleaved tokens when delay is filled
    pub fn process_frame(
        &mut self, user_codes: &[u32; 8], text_token: Option<u32>,
    ) -> Option<Vec<u32>> {
        let mut codes = [0u32; 8];
        codes.copy_from_slice(user_codes);
        self.user_buffer.push(&codes);
        self.frame_count += 1;

        if self.frame_count <= self.tau {
            return None; // delay not yet filled
        }

        let delayed_user = self.user_buffer.pop().unwrap();
        Some(Self::interleave(&delayed_user, &self.asst_prev, text_token))
    }

    /// Update assistant output codes (after Talker generates them).
    pub fn set_assistant_output(&mut self, codes: &[u32; 8]) {
        self.asst_prev.copy_from_slice(codes);
    }

    pub fn reset(&mut self) {
        self.user_buffer = AudioRingBuffer::new(self.tau + 1);
        self.asst_prev = [0u32; 8];
        self.frame_count = 0;
    }
}
```

**Step 3: Run tests**

Run: `cargo test -p sonata-talker -- stream`
Expected: 3 tests PASS

**Step 4: Commit**

```bash
git add src/sonata_talker/src/stream.rs
git commit -m "feat(talker): dual-stream token engine with acoustic delay"
```

---

### Task 7: Latency benchmark

**Files:**

- Create: `src/sonata_talker/benches/talker_latency.rs`
- Modify: `src/sonata_talker/Cargo.toml` (add bench binary)

**Step 1: Add bench binary to Cargo.toml**

```toml
[[bin]]
name = "talker_latency"
path = "benches/talker_latency.rs"
```

**Step 2: Write benchmark**

```rust
//! Talker latency benchmark — measures per-step inference time.
//! Target: <100ms per 12.5Hz step on Apple Silicon Metal.

use candle_core::{DType, Device};
use std::time::Instant;

fn main() {
    let dev = Device::new_metal(0).unwrap_or_else(|_| {
        eprintln!("Metal not available, using CPU");
        Device::Cpu
    });
    println!("Device: {:?}", dev);

    let cfg = sonata_talker::TalkerConfig::default();
    println!("Config: {}M temporal + {}M depth ≈ 512M params",
        340, 72);
    println!("  d_model={}, temporal_layers={}, depth_layers={}",
        cfg.d_model, cfg.n_temporal_layers, cfg.n_depth_layers);

    let mut engine = sonata_talker::TalkerEngine::new_zeros(&cfg, &dev)
        .expect("Failed to create engine");

    // Warmup
    for _ in 0..3 {
        let _ = engine.step(&[0u32; 8], None);
    }
    engine.reset().unwrap();

    // Benchmark
    let n_steps = 100;
    let start = Instant::now();
    for _ in 0..n_steps {
        let _ = engine.step(&[0u32; 8], None).unwrap();
    }
    let elapsed = start.elapsed();
    let per_step_ms = elapsed.as_millis() as f64 / n_steps as f64;
    let frame_ms = 1000.0 / cfg.frame_rate_hz as f64; // 80ms at 12.5Hz

    println!("\n=== Talker Latency ===");
    println!("  Steps: {}", n_steps);
    println!("  Total: {:.1}ms", elapsed.as_millis());
    println!("  Per step: {:.1}ms", per_step_ms);
    println!("  Frame budget: {:.1}ms", frame_ms);
    println!("  RTF: {:.3}", per_step_ms / frame_ms);
    println!("  Target: <100ms (RTF < 1.25)");

    if per_step_ms < 100.0 {
        println!("  ✓ PASS — within latency budget");
    } else {
        println!("  ✗ FAIL — exceeds latency budget");
    }
}
```

**Step 3: Run benchmark**

Run: `cargo run --release -p sonata-talker --bin talker_latency`

**Step 4: Commit**

```bash
git add src/sonata_talker/benches/
git commit -m "bench(talker): latency benchmark targeting <100ms per step"
```

---

## Summary

| Task      | Component               | Files              | Est. LOC |
| --------- | ----------------------- | ------------------ | -------- |
| 1         | Crate scaffold + config | Cargo.toml, lib.rs | 120      |
| 2         | Temporal Transformer    | temporal.rs        | 180      |
| 3         | Depth Transformer       | depth.rs           | 160      |
| 4         | Audio/Text Embedders    | embedder.rs        | 100      |
| 5         | TalkerEngine + FFI      | lib.rs             | 200      |
| 6         | Dual-stream engine      | stream.rs          | 120      |
| 7         | Latency benchmark       | benches/           | 60       |
| **Total** |                         | **7 files**        | **~940** |

All tasks follow existing sonata_lm patterns: GQA, RoPE, KV cache, SwiGLU, cdylib FFI, safetensors loading, Metal GPU.
