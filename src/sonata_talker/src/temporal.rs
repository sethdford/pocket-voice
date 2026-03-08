use crate::TalkerConfig;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Linear, Module, RmsNorm, VarBuilder};

/// Repeat KV heads for GQA (group query attention with KV head repetition for scaling)
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 { return Ok(x.clone()); }
    let (b, h, t, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h, n_rep, t, d))?
        .reshape((b, h * n_rep, t, d))
}

// ─── RoPE (Rotary Position Embedding) ─────────────────────────────────────

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

// ─── Grouped Query Attention ──────────────────────────────────────────────

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

        // Try SDPA (Metal GPU), fall back to manual attention (CPU testing)
        let out = if x.device().is_cpu() {
            // Manual attention for CPU testing: repeat KV heads, compute scores, softmax, output
            let k_exp = repeat_kv(&k_full, self.n_heads / self.n_kv_heads)?;
            let v_exp = repeat_kv(&v_full, self.n_heads / self.n_kv_heads)?;
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let scores = (q.matmul(&k_exp.t()?)? * scale)?;
            let attn = candle_nn::ops::softmax_last_dim(&scores)?;
            attn.matmul(&v_exp)?
        } else {
            // GPU: use fused SDPA Metal kernel — handles GQA natively
            let scale = (self.head_dim as f32).powf(-0.5);
            candle_nn::ops::sdpa(&q, &k_full, &v_full, None, false, scale, 1.0)?
        };

        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, 1, ()))?;
        self.wo.forward(&out)
    }
}

// ─── SwiGLU Feed-Forward Network ──────────────────────────────────────────

pub struct SwiGluFfn { pub w_gate: Linear, pub w_up: Linear, pub w_down: Linear }

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

// ─── Temporal Transformer Block ───────────────────────────────────────────

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

// ─── Temporal Transformer ─────────────────────────────────────────────────

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

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_temporal_block_residual() {
        let dev = &Device::Cpu;
        let cfg = TalkerConfig::default();
        let vb = VarBuilder::zeros(DType::F32, dev);
        let block = TemporalBlock::load(&cfg, vb.pp("block")).unwrap();
        let x = Tensor::zeros((1, 1, 768), DType::F32, dev).unwrap();
        let (cos, sin) = precompute_rope_cache(64, 128, 10000.0, dev).unwrap();
        let mut kc = Tensor::zeros((1, 4, 0, 64), DType::F32, dev).unwrap();
        let mut vc = Tensor::zeros((1, 4, 0, 64), DType::F32, dev).unwrap();
        let out = block.forward(&x, &cos, &sin, 0, &mut kc, &mut vc).unwrap();
        assert_eq!(out.dims(), &[1, 1, 768]);
    }

    #[test]
    fn test_temporal_transformer_forward() {
        let dev = &Device::Cpu;
        let cfg = TalkerConfig::default();
        let vb = VarBuilder::zeros(DType::F32, dev);
        let transformer = TemporalTransformer::load(&cfg, vb.pp("temporal")).unwrap();
        let x = Tensor::zeros((1, 1, 768), DType::F32, dev).unwrap();
        let (cos, sin) = precompute_rope_cache(64, 128, 10000.0, dev).unwrap();
        let mut k_caches = vec![Tensor::zeros((1, 4, 0, 64), DType::F32, dev).unwrap(); cfg.n_temporal_layers];
        let mut v_caches = vec![Tensor::zeros((1, 4, 0, 64), DType::F32, dev).unwrap(); cfg.n_temporal_layers];
        let out = transformer.forward(&x, &cos, &sin, 0, &mut k_caches, &mut v_caches).unwrap();
        assert_eq!(out.dims(), &[1, 1, 768]);
    }
}
