use crate::{TalkerConfig, temporal::SwiGluFfn};
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Linear, Module, RmsNorm, VarBuilder};

/// Standard multi-head attention for depth codebook generation.
/// No GQA (no key/value grouping) — full attention.
/// No RoPE — sequences are very short (max 8 codebooks).
struct DepthAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    n_heads: usize,
    head_dim: usize,
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
            n_heads: cfg.depth_heads,
            head_dim: hd,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;
        let q = self
            .wq
            .forward(x)?
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;  // (b, n_heads, t, head_dim)
        let k = self
            .wk
            .forward(x)?
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;  // (b, n_heads, t, head_dim)
        let v = self
            .wv
            .forward(x)?
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;  // (b, n_heads, t, head_dim)

        // Manual scaled dot-product attention (works on CPU and GPU)
        let scale = (self.head_dim as f32).powf(-0.5);
        let mut scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        scores = scores.broadcast_mul(&Tensor::new(&[scale], q.device())?)?;

        // Apply causal mask: set scores to -inf for future positions
        // Create lower triangular mask
        let mut mask_data = vec![0f32; t * t];
        for i in 0..t {
            for j in 0..t {
                if j > i {
                    mask_data[i * t + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask_data, (t, t), q.device())?;
        scores = scores.broadcast_add(&mask)?;

        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn_weights.matmul(&v)?;  // (b, n_heads, t, head_dim)
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, ()))?;  // (b, t, n_heads*head_dim)
        self.wo.forward(&out)
    }
}

/// Depth transformer block: attn_norm -> attn -> residual -> ffn_norm -> ffn -> residual
struct DepthBlock {
    attn_norm: RmsNorm,
    attn: DepthAttention,
    ffn_norm: RmsNorm,
    ffn: SwiGluFfn,
}

impl DepthBlock {
    fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn_norm: rms_norm(cfg.depth_dim, cfg.norm_eps, vb.pp("attn_norm"))?,
            attn: DepthAttention::load(cfg, vb.pp("attn"))?,
            ffn_norm: rms_norm(cfg.depth_dim, cfg.norm_eps, vb.pp("ffn_norm"))?,
            ffn: SwiGluFfn::load(cfg.depth_dim, cfg.depth_d_ff, vb.pp("ffn"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.attn.forward(&self.attn_norm.forward(x)?)?;
        let x = (x + h)?;
        let h = self.ffn.forward(&self.ffn_norm.forward(&x)?)?;
        x + h
    }
}

/// Depth Transformer: Generates acoustic codes (codebooks 1-7) from semantic code.
/// Input: semantic embedding (1, 1, depth_dim)
/// Output: 7 acoustic code indices (via generate method)
pub struct DepthTransformer {
    blocks: Vec<DepthBlock>,
    norm: RmsNorm,
    codebook_heads: Vec<Linear>,  // one linear head per acoustic codebook
    codebook_embs: Vec<candle_nn::Embedding>,  // embeddings for autoregressive conditioning
    pub project_in: Linear,  // d_model -> depth_dim (for projecting temporal output)
}

impl DepthTransformer {
    pub fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..cfg.n_depth_layers {
            blocks.push(DepthBlock::load(cfg, vb.pp(format!("layer.{}", i)))?);
        }

        // 7 acoustic codebooks (codebook 0 is semantic, handled separately)
        let mut codebook_heads = Vec::new();
        let mut codebook_embs = Vec::new();
        let n_acoustic = cfg.n_codebooks - 1;
        for i in 0..n_acoustic {
            codebook_heads
                .push(linear_no_bias(cfg.depth_dim, cfg.codebook_size, vb.pp(format!("head.{}", i)))?);
            codebook_embs.push(candle_nn::embedding(
                cfg.codebook_size,
                cfg.depth_dim,
                vb.pp(format!("emb.{}", i)),
            )?);
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
    /// Input: (batch, seq_len, depth_dim)
    /// Output: (batch, seq_len, depth_dim)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        self.norm.forward(&h)
    }

    /// Generate acoustic codes in parallel (one forward pass, all codebooks at once).
    /// Input: semantic embedding (1, 1, depth_dim) — output from project_in
    /// Output: Vec of n_acoustic code indices
    ///
    /// Unlike sequential AR, this predicts all 7 acoustic codebooks from a single
    /// Depth Transformer forward pass. Matches Moshi/Qwen3-TTS parallel approach.
    /// Latency: 1 forward pass (~2ms) instead of 7 sequential passes (~14ms).
    pub fn generate(&self, semantic_emb: &Tensor, n_acoustic: usize) -> Result<Vec<u32>> {
        let out = self.forward(semantic_emb)?;
        let mut codes = Vec::with_capacity(n_acoustic);
        for i in 0..n_acoustic {
            let logits = self.codebook_heads[i].forward(&out)?;
            let code = logits
                .argmax(D::Minus1)?
                .squeeze(0)?
                .squeeze(0)?
                .to_scalar::<u32>()?;
            codes.push(code);
        }
        Ok(codes)
    }

    /// Quantize all Linear layers in depth transformer to INT4.
    /// Note: This is a placeholder for future per-layer quantization integration.
    pub fn quantize_layers(&mut self) -> Result<()> {
        // TODO: Extract weights from all Linear layers (wq, wk, wv, wo, w_gate, w_up, w_down,
        // codebook_heads, project_in) and convert to QuantizedLinearInt4
        println!("Depth transformer quantization: placeholder");
        Ok(())
    }

    /// Estimate memory footprint in bytes.
    pub fn estimate_memory_bytes(&self) -> usize {
        // Rough estimate: 6 blocks x (3 Linear for attention + 3 for FFN x avg 50KB)
        // + 7 codebook heads x 50KB + project_in x 50KB
        self.blocks.len() * 300_000 + self.codebook_heads.len() * 50_000 + 50_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_depth_transformer_output_shape() {
        let dev = &Device::Cpu;
        let cfg = TalkerConfig::default();
        let vb = candle_nn::VarBuilder::zeros(DType::F32, dev);
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
        let vb = candle_nn::VarBuilder::zeros(DType::F32, dev);
        let depth = DepthTransformer::load(&cfg, vb.pp("depth")).unwrap();
        let semantic_emb = Tensor::zeros((1, 1, 512), DType::F32, dev).unwrap();
        // Generate 7 acoustic codes autoregressively
        let codes = depth.generate(&semantic_emb, 7).unwrap();
        assert_eq!(codes.len(), 7);
    }

    #[test]
    fn test_depth_project_in_dims() {
        let dev = &Device::Cpu;
        let cfg = TalkerConfig::default();
        let vb = candle_nn::VarBuilder::zeros(DType::F32, dev);
        let depth = DepthTransformer::load(&cfg, vb.pp("depth")).unwrap();
        // Project from d_model to depth_dim
        let d_model_input = Tensor::zeros((1, 1, 768), DType::F32, dev).unwrap();
        let depth_input = depth.project_in.forward(&d_model_input).unwrap();
        assert_eq!(depth_input.dims(), &[1, 1, 512]);
    }
}
