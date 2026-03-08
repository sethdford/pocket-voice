use crate::TalkerConfig;
use candle_core::{DType, Result, Tensor};
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

    #[test]
    fn test_text_embedder_shape() {
        let dev = &Device::Cpu;
        let cfg = TalkerConfig::default();
        let vb = VarBuilder::zeros(DType::F32, dev);
        let emb = TextEmbedder::load(&cfg, vb.pp("text_emb")).unwrap();
        let token_ids = Tensor::zeros((1, 1), DType::U32, dev).unwrap();
        let out = emb.forward(&token_ids).unwrap();
        assert_eq!(out.dims(), &[1, 1, 768]); // (batch, 1, d_model)
    }
}
