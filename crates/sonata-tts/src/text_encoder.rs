//! Text encoder: embedding + positional encoding for TTS input.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};

pub const TEXT_VOCAB_SIZE: usize = 32000;

pub struct TextEncoder {
    token_embed: Embedding,
    #[allow(dead_code)]
    dim: usize,
}

impl TextEncoder {
    pub fn new(dim: usize, dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);
        let token_embed = candle_nn::embedding(TEXT_VOCAB_SIZE, dim, vb.pp("embed"))?;
        Ok(Self { token_embed, dim })
    }

    /// Encode text token IDs to embeddings.
    /// Input: [B, T] u32 token IDs
    /// Output: [B, T, dim] embeddings
    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        self.token_embed.forward(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_text_encoder_shape() {
        let dev = Device::Cpu;
        let enc = TextEncoder::new(512, &dev).unwrap();
        let tokens = Tensor::zeros(&[1, 20], DType::U32, &dev).unwrap();
        let out = enc.forward(&tokens).unwrap();
        assert_eq!(out.dims(), &[1, 20, 512]);
    }
}
