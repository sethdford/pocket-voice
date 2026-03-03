//! Text encoder: embedding + sinusoidal positional encoding for TTS input.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};

pub const TEXT_VOCAB_SIZE: usize = 32000;
const MAX_SEQ_LEN: usize = 2048;

pub struct TextEncoder {
    token_embed: Embedding,
    pos_encoding: Tensor,
    dim: usize,
}

impl TextEncoder {
    /// Generate sinusoidal positional encoding table.
    ///
    /// PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
    /// PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
    ///
    /// Returns shape: [1, max_len, dim]
    fn sinusoidal_positional_encoding(
        dim: usize,
        max_len: usize,
        dev: &Device,
    ) -> Result<Tensor> {
        let mut pe = vec![0.0f32; max_len * dim];
        let half_dim = dim / 2;
        for pos in 0..max_len {
            for i in 0..half_dim {
                let angle =
                    pos as f64 / (10000.0_f64).powf(2.0 * i as f64 / dim as f64);
                pe[pos * dim + 2 * i] = angle.sin() as f32;
                pe[pos * dim + 2 * i + 1] = angle.cos() as f32;
            }
        }
        Tensor::from_vec(pe, (1, max_len, dim), dev)
    }

    pub fn new(dim: usize, dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);
        let token_embed = candle_nn::embedding(TEXT_VOCAB_SIZE, dim, vb.pp("embed"))?;
        let pos_encoding = Self::sinusoidal_positional_encoding(dim, MAX_SEQ_LEN, dev)?;
        Ok(Self {
            token_embed,
            pos_encoding,
            dim,
        })
    }

    /// Encode text token IDs to embeddings with positional encoding.
    /// Input: [B, T] u32 token IDs
    /// Output: [B, T, dim] embeddings + positional encoding
    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let embeddings = self.token_embed.forward(tokens)?;
        let seq_len = tokens.dim(1)?;
        let pos = self.pos_encoding.narrow(1, 0, seq_len)?;
        embeddings.broadcast_add(&pos)
    }

    /// Get the embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
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

    #[test]
    fn test_text_encoder_positional_encoding_applied() {
        let dev = Device::Cpu;
        let enc = TextEncoder::new(512, &dev).unwrap();

        // With zero-initialized embeddings, output should be the positional encoding itself
        let tokens = Tensor::zeros(&[1, 10], DType::U32, &dev).unwrap();
        let out = enc.forward(&tokens).unwrap();

        // Position 0 and position 1 should differ (positional encoding)
        let pos0 = out.narrow(1, 0, 1).unwrap();
        let pos1 = out.narrow(1, 1, 1).unwrap();
        let diff: f32 = (&pos0 - &pos1)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff > 0.0, "Different positions should have different encodings");
    }

    #[test]
    fn test_text_encoder_batch() {
        let dev = Device::Cpu;
        let enc = TextEncoder::new(256, &dev).unwrap();
        let tokens = Tensor::zeros(&[4, 15], DType::U32, &dev).unwrap();
        let out = enc.forward(&tokens).unwrap();
        assert_eq!(out.dims(), &[4, 15, 256]);
    }

    #[test]
    fn test_text_encoder_various_lengths() {
        let dev = Device::Cpu;
        let enc = TextEncoder::new(512, &dev).unwrap();

        for seq_len in &[1, 10, 50, 100] {
            let tokens = Tensor::zeros(&[1, *seq_len], DType::U32, &dev).unwrap();
            let out = enc.forward(&tokens).unwrap();
            assert_eq!(out.dims(), &[1, *seq_len, 512]);
        }
    }
}
