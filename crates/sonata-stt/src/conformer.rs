//! Conformer block: convolution-augmented transformer for speech.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, VarBuilder};
use sonata_common::swiglu::SwiGLU;

pub struct ConformerBlock {
    self_attn: Linear,
    conv: Conv1d,
    ffn: SwiGLU,
    norm1: candle_nn::LayerNorm,
    norm2: candle_nn::LayerNorm,
}

impl ConformerBlock {
    pub fn new(dim: usize, ffn_dim: usize, dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);
        let self_attn = candle_nn::linear(dim, dim, vb.pp("attn"))?;
        let conv_cfg = Conv1dConfig {
            padding: 15,
            ..Default::default()
        };
        let conv = candle_nn::conv1d(dim, dim, 31, conv_cfg, vb.pp("conv"))?;
        let ffn = SwiGLU::new(dim, ffn_dim, vb.pp("ffn"))?;
        let norm1 = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let norm2 = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        Ok(Self {
            self_attn,
            conv,
            ffn,
            norm1,
            norm2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, T, D]
        let residual = x.clone();

        // Self-attention with residual connection
        let x = self.norm1.forward(x)?;
        let attn_out = self.self_attn.forward(&x)?;
        let x = (residual.clone() + attn_out)?;

        // Depthwise convolution with residual connection
        let residual = x.clone();
        let x_conv = x.transpose(1, 2)?; // [B, D, T]
        let conv_out = self.conv.forward(&x_conv)?;
        let conv_out = conv_out.relu()?;
        let x = (residual + conv_out.transpose(1, 2)?)?;

        // FFN with residual connection
        let residual = x.clone();
        let x = self.norm2.forward(&x)?;
        let ffn_out = self.ffn.forward(&x)?;
        residual + ffn_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformer_block_shape() {
        let dev = Device::Cpu;
        let block = ConformerBlock::new(512, 2048, &dev).unwrap();
        let x = Tensor::zeros(&[1, 50, 512], DType::F32, &dev).unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 50, 512]);
    }

    #[test]
    fn test_conformer_block_batch_size() {
        let dev = Device::Cpu;
        let block = ConformerBlock::new(512, 2048, &dev).unwrap();

        // Test multiple batch sizes
        for batch_size in &[1, 2, 4, 8] {
            let x = Tensor::zeros(&[*batch_size, 100, 512], DType::F32, &dev).unwrap();
            let out = block.forward(&x).unwrap();
            assert_eq!(out.dims()[0], *batch_size);
            assert_eq!(out.dims()[1], 100);
            assert_eq!(out.dims()[2], 512);
        }
    }

    #[test]
    fn test_conformer_block_sequence_length() {
        let dev = Device::Cpu;
        let block = ConformerBlock::new(512, 2048, &dev).unwrap();

        // Test various sequence lengths
        for seq_len in &[50, 100, 200, 500] {
            let x = Tensor::zeros(&[2, *seq_len, 512], DType::F32, &dev).unwrap();
            let out = block.forward(&x).unwrap();
            assert_eq!(out.dims()[1], *seq_len);
        }
    }

    #[test]
    fn test_conformer_block_small_model() {
        let dev = Device::Cpu;
        let block = ConformerBlock::new(128, 512, &dev).unwrap();
        let x = Tensor::zeros(&[1, 20, 128], DType::F32, &dev).unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 20, 128]);
    }
}
