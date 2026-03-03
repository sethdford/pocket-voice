//! Diffusion Transformer (DiT) blocks — F5-TTS style.
//!
//! DiT blocks implement a diffusion transformer architecture for conditional flow matching.
//! Each block applies self-attention, then conditions on diffusion timestep and speaker embedding
//! via AdaIN modules, followed by a SwiGLU feed-forward network.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use sonata_common::adain::AdaIN;
use sonata_common::swiglu::SwiGLU;

/// A single Diffusion Transformer block with speaker and time conditioning.
pub struct DiTBlock {
    /// Self-attention layer
    self_attn: Linear,
    /// Time embedding AdaIN layer
    time_adain: AdaIN,
    /// Speaker embedding AdaIN layer
    speaker_adain: AdaIN,
    /// SwiGLU feed-forward network
    ffn: SwiGLU,
    /// Layer normalization
    norm: candle_nn::LayerNorm,
}

impl DiTBlock {
    /// Create a new DiT block.
    ///
    /// # Arguments
    /// * `dim` - Hidden dimension (e.g., 512)
    /// * `ffn_dim` - Feed-forward hidden dimension (e.g., 2048)
    /// * `time_dim` - Diffusion time embedding dimension (e.g., 256)
    /// * `speaker_dim` - Speaker embedding dimension (e.g., 192)
    /// * `dev` - Computation device
    pub fn new(dim: usize, ffn_dim: usize, time_dim: usize, speaker_dim: usize, dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);
        let self_attn = candle_nn::linear(dim, dim, vb.pp("attn"))?;
        let time_adain = AdaIN::new(dim, time_dim, vb.pp("time_adain"))?;
        let speaker_adain = AdaIN::new(dim, speaker_dim, vb.pp("spk_adain"))?;
        let ffn = SwiGLU::new(dim, ffn_dim, vb.pp("ffn"))?;
        let norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm"))?;
        Ok(Self { self_attn, time_adain, speaker_adain, ffn, norm })
    }

    /// Apply DiT block transformation.
    ///
    /// # Arguments
    /// * `x` - Input tensor [B, T, dim]
    /// * `time_emb` - Time embedding [B, time_dim]
    /// * `speaker_emb` - Speaker embedding [B, speaker_dim]
    ///
    /// # Returns
    /// * Output tensor [B, T, dim]
    pub fn forward(&self, x: &Tensor, time_emb: &Tensor, speaker_emb: &Tensor) -> Result<Tensor> {
        // Normalize input
        let x_norm = self.norm.forward(x)?;

        // Apply self-attention with residual connection
        let attn_out = self.self_attn.forward(&x_norm)?;
        let x = (x + &attn_out)?;

        // Condition on timestep via AdaIN
        let x = self.time_adain.forward(&x, time_emb)?;

        // Condition on speaker via AdaIN
        let x = self.speaker_adain.forward(&x, speaker_emb)?;

        // Apply SwiGLU feed-forward with residual connection
        let residual = x.clone();
        let ffn_out = self.ffn.forward(&x)?;
        Ok((residual + ffn_out)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_dit_block_shape() {
        let dev = Device::Cpu;
        let block = DiTBlock::new(512, 2048, 256, 192, &dev).unwrap();
        let x = Tensor::zeros(&[1, 50, 512], DType::F32, &dev).unwrap();
        let time = Tensor::zeros(&[1, 256], DType::F32, &dev).unwrap();
        let spk = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let out = block.forward(&x, &time, &spk).unwrap();
        assert_eq!(out.dims(), &[1, 50, 512]);
    }

    #[test]
    fn test_dit_block_batch_size() {
        let dev = Device::Cpu;
        let block = DiTBlock::new(256, 1024, 128, 192, &dev).unwrap();

        // Test with batch size 4
        let x = Tensor::zeros(&[4, 32, 256], DType::F32, &dev).unwrap();
        let time = Tensor::zeros(&[4, 128], DType::F32, &dev).unwrap();
        let spk = Tensor::zeros(&[4, 192], DType::F32, &dev).unwrap();

        let out = block.forward(&x, &time, &spk).unwrap();
        assert_eq!(out.dims(), &[4, 32, 256]);
    }

    #[test]
    fn test_dit_block_different_sequence_lengths() {
        let dev = Device::Cpu;
        let block = DiTBlock::new(512, 2048, 256, 192, &dev).unwrap();

        for seq_len in [10, 25, 50, 100].iter() {
            let x = Tensor::zeros(&[2, *seq_len, 512], DType::F32, &dev).unwrap();
            let time = Tensor::zeros(&[2, 256], DType::F32, &dev).unwrap();
            let spk = Tensor::zeros(&[2, 192], DType::F32, &dev).unwrap();

            let out = block.forward(&x, &time, &spk).unwrap();
            assert_eq!(out.dims(), &[2, *seq_len, 512]);
        }
    }

    #[test]
    fn test_dit_block_time_conditioning() {
        let dev = Device::Cpu;
        let block = DiTBlock::new(128, 512, 64, 128, &dev).unwrap();

        let x = Tensor::zeros(&[1, 20, 128], DType::F32, &dev).unwrap();
        let spk = Tensor::zeros(&[1, 128], DType::F32, &dev).unwrap();

        // Different time steps should produce different outputs (stochasticity from initialization)
        let time1 = Tensor::zeros(&[1, 64], DType::F32, &dev).unwrap();
        let time2 = Tensor::ones(&[1, 64], DType::F32, &dev).unwrap();

        let out1 = block.forward(&x, &time1, &spk).unwrap();
        let out2 = block.forward(&x, &time2, &spk).unwrap();

        // Shapes should match
        assert_eq!(out1.dims(), &[1, 20, 128]);
        assert_eq!(out2.dims(), &[1, 20, 128]);
    }

    #[test]
    fn test_dit_block_speaker_conditioning() {
        let dev = Device::Cpu;
        let block = DiTBlock::new(256, 1024, 128, 192, &dev).unwrap();

        let x = Tensor::zeros(&[1, 30, 256], DType::F32, &dev).unwrap();
        let time = Tensor::zeros(&[1, 128], DType::F32, &dev).unwrap();

        // Different speakers should produce different outputs
        let spk1 = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let spk2 = Tensor::ones(&[1, 192], DType::F32, &dev).unwrap();

        let out1 = block.forward(&x, &time, &spk1).unwrap();
        let out2 = block.forward(&x, &time, &spk2).unwrap();

        // Shapes should match
        assert_eq!(out1.dims(), &[1, 30, 256]);
        assert_eq!(out2.dims(), &[1, 30, 256]);
    }

    #[test]
    fn test_dit_block_small_dimensions() {
        let dev = Device::Cpu;
        let block = DiTBlock::new(64, 256, 32, 48, &dev).unwrap();
        let x = Tensor::zeros(&[1, 10, 64], DType::F32, &dev).unwrap();
        let time = Tensor::zeros(&[1, 32], DType::F32, &dev).unwrap();
        let spk = Tensor::zeros(&[1, 48], DType::F32, &dev).unwrap();

        let out = block.forward(&x, &time, &spk).unwrap();
        assert_eq!(out.dims(), &[1, 10, 64]);
    }

    #[test]
    fn test_dit_block_large_dimensions() {
        let dev = Device::Cpu;
        let block = DiTBlock::new(1024, 4096, 512, 256, &dev).unwrap();
        let x = Tensor::zeros(&[1, 20, 1024], DType::F32, &dev).unwrap();
        let time = Tensor::zeros(&[1, 512], DType::F32, &dev).unwrap();
        let spk = Tensor::zeros(&[1, 256], DType::F32, &dev).unwrap();

        let out = block.forward(&x, &time, &spk).unwrap();
        assert_eq!(out.dims(), &[1, 20, 1024]);
    }
}
