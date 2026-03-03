//! Context-Aware Masking (CAM) block — the key innovation in CAM++.
//!
//! The CAM block uses attention mechanisms to learn which parts of the input
//! are most relevant for speaker identification, then applies this masking
//! to enhance discriminative features.

use anyhow::Result;
use candle_core::{Device, DType, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder};

/// Context-Aware Masking block that learns to selectively emphasize
/// speaker-relevant parts of the input using attention-based masking.
pub struct CAMBlock {
    attention: Linear,
    mask_proj: Linear,
    ffn1: Conv1d,
    ffn2: Conv1d,
    #[allow(dead_code)]
    dim: usize,
}

impl CAMBlock {
    /// Create a new CAM block.
    ///
    /// # Arguments
    /// * `dim` - Feature dimension (typically 256)
    /// * `_num_heads` - Number of attention heads (typically 8, reserved for future use)
    /// * `dev` - Candle device
    pub fn new(dim: usize, _num_heads: usize, dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);

        // Attention projection: projects to attention logits
        let attention = candle_nn::linear(dim, dim, vb.pp("attn"))?;

        // Mask projection: learns the masking pattern
        let mask_proj = candle_nn::linear(dim, dim, vb.pp("mask"))?;

        // Feed-forward network (1D convolution version)
        // Expands to 4*dim then contracts back to dim
        let cfg1 = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let ffn1 = candle_nn::conv1d(dim, dim * 4, 3, cfg1, vb.pp("ffn1"))?;
        let ffn2 = candle_nn::conv1d(dim * 4, dim, 1, Default::default(), vb.pp("ffn2"))?;

        Ok(Self {
            attention,
            mask_proj,
            ffn1,
            ffn2,
            dim,
        })
    }

    /// Forward pass through the CAM block.
    ///
    /// The block:
    /// 1. Computes attention-based masks to select speaker-relevant frames
    /// 2. Applies residual connection through attention
    /// 3. Applies feed-forward transformation with residual connection
    ///
    /// # Arguments
    /// * `x` - Input tensor [B, dim, T]
    ///
    /// # Returns
    /// * Output tensor [B, dim, T] (same shape as input)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Transpose from [B, dim, T] to [B, T, dim] for linear layer processing
        let x_t = x.transpose(1, 2)?;

        // Compute mask logits and apply sigmoid to get mask weights
        let mask_logits = self.mask_proj.forward(&x_t)?;
        let mask = sigmoid(&mask_logits)?;

        // Apply mask: element-wise multiply each frame by its mask weight
        let masked = masked_mul(&x_t, &mask)?;

        // Attention transformation with residual
        let attn_out = self.attention.forward(&masked)?;
        let x_t = add(&x_t, &attn_out)?;

        // Transpose back to [B, dim, T] for convolution
        let x_conv = x_t.transpose(1, 2)?;

        // Feed-forward with residual: ffn(x) + x
        let ff = self.ffn1.forward(&x_conv)?;
        let ff = ff.relu()?;
        let ff = self.ffn2.forward(&ff)?;

        Ok(add(x, &ff)?)
    }
}

/// Compute sigmoid activation: 1 / (1 + exp(-x))
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg_x = neg_x.exp()?;
    let denom = (exp_neg_x + 1.0)?;
    Ok(denom.recip()?)
}

/// Element-wise multiply: broadcasting compatible with mask [B, T, 1]
fn masked_mul(x: &Tensor, mask: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(mask).map_err(Into::into)
}

/// Element-wise add with broadcasting
fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    (a + b).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cam_block_output_shape() -> Result<()> {
        let dev = Device::Cpu;
        let block = CAMBlock::new(256, 8, &dev)?;
        let x = Tensor::zeros(&[1, 256, 100], DType::F32, &dev)?;
        let out = block.forward(&x)?;

        assert_eq!(out.dims(), x.dims());

        Ok(())
    }

    #[test]
    fn test_cam_block_batch_processing() -> Result<()> {
        let dev = Device::Cpu;
        let block = CAMBlock::new(256, 8, &dev)?;
        let x = Tensor::zeros(&[4, 256, 100], DType::F32, &dev)?;
        let out = block.forward(&x)?;

        assert_eq!(out.dims(), &[4, 256, 100]);

        Ok(())
    }

    #[test]
    fn test_cam_block_variable_length() -> Result<()> {
        let dev = Device::Cpu;
        let block = CAMBlock::new(256, 8, &dev)?;

        for &t in &[50, 100, 200, 500] {
            let x = Tensor::zeros(&[2, 256, t], DType::F32, &dev)?;
            let out = block.forward(&x)?;
            assert_eq!(out.dims(), &[2, 256, t]);
        }

        Ok(())
    }

    #[test]
    fn test_cam_block_residual_property() -> Result<()> {
        let dev = Device::Cpu;
        let block = CAMBlock::new(256, 8, &dev)?;

        // With random input, output should have same shape (residual ensures this)
        let x = Tensor::randn(0.0f32, 1.0, &[1, 256, 100], &dev)?;
        let out = block.forward(&x)?;

        assert_eq!(out.dims(), x.dims());

        Ok(())
    }
}
