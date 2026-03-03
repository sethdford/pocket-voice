//! Adaptive Instance Normalization (AdaIN) module for Kokoro-style voice conditioning.
//!
//! AdaIN allows a single neural network to produce different voices/emotions by conditioning
//! hidden representations with a style embedding (speaker identity or emotion).
//!
//! Formula: `AdaIN(x, style) = gamma(style) * normalize(x) + beta(style)`
//!
//! Where:
//! - `x` is the hidden representation
//! - `style` is the speaker/emotion embedding
//! - `normalize(x)` applies instance normalization (zero mean, unit variance per instance)
//! - `gamma` and `beta` are learned linear projections from the style embedding

use candle_core::{Result, Tensor};
use candle_nn::{Linear, VarBuilder, Module};

const EPSILON: f32 = 1e-5;

/// Adaptive Instance Normalization layer.
///
/// Takes a hidden representation and a style embedding, applies instance normalization
/// to the hidden representation, then scales and shifts using affine parameters derived
/// from the style embedding.
pub struct AdaIN {
    /// Linear layer to project style -> gamma (scale)
    gamma_proj: Linear,
    /// Linear layer to project style -> beta (shift)
    beta_proj: Linear,
    /// Dimension of hidden representation
    #[allow(dead_code)]
    hidden_dim: usize,
}

impl AdaIN {
    /// Create a new AdaIN module.
    ///
    /// # Arguments
    /// * `hidden_dim` - Dimension of the hidden representation to be normalized
    /// * `style_dim` - Dimension of the style/speaker embedding
    /// * `vb` - Variable builder for initializing weights
    pub fn new(hidden_dim: usize, style_dim: usize, vb: VarBuilder) -> Result<Self> {
        let gamma_proj = candle_nn::linear(style_dim, hidden_dim, vb.pp("gamma_proj"))?;
        let beta_proj = candle_nn::linear(style_dim, hidden_dim, vb.pp("beta_proj"))?;

        Ok(Self {
            gamma_proj,
            beta_proj,
            hidden_dim,
        })
    }

    /// Apply adaptive instance normalization.
    ///
    /// # Arguments
    /// * `x` - Hidden representation of shape `[batch, seq_len, hidden_dim]`
    /// * `style` - Style/speaker embedding of shape `[batch, style_dim]`
    ///
    /// # Returns
    /// * Normalized and conditioned output of same shape as `x`
    pub fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        // Instance normalization: normalize across the last dimension (hidden_dim)
        // Shape of x: [batch, seq_len, hidden_dim]
        let x_mean = x.mean_keepdim(candle_core::D::Minus1)?; // [batch, seq_len, 1]

        // Broadcast x_mean to match x's shape for subtraction
        let x_mean_bcast = x_mean.broadcast_as(x.shape())?; // [batch, seq_len, hidden_dim]
        let x_centered = (x - &x_mean_bcast)?; // [batch, seq_len, hidden_dim]

        // Compute variance: mean of squared differences
        let x_var = (&x_centered * &x_centered)?.mean_keepdim(candle_core::D::Minus1)?; // [batch, seq_len, 1]

        // Add epsilon and take square root to get standard deviation
        // Add a small constant to variance for numerical stability
        let epsilon_tensor = Tensor::new(&[[EPSILON]], x.device())?;
        let epsilon_bcast = epsilon_tensor.broadcast_as(x_var.shape())?;
        let x_var_safe = (&x_var + &epsilon_bcast)?.sqrt()?; // [batch, seq_len, 1]
        let x_std = x_var_safe.broadcast_as(x.shape())?; // [batch, seq_len, hidden_dim]

        // Normalize
        let x_norm = (&x_centered / &x_std)?; // [batch, seq_len, hidden_dim]

        // Project style to gamma and beta
        // style shape: [batch, style_dim]
        // gamma/beta shape after projection: [batch, hidden_dim]
        let gamma = self.gamma_proj.forward(style)?; // [batch, hidden_dim]
        let beta = self.beta_proj.forward(style)?; // [batch, hidden_dim]

        // Reshape gamma and beta for broadcasting
        // Need to match x_norm shape [batch, seq_len, hidden_dim]
        // gamma and beta are [batch, hidden_dim], so we unsqueeze to [batch, 1, hidden_dim]
        let gamma = gamma.unsqueeze(1)?; // [batch, 1, hidden_dim]
        let beta = beta.unsqueeze(1)?; // [batch, 1, hidden_dim]

        // Broadcast gamma and beta to match x_norm's shape
        let gamma_bcast = gamma.broadcast_as(x.shape())?; // [batch, seq_len, hidden_dim]
        let beta_bcast = beta.broadcast_as(x.shape())?; // [batch, seq_len, hidden_dim]

        // Apply affine transformation: gamma * x_norm + beta
        let scaled = (x_norm * &gamma_bcast)?; // [batch, seq_len, hidden_dim]
        let output = (scaled + &beta_bcast)?; // [batch, seq_len, hidden_dim]

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_adain_output_shape() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let adain = AdaIN::new(256, 192, vb).unwrap();

        let x = Tensor::zeros(&[1, 10, 256], DType::F32, &dev).unwrap();
        let style = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();

        let out = adain.forward(&x, &style).unwrap();
        assert_eq!(out.dims(), &[1, 10, 256]);
    }

    #[test]
    fn test_adain_batch_shapes() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let adain = AdaIN::new(512, 256, vb).unwrap();

        // Test with larger batch size and longer sequence
        let x = Tensor::zeros(&[4, 20, 512], DType::F32, &dev).unwrap();
        let style = Tensor::zeros(&[4, 256], DType::F32, &dev).unwrap();

        let out = adain.forward(&x, &style).unwrap();
        assert_eq!(out.dims(), &[4, 20, 512]);
    }

    #[test]
    fn test_adain_normalizes() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let adain = AdaIN::new(4, 4, vb).unwrap();

        // Create input with known values to test normalization
        let x_data = [[1.0f32, 2.0, 3.0, 4.0]];
        let x = Tensor::new(&x_data, &dev)
            .unwrap()
            .reshape((1, 1, 4))
            .unwrap();

        let style = Tensor::zeros(&[1, 4], DType::F32, &dev).unwrap();

        let out = adain.forward(&x, &style).unwrap();

        // With zero-init weights (gamma=0, beta=0), output should be zero
        let sum: f32 = out.sum_all().unwrap().to_scalar().unwrap();
        assert!(sum.abs() < 1e-5, "Expected near-zero sum with zero-init weights, got {}", sum);
    }

    #[test]
    fn test_adain_different_sequence_lengths() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let adain = AdaIN::new(128, 64, vb).unwrap();

        // Test with different sequence lengths
        for seq_len in [1, 5, 10, 20, 50].iter() {
            let x = Tensor::zeros(&[2, *seq_len, 128], DType::F32, &dev).unwrap();
            let style = Tensor::zeros(&[2, 64], DType::F32, &dev).unwrap();

            let out = adain.forward(&x, &style).unwrap();
            assert_eq!(out.dims(), &[2, *seq_len, 128]);
        }
    }
}
