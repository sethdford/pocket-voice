//! Attentive Statistics Pooling for speaker embedding extraction.
//!
//! This module implements the pooling mechanism that converts variable-length
//! speaker feature sequences into fixed-size speaker embeddings (192-dim).
//! The attention mechanism learns to weight frames by their relevance for
//! speaker identification.

use anyhow::Result;
use candle_core::{Device, DType, Module, Tensor, D};
use candle_nn::{Linear, VarBuilder};

/// Attentive Statistics Pooling layer.
///
/// This layer:
/// 1. Computes attention weights for each frame (what frames matter for speaker ID?)
/// 2. Computes weighted mean and variance of the features
/// 3. Concatenates mean and std as fixed-size speaker embedding
pub struct AttentiveStatsPooling {
    attention: Linear,
    output_proj: Linear,
}

impl AttentiveStatsPooling {
    /// Create a new attentive statistics pooling layer.
    ///
    /// # Arguments
    /// * `in_dim` - Dimension of input features (typically 256)
    /// * `embed_dim` - Dimension of output embedding (192 for speaker embedding)
    /// * `dev` - Candle device
    pub fn new(in_dim: usize, embed_dim: usize, dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);

        // Attention layer: learns to weight frames
        let attention = candle_nn::linear(in_dim, 1, vb.pp("attn"))?;

        // Output projection: maps concatenated [mean; std] to embedding space
        let output_proj = candle_nn::linear(in_dim * 2, embed_dim, vb.pp("proj"))?;

        Ok(Self {
            attention,
            output_proj,
        })
    }

    /// Forward pass: extract speaker embedding from feature sequence.
    ///
    /// # Arguments
    /// * `x` - Input tensor [B, in_dim, T] of shape (batch, channels, time)
    ///
    /// # Returns
    /// * Output embedding [B, embed_dim] (fixed-size speaker embedding)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Transpose from [B, in_dim, T] to [B, T, in_dim] for linear layer processing
        let x_t = x.transpose(1, 2)?;

        // Compute attention weights: [B, T, 1]
        let attn_logits = self.attention.forward(&x_t)?;

        // Apply softmax to get normalized attention weights
        let attn_weights = softmax(&attn_logits, 1)?;

        // Apply attention weighting to features
        // x_t: [B, T, in_dim], attn_weights: [B, T, 1]
        // Reshape to 2D, multiply, then reshape back
        let dims = x_t.dims();
        let batch = dims[0];
        let time = dims[1];
        let in_dim = dims[2];

        let weighted = {
            // Reshape x_t to [B*T, in_dim]
            let x_2d = x_t.reshape(&[batch * time, in_dim])?;
            // Reshape attn_weights to [B*T, 1]
            let attn_2d = attn_weights.reshape(&[batch * time, 1])?;
            // Multiply [B*T, in_dim] * [B*T, 1] -> broadcasts to [B*T, in_dim]
            let weighted_2d = x_2d.broadcast_mul(&attn_2d)?;
            // Reshape back to [B, T, in_dim]
            weighted_2d.reshape(&[batch, time, in_dim])?
        };

        // Compute weighted mean across time dimension
        // weighted: [B, T, in_dim], sum over T (dim 1) -> [B, in_dim]
        let mean = weighted.sum(1)?;

        // Compute variance: E[(x - mean)^2 * weight]
        // Center each feature around its weighted mean
        let mean_expanded = mean.unsqueeze(1)?;  // [B, in_dim] -> [B, 1, in_dim]

        // Use explicit broadcasting for subtraction
        let mean_bcast = mean_expanded.broadcast_as(x_t.shape())?;
        let centered = (&x_t - &mean_bcast)?;  // Element-wise subtract
        let squared = centered.sqr()?;

        // Weight the squared deviations
        // Reshape to 2D, multiply, then reshape back
        let weighted_var = {
            let squared_2d = squared.reshape(&[batch * time, in_dim])?;
            let attn_2d = attn_weights.reshape(&[batch * time, 1])?;
            let weighted_2d = squared_2d.broadcast_mul(&attn_2d)?;
            weighted_2d.reshape(&[batch, time, in_dim])?
        };

        // Sum weighted variance over time: [B, T, in_dim] -> [B, in_dim]
        let var = weighted_var.sum(1)?;

        // Standard deviation with small epsilon for numerical stability
        let std = ((var + 1e-6)? .sqrt())?;

        // Concatenate mean and std along feature dimension: [B, in_dim*2]
        // mean: [B, in_dim], std: [B, in_dim]
        let stats = Tensor::cat(&[&mean, &std], D::Minus1)?;

        // Project to embedding dimension: [B, embed_dim]
        self.output_proj.forward(&stats).map_err(Into::into)
    }
}

/// Softmax along specified dimension (keeps dimensions for broadcasting)
fn softmax(x: &Tensor, dim: usize) -> Result<Tensor> {
    // Use keepdim to preserve dimensions for broadcasting
    let max = x.max_keepdim(dim)?;
    // Explicitly broadcast max to match x's shape for subtraction
    let max_bcast = max.broadcast_as(x.shape())?;
    let exp = (x - &max_bcast)?.exp()?;
    let sum = exp.sum_keepdim(dim)?;
    // Explicitly broadcast sum to match exp's shape for division
    let sum_bcast = sum.broadcast_as(exp.shape())?;
    (exp / &sum_bcast).map_err(Into::into)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attentive_stats_pooling_output_shape() -> Result<()> {
        let dev = Device::Cpu;
        let pool = AttentiveStatsPooling::new(256, 192, &dev)?;
        let x = Tensor::randn(0.0f32, 1.0, &[1, 256, 100], &dev)?;
        let emb = pool.forward(&x)?;

        assert_eq!(emb.dims(), &[1, 192]);

        Ok(())
    }

    #[test]
    fn test_attentive_stats_pooling_batch() -> Result<()> {
        let dev = Device::Cpu;
        let pool = AttentiveStatsPooling::new(256, 192, &dev)?;
        let x = Tensor::randn(0.0f32, 1.0, &[4, 256, 100], &dev)?;
        let emb = pool.forward(&x)?;

        assert_eq!(emb.dims(), &[4, 192]);

        Ok(())
    }

    #[test]
    fn test_attentive_stats_pooling_variable_length() -> Result<()> {
        let dev = Device::Cpu;
        let pool = AttentiveStatsPooling::new(256, 192, &dev)?;

        // Test with different temporal lengths
        for &t in &[50, 100, 200, 500] {
            let x = Tensor::randn(0.0f32, 1.0, &[2, 256, t], &dev)?;
            let emb = pool.forward(&x)?;
            assert_eq!(emb.dims(), &[2, 192]);
        }

        Ok(())
    }

    #[test]
    fn test_attentive_stats_pooling_fixed_size_output() -> Result<()> {
        let dev = Device::Cpu;
        let pool = AttentiveStatsPooling::new(256, 192, &dev)?;

        // Different batch and time dimensions should all produce same embedding size
        let shapes = vec![
            (1, 256, 50),
            (1, 256, 1000),
            (8, 256, 100),
            (16, 256, 200),
        ];

        for (b, d, t) in shapes {
            let x = Tensor::randn(0.0f32, 1.0, &[b, d, t], &dev)?;
            let emb = pool.forward(&x)?;
            assert_eq!(emb.dims(), &[b, 192]);
        }

        Ok(())
    }
}
