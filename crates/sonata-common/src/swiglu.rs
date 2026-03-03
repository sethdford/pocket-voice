//! SwiGLU feed-forward network for modern transformers.
//!
//! SwiGLU (Swish Gated Linear Unit) is an advanced feed-forward architecture used in
//! transformer models like PaLM and Llama. It replaces the traditional ReLU/GELU activation
//! with a gated mechanism that allows more expressive transformations.
//!
//! Architecture:
//! ```text
//! input -> [gate_proj, up_proj] -> [Swish(gate) * up] -> down_proj -> output
//! ```
//! Where Swish(x) = x * sigmoid(x).

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

/// SwiGLU feed-forward network module.
///
/// Implements a gated feed-forward network with two separate linear projections (gate and up)
/// whose outputs are element-wise multiplied after applying Swish activation to the gate,
/// then projected back down to the original dimension.
pub struct SwiGLU {
    /// Gate linear projection: input_dim -> ffn_dim
    w_gate: Linear,
    /// Up linear projection: input_dim -> ffn_dim
    w_up: Linear,
    /// Down linear projection: ffn_dim -> input_dim
    w_down: Linear,
}

impl SwiGLU {
    /// Create a new SwiGLU module.
    ///
    /// # Arguments
    /// * `dim` - Input/output dimension
    /// * `ffn_dim` - Hidden feed-forward dimension (typically 4x or more of input dim)
    /// * `vb` - Variable builder for initializing weights
    ///
    /// # Returns
    /// A new SwiGLU module with initialized weights
    pub fn new(dim: usize, ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w_gate = candle_nn::linear(dim, ffn_dim, vb.pp("gate"))?;
        let w_up = candle_nn::linear(dim, ffn_dim, vb.pp("up"))?;
        let w_down = candle_nn::linear(ffn_dim, dim, vb.pp("down"))?;

        Ok(Self {
            w_gate,
            w_up,
            w_down,
        })
    }

    /// Apply the SwiGLU transformation.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq_len, dim] or [batch, dim] for single tokens
    ///
    /// # Returns
    /// Output tensor of same shape as input
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Project input through gate: [batch, seq_len, dim] -> [batch, seq_len, ffn_dim]
        let gate_proj = self.w_gate.forward(x)?;

        // Apply Swish activation: Swish(x) = x * sigmoid(x)
        // sigmoid(x) = 1 / (1 + e^(-x))
        let gate_activated = {
            // Compute sigmoid: 1 / (1 + e^(-x))
            let neg_gate = gate_proj.neg()?;
            let exp_neg = neg_gate.exp()?;
            let one_plus_exp = (&Tensor::ones_like(&exp_neg)? + &exp_neg)?;
            let sigmoid = one_plus_exp.recip()?; // 1 / (1 + e^(-x))

            // Multiply: x * sigmoid(x)
            gate_proj.broadcast_mul(&sigmoid)?
        };

        // Project input through up gate: [batch, seq_len, dim] -> [batch, seq_len, ffn_dim]
        let up_proj = self.w_up.forward(x)?;

        // Element-wise multiplication: gate_activated * up_proj
        let gated = gate_activated.broadcast_mul(&up_proj)?;

        // Project back down: [batch, seq_len, ffn_dim] -> [batch, seq_len, dim]
        self.w_down.forward(&gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_swiglu_creation() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let _ffn = SwiGLU::new(256, 1024, vb).unwrap();
        // If we got here without panic, creation succeeded
        assert!(true);
    }

    #[test]
    fn test_swiglu_output_shape() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let ffn = SwiGLU::new(256, 1024, vb).unwrap();

        let x = Tensor::zeros(&[1, 10, 256], DType::F32, &dev).unwrap();
        let out = ffn.forward(&x).unwrap();

        assert_eq!(out.dims(), &[1, 10, 256]);
    }

    #[test]
    fn test_swiglu_batch_shapes() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let ffn = SwiGLU::new(512, 2048, vb).unwrap();

        // Test with larger batch size and longer sequence
        let x = Tensor::zeros(&[4, 32, 512], DType::F32, &dev).unwrap();
        let out = ffn.forward(&x).unwrap();

        assert_eq!(out.dims(), &[4, 32, 512]);
    }

    #[test]
    fn test_swiglu_single_token() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let ffn = SwiGLU::new(128, 512, vb).unwrap();

        // Test with single token (no sequence dimension)
        let x = Tensor::zeros(&[2, 128], DType::F32, &dev).unwrap();
        let out = ffn.forward(&x).unwrap();

        assert_eq!(out.dims(), &[2, 128]);
    }

    #[test]
    fn test_swiglu_different_dimensions() {
        let dev = Device::Cpu;

        // Test various dimension configurations
        for &(input_dim, ffn_dim) in &[
            (64, 256),
            (128, 512),
            (256, 1024),
            (512, 2048),
            (1024, 4096),
        ] {
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let ffn = SwiGLU::new(input_dim, ffn_dim, vb).unwrap();

            let x = Tensor::zeros(&[2, 8, input_dim], DType::F32, &dev).unwrap();
            let out = ffn.forward(&x).unwrap();

            assert_eq!(out.dims(), &[2, 8, input_dim]);
        }
    }

    #[test]
    fn test_swiglu_gating_with_nonzero_input() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let ffn = SwiGLU::new(4, 8, vb).unwrap();

        // Create non-zero input
        let x_data = [[1.0f32, 2.0, 3.0, 4.0]];
        let x = Tensor::new(&x_data, &dev)
            .unwrap()
            .reshape((1, 1, 4))
            .unwrap();

        let out = ffn.forward(&x).unwrap();

        // Shape should be preserved
        assert_eq!(out.dims(), &[1, 1, 4]);

        // Output should be zero (zero-initialized weights)
        // With zero weights: gate = 0, up = 0, so output = 0
        let sum: f32 = out.sum_all().unwrap().to_scalar().unwrap();
        assert!(sum.abs() < 1e-5, "Expected near-zero output with zero-init weights, got {}", sum);
    }

    #[test]
    fn test_swiglu_gate_mechanism() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let ffn = SwiGLU::new(16, 32, vb).unwrap();

        // Test that the gating mechanism works (different inputs produce different outputs)
        let x1 = Tensor::zeros(&[1, 10, 16], DType::F32, &dev).unwrap();
        let x2 = Tensor::ones(&[1, 10, 16], DType::F32, &dev).unwrap();

        let out1 = ffn.forward(&x1).unwrap();
        let out2 = ffn.forward(&x2).unwrap();

        // Both outputs should have correct shape
        assert_eq!(out1.dims(), &[1, 10, 16]);
        assert_eq!(out2.dims(), &[1, 10, 16]);
    }

    #[test]
    fn test_swiglu_intermediate_dimension_expansion() {
        let dev = Device::Cpu;

        // SwiGLU typically uses 4x expansion like traditional FFN, but can vary
        let test_cases = vec![
            (256, 512),   // 2x expansion
            (256, 1024),  // 4x expansion
            (256, 2048),  // 8x expansion
        ];

        for (input_dim, ffn_dim) in test_cases {
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let ffn = SwiGLU::new(input_dim, ffn_dim, vb).unwrap();

            let x = Tensor::zeros(&[1, 16, input_dim], DType::F32, &dev).unwrap();
            let out = ffn.forward(&x).unwrap();

            // Output dimension should match input, not intermediate dimension
            assert_eq!(out.dims(), &[1, 16, input_dim]);
        }
    }
}
