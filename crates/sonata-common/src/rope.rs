//! Rotary Position Embeddings (RoPE) for transformer attention.
//!
//! RoPE is a positional encoding scheme that encodes positions as rotation matrices applied
//! to query and key vectors in attention. This allows the model to capture both absolute and
//! relative position information efficiently.

use candle_core::{Device, Result, Tensor};

/// Rotary Position Embedding module for transformer attention.
///
/// Encodes positional information by rotating query and key vectors in the complex plane.
/// Each dimension pair rotates by a different frequency, allowing the model to learn
/// position relationships without explicit position indices in embeddings.
pub struct RotaryEmbedding {
    /// Precomputed cosine cache for rotations: [max_seq_len, head_dim]
    cos_cache: Tensor,
    /// Precomputed sine cache for rotations: [max_seq_len, head_dim]
    sin_cache: Tensor,
}

impl RotaryEmbedding {
    /// Create a new RoPE module.
    ///
    /// # Arguments
    /// * `head_dim` - Dimension of each attention head (typically 64 or 128)
    /// * `max_seq_len` - Maximum sequence length to precompute cache for
    /// * `dev` - Device to place tensors on
    ///
    /// # Returns
    /// A new RotaryEmbedding module with precomputed sin/cos caches
    pub fn new(head_dim: usize, max_seq_len: usize, dev: &Device) -> Result<Self> {
        // Create inverse frequencies: [head_dim/2]
        // inv_freq[i] = 1.0 / (10000^(2i/head_dim))
        let mut inv_freq = Vec::with_capacity(head_dim / 2);
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / 10000f32.powf(i as f32 / head_dim as f32);
            inv_freq.push(freq);
        }
        let inv_freq = Tensor::new(inv_freq.as_slice(), dev)?;

        // Create position sequence: [max_seq_len]
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), dev)?;

        // Compute frequencies for all (position, dimension) pairs: [max_seq_len, head_dim/2]
        // freqs[pos, dim] = pos * inv_freq[dim]
        let positions_reshaped = positions.unsqueeze(1)?; // [max_seq_len, 1]
        let inv_freq_reshaped = inv_freq.unsqueeze(0)?; // [1, head_dim/2]
        let freqs = positions_reshaped.broadcast_mul(&inv_freq_reshaped)?;

        // Compute cos and sin caches
        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;

        Ok(Self { cos_cache, sin_cache })
    }

    /// Apply rotary embeddings to query and key tensors.
    ///
    /// # Arguments
    /// * `q` - Query tensor of shape [batch, num_heads, seq_len, head_dim]
    /// * `k` - Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]
    /// * `offset` - Starting position offset (for streaming/incremental generation)
    ///
    /// # Returns
    /// Tuple of (rotated_query, rotated_key)
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;

        // Extract the relevant portion of sin/cos caches
        let cos = self.cos_cache.narrow(0, offset, seq_len)?;
        let sin = self.sin_cache.narrow(0, offset, seq_len)?;

        // Apply rotation to both q and k
        let q_rot = Self::apply_rotary(q, &cos, &sin)?;
        let k_rot = Self::apply_rotary(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }

    /// Apply rotary transformation to a single tensor.
    ///
    /// Implements the rotation by pairing adjacent dimensions and applying 2D rotation:
    /// ```text
    /// [x1, x2, x3, x4, ...] -> [x1*cos - x2*sin, x2*cos + x1*sin, ...]
    /// ```
    ///
    /// # Arguments
    /// * `x` - Tensor of shape [batch, num_heads, seq_len, head_dim]
    /// * `cos` - Cosine cache of shape [seq_len, head_dim/2]
    /// * `sin` - Sine cache of shape [seq_len, head_dim/2]
    ///
    /// # Returns
    /// Rotated tensor of same shape as input
    fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let head_dim = x.dim(3)?;
        let half = head_dim / 2;

        // Split into two halves: [batch, num_heads, seq_len, half]
        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;

        // Expand cos/sin for broadcasting: [1, 1, seq_len, half]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // Apply 2D rotation matrix:
        // [cos -sin] [x1]   [x1*cos - x2*sin]
        // [sin  cos] [x2] = [x1*sin + x2*cos]
        let rotated_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rotated_x2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

        // Concatenate rotated halves
        Tensor::cat(&[&rotated_x1, &rotated_x2], 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_rope_creation() {
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 4096, &dev).unwrap();
        // Verify tensors are created (they have the right shape)
        assert_eq!(rope.cos_cache.dims(), &[4096, 32]); // head_dim/2 = 32
        assert_eq!(rope.sin_cache.dims(), &[4096, 32]);
    }

    #[test]
    fn test_rope_output_shape() {
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 4096, &dev).unwrap();

        // Create sample q and k tensors
        let q = Tensor::zeros(&[1, 8, 10, 64], DType::F32, &dev).unwrap();
        let k = Tensor::zeros(&[1, 2, 10, 64], DType::F32, &dev).unwrap();

        let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();

        assert_eq!(q_rot.dims(), &[1, 8, 10, 64]);
        assert_eq!(k_rot.dims(), &[1, 2, 10, 64]);
    }

    #[test]
    fn test_rope_with_offset() {
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(128, 2048, &dev).unwrap();

        // Create non-zero tensors to test rotation with different offsets
        let q = Tensor::ones(&[2, 4, 20, 128], DType::F32, &dev).unwrap();
        let k = Tensor::ones(&[2, 4, 20, 128], DType::F32, &dev).unwrap();

        // Apply with different offsets
        let (q_rot_0, _) = rope.apply(&q, &k, 0).unwrap();
        let (q_rot_100, _) = rope.apply(&q, &k, 100).unwrap();

        // Shapes should be the same
        assert_eq!(q_rot_0.dims(), q_rot_100.dims());

        // Values should be different (different rotation angles from different positions)
        let diff = (q_rot_0 - &q_rot_100).unwrap().abs().unwrap();
        let sum: f32 = diff.sum_all().unwrap().to_scalar().unwrap();
        assert!(sum > 0.1, "Rotations with different offsets should produce different values");
    }

    #[test]
    fn test_rope_different_head_dims() {
        let dev = Device::Cpu;

        // Test common head dimensions
        for head_dim in [64, 128, 256].iter() {
            let rope = RotaryEmbedding::new(*head_dim, 512, &dev).unwrap();
            assert_eq!(rope.cos_cache.dim(1).unwrap(), head_dim / 2);
            assert_eq!(rope.sin_cache.dim(1).unwrap(), head_dim / 2);

            let q = Tensor::zeros(&[1, 8, 32, *head_dim], DType::F32, &dev).unwrap();
            let k = Tensor::zeros(&[1, 2, 32, *head_dim], DType::F32, &dev).unwrap();

            let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();
            assert_eq!(q_rot.dims(), &[1, 8, 32, *head_dim]);
            assert_eq!(k_rot.dims(), &[1, 2, 32, *head_dim]);
        }
    }

    #[test]
    fn test_rope_batch_and_seq_length_variations() {
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 1024, &dev).unwrap();

        // Test various batch sizes and sequence lengths
        for &batch_size in &[1, 2, 4, 8] {
            for &seq_len in &[1, 10, 32, 128] {
                let q = Tensor::zeros(&[batch_size, 8, seq_len, 64], DType::F32, &dev).unwrap();
                let k = Tensor::zeros(&[batch_size, 2, seq_len, 64], DType::F32, &dev).unwrap();

                let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();
                assert_eq!(q_rot.dims(), &[batch_size, 8, seq_len, 64]);
                assert_eq!(k_rot.dims(), &[batch_size, 2, seq_len, 64]);
            }
        }
    }

    #[test]
    fn test_rope_streaming_offset() {
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 2048, &dev).unwrap();

        // Simulate streaming: process one token at offset 0, then one token at offset 1, etc.
        let single_token_q = Tensor::zeros(&[1, 8, 1, 64], DType::F32, &dev).unwrap();
        let single_token_k = Tensor::zeros(&[1, 2, 1, 64], DType::F32, &dev).unwrap();

        for offset in [0, 100, 500, 1000, 2000].iter() {
            if *offset < 2048 {
                let (q_rot, k_rot) = rope.apply(&single_token_q, &single_token_k, *offset).unwrap();
                assert_eq!(q_rot.dims(), &[1, 8, 1, 64]);
                assert_eq!(k_rot.dims(), &[1, 2, 1, 64]);
            }
        }
    }
}
