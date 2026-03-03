//! Sonata CFM (Conditional Flow Matching) Decoder
//!
//! Converts codec token embeddings to mel spectrograms using straight-line flow matching.
//! The CFM decoder is the core audio generation module of Sonata v2, replacing diffusion-based approaches
//! with a more efficient flow matching model.
//!
//! # Architecture
//!
//! The CFM model uses:
//! - **ODE Solver**: Euler integration from noise to target mel spectrogram (4-8 steps)
//! - **DiT Blocks**: Diffusion Transformer blocks parameterizing the velocity field
//! - **Conditioning**: Speaker embedding via AdaIN, timestep via learned embeddings
//!
//! # Example
//!
//! ```no_run
//! use sonata_cfm::SonataCFM;
//! use candle_core::{Device, DType, Tensor};
//!
//! let dev = Device::Cpu;
//! let cfm = SonataCFM::new(&dev).unwrap();
//!
//! // Generate mel spectrogram from speaker embedding
//! let speaker_emb = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
//! let mel = cfm.generate(&speaker_emb, 50, 4).unwrap();
//! // Output shape: [1, 80, 50] (batch, mel_bins, time)
//! ```

pub mod ode;
pub mod dit;

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use dit::DiTBlock;
use sonata_common::{CFM_DIM, CFM_LAYERS, MEL_BINS, SPEAKER_EMBED_DIM};

const TIME_DIM: usize = 256;
const CFM_FFN_DIM: usize = 2048;

/// Time embedding: sinusoidal encoding + MLP for diffusion timestep.
///
/// Standard diffusion model time embedding:
/// 1. Encode scalar timestep t into sinusoidal features (sin/cos at log-spaced frequencies)
/// 2. Pass through a 2-layer MLP with ReLU activation
struct TimeEmbedding {
    mlp1: Linear,
    mlp2: Linear,
    dim: usize,
}

impl TimeEmbedding {
    fn new(dim: usize, dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);
        let mlp1 = candle_nn::linear(dim, dim, vb.pp("time_mlp1"))?;
        let mlp2 = candle_nn::linear(dim, dim, vb.pp("time_mlp2"))?;
        Ok(Self { mlp1, mlp2, dim })
    }

    /// Create sinusoidal embedding for a scalar timestep.
    ///
    /// Uses log-spaced frequencies: freq_i = exp(-i/half_dim * ln(10000))
    /// Output: [1, dim] with first half sin, second half cos.
    fn sinusoidal_embedding(t: f32, dim: usize, dev: &Device) -> Result<Tensor> {
        let half_dim = dim / 2;
        let mut emb = vec![0.0f32; dim];
        for i in 0..half_dim {
            let freq = (-(i as f64 / half_dim as f64) * (10000.0_f64).ln()).exp();
            let angle = t as f64 * freq;
            emb[i] = angle.sin() as f32;
            emb[i + half_dim] = angle.cos() as f32;
        }
        Tensor::from_vec(emb, (1, dim), dev)
    }

    fn forward(&self, t: f32, dev: &Device) -> Result<Tensor> {
        let emb = Self::sinusoidal_embedding(t, self.dim, dev)?;
        let h = self.mlp1.forward(&emb)?.relu()?;
        self.mlp2.forward(&h)
    }
}

/// Sonata CFM Decoder: converts noise to mel spectrograms via flow matching.
///
/// The CFM decoder is a generative model that:
/// 1. Takes a speaker embedding and generates mel spectrograms via ODE integration
/// 2. Uses DiT blocks conditioned on timestep and speaker to parameterize the velocity field
/// 3. Applies Euler integration to solve the ODE from noise to target distribution
pub struct SonataCFM {
    /// Input projection: mel_bins -> CFM_DIM
    input_proj: Linear,
    /// Stack of DiT blocks
    blocks: Vec<DiTBlock>,
    /// Output projection: CFM_DIM -> mel_bins
    output_proj: Linear,
    /// Time step embedding
    time_embed: TimeEmbedding,
}

impl SonataCFM {
    /// Create a new CFM decoder.
    ///
    /// Initializes the model with zero-variance weights. In practice, pretrained weights
    /// should be loaded via safetensors.
    pub fn new(dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);
        let input_proj = candle_nn::linear(MEL_BINS, CFM_DIM, vb.pp("input"))?;
        let mut blocks = Vec::new();
        for _ in 0..CFM_LAYERS {
            blocks.push(DiTBlock::new(CFM_DIM, CFM_FFN_DIM, TIME_DIM, SPEAKER_EMBED_DIM, dev)?);
        }
        let output_proj = candle_nn::linear(CFM_DIM, MEL_BINS, vb.pp("output"))?;
        let time_embed = TimeEmbedding::new(TIME_DIM, dev)?;
        Ok(Self { input_proj, blocks, output_proj, time_embed })
    }

    /// Compute the velocity field v(x, t, speaker).
    ///
    /// The velocity field parameterizes the flow: dx/dt = v(x, t, speaker)
    /// This is learned via the DiT blocks.
    fn velocity(&self, x: &Tensor, t: f32, speaker_emb: &Tensor) -> Result<Tensor> {
        let time_emb = self.time_embed.forward(t, x.device())?;

        // Project input to hidden dimension
        // x is [B, mel, T], reshape to [B, T, mel] for processing
        let x_transposed = x.transpose(1, 2)?; // [B, T, mel]
        let mut h = self.input_proj.forward(&x_transposed)?; // [B, T, CFM_DIM]

        // Apply DiT blocks with conditioning
        for block in &self.blocks {
            h = block.forward(&h, &time_emb, speaker_emb)?;
        }

        // Project back to mel space
        let out = self.output_proj.forward(&h)?; // [B, T, mel]

        // Transpose back to [B, mel, T]
        out.transpose(1, 2)
    }

    /// Generate a mel spectrogram from noise.
    ///
    /// # Arguments
    /// * `speaker_emb` - Speaker embedding of shape [B, 192]
    /// * `num_frames` - Number of time steps (determines mel spectrogram length)
    /// * `steps` - Number of ODE solver steps (typically 4-8)
    ///
    /// # Returns
    /// * Mel spectrogram of shape [B, 80, num_frames]
    pub fn generate(&self, speaker_emb: &Tensor, num_frames: usize, steps: usize) -> Result<Tensor> {
        let batch = speaker_emb.dim(0)?;
        let dev = speaker_emb.device();

        // Start from random noise
        let noise = Tensor::randn(0.0f32, 1.0, &[batch, MEL_BINS, num_frames], dev)?;

        // Solve ODE from noise to mel spectrogram
        let spk = speaker_emb.clone();
        ode::euler_solve(
            |x, t| self.velocity(x, t, &spk),
            &noise,
            steps,
        )
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_cfm_generation_basic() {
        let dev = Device::Cpu;
        let cfm = SonataCFM::new(&dev).unwrap();
        let speaker = Tensor::zeros(&[1, SPEAKER_EMBED_DIM], DType::F32, &dev).unwrap();
        let mel = cfm.generate(&speaker, 50, 4).unwrap();

        assert_eq!(mel.dim(0).unwrap(), 1);
        assert_eq!(mel.dim(1).unwrap(), MEL_BINS);
        assert_eq!(mel.dim(2).unwrap(), 50);
    }

    #[test]
    fn test_cfm_different_lengths() {
        let dev = Device::Cpu;
        let cfm = SonataCFM::new(&dev).unwrap();
        let speaker = Tensor::zeros(&[1, SPEAKER_EMBED_DIM], DType::F32, &dev).unwrap();

        for frames in [10, 25, 50, 100].iter() {
            let mel = cfm.generate(&speaker, *frames, 4).unwrap();
            assert_eq!(mel.dims(), &[1, MEL_BINS, *frames]);
        }
    }

    #[test]
    fn test_cfm_batch_generation() {
        let dev = Device::Cpu;
        let cfm = SonataCFM::new(&dev).unwrap();
        let speaker = Tensor::zeros(&[4, SPEAKER_EMBED_DIM], DType::F32, &dev).unwrap();
        let mel = cfm.generate(&speaker, 50, 4).unwrap();

        assert_eq!(mel.dim(0).unwrap(), 4);
        assert_eq!(mel.dim(1).unwrap(), MEL_BINS);
        assert_eq!(mel.dim(2).unwrap(), 50);
    }

    #[test]
    fn test_cfm_different_step_counts() {
        let dev = Device::Cpu;
        let cfm = SonataCFM::new(&dev).unwrap();
        let speaker = Tensor::zeros(&[1, SPEAKER_EMBED_DIM], DType::F32, &dev).unwrap();

        // Test different ODE solver step counts
        for steps in [2, 4, 8, 16].iter() {
            let mel = cfm.generate(&speaker, 30, *steps).unwrap();
            assert_eq!(mel.dims(), &[1, MEL_BINS, 30]);
        }
    }

    #[test]
    fn test_cfm_large_batch() {
        let dev = Device::Cpu;
        let cfm = SonataCFM::new(&dev).unwrap();
        let speaker = Tensor::zeros(&[8, SPEAKER_EMBED_DIM], DType::F32, &dev).unwrap();
        let mel = cfm.generate(&speaker, 40, 4).unwrap();

        assert_eq!(mel.dims(), &[8, MEL_BINS, 40]);
    }

    #[test]
    fn test_cfm_different_speaker_embeddings() {
        let dev = Device::Cpu;
        let cfm = SonataCFM::new(&dev).unwrap();

        let speaker1 = Tensor::zeros(&[1, SPEAKER_EMBED_DIM], DType::F32, &dev).unwrap();
        let speaker2 = Tensor::ones(&[1, SPEAKER_EMBED_DIM], DType::F32, &dev).unwrap();

        let mel1 = cfm.generate(&speaker1, 30, 4).unwrap();
        let mel2 = cfm.generate(&speaker2, 30, 4).unwrap();

        assert_eq!(mel1.dims(), &[1, MEL_BINS, 30]);
        assert_eq!(mel2.dims(), &[1, MEL_BINS, 30]);
    }

    #[test]
    fn test_cfm_velocity_field_shape() {
        let dev = Device::Cpu;
        let cfm = SonataCFM::new(&dev).unwrap();
        let x = Tensor::zeros(&[1, MEL_BINS, 30], DType::F32, &dev).unwrap();
        let speaker = Tensor::zeros(&[1, SPEAKER_EMBED_DIM], DType::F32, &dev).unwrap();

        let v = cfm.velocity(&x, 0.5, &speaker).unwrap();
        assert_eq!(v.dims(), &[1, MEL_BINS, 30]);
    }

    #[test]
    fn test_cfm_mel_bins_constant() {
        // Verify that MEL_BINS is imported correctly from sonata-common
        assert_eq!(MEL_BINS, 80);
    }

    #[test]
    fn test_cfm_speaker_embed_dim_constant() {
        // Verify that SPEAKER_EMBED_DIM is imported correctly
        assert_eq!(SPEAKER_EMBED_DIM, 192);
    }

    // --- Error path tests ---

    #[test]
    fn test_cfm_wrong_speaker_dim() {
        // Speaker embedding with wrong dimension (128 instead of 192)
        let dev = Device::Cpu;
        let cfm = SonataCFM::new(&dev).unwrap();
        let bad_speaker = Tensor::zeros(&[1, 128], DType::F32, &dev).unwrap();
        let result = cfm.generate(&bad_speaker, 50, 4);
        assert!(result.is_err(), "speaker dim 128 instead of 192 should fail");
    }

    #[test]
    fn test_cfm_single_step() {
        // Single ODE step — should still produce valid output
        let dev = Device::Cpu;
        let cfm = SonataCFM::new(&dev).unwrap();
        let speaker = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let mel = cfm.generate(&speaker, 50, 1).unwrap();
        assert_eq!(mel.dims(), &[1, 80, 50]);
    }

    #[test]
    fn test_cfm_single_frame() {
        // Single mel frame (num_frames=1) — edge case for temporal processing
        let dev = Device::Cpu;
        let cfm = SonataCFM::new(&dev).unwrap();
        let speaker = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let mel = cfm.generate(&speaker, 1, 4).unwrap();
        assert_eq!(mel.dims(), &[1, 80, 1]);
    }

    #[test]
    fn test_cfm_2d_speaker_fails() {
        // 1D speaker embedding (missing batch dimension) — should fail
        let dev = Device::Cpu;
        let cfm = SonataCFM::new(&dev).unwrap();
        let bad_speaker = Tensor::zeros(&[192], DType::F32, &dev).unwrap();
        let result = cfm.generate(&bad_speaker, 50, 4);
        assert!(result.is_err(), "1D speaker embedding should fail");
    }

    #[test]
    fn test_cfm_zero_frames() {
        // num_frames=0 — degenerate case
        let dev = Device::Cpu;
        let cfm = SonataCFM::new(&dev).unwrap();
        let speaker = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let result = cfm.generate(&speaker, 0, 4);
        // Zero frames should either produce empty output or error — not panic
        if let Ok(mel) = result {
            assert_eq!(mel.dim(0).unwrap(), 1);
            assert_eq!(mel.dim(1).unwrap(), 80);
            assert_eq!(mel.dim(2).unwrap(), 0);
        }
    }

    #[test]
    fn test_cfm_zero_steps() {
        // Zero ODE steps — should still produce output (just noise, no denoising)
        let dev = Device::Cpu;
        let cfm = SonataCFM::new(&dev).unwrap();
        let speaker = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let result = cfm.generate(&speaker, 50, 0);
        // Zero steps means no ODE integration — output is initial noise
        if let Ok(mel) = result {
            assert_eq!(mel.dims(), &[1, 80, 50]);
        }
    }

    #[test]
    fn test_cfm_wrong_speaker_dtype() {
        // F64 speaker embedding instead of F32
        let dev = Device::Cpu;
        let cfm = SonataCFM::new(&dev).unwrap();
        let bad_speaker = Tensor::zeros(&[1, 192], DType::F64, &dev).unwrap();
        let result = cfm.generate(&bad_speaker, 50, 4);
        assert!(result.is_err(), "F64 speaker embedding should fail (model expects F32)");
    }
}
