//! Sonata STT Crate v2: Streaming Conformer CTC (~100M params)
//!
//! Transforms codec embeddings [B, 512, T] to text tokens via CTC loss.
//!
//! # Architecture
//! - **Input**: Codec embeddings from Sonata Codec [B, 512, T]
//! - **Processing**: 12-layer Conformer with CTC decoder
//! - **Output**: Text token sequences (vocab_size = 32000)
//!
//! # Example
//! ```ignore
//! let stt = SonataSTT::new(&device)?;
//! let codec_embeddings = /* from codec */;
//! let text_tokens = stt.transcribe(&codec_embeddings)?;
//! ```

pub mod conformer;
pub mod ctc;
pub mod streaming;

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use conformer::ConformerBlock;

const STT_DIM: usize = 512;
const STT_LAYERS: usize = 12;
const STT_FFN_DIM: usize = 2048;

/// Full STT model: Conformer encoder + CTC decoder.
pub struct SonataSTT {
    input_proj: Linear,
    blocks: Vec<ConformerBlock>,
    output_proj: Linear,
}

impl SonataSTT {
    /// Create a new SonataSTT model.
    ///
    /// Initializes:
    /// - Input projection: 512 → 512
    /// - 12 Conformer blocks with 2048 FFN
    /// - Output projection: 512 → 32000 (CTC vocab)
    pub fn new(dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);
        let input_proj = candle_nn::linear(512, STT_DIM, vb.pp("input"))?;

        let mut blocks = Vec::new();
        for _ in 0..STT_LAYERS {
            blocks.push(ConformerBlock::new(
                STT_DIM,
                STT_FFN_DIM,
                dev,
            )?);
        }

        let output_proj =
            candle_nn::linear(STT_DIM, ctc::TEXT_VOCAB_SIZE, vb.pp("output"))?;

        Ok(Self {
            input_proj,
            blocks,
            output_proj,
        })
    }

    /// Process codec embeddings to CTC logits.
    ///
    /// # Arguments
    /// * `codec_embeddings` - Shape [B, 512, T] codec embeddings from Sonata Codec
    ///
    /// # Returns
    /// Shape [B, T, vocab_size] CTC logits for each frame
    ///
    /// # Example
    /// ```ignore
    /// let logits = stt.forward(&codec_embeddings)?;
    /// assert_eq!(logits.dims(), &[batch_size, seq_len, 32000]);
    /// ```
    pub fn forward(&self, codec_embeddings: &Tensor) -> Result<Tensor> {
        // codec_embeddings: [B, 512, T]
        // Transpose to [B, T, 512] for transformer processing
        let x = codec_embeddings.transpose(1, 2)?;

        // Input projection
        let x = self.input_proj.forward(&x)?;

        // Process through Conformer blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Output projection to logits
        self.output_proj.forward(&x)
    }

    /// Full transcription pipeline: codec embeddings → text tokens.
    ///
    /// This is the high-level API that combines forward pass + CTC decoding.
    ///
    /// # Arguments
    /// * `codec_embeddings` - Shape [B, 512, T] codec embeddings
    ///
    /// # Returns
    /// Vector of decoded text token sequences (one per batch element)
    ///
    /// # Example
    /// ```ignore
    /// let text_tokens = stt.transcribe(&codec_embeddings)?;
    /// assert_eq!(text_tokens.len(), batch_size);
    /// ```
    pub fn transcribe(&self, codec_embeddings: &Tensor) -> Result<Vec<Vec<u32>>> {
        let logits = self.forward(codec_embeddings)?;
        ctc::greedy_decode(&logits)
    }

    /// Get model dimensions for debugging/inspection.
    pub fn model_dims() -> ModelDims {
        ModelDims {
            input_dim: 512,
            hidden_dim: STT_DIM,
            num_layers: STT_LAYERS,
            ffn_dim: STT_FFN_DIM,
            vocab_size: ctc::TEXT_VOCAB_SIZE,
        }
    }
}

/// Model dimension information.
#[derive(Debug, Clone)]
pub struct ModelDims {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stt_codec_to_logits() {
        let dev = Device::Cpu;
        let stt = SonataSTT::new(&dev).unwrap();

        // Codec embeddings: [batch=1, 512, seq=50]
        let codec_embeddings = Tensor::zeros(&[1, 512, 50], DType::F32, &dev).unwrap();
        let logits = stt.forward(&codec_embeddings).unwrap();

        // Check output shape
        assert_eq!(logits.dims()[0], 1); // batch
        assert_eq!(logits.dims()[1], 50); // sequence length
        assert_eq!(logits.dims()[2], ctc::TEXT_VOCAB_SIZE); // vocab
    }

    #[test]
    fn test_stt_transcribe() {
        let dev = Device::Cpu;
        let stt = SonataSTT::new(&dev).unwrap();

        let codec_embeddings = Tensor::zeros(&[2, 512, 100], DType::F32, &dev).unwrap();
        let text_tokens = stt.transcribe(&codec_embeddings).unwrap();

        // Should return 2 sequences (one per batch)
        assert_eq!(text_tokens.len(), 2);
    }

    #[test]
    fn test_stt_batch_processing() {
        let dev = Device::Cpu;
        let stt = SonataSTT::new(&dev).unwrap();

        for batch_size in &[1, 2, 4, 8] {
            let codec_embeddings =
                Tensor::zeros(&[*batch_size, 512, 50], DType::F32, &dev).unwrap();
            let logits = stt.forward(&codec_embeddings).unwrap();

            assert_eq!(logits.dims()[0], *batch_size);
            assert_eq!(logits.dims()[1], 50);
            assert_eq!(logits.dims()[2], ctc::TEXT_VOCAB_SIZE);
        }
    }

    #[test]
    fn test_stt_sequence_length_variance() {
        let dev = Device::Cpu;
        let stt = SonataSTT::new(&dev).unwrap();

        for seq_len in &[10, 50, 100, 200, 500] {
            let codec_embeddings = Tensor::zeros(&[1, 512, *seq_len], DType::F32, &dev).unwrap();
            let logits = stt.forward(&codec_embeddings).unwrap();

            assert_eq!(logits.dims()[1], *seq_len);
        }
    }

    #[test]
    fn test_model_dims() {
        let dims = SonataSTT::model_dims();
        assert_eq!(dims.input_dim, 512);
        assert_eq!(dims.hidden_dim, 512);
        assert_eq!(dims.num_layers, 12);
        assert_eq!(dims.ffn_dim, 2048);
        assert_eq!(dims.vocab_size, 32000);
    }

    #[test]
    fn test_stt_zero_initialization() {
        // Models initialized with zeros should produce deterministic output
        let dev = Device::Cpu;
        let stt1 = SonataSTT::new(&dev).unwrap();
        let stt2 = SonataSTT::new(&dev).unwrap();

        let codec_embeddings = Tensor::zeros(&[1, 512, 50], DType::F32, &dev).unwrap();

        let logits1 = stt1.forward(&codec_embeddings).unwrap();
        let logits2 = stt2.forward(&codec_embeddings).unwrap();

        // Both should have identical shapes
        assert_eq!(logits1.dims(), logits2.dims());
    }

    #[test]
    fn test_stt_transcribe_batch() {
        let dev = Device::Cpu;
        let stt = SonataSTT::new(&dev).unwrap();

        let codec_embeddings = Tensor::zeros(&[4, 512, 100], DType::F32, &dev).unwrap();
        let text_tokens = stt.transcribe(&codec_embeddings).unwrap();

        // Should have 4 sequences
        assert_eq!(text_tokens.len(), 4);

        // Each sequence should be a vector of u32
        for seq in text_tokens {
            assert!(seq.iter().all(|&tok| tok < ctc::TEXT_VOCAB_SIZE as u32));
        }
    }
}
