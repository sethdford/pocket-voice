//! Sonata Codec v2: Audio tokenization and reconstruction
//!
//! This crate provides the audio encoder/decoder for the Sonata v2 voice pipeline.
//! It converts 24kHz audio to/from discrete tokens via a learned codec architecture.
//!
//! Architecture:
//! - Encoder: 1D ConvNet with 480x downsampling (24kHz → 50Hz latent space)
//! - Quantizer: Residual Vector Quantizer (8 codebooks, 1024 entries each)
//! - Decoder: Transposed 1D ConvNet with 480x upsampling (50Hz → 24kHz audio)
//!
//! The codec preserves both semantic (books 1-2) and acoustic (books 3-8) information.

pub mod decoder;
pub mod encoder;
pub mod quantizer;
pub mod snake;

use candle_core::{Device, Result, Tensor};
use decoder::CodecDecoder;
use encoder::CodecEncoder;
use quantizer::ResidualVQ;
use sonata_common::{CODEBOOK_DIM, CODEBOOK_SIZE, NUM_CODEBOOKS};

/// Main codec interface: encode 24kHz audio to tokens, decode tokens back to audio.
pub struct SonataCodec {
    encoder: CodecEncoder,
    decoder: CodecDecoder,
    quantizer: ResidualVQ,
}

impl SonataCodec {
    /// Create a new codec instance
    pub fn new(dev: &Device) -> Result<Self> {
        let encoder = CodecEncoder::new(dev)?;
        let decoder = CodecDecoder::new(dev)?;
        let quantizer = ResidualVQ::new(512, NUM_CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM, dev)?;
        Ok(Self {
            encoder,
            decoder,
            quantizer,
        })
    }

    /// Encode 24kHz audio to discrete tokens
    ///
    /// Input: audio tensor of shape [batch, 1, 24000] (1 second at 24kHz)
    /// Output: codes tensor of shape [batch, 8, 50]
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        let latent = self.encoder.forward(audio)?;
        let (codes, _) = self.quantizer.encode(&latent)?;
        Ok(codes)
    }

    /// Decode discrete tokens back to 24kHz audio
    ///
    /// Input: codes tensor of shape [batch, 8, 50]
    /// Output: audio tensor of shape [batch, 1, 24000]
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let latent = self.quantizer.decode(codes)?;
        self.decoder.forward(&latent)
    }

    /// Decode codes to continuous embeddings (RVQ reconstruction) without audio synthesis.
    ///
    /// Input: codes tensor of shape [batch, 8, T] (u32 codebook indices)
    /// Output: embeddings tensor of shape [batch, 512, T]
    ///
    /// This performs the RVQ decode step (sum codebook lookups across books)
    /// but does NOT run the audio decoder. Useful for feeding into STT/TTS.
    pub fn codes_to_embeddings(&self, codes: &Tensor) -> Result<Tensor> {
        self.quantizer.decode(codes)
    }

    /// Split codes into semantic (books 1-2) and acoustic (books 3-8) components
    pub fn split_codes(&self, codes: &Tensor) -> Result<(Tensor, Tensor)> {
        self.quantizer.split_codes(codes)
    }

    /// Get the codebook embedding tensor for a specific codebook index.
    ///
    /// Returns the raw codebook tensor [CODEBOOK_SIZE, CODEBOOK_DIM].
    pub fn get_codebook_embeddings(&self, book_idx: usize) -> Result<&Tensor> {
        self.quantizer.get_codebook_embeddings(book_idx)
    }

    /// Number of codebooks in the quantizer.
    pub fn num_codebooks(&self) -> usize {
        self.quantizer.num_codebooks()
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_codec_roundtrip_shape() {
        let dev = &Device::Cpu;
        let codec = SonataCodec::new(dev).unwrap();
        let audio = Tensor::randn(0.0f32, 0.1, (1, 1, 24000), dev).unwrap();
        let codes = codec.encode(&audio).unwrap();
        assert_eq!(codes.dim(0).unwrap(), 1);
        assert_eq!(codes.dim(1).unwrap(), 8);
        // Allow some variation due to padding arithmetic in encoder
        let code_len = codes.dim(2).unwrap();
        assert!(code_len >= 45 && code_len <= 55, "Expected ~50, got {}", code_len);
        let reconstructed = codec.decode(&codes).unwrap();
        // Decoder reconstructs audio - shape should match original
        assert_eq!(reconstructed.dim(0).unwrap(), audio.dim(0).unwrap());
        assert_eq!(reconstructed.dim(1).unwrap(), audio.dim(1).unwrap());
        // Audio length should be approximately 24000 (within ±1000 due to padding)
        let audio_len = reconstructed.dim(2).unwrap();
        assert!(audio_len >= 23000 && audio_len <= 25000, "Expected ~24000, got {}", audio_len);
    }

    #[test]
    fn test_codec_split_codes() {
        let dev = &Device::Cpu;
        let codec = SonataCodec::new(dev).unwrap();
        let audio = Tensor::randn(0.0f32, 0.1, (1, 1, 24000), dev).unwrap();
        let codes = codec.encode(&audio).unwrap();
        let (semantic, acoustic) = codec.split_codes(&codes).unwrap();
        assert_eq!(semantic.dim(1).unwrap(), 2);
        assert_eq!(acoustic.dim(1).unwrap(), 6);
    }

    #[test]
    fn test_codec_batch_processing() {
        let dev = &Device::Cpu;
        let codec = SonataCodec::new(dev).unwrap();
        let audio = Tensor::randn(0.0f32, 0.1, (4, 1, 12000), dev).unwrap();
        let codes = codec.encode(&audio).unwrap();
        assert_eq!(codes.dim(0).unwrap(), 4);
        let reconstructed = codec.decode(&codes).unwrap();
        assert_eq!(reconstructed.dim(0).unwrap(), 4);
    }

    #[test]
    fn test_codec_codes_to_embeddings_shape() {
        let dev = &Device::Cpu;
        let codec = SonataCodec::new(dev).unwrap();
        let audio = Tensor::randn(0.0f32, 0.1, (1, 1, 24000), dev).unwrap();
        let codes = codec.encode(&audio).unwrap();
        let embeddings = codec.codes_to_embeddings(&codes).unwrap();
        // Should be [B, 512, T] — same shape as encoder output
        assert_eq!(embeddings.dim(0).unwrap(), 1);
        assert_eq!(embeddings.dim(1).unwrap(), 512);
        assert_eq!(embeddings.dim(2).unwrap(), codes.dim(2).unwrap());
    }

    #[test]
    fn test_codec_codes_to_embeddings_deterministic() {
        let dev = &Device::Cpu;
        let codec = SonataCodec::new(dev).unwrap();
        let audio = Tensor::randn(0.0f32, 0.1, (1, 1, 24000), dev).unwrap();
        let codes = codec.encode(&audio).unwrap();
        // Same codes should produce identical embeddings (not random)
        let emb1 = codec.codes_to_embeddings(&codes).unwrap();
        let emb2 = codec.codes_to_embeddings(&codes).unwrap();
        let diff = (&emb1 - &emb2).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(diff, 0.0, "codes_to_embeddings should be deterministic (not random)");
    }

    #[test]
    fn test_codec_get_codebook_embeddings() {
        let dev = &Device::Cpu;
        let codec = SonataCodec::new(dev).unwrap();
        assert_eq!(codec.num_codebooks(), NUM_CODEBOOKS);
        for i in 0..NUM_CODEBOOKS {
            let cb = codec.get_codebook_embeddings(i).unwrap();
            assert_eq!(cb.dim(0).unwrap(), CODEBOOK_SIZE);
            assert_eq!(cb.dim(1).unwrap(), CODEBOOK_DIM);
        }
        // Out of range should error
        assert!(codec.get_codebook_embeddings(NUM_CODEBOOKS).is_err());
    }

    #[test]
    fn test_codec_encode_then_embeddings_roundtrip() {
        // Encode audio → codes → embeddings, verify shapes match
        let dev = &Device::Cpu;
        let codec = SonataCodec::new(dev).unwrap();
        let audio = Tensor::randn(0.0f32, 0.1, (2, 1, 12000), dev).unwrap();
        let codes = codec.encode(&audio).unwrap();
        let embeddings = codec.codes_to_embeddings(&codes).unwrap();
        // Batch preserved
        assert_eq!(embeddings.dim(0).unwrap(), 2);
        // Channel dim = 512 (input_dim of RVQ)
        assert_eq!(embeddings.dim(1).unwrap(), 512);
        // Time dim matches codes
        assert_eq!(embeddings.dim(2).unwrap(), codes.dim(2).unwrap());
    }
}
