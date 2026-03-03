//! Sonata TTS crate: ~100M params, AdaIN + Emotion + Nonverbal token support.
//!
//! Implements text-to-speech prediction with speaker and emotion conditioning.

pub mod emotion;
pub mod nonverbal;
pub mod text_encoder;
pub mod transformer;

use candle_core::{Device, DType, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
use emotion::EmotionStyleEncoder;
use nonverbal::NonverbalEncoder;
use text_encoder::TextEncoder;
use transformer::TTSTransformerLayer;

const TTS_DIM: usize = 512;
const TTS_LAYERS: usize = 12;
const TTS_FFN_DIM: usize = 2048;
const SPEAKER_DIM: usize = 192;
const EMOTION_DIM: usize = 192;
const CODEC_VOCAB: usize = 1024; // codebook size

pub struct SonataTTS {
    text_encoder: TextEncoder,
    layers: Vec<TTSTransformerLayer>,
    emotion_encoder: EmotionStyleEncoder,
    nonverbal_encoder: NonverbalEncoder,
    output_proj: Linear, // projects to codec token logits
}

impl SonataTTS {
    pub fn new(dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);
        let text_encoder = TextEncoder::new(TTS_DIM, dev)?;
        let mut layers = Vec::new();
        for _ in 0..TTS_LAYERS {
            layers.push(TTSTransformerLayer::new(
                TTS_DIM,
                TTS_FFN_DIM,
                SPEAKER_DIM,
                EMOTION_DIM,
                dev,
            )?);
        }
        let emotion_encoder = EmotionStyleEncoder::new(EMOTION_DIM, dev)?;
        let nonverbal_encoder = NonverbalEncoder::new(TTS_DIM, dev)?;
        let output_proj = candle_nn::linear(TTS_DIM, CODEC_VOCAB, vb.pp("output"))?;
        Ok(Self {
            text_encoder,
            layers,
            emotion_encoder,
            nonverbal_encoder,
            output_proj,
        })
    }

    /// Generate codec token logits from text tokens.
    /// Input: text_tokens [B, T], speaker_emb [B, 192], emotion_emb [B, 192]
    /// Output: [B, T, 1024] codec logits
    pub fn forward(
        &self,
        text_tokens: &Tensor,
        speaker_emb: &Tensor,
        emotion_emb: &Tensor,
    ) -> Result<Tensor> {
        let mut x = self.text_encoder.forward(text_tokens)?;
        for layer in &self.layers {
            x = layer.forward(&x, speaker_emb, emotion_emb)?;
        }
        self.output_proj.forward(&x)
    }

    /// Get reference to the emotion encoder (for external emotion encoding).
    pub fn emotion_encoder(&self) -> &EmotionStyleEncoder {
        &self.emotion_encoder
    }

    /// Get reference to the nonverbal encoder (for nonverbal token encoding).
    pub fn nonverbal_encoder(&self) -> &NonverbalEncoder {
        &self.nonverbal_encoder
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    #[test]
    fn test_tts_full_pipeline() {
        let dev = Device::Cpu;
        let tts = SonataTTS::new(&dev).unwrap();
        let tokens = Tensor::zeros(&[1, 20], DType::U32, &dev).unwrap();
        let speaker = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let emotion = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let logits = tts.forward(&tokens, &speaker, &emotion).unwrap();
        assert_eq!(logits.dim(0).unwrap(), 1);
        assert_eq!(logits.dim(1).unwrap(), 20);
        assert_eq!(logits.dim(2).unwrap(), 1024);
    }

    #[test]
    fn test_tts_with_emotion() {
        let dev = Device::Cpu;
        let tts = SonataTTS::new(&dev).unwrap();
        let tokens = Tensor::zeros(&[1, 10], DType::U32, &dev).unwrap();
        let speaker = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        // Encode a happy emotion with 1.5x exaggeration
        let emotion = tts.emotion_encoder.encode(5, 1.5, &dev).unwrap();
        let logits = tts.forward(&tokens, &speaker, &emotion).unwrap();
        assert_eq!(logits.dims(), &[1, 10, 1024]);
    }

    #[test]
    fn test_tts_with_different_batch_sizes() {
        let dev = Device::Cpu;
        let tts = SonataTTS::new(&dev).unwrap();

        // Test batch_size = 2
        let tokens = Tensor::zeros(&[2, 15], DType::U32, &dev).unwrap();
        let speaker = Tensor::zeros(&[2, 192], DType::F32, &dev).unwrap();
        let emotion = Tensor::zeros(&[2, 192], DType::F32, &dev).unwrap();
        let logits = tts.forward(&tokens, &speaker, &emotion).unwrap();
        assert_eq!(logits.dims(), &[2, 15, 1024]);
    }

    #[test]
    fn test_tts_with_nonverbal_tokens() {
        use sonata_common::NonverbalTag;
        let dev = Device::Cpu;
        let tts = SonataTTS::new(&dev).unwrap();

        // Encode various nonverbal tokens
        let laugh = tts.nonverbal_encoder.encode(NonverbalTag::Laugh, &dev).unwrap();
        let sigh = tts.nonverbal_encoder.encode(NonverbalTag::Sigh, &dev).unwrap();

        assert_eq!(laugh.dims(), &[1, 512]);
        assert_eq!(sigh.dims(), &[1, 512]);
    }
}
