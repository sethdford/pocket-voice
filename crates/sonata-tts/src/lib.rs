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

    // --- Error path tests ---

    #[test]
    fn test_tts_wrong_speaker_dim() {
        // Speaker embedding with wrong dimension (128 instead of 192)
        let dev = Device::Cpu;
        let tts = SonataTTS::new(&dev).unwrap();
        let text_tokens = Tensor::zeros(&[1, 10], DType::U32, &dev).unwrap();
        let bad_speaker = Tensor::zeros(&[1, 128], DType::F32, &dev).unwrap();
        let emotion = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let result = tts.forward(&text_tokens, &bad_speaker, &emotion);
        assert!(result.is_err(), "speaker dim 128 instead of 192 should fail");
    }

    #[test]
    fn test_tts_wrong_emotion_dim() {
        // Emotion embedding with wrong dimension (64 instead of 192)
        let dev = Device::Cpu;
        let tts = SonataTTS::new(&dev).unwrap();
        let text_tokens = Tensor::zeros(&[1, 10], DType::U32, &dev).unwrap();
        let speaker = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let bad_emotion = Tensor::zeros(&[1, 64], DType::F32, &dev).unwrap();
        let result = tts.forward(&text_tokens, &speaker, &bad_emotion);
        assert!(result.is_err(), "emotion dim 64 instead of 192 should fail");
    }

    #[test]
    fn test_tts_mismatched_batch_sizes() {
        // Text batch=2 but speaker batch=1 — transformer may broadcast or error
        let dev = Device::Cpu;
        let tts = SonataTTS::new(&dev).unwrap();
        let text_tokens = Tensor::zeros(&[2, 10], DType::U32, &dev).unwrap();
        let speaker = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let emotion = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let result = tts.forward(&text_tokens, &speaker, &emotion);
        // Some layers may broadcast speaker/emotion to match text batch.
        // Either error or correct output shape is acceptable — no panic.
        if let Ok(logits) = result {
            assert_eq!(logits.dims()[1], 10);
            assert_eq!(logits.dims()[2], 1024);
        }
    }

    #[test]
    fn test_tts_single_token() {
        // Single text token — should produce valid output
        let dev = Device::Cpu;
        let tts = SonataTTS::new(&dev).unwrap();
        let text_tokens = Tensor::zeros(&[1, 1], DType::U32, &dev).unwrap();
        let speaker = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let emotion = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let logits = tts.forward(&text_tokens, &speaker, &emotion).unwrap();
        assert_eq!(logits.dims(), &[1, 1, 1024]);
    }

    #[test]
    fn test_tts_empty_text_tokens() {
        // Zero-length text tokens (T=0)
        let dev = Device::Cpu;
        let tts = SonataTTS::new(&dev).unwrap();
        let empty_tokens = Tensor::zeros(&[1, 0], DType::U32, &dev).unwrap();
        let speaker = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let emotion = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let result = tts.forward(&empty_tokens, &speaker, &emotion);
        // Should produce zero-length output or error — not panic
        if let Ok(logits) = result {
            assert_eq!(logits.dim(0).unwrap(), 1);
            assert_eq!(logits.dim(1).unwrap(), 0);
        }
    }

    #[test]
    fn test_tts_extreme_emotion_exaggeration() {
        // Very high exaggeration values should not cause NaN/Inf
        let dev = Device::Cpu;
        let tts = SonataTTS::new(&dev).unwrap();
        let tokens = Tensor::zeros(&[1, 10], DType::U32, &dev).unwrap();
        let speaker = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let extreme_emotion = tts.emotion_encoder.encode(3, 100.0, &dev).unwrap();
        let result = tts.forward(&tokens, &speaker, &extreme_emotion);
        // Should not crash; output may be garbage but shouldn't be NaN
        if let Ok(logits) = result {
            let has_nan = logits.to_vec3::<f32>().unwrap().iter()
                .any(|batch| batch.iter().any(|frame| frame.iter().any(|v| v.is_nan())));
            assert!(!has_nan, "extreme emotion should not produce NaN logits");
        }
    }

    #[test]
    fn test_tts_zero_speaker_embedding() {
        // All-zero speaker embedding — should still produce valid output
        let dev = Device::Cpu;
        let tts = SonataTTS::new(&dev).unwrap();
        let tokens = Tensor::zeros(&[1, 5], DType::U32, &dev).unwrap();
        let zero_speaker = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let emotion = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let logits = tts.forward(&tokens, &zero_speaker, &emotion).unwrap();
        assert_eq!(logits.dims(), &[1, 5, 1024]);
    }
}
