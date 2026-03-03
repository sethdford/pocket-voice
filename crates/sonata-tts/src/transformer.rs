//! TTS Transformer with AdaIN speaker/emotion conditioning.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
use sonata_common::adain::AdaIN;
use sonata_common::swiglu::SwiGLU;

pub struct TTSTransformerLayer {
    self_attn: Linear,
    speaker_adain: AdaIN,
    emotion_adain: AdaIN,
    ffn: SwiGLU,
    norm: candle_nn::LayerNorm,
}

impl TTSTransformerLayer {
    pub fn new(
        dim: usize,
        ffn_dim: usize,
        speaker_dim: usize,
        emotion_dim: usize,
        dev: &Device,
    ) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);
        let self_attn = candle_nn::linear(dim, dim, vb.pp("attn"))?;
        let speaker_adain = AdaIN::new(dim, speaker_dim, vb.pp("spk_adain"))?;
        let emotion_adain = AdaIN::new(dim, emotion_dim, vb.pp("emo_adain"))?;
        let ffn = SwiGLU::new(dim, ffn_dim, vb.pp("ffn"))?;
        let norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm"))?;
        Ok(Self {
            self_attn,
            speaker_adain,
            emotion_adain,
            ffn,
            norm,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        speaker_emb: &Tensor,
        emotion_emb: &Tensor,
    ) -> Result<Tensor> {
        // Self-attention
        let residual = x.clone();
        let x = self.norm.forward(x)?;
        let attn_out = self.self_attn.forward(&x)?;
        let x = (residual.broadcast_add(&attn_out))?;

        // Speaker AdaIN conditioning
        let x = self.speaker_adain.forward(&x, speaker_emb)?;

        // Emotion AdaIN conditioning
        let x = self.emotion_adain.forward(&x, emotion_emb)?;

        // FFN
        let residual = x.clone();
        let ffn_out = self.ffn.forward(&x)?;
        residual.broadcast_add(&ffn_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tts_transformer_layer() {
        let dev = Device::Cpu;
        let layer = TTSTransformerLayer::new(512, 2048, 192, 192, &dev).unwrap();
        let x = Tensor::zeros(&[1, 20, 512], DType::F32, &dev).unwrap();
        let spk = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let emo = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();
        let out = layer.forward(&x, &spk, &emo).unwrap();
        assert_eq!(out.dims(), &[1, 20, 512]);
    }
}
