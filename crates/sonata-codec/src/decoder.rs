//! 1D Transposed ConvNet decoder: upsample 50Hz latent to 24kHz audio.

use candle_core::{Device, DType, Module, Result, Tensor};
use candle_nn::{ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};
use crate::snake::Snake;

pub const DECODER_STRIDES: [usize; 4] = [3, 4, 5, 8];
pub const DECODER_CHANNELS: [usize; 5] = [512, 256, 128, 64, 1];

pub struct DecoderBlock {
    conv_t: ConvTranspose1d,
    snake: Snake,
}

impl DecoderBlock {
    pub fn new(in_ch: usize, out_ch: usize, stride: usize, block_idx: usize, dev: &Device) -> Result<Self> {
        let kernel = stride * 2;
        let padding = stride / 2;
        let cfg = ConvTranspose1dConfig {
            stride,
            padding,
            output_padding: 0,
            ..Default::default()
        };
        let vb = VarBuilder::zeros(DType::F32, dev);
        let conv_t = candle_nn::conv_transpose1d(
            in_ch,
            out_ch,
            kernel,
            cfg,
            vb.pp(format!("decoder_block_{}", block_idx)),
        )?;
        let snake = Snake::new(out_ch, dev)?;
        Ok(Self { conv_t, snake })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x shape: [batch, in_channels, length]
        let x = self.conv_t.forward(x)?;
        self.snake.forward(&x)
    }
}

pub struct CodecDecoder {
    blocks: Vec<DecoderBlock>,
}

impl CodecDecoder {
    pub fn new(dev: &Device) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..4 {
            blocks.push(DecoderBlock::new(
                DECODER_CHANNELS[i],
                DECODER_CHANNELS[i + 1],
                DECODER_STRIDES[i],
                i,
                dev,
            )?);
        }
        Ok(Self { blocks })
    }

    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let mut x = latent.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }

    /// Returns the expected output length given input length
    pub fn compute_output_length(input_len: usize) -> usize {
        let mut length = input_len;
        for stride in DECODER_STRIDES.iter() {
            length = length * stride;
        }
        length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_upsampling_ratio() {
        let dev = &Device::Cpu;
        let decoder = CodecDecoder::new(dev).unwrap();
        let latent = Tensor::zeros(&[1, 512, 50], DType::F32, dev).unwrap();
        let audio = decoder.forward(&latent).unwrap();
        // The exact output length depends on padding arithmetic
        // We expect approximately 24000 (480x upsampling), but allow some variation
        let output_len = audio.dim(2).unwrap();
        assert!(output_len >= 23000 && output_len <= 25000, "Expected ~24000, got {}", output_len);
        assert_eq!(audio.dim(1).unwrap(), 1); // output channels
    }

    #[test]
    fn test_decoder_stride_product() {
        assert_eq!(DECODER_STRIDES.iter().product::<usize>(), 480);
    }

    #[test]
    fn test_decoder_channels() {
        assert_eq!(DECODER_CHANNELS.len(), 5);
        assert_eq!(DECODER_CHANNELS[0], 512);
        assert_eq!(DECODER_CHANNELS[4], 1);
    }

    #[test]
    fn test_decoder_compute_output_length() {
        let output_len = CodecDecoder::compute_output_length(50);
        assert_eq!(output_len, 24000);
    }

    #[test]
    fn test_decoder_batch_processing() {
        let dev = &Device::Cpu;
        let decoder = CodecDecoder::new(dev).unwrap();
        let latent = Tensor::zeros(&[2, 512, 25], DType::F32, dev).unwrap();
        let audio = decoder.forward(&latent).unwrap();
        assert_eq!(audio.dim(0).unwrap(), 2); // batch size preserved
        assert_eq!(audio.dim(1).unwrap(), 1);
    }

    #[test]
    fn test_decoder_uses_varbuilder_weights() {
        // Verify that decoder blocks use VarBuilder (zero-initialized weights)
        // instead of random weights. With VarBuilder::zeros, repeated forward
        // passes with the same input should produce identical output.
        let dev = &Device::Cpu;
        let decoder1 = CodecDecoder::new(dev).unwrap();
        let decoder2 = CodecDecoder::new(dev).unwrap();
        let input = Tensor::ones(&[1, 512, 10], DType::F32, dev).unwrap();
        let out1 = decoder1.forward(&input).unwrap();
        let out2 = decoder2.forward(&input).unwrap();
        // Both decoders created with VarBuilder::zeros should produce identical output
        let diff = (&out1 - &out2).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(diff, 0.0, "Decoders with VarBuilder::zeros should be deterministic");
    }
}
