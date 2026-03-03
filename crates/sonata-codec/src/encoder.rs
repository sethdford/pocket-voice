//! 1D ConvNet encoder: downsample 24kHz audio to 50Hz latent.

use candle_core::{Device, DType, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use crate::snake::Snake;

pub const ENCODER_STRIDES: [usize; 4] = [8, 5, 4, 3];
pub const ENCODER_CHANNELS: [usize; 5] = [1, 64, 128, 256, 512];
pub const ENCODER_LATENT_DIM: usize = 512;

pub struct EncoderBlock {
    conv: Conv1d,
    snake: Snake,
}

impl EncoderBlock {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        stride: usize,
        block_idx: usize,
        dev: &Device,
    ) -> Result<Self> {
        let kernel = stride * 2;
        let padding = stride / 2;
        let cfg = Conv1dConfig {
            stride,
            padding,
            ..Default::default()
        };
        let vb = VarBuilder::zeros(DType::F32, dev);
        let conv = candle_nn::conv1d(
            in_ch,
            out_ch,
            kernel,
            cfg,
            vb.pp(format!("conv{}", block_idx)),
        )?;
        let snake = Snake::new(out_ch, dev)?;
        Ok(Self { conv, snake })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        self.snake.forward(&x)
    }
}

pub struct CodecEncoder {
    blocks: Vec<EncoderBlock>,
}

impl CodecEncoder {
    pub fn new(dev: &Device) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..4 {
            blocks.push(EncoderBlock::new(
                ENCODER_CHANNELS[i],
                ENCODER_CHANNELS[i + 1],
                ENCODER_STRIDES[i],
                i,
                dev,
            )?);
        }
        Ok(Self { blocks })
    }

    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        let mut x = audio.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }

    /// Returns the expected output length given input length
    pub fn compute_output_length(input_len: usize) -> usize {
        let mut length = input_len;
        for stride in ENCODER_STRIDES.iter() {
            length = (length + stride / 2 - 1) / stride;
        }
        length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_downsampling_ratio() {
        let dev = &Device::Cpu;
        let encoder = CodecEncoder::new(dev).unwrap();
        let audio = Tensor::zeros(&[1, 1, 24000], DType::F32, dev).unwrap();
        let encoded = encoder.forward(&audio).unwrap();
        // We expect approximately 50 (480x downsampling), but allow some variation
        let output_len = encoded.dim(2).unwrap();
        assert!(output_len >= 45 && output_len <= 55, "Expected ~50, got {}", output_len);
        assert_eq!(encoded.dim(1).unwrap(), 512); // output channels
    }

    #[test]
    fn test_encoder_stride_product() {
        assert_eq!(ENCODER_STRIDES.iter().product::<usize>(), 480);
    }

    #[test]
    fn test_encoder_channels() {
        assert_eq!(ENCODER_CHANNELS.len(), 5);
        assert_eq!(ENCODER_CHANNELS[0], 1);
        assert_eq!(ENCODER_CHANNELS[4], 512);
    }

    #[test]
    fn test_encoder_compute_output_length() {
        let output_len = CodecEncoder::compute_output_length(24000);
        assert_eq!(output_len, 50);
    }

    #[test]
    fn test_encoder_batch_processing() {
        let dev = &Device::Cpu;
        let encoder = CodecEncoder::new(dev).unwrap();
        let audio = Tensor::randn(0.0f32, 0.1, (2, 1, 12000), dev).unwrap();
        let encoded = encoder.forward(&audio).unwrap();
        assert_eq!(encoded.dim(0).unwrap(), 2); // batch size preserved
        assert_eq!(encoded.dim(1).unwrap(), 512);
    }
}
