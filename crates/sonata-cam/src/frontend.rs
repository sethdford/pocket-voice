//! Multi-Scale Aggregation Frontend for CAM++ speaker encoder.
//!
//! Processes mel spectrograms through a stack of 1D convolutional layers
//! to extract multi-scale features before passing to CAM blocks.

use anyhow::Result;
use candle_core::{Device, Module, Tensor, DType};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

/// Multi-scale frontend that progressively extracts hierarchical features
/// from mel spectrograms using three convolutional layers.
pub struct MultiScaleFrontend {
    conv1: Conv1d,
    conv2: Conv1d,
    conv3: Conv1d,
}

impl MultiScaleFrontend {
    /// Create a new multi-scale frontend.
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels (typically 80 for mel bins)
    /// * `out_channels` - Number of output channels (typically 256)
    /// * `dev` - Candle device
    pub fn new(in_channels: usize, out_channels: usize, dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);

        // Use padding=1 with kernel_size=3 to preserve temporal dimension
        let cfg = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };

        // Progressive expansion: in_channels -> 64 -> 128 -> out_channels
        let conv1 = candle_nn::conv1d(in_channels, 64, 3, cfg, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv1d(64, 128, 3, cfg, vb.pp("conv2"))?;
        let conv3 = candle_nn::conv1d(128, out_channels, 3, cfg, vb.pp("conv3"))?;

        Ok(Self { conv1, conv2, conv3 })
    }

    /// Forward pass through the multi-scale frontend.
    ///
    /// # Arguments
    /// * `mel` - Input mel spectrogram tensor [B, in_channels, T]
    ///
    /// # Returns
    /// * Output feature tensor [B, out_channels, T]
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(mel)?;
        let x = x.relu()?;

        let x = self.conv2.forward(&x)?;
        let x = x.relu()?;

        let x = self.conv3.forward(&x)?;
        Ok(x.relu()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frontend_output_shape() -> Result<()> {
        let dev = Device::Cpu;
        let frontend = MultiScaleFrontend::new(80, 256, &dev)?;
        let mel = Tensor::zeros(&[1, 80, 200], DType::F32, &dev)?;
        let out = frontend.forward(&mel)?;

        assert_eq!(out.dim(0)?, 1);
        assert_eq!(out.dim(1)?, 256);
        assert!(out.dim(2)? > 0);

        Ok(())
    }

    #[test]
    fn test_frontend_batch_processing() -> Result<()> {
        let dev = Device::Cpu;
        let frontend = MultiScaleFrontend::new(80, 256, &dev)?;
        let mel = Tensor::zeros(&[4, 80, 200], DType::F32, &dev)?;
        let out = frontend.forward(&mel)?;

        assert_eq!(out.dims(), &[4, 256, 200]);

        Ok(())
    }

    #[test]
    fn test_frontend_variable_length() -> Result<()> {
        let dev = Device::Cpu;
        let frontend = MultiScaleFrontend::new(80, 256, &dev)?;

        // Test with different temporal lengths
        for &t in &[100, 200, 500] {
            let mel = Tensor::zeros(&[2, 80, t], DType::F32, &dev)?;
            let out = frontend.forward(&mel)?;
            assert_eq!(out.dim(0)?, 2);
            assert_eq!(out.dim(1)?, 256);
            assert_eq!(out.dim(2)?, t);
        }

        Ok(())
    }

    #[test]
    fn test_frontend_random_input() -> Result<()> {
        let dev = Device::Cpu;
        let frontend = MultiScaleFrontend::new(80, 256, &dev)?;
        let mel = Tensor::randn(0.0f32, 1.0, &[2, 80, 100], &dev)?;
        let out = frontend.forward(&mel)?;

        assert_eq!(out.dims(), &[2, 256, 100]);

        Ok(())
    }
}
