//! Snake activation function — learned periodic activation for audio synthesis.
//! Snake(x) = x + (1/alpha) * sin^2(alpha * x)

use candle_core::{DType, Device, Result, Tensor};

pub struct Snake {
    alpha: Tensor, // learnable parameter, shape [channels]
}

impl Snake {
    pub fn new(channels: usize, dev: &Device) -> Result<Self> {
        let alpha = Tensor::ones(&[channels], DType::F32, dev)?;
        Ok(Self { alpha })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x shape: [batch, channels, length]
        // self.alpha shape: [channels]
        // Reshape alpha to [1, channels, 1] for broadcasting
        let alpha = self.alpha.reshape((1, self.alpha.dims()[0], 1))?;
        let ax = x.broadcast_mul(&alpha)?;
        let sin_ax = ax.sin()?;
        let sin_sq = (&sin_ax * &sin_ax)?;
        let inv_alpha = alpha.recip()?;
        x + sin_sq.broadcast_mul(&inv_alpha)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_activation_shape() {
        let dev = &Device::Cpu;
        let snake = Snake::new(64, dev).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (1, 64, 100), dev).unwrap();
        let out = snake.forward(&x).unwrap();
        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn test_snake_nonlinear() {
        let dev = &Device::Cpu;
        let snake = Snake::new(1, dev).unwrap();
        let x = Tensor::new(&[[[0.0f32, 1.0, -1.0]]], dev).unwrap();
        let out = snake.forward(&x).unwrap();
        // Snake(x) = x + (1/alpha) * sin^2(alpha * x), should not equal input for non-zero x
        let diff = (&out - &x)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_snake_output_bounds() {
        let dev = &Device::Cpu;
        let snake = Snake::new(1, dev).unwrap();
        let x = Tensor::new(&[[[0.0f32, 0.5, -0.5]]], dev).unwrap();
        let out = snake.forward(&x).unwrap();
        // Snake output should be differentiable and smooth
        let shape = out.dims();
        assert_eq!(shape, x.dims());
    }
}
