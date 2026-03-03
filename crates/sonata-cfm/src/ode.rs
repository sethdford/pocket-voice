//! Euler ODE solver for straight-line flow matching (F5-TTS/CosyVoice style).

use candle_core::{Result, Tensor};

/// Euler ODE solver: integrates velocity field from t=0 to t=1.
///
/// # Arguments
/// * `v_fn` - velocity function v(x, t) -> dx/dt
/// * `x0` - initial noise [B, C, T]
/// * `steps` - number of Euler steps (typically 4-8)
pub fn euler_solve<F>(v_fn: F, x0: &Tensor, steps: usize) -> Result<Tensor>
where
    F: Fn(&Tensor, f32) -> Result<Tensor>,
{
    let dt = 1.0 / steps as f32;
    let mut x = x0.clone();
    for step in 0..steps {
        let t = step as f32 * dt;
        let v = v_fn(&x, t)?;
        x = (&x + &(v * dt as f64)?)?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_euler_identity() {
        // With constant velocity v=1, x(1) = x(0) + 1
        let dev = Device::Cpu;
        let x0 = Tensor::zeros(&[1, 80, 50], DType::F32, &dev).unwrap();
        let result = euler_solve(
            |_x, _t| Tensor::ones(&[1, 80, 50], DType::F32, &Device::Cpu),
            &x0, 8
        ).unwrap();
        // After 8 steps of dt=0.125, should be near 1.0
        let mean: f32 = result.mean_all().unwrap().to_scalar().unwrap();
        assert!((mean - 1.0).abs() < 0.01, "Expected ~1.0, got {}", mean);
    }

    #[test]
    fn test_euler_converges() {
        let dev = Device::Cpu;
        let x0 = Tensor::zeros(&[1, 80, 50], DType::F32, &dev).unwrap();
        let x1 = Tensor::ones(&[1, 80, 50], DType::F32, &dev).unwrap();
        // Straight-line: v(x,t) = x1 - x0
        let result = euler_solve(
            |_x, _t| {
                let target = Tensor::ones(&[1, 80, 50], DType::F32, &Device::Cpu).unwrap();
                let start = Tensor::zeros(&[1, 80, 50], DType::F32, &Device::Cpu).unwrap();
                Ok((&target - &start).unwrap())
            },
            &x0, 8
        ).unwrap();
        let diff: f32 = (&result - &x1).unwrap().abs().unwrap().mean_all().unwrap().to_scalar().unwrap();
        assert!(diff < 0.01, "Expected convergence, got diff={}", diff);
    }

    #[test]
    fn test_euler_step_count() {
        let dev = Device::Cpu;
        let x0 = Tensor::zeros(&[1, 10, 5], DType::F32, &dev).unwrap();
        // More steps should be more accurate
        for steps in [2, 4, 8, 16] {
            let result = euler_solve(
                |_x, _t| Tensor::ones(&[1, 10, 5], DType::F32, &Device::Cpu),
                &x0, steps
            ).unwrap();
            assert_eq!(result.dims(), &[1, 10, 5]);
        }
    }

    #[test]
    fn test_euler_different_shapes() {
        let dev = Device::Cpu;
        // Test with different tensor shapes
        for &(batch, channels, time) in &[(1, 80, 10), (2, 80, 50), (4, 80, 100)] {
            let x0 = Tensor::zeros(&[batch, channels, time], DType::F32, &dev).unwrap();
            let result = euler_solve(
                |x, _t| {
                    // Return constant velocity field matching input shape
                    Tensor::ones(x.shape(), DType::F32, &dev)
                },
                &x0, 8
            ).unwrap();
            assert_eq!(result.dims(), &[batch, channels, time]);
        }
    }

    #[test]
    fn test_euler_preserves_batch_dimension() {
        let dev = Device::Cpu;
        let x0 = Tensor::zeros(&[3, 80, 25], DType::F32, &dev).unwrap();
        let result = euler_solve(
            |_x, _t| Tensor::zeros(&[3, 80, 25], DType::F32, &Device::Cpu),
            &x0, 4
        ).unwrap();
        // With zero velocity, output should be same as input
        assert_eq!(result.dims(), &[3, 80, 25]);
        let diff: f32 = (&result - &x0).unwrap().abs().unwrap().mean_all().unwrap().to_scalar().unwrap();
        assert!(diff < 1e-5, "Expected identity with zero velocity, got diff={}", diff);
    }
}
