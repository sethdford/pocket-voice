//! Metal GPU device selection utilities.
//!
//! Provides helpers for selecting the best available compute device
//! (Metal GPU on macOS, CPU fallback elsewhere).

use candle_core::Device;

/// Select the best available device for computation.
///
/// On macOS with Metal support compiled in, returns a Metal GPU device.
/// Otherwise falls back to CPU.
pub fn best_device() -> Device {
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(dev) => {
                tracing::info!("Using Metal GPU device");
                dev
            }
            Err(e) => {
                tracing::warn!("Metal GPU not available: {}, falling back to CPU", e);
                Device::Cpu
            }
        }
    }
    #[cfg(not(feature = "metal"))]
    {
        tracing::info!("Metal not compiled in, using CPU");
        Device::Cpu
    }
}

/// Check if Metal GPU is available at runtime.
pub fn is_metal_available() -> bool {
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).is_ok()
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_best_device_returns_device() {
        let dev = best_device();
        // Should always return a valid device (CPU or Metal)
        let _ = candle_core::Tensor::zeros(&[1], candle_core::DType::F32, &dev).unwrap();
    }

    #[test]
    fn test_is_metal_available_returns_bool() {
        let available = is_metal_available();
        // Just verify it returns without panic
        let _ = available;
    }

    #[test]
    fn test_cpu_always_works() {
        let dev = Device::Cpu;
        let t =
            candle_core::Tensor::zeros(&[2, 3], candle_core::DType::F32, &dev).unwrap();
        assert_eq!(t.dims(), &[2, 3]);
    }
}
