//! Quantization utilities for on-device Sonata deployment.
//!
//! Provides 4-bit and 8-bit quantization for reducing model size
//! while maintaining acceptable quality. Used post-training for
//! inference-only deployment on edge devices.

use candle_core::{Result, Tensor};

/// Quantization configuration.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Number of bits for quantization (4 or 8).
    pub bits: u8,
    /// Group size for grouped quantization (default: 128).
    pub group_size: usize,
    /// Whether to use symmetric quantization.
    pub symmetric: bool,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            symmetric: true,
        }
    }
}

/// Quantization statistics for a tensor.
#[derive(Debug, Clone)]
pub struct QuantStats {
    pub original_size_bytes: usize,
    pub quantized_size_bytes: usize,
    pub compression_ratio: f32,
}

/// Compute quantization statistics for a tensor.
pub fn compute_quant_stats(tensor: &Tensor, config: &QuantConfig) -> Result<QuantStats> {
    let numel = tensor.elem_count();
    let original_size = numel * 4; // f32 = 4 bytes
    let bits_per_element = config.bits as usize;
    // Quantized: bits per element + scale/zero per group
    let num_groups = (numel + config.group_size - 1) / config.group_size;
    let quantized_size = (numel * bits_per_element + 7) / 8 + num_groups * 4; // 4 bytes per scale
    let compression_ratio = original_size as f32 / quantized_size as f32;

    Ok(QuantStats {
        original_size_bytes: original_size,
        quantized_size_bytes: quantized_size,
        compression_ratio,
    })
}

/// Compute the quantization range for a tensor.
pub fn compute_range(tensor: &Tensor) -> Result<(f32, f32)> {
    let min_val: f32 = tensor.min(0)?.min(0)?.to_scalar()?;
    let max_val: f32 = tensor.max(0)?.max(0)?.to_scalar()?;
    Ok((min_val, max_val))
}

/// Estimate total model compression from f32 to target quantization.
pub fn estimate_model_compression(total_params: usize, config: &QuantConfig) -> QuantStats {
    let original_size = total_params * 4;
    let num_groups = (total_params + config.group_size - 1) / config.group_size;
    let quantized_size = (total_params * config.bits as usize + 7) / 8 + num_groups * 4;
    let compression_ratio = original_size as f32 / quantized_size as f32;
    QuantStats {
        original_size_bytes: original_size,
        quantized_size_bytes: quantized_size,
        compression_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_quant_config_default() {
        let config = QuantConfig::default();
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
        assert!(config.symmetric);
    }

    #[test]
    fn test_quant_stats_4bit() {
        let config = QuantConfig::default();
        // 1M params: 4MB f32 → ~0.5MB 4-bit
        let stats = estimate_model_compression(1_000_000, &config);
        assert_eq!(stats.original_size_bytes, 4_000_000);
        assert!(stats.compression_ratio > 6.0);
        assert!(stats.compression_ratio < 9.0);
    }

    #[test]
    fn test_quant_stats_8bit() {
        let config = QuantConfig {
            bits: 8,
            ..Default::default()
        };
        let stats = estimate_model_compression(1_000_000, &config);
        assert!(stats.compression_ratio > 3.0);
        assert!(stats.compression_ratio < 5.0);
    }

    #[test]
    fn test_compute_quant_stats_tensor() {
        let dev = Device::Cpu;
        let tensor = Tensor::randn(0.0f32, 1.0, &[256, 512], &dev).unwrap();
        let config = QuantConfig::default();
        let stats = compute_quant_stats(&tensor, &config).unwrap();
        assert!(stats.compression_ratio > 5.0);
    }

    #[test]
    fn test_sonata_v2_total_compression() {
        let config = QuantConfig::default();
        // ~260M total params
        let stats = estimate_model_compression(260_000_000, &config);
        // f32: ~1GB, 4-bit: ~130MB
        println!(
            "Sonata v2 at 4-bit: {} MB → {} MB ({}x compression)",
            stats.original_size_bytes / 1_000_000,
            stats.quantized_size_bytes / 1_000_000,
            stats.compression_ratio
        );
        assert!(stats.quantized_size_bytes < 200_000_000); // Under 200MB
    }

    #[test]
    fn test_compute_range() {
        let dev = Device::Cpu;
        let tensor = Tensor::new(&[[-1.0f32, 2.0], [0.5, -3.5]], &dev).unwrap();
        let (min_val, max_val) = compute_range(&tensor).unwrap();
        assert!((min_val - (-3.5)).abs() < 0.01);
        assert!((max_val - 2.0).abs() < 0.01);
    }
}
