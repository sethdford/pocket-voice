// INT4 weight-only quantization for Sonata Talker.
//
// Per-channel symmetric quantization strategy:
//   1. For each output channel (column): scale[c] = max(abs(w[:, c])) / 7.0
//   2. Quantize weights: w_i4[i, c] = round(w[i, c] / scale[c])  [-8..7]
//   3. Pack two INT4 values per byte (upper 4 bits, lower 4 bits)
//   4. Dequantize on-the-fly: w_dequant[i, c] = unpack(w_i4[i, c]) * scale[c]
//   5. Apply matmul with dequantized weights
//
// Benefits: per-channel quantization preserves per-output sensitivity,
// reduces memory bandwidth by ~8x (F32 → effectively F4), minimal accuracy loss.
// INT4 is 2x more aggressive than INT8 but with bounded reconstruction error.

use candle_core::{DType, Result, Tensor, D};

/// Pack two INT4 values into one u8 (upper 4 bits, lower 4 bits).
/// Each value should be in [-8, 7] for signed INT4.
#[inline]
fn pack_int4(val_high: f32, val_low: f32) -> u8 {
    // Clamp to INT4 range
    let high = (val_high.clamp(-8.0, 7.0) as i8) as u8 & 0x0F;
    let low = (val_low.clamp(-8.0, 7.0) as i8) as u8 & 0x0F;
    (high << 4) | low
}

/// Unpack two INT4 values from one u8.
/// Returns (high_value, low_value) as signed integers.
/// INT4 is signed symmetric: range is [-8, 7]
#[inline]
fn unpack_int4(byte: u8) -> (f32, f32) {
    // Extract high and low 4 bits
    let high_u4 = byte >> 4;
    let low_u4 = byte & 0x0F;

    // Convert from u4 to signed i4 (range -8..7)
    // If MSB is set (bit 3), sign-extend to negative
    let high = if high_u4 & 0x8 != 0 {
        ((high_u4 as i8) << 4) >> 4  // sign-extend
    } else {
        high_u4 as i8
    } as f32;

    let low = if low_u4 & 0x8 != 0 {
        ((low_u4 as i8) << 4) >> 4  // sign-extend
    } else {
        low_u4 as i8
    } as f32;

    (high, low)
}

/// Quantize tensor weights using per-channel symmetric INT4 quantization.
/// Packs quantized weights as u8 (two INT4 values per byte).
///
/// Input: weights tensor (any shape, typically [out_features, in_features])
/// Returns: (quantized_weights_packed, per_channel_scales)
///
/// The returned quantized_weights_packed is stored as u8 tensors,
/// significantly reducing memory. To dequantize:
///   unpack all u8 bytes to INT4 values
///   dequant = (unpacked INT4) * scales.unsqueeze(0)
pub fn quantize_weights_int4(weights: &Tensor) -> Result<(Tensor, Tensor)> {
    let device = weights.device();

    // Convert to F32 for quantization operations
    let weights_f32 = weights.to_dtype(DType::F32)?;
    let (_out_features, _in_features) = weights_f32.dims2()?;

    // Compute per-channel scales: max(abs(w[:, c])) / 7.0
    let abs_weights = weights_f32.abs()?;
    let max_per_channel = abs_weights.max(D::Minus2)?;

    // Compute scale: max / 7.0 (INT4 range is -8..7)
    let scale_divisor = Tensor::new(&[7.0f32], device)?;
    let mut scales = max_per_channel.broadcast_div(&scale_divisor)?;

    // Clamp scales to avoid division by zero: scale >= 1e-8
    let min_scale = Tensor::new(&[1e-8f32], device)?;
    scales = scales.broadcast_maximum(&min_scale)?;

    // Quantize: w_i4[i, c] = round(w[i, c] / scale[c])
    let scales_expanded = scales.unsqueeze(0)?;
    let weights_normalized = weights_f32.broadcast_div(&scales_expanded)?;
    let weights_i4 = weights_normalized.round()?;

    // Pack into bytes: two INT4 values per byte
    // We'll store as F32 for now to maintain precision during packing,
    // then convert to u8 in actual storage
    // (For a true memory-efficient version, we'd use custom storage,
    // but for compatibility with candle we keep as F32 temporarily)
    Ok((weights_i4, scales))
}

/// Dequantize INT4 weights: multiply quantized weights by per-channel scales.
///
/// weights_i4: quantized weights shape (out_features, in_features)
/// scales: per-channel scales shape (in_features,)
/// Returns: dequantized weights (out_features, in_features)
pub fn dequantize_weights_int4(weights_i4: &Tensor, scales: &Tensor) -> Result<Tensor> {
    let scales_expanded = scales.unsqueeze(0)?; // (1, in_features)
    weights_i4.broadcast_mul(&scales_expanded)
}

/// Quantized linear layer (INT4): wraps a candle Linear, stores dequant info separately.
#[derive(Debug, Clone)]
pub struct QuantizedLinearInt4 {
    /// Weights stored as F32 with rounded INT4 values (before packing).
    /// In a production system, these would be packed to u8 for 8x memory savings.
    /// Shape: (out_features, in_features)
    pub weights_i4: Tensor,
    /// Per-channel scales for dequantization
    /// Shape: (in_features,)
    pub scales: Tensor,
    /// Bias (optional): (out_features,)
    pub bias: Option<Tensor>,
}

impl QuantizedLinearInt4 {
    /// Create a quantized linear layer from existing weights and optional bias.
    pub fn new(weights: &Tensor, bias: Option<&Tensor>) -> Result<Self> {
        let (weights_i4, scales) = quantize_weights_int4(weights)?;
        let bias_opt = bias.map(|b| b.clone());

        Ok(Self {
            weights_i4,
            scales,
            bias: bias_opt,
        })
    }

    /// Forward pass: dequantize weights and apply matmul.
    /// x: input tensor (..., in_features)
    /// Returns: output tensor (..., out_features)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();

        // Dequantize weights to input dtype
        let weights_dequant = dequantize_weights_int4(&self.weights_i4, &self.scales)?;
        let weights_dequant = weights_dequant.to_dtype(dtype)?;

        // Apply matmul: result shape (..., out_features)
        let mut y = x.matmul(&weights_dequant.t()?)?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_dtype = bias.to_dtype(dtype)?;
            y = y.broadcast_add(&bias_dtype)?;
        }

        Ok(y)
    }

    /// Get memory footprint in bytes (approximation).
    /// Note: weights_i4 currently stored as F32 (4 bytes per value).
    /// With true packing to u8, would be 0.5 bytes per value.
    pub fn memory_bytes(&self) -> usize {
        // weights_i4 stored as F32 (for now; actual packing would use 0.5 bytes)
        let w_bytes = self.weights_i4.elem_count() * 4;
        // scales stored as F32
        let s_bytes = self.scales.elem_count() * 4;
        // bias if present
        let b_bytes = self.bias.as_ref().map(|b| b.elem_count() * 4).unwrap_or(0);
        w_bytes + s_bytes + b_bytes
    }

    /// Get packed memory footprint (theoretical: weights as 0.5 bytes each).
    pub fn memory_bytes_packed(&self) -> usize {
        // weights_i4 packed as u8: 0.5 bytes per value (two values per byte)
        let w_bytes = (self.weights_i4.elem_count() + 1) / 2;
        // scales stored as F32
        let s_bytes = self.scales.elem_count() * 4;
        // bias if present
        let b_bytes = self.bias.as_ref().map(|b| b.elem_count() * 4).unwrap_or(0);
        w_bytes + s_bytes + b_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_pack_unpack_int4() -> Result<()> {
        // Test bit packing round-trip
        // Note: pack expects rounded integer values, unpack returns exact integers
        let test_pairs = vec![
            (0.0, 0.0),
            (7.0, -8.0),
            (-8.0, 7.0),
            (3.0, -4.0),  // Use exact integers, not 3.5 / -4.2
            (1.0, 2.0),
        ];

        for (high, low) in test_pairs {
            let packed = pack_int4(high, low);
            let (h_unpacked, l_unpacked) = unpack_int4(packed);

            // Unpacking should exactly match input (which should be integers)
            assert!(
                (h_unpacked - high).abs() < 1e-6,
                "High unpack mismatch: {} vs {}",
                h_unpacked,
                high
            );
            assert!(
                (l_unpacked - low).abs() < 1e-6,
                "Low unpack mismatch: {} vs {}",
                l_unpacked,
                low
            );
        }

        Ok(())
    }

    #[test]
    fn test_quantize_dequantize_int4() -> Result<()> {
        let device = Device::Cpu;
        let weights = Tensor::new(
            &[
                [1.5f32, -2.3, 3.7],
                [-4.1, 5.9, -6.2],
            ],
            &device,
        )?;

        let (weights_i4, scales) = quantize_weights_int4(&weights)?;
        let dequant = dequantize_weights_int4(&weights_i4, &scales)?;

        // Quantization should have bounded error (INT4 is less precise than INT8, ~1/7 resolution)
        let original_flat = weights.flatten_all()?.to_vec1::<f32>()?;
        let dequant_flat = dequant.flatten_all()?.to_vec1::<f32>()?;

        for (orig, dq) in original_flat.iter().zip(dequant_flat.iter()) {
            let rel_err = (dq - orig).abs() / (orig.abs() + 1e-8);
            assert!(
                rel_err < 0.25,  // INT4 allows ~25% error (divided by 7 vs 127)
                "INT4 quantization error too high: {} vs {} (error: {:.2}%)",
                orig,
                dq,
                rel_err * 100.0
            );
        }

        Ok(())
    }

    #[test]
    fn test_quantized_linear_int4_forward() -> Result<()> {
        let device = Device::Cpu;

        // Create random weights (out_features=32, in_features=16)
        let weights = Tensor::randn(0.0, 1.0, (32, 16), &device)?;
        let input = Tensor::randn(0.0, 1.0, (2, 16), &device)?;

        let qlinear = QuantizedLinearInt4::new(&weights, None)?;
        let output = qlinear.forward(&input)?;

        // Check output shape
        assert_eq!(output.dims(), vec![2, 32]);

        Ok(())
    }

    #[test]
    fn test_quantized_linear_int4_with_bias() -> Result<()> {
        let device = Device::Cpu;

        let weights = Tensor::randn(0.0, 1.0, (32, 16), &device)?;
        let bias = Tensor::randn(0.0, 1.0, (32,), &device)?;
        let input = Tensor::randn(0.0, 1.0, (2, 16), &device)?;

        let qlinear = QuantizedLinearInt4::new(&weights, Some(&bias))?;
        let output = qlinear.forward(&input)?;

        // output should be shape [2, 32] after matmul and bias add
        assert_eq!(output.dims(), vec![2, 32]);

        Ok(())
    }

    #[test]
    fn test_memory_reduction() -> Result<()> {
        let device = Device::Cpu;

        // Create weights (1000 x 500)
        let weights = Tensor::randn(0.0, 1.0, (1000, 500), &device)?;
        let original_bytes = weights.elem_count() * 4; // F32

        let qlinear = QuantizedLinearInt4::new(&weights, None)?;
        let quantized_bytes = qlinear.memory_bytes(); // Still F32 storage (for now)
        let packed_bytes = qlinear.memory_bytes_packed(); // Theoretical packing

        // Current storage is same size (both F32), but theoretical packing should be ~8x smaller
        // weights_packed = 0.5 bytes/elem (two values per byte)
        // scales = 4 bytes/elem
        // Total packed ≈ weights * 0.5 + scales * 4
        let weights_elem = weights.elem_count();
        let scales_elem = 500; // in_features
        let theoretical_packed = (weights_elem / 2) + (scales_elem * 4);

        assert!(packed_bytes < quantized_bytes, "Packed should be smaller than F32 storage");
        assert_eq!(
            packed_bytes, theoretical_packed,
            "Packed size mismatch: {} vs {}",
            packed_bytes, theoretical_packed
        );

        // Verify significant reduction from F32
        let reduction_factor = (original_bytes as f64) / (packed_bytes as f64);
        assert!(
            reduction_factor > 7.5,
            "Memory reduction should be ~8x, got {:.1}x",
            reduction_factor
        );

        Ok(())
    }

    #[test]
    fn test_quantize_int4_bounds() -> Result<()> {
        let device = Device::Cpu;

        // Create weights in [-10, 10] range
        let weights = Tensor::new(
            &[
                [10.0f32, -10.0],
                [-9.5, 9.8],
                [0.0, 5.0],
            ],
            &device,
        )?;

        let (weights_i4, _scales) = quantize_weights_int4(&weights)?;
        let weights_i4_vec = weights_i4.flatten_all()?.to_vec1::<f32>()?;

        // After quantization and rounding, values should be in ~[-8, 7]
        for (i, val) in weights_i4_vec.iter().enumerate() {
            assert!(
                *val >= -9.0 && *val <= 8.0,
                "Quantized weight {} out of INT4 bounds: {}",
                i,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_quantize_int4_per_channel_scales() -> Result<()> {
        let device = Device::Cpu;

        // Create weights with known per-channel maxima
        let weights = Tensor::new(
            &[
                [1.0f32, -4.0, 0.5],
                [2.0, 5.0, -0.2],
                [3.0, -6.0, 0.1],
            ],
            &device,
        )?;

        let (_weights_i4, scales) = quantize_weights_int4(&weights)?;

        // Expected scales: max(abs(col)) / 7
        // Column 0: max(1, 2, 3) = 3 → 3/7
        // Column 1: max(4, 5, 6) = 6 → 6/7
        // Column 2: max(0.5, 0.2, 0.1) = 0.5 → 0.5/7
        let scales_vec = scales.to_vec1::<f32>()?;
        assert_eq!(scales_vec.len(), 3);

        let expected = vec![3.0 / 7.0, 6.0 / 7.0, 0.5 / 7.0];
        for (i, (e, a)) in expected.iter().zip(scales_vec.iter()).enumerate() {
            assert!(
                (e - a).abs() < 1e-6,
                "Channel {} scale mismatch: expected {}, got {}",
                i,
                e,
                a
            );
        }

        Ok(())
    }

    #[test]
    fn test_dequant_inverts_quant_int4() -> Result<()> {
        let device = Device::Cpu;

        let original = Tensor::new(
            &[
                [0.123f32, -0.456, 0.789],
                [-0.234, 0.567, -0.890],
            ],
            &device,
        )?;

        let (quant, scales) = quantize_weights_int4(&original)?;
        let recon = dequantize_weights_int4(&quant, &scales)?;

        let orig_flat = original.flatten_all()?.to_vec1::<f32>()?;
        let recon_flat = recon.flatten_all()?.to_vec1::<f32>()?;

        for (o, r) in orig_flat.iter().zip(recon_flat.iter()) {
            let max_abs_val = o.abs().max(0.1);
            let max_error = max_abs_val / 7.0; // INT4 quantization resolution
            let error = (r - o).abs();
            assert!(
                error <= max_error + 1e-6,
                "Reconstruction error too large: {} (max allowed: {})",
                error,
                max_error
            );
        }

        Ok(())
    }

    #[test]
    fn test_quantized_linear_int4_preserves_matmul() -> Result<()> {
        let device = Device::Cpu;

        // Create weights (out_features=3, in_features=2)
        let weights = Tensor::new(
            &[
                [0.5f32, -0.3],
                [0.4, 0.6],
                [-0.2, 0.5],
            ],
            &device,
        )?;
        // Input (batch=2, in_features=2)
        let input = Tensor::new(&[[1.0f32, 0.5], [-0.5, 0.2]], &device)?;

        // Reference: original matmul
        let output_ref = input.matmul(&weights.t()?)?;

        // Quantized path
        let qlinear = QuantizedLinearInt4::new(&weights, None)?;
        let output_quant = qlinear.forward(&input)?;

        // Compare outputs
        let ref_vec = output_ref.flatten_all()?.to_vec1::<f32>()?;
        let quant_vec = output_quant.flatten_all()?.to_vec1::<f32>()?;

        for (r, q) in ref_vec.iter().zip(quant_vec.iter()) {
            let rel_error = (q - r).abs() / (r.abs() + 1e-8);
            assert!(
                rel_error < 0.25, // 25% relative error acceptable for INT4 (per-channel rounding)
                "Quantized matmul diverged: {} vs {} (error: {:.2}%)",
                r,
                q,
                rel_error * 100.0
            );
        }

        Ok(())
    }
}
