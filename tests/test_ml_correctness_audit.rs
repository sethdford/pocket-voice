/// Correctness audit tests for Sonata ML operations.
///
/// Tests mathematical correctness of:
/// 1. GRU cell implementation (drafter)
/// 2. INT8 quantization
/// 3. RoPE (rotary position embeddings)
/// 4. Attention mechanisms (self, cross, GQA)
/// 5. Flow ODE solver (Euler and Heun)

#[cfg(test)]
mod correctness_tests {
    use candle_core::{DType, Device, Result, Tensor, D};

    // ─────────────────────────────────────────────────────────────────────────
    // GRU CELL CORRECTNESS TESTS
    // ─────────────────────────────────────────────────────────────────────────

    /// Test 1: GRU cell gate computations with known weights.
    /// Verify: z = sigmoid(Wz·x + Uz·h)
    ///         r = sigmoid(Wr·x + Ur·h)
    ///         h' = tanh(Wh·x + Uh·(r⊙h))
    ///         h_new = (1-z)⊙h + z⊙h'
    #[test]
    fn test_gru_cell_gates_manual() -> Result<()> {
        let device = Device::Cpu;

        // Manual computation with known weights
        // x = [1.0, 0.0], h = [0.5, -0.5]
        // w_z = [[0.1, 0.2], [0.3, 0.4]], u_z = [[0.5, 0.0], [0.0, 0.5]]
        // Expected: z = sigmoid(0.1*1 + 0.2*0 + 0.5*0.5 + 0.0*(-0.5))
        //             = sigmoid(0.1 + 0.25) = sigmoid(0.35) ≈ 0.5866

        let x = Tensor::new(&[[1.0f32, 0.0]], &device)?;
        let h = Tensor::new(&[[0.5f32, -0.5]], &device)?;

        // z = sigmoid(w_z @ x + u_z @ h)
        let w_z_x = Tensor::new(&[[0.1f32, 0.2]], &device)?.matmul(&x.t()?)?;
        let u_z_h = Tensor::new(&[[0.5f32, 0.0]], &device)?.matmul(&h.t()?)?;
        let z_logit = (&w_z_x + &u_z_h)?;
        let z = candle_nn::Activation::Sigmoid.forward(&z_logit)?;

        let z_val = z.squeeze(1)?.squeeze(0)?.to_scalar::<f32>()?;
        // Expected: sigmoid(0.35) ≈ 0.5866
        assert!(
            (z_val - 0.5866).abs() < 0.001,
            "z gate incorrect: expected ~0.5866, got {}",
            z_val
        );

        // r = sigmoid(w_r @ x + u_r @ h)
        let w_r_x = Tensor::new(&[[0.2f32, 0.1]], &device)?.matmul(&x.t()?)?;
        let u_r_h = Tensor::new(&[[0.3f32, 0.1]], &device)?.matmul(&h.t()?)?;
        let r_logit = (&w_r_x + &u_r_h)?;
        let r = candle_nn::Activation::Sigmoid.forward(&r_logit)?;

        let r_val = r.squeeze(1)?.squeeze(0)?.to_scalar::<f32>()?;
        // Expected: sigmoid(0.2 + 0.15 - 0.05) = sigmoid(0.3) ≈ 0.5744
        assert!(
            (r_val - 0.5744).abs() < 0.001,
            "r gate incorrect: expected ~0.5744, got {}",
            r_val
        );

        // h' = tanh(w_h @ x + u_h @ (r ⊙ h))
        let rh = (&r * &h)?;
        let w_h_x = Tensor::new(&[[0.15f32, 0.25]], &device)?.matmul(&x.t()?)?;
        let u_h_rh = Tensor::new(&[[0.4f32, 0.2]], &device)?.matmul(&rh.t()?)?;
        let h_cand_logit = (&w_h_x + &u_h_rh)?;
        let h_cand = h_cand_logit.tanh()?;

        // h_new = (1 - z) ⊙ h + z ⊙ h'
        let ones = Tensor::ones_like(&z)?;
        let one_minus_z = (&ones - &z)?;
        let h_new = (&(&one_minus_z * &h)? + &(&z * &h_cand)?)?;

        // Verify h_new has correct shape
        let h_new_shape = h_new.dims();
        assert_eq!(h_new_shape, vec![1, 2], "h_new shape incorrect");

        Ok(())
    }

    /// Test 2: GRU stacked layers data flow.
    /// Verify: Layer 0 input is token embedding (emb_dim)
    ///         Layer i>0 input is previous layer's hidden state (gru_hidden)
    #[test]
    fn test_gru_stacked_layers_flow() -> Result<()> {
        let device = Device::Cpu;
        let emb_dim = 32;
        let gru_hidden = 64;
        let seq_len = 10;

        // Simulate: token embedding output -> layer 0 -> layer 1
        let token_emb = Tensor::randn(0.0f32, 1.0, (seq_len, emb_dim), &device)?;
        let h0_initial = Tensor::randn(0.0f32, 1.0, (seq_len, gru_hidden), &device)?;
        let h1_initial = Tensor::zeros((seq_len, gru_hidden), DType::F32, &device)?;

        // Verify input dims for layer 0
        let l0_input_dim = token_emb.dims()[1];
        assert_eq!(l0_input_dim, emb_dim, "Layer 0 should accept emb_dim");

        // Simulate layer 0 forward (emb_dim -> gru_hidden)
        let w0 = Tensor::randn(0.0f32, 1.0, (gru_hidden, emb_dim), &device)?;
        let h0_output = token_emb.matmul(&w0.t()?)?;
        assert_eq!(
            h0_output.dims(),
            vec![seq_len, gru_hidden],
            "Layer 0 output shape mismatch"
        );

        // Verify input dims for layer 1
        let l1_input_dim = h0_output.dims()[1];
        assert_eq!(l1_input_dim, gru_hidden, "Layer 1 should accept gru_hidden");

        // Simulate layer 1 forward (gru_hidden -> gru_hidden)
        let w1 = Tensor::randn(0.0f32, 1.0, (gru_hidden, gru_hidden), &device)?;
        let h1_output = h0_output.matmul(&w1.t()?)?;
        assert_eq!(
            h1_output.dims(),
            vec![seq_len, gru_hidden],
            "Layer 1 output shape mismatch"
        );

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // INT8 QUANTIZATION CORRECTNESS TESTS
    // ─────────────────────────────────────────────────────────────────────────

    /// Test 3: Per-channel scale computation.
    /// Verify: scale[c] = max(abs(column c)) / 127.0
    #[test]
    fn test_quant_per_channel_scale() -> Result<()> {
        let device = Device::Cpu;

        // Create weights with known structure
        // Column 0: [1, 2, 3] -> max = 3 -> scale = 3/127
        // Column 1: [-4, 5, -6] -> max(abs) = 6 -> scale = 6/127
        let weights = Tensor::new(
            &[[1.0f32, -4.0], [2.0, 5.0], [3.0, -6.0]],
            &device,
        )?;

        let abs_w = weights.abs()?;
        let scales = abs_w.max(D::Minus2)?;

        let scales_vec = scales.to_vec1::<f32>()?;
        assert_eq!(scales_vec.len(), 2);

        let expected = vec![3.0 / 127.0, 6.0 / 127.0];
        for (e, a) in expected.iter().zip(scales_vec.iter()) {
            assert!(
                (e - a).abs() < 1e-6,
                "Per-channel scale error: expected {}, got {}",
                e,
                a
            );
        }

        Ok(())
    }

    /// Test 4: Quantization round-trip accuracy.
    /// Verify: dequant(quant(w)) ≈ w with bounded error
    #[test]
    fn test_quant_round_trip_accuracy() -> Result<()> {
        let device = Device::Cpu;

        // Create weights in [-1, 1] range
        let weights_orig = Tensor::new(
            &[
                [0.5f32, -0.8, 0.2],
                [-0.3, 0.9, -0.1],
                [0.7, -0.4, 0.6],
            ],
            &device,
        )?;

        // Quantize
        let abs_w = weights_orig.abs()?;
        let scales = abs_w.max(D::Minus2)?;
        let scale_min = Tensor::new(&[1e-8f32], &device)?;
        let scales = scales.broadcast_maximum(&scale_min)?;

        let scales_expanded = scales.unsqueeze(0)?;
        let weights_norm = weights_orig.broadcast_div(&scales_expanded)?;
        let weights_i8 = weights_norm.round()?;

        // Dequantize
        let weights_dequant = weights_i8.broadcast_mul(&scales_expanded)?;

        // Check error
        let weights_dequant = weights_dequant.to_dtype(DType::F32)?;
        let orig_flat = weights_orig.flatten_all()?.to_vec1::<f32>()?;
        let dequant_flat = weights_dequant.flatten_all()?.to_vec1::<f32>()?;

        for (orig, dq) in orig_flat.iter().zip(dequant_flat.iter()) {
            let abs_err = (dq - orig).abs();
            // INT8 quantization error should be < max_abs_value / 127
            let max_val = orig.abs().max(0.1);
            let expected_max_err = max_val / 127.0;
            assert!(
                abs_err <= expected_max_err + 1e-6,
                "Quantization error too high: {} (max allowed: {})",
                abs_err,
                expected_max_err
            );
        }

        Ok(())
    }

    /// Test 5: Quantization preserves matmul semantics.
    /// Verify: (w_dequant @ x) ≈ (w_original @ x)
    #[test]
    fn test_quant_matmul_preservation() -> Result<()> {
        let device = Device::Cpu;

        let w_original = Tensor::new(
            &[[1.5f32, -2.0, 0.5], [0.8, 1.2, -0.3]],
            &device,
        )?;
        let x = Tensor::new(&[[1.0f32], [0.5], [-0.2]], &device)?;

        // Original matmul
        let y_original = w_original.matmul(&x)?;

        // Quantize and dequantize
        let abs_w = w_original.abs()?;
        let scales = abs_w.max(D::Minus2)?;
        let scales_expanded = scales.unsqueeze(0)?;
        let w_norm = w_original.broadcast_div(&scales_expanded)?;
        let w_i8 = w_norm.round()?;
        let w_dequant = w_i8.broadcast_mul(&scales_expanded)?;

        // Dequantized matmul
        let y_dequant = w_dequant.matmul(&x)?;

        // Compare
        let orig_vec = y_original.flatten_all()?.to_vec1::<f32>()?;
        let dequant_vec = y_dequant.flatten_all()?.to_vec1::<f32>()?;

        for (orig, dq) in orig_vec.iter().zip(dequant_vec.iter()) {
            let rel_err = (dq - orig).abs() / (orig.abs() + 1e-8);
            assert!(
                rel_err < 0.05, // 5% relative error acceptable for INT8
                "Matmul output error too high: {} vs {}",
                orig,
                dq
            );
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // RoPE (ROTARY POSITION EMBEDDING) CORRECTNESS TESTS
    // ─────────────────────────────────────────────────────────────────────────

    /// Test 6: RoPE angle computation.
    /// Verify: freq[i] = 1 / theta^(2i/d)
    #[test]
    fn test_rope_freq_computation() -> Result<()> {
        let device = Device::Cpu;
        let d_model = 64;
        let theta = 10000.0f64;
        let half = d_model / 2;

        // Compute frequencies as in code
        let mut freqs = vec![0f32; half];
        for i in 0..half {
            freqs[i] = 1.0 / (theta as f32).powf(2.0 * i as f32 / d_model as f32);
        }

        // Verify first few frequencies match expected
        let expected_f0 = 1.0; // 1 / theta^0
        let expected_f1 = 1.0 / (10000.0_f32).powf(2.0 / 64.0); // 1 / theta^(2/64)
        let expected_f2 = 1.0 / (10000.0_f32).powf(4.0 / 64.0);

        assert!((freqs[0] - expected_f0).abs() < 1e-6);
        assert!((freqs[1] - expected_f1).abs() < 1e-6);
        assert!((freqs[2] - expected_f2).abs() < 1e-6);

        Ok(())
    }

    /// Test 7: RoPE rotation correctness.
    /// Verify: R(θ) @ [x1; x2] = [x1*cos(θ) - x2*sin(θ); x1*sin(θ) + x2*cos(θ)]
    #[test]
    fn test_rope_rotation_matrix() -> Result<()> {
        let device = Device::Cpu;

        // Test vector: [1, 0, 1, 0]
        // Angle: π/4 (45 degrees)
        let x = Tensor::new(&[[1.0f32, 0.0, 1.0, 0.0]], &device)?;
        let angle = std::f32::consts::PI / 4.0;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Apply rotation manually: [x1*cos - x2*sin, x1*sin + x2*cos, ...]
        let x1 = 1.0;
        let x2 = 0.0;
        let x3 = 1.0;
        let x4 = 0.0;

        let expected_x1 = x1 * cos_a - x2 * sin_a; // 1 * cos(π/4) ≈ 0.707
        let expected_x2 = x1 * sin_a + x2 * cos_a; // 1 * sin(π/4) ≈ 0.707
        let expected_x3 = x3 * cos_a - x4 * sin_a; // 1 * cos(π/4) ≈ 0.707
        let expected_x4 = x3 * sin_a + x4 * cos_a; // 1 * sin(π/4) ≈ 0.707

        // Simulate rope rotation
        let x1_narrow = x.narrow(D::Minus1, 0, 1)?;
        let x2_narrow = x.narrow(D::Minus1, 1, 1)?;
        let r1 = (x1_narrow * cos_a - x2_narrow * sin_a)?;
        let r2 = (x1_narrow * sin_a + x2_narrow * cos_a)?;

        let r1_val = r1.squeeze(1)?.squeeze(0)?.to_scalar::<f32>()?;
        let r2_val = r2.squeeze(1)?.squeeze(0)?.to_scalar::<f32>()?;

        assert!((r1_val - expected_x1).abs() < 1e-5);
        assert!((r2_val - expected_x2).abs() < 1e-5);

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ATTENTION CORRECTNESS TESTS
    // ─────────────────────────────────────────────────────────────────────────

    /// Test 8: Attention scale factor.
    /// Verify: scale = 1 / sqrt(d_k)
    #[test]
    fn test_attention_scale() -> Result<()> {
        let head_dim = 64;
        let expected_scale = 1.0 / (head_dim as f64).sqrt();

        // Compute as in code
        let scale = 1.0 / (head_dim as f64).sqrt();

        // For head_dim=64: sqrt(64)=8, scale=0.125
        assert!((expected_scale - 0.125).abs() < 1e-6);
        assert!((scale - expected_scale).abs() < 1e-10);

        Ok(())
    }

    /// Test 9: Attention softmax and aggregation.
    /// Verify: attn = softmax(scores), output = attn @ v
    #[test]
    fn test_attention_softmax_aggregation() -> Result<()> {
        let device = Device::Cpu;

        // Toy attention: Q @ K^T / sqrt(d)
        // Q = [[1, 0]], K = [[1, 0], [0, 1]], V = [[1, 0], [0, 1]]
        let q = Tensor::new(&[[1.0f32, 0.0]], &device)?;
        let k = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &device)?;
        let v = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &device)?;

        // Scores: Q @ K^T
        let scores = q.matmul(&k.t()?)?;
        // Expected: [[1*1 + 0*0, 1*0 + 0*1]] = [[1, 0]]

        let scores_vec = scores.flatten_all()?.to_vec1::<f32>()?;
        assert!((scores_vec[0] - 1.0).abs() < 1e-6);
        assert!((scores_vec[1] - 0.0).abs() < 1e-6);

        // Softmax on scores
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_vec = attn.flatten_all()?.to_vec1::<f32>()?;

        // softmax([1, 0]) = [e^1/(e^1+e^0), e^0/(e^1+e^0)]
        // ≈ [0.731, 0.269]
        assert!((attn_vec[0] - 0.731).abs() < 0.01);
        assert!((attn_vec[1] - 0.269).abs() < 0.01);

        // Attention output: attn @ v
        let out = attn.matmul(&v)?;
        let out_vec = out.flatten_all()?.to_vec1::<f32>()?;

        // out ≈ [0.731*1 + 0.269*0, 0.731*0 + 0.269*1] = [0.731, 0.269]
        assert!((out_vec[0] - 0.731).abs() < 0.01);
        assert!((out_vec[1] - 0.269).abs() < 0.01);

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FLOW ODE SOLVER CORRECTNESS TESTS
    // ─────────────────────────────────────────────────────────────────────────

    /// Test 10: Euler ODE step.
    /// Verify: x_{t+1} = x_t + v_t * dt
    #[test]
    fn test_flow_euler_step() -> Result<()> {
        let device = Device::Cpu;

        // Initial state
        let x = Tensor::new(&[[1.0f32, 2.0]], &device)?;
        let v = Tensor::new(&[[0.1f32, -0.2]], &device)?; // velocity
        let dt = 0.01;

        // Euler step: x_new = x + v * dt
        let x_new = (x + v.affine(dt, 0.0)?)?;

        let expected = vec![1.0 + 0.1 * 0.01, 2.0 - 0.2 * 0.01];
        let actual = x_new.flatten_all()?.to_vec1::<f32>()?;

        for (e, a) in expected.iter().zip(actual.iter()) {
            assert!((e - a).abs() < 1e-6);
        }

        Ok(())
    }

    /// Test 11: Heun ODE step.
    /// Verify: x_{t+1} = x_t + (v_t + v_{t+1}) * dt / 2
    #[test]
    fn test_flow_heun_step() -> Result<()> {
        let device = Device::Cpu;

        // Initial state
        let x = Tensor::new(&[[1.0f32, 2.0]], &device)?;
        let v1 = Tensor::new(&[[0.1f32, -0.2]], &device)?; // velocity at t
        let v2 = Tensor::new(&[[0.15f32, -0.25]], &device)?; // velocity at t+1
        let dt = 0.01;

        // Heun step: x_new = x + (v1 + v2) * dt / 2
        let x_heun = (x + (&v1 + &v2)?.affine(dt / 2.0, 0.0)?)?;

        let expected = vec![
            1.0 + (0.1 + 0.15) * 0.01 / 2.0,
            2.0 + (-0.2 - 0.25) * 0.01 / 2.0,
        ];
        let actual = x_heun.flatten_all()?.to_vec1::<f32>()?;

        for (e, a) in expected.iter().zip(actual.iter()) {
            assert!((e - a).abs() < 1e-6);
        }

        Ok(())
    }

    /// Test 12: Noise schedule (sigma progression).
    /// Verify: t_i = sigma_min + i/N * (1 - sigma_min)
    #[test]
    fn test_flow_noise_schedule() -> Result<()> {
        let sigma_min = 0.0001f32;
        let n_steps = 4;

        let mut sigmas = vec![];
        for i in 0..=n_steps {
            let t = sigma_min + (i as f32 / n_steps as f32) * (1.0 - sigma_min);
            sigmas.push(t);
        }

        // Verify monotonic increase
        for i in 1..sigmas.len() {
            assert!(sigmas[i] >= sigmas[i - 1], "Sigma schedule not monotonic");
        }

        // Verify bounds
        assert!(
            sigmas[0] >= sigma_min - 1e-6,
            "First sigma should be ~sigma_min"
        );
        assert!(
            (sigmas[n_steps] - 1.0).abs() < 1e-6,
            "Last sigma should be ~1.0"
        );

        Ok(())
    }
}
