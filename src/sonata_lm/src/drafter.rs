// GRU-based speculative decoding draft model (ReDrafter).
// Simple K-step linear speculative decoding using a small GRU.
// Loads pre-trained weights from safetensors and runs K draft steps.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, VarBuilder};

/// Manual GRU cell (no built-in GRU in candle).
///
/// Gate computations:
///   z = sigmoid(w_z(x) + u_z(h))
///   r = sigmoid(w_r(x) + u_r(h))
///   h' = tanh(w_h(x) + u_h(r * h))
///   h_new = (1 - z) * h + z * h'
pub struct GruCell {
    w_z: Linear,
    u_z: Linear,
    w_r: Linear,
    u_r: Linear,
    w_h: Linear,
    u_h: Linear,
}

impl GruCell {
    pub fn load(input_dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w_z: linear_no_bias(input_dim, hidden_dim, vb.pp("w_z"))?,
            u_z: linear_no_bias(hidden_dim, hidden_dim, vb.pp("u_z"))?,
            w_r: linear_no_bias(input_dim, hidden_dim, vb.pp("w_r"))?,
            u_r: linear_no_bias(hidden_dim, hidden_dim, vb.pp("u_r"))?,
            w_h: linear_no_bias(input_dim, hidden_dim, vb.pp("w_h"))?,
            u_h: linear_no_bias(hidden_dim, hidden_dim, vb.pp("u_h"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, h: &Tensor) -> Result<Tensor> {
        let z = (self.w_z.forward(x)? + self.u_z.forward(h)?)?;
        let z = candle_nn::Activation::Sigmoid.forward(&z)?;

        let r = (self.w_r.forward(x)? + self.u_r.forward(h)?)?;
        let r = candle_nn::Activation::Sigmoid.forward(&r)?;

        let rh = (&r * h)?;
        let h_cand = (self.w_h.forward(x)? + self.u_h.forward(&rh)?)?;
        let h_cand = h_cand.tanh()?;

        let ones = Tensor::ones_like(&z)?;
        let one_minus_z = (&ones - &z)?;
        ((&one_minus_z * h)? + (&z * &h_cand)?)
    }
}

/// Configuration for GRU drafter.
#[derive(Debug, Clone)]
pub struct DrafterConfig {
    pub gru_hidden: usize,
    pub gru_layers: usize,
    pub emb_dim: usize,
    pub d_model: usize,
    pub vocab_size: usize,
}

/// GRU-based draft model for ReDrafter speculative decoding.
///
/// Architecture:
///   - hidden_proj: Linear(d_model → gru_hidden)  [projects LM hidden state]
///   - token_emb: Embedding(vocab_size, emb_dim)  [draft-specific embeddings]
///   - gru_cells: 2-layer GRU(emb_dim → gru_hidden)
///   - output_head: Linear(gru_hidden → vocab_size)  [logits]
pub struct GruDrafter {
    hidden_proj: Linear,
    token_emb: Embedding,
    gru_cells: Vec<GruCell>,
    output_head: Linear,
    cfg: DrafterConfig,
    device: Device,
    dtype: DType,
}

impl GruDrafter {
    /// Load GRU drafter from VarBuilder (typically from safetensors).
    /// Expects weight names:
    ///   hidden_proj.weight
    ///   token_emb.weight
    ///   gru.{i}.w_z.weight, gru.{i}.u_z.weight, etc.
    ///   output_head.weight
    pub fn load(cfg: &DrafterConfig, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let hidden_proj = linear_no_bias(cfg.d_model, cfg.gru_hidden, vb.pp("hidden_proj"))?;
        let token_emb = embedding(cfg.vocab_size, cfg.emb_dim, vb.pp("token_emb"))?;

        // Load GRU cells
        let mut gru_cells = Vec::new();
        let first_input_dim = cfg.emb_dim; // first layer takes token embedding
        gru_cells.push(GruCell::load(first_input_dim, cfg.gru_hidden, vb.pp("gru.0"))?);
        for i in 1..cfg.gru_layers {
            // Subsequent layers take previous layer's hidden state
            gru_cells.push(GruCell::load(cfg.gru_hidden, cfg.gru_hidden, vb.pp(format!("gru.{i}")))?);
        }

        let output_head = linear_no_bias(cfg.gru_hidden, cfg.vocab_size, vb.pp("output_head"))?;

        Ok(Self {
            hidden_proj,
            token_emb,
            gru_cells,
            output_head,
            cfg: cfg.clone(),
            device: device.clone(),
            dtype,
        })
    }

    /// Run K draft steps starting from the last semantic token and main LM hidden state.
    ///
    /// Process:
    /// 1. Project the main LM hidden state to GRU initial state
    /// 2. For each draft step:
    ///    a. Embed the current token
    ///    b. Run GRU forward pass
    ///    c. Project to logits
    ///    d. Sample next token (greedy argmax)
    ///
    /// Returns vector of K draft token IDs.
    pub fn draft(&self, lm_hidden: &Tensor, current_token: u32, num_steps: usize) -> Result<Vec<u32>> {
        // lm_hidden should be (1, 1, d_model) or (1, d_model)
        let h0_squeezed = if lm_hidden.dims().len() == 3 {
            lm_hidden.squeeze(1)?  // (1, 1, d_model) → (1, d_model)
        } else {
            lm_hidden.clone()
        };

        // Validate h0_squeezed shape before using
        let dims = h0_squeezed.dims();
        if dims.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "Expected 2D tensor (batch, d_model), got shape: {:?}", dims
            )));
        }
        if dims[1] != self.cfg.d_model {
            return Err(candle_core::Error::Msg(format!(
                "Expected d_model={}, got {}", self.cfg.d_model, dims[1]
            )));
        }

        // Project LM hidden state to GRU initial state: (1, d_model) → (1, gru_hidden)
        let h0 = self.hidden_proj.forward(&h0_squeezed)?;

        let mut draft_tokens = Vec::new();
        let mut current_h = h0.clone();
        let mut next_token = current_token;

        for _step in 0..num_steps {
            // Embed current token: (1,) → (1, emb_dim)
            let token_t = Tensor::from_vec(vec![next_token], (1,), &self.device)?;
            let x = self.token_emb.forward(&token_t)?; // (1, emb_dim)

            // Run through GRU layers
            let mut h = current_h.clone();
            for cell in &self.gru_cells {
                h = cell.forward(&x, &h)?;
            }

            // Project to logits: (1, gru_hidden) → (1, vocab_size)
            let logits = self.output_head.forward(&h)?;
            let logits_f32 = logits.to_dtype(DType::F32)?;
            let logits_vec: Vec<f32> = logits_f32.squeeze(0)?.to_vec1()?;

            // Sample greedily (argmax)
            let mut best_idx = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &val) in logits_vec.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_idx = i as u32;
                }
            }

            draft_tokens.push(best_idx);
            next_token = best_idx;
            current_h = h;
        }

        Ok(draft_tokens)
    }

    pub fn cfg(&self) -> &DrafterConfig {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gru_cell_forward() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let cell = GruCell::load(32, 64, vb)?;

        let x = Tensor::randn(0.0, 1.0, (1, 32), &device)?;
        let h = Tensor::randn(0.0, 1.0, (1, 64), &device)?;

        let h_new = cell.forward(&x, &h)?;
        let (b, d) = h_new.dims2()?;

        assert_eq!(b, 1);
        assert_eq!(d, 64);

        Ok(())
    }

    #[test]
    fn test_gru_drafter_draft() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let cfg = DrafterConfig {
            gru_hidden: 64,
            gru_layers: 2,
            emb_dim: 32,
            d_model: 128,
            vocab_size: 256,
        };

        let vb = VarBuilder::zeros(dtype, &device);
        let drafter = GruDrafter::load(&cfg, vb, &device, dtype)?;

        // Test draft with batch=1, hidden state (1, 128)
        let lm_hidden = Tensor::randn(0.0, 1.0, (1, cfg.d_model), &device)?;
        let current_token = 42u32;
        let draft_tokens = drafter.draft(&lm_hidden, current_token, 3)?;

        assert_eq!(draft_tokens.len(), 3);
        for token in draft_tokens {
            assert!(token < cfg.vocab_size as u32);
        }

        Ok(())
    }
}
