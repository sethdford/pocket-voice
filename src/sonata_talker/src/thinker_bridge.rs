/// Thinker↔Talker bridge: connects LLM hidden states to the Talker inference engine.
///
/// The Thinker (LLM like Claude) produces hidden states at token rate (~50 tokens/sec).
/// The Talker consumes them at acoustic frame rate (12.5 Hz).
/// This bridge handles rate adaptation and manages the latest hidden state.
///
/// Typical usage:
/// ```ignore
/// let mut bridge = ThinkerBridge::new(4096, &device)?;
/// bridge.push_hidden(&hidden_buffer)?;
/// let thinker_hidden = bridge.get_hidden();
/// ```

use candle_core::{Device, Result, Tensor};
use std::sync::{Arc, Mutex};

/// Bridge for streaming Thinker (LLM) hidden states to the Talker.
pub struct ThinkerBridge {
    /// Dimension of the Thinker's hidden state (typically 4096 for Claude).
    thinker_dim: usize,
    /// Device (CPU or Metal GPU) for tensor operations.
    device: Arc<Device>,
    /// Most recent thinker hidden state, wrapped as a tensor.
    /// Shape: (1, 1, thinker_dim)
    latest_hidden: Arc<Mutex<Option<Tensor>>>,
}

impl ThinkerBridge {
    /// Create a new bridge with the given thinker hidden dimension.
    ///
    /// # Arguments
    /// * `thinker_dim` - Hidden state dimension (e.g., 4096)
    /// * `device` - Device for tensor allocation (CPU or Metal)
    pub fn new(thinker_dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            thinker_dim,
            device: Arc::new(device.clone()),
            latest_hidden: Arc::new(Mutex::new(None)),
        })
    }

    /// Push a new hidden state from the Thinker (LLM forward pass).
    ///
    /// # Arguments
    /// * `hidden` - Raw f32 buffer of shape [thinker_dim]
    ///
    /// # Returns
    /// * 0 on success
    /// * -1 on error
    pub fn push_hidden(&self, hidden: &[f32]) -> Result<()> {
        if hidden.len() != self.thinker_dim {
            return Err(candle_core::Error::Msg(format!(
                "Hidden state shape mismatch: expected {}, got {}",
                self.thinker_dim,
                hidden.len()
            )));
        }

        // Wrap raw buffer in tensor: (thinker_dim,) → (1, 1, thinker_dim)
        let tensor = Tensor::from_vec(hidden.to_vec(), (self.thinker_dim,), &self.device)?
            .unsqueeze(0)? // (1, thinker_dim)
            .unsqueeze(0)?; // (1, 1, thinker_dim)

        let mut latest = self.latest_hidden.lock().unwrap();
        *latest = Some(tensor);
        Ok(())
    }

    /// Get the most recent hidden state (for use in Talker step).
    ///
    /// Returns None if no hidden state has been pushed yet.
    /// Returns a borrowed reference to the Tensor if available.
    pub fn get_hidden(&self) -> Option<Tensor> {
        let latest = self.latest_hidden.lock().unwrap();
        latest.as_ref().cloned()
    }

    /// Clear the stored hidden state (e.g., on reset).
    pub fn clear(&self) {
        let mut latest = self.latest_hidden.lock().unwrap();
        *latest = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thinker_bridge_push_get() {
        let device = Device::Cpu;
        let dim = 4096;
        let bridge = ThinkerBridge::new(dim, &device).unwrap();

        // Push a hidden state
        let hidden = vec![0.5f32; dim];
        bridge.push_hidden(&hidden).unwrap();

        // Retrieve it
        let retrieved = bridge.get_hidden();
        assert!(retrieved.is_some());

        let tensor = retrieved.unwrap();
        assert_eq!(tensor.dims(), &[1, 1, 4096]);

        // Verify first element
        let val = tensor
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .get(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!((val - 0.5f32).abs() < 1e-6);
    }

    #[test]
    fn test_thinker_bridge_clear() {
        let device = Device::Cpu;
        let bridge = ThinkerBridge::new(4096, &device).unwrap();

        // Push a hidden state
        let hidden = vec![1.0f32; 4096];
        bridge.push_hidden(&hidden).unwrap();
        assert!(bridge.get_hidden().is_some());

        // Clear it
        bridge.clear();
        assert!(bridge.get_hidden().is_none());
    }

    #[test]
    fn test_thinker_bridge_dim_mismatch() {
        let device = Device::Cpu;
        let bridge = ThinkerBridge::new(4096, &device).unwrap();

        // Try to push wrong-sized hidden state
        let hidden = vec![0.5f32; 512]; // Wrong size
        let result = bridge.push_hidden(&hidden);
        assert!(result.is_err());
    }

    #[test]
    fn test_thinker_bridge_no_push() {
        let device = Device::Cpu;
        let bridge = ThinkerBridge::new(4096, &device).unwrap();

        // Without pushing, should return None
        assert!(bridge.get_hidden().is_none());
    }
}
