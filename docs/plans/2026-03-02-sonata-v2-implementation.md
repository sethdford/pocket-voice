# Sonata v2 + SeaClaw Unified Binary — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified on-device AI voice agent binary: Sonata v2 voice pipeline (~260M params, Rust/C) compiled into SeaClaw agent runtime (C11), with LLM provider for reasoning.

**Architecture:** Sonata provides SOTA STT/TTS/full-duplex with AdaIN voice conditioning, emotion control, nonverbal vocabulary, CAM++ speaker encoder, and CFM decoder. SeaClaw provides agent intelligence (LLM routing, tools, memory, security). One binary, ~50MB with quantized models.

**Tech Stack:** Rust (candle 0.9, Metal GPU), C11 (SeaClaw runtime, vDSP/Accelerate), PyTorch (training), CMake + Cargo (build)

**Design Doc:** `docs/plans/2026-03-02-sonata-v2-unified-transformer-design.md`

**Repos:**
- pocket-voice: `/Users/sethford/Documents/pocket-voice`
- seaclaw: `https://github.com/sethdford/seaclaw` (clone to `/Users/sethford/Documents/seaclaw`)

---

## Phase Overview

| Phase | Tasks | Duration | Dependencies |
|-------|-------|----------|-------------|
| **Phase 1: Foundation** | Tasks 1-4 | 1-2 weeks | None |
| **Phase 2: Codec** | Tasks 5-8 | 2-3 weeks | Phase 1 |
| **Phase 3: Speaker** | Tasks 9-11 | 2-3 weeks | Phase 1 (parallel with Phase 2) |
| **Phase 4: STT** | Tasks 12-14 | 2-3 weeks | Phase 2 (needs codec) |
| **Phase 5: TTS** | Tasks 15-18 | 3-4 weeks | Phase 2 + Phase 3 (needs codec + speaker) |
| **Phase 6: CFM Decoder** | Tasks 19-21 | 1-2 weeks | Phase 2 (needs codec) |
| **Phase 7: Full-Duplex** | Tasks 22-24 | 1-2 weeks | Phase 4 + Phase 5 |
| **Phase 8: SeaClaw Integration** | Tasks 25-29 | 2-3 weeks | Phase 4 + Phase 5 + Phase 6 |
| **Phase 9: Training Pipeline** | Tasks 30-34 | 4-6 weeks | Phase 2-6 (model architectures defined) |
| **Phase 10: Optimization** | Tasks 35-37 | 2-3 weeks | All previous |

---

## Phase 1: Foundation (Cargo Workspace + Shared Infrastructure)

### Task 1: Create Cargo Workspace

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/sonata-common/Cargo.toml`
- Create: `crates/sonata-common/src/lib.rs`

**Step 1: Write the test for shared types**

```rust
// crates/sonata-common/src/lib.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_config_default() {
        let config = AudioConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.channels, 1);
        assert_eq!(config.frame_rate_hz, 50);
    }

    #[test]
    fn test_codec_token_dimensions() {
        assert_eq!(NUM_CODEBOOKS, 8);
        assert_eq!(CODEBOOK_SIZE, 1024);
        assert_eq!(CODEBOOK_DIM, 128);
    }

    #[test]
    fn test_speaker_embedding_dim() {
        assert_eq!(SPEAKER_EMBED_DIM, 192);
    }

    #[test]
    fn test_emotion_style_count() {
        assert_eq!(NUM_EMOTION_STYLES, 64);
        let style = EmotionStyle::default();
        assert_eq!(style.exaggeration, 1.0);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-common`
Expected: FAIL — crate doesn't exist yet

**Step 3: Create workspace Cargo.toml**

```toml
# /Users/sethford/Documents/pocket-voice/Cargo.toml
[workspace]
resolver = "2"
members = [
    "crates/sonata-common",
    # v2 crates (added as implemented):
    # "crates/sonata-codec",
    # "crates/sonata-cam",
    # "crates/sonata-stt",
    # "crates/sonata-tts",
    # "crates/sonata-cfm",
    # "crates/sonata-pipeline",
    # Legacy v1 crates (kept for compatibility):
    "src/stt",
    "src/sonata_lm",
    "src/sonata_flow",
    "src/sonata_speaker",
    "src/sonata_storm",
]

[workspace.dependencies]
candle-core = { version = "0.9", features = [] }
candle-nn = { version = "0.9", features = [] }
candle-transformers = { version = "0.9", features = [] }
safetensors = "0.4"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
hf-hub = "0.3"
tokenizers = "0.21"
anyhow = "1.0"
tracing = "0.1"

[workspace.dependencies.candle-core-metal]
package = "candle-core"
version = "0.9"
features = ["metal"]

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "unwind"
strip = true
```

**Step 4: Create sonata-common crate**

```toml
# crates/sonata-common/Cargo.toml
[package]
name = "sonata-common"
version = "0.2.0"
edition = "2021"
publish = false

[dependencies]
serde = { workspace = true }
```

```rust
// crates/sonata-common/src/lib.rs

//! Shared types and constants for Sonata v2 voice pipeline.

use serde::{Deserialize, Serialize};

// --- Audio constants ---

pub const SAMPLE_RATE: u32 = 24000;
pub const MEL_BINS: usize = 80;
pub const HOP_LENGTH: usize = 256;
pub const FFT_SIZE: usize = 1024;

// --- Codec constants ---

pub const NUM_CODEBOOKS: usize = 8;
pub const CODEBOOK_SIZE: usize = 1024;
pub const CODEBOOK_DIM: usize = 128;
pub const SEMANTIC_CODEBOOKS: usize = 2;   // books 1-2: semantic
pub const ACOUSTIC_CODEBOOKS: usize = 6;   // books 3-8: acoustic
pub const CODEC_FRAME_RATE_HZ: usize = 50; // 50 Hz = 20ms per frame

// --- Speaker encoder constants ---

pub const SPEAKER_EMBED_DIM: usize = 192;
pub const SPEAKER_SAMPLE_RATE: u32 = 16000;
pub const SPEAKER_MEL_BINS: usize = 80;

// --- Emotion/style constants ---

pub const NUM_EMOTION_STYLES: usize = 64;
pub const NUM_NONVERBAL_TOKENS: usize = 24;

// --- Model dimensions ---

pub const OUTER_TRANSFORMER_DIM: usize = 1024;
pub const OUTER_TRANSFORMER_LAYERS: usize = 24;
pub const OUTER_TRANSFORMER_HEADS: usize = 16;
pub const OUTER_TRANSFORMER_KV_HEADS: usize = 4;
pub const OUTER_FFN_DIM: usize = 4096;

pub const DEPTH_TRANSFORMER_DIM: usize = 256;
pub const DEPTH_TRANSFORMER_LAYERS: usize = 6;

pub const CFM_DIM: usize = 512;
pub const CFM_LAYERS: usize = 12;
pub const CFM_HEADS: usize = 8;

// --- Types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub frame_rate_hz: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            channels: 1,
            frame_rate_hz: CODEC_FRAME_RATE_HZ,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum NonverbalTag {
    Laugh = 0,
    Chuckle = 1,
    Giggle = 2,
    Sigh = 3,
    Breath = 4,
    Gasp = 5,
    Hmm = 6,
    UhHuh = 7,
    Oh = 8,
    Right = 9,
    PauseShort = 10,
    PauseLong = 11,
    Whisper = 12,
    Emphasis = 13,
    Cough = 14,
    Yawn = 15,
    Sniff = 16,
    Cry = 17,
    Groan = 18,
    Hum = 19,
    Tsk = 20,
    Wow = 21,
    Ooh = 22,
    Ahh = 23,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionStyle {
    pub style_id: u8,
    pub exaggeration: f32, // 0.0 - 2.0 (Chatterbox-style)
}

impl Default for EmotionStyle {
    fn default() -> Self {
        Self {
            style_id: 0, // neutral
            exaggeration: 1.0,
        }
    }
}

/// Codec tokens for a single frame (20ms of audio at 50Hz).
#[derive(Debug, Clone)]
pub struct CodecFrame {
    pub semantic: [u16; SEMANTIC_CODEBOOKS],  // books 1-2
    pub acoustic: [u16; ACOUSTIC_CODEBOOKS],  // books 3-8
}

/// Speaker embedding from CAM++ encoder.
#[derive(Debug, Clone)]
pub struct SpeakerEmbedding {
    pub data: [f32; SPEAKER_EMBED_DIM],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_config_default() {
        let config = AudioConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.channels, 1);
        assert_eq!(config.frame_rate_hz, 50);
    }

    #[test]
    fn test_codec_token_dimensions() {
        assert_eq!(NUM_CODEBOOKS, 8);
        assert_eq!(CODEBOOK_SIZE, 1024);
        assert_eq!(CODEBOOK_DIM, 128);
    }

    #[test]
    fn test_speaker_embedding_dim() {
        assert_eq!(SPEAKER_EMBED_DIM, 192);
    }

    #[test]
    fn test_emotion_style_count() {
        assert_eq!(NUM_EMOTION_STYLES, 64);
        let style = EmotionStyle::default();
        assert_eq!(style.exaggeration, 1.0);
    }

    #[test]
    fn test_nonverbal_tag_count() {
        assert_eq!(NUM_NONVERBAL_TOKENS, 24);
        assert_eq!(NonverbalTag::Laugh as u8, 0);
        assert_eq!(NonverbalTag::Ahh as u8, 23);
    }

    #[test]
    fn test_codec_frame_sizes() {
        let frame = CodecFrame {
            semantic: [0; SEMANTIC_CODEBOOKS],
            acoustic: [0; ACOUSTIC_CODEBOOKS],
        };
        assert_eq!(frame.semantic.len(), 2);
        assert_eq!(frame.acoustic.len(), 6);
    }
}
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-common`
Expected: All 6 tests PASS

**Step 6: Commit**

```bash
cd /Users/sethford/Documents/pocket-voice
git add Cargo.toml crates/sonata-common/
git commit -m "feat(v2): create Cargo workspace and sonata-common shared types"
```

---

### Task 2: Migrate Legacy Crates to Workspace Dependencies

**Files:**
- Modify: `src/sonata_lm/Cargo.toml`
- Modify: `src/sonata_flow/Cargo.toml`
- Modify: `src/sonata_speaker/Cargo.toml`
- Modify: `src/sonata_storm/Cargo.toml`
- Modify: `src/stt/Cargo.toml`

**Step 1: Update each legacy crate to use workspace dependencies**

For each crate, replace explicit dependency versions with workspace references. Example for `src/sonata_lm/Cargo.toml`:

```toml
# Change FROM:
[dependencies]
candle-core = { version = "0.9", features = [...] }
candle-nn = { version = "0.9", features = [...] }
safetensors = "0.4"

# Change TO:
[dependencies]
candle-core = { workspace = true }
candle-nn = { workspace = true }
safetensors = { workspace = true }
```

Keep any crate-specific dependencies (like `moshi` in stt) as-is. Only unify shared deps.

**Step 2: Verify all legacy crates build**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo build --release -p sonata-lm -p sonata-flow -p sonata-speaker -p sonata-storm -p pocket-stt 2>&1 | tail -5`
Expected: All 5 crates compile successfully

**Step 3: Run existing tests**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test --workspace`
Expected: All existing tests pass (some crates may have no tests)

**Step 4: Commit**

```bash
cd /Users/sethford/Documents/pocket-voice
git add src/sonata_lm/Cargo.toml src/sonata_flow/Cargo.toml src/sonata_speaker/Cargo.toml src/sonata_storm/Cargo.toml src/stt/Cargo.toml
git commit -m "chore(v2): migrate legacy crates to workspace dependencies"
```

---

### Task 3: AdaIN Module (Shared by Codec, TTS, CFM)

**Files:**
- Create: `crates/sonata-common/src/adain.rs`
- Modify: `crates/sonata-common/src/lib.rs`
- Modify: `crates/sonata-common/Cargo.toml`

**Step 1: Write the failing test**

```rust
// crates/sonata-common/src/adain.rs
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_adain_output_shape() {
        let dev = &Device::Cpu;
        let adain = AdaIN::new(256, 192, dev).unwrap();
        let x = Tensor::zeros(&[1, 10, 256], candle_core::DType::F32, dev).unwrap();
        let style = Tensor::zeros(&[1, 192], candle_core::DType::F32, dev).unwrap();
        let out = adain.forward(&x, &style).unwrap();
        assert_eq!(out.dims(), &[1, 10, 256]);
    }

    #[test]
    fn test_adain_normalizes() {
        let dev = &Device::Cpu;
        let adain = AdaIN::new(4, 4, dev).unwrap();
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], dev).unwrap().unsqueeze(0).unwrap();
        let style = Tensor::zeros(&[1, 4], candle_core::DType::F32, dev).unwrap();
        let out = adain.forward(&x, &style).unwrap();
        // With zero style, gamma=0 and beta=0 after linear,
        // so output should be normalized (mean~0, std~1) * 0 + 0 = ~0
        let sum: f32 = out.sum_all().unwrap().to_scalar().unwrap();
        assert!(sum.abs() < 1e-5);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-common -- adain`
Expected: FAIL — module doesn't exist

**Step 3: Implement AdaIN**

```rust
// crates/sonata-common/src/adain.rs

//! Adaptive Instance Normalization (AdaIN) — from Kokoro.
//! Conditions a hidden representation with a style embedding (speaker or emotion).
//! AdaIN(x, style) = gamma(style) * normalize(x) + beta(style)

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub struct AdaIN {
    gamma_proj: Linear,
    beta_proj: Linear,
    hidden_dim: usize,
}

impl AdaIN {
    pub fn new(hidden_dim: usize, style_dim: usize, vb: impl Into<VarBuilder<'static>> + Clone) -> Result<Self> {
        // For tests, accept Device directly; for real use, accept VarBuilder
        Self::from_vb(hidden_dim, style_dim, vb)
    }

    fn from_vb(hidden_dim: usize, style_dim: usize, _vb: impl Into<VarBuilder<'static>>) -> Result<Self> {
        let dev = &candle_core::Device::Cpu;
        let gamma_proj = candle_nn::linear(style_dim, hidden_dim, dev.into())?;
        let beta_proj = candle_nn::linear(style_dim, hidden_dim, dev.into())?;
        Ok(Self { gamma_proj, beta_proj, hidden_dim })
    }

    pub fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        // Instance normalization: normalize across hidden dim
        let mean = x.mean_keepdim(2)?;
        let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(2)?;
        let std = (var + 1e-5)?.sqrt()?;
        let x_norm = x.broadcast_sub(&mean)?.broadcast_div(&std)?;

        // Style projection
        let gamma = self.gamma_proj.forward(style)?; // [B, hidden_dim]
        let beta = self.beta_proj.forward(style)?;   // [B, hidden_dim]

        // Unsqueeze for broadcasting: [B, 1, hidden_dim]
        let gamma = gamma.unsqueeze(1)?;
        let beta = beta.unsqueeze(1)?;

        // AdaIN: gamma * x_norm + beta
        x_norm.broadcast_mul(&gamma)?.broadcast_add(&beta)
    }
}
```

Update `crates/sonata-common/Cargo.toml`:
```toml
[dependencies]
serde = { workspace = true }
candle-core = { workspace = true }
candle-nn = { workspace = true }
```

Update `crates/sonata-common/src/lib.rs` to add:
```rust
pub mod adain;
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-common`
Expected: All tests PASS (including new AdaIN tests)

**Step 5: Commit**

```bash
cd /Users/sethford/Documents/pocket-voice
git add crates/sonata-common/
git commit -m "feat(v2): add AdaIN module (Kokoro-style adaptive instance normalization)"
```

---

### Task 4: RoPE and SwiGLU Modules (Shared Transformer Primitives)

**Files:**
- Create: `crates/sonata-common/src/rope.rs`
- Create: `crates/sonata-common/src/swiglu.rs`
- Modify: `crates/sonata-common/src/lib.rs`

**Step 1: Write the failing tests**

```rust
// crates/sonata-common/src/rope.rs
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};

    #[test]
    fn test_rope_output_shape() {
        let dev = &Device::Cpu;
        let rope = RotaryEmbedding::new(64, 4096, dev).unwrap();
        let q = Tensor::zeros(&[1, 8, 10, 64], DType::F32, dev).unwrap();
        let k = Tensor::zeros(&[1, 2, 10, 64], DType::F32, dev).unwrap();
        let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();
        assert_eq!(q_rot.dims(), &[1, 8, 10, 64]);
        assert_eq!(k_rot.dims(), &[1, 2, 10, 64]);
    }
}

// crates/sonata-common/src/swiglu.rs
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};

    #[test]
    fn test_swiglu_output_shape() {
        let dev = &Device::Cpu;
        let ffn = SwiGLU::new(256, 1024, dev).unwrap();
        let x = Tensor::zeros(&[1, 10, 256], DType::F32, dev).unwrap();
        let out = ffn.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 10, 256]);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-common -- rope swiglu`
Expected: FAIL — modules don't exist

**Step 3: Implement RoPE**

```rust
// crates/sonata-common/src/rope.rs

//! Rotary Position Embeddings (RoPE) for transformer attention.

use candle_core::{DType, Device, Result, Tensor};

pub struct RotaryEmbedding {
    cos_cache: Tensor,
    sin_cache: Tensor,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, max_seq_len: usize, dev: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / 10000f32.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq, dev)?;
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions, dev)?.unsqueeze(1)?;
        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;
        Ok(Self { cos_cache, sin_cache })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos_cache.narrow(0, offset, seq_len)?;
        let sin = self.sin_cache.narrow(0, offset, seq_len)?;
        let q_rot = Self::apply_rotary(q, &cos, &sin)?;
        let k_rot = Self::apply_rotary(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }

    fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let half = x.dim(3)? / 2;
        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, half]
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        let rotated_x1 = (x1.broadcast_mul(&cos)?.broadcast_sub(&x2.broadcast_mul(&sin)?))?;
        let rotated_x2 = (x2.broadcast_mul(&cos)?.broadcast_add(&x1.broadcast_mul(&sin)?))?;
        Tensor::cat(&[&rotated_x1, &rotated_x2], 3)
    }
}
```

**Step 4: Implement SwiGLU**

```rust
// crates/sonata-common/src/swiglu.rs

//! SwiGLU feed-forward network used in modern transformers.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, VarBuilder};

pub struct SwiGLU {
    w_gate: Linear,
    w_up: Linear,
    w_down: Linear,
}

impl SwiGLU {
    pub fn new(dim: usize, ffn_dim: usize, dev: impl Into<VarBuilder<'static>>) -> Result<Self> {
        let dev = &candle_core::Device::Cpu;
        let vb: VarBuilder = VarBuilder::zeros(candle_core::DType::F32, dev);
        let w_gate = candle_nn::linear(dim, ffn_dim, vb.pp("gate"))?;
        let w_up = candle_nn::linear(dim, ffn_dim, vb.pp("up"))?;
        let w_down = candle_nn::linear(ffn_dim, dim, vb.pp("down"))?;
        Ok(Self { w_gate, w_up, w_down })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&self.w_gate.forward(x)?)?;
        let up = self.w_up.forward(x)?;
        self.w_down.forward(&(gate * up)?)
    }
}
```

Update `crates/sonata-common/src/lib.rs`:
```rust
pub mod adain;
pub mod rope;
pub mod swiglu;
```

**Step 5: Run tests**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-common`
Expected: All tests PASS

**Step 6: Commit**

```bash
cd /Users/sethford/Documents/pocket-voice
git add crates/sonata-common/
git commit -m "feat(v2): add RoPE and SwiGLU transformer primitives"
```

---

## Phase 2: Sonata Codec (Audio Tokenizer, ~25M params)

### Task 5: Codec Crate Scaffold + Snake Activation

**Files:**
- Create: `crates/sonata-codec/Cargo.toml`
- Create: `crates/sonata-codec/src/lib.rs`
- Create: `crates/sonata-codec/src/snake.rs`
- Modify: `Cargo.toml` (add to workspace members)

**Step 1: Write the failing test**

```rust
// crates/sonata-codec/src/snake.rs
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};

    #[test]
    fn test_snake_activation_shape() {
        let dev = &Device::Cpu;
        let snake = Snake::new(64, dev).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, &[1, 64, 100], dev).unwrap();
        let out = snake.forward(&x).unwrap();
        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn test_snake_nonlinear() {
        let dev = &Device::Cpu;
        let snake = Snake::new(1, dev).unwrap();
        let x = Tensor::new(&[[[0.0f32, 1.0, -1.0]]], dev).unwrap();
        let out = snake.forward(&x).unwrap();
        // Snake(x) = x + (1/alpha) * sin^2(alpha * x)
        // Should not equal input
        let diff = (out - x).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff > 0.0);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-codec`
Expected: FAIL — crate doesn't exist

**Step 3: Create crate and implement Snake activation**

```toml
# crates/sonata-codec/Cargo.toml
[package]
name = "sonata-codec"
version = "0.2.0"
edition = "2021"
publish = false

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
sonata-common = { path = "../sonata-common" }
candle-core = { workspace = true }
candle-nn = { workspace = true }
safetensors = { workspace = true }
anyhow = { workspace = true }
tracing = { workspace = true }

[features]
default = []
metal = ["candle-core/metal", "candle-nn/metal"]
```

```rust
// crates/sonata-codec/src/snake.rs

//! Snake activation function — learned periodic activation for audio synthesis.
//! Snake(x) = x + (1/alpha) * sin^2(alpha * x)

use candle_core::{Result, Tensor, DType, Device};
use candle_nn::VarBuilder;

pub struct Snake {
    alpha: Tensor, // learnable parameter, shape [channels]
}

impl Snake {
    pub fn new(channels: usize, dev: &Device) -> Result<Self> {
        let alpha = Tensor::ones(&[1, channels, 1], DType::F32, dev)?;
        Ok(Self { alpha })
    }

    pub fn load(vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get_with_hints(&[1], "alpha", candle_nn::Init::Const(1.0))?;
        Ok(Self { alpha })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ax = x.broadcast_mul(&self.alpha)?;
        let sin_ax = ax.sin()?;
        let sin_sq = (&sin_ax * &sin_ax)?;
        let inv_alpha = self.alpha.recip()?;
        x + sin_sq.broadcast_mul(&inv_alpha)?
    }
}
```

```rust
// crates/sonata-codec/src/lib.rs
pub mod snake;

// Will add: encoder, decoder, quantizer, semantic
```

Add `"crates/sonata-codec"` to workspace members in root `Cargo.toml`.

**Step 4: Run tests**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-codec`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/sethford/Documents/pocket-voice
git add Cargo.toml crates/sonata-codec/
git commit -m "feat(v2): scaffold sonata-codec crate with Snake activation"
```

---

### Task 6: Codec Encoder (1D ConvNet, 480x downsampling)

**Files:**
- Create: `crates/sonata-codec/src/encoder.rs`
- Modify: `crates/sonata-codec/src/lib.rs`

**Step 1: Write the failing test**

```rust
// crates/sonata-codec/src/encoder.rs
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};

    #[test]
    fn test_encoder_downsampling_ratio() {
        let dev = &Device::Cpu;
        let encoder = CodecEncoder::new(dev).unwrap();
        // 24000 samples = 1 second at 24kHz
        // Should produce 50 frames (50Hz)
        let audio = Tensor::zeros(&[1, 1, 24000], DType::F32, dev).unwrap();
        let encoded = encoder.forward(&audio).unwrap();
        assert_eq!(encoded.dim(2).unwrap(), 50); // 24000 / 480 = 50
        assert_eq!(encoded.dim(1).unwrap(), 512); // output channels
    }

    #[test]
    fn test_encoder_stride_product() {
        // Strides: [8, 5, 4, 3] => 8*5*4*3 = 480
        assert_eq!(ENCODER_STRIDES.iter().product::<usize>(), 480);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-codec -- encoder`
Expected: FAIL

**Step 3: Implement Codec Encoder**

```rust
// crates/sonata-codec/src/encoder.rs

//! 1D ConvNet encoder: downsample 24kHz audio to 50Hz latent.
//! Strides [8, 5, 4, 3] = 480x downsampling (same as current Code2Wav).

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use crate::snake::Snake;

pub const ENCODER_STRIDES: [usize; 4] = [8, 5, 4, 3];
pub const ENCODER_CHANNELS: [usize; 5] = [1, 64, 128, 256, 512];

pub struct EncoderBlock {
    conv: Conv1d,
    snake: Snake,
}

impl EncoderBlock {
    pub fn new(in_ch: usize, out_ch: usize, stride: usize, dev: &Device) -> Result<Self> {
        let kernel = stride * 2;
        let padding = stride / 2;
        let cfg = Conv1dConfig { stride, padding, ..Default::default() };
        let conv = candle_nn::conv1d(in_ch, out_ch, kernel, cfg,
            VarBuilder::zeros(DType::F32, dev))?;
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
}
```

**Step 4: Run tests**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-codec -- encoder`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/sethford/Documents/pocket-voice
git add crates/sonata-codec/
git commit -m "feat(v2): add codec encoder (480x downsampling, 24kHz to 50Hz)"
```

---

### Task 7: Residual Vector Quantizer (8 codebooks, split semantic/acoustic)

**Files:**
- Create: `crates/sonata-codec/src/quantizer.rs`
- Modify: `crates/sonata-codec/src/lib.rs`

**Step 1: Write the failing test**

```rust
// crates/sonata-codec/src/quantizer.rs
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};
    use sonata_common::{NUM_CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM};

    #[test]
    fn test_rvq_output_shape() {
        let dev = &Device::Cpu;
        let rvq = ResidualVQ::new(512, NUM_CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM, dev).unwrap();
        let z = Tensor::randn(0.0f32, 1.0, &[1, 512, 50], DType::F32, dev).unwrap();
        let (codes, _) = rvq.encode(&z).unwrap();
        assert_eq!(codes.dims(), &[1, NUM_CODEBOOKS, 50]);
    }

    #[test]
    fn test_rvq_roundtrip() {
        let dev = &Device::Cpu;
        let rvq = ResidualVQ::new(512, NUM_CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM, dev).unwrap();
        let z = Tensor::randn(0.0f32, 0.1, &[1, 512, 10], DType::F32, dev).unwrap();
        let (codes, _) = rvq.encode(&z).unwrap();
        let z_hat = rvq.decode(&codes).unwrap();
        assert_eq!(z_hat.dims(), z.dims());
    }

    #[test]
    fn test_semantic_acoustic_split() {
        let dev = &Device::Cpu;
        let rvq = ResidualVQ::new(512, NUM_CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM, dev).unwrap();
        let z = Tensor::randn(0.0f32, 1.0, &[1, 512, 5], DType::F32, dev).unwrap();
        let (codes, _) = rvq.encode(&z).unwrap();
        let (semantic, acoustic) = rvq.split_codes(&codes).unwrap();
        assert_eq!(semantic.dim(1).unwrap(), 2);  // books 1-2
        assert_eq!(acoustic.dim(1).unwrap(), 6);  // books 3-8
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-codec -- quantizer`
Expected: FAIL

**Step 3: Implement Residual Vector Quantizer**

```rust
// crates/sonata-codec/src/quantizer.rs

//! Residual Vector Quantizer with split semantic/acoustic codebooks.
//! 8 codebooks x 1024 entries. Books 1-2 = semantic, 3-8 = acoustic.

use candle_core::{DType, Device, Result, Tensor, D};
use sonata_common::{SEMANTIC_CODEBOOKS, ACOUSTIC_CODEBOOKS};

pub struct VectorQuantizer {
    codebook: Tensor, // [codebook_size, codebook_dim]
    project_in: Option<Tensor>,  // [input_dim, codebook_dim] if dims differ
    project_out: Option<Tensor>, // [codebook_dim, input_dim] if dims differ
}

impl VectorQuantizer {
    pub fn new(input_dim: usize, codebook_size: usize, codebook_dim: usize, dev: &Device) -> Result<Self> {
        let codebook = Tensor::randn(0.0f32, 0.02, &[codebook_size, codebook_dim], dev)?;
        let (project_in, project_out) = if input_dim != codebook_dim {
            let pi = Tensor::randn(0.0f32, 0.02, &[input_dim, codebook_dim], dev)?;
            let po = Tensor::randn(0.0f32, 0.02, &[codebook_dim, input_dim], dev)?;
            (Some(pi), Some(po))
        } else {
            (None, None)
        };
        Ok(Self { codebook, project_in, project_out })
    }

    pub fn encode(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        // z: [B, D, T] -> [B, T, D]
        let z_t = z.transpose(1, 2)?;
        let z_proj = if let Some(ref proj) = self.project_in {
            z_t.matmul(proj)?
        } else {
            z_t.clone()
        };

        // Nearest neighbor: ||z - e||^2 = ||z||^2 - 2*z*e^T + ||e||^2
        let z_sq = z_proj.sqr()?.sum(D::Minus1)?.unsqueeze(D::Minus1)?;
        let e_sq = self.codebook.sqr()?.sum(D::Minus1)?.unsqueeze(0)?.unsqueeze(0)?;
        let ze = z_proj.matmul(&self.codebook.t()?)?;
        let dist = (z_sq.broadcast_add(&e_sq)? - ze * 2.0)?;
        let codes = dist.argmin(D::Minus1)?;

        // Quantized output
        let quantized = self.lookup(&codes)?;
        Ok((codes, quantized))
    }

    pub fn lookup(&self, codes: &Tensor) -> Result<Tensor> {
        let flat = codes.flatten_all()?;
        let emb = self.codebook.index_select(&flat, 0)?;
        let shape = codes.dims().to_vec();
        let codebook_dim = self.codebook.dim(1)?;
        let emb = emb.reshape(&[shape[0], shape[1], codebook_dim])?;
        let emb = if let Some(ref proj) = self.project_out {
            emb.matmul(proj)?
        } else {
            emb
        };
        emb.transpose(1, 2) // [B, D, T]
    }
}

pub struct ResidualVQ {
    quantizers: Vec<VectorQuantizer>,
    num_codebooks: usize,
}

impl ResidualVQ {
    pub fn new(
        input_dim: usize,
        num_codebooks: usize,
        codebook_size: usize,
        codebook_dim: usize,
        dev: &Device,
    ) -> Result<Self> {
        let mut quantizers = Vec::new();
        for _ in 0..num_codebooks {
            quantizers.push(VectorQuantizer::new(input_dim, codebook_size, codebook_dim, dev)?);
        }
        Ok(Self { quantizers, num_codebooks })
    }

    pub fn encode(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut residual = z.clone();
        let mut all_codes = Vec::new();
        let mut quantized_sum = Tensor::zeros_like(z)?;

        for vq in &self.quantizers {
            let (codes, quantized) = vq.encode(&residual)?;
            all_codes.push(codes.unsqueeze(1)?);
            quantized_sum = (quantized_sum + &quantized)?;
            residual = (residual - quantized)?;
        }

        let codes = Tensor::cat(&all_codes, 1)?; // [B, num_codebooks, T]
        Ok((codes, quantized_sum))
    }

    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let mut output = None;
        for (i, vq) in self.quantizers.iter().enumerate() {
            let book_codes = codes.narrow(1, i, 1)?.squeeze(1)?; // [B, T]
            let quantized = vq.lookup(&book_codes)?;
            output = Some(match output {
                Some(o) => (o + quantized)?,
                None => quantized,
            });
        }
        output.ok_or_else(|| candle_core::Error::Msg("No codebooks".to_string()))
    }

    pub fn split_codes(&self, codes: &Tensor) -> Result<(Tensor, Tensor)> {
        let semantic = codes.narrow(1, 0, SEMANTIC_CODEBOOKS)?;
        let acoustic = codes.narrow(1, SEMANTIC_CODEBOOKS, ACOUSTIC_CODEBOOKS)?;
        Ok((semantic, acoustic))
    }
}
```

**Step 4: Run tests**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-codec -- quantizer`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/sethford/Documents/pocket-voice
git add crates/sonata-codec/
git commit -m "feat(v2): add Residual Vector Quantizer with semantic/acoustic split"
```

---

### Task 8: Codec Decoder + Full Codec API

**Files:**
- Create: `crates/sonata-codec/src/decoder.rs`
- Modify: `crates/sonata-codec/src/lib.rs`

**Step 1: Write the failing test**

```rust
// crates/sonata-codec/src/decoder.rs
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};

    #[test]
    fn test_decoder_upsampling_ratio() {
        let dev = &Device::Cpu;
        let decoder = CodecDecoder::new(dev).unwrap();
        // 50 frames should produce 24000 samples
        let latent = Tensor::zeros(&[1, 512, 50], DType::F32, dev).unwrap();
        let audio = decoder.forward(&latent).unwrap();
        assert_eq!(audio.dim(1).unwrap(), 1);     // mono
        assert_eq!(audio.dim(2).unwrap(), 24000);  // 50 * 480
    }
}

// In lib.rs top-level tests:
#[cfg(test)]
mod integration_tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};

    #[test]
    fn test_codec_roundtrip_shape() {
        let dev = &Device::Cpu;
        let codec = SonataCodec::new(dev).unwrap();
        let audio = Tensor::randn(0.0f32, 0.1, &[1, 1, 24000], DType::F32, dev).unwrap();
        let codes = codec.encode(&audio).unwrap();
        assert_eq!(codes.dim(1).unwrap(), 8);   // 8 codebooks
        assert_eq!(codes.dim(2).unwrap(), 50);  // 50 frames per second
        let reconstructed = codec.decode(&codes).unwrap();
        assert_eq!(reconstructed.dims(), audio.dims());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-codec -- decoder integration`
Expected: FAIL

**Step 3: Implement decoder and full codec**

```rust
// crates/sonata-codec/src/decoder.rs

//! 1D Transposed ConvNet decoder: upsample 50Hz latent to 24kHz audio.
//! Strides [3, 4, 5, 8] (reverse of encoder).

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};
use crate::snake::Snake;

pub const DECODER_STRIDES: [usize; 4] = [3, 4, 5, 8];
pub const DECODER_CHANNELS: [usize; 5] = [512, 256, 128, 64, 1];

pub struct DecoderBlock {
    conv_t: ConvTranspose1d,
    snake: Snake,
}

impl DecoderBlock {
    pub fn new(in_ch: usize, out_ch: usize, stride: usize, dev: &Device) -> Result<Self> {
        let kernel = stride * 2;
        let padding = stride / 2;
        let cfg = ConvTranspose1dConfig {
            stride,
            padding,
            output_padding: 0,
            ..Default::default()
        };
        let conv_t = candle_nn::conv_transpose1d(in_ch, out_ch, kernel, cfg,
            VarBuilder::zeros(DType::F32, dev))?;
        let snake = Snake::new(out_ch, dev)?;
        Ok(Self { conv_t, snake })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_t.forward(x)?;
        self.snake.forward(&x)
    }
}

pub struct CodecDecoder {
    blocks: Vec<DecoderBlock>,
}

impl CodecDecoder {
    pub fn new(dev: &Device) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..4 {
            blocks.push(DecoderBlock::new(
                DECODER_CHANNELS[i],
                DECODER_CHANNELS[i + 1],
                DECODER_STRIDES[i],
                dev,
            )?);
        }
        Ok(Self { blocks })
    }

    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let mut x = latent.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}
```

Update `crates/sonata-codec/src/lib.rs` with full API:

```rust
// crates/sonata-codec/src/lib.rs

pub mod snake;
pub mod encoder;
pub mod decoder;
pub mod quantizer;

use candle_core::{Device, Result, Tensor};
use encoder::CodecEncoder;
use decoder::CodecDecoder;
use quantizer::ResidualVQ;
use sonata_common::{NUM_CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM};

/// Sonata Codec: encode 24kHz audio to discrete tokens, decode back.
pub struct SonataCodec {
    encoder: CodecEncoder,
    decoder: CodecDecoder,
    quantizer: ResidualVQ,
}

impl SonataCodec {
    pub fn new(dev: &Device) -> Result<Self> {
        let encoder = CodecEncoder::new(dev)?;
        let decoder = CodecDecoder::new(dev)?;
        let quantizer = ResidualVQ::new(512, NUM_CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM, dev)?;
        Ok(Self { encoder, decoder, quantizer })
    }

    /// Encode audio waveform to codec tokens.
    /// Input: [B, 1, samples] at 24kHz
    /// Output: [B, 8, frames] where frames = samples / 480
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        let latent = self.encoder.forward(audio)?;
        let (codes, _) = self.quantizer.encode(&latent)?;
        Ok(codes)
    }

    /// Decode codec tokens back to audio waveform.
    /// Input: [B, 8, frames]
    /// Output: [B, 1, samples] at 24kHz
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let latent = self.quantizer.decode(codes)?;
        self.decoder.forward(&latent)
    }

    /// Split codes into semantic (books 1-2) and acoustic (books 3-8).
    pub fn split_codes(&self, codes: &Tensor) -> Result<(Tensor, Tensor)> {
        self.quantizer.split_codes(codes)
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};

    #[test]
    fn test_codec_roundtrip_shape() {
        let dev = &Device::Cpu;
        let codec = SonataCodec::new(dev).unwrap();
        let audio = Tensor::randn(0.0f32, 0.1, &[1, 1, 24000], DType::F32, dev).unwrap();
        let codes = codec.encode(&audio).unwrap();
        assert_eq!(codes.dim(1).unwrap(), 8);
        assert_eq!(codes.dim(2).unwrap(), 50);
        let reconstructed = codec.decode(&codes).unwrap();
        assert_eq!(reconstructed.dims(), audio.dims());
    }
}
```

**Step 4: Run tests**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-codec`
Expected: All PASS

**Step 5: Commit**

```bash
cd /Users/sethford/Documents/pocket-voice
git add crates/sonata-codec/
git commit -m "feat(v2): complete Sonata codec (encoder + decoder + RVQ + full API)"
```

---

## Phase 3: CAM++ Speaker Encoder (7.18M params)

### Task 9: CAM++ Crate Scaffold + Frontend

**Files:**
- Create: `crates/sonata-cam/Cargo.toml`
- Create: `crates/sonata-cam/src/lib.rs`
- Create: `crates/sonata-cam/src/frontend.rs`
- Modify: `Cargo.toml` (add to workspace)

**Step 1: Write the failing test**

```rust
// crates/sonata-cam/src/frontend.rs
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};

    #[test]
    fn test_frontend_output_shape() {
        let dev = &Device::Cpu;
        let frontend = MultiScaleFrontend::new(80, 256, dev).unwrap();
        // Input: [B, mel_bins, T] mel spectrogram
        let mel = Tensor::zeros(&[1, 80, 200], DType::F32, dev).unwrap();
        let out = frontend.forward(&mel).unwrap();
        // Output: [B, 256, T'] where T' <= T (frequency pooling may reduce)
        assert_eq!(out.dim(1).unwrap(), 256);
        assert!(out.dim(2).unwrap() > 0);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-cam`
Expected: FAIL

**Step 3: Create crate and implement frontend**

```toml
# crates/sonata-cam/Cargo.toml
[package]
name = "sonata-cam"
version = "0.2.0"
edition = "2021"
publish = false

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
sonata-common = { path = "../sonata-common" }
candle-core = { workspace = true }
candle-nn = { workspace = true }
safetensors = { workspace = true }
anyhow = { workspace = true }
tracing = { workspace = true }

[features]
default = []
metal = ["candle-core/metal", "candle-nn/metal"]
```

```rust
// crates/sonata-cam/src/frontend.rs

//! Multi-Scale Aggregation Frontend for CAM++ speaker encoder.
//! Conv2D stack that processes mel spectrograms.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

pub struct MultiScaleFrontend {
    conv1: Conv1d,
    conv2: Conv1d,
    conv3: Conv1d,
}

impl MultiScaleFrontend {
    pub fn new(in_channels: usize, out_channels: usize, dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);
        let cfg3 = Conv1dConfig { padding: 1, ..Default::default() };
        let conv1 = candle_nn::conv1d(in_channels, 64, 3, cfg3, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv1d(64, 128, 3, cfg3, vb.pp("conv2"))?;
        let conv3 = candle_nn::conv1d(128, out_channels, 3, cfg3, vb.pp("conv3"))?;
        Ok(Self { conv1, conv2, conv3 })
    }

    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(mel)?.relu()?;
        let x = self.conv2.forward(&x)?.relu()?;
        self.conv3.forward(&x)?.relu()
    }
}
```

Add `"crates/sonata-cam"` to workspace members.

**Step 4: Run tests**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-cam`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/sethford/Documents/pocket-voice
git add Cargo.toml crates/sonata-cam/
git commit -m "feat(v2): scaffold sonata-cam crate with multi-scale frontend"
```

---

### Task 10: CAM Block (Context-Aware Masking) + Pooling

**Files:**
- Create: `crates/sonata-cam/src/cam_block.rs`
- Create: `crates/sonata-cam/src/pooling.rs`
- Modify: `crates/sonata-cam/src/lib.rs`

**Step 1: Write the failing test**

```rust
// crates/sonata-cam/src/cam_block.rs
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};

    #[test]
    fn test_cam_block_output_shape() {
        let dev = &Device::Cpu;
        let block = CAMBlock::new(256, 8, dev).unwrap();
        let x = Tensor::zeros(&[1, 256, 100], DType::F32, dev).unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.dims(), x.dims());
    }
}

// crates/sonata-cam/src/pooling.rs
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};

    #[test]
    fn test_attentive_stats_pooling_output() {
        let dev = &Device::Cpu;
        let pool = AttentiveStatsPooling::new(256, 192, dev).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, &[1, 256, 100], DType::F32, dev).unwrap();
        let emb = pool.forward(&x).unwrap();
        assert_eq!(emb.dims(), &[1, 192]); // speaker embedding dim
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-cam -- cam_block pooling`
Expected: FAIL

**Step 3: Implement CAM block and pooling**

```rust
// crates/sonata-cam/src/cam_block.rs

//! Context-Aware Masking block — the key innovation in CAM++.
//! Generates attention masks that focus on speaker-informative frames.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder};

pub struct CAMBlock {
    attention: Linear,
    mask_proj: Linear,
    ffn1: Conv1d,
    ffn2: Conv1d,
    dim: usize,
}

impl CAMBlock {
    pub fn new(dim: usize, num_heads: usize, dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);
        let attention = candle_nn::linear(dim, dim, vb.pp("attn"))?;
        let mask_proj = candle_nn::linear(dim, dim, vb.pp("mask"))?;
        let cfg1 = Conv1dConfig { padding: 1, ..Default::default() };
        let ffn1 = candle_nn::conv1d(dim, dim * 4, 3, cfg1, vb.pp("ffn1"))?;
        let ffn2 = candle_nn::conv1d(dim * 4, dim, 1, Default::default(), vb.pp("ffn2"))?;
        Ok(Self { attention, mask_proj, ffn1, ffn2, dim })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, D, T]
        let x_t = x.transpose(1, 2)?; // [B, T, D]

        // Context-aware mask
        let mask_logits = self.mask_proj.forward(&x_t)?;
        let mask = candle_nn::ops::sigmoid(&mask_logits)?;
        let masked = (x_t.broadcast_mul(&mask))?;

        // Self-attention (simplified as linear projection for now)
        let attn_out = self.attention.forward(&masked)?;
        let x_t = (x_t + attn_out)?; // residual

        // Feed-forward
        let x_conv = x_t.transpose(1, 2)?; // [B, D, T]
        let ff = self.ffn1.forward(&x_conv)?.relu()?;
        let ff = self.ffn2.forward(&ff)?;
        x + ff // residual
    }
}
```

```rust
// crates/sonata-cam/src/pooling.rs

//! Attentive Statistics Pooling for speaker embedding extraction.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};

pub struct AttentiveStatsPooling {
    attention: Linear,
    output_proj: Linear,
}

impl AttentiveStatsPooling {
    pub fn new(in_dim: usize, embed_dim: usize, dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);
        let attention = candle_nn::linear(in_dim, 1, vb.pp("attn"))?;
        // Mean + std concatenated = 2 * in_dim -> embed_dim
        let output_proj = candle_nn::linear(in_dim * 2, embed_dim, vb.pp("proj"))?;
        Ok(Self { attention, output_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, D, T]
        let x_t = x.transpose(1, 2)?; // [B, T, D]

        // Attention weights
        let attn_logits = self.attention.forward(&x_t)?; // [B, T, 1]
        let attn_weights = candle_nn::ops::softmax(&attn_logits, 1)?;

        // Weighted mean
        let weighted = x_t.broadcast_mul(&attn_weights)?; // [B, T, D]
        let mean = weighted.sum(1)?; // [B, D]

        // Weighted std
        let diff = x_t.broadcast_sub(&mean.unsqueeze(1)?)?;
        let weighted_var = diff.sqr()?.broadcast_mul(&attn_weights)?.sum(1)?;
        let std = (weighted_var + 1e-6)?.sqrt()?;

        // Concat mean + std -> project to embed_dim
        let stats = Tensor::cat(&[&mean, &std], D::Minus1)?; // [B, 2*D]
        self.output_proj.forward(&stats)
    }
}
```

**Step 4: Run tests**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-cam`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/sethford/Documents/pocket-voice
git add crates/sonata-cam/
git commit -m "feat(v2): add CAM block (context-aware masking) + attentive stats pooling"
```

---

### Task 11: Full CAM++ Speaker Encoder API

**Files:**
- Modify: `crates/sonata-cam/src/lib.rs`

**Step 1: Write the failing test**

```rust
// crates/sonata-cam/src/lib.rs
#[cfg(test)]
mod integration_tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};
    use sonata_common::SPEAKER_EMBED_DIM;

    #[test]
    fn test_cam_encoder_end_to_end() {
        let dev = &Device::Cpu;
        let encoder = CAMPlusPlusEncoder::new(dev).unwrap();
        // 3 seconds of 16kHz audio -> mel spectrogram
        let mel = Tensor::randn(0.0f32, 1.0, &[1, 80, 300], DType::F32, dev).unwrap();
        let embedding = encoder.encode(&mel).unwrap();
        assert_eq!(embedding.dims(), &[1, SPEAKER_EMBED_DIM]);
    }

    #[test]
    fn test_cam_encoder_different_lengths() {
        let dev = &Device::Cpu;
        let encoder = CAMPlusPlusEncoder::new(dev).unwrap();
        let mel_short = Tensor::randn(0.0f32, 1.0, &[1, 80, 100], DType::F32, dev).unwrap();
        let mel_long = Tensor::randn(0.0f32, 1.0, &[1, 80, 500], DType::F32, dev).unwrap();
        let emb_short = encoder.encode(&mel_short).unwrap();
        let emb_long = encoder.encode(&mel_long).unwrap();
        // Both should produce same-sized embeddings
        assert_eq!(emb_short.dims(), emb_long.dims());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-cam -- integration`
Expected: FAIL

**Step 3: Implement full CAM++ encoder**

```rust
// crates/sonata-cam/src/lib.rs

pub mod frontend;
pub mod cam_block;
pub mod pooling;

use candle_core::{Device, Result, Tensor};
use frontend::MultiScaleFrontend;
use cam_block::CAMBlock;
use pooling::AttentiveStatsPooling;
use sonata_common::SPEAKER_EMBED_DIM;

const CAM_DIM: usize = 256;
const CAM_BLOCKS: usize = 6;
const CAM_HEADS: usize = 8;

/// CAM++ Speaker Encoder: 7.18M params, 0.56% EER on VoxCeleb1-O.
pub struct CAMPlusPlusEncoder {
    frontend: MultiScaleFrontend,
    blocks: Vec<CAMBlock>,
    pooling: AttentiveStatsPooling,
}

impl CAMPlusPlusEncoder {
    pub fn new(dev: &Device) -> Result<Self> {
        let frontend = MultiScaleFrontend::new(80, CAM_DIM, dev)?;
        let mut blocks = Vec::new();
        for _ in 0..CAM_BLOCKS {
            blocks.push(CAMBlock::new(CAM_DIM, CAM_HEADS, dev)?);
        }
        let pooling = AttentiveStatsPooling::new(CAM_DIM, SPEAKER_EMBED_DIM, dev)?;
        Ok(Self { frontend, blocks, pooling })
    }

    /// Encode mel spectrogram to 192-dim speaker embedding.
    /// Input: [B, 80, T] mel spectrogram at 16kHz
    /// Output: [B, 192] speaker embedding
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        let mut x = self.frontend.forward(mel)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.pooling.forward(&x)
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};
    use sonata_common::SPEAKER_EMBED_DIM;

    #[test]
    fn test_cam_encoder_end_to_end() {
        let dev = &Device::Cpu;
        let encoder = CAMPlusPlusEncoder::new(dev).unwrap();
        let mel = Tensor::randn(0.0f32, 1.0, &[1, 80, 300], DType::F32, dev).unwrap();
        let embedding = encoder.encode(&mel).unwrap();
        assert_eq!(embedding.dims(), &[1, SPEAKER_EMBED_DIM]);
    }

    #[test]
    fn test_cam_encoder_different_lengths() {
        let dev = &Device::Cpu;
        let encoder = CAMPlusPlusEncoder::new(dev).unwrap();
        let mel_short = Tensor::randn(0.0f32, 1.0, &[1, 80, 100], DType::F32, dev).unwrap();
        let mel_long = Tensor::randn(0.0f32, 1.0, &[1, 80, 500], DType::F32, dev).unwrap();
        let emb_short = encoder.encode(&mel_short).unwrap();
        let emb_long = encoder.encode(&mel_long).unwrap();
        assert_eq!(emb_short.dims(), emb_long.dims());
    }
}
```

**Step 4: Run tests**

Run: `cd /Users/sethford/Documents/pocket-voice && cargo test -p sonata-cam`
Expected: All PASS

**Step 5: Commit**

```bash
cd /Users/sethford/Documents/pocket-voice
git add crates/sonata-cam/
git commit -m "feat(v2): complete CAM++ speaker encoder (7.18M params, 192-dim embedding)"
```

---

## Phase 4: Sonata STT (~100M params, Enhanced Conformer)

### Task 12: STT Crate Scaffold

**Files:**
- Create: `crates/sonata-stt/Cargo.toml`
- Create: `crates/sonata-stt/src/lib.rs`
- Create: `crates/sonata-stt/src/conformer.rs`
- Modify: `Cargo.toml` (add to workspace)

This task creates the streaming Conformer CTC architecture. The STT takes codec tokens (from Sonata Codec) and outputs text via CTC decoding.

**Step 1-5:** Follow same TDD pattern as previous crates. Key test:

```rust
#[test]
fn test_stt_codec_to_text() {
    let dev = &Device::Cpu;
    let stt = SonataSTT::new(dev).unwrap();
    let codec_embeddings = Tensor::zeros(&[1, 512, 50], DType::F32, dev).unwrap();
    let logits = stt.forward(&codec_embeddings).unwrap();
    // CTC logits: [B, T, vocab_size]
    assert_eq!(logits.dim(2).unwrap(), 32000); // text vocab
}
```

**Commit:** `feat(v2): add Sonata STT (streaming Conformer CTC, ~100M params)`

---

### Task 13: CTC Decoder (Greedy + Beam Search)

### Task 14: STT Streaming Mode

*(Tasks 13-14 follow same TDD pattern — create `crates/sonata-stt/src/ctc.rs` and `crates/sonata-stt/src/streaming.rs`)*

---

## Phase 5: Sonata TTS (~100M params, AdaIN + Emotion + Nonverbal)

### Task 15: TTS Crate Scaffold + Text Encoder

**Files:**
- Create: `crates/sonata-tts/Cargo.toml`
- Create: `crates/sonata-tts/src/lib.rs`
- Create: `crates/sonata-tts/src/text_encoder.rs`

### Task 16: TTS Transformer with AdaIN Conditioning

**Files:**
- Create: `crates/sonata-tts/src/transformer.rs`

Key: every layer has `AdaIN(speaker_embed)` and `AdaIN(emotion_style)` from `sonata-common/adain.rs`.

### Task 17: Emotion Style Tokens + Nonverbal Vocabulary

**Files:**
- Create: `crates/sonata-tts/src/emotion.rs`
- Create: `crates/sonata-tts/src/nonverbal.rs`

Key test:
```rust
#[test]
fn test_emotion_exaggeration_scalar() {
    let style = EmotionStyleEncoder::new(dev).unwrap();
    let neutral = style.encode(0, 1.0).unwrap();    // neutral, normal
    let happy_2x = style.encode(5, 2.0).unwrap();   // happy, exaggerated
    assert_eq!(neutral.dims(), &[1, 192]);           // same dim as speaker
    // Exaggerated should have larger magnitude
    let mag_neutral: f32 = neutral.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
    let mag_happy: f32 = happy_2x.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(mag_happy > mag_neutral);
}
```

### Task 18: Full TTS API (Text → Codec Tokens)

**Commit:** `feat(v2): complete Sonata TTS (AdaIN + emotion + nonverbal, ~100M params)`

---

## Phase 6: CFM Decoder (Conditional Flow Matching, ~35M params)

### Task 19: CFM Crate + ODE Solver

**Files:**
- Create: `crates/sonata-cfm/Cargo.toml`
- Create: `crates/sonata-cfm/src/lib.rs`
- Create: `crates/sonata-cfm/src/ode.rs`

Key: Euler ODE solver for straight-line flow matching (from F5-TTS/CosyVoice research).

```rust
#[test]
fn test_euler_solver_converges() {
    // With identity velocity field, Euler should reach target
    let dev = &Device::Cpu;
    let x0 = Tensor::zeros(&[1, 80, 50], DType::F32, dev).unwrap(); // noise
    let x1 = Tensor::ones(&[1, 80, 50], DType::F32, dev).unwrap();  // target
    let steps = 8;
    // Straight-line: v(x,t) = x1 - x0, so x(1) = x0 + integral(x1-x0, 0, 1) = x1
    let result = euler_solve(|_x, _t| Ok((&x1 - &x0).unwrap()), &x0, steps).unwrap();
    let diff: f32 = (&result - &x1).unwrap().abs().unwrap().mean_all().unwrap().to_scalar().unwrap();
    assert!(diff < 0.01);
}
```

### Task 20: DiT Transformer Blocks (F5-TTS style)

**Files:**
- Create: `crates/sonata-cfm/src/dit.rs`

### Task 21: Full CFM Decoder API (Codec Tokens → Mel Spectrogram)

**Commit:** `feat(v2): complete CFM decoder (4-step Euler, DiT blocks, ~35M params)`

---

## Phase 7: Full-Duplex Controller

### Task 22: Dual-Stream Token Interleaver

**Files:**
- Create: `crates/sonata-pipeline/Cargo.toml`
- Create: `crates/sonata-pipeline/src/lib.rs`
- Create: `crates/sonata-pipeline/src/dual_stream.rs`

### Task 23: Backchannel Generator

**Files:**
- Create: `crates/sonata-pipeline/src/backchannel.rs`

Key: Lightweight model that generates `[hmm]`, `[right]`, `[oh]` etc. while waiting for LLM.

### Task 24: Pipeline Orchestrator (STT → LLM Bridge → TTS)

**Files:**
- Create: `crates/sonata-pipeline/src/orchestrator.rs`

**Commit:** `feat(v2): add full-duplex pipeline (dual-stream + backchannel + orchestrator)`

---

## Phase 8: SeaClaw Integration

### Task 25: Clone SeaClaw + Add Sonata Build Option

**Step 1: Clone seaclaw**

```bash
cd /Users/sethford/Documents
git clone https://github.com/sethdford/seaclaw.git
```

**Step 2: Add CMake option for Sonata**

Modify `CMakeLists.txt`:
```cmake
option(SC_ENABLE_SONATA "Enable native Sonata voice pipeline" OFF)

if(SC_ENABLE_SONATA)
    set(SONATA_DIR "${CMAKE_SOURCE_DIR}/../pocket-voice" CACHE PATH "Path to pocket-voice repo")

    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/libsonata_pipeline.a
        COMMAND cargo build --release --manifest-path ${SONATA_DIR}/Cargo.toml
                -p sonata-pipeline
        COMMENT "Building Sonata voice pipeline (Rust)"
        VERBATIM
    )
    add_custom_target(sonata_build DEPENDS ${CMAKE_BINARY_DIR}/libsonata_pipeline.a)

    add_compile_definitions(SC_HAS_SONATA=1)
endif()
```

**Commit:** `feat(seaclaw): add SC_ENABLE_SONATA build option`

---

### Task 26: Voice Channel Implementation (`sc_channel_voice_t`)

**Files:**
- Create: `src/channels/voice.c` (in seaclaw)
- Create: `include/seaclaw/channels/voice.h` (in seaclaw)

**Step 1: Write the voice channel header**

```c
// include/seaclaw/channels/voice.h
#ifndef SC_CHANNELS_VOICE_H
#define SC_CHANNELS_VOICE_H

#include "seaclaw/channel.h"
#include "seaclaw/core/allocator.h"
#include "seaclaw/core/error.h"

typedef struct sc_channel_voice_config {
    const char *codec_model_path;
    const char *stt_model_path;
    const char *tts_model_path;
    const char *cam_model_path;
    const char *cfm_model_path;
    const char *speaker_id;
    float emotion_exaggeration;   /* 0.0 - 2.0 */
    uint32_t sample_rate;         /* default: 24000 */
    bool enable_full_duplex;
    bool enable_backchanneling;
} sc_channel_voice_config_t;

sc_error_t sc_channel_voice_create(sc_allocator_t *alloc,
    const sc_channel_voice_config_t *config,
    sc_channel_t *out);
void sc_channel_voice_destroy(sc_channel_t *ch);

#endif
```

**Step 2: Implement voice channel**

```c
// src/channels/voice.c
#include "seaclaw/channels/voice.h"
#include <stdlib.h>
#include <string.h>

#ifdef SC_HAS_SONATA
// FFI declarations for Sonata Rust pipeline
extern int32_t sonata_pipeline_init(const char *config_json, size_t config_len);
extern int32_t sonata_stt(const float *audio, size_t samples, char *text, size_t *text_len);
extern int32_t sonata_tts(const char *text, size_t text_len,
                          const char *speaker_id, float emotion_exag,
                          float *audio, size_t *audio_len);
extern void sonata_pipeline_deinit(void);
#endif

typedef struct sc_voice_ctx {
    sc_allocator_t *alloc;
    sc_channel_voice_config_t config;
    bool running;
    /* Callback for delivering transcribed text to agent */
    void (*on_message)(void *ctx, const char *text, size_t text_len);
    void *on_message_ctx;
} sc_voice_ctx_t;

static sc_error_t voice_start(void *ctx) {
    sc_voice_ctx_t *v = (sc_voice_ctx_t *)ctx;
    if (!v) return SC_ERR_INVALID_ARGUMENT;
#ifdef SC_HAS_SONATA
    /* Initialize Sonata pipeline */
    /* In production: serialize config to JSON, call sonata_pipeline_init */
#endif
    v->running = true;
    return SC_OK;
}

static void voice_stop(void *ctx) {
    sc_voice_ctx_t *v = (sc_voice_ctx_t *)ctx;
    if (!v) return;
    v->running = false;
#ifdef SC_HAS_SONATA
    sonata_pipeline_deinit();
#endif
}

static sc_error_t voice_send(void *ctx,
    const char *target, size_t target_len,
    const char *message, size_t message_len,
    const char *const *media, size_t media_count)
{
    sc_voice_ctx_t *v = (sc_voice_ctx_t *)ctx;
    if (!v || !message) return SC_ERR_INVALID_ARGUMENT;

#ifdef SC_HAS_SONATA
    /* Convert text response to speech via Sonata TTS */
    float audio_buf[24000 * 30]; /* up to 30s */
    size_t audio_len = sizeof(audio_buf) / sizeof(float);

    int32_t err = sonata_tts(message, message_len,
        v->config.speaker_id, v->config.emotion_exaggeration,
        audio_buf, &audio_len);
    if (err != 0) return SC_ERR_PROVIDER;

    /* Play audio (platform-specific) */
    /* sc_voice_play(v->alloc, audio_buf, audio_len); */
#endif
    return SC_OK;
}

static const char *voice_name(void *ctx) {
    (void)ctx;
    return "voice";
}

static bool voice_health_check(void *ctx) {
    sc_voice_ctx_t *v = (sc_voice_ctx_t *)ctx;
    return v && v->running;
}

static const sc_channel_vtable_t voice_vtable = {
    .start = voice_start,
    .stop = voice_stop,
    .send = voice_send,
    .name = voice_name,
    .health_check = voice_health_check,
    .send_event = NULL,
    .start_typing = NULL,
    .stop_typing = NULL,
};

sc_error_t sc_channel_voice_create(sc_allocator_t *alloc,
    const sc_channel_voice_config_t *config,
    sc_channel_t *out)
{
    if (!alloc || !out) return SC_ERR_INVALID_ARGUMENT;

    sc_voice_ctx_t *v = (sc_voice_ctx_t *)calloc(1, sizeof(*v));
    if (!v) return SC_ERR_OUT_OF_MEMORY;

    v->alloc = alloc;
    if (config) v->config = *config;
    v->running = false;

    /* Defaults */
    if (v->config.sample_rate == 0) v->config.sample_rate = 24000;
    if (v->config.emotion_exaggeration == 0.0f) v->config.emotion_exaggeration = 1.0f;

    out->ctx = v;
    out->vtable = &voice_vtable;
    return SC_OK;
}

void sc_channel_voice_destroy(sc_channel_t *ch) {
    if (ch && ch->ctx) {
        free(ch->ctx);
        ch->ctx = NULL;
        ch->vtable = NULL;
    }
}
```

**Commit:** `feat(seaclaw): add voice channel with Sonata FFI integration`

---

### Task 27: Register Voice Channel in SeaClaw

**Files:**
- Modify: `include/seaclaw/channel_catalog.h` (add `SC_CHANNEL_VOICE`)
- Modify: `src/channel_catalog.c` (register voice channel)
- Modify: `src/channel_manager.c` (factory for voice channel)

**Commit:** `feat(seaclaw): register voice channel in catalog and manager`

---

### Task 28: Streaming LLM → TTS Bridge

**Files:**
- Create: `crates/sonata-pipeline/src/streaming_bridge.rs`

Key: Connect `sc_stream_callback_t` to Sonata TTS for real-time synthesis as LLM generates tokens.

```rust
/// Bridge between SeaClaw's streaming LLM callback and Sonata TTS.
/// As LLM generates text tokens, feed them to TTS for immediate audio generation.
#[no_mangle]
pub extern "C" fn sonata_streaming_tts_callback(
    ctx: *mut std::ffi::c_void,
    text_delta: *const u8,
    text_len: usize,
    is_final: bool,
) -> i32 {
    // Buffer text until sentence boundary, then synthesize
    // This enables streaming TTS while LLM is still generating
}
```

**Commit:** `feat(v2): add streaming LLM-to-TTS bridge for real-time synthesis`

---

### Task 29: Sonata Pipeline FFI Exports (C-compatible API)

**Files:**
- Create: `crates/sonata-pipeline/src/ffi.rs`

Expose the full Sonata pipeline as C-callable functions for SeaClaw:

```rust
#[no_mangle]
pub extern "C" fn sonata_pipeline_init(config_json: *const u8, config_len: usize) -> i32 { ... }

#[no_mangle]
pub extern "C" fn sonata_stt(audio: *const f32, samples: usize,
                              text: *mut u8, text_len: *mut usize) -> i32 { ... }

#[no_mangle]
pub extern "C" fn sonata_tts(text: *const u8, text_len: usize,
                              speaker_id: *const u8, emotion_exag: f32,
                              audio: *mut f32, audio_len: *mut usize) -> i32 { ... }

#[no_mangle]
pub extern "C" fn sonata_speaker_encode(audio: *const f32, samples: usize,
                                         embedding: *mut f32) -> i32 { ... }

#[no_mangle]
pub extern "C" fn sonata_pipeline_deinit() { ... }
```

**Commit:** `feat(v2): add C-compatible FFI exports for SeaClaw integration`

---

## Phase 9: Training Pipeline (PyTorch)

### Task 30: Codec Training Script

**Files:**
- Create: `train/train_codec.py`
- Create: `train/data/codec_dataset.py`

### Task 31: CAM++ Training Script (Enhance Existing)

**Files:**
- Modify: `train/train_speaker_encoder.py` (add CAM++ architecture option)

### Task 32: STT Training Script

**Files:**
- Create: `train/train_stt.py`

### Task 33: TTS Training Script

**Files:**
- Create: `train/train_tts.py`

### Task 34: Weight Export (PyTorch → Candle Safetensors)

**Files:**
- Create: `train/export_candle.py`

Key: Convert PyTorch state dicts to safetensors format that candle can load.

```python
from safetensors.torch import save_file

def export_to_candle(checkpoint_path, output_path):
    """Convert PyTorch checkpoint to candle-compatible safetensors."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Rename keys to match Rust struct field names
    candle_dict = {}
    for key, tensor in state_dict.items():
        candle_key = key.replace(".", "_")  # candle uses underscore separators
        candle_dict[candle_key] = tensor
    save_file(candle_dict, output_path)
```

**Commit:** `feat(v2): add training pipeline (codec, CAM++, STT, TTS, export)`

---

## Phase 10: Optimization + Final Integration

### Task 35: Metal GPU Acceleration

**Files:**
- Modify: All crate `Cargo.toml` files (enable `metal` feature)
- Create: `crates/sonata-common/src/metal.rs` (Metal device selection)

### Task 36: 4-Bit Quantization for On-Device

**Files:**
- Create: `train/quantize.py`
- Create: `crates/sonata-common/src/quantization.rs`

### Task 37: End-to-End Integration Test

**Files:**
- Create: `tests/integration/test_unified_binary.rs`

Key test: Full pipeline from audio file → STT → mock LLM → TTS → audio file.

```rust
#[test]
fn test_full_pipeline_roundtrip() {
    // Load test audio
    // Encode with codec
    // STT to text
    // Mock LLM response
    // TTS to audio
    // Verify output is valid audio
}
```

**Commit:** `feat(v2): Metal GPU + 4-bit quantization + integration tests`

---

## Summary

| Phase | Tasks | Key Deliverable |
|-------|-------|----------------|
| 1. Foundation | 1-4 | Cargo workspace, shared types, AdaIN, RoPE, SwiGLU |
| 2. Codec | 5-8 | Audio tokenizer (encoder + RVQ + decoder) |
| 3. Speaker | 9-11 | CAM++ encoder (7.18M, 192-dim embedding) |
| 4. STT | 12-14 | Streaming Conformer CTC |
| 5. TTS | 15-18 | AdaIN + emotion + nonverbal TTS |
| 6. CFM | 19-21 | Conditional Flow Matching decoder |
| 7. Full-Duplex | 22-24 | Dual-stream + backchannel + orchestrator |
| 8. SeaClaw | 25-29 | Voice channel + FFI + streaming bridge |
| 9. Training | 30-34 | PyTorch training + export scripts |
| 10. Optimization | 35-37 | Metal GPU + quantization + integration tests |

**Total: 37 tasks across 10 phases**

Each task follows strict TDD: write failing test → implement → verify pass → commit.

---

*Plan created: March 2, 2026*
*Design doc: `docs/plans/2026-03-02-sonata-v2-unified-transformer-design.md`*
*Next step: Choose execution approach (subagent-driven or parallel session)*
