//! Sonata v2 Full-Duplex Pipeline
//!
//! Comprehensive voice pipeline for full-duplex conversation with:
//! - **Dual-stream token interleaving**: Merge user (STT) and agent (TTS) token streams
//! - **Backchannel generation**: Natural engagement signals while processing
//! - **Pipeline orchestration**: Unified audio input/output management
//!
//! # Quick Start
//!
//! ```ignore
//! use sonata_pipeline::{PipelineOrchestrator, DualStreamInterleaver, BackchannelGenerator};
//! use candle_core::Device;
//!
//! let dev = Device::new_metal(0)?;
//! let mut pipeline = PipelineOrchestrator::new(&dev)?;
//! let mut interleaver = DualStreamInterleaver::new();
//! let mut backchannel = BackchannelGenerator::new();
//!
//! // Process user audio
//! let user_audio = /* raw audio */;
//! let text_tokens = pipeline.process_audio(&user_audio)?;
//!
//! // Track in interleaver
//! for (i, token) in text_tokens[0].iter().enumerate() {
//!     interleaver.push_user_token(*token, i as u64 * 20); // 20ms per frame
//! }
//!
//! // Check for backchannel while waiting for LLM
//! if backchannel.should_backchannel(1000, false) {
//!     if let Some(tag) = backchannel.generate(1000, 500) {
//!         // Generate backchannel speech
//!     }
//! }
//! ```
//!
//! # Modules
//!
//! - `dual_stream`: Token interleaving for full-duplex conversation
//! - `backchannel`: Engagement signal generation
//! - `orchestrator`: Pipeline state and model management
//! - `streaming_bridge`: LLM streaming → TTS sentence buffering
//! - `ffi`: C-compatible FFI exports for SeaClaw integration

pub mod backchannel;
pub mod dual_stream;
pub mod ffi;
pub mod orchestrator;
pub mod streaming_bridge;

// Re-exports for convenience
pub use backchannel::BackchannelGenerator;
pub use dual_stream::{DualStreamInterleaver, StreamSource, StreamToken};
pub use ffi::{SC_ERR_INTERNAL, SC_ERR_INVALID_ARGUMENT, SC_ERR_NOT_IMPLEMENTED, SC_ERR_NOT_INITIALIZED, SC_OK};
pub use orchestrator::{ModelDims, PipelineConfig, PipelineOrchestrator, PipelineState};
pub use streaming_bridge::StreamingBridge;
