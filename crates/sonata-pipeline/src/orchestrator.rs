//! Pipeline orchestrator for full-duplex conversation.
//!
//! Connects STT → LLM Bridge → TTS in a streaming fashion, managing the complete
//! voice processing pipeline with support for:
//! - Audio input processing (raw audio → text tokens via STT)
//! - Speech generation (text tokens → audio via TTS)
//! - Speaker embedding extraction (reference audio → speaker embedding via CAM++)
//! - Full-duplex state management
//!
//! # Architecture
//!
//! The orchestrator maintains references to all model components:
//! - **Codec**: Audio encoding/decoding (24kHz, 8 codebooks)
//! - **STT**: Speech-to-text (Conformer CTC, 32K vocab)
//! - **TTS**: Text-to-speech (Moshi-style with emotion/speaker)
//! - **CFM**: Conditional Flow Matching for mel spectrogram generation
//! - **CAM++**: Speaker embedding encoder (mel → 192-dim)
//!
//! # Example
//! ```ignore
//! let dev = Device::new_metal(0)?;
//! let mut pipeline = PipelineOrchestrator::new(&dev)?;
//! pipeline.set_emotion(5, 1.2); // Excited emotion
//!
//! // Process user audio
//! let user_audio = /* raw 24kHz audio */;
//! let text_tokens = pipeline.process_audio(&user_audio)?;
//!
//! // Generate response speech
//! let speaker_mel = /* reference audio mel */;
//! let speaker_emb = pipeline.encode_speaker(&speaker_mel)?;
//! let response_audio = pipeline.generate_speech(&text_tokens, &speaker_emb)?;
//! ```

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use sonata_cam::CamPlusPlusEncoder;
use sonata_cfm::SonataCFM;
use sonata_codec::SonataCodec;
use sonata_common::mel::MelSpectrogram;
use sonata_common::{EmotionStyle, NUM_CODEBOOKS, SPEAKER_EMBED_DIM};
use sonata_stt::SonataSTT;
use sonata_tts::SonataTTS;
use tracing::{debug, info};

/// Current state of the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineState {
    /// Idle, waiting for input.
    Idle,
    /// Listening to user audio.
    Listening,
    /// Processing through LLM (not in this crate, but tracked for state).
    Processing,
    /// Generating response audio.
    Speaking,
}

/// Configuration for pipeline operation.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Enable full-duplex mode (allow overlapping speech).
    pub enable_full_duplex: bool,
    /// Enable backchannel generation while waiting for LLM.
    pub enable_backchanneling: bool,
    /// Current emotion style ID (0-63).
    pub emotion_style_id: u8,
    /// Emotion exaggeration factor (0.0-2.0).
    pub emotion_exaggeration: f32,
    /// Number of CFM steps for mel generation.
    pub cfm_steps: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_full_duplex: true,
            enable_backchanneling: true,
            emotion_style_id: 0,
            emotion_exaggeration: 1.0,
            cfm_steps: 20,
        }
    }
}

/// Full-duplex voice pipeline orchestrator.
///
/// Manages audio input/output and speech generation/recognition in a unified pipeline.
pub struct PipelineOrchestrator {
    device: Device,
    codec: SonataCodec,
    stt: SonataSTT,
    tts: SonataTTS,
    cfm: SonataCFM,
    cam: CamPlusPlusEncoder,
    mel: MelSpectrogram,
    state: PipelineState,
    config: PipelineConfig,
}

impl PipelineOrchestrator {
    /// Create a new pipeline orchestrator with all sub-models.
    ///
    /// Initializes codec, STT, TTS, CFM, and CAM++ models on the specified device.
    pub fn new(dev: &Device) -> Result<Self> {
        info!("Initializing PipelineOrchestrator on {:?}", dev);

        let codec = SonataCodec::new(dev)
            .map_err(|e| anyhow!("Failed to initialize SonataCodec: {}", e))?;
        debug!("Initialized SonataCodec");

        let stt = SonataSTT::new(dev)
            .map_err(|e| anyhow!("Failed to initialize SonataSTT: {}", e))?;
        debug!("Initialized SonataSTT");

        let tts = SonataTTS::new(dev)
            .map_err(|e| anyhow!("Failed to initialize SonataTTS: {}", e))?;
        debug!("Initialized SonataTTS");

        let cfm = SonataCFM::new(dev)
            .map_err(|e| anyhow!("Failed to initialize SonataCFM: {}", e))?;
        debug!("Initialized SonataCFM");

        let cam = CamPlusPlusEncoder::new(dev)?;
        debug!("Initialized CamPlusPlusEncoder");

        let mel = MelSpectrogram::new(dev)
            .map_err(|e| anyhow!("Failed to initialize MelSpectrogram: {}", e))?;
        debug!("Initialized MelSpectrogram");

        info!("PipelineOrchestrator initialization complete");

        Ok(Self {
            device: dev.clone(),
            codec,
            stt,
            tts,
            cfm,
            cam,
            mel,
            state: PipelineState::Idle,
            config: PipelineConfig::default(),
        })
    }

    /// Get the device this pipeline was initialized on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the current pipeline state.
    pub fn state(&self) -> PipelineState {
        self.state
    }

    /// Set the current pipeline state.
    pub fn set_state(&mut self, state: PipelineState) {
        debug!("Pipeline state transition: {:?} → {:?}", self.state, state);
        self.state = state;
    }

    /// Set the emotion style and exaggeration.
    ///
    /// # Arguments
    /// * `style_id` - Emotion style ID (0-63)
    /// * `exaggeration` - Emotion exaggeration factor (0.0-2.0, 1.0 = neutral)
    pub fn set_emotion(&mut self, style_id: u8, exaggeration: f32) {
        debug!("Setting emotion: style_id={}, exaggeration={}", style_id, exaggeration);
        self.config.emotion_style_id = style_id;
        self.config.emotion_exaggeration = exaggeration.clamp(0.0, 2.0);
    }

    /// Get the current emotion style.
    pub fn emotion(&self) -> EmotionStyle {
        EmotionStyle {
            style_id: self.config.emotion_style_id,
            exaggeration: self.config.emotion_exaggeration,
        }
    }

    /// Set full-duplex mode.
    pub fn set_full_duplex(&mut self, enabled: bool) {
        self.config.enable_full_duplex = enabled;
        debug!("Full-duplex mode: {}", if enabled { "enabled" } else { "disabled" });
    }

    /// Set backchannel generation.
    pub fn set_backchanneling(&mut self, enabled: bool) {
        self.config.enable_backchanneling = enabled;
        debug!("Backchannel generation: {}", if enabled { "enabled" } else { "disabled" });
    }

    /// Set the number of CFM steps for mel generation.
    pub fn set_cfm_steps(&mut self, steps: usize) {
        self.config.cfm_steps = steps;
        debug!("CFM steps: {}", steps);
    }

    /// Get the pipeline configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Process audio input through codec + STT.
    ///
    /// Converts raw audio to codec embeddings, then decodes via STT to get text tokens.
    ///
    /// # Arguments
    /// * `audio` - Raw audio tensor with shape [B, 1, samples] (24kHz mono)
    ///
    /// # Returns
    /// Vector of text token sequences (one per batch element)
    ///
    /// # Flow
    /// 1. Raw audio [B,1,samples] → codec → [B,8,T]
    /// 2. Codec codes → RVQ codebook lookup → embeddings [B,512,T]
    /// 3. Codec embeddings → STT → text token sequences
    pub fn process_audio(&mut self, audio: &Tensor) -> Result<Vec<Vec<u32>>> {
        debug!("Processing audio, shape: {:?}", audio.dims());
        self.set_state(PipelineState::Listening);

        // Encode audio through codec to get embeddings
        let codec_codes = self.codec.encode(audio)
            .map_err(|e| anyhow!("Codec encoding failed: {}", e))?;
        debug!("Codec encoded, shape: {:?}", codec_codes.dims());

        // RVQ decode: look up codebook embeddings and sum across books
        let codec_embeddings = self.codec_codes_to_embeddings(&codec_codes)?;

        // Decode via STT to get text tokens
        let text_tokens = self.stt.transcribe(&codec_embeddings)
            .map_err(|e| anyhow!("STT transcription failed: {}", e))?;
        debug!("STT decoded {} sequences", text_tokens.len());

        Ok(text_tokens)
    }

    /// Generate speech from text tokens with speaker/emotion conditioning.
    ///
    /// # Arguments
    /// * `text_tokens` - Text token tensor [B, T] with shape [batch, seq_len]
    /// * `speaker_emb` - Speaker embedding tensor [B, 192]
    ///
    /// # Returns
    /// Codec logits tensor [B, T, 1024] ready for codec decoding
    pub fn generate_speech(&self, text_tokens: &Tensor, speaker_emb: &Tensor) -> Result<Tensor> {
        debug!("Generating speech from {} tokens", text_tokens.dims()[1]);

        // Prepare emotion embedding
        let emotion = self.emotion();
        let emotion_emb = self.tts.emotion_encoder()
            .encode(emotion.style_id as u32, emotion.exaggeration, &text_tokens.device())
            .map_err(|e| anyhow!("Emotion encoding failed: {}", e))?;

        // TTS forward pass: text_tokens + speaker_emb + emotion_emb → codec logits
        let logits = self.tts.forward(text_tokens, speaker_emb, &emotion_emb)
            .map_err(|e| anyhow!("TTS forward pass failed: {}", e))?;
        debug!("TTS generated logits, shape: {:?}", logits.dims());

        Ok(logits)
    }

    /// Extract speaker embedding from reference audio.
    ///
    /// # Arguments
    /// * `reference_audio_mel` - Mel spectrogram of reference audio [B, 80, T]
    ///
    /// # Returns
    /// Speaker embedding tensor [B, 192]
    pub fn encode_speaker(&self, reference_audio_mel: &Tensor) -> Result<Tensor> {
        debug!("Encoding speaker from mel spectrogram");
        let speaker_emb = self.cam.forward(reference_audio_mel)
            .map_err(|e| anyhow!("Failed to encode speaker: {}", e))?;
        debug!("Speaker embedding encoded, shape: {:?}", speaker_emb.dims());
        Ok(speaker_emb)
    }

    /// Generate mel spectrogram from speaker embedding using CFM.
    ///
    /// Alternative speech synthesis path using Conditional Flow Matching.
    ///
    /// # Arguments
    /// * `speaker_emb` - Speaker embedding [B, 192]
    /// * `num_frames` - Number of mel frames to generate
    ///
    /// # Returns
    /// Mel spectrogram tensor [B, 80, num_frames]
    pub fn generate_mel_spectrogram(&self, speaker_emb: &Tensor, num_frames: usize) -> Result<Tensor> {
        debug!("Generating mel spectrogram for {} frames using CFM", num_frames);
        let mel = self.cfm.generate(speaker_emb, num_frames, self.config.cfm_steps)
            .map_err(|e| anyhow!("CFM mel generation failed: {}", e))?;
        debug!("CFM generated mel, shape: {:?}", mel.dims());
        Ok(mel)
    }

    /// Compute mel spectrogram from raw audio.
    ///
    /// # Arguments
    /// * `audio` - Raw audio tensor [B, samples] or [B, 1, samples]
    ///
    /// # Returns
    /// Mel spectrogram tensor [B, 80, T]
    pub fn compute_mel(&self, audio: &Tensor) -> Result<Tensor> {
        self.mel.forward(audio)
            .map_err(|e| anyhow!("Mel spectrogram computation failed: {}", e))
    }

    /// Convert TTS logits to audio via codec decoding.
    ///
    /// Takes codec logits [B, T, 1024] from `generate_speech()`, argmaxes to get
    /// first-codebook indices, pads remaining 7 codebooks with zeros, and decodes
    /// via the codec to produce audio.
    ///
    /// # Arguments
    /// * `logits` - Codec logits tensor [B, T, 1024] from TTS
    ///
    /// # Returns
    /// Audio tensor [B, 1, audio_samples] at 24kHz
    pub fn logits_to_audio(&self, logits: &Tensor) -> Result<Tensor> {
        let dev = logits.device();
        let dims = logits.dims();
        let batch = dims[0];
        let seq_len = dims[1];

        // Argmax along vocab dimension → [B, T]
        let codes_first = logits.argmax(candle_core::D::Minus1)
            .map_err(|e| anyhow!("Argmax failed: {}", e))?;
        // codes_first is [B, T] with dtype U32

        // Build [B, 8, T]: first codebook = argmax, rest = zeros
        let zeros = Tensor::zeros(&[batch, seq_len], DType::U32, dev)
            .map_err(|e| anyhow!("Failed to create zero codes: {}", e))?;

        let mut book_slices: Vec<Tensor> = Vec::with_capacity(NUM_CODEBOOKS);
        for i in 0..NUM_CODEBOOKS {
            let book = if i == 0 {
                codes_first.unsqueeze(1)
                    .map_err(|e| anyhow!("Unsqueeze failed: {}", e))?
            } else {
                zeros.unsqueeze(1)
                    .map_err(|e| anyhow!("Unsqueeze failed: {}", e))?
            };
            book_slices.push(book);
        }

        let refs: Vec<&Tensor> = book_slices.iter().collect();
        let codes = Tensor::cat(&refs, 1)
            .map_err(|e| anyhow!("Cat codebooks failed: {}", e))?;
        // codes: [B, 8, T]

        debug!("Decoding {} codec frames to audio", seq_len);
        self.codec.decode(&codes)
            .map_err(|e| anyhow!("Codec decode failed: {}", e))
    }

    // --- Raw FFI helpers: work with slices instead of tensors ---

    /// Process raw audio samples through codec + STT, returning token IDs as text.
    ///
    /// Convenience method for FFI: takes `&[f32]` samples and returns a
    /// space-separated string of decoded token IDs.
    ///
    /// Returns an empty string for empty input (not an error).
    pub fn process_audio_raw(&mut self, audio_samples: &[f32]) -> Result<String> {
        if audio_samples.is_empty() {
            return Ok(String::new());
        }

        // Wrap raw samples as tensor [1, 1, samples]
        let audio_tensor = Tensor::new(audio_samples, &self.device)
            .map_err(|e| anyhow!("Tensor creation failed: {}", e))?
            .reshape((1, 1, audio_samples.len()))
            .map_err(|e| anyhow!("Reshape failed: {}", e))?;

        let token_seqs = self.process_audio(&audio_tensor)?;

        // Convert first batch element's tokens to space-separated string
        let tokens = token_seqs.into_iter().next().unwrap_or_default();
        let text = tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(text)
    }

    /// Generate speech audio from text, returning raw f32 samples.
    ///
    /// Uses simple byte-level tokenization (each UTF-8 byte → token ID).
    /// The generated audio exercises the full TTS → codec pipeline but
    /// requires real model weights for meaningful output.
    ///
    /// Returns an empty vec for empty input (not an error).
    pub fn generate_speech_raw(&mut self, text: &str, emotion_exag: f32) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Simple byte-level tokenization (offset by 1 to avoid CTC blank token 0)
        let token_ids: Vec<u32> = text.bytes().map(|b| (b as u32) + 1).collect();
        let text_tokens = Tensor::new(token_ids.as_slice(), &self.device)
            .map_err(|e| anyhow!("Tensor creation failed: {}", e))?
            .unsqueeze(0)
            .map_err(|e| anyhow!("Unsqueeze failed: {}", e))?; // [1, T]

        // Default speaker embedding (zeros)
        let speaker_emb = Tensor::zeros(
            &[1, SPEAKER_EMBED_DIM],
            DType::F32,
            &self.device,
        ).map_err(|e| anyhow!("Speaker embedding creation failed: {}", e))?;

        // Set emotion
        self.set_emotion(0, emotion_exag);

        // TTS forward → logits [1, T, 1024]
        let logits = self.generate_speech(&text_tokens, &speaker_emb)?;

        // Decode logits to audio via codec
        let audio = self.logits_to_audio(&logits)?;

        // Extract audio samples: [1, 1, samples] → Vec<f32>
        let audio_vec = audio
            .squeeze(0)
            .map_err(|e| anyhow!("Squeeze batch failed: {}", e))?
            .squeeze(0)
            .map_err(|e| anyhow!("Squeeze channel failed: {}", e))?
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("to_vec1 failed: {}", e))?;

        Ok(audio_vec)
    }

    /// Encode speaker embedding from raw audio samples.
    ///
    /// Computes mel spectrogram from audio, then extracts a 192-dim
    /// speaker embedding via CAM++.
    ///
    /// Returns 192 zeros for empty input (not an error).
    pub fn encode_speaker_raw(&self, audio_samples: &[f32]) -> Result<Vec<f32>> {
        if audio_samples.is_empty() {
            return Ok(vec![0.0f32; SPEAKER_EMBED_DIM]);
        }

        // Wrap as [1, samples] for mel computation
        let audio_tensor = Tensor::new(audio_samples, &self.device)
            .map_err(|e| anyhow!("Tensor creation failed: {}", e))?
            .unsqueeze(0)
            .map_err(|e| anyhow!("Unsqueeze failed: {}", e))?;

        // Compute mel spectrogram [1, 80, T]
        let mel = self.compute_mel(&audio_tensor)?;

        // Encode speaker [1, 192]
        let speaker_emb = self.encode_speaker(&mel)?;

        // Extract as Vec<f32>
        speaker_emb
            .squeeze(0)
            .map_err(|e| anyhow!("Squeeze failed: {}", e))?
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("to_vec1 failed: {}", e))
    }

    /// Convert codec codes to continuous embeddings via RVQ codebook lookup.
    ///
    /// Performs real codebook embedding lookup (not random data):
    /// For each codebook, look up the embedding for each code index,
    /// then sum across codebooks (RVQ reconstruction).
    ///
    /// Input: codec_codes [B, 8, T] (u32 codebook indices)
    /// Output: embeddings [B, 512, T] (continuous embeddings)
    fn codec_codes_to_embeddings(&self, codec_codes: &Tensor) -> Result<Tensor> {
        self.codec.codes_to_embeddings(codec_codes)
            .map_err(|e| anyhow!("Codec codes_to_embeddings failed: {}", e))
    }

    /// Get model dimension information.
    pub fn model_dims() -> ModelDims {
        ModelDims {
            audio_sample_rate: 24000,
            codec_frame_rate: 50,
            codec_codebooks: 8,
            stt_vocab_size: 32000,
            tts_codec_vocab: 1024,
            speaker_embed_dim: SPEAKER_EMBED_DIM,
            cfm_steps: 20,
        }
    }
}

/// Model dimension information for reference.
#[derive(Debug, Clone)]
pub struct ModelDims {
    pub audio_sample_rate: u32,
    pub codec_frame_rate: usize,
    pub codec_codebooks: usize,
    pub stt_vocab_size: usize,
    pub tts_codec_vocab: usize,
    pub speaker_embed_dim: usize,
    pub cfm_steps: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_orchestrator_creation() {
        let dev = Device::Cpu;
        let result = PipelineOrchestrator::new(&dev);
        assert!(result.is_ok(), "Failed to create orchestrator: {:?}", result.err());
    }

    #[test]
    fn test_orchestrator_initial_state() {
        let dev = Device::Cpu;
        let orch = PipelineOrchestrator::new(&dev).unwrap();
        assert_eq!(orch.state(), PipelineState::Idle);
    }

    #[test]
    fn test_orchestrator_state_transition() {
        let dev = Device::Cpu;
        let mut orch = PipelineOrchestrator::new(&dev).unwrap();

        orch.set_state(PipelineState::Listening);
        assert_eq!(orch.state(), PipelineState::Listening);

        orch.set_state(PipelineState::Processing);
        assert_eq!(orch.state(), PipelineState::Processing);

        orch.set_state(PipelineState::Speaking);
        assert_eq!(orch.state(), PipelineState::Speaking);

        orch.set_state(PipelineState::Idle);
        assert_eq!(orch.state(), PipelineState::Idle);
    }

    #[test]
    fn test_orchestrator_set_emotion() {
        let dev = Device::Cpu;
        let mut orch = PipelineOrchestrator::new(&dev).unwrap();

        orch.set_emotion(5, 1.2);
        let emotion = orch.emotion();
        assert_eq!(emotion.style_id, 5);
        assert!((emotion.exaggeration - 1.2).abs() < 0.01);
    }

    #[test]
    fn test_orchestrator_emotion_clamping() {
        let dev = Device::Cpu;
        let mut orch = PipelineOrchestrator::new(&dev).unwrap();

        // Should clamp to [0.0, 2.0]
        orch.set_emotion(10, -0.5);
        assert_eq!(orch.emotion().exaggeration, 0.0);

        orch.set_emotion(10, 3.5);
        assert_eq!(orch.emotion().exaggeration, 2.0);
    }

    #[test]
    fn test_orchestrator_config_defaults() {
        let dev = Device::Cpu;
        let orch = PipelineOrchestrator::new(&dev).unwrap();
        let config = orch.config();

        assert!(config.enable_full_duplex);
        assert!(config.enable_backchanneling);
        assert_eq!(config.emotion_style_id, 0);
        assert_eq!(config.emotion_exaggeration, 1.0);
        assert_eq!(config.cfm_steps, 20);
    }

    #[test]
    fn test_orchestrator_full_duplex_toggle() {
        let dev = Device::Cpu;
        let mut orch = PipelineOrchestrator::new(&dev).unwrap();

        assert!(orch.config().enable_full_duplex);
        orch.set_full_duplex(false);
        assert!(!orch.config().enable_full_duplex);
        orch.set_full_duplex(true);
        assert!(orch.config().enable_full_duplex);
    }

    #[test]
    fn test_orchestrator_backchannel_toggle() {
        let dev = Device::Cpu;
        let mut orch = PipelineOrchestrator::new(&dev).unwrap();

        assert!(orch.config().enable_backchanneling);
        orch.set_backchanneling(false);
        assert!(!orch.config().enable_backchanneling);
        orch.set_backchanneling(true);
        assert!(orch.config().enable_backchanneling);
    }

    #[test]
    fn test_orchestrator_cfm_steps() {
        let dev = Device::Cpu;
        let mut orch = PipelineOrchestrator::new(&dev).unwrap();

        assert_eq!(orch.config().cfm_steps, 20);
        orch.set_cfm_steps(50);
        assert_eq!(orch.config().cfm_steps, 50);
    }

    #[test]
    fn test_orchestrator_model_dims() {
        let dims = PipelineOrchestrator::model_dims();
        assert_eq!(dims.audio_sample_rate, 24000);
        assert_eq!(dims.codec_frame_rate, 50);
        assert_eq!(dims.codec_codebooks, 8);
        assert_eq!(dims.stt_vocab_size, 32000);
        assert_eq!(dims.tts_codec_vocab, 1024);
        assert_eq!(dims.speaker_embed_dim, 192);
        assert_eq!(dims.cfm_steps, 20);
    }

    #[test]
    fn test_orchestrator_process_audio() {
        let dev = Device::Cpu;
        let mut orch = PipelineOrchestrator::new(&dev).unwrap();

        // Create minimal audio tensor [1, 1, 24000] (1 second at 24kHz)
        let audio = Tensor::zeros(&[1, 1, 24000], DType::F32, &dev).unwrap();

        let result = orch.process_audio(&audio);
        assert!(result.is_ok(), "process_audio failed: {:?}", result.err());

        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 1); // One batch element
    }

    #[test]
    fn test_orchestrator_generate_speech() {
        let dev = Device::Cpu;
        let orch = PipelineOrchestrator::new(&dev).unwrap();

        // Create text tokens [1, 100] (must be U32 for embedding lookup)
        let text_tokens = Tensor::zeros(&[1, 100], DType::U32, &dev).unwrap();

        // Create speaker embedding [1, 192]
        let speaker_emb = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();

        let result = orch.generate_speech(&text_tokens, &speaker_emb);
        assert!(result.is_ok(), "generate_speech failed: {:?}", result.err());
    }

    #[test]
    fn test_orchestrator_encode_speaker() {
        let dev = Device::Cpu;
        let orch = PipelineOrchestrator::new(&dev).unwrap();

        // Create mel spectrogram [1, 80, 500]
        let mel = Tensor::zeros(&[1, 80, 500], DType::F32, &dev).unwrap();

        let result = orch.encode_speaker(&mel);
        assert!(result.is_ok(), "encode_speaker failed: {:?}", result.err());

        if let Ok(emb) = result {
            let dims = emb.dims();
            assert_eq!(dims.len(), 2);
            assert_eq!(dims[0], 1);
            assert_eq!(dims[1], 192);
        }
    }

    #[test]
    fn test_orchestrator_generate_mel() {
        let dev = Device::Cpu;
        let mut orch = PipelineOrchestrator::new(&dev).unwrap();
        orch.set_cfm_steps(2); // Use minimal steps for test speed

        // Create speaker embedding [1, 192]
        let speaker_emb = Tensor::zeros(&[1, 192], DType::F32, &dev).unwrap();

        let result = orch.generate_mel_spectrogram(&speaker_emb, 10);
        assert!(result.is_ok(), "generate_mel_spectrogram failed: {:?}", result.err());

        if let Ok(mel) = result {
            let dims = mel.dims();
            assert_eq!(dims[0], 1);
            assert_eq!(dims[1], 80);
            assert_eq!(dims[2], 10);
        }
    }
}
