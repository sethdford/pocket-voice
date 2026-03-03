//! Shared types and constants for Sonata v2 voice pipeline.

use serde::{Deserialize, Serialize};

pub mod adain;

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
