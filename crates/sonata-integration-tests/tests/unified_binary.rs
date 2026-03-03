//! End-to-end integration tests for the Sonata v2 unified pipeline.
//!
//! Tests the full pipeline from audio → STT → text → TTS → audio,
//! verifying that all components work together correctly.

use candle_core::{Device, DType, Tensor};
use sonata_codec::SonataCodec;
use sonata_stt::SonataSTT;
use sonata_tts::SonataTTS;
use sonata_cfm::SonataCFM;
use sonata_cam::CamPlusPlusEncoder;
use sonata_pipeline::orchestrator::PipelineOrchestrator;
use sonata_common::{
    SAMPLE_RATE, MEL_BINS, SPEAKER_EMBED_DIM, NUM_CODEBOOKS,
};

#[test]
fn test_codec_roundtrip() {
    let dev = Device::Cpu;
    let codec = SonataCodec::new(&dev).unwrap();

    // Create synthetic audio: 1 second at 24kHz
    let audio = Tensor::randn(0.0f32, 0.1, &[1, 1, SAMPLE_RATE as usize], &dev).unwrap();

    // Encode to tokens
    let codes = codec.encode(&audio).unwrap();
    assert_eq!(codes.dim(1).unwrap(), NUM_CODEBOOKS);

    // Decode back
    let reconstructed = codec.decode(&codes).unwrap();
    assert_eq!(reconstructed.dim(0).unwrap(), 1);
    assert_eq!(reconstructed.dim(1).unwrap(), 1);
    // Length should be approximately 24000
    let recon_len = reconstructed.dim(2).unwrap();
    assert!(
        recon_len >= 20000 && recon_len <= 28000,
        "Reconstructed length {} not in expected range",
        recon_len
    );
}

#[test]
fn test_speaker_embedding_extraction() {
    let dev = Device::Cpu;
    let cam = CamPlusPlusEncoder::new(&dev).unwrap();

    // Create mel spectrogram: 3 seconds of audio
    let mel = Tensor::randn(0.0f32, 1.0, &[1, MEL_BINS, 300], &dev).unwrap();

    let embedding = cam.forward(&mel).unwrap();
    assert_eq!(embedding.dims(), &[1, SPEAKER_EMBED_DIM]);
}

#[test]
fn test_stt_pipeline() {
    let dev = Device::Cpu;
    let stt = SonataSTT::new(&dev).unwrap();

    // Simulate codec embeddings
    let codec_embeddings = Tensor::zeros(&[1, 512, 50], DType::F32, &dev).unwrap();

    let text_tokens = stt.transcribe(&codec_embeddings).unwrap();
    assert_eq!(text_tokens.len(), 1);
}

#[test]
fn test_tts_with_emotion() {
    let dev = Device::Cpu;
    let tts = SonataTTS::new(&dev).unwrap();

    let text_tokens = Tensor::zeros(&[1, 20], DType::U32, &dev).unwrap();
    let speaker_emb = Tensor::zeros(&[1, SPEAKER_EMBED_DIM], DType::F32, &dev).unwrap();
    let emotion_emb = tts.emotion_encoder().encode(5, 1.5, &dev).unwrap();

    let logits = tts.forward(&text_tokens, &speaker_emb, &emotion_emb).unwrap();
    assert_eq!(logits.dim(2).unwrap(), 1024);
}

#[test]
fn test_cfm_mel_generation() {
    let dev = Device::Cpu;
    let cfm = SonataCFM::new(&dev).unwrap();

    let speaker = Tensor::zeros(&[1, SPEAKER_EMBED_DIM], DType::F32, &dev).unwrap();
    let mel = cfm.generate(&speaker, 10, 2).unwrap();

    assert_eq!(mel.dims(), &[1, MEL_BINS, 10]);
}

#[test]
fn test_full_pipeline_orchestrator() {
    let dev = Device::Cpu;
    let mut pipeline = PipelineOrchestrator::new(&dev).unwrap();

    // 1. Encode speaker from reference mel
    let ref_mel = Tensor::randn(0.0f32, 1.0, &[1, MEL_BINS, 200], &dev).unwrap();
    let speaker_emb = pipeline.encode_speaker(&ref_mel).unwrap();
    assert_eq!(speaker_emb.dims(), &[1, SPEAKER_EMBED_DIM]);

    // 2. Process audio through STT
    let audio = Tensor::zeros(&[1, 1, SAMPLE_RATE as usize], DType::F32, &dev).unwrap();
    let text_tokens = pipeline.process_audio(&audio).unwrap();
    assert!(!text_tokens.is_empty());

    // 3. Generate speech with emotion
    pipeline.set_emotion(5, 1.2);
    let text_tensor = Tensor::zeros(&[1, 10], DType::U32, &dev).unwrap();
    let logits = pipeline.generate_speech(&text_tensor, &speaker_emb).unwrap();
    assert_eq!(logits.dim(2).unwrap(), 1024);
}

#[test]
fn test_pipeline_model_dimensions() {
    let dims = PipelineOrchestrator::model_dims();
    assert_eq!(dims.audio_sample_rate, 24000);
    assert_eq!(dims.codec_frame_rate, 50);
    assert_eq!(dims.codec_codebooks, 8);
    assert_eq!(dims.stt_vocab_size, 32000);
    assert_eq!(dims.tts_codec_vocab, 1024);
    assert_eq!(dims.speaker_embed_dim, 192);
}

#[test]
fn test_all_components_share_constants() {
    // Verify constants are consistent across crates
    assert_eq!(SAMPLE_RATE, 24000);
    assert_eq!(MEL_BINS, 80);
    assert_eq!(SPEAKER_EMBED_DIM, 192);
    assert_eq!(NUM_CODEBOOKS, 8);
}
