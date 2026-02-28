/**
 * speaker_encoder_native.h — FFI for native Rust ECAPA-TDNN speaker encoder.
 *
 * High-performance speaker embedding extraction for voice cloning on Apple Silicon.
 * Uses candle-core with Metal GPU acceleration.
 *
 * Architecture: ECAPA-TDNN with SE-Res2Net blocks + attentive statistics pooling
 * Input: 80-bin log-mel spectrogram (16kHz audio)
 * Output: 256-dim L2-normalized speaker d-vector
 *
 * Build: See src/sonata_speaker/Cargo.toml (Rust cdylib)
 * Link: -L src/sonata_speaker/target/release -lsonata_speaker
 */

#ifndef SPEAKER_ENCODER_NATIVE_H
#define SPEAKER_ENCODER_NATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef void* SpeakerEncoderNative;

/**
 * Create a speaker encoder from safetensors weights and config.
 *
 * @param weights_path  Path to .safetensors weight file (e.g., weights.safetensors)
 * @param config_path   Path to JSON config file (e.g., config.json) with architecture params
 * @return              Opaque encoder handle, or NULL on failure
 *
 * Config JSON format:
 * {
 *   "n_mels": 80,
 *   "channels": [1024, 1024, 1024, 1024, 1536],
 *   "kernel_sizes": [5, 3, 3, 3, 1],
 *   "dilations": [1, 2, 3, 4, 1],
 *   "embedding_dim": 256,
 *   "res2net_scale": 8,
 *   "se_channels": 128,
 *   "attention_channels": 128,
 *   "sample_rate": 16000
 * }
 */
SpeakerEncoderNative speaker_encoder_native_create(
    const char *weights_path,
    const char *config_path
);

/** Destroy encoder and free all GPU/CPU resources. */
void speaker_encoder_native_destroy(SpeakerEncoderNative encoder);

/**
 * Get the embedding dimension of this encoder (typically 256 for ECAPA-TDNN).
 */
int speaker_encoder_native_embedding_dim(SpeakerEncoderNative encoder);

/**
 * Get the expected audio sample rate (typically 16000 Hz).
 */
int speaker_encoder_native_sample_rate(SpeakerEncoderNative encoder);

/**
 * Encode from pre-computed mel spectrogram to speaker embedding.
 *
 * The mel spectrogram should be 80-bin log-mel, computed from 16kHz audio.
 * This function is useful when the spectrogram is computed elsewhere (e.g., in STT pipeline).
 *
 * @param encoder    Encoder handle
 * @param mel_data   [n_frames * n_mels] row-major mel spectrogram data (float32)
 *                   Layout: mel_data[frame * n_mels + mel_bin]
 * @param n_frames   Number of frames in spectrogram (>0)
 * @param n_mels     Number of mel bins (typically 80)
 * @param out        Output buffer, must have space for embedding_dim floats
 * @return           Embedding dimension on success, -1 on error
 *
 * Output: out[0..embedding_dim-1] is L2-normalized (unit norm)
 */
int speaker_encoder_native_encode(
    SpeakerEncoderNative encoder,
    const float *mel_data,
    int n_frames,
    int n_mels,
    float *out
);

/**
 * Encode from raw PCM audio to speaker embedding.
 *
 * Convenience function that:
 *   1. Validates audio length and sample rate
 *   2. Resamples to encoder's target sample rate if needed (linear interpolation)
 *   3. Computes 80-bin mel spectrogram internally
 *   4. Encodes mel to 256-dim d-vector
 *
 * @param encoder      Encoder handle
 * @param pcm          Raw mono float32 audio samples (range: -1.0 to 1.0)
 * @param n_samples    Number of audio samples (must be > 0)
 * @param sample_rate  Sample rate of input audio (e.g., 16000, 48000, 8000)
 * @param out          Output buffer, must have space for embedding_dim floats
 * @return             Embedding dimension on success, -1 on error
 *
 * Output: out[0..embedding_dim-1] is L2-normalized
 *
 * Note: For best results, use audio at least 3 seconds long (48000 samples @ 16kHz).
 *       Shorter segments may produce less stable embeddings.
 */
int speaker_encoder_native_encode_audio(
    SpeakerEncoderNative encoder,
    const float *pcm,
    int n_samples,
    int sample_rate,
    float *out
);

#ifdef __cplusplus
}
#endif

#endif /* SPEAKER_ENCODER_NATIVE_H */
