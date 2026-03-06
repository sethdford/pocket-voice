/**
 * speaker_encoder.h — Zero-shot voice cloning via speaker encoder.
 *
 * ECAPA-TDNN speaker encoder for extracting voice characteristics from
 * short audio clips (3-10 seconds). Produces fixed-size embeddings that
 * condition the flow model for voice-prompted speech synthesis.
 *
 * Architecture: SE-Res2Net blocks + Attentive Statistics Pooling → 256D embedding
 * Input: 16kHz PCM audio → Mel spectrogram (80 bins) → Normalized embedding
 * Training: GE2E loss on LibriTTS-R
 *
 * Usage:
 *   SpeakerEncoder *enc = speaker_encoder_create("path/to/speaker_encoder.safetensors");
 *   float embedding[256];
 *   speaker_encoder_encode_audio(enc, pcm, 16000*3, 16000, embedding);
 *   sonata_flow_set_speaker_embedding(flow, embedding, 256);
 *   speaker_encoder_destroy(enc);
 */

#ifndef SPEAKER_ENCODER_H
#define SPEAKER_ENCODER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpeakerEncoder SpeakerEncoder;

/**
 * Create speaker encoder from safetensors weights.
 *
 * @param weights_path  Path to speaker_encoder.safetensors checkpoint
 * @return              Encoder instance, or NULL on failure
 *
 * Loads ECAPA-TDNN architecture with pre-trained weights.
 * Initializes mel spectrogram extractor (80 bins, 16kHz).
 */
SpeakerEncoder *speaker_encoder_create(const char *weights_path);

/**
 * Destroy encoder, freeing all resources.
 */
void speaker_encoder_destroy(SpeakerEncoder *enc);

/**
 * Extract speaker embedding from PCM audio.
 *
 * @param enc         Encoder instance
 * @param pcm         Input audio (float32, mono, at sample_rate)
 * @param n_samples   Number of audio samples
 * @param sample_rate Sample rate of input audio (e.g., 16000, 24000)
 * @param out_emb     Output buffer for embedding [256] L2-normalized
 * @return            0 on success, -1 on error
 *
 * Resamples audio to 16kHz if needed, extracts mel spectrogram,
 * runs through ECAPA-TDNN, and returns L2-normalized 256D embedding.
 *
 * Typical usage: 3-10 seconds of speech for best voice characterization.
 */
int speaker_encoder_encode_audio(SpeakerEncoder *enc, const float *pcm,
                                  int n_samples, int sample_rate,
                                  float *out_emb);

/**
 * Extract embedding from mel spectrogram.
 *
 * @param enc         Encoder instance
 * @param mel         Mel spectrogram [n_frames * 80]
 * @param n_frames    Number of mel frames
 * @param out_emb     Output buffer for embedding [256] L2-normalized
 * @return            0 on success, -1 on error
 *
 * For advanced use: if you already have mel features from another source,
 * pass them directly to the encoder backbone.
 */
int speaker_encoder_encode_mel(SpeakerEncoder *enc, const float *mel,
                                int n_frames, float *out_emb);

/**
 * Get embedding dimension (always 256).
 */
int speaker_encoder_embedding_dim(const SpeakerEncoder *enc);

/**
 * Get sample rate (always 16000 for ECAPA-TDNN input).
 */
int speaker_encoder_sample_rate(const SpeakerEncoder *enc);

#ifdef __cplusplus
}
#endif

#endif /* SPEAKER_ENCODER_H */
