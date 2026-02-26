/**
 * speaker_encoder.h — ONNX-based speaker embedding extraction for voice cloning.
 *
 * Extracts a fixed-dimensional speaker embedding from reference audio using
 * a pre-trained speaker verification model (ECAPA-TDNN, ResNet, etc.).
 *
 * The resulting embedding can be passed to sonata_flow_set_speaker_embedding()
 * for zero-shot voice cloning in the Sonata TTS pipeline.
 *
 * Requires: ONNX Runtime (brew install onnxruntime)
 * Models: Any ONNX speaker encoder that takes float32 audio and outputs embeddings.
 *         Recommended: SpeechBrain ECAPA-TDNN (192-dim) or WavLM-based (256-dim).
 */

#ifndef SPEAKER_ENCODER_H
#define SPEAKER_ENCODER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpeakerEncoder SpeakerEncoder;

/**
 * Create a speaker encoder from an ONNX model.
 *
 * @param model_path  Path to .onnx speaker encoder model
 * @return            Opaque handle, or NULL on failure
 */
SpeakerEncoder *speaker_encoder_create(const char *model_path);

/** Destroy encoder and free all resources. */
void speaker_encoder_destroy(SpeakerEncoder *enc);

/**
 * Extract speaker embedding from audio.
 *
 * Audio should be mono, float32, at the model's expected sample rate (typically 16kHz).
 * The embedding is L2-normalized.
 *
 * @param enc            Encoder handle
 * @param audio          Input audio samples (float32, mono)
 * @param n_samples      Number of audio samples
 * @param embedding_out  Output buffer (must be at least embedding_dim() floats)
 * @return               Embedding dimension on success, -1 on error
 */
int speaker_encoder_extract(SpeakerEncoder *enc, const float *audio, int n_samples,
                            float *embedding_out);

/**
 * Get the embedding dimension of this model.
 * Typical values: 192 (ECAPA-TDNN), 256 (WavLM), 512 (ResNet).
 */
int speaker_encoder_embedding_dim(const SpeakerEncoder *enc);

/**
 * Extract embedding from a WAV file.
 * Convenience function that loads audio, resamples to model sample rate if needed.
 *
 * @param enc            Encoder handle
 * @param wav_path       Path to .wav file
 * @param embedding_out  Output buffer
 * @return               Embedding dimension on success, -1 on error
 */
int speaker_encoder_extract_from_wav(SpeakerEncoder *enc, const char *wav_path,
                                     float *embedding_out);

#ifdef __cplusplus
}
#endif

#endif /* SPEAKER_ENCODER_H */
