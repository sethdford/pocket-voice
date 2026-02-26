#ifndef SILERO_VAD_H
#define SILERO_VAD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SileroVad SileroVad;

/* Create a Silero VAD engine from an ONNX model file. Returns NULL on failure. */
SileroVad *silero_vad_create(const char *model_path);

/* Destroy engine and free all resources. Safe to call with NULL. */
void silero_vad_destroy(SileroVad *vad);

/**
 * Process a 32ms audio chunk and return speech probability.
 *
 * @param vad       Engine handle
 * @param samples   Exactly 512 float32 samples at 16kHz (32ms)
 * @return          Speech probability [0.0, 1.0], or -1.0 on error
 */
float silero_vad_process(SileroVad *vad, const float *samples);

/**
 * Process audio of arbitrary length, returning per-chunk probabilities.
 *
 * @param vad       Engine handle  
 * @param audio     Float32 audio at 16kHz
 * @param n_samples Total number of samples
 * @param probs_out Output array (must be at least n_samples/512 floats)
 * @param max_probs Size of probs_out
 * @return          Number of probability values written, or -1 on error
 */
int silero_vad_process_audio(SileroVad *vad, const float *audio, int n_samples,
                              float *probs_out, int max_probs);

/* Reset the internal LSTM state (call between utterances). */
void silero_vad_reset(SileroVad *vad);

/* Get the required chunk size in samples (always 512 for 16kHz). */
int silero_vad_chunk_size(const SileroVad *vad);

#ifdef __cplusplus
}
#endif

#endif /* SILERO_VAD_H */
