#ifndef NATIVE_VAD_H
#define NATIVE_VAD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NativeVad NativeVad;

/**
 * Create a native C VAD engine from a .nvad binary weights file.
 * Weights are extracted from Silero VAD ONNX via scripts/extract_silero_weights.py.
 * Returns NULL on failure.
 */
NativeVad *native_vad_create(const char *weights_path);

/** Destroy engine and free all resources. Safe to call with NULL. */
void native_vad_destroy(NativeVad *vad);

/**
 * Process a 32ms audio chunk and return speech probability.
 *
 * @param vad       Engine handle
 * @param samples   Exactly 512 float32 samples at 16kHz (32ms)
 * @return          Speech probability [0.0, 1.0], or -1.0 on error
 */
float native_vad_process(NativeVad *vad, const float *samples);

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
int native_vad_process_audio(NativeVad *vad, const float *audio, int n_samples,
                             float *probs_out, int max_probs);

/** Reset LSTM and context state (call between utterances). */
void native_vad_reset(NativeVad *vad);

/** Get the required chunk size in samples (always 512 for 16kHz). */
int native_vad_chunk_size(const NativeVad *vad);

#ifdef __cplusplus
}
#endif

#endif /* NATIVE_VAD_H */
