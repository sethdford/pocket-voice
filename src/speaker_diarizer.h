/**
 * speaker_diarizer.h — Speaker diarization using ECAPA-TDNN embeddings.
 *
 * Identifies and tracks speakers across a conversation via cosine similarity
 * against running centroid embeddings. Uses the speaker_encoder module for
 * embedding extraction.
 *
 * Typical use:
 *   1. Create diarizer with ECAPA-TDNN model path
 *   2. For each audio segment: id = diarizer_identify(d, audio, n_samples)
 *   3. Or use pre-computed embeddings: id = diarizer_identify_embedding(d, emb, 192)
 *   4. Optionally set labels: diarizer_set_label(d, id, "Alice")
 *
 * Requires: speaker_encoder (ONNX), or NULL encoder_path for embedding-only mode.
 */

#ifndef SPEAKER_DIARIZER_H
#define SPEAKER_DIARIZER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpeakerDiarizer SpeakerDiarizer;

/**
 * Create a speaker diarizer.
 * @param encoder_path  Path to ECAPA-TDNN ONNX model (NULL = embedding-only mode)
 * @param threshold    Cosine similarity threshold for same-speaker (default: 0.75)
 * @param max_speakers  Maximum number of speakers to track (default: 8)
 * @return              Handle, or NULL on failure
 */
SpeakerDiarizer *diarizer_create(const char *encoder_path, float threshold, int max_speakers);

/** Destroy diarizer. Safe to call with NULL. */
void diarizer_destroy(SpeakerDiarizer *d);

/**
 * Identify the speaker of an audio segment.
 * Returns a speaker ID (0-based) or assigns a new one if unknown.
 *
 * @param d         Diarizer handle
 * @param audio     Float32 audio at 16kHz
 * @param n_samples Number of samples (minimum ~0.5s recommended)
 * @return          Speaker ID (0-based), or -1 on error
 */
int diarizer_identify(SpeakerDiarizer *d, const float *audio, int n_samples);

/**
 * Identify speaker from a pre-computed embedding.
 * @param d         Diarizer handle
 * @param embedding 192-dim L2-normalized embedding
 * @param dim       Embedding dimension (must be 192)
 * @return          Speaker ID (0-based), or -1 on error
 */
int diarizer_identify_embedding(SpeakerDiarizer *d, const float *embedding, int dim);

/** Get the number of unique speakers seen so far. */
int diarizer_speaker_count(const SpeakerDiarizer *d);

/**
 * Get the centroid embedding for a speaker.
 * @param d          Diarizer handle
 * @param speaker_id Speaker ID (0-based)
 * @param out        Output buffer (must be at least dim floats)
 * @return           Embedding dimension, or -1 on error
 */
int diarizer_get_embedding(const SpeakerDiarizer *d, int speaker_id, float *out);

/**
 * Set a label for a speaker.
 * @param d          Diarizer handle
 * @param speaker_id Speaker ID
 * @param label      Human-readable name (copied internally)
 * @return           0 on success, -1 on error
 */
int diarizer_set_label(SpeakerDiarizer *d, int speaker_id, const char *label);

/**
 * Get the label for a speaker.
 * @return Label string (owned by diarizer), or NULL
 */
const char *diarizer_get_label(const SpeakerDiarizer *d, int speaker_id);

/** Reset: clear all speaker profiles. */
void diarizer_reset(SpeakerDiarizer *d);

#ifdef __cplusplus
}
#endif

#endif /* SPEAKER_DIARIZER_H */
