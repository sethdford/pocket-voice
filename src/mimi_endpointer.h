/**
 * mimi_endpointer.h — Neural audio codec-based end-of-utterance detector.
 *
 * Following arXiv:2506.07081 (Streaming Endpointer for Spoken Dialogue using
 * Neural Audio Codecs and Label-Delayed Training), this module uses Mimi
 * encoder latent features to predict end-of-turn with higher accuracy and
 * lower cutoff error than traditional mel-spectrogram VAD.
 *
 * Architecture:
 *   Mimi encoder latents (D-dim) → LayerNorm → LSTM (hidden_dim) → Linear → sigmoid
 *
 * Predicts 4 classes per frame:
 *   0: silence / non-speech
 *   1: speech active
 *   2: speech ending (transitional)
 *   3: end-of-turn (user is done speaking)
 *
 * Endpoint is triggered when P(end-of-turn) exceeds threshold for N
 * consecutive frames.
 *
 * Advantages over energy VAD:
 *   - Learns linguistic cues for turn completion (question endings, etc.)
 *   - 37-43% lower cutoff error (from the paper)
 *   - Works directly on codec features — no additional feature extraction
 *   - ~0.8M parameters, runs on AMX in < 0.1ms per frame
 *
 * The endpointer runs in parallel with TTS generation since Mimi encoder
 * is already producing latents for voice cloning / codec operations.
 */

#ifndef MIMI_ENDPOINTER_H
#define MIMI_ENDPOINTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MimiEndpointer MimiEndpointer;

/* Endpoint classes */
typedef enum {
    EP_SILENCE   = 0,
    EP_SPEECH    = 1,
    EP_ENDING    = 2,
    EP_END_TURN  = 3,
} EndpointClass;

typedef struct {
    EndpointClass cls;         /* Predicted class */
    float prob_silence;        /* P(silence) */
    float prob_speech;         /* P(speech) */
    float prob_ending;         /* P(ending) */
    float prob_eot;            /* P(end-of-turn) */
    int   consec_eot;          /* Consecutive end-of-turn frames */
    int   triggered;           /* 1 if endpoint threshold exceeded */
} EndpointResult;

/**
 * Create a Mimi endpointer.
 *
 * @param latent_dim    Mimi encoder latent dimension (e.g. 256 or 512)
 * @param hidden_dim    LSTM hidden dimension (e.g. 128 or 256)
 * @param eot_threshold Probability threshold for end-of-turn (e.g. 0.7)
 * @param consec_frames Required consecutive EOT frames to trigger (e.g. 3)
 * @return Opaque handle, or NULL on failure
 */
MimiEndpointer *mimi_ep_create(int latent_dim, int hidden_dim,
                                 float eot_threshold, int consec_frames);

/**
 * Load pre-trained weights from a binary file.
 * Format: [norm_w, norm_b, lstm_Wi, lstm_Wh, lstm_b, linear_w, linear_b]
 * All float32, contiguous.
 */
int mimi_ep_load_weights(MimiEndpointer *ep, const char *path);

/**
 * Initialize with random weights (for testing / fine-tuning from scratch).
 */
void mimi_ep_init_random(MimiEndpointer *ep, uint32_t seed);

void mimi_ep_destroy(MimiEndpointer *ep);

/**
 * Process a single frame of Mimi encoder latents.
 *
 * @param ep       Endpointer instance
 * @param latents  Mimi encoder output for one frame [latent_dim]
 * @return EndpointResult with class probabilities and trigger status
 */
EndpointResult mimi_ep_process(MimiEndpointer *ep, const float *latents);

/**
 * Get the current end-of-turn probability (smoothed).
 * Higher = more confident the user is done speaking.
 */
float mimi_ep_eot_prob(const MimiEndpointer *ep);

/**
 * Check if the endpointer has triggered.
 * Returns 1 if P(EOT) > threshold for consec_frames consecutive frames.
 */
int mimi_ep_triggered(const MimiEndpointer *ep);

/**
 * Reset state for a new utterance. Clears LSTM hidden state and
 * consecutive frame counter.
 */
void mimi_ep_reset(MimiEndpointer *ep);

/**
 * Set the EOT threshold at runtime (for adaptive tuning).
 */
void mimi_ep_set_threshold(MimiEndpointer *ep, float threshold);

/**
 * Get latency in frames since last speech-to-EOT transition.
 * Returns -1 if no EOT has been detected.
 */
int mimi_ep_latency_frames(const MimiEndpointer *ep);

#ifdef __cplusplus
}
#endif

#endif /* MIMI_ENDPOINTER_H */
