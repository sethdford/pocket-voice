/**
 * deep_filter.h — Neural noise suppression via ERB-band gain prediction.
 *
 * DeepFilterNet-inspired architecture:
 *   FFT → ERB-band features → 2-layer GRU (64 units) → gain mask → IFFT
 *
 * Uses Accelerate framework (vDSP for FFT, cblas for matrix ops).
 * Processes 16ms frames (256 samples @ 16kHz) with zero allocations
 * in the hot path. Weights loaded from .dnf binary format.
 *
 * Usage:
 *   DeepFilter *df = deep_filter_create(16000, "models/denoiser.dnf");
 *   deep_filter_process(df, pcm, n_samples);  // in-place
 *   deep_filter_destroy(df);
 */

#ifndef DEEP_FILTER_H
#define DEEP_FILTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DeepFilter DeepFilter;

/**
 * Create a neural noise suppression engine.
 *
 * @param sample_rate  Input sample rate (must be 16000)
 * @param weights_path Path to .dnf weight file, or NULL for passthrough mode
 * @return             Handle, or NULL on failure
 */
DeepFilter *deep_filter_create(int sample_rate, const char *weights_path);

/** Destroy and free all resources. Safe to call with NULL. */
void deep_filter_destroy(DeepFilter *df);

/**
 * Process audio in-place, suppressing non-speech noise.
 *
 * @param df   Engine handle
 * @param pcm  Audio buffer (modified in place)
 * @param n    Number of samples
 */
void deep_filter_process(DeepFilter *df, float *pcm, int n);

/** Reset GRU hidden states (call between utterances or on environment change). */
void deep_filter_reset(DeepFilter *df);

/**
 * Set the noise suppression strength.
 *
 * @param strength  0.0 = passthrough, 1.0 = full suppression (default: 0.8)
 */
void deep_filter_set_strength(DeepFilter *df, float strength);

/**
 * Get the minimum gain floor (prevents over-suppression artifacts).
 * Default: -30 dB (0.0316).
 */
void deep_filter_set_min_gain_db(DeepFilter *df, float min_gain_db);

#ifdef __cplusplus
}
#endif

#endif /* DEEP_FILTER_H */
