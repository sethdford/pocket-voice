/**
 * lufs.h — ITU-R BS.1770 LUFS loudness normalization.
 *
 * Implements perceptual loudness measurement and normalization per the
 * broadcast standard (EBU R128 / ITU-R BS.1770-4). Two-stage K-weighting
 * filter (shelf + highpass) models human loudness perception, followed
 * by sliding-window mean-square measurement with gating.
 *
 * Uses Apple vDSP for vectorized biquad filtering when available.
 */

#ifndef LUFS_H
#define LUFS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LUFSMeter LUFSMeter;

/**
 * Create a LUFS meter.
 * @param sample_rate  Audio sample rate (e.g. 48000)
 * @param window_ms    Measurement window in milliseconds (400 for momentary, 3000 for short-term)
 * @return Opaque handle, or NULL on failure
 */
LUFSMeter *lufs_create(int sample_rate, int window_ms);

/** Destroy and free. NULL-safe. */
void lufs_destroy(LUFSMeter *m);

/** Reset all state (call between utterances). */
void lufs_reset(LUFSMeter *m);

/**
 * Feed audio samples and measure loudness.
 * @param m        Meter handle
 * @param audio    Input samples (mono float32, not modified)
 * @param n        Number of samples
 * @return Current LUFS measurement (negative dB scale, e.g. -23.0)
 */
float lufs_measure(LUFSMeter *m, const float *audio, int n);

/**
 * Normalize audio to target LUFS level.
 * Measures current loudness, computes gain, applies with soft limiting.
 *
 * @param m           Meter handle
 * @param audio       Audio buffer (modified in-place)
 * @param n           Number of samples
 * @param target_lufs Target loudness (e.g. -16.0 for podcast, -23.0 for broadcast)
 * @return Applied gain in dB
 */
float lufs_normalize(LUFSMeter *m, float *audio, int n, float target_lufs);

/**
 * Get the integrated LUFS over all audio fed so far (with gating).
 */
float lufs_integrated(const LUFSMeter *m);

#ifdef __cplusplus
}
#endif

#endif /* LUFS_H */
