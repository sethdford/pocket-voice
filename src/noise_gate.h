/**
 * noise_gate.h — Spectral noise gate for STT preprocessing.
 *
 * Reduces stationary background noise before speech-to-text by:
 *   1. Estimating noise floor during silence frames (adaptive)
 *   2. Per-bin spectral gating: suppress bins below noise floor + margin
 *   3. Smooth gain transitions to avoid musical noise
 *
 * Uses vDSP FFT for Apple Silicon AMX acceleration.
 *
 * Usage:
 *   NoiseGate *ng = noise_gate_create(16000, 512, 256);
 *   noise_gate_process(ng, pcm, n_samples);   // in-place
 *   noise_gate_destroy(ng);
 */

#ifndef NOISE_GATE_H
#define NOISE_GATE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NoiseGate NoiseGate;

/**
 * Create a spectral noise gate.
 *
 * @param sample_rate  Input sample rate (e.g. 16000)
 * @param fft_size     FFT window size (power of 2, e.g. 512)
 * @param hop_size     Hop between frames (e.g. fft_size/2)
 * @return             Handle, or NULL on failure
 */
NoiseGate *noise_gate_create(int sample_rate, int fft_size, int hop_size);

/** Destroy and free resources. */
void noise_gate_destroy(NoiseGate *ng);

/**
 * Process audio in-place, reducing stationary noise.
 * Safe to call on speech — only suppresses bins below noise floor.
 *
 * @param pcm     Audio buffer (modified in place)
 * @param n       Number of samples
 */
void noise_gate_process(NoiseGate *ng, float *pcm, int n);

/** Reset noise estimate (e.g. after environment change). */
void noise_gate_reset(NoiseGate *ng);

/**
 * Set gate parameters.
 *
 * @param threshold_db  Gate threshold above noise floor in dB (default: 6.0)
 * @param attack_ms     Gain smoothing attack time in ms (default: 5.0)
 * @param release_ms    Gain smoothing release time in ms (default: 50.0)
 */
void noise_gate_set_params(NoiseGate *ng, float threshold_db, float attack_ms, float release_ms);

/** Force a noise estimate update from the next N frames of audio. */
void noise_gate_learn_noise(NoiseGate *ng, int n_frames);

#ifdef __cplusplus
}
#endif

#endif /* NOISE_GATE_H */
