/*
 * sonata_istft.h — Sonata iSTFT audio decoder using Apple vDSP.
 *
 * Converts Vocos-style (magnitude, phase) predictions to audio waveform
 * via inverse Short-Time Fourier Transform on the AMX coprocessor.
 *
 * ~100x faster than ConvTranspose decoder. Zero allocations after init.
 */

#ifndef SONATA_ISTFT_H
#define SONATA_ISTFT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SonataISTFT SonataISTFT;

/* Create decoder. n_fft=1024, hop=480 for 24kHz/50Hz. */
SonataISTFT *sonata_istft_create(int n_fft, int hop_length);

/* Destroy decoder and free all resources. */
void sonata_istft_destroy(SonataISTFT *dec);

/* Reset overlap buffer (call between utterances). */
void sonata_istft_reset(SonataISTFT *dec);

/*
 * Enable streaming mode with ring buffer caching.
 * In streaming mode, the overlap-add accumulator uses a ring buffer
 * (zero memmove) instead of linear shift — ~50% less memory bandwidth
 * for typical hop/window ratios (MS-Wavehax, Interspeech 2025).
 *
 * Both modes produce bit-identical output.
 * enable: nonzero to enable, 0 to disable.
 */
void sonata_istft_set_streaming(SonataISTFT *dec, int enable);

/*
 * Decode one frame: magnitude + phase → audio samples.
 *
 * magnitude: n_fft/2+1 positive values (linear scale)
 * phase:     n_fft/2+1 phase values (radians)
 * out_audio: buffer for hop_length samples (e.g., 480 for 50 Hz @ 24kHz)
 *
 * Returns number of samples written (= hop_length).
 */
int sonata_istft_decode_frame(
    SonataISTFT *dec,
    const float *magnitude,
    const float *phase,
    float *out_audio
);

/*
 * Decode a batch of frames.
 *
 * magnitudes: n_frames × (n_fft/2+1) row-major
 * phases:     n_frames × (n_fft/2+1) row-major
 * out_audio:  buffer for n_frames × hop_length samples
 *
 * Returns total samples written.
 */
int sonata_istft_decode_batch(
    SonataISTFT *dec,
    const float *magnitudes,
    const float *phases,
    int n_frames,
    float *out_audio
);

#ifdef __cplusplus
}
#endif

#endif /* SONATA_ISTFT_H */
