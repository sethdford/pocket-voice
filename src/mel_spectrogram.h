/**
 * mel_spectrogram.h — Streaming mel spectrogram extraction using Apple vDSP.
 *
 * Produces 80-bin log-mel spectrogram frames from raw PCM audio, matching
 * the feature format expected by Conformer/FastConformer ASR models.
 *
 * All DSP runs on the AMX coprocessor via Accelerate — zero Python,
 * zero external dependencies.
 */

#ifndef MEL_SPECTROGRAM_H
#define MEL_SPECTROGRAM_H

#include <stdint.h>

typedef struct MelSpectrogram MelSpectrogram;

typedef struct {
    int sample_rate;    /* Input audio sample rate (default 16000) */
    int n_fft;          /* FFT size, must be power of 2 (default 512) */
    int hop_length;     /* Hop between frames in samples (default 160 = 10ms @ 16kHz) */
    int win_length;     /* Analysis window length (default 400 = 25ms @ 16kHz) */
    int n_mels;         /* Number of mel bins (default 80) */
    float fmin;         /* Minimum mel frequency (default 0) */
    float fmax;         /* Maximum mel frequency (default sample_rate / 2) */
    float log_floor;    /* Floor value before log (default 1e-10) */
} MelConfig;

/* Fill config with standard ASR defaults (16kHz, 80 mels, 25ms window, 10ms hop). */
void mel_config_default(MelConfig *cfg);

/* Create a streaming mel spectrogram extractor. Returns NULL on failure. */
MelSpectrogram *mel_create(const MelConfig *cfg);

/* Destroy extractor, freeing all resources. */
void mel_destroy(MelSpectrogram *mel);

/**
 * Feed PCM audio and extract mel frames.
 *
 * @param mel        Extractor instance
 * @param pcm        Input audio (float32, mono, at configured sample rate)
 * @param n_samples  Number of samples in pcm
 * @param out        Output buffer for mel frames [max_frames * n_mels], row-major
 * @param max_frames Maximum frames the output buffer can hold
 * @return           Number of mel frames written, or -1 on error
 */
int mel_process(MelSpectrogram *mel, const float *pcm, int n_samples,
                float *out, int max_frames);

/* Reset internal state for a new utterance. */
void mel_reset(MelSpectrogram *mel);

/* Returns the number of mel bins (n_mels from config). */
int mel_n_mels(const MelSpectrogram *mel);

/* Returns the hop length in samples. */
int mel_hop_length(const MelSpectrogram *mel);

#endif /* MEL_SPECTROGRAM_H */
