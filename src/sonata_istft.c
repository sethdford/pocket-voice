/*
 * sonata_istft.c — Sonata iSTFT decoder via Apple Accelerate/vDSP.
 *
 * Converts (magnitude, phase) STFT frames to audio via inverse FFT.
 * All FFT operations run on the AMX coprocessor — concurrent with GPU inference.
 *
 * Architecture:
 *   1. Construct complex STFT: Z = magnitude * exp(j * phase)
 *   2. Inverse FFT via vDSP_fft_zrip (AMX-accelerated)
 *   3. Apply Hann window
 *   4. Overlap-add with previous frame
 *   5. Output hop_length samples
 *
 * At n_fft=1024, hop=480: each call produces 480 samples (20ms at 24kHz).
 * Latency per frame: ~0.02ms (vs ~2ms for ConvTranspose decoder).
 */

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif

#include "sonata_istft.h"
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct SonataISTFT {
    int n_fft;
    int hop_length;
    int n_bins;         /* n_fft/2 + 1 */

    /* vDSP FFT setup */
    FFTSetup fft_setup;
    int log2n;

    /* Hann window */
    float *window;

    /* Split complex buffers for FFT */
    float *real_buf;
    float *imag_buf;

    /* Time-domain frame buffer */
    float *frame_buf;

    /* Overlap-add buffer (n_fft samples) */
    float *overlap_buf;

    /* Ring buffer state for streaming mode (MS-Wavehax caching) */
    int streaming;      /* nonzero = ring buffer mode */
    int ring_head;      /* read/write position in overlap_buf as ring */
};

SonataISTFT *sonata_istft_create(int n_fft, int hop_length) {
    if (n_fft <= 0 || (n_fft & (n_fft - 1)) != 0) {
        fprintf(stderr, "[sonata_istft] n_fft must be a positive power of 2, got %d\n", n_fft);
        return NULL;
    }
    if (hop_length <= 0 || hop_length > n_fft) {
        fprintf(stderr, "[sonata_istft] hop_size must be in (0, n_fft], got %d (n_fft=%d)\n",
                hop_length, n_fft);
        return NULL;
    }

    SonataISTFT *dec = calloc(1, sizeof(SonataISTFT));
    if (!dec) return NULL;

    dec->n_fft = n_fft;
    dec->hop_length = hop_length;
    dec->n_bins = n_fft / 2 + 1;

    /* FFT setup */
    dec->log2n = 0;
    int tmp = n_fft;
    while (tmp > 1) { dec->log2n++; tmp >>= 1; }
    dec->fft_setup = vDSP_create_fftsetup(dec->log2n, FFT_RADIX2);
    if (!dec->fft_setup) { free(dec); return NULL; }

    /* Hann window */
    dec->window = calloc(n_fft, sizeof(float));
    vDSP_hann_window(dec->window, n_fft, vDSP_HANN_DENORM);

    /* Buffers */
    dec->real_buf = calloc(n_fft / 2, sizeof(float));
    dec->imag_buf = calloc(n_fft / 2, sizeof(float));
    dec->frame_buf = calloc(n_fft, sizeof(float));
    dec->overlap_buf = calloc(n_fft, sizeof(float));

    if (!dec->fft_setup || !dec->window || !dec->real_buf ||
        !dec->imag_buf || !dec->frame_buf || !dec->overlap_buf) {
        sonata_istft_destroy(dec);
        return NULL;
    }

    return dec;
}

void sonata_istft_destroy(SonataISTFT *dec) {
    if (!dec) return;
    if (dec->fft_setup) vDSP_destroy_fftsetup(dec->fft_setup);
    free(dec->window);
    free(dec->real_buf);
    free(dec->imag_buf);
    free(dec->frame_buf);
    free(dec->overlap_buf);
    free(dec);
}

void sonata_istft_reset(SonataISTFT *dec) {
    if (!dec) return;
    memset(dec->overlap_buf, 0, dec->n_fft * sizeof(float));
    dec->ring_head = 0;
}

void sonata_istft_set_streaming(SonataISTFT *dec, int enable) {
    if (!dec) return;
    dec->streaming = enable ? 1 : 0;
    /* Reset state when switching modes to avoid stale overlap data */
    sonata_istft_reset(dec);
}

int sonata_istft_decode_frame(
    SonataISTFT *dec,
    const float *magnitude,
    const float *phase,
    float *out_audio
) {
    if (!dec || !magnitude || !phase || !out_audio) return 0;

    const int n_fft = dec->n_fft;
    const int half = n_fft / 2;

    /*
     * Construct complex spectrum from magnitude and phase.
     * Z[k] = magnitude[k] * exp(j * phase[k])
     *       = magnitude[k] * cos(phase[k]) + j * magnitude[k] * sin(phase[k])
     *
     * vDSP uses split complex format with n_fft/2 elements.
     * Pack DC in real[0] and Nyquist in imag[0] for the packed format.
     */

    /* Batch sin/cos for bins 1..half-1 via vecLib — replaces scalar trig calls
       with a single vectorized pass (AMX-accelerated).
       DC (bin 0) and Nyquist (bin half) are real-only — skip them. */
    int sc_count = half - 1;
    vvsincosf(dec->imag_buf + 1, dec->real_buf + 1, phase + 1, &sc_count);
    vDSP_vmul(dec->real_buf + 1, 1, magnitude + 1, 1, dec->real_buf + 1, 1, half - 1);
    vDSP_vmul(dec->imag_buf + 1, 1, magnitude + 1, 1, dec->imag_buf + 1, 1, half - 1);

    /* Pack DC and Nyquist into the split complex format.
     * vDSP packed format: realp[0] = DC (real), imagp[0] = Nyquist (real) */
    DSPSplitComplex split;
    split.realp = dec->real_buf;
    split.imagp = dec->imag_buf;
    split.realp[0] = magnitude[0] * cosf(phase[0]);
    split.imagp[0] = magnitude[half] * cosf(phase[half]);

    /* Inverse FFT */
    vDSP_fft_zrip(dec->fft_setup, &split, 1, dec->log2n, FFT_INVERSE);

    /* Unpack: vDSP_fft_zrip produces interleaved result in split complex.
     * The time-domain signal is stored as:
     *   frame[2k]   = split.realp[k]
     *   frame[2k+1] = split.imagp[k]
     */
    for (int k = 0; k < half; k++) {
        dec->frame_buf[2 * k]     = split.realp[k];
        dec->frame_buf[2 * k + 1] = split.imagp[k];
    }

    /* Scale by 1/(2*n_fft) — vDSP inverse FFT scale factor */
    float scale = 1.0f / (2.0f * n_fft);
    vDSP_vsmul(dec->frame_buf, 1, &scale, dec->frame_buf, 1, n_fft);

    /* Apply Hann window */
    vDSP_vmul(dec->frame_buf, 1, dec->window, 1, dec->frame_buf, 1, n_fft);

    if (dec->streaming) {
        /*
         * Ring buffer overlap-add (MS-Wavehax caching).
         * Uses overlap_buf as a circular buffer indexed by ring_head.
         * Eliminates the memmove on every frame — pure index arithmetic.
         */
        const int hop = dec->hop_length;
        int head = dec->ring_head;
        int first = n_fft - head; /* samples before wrap */

        /* Add frame_buf into ring buffer at [head, head+n_fft) mod n_fft */
        if (first >= n_fft) {
            /* head == 0: no wrap */
            vDSP_vadd(dec->overlap_buf, 1, dec->frame_buf, 1,
                      dec->overlap_buf, 1, n_fft);
        } else {
            /* Two-part add: [head..n_fft) then [0..head) */
            vDSP_vadd(dec->overlap_buf + head, 1, dec->frame_buf, 1,
                      dec->overlap_buf + head, 1, first);
            vDSP_vadd(dec->overlap_buf, 1, dec->frame_buf + first, 1,
                      dec->overlap_buf, 1, n_fft - first);
        }

        /* Copy hop_length samples from ring_head to output */
        int out_first = n_fft - head;
        if (out_first >= hop) {
            memcpy(out_audio, dec->overlap_buf + head, hop * sizeof(float));
            /* Zero consumed samples */
            memset(dec->overlap_buf + head, 0, hop * sizeof(float));
        } else {
            memcpy(out_audio, dec->overlap_buf + head,
                   out_first * sizeof(float));
            memcpy(out_audio + out_first, dec->overlap_buf,
                   (hop - out_first) * sizeof(float));
            /* Zero consumed samples (two parts) */
            memset(dec->overlap_buf + head, 0, out_first * sizeof(float));
            memset(dec->overlap_buf, 0, (hop - out_first) * sizeof(float));
        }

        /* Advance ring head */
        dec->ring_head = (head + hop) % n_fft;

        return hop;
    }

    /* Non-streaming: linear overlap-add with memmove */
    vDSP_vadd(dec->overlap_buf, 1, dec->frame_buf, 1,
              dec->overlap_buf, 1, n_fft);

    /* Copy first hop_length samples to output */
    memcpy(out_audio, dec->overlap_buf, dec->hop_length * sizeof(float));

    /* Shift overlap buffer left by hop_length */
    memmove(dec->overlap_buf, dec->overlap_buf + dec->hop_length,
            (n_fft - dec->hop_length) * sizeof(float));
    memset(dec->overlap_buf + (n_fft - dec->hop_length), 0,
           dec->hop_length * sizeof(float));

    return dec->hop_length;
}

int sonata_istft_decode_batch(
    SonataISTFT *dec,
    const float *magnitudes,
    const float *phases,
    int n_frames,
    float *out_audio
) {
    if (!dec || !magnitudes || !phases || !out_audio || n_frames <= 0) return 0;

    int total_samples = 0;
    for (int f = 0; f < n_frames; f++) {
        int n = sonata_istft_decode_frame(
            dec,
            magnitudes + f * dec->n_bins,
            phases + f * dec->n_bins,
            out_audio + total_samples
        );
        total_samples += n;
    }
    return total_samples;
}
