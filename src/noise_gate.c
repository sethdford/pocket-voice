#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "noise_gate.h"

struct NoiseGate {
    int fft_size;
    int half_fft;
    int hop_size;
    int sample_rate;

    FFTSetup fft_setup;
    int log2n;
    float *fft_real;
    float *fft_imag;
    float *window;
    float *overlap_out;    /* Overlap-add output accumulation buffer */
    int    overlap_out_cap;

    float *frame_buf;      /* Accumulation buffer for incoming audio */
    int    frame_len;

    float *output_queue;   /* Queue of processed samples ready for output */
    int    output_len;
    int    output_cap;

    float *noise_floor;    /* Per-bin noise magnitude estimate */
    float *gain_smooth;    /* Per-bin smoothed gain */

    /* Pre-allocated working buffers (eliminates VLAs from hot path) */
    float *work_windowed;  /* [fft_size] */
    float *work_mag;       /* [half_fft] */
    float *work_floor_sc;  /* [half_fft] */
    float *work_recon;     /* [fft_size] */
    int    noise_learned;
    int    learn_frames;
    int    noise_frames;

    float  threshold_db;
    float  attack_coeff;
    float  release_coeff;
};

NoiseGate *noise_gate_create(int sample_rate, int fft_size, int hop_size) {
    if (fft_size < 64 || (fft_size & (fft_size - 1)) != 0) return NULL;
    if (hop_size < 1 || hop_size > fft_size) return NULL;

    NoiseGate *ng = calloc(1, sizeof(NoiseGate));
    if (!ng) return NULL;

    ng->fft_size    = fft_size;
    ng->half_fft    = fft_size / 2;
    ng->hop_size    = hop_size;
    ng->sample_rate = sample_rate;

    ng->log2n = 0;
    for (int t = fft_size; t > 1; t >>= 1) ng->log2n++;
    ng->fft_setup = vDSP_create_fftsetup(ng->log2n, kFFTRadix2);
    if (!ng->fft_setup) { free(ng); return NULL; }

    ng->fft_real      = calloc(ng->half_fft, sizeof(float));
    ng->fft_imag      = calloc(ng->half_fft, sizeof(float));
    ng->window        = calloc(fft_size, sizeof(float));
    ng->overlap_out_cap = fft_size * 2;
    ng->overlap_out   = calloc(ng->overlap_out_cap, sizeof(float));
    ng->frame_buf     = calloc(fft_size, sizeof(float));
    ng->output_cap    = sample_rate * 2;  /* 2 second queue — resized dynamically */
    ng->output_queue  = calloc(ng->output_cap, sizeof(float));
    ng->noise_floor   = calloc(ng->half_fft, sizeof(float));
    ng->gain_smooth   = malloc(ng->half_fft * sizeof(float));
    ng->work_windowed = malloc(fft_size * sizeof(float));
    ng->work_mag      = malloc(ng->half_fft * sizeof(float));
    ng->work_floor_sc = malloc(ng->half_fft * sizeof(float));
    ng->work_recon    = malloc(fft_size * sizeof(float));

    if (!ng->fft_real || !ng->fft_imag || !ng->window || !ng->overlap_out ||
        !ng->frame_buf || !ng->output_queue || !ng->noise_floor || !ng->gain_smooth ||
        !ng->work_windowed || !ng->work_mag || !ng->work_floor_sc || !ng->work_recon) {
        noise_gate_destroy(ng);
        return NULL;
    }

    vDSP_hann_window(ng->window, fft_size, vDSP_HANN_DENORM);

    for (int i = 0; i < ng->half_fft; i++) ng->gain_smooth[i] = 1.0f;

    ng->threshold_db  = 6.0f;
    float attack_ms   = 5.0f;
    float release_ms  = 50.0f;
    float frame_rate  = (float)sample_rate / (float)hop_size;
    ng->attack_coeff  = 1.0f - expf(-1.0f / (attack_ms / 1000.0f * frame_rate));
    ng->release_coeff = 1.0f - expf(-1.0f / (release_ms / 1000.0f * frame_rate));

    ng->learn_frames = 10;

    return ng;
}

void noise_gate_destroy(NoiseGate *ng) {
    if (!ng) return;
    if (ng->fft_setup) vDSP_destroy_fftsetup(ng->fft_setup);
    free(ng->fft_real);
    free(ng->fft_imag);
    free(ng->window);
    free(ng->overlap_out);
    free(ng->frame_buf);
    free(ng->output_queue);
    free(ng->noise_floor);
    free(ng->gain_smooth);
    free(ng->work_windowed);
    free(ng->work_mag);
    free(ng->work_floor_sc);
    free(ng->work_recon);
    free(ng);
}

void noise_gate_reset(NoiseGate *ng) {
    if (!ng) return;
    ng->frame_len = 0;
    ng->output_len = 0;
    ng->noise_learned = 0;
    ng->learn_frames = 10;
    ng->noise_frames = 0;
    memset(ng->overlap_out, 0, ng->overlap_out_cap * sizeof(float));
    memset(ng->noise_floor, 0, ng->half_fft * sizeof(float));
    for (int i = 0; i < ng->half_fft; i++) ng->gain_smooth[i] = 1.0f;
}

void noise_gate_set_params(NoiseGate *ng, float threshold_db, float attack_ms, float release_ms) {
    if (!ng) return;
    ng->threshold_db = threshold_db;
    float frame_rate = (float)ng->sample_rate / (float)ng->hop_size;
    ng->attack_coeff  = 1.0f - expf(-1.0f / (attack_ms / 1000.0f * frame_rate));
    ng->release_coeff = 1.0f - expf(-1.0f / (release_ms / 1000.0f * frame_rate));
}

void noise_gate_learn_noise(NoiseGate *ng, int n_frames) {
    if (!ng) return;
    if (n_frames <= 0) {
        ng->learn_frames = 0;
        ng->noise_learned = 0;
        return;
    }
    ng->learn_frames = n_frames;
    ng->noise_learned = 0;
    ng->noise_frames = 0;
    memset(ng->noise_floor, 0, ng->half_fft * sizeof(float));
}

static void process_one_frame(NoiseGate *ng) {
    int n = ng->fft_size;
    int half = ng->half_fft;
    float *windowed = ng->work_windowed;

    vDSP_vmul(ng->frame_buf, 1, ng->window, 1, windowed, 1, n);

    DSPSplitComplex fft_buf = { ng->fft_real, ng->fft_imag };
    vDSP_ctoz((DSPComplex *)windowed, 2, &fft_buf, 1, half);
    vDSP_fft_zrip(ng->fft_setup, &fft_buf, 1, ng->log2n, FFT_FORWARD);

    float *mag = ng->work_mag;
    vDSP_zvabs(&fft_buf, 1, mag, 1, (vDSP_Length)half);

    /* Noise learning phase */
    if (ng->learn_frames > 0) {
        ng->noise_frames++;
        float alpha = 1.0f / (float)ng->noise_frames;
        for (int i = 0; i < half; i++) {
            ng->noise_floor[i] += alpha * (mag[i] - ng->noise_floor[i]);
        }
        ng->learn_frames--;
        if (ng->learn_frames == 0) ng->noise_learned = 1;
    }

    /* Apply spectral gating with vectorized gain application */
    if (ng->noise_learned) {
        float threshold_lin = powf(10.0f, ng->threshold_db / 20.0f);
        float *floor_scaled = ng->work_floor_sc;
        vDSP_vsmul(ng->noise_floor, 1, &threshold_lin, floor_scaled, 1, half);

        for (int i = 0; i < half; i++) {
            float target_gain;
            if (mag[i] > floor_scaled[i]) {
                target_gain = 1.0f;
            } else if (floor_scaled[i] > 0.0f) {
                float ratio = mag[i] / floor_scaled[i];
                target_gain = ratio * ratio;
            } else {
                target_gain = 0.0f;
            }
            float coeff = (target_gain > ng->gain_smooth[i])
                          ? ng->attack_coeff : ng->release_coeff;
            ng->gain_smooth[i] += coeff * (target_gain - ng->gain_smooth[i]);
        }

        vDSP_vmul(fft_buf.realp, 1, ng->gain_smooth, 1, fft_buf.realp, 1, half);
        vDSP_vmul(fft_buf.imagp, 1, ng->gain_smooth, 1, fft_buf.imagp, 1, half);
    }

    /* IFFT — vDSP forward scales by 2, inverse has no scaling.
     * Round-trip = 2N. With Hann analysis at 50% overlap (no synthesis window),
     * COLA sum = 1.0, so normalize by 1/(2N). */
    vDSP_fft_zrip(ng->fft_setup, &fft_buf, 1, ng->log2n, FFT_INVERSE);
    float scale = 1.0f / (2.0f * n);
    float *reconstructed = ng->work_recon;
    vDSP_ztoc(&fft_buf, 1, (DSPComplex *)reconstructed, 2, half);
    vDSP_vsmul(reconstructed, 1, &scale, reconstructed, 1, n);

    /* With Hann analysis window at 50% overlap, WOLA sums to 1.0 without
     * a synthesis window. Applying it again would halve the energy. */

    /* Overlap-add into output accumulation buffer */
    vDSP_vadd(ng->overlap_out, 1, reconstructed, 1, ng->overlap_out, 1, n);

    /* Extract hop_size samples to output queue */
    int avail = ng->output_cap - ng->output_len;
    int to_copy = ng->hop_size < avail ? ng->hop_size : avail;
    memcpy(ng->output_queue + ng->output_len, ng->overlap_out, to_copy * sizeof(float));
    ng->output_len += to_copy;

    /* Shift overlap buffer left by hop_size */
    memmove(ng->overlap_out, ng->overlap_out + ng->hop_size,
            (ng->overlap_out_cap - ng->hop_size) * sizeof(float));
    memset(ng->overlap_out + ng->overlap_out_cap - ng->hop_size, 0,
           ng->hop_size * sizeof(float));
}

void noise_gate_process(NoiseGate *ng, float *pcm, int n) {
    if (!ng || !pcm || n <= 0) return;

    /* Ensure output queue can hold at least n + fft_size samples */
    int needed_cap = n + ng->fft_size;
    if (needed_cap > ng->output_cap) {
        float *new_q = realloc(ng->output_queue, (size_t)needed_cap * sizeof(float));
        if (new_q) {
            ng->output_queue = new_q;
            ng->output_cap = needed_cap;
        } else {
            fprintf(stderr, "[noise_gate] realloc failed for %d samples\n", needed_cap);
            return;
        }
    }

    int read_pos = 0;

    /* Reset output queue */
    ng->output_len = 0;

    /* Feed all input samples into frame buffer, processing when full */
    while (read_pos < n) {
        int need = ng->fft_size - ng->frame_len;
        int avail = n - read_pos;
        int copy = avail < need ? avail : need;

        memcpy(ng->frame_buf + ng->frame_len, pcm + read_pos, copy * sizeof(float));
        ng->frame_len += copy;
        read_pos += copy;

        if (ng->frame_len >= ng->fft_size) {
            process_one_frame(ng);

            /* Shift frame buffer by hop_size (keep overlap) */
            int remain = ng->fft_size - ng->hop_size;
            memmove(ng->frame_buf, ng->frame_buf + ng->hop_size, remain * sizeof(float));
            ng->frame_len = remain;
        }
    }

    /* Copy processed output back to pcm (up to n samples) */
    int out_copy = ng->output_len < n ? ng->output_len : n;
    if (out_copy > 0) {
        memcpy(pcm, ng->output_queue, out_copy * sizeof(float));
    }

    /* If we produced fewer samples than input, zero-fill the rest */
    for (int i = out_copy; i < n; i++) {
        pcm[i] = 0.0f;
    }
}
