/**
 * vdsp_prosody.c — AMX-accelerated prosody processing for SSML.
 *
 * Replaces numpy-based pitch shifting, time stretching, and volume control
 * with native Apple Accelerate vDSP operations running on the AMX coprocessor.
 * All operations are zero-copy where possible, operating on the audio buffer
 * in-place.
 *
 * Features:
 *   - Phase vocoder (STFT→modify→ISTFT) for pitch shift using vDSP FFT
 *   - WSOLA-based time stretching without phase artifacts
 *   - vDSP biquad cascade for parametric EQ (formant preservation)
 *   - Soft-knee limiter using vForce transcendentals
 *   - All operations fused into a single pass where possible
 *
 * Build:
 *   cc -O3 -shared -fPIC -arch arm64 -framework Accelerate \
 *      -o libvdsp_prosody.dylib vdsp_prosody.c
 */

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Configuration
 * ----------------------------------------------------------------------- */

#define MAX_FFT_SIZE      4096
#define MAX_BIQUAD_STAGES 4
#define MAX_RAMP_SIZE     16384

/* -----------------------------------------------------------------------
 * FFT Context (cached DFT setups)
 * ----------------------------------------------------------------------- */

typedef struct {
    FFTSetup fft_setup;
    int fft_log2n;
    int fft_size;
    int hop_size;

    float window[MAX_FFT_SIZE];
    DSPSplitComplex fft_buf;
    float fft_real[MAX_FFT_SIZE / 2];
    float fft_imag[MAX_FFT_SIZE / 2];

    float phase_accum[MAX_FFT_SIZE / 2];
    float prev_phase[MAX_FFT_SIZE / 2];
} FFTContext;

static int fft_ctx_init(FFTContext *ctx, int fft_size) {
    if (fft_size > MAX_FFT_SIZE) return -1;
    ctx->fft_size = fft_size;
    ctx->hop_size = fft_size / 4;

    int log2n = 0, n = fft_size;
    while (n > 1) { n >>= 1; log2n++; }
    ctx->fft_log2n = log2n;

    ctx->fft_setup = vDSP_create_fftsetup(log2n, kFFTRadix2);
    if (!ctx->fft_setup) return -1;

    ctx->fft_buf.realp = ctx->fft_real;
    ctx->fft_buf.imagp = ctx->fft_imag;

    vDSP_hann_window(ctx->window, fft_size, vDSP_HANN_NORM);

    memset(ctx->phase_accum, 0, sizeof(ctx->phase_accum));
    memset(ctx->prev_phase, 0, sizeof(ctx->prev_phase));

    return 0;
}

static void fft_ctx_destroy(FFTContext *ctx) {
    if (ctx->fft_setup) vDSP_destroy_fftsetup(ctx->fft_setup);
    ctx->fft_setup = NULL;
}

/* -----------------------------------------------------------------------
 * Phase Vocoder — Pitch Shifting
 * ----------------------------------------------------------------------- */

int prosody_pitch_shift(const float *input, float *output, int n_samples,
                        float pitch_factor, int fft_size) {
    FFTContext ctx;
    if (fft_ctx_init(&ctx, fft_size) != 0) return -1;

    int hop = ctx.hop_size;
    int half = fft_size / 2;
    float freq_per_bin = 2.0f * (float)M_PI / (float)fft_size;
    float expect = 2.0f * (float)M_PI * (float)hop / (float)fft_size;

    memset(output, 0, n_samples * sizeof(float));

    float frame[MAX_FFT_SIZE];
    float magnitudes[MAX_FFT_SIZE / 2];
    float frequencies[MAX_FFT_SIZE / 2];
    float synth_real[MAX_FFT_SIZE / 2];
    float synth_imag[MAX_FFT_SIZE / 2];
    float synth_frame[MAX_FFT_SIZE];

    int n_frames = (n_samples - fft_size) / hop + 1;

    for (int f = 0; f < n_frames; f++) {
        int offset = f * hop;

        if (offset + fft_size <= n_samples) {
            vDSP_vmul(input + offset, 1, ctx.window, 1, frame, 1, fft_size);
        } else {
            int avail = n_samples - offset;
            memcpy(frame, input + offset, avail * sizeof(float));
            memset(frame + avail, 0, (fft_size - avail) * sizeof(float));
            vDSP_vmul(frame, 1, ctx.window, 1, frame, 1, fft_size);
        }

        vDSP_ctoz((DSPComplex *)frame, 2, &ctx.fft_buf, 1, half);
        vDSP_fft_zrip(ctx.fft_setup, &ctx.fft_buf, 1, ctx.fft_log2n, kFFTDirection_Forward);

        for (int k = 0; k < half; k++) {
            float re = ctx.fft_buf.realp[k];
            float im = ctx.fft_buf.imagp[k];
            magnitudes[k] = sqrtf(re * re + im * im);

            float phase = atan2f(im, re);
            float dp = phase - ctx.prev_phase[k];
            ctx.prev_phase[k] = phase;

            dp -= (float)k * expect;
            dp = fmodf(dp + (float)M_PI, 2.0f * (float)M_PI) - (float)M_PI;

            frequencies[k] = (float)k * freq_per_bin + dp / (float)hop;
        }

        memset(synth_real, 0, half * sizeof(float));
        memset(synth_imag, 0, half * sizeof(float));

        for (int k = 0; k < half; k++) {
            int target = (int)(k * pitch_factor + 0.5f);
            if (target >= 0 && target < half) {
                ctx.phase_accum[target] += frequencies[k] * pitch_factor * (float)hop;
                synth_real[target] += magnitudes[k] * cosf(ctx.phase_accum[target]);
                synth_imag[target] += magnitudes[k] * sinf(ctx.phase_accum[target]);
            }
        }

        DSPSplitComplex synth_buf = { synth_real, synth_imag };
        vDSP_fft_zrip(ctx.fft_setup, &synth_buf, 1, ctx.fft_log2n, kFFTDirection_Inverse);

        vDSP_ztoc(&synth_buf, 1, (DSPComplex *)synth_frame, 2, half);
        float scale = 1.0f / (2.0f * fft_size);
        vDSP_vsmul(synth_frame, 1, &scale, synth_frame, 1, fft_size);

        vDSP_vmul(synth_frame, 1, ctx.window, 1, synth_frame, 1, fft_size);

        int add_len = (offset + fft_size <= n_samples) ? fft_size : n_samples - offset;
        vDSP_vadd(output + offset, 1, synth_frame, 1, output + offset, 1, add_len);
    }

    fft_ctx_destroy(&ctx);
    return 0;
}

/* -----------------------------------------------------------------------
 * WSOLA Time Stretching
 * ----------------------------------------------------------------------- */

int prosody_time_stretch(const float *input, int in_len, float *output,
                         float rate_factor, float window_ms, int sample_rate) {
    int win_size = (int)(window_ms * sample_rate / 1000.0f);
    if (win_size > MAX_FFT_SIZE) win_size = MAX_FFT_SIZE;
    int hop_analysis = win_size / 2;
    int hop_synthesis = (int)(hop_analysis * rate_factor);

    float window[MAX_FFT_SIZE];
    vDSP_hann_window(window, win_size, vDSP_HANN_NORM);

    int out_len = (int)(in_len * rate_factor);
    memset(output, 0, out_len * sizeof(float));

    int out_pos = 0;
    float in_pos_f = 0.0f;

    while (out_pos + win_size < out_len) {
        int in_pos = (int)in_pos_f;
        if (in_pos + win_size > in_len) break;

        int search_range = hop_analysis / 2;
        int best_offset = 0;
        float best_corr = -1e30f;

        for (int delta = -search_range; delta <= search_range; delta++) {
            int pos = in_pos + delta;
            if (pos < 0 || pos + win_size > in_len) continue;

            float corr;
            int cmp_len = (out_pos + win_size <= out_len) ? win_size : out_len - out_pos;
            vDSP_dotpr(input + pos, 1, output + out_pos, 1, &corr, cmp_len);
            if (corr > best_corr) {
                best_corr = corr;
                best_offset = delta;
            }
        }

        int src_pos = in_pos + best_offset;
        if (src_pos < 0) src_pos = 0;
        if (src_pos + win_size > in_len) src_pos = in_len - win_size;

        for (int i = 0; i < win_size && out_pos + i < out_len; i++) {
            output[out_pos + i] += input[src_pos + i] * window[i];
        }

        out_pos += hop_synthesis;
        in_pos_f += hop_analysis;
    }

    return out_pos < out_len ? out_pos : out_len;
}

/* -----------------------------------------------------------------------
 * Biquad Filter Cascade — Parametric EQ
 * ----------------------------------------------------------------------- */

typedef struct {
    vDSP_biquad_Setup setup;
    double coefficients[5 * MAX_BIQUAD_STAGES];
    float delays[2 * MAX_BIQUAD_STAGES + 2];
    int n_stages;
} BiquadCascade;

static void biquad_compute_peaking(double *coeffs, float fc, float gain_db,
                                    float Q, float fs) {
    double A = pow(10.0, (double)gain_db / 40.0);
    double w0 = 2.0 * M_PI * (double)fc / (double)fs;
    double alpha = sin(w0) / (2.0 * (double)Q);

    double b0 = 1.0 + alpha * A;
    double b1 = -2.0 * cos(w0);
    double b2 = 1.0 - alpha * A;
    double a0 = 1.0 + alpha / A;
    double a1 = -2.0 * cos(w0);
    double a2 = 1.0 - alpha / A;

    coeffs[0] = b0 / a0;
    coeffs[1] = b1 / a0;
    coeffs[2] = b2 / a0;
    coeffs[3] = a1 / a0;
    coeffs[4] = a2 / a0;
}

BiquadCascade *prosody_create_formant_eq(float pitch_factor, int sample_rate) {
    BiquadCascade *bc = (BiquadCascade *)calloc(1, sizeof(BiquadCascade));
    if (!bc) return NULL;

    float fs = (float)sample_rate;
    float inv_shift = 1.0f / pitch_factor;

    float formants[3] = {500.0f, 1500.0f, 2500.0f};
    float gains_db[3] = {3.0f, 2.0f, 1.0f};

    bc->n_stages = 3;
    for (int i = 0; i < 3; i++) {
        float correction_db = gains_db[i] * (pitch_factor - 1.0f) * 2.0f;
        biquad_compute_peaking(&bc->coefficients[i * 5],
                               formants[i] * inv_shift, correction_db,
                               2.0f, fs);
    }

    bc->setup = vDSP_biquad_CreateSetup(bc->coefficients, bc->n_stages);
    memset(bc->delays, 0, sizeof(bc->delays));

    return bc;
}

int prosody_apply_biquad(BiquadCascade *bc, float *audio, int n_samples) {
    if (!bc || !bc->setup) return -1;
    vDSP_biquad(bc->setup, bc->delays, audio, 1, audio, 1, n_samples);
    return 0;
}

void prosody_destroy_biquad(BiquadCascade *bc) {
    if (!bc) return;
    if (bc->setup) vDSP_biquad_DestroySetup(bc->setup);
    free(bc);
}

/* -----------------------------------------------------------------------
 * Soft-Knee Limiter — vForce accelerated
 * ----------------------------------------------------------------------- */

void prosody_soft_limit(float *audio, int n_samples,
                        float threshold, float knee_db) {
    float gain = 1.0f / threshold;
    float knee_factor = 1.0f / (knee_db / 6.0f);

    float *temp = (float *)malloc(n_samples * sizeof(float));

    float combined = gain * knee_factor;
    vDSP_vsmul(audio, 1, &combined, temp, 1, n_samples);

    int ni = n_samples;
    vvtanhf(temp, temp, &ni);

    float inv_scale = threshold / knee_factor;
    vDSP_vsmul(temp, 1, &inv_scale, audio, 1, n_samples);

    free(temp);
}

/* -----------------------------------------------------------------------
 * Volume Control — vDSP vectorized
 * ----------------------------------------------------------------------- */

void prosody_volume(float *audio, int n_samples, float volume_db,
                    float fade_ms, int sample_rate) {
    float gain = powf(10.0f, volume_db / 20.0f);

    if (fade_ms <= 0 || n_samples <= 0) {
        vDSP_vsmul(audio, 1, &gain, audio, 1, n_samples);
        return;
    }

    int fade_samples = (int)(fade_ms * sample_rate / 1000.0f);
    if (fade_samples > n_samples) fade_samples = n_samples;
    if (fade_samples > MAX_RAMP_SIZE) fade_samples = MAX_RAMP_SIZE;

    float ramp[MAX_RAMP_SIZE];
    float start = 1.0f;
    float step = (gain - 1.0f) / (float)fade_samples;
    vDSP_vramp(&start, &step, ramp, 1, fade_samples);

    vDSP_vmul(audio, 1, ramp, 1, audio, 1, fade_samples);

    if (n_samples > fade_samples) {
        vDSP_vsmul(audio + fade_samples, 1, &gain,
                   audio + fade_samples, 1, n_samples - fade_samples);
    }
}

/* -----------------------------------------------------------------------
 * Crossfade — Fused ramp + multiply + add via vDSP
 * ----------------------------------------------------------------------- */

void prosody_crossfade(const float *seg_a, const float *seg_b,
                       float *output, int n_samples) {
    float fade_out[MAX_RAMP_SIZE], fade_in[MAX_RAMP_SIZE];
    int n = n_samples > MAX_RAMP_SIZE ? MAX_RAMP_SIZE : n_samples;

    float start_out = 1.0f, step_out = -1.0f / (float)n;
    vDSP_vramp(&start_out, &step_out, fade_out, 1, n);

    float start_in = 0.0f, step_in = 1.0f / (float)n;
    vDSP_vramp(&start_in, &step_in, fade_in, 1, n);

    vDSP_vmul(seg_a, 1, fade_out, 1, output, 1, n);
    vDSP_vma(seg_b, 1, fade_in, 1, output, 1, output, 1, n);
}
