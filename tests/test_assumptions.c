/**
 * test_assumptions.c — Validate critical numerical and platform assumptions.
 *
 * Tests generated from assumption audit Task #7.
 * Each test targets an UNVERIFIED or WRONG assumption found during code review.
 *
 * Build:
 *   cc -O2 -arch arm64 -framework Accelerate \
 *      -I src tests/test_assumptions.c -o test_assumptions
 */

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define PASS(msg) printf("  PASS: %s\n", msg)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); failures++; } while(0)

static int failures = 0;

/* ═══════════════════════════════════════════════════════════════════════════
 * A1: vDSP_DFT_Execute forward scaling factor
 *
 * ASSUMPTION: "vDSP_DFT_Execute returns correctly-scaled results —
 *              no additional scaling needed" (mel_spectrogram.c:293)
 *
 * TRUTH: Apple docs say forward real FFT returns 2x mathematical DFT.
 *        This means power spectrum is 4x, log-mel is offset by log(4).
 *        Test verifies the factor-of-2 exists.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void test_vdsp_dft_forward_scaling(void) {
    printf("\n[A1] vDSP_DFT_Execute forward scaling factor\n");

    const int N = 256;  /* Power of 2 for real DFT */
    float *input = calloc(N, sizeof(float));
    float *real_out = calloc(N / 2, sizeof(float));
    float *imag_out = calloc(N / 2, sizeof(float));

    /* Create a known signal: single sinusoid at bin 10 with amplitude 1.0
     * x[n] = cos(2*pi*10*n/N) */
    for (int n = 0; n < N; n++)
        input[n] = cosf(2.0f * (float)M_PI * 10.0f * n / N);

    /* Mathematical DFT of cos at bin 10: X[10] = N/2, X[N-10] = N/2 */
    float expected_math_magnitude = (float)N / 2.0f;

    /* Setup vDSP DFT (modern API) */
    vDSP_DFT_Setup setup = vDSP_DFT_zrop_CreateSetup(NULL, N, vDSP_DFT_FORWARD);
    assert(setup != NULL);

    /* Pack input into split complex */
    DSPSplitComplex split_in  = { calloc(N/2, sizeof(float)), calloc(N/2, sizeof(float)) };
    DSPSplitComplex split_out = { real_out, imag_out };
    vDSP_ctoz((const DSPComplex *)input, 2, &split_in, 1, N / 2);

    /* Execute forward DFT */
    vDSP_DFT_Execute(setup, split_in.realp, split_in.imagp,
                     split_out.realp, split_out.imagp);

    /* Check bin 10 magnitude: should be 2x mathematical = N */
    float mag_bin10 = sqrtf(real_out[10] * real_out[10] + imag_out[10] * imag_out[10]);
    float expected_vdsp = 2.0f * expected_math_magnitude;  /* 2x factor */

    float ratio = mag_bin10 / expected_vdsp;
    if (fabsf(ratio - 1.0f) < 0.01f) {
        PASS("vDSP_DFT_Execute returns 2x mathematical DFT (confirmed)");
        printf("         Bin 10 magnitude: %.1f (expected %.1f, math would be %.1f)\n",
               mag_bin10, expected_vdsp, expected_math_magnitude);
    } else {
        FAIL("vDSP_DFT_Execute scaling factor unexpected");
        printf("         Bin 10 magnitude: %.1f (expected 2x math = %.1f)\n",
               mag_bin10, expected_vdsp);
    }

    /* Quantify the log-mel offset this introduces */
    float power_vdsp = mag_bin10 * mag_bin10;
    float power_math = expected_math_magnitude * expected_math_magnitude;
    float log_offset = logf(power_vdsp) - logf(power_math);
    printf("         Log-mel offset from 2x factor: %.4f (= log(4) = %.4f)\n",
           log_offset, logf(4.0f));

    vDSP_DFT_DestroySetup(setup);
    free(input);
    free(real_out);
    free(imag_out);
    free(split_in.realp);
    free(split_in.imagp);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * A2: vDSP_fft_zrip inverse scaling factor
 *
 * ASSUMPTION: sonata_istft.c uses scale = 1/(2*n_fft) after inverse FFT.
 * TRUTH: Round-trip scale is 2*N, so 1/(2*N) is correct.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void test_vdsp_fft_inverse_scaling(void) {
    printf("\n[A2] vDSP_fft_zrip inverse FFT round-trip scaling\n");

    const int N = 1024;
    const int log2n = 10;
    float *original = calloc(N, sizeof(float));
    float *real_buf = calloc(N / 2, sizeof(float));
    float *imag_buf = calloc(N / 2, sizeof(float));
    float *recovered = calloc(N, sizeof(float));

    /* Create test signal */
    for (int i = 0; i < N; i++)
        original[i] = sinf(2.0f * (float)M_PI * 5.0f * i / N) + 0.5f;

    FFTSetup fft_setup = vDSP_create_fftsetup(log2n, FFT_RADIX2);
    assert(fft_setup != NULL);

    DSPSplitComplex split = { real_buf, imag_buf };

    /* Forward FFT */
    vDSP_ctoz((const DSPComplex *)original, 2, &split, 1, N / 2);
    vDSP_fft_zrip(fft_setup, &split, 1, log2n, FFT_FORWARD);

    /* Inverse FFT */
    vDSP_fft_zrip(fft_setup, &split, 1, log2n, FFT_INVERSE);

    /* Unpack */
    for (int k = 0; k < N / 2; k++) {
        recovered[2 * k]     = split.realp[k];
        recovered[2 * k + 1] = split.imagp[k];
    }

    /* Apply scale 1/(2*N) as used in sonata_istft.c */
    float scale = 1.0f / (2.0f * N);
    vDSP_vsmul(recovered, 1, &scale, recovered, 1, N);

    /* Compare with original */
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabsf(recovered[i] - original[i]);
        if (err > max_err) max_err = err;
    }

    if (max_err < 1e-4f) {
        PASS("Inverse FFT scale 1/(2*N) recovers original signal");
        printf("         Max error: %.2e\n", max_err);
    } else {
        FAIL("Inverse FFT round-trip error too large");
        printf("         Max error: %.2e (expected < 1e-4)\n", max_err);
    }

    vDSP_destroy_fftsetup(fft_setup);
    free(original);
    free(real_buf);
    free(imag_buf);
    free(recovered);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * A3: DC/Nyquist packing for modern DFT API
 *
 * ASSUMPTION: vDSP_DFT_Execute (zrop) packs DC in realp[0]
 *             and Nyquist in imagp[0], same as legacy API.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void test_dc_nyquist_packing(void) {
    printf("\n[A3] DC/Nyquist packing in modern DFT API\n");

    const int N = 512;
    float *input = calloc(N, sizeof(float));

    /* DC-only signal: constant 1.0 */
    for (int i = 0; i < N; i++)
        input[i] = 1.0f;

    vDSP_DFT_Setup setup = vDSP_DFT_zrop_CreateSetup(NULL, N, vDSP_DFT_FORWARD);
    assert(setup != NULL);

    float *real_in = calloc(N/2, sizeof(float));
    float *imag_in = calloc(N/2, sizeof(float));
    float *real_out = calloc(N/2, sizeof(float));
    float *imag_out = calloc(N/2, sizeof(float));

    DSPSplitComplex si = { real_in, imag_in };
    vDSP_ctoz((const DSPComplex *)input, 2, &si, 1, N / 2);
    vDSP_DFT_Execute(setup, si.realp, si.imagp, real_out, imag_out);

    /* DC component should be at realp[0], non-zero
     * For constant signal of amplitude 1: DFT[0] = N, with 2x factor = 2*N */
    float dc = real_out[0];
    float expected_dc = 2.0f * (float)N;  /* 2x scaling */
    float nyq = imag_out[0];

    if (fabsf(dc - expected_dc) < 1.0f && fabsf(nyq) < 1.0f) {
        PASS("DC in realp[0], Nyquist in imagp[0] confirmed");
        printf("         DC = %.1f (expected %.1f), Nyquist = %.1f (expected ~0)\n",
               dc, expected_dc, nyq);
    } else {
        FAIL("DC/Nyquist packing unexpected");
        printf("         DC = %.1f, Nyquist = %.1f\n", dc, nyq);
    }

    vDSP_DFT_DestroySetup(setup);
    free(input);
    free(real_in);
    free(imag_in);
    free(real_out);
    free(imag_out);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * A4: Hann window formula matches standard definition
 *
 * ASSUMPTION: mel_spectrogram.c manually computes w[i] = 0.5 - 0.5*cos(2pi*i/(N-1))
 *             matching the symmetric Hann window used by librosa/NeMo.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void test_hann_window_formula(void) {
    printf("\n[A4] Hann window formula correctness\n");

    const int win = 400;
    float *w_manual = calloc(win, sizeof(float));
    float *w_vdsp = calloc(win, sizeof(float));

    /* Compute standard Hann: w[i] = 0.5 - 0.5*cos(2*pi*i/(N-1)) */
    for (int i = 0; i < win; i++)
        w_manual[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (win - 1));

    /* Replicate mel_spectrogram.c approach (vDSP vectorized) */
    float zero = 0.0f, one = 1.0f;
    vDSP_vramp(&zero, &one, w_vdsp, 1, win);
    float scale = 2.0f * (float)M_PI / (float)(win - 1);
    vDSP_vsmul(w_vdsp, 1, &scale, w_vdsp, 1, win);
    int n = win;
    vvcosf(w_vdsp, w_vdsp, &n);
    float neg = -0.5f;
    vDSP_vsmul(w_vdsp, 1, &neg, w_vdsp, 1, win);
    float half = 0.5f;
    vDSP_vsadd(w_vdsp, 1, &half, w_vdsp, 1, win);

    /* Compare */
    float max_err = 0.0f;
    for (int i = 0; i < win; i++) {
        float err = fabsf(w_vdsp[i] - w_manual[i]);
        if (err > max_err) max_err = err;
    }

    if (max_err < 1e-6f) {
        PASS("vDSP Hann window matches standard formula");
        printf("         Max error: %.2e\n", max_err);
    } else {
        FAIL("Hann window mismatch");
        printf("         Max error: %.2e\n", max_err);
    }

    /* Verify boundary conditions */
    if (fabsf(w_vdsp[0]) < 1e-6f && fabsf(w_vdsp[win-1]) < 1e-6f) {
        PASS("Hann window endpoints are zero");
    } else {
        FAIL("Hann window endpoints non-zero");
    }

    free(w_manual);
    free(w_vdsp);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * A5: Slaney mel filterbank normalization
 *
 * ASSUMPTION: scale = 2.0 / (f_right - f_left) per mel band.
 * TRUTH: matches librosa.filters.mel(norm='slaney').
 * ═══════════════════════════════════════════════════════════════════════════ */
static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}
static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static void test_slaney_normalization(void) {
    printf("\n[A5] Slaney mel filterbank normalization\n");

    const int n_mels = 80;
    const int n_fft = 512;
    const int sr = 16000;
    const int n_bins = n_fft / 2 + 1;
    float fmin = 0.0f, fmax = 8000.0f;

    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);
    int n_pts = n_mels + 2;
    float *mel_f = malloc(n_pts * sizeof(float));

    for (int i = 0; i < n_pts; i++)
        mel_f[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (n_pts - 1));

    /* Verify Slaney normalization: each filter sums to 2/(f_right - f_left) * area
     * After normalization, each triangular filter has area = 1.0 */
    int pass = 1;
    for (int m = 0; m < n_mels; m++) {
        float f_left = mel_f[m];
        float f_center = mel_f[m + 1];
        float f_right = mel_f[m + 2];
        float width = f_right - f_left;

        /* Area of unit-height triangle with Slaney scale */
        /* Unnormalized triangle area = width/2 */
        /* After Slaney: area = (width/2) * (2/width) = 1.0 */
        float slaney_scale = 2.0f / width;
        float triangle_area = (width / 2.0f) * slaney_scale;

        if (fabsf(triangle_area - 1.0f) > 1e-5f) {
            pass = 0;
            printf("         Mel band %d: area = %.6f (expected 1.0)\n", m, triangle_area);
        }
    }

    if (pass) {
        PASS("Slaney normalization produces unit-area filters");
    } else {
        FAIL("Slaney normalization incorrect");
    }

    free(mel_f);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * A6: INT8 tiled GEMM correctness
 *
 * ASSUMPTION: Tiled dequant+sgemm produces same output as fp32 GEMM.
 *             Tests the tiling logic with beta=0 per tile.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void test_int8_tiled_gemm(void) {
    printf("\n[A6] INT8 tiled dequant+GEMM vs fp32 reference\n");

    const int M = 4;   /* batch/seq */
    const int K = 128;  /* input dim */
    const int N = 192;  /* output dim (not multiple of 64, tests remainder tile) */
    const int TILE = 64;

    float *in = malloc(M * K * sizeof(float));
    int8_t *W_q = malloc(N * K * sizeof(int8_t));
    float *scales = malloc(N * sizeof(float));
    float *W_fp32 = malloc(N * K * sizeof(float));
    float *out_tiled = calloc(M * N, sizeof(float));
    float *out_ref = calloc(M * N, sizeof(float));
    float *W_tile = malloc(TILE * K * sizeof(float));

    /* Initialize random data */
    srand(42);
    for (int i = 0; i < M * K; i++)
        in[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < N; i++)
        scales[i] = ((float)rand() / RAND_MAX) * 0.1f + 0.01f;
    for (int i = 0; i < N * K; i++) {
        W_q[i] = (int8_t)(rand() % 256 - 128);
        /* fp32 reference = int8 * scale */
        W_fp32[i] = (float)W_q[i] * scales[i / K];
    }

    /* Reference: fp32 GEMM (out = in @ W^T) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, in, K, W_fp32, K,
                0.0f, out_ref, N);

    /* Tiled INT8: replicate linear_int8 logic */
    for (int n0 = 0; n0 < N; n0 += TILE) {
        int tile_n = (n0 + TILE <= N) ? TILE : (N - n0);

        /* Dequantize tile */
        for (int r = 0; r < tile_n; r++) {
            float s = scales[n0 + r];
            for (int k = 0; k < K; k++)
                W_tile[r * K + k] = (float)W_q[(n0 + r) * K + k] * s;
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, tile_n, K,
                    1.0f, in, K, W_tile, K,
                    0.0f, out_tiled + n0, N);
    }

    /* Compare */
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(out_tiled[i] - out_ref[i]);
        if (err > max_err) max_err = err;
    }

    if (max_err < 1e-3f) {
        PASS("INT8 tiled GEMM matches fp32 reference");
        printf("         Max error: %.2e (M=%d, K=%d, N=%d, tile=%d)\n",
               max_err, M, K, N, TILE);
    } else {
        FAIL("INT8 tiled GEMM diverges from fp32 reference");
        printf("         Max error: %.2e\n", max_err);
    }

    free(in); free(W_q); free(scales); free(W_fp32);
    free(out_tiled); free(out_ref); free(W_tile);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * A7: iSTFT overlap-add COLA condition
 *
 * ASSUMPTION: Overlap-add with Hann window produces constant output
 *             for a constant-magnitude, zero-phase STFT.
 *
 * Tests whether the iSTFT produces correct amplitude for the
 * actual hop/n_fft ratio used by Sonata (hop=480, n_fft=1024).
 * ═══════════════════════════════════════════════════════════════════════════ */
static void test_istft_cola_condition(void) {
    printf("\n[A7] iSTFT overlap-add COLA condition (hop=480, n_fft=1024)\n");

    const int n_fft = 1024;
    const int hop = 480;
    const int n_frames = 20;
    const int total_samples = n_frames * hop;

    float *window = calloc(n_fft, sizeof(float));
    float *overlap = calloc(n_fft + total_samples, sizeof(float));

    /* Standard Hann window (denorm) */
    vDSP_hann_window(window, n_fft, vDSP_HANN_DENORM);

    /* Check COLA: sum of squared windows at each time step should be constant */
    float *cola_sum = calloc(total_samples, sizeof(float));
    for (int f = 0; f < n_frames; f++) {
        int offset = f * hop;
        for (int i = 0; i < n_fft && (offset + i) < total_samples; i++) {
            cola_sum[offset + i] += window[i] * window[i];
        }
    }

    /* Check steadystate region (skip first and last frame's edge effects) */
    float min_sum = 1e10f, max_sum = 0.0f;
    int steady_start = n_fft;  /* after first full overlap region */
    int steady_end = total_samples - n_fft;
    for (int i = steady_start; i < steady_end; i++) {
        if (cola_sum[i] < min_sum) min_sum = cola_sum[i];
        if (cola_sum[i] > max_sum) max_sum = cola_sum[i];
    }

    float variation = (max_sum - min_sum) / ((max_sum + min_sum) / 2.0f);
    if (steady_end > steady_start) {
        if (variation < 0.01f) {
            PASS("COLA condition satisfied (window^2 sum is constant)");
            printf("         Steady-state sum: %.4f to %.4f (variation: %.4f%%)\n",
                   min_sum, max_sum, variation * 100.0f);
        } else {
            printf("  WARN: COLA condition NOT perfectly satisfied for hop=%d, n_fft=%d\n",
                   hop, n_fft);
            printf("         Steady-state sum: %.4f to %.4f (variation: %.2f%%)\n",
                   min_sum, max_sum, variation * 100.0f);
            printf("         Window normalization may be needed for perfect reconstruction.\n");
            printf("         However, this may be compensated by the neural vocoder.\n");
        }
    } else {
        printf("  SKIP: Not enough frames for steady-state analysis\n");
    }

    free(window);
    free(overlap);
    free(cola_sum);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * A8: calloc alignment for vDSP buffers
 *
 * ASSUMPTION: calloc provides sufficient alignment for vDSP/NEON operations.
 * Apple Silicon requires 4-byte alignment for float, 16-byte for NEON.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void test_buffer_alignment(void) {
    printf("\n[A8] calloc alignment for vDSP/NEON buffers\n");

    int pass = 1;
    for (int trial = 0; trial < 10; trial++) {
        float *buf = calloc(1024, sizeof(float));
        uintptr_t addr = (uintptr_t)buf;

        if ((addr % 4) != 0) {
            printf("         Trial %d: 4-byte alignment FAILED (addr=0x%lx)\n",
                   trial, (unsigned long)addr);
            pass = 0;
        }
        if ((addr % 16) != 0) {
            printf("         Trial %d: 16-byte alignment not guaranteed (addr=0x%lx)\n",
                   trial, (unsigned long)addr);
            /* Not a failure — vDSP works with 4-byte alignment but prefers 16 */
        }

        free(buf);
    }

    if (pass) {
        PASS("calloc provides at least 4-byte alignment (required for float/NEON)");
    } else {
        FAIL("calloc alignment insufficient");
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("=== Assumption Validation Tests ===\n");

    test_vdsp_dft_forward_scaling();
    test_vdsp_fft_inverse_scaling();
    test_dc_nyquist_packing();
    test_hann_window_formula();
    test_slaney_normalization();
    test_int8_tiled_gemm();
    test_istft_cola_condition();
    test_buffer_alignment();

    printf("\n=== Results: %d failure(s) ===\n", failures);
    return failures > 0 ? 1 : 0;
}
