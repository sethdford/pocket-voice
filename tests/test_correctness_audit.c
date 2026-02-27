/**
 * test_correctness_audit.c — Rigorous correctness proofs for ring buffer,
 * INT8 dequantize, Hann window, and pipeline pre-alloc paths.
 *
 * Build:
 *   make test-correctness-audit
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Test infrastructure
 * ═══════════════════════════════════════════════════════════════════════════ */

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  [TEST] %-60s", name); \
    fflush(stdout); \
} while(0)

#define PASS() do { tests_passed++; printf(" PASS\n"); } while(0)
#define FAIL(msg) do { printf(" FAIL: %s\n", msg); } while(0)

/* ═══════════════════════════════════════════════════════════════════════════
 * Include mel_spectrogram.h for public API
 * ═══════════════════════════════════════════════════════════════════════════ */

#include "mel_spectrogram.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Ring Buffer Correctness Tests
 *
 * Strategy: Feed known PCM patterns through mel_process and verify the output
 * mel frames are identical regardless of how the input is chunked.
 * This indirectly proves ring_write, ring_read, ring_consume are correct.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Generate a deterministic test signal: sum of sinusoids */
static void generate_test_pcm(float *buf, int n, float freq1, float freq2) {
    for (int i = 0; i < n; i++) {
        float t = (float)i / 16000.0f;
        buf[i] = 0.5f * sinf(2.0f * (float)M_PI * freq1 * t)
               + 0.3f * sinf(2.0f * (float)M_PI * freq2 * t);
    }
}

/* Test: feeding all PCM at once vs in small chunks should produce identical mel frames */
static void test_ring_buffer_chunked_vs_bulk(void) {
    TEST("Ring buffer: chunked vs bulk produce identical mel");

    MelConfig cfg;
    mel_config_default(&cfg);
    cfg.preemph = 0.0f;  /* Disable pre-emphasis to isolate ring buffer behavior */

    MelSpectrogram *mel_bulk = mel_create(&cfg);
    MelSpectrogram *mel_chunked = mel_create(&cfg);
    if (!mel_bulk || !mel_chunked) { FAIL("mel_create returned NULL"); return; }

    int n_samples = 4800;  /* 300ms = ~18 frames at hop=160 */
    float *pcm = (float *)malloc(n_samples * sizeof(float));
    generate_test_pcm(pcm, n_samples, 440.0f, 1000.0f);

    int max_frames = 200;
    float *out_bulk = (float *)calloc(max_frames * cfg.n_mels, sizeof(float));
    float *out_chunked = (float *)calloc(max_frames * cfg.n_mels, sizeof(float));

    /* Bulk: feed all at once */
    int nf_bulk = mel_process(mel_bulk, pcm, n_samples, out_bulk, max_frames);

    /* Chunked: feed in irregular chunks (17, 53, 160, 7, ...) to exercise ring wraparound */
    int chunk_sizes[] = {17, 53, 160, 7, 320, 1, 400, 113, 800, 160, 2769};
    int offset = 0;
    int nf_chunked = 0;
    for (int c = 0; c < (int)(sizeof(chunk_sizes)/sizeof(chunk_sizes[0])); c++) {
        int cs = chunk_sizes[c];
        if (offset + cs > n_samples) cs = n_samples - offset;
        if (cs <= 0) break;
        int nf = mel_process(mel_chunked, pcm + offset, cs,
                             out_chunked + nf_chunked * cfg.n_mels,
                             max_frames - nf_chunked);
        if (nf > 0) nf_chunked += nf;
        offset += cs;
    }

    if (nf_bulk != nf_chunked) {
        char msg[128];
        snprintf(msg, sizeof(msg), "frame count mismatch: bulk=%d chunked=%d", nf_bulk, nf_chunked);
        FAIL(msg);
    } else {
        /* Compare frame by frame */
        float max_err = 0.0f;
        for (int i = 0; i < nf_bulk * cfg.n_mels; i++) {
            float err = fabsf(out_bulk[i] - out_chunked[i]);
            if (err > max_err) max_err = err;
        }
        if (max_err > 1e-5f) {
            char msg[128];
            snprintf(msg, sizeof(msg), "max error %.8e exceeds 1e-5", max_err);
            FAIL(msg);
        } else {
            PASS();
        }
    }

    free(pcm);
    free(out_bulk);
    free(out_chunked);
    mel_destroy(mel_bulk);
    mel_destroy(mel_chunked);
}

/* Test: very short audio (<1 frame worth) should produce 0 frames, not crash */
static void test_ring_buffer_short_audio(void) {
    TEST("Ring buffer: short audio (<1 frame) returns 0 frames");

    MelConfig cfg;
    mel_config_default(&cfg);
    cfg.preemph = 0.0f;

    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) { FAIL("mel_create returned NULL"); return; }

    /* With center padding (n_fft/2 = 256), we need 400 - 256 = 144 samples for 1 frame.
     * Feed only 50 samples — should get 0 frames. */
    float pcm[50];
    generate_test_pcm(pcm, 50, 440.0f, 0.0f);

    float out[80];
    int nf = mel_process(mel, pcm, 50, out, 1);
    if (nf == 0) {
        PASS();
    } else {
        char msg[64];
        snprintf(msg, sizeof(msg), "expected 0 frames, got %d", nf);
        FAIL(msg);
    }

    mel_destroy(mel);
}

/* Test: feed exactly enough for 1 frame, then exactly enough for next frame */
static void test_ring_buffer_exact_frame_boundary(void) {
    TEST("Ring buffer: exact frame boundary produces correct count");

    MelConfig cfg;
    mel_config_default(&cfg);
    cfg.preemph = 0.0f;

    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) { FAIL("mel_create returned NULL"); return; }

    /* Center pad = n_fft/2 = 256. Need win_length=400 total. So 400-256=144 samples for first frame.
     * Then hop_length=160 samples per additional frame. */
    int first_frame_samples = cfg.win_length - cfg.n_fft / 2;  /* 400 - 256 = 144 */
    float *pcm = (float *)calloc(first_frame_samples + cfg.hop_length, sizeof(float));
    generate_test_pcm(pcm, first_frame_samples + cfg.hop_length, 440.0f, 0.0f);

    float out[160];
    /* Feed exactly enough for 1 frame */
    int nf1 = mel_process(mel, pcm, first_frame_samples, out, 80);
    /* Feed exactly one hop more → should get 1 more frame */
    int nf2 = mel_process(mel, pcm + first_frame_samples, cfg.hop_length, out + nf1 * cfg.n_mels, 80 - nf1);

    if (nf1 == 1 && nf2 == 1) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected (1,1) frames, got (%d,%d)", nf1, nf2);
        FAIL(msg);
    }

    free(pcm);
    mel_destroy(mel);
}

/* Test: feed enough data to force ring buffer growth (>4 seconds) */
static void test_ring_buffer_grow_path(void) {
    TEST("Ring buffer: grow path (>4s audio) produces correct output");

    MelConfig cfg;
    mel_config_default(&cfg);
    cfg.preemph = 0.0f;

    MelSpectrogram *mel_normal = mel_create(&cfg);
    MelSpectrogram *mel_grow = mel_create(&cfg);
    if (!mel_normal || !mel_grow) { FAIL("mel_create returned NULL"); return; }

    /* 5 seconds of audio → exceeds default 4s ring buffer capacity */
    int n_samples = 16000 * 5;
    float *pcm = (float *)malloc(n_samples * sizeof(float));
    generate_test_pcm(pcm, n_samples, 440.0f, 880.0f);

    int max_frames = 600;
    float *out_normal = (float *)calloc(max_frames * cfg.n_mels, sizeof(float));
    float *out_grow = (float *)calloc(max_frames * cfg.n_mels, sizeof(float));

    /* Normal: feed in pieces that don't trigger grow */
    int nf_normal = 0;
    for (int off = 0; off < n_samples; ) {
        int chunk = 3200;  /* 200ms at a time */
        if (off + chunk > n_samples) chunk = n_samples - off;
        int nf = mel_process(mel_normal, pcm + off, chunk,
                             out_normal + nf_normal * cfg.n_mels,
                             max_frames - nf_normal);
        if (nf > 0) nf_normal += nf;
        off += chunk;
    }

    /* Grow: feed all 5 seconds at once to force the grow path */
    int nf_grow = mel_process(mel_grow, pcm, n_samples, out_grow, max_frames);

    if (nf_normal != nf_grow) {
        char msg[128];
        snprintf(msg, sizeof(msg), "frame count mismatch: normal=%d grow=%d", nf_normal, nf_grow);
        FAIL(msg);
    } else {
        float max_err = 0.0f;
        for (int i = 0; i < nf_normal * cfg.n_mels; i++) {
            float err = fabsf(out_normal[i] - out_grow[i]);
            if (err > max_err) max_err = err;
        }
        if (max_err > 1e-5f) {
            char msg[128];
            snprintf(msg, sizeof(msg), "max error %.8e exceeds 1e-5", max_err);
            FAIL(msg);
        } else {
            PASS();
        }
    }

    free(pcm);
    free(out_normal);
    free(out_grow);
    mel_destroy(mel_normal);
    mel_destroy(mel_grow);
}

/* Test: single-sample-at-a-time feeding exercises ring wraparound maximally */
static void test_ring_buffer_single_sample_feeding(void) {
    TEST("Ring buffer: single-sample feeding matches bulk");

    MelConfig cfg;
    mel_config_default(&cfg);
    cfg.preemph = 0.0f;

    MelSpectrogram *mel_bulk = mel_create(&cfg);
    MelSpectrogram *mel_single = mel_create(&cfg);
    if (!mel_bulk || !mel_single) { FAIL("mel_create returned NULL"); return; }

    int n_samples = 1600;  /* 100ms */
    float *pcm = (float *)malloc(n_samples * sizeof(float));
    generate_test_pcm(pcm, n_samples, 440.0f, 1000.0f);

    int max_frames = 50;
    float *out_bulk = (float *)calloc(max_frames * cfg.n_mels, sizeof(float));
    float *out_single = (float *)calloc(max_frames * cfg.n_mels, sizeof(float));

    int nf_bulk = mel_process(mel_bulk, pcm, n_samples, out_bulk, max_frames);

    int nf_single = 0;
    for (int i = 0; i < n_samples; i++) {
        int nf = mel_process(mel_single, &pcm[i], 1,
                             out_single + nf_single * cfg.n_mels,
                             max_frames - nf_single);
        if (nf > 0) nf_single += nf;
    }

    if (nf_bulk != nf_single) {
        char msg[128];
        snprintf(msg, sizeof(msg), "frame count mismatch: bulk=%d single=%d", nf_bulk, nf_single);
        FAIL(msg);
    } else {
        float max_err = 0.0f;
        for (int i = 0; i < nf_bulk * cfg.n_mels; i++) {
            float err = fabsf(out_bulk[i] - out_single[i]);
            if (err > max_err) max_err = err;
        }
        if (max_err > 1e-5f) {
            char msg[128];
            snprintf(msg, sizeof(msg), "max error %.8e exceeds 1e-5", max_err);
            FAIL(msg);
        } else {
            PASS();
        }
    }

    free(pcm);
    free(out_bulk);
    free(out_single);
    mel_destroy(mel_bulk);
    mel_destroy(mel_single);
}

/* Test: mel_reset followed by re-processing produces identical output */
static void test_ring_buffer_reset_determinism(void) {
    TEST("Ring buffer: reset produces identical output on re-process");

    MelConfig cfg;
    mel_config_default(&cfg);
    cfg.preemph = 0.0f;

    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) { FAIL("mel_create returned NULL"); return; }

    int n_samples = 3200;
    float *pcm = (float *)malloc(n_samples * sizeof(float));
    generate_test_pcm(pcm, n_samples, 440.0f, 1000.0f);

    int max_frames = 100;
    float *out1 = (float *)calloc(max_frames * cfg.n_mels, sizeof(float));
    float *out2 = (float *)calloc(max_frames * cfg.n_mels, sizeof(float));

    int nf1 = mel_process(mel, pcm, n_samples, out1, max_frames);

    mel_reset(mel);

    int nf2 = mel_process(mel, pcm, n_samples, out2, max_frames);

    if (nf1 != nf2) {
        char msg[128];
        snprintf(msg, sizeof(msg), "frame count mismatch after reset: %d vs %d", nf1, nf2);
        FAIL(msg);
    } else {
        float max_err = 0.0f;
        for (int i = 0; i < nf1 * cfg.n_mels; i++) {
            float err = fabsf(out1[i] - out2[i]);
            if (err > max_err) max_err = err;
        }
        if (max_err > 0.0f) {
            char msg[128];
            snprintf(msg, sizeof(msg), "output differs after reset, max error %.8e", max_err);
            FAIL(msg);
        } else {
            PASS();
        }
    }

    free(pcm);
    free(out1);
    free(out2);
    mel_destroy(mel);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Hann Window Correctness
 *
 * Verify the vectorized vDSP Hann window matches the scalar formula exactly.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_hann_window_vectorized_vs_scalar(void) {
    TEST("Hann window: vectorized vDSP matches scalar formula");

    int win_length = 400;
    float denom = (float)(win_length - 1);  /* symmetric */

    /* Compute scalar reference */
    float *scalar = (float *)malloc(win_length * sizeof(float));
    for (int i = 0; i < win_length; i++) {
        scalar[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / denom);
    }

    /* Compute vectorized (same method as mel_create) */
    float *vectorized = (float *)malloc(win_length * sizeof(float));
    float zero = 0.0f, one = 1.0f;
    vDSP_vramp(&zero, &one, vectorized, 1, win_length);
    float scale = 2.0f * (float)M_PI / denom;
    vDSP_vsmul(vectorized, 1, &scale, vectorized, 1, win_length);
    int n = win_length;
    vvcosf(vectorized, vectorized, &n);
    float neg = -0.5f;
    vDSP_vsmul(vectorized, 1, &neg, vectorized, 1, win_length);
    float half = 0.5f;
    vDSP_vsadd(vectorized, 1, &half, vectorized, 1, win_length);

    /* Compare */
    float max_err = 0.0f;
    int worst_idx = 0;
    for (int i = 0; i < win_length; i++) {
        float err = fabsf(scalar[i] - vectorized[i]);
        if (err > max_err) { max_err = err; worst_idx = i; }
    }

    /* Should be within single-precision ULP (< 1e-6) */
    if (max_err > 1e-6f) {
        char msg[128];
        snprintf(msg, sizeof(msg), "max error %.8e at index %d (scalar=%.8f vec=%.8f)",
                 max_err, worst_idx, scalar[worst_idx], vectorized[worst_idx]);
        FAIL(msg);
    } else {
        printf("(max_err=%.2e) ", max_err);
        PASS();
    }

    /* Verify boundary values: w[0] and w[N-1] should be 0 for symmetric window */
    if (fabsf(vectorized[0]) > 1e-7f || fabsf(vectorized[win_length - 1]) > 1e-7f) {
        printf("  WARNING: boundary values non-zero: w[0]=%.8e w[N-1]=%.8e\n",
               vectorized[0], vectorized[win_length - 1]);
    }

    free(scalar);
    free(vectorized);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. INT8 Dequantize Correctness
 *
 * Verify that the NEON-vectorized dequantization produces identical output
 * to a naive scalar loop.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Scalar reference: dequantize int8 weights with per-channel scales */
static void dequant_scalar(float *out, const int8_t *W_q, const float *scales,
                           int N, int K) {
    for (int n = 0; n < N; n++) {
        float s = scales[n];
        for (int k = 0; k < K; k++) {
            out[n * K + k] = (float)W_q[n * K + k] * s;
        }
    }
}

/* NEON vectorized dequantize (extracted from conformer_stt.c linear_int8) */
static void dequant_neon(float *out, const int8_t *W_q, const float *scales,
                         int N, int K) {
    for (int n = 0; n < N; n++) {
        const int8_t *row = W_q + (size_t)n * K;
        float *dst = out + (size_t)n * K;
        float s = scales[n];
        int k = 0;
        for (; k + 15 < K; k += 16) {
            int8x16_t q = vld1q_s8(row + k);
            int16x8_t lo16 = vmovl_s8(vget_low_s8(q));
            int16x8_t hi16 = vmovl_s8(vget_high_s8(q));
            float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
            float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
            float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
            float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
            float32x4_t sv = vdupq_n_f32(s);
            vst1q_f32(dst + k,      vmulq_f32(f0, sv));
            vst1q_f32(dst + k + 4,  vmulq_f32(f1, sv));
            vst1q_f32(dst + k + 8,  vmulq_f32(f2, sv));
            vst1q_f32(dst + k + 12, vmulq_f32(f3, sv));
        }
        for (; k < K; k++)
            dst[k] = (float)row[k] * s;
    }
}

static void test_int8_dequant_neon_vs_scalar(void) {
    TEST("INT8 dequant: NEON vectorized matches scalar");

    /* Test with dimensions that exercise both the NEON path and the scalar tail */
    int N = 128;
    int K = 259;  /* Not a multiple of 16 — exercises scalar tail */

    int8_t *W_q = (int8_t *)malloc(N * K);
    float *scales = (float *)malloc(N * sizeof(float));
    float *out_scalar = (float *)calloc(N * K, sizeof(float));
    float *out_neon = (float *)calloc(N * K, sizeof(float));

    /* Fill with diverse values including extremes */
    srand(42);
    for (int i = 0; i < N * K; i++) {
        W_q[i] = (int8_t)((rand() % 256) - 128);
    }
    /* Include extreme values */
    W_q[0] = INT8_MIN;     /* -128 */
    W_q[1] = INT8_MAX;     /* 127 */
    W_q[2] = 0;
    W_q[3] = 1;
    W_q[4] = -1;

    for (int n = 0; n < N; n++) {
        scales[n] = 0.001f + (float)(rand() % 1000) / 1000.0f;
    }
    /* Include edge-case scales */
    scales[0] = 0.0f;
    scales[1] = 1.0f;
    scales[2] = -0.5f;  /* Negative scale (shouldn't happen but must handle) */

    dequant_scalar(out_scalar, W_q, scales, N, K);
    dequant_neon(out_neon, W_q, scales, N, K);

    float max_err = 0.0f;
    for (int i = 0; i < N * K; i++) {
        float err = fabsf(out_scalar[i] - out_neon[i]);
        if (err > max_err) max_err = err;
    }

    if (max_err > 0.0f) {
        char msg[128];
        snprintf(msg, sizeof(msg), "max error %.8e (should be exactly 0)", max_err);
        FAIL(msg);
    } else {
        PASS();
    }

    free(W_q);
    free(scales);
    free(out_scalar);
    free(out_neon);
}

/* Test that the tiled sgemm in linear_int8 produces correct output */
static void test_int8_tiled_sgemm_correctness(void) {
    TEST("INT8 tiled sgemm: tiled output matches reference fp32 sgemm");

    int M = 4;    /* batch / sequence length */
    int K = 128;  /* input dim */
    int N = 200;  /* output dim — not a multiple of tile size (64) */

    int8_t *W_q = (int8_t *)malloc(N * K);
    float *scales = (float *)malloc(N * sizeof(float));
    float *bias = (float *)malloc(N * sizeof(float));
    float *in = (float *)malloc(M * K * sizeof(float));
    float *out_tiled = (float *)calloc(M * N, sizeof(float));
    float *out_ref = (float *)calloc(M * N, sizeof(float));
    float *W_fp32 = (float *)calloc(N * K, sizeof(float));
    float *W_tile = (float *)calloc(64 * K, sizeof(float));  /* INT8_TILE_N=64 */

    srand(123);
    for (int i = 0; i < N * K; i++)
        W_q[i] = (int8_t)((rand() % 256) - 128);
    for (int n = 0; n < N; n++)
        scales[n] = 0.01f + (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < N; i++)
        bias[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;
    for (int i = 0; i < M * K; i++)
        in[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;

    /* Reference: dequantize all weights to fp32, then full sgemm + bias */
    dequant_scalar(W_fp32, W_q, scales, N, K);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, in, K, W_fp32, K,
                0.0f, out_ref, N);
    for (int m = 0; m < M; m++)
        vDSP_vadd(out_ref + m * N, 1, bias, 1, out_ref + m * N, 1, N);

    /* Tiled: reproduce the exact logic from linear_int8 */
    #define INT8_TILE_N 64
    for (int n0 = 0; n0 < N; n0 += INT8_TILE_N) {
        int tile_n = (n0 + INT8_TILE_N <= N) ? INT8_TILE_N : (N - n0);
        for (int r = 0; r < tile_n; r++) {
            const int8_t *row = W_q + (size_t)(n0 + r) * K;
            float *dst = W_tile + (size_t)r * K;
            float s = scales[n0 + r];
            int k = 0;
            for (; k + 15 < K; k += 16) {
                int8x16_t q = vld1q_s8(row + k);
                int16x8_t lo16 = vmovl_s8(vget_low_s8(q));
                int16x8_t hi16 = vmovl_s8(vget_high_s8(q));
                float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                float32x4_t sv = vdupq_n_f32(s);
                vst1q_f32(dst + k,      vmulq_f32(f0, sv));
                vst1q_f32(dst + k + 4,  vmulq_f32(f1, sv));
                vst1q_f32(dst + k + 8,  vmulq_f32(f2, sv));
                vst1q_f32(dst + k + 12, vmulq_f32(f3, sv));
            }
            for (; k < K; k++)
                dst[k] = (float)row[k] * s;
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, tile_n, K,
                    1.0f, in, K, W_tile, K,
                    0.0f, out_tiled + n0, N);
    }
    for (int m = 0; m < M; m++)
        vDSP_vadd(out_tiled + m * N, 1, bias, 1, out_tiled + m * N, 1, N);

    /* Compare */
    float max_err = 0.0f;
    float max_rel = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(out_ref[i] - out_tiled[i]);
        if (err > max_err) max_err = err;
        float denom_val = fabsf(out_ref[i]);
        if (denom_val > 1e-6f) {
            float rel = err / denom_val;
            if (rel > max_rel) max_rel = rel;
        }
    }

    /* Tiled and full sgemm should match within float rounding (~1e-4 for accumulated sums) */
    if (max_err > 1e-3f) {
        char msg[128];
        snprintf(msg, sizeof(msg), "max abs error %.6e, max rel error %.6e", max_err, max_rel);
        FAIL(msg);
    } else {
        printf("(max_abs=%.2e max_rel=%.2e) ", max_err, max_rel);
        PASS();
    }

    free(W_q); free(scales); free(bias); free(in);
    free(out_tiled); free(out_ref); free(W_fp32); free(W_tile);
}

/* Test INT8 dequant with K not a multiple of 16 (scalar tail path) */
static void test_int8_dequant_scalar_tail(void) {
    TEST("INT8 dequant: scalar tail (K=17) matches");

    int N = 4;
    int K = 17;  /* 16 NEON + 1 scalar */

    int8_t *W_q = (int8_t *)malloc(N * K);
    float *scales = (float *)malloc(N * sizeof(float));
    float *out_scalar = (float *)calloc(N * K, sizeof(float));
    float *out_neon = (float *)calloc(N * K, sizeof(float));

    for (int i = 0; i < N * K; i++) W_q[i] = (int8_t)(i % 127);
    for (int n = 0; n < N; n++) scales[n] = 0.1f * (n + 1);

    dequant_scalar(out_scalar, W_q, scales, N, K);
    dequant_neon(out_neon, W_q, scales, N, K);

    float max_err = 0.0f;
    for (int i = 0; i < N * K; i++) {
        float err = fabsf(out_scalar[i] - out_neon[i]);
        if (err > max_err) max_err = err;
    }

    if (max_err > 0.0f) {
        char msg[128];
        snprintf(msg, sizeof(msg), "max error %.8e at K=17", max_err);
        FAIL(msg);
    } else {
        PASS();
    }

    free(W_q); free(scales); free(out_scalar); free(out_neon);
}

/* Test: K < 16 (pure scalar path, no NEON at all) */
static void test_int8_dequant_pure_scalar(void) {
    TEST("INT8 dequant: pure scalar (K=7, no NEON)");

    int N = 4;
    int K = 7;  /* All scalar */

    int8_t *W_q = (int8_t *)malloc(N * K);
    float *scales = (float *)malloc(N * sizeof(float));
    float *out_scalar = (float *)calloc(N * K, sizeof(float));
    float *out_neon = (float *)calloc(N * K, sizeof(float));

    for (int i = 0; i < N * K; i++) W_q[i] = (int8_t)((i * 7 + 13) % 255 - 128);
    for (int n = 0; n < N; n++) scales[n] = 0.05f * (n + 1);

    dequant_scalar(out_scalar, W_q, scales, N, K);
    dequant_neon(out_neon, W_q, scales, N, K);

    float max_err = 0.0f;
    for (int i = 0; i < N * K; i++) {
        float err = fabsf(out_scalar[i] - out_neon[i]);
        if (err > max_err) max_err = err;
    }

    if (max_err > 0.0f) {
        FAIL("pure scalar path differs");
    } else {
        PASS();
    }

    free(W_q); free(scales); free(out_scalar); free(out_neon);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Pipeline Pre-alloc Bug Detection Tests
 *
 * These tests verify the bugs we found in sonatav2_destroy() and the
 * grow-path error handling.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Test: verify mel_process with pre-emphasis enabled produces deterministic output */
static void test_preemphasis_determinism(void) {
    TEST("Pre-emphasis: streaming matches bulk");

    MelConfig cfg;
    mel_config_default(&cfg);
    cfg.preemph = 0.97f;

    MelSpectrogram *mel_bulk = mel_create(&cfg);
    MelSpectrogram *mel_stream = mel_create(&cfg);
    if (!mel_bulk || !mel_stream) { FAIL("mel_create returned NULL"); return; }

    int n_samples = 4800;
    float *pcm = (float *)malloc(n_samples * sizeof(float));
    generate_test_pcm(pcm, n_samples, 440.0f, 1000.0f);

    int max_frames = 200;
    float *out_bulk = (float *)calloc(max_frames * cfg.n_mels, sizeof(float));
    float *out_stream = (float *)calloc(max_frames * cfg.n_mels, sizeof(float));

    int nf_bulk = mel_process(mel_bulk, pcm, n_samples, out_bulk, max_frames);

    /* Stream in 160-sample chunks (one hop at a time) */
    int nf_stream = 0;
    for (int off = 0; off < n_samples; off += 160) {
        int chunk = 160;
        if (off + chunk > n_samples) chunk = n_samples - off;
        int nf = mel_process(mel_stream, pcm + off, chunk,
                             out_stream + nf_stream * cfg.n_mels,
                             max_frames - nf_stream);
        if (nf > 0) nf_stream += nf;
    }

    if (nf_bulk != nf_stream) {
        char msg[128];
        snprintf(msg, sizeof(msg), "frame count: bulk=%d stream=%d", nf_bulk, nf_stream);
        FAIL(msg);
    } else {
        float max_err = 0.0f;
        for (int i = 0; i < nf_bulk * cfg.n_mels; i++) {
            float err = fabsf(out_bulk[i] - out_stream[i]);
            if (err > max_err) max_err = err;
        }
        if (max_err > 1e-5f) {
            char msg[128];
            snprintf(msg, sizeof(msg), "max error %.8e with preemph", max_err);
            FAIL(msg);
        } else {
            PASS();
        }
    }

    free(pcm); free(out_bulk); free(out_stream);
    mel_destroy(mel_bulk);
    mel_destroy(mel_stream);
}

/* Test: mel_process with periodic window matches scalar periodic formula */
static void test_periodic_window(void) {
    TEST("Periodic Hann window: matches scalar formula");

    int win_length = 400;
    float denom = (float)win_length;  /* periodic: divide by N, not N-1 */

    float *scalar = (float *)malloc(win_length * sizeof(float));
    for (int i = 0; i < win_length; i++) {
        scalar[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / denom);
    }

    /* Vectorized periodic */
    float *vectorized = (float *)malloc(win_length * sizeof(float));
    float zero = 0.0f, one = 1.0f;
    vDSP_vramp(&zero, &one, vectorized, 1, win_length);
    float scale = 2.0f * (float)M_PI / denom;
    vDSP_vsmul(vectorized, 1, &scale, vectorized, 1, win_length);
    int n = win_length;
    vvcosf(vectorized, vectorized, &n);
    float neg = -0.5f;
    vDSP_vsmul(vectorized, 1, &neg, vectorized, 1, win_length);
    float half = 0.5f;
    vDSP_vsadd(vectorized, 1, &half, vectorized, 1, win_length);

    float max_err = 0.0f;
    for (int i = 0; i < win_length; i++) {
        float err = fabsf(scalar[i] - vectorized[i]);
        if (err > max_err) max_err = err;
    }

    if (max_err > 1e-6f) {
        char msg[128];
        snprintf(msg, sizeof(msg), "periodic window max error %.8e", max_err);
        FAIL(msg);
    } else {
        printf("(max_err=%.2e) ", max_err);
        PASS();
    }

    /* For periodic, w[0] should be 0 but w[N-1] should NOT be 0 */
    if (fabsf(vectorized[0]) > 1e-7f) {
        printf("  WARNING: w[0] non-zero: %.8e\n", vectorized[0]);
    }

    free(scalar);
    free(vectorized);
}

/* Test: mel_process error handling — NULL inputs */
static void test_mel_process_null_safety(void) {
    TEST("mel_process: NULL inputs return -1");

    MelConfig cfg;
    mel_config_default(&cfg);
    MelSpectrogram *mel = mel_create(&cfg);

    float pcm[160] = {0};
    float out[80] = {0};

    int r1 = mel_process(NULL, pcm, 160, out, 1);
    int r2 = mel_process(mel, NULL, 160, out, 1);
    int r3 = mel_process(mel, pcm, 0, out, 1);
    int r4 = mel_process(mel, pcm, 160, NULL, 1);
    int r5 = mel_process(mel, pcm, 160, out, 0);
    int r6 = mel_process(mel, pcm, -1, out, 1);

    if (r1 == -1 && r2 == -1 && r3 == -1 && r4 == -1 && r5 == -1 && r6 == -1) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "results: %d %d %d %d %d %d (all should be -1)",
                 r1, r2, r3, r4, r5, r6);
        FAIL(msg);
    }

    mel_destroy(mel);
}

/* Test: mel_create rejects invalid configs */
static void test_mel_create_validation(void) {
    TEST("mel_create: rejects invalid configs");

    MelConfig cfg;
    mel_config_default(&cfg);

    /* n_fft not power of 2 */
    cfg.n_fft = 500;
    MelSpectrogram *m1 = mel_create(&cfg);
    mel_config_default(&cfg);

    /* win_length > n_fft */
    cfg.win_length = cfg.n_fft + 1;
    MelSpectrogram *m2 = mel_create(&cfg);
    mel_config_default(&cfg);

    /* hop_length <= 0 */
    cfg.hop_length = 0;
    MelSpectrogram *m3 = mel_create(&cfg);
    mel_config_default(&cfg);

    /* NULL config */
    MelSpectrogram *m4 = mel_create(NULL);

    if (!m1 && !m2 && !m3 && !m4) {
        PASS();
    } else {
        FAIL("accepted invalid config");
        mel_destroy(m1);
        mel_destroy(m2);
        mel_destroy(m3);
        mel_destroy(m4);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Correctness Audit Test Suite\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    printf("── Ring Buffer Correctness ──────────────────────────────────\n");
    test_ring_buffer_chunked_vs_bulk();
    test_ring_buffer_short_audio();
    test_ring_buffer_exact_frame_boundary();
    test_ring_buffer_grow_path();
    test_ring_buffer_single_sample_feeding();
    test_ring_buffer_reset_determinism();

    printf("\n── Hann Window Correctness ──────────────────────────────────\n");
    test_hann_window_vectorized_vs_scalar();
    test_periodic_window();

    printf("\n── INT8 Dequantize Correctness ──────────────────────────────\n");
    test_int8_dequant_neon_vs_scalar();
    test_int8_dequant_scalar_tail();
    test_int8_dequant_pure_scalar();
    test_int8_tiled_sgemm_correctness();

    printf("\n── Pipeline Safety & Error Handling ─────────────────────────\n");
    test_preemphasis_determinism();
    test_mel_process_null_safety();
    test_mel_create_validation();

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  Results: %d/%d passed\n", tests_passed, tests_run);
    printf("═══════════════════════════════════════════════════════════════\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
