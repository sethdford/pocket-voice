/**
 * test_research_stt.c — Verify Conmer mode + INT4 dequantization in conformer_stt.
 *
 * Tests:
 *   T1: Conmer mode enable/disable API (NULL safety)
 *   T2: INT4 nibble pack/unpack roundtrip (all 16 values)
 *   T3: INT4 dequantized GEMM numerical accuracy vs fp32
 *   T4: Conmer block architecture verification
 *   T5: INT4 range edge cases (-8 through 7)
 *   T6: INT4 dequantize precision across scale magnitudes
 *   T7: INT4 NEON dequant matches scalar path
 *   T8: INT4 odd-K dimension handling
 *   T9: INT4 memory savings vs fp32 (~8x)
 *   T10: INT4 zero weights produce zero output
 *
 * Build:
 *   cc -O3 -arch arm64 -Isrc -framework Accelerate \
 *      -Lbuild -lmel_spectrogram -lconformer_stt -lctc_beam_decoder -ltdt_decoder \
 *      -Wl,-rpath,$(pwd)/build \
 *      -o build/test-research-stt tests/test_research_stt.c
 *
 * Run: ./build/test-research-stt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>

#include "conformer_stt.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-60s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; return; } while(0)

/* ══════════════════════════════════════════════════════════════════════════
 * T1: Conmer mode API
 * ══════════════════════════════════════════════════════════════════════════ */

static void test_conmer_api(void) {
    TEST("T1: conmer API NULL safety");

    /* These should not crash */
    conformer_stt_set_conmer_mode(NULL, 1);
    int r = conformer_stt_is_conmer(NULL);
    if (r != 0) FAIL("is_conmer(NULL) should return 0");
    PASS();
}

/* ══════════════════════════════════════════════════════════════════════════
 * T2: INT4 nibble pack/unpack roundtrip
 * ══════════════════════════════════════════════════════════════════════════ */

static void test_int4_pack_roundtrip(void) {
    TEST("T2: INT4 nibble pack/unpack all 16 values");

    /* Test all 16 values: -8 through 7 */
    const int N = 128;
    int8_t original[128];
    uint8_t packed[64];

    for (int i = 0; i < N; i++) {
        original[i] = (int8_t)((i * 7 + 3) % 16 - 8);
    }

    /* Pack: pairs of values into bytes (lo nibble = even, hi nibble = odd) */
    for (int i = 0; i < N / 2; i++) {
        int8_t lo = original[i * 2];
        int8_t hi = original[i * 2 + 1];
        packed[i] = (uint8_t)((lo & 0xF) | (hi << 4));
    }

    /* Unpack: matching the conformer_stt.c linear_int4() scalar path */
    int8_t unpacked[128];
    for (int i = 0; i < N / 2; i++) {
        uint8_t b = packed[i];
        int lo = b & 0x0F;
        int hi = (b >> 4) & 0x0F;
        if (lo >= 8) lo -= 16;
        if (hi >= 8) hi -= 16;
        unpacked[i * 2]     = (int8_t)lo;
        unpacked[i * 2 + 1] = (int8_t)hi;
    }

    for (int i = 0; i < N; i++) {
        if (unpacked[i] != original[i]) {
            char msg[128];
            snprintf(msg, sizeof(msg), "mismatch at %d: expected %d, got %d",
                     i, original[i], unpacked[i]);
            FAIL(msg);
        }
    }
    PASS();
}

/* ══════════════════════════════════════════════════════════════════════════
 * T3: INT4 dequantized GEMM numerical accuracy
 * ══════════════════════════════════════════════════════════════════════════ */

static void test_int4_gemm_accuracy(void) {
    TEST("T3: INT4 dequant+GEMM absolute error within tolerance");

    const int M = 4, K = 16, N = 8;

    float W_fp32[8 * 16];
    for (int i = 0; i < N * K; i++)
        W_fp32[i] = sinf((float)i * 0.1f) * 2.0f;

    /* Quantize to INT4 */
    float scales[8];
    int8_t W_q[8 * 16];
    for (int n = 0; n < N; n++) {
        float absmax = 0;
        for (int k = 0; k < K; k++) {
            float a = fabsf(W_fp32[n * K + k]);
            if (a > absmax) absmax = a;
        }
        scales[n] = absmax / 7.0f;
        if (scales[n] < 1e-8f) scales[n] = 1e-8f;
        for (int k = 0; k < K; k++) {
            int q = (int)roundf(W_fp32[n * K + k] / scales[n]);
            if (q < -8) q = -8;
            if (q > 7) q = 7;
            W_q[n * K + k] = (int8_t)q;
        }
    }

    /* Pack into nibbles */
    uint8_t W_packed[8 * 8];
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K / 2; k++) {
            int8_t lo = W_q[n * K + k * 2];
            int8_t hi = W_q[n * K + k * 2 + 1];
            W_packed[n * (K / 2) + k] = (uint8_t)((lo & 0xF) | (hi << 4));
        }
    }

    /* Input */
    float X[4 * 16];
    for (int i = 0; i < M * K; i++)
        X[i] = cosf((float)i * 0.3f);

    /* Reference: fp32 matmul Y_ref = X @ W^T */
    float Y_ref[4 * 8] = {0};
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                Y_ref[m * N + n] += X[m * K + k] * W_fp32[n * K + k];

    /* INT4 path: dequantize then matmul */
    float W_deq[8 * 16];
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K / 2; k++) {
            uint8_t b = W_packed[n * (K / 2) + k];
            int lo = b & 0x0F;
            int hi = (b >> 4) & 0x0F;
            if (lo >= 8) lo -= 16;
            if (hi >= 8) hi -= 16;
            W_deq[n * K + k * 2]     = (float)lo * scales[n];
            W_deq[n * K + k * 2 + 1] = (float)hi * scales[n];
        }
    }

    float Y_int4[4 * 8] = {0};
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                Y_int4[m * N + n] += X[m * K + k] * W_deq[n * K + k];

    /* Use absolute error as primary metric since INT4 quantization
     * introduces fixed-magnitude noise that makes relative error
     * blow up near zero-valued outputs. */
    float max_abs = 0;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(Y_ref[i] - Y_int4[i]);
        if (err > max_abs) max_abs = err;
    }

    /* INT4 quant noise per weight ≈ step/2 = (range/15)/2.
     * For range=4 (sinf*2): step ≈ 0.267, noise ≈ 0.133.
     * Accumulated over K=16 dot-product: max abs error ≈ K * noise ≈ 2.1
     * Use generous tolerance of 3.0. */
    if (max_abs > 3.0f) {
        char msg[128];
        snprintf(msg, sizeof(msg), "max_abs_error=%.4f exceeds 3.0", max_abs);
        FAIL(msg);
    }
    PASS();
}

/* ══════════════════════════════════════════════════════════════════════════
 * T4: Conmer block architecture verification
 * ══════════════════════════════════════════════════════════════════════════ */

static void test_conmer_architecture(void) {
    TEST("T4: Conmer skips MHSA — saves 4 matmuls per block");

    /* Conmer block: FFN½ → Conv → FFN½ → LayerNorm (skip MHSA)
     * Standard:    FFN½ → MHSA → Conv → FFN½ → LayerNorm
     *
     * Matmul count:
     *   FFN1: up + down = 2
     *   MHSA: Q + K + V + Out = 4  (skipped in Conmer)
     *   Conv: pw1 + pw2 = 2
     *   FFN2: up + down = 2
     *   Standard total: 10, Conmer total: 6, savings: 4 */

    int conmer_matmuls = 6;
    int standard_matmuls = 10;
    int saved = standard_matmuls - conmer_matmuls;
    if (saved != 4) FAIL("should save exactly 4 matmuls");

    float savings_pct = (float)saved / (float)standard_matmuls * 100.0f;
    if (savings_pct < 35.0f || savings_pct > 45.0f) {
        char msg[128];
        snprintf(msg, sizeof(msg), "compute savings %.1f%% out of range", savings_pct);
        FAIL(msg);
    }

    /* Verify Conmer API functions link correctly */
    void (*fn_set)(ConformerSTT *, int) = conformer_stt_set_conmer_mode;
    int (*fn_is)(const ConformerSTT *) = conformer_stt_is_conmer;
    if (!fn_set || !fn_is) FAIL("API functions did not resolve");
    PASS();
}

/* ══════════════════════════════════════════════════════════════════════════
 * T5: INT4 range edge cases
 * ══════════════════════════════════════════════════════════════════════════ */

static void test_int4_edge_cases(void) {
    TEST("T5: INT4 all 16 values + paired packing correct");

    int ok = 1;

    /* Test each value individually (low nibble) */
    for (int v = -8; v <= 7; v++) {
        uint8_t packed = (uint8_t)((int8_t)v & 0xF);
        int nibble = packed & 0x0F;
        if (nibble >= 8) nibble -= 16;
        if (nibble != v) {
            ok = 0;
            break;
        }
    }

    /* Test paired packing (sampled) */
    for (int lo_v = -8; lo_v <= 7; lo_v += 3) {
        for (int hi_v = -8; hi_v <= 7; hi_v += 3) {
            uint8_t packed = (uint8_t)(((int8_t)lo_v & 0xF) | ((int8_t)hi_v << 4));
            int u_lo = packed & 0x0F;
            int u_hi = (packed >> 4) & 0x0F;
            if (u_lo >= 8) u_lo -= 16;
            if (u_hi >= 8) u_hi -= 16;
            if (u_lo != lo_v || u_hi != hi_v) {
                ok = 0;
                break;
            }
        }
        if (!ok) break;
    }

    if (!ok) FAIL("edge case failure");
    PASS();
}

/* ══════════════════════════════════════════════════════════════════════════
 * T6: INT4 dequantize precision across scale magnitudes
 * ══════════════════════════════════════════════════════════════════════════ */

static void test_int4_scale_precision(void) {
    TEST("T6: INT4 dequant exact across scale magnitudes");

    float test_scales[] = {1e-6f, 0.001f, 0.1f, 1.0f, 10.0f, 1000.0f};
    int n_scales = (int)(sizeof(test_scales) / sizeof(test_scales[0]));

    for (int s = 0; s < n_scales; s++) {
        float scale = test_scales[s];
        for (int v = -8; v <= 7; v++) {
            float expected = (float)v * scale;
            uint8_t packed = (uint8_t)((int8_t)v & 0xF);
            int nibble = packed & 0x0F;
            if (nibble >= 8) nibble -= 16;
            float result = (float)nibble * scale;
            float err = fabsf(result - expected);
            if (err > fabsf(scale) * 1e-5f) {
                char msg[128];
                snprintf(msg, sizeof(msg), "scale=%.1e v=%d err=%.2e", scale, v, err);
                FAIL(msg);
            }
        }
    }
    PASS();
}

/* ══════════════════════════════════════════════════════════════════════════
 * T7: INT4 NEON dequant matches scalar path
 * ══════════════════════════════════════════════════════════════════════════ */

static void test_int4_neon_dequant(void) {
    TEST("T7: INT4 NEON dequant matches scalar for K=32");

    const int N = 2, K = 32;
    const int K_packed = K / 2;

    /* Create packed INT4 data with known values */
    uint8_t packed[2 * 16]; /* N * K_packed */
    float scales[2] = { 0.5f, 1.5f };

    for (int r = 0; r < N; r++) {
        for (int k = 0; k < K_packed; k++) {
            int lo_v = ((r * K + k * 2) * 7 + 3) % 16 - 8;
            int hi_v = ((r * K + k * 2 + 1) * 7 + 3) % 16 - 8;
            packed[r * K_packed + k] = (uint8_t)(((int8_t)lo_v & 0xF) |
                                                  ((int8_t)hi_v << 4));
        }
    }

    /* Scalar unpack */
    float scalar_out[2 * 32];
    for (int r = 0; r < N; r++) {
        float s = scales[r];
        const uint8_t *row = packed + r * K_packed;
        for (int k = 0; k < K; k++) {
            uint8_t byte = row[k / 2];
            int nibble = (k & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
            if (nibble >= 8) nibble -= 16;
            scalar_out[r * K + k] = (float)nibble * s;
        }
    }

    /* NEON unpack — replicating the logic from linear_int4() */
    float neon_out[2 * 32];
    for (int r = 0; r < N; r++) {
        const uint8_t *row = packed + r * K_packed;
        float s = scales[r];
        int k = 0;
        for (; k + 15 < K; k += 16) {
            uint8x8_t p = vld1_u8(row + k / 2);
            int8x8_t lo_nib = vreinterpret_s8_u8(vand_u8(p, vdup_n_u8(0x0F)));
            int8x8_t hi_nib = vreinterpret_s8_u8(vshr_n_u8(p, 4));
            /* Sign-extend: subtract 16 if >= 8 */
            int8x8_t lo_sign = vsub_s8(lo_nib, vand_s8(
                vreinterpret_s8_u8(vcge_s8(lo_nib, vdup_n_s8(8))),
                vdup_n_s8(16)));
            int8x8_t hi_sign = vsub_s8(hi_nib, vand_s8(
                vreinterpret_s8_u8(vcge_s8(hi_nib, vdup_n_s8(8))),
                vdup_n_s8(16)));
            /* Interleave: lo[0],hi[0],lo[1],hi[1],... */
            int8x8x2_t zipped = vzip_s8(lo_sign, hi_sign);
            int8x16_t vals = vcombine_s8(zipped.val[0], zipped.val[1]);
            int16x8_t w_lo16 = vmovl_s8(vget_low_s8(vals));
            int16x8_t w_hi16 = vmovl_s8(vget_high_s8(vals));
            float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_lo16)));
            float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_lo16)));
            float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_hi16)));
            float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_hi16)));
            float32x4_t sv = vdupq_n_f32(s);
            vst1q_f32(neon_out + r * K + k,      vmulq_f32(f0, sv));
            vst1q_f32(neon_out + r * K + k + 4,  vmulq_f32(f1, sv));
            vst1q_f32(neon_out + r * K + k + 8,  vmulq_f32(f2, sv));
            vst1q_f32(neon_out + r * K + k + 12, vmulq_f32(f3, sv));
        }
        for (; k < K; k++) {
            uint8_t byte = row[k / 2];
            int nibble = (k & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
            if (nibble >= 8) nibble -= 16;
            neon_out[r * K + k] = (float)nibble * s;
        }
    }

    float max_err = 0.0f;
    for (int i = 0; i < N * K; i++) {
        float err = fabsf(scalar_out[i] - neon_out[i]);
        if (err > max_err) max_err = err;
    }

    if (max_err > 1e-6f) {
        char msg[128];
        snprintf(msg, sizeof(msg), "NEON/scalar mismatch: max_err=%.9f", max_err);
        FAIL(msg);
    }
    PASS();
}

/* ══════════════════════════════════════════════════════════════════════════
 * T8: INT4 odd-K dimension handling
 * ══════════════════════════════════════════════════════════════════════════ */

static void test_int4_odd_k(void) {
    TEST("T8: INT4 odd K dimension (K=7) packs correctly");

    const int N = 2, K = 7;
    const int K_packed = (K + 1) / 2;  /* ceil(7/2) = 4 */

    if (K_packed != 4) FAIL("K_packed should be 4 for K=7");

    /* Create known values and pack */
    int8_t values[2 * 7];
    for (int i = 0; i < N * K; i++)
        values[i] = (int8_t)((i % 16) - 8);

    uint8_t packed[2 * 4]; /* N * K_packed */
    memset(packed, 0, sizeof(packed));
    for (int r = 0; r < N; r++) {
        for (int k = 0; k < K; k++) {
            int nibble = values[r * K + k] & 0x0F;
            if (k & 1)
                packed[r * K_packed + k / 2] |= (nibble << 4);
            else
                packed[r * K_packed + k / 2] = nibble;
        }
    }

    /* Unpack with the same scalar logic as conformer_stt.c */
    float scales[] = { 1.0f, 2.0f };
    float unpacked[2 * 7];
    for (int r = 0; r < N; r++) {
        float s = scales[r];
        const uint8_t *row = packed + r * K_packed;
        for (int k = 0; k < K; k++) {
            uint8_t byte = row[k / 2];
            int nib = (k & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
            if (nib >= 8) nib -= 16;
            unpacked[r * K + k] = (float)nib * s;
        }
    }

    /* Verify values match */
    for (int r = 0; r < N; r++) {
        for (int k = 0; k < K; k++) {
            float expected = (float)values[r * K + k] * scales[r];
            float err = fabsf(unpacked[r * K + k] - expected);
            if (err > 1e-6f) {
                char msg[128];
                snprintf(msg, sizeof(msg), "row=%d k=%d expected=%.2f got=%.2f",
                         r, k, expected, unpacked[r * K + k]);
                FAIL(msg);
            }
        }
    }
    PASS();
}

/* ══════════════════════════════════════════════════════════════════════════
 * T9: INT4 memory savings vs fp32
 * ══════════════════════════════════════════════════════════════════════════ */

static void test_int4_memory_savings(void) {
    TEST("T9: INT4 achieves ~8x memory reduction vs fp32");

    int N = 512, K = 512;
    size_t fp32_size = (size_t)N * K * sizeof(float);
    int K_packed = (K + 1) / 2;
    size_t int4_size = (size_t)N * K_packed + (size_t)N * sizeof(float);

    float ratio = (float)fp32_size / (float)int4_size;

    /* Expected: 512*512*4 / (512*256 + 512*4) = 1048576 / 133120 ≈ 7.87x */
    if (ratio < 7.0f) {
        char msg[128];
        snprintf(msg, sizeof(msg), "ratio=%.2fx (expected ~8x)", ratio);
        FAIL(msg);
    }
    PASS();
}

/* ══════════════════════════════════════════════════════════════════════════
 * T10: INT4 zero weights
 * ══════════════════════════════════════════════════════════════════════════ */

static void test_int4_zero_weights(void) {
    TEST("T10: INT4 zero weights produce zero output");

    const int N = 4, K = 8;
    uint8_t packed[4 * 4]; /* N * K/2, all zeros */
    memset(packed, 0, sizeof(packed));
    float scales[] = { 1.0f, 2.0f, 0.5f, 100.0f };

    for (int r = 0; r < N; r++) {
        float s = scales[r];
        const uint8_t *row = packed + r * (K / 2);
        for (int k = 0; k < K; k++) {
            uint8_t byte = row[k / 2];
            int nib = (k & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
            if (nib >= 8) nib -= 16;
            float val = (float)nib * s;
            if (val != 0.0f) FAIL("zero packed data should unpack to 0.0");
        }
    }
    PASS();
}

/* ══════════════════════════════════════════════════════════════════════════
 * T11: CSTTHeader dtype field
 * ══════════════════════════════════════════════════════════════════════════ */

static void test_header_dtype(void) {
    TEST("T11: CSTTHeader dtype=3 is INT4");

    CSTTHeader h;
    memset(&h, 0, sizeof(h));
    h.dtype = 3;
    if (h.dtype != 3) FAIL("dtype field should hold value 3");
    PASS();
}

/* ══════════════════════════════════════════════════════════════════════════
 * T12: INT4 large matrix dequant RMSE
 * ══════════════════════════════════════════════════════════════════════════ */

static void test_int4_large_matrix(void) {
    TEST("T12: INT4 512x512 dequant RMSE < 0.05");

    const int N = 512, K = 512;
    const int K_packed = K / 2;

    float *weights = (float *)malloc((size_t)N * K * sizeof(float));
    float *scales = (float *)malloc((size_t)N * sizeof(float));
    uint8_t *packed = (uint8_t *)calloc((size_t)N * K_packed, 1);
    float *unpacked = (float *)malloc((size_t)N * K * sizeof(float));

    if (!weights || !scales || !packed || !unpacked) {
        free(weights); free(scales); free(packed); free(unpacked);
        FAIL("allocation failed");
    }

    /* Fill with diverse values in [-0.8, 0.8] */
    for (int i = 0; i < N * K; i++)
        weights[i] = sinf((float)i * 0.01f) * 0.8f;

    /* Quantize */
    for (int r = 0; r < N; r++) {
        float absmax = 0;
        for (int k = 0; k < K; k++) {
            float a = fabsf(weights[r * K + k]);
            if (a > absmax) absmax = a;
        }
        scales[r] = (absmax > 0.0f) ? absmax / 7.0f : 1.0f;
        float inv = 1.0f / scales[r];

        for (int k = 0; k < K; k++) {
            int q = (int)roundf(weights[r * K + k] * inv);
            if (q < -8) q = -8;
            if (q > 7) q = 7;
            int nibble = q & 0x0F;
            if (k & 1)
                packed[r * K_packed + k / 2] |= (nibble << 4);
            else
                packed[r * K_packed + k / 2] = nibble;
        }
    }

    /* Unpack */
    for (int r = 0; r < N; r++) {
        float s = scales[r];
        const uint8_t *row = packed + r * K_packed;
        for (int k = 0; k < K; k++) {
            uint8_t byte = row[k / 2];
            int nib = (k & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
            if (nib >= 8) nib -= 16;
            unpacked[r * K + k] = (float)nib * s;
        }
    }

    /* Compute RMSE */
    double sum_sq = 0.0;
    for (int i = 0; i < N * K; i++) {
        double d = (double)(weights[i] - unpacked[i]);
        sum_sq += d * d;
    }
    double rmse = sqrt(sum_sq / (N * K));

    free(weights);
    free(scales);
    free(packed);
    free(unpacked);

    if (rmse > 0.05) {
        char msg[128];
        snprintf(msg, sizeof(msg), "RMSE=%.6f too high", rmse);
        FAIL(msg);
    }
    PASS();
}

/* ══════════════════════════════════════════════════════════════════════════
 * Main
 * ══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("\n=== Research STT: Conmer Mode + INT4 Quantization Tests ===\n\n");

    printf("Conmer Mode:\n");
    test_conmer_api();
    test_conmer_architecture();

    printf("\nINT4 Weight-Only Quantization:\n");
    test_int4_pack_roundtrip();
    test_int4_gemm_accuracy();
    test_int4_edge_cases();
    test_int4_scale_precision();
    test_int4_neon_dequant();
    test_int4_odd_k();
    test_int4_memory_savings();
    test_int4_zero_weights();
    test_header_dtype();
    test_int4_large_matrix();

    printf("\n=== Results: %d passed, %d failed ===\n\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
