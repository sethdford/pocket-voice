/**
 * test_metal_dispatch.c — Tests for Metal GPU dispatch bridge.
 *
 * Validates:
 *   - Init/cleanup lifecycle
 *   - CPU fallback when Metal unavailable
 *   - fp32 GEMM correctness (GPU path with metallib, CPU fallback otherwise)
 *   - Small matrix CPU fast-path (< 4096 elements)
 *   - NULL safety
 *   - Repeated init/cleanup cycles
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <Accelerate/Accelerate.h>

#include "metal_dispatch.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-60s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; return; } while(0)

/* ── Lifecycle Tests ────────────────────────────────────────────── */

static void test_init_null_path(void) {
    TEST("dispatch: init(NULL) returns 0 (CPU mode)");
    metal_dispatch_cleanup();
    int ret = metal_dispatch_init(NULL);
    if (ret != 0) FAIL("expected 0 for NULL path");
    PASS();
}

static void test_cleanup_without_init(void) {
    TEST("dispatch: cleanup() without init does not crash");
    metal_dispatch_cleanup();
    metal_dispatch_cleanup();
    PASS();
}

static void test_available_without_init(void) {
    TEST("dispatch: available() without init returns 0");
    metal_dispatch_cleanup();
    if (metal_dispatch_available() != 0)
        FAIL("expected 0 before init");
    PASS();
}

static void test_init_cleanup_cycle(void) {
    TEST("dispatch: repeated init/cleanup cycles are safe");
    for (int i = 0; i < 5; i++) {
        metal_dispatch_init(NULL);
        metal_dispatch_cleanup();
    }
    PASS();
}

static void test_double_init(void) {
    TEST("dispatch: double init is a no-op");
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);
    int first = metal_dispatch_available();
    metal_dispatch_init(NULL); /* should be no-op */
    int second = metal_dispatch_available();
    if (first != second) FAIL("availability changed on double init");
    metal_dispatch_cleanup();
    PASS();
}

/* ── CPU Fallback GEMM Tests ────────────────────────────────────── */

static void test_gemm_cpu_fallback_small(void) {
    TEST("dispatch: GEMM cpu fallback (2x3 @ 3x2 = 2x2)");
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL); /* no metallib → CPU mode */

    /* A[2,3] = {{1,2,3},{4,5,6}}
     * B[2,3] = {{7,8,9},{10,11,12}}  (B^T is [3,2])
     * C = A @ B^T = {{1*7+2*8+3*9, 1*10+2*11+3*12},
     *                {4*7+5*8+6*9, 4*10+5*11+6*12}}
     *            = {{50, 68}, {122, 167}}
     */
    float A[] = {1,2,3, 4,5,6};
    float B[] = {7,8,9, 10,11,12};
    float C[4] = {0};
    float expected[] = {50, 68, 122, 167};

    int ret = metal_dispatch_gemm(A, B, C, 2, 2, 3);
    if (ret != 0) FAIL("gemm returned non-zero");

    for (int i = 0; i < 4; i++) {
        if (fabsf(C[i] - expected[i]) > 0.01f) {
            char msg[128];
            snprintf(msg, sizeof(msg), "C[%d]=%.2f expected %.2f", i, C[i], expected[i]);
            FAIL(msg);
        }
    }
    metal_dispatch_cleanup();
    PASS();
}

static void test_gemm_cpu_identity(void) {
    TEST("dispatch: GEMM cpu fallback (identity multiply)");
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);

    /* A[4,4] = eye, B[4,4] = eye → C = eye @ eye^T = eye */
    float eye[16] = {0};
    eye[0] = eye[5] = eye[10] = eye[15] = 1.0f;
    float C[16] = {0};

    int ret = metal_dispatch_gemm(eye, eye, C, 4, 4, 4);
    if (ret != 0) FAIL("gemm returned non-zero");

    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            float expected = (r == c) ? 1.0f : 0.0f;
            if (fabsf(C[r*4+c] - expected) > 1e-5f) FAIL("identity mismatch");
        }
    }
    metal_dispatch_cleanup();
    PASS();
}

static void test_gemm_cpu_alpha(void) {
    TEST("dispatch: GEMM with alpha=2.0 scaling");
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);

    float A[] = {1,0, 0,1};
    float B[] = {3,0, 0,3};
    float C[4] = {0};

    int ret = metal_dispatch_gemm_alpha(A, B, C, 2, 2, 2, 2.0f);
    if (ret != 0) FAIL("gemm_alpha returned non-zero");
    /* C = 2.0 * I @ diag(3,3)^T = 2.0 * diag(3,3) = diag(6,6) */
    if (fabsf(C[0] - 6.0f) > 0.01f || fabsf(C[3] - 6.0f) > 0.01f)
        FAIL("alpha scaling incorrect");
    if (fabsf(C[1]) > 0.01f || fabsf(C[2]) > 0.01f)
        FAIL("off-diagonal should be zero");

    metal_dispatch_cleanup();
    PASS();
}

static void test_gemm_cpu_larger(void) {
    TEST("dispatch: GEMM cpu fallback (32x32 random)");
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);

    int M = 32, N = 32, K = 32;
    float *A = calloc(M * K, sizeof(float));
    float *B = calloc(N * K, sizeof(float));
    float *C = calloc(M * N, sizeof(float));
    float *Cref = calloc(M * N, sizeof(float));

    /* Fill with simple pattern */
    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 7) * 0.1f;
    for (int i = 0; i < N * K; i++) B[i] = (float)(i % 5) * 0.1f;

    /* Reference: cblas */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, Cref, N);

    int ret = metal_dispatch_gemm(A, B, C, M, N, K);
    if (ret != 0) { free(A); free(B); free(C); free(Cref); FAIL("gemm returned non-zero"); }

    float max_err = 0;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(C[i] - Cref[i]);
        if (err > max_err) max_err = err;
    }

    free(A); free(B); free(C); free(Cref);
    if (max_err > 0.01f) {
        char msg[64];
        snprintf(msg, sizeof(msg), "max error=%.6f (limit 0.01)", max_err);
        FAIL(msg);
    }
    PASS();
}

/* ── GPU Path Tests (if metallib available) ──────────────────────── */

static void test_gpu_init(void) {
    TEST("dispatch: init with metallib (GPU path)");
    metal_dispatch_cleanup();
    int ret = metal_dispatch_init("build/tensor_ops.metallib");
    /* GPU may or may not be available — just check no crash */
    if (ret) {
        printf("GPU   ");
    } else {
        printf("CPU   ");
    }
    PASS();
}

static void test_gpu_gemm_correctness(void) {
    TEST("dispatch: GPU GEMM correctness (64x64)");

    /* Re-init to ensure we have GPU if available */
    metal_dispatch_cleanup();
    int gpu = metal_dispatch_init("build/tensor_ops.metallib");

    int M = 64, N = 64, K = 64;
    float *A = calloc(M * K, sizeof(float));
    float *B = calloc(N * K, sizeof(float));
    float *C = calloc(M * N, sizeof(float));
    float *Cref = calloc(M * N, sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 11) * 0.05f;
    for (int i = 0; i < N * K; i++) B[i] = (float)(i % 13) * 0.05f;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, Cref, N);

    int ret = metal_dispatch_gemm(A, B, C, M, N, K);
    if (ret != 0) { free(A); free(B); free(C); free(Cref); FAIL("gemm returned non-zero"); }

    /* fp16 GPU path has ~0.1% relative error; CPU path is exact */
    float tolerance = gpu ? 0.5f : 0.001f;
    float max_err = 0;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(C[i] - Cref[i]);
        if (err > max_err) max_err = err;
    }

    free(A); free(B); free(C); free(Cref);
    if (max_err > tolerance) {
        char msg[64];
        snprintf(msg, sizeof(msg), "max error=%.6f (limit %.3f)", max_err, tolerance);
        FAIL(msg);
    }
    metal_dispatch_cleanup();
    PASS();
}

/* ── fp16 passthrough tests ─────────────────────────────────────── */

static void test_fp16_gemm_no_gpu(void) {
    TEST("dispatch: fp16 GEMM returns -1 when no GPU");
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);

    float dummy[16] = {0};
    int ret = metal_dispatch_gemm_f16(dummy, dummy, dummy, 2, 2, 2, 1.0f);
    if (ret != -1) FAIL("expected -1 for fp16 without GPU");
    metal_dispatch_cleanup();
    PASS();
}

static void test_silu_no_gpu(void) {
    TEST("dispatch: silu_gate returns -1 when no GPU");
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);

    float dummy[16] = {0};
    int ret = metal_dispatch_silu_gate(dummy, dummy, 2, 4);
    if (ret != -1) FAIL("expected -1 without GPU");
    metal_dispatch_cleanup();
    PASS();
}

static void test_flash_attention_no_gpu(void) {
    TEST("dispatch: flash_attention returns -1 when no GPU");
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);

    float dummy[16] = {0};
    int ret = metal_dispatch_flash_attention(dummy, dummy, dummy, dummy, 2, 2, 4);
    if (ret != -1) FAIL("expected -1 without GPU");
    metal_dispatch_cleanup();
    PASS();
}

static void test_layer_norm_no_gpu(void) {
    TEST("dispatch: layer_norm returns -1 when no GPU");
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);

    float dummy[16] = {0};
    int ret = metal_dispatch_layer_norm(dummy, dummy, dummy, dummy, 2, 4, 1e-5f);
    if (ret != -1) FAIL("expected -1 without GPU");
    metal_dispatch_cleanup();
    PASS();
}

/* ── Main ───────────────────────────────────────────────────────── */

int main(void) {
    printf("\n=== Metal Dispatch Bridge Test Suite ===\n\n");

    printf("Lifecycle:\n");
    test_init_null_path();
    test_cleanup_without_init();
    test_available_without_init();
    test_init_cleanup_cycle();
    test_double_init();

    printf("\nCPU Fallback GEMM:\n");
    test_gemm_cpu_fallback_small();
    test_gemm_cpu_identity();
    test_gemm_cpu_alpha();
    test_gemm_cpu_larger();

    printf("\nGPU Path (if metallib available):\n");
    test_gpu_init();
    test_gpu_gemm_correctness();

    printf("\nfp16 Passthrough (no GPU):\n");
    test_fp16_gemm_no_gpu();
    test_silu_no_gpu();
    test_flash_attention_no_gpu();
    test_layer_norm_no_gpu();

    printf("\n=== Results: %d passed, %d failed ===\n\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
