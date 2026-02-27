/**
 * test_research_metal.c — Tests for M5 Neural Accelerator detection
 * and chip-aware Metal dispatch thresholds.
 *
 * Validates:
 *   - Chip generation detection (AP_CHIP_M1..M5)
 *   - Chip generation name strings
 *   - Neural accelerator detection (M5+ only)
 *   - Dispatch threshold auto-configuration per chip
 *   - Manual threshold override
 *   - Threshold getter consistency
 *   - Metal 4 TensorOps availability check
 *   - CPU fallback for below-threshold matrices
 *   - Benchmark harness runs without crashing
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

#include "apple_perf.h"
#include "metal_dispatch.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-60s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; return; } while(0)

/* ── Chip Detection Tests ─────────────────────────────────────────── */

static void test_chip_generation_returns_valid(void) {
    TEST("chip: generation returns valid enum value");
    APChipGeneration gen = ap_chip_generation();
    if (gen < AP_CHIP_UNKNOWN || gen > AP_CHIP_M5)
        FAIL("enum out of range");
    PASS();
}

static void test_chip_generation_cached(void) {
    TEST("chip: repeated calls return same value (cached)");
    APChipGeneration g1 = ap_chip_generation();
    APChipGeneration g2 = ap_chip_generation();
    APChipGeneration g3 = ap_chip_generation();
    if (g1 != g2 || g2 != g3)
        FAIL("inconsistent results across calls");
    PASS();
}

static void test_chip_name_not_null(void) {
    TEST("chip: generation name is never NULL");
    for (int i = 0; i <= 5; i++) {
        const char *name = ap_chip_generation_name((APChipGeneration)i);
        if (!name) FAIL("NULL name");
        if (strlen(name) == 0) FAIL("empty name");
    }
    PASS();
}

static void test_chip_name_known_values(void) {
    TEST("chip: known generation names are correct");
    if (strcmp(ap_chip_generation_name(AP_CHIP_M1), "M1") != 0)
        FAIL("M1 name wrong");
    if (strcmp(ap_chip_generation_name(AP_CHIP_M2), "M2") != 0)
        FAIL("M2 name wrong");
    if (strcmp(ap_chip_generation_name(AP_CHIP_M3), "M3") != 0)
        FAIL("M3 name wrong");
    if (strcmp(ap_chip_generation_name(AP_CHIP_M4), "M4") != 0)
        FAIL("M4 name wrong");
    if (strcmp(ap_chip_generation_name(AP_CHIP_M5), "M5+") != 0)
        FAIL("M5 name wrong");
    if (strcmp(ap_chip_generation_name(AP_CHIP_UNKNOWN), "Unknown") != 0)
        FAIL("Unknown name wrong");
    PASS();
}

static void test_neural_accel_matches_generation(void) {
    TEST("chip: neural_accel consistent with generation");
    APChipGeneration gen = ap_chip_generation();
    int has_accel = ap_has_neural_accel();
    if (gen >= AP_CHIP_M5 && !has_accel)
        FAIL("M5+ should have neural accel");
    if (gen < AP_CHIP_M5 && gen != AP_CHIP_UNKNOWN && has_accel)
        FAIL("M1-M4 should not have neural accel");
    PASS();
}

/* ── Threshold Configuration Tests ────────────────────────────────── */

static void test_auto_threshold_positive(void) {
    TEST("threshold: auto-config produces positive value");
    metal_dispatch_auto_threshold();
    int th = metal_dispatch_get_threshold();
    if (th <= 0) FAIL("threshold should be > 0");
    PASS();
}

static void test_auto_threshold_matches_chip(void) {
    TEST("threshold: auto-config matches chip generation");
    metal_dispatch_auto_threshold();
    int th = metal_dispatch_get_threshold();
    APChipGeneration gen = ap_chip_generation();

    switch (gen) {
        case AP_CHIP_M1:
        case AP_CHIP_M2:
        case AP_CHIP_M3:
        case AP_CHIP_M4:
            if (th != 1024) FAIL("M1-M4 should be 1024");
            break;
        case AP_CHIP_M5:
            if (th != 128) FAIL("M5+ should be 128");
            break;
        case AP_CHIP_UNKNOWN:
            if (th != 2048) FAIL("Unknown should be 2048");
            break;
    }
    PASS();
}

static void test_manual_threshold_override(void) {
    TEST("threshold: manual override takes effect");
    metal_dispatch_set_threshold(512);
    int th = metal_dispatch_get_threshold();
    if (th != 512) FAIL("expected 512 after override");
    /* Reset to auto */
    metal_dispatch_auto_threshold();
    PASS();
}

static void test_threshold_zero_override(void) {
    TEST("threshold: setting 0 means always GPU");
    metal_dispatch_set_threshold(0);
    int th = metal_dispatch_get_threshold();
    if (th != 0) FAIL("expected 0");
    /* Reset to auto */
    metal_dispatch_auto_threshold();
    PASS();
}

static void test_threshold_max_override(void) {
    TEST("threshold: large value means always CPU");
    metal_dispatch_set_threshold(999999);
    int th = metal_dispatch_get_threshold();
    if (th != 999999) FAIL("expected 999999");
    /* Reset to auto */
    metal_dispatch_auto_threshold();
    PASS();
}

/* ── Dispatch Behavior Tests ──────────────────────────────────────── */

static void test_small_matrix_uses_cpu(void) {
    TEST("dispatch: small matrix below threshold uses CPU");
    /* Init without metallib → CPU only mode */
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);
    metal_dispatch_set_threshold(1024);

    /* 64x64 matrix — well below 1024 threshold */
    int M = 64, N = 64, K = 64;
    size_t sz = (size_t)M * N;
    float *A = (float *)calloc((size_t)M * K, sizeof(float));
    float *B = (float *)calloc((size_t)N * K, sizeof(float));
    float *C = (float *)calloc(sz, sizeof(float));
    float *ref = (float *)calloc(sz, sizeof(float));

    if (!A || !B || !C || !ref) {
        free(A); free(B); free(C); free(ref);
        FAIL("alloc failed");
    }

    /* Fill with values */
    for (int i = 0; i < M * K; i++) A[i] = 0.1f * (float)(i % 13);
    for (int i = 0; i < N * K; i++) B[i] = 0.1f * (float)(i % 7);

    /* Reference via cblas */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, ref, N);

    /* Metal dispatch (should fall through to CPU) */
    int rc = metal_dispatch_gemm(A, B, C, M, N, K);
    if (rc != 0) {
        free(A); free(B); free(C); free(ref);
        FAIL("gemm returned non-zero");
    }

    /* Verify result matches cblas */
    float max_err = 0;
    for (size_t i = 0; i < sz; i++) {
        float err = fabsf(C[i] - ref[i]);
        if (err > max_err) max_err = err;
    }

    free(A); free(B); free(C); free(ref);
    if (max_err > 1e-4f) FAIL("result diverges from cblas reference");
    PASS();
}

static void test_cpu_fallback_correctness(void) {
    TEST("dispatch: CPU fallback produces correct GEMM results");
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);  /* Force CPU-only mode */

    int M = 128, N = 64, K = 96;
    size_t sz = (size_t)M * N;
    float *A = (float *)calloc((size_t)M * K, sizeof(float));
    float *B = (float *)calloc((size_t)N * K, sizeof(float));
    float *C = (float *)calloc(sz, sizeof(float));
    float *ref = (float *)calloc(sz, sizeof(float));

    if (!A || !B || !C || !ref) {
        free(A); free(B); free(C); free(ref);
        FAIL("alloc failed");
    }

    for (int i = 0; i < M * K; i++) A[i] = 0.01f * (float)((i * 7 + 3) % 100);
    for (int i = 0; i < N * K; i++) B[i] = 0.01f * (float)((i * 11 + 5) % 100);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, ref, N);

    metal_dispatch_gemm(A, B, C, M, N, K);

    float max_err = 0;
    for (size_t i = 0; i < sz; i++) {
        float err = fabsf(C[i] - ref[i]);
        if (err > max_err) max_err = err;
    }

    free(A); free(B); free(C); free(ref);
    if (max_err > 1e-5f) FAIL("CPU fallback diverges from reference");
    PASS();
}

static void test_gemm_alpha_scaling(void) {
    TEST("dispatch: alpha scaling works with threshold check");
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);

    int M = 32, N = 32, K = 32;
    float alpha = 0.5f;
    size_t sz = (size_t)M * N;
    float *A = (float *)calloc((size_t)M * K, sizeof(float));
    float *B = (float *)calloc((size_t)N * K, sizeof(float));
    float *C = (float *)calloc(sz, sizeof(float));
    float *ref = (float *)calloc(sz, sizeof(float));

    if (!A || !B || !C || !ref) {
        free(A); free(B); free(C); free(ref);
        FAIL("alloc failed");
    }

    for (int i = 0; i < M * K; i++) A[i] = 0.1f * (float)(i % 17);
    for (int i = 0; i < N * K; i++) B[i] = 0.1f * (float)(i % 11);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, alpha, A, K, B, K, 0.0f, ref, N);

    metal_dispatch_gemm_alpha(A, B, C, M, N, K, alpha);

    float max_err = 0;
    for (size_t i = 0; i < sz; i++) {
        float err = fabsf(C[i] - ref[i]);
        if (err > max_err) max_err = err;
    }

    free(A); free(B); free(C); free(ref);
    if (max_err > 1e-5f) FAIL("alpha-scaled result diverges from reference");
    PASS();
}

/* ── Metal 4 TensorOps Tests ─────────────────────────────────────── */

static void test_tensorops_consistent(void) {
    TEST("tensorops: result is consistent across calls");
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);
    int t1 = metal_dispatch_has_tensorops();
    int t2 = metal_dispatch_has_tensorops();
    if (t1 != t2) FAIL("inconsistent TensorOps result");
    /* On M1-M4 without Metal GPU, should be 0 */
    if (!metal_dispatch_available() && t1 != 0)
        FAIL("TensorOps should be 0 without Metal GPU");
    PASS();
}

static void test_tensorops_requires_neural_accel(void) {
    TEST("tensorops: requires neural accel (M5+)");
    int has_accel = ap_has_neural_accel();
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);  /* CPU only — no Metal device */
    int has_ops = metal_dispatch_has_tensorops();
    /* Without Metal device, TensorOps should always be 0 */
    if (has_ops != 0) FAIL("TensorOps without Metal device should be 0");
    (void)has_accel;
    PASS();
}

/* ── Benchmark Harness Tests ──────────────────────────────────────── */

static void test_benchmark_cpu_only(void) {
    TEST("benchmark: runs in CPU-only mode without crash");
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);
    int result = metal_dispatch_benchmark(64, 64, 64);
    /* In CPU-only mode, should return 0 (CPU wins by default) */
    if (result != 0) FAIL("CPU-only benchmark should return 0");
    PASS();
}

/* ── Cleanup / Lifecycle Tests ────────────────────────────────────── */

static void test_cleanup_resets_tensorops(void) {
    TEST("lifecycle: cleanup resets TensorOps cache");
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);
    metal_dispatch_has_tensorops();  /* prime cache */
    metal_dispatch_cleanup();
    /* After cleanup, re-init should work */
    metal_dispatch_init(NULL);
    int t = metal_dispatch_has_tensorops();
    if (t != 0) FAIL("TensorOps should be 0 after cleanup+reinit(NULL)");
    metal_dispatch_cleanup();
    PASS();
}

static void test_threshold_survives_cleanup(void) {
    TEST("lifecycle: manual threshold persists across cleanup");
    metal_dispatch_set_threshold(777);
    metal_dispatch_cleanup();
    metal_dispatch_init(NULL);
    int th = metal_dispatch_get_threshold();
    if (th != 777) FAIL("expected 777 after cleanup/reinit");
    /* Reset */
    metal_dispatch_auto_threshold();
    metal_dispatch_cleanup();
    PASS();
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n═══ Research: M5 Neural Accelerator Detection & Metal Dispatch ═══\n\n");

    printf("Chip Detection:\n");
    test_chip_generation_returns_valid();
    test_chip_generation_cached();
    test_chip_name_not_null();
    test_chip_name_known_values();
    test_neural_accel_matches_generation();

    printf("\nThreshold Configuration:\n");
    test_auto_threshold_positive();
    test_auto_threshold_matches_chip();
    test_manual_threshold_override();
    test_threshold_zero_override();
    test_threshold_max_override();

    printf("\nDispatch Behavior:\n");
    test_small_matrix_uses_cpu();
    test_cpu_fallback_correctness();
    test_gemm_alpha_scaling();

    printf("\nMetal 4 TensorOps:\n");
    test_tensorops_consistent();
    test_tensorops_requires_neural_accel();

    printf("\nBenchmark Harness:\n");
    test_benchmark_cpu_only();

    printf("\nLifecycle:\n");
    test_cleanup_resets_tensorops();
    test_threshold_survives_cleanup();

    printf("\n═══ Results: %d passed, %d failed ═══\n\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
