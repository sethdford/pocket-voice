/**
 * test_apple_perf.c — Tests for Apple Silicon performance primitives.
 *
 * Verifies correctness and measures throughput of:
 *   - Real-time thread scheduling
 *   - NEON softmax, GELU, SiLU, layernorm, rmsnorm
 *   - IOSurface zero-copy buffers
 *   - Model mmap with prefetch
 *   - AMX alignment utilities
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>
#include "apple_perf.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define PASS(msg) do { \
    fprintf(stderr, "  PASS: %s\n", msg); \
    tests_passed++; \
} while(0)

static double ns_elapsed(uint64_t start, uint64_t end)
{
    static mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0) mach_timebase_info(&info);
    return (double)(end - start) * info.numer / info.denom;
}

/* Reference implementations for correctness checking */

static void ref_softmax(const float *in, float *out, int n) {
    float max_val = -HUGE_VALF;
    for (int i = 0; i < n; i++)
        if (in[i] > max_val) max_val = in[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        out[i] = expf(in[i] - max_val);
        sum += out[i];
    }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

static void ref_gelu(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float x = in[i];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        out[i] = x * 0.5f * (1.0f + tanhf(inner));
    }
}

static void ref_silu(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float x = in[i];
        out[i] = x / (1.0f + expf(-x));
    }
}

static void ref_layernorm(const float *in, float *out, const float *g,
                           const float *b, int n, float eps) {
    float mean = 0;
    for (int i = 0; i < n; i++) mean += in[i];
    mean /= n;
    float var = 0;
    for (int i = 0; i < n; i++) { float d = in[i] - mean; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++)
        out[i] = (in[i] - mean) * inv * g[i] + b[i];
}

static void ref_rmsnorm(const float *in, float *out, const float *g,
                          int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += in[i] * in[i];
    float inv = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = in[i] * inv * g[i];
}

/* ═══════════════════════════════════════════════════════════════════════════ */

static void test_realtime_priority(void)
{
    fprintf(stderr, "\n=== Real-Time Thread Priority ===\n");

    int rc = ap_set_realtime_audio();
    ASSERT(rc == 0, "ap_set_realtime_audio() succeeds");
    PASS("Real-time audio priority set");

    rc = ap_set_realtime_inference();
    ASSERT(rc == 0, "ap_set_realtime_inference() succeeds");
    PASS("Real-time inference priority set");

    rc = ap_set_qos_user_interactive();
    ASSERT(rc == 0, "ap_set_qos_user_interactive() succeeds");
    PASS("QoS User Interactive set");
}

static void test_neon_softmax(void)
{
    fprintf(stderr, "\n=== NEON Softmax ===\n");

    const int N = 1024;
    float *in  = malloc(N * sizeof(float));
    float *ref = malloc(N * sizeof(float));
    float *out = malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < N; i++)
        in[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;

    ref_softmax(in, ref, N);
    ap_neon_softmax(in, out, N);

    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float err = fabsf(ref[i] - out[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT(max_err < 1e-3f, "NEON softmax matches reference within 1e-3");
    fprintf(stderr, "  max error: %.2e\n", max_err);

    /* Benchmark */
    const int ITERS = 10000;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < ITERS; i++)
        ap_neon_softmax(in, out, N);
    uint64_t t1 = mach_absolute_time();
    double ns_per = ns_elapsed(t0, t1) / ITERS;
    fprintf(stderr, "  throughput: %.1f ns/call (%d elements)\n", ns_per, N);
    PASS("NEON softmax correct and fast");

    free(in); free(ref); free(out);
}

static void test_neon_gelu(void)
{
    fprintf(stderr, "\n=== NEON GELU ===\n");

    const int N = 1024;
    float *in  = malloc(N * sizeof(float));
    float *ref = malloc(N * sizeof(float));
    float *out = malloc(N * sizeof(float));

    srand(123);
    for (int i = 0; i < N; i++)
        in[i] = ((float)rand() / RAND_MAX) * 6.0f - 3.0f;

    ref_gelu(in, ref, N);
    ap_neon_gelu(in, out, N);

    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float err = fabsf(ref[i] - out[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT(max_err < 1e-3f, "NEON GELU matches reference within 1e-3");
    fprintf(stderr, "  max error: %.2e\n", max_err);

    const int ITERS = 10000;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < ITERS; i++)
        ap_neon_gelu(in, out, N);
    uint64_t t1 = mach_absolute_time();
    fprintf(stderr, "  throughput: %.1f ns/call (%d elements)\n",
            ns_elapsed(t0, t1) / ITERS, N);
    PASS("NEON GELU correct and fast");

    free(in); free(ref); free(out);
}

static void test_neon_silu(void)
{
    fprintf(stderr, "\n=== NEON SiLU ===\n");

    const int N = 1024;
    float *in  = malloc(N * sizeof(float));
    float *ref = malloc(N * sizeof(float));
    float *out = malloc(N * sizeof(float));

    srand(456);
    for (int i = 0; i < N; i++)
        in[i] = ((float)rand() / RAND_MAX) * 8.0f - 4.0f;

    ref_silu(in, ref, N);
    ap_neon_silu(in, out, N);

    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float err = fabsf(ref[i] - out[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT(max_err < 1e-3f, "NEON SiLU matches reference within 1e-3");
    fprintf(stderr, "  max error: %.2e\n", max_err);

    const int ITERS = 10000;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < ITERS; i++)
        ap_neon_silu(in, out, N);
    uint64_t t1 = mach_absolute_time();
    fprintf(stderr, "  throughput: %.1f ns/call (%d elements)\n",
            ns_elapsed(t0, t1) / ITERS, N);
    PASS("NEON SiLU correct and fast");

    free(in); free(ref); free(out);
}

static void test_neon_layernorm(void)
{
    fprintf(stderr, "\n=== NEON LayerNorm ===\n");

    const int N = 512;
    float *in  = malloc(N * sizeof(float));
    float *g   = malloc(N * sizeof(float));
    float *b   = malloc(N * sizeof(float));
    float *ref = malloc(N * sizeof(float));
    float *out = malloc(N * sizeof(float));

    srand(789);
    for (int i = 0; i < N; i++) {
        in[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        g[i]  = 0.8f + ((float)rand() / RAND_MAX) * 0.4f;
        b[i]  = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }

    ref_layernorm(in, ref, g, b, N, 1e-5f);
    ap_neon_layernorm(in, out, g, b, N, 1e-5f);

    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float err = fabsf(ref[i] - out[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT(max_err < 1e-4f, "NEON LayerNorm matches reference within 1e-4");
    fprintf(stderr, "  max error: %.2e\n", max_err);

    const int ITERS = 10000;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < ITERS; i++)
        ap_neon_layernorm(in, out, g, b, N, 1e-5f);
    uint64_t t1 = mach_absolute_time();
    fprintf(stderr, "  throughput: %.1f ns/call (%d elements)\n",
            ns_elapsed(t0, t1) / ITERS, N);
    PASS("NEON LayerNorm correct and fast");

    free(in); free(g); free(b); free(ref); free(out);
}

static void test_neon_rmsnorm(void)
{
    fprintf(stderr, "\n=== NEON RMSNorm ===\n");

    const int N = 512;
    float *in  = malloc(N * sizeof(float));
    float *g   = malloc(N * sizeof(float));
    float *ref = malloc(N * sizeof(float));
    float *out = malloc(N * sizeof(float));

    srand(321);
    for (int i = 0; i < N; i++) {
        in[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        g[i]  = 0.9f + ((float)rand() / RAND_MAX) * 0.2f;
    }

    ref_rmsnorm(in, ref, g, N, 1e-5f);
    ap_neon_rmsnorm(in, out, g, N, 1e-5f);

    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float err = fabsf(ref[i] - out[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT(max_err < 1e-4f, "NEON RMSNorm matches reference within 1e-4");
    fprintf(stderr, "  max error: %.2e\n", max_err);

    const int ITERS = 10000;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < ITERS; i++)
        ap_neon_rmsnorm(in, out, g, N, 1e-5f);
    uint64_t t1 = mach_absolute_time();
    fprintf(stderr, "  throughput: %.1f ns/call (%d elements)\n",
            ns_elapsed(t0, t1) / ITERS, N);
    PASS("NEON RMSNorm correct and fast");

    free(in); free(g); free(ref); free(out);
}

static void test_residual_layernorm(void)
{
    fprintf(stderr, "\n=== NEON Residual + LayerNorm (Fused) ===\n");

    const int N = 512;
    float *x   = malloc(N * sizeof(float));
    float *res = malloc(N * sizeof(float));
    float *g   = malloc(N * sizeof(float));
    float *b   = malloc(N * sizeof(float));
    float *ref = malloc(N * sizeof(float));
    float *ref_added = malloc(N * sizeof(float));
    float *out = malloc(N * sizeof(float));

    srand(654);
    for (int i = 0; i < N; i++) {
        x[i]   = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        res[i]  = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        g[i]    = 0.8f + ((float)rand() / RAND_MAX) * 0.4f;
        b[i]    = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }

    /* Reference: add then layernorm */
    for (int i = 0; i < N; i++) ref_added[i] = x[i] + res[i];
    ref_layernorm(ref_added, ref, g, b, N, 1e-5f);

    ap_neon_residual_layernorm(x, res, out, g, b, N, 1e-5f);

    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float err = fabsf(ref[i] - out[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT(max_err < 1e-4f, "Fused residual+layernorm matches reference");
    fprintf(stderr, "  max error: %.2e\n", max_err);

    const int ITERS = 10000;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < ITERS; i++)
        ap_neon_residual_layernorm(x, res, out, g, b, N, 1e-5f);
    uint64_t t1 = mach_absolute_time();
    fprintf(stderr, "  throughput: %.1f ns/call (%d elements)\n",
            ns_elapsed(t0, t1) / ITERS, N);
    PASS("Fused residual+layernorm correct and fast");

    free(x); free(res); free(g); free(b); free(ref); free(ref_added); free(out);
}

static void test_zerocopy_buffer(void)
{
    fprintf(stderr, "\n=== IOSurface Zero-Copy Buffer ===\n");

    APZeroCopyBuffer buf = ap_zerocopy_create(64 * 1024);
    ASSERT(buf.cpu_ptr != NULL, "IOSurface buffer created");
    ASSERT(buf.size >= 64 * 1024, "Buffer size correct");

    /* Write + read back */
    float *fp = (float *)buf.cpu_ptr;
    for (int i = 0; i < 1024; i++)
        fp[i] = (float)i * 0.001f;

    float sum = 0;
    for (int i = 0; i < 1024; i++)
        sum += fp[i];

    ASSERT(sum > 0, "IOSurface read/write works");
    PASS("IOSurface zero-copy buffer functional");

    ap_zerocopy_destroy(&buf);
    ASSERT(buf.cpu_ptr == NULL, "Buffer destroyed cleanly");
    PASS("IOSurface cleanup successful");
}

static void test_amx_alignment(void)
{
    fprintf(stderr, "\n=== AMX Alignment Utilities ===\n");

    ASSERT(ap_amx_align(1) == 32, "align(1) == 32");
    ASSERT(ap_amx_align(32) == 32, "align(32) == 32");
    ASSERT(ap_amx_align(33) == 64, "align(33) == 64");
    ASSERT(ap_amx_align(512) == 512, "align(512) == 512");
    PASS("AMX alignment correct");

    float *p = ap_amx_alloc(256);
    ASSERT(p != NULL, "AMX alloc succeeded");
    ASSERT(((uintptr_t)p & 127) == 0, "128-byte aligned");
    free(p);
    PASS("AMX-aligned allocation correct");
}

static void test_prefetch(void)
{
    fprintf(stderr, "\n=== Weight Prefetch ===\n");

    float *weights = malloc(4096 * sizeof(float));
    for (int i = 0; i < 4096; i++) weights[i] = (float)i;

    /* Should not crash, even with large counts */
    ap_prefetch_weights(weights, 4096);
    PASS("Weight prefetch completed without crash");

    ap_model_prefetch(weights, 4096 * sizeof(float));
    PASS("Model prefetch completed without crash");

    free(weights);
}

static void test_softmax_sizes(void)
{
    fprintf(stderr, "\n=== Softmax Edge Cases ===\n");

    /* Test various sizes including non-multiple-of-4 */
    int sizes[] = {1, 3, 4, 7, 15, 16, 31, 32, 127, 128, 255, 256, 1023};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < n_sizes; s++) {
        int N = sizes[s];
        float *in  = malloc(N * sizeof(float));
        float *ref = malloc(N * sizeof(float));
        float *out = malloc(N * sizeof(float));

        for (int i = 0; i < N; i++)
            in[i] = ((float)(i * 7 + 3) / N) - 0.5f;

        ref_softmax(in, ref, N);
        ap_neon_softmax(in, out, N);

        float max_err = 0;
        for (int i = 0; i < N; i++) {
            float err = fabsf(ref[i] - out[i]);
            if (err > max_err) max_err = err;
        }
        ASSERT(max_err < 1e-3f, "Softmax correct for non-aligned size");

        free(in); free(ref); free(out);
    }
    PASS("Softmax correct for all tested sizes");
}

/* ── Timing Accuracy Tests ──────────────────────────────────────────────── */

static void test_timing_accuracy(void)
{
    fprintf(stderr, "\n=== Timing Accuracy ===\n");

    /* Measure a known delay and verify it's in the right ballpark */
    uint64_t t0 = mach_absolute_time();
    /* Busy-wait spin for ~1ms by doing computation */
    volatile float x = 1.0f;
    for (int i = 0; i < 100000; i++) x *= 1.000001f;
    uint64_t t1 = mach_absolute_time();

    double elapsed_ns = ns_elapsed(t0, t1);
    ASSERT(elapsed_ns > 0, "elapsed time should be positive");
    ASSERT(elapsed_ns < 1e9, "elapsed time should be < 1 second for busy loop");
    fprintf(stderr, "  busy-wait elapsed: %.0f ns\n", elapsed_ns);
    PASS("Timing measurement returns positive, reasonable values");
}

static void test_timing_nested(void)
{
    fprintf(stderr, "\n=== Nested Timing Measurements ===\n");

    uint64_t outer_start = mach_absolute_time();

    /* Inner measurement 1 */
    uint64_t inner1_start = mach_absolute_time();
    volatile float x = 1.0f;
    for (int i = 0; i < 50000; i++) x *= 1.000001f;
    uint64_t inner1_end = mach_absolute_time();
    double inner1_ns = ns_elapsed(inner1_start, inner1_end);

    /* Inner measurement 2 */
    uint64_t inner2_start = mach_absolute_time();
    for (int i = 0; i < 50000; i++) x *= 1.000001f;
    uint64_t inner2_end = mach_absolute_time();
    double inner2_ns = ns_elapsed(inner2_start, inner2_end);

    uint64_t outer_end = mach_absolute_time();
    double outer_ns = ns_elapsed(outer_start, outer_end);

    ASSERT(inner1_ns > 0, "inner1 elapsed should be positive");
    ASSERT(inner2_ns > 0, "inner2 elapsed should be positive");
    ASSERT(outer_ns >= inner1_ns, "outer should be >= inner1");
    ASSERT(outer_ns >= inner2_ns, "outer should be >= inner2");
    fprintf(stderr, "  inner1=%.0f ns, inner2=%.0f ns, outer=%.0f ns\n",
            inner1_ns, inner2_ns, outer_ns);
    PASS("Nested timing measurements are consistent");
}

static void test_timing_reset_reuse(void)
{
    fprintf(stderr, "\n=== Timing Reset and Reuse ===\n");

    /* Take multiple independent measurements and verify they're all reasonable */
    double measurements[10];
    for (int m = 0; m < 10; m++) {
        uint64_t t0 = mach_absolute_time();
        volatile float x = 1.0f;
        for (int i = 0; i < 10000; i++) x *= 1.000001f;
        uint64_t t1 = mach_absolute_time();
        measurements[m] = ns_elapsed(t0, t1);
    }

    /* All should be positive and within 100x of each other */
    double min_m = measurements[0], max_m = measurements[0];
    for (int i = 1; i < 10; i++) {
        if (measurements[i] < min_m) min_m = measurements[i];
        if (measurements[i] > max_m) max_m = measurements[i];
    }
    ASSERT(min_m > 0, "all measurements should be positive");
    ASSERT(max_m / min_m < 100.0, "measurement variance should be < 100x");
    fprintf(stderr, "  min=%.0f ns, max=%.0f ns, ratio=%.1f\n",
            min_m, max_m, max_m / min_m);
    PASS("Repeated independent measurements are stable");
}

/* ── AMX Alignment Extended Tests ──────────────────────────────────────── */

static void test_amx_alignment_extended(void)
{
    fprintf(stderr, "\n=== AMX Alignment Extended ===\n");

    /* Edge cases */
    ASSERT(ap_amx_align(0) == 0, "align(0) == 0");
    ASSERT(ap_amx_align(1) == 32, "align(1) == 32");
    ASSERT(ap_amx_align(31) == 32, "align(31) == 32");
    ASSERT(ap_amx_align(32) == 32, "align(32) == 32");
    ASSERT(ap_amx_align(33) == 64, "align(33) == 64");
    ASSERT(ap_amx_align(64) == 64, "align(64) == 64");
    ASSERT(ap_amx_align(65) == 96, "align(65) == 96");
    ASSERT(ap_amx_align(1024) == 1024, "align(1024) == 1024");
    ASSERT(ap_amx_align(1025) == 1056, "align(1025) == 1056");
    PASS("AMX alignment extended cases correct");

    /* Alloc with various sizes */
    int sizes[] = {1, 32, 64, 128, 256, 512, 1024};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    for (int i = 0; i < n_sizes; i++) {
        float *p = ap_amx_alloc(sizes[i]);
        ASSERT(p != NULL, "AMX alloc should succeed");
        ASSERT(((uintptr_t)p & 127) == 0, "should be 128-byte aligned");
        /* Write and read back */
        for (int j = 0; j < sizes[i]; j++) p[j] = (float)j;
        float check = p[sizes[i] - 1];
        ASSERT(check == (float)(sizes[i] - 1), "read-back should match");
        free(p);
    }
    PASS("AMX-aligned allocations all correct for various sizes");
}

/* ── IOSurface Extended Tests ──────────────────────────────────────────── */

static void test_zerocopy_sizes(void)
{
    fprintf(stderr, "\n=== IOSurface Zero-Copy Buffer Sizes ===\n");

    /* Test various buffer sizes */
    size_t sizes[] = {4096, 16 * 1024, 64 * 1024, 256 * 1024};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < n_sizes; i++) {
        APZeroCopyBuffer buf = ap_zerocopy_create(sizes[i]);
        ASSERT(buf.cpu_ptr != NULL, "buffer created");
        ASSERT(buf.size >= sizes[i], "buffer size >= requested");

        /* Write pattern and verify */
        unsigned char *p = (unsigned char *)buf.cpu_ptr;
        for (size_t j = 0; j < sizes[i] && j < 256; j++)
            p[j] = (unsigned char)(j & 0xFF);
        for (size_t j = 0; j < 256 && j < sizes[i]; j++)
            ASSERT(p[j] == (unsigned char)(j & 0xFF), "pattern mismatch");

        ap_zerocopy_destroy(&buf);
        ASSERT(buf.cpu_ptr == NULL, "destroyed cleanly");
    }
    PASS("IOSurface buffers work at various sizes");
}

static void test_zerocopy_lifecycle(void)
{
    fprintf(stderr, "\n=== IOSurface Lifecycle ===\n");

    for (int i = 0; i < 20; i++) {
        APZeroCopyBuffer buf = ap_zerocopy_create(4096);
        ASSERT(buf.cpu_ptr != NULL, "buffer created in cycle");
        ap_zerocopy_destroy(&buf);
    }
    PASS("20x IOSurface create/destroy cycles stable");
}

/* ── NEON Edge Cases ───────────────────────────────────────────────────── */

static void test_neon_single_element(void)
{
    fprintf(stderr, "\n=== NEON Single Element ===\n");

    float in = 1.0f, out = 0.0f;

    /* Softmax of single element should be 1.0 */
    ap_neon_softmax(&in, &out, 1);
    ASSERT(fabsf(out - 1.0f) < 1e-5f, "softmax(single) == 1.0");

    /* GELU of 0 should be 0 */
    float zero = 0.0f, gelu_out = 999.0f;
    ap_neon_gelu(&zero, &gelu_out, 1);
    ASSERT(fabsf(gelu_out) < 1e-5f, "gelu(0) == 0");

    /* SiLU of 0 should be 0 */
    float silu_out = 999.0f;
    ap_neon_silu(&zero, &silu_out, 1);
    ASSERT(fabsf(silu_out) < 1e-5f, "silu(0) == 0");

    PASS("NEON single-element edge cases correct");
}

static void test_neon_gelu_properties(void)
{
    fprintf(stderr, "\n=== NEON GELU Properties ===\n");

    /* GELU should be monotonically increasing for large positive inputs */
    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4];
    ap_neon_gelu(in, out, 4);
    for (int i = 1; i < 4; i++)
        ASSERT(out[i] > out[i - 1], "GELU should be increasing for positive x");

    /* GELU(x) ≈ x for large positive x */
    float big_in = 10.0f, big_out = 0.0f;
    ap_neon_gelu(&big_in, &big_out, 1);
    ASSERT(fabsf(big_out - big_in) < 0.1f, "GELU(10) ≈ 10");

    PASS("GELU properties verified");
}

static void test_neon_silu_properties(void)
{
    fprintf(stderr, "\n=== NEON SiLU Properties ===\n");

    /* SiLU should be monotonically increasing for positive inputs */
    float in[4] = {0.5f, 1.0f, 2.0f, 4.0f};
    float out[4];
    ap_neon_silu(in, out, 4);
    for (int i = 1; i < 4; i++)
        ASSERT(out[i] > out[i - 1], "SiLU should be increasing for positive x");

    /* SiLU(x) ≈ x for large positive x */
    float big_in = 10.0f, big_out = 0.0f;
    ap_neon_silu(&big_in, &big_out, 1);
    ASSERT(fabsf(big_out - big_in) < 0.1f, "SiLU(10) ≈ 10");

    /* SiLU is negative for x < 0 but bounded */
    float neg_in = -5.0f, neg_out = 0.0f;
    ap_neon_silu(&neg_in, &neg_out, 1);
    ASSERT(neg_out < 0.0f, "SiLU(-5) < 0");
    ASSERT(neg_out > -1.0f, "SiLU(-5) > -1 (bounded)");

    PASS("SiLU properties verified");
}

static void test_neon_layernorm_constant(void)
{
    fprintf(stderr, "\n=== NEON LayerNorm Constant Input ===\n");

    const int N = 64;
    float in[64], out[64], g[64], b[64];

    /* Constant input: layernorm should produce gamma * 0 + beta = beta
     * because (x - mean) = 0 for all elements */
    for (int i = 0; i < N; i++) {
        in[i] = 5.0f;  /* constant */
        g[i] = 2.0f;
        b[i] = 3.0f;
    }
    ap_neon_layernorm(in, out, g, b, N, 1e-5f);

    /* With constant input, var=0, so (x-mean)/sqrt(var+eps) ≈ 0 */
    /* out[i] ≈ 0 * gamma + beta = beta = 3.0 */
    for (int i = 0; i < N; i++)
        ASSERT(fabsf(out[i] - 3.0f) < 0.1f, "layernorm of constant should be ~beta");

    PASS("LayerNorm with constant input produces beta");
}

static void test_neon_rmsnorm_unit(void)
{
    fprintf(stderr, "\n=== NEON RMSNorm Unit Vector ===\n");

    const int N = 64;
    float in[64], out[64], g[64];

    /* Input: uniform 1/sqrt(N) so RMS = 1/sqrt(N), sqrt(mean(x^2)) = 1/sqrt(N)
     * rmsnorm = x / rms * gamma. For gamma=1, out = x * sqrt(N) */
    float val = 1.0f / sqrtf((float)N);
    for (int i = 0; i < N; i++) {
        in[i] = val;
        g[i] = 1.0f;
    }
    ap_neon_rmsnorm(in, out, g, N, 1e-5f);

    /* out[i] should be ≈ 1.0 (x / rms(x) = 1 when all elements equal) */
    for (int i = 0; i < N; i++)
        ASSERT(fabsf(out[i] - 1.0f) < 0.01f, "rmsnorm of uniform should be ~1.0");

    PASS("RMSNorm with uniform input produces ~1.0");
}

/* ── Model Mmap Tests ──────────────────────────────────────────────────── */

static void test_model_mmap_nonexistent(void)
{
    fprintf(stderr, "\n=== Model Mmap Nonexistent File ===\n");

    APModelMap m = ap_model_mmap("/nonexistent/file.bin");
    ASSERT(m.base == NULL, "mmap of nonexistent file returns NULL base");
    ASSERT(m.size == 0, "mmap of nonexistent file has size 0");
    /* Cleanup should be safe even on failed mmap */
    ap_model_munmap(&m);
    PASS("Model mmap handles nonexistent file safely");
}

static void bench_comparison(void)
{
    fprintf(stderr, "\n=== Throughput Summary (1024 elements, 10K iterations) ===\n");

    const int N = 1024;
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *g = malloc(N * sizeof(float));
    float *beta = malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        g[i] = 1.0f;
        beta[i] = 0.0f;
    }

    const int ITERS = 10000;

    /* Softmax */
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < ITERS; i++) ap_neon_softmax(a, b, N);
    uint64_t t1 = mach_absolute_time();
    fprintf(stderr, "  softmax:   %7.0f ns\n", ns_elapsed(t0, t1) / ITERS);

    /* GELU */
    t0 = mach_absolute_time();
    for (int i = 0; i < ITERS; i++) ap_neon_gelu(a, b, N);
    t1 = mach_absolute_time();
    fprintf(stderr, "  gelu:      %7.0f ns\n", ns_elapsed(t0, t1) / ITERS);

    /* SiLU */
    t0 = mach_absolute_time();
    for (int i = 0; i < ITERS; i++) ap_neon_silu(a, b, N);
    t1 = mach_absolute_time();
    fprintf(stderr, "  silu:      %7.0f ns\n", ns_elapsed(t0, t1) / ITERS);

    /* LayerNorm */
    t0 = mach_absolute_time();
    for (int i = 0; i < ITERS; i++) ap_neon_layernorm(a, b, g, beta, N, 1e-5f);
    t1 = mach_absolute_time();
    fprintf(stderr, "  layernorm: %7.0f ns\n", ns_elapsed(t0, t1) / ITERS);

    /* RMSNorm */
    t0 = mach_absolute_time();
    for (int i = 0; i < ITERS; i++) ap_neon_rmsnorm(a, b, g, N, 1e-5f);
    t1 = mach_absolute_time();
    fprintf(stderr, "  rmsnorm:   %7.0f ns\n", ns_elapsed(t0, t1) / ITERS);

    /* Residual + LayerNorm */
    t0 = mach_absolute_time();
    for (int i = 0; i < ITERS; i++) ap_neon_residual_layernorm(a, a, b, g, beta, N, 1e-5f);
    t1 = mach_absolute_time();
    fprintf(stderr, "  res+ln:    %7.0f ns\n", ns_elapsed(t0, t1) / ITERS);

    free(a); free(b); free(g); free(beta);
}

int main(void)
{
    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║   Apple Silicon Performance Primitives Test  ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════╝\n");

    test_realtime_priority();
    test_neon_softmax();
    test_neon_gelu();
    test_neon_silu();
    test_neon_layernorm();
    test_neon_rmsnorm();
    test_residual_layernorm();
    test_zerocopy_buffer();
    test_amx_alignment();
    test_prefetch();
    test_softmax_sizes();

    /* New tests */
    test_timing_accuracy();
    test_timing_nested();
    test_timing_reset_reuse();
    test_amx_alignment_extended();
    test_zerocopy_sizes();
    test_zerocopy_lifecycle();
    test_neon_single_element();
    test_neon_gelu_properties();
    test_neon_silu_properties();
    test_neon_layernorm_constant();
    test_neon_rmsnorm_unit();
    test_model_mmap_nonexistent();

    bench_comparison();

    fprintf(stderr, "\n══════════════════════════════════════════════\n");
    fprintf(stderr, "Results: %d passed, %d failed\n",
            tests_passed, tests_failed);
    fprintf(stderr, "══════════════════════════════════════════════\n");

    return tests_failed > 0 ? 1 : 0;
}
