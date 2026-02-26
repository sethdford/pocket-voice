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
    bench_comparison();

    fprintf(stderr, "\n══════════════════════════════════════════════\n");
    fprintf(stderr, "Results: %d passed, %d failed\n",
            tests_passed, tests_failed);
    fprintf(stderr, "══════════════════════════════════════════════\n");

    return tests_failed > 0 ? 1 : 0;
}
