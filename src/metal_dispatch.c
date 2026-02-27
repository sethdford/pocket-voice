/**
 * metal_dispatch.c — High-level Metal GPU dispatch with automatic CPU fallback.
 *
 * Singleton wrapper around metal_loader.c that provides:
 *   - One-time init with metal_dispatch_init()
 *   - fp32 GEMM with automatic fp16 conversion for GPU path
 *   - Transparent fallback to cblas_sgemm when Metal is unavailable
 *   - Passthrough for native fp16 operations (silu_gate, layer_norm, etc.)
 */

#include "metal_dispatch.h"
#include "metal_loader.h"
#include "apple_perf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdatomic.h>
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <mach/mach_time.h>

/* Singleton state */
static MetalKernels *g_metal = NULL;
static atomic_int g_initialized = 0;

/* Chip-aware dispatch threshold (min dimension for GPU path) */
static int g_threshold = 0;        /* 0 = not yet configured */
static int g_threshold_set = 0;    /* 1 = user override via set_threshold */
static int g_has_tensorops = -1;   /* -1 = not checked, 0 = no, 1 = yes */

int metal_dispatch_init(const char *metallib_path) {
    int expected = 0;
    if (!atomic_compare_exchange_strong(&g_initialized, &expected, 1))
        return metal_dispatch_available(); /* Another thread already initialized */

    /* Auto-configure threshold if not manually set */
    if (!g_threshold_set) {
        metal_dispatch_auto_threshold();
    }

    if (!metallib_path) {
        fprintf(stderr, "[metal_dispatch] No metallib path provided — CPU only\n");
        return 0;
    }

    g_metal = metal_kernels_load(metallib_path);
    if (metal_kernels_available(g_metal)) {
        const char *names[16];
        int n = metal_kernels_list(g_metal, names, 16);
        fprintf(stderr, "[metal_dispatch] GPU dispatch ready — %d kernel(s), "
                "threshold=%d (%s)\n", n, g_threshold,
                ap_chip_generation_name(ap_chip_generation()));
        return 1;
    }

    fprintf(stderr, "[metal_dispatch] Metal load failed — CPU fallback\n");
    return 0;
}

int metal_dispatch_available(void) {
    return g_metal && metal_kernels_available(g_metal);
}

/* ── fp32 ↔ fp16 conversion using NEON ──────────────────────────────────── */

static void f32_to_f16(const float *src, __fp16 *dst, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        float16x4_t h = vcvt_f16_f32(v);
        vst1_f16(dst + i, h);
    }
    for (; i < n; i++) {
        dst[i] = (__fp16)src[i];
    }
}

static void f16_to_f32(const __fp16 *src, float *dst, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float16x4_t h = vld1_f16(src + i);
        float32x4_t v = vcvt_f32_f16(h);
        vst1q_f32(dst + i, v);
    }
    for (; i < n; i++) {
        dst[i] = (float)src[i];
    }
}

/* ── CPU fallback: cblas_sgemm C = alpha * A @ B^T ──────────────────────── */

static void cpu_gemm(const float *A, const float *B, float *C,
                     int M, int N, int K, float alpha) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, alpha, A, K, B, K, 0.0f, C, N);
}

/* ── Public API ─────────────────────────────────────────────────────────── */

int metal_dispatch_gemm(const float *A, const float *B, float *C,
                        int M, int N, int K) {
    return metal_dispatch_gemm_alpha(A, B, C, M, N, K, 1.0f);
}

int metal_dispatch_gemm_alpha(const float *A, const float *B, float *C,
                              int M, int N, int K, float alpha) {
    if (!metal_dispatch_available()) {
        cpu_gemm(A, B, C, M, N, K, alpha);
        return 0;
    }

    /* GPU path: convert fp32 → fp16, dispatch, convert back */
    size_t a_elems = (size_t)M * K;
    size_t b_elems = (size_t)N * K;
    size_t c_elems = (size_t)M * N;

    /* Chip-aware threshold: use CPU for matrices below the threshold.
     * On M1-M4 the audit proved Metal adds +3228% to +568006% overhead
     * for conformer-sized matrices. On M5+ neural accelerators make
     * small GPU dispatch viable. */
    int min_dim = M < N ? M : N;
    if (K < min_dim) min_dim = K;
    int threshold = g_threshold > 0 ? g_threshold : 1024; /* safe default */
    if (min_dim < threshold) {
        cpu_gemm(A, B, C, M, N, K, alpha);
        return 0;
    }

    /* Legacy small-matrix guard (catches edge cases the threshold misses) */
    if (a_elems + b_elems + c_elems < 4096) {
        cpu_gemm(A, B, C, M, N, K, alpha);
        return 0;
    }

    /* Metal buffers need page-aligned memory for newBufferWithBytesNoCopy */
    size_t a_bytes = a_elems * sizeof(__fp16);
    size_t b_bytes = b_elems * sizeof(__fp16);
    size_t c_bytes = c_elems * sizeof(__fp16);

    /* Round up to page size for Metal shared memory */
    size_t page = 16384; /* Apple Silicon page size */
    size_t a_alloc = (a_bytes + page - 1) & ~(page - 1);
    size_t b_alloc = (b_bytes + page - 1) & ~(page - 1);
    size_t c_alloc = (c_bytes + page - 1) & ~(page - 1);

    void *a_buf = NULL, *b_buf = NULL, *c_buf = NULL;
    posix_memalign(&a_buf, page, a_alloc);
    posix_memalign(&b_buf, page, b_alloc);
    posix_memalign(&c_buf, page, c_alloc);

    if (!a_buf || !b_buf || !c_buf) {
        free(a_buf); free(b_buf); free(c_buf);
        cpu_gemm(A, B, C, M, N, K, alpha);
        return 0;
    }

    /* Zero padding bytes */
    memset(a_buf, 0, a_alloc);
    memset(b_buf, 0, b_alloc);
    memset(c_buf, 0, c_alloc);

    f32_to_f16(A, (__fp16 *)a_buf, (int)a_elems);
    f32_to_f16(B, (__fp16 *)b_buf, (int)b_elems);

    int rc = metal_gemm_f16(g_metal, a_buf, b_buf, c_buf,
                            (uint32_t)M, (uint32_t)N, (uint32_t)K, alpha);

    if (rc == 0) {
        f16_to_f32((__fp16 *)c_buf, C, (int)c_elems);
    } else {
        /* GPU dispatch failed — fall back to CPU */
        cpu_gemm(A, B, C, M, N, K, alpha);
        rc = 0;
    }

    free(a_buf);
    free(b_buf);
    free(c_buf);
    return rc;
}

int metal_dispatch_gemm_f16(const void *A, const void *B, void *C,
                            uint32_t M, uint32_t N, uint32_t K, float alpha) {
    if (!metal_dispatch_available()) return -1;
    return metal_gemm_f16(g_metal, A, B, C, M, N, K, alpha);
}

int metal_dispatch_silu_gate(const void *input, void *output,
                             uint32_t N, uint32_t D) {
    if (!metal_dispatch_available()) return -1;
    return metal_silu_gate(g_metal, input, output, N, D);
}

int metal_dispatch_flash_attention(const void *Q, const void *K,
                                   const void *V, void *O,
                                   uint32_t M, uint32_t N, uint32_t head_dim) {
    if (!metal_dispatch_available()) return -1;
    return metal_flash_attention(g_metal, Q, K, V, O, M, N, head_dim);
}

int metal_dispatch_layer_norm(const void *input, void *output,
                              const void *gamma, const void *beta,
                              uint32_t N, uint32_t D, float eps) {
    if (!metal_dispatch_available()) return -1;
    return metal_layer_norm(g_metal, input, output, gamma, beta, N, D, eps);
}

/* ── Chip-Aware Threshold Configuration ────────────────────────────────── */

void metal_dispatch_set_threshold(int min_dim) {
    g_threshold = min_dim;
    g_threshold_set = 1;
    fprintf(stderr, "[metal_dispatch] Manual threshold override: %d\n", min_dim);
}

int metal_dispatch_get_threshold(void) {
    if (g_threshold == 0 && !g_threshold_set) {
        metal_dispatch_auto_threshold();
    }
    return g_threshold;
}

void metal_dispatch_auto_threshold(void) {
    APChipGeneration gen = ap_chip_generation();
    switch (gen) {
        case AP_CHIP_M5:
            /* M5+ neural accelerators: GPU viable for small matrices */
            g_threshold = 128;
            break;
        case AP_CHIP_M1:
        case AP_CHIP_M2:
        case AP_CHIP_M3:
        case AP_CHIP_M4:
            /* Audit-proven: Metal adds massive overhead for small matrices */
            g_threshold = 1024;
            break;
        default:
            /* Unknown chip — conservative, prefer CPU */
            g_threshold = 2048;
            break;
    }
    fprintf(stderr, "[metal_dispatch] Auto threshold for %s: %d\n",
            ap_chip_generation_name(gen), g_threshold);
}

int metal_dispatch_has_tensorops(void) {
    if (g_has_tensorops >= 0) return g_has_tensorops;

    /* Metal 4 TensorOps require M5+ hardware.
     * Runtime check: Metal 4 GPU family support is only available on M5+.
     * We check the chip generation first as a fast path — the actual
     * MTLDevice.supportsFamily check requires Objective-C and is deferred
     * to metal_loader.c if needed. */
    if (!ap_has_neural_accel()) {
        g_has_tensorops = 0;
        return 0;
    }

    /* On M5+, Metal 4 TensorOps are available if Metal GPU is initialized.
     * The actual MTLGPUFamilyMetal4 check is done in metal_loader at init time.
     * For now, we gate on chip generation — Metal 4 launched with M5. */
    g_has_tensorops = metal_dispatch_available() ? 1 : 0;
    if (g_has_tensorops) {
        fprintf(stderr, "[metal_dispatch] Metal 4 TensorOps available — "
                "using neural accelerator path\n");
    }
    return g_has_tensorops;
}

int metal_dispatch_benchmark(int M, int K, int N) {
    if (!metal_dispatch_available()) return 0; /* CPU only */

    const int WARMUP = 3;
    const int ITERS  = 10;
    size_t a_sz = (size_t)M * K;
    size_t b_sz = (size_t)N * K;
    size_t c_sz = (size_t)M * N;

    float *A = (float *)calloc(a_sz, sizeof(float));
    float *B = (float *)calloc(b_sz, sizeof(float));
    float *C = (float *)calloc(c_sz, sizeof(float));

    if (!A || !B || !C) {
        free(A); free(B); free(C);
        return -1;
    }

    /* Fill with small random-ish values */
    for (size_t i = 0; i < a_sz; i++) A[i] = 0.01f * (float)(i % 97);
    for (size_t i = 0; i < b_sz; i++) B[i] = 0.01f * (float)(i % 89);

    mach_timebase_info_data_t tbi;
    mach_timebase_info(&tbi);

    /* Benchmark CPU */
    for (int i = 0; i < WARMUP; i++) cpu_gemm(A, B, C, M, N, K, 1.0f);
    uint64_t cpu_start = mach_absolute_time();
    for (int i = 0; i < ITERS; i++) cpu_gemm(A, B, C, M, N, K, 1.0f);
    uint64_t cpu_end = mach_absolute_time();
    double cpu_ns = (double)(cpu_end - cpu_start) * tbi.numer / tbi.denom;

    /* Benchmark GPU (use internal Metal path directly, bypass threshold) */
    /* We need to allocate fp16 buffers and do conversions like the real path */
    size_t page = 16384;
    size_t a_bytes = a_sz * sizeof(__fp16);
    size_t b_bytes = b_sz * sizeof(__fp16);
    size_t c_bytes = c_sz * sizeof(__fp16);
    size_t a_alloc = (a_bytes + page - 1) & ~(page - 1);
    size_t b_alloc = (b_bytes + page - 1) & ~(page - 1);
    size_t c_alloc = (c_bytes + page - 1) & ~(page - 1);

    void *a16 = NULL, *b16 = NULL, *c16 = NULL;
    posix_memalign(&a16, page, a_alloc);
    posix_memalign(&b16, page, b_alloc);
    posix_memalign(&c16, page, c_alloc);

    if (!a16 || !b16 || !c16) {
        free(A); free(B); free(C);
        free(a16); free(b16); free(c16);
        return -1;
    }

    memset(a16, 0, a_alloc);
    memset(b16, 0, b_alloc);
    memset(c16, 0, c_alloc);
    f32_to_f16(A, (__fp16 *)a16, (int)a_sz);
    f32_to_f16(B, (__fp16 *)b16, (int)b_sz);

    for (int i = 0; i < WARMUP; i++)
        metal_gemm_f16(g_metal, a16, b16, c16, (uint32_t)M, (uint32_t)N, (uint32_t)K, 1.0f);
    uint64_t gpu_start = mach_absolute_time();
    for (int i = 0; i < ITERS; i++)
        metal_gemm_f16(g_metal, a16, b16, c16, (uint32_t)M, (uint32_t)N, (uint32_t)K, 1.0f);
    uint64_t gpu_end = mach_absolute_time();
    double gpu_ns = (double)(gpu_end - gpu_start) * tbi.numer / tbi.denom;

    free(A); free(B); free(C);
    free(a16); free(b16); free(c16);

    double cpu_avg = cpu_ns / ITERS;
    double gpu_avg = gpu_ns / ITERS;

    fprintf(stderr, "[metal_dispatch] Benchmark %dx%dx%d: CPU=%.1fus GPU=%.1fus → %s\n",
            M, K, N, cpu_avg / 1000.0, gpu_avg / 1000.0,
            cpu_avg <= gpu_avg ? "CPU wins" : "GPU wins");

    return (gpu_avg < cpu_avg) ? 1 : 0;
}

void metal_dispatch_cleanup(void) {
    if (g_metal) {
        metal_kernels_destroy(g_metal);
        g_metal = NULL;
    }
    atomic_store(&g_initialized, 0);
    g_has_tensorops = -1;
}
