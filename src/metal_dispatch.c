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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

/* Singleton state */
static MetalKernels *g_metal = NULL;
static int g_initialized = 0;

int metal_dispatch_init(const char *metallib_path) {
    if (g_initialized) return metal_dispatch_available();
    g_initialized = 1;

    if (!metallib_path) {
        fprintf(stderr, "[metal_dispatch] No metallib path provided — CPU only\n");
        return 0;
    }

    g_metal = metal_kernels_load(metallib_path);
    if (metal_kernels_available(g_metal)) {
        const char *names[16];
        int n = metal_kernels_list(g_metal, names, 16);
        fprintf(stderr, "[metal_dispatch] GPU dispatch ready — %d kernel(s)\n", n);
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

    /* For small matrices, CPU is faster due to conversion overhead */
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

void metal_dispatch_cleanup(void) {
    if (g_metal) {
        metal_kernels_destroy(g_metal);
        g_metal = NULL;
    }
    g_initialized = 0;
}
