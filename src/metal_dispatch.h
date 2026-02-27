/**
 * metal_dispatch.h — High-level Metal GPU dispatch with automatic CPU fallback.
 *
 * Wraps metal_loader.h with a singleton pattern, fp32<->fp16 conversion,
 * and transparent fallback to cblas_sgemm when Metal is unavailable.
 *
 * Usage:
 *   metal_dispatch_init("build/tensor_ops.metallib");
 *   if (metal_dispatch_available()) { ... }
 *   metal_dispatch_gemm(A, B, C, M, N, K);  // GPU or CPU automatically
 *   metal_dispatch_cleanup();
 */

#ifndef METAL_DISPATCH_H
#define METAL_DISPATCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize Metal dispatch by loading the metallib and checking GPU.
 * Safe to call multiple times — subsequent calls are no-ops.
 * @param metallib_path  Path to tensor_ops.metallib
 * @return 1 if Metal GPU is available, 0 if falling back to CPU
 */
int metal_dispatch_init(const char *metallib_path);

/**
 * Check if Metal GPU dispatch is available.
 * @return 1 if Metal is initialized and kernels are loaded, 0 otherwise
 */
int metal_dispatch_available(void);

/**
 * GEMM: C = A @ B^T (fp32 interface with automatic fp16 conversion for GPU).
 * A[M,K], B[N,K], C[M,N]. Falls back to cblas_sgemm if Metal unavailable.
 * @return 0 on success
 */
int metal_dispatch_gemm(const float *A, const float *B, float *C,
                        int M, int N, int K);

/**
 * GEMM with alpha scaling: C = alpha * A @ B^T.
 * @return 0 on success
 */
int metal_dispatch_gemm_alpha(const float *A, const float *B, float *C,
                              int M, int N, int K, float alpha);

/**
 * Native fp16 GEMM (zero-copy GPU path when data is already fp16).
 * C = alpha * A @ B^T.  A[M,K], B[N,K], C[M,N] all fp16.
 * @return 0 on success, -1 on error (no CPU fallback for fp16)
 */
int metal_dispatch_gemm_f16(const void *A, const void *B, void *C,
                            uint32_t M, uint32_t N, uint32_t K, float alpha);

/**
 * Fused SiLU+gate: out[n,d] = silu(in[n,d]) * in[n,D+d].
 * input[N,2D] fp16, output[N,D] fp16.
 * @return 0 on success, -1 if Metal unavailable
 */
int metal_dispatch_silu_gate(const void *input, void *output,
                             uint32_t N, uint32_t D);

/**
 * Flash attention: O = softmax(Q*K^T / sqrt(hd)) * V.
 * Q[M,hd], K[N,hd], V[N,hd], O[M,hd] all fp16.
 * @return 0 on success, -1 if Metal unavailable
 */
int metal_dispatch_flash_attention(const void *Q, const void *K,
                                   const void *V, void *O,
                                   uint32_t M, uint32_t N, uint32_t head_dim);

/**
 * Fused layer norm: out = (x - mean) / sqrt(var + eps) * gamma + beta.
 * input/output[N,D], gamma/beta[D] all fp16.
 * @return 0 on success, -1 if Metal unavailable
 */
int metal_dispatch_layer_norm(const void *input, void *output,
                              const void *gamma, const void *beta,
                              uint32_t N, uint32_t D, float eps);

/**
 * Release all Metal resources. Safe to call if not initialized.
 */
void metal_dispatch_cleanup(void);

/* ═══════════════════════════════════════════════════════════════════════════
 * Chip-Aware Dispatch Thresholds
 *
 * Metal dispatch overhead varies dramatically by chip generation:
 *   M1-M4: +3228% to +568006% overhead for conformer-sized matrices
 *   M5+:   Neural accelerators in GPU cores make small dispatches viable
 *
 * Default thresholds (minimum dimension for GPU path):
 *   M1-M4:  1024  (only large matrices benefit from GPU)
 *   M5+:    128   (neural accelerators handle small matrices well)
 *   Unknown: 2048  (conservative — prefer CPU)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Set the minimum matrix dimension for Metal GPU dispatch.
 * Matrices where min(M,N,K) < threshold use CPU (cblas_sgemm).
 * Set to 0 to always use GPU, INT_MAX to always use CPU.
 * @param min_dim  Minimum dimension threshold
 */
void metal_dispatch_set_threshold(int min_dim);

/**
 * Get the current Metal dispatch threshold.
 * @return Current minimum dimension for GPU dispatch
 */
int metal_dispatch_get_threshold(void);

/**
 * Auto-configure the dispatch threshold based on detected chip generation.
 * Called automatically by metal_dispatch_init(), but can be called manually
 * to re-detect (e.g. after testing override via metal_dispatch_set_threshold).
 */
void metal_dispatch_auto_threshold(void);

/**
 * Check if Metal 4 TensorOps (neural accelerator path) is available.
 * Requires M5+ hardware with Metal 4 GPU family support.
 * @return 1 if Metal 4 TensorOps available, 0 otherwise
 */
int metal_dispatch_has_tensorops(void);

/**
 * Benchmark CPU vs GPU for a given matrix size on this hardware.
 * Runs multiple iterations and returns the faster path.
 * @param M  Rows of A / rows of C
 * @param K  Cols of A / cols of B
 * @param N  Rows of B / cols of C
 * @return 0 if CPU is faster, 1 if GPU is faster, -1 on error
 */
int metal_dispatch_benchmark(int M, int K, int N);

#ifdef __cplusplus
}
#endif

#endif /* METAL_DISPATCH_H */
