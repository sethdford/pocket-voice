/**
 * metal_loader.h — Load and dispatch custom Metal compute kernels at runtime.
 *
 * Loads tensor_ops.metallib and provides C-callable wrappers for:
 *   - flash_attention (fused Q·K^T → softmax → V)
 *   - gemm_f16 (half-precision GEMM)
 *   - silu_gate (fused SiLU + gating)
 *   - layer_norm (fused layer normalization)
 *
 * Thread-safe: the library is loaded once and cached.
 */

#ifndef METAL_LOADER_H
#define METAL_LOADER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MetalKernels MetalKernels;

/**
 * Load tensor_ops.metallib from the given path.
 * @param metallib_path  Path to .metallib file
 * @return               Handle or NULL on failure
 */
MetalKernels *metal_kernels_load(const char *metallib_path);

/**
 * Check if custom kernels are available.
 * @return 1 if loaded, 0 otherwise
 */
int metal_kernels_available(const MetalKernels *mk);

/**
 * Get the names of available kernel functions.
 * @param mk     Handle
 * @param names  Output: array of const char* (caller provides array)
 * @param max_n  Maximum names to return
 * @return       Number of names written
 */
int metal_kernels_list(const MetalKernels *mk, const char **names, int max_n);

/**
 * Dispatch a GEMM: C = alpha * A @ B^T,  A=[M,K], B=[N,K], C=[M,N] (all fp16).
 * @return 0 on success, -1 if kernel unavailable or dispatch error.
 */
int metal_gemm_f16(MetalKernels *mk,
                   const void *a_buf, const void *b_buf, void *c_buf,
                   uint32_t M, uint32_t N, uint32_t K, float alpha);

/**
 * Dispatch fused SiLU+gate: out[n,d] = silu(input[n,d]) * input[n,D+d].
 * input=[N,2D], output=[N,D] (fp16).
 */
int metal_silu_gate(MetalKernels *mk,
                    const void *input_buf, void *output_buf,
                    uint32_t N, uint32_t D);

/**
 * Dispatch fused layer norm: out = (x-mean)/sqrt(var+eps) * gamma + beta.
 * input/output=[N,D], gamma/beta=[D] (fp16).
 */
int metal_layer_norm(MetalKernels *mk,
                     const void *input_buf, void *output_buf,
                     const void *gamma_buf, const void *beta_buf,
                     uint32_t N, uint32_t D, float eps);

/**
 * Dispatch flash attention: O = softmax(Q·K^T / sqrt(d)) · V.
 * Q=[M,hd], K=[N,hd], V=[N,hd], O=[M,hd] (fp16).
 */
int metal_flash_attention(MetalKernels *mk,
                          const void *q_buf, const void *k_buf,
                          const void *v_buf, void *o_buf,
                          uint32_t M, uint32_t N, uint32_t head_dim);

/**
 * Destroy and release all Metal resources.
 */
void metal_kernels_destroy(MetalKernels *mk);

#ifdef __cplusplus
}
#endif

#endif /* METAL_LOADER_H */
