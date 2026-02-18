/**
 * metal4_ml.h — Metal 4 ML Encoder interface for pocket-tts (macOS 26+).
 *
 * Metal 4 (WWDC25) introduces native tensor types and inline ML inference
 * operations directly in Metal Shading Language. This allows embedding the
 * entire flow network computation inside a Metal command buffer, eliminating
 * ALL framework overhead between GPU operations.
 *
 * Architecture:
 *   Instead of dispatching separate kernels for each matmul/norm/activation,
 *   Metal 4's MLTensor and MLEncoder let us express the full flow network
 *   as a single Metal encoder — the GPU processes the entire multi-layer
 *   network without returning to the CPU.
 *
 * Requirements:
 *   - macOS 26 / iOS 26 or later
 *   - Apple Silicon M1+ (MTLGPUFamily.metal4 support)
 *   - Xcode 26 with Metal 4 SDK
 *
 * This file defines the C interface. The implementation will be in
 * metal4_ml.metal using Metal 4's tensor types.
 *
 * Reference: developer.apple.com/videos/play/wwdc2025/262/
 */

#pragma once

#ifdef __APPLE__
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------
 * Metal 4 Flow Network Context
 * ----------------------------------------------------------------------- */

typedef struct Metal4FlowNet Metal4FlowNet;

/**
 * Create a Metal 4 flow network context.
 *
 * Loads weights and creates the Metal 4 ML pipeline. The pipeline
 * encodes the entire flow network (4 Euler steps × MLP forward) into
 * a single Metal command buffer.
 *
 * @param weights_data    Packed float32 weight buffer (same layout as amx_flow_fused.c)
 * @param n_floats        Number of floats in weight buffer
 * @param hidden_dim      Flow network hidden dimension (typically 512)
 * @param cond_dim        Conditioning dimension (typically 1024)
 * @param output_dim      Output dimension (typically 512)
 * @param n_layers        Number of MLP layers (typically 3)
 * @param n_euler_steps   Number of Euler integration steps (typically 4)
 * @return                Context pointer, or NULL if Metal 4 unavailable
 */
Metal4FlowNet *metal4_flow_create(
    const float *weights_data, uint64_t n_floats,
    int hidden_dim, int cond_dim, int output_dim,
    int n_layers, int n_euler_steps);

/**
 * Run LSD decode using Metal 4 ML Encoder.
 *
 * Encodes the full LSD decode loop (4 Euler steps) as a single Metal
 * command buffer submission. The GPU processes all steps without
 * returning to CPU.
 *
 * Equivalent to amx_flow_fused.c::lsd_decode_fused() but running on GPU
 * via Metal 4's native tensor operations instead of CPU AMX.
 *
 * @param ctx             Metal 4 context
 * @param noise           (output_dim,) noise vector for flow matching
 * @param conditioning    (cond_dim,) conditioning vector from transformer
 * @param output          (output_dim,) decoded output (caller-allocated)
 * @return                0 on success, -1 on failure
 */
int metal4_flow_decode(
    Metal4FlowNet *ctx,
    const float *noise,
    const float *conditioning,
    float *output);

/**
 * Check if Metal 4 ML Encoder is available.
 *
 * @return  1 if available (macOS 26+, M1+), 0 otherwise
 */
int metal4_is_available(void);

/**
 * Destroy Metal 4 flow network context.
 */
void metal4_flow_destroy(Metal4FlowNet *ctx);

/* -----------------------------------------------------------------------
 * Metal 4 Fused Attention (Future)
 *
 * When available, Metal 4's MLEncoder can also fuse the full attention
 * layer (LN → QKV → RoPE → SDPA → out_proj → residual) into a single
 * encoder pass, further reducing kernel dispatch overhead.
 * ----------------------------------------------------------------------- */

typedef struct Metal4Attention Metal4Attention;

Metal4Attention *metal4_attention_create(
    int embed_dim, int num_heads, int head_dim,
    float rope_base, int max_seq_len);

int metal4_attention_forward(
    Metal4Attention *ctx,
    const float *x,              /* (1, 1, E) input */
    const float *k_cache,        /* (1, H, T, D) */
    const float *v_cache,        /* (1, H, T, D) */
    int cache_len,
    int offset,
    float *output,               /* (1, 1, E) */
    float *new_k_cache,          /* (1, H, T+1, D) */
    float *new_v_cache);         /* (1, H, T+1, D) */

void metal4_attention_destroy(Metal4Attention *ctx);

#ifdef __cplusplus
}
#endif

#endif /* __APPLE__ */
