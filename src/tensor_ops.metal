/*
 * tensor_ops.metal — GPU compute kernels for Apple Silicon M-series.
 *
 * Kernels:
 *   1. gemm_f16:     Half-precision GEMM C = alpha * A @ B^T
 *   2. flash_attention: Fused attention with online softmax
 *   3. silu_gate:    Fused SiLU + gating (GLU-style)
 *   4. layer_norm:   Fused layer normalization
 */

#include <metal_stdlib>
using namespace metal;

// ─── 1. gemm_f16: Half-precision GEMM C = alpha * A @ B^T ─────────────────
//
// A[M,K], B[N,K], C[M,N]. Uses 32x32 threadgroup tiling for cache efficiency.
// Buffers: A@0, B@1, C@2, M@3, N@4, K@5, alpha@6
// M,N,K are uint32_t; alpha is float.
kernel void gemm_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    device const uint* M [[buffer(3)]],
    device const uint* N [[buffer(4)]],
    device const uint* K [[buffer(5)]],
    device const float* alpha_ptr [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint m = *M;
    uint n = *N;
    uint k = *K;
    float alpha = *alpha_ptr;

    // Bounds check: each thread produces one output element
    if (gid.x >= n || gid.y >= m) return;

    uint row = gid.y;
    uint col = gid.x;

    // Accumulate in fp32 for precision
    float acc = 0.0f;

    // Tile size 32 for threadgroup memory
    constexpr uint TILE = 32;
    threadgroup half As[TILE][TILE];
    threadgroup half Bs[TILE][TILE];

    // Iterate over K in tiles of 32
    for (uint ko = 0; ko < k; ko += TILE) {
        // Cooperative load: each thread loads one element into shared memory
        // A tile: rows [row_base : row_base+32], cols [ko : ko+32]
        uint row_base = (tgid.y * TILE);
        uint col_base = (tgid.x * TILE);

        // Load A tile: A[row, ko:ko+32] for our row
        uint a_row = row_base + tid.y;
        uint a_col = ko + tid.x;
        if (a_row < m && a_col < k) {
            As[tid.y][tid.x] = A[a_row * k + a_col];
        } else {
            As[tid.y][tid.x] = 0.0h;
        }

        // Load B tile: B[col_base:col_base+32, ko:ko+32]
        // B is [N,K], C[row,col] = A[row,:]·B[col,:]. Store Bs[j][k] = B[col_base+j, ko+k]
        uint b_row = col_base + tid.x;
        uint b_col = ko + tid.y;
        if (b_row < n && b_col < k) {
            Bs[tid.x][tid.y] = B[b_row * k + b_col];
        } else {
            Bs[tid.x][tid.y] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product for this tile
        uint k_tile_end = min(ko + TILE, k);
        for (uint ki = 0; ki < k_tile_end - ko; ki++) {
            acc += float(As[tid.y][ki]) * float(Bs[tid.x][ki]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    C[row * n + col] = half(alpha * acc);
}

// ─── 2. flash_attention: Fused attention with online softmax ────────────────
//
// O = softmax(Q·K^T / scale) · V
// Q[M,hd], K[N,hd], V[N,hd], O[M,hd] all fp16.
// Buffers: Q@0, K@1, V@2, O@3, M@4, N@5, scale@6, hd@7
// Dispatch: grid (hd, M), threadgroup (1,1) or (min(256,hd), 1)
kernel void flash_attention(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device const uint* M_ptr [[buffer(4)]],
    device const uint* N_ptr [[buffer(5)]],
    device const float* scale_ptr [[buffer(6)]],
    device const uint* hd_ptr [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint M = *M_ptr;
    uint N = *N_ptr;
    uint hd = *hd_ptr;
    float scale = *scale_ptr;

    uint m = gid.y;
    uint d = gid.x;

    if (m >= M || d >= hd) return;

    // Online softmax for numerical stability (single-thread per output element)
    float m_max = -1e30f;
    float l_sum = 0.0f;
    float o_acc = 0.0f;

    for (uint n = 0; n < N; n++) {
        float score = 0.0f;
        for (uint k = 0; k < hd; k++) {
            score += float(Q[m * hd + k]) * float(K[n * hd + k]);
        }
        score *= scale;

        float m_new = max(m_max, score);
        float alpha = exp(m_max - m_new);
        float beta = exp(score - m_new);

        l_sum = l_sum * alpha + beta;
        m_max = m_new;

        o_acc = o_acc * alpha + beta * float(V[n * hd + d]);
    }

    if (l_sum > 1e-12f) {
        O[m * hd + d] = half(o_acc / l_sum);
    } else {
        O[m * hd + d] = 0.0h;
    }
}

// ─── 3. silu_gate: Fused SiLU + gating (GLU-style) ─────────────────────────
//
// out[n,d] = silu(input[n,d]) * input[n,D+d]
// input[N,2D] fp16, output[N,D] fp16. SiLU(x) = x * sigmoid(x)
// Buffers: input@0, output@1, N@2, D@3
kernel void silu_gate(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const uint* N_ptr [[buffer(2)]],
    device const uint* D_ptr [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint N = *N_ptr;
    uint D = *D_ptr;

    if (gid >= N * D) return;

    uint n = gid / D;
    uint d = gid % D;

    float x = float(input[n * (2 * D) + d]);
    float gate = float(input[n * (2 * D) + D + d]);

    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float silu = x / (1.0f + exp(-x));
    output[gid] = half(silu * gate);
}

// ─── 4. layer_norm: Fused layer normalization ───────────────────────────────
//
// out = (x - mean) / sqrt(var + eps) * gamma + beta
// input/output[N,D] fp16, gamma/beta[D] fp16
// Buffers: input@0, output@1, gamma@2, beta@3, N@4, D@5, eps@6
// Dispatch: grid (ceildiv(D,256), N), threadgroup (256, 1)
kernel void layer_norm(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const half* gamma [[buffer(2)]],
    device const half* beta [[buffer(3)]],
    device const uint* N_ptr [[buffer(4)]],
    device const uint* D_ptr [[buffer(5)]],
    device const float* eps_ptr [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    uint N = *N_ptr;
    uint D = *D_ptr;
    float eps = *eps_ptr;

    if (gid.y >= N) return;

    uint n = gid.y;

    constexpr uint TG = 256;
    uint lane = tid.x;
    threadgroup float reduce_buf[TG + 1];  // [0..255] for reduction, [256] for mean

    // Phase 1: Sum reduction for mean
    float sum = 0.0f;
    for (uint d = lane; d < D; d += TG) {
        sum += float(input[n * D + d]);
    }
    reduce_buf[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = TG / 2; s > 0; s >>= 1) {
        if (lane < s) {
            reduce_buf[lane] += reduce_buf[lane + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        reduce_buf[TG] = reduce_buf[0] / float(D);  // mean
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = reduce_buf[TG];

    // Phase 2: Variance reduction
    float var_sum = 0.0f;
    for (uint d = lane; d < D; d += TG) {
        float diff = float(input[n * D + d]) - mean;
        var_sum += diff * diff;
    }
    reduce_buf[lane] = var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = TG / 2; s > 0; s >>= 1) {
        if (lane < s) {
            reduce_buf[lane] += reduce_buf[lane + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = 1.0f;
    if (lane == 0) {
        float var = reduce_buf[0] / float(D);
        inv_std = 1.0f / sqrt(var + eps);
        reduce_buf[TG] = inv_std;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    inv_std = reduce_buf[TG];

    // Phase 3: Normalize and write
    for (uint d = lane; d < D; d += TG) {
        float x = float(input[n * D + d]);
        float norm = (x - mean) * inv_std;
        output[n * D + d] = half(norm * float(gamma[d]) + float(beta[d]));
    }
}
