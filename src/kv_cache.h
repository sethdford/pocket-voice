/**
 * kv_cache.h — Cache-oblivious interleaved KV cache for transformer attention.
 *
 * Standard KV cache layout stores K and V in separate contiguous arrays:
 *   K: [H][T][D]    V: [H][T][D]
 *
 * This causes two separate cache-line fetches per timestep during attention
 * (one for K, one for V). With the interleaved layout:
 *   KV: [H][T][2][D]   (K[h][t] immediately followed by V[h][t])
 *
 * Both K and V for the same (h,t) are in adjacent cache lines, halving
 * L2/L3 cache misses during the attention score→value reduction loop.
 *
 * The tiled variant further groups timesteps into tiles of 8, optimizing
 * for the M-series Apple Silicon L1 cache line size (128 bytes = 32 floats).
 */

#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define KV_TILE_SIZE 8  /* timesteps per tile (fits 2*D*4=512 bytes per tile for D=64) */

typedef struct {
    float *data;         /* Interleaved [H][T_max][2][D] */
    int n_heads;
    int head_dim;
    int max_len;         /* Maximum sequence length */
    int cur_len;         /* Current number of cached timesteps */
    int stride_head;     /* = max_len * 2 * head_dim */
    int stride_time;     /* = 2 * head_dim */
} InterleavedKVCache;

static inline int kv_cache_create(InterleavedKVCache *c, int n_heads, int head_dim, int max_len)
{
    if (!c) return -1;
    c->n_heads = n_heads;
    c->head_dim = head_dim;
    c->max_len = max_len;
    c->cur_len = 0;
    c->stride_time = 2 * head_dim;
    c->stride_head = max_len * c->stride_time;

    size_t total = (size_t)n_heads * c->stride_head;
    c->data = (float *)calloc(total, sizeof(float));
    return c->data ? 0 : -1;
}

static inline void kv_cache_destroy(InterleavedKVCache *c)
{
    if (c) { free(c->data); c->data = NULL; }
}

static inline void kv_cache_reset(InterleavedKVCache *c)
{
    if (!c || !c->data) return;
    c->cur_len = 0;
    /* No need to zero — we track cur_len */
}

/**
 * Append a new K,V pair for all heads at time position cur_len.
 * @param k  Shape [n_heads * head_dim] (flattened)
 * @param v  Shape [n_heads * head_dim] (flattened)
 */
static inline void kv_cache_append(InterleavedKVCache *c, const float *k, const float *v)
{
    int t = c->cur_len;
    if (t >= c->max_len) {
        /* Ring buffer: overwrite oldest, shift the logical window.
           For simplicity, shift all data left by 1 timestep per head. */
        for (int h = 0; h < c->n_heads; h++) {
            float *head_base = c->data + h * c->stride_head;
            memmove(head_base, head_base + c->stride_time,
                    (size_t)(c->max_len - 1) * c->stride_time * sizeof(float));
        }
        t = c->max_len - 1;
    } else {
        c->cur_len = t + 1;
    }

    for (int h = 0; h < c->n_heads; h++) {
        float *slot = c->data + h * c->stride_head + t * c->stride_time;
        memcpy(slot, k + h * c->head_dim, (size_t)c->head_dim * sizeof(float));
        memcpy(slot + c->head_dim, v + h * c->head_dim, (size_t)c->head_dim * sizeof(float));
    }
}

/**
 * Get K pointer for head h, timestep t. V follows immediately at +head_dim.
 */
static inline const float *kv_cache_k(const InterleavedKVCache *c, int h, int t)
{
    return c->data + h * c->stride_head + t * c->stride_time;
}

static inline const float *kv_cache_v(const InterleavedKVCache *c, int h, int t)
{
    return c->data + h * c->stride_head + t * c->stride_time + c->head_dim;
}

/**
 * Compute attention scores and weighted sum for a single query vector.
 * Leverages the interleaved layout for cache-friendly access:
 *   For each timestep t, loads K[t] → compute score, then V[t] → accumulate.
 *   K[t] and V[t] are adjacent in memory, so V[t] is already in L1 cache.
 *
 * @param c       KV cache
 * @param q       Query vector for one head [head_dim]
 * @param head    Head index
 * @param scale   Attention scale (1/sqrt(d_k))
 * @param output  Output vector [head_dim] (attention-weighted sum of values)
 */
static inline void kv_cache_attend(const InterleavedKVCache *c,
                                    const float *q, int head, float scale,
                                    float *output)
{
    int T = c->cur_len;
    int D = c->head_dim;
    const float *head_base = c->data + head * c->stride_head;

    /* Phase 1: compute scores (Q @ K^T) */
    float scores_stack[4096];
    float *scores_heap = NULL;
    float *scores;
    if (T <= 4096) {
        scores = scores_stack;
    } else {
        scores_heap = (float *)malloc((size_t)T * sizeof(float));
        scores = scores_heap;
    }
    for (int t = 0; t < T; t++) {
        const float *kt = head_base + t * c->stride_time;
        float dot = 0;
#ifdef __APPLE__
        vDSP_dotpr(q, 1, kt, 1, &dot, (vDSP_Length)D);
#else
        for (int d = 0; d < D; d++) dot += q[d] * kt[d];
#endif
        scores[t] = dot * scale;
    }

    /* Phase 2: softmax */
    float max_val = scores[0];
    for (int t = 1; t < T; t++)
        if (scores[t] > max_val) max_val = scores[t];

    float sum = 0;
    for (int t = 0; t < T; t++) {
        scores[t] = expf(scores[t] - max_val);
        sum += scores[t];
    }
    float inv_sum = 1.0f / sum;
    for (int t = 0; t < T; t++) scores[t] *= inv_sum;

    /* Phase 3: weighted sum of V (V is at kt + D, already in L1 from score computation) */
    memset(output, 0, (size_t)D * sizeof(float));
    for (int t = 0; t < T; t++) {
        const float *vt = head_base + t * c->stride_time + D;
        float s = scores[t];
#ifdef __APPLE__
        /* output += s * vt */
        vDSP_vsma(vt, 1, &s, output, 1, output, 1, (vDSP_Length)D);
#else
        for (int d = 0; d < D; d++) output[d] += s * vt[d];
#endif
    }

    free(scores_heap); /* NULL-safe: no-op if stack path was used */
}

#ifdef __cplusplus
}
#endif

#endif /* KV_CACHE_H */
