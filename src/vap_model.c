/**
 * vap_model.c — Voice Activity Projection (VAP) inference engine.
 *
 * Transformer-based turn-taking predictor. Input: user + system mel features [160].
 * Output: 4 sigmoid heads (user_speaking, system_turn, backchannel, eou).
 *
 * All matrix ops via cblas_sgemm/cblas_sgemv (AMX-accelerated).
 * KV cache for streaming causal attention. Zero allocations in vap_feed.
 *
 * Weight file: .vap (see train/sonata/train_vap.py export)
 */

#include "vap_model.h"

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════════════ */

#define VAP_MAGIC      0x00504156u   /* "VAP\0" little-endian */
#define VAP_VERSION   1
#define VAP_INPUT_DIM  160
#define VAP_MEL_DIM    80
#define VAP_CONTEXT_LEN 250   /* 5s at 50Hz */

/* ═══════════════════════════════════════════════════════════════════════════
 * Binary header
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t d_model;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t ff_dim;
    uint64_t n_weights;
} VAPHeader;

/* ═══════════════════════════════════════════════════════════════════════════
 * Layer weights
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const float *attn_norm_w;
    const float *wq_w, *wk_w, *wv_w, *wo_w;
    const float *ffn_norm_w;
    const float *ffn_up_w, *ffn_up_b;
    const float *ffn_down_w, *ffn_down_b;
} VAPLayerWeights;

/* ═══════════════════════════════════════════════════════════════════════════
 * Engine state
 * ═══════════════════════════════════════════════════════════════════════════ */

struct VAPModel {
    int d_model;
    int n_layers;
    int n_heads;
    int head_dim;
    int ff_dim;
    int context_len;
    int pos;        /* next write slot (0..context_len-1) */
    int n_valid;    /* number of valid cache positions (1..context_len) */

    const float *input_proj_w;
    const float *input_proj_b;
    VAPLayerWeights *layers;
    const float *final_norm_w;
    const float *head_user_w, *head_user_b;
    const float *head_system_w, *head_system_b;
    const float *head_backchannel_w, *head_backchannel_b;
    const float *head_eou_w, *head_eou_b;

    float *kv_cache;
    float *buf_x;
    float *buf_norm;
    float *buf_qkv;
    float *buf_attn;
    float *buf_ff;
    float *silu_ws;

    float alpha;
    VAPPrediction smoothed;

    int owns_weights;
    void *mmap_base;
    size_t mmap_size;
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Math helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static void rms_norm(float *out, const float *x, const float *w, int n, float eps) {
    float sum_sq;
    vDSP_svesq(x, 1, &sum_sq, n);
    float rms = 1.0f / sqrtf(sum_sq / (float)n + eps);
    vDSP_vsmul(x, 1, &rms, out, 1, n);
    vDSP_vmul(out, 1, w, 1, out, 1, n);
}

static void linear_vec(float *out, const float *x, const float *W, const float *b,
                       int out_dim, int in_dim) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, out_dim, in_dim,
                1.0f, W, in_dim, x, 1, 0.0f, out, 1);
    if (b)
        vDSP_vadd(out, 1, b, 1, out, 1, out_dim);
}

static void silu_inplace(float *x, int n, float *workspace) {
    vDSP_vneg(x, 1, workspace, 1, n);
    int ni = n;
    vvexpf(workspace, workspace, &ni);
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    int i = 0;
    float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t xi = vld1q_f32(x + i);
        float32x4_t ei = vld1q_f32(workspace + i);
        vst1q_f32(x + i, vdivq_f32(xi, vaddq_f32(one, ei)));
    }
    for (; i < n; i++) x[i] = x[i] / (1.0f + workspace[i]);
#else
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + workspace[i]);
#endif
}

static void softmax_row(float *x, int len) {
    if (len <= 0) return;
    float mx = x[0];
    for (int i = 1; i < len; i++) if (x[i] > mx) mx = x[i];
    float sum = 0;
    for (int i = 0; i < len; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    vDSP_vsmul(x, 1, &inv, x, 1, len);
}

static inline float sigmoidf(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/* Sinusoidal positional encoding for position pos, add to x (d_model elements) */
static void add_pos_enc(float *x, int pos, int d_model) {
    float div = 10000.0f;
    for (int i = 0; i < d_model; i += 2) {
        float freq = 1.0f / powf(div, (float)i / (float)d_model);
        float angle = (float)pos * freq;
        x[i] += cosf(angle);
        if (i + 1 < d_model)
            x[i + 1] += sinf(angle);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Single-step causal attention with KV cache
 * ═══════════════════════════════════════════════════════════════════════════ */

static void attn_step(float *out, const float *x, const VAPLayerWeights *lw,
                      float *kv_cache_k, float *kv_cache_v,
                      int cache_len, int pos, int n_ctx_in, int n_valid,
                      float *buf_norm, float *buf_qkv, float *buf_attn,
                      int D, int n_heads, int head_dim) {
    rms_norm(buf_norm, x, lw->attn_norm_w, D, 1e-5f);

    float *q = buf_qkv;
    float *k = buf_qkv + n_heads * head_dim;
    float *v = buf_qkv + n_heads * head_dim * 2;

    linear_vec(q, buf_norm, lw->wq_w, NULL, n_heads * head_dim, D);
    linear_vec(k, buf_norm, lw->wk_w, NULL, n_heads * head_dim, D);
    linear_vec(v, buf_norm, lw->wv_w, NULL, n_heads * head_dim, D);

    /* Store K, V in cache at pos */
    size_t kv_sz = (size_t)n_heads * head_dim * sizeof(float);
    memcpy(kv_cache_k + pos * n_heads * head_dim, k, kv_sz);
    memcpy(kv_cache_v + pos * n_heads * head_dim, v, kv_sz);

    if (n_ctx_in <= 0 || n_ctx_in > VAP_CONTEXT_LEN) {
        memset(out, 0, D * sizeof(float));
    } else {
        /* Ring buffer: before wrap use slots 0..n_ctx-1. After wrap (n_valid>=cache_len),
         * start_slot = (pos+1)%cache_len gives oldest; pos holds newest. */
        int start_slot = (n_ctx_in == cache_len && n_valid >= cache_len)
                        ? (pos + 1) % cache_len : 0;

        float scale = 1.0f / sqrtf((float)head_dim);
        memset(out, 0, D * sizeof(float));

        for (int h = 0; h < n_heads; h++) {
            float *scores = buf_attn;
            for (int ti = 0; ti < n_ctx_in; ti++) {
                int slot = (start_slot + ti) % cache_len;
                float dot = 0;
                const float *kr = kv_cache_k + slot * n_heads * head_dim + h * head_dim;
                vDSP_dotpr(q + h * head_dim, 1, kr, 1, &dot, head_dim);
                scores[ti] = dot * scale;
            }
            softmax_row(scores, n_ctx_in);
            for (int ti = 0; ti < n_ctx_in; ti++) {
                int slot = (start_slot + ti) % cache_len;
                float w = scores[ti];
                const float *vr = kv_cache_v + slot * n_heads * head_dim + h * head_dim;
                for (int d = 0; d < head_dim; d++)
                    out[h * head_dim + d] += w * vr[d];
            }
        }
    }
    linear_vec(buf_norm, out, lw->wo_w, NULL, D, n_heads * head_dim);
    memcpy(out, buf_norm, (size_t)D * sizeof(float));
}

static void ffn_step(float *out, const float *x, const VAPLayerWeights *lw,
                     float *buf_norm, float *buf_ff, float *silu_ws,
                     int D, int ff) {
    rms_norm(buf_norm, x, lw->ffn_norm_w, D, 1e-5f);
    linear_vec(buf_ff, buf_norm, lw->ffn_up_w, lw->ffn_up_b, ff, D);
    silu_inplace(buf_ff, ff, silu_ws);
    linear_vec(out, buf_ff, lw->ffn_down_w, lw->ffn_down_b, D, ff);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Weight loading (advance pointer)
 * ═══════════════════════════════════════════════════════════════════════════ */

static const float *advance(const float **ptr, int count) {
    const float *p = *ptr;
    *ptr += count;
    return p;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * vap_create (from file)
 * ═══════════════════════════════════════════════════════════════════════════ */

VAPModel *vap_create(const char *weights_path) {
    if (!weights_path) return NULL;

    int fd = open(weights_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[vap] Cannot open %s\n", weights_path);
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return NULL;
    }
    size_t file_size = st.st_size;
    void *mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "[vap] mmap failed\n");
        return NULL;
    }

    const char *hdr_bytes = (const char *)mapped;
    uint32_t magic = *(const uint32_t *)hdr_bytes;
    if (magic != VAP_MAGIC) {
        fprintf(stderr, "[vap] Invalid magic 0x%08x (expected VAP)\n", magic);
        munmap(mapped, file_size);
        return NULL;
    }
    uint32_t version = *(const uint32_t *)(hdr_bytes + 4);
    if (version != VAP_VERSION) {
        fprintf(stderr, "[vap] Unsupported version %u\n", version);
        munmap(mapped, file_size);
        return NULL;
    }

    int d_model = (int)*(const uint32_t *)(hdr_bytes + 8);
    int n_layers = (int)*(const uint32_t *)(hdr_bytes + 12);
    int n_heads = (int)*(const uint32_t *)(hdr_bytes + 16);
    int ff_dim = (int)*(const uint32_t *)(hdr_bytes + 20);

    if (d_model <= 0 || n_layers <= 0 || n_heads <= 0 || ff_dim <= 0 ||
        d_model % n_heads != 0) {
        fprintf(stderr, "[vap] Invalid config d=%d L=%d H=%d ff=%d\n",
                d_model, n_layers, n_heads, ff_dim);
        munmap(mapped, file_size);
        return NULL;
    }
    int head_dim = d_model / n_heads;

    const float *ptr = (const float *)(hdr_bytes + 32);

    VAPModel *vap = calloc(1, sizeof(VAPModel));
    if (!vap) {
        munmap(mapped, file_size);
        return NULL;
    }
    vap->d_model = d_model;
    vap->n_layers = n_layers;
    vap->n_heads = n_heads;
    vap->head_dim = head_dim;
    vap->ff_dim = ff_dim;
    vap->context_len = VAP_CONTEXT_LEN;
    vap->pos = 0;
    vap->n_valid = 0;
    vap->alpha = 0.3f;
    vap->smoothed = (VAPPrediction){0.0f, 0.0f, 0.0f, 0.0f};
    vap->owns_weights = 0;
    vap->mmap_base = mapped;
    vap->mmap_size = file_size;

    vap->input_proj_w = advance(&ptr, d_model * VAP_INPUT_DIM);
    vap->input_proj_b = advance(&ptr, d_model);

    vap->layers = calloc((size_t)n_layers, sizeof(VAPLayerWeights));
    if (!vap->layers) {
        free(vap);
        munmap(mapped, file_size);
        return NULL;
    }
    for (int l = 0; l < n_layers; l++) {
        VAPLayerWeights *lw = &vap->layers[l];
        lw->attn_norm_w = advance(&ptr, d_model);
        lw->wq_w = advance(&ptr, n_heads * head_dim * d_model);
        lw->wk_w = advance(&ptr, n_heads * head_dim * d_model);
        lw->wv_w = advance(&ptr, n_heads * head_dim * d_model);
        lw->wo_w = advance(&ptr, d_model * n_heads * head_dim);
        lw->ffn_norm_w = advance(&ptr, d_model);
        lw->ffn_up_w = advance(&ptr, ff_dim * d_model);
        lw->ffn_up_b = advance(&ptr, ff_dim);
        lw->ffn_down_w = advance(&ptr, d_model * ff_dim);
        lw->ffn_down_b = advance(&ptr, d_model);
    }
    vap->final_norm_w = advance(&ptr, d_model);
    vap->head_user_w = advance(&ptr, d_model);
    vap->head_user_b = advance(&ptr, 1);
    vap->head_system_w = advance(&ptr, d_model);
    vap->head_system_b = advance(&ptr, 1);
    vap->head_backchannel_w = advance(&ptr, d_model);
    vap->head_backchannel_b = advance(&ptr, 1);
    vap->head_eou_w = advance(&ptr, d_model);
    vap->head_eou_b = advance(&ptr, 1);

    size_t expected = (size_t)((const char *)ptr - (const char *)mapped);
    if (expected > file_size) {
        fprintf(stderr, "[vap] Weight file too small: need %zu, got %zu\n",
                expected, file_size);
        free(vap->layers);
        free(vap);
        munmap(mapped, file_size);
        return NULL;
    }

    /* Allocate KV cache and working buffers */
    size_t kv_sz = (size_t)2 * n_layers * VAP_CONTEXT_LEN * n_heads * head_dim * sizeof(float);
    vap->kv_cache = calloc(1, kv_sz);
    if (!vap->kv_cache) {
        free(vap->layers);
        free(vap);
        munmap(mapped, file_size);
        return NULL;
    }

    size_t buf_x_sz = (size_t)d_model * sizeof(float);
    size_t buf_norm_sz = buf_x_sz;
    size_t buf_qkv_sz = (size_t)3 * n_heads * head_dim * sizeof(float);
    size_t buf_attn_sz = (size_t)VAP_CONTEXT_LEN * sizeof(float);
    size_t buf_ff_sz = (size_t)ff_dim * sizeof(float);
    size_t silu_sz = (size_t)ff_dim * sizeof(float);

    vap->buf_x = malloc(buf_x_sz + buf_norm_sz + buf_qkv_sz + buf_attn_sz +
                        buf_ff_sz + silu_sz);
    if (!vap->buf_x) {
        free(vap->kv_cache);
        free(vap->layers);
        free(vap);
        munmap(mapped, file_size);
        return NULL;
    }
    vap->buf_norm = vap->buf_x + d_model;
    vap->buf_qkv = vap->buf_norm + d_model;
    vap->buf_attn = vap->buf_qkv + 3 * n_heads * head_dim;
    vap->buf_ff = vap->buf_attn + VAP_CONTEXT_LEN;
    vap->silu_ws = vap->buf_ff + ff_dim;

    return vap;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * vap_create_config (for testing, zero weights)
 * ═══════════════════════════════════════════════════════════════════════════ */

static float *alloc_zeros(size_t n) {
    float *p = calloc(n, sizeof(float));
    if (!p) {
        fprintf(stderr, "[vap] calloc failed for %zu floats\n", n);
    }
    return p;
}

VAPModel *vap_create_config(int d_model, int n_layers, int n_heads, int ff_dim) {
    if (d_model <= 0 || n_layers <= 0 || n_heads <= 0 || ff_dim <= 0 ||
        d_model % n_heads != 0) return NULL;

    int head_dim = d_model / n_heads;

    VAPModel *vap = calloc(1, sizeof(VAPModel));
    if (!vap) return NULL;
    vap->d_model = d_model;
    vap->n_layers = n_layers;
    vap->n_heads = n_heads;
    vap->head_dim = head_dim;
    vap->ff_dim = ff_dim;
    vap->context_len = VAP_CONTEXT_LEN;
    vap->pos = 0;
    vap->n_valid = 0;
    vap->alpha = 0.3f;
    vap->smoothed = (VAPPrediction){0.0f, 0.0f, 0.0f, 0.0f};
    vap->owns_weights = 1;

    /* Allocate zero-initialized weights */
    vap->input_proj_w = (const float *)alloc_zeros((size_t)d_model * VAP_INPUT_DIM);
    vap->input_proj_b = (const float *)alloc_zeros((size_t)d_model);
    if (!vap->input_proj_w || !vap->input_proj_b) goto err;

    vap->layers = calloc((size_t)n_layers, sizeof(VAPLayerWeights));
    if (!vap->layers) goto err;
    for (int l = 0; l < n_layers; l++) {
        VAPLayerWeights *lw = &vap->layers[l];
        lw->attn_norm_w = (const float *)alloc_zeros((size_t)d_model);
        lw->wq_w = (const float *)alloc_zeros((size_t)n_heads * head_dim * d_model);
        lw->wk_w = (const float *)alloc_zeros((size_t)n_heads * head_dim * d_model);
        lw->wv_w = (const float *)alloc_zeros((size_t)n_heads * head_dim * d_model);
        lw->wo_w = (const float *)alloc_zeros((size_t)d_model * n_heads * head_dim);
        lw->ffn_norm_w = (const float *)alloc_zeros((size_t)d_model);
        lw->ffn_up_w = (const float *)alloc_zeros((size_t)ff_dim * d_model);
        lw->ffn_up_b = (const float *)alloc_zeros((size_t)ff_dim);
        lw->ffn_down_w = (const float *)alloc_zeros((size_t)d_model * ff_dim);
        lw->ffn_down_b = (const float *)alloc_zeros((size_t)d_model);
        if (!lw->attn_norm_w || !lw->wq_w || !lw->wk_w || !lw->wv_w || !lw->wo_w ||
            !lw->ffn_norm_w || !lw->ffn_up_w || !lw->ffn_up_b ||
            !lw->ffn_down_w || !lw->ffn_down_b) goto err_layers;
    }
    vap->final_norm_w = (const float *)alloc_zeros((size_t)d_model);
    vap->head_user_w = (const float *)alloc_zeros((size_t)d_model);
    vap->head_user_b = (const float *)alloc_zeros(1);
    vap->head_system_w = (const float *)alloc_zeros((size_t)d_model);
    vap->head_system_b = (const float *)alloc_zeros(1);
    vap->head_backchannel_w = (const float *)alloc_zeros((size_t)d_model);
    vap->head_backchannel_b = (const float *)alloc_zeros(1);
    vap->head_eou_w = (const float *)alloc_zeros((size_t)d_model);
    vap->head_eou_b = (const float *)alloc_zeros(1);
    if (!vap->final_norm_w || !vap->head_user_w || !vap->head_user_b ||
        !vap->head_system_w || !vap->head_system_b ||
        !vap->head_backchannel_w || !vap->head_backchannel_b ||
        !vap->head_eou_w || !vap->head_eou_b) goto err_heads;

    size_t kv_sz = (size_t)2 * n_layers * VAP_CONTEXT_LEN * n_heads * head_dim * sizeof(float);
    vap->kv_cache = calloc(1, kv_sz);
    if (!vap->kv_cache) goto err_heads;

    size_t buf_total = (size_t)d_model * 2 + 3 * n_heads * head_dim +
                       VAP_CONTEXT_LEN + ff_dim * 2;
    vap->buf_x = malloc(buf_total * sizeof(float));
    if (!vap->buf_x) goto err_kv;
    vap->buf_norm = vap->buf_x + d_model;
    vap->buf_qkv = vap->buf_norm + d_model;
    vap->buf_attn = vap->buf_qkv + 3 * n_heads * head_dim;
    vap->buf_ff = vap->buf_attn + VAP_CONTEXT_LEN;
    vap->silu_ws = vap->buf_ff + ff_dim;

    return vap;

err_heads:
    free((void *)vap->final_norm_w);
    free((void *)vap->head_user_w);
    free((void *)vap->head_user_b);
    free((void *)vap->head_system_w);
    free((void *)vap->head_system_b);
    free((void *)vap->head_backchannel_w);
    free((void *)vap->head_backchannel_b);
    free((void *)vap->head_eou_w);
    free((void *)vap->head_eou_b);
err_layers:
    for (int l = 0; l < n_layers; l++) {
        VAPLayerWeights *lw = &vap->layers[l];
        free((void *)lw->attn_norm_w);
        free((void *)lw->wq_w);
        free((void *)lw->wk_w);
        free((void *)lw->wv_w);
        free((void *)lw->wo_w);
        free((void *)lw->ffn_norm_w);
        free((void *)lw->ffn_up_w);
        free((void *)lw->ffn_up_b);
        free((void *)lw->ffn_down_w);
        free((void *)lw->ffn_down_b);
    }
    free(vap->layers);
    free((void *)vap->input_proj_w);
    free((void *)vap->input_proj_b);
err_kv:
    free(vap->kv_cache);
err:
    free(vap);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * vap_feed
 * ═══════════════════════════════════════════════════════════════════════════ */

VAPPrediction vap_feed(VAPModel *vap, const float *user_mel, const float *system_mel) {
    if (!vap) return (VAPPrediction){0.0f, 0.0f, 0.0f, 0.0f};

    float input[VAP_INPUT_DIM];
    memcpy(input, user_mel, (size_t)VAP_MEL_DIM * sizeof(float));
    if (system_mel)
        memcpy(input + VAP_MEL_DIM, system_mel, (size_t)VAP_MEL_DIM * sizeof(float));
    else
        memset(input + VAP_MEL_DIM, 0, (size_t)VAP_MEL_DIM * sizeof(float));

    linear_vec(vap->buf_x, input, vap->input_proj_w, vap->input_proj_b,
              vap->d_model, VAP_INPUT_DIM);
    add_pos_enc(vap->buf_x, vap->pos, vap->d_model);

    int D = vap->d_model;
    int n_heads = vap->n_heads;
    int head_dim = vap->head_dim;
    int ff_dim = vap->ff_dim;
    int cache_len = vap->context_len;
    int n_ctx = (vap->n_valid + 1) < cache_len ? vap->n_valid + 1 : cache_len;

    for (int l = 0; l < vap->n_layers; l++) {
        float *kv_k = vap->kv_cache + 2 * l * cache_len * n_heads * head_dim;
        float *kv_v = vap->kv_cache + (2 * l + 1) * cache_len * n_heads * head_dim;

        float attn_out[D];
        attn_step(attn_out, vap->buf_x, &vap->layers[l], kv_k, kv_v,
                  cache_len, vap->pos, n_ctx, vap->n_valid, vap->buf_norm, vap->buf_qkv,
                  vap->buf_attn, D, n_heads, head_dim);
        for (int i = 0; i < D; i++) vap->buf_x[i] += attn_out[i];

        float ffn_out[D];
        ffn_step(ffn_out, vap->buf_x, &vap->layers[l], vap->buf_norm,
                 vap->buf_ff, vap->silu_ws, D, ff_dim);
        for (int i = 0; i < D; i++) vap->buf_x[i] += ffn_out[i];
    }

    rms_norm(vap->buf_x, vap->buf_x, vap->final_norm_w, D, 1e-5f);

    float lu, ls, lb, le;
    vDSP_dotpr(vap->buf_x, 1, vap->head_user_w, 1, &lu, D);
    lu += vap->head_user_b[0];
    vDSP_dotpr(vap->buf_x, 1, vap->head_system_w, 1, &ls, D);
    ls += vap->head_system_b[0];
    vDSP_dotpr(vap->buf_x, 1, vap->head_backchannel_w, 1, &lb, D);
    lb += vap->head_backchannel_b[0];
    vDSP_dotpr(vap->buf_x, 1, vap->head_eou_w, 1, &le, D);
    le += vap->head_eou_b[0];

    VAPPrediction pred = {
        sigmoidf(lu),
        sigmoidf(ls),
        sigmoidf(lb),
        sigmoidf(le)
    };

    /* EMA smoothing */
    float a = vap->alpha;
    float b = 1.0f - a;
    vap->smoothed.p_user_speaking = a * vap->smoothed.p_user_speaking + b * pred.p_user_speaking;
    vap->smoothed.p_system_turn = a * vap->smoothed.p_system_turn + b * pred.p_system_turn;
    vap->smoothed.p_backchannel = a * vap->smoothed.p_backchannel + b * pred.p_backchannel;
    vap->smoothed.p_eou = a * vap->smoothed.p_eou + b * pred.p_eou;

    vap->n_valid++;
    if (vap->n_valid > cache_len) vap->n_valid = cache_len;
    vap->pos++;
    if (vap->pos >= cache_len) vap->pos = 0;

    return pred;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * vap_reset, vap_get_smoothed, vap_set_smoothing, vap_destroy
 * ═══════════════════════════════════════════════════════════════════════════ */

void vap_reset(VAPModel *vap) {
    if (!vap) return;
    vap->pos = 0;
    vap->n_valid = 0;
    if (vap->kv_cache) {
        size_t sz = (size_t)2 * vap->n_layers * vap->context_len *
                    vap->n_heads * vap->head_dim * sizeof(float);
        memset(vap->kv_cache, 0, sz);
    }
    vap->smoothed = (VAPPrediction){0.0f, 0.0f, 0.0f, 0.0f};
}

VAPPrediction vap_get_smoothed(const VAPModel *vap) {
    if (!vap) return (VAPPrediction){0.0f, 0.0f, 0.0f, 0.0f};
    return vap->smoothed;
}

void vap_set_smoothing(VAPModel *vap, float alpha) {
    if (vap) vap->alpha = alpha;
}

void vap_destroy(VAPModel *vap) {
    if (!vap) return;
    free(vap->kv_cache);
    free(vap->buf_x);
    if (vap->owns_weights && vap->layers) {
        free((void *)vap->input_proj_w);
        free((void *)vap->input_proj_b);
        for (int l = 0; l < vap->n_layers; l++) {
            VAPLayerWeights *lw = &vap->layers[l];
            free((void *)lw->attn_norm_w);
            free((void *)lw->wq_w);
            free((void *)lw->wk_w);
            free((void *)lw->wv_w);
            free((void *)lw->wo_w);
            free((void *)lw->ffn_norm_w);
            free((void *)lw->ffn_up_w);
            free((void *)lw->ffn_up_b);
            free((void *)lw->ffn_down_w);
            free((void *)lw->ffn_down_b);
        }
        free((void *)vap->final_norm_w);
        free((void *)vap->head_user_w);
        free((void *)vap->head_user_b);
        free((void *)vap->head_system_w);
        free((void *)vap->head_system_b);
        free((void *)vap->head_backchannel_w);
        free((void *)vap->head_backchannel_b);
        free((void *)vap->head_eou_w);
        free((void *)vap->head_eou_b);
    } else if (vap->mmap_base) {
        munmap(vap->mmap_base, vap->mmap_size);
    }
    free(vap->layers);
    free(vap);
}
