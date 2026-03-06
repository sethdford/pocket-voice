/**
 * sonata_refiner.c — Sonata STT Pass 2: semantic tokens → text.
 *
 * Encoder-decoder transformer. Encoder runs bidirectional on semantic tokens;
 * decoder generates text autoregressively with cross-attention.
 * All matrix ops via cblas_sgemm/cblas_sgemv (AMX-accelerated).
 *
 * Weight file: .cref (see scripts/export_sonata_refiner.py)
 */

#include "sonata_refiner.h"

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

/* ═══════════════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════════════ */

#define CREF_MAGIC     0x46455243  /* "CREF" little-endian */
#define CREF_VERSION   1
#define CREF_VERSION_2 2  /* v2: header includes enc_d_ff, dec_d_ff */
#define MAX_AUDIO_LEN  2048
#define MAX_TEXT_LEN   512
#define TEXT_BOS_ID    1
#define TEXT_EOS_ID    2
#define TEXT_PAD_ID    0

/* Simple character mapping for token IDs (initial impl; BPE needs vocab file).
 * 0=pad, 1=bos, 2=eos, 3=space, 4-29=a-z, 30=' */
static char token_to_char(int id) {
    if (id == 0 || id == 1 || id == 2) return '\0';  /* skip */
    if (id == 3) return ' ';
    if (id >= 4 && id <= 29) return (char)('a' + id - 4);
    if (id == 30) return '\'';
    if (id >= 32 && id <= 126) return (char)id;  /* printable ASCII */
    return '?';
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Binary header
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t semantic_vocab_size;
    uint32_t text_vocab_size;
    uint32_t enc_d_model;
    uint32_t enc_n_layers;
    uint32_t enc_n_heads;
    uint32_t dec_d_model;
    uint32_t dec_n_layers;
    uint32_t dec_n_heads;
    uint32_t dec_n_kv_heads;
    uint32_t max_audio_len;
    uint64_t n_weights;
} CRefHeader;

/* ═══════════════════════════════════════════════════════════════════════════
 * Encoder layer weights
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const float *attn_norm_w;
    const float *attn_in_proj_w;
    const float *attn_in_proj_b;
    const float *attn_out_proj_w;
    const float *attn_out_proj_b;
    const float *ffn_norm_w;
    const float *ffn_up_w;
    const float *ffn_up_b;
    const float *ffn_down_w;
    const float *ffn_down_b;
} EncLayerWeights;

/* ═══════════════════════════════════════════════════════════════════════════
 * Decoder layer weights
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const float *self_attn_norm_w;
    const float *wq_w;
    const float *wk_w;
    const float *wv_w;
    const float *wo_w;
    const float *cross_norm_w;
    const float *cross_q_w;
    const float *cross_k_w;
    const float *cross_v_w;
    const float *cross_o_w;
    const float *ffn_norm_w;
    const float *ffn_up_w;
    const float *ffn_up_b;
    const float *ffn_down_w;
    const float *ffn_down_b;
} DecLayerWeights;

/* ═══════════════════════════════════════════════════════════════════════════
 * Engine state
 * ═══════════════════════════════════════════════════════════════════════════ */

struct SonataRefiner {
    int semantic_vocab_size;
    int text_vocab_size;
    int enc_d_model;
    int enc_n_layers;
    int enc_n_heads;
    int enc_d_ff;
    int dec_d_model;
    int dec_n_layers;
    int dec_n_heads;
    int dec_n_kv_heads;
    int dec_head_dim;
    int dec_d_ff;
    int max_audio_len;

    const float *sem_emb;
    const float *sem_pos;
    EncLayerWeights *enc_layers;
    const float *enc_norm_w;

    const float *text_emb;
    DecLayerWeights *dec_layers;
    const float *dec_norm_w;
    const float *output_proj_w;

    float *rope_cos;
    float *rope_sin;
    int rope_max_len;

    void *mmap_base;
    size_t mmap_size;

    float *enc_out;      /* T_enc × enc_d_model */
    float *dec_buf;      /* working buffer for decoder */
    float *buf_a;
    float *buf_b;
    float *silu_ws;      /* workspace for vectorized SiLU */
    float *buf_attn;     /* encoder MHSA: Qh,Kh,Vh,scores */
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Math helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static void rms_norm(float *out, const float *x, const float *w,
                     int n, float eps) {
    float sum_sq;
    vDSP_svesq(x, 1, &sum_sq, n);
    float rms = 1.0f / sqrtf(sum_sq / (float)n + eps);
    vDSP_vsmul(x, 1, &rms, out, 1, n);
    vDSP_vmul(out, 1, w, 1, out, 1, n);
}

static void rms_norm_rows(float *out, const float *x, const float *w,
                         int T, int D, float eps) {
    for (int t = 0; t < T; t++) {
        rms_norm(out + t * D, x + t * D, w, D, eps);
    }
}

static void linear(float *out, const float *x, const float *W, const float *b,
                   int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K,
                1.0f, x, K, W, K, 0.0f, out, N);
    if (b) {
        for (int i = 0; i < M; i++)
            vDSP_vadd(out + i * N, 1, b, 1, out + i * N, 1, N);
    }
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
    float mx = x[0];
    for (int i = 1; i < len; i++) if (x[i] > mx) mx = x[i];
    float sum = 0;
    for (int i = 0; i < len; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    vDSP_vsmul(x, 1, &inv, x, 1, len);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Encoder
 * ═══════════════════════════════════════════════════════════════════════════ */

static void encoder_mhsa(float *out, const float *x, const EncLayerWeights *lw,
                         float *buf_norm, float *buf_qkv, float *buf_attn,
                         int T, int D, int n_heads) {
    int head_dim = D / n_heads;
    rms_norm_rows(buf_norm, x, lw->attn_norm_w, T, D, 1e-5f);
    linear(buf_qkv, buf_norm, lw->attn_in_proj_w, lw->attn_in_proj_b, T, D, 3 * D);

    float *Q = buf_qkv;
    float *K = buf_qkv + T * D;
    float *V = buf_qkv + T * 2 * D;
    float scale = 1.0f / sqrtf((float)head_dim);

    memset(out, 0, T * D * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        /* Gather head-strided Q,K,V into contiguous blocks for GEMM */
        float *Qh = buf_attn;
        float *Kh = buf_attn + T * head_dim;
        float *Vh = buf_attn + 2 * T * head_dim;
        float *scores = buf_attn + 3 * T * head_dim;

        for (int t = 0; t < T; t++) {
            memcpy(Qh + t * head_dim, Q + t * D + h * head_dim, head_dim * sizeof(float));
            memcpy(Kh + t * head_dim, K + t * D + h * head_dim, head_dim * sizeof(float));
            memcpy(Vh + t * head_dim, V + t * D + h * head_dim, head_dim * sizeof(float));
        }

        /* scores = scale * Qh @ Kh^T */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, T, head_dim,
                    scale, Qh, head_dim, Kh, head_dim,
                    0.0f, scores, T);

        for (int qi = 0; qi < T; qi++)
            softmax_row(scores + qi * T, T);

        /* context = scores @ Vh */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, head_dim, T,
                    1.0f, scores, T, Vh, head_dim,
                    0.0f, Qh, head_dim);

        for (int t = 0; t < T; t++)
            memcpy(out + t * D + h * head_dim, Qh + t * head_dim, head_dim * sizeof(float));
    }

    memcpy(buf_norm, out, T * D * sizeof(float));
    linear(out, buf_norm, lw->attn_out_proj_w, lw->attn_out_proj_b, T, D, D);
}

static void encoder_ffn(float *out, const float *x, const EncLayerWeights *lw,
                       float *buf_norm, float *buf_ff, float *silu_ws,
                       int T, int D, int ff) {
    rms_norm_rows(buf_norm, x, lw->ffn_norm_w, T, D, 1e-5f);
    linear(buf_ff, buf_norm, lw->ffn_up_w, lw->ffn_up_b, T, D, ff);
    silu_inplace(buf_ff, T * ff, silu_ws);
    linear(out, buf_ff, lw->ffn_down_w, lw->ffn_down_b, T, ff, D);
}

static void encoder_forward(SonataRefiner *ref, const float *x, int T,
                            float *enc_out) {
    int D = ref->enc_d_model;
    int ff = ref->enc_d_ff;
    int n_heads = ref->enc_n_heads;

    float *buf = ref->buf_a;
    float *buf2 = ref->buf_b;
    float *tmp = ref->dec_buf;

    memcpy(enc_out, x, (size_t)T * D * sizeof(float));

    for (int l = 0; l < ref->enc_n_layers; l++) {
        const EncLayerWeights *lw = &ref->enc_layers[l];
        encoder_mhsa(buf, enc_out, lw, buf2, tmp, ref->buf_attn, T, D, n_heads);
        for (int i = 0; i < T * D; i++) enc_out[i] += buf[i];
        encoder_ffn(buf, enc_out, lw, buf2, tmp, ref->silu_ws, T, D, ff);
        for (int i = 0; i < T * D; i++) enc_out[i] += buf[i];
    }
    rms_norm_rows(enc_out, enc_out, ref->enc_norm_w, T, D, 1e-5f);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Decoder (single step, autoregressive)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void decoder_self_attn_step(float *out, const float *x,
                                   const DecLayerWeights *lw,
                                   float *kv_cache_k, float *kv_cache_v,
                                   int cache_len, int pos,
                                   const float *rope_cos, const float *rope_sin,
                                   float *buf_norm, float *buf_qkv,
                                   float *buf_attn, int D,
                                   int n_heads, int n_kv_heads, int head_dim) {
    int n_rep = n_heads / n_kv_heads;
    rms_norm(buf_norm, x, lw->self_attn_norm_w, D, 1e-5f);

    float *q = buf_qkv;
    float *k = buf_qkv + n_heads * head_dim;
    float *v = buf_qkv + (n_heads + n_kv_heads) * head_dim;

    cblas_sgemv(CblasRowMajor, CblasNoTrans, n_heads * head_dim, D,
                1.0f, lw->wq_w, D, buf_norm, 1, 0.0f, q, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n_kv_heads * head_dim, D,
                1.0f, lw->wk_w, D, buf_norm, 1, 0.0f, k, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n_kv_heads * head_dim, D,
                1.0f, lw->wv_w, D, buf_norm, 1, 0.0f, v, 1);

    /* Apply RoPE to q (each head) and k (each kv head) */
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float cosv = rope_cos[pos * half + i];
            float sinv = rope_sin[pos * half + i];
            int i0 = i * 2, i1 = i * 2 + 1;
            float q0 = qh[i0], q1 = qh[i1];
            qh[i0] = q0 * cosv - q1 * sinv;
            qh[i1] = q0 * sinv + q1 * cosv;
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            float cosv = rope_cos[pos * half + i];
            float sinv = rope_sin[pos * half + i];
            int i0 = i * 2, i1 = i * 2 + 1;
            float k0 = kh[i0], k1 = kh[i1];
            kh[i0] = k0 * cosv - k1 * sinv;
            kh[i1] = k0 * sinv + k1 * cosv;
        }
    }

    memcpy(kv_cache_k + cache_len * n_kv_heads * head_dim, k,
           (size_t)n_kv_heads * head_dim * sizeof(float));
    memcpy(kv_cache_v + cache_len * n_kv_heads * head_dim, v,
           (size_t)n_kv_heads * head_dim * sizeof(float));

    if (cache_len >= MAX_TEXT_LEN) return;  /* bounds guard for scores[] */

    float scale = 1.0f / sqrtf((float)head_dim);
    memset(out, 0, D * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / n_rep;
        float scores[MAX_TEXT_LEN];
        for (int ti = 0; ti <= cache_len; ti++) {
            float dot = 0;
            const float *kr = kv_cache_k + ti * n_kv_heads * head_dim + kv_h * head_dim;
            vDSP_dotpr(q + h * head_dim, 1, kr, 1, &dot, head_dim);
            scores[ti] = dot * scale;
        }
        softmax_row(scores, cache_len + 1);
        for (int ti = 0; ti <= cache_len; ti++) {
            float w = scores[ti];
            const float *vr = kv_cache_v + ti * n_kv_heads * head_dim + kv_h * head_dim;
            for (int d = 0; d < head_dim; d++)
                out[h * head_dim + d] += w * vr[d];
        }
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, D, n_heads * head_dim,
                1.0f, lw->wo_w, n_heads * head_dim, out, 1, 0.0f, buf_norm, 1);
    memcpy(out, buf_norm, D * sizeof(float));
}

static void decoder_cross_attn_step(float *out, const float *x,
                                    const float *enc_out, int T_enc,
                                    const DecLayerWeights *lw,
                                    float *buf_norm, float *buf_qkv,
                                    int D, int n_heads, int n_kv_heads, int head_dim) {
    if (T_enc >= MAX_AUDIO_LEN) return;  /* bounds guard for scores[] */

    int n_rep = n_heads / n_kv_heads;
    int q_sz = n_heads * head_dim;
    int kv_sz = T_enc * n_kv_heads * head_dim;

    rms_norm(buf_norm, x, lw->cross_norm_w, D, 1e-5f);

    float *q = buf_qkv;
    float *k = buf_qkv + q_sz;
    float *v = buf_qkv + q_sz + kv_sz;

    cblas_sgemv(CblasRowMajor, CblasNoTrans, n_heads * head_dim, D,
                1.0f, lw->cross_q_w, D, buf_norm, 1, 0.0f, q, 1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T_enc, n_kv_heads * head_dim, D,
                1.0f, enc_out, D, lw->cross_k_w, D, 0.0f, k, n_kv_heads * head_dim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T_enc, n_kv_heads * head_dim, D,
                1.0f, enc_out, D, lw->cross_v_w, D, 0.0f, v, n_kv_heads * head_dim);

    float scale = 1.0f / sqrtf((float)head_dim);
    memset(out, 0, D * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / n_rep;
        float scores[MAX_AUDIO_LEN];
        for (int ti = 0; ti < T_enc; ti++) {
            float dot = 0;
            const float *kr = k + ti * n_kv_heads * head_dim + kv_h * head_dim;
            vDSP_dotpr(q + h * head_dim, 1, kr, 1, &dot, head_dim);
            scores[ti] = dot * scale;
        }
        softmax_row(scores, T_enc);
        for (int ti = 0; ti < T_enc; ti++) {
            float w = scores[ti];
            const float *vr = v + ti * n_kv_heads * head_dim + kv_h * head_dim;
            for (int d = 0; d < head_dim; d++)
                out[h * head_dim + d] += w * vr[d];
        }
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, D, n_heads * head_dim,
                1.0f, lw->cross_o_w, n_heads * head_dim, out, 1, 0.0f, buf_norm, 1);
    memcpy(out, buf_norm, D * sizeof(float));
}

static void decoder_ffn_step(float *out, const float *x,
                             const DecLayerWeights *lw, float *buf_norm,
                             float *buf_ff, float *silu_ws, int D, int ff) {
    rms_norm(buf_norm, x, lw->ffn_norm_w, D, 1e-5f);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, ff, D, 1.0f, lw->ffn_up_w, D,
                buf_norm, 1, 0.0f, buf_ff, 1);
    vDSP_vadd(buf_ff, 1, lw->ffn_up_b, 1, buf_ff, 1, ff);
    silu_inplace(buf_ff, ff, silu_ws);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, D, ff, 1.0f, lw->ffn_down_w, ff,
                buf_ff, 1, 0.0f, out, 1);
    vDSP_vadd(out, 1, lw->ffn_down_b, 1, out, 1, D);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Weight loading
 * ═══════════════════════════════════════════════════════════════════════════ */

static const float *advance(const float **ptr, int count) {
    const float *p = *ptr;
    *ptr += count;
    return p;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════════ */

SonataRefiner *sonata_refiner_create(const char *model_path) {
    if (!model_path) return NULL;

    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[sonata_refiner] Cannot open %s\n", model_path);
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
        fprintf(stderr, "[sonata_refiner] mmap failed\n");
        return NULL;
    }

    const char *hdr_bytes = (const char *)mapped;
    if (memcmp(hdr_bytes, "CREF", 4) != 0) {
        fprintf(stderr, "[sonata_refiner] Invalid magic (expected CREF)\n");
        munmap(mapped, file_size);
        return NULL;
    }
    uint32_t version = *(const uint32_t *)(hdr_bytes + 4);
    if (version != CREF_VERSION && version != CREF_VERSION_2) {
        fprintf(stderr, "[sonata_refiner] Unsupported version %u\n", version);
        munmap(mapped, file_size);
        return NULL;
    }

    const uint32_t *u32 = (const uint32_t *)(hdr_bytes + 8);
    int semantic_vocab = (int)u32[0];
    int text_vocab = (int)u32[1];
    int enc_d_model = (int)u32[2];
    int enc_n_layers = (int)u32[3];
    int enc_n_heads = (int)u32[4];
    int dec_d_model = (int)u32[5];
    int dec_n_layers = (int)u32[6];
    int dec_n_heads = (int)u32[7];
    int dec_n_kv_heads = (int)u32[8];
    int max_audio_len = (int)u32[9];
    uint64_t n_weights = *(const uint64_t *)(hdr_bytes + 48);

    int enc_d_ff, dec_d_ff;
    size_t weights_offset;
    if (version >= CREF_VERSION_2) {
        enc_d_ff = (int)*(const uint32_t *)(hdr_bytes + 56);
        dec_d_ff = (int)*(const uint32_t *)(hdr_bytes + 60);
        weights_offset = 64;
    } else {
        enc_d_ff = enc_d_model * 4;
        dec_d_ff = dec_d_model * 4;
        weights_offset = 56;
    }
    int dec_head_dim = dec_d_model / dec_n_heads;
    int sem_emb_size = semantic_vocab + 4;
    int text_emb_size = text_vocab + 4;

    const float *ptr = (const float *)(hdr_bytes + weights_offset);

    SonataRefiner *ref = calloc(1, sizeof(SonataRefiner));
    if (!ref) { munmap(mapped, file_size); return NULL; }

    ref->semantic_vocab_size = semantic_vocab;
    ref->text_vocab_size = text_vocab;
    ref->enc_d_model = enc_d_model;
    ref->enc_n_layers = enc_n_layers;
    ref->enc_n_heads = enc_n_heads;
    ref->enc_d_ff = enc_d_ff;
    ref->dec_d_model = dec_d_model;
    ref->dec_n_layers = dec_n_layers;
    ref->dec_n_heads = dec_n_heads;
    ref->dec_n_kv_heads = dec_n_kv_heads;
    ref->dec_head_dim = dec_head_dim;
    ref->dec_d_ff = dec_d_ff;
    ref->max_audio_len = max_audio_len;
    ref->mmap_base = mapped;
    ref->mmap_size = file_size;

    ref->sem_emb = advance(&ptr, sem_emb_size * enc_d_model);
    ref->sem_pos = advance(&ptr, max_audio_len * enc_d_model);

    ref->enc_layers = calloc(enc_n_layers, sizeof(EncLayerWeights));
    if (!ref->enc_layers) {
        free(ref);
        munmap(mapped, file_size);
        return NULL;
    }
    for (int l = 0; l < enc_n_layers; l++) {
        EncLayerWeights *lw = &ref->enc_layers[l];
        lw->attn_norm_w = advance(&ptr, enc_d_model);
        lw->attn_in_proj_w = advance(&ptr, 3 * enc_d_model * enc_d_model);
        lw->attn_in_proj_b = advance(&ptr, 3 * enc_d_model);
        lw->attn_out_proj_w = advance(&ptr, enc_d_model * enc_d_model);
        lw->attn_out_proj_b = advance(&ptr, enc_d_model);
        lw->ffn_norm_w = advance(&ptr, enc_d_model);
        lw->ffn_up_w = advance(&ptr, enc_d_ff * enc_d_model);
        lw->ffn_up_b = advance(&ptr, enc_d_ff);
        lw->ffn_down_w = advance(&ptr, enc_d_model * enc_d_ff);
        lw->ffn_down_b = advance(&ptr, enc_d_model);
    }
    ref->enc_norm_w = advance(&ptr, enc_d_model);

    ref->text_emb = advance(&ptr, text_emb_size * dec_d_model);

    ref->dec_layers = calloc(dec_n_layers, sizeof(DecLayerWeights));
    if (!ref->dec_layers) {
        free(ref->enc_layers);
        free(ref);
        munmap(mapped, file_size);
        return NULL;
    }
    for (int l = 0; l < dec_n_layers; l++) {
        DecLayerWeights *lw = &ref->dec_layers[l];
        lw->self_attn_norm_w = advance(&ptr, dec_d_model);
        lw->wq_w = advance(&ptr, dec_n_heads * dec_head_dim * dec_d_model);
        lw->wk_w = advance(&ptr, dec_n_kv_heads * dec_head_dim * dec_d_model);
        lw->wv_w = advance(&ptr, dec_n_kv_heads * dec_head_dim * dec_d_model);
        lw->wo_w = advance(&ptr, dec_d_model * dec_n_heads * dec_head_dim);
        lw->cross_norm_w = advance(&ptr, dec_d_model);
        lw->cross_q_w = advance(&ptr, dec_n_heads * dec_head_dim * dec_d_model);
        lw->cross_k_w = advance(&ptr, dec_n_kv_heads * dec_head_dim * enc_d_model);
        lw->cross_v_w = advance(&ptr, dec_n_kv_heads * dec_head_dim * enc_d_model);
        lw->cross_o_w = advance(&ptr, dec_d_model * dec_n_heads * dec_head_dim);
        lw->ffn_norm_w = advance(&ptr, dec_d_model);
        lw->ffn_up_w = advance(&ptr, dec_d_ff * dec_d_model);
        lw->ffn_up_b = advance(&ptr, dec_d_ff);
        lw->ffn_down_w = advance(&ptr, dec_d_model * dec_d_ff);
        lw->ffn_down_b = advance(&ptr, dec_d_model);
    }
    ref->dec_norm_w = advance(&ptr, dec_d_model);
    ref->output_proj_w = advance(&ptr, text_emb_size * dec_d_model);

    /* Validate weight file size — ensure we didn't overrun the mmap region */
    {
        size_t expected_size = (size_t)((const char *)ptr - (const char *)mapped);
        if (expected_size > file_size) {
            fprintf(stderr, "[sonata_refiner] Weight file too small: need %zu bytes, got %zu\n",
                    expected_size, file_size);
            free(ref->dec_layers);
            free(ref->enc_layers);
            free(ref);
            munmap(mapped, file_size);
            return NULL;
        }
    }

    /* Precompute RoPE cos/sin */
    ref->rope_max_len = MAX_TEXT_LEN;
    int half = dec_head_dim / 2;
    ref->rope_cos = malloc((size_t)ref->rope_max_len * half * sizeof(float));
    ref->rope_sin = malloc((size_t)ref->rope_max_len * half * sizeof(float));
    if (!ref->rope_cos || !ref->rope_sin) {
        free(ref->rope_cos);
        free(ref->rope_sin);
        free(ref->dec_layers);
        free(ref->enc_layers);
        free(ref);
        munmap(mapped, file_size);
        return NULL;
    }
    float theta = 10000.0f;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(theta, 2.0f * i / (float)dec_head_dim);
        for (int pos = 0; pos < ref->rope_max_len; pos++) {
            float angle = pos * freq;
            ref->rope_cos[pos * half + i] = cosf(angle);
            ref->rope_sin[pos * half + i] = sinf(angle);
        }
    }

    int enc_head_dim = enc_d_model / enc_n_heads;
    size_t silu_ws_sz = (size_t)MAX_AUDIO_LEN * enc_d_ff;
    if (silu_ws_sz < (size_t)MAX_TEXT_LEN * dec_d_ff)
        silu_ws_sz = (size_t)MAX_TEXT_LEN * dec_d_ff;
    size_t buf_attn_sz = (size_t)3 * MAX_AUDIO_LEN * enc_head_dim +
                        (size_t)MAX_AUDIO_LEN * MAX_AUDIO_LEN;

    ref->enc_out = malloc((size_t)MAX_AUDIO_LEN * enc_d_model * sizeof(float));
    /* Scratch must fit encoder MHSA (T*3*D + buf_attn) and decoder KV+scratch */
    size_t enc_scratch = (size_t)MAX_AUDIO_LEN * enc_d_model * 3 +
                         (size_t)MAX_AUDIO_LEN * MAX_AUDIO_LEN;
    size_t dec_scratch = (size_t)MAX_AUDIO_LEN * enc_d_model * 2 +
                         (size_t)MAX_TEXT_LEN * dec_d_model * 4 +
                         (size_t)MAX_TEXT_LEN * MAX_TEXT_LEN;
    size_t scratch_sz = enc_scratch > dec_scratch ? enc_scratch : dec_scratch;
    ref->dec_buf = malloc(scratch_sz * sizeof(float));
    ref->buf_a = malloc((size_t)MAX_AUDIO_LEN * enc_d_model * 4 * sizeof(float));
    ref->buf_b = malloc((size_t)MAX_AUDIO_LEN * enc_d_model * 4 * sizeof(float));
    ref->silu_ws = malloc(silu_ws_sz * sizeof(float));
    ref->buf_attn = malloc(buf_attn_sz * sizeof(float));

    if (!ref->enc_out || !ref->dec_buf || !ref->buf_a || !ref->buf_b ||
        !ref->silu_ws || !ref->buf_attn) {
        free(ref->enc_out);
        free(ref->dec_buf);
        free(ref->buf_a);
        free(ref->buf_b);
        free(ref->silu_ws);
        free(ref->buf_attn);
        free(ref->rope_cos);
        free(ref->rope_sin);
        free(ref->dec_layers);
        free(ref->enc_layers);
        free(ref);
        munmap(mapped, file_size);
        return NULL;
    }

    fprintf(stderr, "[sonata_refiner] Loaded: enc %dL×d=%d, dec %dL×d=%d (%.1fM)\n",
            enc_n_layers, enc_d_model, dec_n_layers, dec_d_model,
            (float)n_weights / 1e6f);

    return ref;
}

void sonata_refiner_destroy(SonataRefiner *ref) {
    if (!ref) return;
    free(ref->enc_layers);
    free(ref->dec_layers);
    free(ref->rope_cos);
    free(ref->rope_sin);
    free(ref->enc_out);
    free(ref->dec_buf);
    free(ref->buf_a);
    free(ref->buf_b);
    free(ref->silu_ws);
    free(ref->buf_attn);
    if (ref->mmap_base)
        munmap(ref->mmap_base, ref->mmap_size);
    free(ref);
}

void sonata_refiner_reset(SonataRefiner *ref) {
    if (!ref || !ref->dec_buf) return;
    /* Zero the decoder KV cache: [dec_n_layers][2][MAX_TEXT_LEN][n_kv_heads][head_dim] */
    size_t kv_size = (size_t)ref->dec_n_layers * 2 * MAX_TEXT_LEN *
                    (size_t)ref->dec_n_kv_heads * (size_t)ref->dec_head_dim;
    memset(ref->dec_buf, 0, kv_size * sizeof(float));
}

int sonata_refiner_process(SonataRefiner *ref,
                            const int *semantic_ids, int n_tokens,
                            char *out_text, int max_len) {
    if (!ref || !out_text || max_len <= 0) return -1;
    if (!semantic_ids && n_tokens > 0) return -1;
    if (n_tokens <= 0) { out_text[0] = '\0'; return 0; }
    if (n_tokens > ref->max_audio_len) n_tokens = ref->max_audio_len;

    int enc_D = ref->enc_d_model;
    int dec_D = ref->dec_d_model;
    int text_V = ref->text_vocab_size + 4;

    /* Encode semantic tokens */
    float *enc_in = ref->buf_a;
    for (int t = 0; t < n_tokens; t++) {
        int id = semantic_ids[t];
        if (id < 0 || id >= ref->semantic_vocab_size + 4) id = 0;
        const float *emb = ref->sem_emb + id * enc_D;
        const float *pos = ref->sem_pos + t * enc_D;
        for (int d = 0; d < enc_D; d++)
            enc_in[t * enc_D + d] = emb[d] + pos[d];
    }
    encoder_forward(ref, enc_in, n_tokens, ref->enc_out);

    /* KV cache for decoder self-attention: [max_len][n_kv_heads][head_dim] per layer */
    int n_kv = ref->dec_n_kv_heads;
    int hd = ref->dec_head_dim;
    float *kv_cache = ref->dec_buf;

    /* Decoder autoregressive generation */
    int generated[MAX_TEXT_LEN];
    int gen_len = 0;
    generated[gen_len++] = TEXT_BOS_ID;

    float *dec_hidden = ref->buf_a;  /* current decoder hidden state (1 × dec_D) */
    float *dec_tmp = ref->buf_b;

    for (int step = 0; step < MAX_TEXT_LEN - 1; step++) {
        int pos = step;
        int token_id = generated[step];
        if (token_id < 0 || token_id >= text_V) token_id = 0;
        const float *emb = ref->text_emb + token_id * dec_D;
        memcpy(dec_hidden, emb, (size_t)dec_D * sizeof(float));

        int cache_len = step;

        for (int l = 0; l < ref->dec_n_layers; l++) {
            const DecLayerWeights *lw = &ref->dec_layers[l];
            float *kv_k = kv_cache + (size_t)l * 2 * MAX_TEXT_LEN * n_kv * hd;
            float *kv_v = kv_cache + (size_t)l * 2 * MAX_TEXT_LEN * n_kv * hd +
                         MAX_TEXT_LEN * n_kv * hd;

            float *sa_out = dec_tmp;
            decoder_self_attn_step(sa_out, dec_hidden, lw, kv_k, kv_v, cache_len, pos,
                                  ref->rope_cos, ref->rope_sin,
                                  dec_tmp + dec_D, dec_tmp + dec_D * 2,
                                  dec_tmp + dec_D * 3, dec_D,
                                  ref->dec_n_heads, ref->dec_n_kv_heads, hd);
            for (int i = 0; i < dec_D; i++) dec_hidden[i] += sa_out[i];

            float *ca_out = dec_tmp;
            decoder_cross_attn_step(ca_out, dec_hidden, ref->enc_out, n_tokens, lw,
                                   dec_tmp + dec_D, dec_tmp + dec_D * 2,
                                   dec_D, ref->dec_n_heads, ref->dec_n_kv_heads, hd);
            for (int i = 0; i < dec_D; i++) dec_hidden[i] += ca_out[i];

            float *ff_out = dec_tmp;
            decoder_ffn_step(ff_out, dec_hidden, lw, dec_tmp + dec_D,
                            dec_tmp + dec_D * 2, ref->silu_ws, dec_D, ref->dec_d_ff);
            for (int i = 0; i < dec_D; i++) dec_hidden[i] += ff_out[i];
        }
        rms_norm(dec_tmp, dec_hidden, ref->dec_norm_w, dec_D, 1e-5f);
        float *logits = ref->buf_a;  /* NOT dec_buf — that's the KV cache */
        cblas_sgemv(CblasRowMajor, CblasNoTrans, text_V, dec_D,
                    1.0f, ref->output_proj_w, dec_D, dec_tmp, 1, 0.0f, logits, 1);

        int next = 0;
        float best = logits[0];
        for (int v = 1; v < text_V; v++) {
            if (logits[v] > best) { best = logits[v]; next = v; }
        }
        generated[gen_len++] = next;
        if (next == TEXT_EOS_ID) break;
    }

    /* Decode token IDs to string */
    int out_pos = 0;
    for (int i = 1; i < gen_len && out_pos < max_len - 1; i++) {
        int id = generated[i];
        if (id == TEXT_EOS_ID) break;
        char c = token_to_char(id);
        if (c != '\0') out_text[out_pos++] = c;
    }
    out_text[out_pos] = '\0';
    return out_pos;
}

int sonata_refiner_vocab_size(const SonataRefiner *ref) {
    return ref ? (ref->text_vocab_size + 4) : 0;
}
