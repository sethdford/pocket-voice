/**
 * conformer_stt.c — Pure C Conformer CTC STT engine for Apple Silicon.
 *
 * Implements a FastConformer encoder with CTC greedy decoding. Every matrix
 * multiply runs on the AMX coprocessor via cblas_sgemm. Element-wise ops
 * use vDSP and vecLib (vvexpf, vvlog10f, etc). Depthwise convolutions
 * are hand-rolled with NEON intrinsics.
 *
 * Architecture (per block):
 *   input → FFN½ → MHSA → Conv → FFN½ → LayerNorm → output
 *
 * Supports NeMo FastConformer models (dw_striding subsampling, relative PE)
 * as well as generic Conv1D-based conformers.
 *
 * The model weights are mmap'd from a .cstt file for zero-copy access.
 * On Apple Silicon, mmap'd memory is accessible from both CPU and GPU
 * through unified memory — no copies needed.
 *
 * Build:
 *   cc -O3 -shared -fPIC -arch arm64 -framework Accelerate \
 *      -L$(BUILD) -lmel_spectrogram \
 *      -install_name @rpath/libconformer_stt.dylib \
 *      -o libconformer_stt.dylib conformer_stt.c
 */

#include "conformer_stt.h"
#include "mel_spectrogram.h"

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Constants and limits
 * ═══════════════════════════════════════════════════════════════════════════ */

#define MAX_SEQ_LEN     2048
#define MAX_D_MODEL     1024
#define MAX_VOCAB       8192
#define MAX_TRANSCRIPT  16384
#define CHUNK_FRAMES    400    /* Mel frames per chunk (~4s at 10ms hop) */
#define MAX_SUB_CONVS   6      /* Max conv layers in subsampling */

/* ═══════════════════════════════════════════════════════════════════════════
 * Weight pointers for one Conformer block
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const float *ff1_norm_w, *ff1_norm_b;
    const float *ff1_up_w, *ff1_up_b;
    const float *ff1_down_w, *ff1_down_b;

    const float *attn_norm_w, *attn_norm_b;
    const float *attn_q_w, *attn_q_b;
    const float *attn_k_w, *attn_k_b;
    const float *attn_v_w, *attn_v_b;
    const float *attn_out_w, *attn_out_b;
    const float *attn_pos_w;                   /* [D, D] — positional projection (no bias) */
    const float *attn_pos_bias_u;              /* [n_heads, d_head] — relative PE */
    const float *attn_pos_bias_v;              /* [n_heads, d_head] — relative PE */

    const float *conv_norm_w, *conv_norm_b;
    const float *conv_pw1_w, *conv_pw1_b;
    const float *conv_dw_w, *conv_dw_b;
    const float *conv_bn_gamma, *conv_bn_beta;
    const float *conv_bn_mean, *conv_bn_var;
    const float *conv_pw2_w, *conv_pw2_b;

    const float *ff2_norm_w, *ff2_norm_b;
    const float *ff2_up_w, *ff2_up_b;
    const float *ff2_down_w, *ff2_down_b;

    const float *final_norm_w, *final_norm_b;
} ConformerBlockWeights;

/* Subsampling weight pointers: supports both Conv1D and NeMo dw_striding */
typedef struct {
    int n_convs;                                   /* Number of conv layers */
    struct {
        const float *w, *b;
        int c_in, c_out, kernel, stride, groups;   /* Conv2D params */
    } convs[MAX_SUB_CONVS];
    const float *proj_w, *proj_b;                  /* Final linear projection */
    int proj_in, proj_out;
} SubsamplingWeights;

typedef struct {
    SubsamplingWeights sub;
    ConformerBlockWeights *blocks;
    int n_blocks;
    const float *ctc_w, *ctc_b;
} ModelWeights;

/* ═══════════════════════════════════════════════════════════════════════════
 * Working memory
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float *buf_a;
    float *buf_b;
    float *residual;
    float *qkv;
    float *attn_scores;
    float *ff_mid;
    float *conv_mid;
    float *logits;
    float *sub_work;       /* Large buffer for subsampling intermediates */
    float *rel_pe;         /* Pre-computed sinusoidal relative PE [2*MAX_SEQ_LEN, d_head] */
} Workspace;

typedef struct {
    char **tokens;
    int    size;
    int    blank_id;
} Vocabulary;

/* ═══════════════════════════════════════════════════════════════════════════
 * Per-layer activation cache for cache-aware streaming.
 * Stores K/V projections and depthwise conv overlap from previous chunks.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define CACHE_MAX_CONTEXT 70   /* Parakeet-style: [70, 1] attention context */

typedef struct {
    float *k_cache;            /* [CACHE_MAX_CONTEXT, D] — cached K projections */
    float *v_cache;            /* [CACHE_MAX_CONTEXT, D] — cached V projections */
    int    k_len;              /* Current number of cached K frames */
    int    v_len;              /* Current number of cached V frames */
    float *conv_state;         /* [conv_kernel-1, D] — depthwise conv overlap */
    int    conv_state_len;     /* Stored overlap frames */
} LayerCache;

struct ConformerSTT {
    CSTTHeader header;
    ModelWeights weights;
    Workspace work;
    Vocabulary vocab;
    MelSpectrogram *mel;

    void *mmap_base;
    size_t mmap_size;
    int mmap_fd;

    float *mel_accum;
    int    mel_accum_len;
    int    mel_accum_cap;

    char   transcript[MAX_TRANSCRIPT];
    int    transcript_len;
    int    prev_token;

    /* EOU detection state */
    int    eou_token_id;       /* Vocab index of <eou>/<|endofutterance|> token, or -1 */
    int    eou_detected;       /* 1 if EOU was detected in last decode */
    int    eou_frame;          /* Frame index of last EOU detection, or -1 */
    float  eou_prob;           /* Trailing average EOU probability */
    float *eou_probs;          /* Per-frame EOU probabilities from last forward pass */
    int    eou_probs_len;      /* Number of frames in eou_probs */
    int    eou_probs_cap;      /* Capacity of eou_probs */

    /* Cache-aware streaming */
    int          cache_aware;  /* 1 = caching enabled */
    LayerCache  *layer_caches; /* [n_layers] per-layer caches */
    int          total_frames_processed; /* Running count for PE offset */
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Low-level neural network operations — all via Accelerate (AMX/vDSP)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void layer_norm(float *out, const float *in,
                       const float *gamma, const float *beta,
                       int T, int D) {
    const float eps = 1e-5f;
    for (int t = 0; t < T; t++) {
        const float *x = in + t * D;
        float *y = out + t * D;
        float mean;
        vDSP_meanv(x, 1, &mean, D);
        float neg_mean = -mean;
        vDSP_vsadd(x, 1, &neg_mean, y, 1, D);
        float var;
        vDSP_measqv(y, 1, &var, D);
        float inv_std = 1.0f / sqrtf(var + eps);
        vDSP_vsmul(y, 1, &inv_std, y, 1, D);
        vDSP_vma(y, 1, gamma, 1, beta, 1, y, 1, D);
    }
}

static void linear(float *out, const float *in, const float *W,
                   const float *bias, int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, in, K, W, K,
                0.0f, out, N);
    if (bias) {
        for (int m = 0; m < M; m++)
            vDSP_vadd(out + m * N, 1, bias, 1, out + m * N, 1, N);
    }
}

static void silu_inplace(float *x, int N) {
    float *tmp = (float *)malloc(N * sizeof(float));
    float neg = -1.0f;
    vDSP_vsmul(x, 1, &neg, tmp, 1, N);
    int n = N;
    vvexpf(tmp, tmp, &n);
    float one = 1.0f;
    vDSP_vsadd(tmp, 1, &one, tmp, 1, N);
    vDSP_vdiv(tmp, 1, x, 1, x, 1, N);
    free(tmp);
}

static void glu(float *out, const float *in, int T, int D) {
    for (int t = 0; t < T; t++) {
        const float *a = in + t * 2 * D;
        const float *b = a + D;
        float *o = out + t * D;
        float neg = -1.0f;
        vDSP_vsmul(b, 1, &neg, o, 1, D);
        int n = D;
        vvexpf(o, o, &n);
        float one = 1.0f;
        vDSP_vsadd(o, 1, &one, o, 1, D);
        vDSP_svdiv(&one, o, 1, o, 1, D);
        vDSP_vmul(a, 1, o, 1, o, 1, D);
    }
}

static void softmax_row(float *x, int N) {
    float max_val;
    vDSP_maxv(x, 1, &max_val, N);
    float neg_max = -max_val;
    vDSP_vsadd(x, 1, &neg_max, x, 1, N);
    int n = N;
    vvexpf(x, x, &n);
    float sum;
    vDSP_sve(x, 1, &sum, N);
    if (sum > 0.0f)
        vDSP_vsdiv(x, 1, &sum, x, 1, N);
}

static void relu_inplace(float *x, int N) {
    float zero = 0.0f;
    vDSP_vthres(x, 1, &zero, x, 1, N);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Sinusoidal positional encoding
 * ═══════════════════════════════════════════════════════════════════════════ */

static void add_sinusoidal_pe(float *x, int T, int D) {
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < D; d += 2) {
            float freq = 1.0f / powf(10000.0f, (float)d / D);
            float angle = (float)t * freq;
            x[t * D + d]     += sinf(angle);
            if (d + 1 < D)
                x[t * D + d + 1] += cosf(angle);
        }
    }
}

/**
 * Pre-compute sinusoidal relative positional encoding table.
 * Table: [2 * max_len, d_model], where row i encodes relative position i-max_len.
 */
static void precompute_rel_pe(float *table, int max_len, int D) {
    int total = 2 * max_len;
    for (int i = 0; i < total; i++) {
        float pos = (float)(i - max_len);
        for (int d = 0; d < D; d += 2) {
            float freq = 1.0f / powf(10000.0f, (float)d / D);
            table[i * D + d] = sinf(pos * freq);
            if (d + 1 < D)
                table[i * D + d + 1] = cosf(pos * freq);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Depthwise 1D convolution (for conformer conv module)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void depthwise_conv1d(float *out, const float *in,
                             const float *kernel, const float *bias,
                             int T, int D, int K) {
    int pad_left = K - 1;
    for (int d = 0; d < D; d++) {
        for (int t = 0; t < T; t++) {
            float sum = bias ? bias[d] : 0.0f;
            for (int k = 0; k < K; k++) {
                int src_t = t - pad_left + k;
                if (src_t >= 0 && src_t < T)
                    sum += in[src_t * D + d] * kernel[d * K + k];
            }
            out[t * D + d] = sum;
        }
    }
}

static void batch_norm(float *out, const float *in,
                       const float *gamma, const float *beta,
                       const float *running_mean, const float *running_var,
                       int T, int D) {
    const float eps = 1e-5f;
    float *scale = (float *)malloc(D * sizeof(float));
    float *shift = (float *)malloc(D * sizeof(float));
    for (int d = 0; d < D; d++) {
        scale[d] = gamma[d] / sqrtf(running_var[d] + eps);
        shift[d] = beta[d] - scale[d] * running_mean[d];
    }
    for (int t = 0; t < T; t++)
        vDSP_vma(in + t * D, 1, scale, 1, shift, 1, out + t * D, 1, D);
    free(scale);
    free(shift);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Conv2D operations for subsampling
 *
 * Data layout: [C, T, F] (channels-first, matching PyTorch)
 * Weight layout: [C_out, C_in, Kh, Kw] (PyTorch convention)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void conv2d_forward(float *out, const float *in,
                           const float *W, const float *bias,
                           int C_in, int C_out, int T_in, int F_in,
                           int Kh, int Kw, int Sh, int Sw, int groups) {
    int pad_h = Kh / 2, pad_w = Kw / 2;
    int T_out = (T_in + 2 * pad_h - Kh) / Sh + 1;
    int F_out = (F_in + 2 * pad_w - Kw) / Sw + 1;
    int c_per_group_in  = C_in / groups;
    int c_per_group_out = C_out / groups;

    for (int co = 0; co < C_out; co++) {
        int g = co / c_per_group_out;
        float b = bias ? bias[co] : 0.0f;
        for (int t = 0; t < T_out; t++) {
            for (int f = 0; f < F_out; f++) {
                float sum = b;
                for (int ci = 0; ci < c_per_group_in; ci++) {
                    int ci_abs = g * c_per_group_in + ci;
                    for (int kh = 0; kh < Kh; kh++) {
                        int tt = t * Sh - pad_h + kh;
                        if (tt < 0 || tt >= T_in) continue;
                        for (int kw = 0; kw < Kw; kw++) {
                            int ff = f * Sw - pad_w + kw;
                            if (ff < 0 || ff >= F_in) continue;
                            sum += in[ci_abs * T_in * F_in + tt * F_in + ff]
                                 * W[co * c_per_group_in * Kh * Kw + ci * Kh * Kw + kh * Kw + kw];
                        }
                    }
                }
                out[co * T_out * F_out + t * F_out + f] = sum;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Subsampling: dispatch based on header sub_type
 *
 * Converts [T, n_mels] mel frames → [T/factor, D] encoder input.
 * ═══════════════════════════════════════════════════════════════════════════ */

static int conv1d_subsample_forward(float *out, const float *mel_in, int T,
                                    const SubsamplingWeights *sw, int n_mels, int D) {
    int K = sw->convs[0].kernel;
    int pad = K / 2;

    int T1 = (T + 2 * pad - K) / 2 + 1;
    float *mid = (float *)calloc((size_t)T1 * D, sizeof(float));
    if (!mid) return -1;

    /* Conv1: [T, n_mels] → [T/2, D] with ReLU (reuse conv2d as 1D: F=1) */
    /* Simple loop-based Conv1D stride 2 */
    for (int t = 0; t < T1; t++) {
        int t_in = t * 2 - pad;
        for (int co = 0; co < D; co++) {
            float sum = sw->convs[0].b ? sw->convs[0].b[co] : 0.0f;
            for (int k = 0; k < K; k++) {
                int tt = t_in + k;
                if (tt >= 0 && tt < T) {
                    for (int ci = 0; ci < n_mels; ci++)
                        sum += mel_in[tt * n_mels + ci]
                             * sw->convs[0].w[co * n_mels * K + ci * K + k];
                }
            }
            mid[t * D + co] = sum > 0.0f ? sum : 0.0f;
        }
    }

    /* Conv2: [T/2, D] → [T/4, D] with ReLU */
    int T2 = (T1 + 2 * pad - K) / 2 + 1;
    for (int t = 0; t < T2; t++) {
        int t_in = t * 2 - pad;
        for (int co = 0; co < D; co++) {
            float sum = sw->convs[1].b ? sw->convs[1].b[co] : 0.0f;
            for (int k = 0; k < K; k++) {
                int tt = t_in + k;
                if (tt >= 0 && tt < T1) {
                    for (int ci = 0; ci < D; ci++)
                        sum += mid[tt * D + ci]
                             * sw->convs[1].w[co * D * K + ci * K + k];
                }
            }
            out[t * D + co] = sum > 0.0f ? sum : 0.0f;
        }
    }
    free(mid);

    /* Linear projection */
    if (sw->proj_w) {
        float *tmp = (float *)malloc((size_t)T2 * D * sizeof(float));
        if (!tmp) return -1;
        linear(tmp, out, sw->proj_w, sw->proj_b, T2, sw->proj_in, sw->proj_out);
        memcpy(out, tmp, (size_t)T2 * sw->proj_out * sizeof(float));
        free(tmp);
    }
    return T2;
}

/**
 * NeMo dw_striding subsampling:
 *   For each stage: Conv2D(s=1) → ReLU → DW_Conv2D(s=2) → PW_Conv2D(s=1)
 *   Then: flatten [D, T/factor, F/factor] → [T/factor, D * F/factor]
 *   Then: Linear(D * F/factor, D)
 */
static int dw_striding_forward(float *out, const float *mel_in, int T,
                               const SubsamplingWeights *sw, int n_mels, int D,
                               float *work) {
    /* NeMo convention: [C, H=n_mels, W=T] — mels as height, time as width.
     * Input mel_in is [T, n_mels] (row-major). Transpose to [1, n_mels, T]. */
    int C_cur = 1, T_cur = n_mels, F_cur = T;

    float *cur  = work;
    size_t max_elems = (size_t)D * T * n_mels;
    float *next = work + max_elems;
    for (int t = 0; t < T; t++)
        for (int f = 0; f < n_mels; f++)
            cur[f * T + t] = mel_in[t * n_mels + f];

    /* Process conv layers in pairs/triples as defined by the subsampling config */
    for (int i = 0; i < sw->n_convs; i++) {
        int c_in  = sw->convs[i].c_in;
        int c_out = sw->convs[i].c_out;
        int K     = sw->convs[i].kernel;
        int S     = sw->convs[i].stride;
        int G     = sw->convs[i].groups;

        int pad_h = K / 2, pad_w = K / 2;
        int T_next = (T_cur + 2 * pad_h - K) / S + 1;
        int F_next = (F_cur + 2 * pad_w - K) / S + 1;

#ifdef CSTT_DEBUG
        fprintf(stderr, "[debug] sub conv[%d]: cin=%d cout=%d K=%d S=%d G=%d "
                "in=[%d,%d,%d] → out=[%d,%d,%d]\n",
                i, c_in, c_out, K, S, G, C_cur, T_cur, F_cur, c_out, T_next, F_next);
        {
            float mn = cur[0], mx = cur[0], sm = 0;
            int nn = C_cur * T_cur * F_cur;
            for (int j = 0; j < nn; j++) {
                if (cur[j] < mn) mn = cur[j];
                if (cur[j] > mx) mx = cur[j];
                sm += cur[j];
            }
            fprintf(stderr, "[debug]   input: min=%.4f max=%.4f mean=%.6f\n",
                    mn, mx, sm / nn);
            /* weight stats */
            int wn = c_out * (c_in / G) * K * K;
            float wmn = sw->convs[i].w[0], wmx = sw->convs[i].w[0], wsm = 0;
            for (int j = 0; j < wn; j++) {
                if (sw->convs[i].w[j] < wmn) wmn = sw->convs[i].w[j];
                if (sw->convs[i].w[j] > wmx) wmx = sw->convs[i].w[j];
                wsm += sw->convs[i].w[j];
            }
            fprintf(stderr, "[debug]   weight: n=%d min=%.4f max=%.4f mean=%.6f\n",
                    wn, wmn, wmx, wsm / wn);
        }
#endif

        conv2d_forward(next, cur, sw->convs[i].w, sw->convs[i].b,
                       c_in, c_out, T_cur, F_cur, K, K, S, S, G);

        /* Apply ReLU after non-depthwise conv layers (regular and pointwise).
         * NeMo dw_striding pattern: regular→ReLU, dw→pw→ReLU, dw→pw→ReLU */
        if (G == 1)
            relu_inplace(next, c_out * T_next * F_next);

#ifdef CSTT_DEBUG
        {
            float mn = next[0], mx = next[0], sm = 0;
            int nn = c_out * T_next * F_next;
            for (int j = 0; j < nn; j++) {
                if (next[j] < mn) mn = next[j];
                if (next[j] > mx) mx = next[j];
                sm += next[j];
            }
            fprintf(stderr, "[debug]   output: min=%.4f max=%.4f mean=%.6f\n",
                    mn, mx, sm / nn);
        }
#endif

        /* Swap buffers */
        float *tmp = cur; cur = next; next = tmp;
        C_cur = c_out;
        T_cur = T_next;
        F_cur = F_next;
    }

    /* After convolutions: [C, H=mels', W=T'] (NeMo layout).
     * NeMo does: view(B, C*H, W) → transpose(1,2) → [B, W, C*H] = [T', C*mels'].
     * Here: C_cur=C, T_cur=mels' (H dim), F_cur=T' (W dim = actual time). */
    int T_actual = F_cur;     /* Time frames after subsampling */
    int mels_reduced = T_cur; /* Reduced mel dimension */
    int feat_dim = C_cur * mels_reduced;
    for (int t = 0; t < T_actual; t++) {
        for (int c = 0; c < C_cur; c++) {
            for (int m = 0; m < mels_reduced; m++) {
                next[t * feat_dim + c * mels_reduced + m] =
                    cur[c * mels_reduced * T_actual + m * T_actual + t];
            }
        }
    }

    /* Linear projection: [T_actual, feat_dim] → [T_actual, D] */
    linear(out, next, sw->proj_w, sw->proj_b, T_actual, sw->proj_in, sw->proj_out);

    return T_actual;
}

static int subsample_forward(float *out, const float *mel_in, int T,
                             const ConformerSTT *stt) {
    int D = (int)stt->header.d_model;
    int n_mels = (int)stt->header.n_mels;
    int sub_type = (int)stt->header.sub_type;
    const SubsamplingWeights *sw = &stt->weights.sub;

    switch (sub_type) {
    case CSTT_SUB_DW_STRIDING:
    case CSTT_SUB_CONV2D:
        return dw_striding_forward(out, mel_in, T, sw, n_mels, D,
                                   stt->work.sub_work);
    default:
        return conv1d_subsample_forward(out, mel_in, T, sw, n_mels, D);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Multi-Head Self-Attention (with optional relative positional encoding)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void mhsa_forward(float *out, const float *in,
                         const ConformerBlockWeights *bw,
                         Workspace *ws, int T, int D, int n_heads,
                         int use_rel_pe) {
    int d_head = D / n_heads;

    float *Q = ws->qkv;
    float *K = ws->qkv + T * D;
    float *V = ws->qkv + T * 2 * D;

    linear(Q, in, bw->attn_q_w, bw->attn_q_b, T, D, D);
    linear(K, in, bw->attn_k_w, bw->attn_k_b, T, D, D);
    linear(V, in, bw->attn_v_w, bw->attn_v_b, T, D, D);

    float scale = 1.0f / sqrtf((float)d_head);
    float *attn_out = ws->buf_b;

    for (int h = 0; h < n_heads; h++) {
        float *Qh = ws->attn_scores;
        float *Kh = ws->attn_scores + T * d_head;
        float *Vh = ws->attn_scores + T * d_head * 2;
        float *scores = ws->attn_scores + T * d_head * 3;

        for (int t = 0; t < T; t++) {
            memcpy(Qh + t * d_head, Q + t * D + h * d_head, d_head * sizeof(float));
            memcpy(Kh + t * d_head, K + t * D + h * d_head, d_head * sizeof(float));
            memcpy(Vh + t * d_head, V + t * D + h * d_head, d_head * sizeof(float));
        }

        /* Add pos_bias_u to Q for content-based attention */
        if (use_rel_pe && bw->attn_pos_bias_u) {
            const float *bias_u = bw->attn_pos_bias_u + h * d_head;
            for (int t = 0; t < T; t++)
                vDSP_vadd(Qh + t * d_head, 1, bias_u, 1, Qh + t * d_head, 1, d_head);
        }

        /* Content attention: (Q + pos_bias_u) @ K^T */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, T, d_head,
                    scale, Qh, d_head, Kh, d_head,
                    0.0f, scores, T);

        /* Relative positional attention with linear_pos projection.
         * NeMo: P = linear_pos(pos_emb), then score_pos = (Q + bias_v) @ P^T
         * with relative position indexing (equivalent to rel_shift). */
        if (use_rel_pe && bw->attn_pos_bias_v && ws->rel_pe && bw->attn_pos_w) {
            /* Project PE through linear_pos: P = PE @ W_pos^T
             * PE is [2*MAX_SEQ_LEN, D], W_pos is [D, D], P is [2*MAX_SEQ_LEN, D].
             * We only need PE entries for relative positions -(T-1) to +(T-1),
             * centered at MAX_SEQ_LEN. That's 2*T-1 entries. */
            int pe_len = 2 * T - 1;
            int pe_start = MAX_SEQ_LEN - (T - 1);
            float *P_proj = (float *)malloc((size_t)pe_len * D * sizeof(float));
            if (P_proj) {
                /* P_proj = PE_slice @ W_pos^T  (W_pos is [D, D]) */
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            pe_len, D, D,
                            1.0f, ws->rel_pe + pe_start * D, D,
                            bw->attn_pos_w, D,
                            0.0f, P_proj, D);

                /* Extract head h's slice of projected PE: P_h[i, :] = P_proj[i, h*d_head:(h+1)*d_head] */
                float *Qv = Vh; /* reuse Vh temporarily */
                for (int t = 0; t < T; t++) {
                    memcpy(Qv + t * d_head, Q + t * D + h * d_head, d_head * sizeof(float));
                    const float *bias_v = bw->attn_pos_bias_v + h * d_head;
                    vDSP_vadd(Qv + t * d_head, 1, bias_v, 1, Qv + t * d_head, 1, d_head);
                }

                /* Compute position scores: for each (i, j), dot(Q_v_h[i], P_h[i-j+(T-1)]) */
                for (int i = 0; i < T; i++) {
                    for (int j = 0; j < T; j++) {
                        int pe_idx = i - j + (T - 1);
                        const float *pe_h = P_proj + pe_idx * D + h * d_head;
                        float dot = 0.0f;
                        vDSP_dotpr(Qv + i * d_head, 1, pe_h, 1, &dot, d_head);
                        scores[i * T + j] += dot * scale;
                    }
                }

                free(P_proj);

                /* Restore Vh */
                for (int t = 0; t < T; t++)
                    memcpy(Vh + t * d_head, V + t * D + h * d_head, d_head * sizeof(float));
            }
        }

        for (int t = 0; t < T; t++)
            softmax_row(scores + t * T, T);

        float *ctx = Qh;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, d_head, T,
                    1.0f, scores, T, Vh, d_head,
                    0.0f, ctx, d_head);

        for (int t = 0; t < T; t++)
            memcpy(attn_out + t * D + h * d_head, ctx + t * d_head,
                   d_head * sizeof(float));
    }

    linear(out, attn_out, bw->attn_out_w, bw->attn_out_b, T, D, D);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Convolution module
 * ═══════════════════════════════════════════════════════════════════════════ */

static int conv_debug_count = 0;

static void conv_module_forward(float *out, const float *in,
                                const ConformerBlockWeights *bw,
                                Workspace *ws, int T, int D, int K) {
    int dbg = 0;
#ifdef CSTT_DEBUG
    dbg = (conv_debug_count++ == 0);
#endif
    float *normed = ws->buf_b;
    layer_norm(normed, in, bw->conv_norm_w, bw->conv_norm_b, T, D);
    if (dbg) fprintf(stderr, "[conv0] after LN f0[:4]=%.4f,%.4f,%.4f,%.4f\n",
                     normed[0], normed[1], normed[2], normed[3]);

    linear(ws->conv_mid, normed, bw->conv_pw1_w, bw->conv_pw1_b, T, D, 2 * D);
    if (dbg) fprintf(stderr, "[conv0] after PW1 f0[:4]=%.4f,%.4f,%.4f,%.4f (2D: %.4f,%.4f)\n",
                     ws->conv_mid[0], ws->conv_mid[1], ws->conv_mid[2], ws->conv_mid[3],
                     ws->conv_mid[D], ws->conv_mid[D+1]);

    glu(normed, ws->conv_mid, T, D);
    if (dbg) fprintf(stderr, "[conv0] after GLU f0[:4]=%.4f,%.4f,%.4f,%.4f\n",
                     normed[0], normed[1], normed[2], normed[3]);

    depthwise_conv1d(out, normed, bw->conv_dw_w, bw->conv_dw_b, T, D, K);
    if (dbg) fprintf(stderr, "[conv0] after DW f0[:4]=%.4f,%.4f,%.4f,%.4f\n",
                     out[0], out[1], out[2], out[3]);

    batch_norm(normed, out, bw->conv_bn_gamma, bw->conv_bn_beta,
               bw->conv_bn_mean, bw->conv_bn_var, T, D);
    if (dbg) fprintf(stderr, "[conv0] after BN f0[:4]=%.4f,%.4f,%.4f,%.4f\n",
                     normed[0], normed[1], normed[2], normed[3]);

    silu_inplace(normed, T * D);
    if (dbg) fprintf(stderr, "[conv0] after SiLU f0[:4]=%.4f,%.4f,%.4f,%.4f\n",
                     normed[0], normed[1], normed[2], normed[3]);

    linear(out, normed, bw->conv_pw2_w, bw->conv_pw2_b, T, D, D);
    if (dbg) fprintf(stderr, "[conv0] after PW2 f0[:4]=%.4f,%.4f,%.4f,%.4f\n",
                     out[0], out[1], out[2], out[3]);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Feed-Forward module (Macaron-style half-step)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void ffn_forward(float *out, const float *in,
                        const float *norm_w, const float *norm_b,
                        const float *up_w, const float *up_b,
                        const float *down_w, const float *down_b,
                        Workspace *ws, int T, int D, int ff_dim) {
    float *normed = ws->buf_b;
    layer_norm(normed, in, norm_w, norm_b, T, D);
    linear(ws->ff_mid, normed, up_w, up_b, T, D, ff_dim);
    silu_inplace(ws->ff_mid, T * ff_dim);
    linear(out, ws->ff_mid, down_w, down_b, T, ff_dim, D);
    float half = 0.5f;
    vDSP_vsmul(out, 1, &half, out, 1, T * D);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Single Conformer block
 * ═══════════════════════════════════════════════════════════════════════════ */

static void conformer_block_forward(float *x, const ConformerBlockWeights *bw,
                                    Workspace *ws, int T, int D,
                                    int n_heads, int ff_mult, int conv_kernel,
                                    int use_rel_pe) {
    static int block_call_count = 0;
    int block_idx = block_call_count++;
    int ff_dim = D * ff_mult;
    float *tmp = ws->residual;

    ffn_forward(tmp, x, bw->ff1_norm_w, bw->ff1_norm_b,
                bw->ff1_up_w, bw->ff1_up_b,
                bw->ff1_down_w, bw->ff1_down_b,
                ws, T, D, ff_dim);
#ifdef CSTT_DEBUG
    if (block_idx == 0) {
        fprintf(stderr, "[blk0] FFN1 raw f0[:4]=%.4f,%.4f,%.4f,%.4f\n",
                tmp[0], tmp[1], tmp[2], tmp[3]);
    }
#endif
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);
#ifdef CSTT_DEBUG
    if (block_idx == 0) {
        fprintf(stderr, "[blk0] After FFN1 f0[:4]=%.4f,%.4f,%.4f,%.4f\n",
                x[0], x[1], x[2], x[3]);
    }
#endif

    float *attn_in = ws->buf_b;
    layer_norm(attn_in, x, bw->attn_norm_w, bw->attn_norm_b, T, D);
    mhsa_forward(tmp, attn_in, bw, ws, T, D, n_heads, use_rel_pe);
#ifdef CSTT_DEBUG
    if (block_idx == 0) {
        fprintf(stderr, "[blk0] MHSA out f0[:4]=%.4f,%.4f,%.4f,%.4f\n",
                tmp[0], tmp[1], tmp[2], tmp[3]);
    }
#endif
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);
#ifdef CSTT_DEBUG
    if (block_idx == 0) {
        fprintf(stderr, "[blk0] After MHSA f0[:4]=%.4f,%.4f,%.4f,%.4f\n",
                x[0], x[1], x[2], x[3]);
    }
#endif

    conv_module_forward(tmp, x, bw, ws, T, D, conv_kernel);
#ifdef CSTT_DEBUG
    if (block_idx == 0) {
        fprintf(stderr, "[blk0] Conv out f0[:4]=%.4f,%.4f,%.4f,%.4f\n",
                tmp[0], tmp[1], tmp[2], tmp[3]);
    }
#endif
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);
#ifdef CSTT_DEBUG
    if (block_idx == 0) {
        fprintf(stderr, "[blk0] After Conv f0[:4]=%.4f,%.4f,%.4f,%.4f\n",
                x[0], x[1], x[2], x[3]);
    }
#endif

    ffn_forward(tmp, x, bw->ff2_norm_w, bw->ff2_norm_b,
                bw->ff2_up_w, bw->ff2_up_b,
                bw->ff2_down_w, bw->ff2_down_b,
                ws, T, D, ff_dim);
#ifdef CSTT_DEBUG
    if (block_idx == 0) {
        fprintf(stderr, "[blk0] FFN2 raw f0[:4]=%.4f,%.4f,%.4f,%.4f\n",
                tmp[0], tmp[1], tmp[2], tmp[3]);
    }
#endif
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);

    layer_norm(x, x, bw->final_norm_w, bw->final_norm_b, T, D);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CTC Greedy Decoder
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * CTC greedy decode with EOU token detection.
 *
 * When eou_token_id >= 0, the decoder tracks:
 *   - Whether <eou> was emitted (sets *eou_detected = 1)
 *   - The frame at which <eou> was first detected (*eou_frame)
 *   - Per-frame softmax probabilities of the <eou> token (eou_probs)
 *
 * The <eou> token is NOT emitted into the text output.
 */
static int ctc_greedy_decode(char *out, int out_cap,
                             const float *logits, int T, int vocab_size,
                             const Vocabulary *vocab, int *prev_token,
                             int eou_token_id,
                             int *eou_detected, int *eou_frame,
                             float *eou_probs) {
    int written = 0;
    if (eou_detected) *eou_detected = 0;
    if (eou_frame) *eou_frame = -1;

    for (int t = 0; t < T; t++) {
        const float *row = logits + t * vocab_size;

        /* Extract EOU probability via softmax approximation.
         * Full softmax is expensive; we use the log-sum-exp trick
         * to get P(eou) = exp(logit_eou) / sum(exp(logits)). */
        if (eou_probs && eou_token_id >= 0 && eou_token_id < vocab_size) {
            float max_logit = row[0];
            for (int v = 1; v < vocab_size; v++)
                if (row[v] > max_logit) max_logit = row[v];

            float sum_exp = 0.0f;
            for (int v = 0; v < vocab_size; v++)
                sum_exp += expf(row[v] - max_logit);

            eou_probs[t] = expf(row[eou_token_id] - max_logit) / sum_exp;
        }

        /* Find best token */
        float best_val = row[0];
        int best_idx = 0;
        for (int v = 1; v < vocab_size; v++) {
            if (row[v] > best_val) { best_val = row[v]; best_idx = v; }
        }

        /* EOU detection: if best token is <eou>, record it */
        if (eou_token_id >= 0 && best_idx == eou_token_id && best_idx != *prev_token) {
            if (eou_detected) *eou_detected = 1;
            if (eou_frame && *eou_frame < 0) *eou_frame = t;
            *prev_token = best_idx;
            continue; /* Don't emit <eou> into text */
        }

        if (best_idx == vocab->blank_id || best_idx == *prev_token) {
            if (best_idx != *prev_token) *prev_token = best_idx;
            continue;
        }
        *prev_token = best_idx;
        if (best_idx >= 0 && best_idx < vocab->size && vocab->tokens[best_idx]) {
            const char *tok = vocab->tokens[best_idx];
            int tok_len = (int)strlen(tok);
            if (written + tok_len < out_cap) {
                memcpy(out + written, tok, tok_len);
                written += tok_len;
            }
        }
    }
    if (written < out_cap) out[written] = '\0';
    return written;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Full forward pass
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Cache-aware depthwise conv: prepends cached overlap from previous chunk.
 * After convolution, saves the trailing (K-1) frames as the new overlap.
 */
static void depthwise_conv1d_cached(float *out, const float *in,
                                     const float *kernel, const float *bias,
                                     int T, int D, int K,
                                     LayerCache *cache) {
    if (!cache || cache->conv_state_len == 0) {
        depthwise_conv1d(out, in, kernel, bias, T, D, K);
    } else {
        /* Prepend cached overlap: [overlap | in] */
        int overlap = cache->conv_state_len;
        int total_T = overlap + T;
        float *merged = (float *)malloc((size_t)total_T * D * sizeof(float));
        memcpy(merged, cache->conv_state, (size_t)overlap * D * sizeof(float));
        memcpy(merged + overlap * D, in, (size_t)T * D * sizeof(float));

        /* Run conv on merged input (causal: pad_left = K-1) */
        float *full_out = (float *)malloc((size_t)total_T * D * sizeof(float));
        depthwise_conv1d(full_out, merged, kernel, bias, total_T, D, K);

        /* Take only the last T frames (corresponding to new input) */
        memcpy(out, full_out + overlap * D, (size_t)T * D * sizeof(float));

        free(merged);
        free(full_out);
    }

    /* Save trailing (K-1) frames as overlap for next chunk */
    if (cache) {
        int save = K - 1;
        if (save > T) save = T;
        if (!cache->conv_state) {
            cache->conv_state = (float *)malloc((size_t)(K - 1) * D * sizeof(float));
        }
        memcpy(cache->conv_state, in + (T - save) * D, (size_t)save * D * sizeof(float));
        cache->conv_state_len = save;
    }
}

/**
 * Cache-aware MHSA: uses cached K/V from previous chunks to extend context.
 * Only the new Q frames attend to [cached_K | new_K] and [cached_V | new_V].
 * After attention, the new K/V are appended to the cache (with eviction if full).
 */
static void mhsa_forward_cached(float *out, const float *in,
                                 const ConformerBlockWeights *bw,
                                 Workspace *ws, int T, int D, int n_heads,
                                 int use_rel_pe, LayerCache *cache) {
    if (!cache || !cache->k_cache) {
        mhsa_forward(out, in, bw, ws, T, D, n_heads, use_rel_pe);
        return;
    }

    int d_head = D / n_heads;
    int cached_T = cache->k_len;
    int total_T = cached_T + T;

    /* Project new Q, K, V */
    float *Q_new = ws->qkv;
    float *K_new = ws->qkv + T * D;
    float *V_new = ws->qkv + T * 2 * D;

    linear(Q_new, in, bw->attn_q_w, bw->attn_q_b, T, D, D);
    linear(K_new, in, bw->attn_k_w, bw->attn_k_b, T, D, D);
    linear(V_new, in, bw->attn_v_w, bw->attn_v_b, T, D, D);

    /* Build full K and V: [cached | new] */
    float *K_full = (float *)malloc((size_t)total_T * D * sizeof(float));
    float *V_full = (float *)malloc((size_t)total_T * D * sizeof(float));

    if (cached_T > 0) {
        memcpy(K_full, cache->k_cache, (size_t)cached_T * D * sizeof(float));
        memcpy(V_full, cache->v_cache, (size_t)cached_T * D * sizeof(float));
    }
    memcpy(K_full + cached_T * D, K_new, (size_t)T * D * sizeof(float));
    memcpy(V_full + cached_T * D, V_new, (size_t)T * D * sizeof(float));

    /* Per-head attention: Q[T] @ K_full[total_T]^T → scores[T, total_T] */
    float scale = 1.0f / sqrtf((float)d_head);
    float *attn_out = ws->buf_b;
    float *scores = (float *)malloc((size_t)T * total_T * sizeof(float));
    float *Qh = (float *)malloc((size_t)T * d_head * sizeof(float));
    float *Kh = (float *)malloc((size_t)total_T * d_head * sizeof(float));
    float *Vh = (float *)malloc((size_t)total_T * d_head * sizeof(float));
    float *ctx = (float *)malloc((size_t)T * d_head * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < T; t++)
            memcpy(Qh + t * d_head, Q_new + t * D + h * d_head, d_head * sizeof(float));
        for (int t = 0; t < total_T; t++) {
            memcpy(Kh + t * d_head, K_full + t * D + h * d_head, d_head * sizeof(float));
            memcpy(Vh + t * d_head, V_full + t * D + h * d_head, d_head * sizeof(float));
        }

        if (use_rel_pe && bw->attn_pos_bias_u) {
            const float *bias_u = bw->attn_pos_bias_u + h * d_head;
            for (int t = 0; t < T; t++)
                vDSP_vadd(Qh + t * d_head, 1, bias_u, 1, Qh + t * d_head, 1, d_head);
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, total_T, d_head,
                    scale, Qh, d_head, Kh, d_head,
                    0.0f, scores, total_T);

        for (int t = 0; t < T; t++)
            softmax_row(scores + t * total_T, total_T);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, d_head, total_T,
                    1.0f, scores, total_T, Vh, d_head,
                    0.0f, ctx, d_head);

        for (int t = 0; t < T; t++)
            memcpy(attn_out + t * D + h * d_head, ctx + t * d_head, d_head * sizeof(float));
    }

    linear(out, attn_out, bw->attn_out_w, bw->attn_out_b, T, D, D);

    /* Update cache: append new K/V, evict oldest if over capacity */
    int new_total = cached_T + T;
    if (new_total > CACHE_MAX_CONTEXT) {
        int evict = new_total - CACHE_MAX_CONTEXT;
        int keep = cached_T - evict;
        if (keep > 0) {
            memmove(cache->k_cache, cache->k_cache + evict * D, (size_t)keep * D * sizeof(float));
            memmove(cache->v_cache, cache->v_cache + evict * D, (size_t)keep * D * sizeof(float));
        }
        cached_T = keep > 0 ? keep : 0;
    }
    memcpy(cache->k_cache + cached_T * D, K_new, (size_t)T * D * sizeof(float));
    memcpy(cache->v_cache + cached_T * D, V_new, (size_t)T * D * sizeof(float));
    cache->k_len = cached_T + T;
    cache->v_len = cached_T + T;
    if (cache->k_len > CACHE_MAX_CONTEXT) cache->k_len = CACHE_MAX_CONTEXT;
    if (cache->v_len > CACHE_MAX_CONTEXT) cache->v_len = CACHE_MAX_CONTEXT;

    free(K_full); free(V_full);
    free(scores); free(Qh); free(Kh); free(Vh); free(ctx);
}

/**
 * Cache-aware conformer block: uses cached K/V and conv state.
 */
static void conformer_block_forward_cached(float *x, const ConformerBlockWeights *bw,
                                            Workspace *ws, int T, int D,
                                            int n_heads, int ff_mult, int conv_kernel,
                                            int use_rel_pe, LayerCache *cache) {
    int ff_dim = D * ff_mult;
    float *tmp = ws->residual;

    ffn_forward(tmp, x, bw->ff1_norm_w, bw->ff1_norm_b,
                bw->ff1_up_w, bw->ff1_up_b,
                bw->ff1_down_w, bw->ff1_down_b,
                ws, T, D, ff_dim);
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);

    float *attn_in = ws->buf_b;
    layer_norm(attn_in, x, bw->attn_norm_w, bw->attn_norm_b, T, D);
    mhsa_forward_cached(tmp, attn_in, bw, ws, T, D, n_heads, use_rel_pe, cache);
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);

    /* Conv module with cached overlap */
    float *conv_normed = ws->buf_b;
    layer_norm(conv_normed, x, bw->conv_norm_w, bw->conv_norm_b, T, D);
    linear(ws->conv_mid, conv_normed, bw->conv_pw1_w, bw->conv_pw1_b, T, D, 2 * D);
    glu(conv_normed, ws->conv_mid, T, D);
    depthwise_conv1d_cached(tmp, conv_normed, bw->conv_dw_w, bw->conv_dw_b,
                             T, D, conv_kernel, cache);
    batch_norm(conv_normed, tmp, bw->conv_bn_gamma, bw->conv_bn_beta,
               bw->conv_bn_mean, bw->conv_bn_var, T, D);
    silu_inplace(conv_normed, T * D);
    linear(tmp, conv_normed, bw->conv_pw2_w, bw->conv_pw2_b, T, D, D);
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);

    ffn_forward(tmp, x, bw->ff2_norm_w, bw->ff2_norm_b,
                bw->ff2_up_w, bw->ff2_up_b,
                bw->ff2_down_w, bw->ff2_down_b,
                ws, T, D, ff_dim);
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);

    layer_norm(x, x, bw->final_norm_w, bw->final_norm_b, T, D);
}

static void normalize_per_feature(float *mel, int T, int n_mels) {
    float eps = 1e-5f;
    for (int f = 0; f < n_mels; f++) {
        float sum = 0.0f;
        for (int t = 0; t < T; t++)
            sum += mel[t * n_mels + f];
        float mean = sum / (float)T;

        float var_sum = 0.0f;
        for (int t = 0; t < T; t++) {
            float d = mel[t * n_mels + f] - mean;
            var_sum += d * d;
        }
        float inv_std = 1.0f / sqrtf(var_sum / (float)T + eps);

        for (int t = 0; t < T; t++)
            mel[t * n_mels + f] = (mel[t * n_mels + f] - mean) * inv_std;
    }
}

static int full_forward(ConformerSTT *stt, const float *mel_in, int T) {
    int D         = (int)stt->header.d_model;
    int n_mels    = (int)stt->header.n_mels;
    int n_heads   = (int)stt->header.n_heads;
    int ff_mult   = (int)stt->header.ff_mult;
    int conv_kern = (int)stt->header.conv_kernel;
    int vocab     = (int)stt->header.vocab_size;
    int use_rel_pe = (stt->header.flags & CSTT_FLAG_REL_PE) ? 1 : 0;
    ModelWeights *w = &stt->weights;
    Workspace *ws   = &stt->work;

    /* Per-feature normalization (matching NeMo's per_feature normalize) */
    float *mel_norm = (float *)malloc((size_t)T * n_mels * sizeof(float));
    if (!mel_norm) return -1;
    memcpy(mel_norm, mel_in, (size_t)T * n_mels * sizeof(float));
    normalize_per_feature(mel_norm, T, n_mels);

    int T_sub = subsample_forward(ws->buf_a, mel_norm, T, stt);
    free(mel_norm);
    if (T_sub <= 0) return -1;
    if (T_sub > MAX_SEQ_LEN) T_sub = MAX_SEQ_LEN;

#ifdef CSTT_DEBUG
    {
        float mn = ws->buf_a[0], mx = ws->buf_a[0], sm = 0;
        for (int i = 0; i < T_sub * D; i++) {
            if (ws->buf_a[i] < mn) mn = ws->buf_a[i];
            if (ws->buf_a[i] > mx) mx = ws->buf_a[i];
            sm += ws->buf_a[i];
        }
        fprintf(stderr, "[debug] post-subsample: T=%d D=%d min=%.4f max=%.4f mean=%.6f\n",
                T_sub, D, mn, mx, sm / (T_sub * D));
    }
#endif

    if (!use_rel_pe)
        add_sinusoidal_pe(ws->buf_a, T_sub, D);

    if (stt->cache_aware && stt->layer_caches) {
        for (int i = 0; i < w->n_blocks; i++) {
            conformer_block_forward_cached(ws->buf_a, &w->blocks[i],
                                            ws, T_sub, D, n_heads, ff_mult, conv_kern,
                                            use_rel_pe, &stt->layer_caches[i]);
        }
    } else {
        for (int i = 0; i < w->n_blocks; i++) {
            conformer_block_forward(ws->buf_a, &w->blocks[i],
                                    ws, T_sub, D, n_heads, ff_mult, conv_kern,
                                    use_rel_pe);
#ifdef CSTT_DEBUG
            if (i == 0 || i == w->n_blocks - 1) {
                float mn = ws->buf_a[0], mx = ws->buf_a[0], sm = 0;
                for (int j = 0; j < T_sub * D; j++) {
                    if (ws->buf_a[j] < mn) mn = ws->buf_a[j];
                    if (ws->buf_a[j] > mx) mx = ws->buf_a[j];
                    sm += ws->buf_a[j];
                }
                fprintf(stderr, "[debug] block[%d]: min=%.4f max=%.4f mean=%.6f\n",
                        i, mn, mx, sm / (T_sub * D));
            }
#endif
        }
    }

    linear(ws->logits, ws->buf_a, w->ctc_w, w->ctc_b, T_sub, D, vocab);

#ifdef CSTT_DEBUG
    {
        /* Print argmax per frame and logit stats */
        fprintf(stderr, "[debug] CTC logits: T=%d V=%d\n", T_sub, vocab);
        for (int t = 0; t < T_sub && t < 10; t++) {
            const float *row = ws->logits + t * vocab;
            float best = row[0]; int bidx = 0;
            for (int v = 1; v < vocab; v++)
                if (row[v] > best) { best = row[v]; bidx = v; }
            fprintf(stderr, "[debug]   t=%d: argmax=%d (val=%.3f) blank_val=%.3f\n",
                    t, bidx, best, row[stt->vocab.blank_id]);
        }
    }
#endif
    stt->total_frames_processed += T_sub;
    return T_sub;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Vocabulary
 * ═══════════════════════════════════════════════════════════════════════════ */

static void vocab_init_charset(Vocabulary *v) {
    static const char *charset[] = {
        "<blank>",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z", " ", "'", ".", ",",
        "?", "!", "-"
    };
    int n = (int)(sizeof(charset) / sizeof(charset[0]));
    v->tokens = (char **)malloc(n * sizeof(char *));
    for (int i = 0; i < n; i++)
        v->tokens[i] = strdup(charset[i]);
    v->size = n;
    v->blank_id = 0;
}

/**
 * Load vocabulary from file. Also detects special tokens:
 *   - <blank> / <blk>                → blank_id
 *   - <eou> / <|endofutterance|> / <eos> → eou_id (returned via *out_eou_id)
 */
static int vocab_load(Vocabulary *v, const char *path, int expected_size,
                      int *out_eou_id) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int cap = expected_size > 0 ? expected_size : 1024;
    v->tokens = (char **)calloc(cap, sizeof(char *));
    v->size = 0;
    v->blank_id = 0;
    if (out_eou_id) *out_eou_id = -1;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;
        if (v->size >= cap) {
            cap *= 2;
            v->tokens = (char **)realloc(v->tokens, cap * sizeof(char *));
        }
        v->tokens[v->size] = strdup(line);
        if (strcmp(line, "<blank>") == 0 || strcmp(line, "<blk>") == 0)
            v->blank_id = v->size;
        if (out_eou_id &&
            (strcmp(line, "<eou>") == 0 ||
             strcmp(line, "<|endofutterance|>") == 0 ||
             strcmp(line, "<eos>") == 0))
            *out_eou_id = v->size;
        v->size++;
    }
    fclose(f);
    return v->size;
}

static void vocab_free(Vocabulary *v) {
    if (!v->tokens) return;
    for (int i = 0; i < v->size; i++) free(v->tokens[i]);
    free(v->tokens);
    v->tokens = NULL;
    v->size = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Weight loading
 * ═══════════════════════════════════════════════════════════════════════════ */

static const float *read_weight(const float **cursor, int count) {
    const float *ptr = *cursor;
    *cursor += count;
    return ptr;
}

static int load_block_weights(ConformerBlockWeights *bw, const float **cursor,
                              int D, int ff_dim, int K, int n_heads, int has_rel_pe) {
    bw->ff1_norm_w    = read_weight(cursor, D);
    bw->ff1_norm_b    = read_weight(cursor, D);
    bw->ff1_up_w      = read_weight(cursor, D * ff_dim);
    bw->ff1_up_b      = read_weight(cursor, ff_dim);
    bw->ff1_down_w    = read_weight(cursor, ff_dim * D);
    bw->ff1_down_b    = read_weight(cursor, D);

    bw->attn_norm_w   = read_weight(cursor, D);
    bw->attn_norm_b   = read_weight(cursor, D);
    bw->attn_q_w      = read_weight(cursor, D * D);
    bw->attn_q_b      = read_weight(cursor, D);
    bw->attn_k_w      = read_weight(cursor, D * D);
    bw->attn_k_b      = read_weight(cursor, D);
    bw->attn_v_w      = read_weight(cursor, D * D);
    bw->attn_v_b      = read_weight(cursor, D);
    bw->attn_out_w    = read_weight(cursor, D * D);
    bw->attn_out_b    = read_weight(cursor, D);

    if (has_rel_pe) {
        bw->attn_pos_w      = read_weight(cursor, D * D);
        bw->attn_pos_bias_u = read_weight(cursor, n_heads * (D / n_heads));
        bw->attn_pos_bias_v = read_weight(cursor, n_heads * (D / n_heads));
    } else {
        bw->attn_pos_w = NULL;
        bw->attn_pos_bias_u = NULL;
        bw->attn_pos_bias_v = NULL;
    }

    bw->conv_norm_w   = read_weight(cursor, D);
    bw->conv_norm_b   = read_weight(cursor, D);
    bw->conv_pw1_w    = read_weight(cursor, 2 * D * D);
    bw->conv_pw1_b    = read_weight(cursor, 2 * D);
    bw->conv_dw_w     = read_weight(cursor, D * K);
    bw->conv_dw_b     = read_weight(cursor, D);
    bw->conv_bn_gamma = read_weight(cursor, D);
    bw->conv_bn_beta  = read_weight(cursor, D);
    bw->conv_bn_mean  = read_weight(cursor, D);
    bw->conv_bn_var   = read_weight(cursor, D);
    bw->conv_pw2_w    = read_weight(cursor, D * D);
    bw->conv_pw2_b    = read_weight(cursor, D);

    bw->ff2_norm_w    = read_weight(cursor, D);
    bw->ff2_norm_b    = read_weight(cursor, D);
    bw->ff2_up_w      = read_weight(cursor, D * ff_dim);
    bw->ff2_up_b      = read_weight(cursor, ff_dim);
    bw->ff2_down_w    = read_weight(cursor, ff_dim * D);
    bw->ff2_down_b    = read_weight(cursor, D);

    bw->final_norm_w  = read_weight(cursor, D);
    bw->final_norm_b  = read_weight(cursor, D);

    return 0;
}

/**
 * Load subsampling weights.
 *
 * The .cstt file stores subsampling convolutions as a sequence of:
 *   [c_in, c_out, kernel, stride, groups] descriptor (5 uint32 per conv)
 * followed by the weight data for each conv layer, then the projection.
 *
 * For Conv1D mode, the old format is used (n_sub_convs == 0 in header).
 */
static int load_subsampling_weights(SubsamplingWeights *sw, const float **cursor,
                                    const CSTTHeader *h) {
    int D = (int)h->d_model;
    int n_mels = (int)h->n_mels;

    if (h->sub_type == CSTT_SUB_CONV1D && h->n_sub_convs == 0) {
        /* Legacy Conv1D format */
        int K = h->sub_conv_kernel > 0 ? (int)h->sub_conv_kernel : 3;
        sw->n_convs = 2;
        sw->convs[0].w = read_weight(cursor, D * n_mels * K);
        sw->convs[0].b = read_weight(cursor, D);
        sw->convs[0].c_in = n_mels; sw->convs[0].c_out = D;
        sw->convs[0].kernel = K; sw->convs[0].stride = 2; sw->convs[0].groups = 1;

        sw->convs[1].w = read_weight(cursor, D * D * K);
        sw->convs[1].b = read_weight(cursor, D);
        sw->convs[1].c_in = D; sw->convs[1].c_out = D;
        sw->convs[1].kernel = K; sw->convs[1].stride = 2; sw->convs[1].groups = 1;

        sw->proj_w  = read_weight(cursor, D * D);
        sw->proj_b  = read_weight(cursor, D);
        sw->proj_in = D;
        sw->proj_out = D;
    } else {
        /* General format: conv descriptors are stored as metadata */
        sw->n_convs = (int)h->n_sub_convs;
        if (sw->n_convs > MAX_SUB_CONVS) return -1;

        /* Read conv descriptors (5 uint32 per conv, stored as floats) */
        for (int i = 0; i < sw->n_convs; i++) {
            const uint32_t *desc = (const uint32_t *)(*cursor);
            sw->convs[i].c_in   = (int)desc[0];
            sw->convs[i].c_out  = (int)desc[1];
            sw->convs[i].kernel = (int)desc[2];
            sw->convs[i].stride = (int)desc[3];
            sw->convs[i].groups = (int)desc[4];
            *cursor += 5;  /* 5 uint32 = 5 floats */
        }

        /* Read weight data for each conv */
        for (int i = 0; i < sw->n_convs; i++) {
            int ci = sw->convs[i].c_in / sw->convs[i].groups;
            int co = sw->convs[i].c_out;
            int K2 = sw->convs[i].kernel * sw->convs[i].kernel;
            sw->convs[i].w = read_weight(cursor, co * ci * K2);
            sw->convs[i].b = read_weight(cursor, co);
        }

        /* Projection */
        int feat_in = (int)h->sub_feat_in;
        sw->proj_w  = read_weight(cursor, D * feat_in);
        sw->proj_b  = read_weight(cursor, D);
        sw->proj_in = feat_in;
        sw->proj_out = D;
    }

    return 0;
}

static int load_weights(ConformerSTT *stt) {
    const CSTTHeader *h = &stt->header;
    int D       = (int)h->d_model;
    int ff_dim  = D * (int)h->ff_mult;
    int K       = (int)h->conv_kernel;
    int vocab   = (int)h->vocab_size;
    int n_layers = (int)h->n_layers;
    int n_heads = (int)h->n_heads;
    int has_rel_pe = (h->flags & CSTT_FLAG_REL_PE) ? 1 : 0;

    const float *cursor = (const float *)((char *)stt->mmap_base + sizeof(CSTTHeader));
    ModelWeights *w = &stt->weights;

    if (load_subsampling_weights(&w->sub, &cursor, h) != 0)
        return -1;

    w->n_blocks = n_layers;
    w->blocks = (ConformerBlockWeights *)calloc(n_layers, sizeof(ConformerBlockWeights));
    if (!w->blocks) return -1;

    for (int i = 0; i < n_layers; i++) {
        if (load_block_weights(&w->blocks[i], &cursor, D, ff_dim, K,
                               n_heads, has_rel_pe) != 0)
            return -1;
    }

    w->ctc_w = read_weight(&cursor, D * vocab);
    w->ctc_b = read_weight(&cursor, vocab);

    size_t consumed = (size_t)((char *)cursor - (char *)stt->mmap_base);
    if (consumed > stt->mmap_size) {
        fprintf(stderr, "[conformer_stt] Weight file too small: need %zu, have %zu\n",
                consumed, stt->mmap_size);
        return -1;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Workspace allocation
 * ═══════════════════════════════════════════════════════════════════════════ */

static int workspace_alloc(Workspace *ws, const CSTTHeader *h) {
    int D = (int)h->d_model;
    int n_heads = (int)h->n_heads;
    int ff_dim = D * (int)h->ff_mult;
    int vocab = (int)h->vocab_size;
    int n_mels = (int)h->n_mels;
    int d_head = D / n_heads;
    size_t seq_d  = (size_t)MAX_SEQ_LEN * D;
    size_t seq_ff = (size_t)MAX_SEQ_LEN * ff_dim;
    size_t seq_2d = (size_t)MAX_SEQ_LEN * 2 * D;
    size_t attn_sz = (size_t)MAX_SEQ_LEN * D * 3
                   + (size_t)MAX_SEQ_LEN * MAX_SEQ_LEN;

    ws->buf_a       = (float *)calloc(seq_d, sizeof(float));
    ws->buf_b       = (float *)calloc(seq_d, sizeof(float));
    ws->residual    = (float *)calloc(seq_d, sizeof(float));
    ws->qkv         = (float *)calloc((size_t)MAX_SEQ_LEN * 3 * D, sizeof(float));
    ws->attn_scores = (float *)calloc(attn_sz, sizeof(float));
    ws->ff_mid      = (float *)calloc(seq_ff, sizeof(float));
    ws->conv_mid    = (float *)calloc(seq_2d, sizeof(float));
    ws->logits      = (float *)calloc((size_t)MAX_SEQ_LEN * vocab, sizeof(float));

    /* Large buffer for Conv2D subsampling intermediates */
    size_t sub_sz = 2 * (size_t)D * CHUNK_FRAMES * n_mels;
    ws->sub_work = (float *)calloc(sub_sz, sizeof(float));

    /* Relative PE table — full model dimension for linear_pos projection */
    if (h->flags & CSTT_FLAG_REL_PE) {
        ws->rel_pe = (float *)calloc(2 * (size_t)MAX_SEQ_LEN * D, sizeof(float));
        if (ws->rel_pe)
            precompute_rel_pe(ws->rel_pe, MAX_SEQ_LEN, D);
    } else {
        ws->rel_pe = NULL;
    }

    if (!ws->buf_a || !ws->buf_b || !ws->residual || !ws->qkv ||
        !ws->attn_scores || !ws->ff_mid || !ws->conv_mid ||
        !ws->logits || !ws->sub_work)
        return -1;

    return 0;
}

static void workspace_free(Workspace *ws) {
    free(ws->buf_a);
    free(ws->buf_b);
    free(ws->residual);
    free(ws->qkv);
    free(ws->attn_scores);
    free(ws->ff_mid);
    free(ws->conv_mid);
    free(ws->logits);
    free(ws->sub_work);
    free(ws->rel_pe);
    memset(ws, 0, sizeof(Workspace));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════════ */

ConformerSTT *conformer_stt_create(const char *model_path) {
    if (!model_path) return NULL;

    ConformerSTT *stt = (ConformerSTT *)calloc(1, sizeof(ConformerSTT));
    if (!stt) return NULL;
    stt->mmap_fd = -1;
    stt->prev_token = -1;

    stt->mmap_fd = open(model_path, O_RDONLY);
    if (stt->mmap_fd < 0) {
        fprintf(stderr, "[conformer_stt] Cannot open model: %s\n", model_path);
        goto fail;
    }

    struct stat st;
    if (fstat(stt->mmap_fd, &st) != 0) goto fail;
    stt->mmap_size = (size_t)st.st_size;

    if (stt->mmap_size < sizeof(CSTTHeader)) {
        fprintf(stderr, "[conformer_stt] Model file too small\n");
        goto fail;
    }

    stt->mmap_base = mmap(NULL, stt->mmap_size, PROT_READ, MAP_PRIVATE,
                          stt->mmap_fd, 0);
    if (stt->mmap_base == MAP_FAILED) {
        stt->mmap_base = NULL;
        goto fail;
    }

    memcpy(&stt->header, stt->mmap_base, sizeof(CSTTHeader));
    if (stt->header.magic != CSTT_MAGIC) {
        fprintf(stderr, "[conformer_stt] Invalid magic: 0x%08X\n", stt->header.magic);
        goto fail;
    }
    if (stt->header.version > 1) {
        fprintf(stderr, "[conformer_stt] Unsupported version: %u\n", stt->header.version);
        goto fail;
    }

    fprintf(stderr, "[conformer_stt] Model: %u layers, d=%u, heads=%u, "
            "conv_k=%u, vocab=%u, mels=%u, sub=%u\n",
            stt->header.n_layers, stt->header.d_model, stt->header.n_heads,
            stt->header.conv_kernel, stt->header.vocab_size, stt->header.n_mels,
            stt->header.sub_type);

    if (load_weights(stt) != 0) {
        fprintf(stderr, "[conformer_stt] Failed to map weights\n");
        goto fail;
    }

    if (workspace_alloc(&stt->work, &stt->header) != 0) {
        fprintf(stderr, "[conformer_stt] Failed to allocate workspace\n");
        goto fail;
    }

    MelConfig mel_cfg;
    mel_config_default(&mel_cfg);
    mel_cfg.sample_rate = (int)stt->header.sample_rate;
    mel_cfg.n_fft       = (int)stt->header.n_fft;
    mel_cfg.hop_length  = (int)stt->header.hop_length;
    mel_cfg.win_length  = (int)stt->header.win_length;
    mel_cfg.n_mels      = (int)stt->header.n_mels;

    stt->mel = mel_create(&mel_cfg);
    if (!stt->mel) {
        fprintf(stderr, "[conformer_stt] Failed to create mel extractor\n");
        goto fail;
    }

    stt->mel_accum_cap = CHUNK_FRAMES * 2 * (int)stt->header.n_mels;
    stt->mel_accum = (float *)calloc(stt->mel_accum_cap, sizeof(float));
    if (!stt->mel_accum) goto fail;

    char vocab_path[4096];
    snprintf(vocab_path, sizeof(vocab_path), "%s", model_path);
    char *dot = strrchr(vocab_path, '.');
    if (dot) strcpy(dot, ".vocab");
    else strcat(vocab_path, ".vocab");

    int eou_id = -1;
    if (vocab_load(&stt->vocab, vocab_path, (int)stt->header.vocab_size, &eou_id) <= 0) {
        fprintf(stderr, "[conformer_stt] No vocab at %s, using built-in charset\n", vocab_path);
        vocab_init_charset(&stt->vocab);
        eou_id = -1;
    } else {
        fprintf(stderr, "[conformer_stt] Loaded vocabulary: %d tokens\n", stt->vocab.size);
    }

    /* EOU token detection */
    stt->eou_token_id = eou_id;
    stt->eou_detected = 0;
    stt->eou_frame = -1;
    stt->eou_prob = 0.0f;
    stt->eou_probs_cap = MAX_SEQ_LEN;
    stt->eou_probs = (float *)calloc(stt->eou_probs_cap, sizeof(float));
    stt->eou_probs_len = 0;

    if (eou_id >= 0) {
        fprintf(stderr, "[conformer_stt] EOU token detected at vocab[%d] = '%s'\n",
                eou_id, stt->vocab.tokens[eou_id]);
    }

    /* Activation caching — allocate per-layer caches */
    int n_layers = (int)stt->header.n_layers;
    int D = (int)stt->header.d_model;
    stt->cache_aware = (stt->header.flags & CSTT_FLAG_CACHE_AWARE) ? 1 : 0;
    stt->layer_caches = (LayerCache *)calloc(n_layers, sizeof(LayerCache));
    if (stt->layer_caches) {
        for (int i = 0; i < n_layers; i++) {
            stt->layer_caches[i].k_cache = (float *)calloc((size_t)CACHE_MAX_CONTEXT * D, sizeof(float));
            stt->layer_caches[i].v_cache = (float *)calloc((size_t)CACHE_MAX_CONTEXT * D, sizeof(float));
            stt->layer_caches[i].k_len = 0;
            stt->layer_caches[i].v_len = 0;
            int conv_k = (int)stt->header.conv_kernel;
            stt->layer_caches[i].conv_state = (float *)calloc((size_t)(conv_k - 1) * D, sizeof(float));
            stt->layer_caches[i].conv_state_len = 0;
        }
    }
    stt->total_frames_processed = 0;

    stt->transcript[0] = '\0';
    fprintf(stderr, "[conformer_stt] Ready. (cache_aware=%d, eou=%s)\n",
            stt->cache_aware, eou_id >= 0 ? "yes" : "no");
    return stt;

fail:
    conformer_stt_destroy(stt);
    return NULL;
}

void conformer_stt_destroy(ConformerSTT *stt) {
    if (!stt) return;
    mel_destroy(stt->mel);
    workspace_free(&stt->work);
    vocab_free(&stt->vocab);
    free(stt->weights.blocks);
    free(stt->mel_accum);
    free(stt->eou_probs);
    if (stt->layer_caches) {
        int n = (int)stt->header.n_layers;
        for (int i = 0; i < n; i++) {
            free(stt->layer_caches[i].k_cache);
            free(stt->layer_caches[i].v_cache);
            free(stt->layer_caches[i].conv_state);
        }
        free(stt->layer_caches);
    }
    if (stt->mmap_base && stt->mmap_base != MAP_FAILED)
        munmap(stt->mmap_base, stt->mmap_size);
    if (stt->mmap_fd >= 0)
        close(stt->mmap_fd);
    free(stt);
}

int conformer_stt_process(ConformerSTT *stt, const float *pcm, int n_samples) {
    if (!stt || !pcm || n_samples <= 0) return -1;

    int n_mels = (int)stt->header.n_mels;
    int sub_factor = (int)stt->header.subsample_factor;
    if (sub_factor < 1) sub_factor = 4;

    int max_new_frames = n_samples / mel_hop_length(stt->mel) + 2;
    float *new_mels = (float *)malloc((size_t)max_new_frames * n_mels * sizeof(float));
    if (!new_mels) return -1;

    int n_frames = mel_process(stt->mel, pcm, n_samples, new_mels, max_new_frames);
    if (n_frames <= 0) { free(new_mels); return 0; }

    int needed = (stt->mel_accum_len + n_frames) * n_mels;
    if (needed > stt->mel_accum_cap) {
        stt->mel_accum_cap = needed + CHUNK_FRAMES * n_mels;
        stt->mel_accum = (float *)realloc(stt->mel_accum,
                                           stt->mel_accum_cap * sizeof(float));
        if (!stt->mel_accum) { free(new_mels); return -1; }
    }
    memcpy(stt->mel_accum + stt->mel_accum_len * n_mels,
           new_mels, (size_t)n_frames * n_mels * sizeof(float));
    stt->mel_accum_len += n_frames;
    free(new_mels);

    int total_new_chars = 0;
    int min_frames = sub_factor * 4;

    while (stt->mel_accum_len >= min_frames) {
        int process_frames = stt->mel_accum_len;
        if (process_frames > CHUNK_FRAMES)
            process_frames = CHUNK_FRAMES;
        process_frames = (process_frames / sub_factor) * sub_factor;
        if (process_frames < min_frames) break;

        int T_out = full_forward(stt, stt->mel_accum, process_frames);
        if (T_out <= 0) break;

        /* Ensure eou_probs buffer is large enough */
        if (T_out > stt->eou_probs_cap) {
            stt->eou_probs_cap = T_out + 256;
            stt->eou_probs = (float *)realloc(stt->eou_probs,
                                               stt->eou_probs_cap * sizeof(float));
        }

        char decode_buf[4096];
        int eou_det = 0, eou_fr = -1;
        int n_chars = ctc_greedy_decode(decode_buf, sizeof(decode_buf),
                                        stt->work.logits, T_out,
                                        (int)stt->header.vocab_size,
                                        &stt->vocab, &stt->prev_token,
                                        stt->eou_token_id,
                                        &eou_det, &eou_fr,
                                        stt->eou_probs);
        stt->eou_probs_len = T_out;
        if (eou_det) {
            stt->eou_detected = 1;
            stt->eou_frame = stt->total_frames_processed + eou_fr;
        }

        /* Compute trailing EOU probability average */
        if (stt->eou_token_id >= 0 && T_out > 0) {
            int horizon = 4;
            if (horizon > T_out) horizon = T_out;
            float sum = 0.0f;
            for (int i = T_out - horizon; i < T_out; i++)
                sum += stt->eou_probs[i];
            stt->eou_prob = sum / (float)horizon;
        }

        if (n_chars > 0) {
            int space = MAX_TRANSCRIPT - stt->transcript_len - 1;
            int copy = n_chars < space ? n_chars : space;
            if (copy > 0) {
                memcpy(stt->transcript + stt->transcript_len, decode_buf, copy);
                stt->transcript_len += copy;
                stt->transcript[stt->transcript_len] = '\0';
                total_new_chars += copy;
            }
        }

        int remaining = stt->mel_accum_len - process_frames;
        if (remaining > 0)
            memmove(stt->mel_accum,
                    stt->mel_accum + process_frames * n_mels,
                    (size_t)remaining * n_mels * sizeof(float));
        stt->mel_accum_len = remaining;
    }

    return total_new_chars;
}

int conformer_stt_flush(ConformerSTT *stt) {
    if (!stt) return -1;
    int n_mels = (int)stt->header.n_mels;
    int sub_factor = (int)stt->header.subsample_factor;
    if (sub_factor < 1) sub_factor = 4;
    int min_frames = sub_factor * 4;

    if (stt->mel_accum_len < min_frames) {
        int pad = min_frames - stt->mel_accum_len;
        int needed = (stt->mel_accum_len + pad) * n_mels;
        if (needed > stt->mel_accum_cap) {
            stt->mel_accum_cap = needed;
            stt->mel_accum = (float *)realloc(stt->mel_accum,
                                               stt->mel_accum_cap * sizeof(float));
            if (!stt->mel_accum) return -1;
        }
        memset(stt->mel_accum + stt->mel_accum_len * n_mels, 0,
               (size_t)pad * n_mels * sizeof(float));
        stt->mel_accum_len += pad;
    }

    int T_out = full_forward(stt, stt->mel_accum, stt->mel_accum_len);
    stt->mel_accum_len = 0;
    if (T_out <= 0) return 0;

    if (T_out > stt->eou_probs_cap) {
        stt->eou_probs_cap = T_out + 256;
        stt->eou_probs = (float *)realloc(stt->eou_probs,
                                           stt->eou_probs_cap * sizeof(float));
    }

    char decode_buf[4096];
    int eou_det = 0, eou_fr = -1;
    int n_chars = ctc_greedy_decode(decode_buf, sizeof(decode_buf),
                                    stt->work.logits, T_out,
                                    (int)stt->header.vocab_size,
                                    &stt->vocab, &stt->prev_token,
                                    stt->eou_token_id,
                                    &eou_det, &eou_fr,
                                    stt->eou_probs);
    stt->eou_probs_len = T_out;
    if (eou_det) {
        stt->eou_detected = 1;
        stt->eou_frame = stt->total_frames_processed + eou_fr;
    }

    if (n_chars > 0) {
        int space = MAX_TRANSCRIPT - stt->transcript_len - 1;
        int copy = n_chars < space ? n_chars : space;
        if (copy > 0) {
            memcpy(stt->transcript + stt->transcript_len, decode_buf, copy);
            stt->transcript_len += copy;
            stt->transcript[stt->transcript_len] = '\0';
        }
    }
    return n_chars;
}

int conformer_stt_get_text(const ConformerSTT *stt, char *buf, int buf_size) {
    if (!stt || !buf || buf_size <= 0) return -1;
    int copy = stt->transcript_len;
    if (copy >= buf_size) copy = buf_size - 1;
    memcpy(buf, stt->transcript, copy);
    buf[copy] = '\0';
    return copy;
}

void conformer_stt_reset(ConformerSTT *stt) {
    if (!stt) return;
    stt->transcript[0] = '\0';
    stt->transcript_len = 0;
    stt->mel_accum_len = 0;
    stt->prev_token = -1;
    stt->eou_detected = 0;
    stt->eou_frame = -1;
    stt->eou_prob = 0.0f;
    stt->eou_probs_len = 0;
    stt->total_frames_processed = 0;
    mel_reset(stt->mel);

    /* Reset layer caches */
    if (stt->layer_caches) {
        int n = (int)stt->header.n_layers;
        for (int i = 0; i < n; i++) {
            stt->layer_caches[i].k_len = 0;
            stt->layer_caches[i].v_len = 0;
            stt->layer_caches[i].conv_state_len = 0;
        }
    }
}

/* ─── EOU Detection API ─────────────────────────────────────────────────── */

int conformer_stt_has_eou(const ConformerSTT *stt) {
    return stt ? stt->eou_detected : 0;
}

float conformer_stt_eou_prob(const ConformerSTT *stt, int horizon) {
    if (!stt || stt->eou_token_id < 0 || stt->eou_probs_len <= 0) return 0.0f;
    if (horizon <= 0) horizon = 4;
    if (horizon > stt->eou_probs_len) horizon = stt->eou_probs_len;
    float sum = 0.0f;
    for (int i = stt->eou_probs_len - horizon; i < stt->eou_probs_len; i++)
        sum += stt->eou_probs[i];
    return sum / (float)horizon;
}

int conformer_stt_eou_frame(const ConformerSTT *stt) {
    return stt ? stt->eou_frame : -1;
}

/* ─── Cache-Aware Streaming API ──────────────────────────────────────────── */

void conformer_stt_set_cache_aware(ConformerSTT *stt, int enable) {
    if (!stt) return;
    stt->cache_aware = enable ? 1 : 0;
    if (enable)
        fprintf(stderr, "[conformer_stt] Cache-aware streaming enabled\n");
}

int conformer_stt_stride_ms(const ConformerSTT *stt) {
    if (!stt) return 0;
    int hop = (int)stt->header.hop_length;
    int sr = (int)stt->header.sample_rate;
    int sub = (int)stt->header.subsample_factor;
    if (sub < 1) sub = 4;
    if (stt->cache_aware) {
        return (hop * sub * 1000) / sr; /* Single subsampled frame time */
    }
    return (hop * CHUNK_FRAMES * 1000) / sr; /* Full chunk time */
}

/* ─── Info API ──────────────────────────────────────────────────────────── */

int conformer_stt_sample_rate(const ConformerSTT *stt) {
    return stt ? (int)stt->header.sample_rate : 0;
}
int conformer_stt_d_model(const ConformerSTT *stt) {
    return stt ? (int)stt->header.d_model : 0;
}
int conformer_stt_n_layers(const ConformerSTT *stt) {
    return stt ? (int)stt->header.n_layers : 0;
}
int conformer_stt_vocab_size(const ConformerSTT *stt) {
    return stt ? (int)stt->header.vocab_size : 0;
}
int conformer_stt_has_eou_support(const ConformerSTT *stt) {
    return (stt && stt->eou_token_id >= 0) ? 1 : 0;
}

int conformer_stt_forward_normalized_mel(ConformerSTT *stt,
                                          const float *mel, int T) {
    if (!stt || !mel || T <= 0) return -1;

    int D         = (int)stt->header.d_model;
    int n_mels    = (int)stt->header.n_mels;
    int n_heads   = (int)stt->header.n_heads;
    int ff_mult   = (int)stt->header.ff_mult;
    int conv_kern = (int)stt->header.conv_kernel;
    int vocab     = (int)stt->header.vocab_size;
    int use_rel_pe = (stt->header.flags & CSTT_FLAG_REL_PE) ? 1 : 0;
    ModelWeights *w = &stt->weights;
    Workspace *ws   = &stt->work;

    /* Skip mel extraction + normalization — feed directly to subsampling */
    int T_sub = subsample_forward(ws->buf_a, mel, T, stt);
    if (T_sub <= 0) return -1;
    if (T_sub > MAX_SEQ_LEN) T_sub = MAX_SEQ_LEN;

#ifdef CSTT_DEBUG
    {
        float mn = ws->buf_a[0], mx = ws->buf_a[0], sm = 0;
        for (int i = 0; i < T_sub * D; i++) {
            if (ws->buf_a[i] < mn) mn = ws->buf_a[i];
            if (ws->buf_a[i] > mx) mx = ws->buf_a[i];
            sm += ws->buf_a[i];
        }
        fprintf(stderr, "[debug-mel] post-subsample: T=%d D=%d min=%.4f max=%.4f mean=%.6f\n",
                T_sub, D, mn, mx, sm / (T_sub * D));
    }
#endif

    if (!use_rel_pe)
        add_sinusoidal_pe(ws->buf_a, T_sub, D);

    for (int i = 0; i < w->n_blocks; i++) {
        conformer_block_forward(ws->buf_a, &w->blocks[i],
                                ws, T_sub, D, n_heads, ff_mult, conv_kern,
                                use_rel_pe);
#ifdef CSTT_DEBUG
        if (i == 0 || i == w->n_blocks - 1) {
            float mn = ws->buf_a[0], mx = ws->buf_a[0], sm = 0;
            for (int j = 0; j < T_sub * D; j++) {
                if (ws->buf_a[j] < mn) mn = ws->buf_a[j];
                if (ws->buf_a[j] > mx) mx = ws->buf_a[j];
                sm += ws->buf_a[j];
            }
            fprintf(stderr, "[debug-mel] block[%d]: min=%.4f max=%.4f mean=%.6f\n",
                    i, mn, mx, sm / (T_sub * D));
        }
#endif
    }

    linear(ws->logits, ws->buf_a, w->ctc_w, w->ctc_b, T_sub, D, vocab);

#ifdef CSTT_DEBUG
    {
        fprintf(stderr, "[debug-mel] CTC logits: T=%d V=%d\n", T_sub, vocab);
        for (int t = 0; t < T_sub && t < 10; t++) {
            const float *row = ws->logits + t * vocab;
            float best = row[0]; int bidx = 0;
            for (int v = 1; v < vocab; v++)
                if (row[v] > best) { best = row[v]; bidx = v; }
            fprintf(stderr, "[debug-mel]   t=%d: argmax=%d (val=%.3f) blank_val=%.3f\n",
                    t, bidx, best, row[stt->vocab.blank_id]);
        }
    }
#endif
    stt->total_frames_processed += T_sub;

    /* CTC greedy decode */
    char decode_buf[4096];
    int eou_det = 0, eou_fr = -1;
    int n_chars = ctc_greedy_decode(decode_buf, sizeof(decode_buf),
                                    ws->logits, T_sub, vocab,
                                    &stt->vocab, &stt->prev_token,
                                    stt->eou_token_id,
                                    &eou_det, &eou_fr, NULL);

    if (n_chars > 0) {
        int space = MAX_TRANSCRIPT - stt->transcript_len - 1;
        int copy = n_chars < space ? n_chars : space;
        if (copy > 0) {
            memcpy(stt->transcript + stt->transcript_len, decode_buf, copy);
            stt->transcript_len += copy;
            stt->transcript[stt->transcript_len] = '\0';
        }
    }

    return n_chars;
}

int conformer_stt_forward_subsample_output(ConformerSTT *stt,
                                            const float *sub_out, int T_sub) {
    if (!stt || !sub_out || T_sub <= 0) return -1;

    int D         = (int)stt->header.d_model;
    int n_heads   = (int)stt->header.n_heads;
    int ff_mult   = (int)stt->header.ff_mult;
    int conv_kern = (int)stt->header.conv_kernel;
    int vocab     = (int)stt->header.vocab_size;
    int use_rel_pe = (stt->header.flags & CSTT_FLAG_REL_PE) ? 1 : 0;
    ModelWeights *w = &stt->weights;
    Workspace *ws   = &stt->work;

    if (T_sub > MAX_SEQ_LEN) T_sub = MAX_SEQ_LEN;
    memcpy(ws->buf_a, sub_out, (size_t)T_sub * D * sizeof(float));

    if (!use_rel_pe)
        add_sinusoidal_pe(ws->buf_a, T_sub, D);

    for (int i = 0; i < w->n_blocks; i++) {
        conformer_block_forward(ws->buf_a, &w->blocks[i],
                                ws, T_sub, D, n_heads, ff_mult, conv_kern,
                                use_rel_pe);
#ifdef CSTT_DEBUG
        {
            float mn = ws->buf_a[0], mx = ws->buf_a[0], sm = 0;
            for (int j = 0; j < T_sub * D; j++) {
                if (ws->buf_a[j] < mn) mn = ws->buf_a[j];
                if (ws->buf_a[j] > mx) mx = ws->buf_a[j];
                sm += ws->buf_a[j];
            }
            fprintf(stderr, "[debug-sub] block[%d]: min=%.4f max=%.4f mean=%.6f",
                    i, mn, mx, sm / (T_sub * D));
            fprintf(stderr, "  f[0][:4]=%.4f,%.4f,%.4f,%.4f\n",
                    ws->buf_a[0], ws->buf_a[1], ws->buf_a[2], ws->buf_a[3]);
        }
#endif
    }

    linear(ws->logits, ws->buf_a, w->ctc_w, w->ctc_b, T_sub, D, vocab);

#ifdef CSTT_DEBUG
    {
        fprintf(stderr, "[debug-sub] CTC logits: T=%d V=%d\n", T_sub, vocab);
        for (int t = 0; t < T_sub && t < 10; t++) {
            const float *row = ws->logits + t * vocab;
            float best = row[0]; int bidx = 0;
            for (int v = 1; v < vocab; v++)
                if (row[v] > best) { best = row[v]; bidx = v; }
            fprintf(stderr, "[debug-sub]   t=%d: argmax=%d (val=%.3f) blank_val=%.3f\n",
                    t, bidx, best, row[stt->vocab.blank_id]);
        }
    }
#endif
    stt->total_frames_processed += T_sub;

    char decode_buf[4096];
    int eou_det = 0, eou_fr = -1;
    int n_chars = ctc_greedy_decode(decode_buf, sizeof(decode_buf),
                                    ws->logits, T_sub, vocab,
                                    &stt->vocab, &stt->prev_token,
                                    stt->eou_token_id,
                                    &eou_det, &eou_fr, NULL);

    if (n_chars > 0) {
        int space = MAX_TRANSCRIPT - stt->transcript_len - 1;
        int copy = n_chars < space ? n_chars : space;
        if (copy > 0) {
            memcpy(stt->transcript + stt->transcript_len, decode_buf, copy);
            stt->transcript_len += copy;
            stt->transcript[stt->transcript_len] = '\0';
        }
    }

    return n_chars;
}
