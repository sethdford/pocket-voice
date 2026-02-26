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
#include "ctc_beam_decoder.h"
#include "tdt_decoder.h"
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

/* Undocumented Accelerate fp16 GEMM — available on macOS 15+ Apple Silicon.
 * Resolved at runtime via dlsym to avoid link failures on older systems. */
#include <dlfcn.h>
typedef void (*cblas_hgemm_fn)(enum CBLAS_ORDER, enum CBLAS_TRANSPOSE, enum CBLAS_TRANSPOSE,
                               int, int, int,
                               __fp16, const __fp16 *, int,
                               const __fp16 *, int,
                               __fp16, __fp16 *, int);
static cblas_hgemm_fn _cblas_hgemm = NULL;
static int _hgemm_resolved = 0;

static cblas_hgemm_fn get_cblas_hgemm(void) {
    if (!_hgemm_resolved) {
        _cblas_hgemm = (cblas_hgemm_fn)dlsym(RTLD_DEFAULT, "cblas_hgemm");
        _hgemm_resolved = 1;
    }
    return _cblas_hgemm;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Constants and limits
 * ═══════════════════════════════════════════════════════════════════════════ */

#define MAX_SEQ_LEN     2048
#define MAX_D_MODEL     1024
#define MAX_VOCAB       8192
#define MAX_TRANSCRIPT  16384
#define CHUNK_FRAMES    8000   /* Mel frames per chunk (~80s at 10ms hop) */
#define MAX_SUB_CONVS   6      /* Max conv layers in subsampling */
#define MAX_PRED_LAYERS 4      /* Max LSTM layers in TDT prediction net */

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
    const float *attn_linear_pos_w;            /* [D, D] — PE projection (NeMo linear_pos) */
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

    /* Raw fp16 pointers for AMX cblas_hgemm (NULL when fp32 model) */
    const __fp16 *ff1_up_w_h, *ff1_down_w_h;
    const __fp16 *attn_q_w_h, *attn_k_w_h, *attn_v_w_h, *attn_out_w_h;
    const __fp16 *attn_linear_pos_w_h;
    const __fp16 *conv_pw1_w_h, *conv_pw2_w_h;
    const __fp16 *ff2_up_w_h, *ff2_down_w_h;
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
    const __fp16 *ctc_w_h;          /* Raw fp16 CTC weight (NULL for fp32) */
    const __fp16 *sub_proj_w_h;     /* Raw fp16 subsampling projection */
    int use_fp16_gemm;               /* 1 = use cblas_hgemm for large matmuls */
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
    float *rel_pe;         /* Scratch for relative PE [2*T, D] — allocated per max T */
    float *rel_pe_proj;    /* Scratch for projected PE [2*T, D] */

    /* Pre-allocated scratch buffers for hot paths (no malloc in inference) */
    float *silu_tmp;       /* [MAX_SEQ_LEN * ff_dim] for silu_inplace */
    float *bn_scale;       /* [MAX_D_MODEL] for batch_norm */
    float *bn_shift;       /* [MAX_D_MODEL] for batch_norm */
    float *lsp_tmp;        /* [MAX_VOCAB] for logits_to_log_probs */
    float *mel_norm_buf;   /* [CHUNK_FRAMES * n_mels] for full_forward mel copy */

    /* Cached MHSA scratch: sized for (CACHE_MAX_CONTEXT + MAX_SEQ_LEN) */
    float *cache_k_full;   /* [total_T_max * D] */
    float *cache_v_full;   /* [total_T_max * D] */
    float *cache_scores;   /* [MAX_SEQ_LEN * total_T_max] */
    float *cache_qh;       /* [MAX_SEQ_LEN * d_head] */
    float *cache_kh;       /* [total_T_max * d_head] */
    float *cache_vh;       /* [total_T_max * d_head] */
    float *cache_ctx;      /* [MAX_SEQ_LEN * d_head] */

    /* Cached conv scratch: sized for (conv_kernel-1 + MAX_SEQ_LEN) */
    float *cache_conv_merged;  /* [total_conv_max * D] */
    float *cache_conv_out;     /* [total_conv_max * D] */

    /* Feature normalization transpose buffer */
    float *feat_col;       /* [max(CHUNK_FRAMES, MAX_SEQ_LEN)] for per_feature_normalize */

    /* fp16 mixed-precision scratch buffers */
    __fp16 *fp16_in;       /* [MAX_SEQ_LEN * max(D, ff_dim)] activation input (fp32→fp16) */
    __fp16 *fp16_out;      /* [MAX_SEQ_LEN * max(D, ff_dim)] GEMM output (fp16→fp32) */
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

#define CACHE_MAX_CONTEXT 256  /* Maximum cached K/V frames for streaming attention */

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
    int          chunk_frames; /* Mel frames per encoder chunk (0 = use CHUNK_FRAMES) */

    int    skip_mel_normalize; /* 1 = caller pre-normalized mel input */

    /* Beam search decoder (optional — NULL means greedy) */
    CTCBeamDecoder *beam_decoder;
    float *beam_logits;     /* Accumulated logits for full-utterance beam decode */
    int    beam_logits_len; /* Number of accumulated time steps */
    int    beam_logits_cap; /* Capacity in time steps */

    /* Running normalization for streaming (Welford's online algorithm) */
    double *running_sum;    /* [n_mels] running sum per feature */
    double *running_sum_sq; /* [n_mels] running sum of squares per feature */
    int     running_count;  /* Total frames seen across all chunks */

    /* fp16 weight promotion buffer */
    float *fp16_arena;      /* Bulk fp32 buffer for promoted fp16 weights */

    /* TDT transducer decoder (optional — NULL for CTC-only models) */
    TDTDecoder *tdt;
    int         tdt_n_durations;
    int        *tdt_token_buf;   /* Scratch for decoded token IDs */
    int         tdt_token_cap;
    float      *tdt_enc_accum;   /* Accumulated encoder output [T, D] */
    int         tdt_enc_len;     /* Number of accumulated time steps */
    int         tdt_enc_cap;     /* Capacity in time steps */

    /* External forward-pass hook (BNNS, Metal, etc.) */
    conformer_external_forward_fn external_forward;
    void                         *external_forward_ctx;
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

static void per_feature_normalize(float *mel, int T, int n_mels) {
    const float eps = 1e-5f;
    for (int f = 0; f < n_mels; f++) {
        float *col = mel + f;
        float mean;
        vDSP_meanv(col, n_mels, &mean, T);
        float neg_mean = -mean;
        vDSP_vsadd(col, n_mels, &neg_mean, col, n_mels, T);
        float var;
        vDSP_measqv(col, n_mels, &var, T);
        float inv_std = 1.0f / sqrtf(var + eps);
        vDSP_vsmul(col, n_mels, &inv_std, col, n_mels, T);
    }
}

/**
 * Running per-feature normalization for streaming.
 * Uses Welford's online algorithm to track mean/variance across all chunks
 * seen so far. Each new chunk updates the running stats, then normalization
 * uses the accumulated statistics instead of per-chunk local stats.
 */
static void per_feature_normalize_running(float *mel, int T, int n_mels,
                                          double *running_sum,
                                          double *running_sum_sq,
                                          int *running_count) {
    const float eps = 1e-5f;

    for (int t = 0; t < T; t++)
        for (int f = 0; f < n_mels; f++) {
            double v = (double)mel[t * n_mels + f];
            running_sum[f] += v;
            running_sum_sq[f] += v * v;
        }
    *running_count += T;

    int N = *running_count;
    for (int f = 0; f < n_mels; f++) {
        float mean = (float)(running_sum[f] / N);
        float var = (float)(running_sum_sq[f] / N - (double)mean * mean);
        float inv_std = 1.0f / sqrtf(var + eps);
        float *col = mel + f;
        float neg_mean = -mean;
        vDSP_vsadd(col, n_mels, &neg_mean, col, n_mels, T);
        vDSP_vsmul(col, n_mels, &inv_std, col, n_mels, T);
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

/* Native AMX fp16 GEMM path — fastest when cblas_hgemm is available (macOS 15+). */
static void linear_fp16_native(float *out, const float *in, const __fp16 *W_h,
                               const float *bias, int M, int K, int N,
                               __fp16 *tmp_in, __fp16 *tmp_out) {
    int in_count = M * K;
    int out_count = M * N;

    for (int i = 0; i + 3 < in_count; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vst1_f16(tmp_in + i, vcvt_f16_f32(v));
    }
    for (int i = (in_count & ~3); i < in_count; i++)
        tmp_in[i] = (__fp16)in[i];

    get_cblas_hgemm()(CblasRowMajor, CblasNoTrans, CblasTrans,
                      M, N, K,
                      (__fp16)1.0f, tmp_in, K, W_h, K,
                      (__fp16)0.0f, tmp_out, N);

    for (int i = 0; i + 3 < out_count; i += 4) {
        vst1q_f32(out + i, vcvt_f32_f16(vld1_f16(tmp_out + i)));
    }
    for (int i = (out_count & ~3); i < out_count; i++)
        out[i] = (float)tmp_out[i];

    if (bias) {
        for (int m = 0; m < M; m++)
            vDSP_vadd(out + m * N, 1, bias, 1, out + m * N, 1, N);
    }
}

/* Fallback: fp16 weights → on-the-fly fp32 conversion + sgemm.
 * Still 2x less memory traffic from weight reads (fp16 is half the cache footprint).
 * Converts weight rows in tiles to amortize conversion overhead. */
#define FP16_TILE_N 128
static void linear_fp16_fallback(float *out, const float *in, const __fp16 *W_h,
                                 const float *bias, int M, int K, int N,
                                 float *W_tile) {
    for (int n0 = 0; n0 < N; n0 += FP16_TILE_N) {
        int tile_n = (n0 + FP16_TILE_N <= N) ? FP16_TILE_N : (N - n0);

        /* Convert tile of fp16 weights → fp32 (NEON vectorized, 4-wide) */
        for (int r = 0; r < tile_n; r++) {
            const __fp16 *row = W_h + (size_t)(n0 + r) * K;
            float *dst = W_tile + (size_t)r * K;
            int k = 0;
            for (; k + 3 < K; k += 4) {
                float16x4_t h = vld1_f16(row + k);
                vst1q_f32(dst + k, vcvt_f32_f16(h));
            }
            for (; k < K; k++)
                dst[k] = (float)row[k];
        }

        /* GEMM: out[:,n0:n0+tile_n] = in @ W_tile^T */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, tile_n, K,
                    1.0f, in, K, W_tile, K,
                    0.0f, out + n0, N);
    }

    if (bias) {
        for (int m = 0; m < M; m++)
            vDSP_vadd(out + m * N, 1, bias, 1, out + m * N, 1, N);
    }
}

/* INT8 dequantize-and-GEMM: per-channel symmetric.
 * Weight layout: int8 data [N, K] followed by fp32 scales [N].
 * Dequantizes tiles of weight rows to fp32, then runs sgemm.
 * ~4x smaller than fp32 with ~1% accuracy loss for Conformer weights. */
#define INT8_TILE_N 64
static void linear_int8(float *out, const float *in, const int8_t *W_q,
                        const float *scales, const float *bias,
                        int M, int K, int N, float *W_tile) {
    for (int n0 = 0; n0 < N; n0 += INT8_TILE_N) {
        int tile_n = (n0 + INT8_TILE_N <= N) ? INT8_TILE_N : (N - n0);

        for (int r = 0; r < tile_n; r++) {
            const int8_t *row = W_q + (size_t)(n0 + r) * K;
            float *dst = W_tile + (size_t)r * K;
            float s = scales[n0 + r];
            int k = 0;
            /* NEON vectorized dequantize: 16 int8 → 16 fp32 per iteration */
            for (; k + 15 < K; k += 16) {
                int8x16_t q = vld1q_s8(row + k);
                int16x8_t lo16 = vmovl_s8(vget_low_s8(q));
                int16x8_t hi16 = vmovl_s8(vget_high_s8(q));
                float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                float32x4_t sv = vdupq_n_f32(s);
                vst1q_f32(dst + k,      vmulq_f32(f0, sv));
                vst1q_f32(dst + k + 4,  vmulq_f32(f1, sv));
                vst1q_f32(dst + k + 8,  vmulq_f32(f2, sv));
                vst1q_f32(dst + k + 12, vmulq_f32(f3, sv));
            }
            for (; k < K; k++)
                dst[k] = (float)row[k] * s;
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, tile_n, K,
                    1.0f, in, K, W_tile, K,
                    0.0f, out + n0, N);
    }

    if (bias) {
        for (int m = 0; m < M; m++)
            vDSP_vadd(out + m * N, 1, bias, 1, out + m * N, 1, N);
    }
}

/* Dispatch: native hgemm > fp16 fallback > int8 dequant > fp32 */
static void linear_dispatch(float *out, const float *in, const float *W,
                            const __fp16 *W_h, const float *bias,
                            int M, int K, int N, Workspace *ws) {
    if (W_h) {
        if (ws->fp16_in && ws->fp16_out && get_cblas_hgemm() != NULL) {
            linear_fp16_native(out, in, W_h, bias, M, K, N, ws->fp16_in, ws->fp16_out);
        } else {
            linear_fp16_fallback(out, in, W_h, bias, M, K, N, (float *)ws->fp16_in);
        }
    } else {
        linear(out, in, W, bias, M, K, N);
    }
}

/* Fused SiLU: 3 passes (vneg + vvexpf + NEON divide) instead of 5.
   Reduces memory round-trips from 5 full-buffer scans to 3. */
static void silu_inplace_ws(float *x, int N, float *tmp) {
    vDSP_vneg(x, 1, tmp, 1, N);
    int n = N;
    vvexpf(tmp, tmp, &n);

    int i = 0;
    float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 4 <= N; i += 4) {
        float32x4_t xi = vld1q_f32(x + i);
        float32x4_t ei = vld1q_f32(tmp + i);
        vst1q_f32(x + i, vdivq_f32(xi, vaddq_f32(one, ei)));
    }
    for (; i < N; i++)
        x[i] = x[i] / (1.0f + tmp[i]);
}

/* Fused GLU: 3 passes per row (vneg + vvexpf + NEON a*sigmoid(b)) instead of 5. */
static void glu(float *out, const float *in, int T, int D) {
    for (int t = 0; t < T; t++) {
        const float *a = in + t * 2 * D;
        const float *b = a + D;
        float *o = out + t * D;

        vDSP_vneg(b, 1, o, 1, D);
        int n = D;
        vvexpf(o, o, &n);

        int i = 0;
        float32x4_t one_v = vdupq_n_f32(1.0f);
        for (; i + 4 <= D; i += 4) {
            float32x4_t ai = vld1q_f32(a + i);
            float32x4_t ei = vld1q_f32(o + i);
            vst1q_f32(o + i, vdivq_f32(ai, vaddq_f32(one_v, ei)));
        }
        for (; i < D; i++)
            o[i] = a[i] / (1.0f + o[i]);
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
 * Generate sinusoidal relative positional encoding.
 * For Q length q_len and KV length kv_len, output has q_len + kv_len - 1 entries.
 * Entry k encodes relative position k - (q_len - 1).
 * When q_len == kv_len, this reduces to the standard 2T-1 PE.
 */
static void generate_rel_pe_asym(float *table, int q_len, int kv_len, int D) {
    int pe_len = q_len + kv_len - 1;
    for (int k = 0; k < pe_len; k++) {
        float pos = (float)(k - (q_len - 1));
        for (int d = 0; d < D; d += 2) {
            float freq = 1.0f / powf(10000.0f, (float)d / (float)D);
            table[k * D + d] = sinf(pos * freq);
            if (d + 1 < D)
                table[k * D + d + 1] = cosf(pos * freq);
        }
    }
}

static void generate_rel_pe(float *table, int T, int D) {
    generate_rel_pe_asym(table, T, T, D);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Depthwise 1D convolution — gather/convolve/scatter with 4-channel batching.
 *
 * The original per-element vDSP_dotpr with stride-D input caused T*D function
 * calls and cache-hostile strided reads (4KB apart for D=1024). This version:
 *   1. Gathers 4 adjacent channels per cache line access into contiguous buffers
 *   2. Uses vDSP_conv (AMX-accelerated) for contiguous 1D correlation
 *   3. Scatters results back with NEON 4-wide stores
 *
 * Reduces function calls from T*D to D, and memory traffic by ~4x.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define DW_PAD_T (MAX_SEQ_LEN + 128)

static void depthwise_conv1d(float *out, const float *in,
                             const float *kernel, const float *bias,
                             int T, int D, int K, int causal) {
    int pad_left = causal ? (K - 1) : (K / 2);
    int padded_T = T + K - 1;

    float col0[DW_PAD_T], col1[DW_PAD_T], col2[DW_PAD_T], col3[DW_PAD_T];
    float cv0[DW_PAD_T],  cv1[DW_PAD_T],  cv2[DW_PAD_T],  cv3[DW_PAD_T];

    int d = 0;

    for (; d + 4 <= D; d += 4) {
        memset(col0, 0, (size_t)padded_T * sizeof(float));
        memset(col1, 0, (size_t)padded_T * sizeof(float));
        memset(col2, 0, (size_t)padded_T * sizeof(float));
        memset(col3, 0, (size_t)padded_T * sizeof(float));

        for (int t = 0; t < T; t++) {
            const float *row = in + t * D + d;
            if (t + 8 < T) __builtin_prefetch(in + (t + 8) * D + d, 0, 0);
            col0[pad_left + t] = row[0];
            col1[pad_left + t] = row[1];
            col2[pad_left + t] = row[2];
            col3[pad_left + t] = row[3];
        }

        vDSP_conv(col0, 1, kernel + d * K, 1,
                  cv0, 1, (vDSP_Length)T, (vDSP_Length)K);
        vDSP_conv(col1, 1, kernel + (d + 1) * K, 1,
                  cv1, 1, (vDSP_Length)T, (vDSP_Length)K);
        vDSP_conv(col2, 1, kernel + (d + 2) * K, 1,
                  cv2, 1, (vDSP_Length)T, (vDSP_Length)K);
        vDSP_conv(col3, 1, kernel + (d + 3) * K, 1,
                  cv3, 1, (vDSP_Length)T, (vDSP_Length)K);

        float32x4_t vb = bias
            ? (float32x4_t){bias[d], bias[d+1], bias[d+2], bias[d+3]}
            : vdupq_n_f32(0.0f);

        for (int t = 0; t < T; t++) {
            float32x4_t v = {cv0[t], cv1[t], cv2[t], cv3[t]};
            vst1q_f32(out + t * D + d, vaddq_f32(v, vb));
        }
    }

    for (; d < D; d++) {
        memset(col0, 0, (size_t)padded_T * sizeof(float));
        for (int t = 0; t < T; t++)
            col0[pad_left + t] = in[t * D + d];
        vDSP_conv(col0, 1, kernel + d * K, 1,
                  cv0, 1, (vDSP_Length)T, (vDSP_Length)K);
        float b = bias ? bias[d] : 0.0f;
        for (int t = 0; t < T; t++)
            out[t * D + d] = cv0[t] + b;
    }
}

static void batch_norm_ws(float *out, const float *in,
                          const float *gamma, const float *beta,
                          const float *running_mean, const float *running_var,
                          int T, int D, float *scale, float *shift) {
    const float eps = 1e-5f;
    for (int d = 0; d < D; d++) {
        scale[d] = gamma[d] / sqrtf(running_var[d] + eps);
        shift[d] = beta[d] - scale[d] * running_mean[d];
    }
    for (int t = 0; t < T; t++)
        vDSP_vma(in + t * D, 1, scale, 1, shift, 1, out + t * D, 1, D);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * Conv2D operations for subsampling
 *
 * Data layout: [C, T, F] (channels-first, matching PyTorch)
 * Weight layout: [C_out, C_in, Kh, Kw] (PyTorch convention)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* im2col: extract input patches into a column matrix for GEMM-based conv2d.
   col layout: [C_in * Kh * Kw, T_out * F_out] */
static void im2col(const float *in, float *col,
                   int C_in, int T_in, int F_in,
                   int Kh, int Kw, int Sh, int Sw,
                   int T_out, int F_out) {
    int pad_h = Kh / 2, pad_w = Kw / 2;
    int col_w = T_out * F_out;
    for (int ci = 0; ci < C_in; ci++) {
        for (int kh = 0; kh < Kh; kh++) {
            for (int kw = 0; kw < Kw; kw++) {
                int row = (ci * Kh + kh) * Kw + kw;
                for (int t = 0; t < T_out; t++) {
                    int tt = t * Sh - pad_h + kh;
                    for (int f = 0; f < F_out; f++) {
                        int ff = f * Sw - pad_w + kw;
                        float val = 0.0f;
                        if (tt >= 0 && tt < T_in && ff >= 0 && ff < F_in)
                            val = in[ci * T_in * F_in + tt * F_in + ff];
                        col[row * col_w + t * F_out + f] = val;
                    }
                }
            }
        }
    }
}

/* Conv2D with three dispatch paths:
   1. K=1, G=1: direct cblas_sgemm (pointwise projection, no im2col)
   2. K>1, G=1: im2col + cblas_sgemm (AMX-accelerated)
   3. G>1: per-output scalar (depthwise — small per-channel work) */
static void conv2d_forward(float *out, const float *in,
                           const float *W, const float *bias,
                           int C_in, int C_out, int T_in, int F_in,
                           int Kh, int Kw, int Sh, int Sw, int groups,
                           float *work_buf, size_t work_sz) {
    int pad_h = Kh / 2, pad_w = Kw / 2;
    int T_out = (T_in + 2 * pad_h - Kh) / Sh + 1;
    int F_out = (F_in + 2 * pad_w - Kw) / Sw + 1;
    int out_spatial = T_out * F_out;
    int c_per_group_in  = C_in / groups;
    int c_per_group_out = C_out / groups;

    if (groups == 1 && Kh == 1 && Kw == 1 && Sh == 1 && Sw == 1) {
        /* Pointwise: out[C_out, T*F] = W[C_out, C_in] @ in[C_in, T*F] */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    C_out, out_spatial, C_in,
                    1.0f, W, C_in, in, out_spatial,
                    0.0f, out, out_spatial);
        if (bias) {
            for (int co = 0; co < C_out; co++) {
                float b = bias[co];
                vDSP_vsadd(out + co * out_spatial, 1, &b,
                           out + co * out_spatial, 1, out_spatial);
            }
        }
    } else if (groups == 1) {
        /* Standard conv: im2col + GEMM (AMX-accelerated) */
        int col_h = C_in * Kh * Kw;
        size_t col_need = (size_t)col_h * out_spatial;
        float *col = (col_need <= work_sz && work_buf) ? work_buf
                     : (float *)malloc(col_need * sizeof(float));
        if (col) {
            im2col(in, col, C_in, T_in, F_in, Kh, Kw, Sh, Sw, T_out, F_out);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        C_out, out_spatial, col_h,
                        1.0f, W, col_h, col, out_spatial,
                        0.0f, out, out_spatial);
            if (col != work_buf) free(col);
            if (bias) {
                for (int co = 0; co < C_out; co++) {
                    float b = bias[co];
                    vDSP_vsadd(out + co * out_spatial, 1, &b,
                               out + co * out_spatial, 1, out_spatial);
                }
            }
            return;
        }
        /* Fall through to scalar if malloc fails */
    }

    if (groups > 1 || (groups == 1 && Kh > 1)) {
        /* Depthwise or scalar fallback */
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
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Subsampling: dispatch based on header sub_type
 *
 * Converts [T, n_mels] mel frames → [T/factor, D] encoder input.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Conv1D subsample using im2col + cblas_sgemm per stage.
   Replaces per-call calloc/free with reuse of workspace sub_work buffer. */
static int conv1d_subsample_forward(float *out, const float *mel_in, int T,
                                    const SubsamplingWeights *sw, int n_mels, int D,
                                    float *work_buf) {
    int K = sw->convs[0].kernel;
    int pad = K / 2;

    int T1 = (T + 2 * pad - K) / 2 + 1;

    /* Workspace layout: mid[T1*D], im2col[max(n_mels,D)*K*max(T1,T2)], proj_tmp[T2*D] */
    float *mid = work_buf;
    float *im2col_buf = mid + T1 * D;

    /* Conv1: im2col + GEMM for [T, n_mels] → [T1, D] */
    int col_h1 = n_mels * K;
    for (int ci = 0; ci < n_mels; ci++) {
        for (int k = 0; k < K; k++) {
            int row = ci * K + k;
            for (int t = 0; t < T1; t++) {
                int tt = t * 2 - pad + k;
                im2col_buf[row * T1 + t] = (tt >= 0 && tt < T)
                    ? mel_in[tt * n_mels + ci] : 0.0f;
            }
        }
    }
    /* W: [D, n_mels*K] (co varies fastest over ci*K+k) × col: [n_mels*K, T1] → mid: [D, T1] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                D, T1, col_h1,
                1.0f, sw->convs[0].w, col_h1, im2col_buf, T1,
                0.0f, mid, T1);
    /* Transpose [D, T1] → mid_row[T1, D] and apply bias + ReLU */
    {
        float *mid_row = im2col_buf;
        for (int t = 0; t < T1; t++)
            for (int co = 0; co < D; co++)
                mid_row[t * D + co] = mid[co * T1 + t];
        if (sw->convs[0].b) {
            for (int t = 0; t < T1; t++)
                vDSP_vadd(mid_row + t * D, 1, sw->convs[0].b, 1,
                          mid_row + t * D, 1, D);
        }
        float zero = 0.0f;
        vDSP_vthres(mid_row, 1, &zero, mid_row, 1, T1 * D);
        memcpy(mid, mid_row, (size_t)T1 * D * sizeof(float));
    }

    /* Conv2: [T1, D] → [T2, D] with ReLU */
    int T2 = (T1 + 2 * pad - K) / 2 + 1;
    int col_h2 = D * K;
    for (int ci = 0; ci < D; ci++) {
        for (int k = 0; k < K; k++) {
            int row = ci * K + k;
            for (int t = 0; t < T2; t++) {
                int tt = t * 2 - pad + k;
                im2col_buf[row * T2 + t] = (tt >= 0 && tt < T1)
                    ? mid[tt * D + ci] : 0.0f;
            }
        }
    }
    float *out_col = mid;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                D, T2, col_h2,
                1.0f, sw->convs[1].w, col_h2, im2col_buf, T2,
                0.0f, out_col, T2);
    for (int t = 0; t < T2; t++)
        for (int co = 0; co < D; co++)
            out[t * D + co] = out_col[co * T2 + t];
    if (sw->convs[1].b) {
        for (int t = 0; t < T2; t++)
            vDSP_vadd(out + t * D, 1, sw->convs[1].b, 1,
                      out + t * D, 1, D);
    }
    {
        float zero = 0.0f;
        vDSP_vthres(out, 1, &zero, out, 1, T2 * D);
    }

    /* Linear projection (reuse mid as scratch) */
    if (sw->proj_w) {
        float *tmp = mid;
        linear(tmp, out, sw->proj_w, sw->proj_b, T2, sw->proj_in, sw->proj_out);
        memcpy(out, tmp, (size_t)T2 * sw->proj_out * sizeof(float));
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
    int C_cur = 1, T_cur = T, F_cur = n_mels;

    float *cur  = work;
    size_t max_elems = (size_t)D * T * n_mels;
    float *next = work + max_elems;
    memcpy(cur, mel_in, (size_t)T * n_mels * sizeof(float));

    for (int i = 0; i < sw->n_convs; i++) {
        int c_in  = sw->convs[i].c_in;
        int c_out = sw->convs[i].c_out;
        int K     = sw->convs[i].kernel;
        int S     = sw->convs[i].stride;
        int G     = sw->convs[i].groups;

        int pad_h = K / 2, pad_w = K / 2;
        int T_next = (T_cur + 2 * pad_h - K) / S + 1;
        int F_next = (F_cur + 2 * pad_w - K) / S + 1;

        conv2d_forward(next, cur, sw->convs[i].w, sw->convs[i].b,
                       c_in, c_out, T_cur, F_cur, K, K, S, S, G,
                       NULL, 0);

        int out_n = c_out * T_next * F_next;

        /* NeMo dw_striding: ReLU after standard conv (G==1 && K>1)
           and after pointwise conv (G==1 && K==1) that ends a dw→pw→relu triplet.
           Depthwise convs (G>1) have no activation.
           Simplified: ReLU after all non-depthwise convs. */
        if (G == 1)
            relu_inplace(next, out_n);

        float *tmp = cur; cur = next; next = tmp;
        C_cur = c_out;
        T_cur = T_next;
        F_cur = F_next;
    }

    /* Flatten [C_cur, T_cur, F_cur] → [T_cur, C_cur * F_cur] (channels-first to row-major) */
    int feat_dim = C_cur * F_cur;
    for (int t = 0; t < T_cur; t++) {
        for (int c = 0; c < C_cur; c++) {
            for (int f = 0; f < F_cur; f++) {
                next[t * feat_dim + c * F_cur + f] = cur[c * T_cur * F_cur + t * F_cur + f];
            }
        }
    }

    /* Linear projection: [T_cur, feat_dim] → [T_cur, D] */
    linear(out, next, sw->proj_w, sw->proj_b, T_cur, sw->proj_in, sw->proj_out);

    return T_cur;
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
        return conv1d_subsample_forward(out, mel_in, T, sw, n_mels, D,
                                               stt->work.sub_work);
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

    linear_dispatch(Q, in, bw->attn_q_w, bw->attn_q_w_h, bw->attn_q_b, T, D, D, ws);
    linear_dispatch(K, in, bw->attn_k_w, bw->attn_k_w_h, bw->attn_k_b, T, D, D, ws);
    linear_dispatch(V, in, bw->attn_v_w, bw->attn_v_w_h, bw->attn_v_b, T, D, D, ws);

    float scale = 1.0f / sqrtf((float)d_head);
    float *attn_out = ws->buf_b;

    /* NeMo relative PE: generate sinusoidal PE [2T-1, D], project through linear_pos */
    int pe_len = 2 * T - 1;
    float *pe_proj = NULL;
    if (use_rel_pe && bw->attn_linear_pos_w && ws->rel_pe) {
        generate_rel_pe(ws->rel_pe, T, D);
        pe_proj = ws->rel_pe_proj;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    pe_len, D, D,
                    1.0f, ws->rel_pe, D, bw->attn_linear_pos_w, D,
                    0.0f, pe_proj, D);
    }

    for (int h = 0; h < n_heads; h++) {
        float *Qh = ws->attn_scores;
        float *Kh = ws->attn_scores + T * d_head;
        float *Vh = ws->attn_scores + T * d_head * 2;
        float *scores = ws->attn_scores + T * d_head * 3;

        for (int t = 0; t < T; t++) {
            if (t + 4 < T) {
                __builtin_prefetch(Q + (t + 4) * D + h * d_head, 0, 1);
                __builtin_prefetch(K + (t + 4) * D + h * d_head, 0, 1);
                __builtin_prefetch(V + (t + 4) * D + h * d_head, 0, 1);
            }
            memcpy(Qh + t * d_head, Q + t * D + h * d_head, d_head * sizeof(float));
            memcpy(Kh + t * d_head, K + t * D + h * d_head, d_head * sizeof(float));
            memcpy(Vh + t * d_head, V + t * D + h * d_head, d_head * sizeof(float));
        }

        /* Content attention: (Q + pos_bias_u) @ K^T */
        if (use_rel_pe && bw->attn_pos_bias_u) {
            float *Qu = Qh;
            const float *bias_u = bw->attn_pos_bias_u + h * d_head;
            for (int t = 0; t < T; t++)
                vDSP_vadd(Qu + t * d_head, 1, bias_u, 1, Qu + t * d_head, 1, d_head);
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, T, d_head,
                    scale, Qh, d_head, Kh, d_head,
                    0.0f, scores, T);

        /* Positional attention: (Q + pos_bias_v) @ PE_proj^T with rel_shift */
        if (pe_proj && bw->attn_pos_bias_v) {
            /* Extract per-head PE: pe_proj[:, h*d_head:(h+1)*d_head] */
            float *PEh = ws->attn_scores + T * d_head * 3 + T * T;
            for (int p = 0; p < pe_len; p++)
                memcpy(PEh + p * d_head, pe_proj + p * D + h * d_head,
                       d_head * sizeof(float));

            /* Qv = Q_orig + pos_bias_v */
            float *Qv = ws->attn_scores + T * d_head * 3 + T * T + pe_len * d_head;
            for (int t = 0; t < T; t++) {
                memcpy(Qv + t * d_head, Q + t * D + h * d_head, d_head * sizeof(float));
                const float *bias_v = bw->attn_pos_bias_v + h * d_head;
                vDSP_vadd(Qv + t * d_head, 1, bias_v, 1, Qv + t * d_head, 1, d_head);
            }

            /* pos_scores = Qv @ PEh^T → [T, pe_len] */
            float *pos_raw = Qv + T * d_head;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, pe_len, d_head,
                        scale, Qv, d_head, PEh, d_head,
                        0.0f, pos_raw, pe_len);

            /* rel_shift via vectorized reversed-stride add.
               k = (T-1)+i-j is always in [0, pe_len) for i,j ∈ [0,T),
               and decreases as j increases — use vDSP_vadd with stride -1. */
            for (int i = 0; i < T; i++) {
                vDSP_vadd(scores + i * T, 1,
                          pos_raw + i * pe_len + (T - 1 + i), -1,
                          scores + i * T, 1, (vDSP_Length)T);
            }
        }

        for (int t = 0; t < T; t++) {
            if (t + 2 < T) __builtin_prefetch(scores + (t + 2) * T, 0, 1);
            softmax_row(scores + t * T, T);
        }

        float *ctx = Qh;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, d_head, T,
                    1.0f, scores, T, Vh, d_head,
                    0.0f, ctx, d_head);

        for (int t = 0; t < T; t++)
            memcpy(attn_out + t * D + h * d_head, ctx + t * d_head,
                   d_head * sizeof(float));
    }

    linear_dispatch(out, attn_out, bw->attn_out_w, bw->attn_out_w_h,
                    bw->attn_out_b, T, D, D, ws);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Convolution module
 * ═══════════════════════════════════════════════════════════════════════════ */

static void conv_module_forward(float *out, const float *in,
                                const ConformerBlockWeights *bw,
                                Workspace *ws, int T, int D, int K) {
    float *normed = ws->buf_b;
    layer_norm(normed, in, bw->conv_norm_w, bw->conv_norm_b, T, D);
    linear_dispatch(ws->conv_mid, normed, bw->conv_pw1_w, bw->conv_pw1_w_h,
                    bw->conv_pw1_b, T, D, 2 * D, ws);
    glu(normed, ws->conv_mid, T, D);
    depthwise_conv1d(out, normed, bw->conv_dw_w, bw->conv_dw_b, T, D, K, 0);
    batch_norm_ws(normed, out, bw->conv_bn_gamma, bw->conv_bn_beta,
                  bw->conv_bn_mean, bw->conv_bn_var, T, D,
                  ws->bn_scale, ws->bn_shift);
    silu_inplace_ws(normed, T * D, ws->silu_tmp);
    linear_dispatch(out, normed, bw->conv_pw2_w, bw->conv_pw2_w_h,
                    bw->conv_pw2_b, T, D, D, ws);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Feed-Forward module (Macaron-style half-step)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void ffn_forward(float *out, const float *in,
                        const float *norm_w, const float *norm_b,
                        const float *up_w, const float *up_b,
                        const float *down_w, const float *down_b,
                        const __fp16 *up_w_h, const __fp16 *down_w_h,
                        Workspace *ws, int T, int D, int ff_dim) {
    float *normed = ws->buf_b;
    layer_norm(normed, in, norm_w, norm_b, T, D);
    linear_dispatch(ws->ff_mid, normed, up_w, up_w_h, up_b, T, D, ff_dim, ws);
    silu_inplace_ws(ws->ff_mid, T * ff_dim, ws->silu_tmp);
    linear_dispatch(out, ws->ff_mid, down_w, down_w_h, down_b, T, ff_dim, D, ws);
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
    int ff_dim = D * ff_mult;
    float *tmp = ws->residual;

    ffn_forward(tmp, x, bw->ff1_norm_w, bw->ff1_norm_b,
                bw->ff1_up_w, bw->ff1_up_b,
                bw->ff1_down_w, bw->ff1_down_b,
                bw->ff1_up_w_h, bw->ff1_down_w_h,
                ws, T, D, ff_dim);
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);

    float *attn_in = ws->buf_b;
    layer_norm(attn_in, x, bw->attn_norm_w, bw->attn_norm_b, T, D);
    mhsa_forward(tmp, attn_in, bw, ws, T, D, n_heads, use_rel_pe);
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);

    conv_module_forward(tmp, x, bw, ws, T, D, conv_kernel);
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);

    ffn_forward(tmp, x, bw->ff2_norm_w, bw->ff2_norm_b,
                bw->ff2_up_w, bw->ff2_up_b,
                bw->ff2_down_w, bw->ff2_down_b,
                bw->ff2_up_w_h, bw->ff2_down_w_h,
                ws, T, D, ff_dim);
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

/**
 * Convert logits [T, V] to log-softmax probabilities in-place.
 * log_softmax(x_i) = x_i - log(sum(exp(x_j - max)))  - max
 *                   = x_i - (log_sum_exp)
 */
static void logits_to_log_probs_ws(float *logits, int T, int V, float *tmp) {
    for (int t = 0; t < T; t++) {
        float *row = logits + t * V;
        float mx;
        vDSP_maxv(row, 1, &mx, V);
        float neg_mx = -mx;
        vDSP_vsadd(row, 1, &neg_mx, tmp, 1, V);
        int Vn = V;
        vvexpf(tmp, tmp, &Vn);
        float sum;
        vDSP_sve(tmp, 1, &sum, V);
        float lse = logf(sum) + mx;
        float neg_lse = -lse;
        vDSP_vsadd(row, 1, &neg_lse, row, 1, V);
    }
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
                                     LayerCache *cache, Workspace *ws) {
    /* Must use causal=0 (symmetric padding) to match non-cached behavior */
    if (!cache || cache->conv_state_len == 0) {
        depthwise_conv1d(out, in, kernel, bias, T, D, K, 0);
    } else {
        int overlap = cache->conv_state_len;
        int total_T = overlap + T;
        float *merged = ws->cache_conv_merged;
        float *full_out = ws->cache_conv_out;
        memcpy(merged, cache->conv_state, (size_t)overlap * D * sizeof(float));
        memcpy(merged + overlap * D, in, (size_t)T * D * sizeof(float));
        depthwise_conv1d(full_out, merged, kernel, bias, total_T, D, K, 0);
        memcpy(out, full_out + overlap * D, (size_t)T * D * sizeof(float));
    }

    if (cache) {
        int save = K / 2;
        if (save > T) save = T;
        if (!cache->conv_state) {
            cache->conv_state = (float *)malloc((size_t)(K - 1) * D * sizeof(float));
            if (!cache->conv_state) { cache->conv_state_len = 0; return; }
        }
        memcpy(cache->conv_state, in + (T - save) * D, (size_t)save * D * sizeof(float));
        cache->conv_state_len = save;
    }
}

/**
 * Cache-aware MHSA with full relative positional encoding.
 * Q from new chunk attends to [cached_K | new_K] and [cached_V | new_V].
 * Relative PE covers the asymmetric range: Q has T frames, KV has total_T.
 * After attention, new K/V are appended to the cache (with eviction if full).
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

    float *Q_new = ws->qkv;
    float *K_new = ws->qkv + T * D;
    float *V_new = ws->qkv + T * 2 * D;

    linear_dispatch(Q_new, in, bw->attn_q_w, bw->attn_q_w_h, bw->attn_q_b, T, D, D, ws);
    linear_dispatch(K_new, in, bw->attn_k_w, bw->attn_k_w_h, bw->attn_k_b, T, D, D, ws);
    linear_dispatch(V_new, in, bw->attn_v_w, bw->attn_v_w_h, bw->attn_v_b, T, D, D, ws);

    float *K_full = ws->cache_k_full;
    float *V_full = ws->cache_v_full;

    if (cached_T > 0) {
        memcpy(K_full, cache->k_cache, (size_t)cached_T * D * sizeof(float));
        memcpy(V_full, cache->v_cache, (size_t)cached_T * D * sizeof(float));
    }
    memcpy(K_full + cached_T * D, K_new, (size_t)T * D * sizeof(float));
    memcpy(V_full + cached_T * D, V_new, (size_t)T * D * sizeof(float));

    float scale = 1.0f / sqrtf((float)d_head);
    float *attn_out = ws->buf_b;

    /* Relative PE for asymmetric Q[T] × K[total_T] attention */
    int pe_len = T + total_T - 1;
    float *pe_proj = NULL;
    if (use_rel_pe && bw->attn_linear_pos_w && ws->rel_pe) {
        generate_rel_pe_asym(ws->rel_pe, T, total_T, D);
        pe_proj = ws->rel_pe_proj;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    pe_len, D, D,
                    1.0f, ws->rel_pe, D, bw->attn_linear_pos_w, D,
                    0.0f, pe_proj, D);
    }

    float *scores = ws->cache_scores;
    float *Qh  = ws->cache_qh;
    float *Kh  = ws->cache_kh;
    float *Vh  = ws->cache_vh;
    float *ctx = ws->cache_ctx;

    /* Positional attention scratch — reuse tail end of attn_scores buffer */
    float *PEh     = NULL;
    float *Qv      = NULL;
    float *pos_raw = NULL;
    if (pe_proj) {
        size_t off = (size_t)MAX_SEQ_LEN * (CACHE_MAX_CONTEXT + MAX_SEQ_LEN);
        PEh     = ws->cache_scores + off;
        Qv      = PEh + pe_len * d_head;
        pos_raw = Qv + T * d_head;
    }

    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < T; t++) {
            if (t + 4 < T)
                __builtin_prefetch(Q_new + (t + 4) * D + h * d_head, 0, 1);
            memcpy(Qh + t * d_head, Q_new + t * D + h * d_head,
                   d_head * sizeof(float));
        }
        for (int t = 0; t < total_T; t++) {
            if (t + 4 < total_T) {
                __builtin_prefetch(K_full + (t + 4) * D + h * d_head, 0, 1);
                __builtin_prefetch(V_full + (t + 4) * D + h * d_head, 0, 1);
            }
            memcpy(Kh + t * d_head, K_full + t * D + h * d_head,
                   d_head * sizeof(float));
            memcpy(Vh + t * d_head, V_full + t * D + h * d_head,
                   d_head * sizeof(float));
        }

        /* Content attention: (Q + pos_bias_u) @ K^T */
        if (use_rel_pe && bw->attn_pos_bias_u) {
            const float *bias_u = bw->attn_pos_bias_u + h * d_head;
            for (int t = 0; t < T; t++)
                vDSP_vadd(Qh + t * d_head, 1, bias_u, 1,
                          Qh + t * d_head, 1, d_head);
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, total_T, d_head,
                    scale, Qh, d_head, Kh, d_head,
                    0.0f, scores, total_T);

        /* Positional attention: (Q + pos_bias_v) @ PE_proj^T with rel_shift */
        if (pe_proj && bw->attn_pos_bias_v) {
            for (int p = 0; p < pe_len; p++)
                memcpy(PEh + p * d_head, pe_proj + p * D + h * d_head,
                       d_head * sizeof(float));

            for (int t = 0; t < T; t++) {
                memcpy(Qv + t * d_head, Q_new + t * D + h * d_head,
                       d_head * sizeof(float));
                const float *bias_v = bw->attn_pos_bias_v + h * d_head;
                vDSP_vadd(Qv + t * d_head, 1, bias_v, 1,
                          Qv + t * d_head, 1, d_head);
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, pe_len, d_head,
                        scale, Qv, d_head, PEh, d_head,
                        0.0f, pos_raw, pe_len);

            /* rel_shift via vectorized reversed-stride add (cached variant).
               k = (T-1)+cached_T+i-j is always in [0, pe_len). */
            for (int i = 0; i < T; i++) {
                int start_k = (T - 1) + cached_T + i;
                vDSP_vadd(scores + i * total_T, 1,
                          pos_raw + i * pe_len + start_k, -1,
                          scores + i * total_T, 1, (vDSP_Length)total_T);
            }
        }

        for (int t = 0; t < T; t++) {
            if (t + 2 < T) __builtin_prefetch(scores + (t + 2) * total_T, 0, 1);
            softmax_row(scores + t * total_T, total_T);
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, d_head, total_T,
                    1.0f, scores, total_T, Vh, d_head,
                    0.0f, ctx, d_head);

        for (int t = 0; t < T; t++)
            memcpy(attn_out + t * D + h * d_head, ctx + t * d_head,
                   d_head * sizeof(float));
    }

    linear_dispatch(out, attn_out, bw->attn_out_w, bw->attn_out_w_h,
                    bw->attn_out_b, T, D, D, ws);

    /* Update cache: keep the most recent CACHE_MAX_CONTEXT frames of K/V */
    if (T >= CACHE_MAX_CONTEXT) {
        /* New chunk alone fills the cache — keep its last CACHE_MAX_CONTEXT frames */
        int skip = T - CACHE_MAX_CONTEXT;
        memcpy(cache->k_cache, K_new + skip * D,
               (size_t)CACHE_MAX_CONTEXT * D * sizeof(float));
        memcpy(cache->v_cache, V_new + skip * D,
               (size_t)CACHE_MAX_CONTEXT * D * sizeof(float));
        cache->k_len = CACHE_MAX_CONTEXT;
        cache->v_len = CACHE_MAX_CONTEXT;
    } else {
        int new_total = cached_T + T;
        if (new_total > CACHE_MAX_CONTEXT) {
            int evict = new_total - CACHE_MAX_CONTEXT;
            int keep = cached_T - evict;
            if (keep > 0) {
                memmove(cache->k_cache, cache->k_cache + evict * D,
                        (size_t)keep * D * sizeof(float));
                memmove(cache->v_cache, cache->v_cache + evict * D,
                        (size_t)keep * D * sizeof(float));
            }
            cached_T = keep > 0 ? keep : 0;
        }
        memcpy(cache->k_cache + cached_T * D, K_new,
               (size_t)T * D * sizeof(float));
        memcpy(cache->v_cache + cached_T * D, V_new,
               (size_t)T * D * sizeof(float));
        cache->k_len = cached_T + T;
        cache->v_len = cached_T + T;
    }

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
                bw->ff1_up_w_h, bw->ff1_down_w_h,
                ws, T, D, ff_dim);
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);

    float *attn_in = ws->buf_b;
    layer_norm(attn_in, x, bw->attn_norm_w, bw->attn_norm_b, T, D);
    mhsa_forward_cached(tmp, attn_in, bw, ws, T, D, n_heads, use_rel_pe, cache);
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);

    /* Conv module with cached overlap */
    float *conv_normed = ws->buf_b;
    layer_norm(conv_normed, x, bw->conv_norm_w, bw->conv_norm_b, T, D);
    linear_dispatch(ws->conv_mid, conv_normed, bw->conv_pw1_w, bw->conv_pw1_w_h,
                    bw->conv_pw1_b, T, D, 2 * D, ws);
    glu(conv_normed, ws->conv_mid, T, D);
    depthwise_conv1d_cached(tmp, conv_normed, bw->conv_dw_w, bw->conv_dw_b,
                             T, D, conv_kernel, cache, ws);
    batch_norm_ws(conv_normed, tmp, bw->conv_bn_gamma, bw->conv_bn_beta,
                  bw->conv_bn_mean, bw->conv_bn_var, T, D,
                  ws->bn_scale, ws->bn_shift);
    silu_inplace_ws(conv_normed, T * D, ws->silu_tmp);
    linear_dispatch(tmp, conv_normed, bw->conv_pw2_w, bw->conv_pw2_w_h,
                    bw->conv_pw2_b, T, D, D, ws);
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);

    ffn_forward(tmp, x, bw->ff2_norm_w, bw->ff2_norm_b,
                bw->ff2_up_w, bw->ff2_up_b,
                bw->ff2_down_w, bw->ff2_down_b,
                bw->ff2_up_w_h, bw->ff2_down_w_h,
                ws, T, D, ff_dim);
    vDSP_vadd(x, 1, tmp, 1, x, 1, T * D);

    layer_norm(x, x, bw->final_norm_w, bw->final_norm_b, T, D);
}

static int full_forward(ConformerSTT *stt, const float *mel_in, int T) {
    int D         = (int)stt->header.d_model;
    int n_heads   = (int)stt->header.n_heads;
    int ff_mult   = (int)stt->header.ff_mult;
    int conv_kern = (int)stt->header.conv_kernel;
    int vocab     = (int)stt->header.vocab_size;
    int use_rel_pe = (stt->header.flags & CSTT_FLAG_REL_PE) ? 1 : 0;
    ModelWeights *w = &stt->weights;
    Workspace *ws   = &stt->work;

    int n_mels = (int)stt->header.n_mels;
    float *mel_norm = ws->mel_norm_buf;
    memcpy(mel_norm, mel_in, (size_t)T * n_mels * sizeof(float));
    if (!stt->skip_mel_normalize) {
        if (stt->cache_aware && stt->running_sum) {
            per_feature_normalize_running(mel_norm, T, n_mels,
                                          stt->running_sum,
                                          stt->running_sum_sq,
                                          &stt->running_count);
        } else {
            per_feature_normalize(mel_norm, T, n_mels);
        }
    }

    /* Try external forward hook (BNNS/ANE, Metal, etc.) — covers the
     * entire encoder: subsampling + conformer blocks + CTC head. */
    if (stt->external_forward) {
        int max_T_sub = T / (int)stt->header.subsample_factor + 1;
        if (max_T_sub > MAX_SEQ_LEN) max_T_sub = MAX_SEQ_LEN;
        int ext_T_sub = stt->external_forward(
            stt->external_forward_ctx,
            mel_norm, T, n_mels,
            ws->logits, max_T_sub);
        if (ext_T_sub > 0) {
            stt->total_frames_processed += ext_T_sub;
            return ext_T_sub;
        }
        /* ext returned -1: fallback to built-in */
    }

    int T_sub = subsample_forward(ws->buf_a, mel_norm, T, stt);
    if (T_sub <= 0) return -1;
    if (T_sub > MAX_SEQ_LEN) T_sub = MAX_SEQ_LEN;

    if (stt->header.flags & CSTT_FLAG_XSCALING) {
        float scale = sqrtf((float)D);
        vDSP_vsmul(ws->buf_a, 1, &scale, ws->buf_a, 1, T_sub * D);
    }

    if (!use_rel_pe)
        add_sinusoidal_pe(ws->buf_a, T_sub, D);

    if (stt->cache_aware && stt->layer_caches) {
        for (int i = 0; i < w->n_blocks; i++) {
            /* Prefetch next layer's weights into L2 while computing this layer */
            if (i + 1 < w->n_blocks) {
                const float *next_w = w->blocks[i + 1].ff1_up_w;
                if (next_w) {
                    for (int p = 0; p < D * ff_mult * D; p += 16)
                        __builtin_prefetch(next_w + p, 0, 1);
                }
            }
            conformer_block_forward_cached(ws->buf_a, &w->blocks[i],
                                            ws, T_sub, D, n_heads, ff_mult, conv_kern,
                                            use_rel_pe, &stt->layer_caches[i]);
        }
    } else {
        for (int i = 0; i < w->n_blocks; i++) {
            if (i + 1 < w->n_blocks) {
                const float *next_w = w->blocks[i + 1].ff1_up_w;
                if (next_w) {
                    for (int p = 0; p < D * ff_mult * D; p += 16)
                        __builtin_prefetch(next_w + p, 0, 1);
                }
            }
            conformer_block_forward(ws->buf_a, &w->blocks[i],
                                    ws, T_sub, D, n_heads, ff_mult, conv_kern,
                                    use_rel_pe);
        }
    }

    linear_dispatch(ws->logits, ws->buf_a, w->ctc_w, w->ctc_w_h,
                    w->ctc_b, T_sub, D, vocab, ws);
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
    if (!v->tokens) { v->size = 0; return; }
    for (int i = 0; i < n; i++) {
        v->tokens[i] = strdup(charset[i]);
        if (!v->tokens[i]) { v->size = i; return; }
    }
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
            char **tmp = (char **)realloc(v->tokens, cap * sizeof(char *));
            if (!tmp) break;
            v->tokens = tmp;
        }
        v->tokens[v->size] = strdup(line);
        if (!v->tokens[v->size]) break;
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

/*
 * Weight reading: supports fp32 (direct mmap pointer) and fp16 (promote to fp32 buffer).
 * For fp16, we advance a byte cursor by count*2 and promote into a pre-allocated fp32 arena.
 */
static const float *read_weight_fp32(const char **cursor, int count) {
    const float *ptr = (const float *)*cursor;
    *cursor += (size_t)count * sizeof(float);
    return ptr;
}

static const float *read_weight_fp16(const char **cursor, float **arena, int count) {
    const uint16_t *src = (const uint16_t *)*cursor;
    float *dst = *arena;
    for (int i = 0; i < count; i++) {
        uint32_t h = src[i];
        uint32_t sign = (h >> 15) & 1;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f;
        if (exp == 0) {
            if (mant == 0) { f = sign << 31; }
            else {
                exp = 1;
                while (!(mant & 0x400)) { mant <<= 1; exp--; }
                mant &= 0x3FF;
                f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            f = (sign << 31) | 0x7F800000 | (mant << 13);
        } else {
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        memcpy(&dst[i], &f, 4);
    }
    *cursor += (size_t)count * sizeof(uint16_t);
    *arena += count;
    return dst;
}

/* INT8 weight reading: per-channel symmetric quantization.
 * Data layout: int8_t[N*K] followed by float scales[N].
 * Dequantizes into the fp32 arena for use with cblas_sgemm. */
static const float *read_weight_int8(const char **cursor, float **arena,
                                      int count, int n_rows, const int8_t **q_out,
                                      const float **scales_out) {
    int K = count / n_rows;
    const int8_t *q = (const int8_t *)*cursor;
    *cursor += (size_t)count * sizeof(int8_t);
    const float *sc = (const float *)*cursor;
    *cursor += (size_t)n_rows * sizeof(float);

    if (q_out) *q_out = q;
    if (scales_out) *scales_out = sc;

    float *dst = *arena;
    for (int r = 0; r < n_rows; r++) {
        float s = sc[r];
        for (int k = 0; k < K; k++)
            dst[r * K + k] = (float)q[r * K + k] * s;
    }
    *arena += count;
    return dst;
}

/* Read a bias/norm weight as fp32 regardless of model dtype (biases are never quantized) */
static const float *read_weight_always_fp32(const char **cursor, int count) {
    const float *ptr = (const float *)*cursor;
    *cursor += (size_t)count * sizeof(float);
    return ptr;
}

/* Unified read_weight: dispatches to fp32, fp16, or int8 based on dtype flag. */
typedef struct {
    const char *cursor;
    float *arena;
    int is_fp16;
    int is_int8;
    int current_n_rows;   /* For INT8: number of output rows (set before calling read_weight for matrices) */
    const __fp16 *last_fp16;  /* Raw fp16 address of last read (for GEMM path) */
    const int8_t *last_int8;  /* Raw INT8 quantized weights of last read */
    const float  *last_int8_scales; /* Per-channel scales of last INT8 read */
} WeightReader;

static const float *read_weight(WeightReader *r, int count) {
    if (r->is_int8 && r->current_n_rows > 0) {
        r->last_fp16 = NULL;
        const float *w = read_weight_int8(&r->cursor, &r->arena, count,
                                           r->current_n_rows, &r->last_int8,
                                           &r->last_int8_scales);
        r->current_n_rows = 0;
        return w;
    } else if (r->is_fp16) {
        r->last_fp16 = (const __fp16 *)r->cursor;
        r->last_int8 = NULL;
        r->last_int8_scales = NULL;
        return read_weight_fp16(&r->cursor, &r->arena, count);
    } else {
        r->last_fp16 = NULL;
        r->last_int8 = NULL;
        r->last_int8_scales = NULL;
        return read_weight_fp32(&r->cursor, count);
    }
}

/* For INT8, biases and norms are stored as fp32. This macro reads them as fp32
 * regardless of dtype, then advances the reader cursor. */
static const float *read_bias(WeightReader *r, int count) {
    if (r->is_int8) {
        return read_weight_always_fp32(&r->cursor, count);
    }
    return read_weight(r, count);
}

static int load_block_weights(ConformerBlockWeights *bw, WeightReader *r,
                              int D, int ff_dim, int K, int n_heads, int has_rel_pe) {
    bw->ff1_norm_w    = read_bias(r, D);
    bw->ff1_norm_b    = read_bias(r, D);
    r->current_n_rows = ff_dim;
    bw->ff1_up_w      = read_weight(r, D * ff_dim);
    bw->ff1_up_w_h    = r->last_fp16;
    bw->ff1_up_b      = read_bias(r, ff_dim);
    r->current_n_rows = D;
    bw->ff1_down_w    = read_weight(r, ff_dim * D);
    bw->ff1_down_w_h  = r->last_fp16;
    bw->ff1_down_b    = read_bias(r, D);

    bw->attn_norm_w   = read_bias(r, D);
    bw->attn_norm_b   = read_bias(r, D);
    r->current_n_rows = D;
    bw->attn_q_w      = read_weight(r, D * D);
    bw->attn_q_w_h    = r->last_fp16;
    bw->attn_q_b      = read_bias(r, D);
    r->current_n_rows = D;
    bw->attn_k_w      = read_weight(r, D * D);
    bw->attn_k_w_h    = r->last_fp16;
    bw->attn_k_b      = read_bias(r, D);
    r->current_n_rows = D;
    bw->attn_v_w      = read_weight(r, D * D);
    bw->attn_v_w_h    = r->last_fp16;
    bw->attn_v_b      = read_bias(r, D);
    r->current_n_rows = D;
    bw->attn_out_w    = read_weight(r, D * D);
    bw->attn_out_w_h  = r->last_fp16;
    bw->attn_out_b    = read_bias(r, D);

    if (has_rel_pe) {
        r->current_n_rows = D;
        bw->attn_linear_pos_w = read_weight(r, D * D);
        bw->attn_linear_pos_w_h = r->last_fp16;
        bw->attn_pos_bias_u = read_bias(r, n_heads * (D / n_heads));
        bw->attn_pos_bias_v = read_bias(r, n_heads * (D / n_heads));
    } else {
        bw->attn_linear_pos_w = NULL;
        bw->attn_linear_pos_w_h = NULL;
        bw->attn_pos_bias_u = NULL;
        bw->attn_pos_bias_v = NULL;
    }

    bw->conv_norm_w   = read_bias(r, D);
    bw->conv_norm_b   = read_bias(r, D);
    r->current_n_rows = 2 * D;
    bw->conv_pw1_w    = read_weight(r, 2 * D * D);
    bw->conv_pw1_w_h  = r->last_fp16;
    bw->conv_pw1_b    = read_bias(r, 2 * D);
    bw->conv_dw_w     = read_bias(r, D * K);  /* Depthwise conv: small, kept as fp32 */
    bw->conv_dw_b     = read_bias(r, D);
    bw->conv_bn_gamma = read_bias(r, D);
    bw->conv_bn_beta  = read_bias(r, D);
    bw->conv_bn_mean  = read_bias(r, D);
    bw->conv_bn_var   = read_bias(r, D);
    r->current_n_rows = D;
    bw->conv_pw2_w    = read_weight(r, D * D);
    bw->conv_pw2_w_h  = r->last_fp16;
    bw->conv_pw2_b    = read_bias(r, D);

    bw->ff2_norm_w    = read_bias(r, D);
    bw->ff2_norm_b    = read_bias(r, D);
    r->current_n_rows = ff_dim;
    bw->ff2_up_w      = read_weight(r, D * ff_dim);
    bw->ff2_up_w_h    = r->last_fp16;
    bw->ff2_up_b      = read_bias(r, ff_dim);
    r->current_n_rows = D;
    bw->ff2_down_w    = read_weight(r, ff_dim * D);
    bw->ff2_down_w_h  = r->last_fp16;
    bw->ff2_down_b    = read_bias(r, D);

    bw->final_norm_w  = read_bias(r, D);
    bw->final_norm_b  = read_bias(r, D);

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
static int load_subsampling_weights(SubsamplingWeights *sw, WeightReader *r,
                                    const CSTTHeader *h) {
    int D = (int)h->d_model;
    int n_mels = (int)h->n_mels;

    if (h->sub_type == CSTT_SUB_CONV1D && h->n_sub_convs == 0) {
        /* Legacy Conv1D format */
        int K = h->sub_conv_kernel > 0 ? (int)h->sub_conv_kernel : 3;
        sw->n_convs = 2;
        sw->convs[0].w = read_weight(r, D * n_mels * K);
        sw->convs[0].b = read_weight(r, D);
        sw->convs[0].c_in = n_mels; sw->convs[0].c_out = D;
        sw->convs[0].kernel = K; sw->convs[0].stride = 2; sw->convs[0].groups = 1;

        sw->convs[1].w = read_weight(r, D * D * K);
        sw->convs[1].b = read_weight(r, D);
        sw->convs[1].c_in = D; sw->convs[1].c_out = D;
        sw->convs[1].kernel = K; sw->convs[1].stride = 2; sw->convs[1].groups = 1;

        sw->proj_w  = read_weight(r, D * D);
        sw->proj_b  = read_weight(r, D);
        sw->proj_in = D;
        sw->proj_out = D;
    } else {
        /* General format: conv descriptors are always uint32 metadata (not fp16) */
        sw->n_convs = (int)h->n_sub_convs;
        if (sw->n_convs > MAX_SUB_CONVS) return -1;

        for (int i = 0; i < sw->n_convs; i++) {
            const uint32_t *desc = (const uint32_t *)r->cursor;
            sw->convs[i].c_in   = (int)desc[0];
            sw->convs[i].c_out  = (int)desc[1];
            sw->convs[i].kernel = (int)desc[2];
            sw->convs[i].stride = (int)desc[3];
            sw->convs[i].groups = (int)desc[4];
            r->cursor += 5 * sizeof(uint32_t);
        }

        for (int i = 0; i < sw->n_convs; i++) {
            int ci = sw->convs[i].c_in / sw->convs[i].groups;
            int co = sw->convs[i].c_out;
            int K2 = sw->convs[i].kernel * sw->convs[i].kernel;
            sw->convs[i].w = read_weight(r, co * ci * K2);
            sw->convs[i].b = read_weight(r, co);
        }

        int feat_in = (int)h->sub_feat_in;
        sw->proj_w  = read_weight(r, D * feat_in);
        sw->proj_b  = read_weight(r, D);
        sw->proj_in = feat_in;
        sw->proj_out = D;
    }

    return 0;
}

static size_t estimate_total_weights(const CSTTHeader *h) {
    int D = (int)h->d_model;
    int ff_dim = D * (int)h->ff_mult;
    int K = (int)h->conv_kernel;
    int V = (int)h->vocab_size;
    int n = (int)h->n_layers;
    int has_rel = (h->flags & CSTT_FLAG_REL_PE) ? 1 : 0;

    size_t per_block = (size_t)(
        6*D + 2*D*ff_dim + 2*ff_dim +    /* FFN1 */
        2*D + 4*(size_t)D*D + 4*D +      /* MHSA */
        (has_rel ? (D*D + 2*D) : 0) +    /* Rel PE */
        2*D + 2*D*(size_t)D + 2*D +      /* Conv pw1 */
        D*K + D +                          /* Conv dw */
        4*D +                              /* Conv BN */
        D*(size_t)D + D +                 /* Conv pw2 */
        6*D + 2*D*(size_t)ff_dim + 2*ff_dim + /* FFN2 */
        2*D                                /* Final norm */
    );
    size_t ctc = (size_t)D * V + V;
    /* Rough estimate for subsampling — overallocate by 2x is fine */
    size_t sub = (size_t)D * D * 4;
    size_t total = per_block * n + ctc + sub;

    /* TDT decoder weights (if present) */
    if (h->flags & CSTT_FLAG_TDT) {
        int pred_h = (int)h->reserved[0];
        int pred_l = (int)h->reserved[1];
        int n_dur  = (int)h->reserved[2];
        int joint_d = (int)h->reserved[3];
        if (pred_h > 0) {
            size_t embed = (size_t)V * pred_h;
            size_t lstm = (size_t)pred_l * (4 * pred_h * pred_h * 2 + 4 * pred_h * 2);
            size_t joint = (size_t)joint_d * D + joint_d +
                           (size_t)joint_d * pred_h + joint_d +
                           (size_t)(V + n_dur) * joint_d + (V + n_dur);
            total += embed + lstm + joint;
        }
    }
    return total;
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
    int is_fp16 = (h->dtype == 1) ? 1 : 0;
    int is_int8 = (h->dtype == 2) ? 1 : 0;

    WeightReader reader;
    reader.cursor = (const char *)stt->mmap_base + sizeof(CSTTHeader);
    reader.is_fp16 = is_fp16;
    reader.is_int8 = is_int8;
    reader.arena = NULL;
    reader.current_n_rows = 0;
    reader.last_int8 = NULL;
    reader.last_int8_scales = NULL;

    if (is_fp16 || is_int8) {
        size_t total = estimate_total_weights(h);
        stt->fp16_arena = (float *)malloc(total * sizeof(float));
        if (!stt->fp16_arena) return -1;
        reader.arena = stt->fp16_arena;
        const char *dtype_label = is_int8 ? "int8" : "fp16";
        fprintf(stderr, "[conformer_stt] %s mode: dequantizing to fp32 (%.1f MB arena)\n",
                dtype_label, (float)(total * 4) / (1024 * 1024));
    }

    ModelWeights *w = &stt->weights;

    if (load_subsampling_weights(&w->sub, &reader, h) != 0)
        return -1;

    w->n_blocks = n_layers;
    w->blocks = (ConformerBlockWeights *)calloc(n_layers, sizeof(ConformerBlockWeights));
    if (!w->blocks) return -1;

    for (int i = 0; i < n_layers; i++) {
        if (load_block_weights(&w->blocks[i], &reader, D, ff_dim, K,
                               n_heads, has_rel_pe) != 0)
            return -1;
    }

    w->ctc_w = read_weight(&reader, D * vocab);
    w->ctc_w_h = reader.last_fp16;
    w->ctc_b = read_weight(&reader, vocab);
    w->use_fp16_gemm = is_fp16;

    /* TDT transducer decoder weights */
    if (h->flags & CSTT_FLAG_TDT) {
        /* Read TDT sub-header: [pred_hidden, pred_layers, n_durations, joint_dim] */
        uint32_t tdt_hdr[4];
        memcpy(tdt_hdr, reader.cursor, sizeof(tdt_hdr));
        reader.cursor += sizeof(tdt_hdr);

        int pred_h  = (int)tdt_hdr[0];
        int pred_l  = (int)tdt_hdr[1];
        int n_dur   = (int)tdt_hdr[2];
        int joint_d = (int)tdt_hdr[3];

        /* Read duration values array (n_dur uint32s) */
        int dur_values[16] = {0};
        if (n_dur > 16) n_dur = 16;
        uint32_t dur_raw[16];
        memcpy(dur_raw, reader.cursor, n_dur * sizeof(uint32_t));
        reader.cursor += n_dur * sizeof(uint32_t);
        for (int i = 0; i < n_dur; i++)
            dur_values[i] = (int)dur_raw[i];

        fprintf(stderr, "[conformer_stt] Loading TDT decoder: LSTM(%d layers, h=%d), "
                "joint=%d, %d durations [", pred_l, pred_h, joint_d, n_dur);
        for (int i = 0; i < n_dur; i++)
            fprintf(stderr, "%s%d", i ? "," : "", dur_values[i]);
        fprintf(stderr, "]\n");

        const float *embed_w = read_weight(&reader, vocab * pred_h);

        const float *lstm_wi[MAX_PRED_LAYERS];
        const float *lstm_bi[MAX_PRED_LAYERS];
        const float *lstm_wh[MAX_PRED_LAYERS];
        const float *lstm_bh[MAX_PRED_LAYERS];
        for (int l = 0; l < pred_l; l++) {
            lstm_wi[l] = read_weight(&reader, 4 * pred_h * pred_h);
            lstm_bi[l] = read_weight(&reader, 4 * pred_h);
            lstm_wh[l] = read_weight(&reader, 4 * pred_h * pred_h);
            lstm_bh[l] = read_weight(&reader, 4 * pred_h);
        }

        const float *jenc_w  = read_weight(&reader, joint_d * D);
        const float *jenc_b  = read_weight(&reader, joint_d);
        const float *jpred_w = read_weight(&reader, joint_d * pred_h);
        const float *jpred_b = read_weight(&reader, joint_d);
        const float *jout_w  = read_weight(&reader, (vocab + n_dur) * joint_d);
        const float *jout_b  = read_weight(&reader, vocab + n_dur);

        TDTConfig tdt_cfg = {
            .pred_hidden = pred_h,
            .pred_layers = pred_l,
            .vocab_size  = vocab,
            .n_durations = n_dur,
            .joint_dim   = joint_d,
            .encoder_dim = D,
            .blank_id    = vocab - 1,
            .duration_values = {0},
        };
        for (int i = 0; i < n_dur && i < 16; i++)
            tdt_cfg.duration_values[i] = dur_values[i];

        stt->tdt = tdt_decoder_create(&tdt_cfg, embed_w,
                                       lstm_wi, lstm_bi, lstm_wh, lstm_bh,
                                       jenc_w, jenc_b, jpred_w, jpred_b,
                                       jout_w, jout_b);
        if (!stt->tdt) {
            fprintf(stderr, "[conformer_stt] Failed to create TDT decoder\n");
            return -1;
        }

        stt->tdt_n_durations = n_dur;
        stt->tdt_token_cap = MAX_SEQ_LEN;
        stt->tdt_token_buf = (int *)malloc(stt->tdt_token_cap * sizeof(int));
        if (!stt->tdt_token_buf) return -1;

        fprintf(stderr, "[conformer_stt] TDT decoder ready\n");
    }

    size_t consumed = (size_t)(reader.cursor - (const char *)stt->mmap_base);
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
    int conv_k = (int)h->conv_kernel;
    size_t seq_d  = (size_t)MAX_SEQ_LEN * D;
    size_t seq_ff = (size_t)MAX_SEQ_LEN * ff_dim;
    size_t seq_2d = (size_t)MAX_SEQ_LEN * 2 * D;
    /* Per-head scratch in mhsa_forward: Qh,Kh,Vh [3*T*d_head] + scores [T*T]
       + PEh [(2T-1)*d_head] + Qv [T*d_head] + pos_raw [T*(2T-1)] */
    size_t pe_len_max = 2 * (size_t)MAX_SEQ_LEN - 1;
    size_t attn_sz = (size_t)MAX_SEQ_LEN * D * 3
                   + (size_t)MAX_SEQ_LEN * MAX_SEQ_LEN
                   + pe_len_max * d_head
                   + (size_t)MAX_SEQ_LEN * d_head
                   + (size_t)MAX_SEQ_LEN * pe_len_max;

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

    /* Relative PE scratch: max pe_len = 2*MAX_SEQ_LEN + CACHE_MAX_CONTEXT - 1
       (asymmetric case: Q[T] x K[T + CACHE_MAX_CONTEXT]) */
    if (h->flags & CSTT_FLAG_REL_PE) {
        size_t pe_max = 2 * (size_t)MAX_SEQ_LEN + CACHE_MAX_CONTEXT;
        size_t pe_elems = pe_max * D;
        ws->rel_pe = (float *)calloc(pe_elems, sizeof(float));
        ws->rel_pe_proj = (float *)calloc(pe_elems, sizeof(float));
    } else {
        ws->rel_pe = NULL;
        ws->rel_pe_proj = NULL;
    }

    /* --- Pre-allocated hot-path scratch (zero malloc during inference) --- */
    ws->silu_tmp  = (float *)calloc(seq_ff, sizeof(float));
    ws->bn_scale  = (float *)calloc(D, sizeof(float));
    ws->bn_shift  = (float *)calloc(D, sizeof(float));
    ws->lsp_tmp   = (float *)calloc(vocab, sizeof(float));
    ws->mel_norm_buf = (float *)calloc((size_t)CHUNK_FRAMES * n_mels, sizeof(float));

    /* Cached MHSA scratch: total_T_max = CACHE_MAX_CONTEXT + MAX_SEQ_LEN */
    size_t total_T_max = (size_t)CACHE_MAX_CONTEXT + MAX_SEQ_LEN;
    size_t cached_pe_max = 2 * (size_t)MAX_SEQ_LEN + CACHE_MAX_CONTEXT;
    ws->cache_k_full  = (float *)calloc(total_T_max * D, sizeof(float));
    ws->cache_v_full  = (float *)calloc(total_T_max * D, sizeof(float));
    /* scores [T * total_T] + PEh [pe_len * d_head] + Qv [T * d_head]
       + pos_raw [T * pe_len] */
    size_t cache_attn_sz = (size_t)MAX_SEQ_LEN * total_T_max
                         + cached_pe_max * d_head
                         + (size_t)MAX_SEQ_LEN * d_head
                         + (size_t)MAX_SEQ_LEN * cached_pe_max;
    ws->cache_scores  = (float *)calloc(cache_attn_sz, sizeof(float));
    ws->cache_qh      = (float *)calloc((size_t)MAX_SEQ_LEN * d_head, sizeof(float));
    ws->cache_kh      = (float *)calloc(total_T_max * d_head, sizeof(float));
    ws->cache_vh      = (float *)calloc(total_T_max * d_head, sizeof(float));
    ws->cache_ctx     = (float *)calloc((size_t)MAX_SEQ_LEN * d_head, sizeof(float));

    /* Cached conv scratch: overlap = conv_kernel-1, total = overlap + MAX_SEQ_LEN */
    size_t conv_total = (size_t)(conv_k - 1) + MAX_SEQ_LEN;
    ws->cache_conv_merged = (float *)calloc(conv_total * D, sizeof(float));
    ws->cache_conv_out    = (float *)calloc(conv_total * D, sizeof(float));

    ws->feat_col = NULL;

    /* fp16 mixed-precision scratch (allocate if model is fp16).
     * fp16_in needs M*K (largest K = ff_dim for FFN, D for CTC).
     * fp16_out needs M*N (largest N = vocab for CTC head, ff_dim for FFN).
     * Must account for vocab > ff_dim. */
    if (h->dtype == 1) {
        size_t max_in_dim = (size_t)(ff_dim > D ? ff_dim : D);
        size_t max_out_dim = (size_t)(vocab > ff_dim ? vocab : ff_dim);
        if ((size_t)D > max_out_dim) max_out_dim = (size_t)D;
        ws->fp16_in  = (__fp16 *)calloc((size_t)MAX_SEQ_LEN * max_in_dim, sizeof(__fp16));
        ws->fp16_out = (__fp16 *)calloc((size_t)MAX_SEQ_LEN * max_out_dim, sizeof(__fp16));
    } else {
        ws->fp16_in = NULL;
        ws->fp16_out = NULL;
    }

    if (!ws->buf_a || !ws->buf_b || !ws->residual || !ws->qkv ||
        !ws->attn_scores || !ws->ff_mid || !ws->conv_mid ||
        !ws->logits || !ws->sub_work || !ws->silu_tmp ||
        !ws->cache_k_full || !ws->cache_v_full || !ws->cache_scores ||
        !ws->cache_qh || !ws->cache_kh || !ws->cache_vh || !ws->cache_ctx ||
        !ws->cache_conv_merged || !ws->cache_conv_out ||
        !ws->bn_scale || !ws->bn_shift || !ws->lsp_tmp || !ws->mel_norm_buf ||
        ((h->flags & CSTT_FLAG_REL_PE) && (!ws->rel_pe || !ws->rel_pe_proj)) ||
        (h->dtype == 1 && (!ws->fp16_in || !ws->fp16_out)))
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
    free(ws->rel_pe_proj);
    free(ws->silu_tmp);
    free(ws->bn_scale);
    free(ws->bn_shift);
    free(ws->lsp_tmp);
    free(ws->mel_norm_buf);
    free(ws->cache_k_full);
    free(ws->cache_v_full);
    free(ws->cache_scores);
    free(ws->cache_qh);
    free(ws->cache_kh);
    free(ws->cache_vh);
    free(ws->cache_ctx);
    free(ws->cache_conv_merged);
    free(ws->cache_conv_out);
    free(ws->feat_col);
    free(ws->fp16_in);
    free(ws->fp16_out);
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

    /* Aggressive VM hints: prefetch + wire pages into physical memory */
    posix_madvise(stt->mmap_base, stt->mmap_size, POSIX_MADV_WILLNEED);
    posix_madvise(stt->mmap_base, stt->mmap_size, POSIX_MADV_RANDOM);
    if (mlock(stt->mmap_base, stt->mmap_size) == 0) {
        fprintf(stderr, "[conformer_stt] Wired %zu MB into physical memory\n",
                stt->mmap_size / (1024 * 1024));
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
    mel_cfg.preemph     = 0.97f;  /* NeMo default pre-emphasis */

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
    stt->chunk_frames = 0;  /* 0 = use CHUNK_FRAMES default */

    int n_mels_init = (int)stt->header.n_mels;
    stt->running_sum    = (double *)calloc(n_mels_init, sizeof(double));
    stt->running_sum_sq = (double *)calloc(n_mels_init, sizeof(double));
    stt->running_count  = 0;

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
    tdt_decoder_destroy(stt->tdt);
    free(stt->tdt_token_buf);
    free(stt->tdt_enc_accum);
    ctc_beam_destroy(stt->beam_decoder);
    free(stt->beam_logits);
    free(stt->fp16_arena);
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
    free(stt->running_sum);
    free(stt->running_sum_sq);
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

    int chunk_limit = stt->chunk_frames > 0 ? stt->chunk_frames : CHUNK_FRAMES;
    while (stt->mel_accum_len >= min_frames) {
        int process_frames = stt->mel_accum_len;
        if (process_frames > chunk_limit)
            process_frames = chunk_limit;
        process_frames = (process_frames / sub_factor) * sub_factor;
        if (process_frames < min_frames) break;

        int frames_before = stt->total_frames_processed;
        int T_out = full_forward(stt, stt->mel_accum, process_frames);
        if (T_out <= 0) break;

        int V = (int)stt->header.vocab_size;
        int D = (int)stt->header.d_model;

        if (stt->tdt) {
            /* Accumulate encoder output for full-utterance TDT decode in flush() */
            int need = stt->tdt_enc_len + T_out;
            if (need > stt->tdt_enc_cap) {
                stt->tdt_enc_cap = need + 256;
                float *new_buf = (float *)realloc(stt->tdt_enc_accum,
                    (size_t)stt->tdt_enc_cap * D * sizeof(float));
                if (!new_buf) return -1;
                stt->tdt_enc_accum = new_buf;
            }
            memcpy(stt->tdt_enc_accum + (size_t)stt->tdt_enc_len * D,
                   stt->work.buf_a, (size_t)T_out * D * sizeof(float));
            stt->tdt_enc_len += T_out;
        } else if (stt->beam_decoder) {
            /* Accumulate logits for full-utterance beam decode in flush() */
            int need = stt->beam_logits_len + T_out;
            if (need > stt->beam_logits_cap) {
                stt->beam_logits_cap = need + 256;
                float *new_buf = (float *)realloc(stt->beam_logits,
                    (size_t)stt->beam_logits_cap * V * sizeof(float));
                if (!new_buf) return -1;
                stt->beam_logits = new_buf;
            }
            memcpy(stt->beam_logits + (size_t)stt->beam_logits_len * V,
                   stt->work.logits, (size_t)T_out * V * sizeof(float));
            stt->beam_logits_len += T_out;
        } else {
            /* Ensure eou_probs buffer is large enough */
            if (T_out > stt->eou_probs_cap) {
                stt->eou_probs_cap = T_out + 256;
                float *new_buf = (float *)realloc(stt->eou_probs,
                                                   stt->eou_probs_cap * sizeof(float));
                if (!new_buf) return -1;
                stt->eou_probs = new_buf;
            }

            char decode_buf[4096];
            int eou_det = 0, eou_fr = -1;
            int n_chars = ctc_greedy_decode(decode_buf, sizeof(decode_buf),
                                            stt->work.logits, T_out, V,
                                            &stt->vocab, &stt->prev_token,
                                            stt->eou_token_id,
                                            &eou_det, &eou_fr,
                                            stt->eou_probs);
            stt->eou_probs_len = T_out;
            if (eou_det) {
                stt->eou_detected = 1;
                stt->eou_frame = frames_before + eou_fr;
            }

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

/**
 * Helper: run beam decode on all accumulated logits, append result to transcript.
 * Returns number of new chars, or 0.
 */
static int beam_decode_accumulated(ConformerSTT *stt) {
    if (!stt->beam_decoder || stt->beam_logits_len == 0) return 0;
    int V = (int)stt->header.vocab_size;
    logits_to_log_probs_ws(stt->beam_logits, stt->beam_logits_len, V, stt->work.lsp_tmp);
    char decode_buf[4096];
    int n_chars = ctc_beam_decode(stt->beam_decoder,
                                   stt->beam_logits, stt->beam_logits_len, V,
                                   decode_buf, sizeof(decode_buf));
    stt->beam_logits_len = 0;
    if (n_chars <= 0) return 0;
    int space = MAX_TRANSCRIPT - stt->transcript_len - 1;
    int copy = n_chars < space ? n_chars : space;
    if (copy > 0) {
        memcpy(stt->transcript + stt->transcript_len, decode_buf, copy);
        stt->transcript_len += copy;
        stt->transcript[stt->transcript_len] = '\0';
    }
    return copy;
}

/**
 * Helper: run TDT decode on accumulated encoder output, append result to transcript.
 * Returns number of new chars, or 0.
 */
static int tdt_decode_accumulated(ConformerSTT *stt) {
    if (!stt->tdt || stt->tdt_enc_len == 0) return 0;

    int n_tokens = tdt_decoder_decode(stt->tdt,
                                       stt->tdt_enc_accum, stt->tdt_enc_len,
                                       stt->tdt_token_buf, stt->tdt_token_cap);
    stt->tdt_enc_len = 0;
    if (n_tokens <= 0) return 0;

    /* Convert token IDs to text using vocabulary */
    int blank_id = stt->vocab.size - 1;
    char decode_buf[8192];
    int pos = 0;
    for (int i = 0; i < n_tokens && pos < (int)sizeof(decode_buf) - 64; i++) {
        int tok = stt->tdt_token_buf[i];
        if (tok <= 0 || tok >= stt->vocab.size || tok == blank_id) continue;
        const char *piece = stt->vocab.tokens[tok];
        if (!piece) continue;

        /* SentencePiece uses ▁ (U+2581) as word separator */
        const char *p = piece;
        if ((unsigned char)p[0] == 0xE2 && (unsigned char)p[1] == 0x96 &&
            (unsigned char)p[2] == 0x81) {
            if (pos > 0) decode_buf[pos++] = ' ';
            p += 3;
        }
        int len = (int)strlen(p);
        if (pos + len < (int)sizeof(decode_buf) - 1) {
            memcpy(decode_buf + pos, p, len);
            pos += len;
        }
    }
    decode_buf[pos] = '\0';

    if (pos > 0) {
        int space = MAX_TRANSCRIPT - stt->transcript_len - 1;
        int copy = pos < space ? pos : space;
        if (copy > 0) {
            memcpy(stt->transcript + stt->transcript_len, decode_buf, copy);
            stt->transcript_len += copy;
            stt->transcript[stt->transcript_len] = '\0';
        }
        return copy;
    }
    return 0;
}

int conformer_stt_flush(ConformerSTT *stt) {
    if (!stt) return -1;

    /* Early return for deferred decoders when no mel remains */
    if (stt->mel_accum_len == 0) {
        if (stt->tdt)          return tdt_decode_accumulated(stt);
        if (stt->beam_decoder) return beam_decode_accumulated(stt);
        return 0;
    }

    int n_mels = (int)stt->header.n_mels;
    int sub_factor = (int)stt->header.subsample_factor;
    if (sub_factor < 1) sub_factor = 4;
    int min_frames = sub_factor * 4;

    /* Skip flush if remaining frames are fewer than one subsampled step */
    if (stt->mel_accum_len < sub_factor) {
        stt->mel_accum_len = 0;
        if (stt->tdt)          return tdt_decode_accumulated(stt);
        if (stt->beam_decoder) return beam_decode_accumulated(stt);
        return 0;
    }

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

    int frames_before = stt->total_frames_processed;
    int T_out = full_forward(stt, stt->mel_accum, stt->mel_accum_len);
    stt->mel_accum_len = 0;
    if (T_out <= 0) {
        if (stt->tdt)          return tdt_decode_accumulated(stt);
        if (stt->beam_decoder) return beam_decode_accumulated(stt);
        return 0;
    }

    int V = (int)stt->header.vocab_size;
    int D = (int)stt->header.d_model;
    char decode_buf[4096];
    int n_chars;

    if (stt->tdt) {
        /* Accumulate final chunk's encoder output then TDT decode everything */
        int need = stt->tdt_enc_len + T_out;
        if (need > stt->tdt_enc_cap) {
            stt->tdt_enc_cap = need + 256;
            float *new_buf = (float *)realloc(stt->tdt_enc_accum,
                (size_t)stt->tdt_enc_cap * D * sizeof(float));
            if (!new_buf) return -1;
            stt->tdt_enc_accum = new_buf;
        }
        memcpy(stt->tdt_enc_accum + (size_t)stt->tdt_enc_len * D,
               stt->work.buf_a, (size_t)T_out * D * sizeof(float));
        stt->tdt_enc_len += T_out;
        return tdt_decode_accumulated(stt);
    }

    if (stt->beam_decoder) {
        /* Accumulate final chunk's logits then beam decode everything */
        int need = stt->beam_logits_len + T_out;
        if (need > stt->beam_logits_cap) {
            stt->beam_logits_cap = need + 256;
            float *new_buf = (float *)realloc(stt->beam_logits,
                (size_t)stt->beam_logits_cap * V * sizeof(float));
            if (!new_buf) return -1;
            stt->beam_logits = new_buf;
        }
        memcpy(stt->beam_logits + (size_t)stt->beam_logits_len * V,
               stt->work.logits, (size_t)T_out * V * sizeof(float));
        stt->beam_logits_len += T_out;
        return beam_decode_accumulated(stt);
    }

    if (T_out > stt->eou_probs_cap) {
        stt->eou_probs_cap = T_out + 256;
        float *new_buf = (float *)realloc(stt->eou_probs,
                                           stt->eou_probs_cap * sizeof(float));
        if (!new_buf) return -1;
        stt->eou_probs = new_buf;
    }

    int eou_det = 0, eou_fr = -1;
    n_chars = ctc_greedy_decode(decode_buf, sizeof(decode_buf),
                                stt->work.logits, T_out, V,
                                &stt->vocab, &stt->prev_token,
                                stt->eou_token_id,
                                &eou_det, &eou_fr,
                                stt->eou_probs);
    stt->eou_probs_len = T_out;
    if (eou_det) {
        stt->eou_detected = 1;
        stt->eou_frame = frames_before + eou_fr;
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

int conformer_stt_debug_logits(const ConformerSTT *stt, float *out, int max_frames, int *out_frames) {
    if (!stt || !out) return -1;
    int vocab = (int)stt->header.vocab_size;
    int T = stt->total_frames_processed;
    if (T > max_frames) T = max_frames;
    if (out_frames) *out_frames = T;
    memcpy(out, stt->work.logits, (size_t)T * vocab * sizeof(float));
    return 0;
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
    stt->beam_logits_len = 0;
    stt->tdt_enc_len = 0;
    if (stt->tdt) tdt_decoder_reset(stt->tdt);
    stt->total_frames_processed = 0;
    mel_reset(stt->mel);

    /* Reset running normalization stats */
    if (stt->running_sum) {
        int n_mels = (int)stt->header.n_mels;
        memset(stt->running_sum,    0, n_mels * sizeof(double));
        memset(stt->running_sum_sq, 0, n_mels * sizeof(double));
        stt->running_count = 0;
    }

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

void conformer_stt_set_chunk_frames(ConformerSTT *stt, int frames) {
    if (!stt) return;
    stt->chunk_frames = frames;
    fprintf(stderr, "[conformer_stt] Chunk frames set to %d (%d ms)\n",
            frames, frames * 10);
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

/* ─── Beam Search API ──────────────────────────────────────────────────── */

int conformer_stt_enable_beam_search(ConformerSTT *stt,
                                      const char *lm_path,
                                      int beam_size,
                                      float lm_weight,
                                      float word_score) {
    if (!stt) return -1;

    /* Tear down existing beam decoder if any */
    if (stt->beam_decoder) {
        ctc_beam_destroy(stt->beam_decoder);
        stt->beam_decoder = NULL;
    }

    CTCBeamConfig cfg = ctc_beam_config_default();
    if (beam_size > 0) cfg.beam_size = beam_size;
    if (lm_weight >= 0.0f) cfg.lm_weight = lm_weight;
    cfg.word_score = word_score;

    stt->beam_decoder = ctc_beam_create(
        lm_path,
        (const char *const *)stt->vocab.tokens,
        stt->vocab.size,
        stt->vocab.blank_id,
        &cfg);

    if (!stt->beam_decoder) {
        fprintf(stderr, "[conformer_stt] Failed to create beam decoder\n");
        return -1;
    }

    fprintf(stderr, "[conformer_stt] Beam search enabled: beam=%d, lm_weight=%.2f, "
            "word_score=%.2f, lm=%s\n",
            cfg.beam_size, cfg.lm_weight, cfg.word_score,
            lm_path ? lm_path : "(none)");
    return 0;
}

void conformer_stt_disable_beam_search(ConformerSTT *stt) {
    if (!stt) return;
    ctc_beam_destroy(stt->beam_decoder);
    stt->beam_decoder = NULL;
    fprintf(stderr, "[conformer_stt] Beam search disabled, using greedy decode\n");
}

int conformer_stt_is_tdt(const ConformerSTT *stt) {
    return (stt && stt->tdt) ? 1 : 0;
}

void conformer_stt_set_external_forward(ConformerSTT *stt,
    conformer_external_forward_fn fn, void *user_ctx) {
    if (!stt) return;
    stt->external_forward = fn;
    stt->external_forward_ctx = user_ctx;
}

float *conformer_stt_get_logits_buf(ConformerSTT *stt, int *out_vocab) {
    if (!stt) return NULL;
    if (out_vocab) *out_vocab = (int)stt->header.vocab_size;
    return stt->work.logits;
}
