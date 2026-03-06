/**
 * sonata_stt.c — Sonata STT CTC inference engine for Apple Silicon.
 *
 * Conformer encoder (4-12 layers, d=256-512, RoPE) + CTC projection.
 * All matrix ops via cblas_sgemm/sgemv (AMX coprocessor).
 * Mel spectrogram via the shared mel_spectrogram library.
 *
 * Features:
 *   - Streaming: growing-window audio buffer with feed/flush
 *   - Beam search: pluggable CTCBeamDecoder for LM-boosted decode
 *   - EOU token: id=29 for inline end-of-utterance detection
 *   - Supports base (4L d=256) and large (12L d=512) encoders
 *
 * Build:
 *   cc -O3 -shared -fPIC -arch arm64 -framework Accelerate \
 *      -L$(BUILD) -lmel_spectrogram \
 *      -install_name @rpath/libsonata_stt.dylib \
 *      -o libsonata_stt.dylib sonata_stt.c
 */

#include "sonata_stt.h"
#include "mel_spectrogram.h"

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

#define SONATA_STT_MAGIC      0x53545453  /* "STTS" */
#define SONATA_STT_VERSION    1
#define MAX_FRAMES            2048
#define HOP_LENGTH            480
#define SAMPLE_RATE          24000
#define FRAME_DURATION_SEC    (1.0f / 50.0f)  /* 50 Hz = 20ms */
#define MAX_TEXT_VOCAB      64
#define MAX_ENC_DIM         512
#define MAX_FF_DIM          2048
#define MAX_CONV_KERNEL     63
#define MAX_TRANSCRIPT_LEN  4096

/* CTC character vocabulary: blank(0) space(1) a-z(2-27) '(28) <eou>(29) */
static const char CTC_CHARS[] = "\0 abcdefghijklmnopqrstuvwxyz'";
#define CTC_BLANK   0
#define CTC_SPACE   1
#define CTC_EOU_ID  29

/* ═══════════════════════════════════════════════════════════════════════════
 * Weight file header
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t enc_dim;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_mels;
    uint32_t conv_kernel;
    uint32_t text_vocab;
    uint32_t n_weights;
    uint32_t pad;
} SonataSTTHeader;

/* ═══════════════════════════════════════════════════════════════════════════
 * Conformer block weight pointers
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* FF1: LayerNorm → Linear(D,4D) → SiLU → Linear(4D,D) */
    const float *ff1_ln_w, *ff1_ln_b;
    const float *ff1_up_w, *ff1_up_b;
    const float *ff1_down_w, *ff1_down_b;

    /* MHSA: LayerNorm → Q/K/V/Out */
    const float *attn_ln_w, *attn_ln_b;
    const float *attn_qkv_w, *attn_qkv_b;
    const float *attn_out_w, *attn_out_b;

    /* Conv: LayerNorm → pw1(D,2D) → GLU → dw(D,k) → BN → SiLU → pw2(D,D) */
    const float *conv_ln_w, *conv_ln_b;
    const float *conv_pw1_w, *conv_pw1_b;
    const float *conv_dw_w, *conv_dw_b;
    const float *conv_bn_g, *conv_bn_b;
    const float *conv_bn_mean, *conv_bn_var;
    const float *conv_pw2_w, *conv_pw2_b;

    /* FF2 */
    const float *ff2_ln_w, *ff2_ln_b;
    const float *ff2_up_w, *ff2_up_b;
    const float *ff2_down_w, *ff2_down_b;

    /* Final LayerNorm */
    const float *final_ln_w, *final_ln_b;
} BlockWeights;

/* ═══════════════════════════════════════════════════════════════════════════
 * Engine state
 * ═══════════════════════════════════════════════════════════════════════════ */

struct SonataSTT {
    /* Config */
    int enc_dim;
    int n_layers;
    int n_heads;
    int head_dim;
    int n_mels;
    int conv_kernel;
    int text_vocab;
    int ff_dim;

    /* Mel spectrogram extractor */
    MelSpectrogram *mel;

    /* Weight pointers (into mmap'd region) */
    const float *input_proj_w, *input_proj_b;
    BlockWeights *blocks;
    const float *adapter_ln_w, *adapter_ln_b;
    const float *adapter_w, *adapter_b;
    const float *ctc_w, *ctc_b;

    /* mmap state */
    void *mmap_base;
    size_t mmap_size;

    /* Working buffers */
    float *buf_a;
    float *buf_b;
    float *buf_ff;
    float *buf_qkv;
    float *buf_attn;
    float *buf_conv;
    float *buf_pw1;    /* separate pw1 output buffer (avoids SGEMM aliasing in conv module) */
    float *mel_buf;
    float *logits_buf;

    /* Precomputed RoPE (cos/sin) and SiLU/GLU workspace */
    float *rope_cos;
    float *rope_sin;
    float *silu_tmp;

    /* Streaming state */
    float *stream_audio;
    int    stream_pos;
    int    stream_cap;
    int    streaming;

    /* Last decode state (for EOU extraction) */
    int    last_n_frames;

    /* CTC alignment for word timestamps (greedy decode only) */
    int    *align_token_id;
    int    *align_frame;
    float  *align_prob;
    int    align_len;

    /* Optional beam decoder (not owned) */
    CTCBeamDecoder *beam_dec;

    /* FP16 acceleration */
    _Float16 *fp16_weights;
    int fp16_n_weights;
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Math helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static void layer_norm(float *out, const float *x, const float *w, const float *b,
                       int T, int D) {
    for (int t = 0; t < T; t++) {
        const float *xt = x + t * D;
        float *ot = out + t * D;
        float mean, meansq;
        vDSP_meanv(xt, 1, &mean, D);
        vDSP_measqv(xt, 1, &meansq, D);
        float var = meansq - mean * mean;
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        float neg_mean = -mean;
        vDSP_vsadd(xt, 1, &neg_mean, ot, 1, D);
        vDSP_vsmul(ot, 1, &inv_std, ot, 1, D);
        vDSP_vmul(ot, 1, w, 1, ot, 1, D);
        vDSP_vadd(ot, 1, b, 1, ot, 1, D);
    }
}

static void linear_forward(float *out, const float *x, const float *W,
                            const float *bias, int T, int in_dim, int out_dim) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, out_dim, in_dim,
                1.0f, x, in_dim, W, in_dim,
                0.0f, out, out_dim);
    if (bias) {
        for (int t = 0; t < T; t++)
            vDSP_vadd(out + t * out_dim, 1, bias, 1, out + t * out_dim, 1, out_dim);
    }
}

static void silu_inplace(float *x, int n, float *workspace) {
    /* Fused SiLU via vvexpf + NEON — 3 passes instead of per-element expf */
    float *t = workspace;
    vDSP_vneg(x, 1, t, 1, n);
    int ni = n;
    vvexpf(t, t, &ni);
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    int i = 0;
    float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t xi = vld1q_f32(x + i);
        float32x4_t ei = vld1q_f32(t + i);
        vst1q_f32(x + i, vdivq_f32(xi, vaddq_f32(one, ei)));
    }
    for (; i < n; i++) x[i] = x[i] / (1.0f + t[i]);
#else
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + t[i]);
#endif
}

static void softmax_row(float *x, int len) {
    float mx;
    vDSP_maxv(x, 1, &mx, len);
    float neg_mx = -mx;
    vDSP_vsadd(x, 1, &neg_mx, x, 1, len);
    int ni = len;
    vvexpf(x, x, &ni);
    float sum;
    vDSP_sve(x, 1, &sum, len);
    float inv = 1.0f / sum;
    vDSP_vsmul(x, 1, &inv, x, 1, len);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Conformer blocks
 * ═══════════════════════════════════════════════════════════════════════════ */

static void feed_forward(float *out, const float *x, const BlockWeights *bw,
                          const float *ln_w, const float *ln_b,
                          const float *up_w, const float *up_b,
                          const float *down_w, const float *down_b,
                          float *buf_norm, float *buf_ff,
                          float *silu_ws,
                          int T, int D, int ff_dim) {
    layer_norm(buf_norm, x, ln_w, ln_b, T, D);
    linear_forward(buf_ff, buf_norm, up_w, up_b, T, D, ff_dim);
    silu_inplace(buf_ff, T * ff_dim, silu_ws);
    linear_forward(out, buf_ff, down_w, down_b, T, ff_dim, D);
}

static void mhsa_forward(float *out, const float *x, const BlockWeights *bw,
                          float *buf_norm, float *buf_qkv, float *buf_attn,
                          const float *rope_cos, const float *rope_sin,
                          int T, int D, int n_heads) {
    int head_dim = D / n_heads;
    int half_dim = head_dim / 2;
    layer_norm(buf_norm, x, bw->attn_ln_w, bw->attn_ln_b, T, D);

    linear_forward(buf_qkv, buf_norm, bw->attn_qkv_w, bw->attn_qkv_b, T, D, 3 * D);

    float *Q = buf_qkv;
    float *K = buf_qkv + T * D;
    float *V = buf_qkv + T * 2 * D;

    /* Apply RoPE (Rotary Position Encoding) to Q and K — precomputed tables */
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *qt = Q + t * D + h * head_dim;
            float *kt = K + t * D + h * head_dim;
            for (int i = 0; i < half_dim; i++) {
                float cos_t = rope_cos[t * half_dim + i];
                float sin_t = rope_sin[t * half_dim + i];
                float q0 = qt[2*i], q1 = qt[2*i+1];
                qt[2*i]   = q0 * cos_t - q1 * sin_t;
                qt[2*i+1] = q0 * sin_t + q1 * cos_t;
                float k0 = kt[2*i], k1 = kt[2*i+1];
                kt[2*i]   = k0 * cos_t - k1 * sin_t;
                kt[2*i+1] = k0 * sin_t + k1 * cos_t;
            }
        }
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    memset(out, 0, T * D * sizeof(float));

    /* GEMM-based attention: Q[T,hd] @ K[T,hd]^T → scores[T,T], then scores @ V */
    for (int h = 0; h < n_heads; h++) {
        /* Gather head-strided Q,K,V into contiguous blocks for GEMM.
           Q/K/V are [T, D] with head data at stride D, offset h*head_dim. */
        float *Qh = buf_attn;                       /* reuse buf_attn [T*head_dim] */
        float *Kh = buf_attn + T * head_dim;        /* [T*head_dim] */
        float *Vh = buf_attn + 2 * T * head_dim;    /* [T*head_dim] */
        float *scores = buf_attn + 3 * T * head_dim; /* [T*T] */

        for (int t = 0; t < T; t++) {
            memcpy(Qh + t * head_dim, Q + t * D + h * head_dim, head_dim * sizeof(float));
            memcpy(Kh + t * head_dim, K + t * D + h * head_dim, head_dim * sizeof(float));
            memcpy(Vh + t * head_dim, V + t * D + h * head_dim, head_dim * sizeof(float));
        }

        /* scores = scale * Qh @ Kh^T   [T x head_dim] @ [head_dim x T] = [T x T] */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, T, head_dim,
                    scale, Qh, head_dim, Kh, head_dim,
                    0.0f, scores, T);

        for (int qi = 0; qi < T; qi++)
            softmax_row(scores + qi * T, T);

        /* context = scores @ Vh   [T x T] @ [T x head_dim] = [T x head_dim] */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, head_dim, T,
                    1.0f, scores, T, Vh, head_dim,
                    0.0f, Qh, head_dim);  /* reuse Qh for output */

        /* Scatter back to interleaved output */
        for (int t = 0; t < T; t++)
            memcpy(out + t * D + h * head_dim, Qh + t * head_dim, head_dim * sizeof(float));
    }

    float *tmp = buf_norm;
    memcpy(tmp, out, T * D * sizeof(float));
    linear_forward(out, tmp, bw->attn_out_w, bw->attn_out_b, T, D, D);
}

static void conv_module(float *out, const float *x, const BlockWeights *bw,
                         float *buf_norm, float *buf_conv, float *buf_pw1,
                         float *silu_ws,
                         int T, int D, int kernel) {
    if (T > 2048) return;  /* stack buffers below are sized for T <= 2048 */
    layer_norm(buf_norm, x, bw->conv_ln_w, bw->conv_ln_b, T, D);

    /* Use separate buf_pw1 for pw1 output to avoid SGEMM aliasing with buf_norm */
    float *pw1_out = buf_pw1;
    linear_forward(pw1_out, buf_norm, bw->conv_pw1_w, bw->conv_pw1_b, T, D, 2 * D);

    /* Vectorized GLU: dest = a * sigmoid(b) */
    {
        int total = T * D;
        float *sig_buf = silu_ws;
        /* Gather all b_half values contiguously */
        for (int t = 0; t < T; t++)
            memcpy(sig_buf + t * D, pw1_out + t * 2 * D + D, (size_t)D * sizeof(float));
        /* sigmoid(b) = 1/(1+exp(-b)) via vDSP */
        vDSP_vneg(sig_buf, 1, sig_buf, 1, total);
        int ni = total;
        vvexpf(sig_buf, sig_buf, &ni);
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
        int k = 0;
        float32x4_t one = vdupq_n_f32(1.0f);
        for (; k + 4 <= total; k += 4) {
            float32x4_t e = vld1q_f32(sig_buf + k);
            vst1q_f32(sig_buf + k, vdivq_f32(one, vaddq_f32(one, e)));
        }
        for (; k < total; k++) sig_buf[k] = 1.0f / (1.0f + sig_buf[k]);
#else
        for (int k = 0; k < total; k++) sig_buf[k] = 1.0f / (1.0f + sig_buf[k]);
#endif
        /* dest = a * sigmoid(b) */
        for (int t = 0; t < T; t++) {
            float *a = pw1_out + t * 2 * D;
            float *dest = buf_norm + t * D;
            vDSP_vmul(a, 1, sig_buf + t * D, 1, dest, 1, D);
        }
    }

    int pad = kernel / 2;
    float *dw_out = buf_conv;
    memset(dw_out, 0, T * D * sizeof(float));

    for (int ch = 0; ch < D; ch++) {
        const float *dw_k = bw->conv_dw_w + ch * kernel;
        float dw_bias = bw->conv_dw_b[ch];
        /* Extract channel column, convolve with vDSP, scatter back */
        float col_in[2048 + 64];  /* T + pad*2 */
        float col_out[2048];
        int padded = T + 2 * pad;
        memset(col_in, 0, padded * sizeof(float));
        for (int t = 0; t < T; t++) col_in[t + pad] = buf_norm[t * D + ch];

        /* Weights stored in correlation order; vDSP_conv performs correlation */
        vDSP_conv(col_in, 1, dw_k, 1, col_out, 1, T, kernel);

        for (int t = 0; t < T; t++)
            dw_out[t * D + ch] = col_out[t] + dw_bias;
    }

    /* Fused batch norm: (x - mean) / sqrt(var + eps) * gamma + beta */
    for (int i = 0; i < D; i++) {
        float inv_std = 1.0f / sqrtf(bw->conv_bn_var[i] + 1e-5f);
        float scale = bw->conv_bn_g[i] * inv_std;
        float bias = bw->conv_bn_b[i] - bw->conv_bn_mean[i] * scale;
        for (int t = 0; t < T; t++)
            dw_out[t * D + i] = dw_out[t * D + i] * scale + bias;
    }

    silu_inplace(dw_out, T * D, silu_ws);
    linear_forward(out, dw_out, bw->conv_pw2_w, bw->conv_pw2_b, T, D, D);
}

static void conformer_block(float *x, const BlockWeights *bw,
                              float *buf_b, float *buf_ff, float *buf_qkv,
                              float *buf_attn, float *buf_conv, float *buf_pw1,
                              const float *rope_cos, const float *rope_sin,
                              float *silu_ws,
                              int T, int D, int ff_dim, int n_heads, int conv_kernel) {
    feed_forward(buf_b, x, bw,
                 bw->ff1_ln_w, bw->ff1_ln_b,
                 bw->ff1_up_w, bw->ff1_up_b,
                 bw->ff1_down_w, bw->ff1_down_b,
                 buf_conv, buf_ff, silu_ws, T, D, ff_dim);
    for (int i = 0; i < T * D; i++) x[i] += 0.5f * buf_b[i];

    mhsa_forward(buf_b, x, bw, buf_conv, buf_qkv, buf_attn, rope_cos, rope_sin, T, D, n_heads);
    vDSP_vadd(x, 1, buf_b, 1, x, 1, T * D);

    conv_module(buf_b, x, bw, buf_conv + T * D, buf_conv, buf_pw1, silu_ws, T, D, conv_kernel);
    vDSP_vadd(x, 1, buf_b, 1, x, 1, T * D);

    feed_forward(buf_b, x, bw,
                 bw->ff2_ln_w, bw->ff2_ln_b,
                 bw->ff2_up_w, bw->ff2_up_b,
                 bw->ff2_down_w, bw->ff2_down_b,
                 buf_conv, buf_ff, silu_ws, T, D, ff_dim);
    for (int i = 0; i < T * D; i++) x[i] += 0.5f * buf_b[i];

    layer_norm(buf_b, x, bw->final_ln_w, bw->final_ln_b, T, D);
    memcpy(x, buf_b, T * D * sizeof(float));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CTC greedy decode (with EOU support and optional alignment)
 * ═══════════════════════════════════════════════════════════════════════════ */

static int ctc_greedy_decode_impl(const float *logits, int T, int vocab_size,
                                   char *out, int max_len,
                                   int *align_token_id, int *align_frame,
                                   float *align_prob, int *align_len_out) {
    int prev = -1;
    int pos = 0;
    int align_pos = 0;
    int store_align = (align_token_id && align_frame && align_prob && align_len_out);

    for (int t = 0; t < T && pos < max_len - 1; t++) {
        const float *row = logits + t * vocab_size;
        int best = 0;
        float best_val = row[0];
        for (int v = 1; v < vocab_size; v++) {
            if (row[v] > best_val) { best_val = row[v]; best = v; }
        }
        if (best == CTC_BLANK) { prev = best; continue; }
        if (best == prev) continue;
        if (best == CTC_EOU_ID) break;
        prev = best;
        if (best > 0 && best < (int)sizeof(CTC_CHARS)) {
            out[pos++] = CTC_CHARS[best];
            if (store_align && align_pos < MAX_FRAMES) {
                /* Softmax prob for best token */
                float mx = row[0];
                for (int v = 1; v < vocab_size; v++) if (row[v] > mx) mx = row[v];
                float sum = 0;
                for (int v = 0; v < vocab_size; v++) sum += expf(row[v] - mx);
                float prob = expf(row[best] - mx) / sum;

                align_token_id[align_pos] = best;
                align_frame[align_pos] = t;
                align_prob[align_pos] = prob;
                align_pos++;
            }
        }
    }
    out[pos] = '\0';
    if (store_align) *align_len_out = align_pos;
    return pos;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * Weight loading
 * ═══════════════════════════════════════════════════════════════════════════ */

static const float *advance(const float **ptr, int count) {
    const float *p = *ptr;
    *ptr += count;
    return p;
}

static int load_block_weights(BlockWeights *bw, const float **ptr, int D, int ff_dim, int kernel) {
    bw->ff1_ln_w = advance(ptr, D);
    bw->ff1_ln_b = advance(ptr, D);
    bw->ff1_up_w = advance(ptr, ff_dim * D);
    bw->ff1_up_b = advance(ptr, ff_dim);
    bw->ff1_down_w = advance(ptr, D * ff_dim);
    bw->ff1_down_b = advance(ptr, D);

    bw->attn_ln_w = advance(ptr, D);
    bw->attn_ln_b = advance(ptr, D);
    bw->attn_qkv_w = advance(ptr, 3 * D * D);
    bw->attn_qkv_b = advance(ptr, 3 * D);
    bw->attn_out_w = advance(ptr, D * D);
    bw->attn_out_b = advance(ptr, D);

    bw->conv_ln_w = advance(ptr, D);
    bw->conv_ln_b = advance(ptr, D);
    bw->conv_pw1_w = advance(ptr, 2 * D * D);
    bw->conv_pw1_b = advance(ptr, 2 * D);
    bw->conv_dw_w = advance(ptr, D * kernel);
    bw->conv_dw_b = advance(ptr, D);
    bw->conv_bn_g = advance(ptr, D);
    bw->conv_bn_b = advance(ptr, D);
    bw->conv_bn_mean = advance(ptr, D);
    bw->conv_bn_var = advance(ptr, D);
    bw->conv_pw2_w = advance(ptr, D * D);
    bw->conv_pw2_b = advance(ptr, D);

    bw->ff2_ln_w = advance(ptr, D);
    bw->ff2_ln_b = advance(ptr, D);
    bw->ff2_up_w = advance(ptr, ff_dim * D);
    bw->ff2_up_b = advance(ptr, ff_dim);
    bw->ff2_down_w = advance(ptr, D * ff_dim);
    bw->ff2_down_b = advance(ptr, D);

    bw->final_ln_w = advance(ptr, D);
    bw->final_ln_b = advance(ptr, D);

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Internal: run encoder + CTC proj, store logits in stt->logits_buf
 * ═══════════════════════════════════════════════════════════════════════════ */

static int run_encoder(SonataSTT *stt, const float *pcm, int n_samples,
                       float *out_logits, int max_frames) {
    if (!stt || !pcm || n_samples <= 0) return -1;

    int D = stt->enc_dim;
    int ff = stt->ff_dim;

    int n_frames = mel_process(stt->mel, pcm, n_samples, stt->mel_buf, MAX_FRAMES);
    if (n_frames <= 0) return 0;
    if (n_frames > max_frames) n_frames = max_frames;

    float *x = stt->buf_a;
    linear_forward(x, stt->mel_buf, stt->input_proj_w, stt->input_proj_b,
                   n_frames, stt->n_mels, D);

    for (int l = 0; l < stt->n_layers; l++) {
        conformer_block(x, &stt->blocks[l],
                        stt->buf_b, stt->buf_ff, stt->buf_qkv,
                        stt->buf_attn, stt->buf_conv, stt->buf_pw1,
                        stt->rope_cos, stt->rope_sin, stt->silu_tmp,
                        n_frames, D, ff, stt->n_heads, stt->conv_kernel);
    }

    layer_norm(stt->buf_b, x, stt->adapter_ln_w, stt->adapter_ln_b, n_frames, D);
    linear_forward(x, stt->buf_b, stt->adapter_w, stt->adapter_b, n_frames, D, D);
    silu_inplace(x, n_frames * D, stt->silu_tmp);

    linear_forward(out_logits, x, stt->ctc_w, stt->ctc_b,
                   n_frames, D, stt->text_vocab);

    stt->last_n_frames = n_frames;
    return n_frames;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Beam search (weak link to ctc_beam_decoder)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Weak-linked so the library compiles without ctc_beam_decoder */
int ctc_beam_decode(CTCBeamDecoder *dec,
                    const float *log_probs,
                    int T, int vocab_size,
                    char *out, int out_cap) __attribute__((weak));
int ctc_beam_decode(CTCBeamDecoder *dec __attribute__((unused)),
                    const float *log_probs __attribute__((unused)),
                    int T __attribute__((unused)),
                    int vocab_size __attribute__((unused)),
                    char *out __attribute__((unused)),
                    int out_cap __attribute__((unused))) {
    return -1;
}

static int decode_with_beam(SonataSTT *stt, const float *logits,
                            int n_frames, char *out, int max_len) {
    if (!stt->beam_dec || !ctc_beam_decode) {
        return ctc_greedy_decode_impl(logits, n_frames, stt->text_vocab, out, max_len,
                                      stt->align_token_id, stt->align_frame,
                                      stt->align_prob, &stt->align_len);
    }

    /* Beam decoder: no per-frame alignment available */
    stt->align_len = 0;

    /* Beam decoder expects log-softmax probabilities */
    int V = stt->text_vocab;
    float *log_probs = stt->buf_ff;
    for (int t = 0; t < n_frames; t++) {
        const float *row = logits + t * V;
        float *lp = log_probs + t * V;
        float mx = row[0];
        for (int v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
        float sum = 0;
        for (int v = 0; v < V; v++) { lp[v] = expf(row[v] - mx); sum += lp[v]; }
        float log_sum = logf(sum) + mx;
        for (int v = 0; v < V; v++) lp[v] = row[v] - log_sum;
    }

    return ctc_beam_decode(stt->beam_dec, log_probs, n_frames, V, out, max_len);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API — Lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

SonataSTT *sonata_stt_create(const char *weights_path) {
    if (!weights_path) return NULL;

    int fd = open(weights_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[sonata_stt] Cannot open %s\n", weights_path);
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "[sonata_stt] fstat failed for %s\n", weights_path);
        close(fd);
        return NULL;
    }
    size_t file_size = st.st_size;
    if (file_size < sizeof(SonataSTTHeader)) {
        fprintf(stderr, "[sonata_stt] File too small for header: %zu bytes\n", file_size);
        close(fd);
        return NULL;
    }

    void *mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "[sonata_stt] mmap failed\n");
        return NULL;
    }

    const SonataSTTHeader *hdr = (const SonataSTTHeader *)mapped;
    if (hdr->magic != SONATA_STT_MAGIC || hdr->version != SONATA_STT_VERSION) {
        fprintf(stderr, "[sonata_stt] Invalid weight file (magic=0x%08x ver=%u)\n",
                hdr->magic, hdr->version);
        munmap(mapped, file_size);
        return NULL;
    }

    SonataSTT *stt = calloc(1, sizeof(SonataSTT));
    if (!stt) { munmap(mapped, file_size); return NULL; }

    stt->enc_dim = hdr->enc_dim;
    stt->n_layers = hdr->n_layers;
    stt->n_heads = hdr->n_heads;
    stt->n_mels = hdr->n_mels;
    stt->conv_kernel = hdr->conv_kernel;
    stt->text_vocab = hdr->text_vocab;

    /* Bug 1: Validate header fields to prevent division by zero and buffer overflows */
    if (stt->n_heads == 0 || stt->enc_dim == 0 || stt->enc_dim % stt->n_heads != 0) {
        fprintf(stderr, "[sonata_stt] Invalid model header (n_heads=%d, enc_dim=%d)\n",
                stt->n_heads, stt->enc_dim);
        munmap(mapped, file_size);
        free(stt);
        return NULL;
    }
    if (stt->n_layers == 0 || stt->text_vocab == 0 || stt->n_mels == 0 || stt->conv_kernel == 0) {
        fprintf(stderr, "[sonata_stt] Invalid model header (n_layers=%d, text_vocab=%d, n_mels=%d, conv_kernel=%d)\n",
                stt->n_layers, stt->text_vocab, stt->n_mels, stt->conv_kernel);
        munmap(mapped, file_size);
        free(stt);
        return NULL;
    }

    /* Bug 4b: Validate file contains enough data for declared weights */
    size_t expected_size = sizeof(SonataSTTHeader) + (size_t)hdr->n_weights * sizeof(float);
    if (expected_size > file_size) {
        fprintf(stderr, "[sonata_stt] Weight file truncated (expected %zu bytes, got %zu)\n",
                expected_size, file_size);
        munmap(mapped, file_size);
        free(stt);
        return NULL;
    }

    stt->head_dim = hdr->enc_dim / hdr->n_heads;
    stt->ff_dim = (int)(hdr->enc_dim * 4);
    stt->mmap_base = mapped;
    stt->mmap_size = file_size;

    int D = stt->enc_dim;
    int ff = stt->ff_dim;
    int K = stt->conv_kernel;

    const float *ptr = (const float *)((const char *)mapped + sizeof(SonataSTTHeader));

    stt->input_proj_w = advance(&ptr, D * stt->n_mels);
    stt->input_proj_b = advance(&ptr, D);

    stt->blocks = calloc(stt->n_layers, sizeof(BlockWeights));
    for (int l = 0; l < stt->n_layers; l++)
        load_block_weights(&stt->blocks[l], &ptr, D, ff, K);

    stt->adapter_ln_w = advance(&ptr, D);
    stt->adapter_ln_b = advance(&ptr, D);
    stt->adapter_w = advance(&ptr, D * D);
    stt->adapter_b = advance(&ptr, D);

    stt->ctc_w = advance(&ptr, stt->text_vocab * D);
    stt->ctc_b = advance(&ptr, stt->text_vocab);

    /* Bug 4c: Verify pointer didn't advance past the mmap'd region */
    const float *mmap_end = (const float *)((const char *)mapped + file_size);
    if (ptr > mmap_end) {
        fprintf(stderr, "[sonata_stt] Weight file truncated: pointer overran mmap region\n");
        free(stt->blocks);
        free(stt);
        munmap(mapped, file_size);
        return NULL;
    }

    MelConfig mel_cfg;
    mel_config_default(&mel_cfg);
    mel_cfg.sample_rate = 24000;
    mel_cfg.n_fft = 1024;
    mel_cfg.hop_length = 480;
    mel_cfg.win_length = 1024;
    mel_cfg.n_mels = stt->n_mels;
    mel_cfg.preemph = 0.0f;
    mel_cfg.log_floor = 1e-7f;
    mel_cfg.slaney_norm = 0;
    mel_cfg.periodic_window = 1;
    stt->mel = mel_create(&mel_cfg);
    if (!stt->mel) {
        fprintf(stderr, "[sonata_stt] Failed to create mel extractor\n");
        free(stt->blocks);
        free(stt);
        munmap(mapped, file_size);
        return NULL;
    }

    stt->buf_a = calloc(MAX_FRAMES * MAX_ENC_DIM, sizeof(float));
    stt->buf_b = calloc(MAX_FRAMES * MAX_ENC_DIM, sizeof(float));
    stt->buf_ff = calloc(MAX_FRAMES * MAX_FF_DIM, sizeof(float));
    stt->buf_qkv = calloc(MAX_FRAMES * 3 * MAX_ENC_DIM, sizeof(float));
    /* Sized for GEMM-based MHSA: 3*T*head_dim + T*T (head_dim can be up to enc_dim when n_heads=1) */
    stt->buf_attn = calloc(3 * MAX_FRAMES * (size_t)stt->head_dim + MAX_FRAMES * MAX_FRAMES, sizeof(float));
    stt->buf_conv = calloc(MAX_FRAMES * MAX_ENC_DIM * 2, sizeof(float));
    stt->buf_pw1 = calloc(MAX_FRAMES * MAX_ENC_DIM * 2, sizeof(float));
    stt->mel_buf = calloc(MAX_FRAMES * stt->n_mels, sizeof(float));
    stt->logits_buf = calloc(MAX_FRAMES * (size_t)stt->text_vocab, sizeof(float));

    /* RoPE precomputation */
    int half_dim = stt->head_dim / 2;
    stt->rope_cos = malloc((size_t)MAX_FRAMES * half_dim * sizeof(float));
    stt->rope_sin = malloc((size_t)MAX_FRAMES * half_dim * sizeof(float));
    if (!stt->rope_cos || !stt->rope_sin) {
        fprintf(stderr, "[sonata_stt] Failed to allocate RoPE tables\n");
        free(stt->rope_cos);
        free(stt->rope_sin);
        goto cleanup_create;
    }
    for (int i = 0; i < half_dim; i++) {
        float freq = 1.0f / powf(10000.0f, 2.0f * i / (float)stt->head_dim);
        for (int t = 0; t < MAX_FRAMES; t++) {
            float angle = (float)t * freq;
            stt->rope_cos[t * half_dim + i] = cosf(angle);
            stt->rope_sin[t * half_dim + i] = sinf(angle);
        }
    }

    /* SiLU/GLU workspace (avoid malloc in hot path) */
    stt->silu_tmp = calloc((size_t)MAX_FRAMES * MAX_FF_DIM, sizeof(float));
    if (!stt->silu_tmp) {
        fprintf(stderr, "[sonata_stt] Failed to allocate silu workspace\n");
        goto cleanup_create;
    }

    stt->align_token_id = calloc(MAX_FRAMES, sizeof(int));
    stt->align_frame = calloc(MAX_FRAMES, sizeof(int));
    stt->align_prob = calloc(MAX_FRAMES, sizeof(float));
    stt->align_len = 0;

    stt->beam_dec = NULL;
    stt->stream_audio = NULL;
    stt->stream_pos = 0;
    stt->stream_cap = 0;
    stt->streaming = 0;
    stt->last_n_frames = 0;
    stt->fp16_weights = NULL;
    stt->fp16_n_weights = 0;

    fprintf(stderr, "[sonata_stt] Loaded: %d layers, d=%d, %d heads, vocab=%d (%.1fM params)\n",
            stt->n_layers, D, stt->n_heads, stt->text_vocab,
            (float)hdr->n_weights / 1e6f);

    return stt;

cleanup_create:
    mel_destroy(stt->mel);
    free(stt->blocks);
    free(stt->buf_a);
    free(stt->buf_b);
    free(stt->buf_ff);
    free(stt->buf_qkv);
    free(stt->buf_attn);
    free(stt->buf_conv);
    free(stt->buf_pw1);
    free(stt->mel_buf);
    free(stt->logits_buf);
    free(stt->rope_cos);
    free(stt->rope_sin);
    free(stt->silu_tmp);
    if (stt->mmap_base)
        munmap(stt->mmap_base, stt->mmap_size);
    free(stt);
    return NULL;
}

void sonata_stt_destroy(SonataSTT *stt) {
    if (!stt) return;
    mel_destroy(stt->mel);
    free(stt->blocks);
    free(stt->buf_a);
    free(stt->buf_b);
    free(stt->buf_ff);
    free(stt->buf_qkv);
    free(stt->buf_attn);
    free(stt->buf_conv);
    free(stt->buf_pw1);
    free(stt->mel_buf);
    free(stt->logits_buf);
    free(stt->rope_cos);
    free(stt->rope_sin);
    free(stt->silu_tmp);
    free(stt->align_token_id);
    free(stt->align_frame);
    free(stt->align_prob);
    free(stt->stream_audio);
    free(stt->fp16_weights);
    if (stt->mmap_base)
        munmap(stt->mmap_base, stt->mmap_size);
    free(stt);
}

void sonata_stt_reset(SonataSTT *stt) {
    if (!stt) return;
    if (stt->mel) mel_reset(stt->mel);
    stt->last_n_frames = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API — Batch processing
 * ═══════════════════════════════════════════════════════════════════════════ */

int sonata_stt_get_logits(SonataSTT *stt, const float *pcm, int n_samples,
                           float *out_logits, int max_frames) {
    return run_encoder(stt, pcm, n_samples, out_logits, max_frames);
}

int sonata_stt_process(SonataSTT *stt, const float *pcm, int n_samples,
                        char *out_text, int max_len) {
    if (!stt || !pcm || !out_text || max_len <= 0) return -1;

    int n_frames = run_encoder(stt, pcm, n_samples, stt->logits_buf, MAX_FRAMES);
    if (n_frames <= 0) {
        out_text[0] = '\0';
        stt->align_len = 0;
        return 0;
    }

    return ctc_greedy_decode_impl(stt->logits_buf, n_frames, stt->text_vocab,
                                   out_text, max_len,
                                   stt->align_token_id, stt->align_frame,
                                   stt->align_prob, &stt->align_len);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API — Streaming
 * ═══════════════════════════════════════════════════════════════════════════ */

int sonata_stt_stream_start(SonataSTT *stt, float max_seconds) {
    if (!stt) return -1;
    if (max_seconds <= 0) max_seconds = 30.0f;
    if (max_seconds > 3600.0f) max_seconds = 3600.0f;

    int cap = (int)(24000.0f * max_seconds);
    free(stt->stream_audio);
    stt->stream_audio = calloc(cap, sizeof(float));
    if (!stt->stream_audio) return -1;
    stt->stream_cap = cap;
    stt->stream_pos = 0;
    stt->streaming = 1;
    stt->last_n_frames = 0;
    mel_reset(stt->mel);
    return 0;
}

int sonata_stt_stream_feed(SonataSTT *stt, const float *pcm, int n_samples) {
    if (!stt || !stt->streaming || !pcm || n_samples <= 0) return -1;
    if (stt->stream_pos + n_samples > stt->stream_cap) return -1;

    memcpy(stt->stream_audio + stt->stream_pos, pcm, n_samples * sizeof(float));
    stt->stream_pos += n_samples;
    return 0;
}

int sonata_stt_stream_flush(SonataSTT *stt, char *out_text, int max_len) {
    if (!stt || !stt->streaming || !out_text || max_len <= 0) return -1;
    if (stt->stream_pos <= 0) { out_text[0] = '\0'; stt->align_len = 0; return 0; }

    mel_reset(stt->mel);

    int n_frames = run_encoder(stt, stt->stream_audio, stt->stream_pos,
                               stt->logits_buf, MAX_FRAMES);
    if (n_frames <= 0) { out_text[0] = '\0'; stt->align_len = 0; return 0; }

    return decode_with_beam(stt, stt->logits_buf, n_frames, out_text, max_len);
}

void sonata_stt_stream_end(SonataSTT *stt) {
    if (!stt) return;
    stt->streaming = 0;
    stt->stream_pos = 0;
    stt->last_n_frames = 0;
    mel_reset(stt->mel);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API — Beam search
 * ═══════════════════════════════════════════════════════════════════════════ */

void sonata_stt_set_beam_decoder(SonataSTT *stt, CTCBeamDecoder *beam) {
    if (stt) stt->beam_dec = beam;
}

int sonata_stt_process_beam(SonataSTT *stt, const float *pcm, int n_samples,
                             char *out_text, int max_len) {
    if (!stt || !pcm || !out_text || max_len <= 0) return -1;

    int n_frames = run_encoder(stt, pcm, n_samples, stt->logits_buf, MAX_FRAMES);
    if (n_frames <= 0) { out_text[0] = '\0'; stt->align_len = 0; return 0; }

    return decode_with_beam(stt, stt->logits_buf, n_frames, out_text, max_len);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API — Word-level timestamps
 * ═══════════════════════════════════════════════════════════════════════════ */

int sonata_stt_get_words(const SonataSTT *stt, SonataSTTWord *out, int max_words) {
    if (!stt) return -1;
    if (!out || max_words <= 0) return -1;

    if (stt->align_len <= 0)
        return 0;  /* Silence, or beam decode (no alignment) */

    const int *tid = stt->align_token_id;
    const int *fr = stt->align_frame;
    const float *pr = stt->align_prob;
    int n = stt->align_len;

    int n_words = 0;
    int i = 0;

    while (i < n && n_words < max_words) {
        /* Skip leading spaces */
        while (i < n && tid[i] == CTC_SPACE) i++;
        if (i >= n) break;

        /* Word spans from i to j (exclusive) where next space or end */
        int j = i;
        while (j < n && tid[j] != CTC_SPACE) j++;

        /* Build word string from tokens i..j-1 */
        char word[64];
        int wi = 0;
        for (int k = i; k < j && wi < 63; k++) {
            if (tid[k] > 0 && tid[k] < (int)sizeof(CTC_CHARS) && CTC_CHARS[tid[k]] != ' ')
                word[wi++] = CTC_CHARS[tid[k]];
        }
        word[wi] = '\0';

        if (wi > 0) {
            SonataSTTWord *w = &out[n_words];
            strncpy(w->word, word, 63);
            w->word[63] = '\0';
            w->start_sec = (float)fr[i] * (float)HOP_LENGTH / (float)SAMPLE_RATE;
            w->end_sec = (float)(fr[j - 1] + 1) * (float)HOP_LENGTH / (float)SAMPLE_RATE;
            /* Geometric mean of non-blank probs for this word */
            float log_sum = 0.0f;
            int count = 0;
            for (int k = i; k < j; k++) {
                float p = pr[k];
                if (p < 1e-10f) p = 1e-10f;
                log_sum += logf(p);
                count++;
            }
            w->confidence = (count > 0) ? expf(log_sum / (float)count) : 0.0f;
            n_words++;
        }
        i = j;
    }
    return n_words;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API — EOU detection
 * ═══════════════════════════════════════════════════════════════════════════ */

int sonata_stt_eou_probs(SonataSTT *stt, float *out_probs, int max_frames) {
    if (!stt || !out_probs || stt->last_n_frames <= 0) return -1;

    int T = stt->last_n_frames;
    if (T > max_frames) T = max_frames;
    int V = stt->text_vocab;

    if (CTC_EOU_ID >= V) return -1;

    for (int t = 0; t < T; t++) {
        const float *row = stt->logits_buf + t * V;
        float mx = row[0];
        for (int v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
        float sum = 0;
        for (int v = 0; v < V; v++) sum += expf(row[v] - mx);
        out_probs[t] = expf(row[CTC_EOU_ID] - mx) / sum;
    }
    return T;
}

float sonata_stt_eou_peak(SonataSTT *stt, int window_frames) {
    if (!stt || stt->last_n_frames <= 0) return -1.0f;

    int T = stt->last_n_frames;
    int V = stt->text_vocab;
    if (CTC_EOU_ID >= V) return -1.0f;

    int start = 0;
    if (window_frames > 0 && window_frames < T)
        start = T - window_frames;

    float peak = 0.0f;
    for (int t = start; t < T; t++) {
        const float *row = stt->logits_buf + t * V;
        float mx = row[0];
        for (int v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
        float sum = 0;
        for (int v = 0; v < V; v++) sum += expf(row[v] - mx);
        float p = expf(row[CTC_EOU_ID] - mx) / sum;
        if (p > peak) peak = p;
    }
    return peak;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API — FP16 acceleration
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * FP16 support on Apple Silicon:
 *
 * The AMX coprocessor supports cblas_hgemm (_Float16 GEMM) which provides
 * ~2x throughput by halving memory bandwidth. We convert FP32 mmap'd weights
 * to FP16 at enable-time, then use vDSP for F16↔F32 conversion at boundaries.
 *
 * Strategy:
 *   - Weights: stored in FP16 (halves model memory from ~24MB to ~12MB)
 *   - Activations: remain FP32 for numerical stability
 *   - GEMM: FP32 input × FP16 weights → FP32 output (mixed precision)
 *     Apple's cblas_sgemm with FP16 weight pointer achieves this via AMX
 *   - LayerNorm/SiLU/Softmax: stay FP32
 *
 * This is equivalent to what Sonata LM/Flow do on Metal GPU (FP16 weights,
 * FP32 compute), but on the AMX coprocessor.
 *
 * NOTE: FP16 weights are pre-allocated and converted here, but inference
 * remains FP32 until linear_forward_fp16 is implemented.
 */

int sonata_stt_enable_fp16(SonataSTT *stt) {
    if (!stt) return -1;
    if (stt->fp16_weights) return 0;

    /* Count total weight floats from the header */
    const SonataSTTHeader *hdr = (const SonataSTTHeader *)stt->mmap_base;
    int n_weights = (int)hdr->n_weights;
    if (n_weights <= 0) return -1;

    /* Allocate FP16 buffer and convert (ARM64 supports native __fp16 casts) */
    _Float16 *fp16 = (_Float16 *)malloc(n_weights * sizeof(_Float16));
    if (!fp16) return -1;

    const float *src = (const float *)((const char *)stt->mmap_base + sizeof(SonataSTTHeader));
    for (int i = 0; i < n_weights; i++)
        fp16[i] = (_Float16)src[i];

    stt->fp16_weights = fp16;
    stt->fp16_n_weights = n_weights;

    /* Repoint all weight pointers into the fp16 buffer at the same offsets */
    /* The pointers stay as float* for the existing code paths — the FP16
     * path is selected in linear_forward_fp16 which casts internally. */

    fprintf(stderr, "[sonata-stt] Warning: FP16 storage enabled but inference uses FP32 (FP16 kernels not yet implemented)\n");
    fprintf(stderr, "[sonata_stt] FP16 enabled: %.1fMB → %.1fMB weights\n",
            n_weights * 4.0f / 1e6f, n_weights * 2.0f / 1e6f);
    return 0;
}

int sonata_stt_is_fp16(const SonataSTT *stt) {
    return (stt && stt->fp16_weights) ? 1 : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API — Properties
 * ═══════════════════════════════════════════════════════════════════════════ */

int sonata_stt_vocab_size(const SonataSTT *stt) {
    return stt ? stt->text_vocab : 0;
}

int sonata_stt_enc_dim(const SonataSTT *stt) {
    return stt ? stt->enc_dim : 0;
}

int sonata_stt_eou_id(const SonataSTT *stt) {
    return stt ? CTC_EOU_ID : -1;
}
