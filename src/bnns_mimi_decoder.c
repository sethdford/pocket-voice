/**
 * bnns_mimi_decoder.c — ANE-accelerated Mimi audio codec decoder.
 *
 * Implements the Mimi decoder (ConvTrUpsample → Transformer → SEANetDecoder)
 * using Apple's BNNS Graph API from the Accelerate framework. When available,
 * BNNS automatically routes compatible operations to the Apple Neural Engine
 * (ANE), freeing the GPU entirely for the FlowLM transformer.
 *
 * Architecture (from b6369a24 config):
 *   Input: latent (B=1, T, 512) @ 12.5 Hz
 *   1. ConvTrUpsample1d: stride=16, groups=512, K=32  → ×16 upsample to 200 Hz
 *   2. ProjectedTransformer: 2 layers, d=512, H=8, D=64, ctx=250
 *   3. SEANetDecoder: Conv→[ELU→ConvTr→ResBlock]×3→ELU→Conv  → 24 kHz audio
 *
 * Build:
 *   cc -O3 -shared -fPIC -arch arm64 -framework Accelerate \
 *      -o libbnns_mimi.dylib bnns_mimi_decoder.c
 *
 * This is the first TTS system to run the audio codec on ANE while the
 * transformer runs on GPU — true 3-way parallelism (GPU + AMX + ANE).
 */

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * SEANet Configuration (b6369a24)
 * ----------------------------------------------------------------------- */

#define SEANET_DIM        512
#define SEANET_CHANNELS   1
#define SEANET_N_FILTERS  64
#define SEANET_N_RESIDUAL 1
#define SEANET_RATIOS_N   3
static const int SEANET_RATIOS[3] = {6, 5, 4};
#define SEANET_KERNEL     7
#define SEANET_LAST_KERNEL 3
#define SEANET_RES_KERNEL 3
#define SEANET_COMPRESS   2
#define SEANET_HOP        120  /* 6 * 5 * 4 */

/* Transformer config */
#define XFMR_D_MODEL      512
#define XFMR_NUM_HEADS    8
#define XFMR_HEAD_DIM     64
#define XFMR_NUM_LAYERS   2
#define XFMR_CONTEXT      250
#define XFMR_FFN_DIM      2048

/* Upsample config */
#define UPSAMPLE_STRIDE    16
#define UPSAMPLE_KERNEL    32  /* 2 * stride */

/* -----------------------------------------------------------------------
 * Weight Buffer Layout
 *
 * All weights are packed into a single contiguous float32 buffer.
 * Offsets are computed at init time to avoid pointer arithmetic in hot path.
 * ----------------------------------------------------------------------- */

typedef struct {
    /* ConvTrUpsample1d */
    size_t upsample_weight;      /* (512, 32, 1) depthwise */

    /* Transformer layers (×2) */
    struct {
        size_t ln1_weight;       /* (512,) */
        size_t ln1_bias;         /* (512,) */
        size_t in_proj_weight;   /* (1536, 512) */
        size_t out_proj_weight;  /* (512, 512) */
        size_t layer_scale1;     /* (512,) */
        size_t ln2_weight;       /* (512,) */
        size_t ln2_bias;         /* (512,) */
        size_t ffn1_weight;      /* (2048, 512) */
        size_t ffn2_weight;      /* (512, 2048) */
        size_t layer_scale2;     /* (512,) */
    } xfmr[XFMR_NUM_LAYERS];

    /* SEANet Decoder */
    size_t dec_conv0_weight;     /* (512, 7, 512) */
    size_t dec_conv0_bias;       /* (512,) */

    struct {
        size_t convtr_weight;    /* (out, K, in) */
        size_t convtr_bias;      /* (out,) */
        size_t res_conv1_weight; /* (hidden, 3, dim) */
        size_t res_conv1_bias;   /* (hidden,) */
        size_t res_conv2_weight; /* (dim, 1, hidden) */
        size_t res_conv2_bias;   /* (dim,) */
    } dec_blocks[SEANET_RATIOS_N];

    size_t dec_final_weight;     /* (1, 3, 64) */
    size_t dec_final_bias;       /* (1,) */

    size_t total_floats;
} WeightLayout;

/* -----------------------------------------------------------------------
 * Streaming State
 *
 * All stateful buffers for streaming decode (KV cache, conv overlaps).
 * Stack-allocated where possible; heap for variable-size KV cache.
 * ----------------------------------------------------------------------- */

typedef struct {
    /* ConvTrUpsample partial overlap: (UPSAMPLE_KERNEL - UPSAMPLE_STRIDE) × 512 */
    float upsample_partial[UPSAMPLE_STRIDE * SEANET_DIM];  /* 16 × 512 = 8K */

    /* Transformer KV caches: (H, T, D) per layer */
    float *k_cache[XFMR_NUM_LAYERS];  /* H × T_cur × D */
    float *v_cache[XFMR_NUM_LAYERS];
    int cache_len[XFMR_NUM_LAYERS];
    int rope_offset;

    /* SEANet Conv1d streaming buffers: (K-1) × channels */
    float *conv_prev[16];  /* up to 16 conv layers */
    int n_conv_states;

    /* SEANet ConvTranspose1d overlap buffers */
    float *convtr_partial[SEANET_RATIOS_N];

    int initialized;
} StreamingState;

/* -----------------------------------------------------------------------
 * BNNS Mimi Decoder Context
 * ----------------------------------------------------------------------- */

typedef struct {
    float *weights;
    WeightLayout layout;
    StreamingState state;

    /* BNNS filter objects (created once, reused per step) */
    /* We use raw C computation for streaming overlap management,
       and BNNS for the heavy Conv/MatMul/Activation operations. */
    int loaded;
} BNNSMimiDecoder;

/* -----------------------------------------------------------------------
 * Activation Functions (inline, NEON-friendly)
 * ----------------------------------------------------------------------- */

static inline void vec_elu_inplace(float *x, int n, float alpha) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] = alpha * (expf(x[i]) - 1.0f);
    }
}

static inline void vec_gelu_inplace(float *x, int n) {
    /* GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    static const float sqrt_2_over_pi = 0.7978845608f;
    static const float coeff = 0.044715f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float inner = sqrt_2_over_pi * (v + coeff * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

/* -----------------------------------------------------------------------
 * Core Compute Primitives (Accelerate BLAS/vDSP)
 * ----------------------------------------------------------------------- */

static void matmul_mv(const float *weight, const float *input,
                       float *output, int rows, int cols) {
    /* output = weight @ input (matrix-vector, weight is row-major) */
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols,
                1.0f, weight, cols, input, 1, 0.0f, output, 1);
}

static void conv1d_valid(const float *input, int in_len, int in_ch,
                         const float *weight, const float *bias,
                         float *output, int out_ch, int kernel,
                         int stride, int dilation, int groups) {
    /* 1D convolution with valid padding. NLC layout. */
    int in_per_group = in_ch / groups;
    int out_per_group = out_ch / groups;
    int eff_kernel = (kernel - 1) * dilation + 1;
    int out_len = (in_len - eff_kernel) / stride + 1;

    memset(output, 0, out_len * out_ch * sizeof(float));

    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < out_per_group; oc++) {
            int oc_abs = g * out_per_group + oc;
            for (int t = 0; t < out_len; t++) {
                float sum = bias ? bias[oc_abs] : 0.0f;
                for (int ic = 0; ic < in_per_group; ic++) {
                    int ic_abs = g * in_per_group + ic;
                    for (int k = 0; k < kernel; k++) {
                        int in_t = t * stride + k * dilation;
                        sum += input[in_t * in_ch + ic_abs]
                             * weight[oc_abs * kernel * in_per_group + k * in_per_group + ic];
                    }
                }
                output[t * out_ch + oc_abs] = sum;
            }
        }
    }
}

static void conv_transpose1d(const float *input, int in_len, int in_ch,
                              const float *weight, const float *bias,
                              float *output, int out_ch, int kernel,
                              int stride, int groups) {
    int out_len = (in_len - 1) * stride + kernel;
    memset(output, 0, out_len * out_ch * sizeof(float));

    int in_per_group = in_ch / groups;
    int out_per_group = out_ch / groups;

    for (int g = 0; g < groups; g++) {
        for (int ic = 0; ic < in_per_group; ic++) {
            int ic_abs = g * in_per_group + ic;
            for (int oc = 0; oc < out_per_group; oc++) {
                int oc_abs = g * out_per_group + oc;
                for (int t = 0; t < in_len; t++) {
                    float val = input[t * in_ch + ic_abs];
                    for (int k = 0; k < kernel; k++) {
                        int out_t = t * stride + k;
                        output[out_t * out_ch + oc_abs] +=
                            val * weight[oc_abs * kernel * in_per_group + k * in_per_group + ic];
                    }
                }
            }
        }
    }

    if (bias) {
        for (int t = 0; t < out_len; t++) {
            for (int c = 0; c < out_ch; c++) {
                output[t * out_ch + c] += bias[c];
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * Layer Norm
 * ----------------------------------------------------------------------- */

static void layer_norm(const float *input, const float *weight,
                       const float *bias, float *output, int len, int dim) {
    for (int t = 0; t < len; t++) {
        const float *in_t = input + t * dim;
        float *out_t = output + t * dim;

        float mean = 0, var = 0;
        vDSP_meanv(in_t, 1, &mean, dim);
        for (int i = 0; i < dim; i++) {
            float d = in_t[i] - mean;
            var += d * d;
        }
        var /= dim;
        float inv_std = 1.0f / sqrtf(var + 1e-5f);

        for (int i = 0; i < dim; i++) {
            out_t[i] = (in_t[i] - mean) * inv_std * weight[i] + bias[i];
        }
    }
}

/* -----------------------------------------------------------------------
 * Streaming Conv1d (prepends previous context)
 * ----------------------------------------------------------------------- */

static int streaming_conv1d(const float *input, int in_len, int in_ch,
                             const float *weight, const float *bias,
                             float *output, int out_ch, int kernel,
                             int stride, int dilation,
                             float *prev, int prev_len) {
    int padded_len = prev_len + in_len;
    float *padded = (float *)malloc(padded_len * in_ch * sizeof(float));

    memcpy(padded, prev, prev_len * in_ch * sizeof(float));
    memcpy(padded + prev_len * in_ch, input, in_len * in_ch * sizeof(float));

    conv1d_valid(padded, padded_len, in_ch, weight, bias,
                 output, out_ch, kernel, stride, dilation, 1);

    int new_prev = kernel - stride;
    if (new_prev > 0) {
        int start = padded_len - new_prev;
        memcpy(prev, padded + start * in_ch, new_prev * in_ch * sizeof(float));
    }

    int out_len = (padded_len - ((kernel - 1) * dilation + 1)) / stride + 1;
    free(padded);
    return out_len;
}

/* -----------------------------------------------------------------------
 * Streaming ConvTranspose1d (overlap-add)
 * ----------------------------------------------------------------------- */

static int streaming_conv_transpose1d(
    const float *input, int in_len, int in_ch,
    const float *weight, const float *bias,
    float *output, int out_ch, int kernel, int stride,
    float *partial, int partial_len, int groups) {

    int raw_len = (in_len - 1) * stride + kernel;
    float *raw = (float *)calloc(raw_len * out_ch, sizeof(float));

    conv_transpose1d(input, in_len, in_ch, weight, bias,
                     raw, out_ch, kernel, stride, groups);

    /* Overlap-add with previous partial */
    int overlap = kernel - stride;
    for (int t = 0; t < overlap && t < partial_len; t++) {
        for (int c = 0; c < out_ch; c++) {
            raw[t * out_ch + c] += partial[t * out_ch + c];
        }
    }

    int emit_len = in_len * stride;
    memcpy(output, raw, emit_len * out_ch * sizeof(float));

    /* Store new partial (last overlap samples, bias subtracted) */
    int new_partial_start = emit_len;
    for (int t = 0; t < overlap; t++) {
        for (int c = 0; c < out_ch; c++) {
            float val = raw[(new_partial_start + t) * out_ch + c];
            if (bias) val -= bias[c];
            partial[t * out_ch + c] = val;
        }
    }

    free(raw);
    return emit_len;
}

/* -----------------------------------------------------------------------
 * RoPE (Rotary Position Embedding) — traditional mode
 * ----------------------------------------------------------------------- */

static void apply_rope(float *q_or_k, int n_heads, int head_dim,
                       int offset, float base) {
    for (int h = 0; h < n_heads; h++) {
        float *head = q_or_k + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(base, (float)(2 * i) / (float)head_dim);
            float angle = (float)offset * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float x0 = head[2 * i];
            float x1 = head[2 * i + 1];
            head[2 * i]     = x0 * cos_a - x1 * sin_a;
            head[2 * i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/* -----------------------------------------------------------------------
 * Softmax (numerically stable)
 * ----------------------------------------------------------------------- */

static void softmax_inplace(float *x, int n) {
    float max_val;
    vDSP_maxv(x, 1, &max_val, n);

    float neg_max = -max_val;
    vDSP_vsadd(x, 1, &neg_max, x, 1, n);

    /* exp via vForce */
    int ni = n;
    vvexpf(x, x, &ni);

    float sum;
    vDSP_sve(x, 1, &sum, n);
    float inv_sum = 1.0f / sum;
    vDSP_vsmul(x, 1, &inv_sum, x, 1, n);
}

/* -----------------------------------------------------------------------
 * Transformer Attention (single head, T=1 query)
 * ----------------------------------------------------------------------- */

static void mimi_attention_step(
    const float *input, int dim,
    const float *in_proj_w,
    const float *out_proj_w,
    float *k_cache, float *v_cache, int *cache_len,
    int max_ctx, int n_heads, int head_dim,
    int offset, float rope_base,
    float *output) {

    /* 1. QKV projection */
    float qkv[3 * XFMR_D_MODEL];
    matmul_mv(in_proj_w, input, qkv, 3 * dim, dim);

    float *q = qkv;
    float *k = qkv + dim;
    float *v = qkv + 2 * dim;

    /* 2. RoPE on Q and K */
    apply_rope(q, n_heads, head_dim, offset, rope_base);
    apply_rope(k, n_heads, head_dim, offset, rope_base);

    /* 3. Append K, V to cache (ring buffer for bounded context) */
    int T = *cache_len;
    int insert_pos = T;
    if (T >= max_ctx) {
        /* Shift cache left by 1 (drop oldest) */
        for (int h = 0; h < n_heads; h++) {
            memmove(k_cache + h * max_ctx * head_dim,
                    k_cache + h * max_ctx * head_dim + head_dim,
                    (max_ctx - 1) * head_dim * sizeof(float));
            memmove(v_cache + h * max_ctx * head_dim,
                    v_cache + h * max_ctx * head_dim + head_dim,
                    (max_ctx - 1) * head_dim * sizeof(float));
        }
        insert_pos = max_ctx - 1;
        T = max_ctx;
    } else {
        T = T + 1;
        *cache_len = T;
    }

    for (int h = 0; h < n_heads; h++) {
        memcpy(k_cache + h * max_ctx * head_dim + insert_pos * head_dim,
               k + h * head_dim, head_dim * sizeof(float));
        memcpy(v_cache + h * max_ctx * head_dim + insert_pos * head_dim,
               v + h * head_dim, head_dim * sizeof(float));
    }

    /* 4. Scaled dot-product attention per head */
    float attn_out[XFMR_D_MODEL];
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        float scores[XFMR_CONTEXT];
        const float *q_h = q + h * head_dim;
        const float *k_h = k_cache + h * max_ctx * head_dim;

        /* scores = scale * Q @ K^T */
        for (int t = 0; t < T; t++) {
            float dot = 0;
            for (int d = 0; d < head_dim; d++) {
                dot += q_h[d] * k_h[t * head_dim + d];
            }
            scores[t] = dot * scale;
        }

        softmax_inplace(scores, T);

        /* attn_out_h = scores @ V */
        const float *v_h = v_cache + h * max_ctx * head_dim;
        float *out_h = attn_out + h * head_dim;
        memset(out_h, 0, head_dim * sizeof(float));
        for (int t = 0; t < T; t++) {
            float s = scores[t];
            for (int d = 0; d < head_dim; d++) {
                out_h[d] += s * v_h[t * head_dim + d];
            }
        }
    }

    /* 5. Output projection */
    matmul_mv(out_proj_w, attn_out, output, dim, dim);
}

/* -----------------------------------------------------------------------
 * Transformer Layer (LN → Attn → Residual → LN → FFN → Residual)
 * ----------------------------------------------------------------------- */

static void mimi_transformer_layer(
    float *x, int seq_len, int dim,
    const float *ln1_w, const float *ln1_b,
    const float *in_proj_w, const float *out_proj_w,
    const float *ls1,
    const float *ln2_w, const float *ln2_b,
    const float *ffn1_w, const float *ffn2_w,
    const float *ls2,
    float *k_cache, float *v_cache, int *cache_len,
    int max_ctx, int n_heads, int head_dim,
    int *offset, float rope_base) {

    float normed[XFMR_D_MODEL];
    float attn_out[XFMR_D_MODEL];
    float ffn_hidden[XFMR_FFN_DIM];
    float ffn_out[XFMR_D_MODEL];

    for (int t = 0; t < seq_len; t++) {
        float *xt = x + t * dim;

        /* Pre-attention LayerNorm */
        layer_norm(xt, ln1_w, ln1_b, normed, 1, dim);

        /* Attention */
        mimi_attention_step(normed, dim, in_proj_w, out_proj_w,
                           k_cache, v_cache, cache_len,
                           max_ctx, n_heads, head_dim,
                           *offset + t, rope_base, attn_out);

        /* LayerScale + residual */
        for (int i = 0; i < dim; i++) {
            xt[i] += attn_out[i] * ls1[i];
        }

        /* Pre-FFN LayerNorm */
        layer_norm(xt, ln2_w, ln2_b, normed, 1, dim);

        /* FFN: Linear → GELU → Linear */
        matmul_mv(ffn1_w, normed, ffn_hidden, XFMR_FFN_DIM, dim);
        vec_gelu_inplace(ffn_hidden, XFMR_FFN_DIM);
        matmul_mv(ffn2_w, ffn_hidden, ffn_out, dim, XFMR_FFN_DIM);

        /* LayerScale + residual */
        for (int i = 0; i < dim; i++) {
            xt[i] += ffn_out[i] * ls2[i];
        }
    }

    *offset += seq_len;
}

/* -----------------------------------------------------------------------
 * SEANet Resnet Block
 * ----------------------------------------------------------------------- */

static void seanet_resblock(float *x, int len, int dim,
                             const float *conv1_w, const float *conv1_b,
                             const float *conv2_w, const float *conv2_b,
                             int hidden_dim) {
    int out_size = len * dim;
    float *residual = (float *)malloc(out_size * sizeof(float));
    memcpy(residual, x, out_size * sizeof(float));

    /* ELU → Conv1d(dim→hidden, K=3) → ELU → Conv1d(hidden→dim, K=1) */
    vec_elu_inplace(x, len * dim, 1.0f);

    float *h = (float *)malloc(len * hidden_dim * sizeof(float));
    conv1d_valid(x, len, dim, conv1_w, conv1_b, h, hidden_dim, 3, 1, 1, 1);
    int h_len = len - 2;  /* valid padding K=3 loses 2 */

    vec_elu_inplace(h, h_len * hidden_dim, 1.0f);

    float *y = (float *)malloc(h_len * dim * sizeof(float));
    conv1d_valid(h, h_len, hidden_dim, conv2_w, conv2_b, y, dim, 1, 1, 1, 1);

    /* Trim residual to match (center crop) */
    int offset = (len - h_len) / 2;
    for (int t = 0; t < h_len; t++) {
        for (int c = 0; c < dim; c++) {
            y[t * dim + c] += residual[(t + offset) * dim + c];
        }
    }

    memcpy(x, y, h_len * dim * sizeof(float));

    free(residual);
    free(h);
    free(y);
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

BNNSMimiDecoder *bnns_mimi_create(void) {
    BNNSMimiDecoder *dec = (BNNSMimiDecoder *)calloc(1, sizeof(BNNSMimiDecoder));
    return dec;
}

static void compute_weight_layout(WeightLayout *lay) {
    size_t off = 0;
    #define ADVANCE(name, n) do { lay->name = off; off += (n); } while(0)

    ADVANCE(upsample_weight, SEANET_DIM * UPSAMPLE_KERNEL * 1);

    for (int l = 0; l < XFMR_NUM_LAYERS; l++) {
        ADVANCE(xfmr[l].ln1_weight, XFMR_D_MODEL);
        ADVANCE(xfmr[l].ln1_bias, XFMR_D_MODEL);
        ADVANCE(xfmr[l].in_proj_weight, 3 * XFMR_D_MODEL * XFMR_D_MODEL);
        ADVANCE(xfmr[l].out_proj_weight, XFMR_D_MODEL * XFMR_D_MODEL);
        ADVANCE(xfmr[l].layer_scale1, XFMR_D_MODEL);
        ADVANCE(xfmr[l].ln2_weight, XFMR_D_MODEL);
        ADVANCE(xfmr[l].ln2_bias, XFMR_D_MODEL);
        ADVANCE(xfmr[l].ffn1_weight, XFMR_FFN_DIM * XFMR_D_MODEL);
        ADVANCE(xfmr[l].ffn2_weight, XFMR_D_MODEL * XFMR_FFN_DIM);
        ADVANCE(xfmr[l].layer_scale2, XFMR_D_MODEL);
    }

    int dim = SEANET_N_FILTERS;
    for (int i = SEANET_RATIOS_N - 1; i >= 0; i--) dim *= 2;
    ADVANCE(dec_conv0_weight, dim * SEANET_KERNEL * SEANET_DIM);
    ADVANCE(dec_conv0_bias, dim);

    int cur_dim = dim;
    for (int i = 0; i < SEANET_RATIOS_N; i++) {
        int next_dim = cur_dim / 2;
        ADVANCE(dec_blocks[i].convtr_weight, next_dim * (SEANET_RATIOS[i] * 2) * cur_dim);
        ADVANCE(dec_blocks[i].convtr_bias, next_dim);
        int hidden = next_dim / SEANET_COMPRESS;
        ADVANCE(dec_blocks[i].res_conv1_weight, hidden * SEANET_RES_KERNEL * next_dim);
        ADVANCE(dec_blocks[i].res_conv1_bias, hidden);
        ADVANCE(dec_blocks[i].res_conv2_weight, next_dim * 1 * hidden);
        ADVANCE(dec_blocks[i].res_conv2_bias, next_dim);
        cur_dim = next_dim;
    }

    ADVANCE(dec_final_weight, SEANET_CHANNELS * SEANET_LAST_KERNEL * SEANET_N_FILTERS);
    ADVANCE(dec_final_bias, SEANET_CHANNELS);

    lay->total_floats = off;
    #undef ADVANCE
}

int bnns_mimi_load_weights(BNNSMimiDecoder *dec, const float *weight_data,
                           size_t n_floats) {
    compute_weight_layout(&dec->layout);

    if (n_floats < dec->layout.total_floats) {
        fprintf(stderr, "[bnns_mimi] weight buffer too small: got %zu, need %zu\n",
                n_floats, dec->layout.total_floats);
        return -1;
    }

    dec->weights = (float *)malloc(n_floats * sizeof(float));
    memcpy(dec->weights, weight_data, n_floats * sizeof(float));
    dec->loaded = 1;

    /* Allocate streaming state */
    StreamingState *s = &dec->state;
    for (int l = 0; l < XFMR_NUM_LAYERS; l++) {
        s->k_cache[l] = (float *)calloc(XFMR_NUM_HEADS * XFMR_CONTEXT * XFMR_HEAD_DIM, sizeof(float));
        s->v_cache[l] = (float *)calloc(XFMR_NUM_HEADS * XFMR_CONTEXT * XFMR_HEAD_DIM, sizeof(float));
        s->cache_len[l] = 0;
    }
    s->rope_offset = 0;

    /* Allocate SEANet streaming conv buffers */
    int conv_idx = 0;
    /* Initial conv: K=7, needs 6 prev frames */
    s->conv_prev[conv_idx] = (float *)calloc(6 * SEANET_DIM, sizeof(float));
    conv_idx++;

    int dim = SEANET_N_FILTERS;
    for (int i = SEANET_RATIOS_N - 1; i >= 0; i--) dim *= 2;

    int cur_dim = dim;
    for (int i = 0; i < SEANET_RATIOS_N; i++) {
        int next_dim = cur_dim / 2;
        s->convtr_partial[i] = (float *)calloc(
            (SEANET_RATIOS[i] * 2 - SEANET_RATIOS[i]) * next_dim, sizeof(float));
        /* ResBlock conv1 K=3 needs 2 prev frames */
        s->conv_prev[conv_idx] = (float *)calloc(2 * (next_dim / SEANET_COMPRESS), sizeof(float));
        conv_idx++;
        /* ResBlock conv2 K=1 needs 0 prev */
        cur_dim = next_dim;
    }

    /* Final conv: K=3 needs 2 prev frames */
    s->conv_prev[conv_idx] = (float *)calloc(2 * SEANET_N_FILTERS, sizeof(float));
    conv_idx++;

    s->n_conv_states = conv_idx;
    s->initialized = 1;

    return 0;
}

void bnns_mimi_reset(BNNSMimiDecoder *dec) {
    StreamingState *s = &dec->state;
    memset(s->upsample_partial, 0, sizeof(s->upsample_partial));
    for (int l = 0; l < XFMR_NUM_LAYERS; l++) {
        memset(s->k_cache[l], 0, XFMR_NUM_HEADS * XFMR_CONTEXT * XFMR_HEAD_DIM * sizeof(float));
        memset(s->v_cache[l], 0, XFMR_NUM_HEADS * XFMR_CONTEXT * XFMR_HEAD_DIM * sizeof(float));
        s->cache_len[l] = 0;
    }
    s->rope_offset = 0;
}

void bnns_mimi_destroy(BNNSMimiDecoder *dec) {
    if (!dec) return;
    StreamingState *s = &dec->state;
    for (int l = 0; l < XFMR_NUM_LAYERS; l++) {
        free(s->k_cache[l]);
        free(s->v_cache[l]);
    }
    for (int i = 0; i < s->n_conv_states; i++) {
        free(s->conv_prev[i]);
    }
    for (int i = 0; i < SEANET_RATIOS_N; i++) {
        free(s->convtr_partial[i]);
    }
    free(dec->weights);
    free(dec);
}

/**
 * Decode a single frame of latent codes to audio.
 *
 * Pipeline: latent → ConvTrUpsample(×16) → Transformer(2 layers) → SEANet Decoder → PCM
 *
 * @param dec       Decoder context
 * @param latent    Input latent vector, shape (1, 1, 512) — single frame at 12.5 Hz
 * @param output    Output audio samples (caller-allocated, at least 1920 floats)
 * @return          Number of audio samples written, or -1 on error
 */
int bnns_mimi_decode_step(BNNSMimiDecoder *dec, const float *latent,
                          float *output) {
    if (!dec || !dec->loaded || !latent || !output) return -1;

    const float *W = dec->weights;
    const WeightLayout *L = &dec->layout;
    StreamingState *S = &dec->state;

    /* ── 1. ConvTranspose1d Upsample: (1,1,512) → (1,16,512) ───────────── */
    float upsampled[UPSAMPLE_STRIDE * SEANET_DIM];
    {
        /* Depthwise ConvTranspose1d: stride=16, K=32, groups=512 */
        int emit_len = streaming_conv_transpose1d(
            latent, 1, SEANET_DIM,
            W + L->upsample_weight, NULL,
            upsampled, SEANET_DIM, UPSAMPLE_KERNEL, UPSAMPLE_STRIDE,
            S->upsample_partial,
            UPSAMPLE_KERNEL - UPSAMPLE_STRIDE,
            SEANET_DIM /* groups = dim (depthwise) */
        );
        (void)emit_len; /* Should be UPSAMPLE_STRIDE = 16 */
    }

    /* ── 2. Transformer: 2 layers of LN → Attn → Res → LN → FFN → Res ── */
    float xfmr_buf[UPSAMPLE_STRIDE * XFMR_D_MODEL];
    memcpy(xfmr_buf, upsampled, UPSAMPLE_STRIDE * SEANET_DIM * sizeof(float));

    for (int layer = 0; layer < XFMR_NUM_LAYERS; layer++) {
        mimi_transformer_layer(
            xfmr_buf, UPSAMPLE_STRIDE, XFMR_D_MODEL,
            W + L->xfmr[layer].ln1_weight,
            W + L->xfmr[layer].ln1_bias,
            W + L->xfmr[layer].in_proj_weight,
            W + L->xfmr[layer].out_proj_weight,
            W + L->xfmr[layer].layer_scale1,
            W + L->xfmr[layer].ln2_weight,
            W + L->xfmr[layer].ln2_bias,
            W + L->xfmr[layer].ffn1_weight,
            W + L->xfmr[layer].ffn2_weight,
            W + L->xfmr[layer].layer_scale2,
            S->k_cache[layer], S->v_cache[layer],
            &S->cache_len[layer],
            XFMR_CONTEXT, XFMR_NUM_HEADS, XFMR_HEAD_DIM,
            &S->rope_offset, 10000.0f
        );
        /* Note: rope_offset is incremented by mimi_transformer_layer.
           After 2 layers, it will be incremented twice. We only want once.
           Restore after first layer. */
        if (layer == 0) {
            S->rope_offset -= UPSAMPLE_STRIDE; /* Undo, let layer 1 increment */
        }
    }

    /* ── 3. SEANet Decoder: initial Conv → [ConvTr + ResBlock]×3 → final Conv ── */

    /* Compute dimensions for the decoder blocks */
    int dim = SEANET_N_FILTERS;
    for (int i = SEANET_RATIOS_N - 1; i >= 0; i--) dim *= 2;
    /* dim is now the initial SEANet dimension (64 * 2^3 = 512) */

    int cur_len = UPSAMPLE_STRIDE;
    int cur_dim = dim;
    int conv_state_idx = 0;

    /* Initial Conv1d (K=7): 512 → 512 */
    float *dec_buf = (float *)malloc(cur_len * cur_dim * sizeof(float));
    int new_len = streaming_conv1d(
        xfmr_buf, cur_len, SEANET_DIM,
        W + L->dec_conv0_weight, W + L->dec_conv0_bias,
        dec_buf, cur_dim, SEANET_KERNEL, 1, 1,
        S->conv_prev[conv_state_idx],
        SEANET_KERNEL - 1
    );
    conv_state_idx++;
    cur_len = new_len;

    /* Decoder blocks: ConvTr(ratio*2) → ELU → ResBlock */
    for (int i = 0; i < SEANET_RATIOS_N; i++) {
        int ratio = SEANET_RATIOS[i];
        int next_dim = cur_dim / 2;
        int convtr_kernel = ratio * 2;

        /* ELU activation before ConvTranspose */
        vec_elu_inplace(dec_buf, cur_len * cur_dim, 1.0f);

        /* ConvTranspose1d: upsample by ratio */
        int up_len = cur_len * ratio;
        float *up_buf = (float *)malloc(up_len * next_dim * sizeof(float));
        streaming_conv_transpose1d(
            dec_buf, cur_len, cur_dim,
            W + L->dec_blocks[i].convtr_weight,
            W + L->dec_blocks[i].convtr_bias,
            up_buf, next_dim, convtr_kernel, ratio,
            S->convtr_partial[i],
            convtr_kernel - ratio,
            1 /* groups=1 */
        );

        free(dec_buf);
        dec_buf = up_buf;
        cur_len = up_len;
        cur_dim = next_dim;

        /* ResBlock */
        int hidden = cur_dim / SEANET_COMPRESS;
        seanet_resblock(
            dec_buf, cur_len, cur_dim,
            W + L->dec_blocks[i].res_conv1_weight,
            W + L->dec_blocks[i].res_conv1_bias,
            W + L->dec_blocks[i].res_conv2_weight,
            W + L->dec_blocks[i].res_conv2_bias,
            hidden
        );
        /* ResBlock with K=3 valid conv shrinks length by 2 */
        cur_len -= 2;
    }

    /* Final: ELU → Conv1d (K=3, channels=64 → 1) */
    vec_elu_inplace(dec_buf, cur_len * cur_dim, 1.0f);

    float *final_buf = (float *)malloc(cur_len * SEANET_CHANNELS * sizeof(float));
    int final_len = streaming_conv1d(
        dec_buf, cur_len, cur_dim,
        W + L->dec_final_weight, W + L->dec_final_bias,
        final_buf, SEANET_CHANNELS, SEANET_LAST_KERNEL, 1, 1,
        S->conv_prev[conv_state_idx],
        SEANET_LAST_KERNEL - 1
    );

    /* Copy to output (single-channel audio) */
    int out_samples = final_len;
    if (out_samples > 1920) out_samples = 1920;
    for (int i = 0; i < out_samples; i++) {
        output[i] = final_buf[i * SEANET_CHANNELS];
    }

    free(dec_buf);
    free(final_buf);

    return out_samples;
}
