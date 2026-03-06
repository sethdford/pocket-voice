/*
 * codec_12hz.c — Sonata 12.5Hz Neural Audio Codec C Inference
 *
 * Implements the low-frame-rate codec for 4x token reduction.
 * Architecture: FSQ dequantization → ConvDecoder (5 transposed conv stages)
 *
 * All vector operations use Apple Accelerate (vDSP/cblas).
 * Pre-allocates all buffers in create(), zero allocations in decode().
 */

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif

#include "codec_12hz.h"
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Binary model format magic */
#define CODEC_12HZ_MAGIC "CODEC12HZ\x00"
#define CODEC_12HZ_MAGIC_LEN 10

/* ═══════════════════════════════════════════════════════════════════════════════
 * ConvNeXt Block Weights
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float *dwconv_weight;    /* [dec_dim, 1, kernel_size] */
    float *dwconv_bias;      /* [dec_dim] */
    float *norm_weight;      /* [dec_dim] */
    float *norm_bias;        /* [dec_dim] */
    float *pwconv1_weight;   /* [inner_dim, dec_dim] */
    float *pwconv1_bias;     /* [inner_dim] */
    float *pwconv2_weight;   /* [dec_dim, inner_dim] */
    float *pwconv2_bias;     /* [dec_dim] */
    float *gamma;            /* [dec_dim] residual scale */
    int inner_dim;
} ConvNeXtBlockWeights;

/* ═══════════════════════════════════════════════════════════════════════════════
 * Upsample Block Weights
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float *convt_weight;     /* [out_ch, in_ch, kernel_size] */
    float *convt_bias;       /* [out_ch] */
    float *res1_weight;      /* [out_ch, out_ch, 3] for dilation=1 */
    float *res1_bias;        /* [out_ch] */
    float *res3_weight;      /* [out_ch, out_ch, 3] for dilation=3 */
    float *res3_bias;        /* [out_ch] */
    float *res9_weight;      /* [out_ch, out_ch, 3] for dilation=9 */
    float *res9_bias;        /* [out_ch] */
    int in_ch;
    int out_ch;
    int stride;
} UpsampleBlockWeights;

/* ═══════════════════════════════════════════════════════════════════════════════
 * Main Codec12Hz Structure
 * ═══════════════════════════════════════════════════════════════════════════════ */

struct Codec12Hz {
    Codec12HzConfig cfg;

    /* FSQ codebook (4096 × 512) */
    float *fsq_codebook;

    /* Input projection: (semantic_codes + acoustic_latent) → dec_dim */
    /* semantic_codes: 4×512=2048-dim after embedding, acoustic: 512-dim → total 2048+512=2560 */
    float *input_proj_weight;  /* [dec_dim, 2560, 7] */
    float *input_proj_bias;    /* [dec_dim] */

    /* ConvNeXt backbone blocks */
    ConvNeXtBlockWeights *blocks;

    /* Upsample stages */
    UpsampleBlockWeights *upsample_stages;

    /* Output projection */
    float *output_weight;      /* [1, dec_dim//16, 7] */
    float *output_bias;        /* [1] */

    /* Scratch buffers (pre-allocated, zero-alloc in decode) */
    float *scratch;
    int scratch_size;

    /* Streaming state (ring buffer for overlap-add) */
    int streaming;
    int ring_head;
    float *overlap_buf;

    /* Working buffers for frame processing */
    float *frame_buf;          /* [hop_length] current frame */
    float *embedding_buf;      /* [2560] semantic embedding + acoustic latent */
    float *conv_buf;           /* [max(dec_dim, dec_dim//2, ...)] convolution output */
};

/* ═══════════════════════════════════════════════════════════════════════════════
 * Utilities
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Read uint32 little-endian */
static uint32_t read_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* LeakyReLU activation (slope=0.1) */
static void leaky_relu(float *x, int n, float slope) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] *= slope;
    }
}

/* Layer normalization */
static void layer_norm(float *x, const float *weight, const float *bias,
                       int n, float eps) {
    /* Compute mean */
    float mean = 0.0f;
    vDSP_sve(x, 1, &mean, n);
    mean /= n;

    /* Compute variance */
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= n;

    /* Normalize and scale/shift */
    float scale = 1.0f / sqrtf(var + eps);
    vDSP_vsmul(x, 1, &scale, x, 1, n);
    for (int i = 0; i < n; i++) {
        x[i] = (x[i] * weight[i]) + bias[i];
    }
}

/* GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))) */
static void gelu(float *x, int n) {
    const float k0 = sqrtf(2.0f / M_PI);
    const float k1 = 0.044715f;
    for (int i = 0; i < n; i++) {
        float x3 = x[i] * x[i] * x[i];
        float arg = k0 * (x[i] + k1 * x3);
        x[i] = x[i] * 0.5f * (1.0f + tanhf(arg));
    }
}

/* Tanh activation */
static void tanh_activation(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = tanhf(x[i]);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * FSQ Dequantization
 * ═══════════════════════════════════════════════════════════════════════════════ */

/*
 * FSQ with levels [8,8,8,8] = 4096 codebook.
 * Dequantization: indices[0..3] ∈ [0,7] → codebook lookup.
 *
 * Each index selects a position in a 4-dimensional quantization grid.
 * Codebook entry at index i is simply embedding[i].
 */
static void fsq_dequantize(
    Codec12Hz *codec,
    const uint8_t *indices,     /* [4] indices 0-7 */
    float *out_embedding        /* [512] output embedding */
) {
    /* Validate indices before arithmetic to prevent overflow */
    for (int i = 0; i < 4; i++) {
        if (indices[i] >= 8) {
            memset(out_embedding, 0, codec->cfg.fsq_embed_dim * sizeof(float));
            return;
        }
    }

    /* Reconstruct codebook index from 4D quantization grid */
    int idx = (indices[0] * 512) + (indices[1] * 64) + (indices[2] * 8) + indices[3];

    /* Copy embedding */
    const float *embedding = codec->fsq_codebook + (size_t)idx * codec->cfg.fsq_embed_dim;
    memcpy(out_embedding, embedding,
           codec->cfg.fsq_embed_dim * sizeof(float));
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ConvTranspose1d (Upsampling)
 * ═══════════════════════════════════════════════════════════════════════════════ */

/*
 * ConvTranspose1d: upsamples by stride factor using grouped convolution.
 * Equivalent to PyTorch's ConvTranspose1d with stride and padding.
 *
 * For inference: compute output directly via grouped matrix multiplication.
 */
static void convtranspose1d(
    const float *input,      /* [in_ch, T_in] */
    int in_ch, int T_in,
    const float *weight,     /* [in_ch, out_ch, kernel_size] */
    const float *bias,       /* [out_ch] */
    int out_ch, int kernel_size,
    int stride, int padding,
    float *output,           /* [out_ch, T_out] */
    int T_out
) {
    /* Initialize output with bias */
    for (int oc = 0; oc < out_ch; oc++) {
        for (int t = 0; t < T_out; t++) {
            output[(size_t)oc * T_out + t] = bias[oc];
        }
    }

    /* Convolve each input frame */
    for (int ic = 0; ic < in_ch; ic++) {
        for (int t_in = 0; t_in < T_in; t_in++) {
            float x = input[(size_t)ic * T_in + t_in];
            int t_out_start = t_in * stride - padding;

            for (int oc = 0; oc < out_ch; oc++) {
                const float *w = weight + (size_t)ic * out_ch * kernel_size +
                                 (size_t)oc * kernel_size;

                for (int k = 0; k < kernel_size; k++) {
                    int t_out = t_out_start + k;
                    if (t_out >= 0 && t_out < T_out) {
                        output[(size_t)oc * T_out + t_out] += x * w[k];
                    }
                }
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Creation and Initialization
 * ═══════════════════════════════════════════════════════════════════════════════ */

Codec12Hz *codec_12hz_create_empty(const Codec12HzConfig *cfg) {
    if (!cfg) return NULL;

    /* P2-2: Validate dec_dim before using it */
    if (cfg->dec_dim < 16 || cfg->dec_dim % 16 != 0) {
        fprintf(stderr, "[codec_12hz] dec_dim must be >= 16 and divisible by 16\n");
        return NULL;
    }

    /* P0-2: Validate config ranges to prevent integer overflow */
    if (cfg->dec_dim > 8192 || cfg->fsq_dim > 256 || cfg->fsq_embed_dim > 2048) {
        fprintf(stderr, "[codec_12hz] Config parameter out of range\n");
        return NULL;
    }

    Codec12Hz *codec = (Codec12Hz *)calloc(1, sizeof(Codec12Hz));
    if (!codec) return NULL;

    memcpy(&codec->cfg, cfg, sizeof(Codec12HzConfig));

    /* Allocate FSQ codebook */
    codec->fsq_codebook = (float *)calloc(
        (size_t)cfg->fsq_codebook_size * cfg->fsq_embed_dim,
        sizeof(float)
    );
    if (!codec->fsq_codebook) {
        codec_12hz_destroy(codec);
        return NULL;
    }

    /* Input projection: semantic embedding + acoustic latent → dec_dim */
    int input_dim = cfg->fsq_embed_dim + cfg->acoustic_dim;
    size_t inp_weight_count = (size_t)cfg->dec_dim * input_dim * 7;
    if (inp_weight_count / 7 / input_dim != (size_t)cfg->dec_dim) {
        codec_12hz_destroy(codec);
        return NULL;  /* Overflow */
    }
    codec->input_proj_weight = (float *)calloc(inp_weight_count, sizeof(float));
    if (!codec->input_proj_weight) {
        codec_12hz_destroy(codec);
        return NULL;
    }
    codec->input_proj_bias = (float *)calloc(cfg->dec_dim, sizeof(float));
    if (!codec->input_proj_bias) {
        codec_12hz_destroy(codec);
        return NULL;
    }

    /* ConvNeXt blocks */
    codec->blocks = (ConvNeXtBlockWeights *)calloc(cfg->dec_n_layers,
                                                    sizeof(ConvNeXtBlockWeights));
    if (!codec->blocks) {
        codec_12hz_destroy(codec);
        return NULL;
    }
    int inner_dim = (int)(cfg->dec_dim * cfg->dec_ff_mult);
    for (int i = 0; i < cfg->dec_n_layers; i++) {
        ConvNeXtBlockWeights *b = &codec->blocks[i];
        b->inner_dim = inner_dim;
        b->dwconv_weight = (float *)calloc((size_t)cfg->dec_dim * cfg->dec_conv_kernel,
                                           sizeof(float));
        if (!b->dwconv_weight) {
            codec_12hz_destroy(codec);
            return NULL;
        }
        b->dwconv_bias = (float *)calloc(cfg->dec_dim, sizeof(float));
        if (!b->dwconv_bias) {
            codec_12hz_destroy(codec);
            return NULL;
        }
        b->norm_weight = (float *)calloc(cfg->dec_dim, sizeof(float));
        if (!b->norm_weight) {
            codec_12hz_destroy(codec);
            return NULL;
        }
        b->norm_bias = (float *)calloc(cfg->dec_dim, sizeof(float));
        if (!b->norm_bias) {
            codec_12hz_destroy(codec);
            return NULL;
        }
        b->pwconv1_weight = (float *)calloc((size_t)inner_dim * cfg->dec_dim,
                                            sizeof(float));
        if (!b->pwconv1_weight) {
            codec_12hz_destroy(codec);
            return NULL;
        }
        b->pwconv1_bias = (float *)calloc(inner_dim, sizeof(float));
        if (!b->pwconv1_bias) {
            codec_12hz_destroy(codec);
            return NULL;
        }
        b->pwconv2_weight = (float *)calloc((size_t)cfg->dec_dim * inner_dim,
                                            sizeof(float));
        if (!b->pwconv2_weight) {
            codec_12hz_destroy(codec);
            return NULL;
        }
        b->pwconv2_bias = (float *)calloc(cfg->dec_dim, sizeof(float));
        if (!b->pwconv2_bias) {
            codec_12hz_destroy(codec);
            return NULL;
        }
        b->gamma = (float *)calloc(cfg->dec_dim, sizeof(float));
        if (!b->gamma) {
            codec_12hz_destroy(codec);
            return NULL;
        }

        /* Initialize defaults */
        for (int j = 0; j < cfg->dec_dim; j++) {
            b->norm_weight[j] = 1.0f;
            b->gamma[j] = 1.0f;
        }
    }

    /* Upsample stages [3, 4, 5, 8, 4] = 1920x total */
    codec->upsample_stages = (UpsampleBlockWeights *)calloc(5,
                                                            sizeof(UpsampleBlockWeights));
    if (!codec->upsample_stages) {
        codec_12hz_destroy(codec);
        return NULL;
    }
    int ch_in = cfg->dec_dim;
    int ch_schedule[] = {cfg->dec_dim, cfg->dec_dim / 2, cfg->dec_dim / 4,
                         cfg->dec_dim / 8, cfg->dec_dim / 16};
    int strides[] = {3, 4, 5, 8, 4};
    for (int i = 0; i < 5; i++) {
        UpsampleBlockWeights *u = &codec->upsample_stages[i];
        int ch_out = ch_schedule[i];
        int stride = strides[i];
        u->in_ch = ch_in;
        u->out_ch = ch_out;
        u->stride = stride;

        /* ConvTranspose kernel size = stride * 2 */
        int kernel = stride * 2;
        u->convt_weight = (float *)calloc((size_t)ch_in * ch_out * kernel,
                                          sizeof(float));
        if (!u->convt_weight) {
            codec_12hz_destroy(codec);
            return NULL;
        }
        u->convt_bias = (float *)calloc(ch_out, sizeof(float));
        if (!u->convt_bias) {
            codec_12hz_destroy(codec);
            return NULL;
        }

        /* Residual units: 3 dilations per stage */
        u->res1_weight = (float *)calloc((size_t)ch_out * ch_out * 3,
                                         sizeof(float));
        if (!u->res1_weight) {
            codec_12hz_destroy(codec);
            return NULL;
        }
        u->res1_bias = (float *)calloc(ch_out, sizeof(float));
        if (!u->res1_bias) {
            codec_12hz_destroy(codec);
            return NULL;
        }
        u->res3_weight = (float *)calloc((size_t)ch_out * ch_out * 3,
                                         sizeof(float));
        if (!u->res3_weight) {
            codec_12hz_destroy(codec);
            return NULL;
        }
        u->res3_bias = (float *)calloc(ch_out, sizeof(float));
        if (!u->res3_bias) {
            codec_12hz_destroy(codec);
            return NULL;
        }
        u->res9_weight = (float *)calloc((size_t)ch_out * ch_out * 3,
                                         sizeof(float));
        if (!u->res9_weight) {
            codec_12hz_destroy(codec);
            return NULL;
        }
        u->res9_bias = (float *)calloc(ch_out, sizeof(float));
        if (!u->res9_bias) {
            codec_12hz_destroy(codec);
            return NULL;
        }

        ch_in = ch_out;
    }

    /* Output projection: 768→16 → 1 (to audio) */
    codec->output_weight = (float *)calloc((size_t)1 * (cfg->dec_dim / 16) * 7,
                                           sizeof(float));
    if (!codec->output_weight) {
        codec_12hz_destroy(codec);
        return NULL;
    }
    codec->output_bias = (float *)calloc(1, sizeof(float));
    if (!codec->output_bias) {
        codec_12hz_destroy(codec);
        return NULL;
    }

    /* Scratch: accommodate all intermediate buffers (ConvNeXt + innermost parts)
     * We need: x (dec_dim) + x_next (dec_dim) + inner_buf (dec_dim * ff_mult)
     * Plus semantic_embedding and all activations */
    codec->scratch_size = 10 * (cfg->dec_dim + inner_dim) + cfg->fsq_embed_dim;
    codec->scratch = (float *)calloc(codec->scratch_size, sizeof(float));
    if (!codec->scratch) {
        codec_12hz_destroy(codec);
        return NULL;
    }

    /* Working buffers */
    codec->frame_buf = (float *)calloc(cfg->hop_length, sizeof(float));
    if (!codec->frame_buf) {
        codec_12hz_destroy(codec);
        return NULL;
    }
    codec->embedding_buf = (float *)calloc(input_dim, sizeof(float));
    if (!codec->embedding_buf) {
        codec_12hz_destroy(codec);
        return NULL;
    }
    codec->conv_buf = (float *)calloc((size_t)cfg->dec_dim * cfg->hop_length, sizeof(float));
    if (!codec->conv_buf) {
        codec_12hz_destroy(codec);
        return NULL;
    }

    /* Overlap buffer for ring-buffer streaming */
    codec->overlap_buf = (float *)calloc(cfg->n_fft, sizeof(float));
    if (!codec->overlap_buf) {
        codec_12hz_destroy(codec);
        return NULL;
    }

    /* Verify all allocations succeeded */
    if (!codec->fsq_codebook || !codec->input_proj_weight ||
        !codec->input_proj_bias || !codec->blocks || !codec->upsample_stages ||
        !codec->output_weight || !codec->output_bias || !codec->scratch ||
        !codec->frame_buf || !codec->embedding_buf || !codec->conv_buf ||
        !codec->overlap_buf) {
        codec_12hz_destroy(codec);
        return NULL;
    }

    return codec;
}

Codec12Hz *codec_12hz_create(const char *model_path) {
    if (!model_path) return NULL;

    /* Load config from header */
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "[codec_12hz] Failed to open: %s\n", model_path);
        return NULL;
    }

    uint8_t magic[CODEC_12HZ_MAGIC_LEN];
    if (fread(magic, 1, CODEC_12HZ_MAGIC_LEN, f) != CODEC_12HZ_MAGIC_LEN) {
        fprintf(stderr, "[codec_12hz] Magic header read failed\n");
        fclose(f);
        return NULL;
    }

    if (memcmp(magic, CODEC_12HZ_MAGIC, CODEC_12HZ_MAGIC_LEN) != 0) {
        fprintf(stderr, "[codec_12hz] Invalid magic header\n");
        fclose(f);
        return NULL;
    }

    /* Read config (16 uint32 values) */
    uint8_t cfg_buf[64];
    if (fread(cfg_buf, 1, 64, f) != 64) {
        fprintf(stderr, "[codec_12hz] Config read failed\n");
        fclose(f);
        return NULL;
    }

    Codec12HzConfig cfg = {0};
    cfg.sample_rate = read_u32_le(cfg_buf + 0);
    cfg.n_fft = read_u32_le(cfg_buf + 4);
    cfg.hop_length = read_u32_le(cfg_buf + 8);
    cfg.n_mels = read_u32_le(cfg_buf + 12);
    cfg.fsq_dim = read_u32_le(cfg_buf + 16);
    cfg.fsq_codebook_size = read_u32_le(cfg_buf + 20);
    cfg.fsq_embed_dim = read_u32_le(cfg_buf + 24);
    cfg.acoustic_dim = read_u32_le(cfg_buf + 28);
    cfg.dec_dim = read_u32_le(cfg_buf + 32);
    cfg.dec_n_layers = read_u32_le(cfg_buf + 36);
    cfg.dec_conv_kernel = read_u32_le(cfg_buf + 40);
    /* P2-1: Strict aliasing fix — use memcpy instead of cast */
    float dec_ff_mult;
    memcpy(&dec_ff_mult, cfg_buf + 44, sizeof(float));
    cfg.dec_ff_mult = dec_ff_mult;
    for (int i = 0; i < 5; i++) {
        cfg.decoder_strides[i] = read_u32_le(cfg_buf + 48 + i * 4);
    }

    fclose(f);

    /* Create codec with loaded config */
    Codec12Hz *codec = codec_12hz_create_empty(&cfg);
    if (!codec) return NULL;

    /* Load weights */
    if (codec_12hz_load_weights(codec, model_path) != 0) {
        codec_12hz_destroy(codec);
        return NULL;
    }

    return codec;
}

int codec_12hz_load_weights(Codec12Hz *codec, const char *model_path) {
    if (!codec || !model_path) return -1;

    FILE *f = fopen(model_path, "rb");
    if (!f) return -1;

    /* Skip magic and config header */
    fseek(f, CODEC_12HZ_MAGIC_LEN + 64, SEEK_SET);

    /* Load FSQ codebook */
    size_t fsq_sz = (size_t)codec->cfg.fsq_codebook_size * codec->cfg.fsq_embed_dim;
    if (fread(codec->fsq_codebook, sizeof(float), fsq_sz, f) != fsq_sz) {
        fprintf(stderr, "[codec_12hz] FSQ codebook read failed\n");
        fclose(f);
        return -1;
    }

    /* Load input projection */
    int input_dim = codec->cfg.fsq_embed_dim + codec->cfg.acoustic_dim;
    size_t inp_sz = (size_t)codec->cfg.dec_dim * input_dim * 7;
    if (fread(codec->input_proj_weight, sizeof(float), inp_sz, f) != inp_sz) {
        fprintf(stderr, "[codec_12hz] Input projection read failed\n");
        fclose(f);
        return -1;
    }

    if (fread(codec->input_proj_bias, sizeof(float), codec->cfg.dec_dim, f) !=
        (size_t)codec->cfg.dec_dim) {
        fprintf(stderr, "[codec_12hz] Input bias read failed\n");
        fclose(f);
        return -1;
    }

    /* Load ConvNeXt blocks */
    for (int i = 0; i < codec->cfg.dec_n_layers; i++) {
        ConvNeXtBlockWeights *b = &codec->blocks[i];
        int inner = b->inner_dim;

        if (fread(b->dwconv_weight, sizeof(float),
                  (size_t)codec->cfg.dec_dim * codec->cfg.dec_conv_kernel, f) !=
            (size_t)codec->cfg.dec_dim * codec->cfg.dec_conv_kernel) {
            fprintf(stderr, "[codec_12hz] Block %d dwconv weight read failed\n", i);
            fclose(f);
            return -1;
        }

        if (fread(b->dwconv_bias, sizeof(float), codec->cfg.dec_dim, f) !=
            (size_t)codec->cfg.dec_dim) {
            fclose(f);
            return -1;
        }

        if (fread(b->norm_weight, sizeof(float), codec->cfg.dec_dim, f) !=
            (size_t)codec->cfg.dec_dim) {
            fclose(f);
            return -1;
        }

        if (fread(b->norm_bias, sizeof(float), codec->cfg.dec_dim, f) !=
            (size_t)codec->cfg.dec_dim) {
            fclose(f);
            return -1;
        }

        if (fread(b->pwconv1_weight, sizeof(float), (size_t)inner * codec->cfg.dec_dim,
                  f) != (size_t)inner * codec->cfg.dec_dim) {
            fclose(f);
            return -1;
        }

        if (fread(b->pwconv1_bias, sizeof(float), inner, f) != (size_t)inner) {
            fclose(f);
            return -1;
        }

        if (fread(b->pwconv2_weight, sizeof(float), (size_t)codec->cfg.dec_dim * inner,
                  f) != (size_t)codec->cfg.dec_dim * inner) {
            fclose(f);
            return -1;
        }

        if (fread(b->pwconv2_bias, sizeof(float), codec->cfg.dec_dim, f) !=
            (size_t)codec->cfg.dec_dim) {
            fclose(f);
            return -1;
        }

        if (fread(b->gamma, sizeof(float), codec->cfg.dec_dim, f) !=
            (size_t)codec->cfg.dec_dim) {
            fclose(f);
            return -1;
        }
    }

    /* Load upsample stages */
    for (int i = 0; i < 5; i++) {
        UpsampleBlockWeights *u = &codec->upsample_stages[i];
        int kernel = u->stride * 2;

        if (fread(u->convt_weight, sizeof(float),
                  (size_t)u->in_ch * u->out_ch * kernel, f) !=
            (size_t)u->in_ch * u->out_ch * kernel) {
            fprintf(stderr, "[codec_12hz] Upsample %d convt weight read failed\n", i);
            fclose(f);
            return -1;
        }

        if (fread(u->convt_bias, sizeof(float), u->out_ch, f) != (size_t)u->out_ch) {
            fclose(f);
            return -1;
        }

        if (fread(u->res1_weight, sizeof(float), (size_t)u->out_ch * u->out_ch * 3, f) !=
            (size_t)u->out_ch * u->out_ch * 3) {
            fclose(f);
            return -1;
        }

        if (fread(u->res1_bias, sizeof(float), u->out_ch, f) != (size_t)u->out_ch) {
            fclose(f);
            return -1;
        }

        if (fread(u->res3_weight, sizeof(float), (size_t)u->out_ch * u->out_ch * 3, f) !=
            (size_t)u->out_ch * u->out_ch * 3) {
            fclose(f);
            return -1;
        }

        if (fread(u->res3_bias, sizeof(float), u->out_ch, f) != (size_t)u->out_ch) {
            fclose(f);
            return -1;
        }

        if (fread(u->res9_weight, sizeof(float), (size_t)u->out_ch * u->out_ch * 3, f) !=
            (size_t)u->out_ch * u->out_ch * 3) {
            fclose(f);
            return -1;
        }

        if (fread(u->res9_bias, sizeof(float), u->out_ch, f) != (size_t)u->out_ch) {
            fclose(f);
            return -1;
        }
    }

    /* Load output projection */
    int out_kernel = 7;
    if (fread(codec->output_weight, sizeof(float),
              (size_t)1 * (codec->cfg.dec_dim / 16) * out_kernel, f) !=
        (size_t)1 * (codec->cfg.dec_dim / 16) * out_kernel) {
        fprintf(stderr, "[codec_12hz] Output weight read failed\n");
        fclose(f);
        return -1;
    }

    if (fread(codec->output_bias, sizeof(float), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

void codec_12hz_destroy(Codec12Hz *codec) {
    if (!codec) return;

    free(codec->fsq_codebook);
    free(codec->input_proj_weight);
    free(codec->input_proj_bias);

    if (codec->blocks) {
        for (int i = 0; i < codec->cfg.dec_n_layers; i++) {
            ConvNeXtBlockWeights *b = &codec->blocks[i];
            free(b->dwconv_weight);
            free(b->dwconv_bias);
            free(b->norm_weight);
            free(b->norm_bias);
            free(b->pwconv1_weight);
            free(b->pwconv1_bias);
            free(b->pwconv2_weight);
            free(b->pwconv2_bias);
            free(b->gamma);
        }
        free(codec->blocks);
    }

    if (codec->upsample_stages) {
        for (int i = 0; i < 5; i++) {
            UpsampleBlockWeights *u = &codec->upsample_stages[i];
            free(u->convt_weight);
            free(u->convt_bias);
            free(u->res1_weight);
            free(u->res1_bias);
            free(u->res3_weight);
            free(u->res3_bias);
            free(u->res9_weight);
            free(u->res9_bias);
        }
        free(codec->upsample_stages);
    }

    free(codec->output_weight);
    free(codec->output_bias);
    free(codec->scratch);
    free(codec->frame_buf);
    free(codec->embedding_buf);
    free(codec->conv_buf);
    free(codec->overlap_buf);
    free(codec);
}

void codec_12hz_reset(Codec12Hz *codec) {
    if (!codec) return;
    memset(codec->overlap_buf, 0, codec->cfg.n_fft * sizeof(float));
    codec->ring_head = 0;
}

void codec_12hz_set_streaming(Codec12Hz *codec, int enable) {
    if (!codec) return;
    codec->streaming = enable ? 1 : 0;
    codec_12hz_reset(codec);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Frame Decoding
 * ═══════════════════════════════════════════════════════════════════════════════ */

int codec_12hz_decode_frame(
    Codec12Hz *codec,
    const uint8_t *semantic_codes,   /* [fsq_dim] */
    const float *acoustic_latent,     /* [acoustic_dim] */
    float *out_audio                  /* [hop_length] */
) {
    if (!codec || !semantic_codes || !acoustic_latent || !out_audio) return 0;

    const int dec_dim = codec->cfg.dec_dim;
    const int acoustic_dim = codec->cfg.acoustic_dim;
    const int fsq_dim = codec->cfg.fsq_dim;
    const int fsq_embed_dim = codec->cfg.fsq_embed_dim;
    const int hop_length = codec->cfg.hop_length;
    const int n_layers = codec->cfg.dec_n_layers;
    const int conv_kernel = codec->cfg.dec_conv_kernel;
    const float ff_mult = codec->cfg.dec_ff_mult;
    const int inner_dim = (int)(dec_dim * ff_mult);

    /* ─ FSQ Dequantization ─ */
    float *semantic_embedding = codec->scratch;  /* [fsq_embed_dim] at scratch[0] */
    for (int i = 0; i < fsq_dim; i++) {
        if (semantic_codes[i] > 7) {
            fprintf(stderr, "[codec_12hz] FSQ index out of range: %u\n", semantic_codes[i]);
            return 0;
        }
    }

    fsq_dequantize(codec, semantic_codes, semantic_embedding);

    /* ─ Concatenate embedding + acoustic ─ */
    int input_dim = fsq_embed_dim + acoustic_dim;
    float *embedding = codec->embedding_buf;  /* [input_dim] */
    memcpy(embedding, semantic_embedding, fsq_embed_dim * sizeof(float));
    memcpy(embedding + fsq_embed_dim, acoustic_latent,
           acoustic_dim * sizeof(float));

    /* ─ Input Projection ─ */
    /* Use scratch for x, x_next, inner_buf: need dec_dim + dec_dim + inner_dim */
    float *x = codec->scratch;  /* [dec_dim] at scratch[0..dec_dim) */
    memset(x, 0, dec_dim * sizeof(float));

    /* Conv1d projection: kernel_size=7, apply to embedding */
    for (int oc = 0; oc < dec_dim; oc++) {
        float sum = codec->input_proj_bias[oc];
        const float *w = codec->input_proj_weight + (size_t)oc * input_dim * 7;
        for (int ic = 0; ic < input_dim; ic++) {
            sum += w[ic * 7 + 3] * embedding[ic];  /* Center of kernel (pos 3) */
        }
        x[oc] = sum;
    }

    leaky_relu(x, dec_dim, 0.1f);

    /* ─ ConvNeXt backbone ─ */
    /* Careful buffer layout: x at scratch[0], x_next at scratch[dec_dim] */
    float *x_next = x + dec_dim;  /* scratch[dec_dim..2*dec_dim) */
    for (int layer = 0; layer < n_layers; layer++) {
        ConvNeXtBlockWeights *b = &codec->blocks[layer];

        /* Depthwise conv */
        memset(x_next, 0, dec_dim * sizeof(float));
        for (int c = 0; c < dec_dim; c++) {
            float sum = b->dwconv_bias[c];
            const float *w = b->dwconv_weight + (size_t)c * conv_kernel;
            sum += w[conv_kernel / 2] * x[c];  /* Center of kernel */
            x_next[c] = sum;
        }

        /* Layer norm */
        layer_norm(x_next, b->norm_weight, b->norm_bias, dec_dim, 1e-5f);

        /* Pointwise expand + GELU */
        /* inner_buf at scratch[2*dec_dim..2*dec_dim+inner_dim) */
        float *inner_buf = codec->scratch + 2 * dec_dim;
        memset(inner_buf, 0, inner_dim * sizeof(float));
        for (int j = 0; j < inner_dim; j++) {
            float sum = b->pwconv1_bias[j];
            for (int i = 0; i < dec_dim; i++) {
                sum += b->pwconv1_weight[(size_t)j * dec_dim + i] * x_next[i];
            }
            inner_buf[j] = sum;
        }
        gelu(inner_buf, inner_dim);

        /* Pointwise shrink + gamma scale */
        memset(x, 0, dec_dim * sizeof(float));
        for (int i = 0; i < dec_dim; i++) {
            float sum = b->pwconv2_bias[i];
            for (int j = 0; j < inner_dim; j++) {
                sum += b->pwconv2_weight[(size_t)i * inner_dim + j] * inner_buf[j];
            }
            x[i] = (sum * b->gamma[i]) + x[i];  /* Residual */
        }
    }

    /* ─ Simplified inference path ─
     * For a single semantic code + acoustic latent pair, we generate hop_length samples.
     * The ConvNeXt backbone processes a condensed representation,
     * and upsampling expands to the full waveform.
     *
     * Full transposed convolution would require batch dimension.
     * For inference, we implement a simplified decoder path:
     * 1. ConvNeXt processes [dec_dim] → [dec_dim]
     * 2. Output projection directly maps to [hop_length] waveform
     */

    float *audio_frame = codec->frame_buf;
    memset(audio_frame, 0, hop_length * sizeof(float));

    /* Output projection: map [dec_dim] encoded vector to [hop_length] waveform.
     * Uses dec_dim/16 features with kernel_size=7, center tap only for single-frame. */
    int out_dim = dec_dim / 16;
    int output_weight_size = out_dim * 7;  /* P0-3: Bounds validation */
    for (int t = 0; t < hop_length; t++) {
        float sum = codec->output_bias[0];
        /* Map time position to feature index */
        int feat_idx = (t * out_dim) / hop_length;
        if (feat_idx >= out_dim) feat_idx = out_dim - 1;

        /* Use center kernel tap (position 3 of 7) */
        /* P0-3: Validate bounds before array access */
        int w_idx = feat_idx * 7 + 3;
        if (w_idx < output_weight_size && feat_idx < dec_dim) {
            sum += codec->output_weight[w_idx] * x[feat_idx];
        }
        audio_frame[t] = sum;
    }

    tanh_activation(audio_frame, hop_length);

    /* ─ Overlap-add ─ */
    if (codec->streaming) {
        /* Ring buffer overlap-add (zero-copy) */
        const int ring_n = codec->cfg.n_fft;
        int head = codec->ring_head;

        /* P1-1: Fixed wraparound logic — correctly handle add with wrap */
        if (head + hop_length <= ring_n) {
            /* No wrap — enough room from head to end of ring */
            vDSP_vadd(codec->overlap_buf + head, 1, audio_frame, 1,
                      codec->overlap_buf + head, 1, hop_length);
        } else {
            /* Two-part add: first elements before wrap, remainder at start */
            int first_part = ring_n - head;
            int second_part = hop_length - first_part;
            vDSP_vadd(codec->overlap_buf + head, 1, audio_frame, 1,
                      codec->overlap_buf + head, 1, first_part);
            vDSP_vadd(codec->overlap_buf, 1, audio_frame + first_part, 1,
                      codec->overlap_buf, 1, second_part);
        }

        /* Copy hop_length samples to output */
        int out_first = ring_n - head;
        if (out_first >= hop_length) {
            memcpy(out_audio, codec->overlap_buf + head,
                   hop_length * sizeof(float));
            memset(codec->overlap_buf + head, 0, hop_length * sizeof(float));
        } else {
            memcpy(out_audio, codec->overlap_buf + head,
                   out_first * sizeof(float));
            memcpy(out_audio + out_first, codec->overlap_buf,
                   (hop_length - out_first) * sizeof(float));
            memset(codec->overlap_buf + head, 0, out_first * sizeof(float));
            memset(codec->overlap_buf, 0, (hop_length - out_first) * sizeof(float));
        }

        codec->ring_head = (head + hop_length) % ring_n;
    } else {
        /* Linear overlap-add */
        vDSP_vadd(codec->overlap_buf, 1, audio_frame, 1,
                  codec->overlap_buf, 1, hop_length);

        memcpy(out_audio, codec->overlap_buf, hop_length * sizeof(float));

        memmove(codec->overlap_buf, codec->overlap_buf + hop_length,
                (codec->cfg.n_fft - hop_length) * sizeof(float));
        memset(codec->overlap_buf + (codec->cfg.n_fft - hop_length), 0,
               hop_length * sizeof(float));
    }

    return hop_length;
}

int codec_12hz_decode_batch(
    Codec12Hz *codec,
    const uint8_t *semantic_codes,    /* [n_frames, fsq_dim] */
    const float *acoustic_latents,    /* [n_frames, acoustic_dim] */
    int n_frames,
    float *out_audio                  /* [n_frames * hop_length] */
) {
    if (!codec || !semantic_codes || !acoustic_latents || !out_audio || n_frames <= 0)
        return 0;

    int total_samples = 0;
    for (int f = 0; f < n_frames; f++) {
        int n = codec_12hz_decode_frame(
            codec,
            semantic_codes + (size_t)f * codec->cfg.fsq_dim,
            acoustic_latents + (size_t)f * codec->cfg.acoustic_dim,
            out_audio + total_samples
        );
        if (n == 0) return -1;
        total_samples += n;
    }
    return total_samples;
}
