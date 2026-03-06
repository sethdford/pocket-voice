/**
 * bnns_convnext_decoder.c — ANE-accelerated ConvNeXt decoder for Sonata TTS.
 *
 * Implements the ConvNeXt decoder using Apple's BNNS framework, targeting
 * the Neural Engine for inference while the GPU handles the flow network.
 *
 * Architecture per block:
 *   Input → DW Conv1D → LayerNorm → PW Conv1 (expand) → GELU → PW Conv2 → Gamma scale → Residual add
 *
 * Final heads: decoder hidden → magnitude (exp) + instantaneous frequency
 */

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif

#include <Accelerate/Accelerate.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "bnns_convnext_decoder.h"
#include "cJSON.h"

typedef struct ConvNeXtBlockWeights {
    float *dwconv_weight;   /* [dec_dim, 1, kernel_size] */
    float *dwconv_bias;     /* [dec_dim] */
    float *norm_weight;     /* [dec_dim] */
    float *norm_bias;       /* [dec_dim] */
    float *pwconv1_weight;  /* [inner_dim, dec_dim] */
    float *pwconv1_bias;    /* [inner_dim] */
    float *pwconv2_weight;  /* [dec_dim, inner_dim] */
    float *pwconv2_bias;    /* [dec_dim] */
    float *gamma;           /* [dec_dim] */
    int inner_dim;
} ConvNeXtBlockWeights;

#define BNNS_CONVNEXT_MAX_ARGS 8

struct BNNSConvNeXtDecoder {
    int n_layers;
    int dec_dim;
    int conv_kernel;
    float ff_mult;
    int input_dim;
    int n_fft;
    int n_bins;

    float *input_proj_weight;  /* [dec_dim, input_dim, 7] */
    float *input_proj_bias;    /* [dec_dim] */

    ConvNeXtBlockWeights *blocks;

    float *mag_proj_weight;    /* [n_bins, dec_dim] */
    float *mag_proj_bias;      /* [n_bins] */
    float *phase_proj_weight;  /* [n_bins, dec_dim] */
    float *phase_proj_bias;    /* [n_bins] */

    float *scratch;
    int scratch_size;

    /* BNNS Graph (ANE) path — macOS 15+ */
    bnns_graph_t graph;
    bnns_graph_context_t ctx;
    size_t n_args;
    size_t input_semantic_pos;
    size_t input_acoustic_pos;
    size_t output_mag_pos;
    size_t output_phase_pos;
    size_t workspace_size;
    char *workspace;
    int use_ane;
};

BNNSConvNeXtDecoder *bnns_convnext_create(int n_layers, int dec_dim,
                                            int conv_kernel, float ff_mult,
                                            int input_dim, int n_fft) {
    BNNSConvNeXtDecoder *dec = (BNNSConvNeXtDecoder *)calloc(1, sizeof(BNNSConvNeXtDecoder));
    if (!dec) return NULL;

    dec->n_layers = n_layers;
    dec->dec_dim = dec_dim;
    dec->conv_kernel = conv_kernel;
    dec->ff_mult = ff_mult;
    dec->input_dim = input_dim;
    dec->n_fft = n_fft;
    dec->n_bins = n_fft / 2 + 1;

    dec->input_proj_weight = (float *)calloc(dec_dim * input_dim * 7, sizeof(float));
    dec->input_proj_bias = (float *)calloc(dec_dim, sizeof(float));

    dec->blocks = (ConvNeXtBlockWeights *)calloc(n_layers, sizeof(ConvNeXtBlockWeights));
    int inner = (int)(dec_dim * ff_mult);
    for (int i = 0; i < n_layers; i++) {
        ConvNeXtBlockWeights *b = &dec->blocks[i];
        b->inner_dim = inner;
        b->dwconv_weight = (float *)calloc(dec_dim * conv_kernel, sizeof(float));
        b->dwconv_bias = (float *)calloc(dec_dim, sizeof(float));
        b->norm_weight = (float *)calloc(dec_dim, sizeof(float));
        b->norm_bias = (float *)calloc(dec_dim, sizeof(float));
        b->pwconv1_weight = (float *)calloc(inner * dec_dim, sizeof(float));
        b->pwconv1_bias = (float *)calloc(inner, sizeof(float));
        b->pwconv2_weight = (float *)calloc(dec_dim * inner, sizeof(float));
        b->pwconv2_bias = (float *)calloc(dec_dim, sizeof(float));
        b->gamma = (float *)calloc(dec_dim, sizeof(float));
        for (int j = 0; j < dec_dim; j++) {
            b->norm_weight[j] = 1.0f;
            b->gamma[j] = 1.0f;
        }
    }

    dec->mag_proj_weight = (float *)calloc(dec->n_bins * dec_dim, sizeof(float));
    dec->mag_proj_bias = (float *)calloc(dec->n_bins, sizeof(float));
    dec->phase_proj_weight = (float *)calloc(dec->n_bins * dec_dim, sizeof(float));
    dec->phase_proj_bias = (float *)calloc(dec->n_bins, sizeof(float));

    /* Scratch: 2*n_frames*dim + n_frames*inner for block intermediates */
    dec->scratch_size = (2 * dec_dim + inner) * 4096;
    dec->scratch = (float *)calloc(dec->scratch_size, sizeof(float));

    fprintf(stderr, "[bnns_convnext] Created: %d layers, dim=%d, kernel=%d, n_fft=%d\n",
            n_layers, dec_dim, conv_kernel, n_fft);
    return dec;
}

/* Binary format magic (8 bytes) */
#define BNNS_DEC_MAGIC "BNNSDEC\x00"

/* Read uint64 little-endian from buffer */
static uint64_t read_u64_le(const uint8_t *p) {
    return (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

static int load_tensor_from_safetensors(cJSON *root, const char *key, void *dest,
                                         size_t n_floats, const uint8_t *data, size_t data_base) {
    cJSON *obj = cJSON_GetObjectItem(root, key);
    if (!obj || !cJSON_IsObject(obj)) return -1;
    cJSON *offsets = cJSON_GetObjectItem(obj, "data_offsets");
    if (!offsets || !cJSON_IsArray(offsets)) return -1;
    cJSON *o0 = cJSON_GetArrayItem(offsets, 0);
    cJSON *o1 = cJSON_GetArrayItem(offsets, 1);
    if (!o0 || !o1) return -1;
    size_t start = (size_t)o0->valuedouble;
    size_t end = (size_t)o1->valuedouble;
    if (end - start != n_floats * sizeof(float)) return -1;
    memcpy(dest, data + data_base + start, n_floats * sizeof(float));
    return 0;
}

static int load_weights_binary(BNNSConvNeXtDecoder *dec, FILE *f) {
    size_t n = 0;
    int dim = dec->dec_dim;
    int input_dim = dec->input_dim;
    int inner = (int)(dim * dec->ff_mult);
    int n_bins = dec->n_bins;
    int kernel = dec->conv_kernel;

    n += fread(dec->input_proj_weight, sizeof(float), (size_t)dim * input_dim * 7, f);
    n += fread(dec->input_proj_bias, sizeof(float), (size_t)dim, f);

    for (int i = 0; i < dec->n_layers; i++) {
        ConvNeXtBlockWeights *b = &dec->blocks[i];
        n += fread(b->dwconv_weight, sizeof(float), (size_t)dim * kernel, f);
        n += fread(b->dwconv_bias, sizeof(float), (size_t)dim, f);
        n += fread(b->norm_weight, sizeof(float), (size_t)dim, f);
        n += fread(b->norm_bias, sizeof(float), (size_t)dim, f);
        n += fread(b->pwconv1_weight, sizeof(float), (size_t)inner * dim, f);
        n += fread(b->pwconv1_bias, sizeof(float), (size_t)inner, f);
        n += fread(b->pwconv2_weight, sizeof(float), (size_t)dim * inner, f);
        n += fread(b->pwconv2_bias, sizeof(float), (size_t)dim, f);
        n += fread(b->gamma, sizeof(float), (size_t)dim, f);
    }

    n += fread(dec->mag_proj_weight, sizeof(float), (size_t)n_bins * dim, f);
    n += fread(dec->mag_proj_bias, sizeof(float), (size_t)n_bins, f);
    n += fread(dec->phase_proj_weight, sizeof(float), (size_t)n_bins * dim, f);
    n += fread(dec->phase_proj_bias, sizeof(float), (size_t)n_bins, f);

    size_t expected = (size_t)(dim * input_dim * 7 + dim +
         dec->n_layers * (dim * kernel + dim + dim + dim + inner * dim + inner + dim * inner + dim + dim) +
         n_bins * dim + n_bins + n_bins * dim + n_bins);
    if (n != expected) {
        fprintf(stderr, "[bnns_convnext] Binary weight size mismatch: got %zu, expected %zu\n", n, expected);
        return -1;
    }
    fprintf(stderr, "[bnns_convnext] Loaded binary weights: %zu floats\n", n);
    return 0;
}

int bnns_convnext_load_weights(BNNSConvNeXtDecoder *dec, const char *path) {
    if (!dec || !path) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[bnns_convnext] Cannot open weights: %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size < 8) {
        fclose(f);
        fprintf(stderr, "[bnns_convnext] File too small: %ld bytes\n", file_size);
        return -1;
    }

    uint8_t header[16];
    if (fread(header, 1, 16, f) != 16) {
        fclose(f);
        return -1;
    }
    fseek(f, 0, SEEK_SET);

    /* Check for BNNS binary format magic */
    if (memcmp(header, BNNS_DEC_MAGIC, 8) == 0) {
        fseek(f, 8, SEEK_SET);  /* Skip magic */
        int ret = load_weights_binary(dec, f);
        fclose(f);
        return ret;
    }

    /* Check for safetensors: 8-byte LE header length, then JSON starting with '{' */
    uint64_t header_len = read_u64_le(header);
    if (header_len > 0 && header_len < 50 * 1024 * 1024 && header[8] == '{') {
        /* Safetensors format: parse with cJSON */
        uint8_t *data = (uint8_t *)malloc((size_t)file_size);
        if (!data) { fclose(f); return -1; }
        fseek(f, 0, SEEK_SET);
        fread(data, 1, (size_t)file_size, f);
        fclose(f);

        char *json_str = (char *)malloc((size_t)header_len + 1);
        if (!json_str) { free(data); return -1; }
        memcpy(json_str, data + 8, (size_t)header_len);
        json_str[header_len] = '\0';

        cJSON *root = cJSON_ParseWithLength(json_str, (size_t)header_len);
        free(json_str);
        if (!root) {
            fprintf(stderr, "[bnns_convnext] Safetensors JSON parse failed, trying binary\n");
            free(data);
            f = fopen(path, "rb");
            if (!f) return -1;
            int ret = load_weights_binary(dec, f);
            fclose(f);
            return ret;
        }

        size_t data_base = (size_t)(8 + header_len);
        int n_blocks = dec->n_layers;
        int dim = dec->dec_dim;
        int input_dim = dec->input_dim;
        int inner = (int)(dim * dec->ff_mult);
        int n_bins = dec->n_bins;
        int kernel = dec->conv_kernel;

        int err = 0;
        err |= load_tensor_from_safetensors(root, "decoder.input_proj.weight", dec->input_proj_weight, (size_t)dim * input_dim * 7, data, data_base);
        err |= load_tensor_from_safetensors(root, "decoder.input_proj.bias", dec->input_proj_bias, (size_t)dim, data, data_base);
        for (int i = 0; i < n_blocks && !err; i++) {
            char keybuf[80];
            ConvNeXtBlockWeights *b = &dec->blocks[i];
            snprintf(keybuf, sizeof(keybuf), "decoder.backbone.%d.dwconv.weight", i);
            err |= load_tensor_from_safetensors(root, keybuf, b->dwconv_weight, (size_t)dim * kernel, data, data_base);
            snprintf(keybuf, sizeof(keybuf), "decoder.backbone.%d.dwconv.bias", i);
            err |= load_tensor_from_safetensors(root, keybuf, b->dwconv_bias, (size_t)dim, data, data_base);
            snprintf(keybuf, sizeof(keybuf), "decoder.backbone.%d.norm.weight", i);
            err |= load_tensor_from_safetensors(root, keybuf, b->norm_weight, (size_t)dim, data, data_base);
            snprintf(keybuf, sizeof(keybuf), "decoder.backbone.%d.norm.bias", i);
            err |= load_tensor_from_safetensors(root, keybuf, b->norm_bias, (size_t)dim, data, data_base);
            snprintf(keybuf, sizeof(keybuf), "decoder.backbone.%d.pwconv1.weight", i);
            err |= load_tensor_from_safetensors(root, keybuf, b->pwconv1_weight, (size_t)inner * dim, data, data_base);
            snprintf(keybuf, sizeof(keybuf), "decoder.backbone.%d.pwconv1.bias", i);
            err |= load_tensor_from_safetensors(root, keybuf, b->pwconv1_bias, (size_t)inner, data, data_base);
            snprintf(keybuf, sizeof(keybuf), "decoder.backbone.%d.pwconv2.weight", i);
            err |= load_tensor_from_safetensors(root, keybuf, b->pwconv2_weight, (size_t)dim * inner, data, data_base);
            snprintf(keybuf, sizeof(keybuf), "decoder.backbone.%d.pwconv2.bias", i);
            err |= load_tensor_from_safetensors(root, keybuf, b->pwconv2_bias, (size_t)dim, data, data_base);
            snprintf(keybuf, sizeof(keybuf), "decoder.backbone.%d.gamma", i);
            err |= load_tensor_from_safetensors(root, keybuf, b->gamma, (size_t)dim, data, data_base);
        }
        err |= load_tensor_from_safetensors(root, "decoder.head.mag_proj.weight", dec->mag_proj_weight, (size_t)n_bins * dim, data, data_base);
        err |= load_tensor_from_safetensors(root, "decoder.head.mag_proj.bias", dec->mag_proj_bias, (size_t)n_bins, data, data_base);
        err |= load_tensor_from_safetensors(root, "decoder.head.phase_proj.weight", dec->phase_proj_weight, (size_t)n_bins * dim, data, data_base);
        err |= load_tensor_from_safetensors(root, "decoder.head.phase_proj.bias", dec->phase_proj_bias, (size_t)n_bins, data, data_base);

        cJSON_Delete(root);
        free(data);
        if (err) {
            fprintf(stderr, "[bnns_convnext] Safetensors load failed (tensor missing/mismatch)\n");
            return -1;
        }
        fprintf(stderr, "[bnns_convnext] Loaded safetensors weights from %s\n", path);
        return 0;
    }

    /* Fallback: raw binary (no magic) - e.g. exported from Python */
    int ret = load_weights_binary(dec, f);
    fclose(f);
    return ret;
}

int bnns_convnext_load_mlmodelc(BNNSConvNeXtDecoder *dec, const char *path) {
    if (!dec || !path) return -1;

#ifdef __APPLE__
    fprintf(stderr, "[bnns_convnext] Compiling BNNS graph from: %s\n", path);

    bnns_graph_compile_options_t opts = BNNSGraphCompileOptionsMakeDefault();
    BNNSGraphCompileOptionsSetOptimizationPreference(
        opts, BNNSGraphOptimizationPreferencePerformance);

    dec->graph = BNNSGraphCompileFromFile(path, NULL, opts);
    BNNSGraphCompileOptionsDestroy(opts);

    if (!dec->graph.data) {
        fprintf(stderr, "[bnns_convnext] Failed to compile graph from %s (macOS 15+ required)\n", path);
        return -1;
    }

    dec->n_args = BNNSGraphGetArgumentCount(dec->graph, NULL);
    if (dec->n_args > BNNS_CONVNEXT_MAX_ARGS) {
        fprintf(stderr, "[bnns_convnext] Graph has too many args: %zu\n", dec->n_args);
        free(dec->graph.data);
        dec->graph.data = NULL;
        return -1;
    }

    /* Resolve input/output positions by name. Expected: semantic, acoustic (inputs);
     * magnitude, inst_freq (outputs). Actual names depend on CoreML export. */
    const char *in_semantic = "semantic";
    const char *in_acoustic = "acoustic";
    const char *out_mag = "magnitude";
    const char *out_phase = "inst_freq";
    dec->input_semantic_pos = BNNSGraphGetArgumentPosition(dec->graph, NULL, in_semantic);
    dec->input_acoustic_pos = BNNSGraphGetArgumentPosition(dec->graph, NULL, in_acoustic);
    dec->output_mag_pos = BNNSGraphGetArgumentPosition(dec->graph, NULL, out_mag);
    dec->output_phase_pos = BNNSGraphGetArgumentPosition(dec->graph, NULL, out_phase);

    dec->ctx = BNNSGraphContextMake(dec->graph);
    if (!dec->ctx.data) {
        fprintf(stderr, "[bnns_convnext] Failed to create execution context\n");
        free(dec->graph.data);
        dec->graph.data = NULL;
        return -1;
    }

    dec->workspace_size = BNNSGraphContextGetWorkspaceSize(dec->ctx, NULL);
    if (dec->workspace_size > 0 && dec->workspace_size != SIZE_MAX) {
        if (posix_memalign((void **)&dec->workspace, 4096, dec->workspace_size) != 0) {
            dec->workspace = NULL;
            BNNSGraphContextDestroy(dec->ctx);
            free(dec->graph.data);
            dec->graph.data = NULL;
            dec->ctx.data = NULL;
            return -1;
        }
        fprintf(stderr, "[bnns_convnext] ANE workspace: %zu bytes\n", dec->workspace_size);
    }

    dec->use_ane = 1;
    fprintf(stderr, "[bnns_convnext] BNNS Graph loaded for ANE execution\n");
    return 0;
#else
    (void)dec;
    (void)path;
    fprintf(stderr, "[bnns_convnext] BNNS Graph requires macOS\n");
    return -1;
#endif
}

static void convnext_layernorm(const float *input, float *output,
                                const float *weight, const float *bias,
                                int n_frames, int dim) {
    const float eps = 1e-5f;
    for (int t = 0; t < n_frames; t++) {
        const float *in_row = input + t * dim;
        float *out_row = output + t * dim;

        float mean = 0;
        vDSP_meanv(in_row, 1, &mean, dim);
        float neg_mean = -mean;
        vDSP_vsadd(in_row, 1, &neg_mean, out_row, 1, dim);
        float var = 0;
        vDSP_svesq(out_row, 1, &var, dim);
        var = var / (float)dim + eps;
        float inv_std = 1.0f / sqrtf(var);

        vDSP_vsmul(out_row, 1, &inv_std, out_row, 1, dim);
        vDSP_vmul(out_row, 1, weight, 1, out_row, 1, dim);
        vDSP_vadd(out_row, 1, bias, 1, out_row, 1, dim);
    }
}

static void depthwise_conv1d(const float *input, float *output,
                              const float *weight, const float *bias,
                              int n_frames, int channels, int kernel_size) {
    int pad = kernel_size / 2;
    for (int c = 0; c < channels; c++) {
        for (int t = 0; t < n_frames; t++) {
            float sum = bias[c];
            for (int k = 0; k < kernel_size; k++) {
                int ti = t + k - pad;
                if (ti >= 0 && ti < n_frames) {
                    sum += input[ti * channels + c] * weight[c * kernel_size + k];
                }
            }
            output[t * channels + c] = sum;
        }
    }
}

static void linear_forward(const float *input, float *output,
                            const float *weight, const float *bias,
                            int n_frames, int in_dim, int out_dim) {
    for (int t = 0; t < n_frames; t++) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    out_dim, in_dim, 1.0f,
                    weight, in_dim,
                    input + t * in_dim, 1,
                    0.0f,
                    output + t * out_dim, 1);
        if (bias) {
            vDSP_vadd(output + t * out_dim, 1, bias, 1,
                      output + t * out_dim, 1, out_dim);
        }
    }
}

static void gelu_inplace(float *data, int n) {
    /* GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    for (int i = 0; i < n; i++) {
        float x = data[i];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

static void convnext_block_forward(ConvNeXtBlockWeights *b,
                                    const float *input, float *output,
                                    float *scratch, int n_frames, int dim,
                                    int kernel_size) {
    float *dw_out = scratch;
    float *norm_out = scratch + n_frames * dim;
    float *pw1_out = scratch + 2 * n_frames * dim;

    /* Depthwise conv1d (channels-last → conv → channels-last) */
    depthwise_conv1d(input, dw_out, b->dwconv_weight, b->dwconv_bias,
                     n_frames, dim, kernel_size);

    /* LayerNorm */
    convnext_layernorm(dw_out, norm_out, b->norm_weight, b->norm_bias,
                       n_frames, dim);

    /* Pointwise conv1 (expand) + GELU */
    linear_forward(norm_out, pw1_out, b->pwconv1_weight, b->pwconv1_bias,
                   n_frames, dim, b->inner_dim);
    gelu_inplace(pw1_out, n_frames * b->inner_dim);

    /* Pointwise conv2 (contract) */
    float *pw2_out = dw_out;
    linear_forward(pw1_out, pw2_out, b->pwconv2_weight, b->pwconv2_bias,
                   n_frames, b->inner_dim, dim);

    /* Gamma scale + residual */
    for (int t = 0; t < n_frames; t++) {
        for (int d = 0; d < dim; d++) {
            int idx = t * dim + d;
            output[idx] = input[idx] + pw2_out[idx] * b->gamma[d];
        }
    }
}

int bnns_convnext_forward(BNNSConvNeXtDecoder *dec,
                           const float *semantic, const float *acoustic,
                           int n_frames,
                           float *out_magnitude, float *out_inst_freq) {
    if (!dec || !semantic || !acoustic || !out_magnitude || !out_inst_freq) return 0;
    if (n_frames <= 0) return 0;

#ifdef __APPLE__
    /* ANE path: dispatch to BNNS Graph */
    if (dec->use_ane && dec->ctx.data) {
        int fsq_dim = dec->input_dim - 256;
        int acoustic_dim = 256;
        int n_bins = dec->n_bins;

        bnns_graph_argument_t args[BNNS_CONVNEXT_MAX_ARGS];
        memset(args, 0, sizeof(args));
        args[dec->input_semantic_pos].data_ptr = (void *)semantic;
        args[dec->input_semantic_pos].data_ptr_size = (size_t)n_frames * fsq_dim * sizeof(float);
        args[dec->input_acoustic_pos].data_ptr = (void *)acoustic;
        args[dec->input_acoustic_pos].data_ptr_size = (size_t)n_frames * acoustic_dim * sizeof(float);
        args[dec->output_mag_pos].data_ptr = out_magnitude;
        args[dec->output_mag_pos].data_ptr_size = (size_t)n_frames * n_bins * sizeof(float);
        args[dec->output_phase_pos].data_ptr = out_inst_freq;
        args[dec->output_phase_pos].data_ptr_size = (size_t)n_frames * n_bins * sizeof(float);

        int rc = BNNSGraphContextExecute(dec->ctx, NULL, dec->n_args, args,
                                          dec->workspace_size, dec->workspace);
        if (rc != 0) {
            fprintf(stderr, "[bnns_convnext] BNNSGraphContextExecute failed: %d\n", rc);
            return 0;
        }
        return n_bins;
    }
#endif

    /* CPU path: cblas + vDSP */
    int dim = dec->dec_dim;
    int n_bins = dec->n_bins;
    int inner = (int)(dim * dec->ff_mult);
    size_t scratch_needed = (size_t)(2 * dim + inner) * (size_t)n_frames;
    if (scratch_needed > (size_t)dec->scratch_size) return 0;

    float *concat = (float *)calloc(n_frames * dec->input_dim, sizeof(float));
    float *hidden = (float *)calloc(n_frames * dim, sizeof(float));
    float *hidden2 = (float *)calloc(n_frames * dim, sizeof(float));
    if (!concat || !hidden || !hidden2) {
        free(concat); free(hidden); free(hidden2);
        return 0;
    }

    /* Concatenate semantic + acoustic along feature dim */
    int fsq_dim = dec->input_dim - 256;
    for (int t = 0; t < n_frames; t++) {
        memcpy(concat + t * dec->input_dim,
               semantic + t * fsq_dim, fsq_dim * sizeof(float));
        memcpy(concat + t * dec->input_dim + fsq_dim,
               acoustic + t * 256, 256 * sizeof(float));
    }

    /* Input projection (1D conv with kernel 7) */
    int pad = 3;
    for (int t = 0; t < n_frames; t++) {
        for (int d = 0; d < dim; d++) {
            float sum = dec->input_proj_bias[d];
            for (int k = 0; k < 7; k++) {
                int ti = t + k - pad;
                if (ti >= 0 && ti < n_frames) {
                    for (int c = 0; c < dec->input_dim; c++) {
                        sum += concat[ti * dec->input_dim + c] *
                               dec->input_proj_weight[(d * dec->input_dim + c) * 7 + k];
                    }
                }
            }
            hidden[t * dim + d] = sum;
        }
    }

    /* LeakyReLU(0.1) as in VocosDecoder */
    {
        const float slope = 0.1f;
        int n = n_frames * dim;
        for (int i = 0; i < n; i++) {
            float x = hidden[i];
            hidden[i] = (x >= 0) ? x : (slope * x);
        }
    }

    /* ConvNeXt blocks */
    float *cur_in = hidden;
    float *cur_out = hidden2;
    for (int i = 0; i < dec->n_layers; i++) {
        convnext_block_forward(&dec->blocks[i], cur_in, cur_out,
                               dec->scratch, n_frames, dim, dec->conv_kernel);
        float *tmp = cur_in;
        cur_in = cur_out;
        cur_out = tmp;
    }

    /* Magnitude head: linear → exp */
    linear_forward(cur_in, out_magnitude, dec->mag_proj_weight, dec->mag_proj_bias,
                   n_frames, dim, n_bins);
    int total = n_frames * n_bins;
    vvexpf(out_magnitude, out_magnitude, &total);

    /* Phase head: linear */
    linear_forward(cur_in, out_inst_freq, dec->phase_proj_weight, dec->phase_proj_bias,
                   n_frames, dim, n_bins);

    free(concat);
    free(hidden);
    free(hidden2);
    return n_bins;
}

void bnns_convnext_destroy(BNNSConvNeXtDecoder *dec) {
    if (!dec) return;
#ifdef __APPLE__
    if (dec->ctx.data)
        BNNSGraphContextDestroy(dec->ctx);
    if (dec->graph.data)
        free(dec->graph.data);
    if (dec->workspace) free(dec->workspace);
#endif
    free(dec->input_proj_weight);
    free(dec->input_proj_bias);
    for (int i = 0; i < dec->n_layers; i++) {
        ConvNeXtBlockWeights *b = &dec->blocks[i];
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
    free(dec->blocks);
    free(dec->mag_proj_weight);
    free(dec->mag_proj_bias);
    free(dec->phase_proj_weight);
    free(dec->phase_proj_bias);
    free(dec->scratch);
    free(dec);
}
