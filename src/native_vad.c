/**
 * native_vad.c — Pure C Silero-compatible VAD with AMX-accelerated inference.
 *
 * Architecture (matches Silero VAD v5):
 *   Learned STFT → 4× Conv1d+ReLU encoder → 1-layer LSTM → Linear → Sigmoid
 *
 * All matrix ops use cblas_sgemm/sgemv (AMX-accelerated on Apple Silicon).
 * Processes 512-sample (32ms @ 16kHz) chunks with 64-sample context overlap.
 * Weights extracted from ONNX via scripts/extract_silero_weights.py.
 *
 * Build: cc -O3 -shared -fPIC -arch arm64 -DACCELERATE_NEW_LAPACK
 *        -framework Accelerate -install_name @rpath/libnative_vad.dylib
 *        -o libnative_vad.dylib native_vad.c
 */

#include "native_vad.h"
#include "lstm_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ── Architecture constants (Silero VAD v5, 16kHz) ──────────────────────── */

#define NVAD_MAGIC       0x4441564E  /* "NVAD" little-endian */
#define NVAD_CHUNK       512
#define NVAD_CONTEXT     64
#define NVAD_FULL_LEN    (NVAD_CONTEXT + NVAD_CHUNK)  /* 576 */
#define NVAD_FILTER_LEN  256
#define NVAD_HOP_LEN     128
#define NVAD_PAD_LEN     (NVAD_FILTER_LEN / 2)        /* 128 */
#define NVAD_PADDED_LEN  (NVAD_FULL_LEN + 2 * NVAD_PAD_LEN)  /* 832 */
#define NVAD_STFT_CHANS  258        /* filter_length + 2 */
#define NVAD_FREQ_BINS   129        /* filter_length/2 + 1 */
#define NVAD_STFT_FRAMES 5          /* (832 - 256) / 128 + 1 */
#define NVAD_LSTM_HIDDEN 128
#define NVAD_LSTM_GATES  (4 * NVAD_LSTM_HIDDEN)  /* 512 */
#define NVAD_N_ENC       4

/* Encoder layer descriptors */
static const int ENC_IN_CH[NVAD_N_ENC]  = {129, 128, 64, 64};
static const int ENC_OUT_CH[NVAD_N_ENC] = {128,  64, 64, 128};
static const int ENC_STRIDE[NVAD_N_ENC] = {  1,   2,  2,   1};
/* All use kernel=3, padding=1 */

/* ── Working buffer sizes ────────────────────────────────────────────────── */

#define WORK_BUF_SIZE 2048  /* covers all intermediate activations */

/* ── Binary file header ──────────────────────────────────────────────────── */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t sample_rate;
    uint32_t filter_length;
    uint32_t hop_length;
    uint32_t n_freq_bins;
    uint32_t n_enc_layers;
    uint32_t lstm_hidden;
    uint32_t context_size;
} NvadHeader;

typedef struct {
    uint32_t in_ch;
    uint32_t out_ch;
    uint32_t stride;
} NvadEncDesc;

/* ── Engine struct ───────────────────────────────────────────────────────── */

struct NativeVad {
    /* STFT learned basis [STFT_CHANS × FILTER_LEN] = [258 × 256] */
    float *stft_basis;

    /* Encoder: 4 conv layers, weights stored as [out_ch, in_ch * 3] */
    float *enc_w[NVAD_N_ENC];
    float *enc_b[NVAD_N_ENC];

    /* LSTM: 1 layer, hidden=128 */
    float *lstm_wi;    /* [512, 128] input-to-hidden */
    float *lstm_wh;    /* [512, 128] hidden-to-hidden */
    float *lstm_bias;  /* [512] combined bias (ih + hh) */

    /* Output projection */
    float *out_w;      /* [128] */
    float  out_b;

    /* LSTM persistent state */
    float h[NVAD_LSTM_HIDDEN];
    float c[NVAD_LSTM_HIDDEN];

    /* Audio context (last 64 samples for chunk-to-chunk continuity) */
    float context[NVAD_CONTEXT];
    int   has_context;

    /* Pre-allocated working memory (no allocs in hot path) */
    float padded[NVAD_PADDED_LEN];
    float stft_patches[NVAD_FILTER_LEN * NVAD_STFT_FRAMES];
    float stft_out[NVAD_STFT_CHANS * NVAD_STFT_FRAMES];
    float mag[NVAD_FREQ_BINS * NVAD_STFT_FRAMES];
    float enc_a[WORK_BUF_SIZE];
    float enc_b_buf[WORK_BUF_SIZE];
    float im2col[WORK_BUF_SIZE];
    float gates[NVAD_LSTM_GATES];
    float relu_h[NVAD_LSTM_HIDDEN];
};

/* ── Helpers ─────────────────────────────────────────────────────────────── */

/* Use lstm_sigmoid from lstm_ops.h for consistency */
#define nvad_sigmoid lstm_sigmoid

/**
 * Reflection-pad audio: pad PAD_LEN samples on each side.
 * reflect([-3,-2,-1, 0,1,...,N-1, N,N+1,N+2]) = input[3,2,1, 0,1,...,N-1, N-2,N-3,N-4]
 */
static void reflection_pad(float *out, const float *in, int n, int pad) {
    /* Left reflection: out[i] = in[pad - i] for i = 0..pad-1 */
    for (int i = 0; i < pad; i++)
        out[i] = in[pad - i];

    /* Center: copy input */
    memcpy(out + pad, in, (size_t)n * sizeof(float));

    /* Right reflection: out[pad+n+i] = in[n - 1 - i] for i = 0..pad-1
     * (mirrors the last sample first, matching PyTorch reflect padding) */
    for (int i = 0; i < pad; i++)
        out[pad + n + i] = in[n - 1 - i];
}

/**
 * Build im2col matrix for Conv1d(kernel=3, padding=1).
 * Input layout: [C_in, T_in] row-major.
 * Output layout: [C_in * 3, T_out] row-major.
 */
static void im2col_conv1d(const float *input, int C_in, int T_in,
                           int stride, float *col, int T_out) {
    const int kernel = 3;
    const int padding = 1;
    int rows = C_in * kernel;

    memset(col, 0, sizeof(float) * (size_t)rows * (size_t)T_out);

    for (int t = 0; t < T_out; t++) {
        for (int k = 0; k < kernel; k++) {
            int t_in = t * stride - padding + k;
            if (t_in >= 0 && t_in < T_in) {
                for (int ci = 0; ci < C_in; ci++) {
                    col[(ci * kernel + k) * T_out + t] =
                        input[ci * T_in + t_in];
                }
            }
        }
    }
}

/**
 * Conv1d forward: weight[C_out, C_in*3] @ im2col[C_in*3, T_out] + bias.
 * Applies ReLU in-place after bias addition.
 */
static void conv1d_relu(const float *weight, const float *bias,
                         int C_out, int C_in, int T_in, int stride,
                         const float *input, float *im2col_buf,
                         float *output, int *p_T_out) {
    int T_out = (T_in + 2 - 3) / stride + 1;
    *p_T_out = T_out;
    int K = C_in * 3;

    im2col_conv1d(input, C_in, T_in, stride, im2col_buf, T_out);

#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                C_out, T_out, K,
                1.0f, weight, K, im2col_buf, T_out,
                0.0f, output, T_out);
#else
    for (int co = 0; co < C_out; co++)
        for (int t = 0; t < T_out; t++) {
            float s = 0.0f;
            for (int ki = 0; ki < K; ki++)
                s += weight[co * K + ki] * im2col_buf[ki * T_out + t];
            output[co * T_out + t] = s;
        }
#endif

    /* Add bias and ReLU */
    for (int co = 0; co < C_out; co++) {
        float b = bias[co];
        for (int t = 0; t < T_out; t++) {
            float v = output[co * T_out + t] + b;
            output[co * T_out + t] = v > 0.0f ? v : 0.0f;
        }
    }
}

/* ── STFT Forward ────────────────────────────────────────────────────────── */

/**
 * Learned STFT: 1D convolution with basis[258, 256], stride 128.
 * Input: padded audio [832]. Output: magnitude spectrogram [129, 5].
 */
static void stft_magnitude(const NativeVad *vad, const float *padded,
                            float *mag_out) {
    const int n_frames = NVAD_STFT_FRAMES;

    /* Build patch matrix [FILTER_LEN, n_frames] from padded audio.
     * Each column is a 256-sample window at stride 128. */
    float *patches = (float *)vad->stft_patches;
    for (int t = 0; t < n_frames; t++) {
        const float *win = padded + t * NVAD_HOP_LEN;
        for (int k = 0; k < NVAD_FILTER_LEN; k++)
            patches[k * n_frames + t] = win[k];
    }

    /* stft_out[258, 5] = basis[258, 256] × patches[256, 5] */
    float *stft = (float *)vad->stft_out;
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NVAD_STFT_CHANS, n_frames, NVAD_FILTER_LEN,
                1.0f, vad->stft_basis, NVAD_FILTER_LEN,
                patches, n_frames,
                0.0f, stft, n_frames);
#else
    for (int c = 0; c < NVAD_STFT_CHANS; c++)
        for (int t = 0; t < n_frames; t++) {
            float s = 0.0f;
            for (int k = 0; k < NVAD_FILTER_LEN; k++)
                s += vad->stft_basis[c * NVAD_FILTER_LEN + k] * patches[k * n_frames + t];
            stft[c * n_frames + t] = s;
        }
#endif

    /* Magnitude: sqrt(real² + imag²) for 129 frequency bins */
    for (int f = 0; f < NVAD_FREQ_BINS; f++) {
        for (int t = 0; t < n_frames; t++) {
            float re = stft[f * n_frames + t];
            float im = stft[(f + NVAD_FREQ_BINS) * n_frames + t];
            mag_out[f * n_frames + t] = sqrtf(re * re + im * im);
        }
    }
}

/* ── LSTM Step (delegates to shared lstm_ops.h) ──────────────────────────── */

static void nvad_lstm_step(NativeVad *vad, const float *enc_out, int t, int T) {
    lstm_step(vad->lstm_wi, vad->lstm_wh, vad->lstm_bias,
              enc_out + t, T,  /* stride-T access into encoder output columns */
              vad->h, vad->c, vad->gates,
              NVAD_LSTM_HIDDEN, NVAD_LSTM_HIDDEN);
}

/* ── Weight Loading ──────────────────────────────────────────────────────── */

static int load_weights(NativeVad *vad, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[native_vad] Cannot open weights: %s\n", path);
        return -1;
    }

    NvadHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "[native_vad] Failed to read header\n");
        fclose(f);
        return -1;
    }

    if (hdr.magic != NVAD_MAGIC) {
        fprintf(stderr, "[native_vad] Bad magic: 0x%08X (expected 0x%08X)\n",
                hdr.magic, NVAD_MAGIC);
        fclose(f);
        return -1;
    }
    if (hdr.version != 1) {
        fprintf(stderr, "[native_vad] Unsupported version: %u\n", hdr.version);
        fclose(f);
        return -1;
    }
    if (hdr.filter_length != NVAD_FILTER_LEN || hdr.lstm_hidden != NVAD_LSTM_HIDDEN ||
        hdr.n_enc_layers != NVAD_N_ENC || hdr.n_freq_bins != NVAD_FREQ_BINS) {
        fprintf(stderr, "[native_vad] Architecture mismatch\n");
        fclose(f);
        return -1;
    }

    /* Read encoder layer descriptors (validate but don't use — hardcoded) */
    NvadEncDesc descs[NVAD_N_ENC];
    if (fread(descs, sizeof(NvadEncDesc), NVAD_N_ENC, f) != NVAD_N_ENC) {
        fprintf(stderr, "[native_vad] Failed to read encoder descriptors\n");
        fclose(f);
        return -1;
    }

    /* ── Allocate and read weights ─── */

    size_t total = 0;
    size_t got;
    int read_ok = 1;

#define READ_WEIGHTS(dst, count) do { \
    got = fread((dst), sizeof(float), (count), f); \
    total += got; \
    if (got != (size_t)(count)) read_ok = 0; \
} while (0)

    /* STFT basis [258, 256] */
    size_t stft_sz = (size_t)NVAD_STFT_CHANS * NVAD_FILTER_LEN;
    vad->stft_basis = (float *)malloc(stft_sz * sizeof(float));
    if (!vad->stft_basis) { fclose(f); return -1; }
    READ_WEIGHTS(vad->stft_basis, stft_sz);

    /* Encoder conv layers */
    for (int i = 0; i < NVAD_N_ENC; i++) {
        size_t w_sz = (size_t)ENC_OUT_CH[i] * ENC_IN_CH[i] * 3;
        size_t b_sz = (size_t)ENC_OUT_CH[i];
        vad->enc_w[i] = (float *)malloc(w_sz * sizeof(float));
        vad->enc_b[i] = (float *)malloc(b_sz * sizeof(float));
        if (!vad->enc_w[i] || !vad->enc_b[i]) {
            /* Free partially allocated encoder layers */
            for (int j = 0; j <= i; j++) { free(vad->enc_w[j]); free(vad->enc_b[j]); }
            fclose(f);
            return -1;
        }
        READ_WEIGHTS(vad->enc_w[i], w_sz);
        READ_WEIGHTS(vad->enc_b[i], b_sz);
    }

    /* LSTM weights */
    size_t lstm_w_sz = (size_t)NVAD_LSTM_GATES * NVAD_LSTM_HIDDEN;
    vad->lstm_wi   = (float *)malloc(lstm_w_sz * sizeof(float));
    vad->lstm_wh   = (float *)malloc(lstm_w_sz * sizeof(float));
    vad->lstm_bias = (float *)malloc(NVAD_LSTM_GATES * sizeof(float));
    if (!vad->lstm_wi || !vad->lstm_wh || !vad->lstm_bias) { fclose(f); return -1; }
    READ_WEIGHTS(vad->lstm_wi,   lstm_w_sz);
    READ_WEIGHTS(vad->lstm_wh,   lstm_w_sz);
    READ_WEIGHTS(vad->lstm_bias, NVAD_LSTM_GATES);

    /* Output projection */
    vad->out_w = (float *)malloc(NVAD_LSTM_HIDDEN * sizeof(float));
    if (!vad->out_w) { fclose(f); return -1; }
    READ_WEIGHTS(vad->out_w, NVAD_LSTM_HIDDEN);
    READ_WEIGHTS(&vad->out_b, 1);

#undef READ_WEIGHTS

    fclose(f);

    if (!read_ok) {
        fprintf(stderr, "[native_vad] Truncated weight file\n");
        return -1;
    }

    /* Validate total read count */
    size_t expected = stft_sz;
    for (int i = 0; i < NVAD_N_ENC; i++)
        expected += (size_t)ENC_OUT_CH[i] * ENC_IN_CH[i] * 3 + ENC_OUT_CH[i];
    expected += 2 * lstm_w_sz + NVAD_LSTM_GATES + NVAD_LSTM_HIDDEN + 1;

    if (total != expected) {
        fprintf(stderr, "[native_vad] Weight count mismatch: got %zu, expected %zu\n",
                total, expected);
        return -1;
    }

    fprintf(stderr, "[native_vad] Loaded %s: %zu params (%.1f KB)\n",
            path, expected, (float)(expected * sizeof(float)) / 1024.0f);
    return 0;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

NativeVad *native_vad_create(const char *weights_path) {
    if (!weights_path) return NULL;

    NativeVad *vad = (NativeVad *)calloc(1, sizeof(NativeVad));
    if (!vad) return NULL;

    if (load_weights(vad, weights_path) != 0) {
        native_vad_destroy(vad);
        return NULL;
    }

    return vad;
}

void native_vad_destroy(NativeVad *vad) {
    if (!vad) return;
    free(vad->stft_basis);
    for (int i = 0; i < NVAD_N_ENC; i++) {
        free(vad->enc_w[i]);
        free(vad->enc_b[i]);
    }
    free(vad->lstm_wi);
    free(vad->lstm_wh);
    free(vad->lstm_bias);
    free(vad->out_w);
    free(vad);
}

float native_vad_process(NativeVad *vad, const float *samples) {
    if (!vad || !samples) return -1.0f;

    /* ── 1. Prepend context (64 samples) + current chunk (512) = 576 ──── */
    float input[NVAD_FULL_LEN];
    if (vad->has_context)
        memcpy(input, vad->context, NVAD_CONTEXT * sizeof(float));
    else
        memset(input, 0, NVAD_CONTEXT * sizeof(float));
    memcpy(input + NVAD_CONTEXT, samples, NVAD_CHUNK * sizeof(float));

    /* Update context for next call: last 64 of the 576-sample input */
    memcpy(vad->context, input + NVAD_FULL_LEN - NVAD_CONTEXT,
           NVAD_CONTEXT * sizeof(float));
    vad->has_context = 1;

    /* ── 2. Reflection-pad 128 on each side → 832 samples ────────────── */
    reflection_pad(vad->padded, input, NVAD_FULL_LEN, NVAD_PAD_LEN);

    /* ── 3. Learned STFT → magnitude spectrogram [129, 5] ────────────── */
    stft_magnitude(vad, vad->padded, vad->mag);

    /* ── 4. Encoder: 4× Conv1d(k=3)+ReLU ─────────────────────────────── */
    const float *enc_in = vad->mag;
    float *buf_a = vad->enc_a;
    float *buf_b = vad->enc_b_buf;
    int T_in = NVAD_STFT_FRAMES;

    for (int i = 0; i < NVAD_N_ENC; i++) {
        float *enc_out = (i % 2 == 0) ? buf_a : buf_b;
        int T_out;
        conv1d_relu(vad->enc_w[i], vad->enc_b[i],
                    ENC_OUT_CH[i], ENC_IN_CH[i], T_in, ENC_STRIDE[i],
                    enc_in, vad->im2col, enc_out, &T_out);
        enc_in = enc_out;
        T_in = T_out;
    }
    /* enc_in now points to encoder output [128, 2] in the last used buffer */
    int T_final = T_in;  /* should be 2 */

    /* ── 5. LSTM: process T_final time steps ──────────────────────────── */
    for (int t = 0; t < T_final; t++)
        nvad_lstm_step(vad, enc_in, t, T_final);

    /* ── 6. Output: ReLU → Linear(128→1) → Sigmoid ───────────────────── */
    for (int i = 0; i < NVAD_LSTM_HIDDEN; i++)
        vad->relu_h[i] = vad->h[i] > 0.0f ? vad->h[i] : 0.0f;

    float logit;
#ifdef __APPLE__
    logit = cblas_sdot(NVAD_LSTM_HIDDEN, vad->out_w, 1, vad->relu_h, 1);
#else
    logit = 0.0f;
    for (int i = 0; i < NVAD_LSTM_HIDDEN; i++)
        logit += vad->out_w[i] * vad->relu_h[i];
#endif
    logit += vad->out_b;

    return nvad_sigmoid(logit);
}

int native_vad_process_audio(NativeVad *vad, const float *audio, int n_samples,
                              float *probs_out, int max_probs) {
    if (!vad || !audio || !probs_out || max_probs < 0) return -1;
    if (max_probs == 0) return 0;

    int n_chunks = n_samples / NVAD_CHUNK;
    if (n_chunks <= 0) return 0;
    if (n_chunks > max_probs) n_chunks = max_probs;

    for (int i = 0; i < n_chunks; i++) {
        float p = native_vad_process(vad, audio + i * NVAD_CHUNK);
        if (p < 0.0f) return -1;
        probs_out[i] = p;
    }

    return n_chunks;
}

void native_vad_reset(NativeVad *vad) {
    if (!vad) return;
    memset(vad->h, 0, sizeof(vad->h));
    memset(vad->c, 0, sizeof(vad->c));
    memset(vad->context, 0, sizeof(vad->context));
    vad->has_context = 0;
}

int native_vad_chunk_size(const NativeVad *vad) {
    (void)vad;
    return NVAD_CHUNK;
}
