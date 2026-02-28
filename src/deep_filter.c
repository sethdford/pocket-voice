/**
 * deep_filter.c — Neural noise suppression via ERB-band gain prediction.
 *
 * Architecture (DeepFilterNet-inspired):
 *   512-pt FFT → 32 ERB bands → 2-layer GRU(64) → sigmoid gain mask → IFFT
 *
 * All matrix ops use cblas_sgemv (AMX-accelerated on Apple Silicon).
 * All vector ops use vDSP. Zero allocations after create().
 *
 * Weight file format (.dnf):
 *   Header (DnfHeader) → ERB filter bank → GRU L1 → GRU L2 → output linear
 *
 * Build: cc -O3 -shared -fPIC -arch arm64 -DACCELERATE_NEW_LAPACK
 *        -framework Accelerate -install_name @rpath/libdeep_filter.dylib
 *        -o libdeep_filter.dylib deep_filter.c
 */

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "deep_filter.h"

/* ── Architecture constants ──────────────────────────────────────────────── */

#define DF_MAGIC         0x46464E44  /* "DNFF" little-endian */
#define DF_VERSION       1
#define DF_SAMPLE_RATE   16000
#define DF_FFT_SIZE      512
#define DF_HOP_SIZE      256        /* 16ms @ 16kHz */
#define DF_HALF_FFT      (DF_FFT_SIZE / 2)  /* 256 bins (DC to Nyquist-1) */
#define DF_FREQ_BINS     (DF_FFT_SIZE / 2 + 1)  /* 257 bins (DC to Nyquist) */
#define DF_N_ERB         32
#define DF_GRU_HIDDEN    64
#define DF_GRU_GATES     (3 * DF_GRU_HIDDEN)  /* 192: z, r, n gates */
#define DF_N_GRU_LAYERS  2

/* ── Binary file header ──────────────────────────────────────────────────── */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t sample_rate;
    uint32_t fft_size;
    uint32_t hop_size;
    uint32_t n_erb;
    uint32_t n_gru_layers;
    uint32_t gru_hidden;
    uint32_t reserved[4];
} DnfHeader;

/* ── GRU layer weights ───────────────────────────────────────────────────── */

typedef struct {
    float *W_ih;   /* [3*H, D] input-to-hidden (z,r,n gates) */
    float *W_hh;   /* [3*H, H] hidden-to-hidden */
    float *b_ih;   /* [3*H] input bias */
    float *b_hh;   /* [3*H] hidden bias */
    int input_dim;
} GruLayer;

/* ── Engine struct ───────────────────────────────────────────────────────── */

struct DeepFilter {
    /* FFT */
    FFTSetup fft_setup;
    int log2n;
    float *window;          /* [FFT_SIZE] Hann window */

    /* ERB filter bank: maps FFT bins → ERB bands */
    float *erb_fb;          /* [N_ERB × FREQ_BINS] row-major */
    float *erb_fb_inv;      /* [FREQ_BINS × N_ERB] for interpolation back */

    /* GRU layers */
    GruLayer gru[DF_N_GRU_LAYERS];

    /* Output projection: Linear(hidden → n_erb) + sigmoid */
    float *out_w;           /* [N_ERB, GRU_HIDDEN] */
    float *out_b;           /* [N_ERB] */

    /* GRU persistent hidden states */
    float h[DF_N_GRU_LAYERS][DF_GRU_HIDDEN];

    /* ── Pre-allocated working buffers (zero allocs in hot path) ─── */
    float *frame_buf;       /* [FFT_SIZE] accumulation buffer */
    int    frame_len;       /* current samples in frame_buf */

    float *output_queue;    /* processed output samples */
    int    output_len;
    int    output_cap;

    float *overlap_out;     /* [FFT_SIZE * 2] overlap-add buffer */
    int    overlap_out_cap;

    /* Per-frame scratch */
    float *work_windowed;   /* [FFT_SIZE] */
    float *fft_real;        /* [HALF_FFT] */
    float *fft_imag;        /* [HALF_FFT] */
    float *work_mag;        /* [FREQ_BINS] */
    float *work_erb;        /* [N_ERB] */
    float *work_gains_erb;  /* [N_ERB] */
    float *work_gains_bin;  /* [FREQ_BINS] */
    float *gain_smooth;     /* [FREQ_BINS] smoothed per-bin gains */
    float *work_recon;      /* [FFT_SIZE] */
    float *gru_scratch;     /* [GRU_GATES] for GRU gate computation */
    float *gru_scratch2;    /* [GRU_GATES] for hidden gate computation */
    float *gru_input;       /* [GRU_HIDDEN] intermediate between GRU layers */

    /* Parameters */
    float strength;         /* 0.0 = passthrough, 1.0 = full suppression */
    float min_gain;         /* minimum gain floor (linear) */
    float smooth_coeff;     /* gain smoothing coefficient */
    int   passthrough;      /* 1 if no weights loaded */
};

/* ── ERB filter bank computation ─────────────────────────────────────────── */

static inline float hz_to_erb_rate(float hz) {
    return 21.4f * log10f(0.00437f * hz + 1.0f);
}

/**
 * Compute triangular ERB filter bank.
 * Maps FREQ_BINS linear frequency bins to N_ERB perceptual bands.
 * Each filter is triangular with peak = 1.0 at center, normalized by bandwidth.
 */
static void compute_erb_filter_bank(float *fb, int n_erb, int n_bins,
                                     int sample_rate) {
    float nyquist = (float)sample_rate / 2.0f;
    float erb_low = hz_to_erb_rate(0.0f);
    float erb_high = hz_to_erb_rate(nyquist);
    float erb_step = (erb_high - erb_low) / (float)(n_erb + 1);

    /* Compute ERB center frequencies */
    float centers[DF_N_ERB + 2];
    for (int i = 0; i < n_erb + 2; i++) {
        float erb_rate = erb_low + (float)i * erb_step;
        centers[i] = (powf(10.0f, erb_rate / 21.4f) - 1.0f) / 0.00437f;
    }

    float bin_hz = nyquist / (float)(n_bins - 1);

    memset(fb, 0, (size_t)n_erb * (size_t)n_bins * sizeof(float));

    for (int band = 0; band < n_erb; band++) {
        float lo = centers[band];
        float mid = centers[band + 1];
        float hi = centers[band + 2];

        float sum = 0.0f;
        for (int bin = 0; bin < n_bins; bin++) {
            float freq = (float)bin * bin_hz;
            float val = 0.0f;
            if (freq >= lo && freq <= mid && mid > lo) {
                val = (freq - lo) / (mid - lo);
            } else if (freq > mid && freq <= hi && hi > mid) {
                val = (hi - freq) / (hi - mid);
            }
            fb[band * n_bins + bin] = val;
            sum += val;
        }
        /* Normalize so each band sums to 1 */
        if (sum > 0.0f) {
            float inv_sum = 1.0f / sum;
            vDSP_vsmul(&fb[band * n_bins], 1, &inv_sum, &fb[band * n_bins], 1,
                        (vDSP_Length)n_bins);
        }
    }
}

/**
 * Compute pseudo-inverse of ERB filter bank for interpolation back to bins.
 * Uses simple transpose + per-bin normalization.
 */
static void compute_erb_inverse(const float *fb, float *fb_inv,
                                 int n_erb, int n_bins) {
    /* Transpose: fb_inv[bin, band] = fb[band, bin] */
    for (int band = 0; band < n_erb; band++) {
        for (int bin = 0; bin < n_bins; bin++) {
            fb_inv[bin * n_erb + band] = fb[band * n_bins + bin];
        }
    }
    /* Normalize each bin (row) so filters sum to 1 */
    for (int bin = 0; bin < n_bins; bin++) {
        float sum = 0.0f;
        vDSP_sve(&fb_inv[bin * n_erb], 1, &sum, (vDSP_Length)n_erb);
        if (sum > 1e-8f) {
            float inv = 1.0f / sum;
            vDSP_vsmul(&fb_inv[bin * n_erb], 1, &inv,
                        &fb_inv[bin * n_erb], 1, (vDSP_Length)n_erb);
        }
    }
}

/* ── GRU step ────────────────────────────────────────────────────────────── */

/**
 * Single GRU time step using cblas for AMX acceleration.
 *
 * GRU equations:
 *   z = sigmoid(W_ih_z @ x + b_ih_z + W_hh_z @ h + b_hh_z)  — update gate
 *   r = sigmoid(W_ih_r @ x + b_ih_r + W_hh_r @ h + b_hh_r)  — reset gate
 *   n = tanh(W_ih_n @ x + b_ih_n + r * (W_hh_n @ h + b_hh_n)) — new gate
 *   h = (1 - z) * n + z * h_prev
 */
static void gru_step(const GruLayer *layer, const float *x, float *h,
                      float *gates_i, float *gates_h, int H) {
    int D = layer->input_dim;
    int G = 3 * H;

    /* gates_i = W_ih @ x + b_ih */
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                G, D, 1.0f, layer->W_ih, D, x, 1, 0.0f, gates_i, 1);
    vDSP_vadd(gates_i, 1, layer->b_ih, 1, gates_i, 1, (vDSP_Length)G);

    /* gates_h = W_hh @ h + b_hh */
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                G, H, 1.0f, layer->W_hh, H, h, 1, 0.0f, gates_h, 1);
    vDSP_vadd(gates_h, 1, layer->b_hh, 1, gates_h, 1, (vDSP_Length)G);

    /* Split gates: [z, r, n] each of size H */
    float *zi = gates_i;           float *zh = gates_h;
    float *ri = gates_i + H;      float *rh = gates_h + H;
    float *ni = gates_i + 2 * H;  float *nh = gates_h + 2 * H;

    for (int i = 0; i < H; i++) {
        /* z = sigmoid(zi + zh) */
        float z_val = zi[i] + zh[i];
        if (z_val > 10.0f) z_val = 1.0f;
        else if (z_val < -10.0f) z_val = 0.0f;
        else z_val = 1.0f / (1.0f + expf(-z_val));

        /* r = sigmoid(ri + rh) */
        float r_val = ri[i] + rh[i];
        if (r_val > 10.0f) r_val = 1.0f;
        else if (r_val < -10.0f) r_val = 0.0f;
        else r_val = 1.0f / (1.0f + expf(-r_val));

        /* n = tanh(ni + r * nh) */
        float n_val = tanhf(ni[i] + r_val * nh[i]);

        /* h = (1 - z) * n + z * h_prev */
        h[i] = (1.0f - z_val) * n_val + z_val * h[i];
    }
}

/* ── Weight loading ──────────────────────────────────────────────────────── */

static void free_gru_layer(GruLayer *layer) {
    free(layer->W_ih);
    free(layer->W_hh);
    free(layer->b_ih);
    free(layer->b_hh);
    layer->W_ih = layer->W_hh = layer->b_ih = layer->b_hh = NULL;
}

static int load_weights(DeepFilter *df, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[deep_filter] Cannot open weights: %s\n", path);
        return -1;
    }

    DnfHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "[deep_filter] Failed to read header\n");
        fclose(f);
        return -1;
    }

    if (hdr.magic != DF_MAGIC) {
        fprintf(stderr, "[deep_filter] Bad magic: 0x%08X (expected 0x%08X)\n",
                hdr.magic, DF_MAGIC);
        fclose(f);
        return -1;
    }
    if (hdr.version != DF_VERSION) {
        fprintf(stderr, "[deep_filter] Unsupported version: %u\n", hdr.version);
        fclose(f);
        return -1;
    }
    if (hdr.fft_size != DF_FFT_SIZE || hdr.n_erb != DF_N_ERB ||
        hdr.gru_hidden != DF_GRU_HIDDEN || hdr.n_gru_layers != DF_N_GRU_LAYERS) {
        fprintf(stderr, "[deep_filter] Architecture mismatch\n");
        fclose(f);
        return -1;
    }

    size_t total = 0;
    int read_ok = 1;

#define READ_F(dst, count) do { \
    size_t _n = fread((dst), sizeof(float), (count), f); \
    total += _n; \
    if (_n != (size_t)(count)) read_ok = 0; \
} while (0)

    /* ERB filter bank [N_ERB × FREQ_BINS] */
    size_t erb_sz = (size_t)DF_N_ERB * DF_FREQ_BINS;
    df->erb_fb = (float *)malloc(erb_sz * sizeof(float));
    if (!df->erb_fb) { fclose(f); return -1; }
    READ_F(df->erb_fb, erb_sz);

    /* GRU layers */
    int gru_input_dims[DF_N_GRU_LAYERS] = { DF_N_ERB, DF_GRU_HIDDEN };
    for (int i = 0; i < DF_N_GRU_LAYERS; i++) {
        int D = gru_input_dims[i];
        int H = DF_GRU_HIDDEN;
        int G = 3 * H;

        df->gru[i].input_dim = D;
        df->gru[i].W_ih = (float *)malloc((size_t)G * D * sizeof(float));
        df->gru[i].W_hh = (float *)malloc((size_t)G * H * sizeof(float));
        df->gru[i].b_ih = (float *)malloc((size_t)G * sizeof(float));
        df->gru[i].b_hh = (float *)malloc((size_t)G * sizeof(float));

        if (!df->gru[i].W_ih || !df->gru[i].W_hh ||
            !df->gru[i].b_ih || !df->gru[i].b_hh) {
            fclose(f);
            return -1;
        }

        READ_F(df->gru[i].W_ih, (size_t)G * D);
        READ_F(df->gru[i].W_hh, (size_t)G * H);
        READ_F(df->gru[i].b_ih, G);
        READ_F(df->gru[i].b_hh, G);
    }

    /* Output projection [N_ERB, GRU_HIDDEN] + bias [N_ERB] */
    df->out_w = (float *)malloc((size_t)DF_N_ERB * DF_GRU_HIDDEN * sizeof(float));
    df->out_b = (float *)malloc((size_t)DF_N_ERB * sizeof(float));
    if (!df->out_w || !df->out_b) { fclose(f); return -1; }
    READ_F(df->out_w, (size_t)DF_N_ERB * DF_GRU_HIDDEN);
    READ_F(df->out_b, DF_N_ERB);

#undef READ_F

    fclose(f);

    if (!read_ok) {
        fprintf(stderr, "[deep_filter] Truncated weight file\n");
        return -1;
    }

    /* Compute ERB inverse from loaded filter bank */
    df->erb_fb_inv = (float *)malloc((size_t)DF_FREQ_BINS * DF_N_ERB * sizeof(float));
    if (!df->erb_fb_inv) return -1;
    compute_erb_inverse(df->erb_fb, df->erb_fb_inv, DF_N_ERB, DF_FREQ_BINS);

    /* Validate total */
    size_t expected = erb_sz;
    for (int i = 0; i < DF_N_GRU_LAYERS; i++) {
        int D = gru_input_dims[i];
        int H = DF_GRU_HIDDEN;
        int G = 3 * H;
        expected += (size_t)G * D + (size_t)G * H + G + G;
    }
    expected += (size_t)DF_N_ERB * DF_GRU_HIDDEN + DF_N_ERB;

    if (total != expected) {
        fprintf(stderr, "[deep_filter] Weight count mismatch: got %zu, expected %zu\n",
                total, expected);
        return -1;
    }

    fprintf(stderr, "[deep_filter] Loaded %s: %zu params (%.1f KB)\n",
            path, expected, (float)(expected * sizeof(float)) / 1024.0f);
    return 0;
}

/* ── Process one FFT frame ───────────────────────────────────────────────── */

static void process_one_frame(DeepFilter *df) {
    int n = DF_FFT_SIZE;
    int half = DF_HALF_FFT;

    /* Window the frame */
    vDSP_vmul(df->frame_buf, 1, df->window, 1, df->work_windowed, 1,
              (vDSP_Length)n);

    /* Forward FFT */
    DSPSplitComplex fft_buf = { df->fft_real, df->fft_imag };
    vDSP_ctoz((DSPComplex *)df->work_windowed, 2, &fft_buf, 1,
              (vDSP_Length)half);
    vDSP_fft_zrip(df->fft_setup, &fft_buf, 1, df->log2n, FFT_FORWARD);

    if (df->passthrough) {
        /* No neural processing — just pass through */
        goto do_ifft;
    }

    /* Compute magnitude spectrum [FREQ_BINS] */
    {
        /* DC bin: stored in fft_real[0] (packed format: DC real in realp[0]) */
        df->work_mag[0] = fabsf(fft_buf.realp[0]);
        /* Nyquist bin: stored in fft_imag[0] (packed format: Nyquist real in imagp[0]) */
        df->work_mag[half] = fabsf(fft_buf.imagp[0]);
        /* Bins 1..(half-1): offset past packed DC/Nyquist element 0 */
        DSPSplitComplex sc_offset = { fft_buf.realp + 1, fft_buf.imagp + 1 };
        vDSP_zvabs(&sc_offset, 1, df->work_mag + 1, 1, (vDSP_Length)(half - 1));
    }

    /* Map magnitude to ERB bands: erb[N_ERB] = erb_fb[N_ERB, FREQ_BINS] @ mag[FREQ_BINS] */
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                DF_N_ERB, DF_FREQ_BINS,
                1.0f, df->erb_fb, DF_FREQ_BINS,
                df->work_mag, 1,
                0.0f, df->work_erb, 1);

    /* Log-compress ERB features for better GRU input range */
    {
        float one = 1.0f;
        /* erb = log(1 + erb) */
        vDSP_vsadd(df->work_erb, 1, &one, df->work_erb, 1, (vDSP_Length)DF_N_ERB);
        int n_erb_i = DF_N_ERB;
        vvlogf(df->work_erb, df->work_erb, &n_erb_i);
    }

    /* GRU Layer 1: input = ERB features [N_ERB] → hidden [GRU_HIDDEN] */
    gru_step(&df->gru[0], df->work_erb, df->h[0],
             df->gru_scratch, df->gru_scratch2, DF_GRU_HIDDEN);

    /* NaN guard: if GRU L1 state is corrupted, reset and zero input to L2 */
    {
        int nan_found = 0;
        for (int i = 0; i < DF_GRU_HIDDEN; i++) {
            if (!isfinite(df->h[0][i])) { nan_found = 1; break; }
        }
        if (__builtin_expect(nan_found, 0)) {
            memset(df->h[0], 0, DF_GRU_HIDDEN * sizeof(float));
        }
    }

    /* GRU Layer 2: input = h1 [GRU_HIDDEN] → hidden [GRU_HIDDEN] */
    gru_step(&df->gru[1], df->h[0], df->h[1],
             df->gru_scratch, df->gru_scratch2, DF_GRU_HIDDEN);

    /* NaN guard: if GRU L2 state is corrupted, reset */
    {
        int nan_found = 0;
        for (int i = 0; i < DF_GRU_HIDDEN; i++) {
            if (!isfinite(df->h[1][i])) { nan_found = 1; break; }
        }
        if (__builtin_expect(nan_found, 0)) {
            memset(df->h[1], 0, DF_GRU_HIDDEN * sizeof(float));
        }
    }

    /* Output projection: gains_erb = sigmoid(out_w @ h2 + out_b) */
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                DF_N_ERB, DF_GRU_HIDDEN,
                1.0f, df->out_w, DF_GRU_HIDDEN,
                df->h[1], 1,
                0.0f, df->work_gains_erb, 1);
    vDSP_vadd(df->work_gains_erb, 1, df->out_b, 1,
              df->work_gains_erb, 1, (vDSP_Length)DF_N_ERB);

    /* Sigmoid activation */
    for (int i = 0; i < DF_N_ERB; i++) {
        float v = df->work_gains_erb[i];
        if (v > 10.0f) df->work_gains_erb[i] = 1.0f;
        else if (v < -10.0f) df->work_gains_erb[i] = 0.0f;
        else df->work_gains_erb[i] = 1.0f / (1.0f + expf(-v));
    }

    /* Apply strength: gain = strength * predicted_gain + (1 - strength) * 1.0 */
    if (df->strength < 1.0f) {
        float s = df->strength;
        float one_minus_s = 1.0f - s;
        vDSP_vsmsa(df->work_gains_erb, 1, &s, &one_minus_s,
                    df->work_gains_erb, 1, (vDSP_Length)DF_N_ERB);
    }

    /* Clamp to minimum gain floor */
    {
        float mg = df->min_gain;
        vDSP_vthr(df->work_gains_erb, 1, &mg, df->work_gains_erb, 1,
                   (vDSP_Length)DF_N_ERB);
    }

    /* Interpolate ERB gains back to FFT bins:
     * gains_bin[FREQ_BINS] = erb_fb_inv[FREQ_BINS, N_ERB] @ gains_erb[N_ERB] */
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                DF_FREQ_BINS, DF_N_ERB,
                1.0f, df->erb_fb_inv, DF_N_ERB,
                df->work_gains_erb, 1,
                0.0f, df->work_gains_bin, 1);

    /* Smooth gains over time to prevent musical noise artifacts */
    {
        float alpha = df->smooth_coeff;
        float one_minus_alpha = 1.0f - alpha;
        /* gain_smooth = alpha * new_gain + (1-alpha) * gain_smooth */
        vDSP_vsmul(df->work_gains_bin, 1, &alpha, df->work_gains_bin, 1,
                    (vDSP_Length)DF_FREQ_BINS);
        vDSP_vsma(df->gain_smooth, 1, &one_minus_alpha,
                   df->work_gains_bin, 1, df->work_gains_bin, 1,
                   (vDSP_Length)DF_FREQ_BINS);
        memcpy(df->gain_smooth, df->work_gains_bin,
               (size_t)DF_FREQ_BINS * sizeof(float));
    }

    /* Apply per-bin gains to complex spectrum */
    /* DC: stored in fft_real[0] */
    fft_buf.realp[0] *= df->work_gains_bin[0];
    /* Nyquist: stored in fft_imag[0] */
    fft_buf.imagp[0] *= df->work_gains_bin[half];
    /* Bins 1..(half-1) */
    vDSP_vmul(fft_buf.realp + 1, 1, df->work_gains_bin + 1, 1,
              fft_buf.realp + 1, 1, (vDSP_Length)(half - 1));
    vDSP_vmul(fft_buf.imagp + 1, 1, df->work_gains_bin + 1, 1,
              fft_buf.imagp + 1, 1, (vDSP_Length)(half - 1));

do_ifft:
    /* IFFT — same scaling as noise_gate.c */
    vDSP_fft_zrip(df->fft_setup, &fft_buf, 1, df->log2n, FFT_INVERSE);
    {
        float scale = 1.0f / (2.0f * (float)n);
        float *reconstructed = df->work_recon;
        vDSP_ztoc(&fft_buf, 1, (DSPComplex *)reconstructed, 2,
                   (vDSP_Length)half);
        vDSP_vsmul(reconstructed, 1, &scale, reconstructed, 1,
                    (vDSP_Length)n);

        /* Overlap-add into output accumulation buffer */
        vDSP_vadd(df->overlap_out, 1, reconstructed, 1,
                   df->overlap_out, 1, (vDSP_Length)n);
    }

    /* Extract hop_size samples to output queue */
    int avail = df->output_cap - df->output_len;
    int to_copy = DF_HOP_SIZE < avail ? DF_HOP_SIZE : avail;
    memcpy(df->output_queue + df->output_len, df->overlap_out,
           (size_t)to_copy * sizeof(float));
    df->output_len += to_copy;

    /* Shift overlap buffer left by hop_size */
    memmove(df->overlap_out, df->overlap_out + DF_HOP_SIZE,
            (size_t)(df->overlap_out_cap - DF_HOP_SIZE) * sizeof(float));
    memset(df->overlap_out + df->overlap_out_cap - DF_HOP_SIZE, 0,
           (size_t)DF_HOP_SIZE * sizeof(float));
}

/* ── Public API ──────────────────────────────────────────────────────────── */

DeepFilter *deep_filter_create(int sample_rate, const char *weights_path) {
    if (sample_rate != DF_SAMPLE_RATE) {
        fprintf(stderr, "[deep_filter] Unsupported sample rate: %d (need %d)\n",
                sample_rate, DF_SAMPLE_RATE);
        return NULL;
    }

    DeepFilter *df = (DeepFilter *)calloc(1, sizeof(DeepFilter));
    if (!df) return NULL;

    /* FFT setup */
    df->log2n = 0;
    for (int t = DF_FFT_SIZE; t > 1; t >>= 1) df->log2n++;
    df->fft_setup = vDSP_create_fftsetup(df->log2n, kFFTRadix2);
    if (!df->fft_setup) { free(df); return NULL; }

    /* Hann window */
    df->window = (float *)malloc(DF_FFT_SIZE * sizeof(float));
    if (!df->window) { deep_filter_destroy(df); return NULL; }
    vDSP_hann_window(df->window, DF_FFT_SIZE, vDSP_HANN_DENORM);

    /* Allocate working buffers */
    df->fft_real      = (float *)calloc(DF_HALF_FFT, sizeof(float));
    df->fft_imag      = (float *)calloc(DF_HALF_FFT, sizeof(float));
    df->frame_buf     = (float *)calloc(DF_FFT_SIZE, sizeof(float));
    df->overlap_out_cap = DF_FFT_SIZE * 2;
    df->overlap_out   = (float *)calloc(df->overlap_out_cap, sizeof(float));
    df->output_cap    = sample_rate * 2;
    df->output_queue  = (float *)calloc(df->output_cap, sizeof(float));
    df->work_windowed = (float *)malloc(DF_FFT_SIZE * sizeof(float));
    df->work_mag      = (float *)malloc(DF_FREQ_BINS * sizeof(float));
    df->work_erb      = (float *)malloc(DF_N_ERB * sizeof(float));
    df->work_gains_erb = (float *)malloc(DF_N_ERB * sizeof(float));
    df->work_gains_bin = (float *)malloc(DF_FREQ_BINS * sizeof(float));
    df->gain_smooth   = (float *)malloc(DF_FREQ_BINS * sizeof(float));
    df->work_recon    = (float *)malloc(DF_FFT_SIZE * sizeof(float));
    df->gru_scratch   = (float *)malloc(DF_GRU_GATES * sizeof(float));
    df->gru_scratch2  = (float *)malloc(DF_GRU_GATES * sizeof(float));
    df->gru_input     = (float *)malloc(DF_GRU_HIDDEN * sizeof(float));

    if (!df->fft_real || !df->fft_imag || !df->frame_buf ||
        !df->overlap_out || !df->output_queue || !df->work_windowed ||
        !df->work_mag || !df->work_erb || !df->work_gains_erb ||
        !df->work_gains_bin || !df->gain_smooth || !df->work_recon ||
        !df->gru_scratch || !df->gru_scratch2 || !df->gru_input) {
        deep_filter_destroy(df);
        return NULL;
    }

    /* Initialize smoothed gains to 1.0 (passthrough) */
    for (int i = 0; i < DF_FREQ_BINS; i++) df->gain_smooth[i] = 1.0f;

    /* Default parameters */
    df->strength = 0.8f;
    df->min_gain = powf(10.0f, -30.0f / 20.0f);  /* -30 dB */
    df->smooth_coeff = 0.7f;  /* temporal smoothing */

    /* Load weights or fall back to passthrough */
    if (weights_path) {
        if (load_weights(df, weights_path) != 0) {
            fprintf(stderr, "[deep_filter] Weight loading failed, using passthrough\n");
            df->passthrough = 1;
            /* Generate ERB filter bank for potential future use */
            df->erb_fb = (float *)malloc((size_t)DF_N_ERB * DF_FREQ_BINS * sizeof(float));
            df->erb_fb_inv = (float *)malloc((size_t)DF_FREQ_BINS * DF_N_ERB * sizeof(float));
            if (df->erb_fb && df->erb_fb_inv) {
                compute_erb_filter_bank(df->erb_fb, DF_N_ERB, DF_FREQ_BINS,
                                         DF_SAMPLE_RATE);
                compute_erb_inverse(df->erb_fb, df->erb_fb_inv,
                                     DF_N_ERB, DF_FREQ_BINS);
            }
        } else {
            df->passthrough = 0;
        }
    } else {
        df->passthrough = 1;
        /* Generate ERB filter bank */
        df->erb_fb = (float *)malloc((size_t)DF_N_ERB * DF_FREQ_BINS * sizeof(float));
        df->erb_fb_inv = (float *)malloc((size_t)DF_FREQ_BINS * DF_N_ERB * sizeof(float));
        if (df->erb_fb && df->erb_fb_inv) {
            compute_erb_filter_bank(df->erb_fb, DF_N_ERB, DF_FREQ_BINS,
                                     DF_SAMPLE_RATE);
            compute_erb_inverse(df->erb_fb, df->erb_fb_inv,
                                 DF_N_ERB, DF_FREQ_BINS);
        }
    }

    return df;
}

void deep_filter_destroy(DeepFilter *df) {
    if (!df) return;
    if (df->fft_setup) vDSP_destroy_fftsetup(df->fft_setup);
    free(df->window);
    free(df->fft_real);
    free(df->fft_imag);
    free(df->frame_buf);
    free(df->overlap_out);
    free(df->output_queue);
    free(df->work_windowed);
    free(df->work_mag);
    free(df->work_erb);
    free(df->work_gains_erb);
    free(df->work_gains_bin);
    free(df->gain_smooth);
    free(df->work_recon);
    free(df->gru_scratch);
    free(df->gru_scratch2);
    free(df->gru_input);
    free(df->erb_fb);
    free(df->erb_fb_inv);
    for (int i = 0; i < DF_N_GRU_LAYERS; i++) {
        free_gru_layer(&df->gru[i]);
    }
    free(df->out_w);
    free(df->out_b);
    free(df);
}

void deep_filter_process(DeepFilter *df, float *pcm, int n) {
    if (!df || !pcm || n <= 0) return;

    /* NaN guard: clamp NaN/Inf in input to prevent GRU state poisoning */
    {
        int has_bad = 0;
        for (int i = 0; i < n; i++) {
            if (__builtin_expect(!isfinite(pcm[i]), 0)) {
                pcm[i] = 0.0f;
                has_bad = 1;
            }
        }
        if (has_bad) {
            fprintf(stderr, "[deep_filter] NaN/Inf detected in input, clamped to zero\n");
        }
    }

    /* Ensure output queue can hold at least n + FFT_SIZE samples */
    int needed_cap = n + DF_FFT_SIZE;
    if (needed_cap > df->output_cap) {
        float *new_q = realloc(df->output_queue, (size_t)needed_cap * sizeof(float));
        if (new_q) {
            df->output_queue = new_q;
            df->output_cap = needed_cap;
        } else {
            fprintf(stderr, "[deep_filter] realloc failed for %d samples\n", needed_cap);
            return;
        }
    }

    int read_pos = 0;
    df->output_len = 0;

    while (read_pos < n) {
        int need = DF_FFT_SIZE - df->frame_len;
        int avail = n - read_pos;
        int copy = avail < need ? avail : need;

        memcpy(df->frame_buf + df->frame_len, pcm + read_pos,
               (size_t)copy * sizeof(float));
        df->frame_len += copy;
        read_pos += copy;

        if (df->frame_len >= DF_FFT_SIZE) {
            process_one_frame(df);

            /* Shift frame buffer by hop_size (keep overlap) */
            int remain = DF_FFT_SIZE - DF_HOP_SIZE;
            memmove(df->frame_buf, df->frame_buf + DF_HOP_SIZE,
                    (size_t)remain * sizeof(float));
            df->frame_len = remain;
        }
    }

    /* Copy processed output back to pcm */
    int out_copy = df->output_len < n ? df->output_len : n;
    if (out_copy > 0) {
        memcpy(pcm, df->output_queue, (size_t)out_copy * sizeof(float));
    }
    /* Zero-fill if fewer output samples than input */
    if (out_copy < n) {
        memset(pcm + out_copy, 0, (size_t)(n - out_copy) * sizeof(float));
    }
}

void deep_filter_reset(DeepFilter *df) {
    if (!df) return;
    memset(df->h, 0, sizeof(df->h));
    df->frame_len = 0;
    df->output_len = 0;
    memset(df->overlap_out, 0, (size_t)df->overlap_out_cap * sizeof(float));
    for (int i = 0; i < DF_FREQ_BINS; i++) df->gain_smooth[i] = 1.0f;
}

void deep_filter_set_strength(DeepFilter *df, float strength) {
    if (!df) return;
    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;
    df->strength = strength;
}

void deep_filter_set_min_gain_db(DeepFilter *df, float min_gain_db) {
    if (!df) return;
    if (min_gain_db > 0.0f) min_gain_db = 0.0f;
    if (min_gain_db < -60.0f) min_gain_db = -60.0f;
    df->min_gain = powf(10.0f, min_gain_db / 20.0f);
}
