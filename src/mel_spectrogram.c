/**
 * mel_spectrogram.c — Streaming log-mel spectrogram via Apple Accelerate.
 *
 * Pipeline per frame:
 *   PCM → Hann window → vDSP real FFT → power spectrum → mel filterbank → log
 *
 * All heavy operations run on the AMX coprocessor through vDSP.
 * The mel filterbank is pre-computed as a sparse matrix (triangular filters).
 *
 * Build:
 *   cc -O3 -shared -fPIC -arch arm64 -framework Accelerate \
 *      -install_name @rpath/libmel_spectrogram.dylib \
 *      -o libmel_spectrogram.dylib mel_spectrogram.c
 */

#include "mel_spectrogram.h"

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Mel Spectrogram State
 * ═══════════════════════════════════════════════════════════════════════════ */

struct MelSpectrogram {
    MelConfig cfg;

    /* Pre-computed */
    float *window;          /* Hann window [win_length] */
    float *mel_bank;        /* Mel filterbank [n_mels * n_bins] where n_bins = n_fft/2 + 1 */
    int    n_bins;          /* n_fft / 2 + 1 */

    /* vDSP DFT (modern API, replaces deprecated FFTSetup) */
    vDSP_DFT_Setup dft_setup;
    DSPSplitComplex fft_split;
    float *fft_real;        /* [n_fft / 2] */
    float *fft_imag;        /* [n_fft / 2] */

    /* Streaming PCM accumulator */
    float *pcm_buf;
    int    pcm_len;
    int    pcm_cap;

    /* Working buffers */
    float *windowed;        /* [n_fft] */
    float *power_spec;      /* [n_bins] */
    float *mel_frame;       /* [n_mels] */

    /* Pre-emphasis state */
    float preemph_last;     /* Last sample from previous call (for streaming) */
    int   preemph_init;     /* 1 if preemph_last has been initialized */

    /* Pre-allocated pre-emphasis buffer (avoids malloc per mel_process call) */
    float *preemph_buf;
    int    preemph_cap;
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Mel scale conversion
 * ═══════════════════════════════════════════════════════════════════════════ */

static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Pre-compute mel filterbank (triangular filters)
 *
 * Each of n_mels filters is a triangle spanning 3 center frequencies.
 * Output: row-major matrix [n_mels, n_bins].
 * ═══════════════════════════════════════════════════════════════════════════ */

static int build_mel_filterbank(float *bank, int n_mels, int n_fft,
                                 int sample_rate, float fmin, float fmax,
                                 int slaney_norm) {
    int n_bins = n_fft / 2 + 1;
    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);

    int n_pts = n_mels + 2;
    float *mel_f = (float *)malloc(n_pts * sizeof(float));
    float *fft_freqs = (float *)malloc(n_bins * sizeof(float));
    if (!mel_f || !fft_freqs) {
        free(mel_f);
        free(fft_freqs);
        return -1;
    }

    for (int i = 0; i < n_pts; i++)
        mel_f[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (n_pts - 1));
    for (int k = 0; k < n_bins; k++)
        fft_freqs[k] = (float)k * (float)sample_rate / (float)n_fft;

    memset(bank, 0, (size_t)n_mels * n_bins * sizeof(float));

    for (int m = 0; m < n_mels; m++) {
        float f_left   = mel_f[m];
        float f_center = mel_f[m + 1];
        float f_right  = mel_f[m + 2];

        for (int k = 0; k < n_bins; k++) {
            float freq = fft_freqs[k];
            if (freq >= f_left && freq <= f_center && f_center > f_left)
                bank[m * n_bins + k] = (freq - f_left) / (f_center - f_left);
            else if (freq > f_center && freq <= f_right && f_right > f_center)
                bank[m * n_bins + k] = (f_right - freq) / (f_right - f_center);
        }

        if (slaney_norm) {
            float width = f_right - f_left;
            if (width > 0.0f) {
                float scale = 2.0f / width;
                vDSP_vsmul(bank + m * n_bins, 1, &scale, bank + m * n_bins, 1, n_bins);
            }
        }
    }

    free(mel_f);
    free(fft_freqs);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════════ */

void mel_config_default(MelConfig *cfg) {
    cfg->sample_rate = 16000;
    cfg->n_fft       = 512;
    cfg->hop_length  = 160;
    cfg->win_length  = 400;
    cfg->n_mels      = 80;
    cfg->fmin         = 0.0f;
    cfg->fmax         = 0.0f;  /* 0 → use sample_rate / 2 */
    cfg->log_floor    = 5.96046448e-8f;
    cfg->preemph      = 0.0f;
    cfg->slaney_norm  = 1;
    cfg->periodic_window = 0;
}

MelSpectrogram *mel_create(const MelConfig *cfg) {
    if (!cfg || cfg->n_fft <= 0 || (cfg->n_fft & (cfg->n_fft - 1)) != 0)
        return NULL;
    if (cfg->win_length > cfg->n_fft || cfg->hop_length <= 0)
        return NULL;

    MelSpectrogram *mel = (MelSpectrogram *)calloc(1, sizeof(MelSpectrogram));
    if (!mel) return NULL;

    mel->cfg = *cfg;
    if (mel->cfg.fmax <= 0.0f)
        mel->cfg.fmax = (float)mel->cfg.sample_rate / 2.0f;
    if (mel->cfg.log_floor <= 0.0f)
        mel->cfg.log_floor = 1e-10f;

    int n_fft = mel->cfg.n_fft;
    mel->n_bins = n_fft / 2 + 1;

    /* Hann window — symmetric, matching NeMo / librosa / PyTorch(periodic=False).
       vDSP_HANN_NORM scales to unit energy which inflates the power spectrum by 8/3;
       compute the standard formula directly instead. */
    mel->window = (float *)malloc(cfg->win_length * sizeof(float));
    if (!mel->window) goto fail;
    {
        float denom = cfg->periodic_window ? (float)cfg->win_length : (float)(cfg->win_length - 1);
        for (int i = 0; i < cfg->win_length; i++)
            mel->window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / denom));
    }

    /* Mel filterbank */
    mel->mel_bank = (float *)calloc((size_t)mel->cfg.n_mels * mel->n_bins, sizeof(float));
    if (!mel->mel_bank) goto fail;
    if (build_mel_filterbank(mel->mel_bank, mel->cfg.n_mels, n_fft,
                             mel->cfg.sample_rate, mel->cfg.fmin, mel->cfg.fmax,
                             mel->cfg.slaney_norm) != 0) {
        goto fail;
    }

    /* vDSP DFT setup (modern API — better AMX scheduling, non-power-of-2 support) */
    mel->dft_setup = vDSP_DFT_zrop_CreateSetup(NULL, n_fft, vDSP_DFT_FORWARD);
    if (!mel->dft_setup) goto fail;

    mel->fft_real = (float *)calloc(n_fft / 2, sizeof(float));
    mel->fft_imag = (float *)calloc(n_fft / 2, sizeof(float));
    if (!mel->fft_real || !mel->fft_imag) goto fail;
    mel->fft_split.realp = mel->fft_real;
    mel->fft_split.imagp = mel->fft_imag;

    /* PCM accumulator — hold up to 4 seconds of audio */
    mel->pcm_cap = mel->cfg.sample_rate * 4;
    mel->pcm_buf = (float *)calloc(mel->pcm_cap, sizeof(float));
    if (!mel->pcm_buf) goto fail;
    mel->pcm_len = 0;

    /* Working buffers */
    mel->windowed   = (float *)calloc(n_fft, sizeof(float));
    mel->power_spec = (float *)calloc(mel->n_bins, sizeof(float));
    mel->mel_frame  = (float *)calloc(mel->cfg.n_mels, sizeof(float));
    if (!mel->windowed || !mel->power_spec || !mel->mel_frame) goto fail;

    return mel;

fail:
    mel_destroy(mel);
    return NULL;
}

void mel_destroy(MelSpectrogram *mel) {
    if (!mel) return;
    if (mel->dft_setup) vDSP_DFT_DestroySetup(mel->dft_setup);
    free(mel->window);
    free(mel->mel_bank);
    free(mel->fft_real);
    free(mel->fft_imag);
    free(mel->pcm_buf);
    free(mel->windowed);
    free(mel->power_spec);
    free(mel->mel_frame);
    free(mel->preemph_buf);
    free(mel);
}

/**
 * Extract one mel frame from a windowed segment of PCM.
 *
 * Steps:
 *   1. Apply Hann window               (vDSP_vmul)
 *   2. Zero-pad to n_fft               (memset)
 *   3. Real FFT                         (vDSP_fft_zrip)
 *   4. Power spectrum |X[k]|^2          (vDSP_zvmags)
 *   5. Mel filterbank multiply          (cblas_sgemv — runs on AMX)
 *   6. Log                              (vvlog10f + scale)
 */
static void extract_one_frame(MelSpectrogram *mel, const float *pcm_frame) {
    int n_fft      = mel->cfg.n_fft;
    int win_length = mel->cfg.win_length;
    int n_mels     = mel->cfg.n_mels;
    int n_bins     = mel->n_bins;

    /* 1. Window the PCM */
    memset(mel->windowed, 0, n_fft * sizeof(float));
    vDSP_vmul(pcm_frame, 1, mel->window, 1, mel->windowed, 1, win_length);

    /* 2. Pack into split complex for vDSP real DFT */
    vDSP_ctoz((const DSPComplex *)mel->windowed, 2,
              &mel->fft_split, 1, n_fft / 2);

    /* 3. Forward real DFT (in-place via modern vDSP_DFT API) */
    vDSP_DFT_Execute(mel->dft_setup,
                     mel->fft_split.realp, mel->fft_split.imagp,
                     mel->fft_split.realp, mel->fft_split.imagp);

    /* 4. Power spectrum: |X[k]|^2 = real^2 + imag^2
     *    Bin 0 (DC) and bin N/2 (Nyquist) are packed in realp[0] and imagp[0]. */
    float dc_power  = mel->fft_split.realp[0] * mel->fft_split.realp[0];
    float nyq_power = mel->fft_split.imagp[0] * mel->fft_split.imagp[0];

    /* Bins 1..N/2-1: offset past the packed DC/Nyquist at index 0 */
    DSPSplitComplex bins_offset = { mel->fft_split.realp + 1, mel->fft_split.imagp + 1 };
    vDSP_zvmags(&bins_offset, 1, mel->power_spec + 1, 1, n_fft / 2 - 1);

    /* Fixup DC and Nyquist */
    mel->power_spec[0]          = dc_power;
    mel->power_spec[n_bins - 1] = nyq_power;

    /* vDSP_DFT_Execute (modern API) returns correctly-scaled results —
       no additional scaling needed. */

    /* 5. Mel filterbank: mel_frame = mel_bank @ power_spec
     *    mel_bank is [n_mels, n_bins], power_spec is [n_bins], output is [n_mels]
     *    cblas_sgemv runs on the AMX coprocessor. */
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                n_mels, n_bins,
                1.0f, mel->mel_bank, n_bins,
                mel->power_spec, 1,
                0.0f, mel->mel_frame, 1);

    /* 6. Log mel: log(mel + eps), NeMo default log_zero_guard_value = 2^-24 */
    float eps = mel->cfg.log_floor;
    vDSP_vsadd(mel->mel_frame, 1, &eps, mel->mel_frame, 1, n_mels);
    int n = n_mels;
    vvlogf(mel->mel_frame, mel->mel_frame, &n);
}

int mel_process(MelSpectrogram *mel, const float *pcm, int n_samples,
                float *out, int max_frames) {
    if (!mel || !pcm || !out || n_samples <= 0 || max_frames <= 0)
        return -1;

    int hop = mel->cfg.hop_length;
    int win = mel->cfg.win_length;
    int n_mels = mel->cfg.n_mels;
    int frames_out = 0;

    /* Apply pre-emphasis: y[t] = x[t] - coeff * x[t-1] (using pre-allocated buffer) */
    const float *src = pcm;
    if (mel->cfg.preemph > 0.0f) {
        if (n_samples > mel->preemph_cap) {
            free(mel->preemph_buf);
            mel->preemph_cap = n_samples + mel->cfg.sample_rate;
            mel->preemph_buf = (float *)malloc(mel->preemph_cap * sizeof(float));
            if (!mel->preemph_buf) { mel->preemph_cap = 0; return -1; }
        }
        float prev = mel->preemph_init ? mel->preemph_last : pcm[0];
        float coeff = mel->cfg.preemph;
        for (int i = 0; i < n_samples; i++) {
            mel->preemph_buf[i] = pcm[i] - coeff * prev;
            prev = pcm[i];
        }
        mel->preemph_last = pcm[n_samples - 1];
        mel->preemph_init = 1;
        src = mel->preemph_buf;
    }

    /* Center padding: on first call, prepend n_fft/2 zeros (matches torch.stft center=True) */
    int center_pad = 0;
    if (mel->pcm_len == 0)
        center_pad = mel->cfg.n_fft / 2;

    int total_new = center_pad + n_samples;
    int space = mel->pcm_cap - mel->pcm_len;
    if (total_new > space) {
        int new_cap = mel->pcm_len + total_new + mel->cfg.sample_rate;
        float *new_buf = (float *)realloc(mel->pcm_buf, new_cap * sizeof(float));
        if (!new_buf) return -1;
        mel->pcm_buf = new_buf;
        mel->pcm_cap = new_cap;
    }
    if (center_pad > 0) {
        memset(mel->pcm_buf + mel->pcm_len, 0, center_pad * sizeof(float));
        mel->pcm_len += center_pad;
    }
    memcpy(mel->pcm_buf + mel->pcm_len, src, n_samples * sizeof(float));
    mel->pcm_len += n_samples;

    /* Extract frames: need at least win_length samples per frame */
    int pos = 0;
    while (pos + win <= mel->pcm_len && frames_out < max_frames) {
        extract_one_frame(mel, mel->pcm_buf + pos);
        memcpy(out + frames_out * n_mels, mel->mel_frame, n_mels * sizeof(float));
        frames_out++;
        pos += hop;
    }

    /* Shift unconsumed samples to front of accumulator */
    if (pos > 0) {
        int remaining = mel->pcm_len - pos;
        if (remaining > 0)
            memmove(mel->pcm_buf, mel->pcm_buf + pos, remaining * sizeof(float));
        mel->pcm_len = remaining;
    }

    return frames_out;
}

void mel_reset(MelSpectrogram *mel) {
    if (!mel) return;
    mel->pcm_len = 0;
    mel->preemph_last = 0.0f;
    mel->preemph_init = 0;
}

int mel_n_mels(const MelSpectrogram *mel) {
    return mel ? mel->cfg.n_mels : 0;
}

int mel_hop_length(const MelSpectrogram *mel) {
    return mel ? mel->cfg.hop_length : 0;
}
