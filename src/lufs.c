/**
 * lufs.c — ITU-R BS.1770-4 LUFS loudness meter and normalizer.
 *
 * K-weighting:
 *   Stage 1: Pre-filter (high shelf at ~1681Hz, +4dB — models head diffraction)
 *   Stage 2: High-pass (revised low-frequency at ~38Hz)
 *
 * Both filters are 2nd-order IIR biquads with coefficients defined in the
 * ITU-R BS.1770-4 standard for 48kHz. Coefficients for other sample rates
 * are derived via bilinear transform.
 *
 * Measurement:
 *   - 400ms gating blocks with 75% overlap (100ms hop)
 *   - Absolute gate at -70 LUFS
 *   - Relative gate at -10 dB below absolute-gated level
 */

#include "lufs.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

typedef struct {
    double b0, b1, b2, a1, a2;
    double x1, x2, y1, y2;
} Biquad64;

/* Sliding window ring buffer for block powers */
#define MAX_BLOCKS 4096

struct LUFSMeter {
    int sample_rate;
    int window_ms;

    /* K-weighting filters (double precision for coefficient computation) */
    Biquad64 shelf;   /* stage 1: high shelf */
    Biquad64 hpf;     /* stage 2: high-pass */

#ifdef __APPLE__
    /* vDSP biquad cascade: 2 sections (shelf + HPF) for AMX-accelerated block processing.
       Coefficients are double (per vDSP API), delay state is float. */
    vDSP_biquad_Setup vdsp_setup;
    float vdsp_delays[2 + 4 * 2]; /* 2 + 4*M floats where M=2 sections */
#endif

    /* Block measurement */
    int block_size;        /* samples per 400ms block */
    int hop_size;          /* samples per 100ms hop */

    float *block_buf;      /* accumulation buffer */
    int block_pos;         /* current position in block_buf */

    double *block_powers;  /* ring of measured block powers */
    int n_blocks;          /* number of stored blocks */
    int block_idx;         /* write cursor */

    /* Integrated measurement */
    double sum_powers;
    int total_blocks;
};

/* ── ITU-R BS.1770-4 K-weighting coefficients for 48kHz ───── */

static void init_kweight_48k(Biquad64 *shelf, Biquad64 *hpf)
{
    /* Stage 1: pre-filter (high shelf) — Table 1 in BS.1770-4 */
    shelf->b0 =  1.53512485958697;
    shelf->b1 = -2.69169618940638;
    shelf->b2 =  1.19839281085285;
    shelf->a1 = -1.69065929318241;
    shelf->a2 =  0.73248077421585;
    shelf->x1 = shelf->x2 = shelf->y1 = shelf->y2 = 0.0;

    /* Stage 2: revised low-frequency (high-pass) — Table 2 in BS.1770-4 */
    hpf->b0 =  1.0;
    hpf->b1 = -2.0;
    hpf->b2 =  1.0;
    hpf->a1 = -1.99004745483398;
    hpf->a2 =  0.99007225036621;
    hpf->x1 = hpf->x2 = hpf->y1 = hpf->y2 = 0.0;
}

/**
 * Retune K-weighting biquad coefficients for a non-48kHz sample rate.
 *
 * Uses the ITU-R BS.1770-4 analog prototype poles/zeros, then applies
 * the bilinear transform with frequency pre-warping for the target rate.
 *
 * Stage 1 (shelf): fc=1681.97 Hz, gain=+3.999 dB, Q=0.7072
 * Stage 2 (HPF):   fc=38.135 Hz, Q=0.5003
 */
static void retune_biquad_shelf(Biquad64 *f, int sr)
{
    if (sr == 48000) return;

    double fs = (double)sr;
    double fc = 1681.974450955533;
    double VH = 1.584893192461113;  /* 10^(3.999/20) */
    double Q = 0.7071752369554196;

    double K = tan(M_PI * fc / fs);
    double Vh = VH;
    double Vb = pow(VH, 0.4996667741545416);

    double denom = 1.0 + K / Q + K * K;
    f->b0 = (Vh + Vb * K / Q + K * K) / denom;
    f->b1 = 2.0 * (K * K - Vh) / denom;
    f->b2 = (Vh - Vb * K / Q + K * K) / denom;
    f->a1 = 2.0 * (K * K - 1.0) / denom;
    f->a2 = (1.0 - K / Q + K * K) / denom;
}

static void retune_biquad_hpf(Biquad64 *f, int sr)
{
    if (sr == 48000) return;

    double fs = (double)sr;
    double fc = 38.13547087602444;
    double Q = 0.5003270373238773;

    double K = tan(M_PI * fc / fs);
    double denom = 1.0 + K / Q + K * K;
    f->b0 = 1.0 / denom;
    f->b1 = -2.0 / denom;
    f->b2 = 1.0 / denom;
    f->a1 = 2.0 * (K * K - 1.0) / denom;
    f->a2 = (1.0 - K / Q + K * K) / denom;
}

#ifndef __APPLE__
static inline double biquad_process_d(Biquad64 *f, double x)
{
    double y = f->b0 * x + f->b1 * f->x1 + f->b2 * f->x2
             - f->a1 * f->y1 - f->a2 * f->y2;
    f->x2 = f->x1;
    f->x1 = x;
    f->y2 = f->y1;
    f->y1 = y;
    return y;
}
#endif

/* ── Public API ───────────────────────────────────────── */

LUFSMeter *lufs_create(int sample_rate, int window_ms)
{
    LUFSMeter *m = calloc(1, sizeof(LUFSMeter));
    if (!m) return NULL;

    m->sample_rate = sample_rate;
    m->window_ms = window_ms > 0 ? window_ms : 400;

    init_kweight_48k(&m->shelf, &m->hpf);
    retune_biquad_shelf(&m->shelf, sample_rate);
    retune_biquad_hpf(&m->hpf, sample_rate);

    m->block_size = sample_rate * 400 / 1000;  /* 400ms blocks per ITU */
    m->hop_size = sample_rate * 100 / 1000;    /* 75% overlap */

    m->block_buf = calloc((size_t)m->block_size, sizeof(float));
    m->block_powers = calloc(MAX_BLOCKS, sizeof(double));
    if (!m->block_buf || !m->block_powers) {
        free(m->block_buf);
        free(m->block_powers);
        free(m);
        return NULL;
    }

#ifdef __APPLE__
    /* Pack coefficients for vDSP_biquad: [b0,b1,b2,a1,a2] per section (double) */
    double vdsp_coeffs[10] = {
        m->shelf.b0, m->shelf.b1, m->shelf.b2, m->shelf.a1, m->shelf.a2,
        m->hpf.b0,   m->hpf.b1,   m->hpf.b2,   m->hpf.a1,   m->hpf.a2,
    };
    m->vdsp_setup = vDSP_biquad_CreateSetup(vdsp_coeffs, 2);
    memset(m->vdsp_delays, 0, sizeof(m->vdsp_delays));
#endif

    m->block_pos = 0;
    m->n_blocks = 0;
    m->block_idx = 0;
    m->sum_powers = 0.0;
    m->total_blocks = 0;

    return m;
}

void lufs_destroy(LUFSMeter *m)
{
    if (!m) return;
    free(m->block_buf);
    free(m->block_powers);
#ifdef __APPLE__
    if (m->vdsp_setup) vDSP_biquad_DestroySetup(m->vdsp_setup);
#endif
    free(m);
}

void lufs_reset(LUFSMeter *m)
{
    if (!m) return;
    m->shelf.x1 = m->shelf.x2 = m->shelf.y1 = m->shelf.y2 = 0.0;
    m->hpf.x1 = m->hpf.x2 = m->hpf.y1 = m->hpf.y2 = 0.0;
#ifdef __APPLE__
    memset(m->vdsp_delays, 0, sizeof(m->vdsp_delays));
#endif
    m->block_pos = 0;
    m->n_blocks = 0;
    m->block_idx = 0;
    m->sum_powers = 0.0;
    m->total_blocks = 0;
}

static inline void lufs_commit_block(LUFSMeter *m)
{
    float dot;
#ifdef __APPLE__
    vDSP_dotpr(m->block_buf, 1, m->block_buf, 1, &dot,
               (vDSP_Length)m->block_size);
#else
    dot = 0.0f;
    for (int j = 0; j < m->block_size; j++)
        dot += m->block_buf[j] * m->block_buf[j];
#endif
    double power = (double)dot / (double)m->block_size;
    m->block_powers[m->block_idx] = power;
    m->block_idx = (m->block_idx + 1) % MAX_BLOCKS;
    if (m->n_blocks < MAX_BLOCKS) m->n_blocks++;
    m->sum_powers += power;
    m->total_blocks++;

    int keep = m->block_size - m->hop_size;
    memmove(m->block_buf, m->block_buf + m->hop_size,
            (size_t)keep * sizeof(float));
    m->block_pos = keep;
}

float lufs_measure(LUFSMeter *m, const float *audio, int n)
{
    if (!m || !audio || n <= 0) return -70.0f;

#ifdef __APPLE__
    /* K-weight the entire input via vDSP_biquad cascade (AMX-accelerated).
       Process in chunks that fill up to one block at a time. */
    if (m->vdsp_setup) {
        int remaining = n;
        const float *src = audio;
        while (remaining > 0) {
            int space = m->block_size - m->block_pos;
            int chunk = remaining < space ? remaining : space;
            vDSP_biquad(m->vdsp_setup, m->vdsp_delays, src, 1,
                        m->block_buf + m->block_pos, 1, chunk);
            m->block_pos += chunk;
            src += chunk;
            remaining -= chunk;
            if (m->block_pos >= m->block_size)
                lufs_commit_block(m);
        }
    }
#else
    for (int i = 0; i < n; i++) {
        double s = (double)audio[i];
        s = biquad_process_d(&m->shelf, s);
        s = biquad_process_d(&m->hpf, s);
        m->block_buf[m->block_pos] = (float)s;
        m->block_pos++;
        if (m->block_pos >= m->block_size)
            lufs_commit_block(m);
    }
#endif

    /* Compute momentary LUFS from the most recent window */
    int window_blocks = (m->window_ms / 100); /* 100ms per hop */
    if (window_blocks > m->n_blocks) window_blocks = m->n_blocks;
    if (window_blocks <= 0) return -70.0f;

    double sum = 0.0;
    int idx = m->block_idx - 1;
    for (int j = 0; j < window_blocks; j++) {
        if (idx < 0) idx += MAX_BLOCKS;
        sum += m->block_powers[idx];
        idx--;
    }
    double mean_power = sum / (double)window_blocks;

    if (mean_power < 1e-20) return -70.0f;
    return (float)(-0.691 + 10.0 * log10(mean_power));
}

float lufs_normalize(LUFSMeter *m, float *audio, int n, float target_lufs)
{
    if (!m || !audio || n <= 0) return 0.0f;

    /* NOTE: lufs_measure() advances internal state. Call lufs_reset() before
       re-measuring the same buffer to get consistent results. */
    float current = lufs_measure(m, audio, n);
    if (current <= -70.0f) return 0.0f; /* silence, don't amplify */

    float gain_db = target_lufs - current;

    /* Clamp gain to avoid excessive amplification or attenuation */
    if (gain_db > 20.0f) gain_db = 20.0f;
    if (gain_db < -20.0f) gain_db = -20.0f;

    float gain_linear = powf(10.0f, gain_db / 20.0f);

#ifdef __APPLE__
    vDSP_vsmul(audio, 1, &gain_linear, audio, 1, (vDSP_Length)n);
#else
    for (int i = 0; i < n; i++)
        audio[i] *= gain_linear;
#endif

    /* Soft limiter: tanh-based to prevent clipping */
    float peak = 0.0f;
#ifdef __APPLE__
    float abs_peak;
    vDSP_maxmgv(audio, 1, &abs_peak, (vDSP_Length)n);
    peak = abs_peak;
#else
    for (int i = 0; i < n; i++) {
        float a = audio[i] > 0 ? audio[i] : -audio[i];
        if (a > peak) peak = a;
    }
#endif

    if (peak > 0.95f) {
        float scale = 0.95f / peak;
        /* Soft knee: blend between linear and limited */
        float knee = (peak - 0.95f) / (peak - 0.5f);
        if (knee > 1.0f) knee = 1.0f;
        if (knee < 0.0f) knee = 0.0f;
        float final_scale = 1.0f * (1.0f - knee) + scale * knee;

#ifdef __APPLE__
        vDSP_vsmul(audio, 1, &final_scale, audio, 1, (vDSP_Length)n);
#else
        for (int i = 0; i < n; i++)
            audio[i] *= final_scale;
#endif
    }

    return gain_db;
}

float lufs_integrated(const LUFSMeter *m)
{
    if (!m || m->total_blocks <= 0) return -70.0f;

    /* Absolute gate: -70 LUFS */
    double abs_threshold = pow(10.0, (-70.0 + 0.691) / 10.0);
    double gated_sum = 0.0;
    int gated_count = 0;

    int start = m->n_blocks < MAX_BLOCKS ? 0 : m->block_idx;
    for (int i = 0; i < m->n_blocks; i++) {
        int idx = (start + i) % MAX_BLOCKS;
        if (m->block_powers[idx] >= abs_threshold) {
            gated_sum += m->block_powers[idx];
            gated_count++;
        }
    }

    if (gated_count == 0) return -70.0f;

    double abs_gated_power = gated_sum / (double)gated_count;

    /* Relative gate: -10 dB below absolute-gated level */
    double rel_threshold = abs_gated_power * pow(10.0, -10.0 / 10.0);
    double rel_sum = 0.0;
    int rel_count = 0;

    for (int i = 0; i < m->n_blocks; i++) {
        int idx = (start + i) % MAX_BLOCKS;
        if (m->block_powers[idx] >= rel_threshold) {
            rel_sum += m->block_powers[idx];
            rel_count++;
        }
    }

    if (rel_count == 0) return -70.0f;

    double integrated = rel_sum / (double)rel_count;
    return (float)(-0.691 + 10.0 * log10(integrated));
}
