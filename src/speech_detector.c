/**
 * speech_detector.c — Unified speech detection and end-of-utterance.
 *
 * Wraps native_vad, mimi_endpointer, and fused_eou into a single module
 * that owns all buffer management.
 */

#include "speech_detector.h"
#include "native_vad.h"
#include "mimi_endpointer.h"
#include "fused_eou.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#endif

/* ── Constants ─────────────────────────────────────────────────────────── */

#define VAD_CHUNK        512    /* 32ms @ 16kHz */
#define EP_FRAME_SIZE   1920    /* 80ms @ 24kHz */
#define EP_N_BANDS        80    /* Mel-energy feature dimension */
#define BUF_16K_CAP     2048    /* ~128ms accumulation */
#define BUF_24K_CAP     (EP_FRAME_SIZE * 4)

/* ── Struct ────────────────────────────────────────────────────────────── */

struct SpeechDetector {
    /* Neural VAD */
    NativeVad   *native_vad;

    /* Mimi endpointer */
    MimiEndpointer *mimi_ep;

    /* Fused EOU */
    FusedEOU    *fused_eou;

    /* 16kHz accumulation buffer for VAD */
    float       *buf_16k;
    int          buf_16k_len;

    /* 24kHz accumulation buffer for endpointer */
    float       *buf_24k;
    int          buf_24k_len;

    /* Mel-energy feature scratch */
    float        features[EP_N_BANDS];

    /* Resample scratch: 24kHz → 16kHz */
    float        resample_buf[BUF_24K_CAP];

    /* Anti-aliasing filter scratch for 24→16kHz */
    float       *aa_buf;

    /* Latest probabilities */
    float        last_speech_prob;   /* -1 = no data yet */
    float        last_eot_prob;
};

/* ── Anti-aliasing FIR lowpass for 24→16kHz resampling ─────────────────
 * 15-tap windowed sinc (Hamming), cutoff at 8kHz (Nyquist of 16kHz).
 * Normalized so DC gain = 1.0.  Symmetric: h[k] == h[14-k].           */

static const float aa_fir[15] = {
     0.00314f,  0.00000f, -0.01391f,  0.03004f,  0.00000f,
    -0.11347f,  0.26211f,  0.66419f,  0.26211f, -0.11347f,
     0.00000f,  0.03004f, -0.01391f,  0.00000f,  0.00314f
};
#define AA_FIR_LEN  15
#define AA_FIR_HALF 7

static void apply_aa_lowpass(const float *in, int n, float *out) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int k = 0; k < AA_FIR_LEN; k++) {
            int j = i - AA_FIR_HALF + k;
            if (j < 0) j = 0;
            else if (j >= n) j = n - 1;
            sum += in[j] * aa_fir[k];
        }
        out[i] = sum;
    }
}

/* ── Internal: linear resample 24→16kHz ───────────────────────────────── */

static int resample_24_to_16(const float *in, int n_in,
                             float *out, int max_out)
{
    const double ratio = 24000.0 / 16000.0;  /* 1.5 */
    int n_out = (int)((double)n_in / ratio);
    if (n_out > max_out) n_out = max_out;
    for (int i = 0; i < n_out; i++) {
        double src = (double)i * ratio;
        int idx = (int)src;
        double frac = src - idx;
        if (idx + 1 < n_in)
            out[i] = (float)((1.0 - frac) * in[idx] + frac * in[idx + 1]);
        else
            out[i] = in[idx < n_in ? idx : n_in - 1];
    }
    return n_out;
}

/* ── Internal: feed 16kHz chunks to VAD ───────────────────────────────── */

static void feed_vad_16k(SpeechDetector *sd, const float *pcm16, int n) {
    int space = BUF_16K_CAP - sd->buf_16k_len;
    int to_add = n < space ? n : space;
    memcpy(sd->buf_16k + sd->buf_16k_len, pcm16, (size_t)to_add * sizeof(float));
    sd->buf_16k_len += to_add;

    while (sd->buf_16k_len >= VAD_CHUNK) {
        float p;
        if (sd->native_vad)
            p = native_vad_process(sd->native_vad, sd->buf_16k);
        else
            break;

        if (p >= 0.0f) sd->last_speech_prob = p;

        int rem = sd->buf_16k_len - VAD_CHUNK;
        if (rem > 0)
            memmove(sd->buf_16k, sd->buf_16k + VAD_CHUNK,
                    (size_t)rem * sizeof(float));
        sd->buf_16k_len = rem;
    }
}

/* ── Internal: feed 24kHz frames to endpointer ────────────────────────── */

static void feed_endpointer_24k(SpeechDetector *sd, const float *pcm24, int n) {
    if (!sd->mimi_ep) return;

    int space = BUF_24K_CAP - sd->buf_24k_len;
    int to_add = n < space ? n : space;
    memcpy(sd->buf_24k + sd->buf_24k_len, pcm24, (size_t)to_add * sizeof(float));
    sd->buf_24k_len += to_add;

    while (sd->buf_24k_len >= EP_FRAME_SIZE) {
        int spb = EP_FRAME_SIZE / EP_N_BANDS;
        for (int b = 0; b < EP_N_BANDS; b++) {
#ifdef __APPLE__
            float rms;
            vDSP_rmsqv(sd->buf_24k + b * spb, 1, &rms, spb);
            sd->features[b] = rms;
#else
            float sum = 0.0f;
            for (int j = 0; j < spb; j++) {
                float v = sd->buf_24k[b * spb + j];
                sum += v * v;
            }
            sd->features[b] = sqrtf(sum / spb);
#endif
        }
        mimi_ep_process(sd->mimi_ep, sd->features);
        sd->last_eot_prob = mimi_ep_eot_prob(sd->mimi_ep);

        int rem = sd->buf_24k_len - EP_FRAME_SIZE;
        if (rem > 0)
            memmove(sd->buf_24k, sd->buf_24k + EP_FRAME_SIZE,
                    (size_t)rem * sizeof(float));
        sd->buf_24k_len = rem;
    }
}

/* ── Public API ────────────────────────────────────────────────────────── */

SpeechDetector *speech_detector_create(const SpeechDetectorConfig *cfg) {
    SpeechDetector *sd = (SpeechDetector *)calloc(1, sizeof(*sd));
    if (!sd) return NULL;

    sd->last_speech_prob = -1.0f;
    sd->last_eot_prob = 0.0f;

    /* Allocate buffers */
    sd->buf_16k = (float *)calloc(BUF_16K_CAP, sizeof(float));
    sd->buf_24k = (float *)calloc(BUF_24K_CAP, sizeof(float));
    sd->aa_buf  = (float *)calloc(BUF_24K_CAP, sizeof(float));
    if (!sd->buf_16k || !sd->buf_24k || !sd->aa_buf) {
        speech_detector_destroy(sd);
        return NULL;
    }

    if (!cfg) return sd;

    /* Native VAD (preferred) */
    if (cfg->native_vad_path)
        sd->native_vad = native_vad_create(cfg->native_vad_path);

    /* Mimi endpointer */
    int ldim = cfg->mimi_latent_dim > 0 ? cfg->mimi_latent_dim : EP_N_BANDS;
    int hdim = cfg->mimi_hidden_dim > 0 ? cfg->mimi_hidden_dim : 64;
    float thr = cfg->eot_threshold > 0.0f ? cfg->eot_threshold : 0.7f;
    int consec = cfg->eot_consec_frames > 0 ? cfg->eot_consec_frames : 3;
    sd->mimi_ep = mimi_ep_create(ldim, hdim, thr, consec);
    if (sd->mimi_ep)
        mimi_ep_init_random(sd->mimi_ep, 42);

    /* Fused EOU */
    float fuse_thr = cfg->eot_threshold > 0.0f ? cfg->eot_threshold : 0.6f;
    int fuse_consec = cfg->eot_consec_frames > 0 ? cfg->eot_consec_frames : 2;
    sd->fused_eou = fused_eou_create(fuse_thr, fuse_consec, 80.0f);

    /* Disable mimi weight — endpointer uses random/uninitialized weights.
     * Redistribute weight to energy + STT only. */
    if (sd->fused_eou)
        fused_eou_set_weights(sd->fused_eou, 0.35f, 0.0f, 0.65f);

    return sd;
}

void speech_detector_destroy(SpeechDetector *sd) {
    if (!sd) return;
    if (sd->native_vad) native_vad_destroy(sd->native_vad);
    if (sd->mimi_ep) mimi_ep_destroy(sd->mimi_ep);
    if (sd->fused_eou) fused_eou_destroy(sd->fused_eou);
    free(sd->buf_16k);
    free(sd->buf_24k);
    free(sd->aa_buf);
    free(sd);
}

void speech_detector_reset(SpeechDetector *sd) {
    if (!sd) return;
    if (sd->native_vad) native_vad_reset(sd->native_vad);
    if (sd->mimi_ep) mimi_ep_reset(sd->mimi_ep);
    if (sd->fused_eou) fused_eou_reset(sd->fused_eou);
    sd->buf_16k_len = 0;
    sd->buf_24k_len = 0;
    sd->last_speech_prob = -1.0f;
    sd->last_eot_prob = 0.0f;
}

void speech_detector_feed(SpeechDetector *sd,
                          const float *pcm24, int n_samples)
{
    if (!sd || !pcm24 || n_samples <= 0) return;

    /* Feed 24kHz to endpointer */
    feed_endpointer_24k(sd, pcm24, n_samples);

    /* Anti-alias filter then resample 24→16kHz for VAD */
    if (sd->native_vad && sd->aa_buf) {
        int n_filt = n_samples < BUF_24K_CAP ? n_samples : BUF_24K_CAP;
        apply_aa_lowpass(pcm24, n_filt, sd->aa_buf);
        int n16 = resample_24_to_16(sd->aa_buf, n_filt,
                                    sd->resample_buf, BUF_24K_CAP);
        if (n16 > 0) feed_vad_16k(sd, sd->resample_buf, n16);
    }
}

void speech_detector_feed_16k(SpeechDetector *sd,
                              const float *pcm16, int n_samples)
{
    if (!sd || !pcm16 || n_samples <= 0) return;
    if (sd->native_vad)
        feed_vad_16k(sd, pcm16, n_samples);
}

float speech_detector_speech_prob(const SpeechDetector *sd) {
    return sd ? sd->last_speech_prob : -1.0f;
}

int speech_detector_speech_active(const SpeechDetector *sd, int energy_vad) {
    if (!sd) return energy_vad >= 1;

    int vad = energy_vad;
    if ((sd->native_vad) && sd->last_speech_prob >= 0.0f) {
        if (sd->last_speech_prob > 0.5f)  vad = 1;
        else if (sd->last_speech_prob < 0.1f) vad = 0;
    }
    return vad >= 1;
}

float speech_detector_eot_prob(const SpeechDetector *sd) {
    if (!sd || !sd->mimi_ep) return 0.0f;
    return sd->last_eot_prob;
}

EOUResult speech_detector_eou(SpeechDetector *sd,
                              int energy_vad, float stt_eou_prob)
{
    EOUResult empty = {0};
    if (!sd) return empty;

    EOUSignals sig = {0};

    /* Signal 1: Energy VAD → silence probability */
    sig.energy_signal = (energy_vad == 3) ? 1.0f :
                        (energy_vad == 2) ? 0.0f :
                        (energy_vad == 0) ? 0.5f : 0.0f;

    /* Neural VAD reinforcement/softening */
    if ((sd->native_vad) && sd->last_speech_prob >= 0.0f) {
        if (sd->last_speech_prob < 0.1f)
            sig.energy_signal = (sig.energy_signal + 1.0f) * 0.5f;
        else if (sd->last_speech_prob > 0.5f)
            sig.energy_signal *= 0.5f;
    }

    /* Signal 2: Mimi endpointer — disabled (random/uninitialized weights) */
    sig.mimi_eot_prob = 0.0f;

    /* Signal 3: STT semantic EOU */
    sig.stt_eou_prob = stt_eou_prob;

    /* Fuse */
    if (sd->fused_eou)
        return fused_eou_process(sd->fused_eou, sig);

    return empty;
}

int speech_detector_has_vad(const SpeechDetector *sd) {
    return sd && (sd->native_vad);
}

int speech_detector_has_endpointer(const SpeechDetector *sd) {
    return sd && sd->mimi_ep;
}
