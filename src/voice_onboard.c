/**
 * voice_onboard.c — Real-time voice onboarding for prosody transfer.
 *
 * Captures mic audio, extracts speaker embedding (via external ONNX encoder)
 * and prosody profile (F0, energy, rate) for voice cloning.
 */

#include "voice_onboard.h"
#include "speaker_encoder.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#define MAX_CAPTURE_SAMPLES (16000 * 30) /* 30 seconds max */
#define MIN_SPEECH_SAMPLES  (16000 * 1)  /* 1 second min */

struct VoiceOnboardSession {
    float *audio_buf;
    int    buf_len;
    int    buf_cap;
    int    sample_rate;
    float  target_duration;
    int    target_samples;
    const char *encoder_path;
    int    done;
};

VoiceOnboardSession *voice_onboard_create(
    const char *speaker_encoder_path,
    float capture_duration_sec,
    int sample_rate)
{
    VoiceOnboardSession *s = (VoiceOnboardSession *)calloc(1, sizeof(*s));
    if (!s) return NULL;

    if (capture_duration_sec <= 0) capture_duration_sec = 5.0f;
    if (sample_rate <= 0) sample_rate = 16000;

    s->sample_rate = sample_rate;
    s->target_duration = capture_duration_sec;
    s->target_samples = (int)(capture_duration_sec * sample_rate);
    if (s->target_samples > MAX_CAPTURE_SAMPLES)
        s->target_samples = MAX_CAPTURE_SAMPLES;

    s->buf_cap = s->target_samples;
    s->audio_buf = (float *)calloc((size_t)s->buf_cap, sizeof(float));
    if (!s->audio_buf) { free(s); return NULL; }

    s->encoder_path = speaker_encoder_path;
    return s;
}

int voice_onboard_feed(VoiceOnboardSession *session,
                       const float *pcm, int n_samples)
{
    if (!session || session->done || !pcm || n_samples <= 0) return 0;

    int room = session->buf_cap - session->buf_len;
    int to_copy = n_samples < room ? n_samples : room;
    if (to_copy > 0) {
        memcpy(session->audio_buf + session->buf_len, pcm,
               (size_t)to_copy * sizeof(float));
        session->buf_len += to_copy;
    }

    if (session->buf_len >= session->target_samples) {
        session->done = 1;
        return 1;
    }
    return 0;
}

float voice_onboard_progress(const VoiceOnboardSession *session) {
    if (!session || session->target_samples <= 0) return 0.0f;
    float p = (float)session->buf_len / (float)session->target_samples;
    return p > 1.0f ? 1.0f : p;
}

float voice_onboard_estimate_f0(const float *pcm, int n_samples, int sample_rate) {
    if (!pcm || n_samples < 512 || sample_rate <= 0) return 0.0f;

    /* Autocorrelation-based pitch estimation */
    int min_lag = sample_rate / 500; /* 500 Hz max */
    int max_lag = sample_rate / 60;  /* 60 Hz min */
    if (max_lag > n_samples / 2) max_lag = n_samples / 2;
    if (min_lag >= max_lag) return 0.0f;

    float best_corr = 0.0f;
    int best_lag = 0;

    /* Compute RMS for normalization */
    float rms;
    vDSP_rmsqv(pcm, 1, &rms, (vDSP_Length)n_samples);
    if (rms < 1e-6f) return 0.0f;

    for (int lag = min_lag; lag <= max_lag; lag++) {
        float corr = 0.0f;
        vDSP_dotpr(pcm, 1, pcm + lag, 1, &corr,
                   (vDSP_Length)(n_samples - lag));
        corr /= (float)(n_samples - lag);
        if (corr > best_corr) {
            best_corr = corr;
            best_lag = lag;
        }
    }

    if (best_lag <= 0 || best_corr < rms * rms * 0.3f) return 0.0f;

    /* Subharmonic correction: find the shortest lag (highest F0) that still
     * has >= 80% of best correlation, checking divisors 2 through 4. */
    for (int div = 2; div <= 4; div++) {
        int sub_lag = best_lag / div;
        if (sub_lag < min_lag) break;
        float corr_s = 0.0f;
        vDSP_dotpr(pcm, 1, pcm + sub_lag, 1, &corr_s,
                   (vDSP_Length)(n_samples - sub_lag));
        corr_s /= (float)(n_samples - sub_lag);
        if (corr_s >= best_corr * 0.8f) {
            best_lag = sub_lag;
            break;
        }
    }

    return (float)sample_rate / (float)best_lag;
}

static void compute_prosody_profile(const float *pcm, int n_samples,
                                     int sample_rate, VoiceProsodyProfile *out)
{
    memset(out, 0, sizeof(*out));
    out->sample_rate = sample_rate;
    out->duration_sec = (float)n_samples / (float)sample_rate;

    /* Energy: RMS in dB */
    float rms;
    vDSP_rmsqv(pcm, 1, &rms, (vDSP_Length)n_samples);
    out->energy_mean_db = (rms > 1e-10f) ? 20.0f * log10f(rms) : -100.0f;

    /* F0 analysis: estimate on overlapping 50ms windows */
    int window = sample_rate / 20; /* 50ms */
    int hop = window / 2;
    int n_windows = 0;
    float f0_sum = 0.0f;
    float f0_min = 1e6f, f0_max = 0.0f;

    for (int i = 0; i + window <= n_samples; i += hop) {
        float f0 = voice_onboard_estimate_f0(pcm + i, window, sample_rate);
        if (f0 > 60.0f && f0 < 500.0f) {
            f0_sum += f0;
            if (f0 < f0_min) f0_min = f0;
            if (f0 > f0_max) f0_max = f0;
            n_windows++;
        }
    }

    if (n_windows > 0) {
        out->f0_mean = f0_sum / (float)n_windows;
        out->f0_range = f0_max - f0_min;
    }

    /* Speaking rate: estimate syllables from energy contour transitions */
    int transitions = 0;
    float energy_thresh = rms * 0.5f;
    int in_speech = 0;
    for (int i = 0; i + 160 <= n_samples; i += 160) {
        float frame_rms;
        vDSP_rmsqv(pcm + i, 1, &frame_rms, 160);
        int speech = (frame_rms > energy_thresh);
        if (speech && !in_speech) transitions++;
        in_speech = speech;
    }
    /* Rough syllable rate → word rate (~1.5 syllables/word) */
    out->speaking_rate = (transitions > 0 && out->duration_sec > 0.1f)
        ? (float)transitions / (1.5f * out->duration_sec) : 0.0f;
}

VoiceOnboardResult voice_onboard_finalize(VoiceOnboardSession *session) {
    VoiceOnboardResult result = {0};
    if (!session) return result;

    if (session->buf_len < MIN_SPEECH_SAMPLES) {
        fprintf(stderr, "[onboard] Not enough speech captured (%.1fs < 1.0s)\n",
                (float)session->buf_len / (float)session->sample_rate);
        result.success = 0;
        return result;
    }

    compute_prosody_profile(session->audio_buf, session->buf_len,
                            session->sample_rate, &result.prosody);

    fprintf(stderr, "[onboard] Prosody profile: F0=%.0fHz (range=%.0f), energy=%.1fdB, rate=%.1f wps\n",
            result.prosody.f0_mean, result.prosody.f0_range,
            result.prosody.energy_mean_db, result.prosody.speaking_rate);

    /* Extract speaker embedding if encoder path provided */
    if (session->encoder_path && session->encoder_path[0] != '\0') {
        SpeakerEncoder *enc = speaker_encoder_create(session->encoder_path);
        if (enc) {
            int dim = speaker_encoder_embedding_dim(enc);
            if (dim > 0) {
                result.embedding = (float *)malloc((size_t)dim * sizeof(float));
                if (result.embedding) {
                    int ret = speaker_encoder_extract(enc, session->audio_buf,
                                                     session->buf_len, result.embedding);
                    if (ret > 0) {
                        result.embedding_dim = ret;
                        fprintf(stderr, "[onboard] Speaker embedding extracted (dim=%d)\n", ret);
                    } else {
                        free(result.embedding);
                        result.embedding = NULL;
                        result.embedding_dim = 0;
                        fprintf(stderr, "[onboard] Speaker embedding extraction failed\n");
                    }
                }
            }
            speaker_encoder_destroy(enc);
        } else {
            fprintf(stderr, "[onboard] Could not load speaker encoder: %s\n",
                    session->encoder_path);
            result.success = 0;
            return result;
        }
    }

    result.success = 1;
    return result;
}

void voice_onboard_destroy(VoiceOnboardSession *session) {
    if (!session) return;
    free(session->audio_buf);
    free(session);
}
