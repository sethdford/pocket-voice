/**
 * audio_emotion.c — Real-time emotion detection from audio features.
 *
 * Extracts acoustic features (pitch, energy, speaking rate, jitter,
 * spectral tilt) and maps them to emotion labels + valence/arousal space.
 *
 * Runs entirely on AMX via vDSP. No neural network required — uses
 * rule-based mapping from psychoacoustic research on vocal emotion cues.
 *
 * Build: cc -O3 -shared -fPIC -framework Accelerate -o libaudio_emotion.dylib audio_emotion.c
 */

#include "audio_emotion.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#define AE_WINDOW_FRAMES  50   /* ~4s analysis window at 80ms/frame */
#define AE_MIN_FRAMES     3    /* Minimum frames before reporting */
#define AE_EMA_ALPHA      0.3f

struct AudioEmotionDetector {
    int     sample_rate;
    int     frame_count;

    /* Running feature accumulators */
    float   pitch_buf[AE_WINDOW_FRAMES];
    float   energy_buf[AE_WINDOW_FRAMES];
    float   rate_buf[AE_WINDOW_FRAMES];    /* voiced frame ratio → proxy for rate */
    float   tilt_buf[AE_WINDOW_FRAMES];
    int     buf_pos;
    int     buf_count;

    /* Smoothed output */
    AudioEmotionResult smoothed;

    /* Baseline tracking (calibrated over first ~2s) */
    float   baseline_pitch;
    float   baseline_energy;
    int     baseline_frames;
    int     baseline_pitch_frames;
    int     baseline_ready;
};

static float estimate_pitch_ac(const float *audio, int n, int sr) {
    if (n < 256) return 0.0f;
    int min_lag = sr / 500;
    int max_lag = sr / 60;
    if (max_lag > n / 2) max_lag = n / 2;
    if (min_lag >= max_lag) return 0.0f;

    float best_corr = -1.0f;
    int best_lag = 0;
    int window = n < 512 ? n : 512;

    for (int lag = min_lag; lag < max_lag; lag += 2) {
        float corr = 0.0f, e1 = 0.0f, e2 = 0.0f;
        int end = window - lag;
        if (end <= 0) continue;
        vDSP_dotpr(audio, 1, audio + lag, 1, &corr, end);
        vDSP_dotpr(audio, 1, audio, 1, &e1, end);
        vDSP_dotpr(audio + lag, 1, audio + lag, 1, &e2, end);
        float denom = sqrtf(e1 * e2);
        if (denom > 1e-8f) corr /= denom;
        if (corr > best_corr) {
            best_corr = corr;
            best_lag = lag;
        }
    }
    if (best_corr < 0.3f || best_lag == 0) return 0.0f;

    /* Subharmonic correction: find the shortest lag (highest F0) that still has
     * good correlation, checking divisors 2 through 4. */
    for (int div = 2; div <= 4; div++) {
        int sub_lag = best_lag / div;
        if (sub_lag < min_lag) break;
        float corr_s = 0.0f, e1_s = 0.0f, e2_s = 0.0f;
        int end_s = window - sub_lag;
        if (end_s <= 0) continue;
        vDSP_dotpr(audio, 1, audio + sub_lag, 1, &corr_s, end_s);
        vDSP_dotpr(audio, 1, audio, 1, &e1_s, end_s);
        vDSP_dotpr(audio + sub_lag, 1, audio + sub_lag, 1, &e2_s, end_s);
        float denom_s = sqrtf(e1_s * e2_s);
        if (denom_s > 1e-8f) corr_s /= denom_s;
        if (corr_s >= best_corr * 0.8f) {
            best_lag = sub_lag;
            break;
        }
    }

    return (float)sr / (float)best_lag;
}

static float compute_spectral_tilt(const float *audio, int n, int sr) {
    /* Estimate spectral tilt via ratio of high-frequency to low-frequency energy */
    if (n < 512) return 0.0f;
    int half = n / 2;
    float lo_energy = 0.0f, hi_energy = 0.0f;
    vDSP_dotpr(audio, 1, audio, 1, &lo_energy, half);
    vDSP_dotpr(audio + half, 1, audio + half, 1, &hi_energy, n - half);
    (void)sr;
    if (lo_energy < 1e-10f) return 0.0f;
    return 10.0f * log10f((hi_energy + 1e-10f) / (lo_energy + 1e-10f));
}

static float compute_jitter(const float *audio, int n, int sr) {
    /* Compute pitch perturbation (jitter) via consecutive period differences */
    if (n < 1024) return 0.0f;

    int chunk = n / 4;
    float pitches[4];
    int n_voiced = 0;
    for (int i = 0; i < 4; i++) {
        pitches[i] = estimate_pitch_ac(audio + i * chunk, chunk, sr);
        if (pitches[i] > 50.0f) n_voiced++;
    }
    if (n_voiced < 3) return 0.0f;

    float sum_diff = 0.0f;
    float sum_pitch = 0.0f;
    int n_diffs = 0;
    for (int i = 1; i < 4; i++) {
        if (pitches[i] > 50.0f && pitches[i-1] > 50.0f) {
            sum_diff += fabsf(pitches[i] - pitches[i-1]);
            sum_pitch += pitches[i];
            n_diffs++;
        }
    }
    if (n_diffs == 0 || sum_pitch < 1e-6f) return 0.0f;
    return (sum_diff / n_diffs) / (sum_pitch / n_diffs);
}

AudioEmotionDetector *audio_emotion_create(int sample_rate) {
    AudioEmotionDetector *det = (AudioEmotionDetector *)calloc(1, sizeof(AudioEmotionDetector));
    if (!det) return NULL;
    det->sample_rate = sample_rate;
    det->smoothed.primary = AUDIO_EMO_NEUTRAL;
    det->smoothed.valence = 0.0f;
    det->smoothed.arousal = 0.0f;
    det->smoothed.speaking_rate = 1.0f;
    return det;
}

void audio_emotion_destroy(AudioEmotionDetector *det) {
    free(det);
}

void audio_emotion_feed(AudioEmotionDetector *det, const float *audio, int n_samples) {
    if (!det || !audio || n_samples <= 0) return;

    /* Extract features */
    float rms = 0.0f;
    vDSP_rmsqv(audio, 1, &rms, n_samples);
    float energy_db = (rms > 1e-8f) ? 20.0f * log10f(rms) : -80.0f;
    float pitch = estimate_pitch_ac(audio, n_samples, det->sample_rate);
    float tilt = compute_spectral_tilt(audio, n_samples, det->sample_rate);
    float voiced = (pitch > 50.0f) ? 1.0f : 0.0f;

    /* Update baseline (first ~2s = 25 frames) */
    if (!det->baseline_ready) {
        if (energy_db > -50.0f) {
            det->baseline_energy = det->baseline_energy * det->baseline_frames
                                  + energy_db;
            det->baseline_frames++;
            det->baseline_energy /= det->baseline_frames;

            /* Only include voiced frames in pitch baseline to avoid
               dragging the average toward zero with unvoiced frames */
            if (pitch > 50.0f) {
                det->baseline_pitch = det->baseline_pitch * det->baseline_pitch_frames
                                    + pitch;
                det->baseline_pitch_frames++;
                det->baseline_pitch /= det->baseline_pitch_frames;
            }

            if (det->baseline_frames >= 25) det->baseline_ready = 1;
        }
    }

    /* Store in circular buffer */
    int idx = det->buf_pos % AE_WINDOW_FRAMES;
    det->pitch_buf[idx] = pitch;
    det->energy_buf[idx] = energy_db;
    det->rate_buf[idx] = voiced;
    det->tilt_buf[idx] = tilt;
    det->buf_pos++;
    if (det->buf_count < AE_WINDOW_FRAMES) det->buf_count++;
    det->frame_count++;
}

AudioEmotionResult audio_emotion_get(const AudioEmotionDetector *det) {
    AudioEmotionResult r = {0};
    r.primary = AUDIO_EMO_NEUTRAL;
    r.speaking_rate = 1.0f;

    if (!det || det->frame_count < AE_MIN_FRAMES) return r;

    int n = det->buf_count;

    /* Compute statistics over window */
    float pitch_sum = 0.0f, pitch_min = 1e6f, pitch_max = 0.0f;
    float energy_sum = 0.0f;
    float voiced_sum = 0.0f;
    float tilt_sum = 0.0f;
    int pitch_count = 0;

    for (int i = 0; i < n; i++) {
        energy_sum += det->energy_buf[i];
        voiced_sum += det->rate_buf[i];
        tilt_sum += det->tilt_buf[i];
        if (det->pitch_buf[i] > 50.0f) {
            pitch_sum += det->pitch_buf[i];
            if (det->pitch_buf[i] < pitch_min) pitch_min = det->pitch_buf[i];
            if (det->pitch_buf[i] > pitch_max) pitch_max = det->pitch_buf[i];
            pitch_count++;
        }
    }

    r.pitch_mean = (pitch_count > 0) ? pitch_sum / pitch_count : 0.0f;
    r.pitch_range = (pitch_count > 2) ? pitch_max - pitch_min : 0.0f;
    r.energy_mean = energy_sum / n;
    r.speaking_rate = voiced_sum / n; /* Fraction of voiced frames */
    r.spectral_tilt = tilt_sum / n;

    /* Compute jitter from most recent frame */
    /* (already averaged across the window via feature buffer) */
    r.jitter = 0.0f;
    if (pitch_count > 2) {
        float jit_sum = 0.0f;
        float prev_p = 0.0f;
        int jit_count = 0;
        for (int i = 0; i < n; i++) {
            if (det->pitch_buf[i] > 50.0f) {
                if (prev_p > 50.0f) {
                    jit_sum += fabsf(det->pitch_buf[i] - prev_p) / prev_p;
                    jit_count++;
                }
                prev_p = det->pitch_buf[i];
            }
        }
        if (jit_count > 0) r.jitter = jit_sum / jit_count;
    }

    /* ── Rule-based emotion mapping ──
     * Based on psychoacoustic research (Scherer 2003, Juslin & Laukka 2003):
     *
     * Happy/Excited: high pitch, wide range, high energy, fast rate
     * Sad:           low pitch, narrow range, low energy, slow rate
     * Angry:         high pitch, wide range, high energy, fast rate, harsh
     * Fearful:       high pitch, wide range, high jitter, fast rate
     * Calm:          low-mid pitch, narrow range, moderate energy, slow rate
     * Frustrated:    moderate pitch, irregular, moderate-high energy
     * Hesitant:      irregular pitch, low energy, many pauses
     */

    float bp = det->baseline_ready ? det->baseline_pitch : 150.0f;
    float be = det->baseline_ready ? det->baseline_energy : -25.0f;

    float pitch_dev = (bp > 50.0f && r.pitch_mean > 50.0f)
                    ? (r.pitch_mean - bp) / bp : 0.0f;
    float energy_dev = r.energy_mean - be;
    float rate_norm = r.speaking_rate; /* 0-1, higher = more voiced = faster */

    /* Valence-arousal mapping */
    r.arousal = 0.5f;
    r.valence = 0.0f;

    /* Arousal: high energy + high pitch + wide range + fast rate */
    r.arousal = 0.5f + 0.2f * (energy_dev / 10.0f)
              + 0.15f * pitch_dev
              + 0.1f * (r.pitch_range / 100.0f)
              + 0.05f * (rate_norm - 0.5f);
    if (r.arousal < 0.0f) r.arousal = 0.0f;
    if (r.arousal > 1.0f) r.arousal = 1.0f;

    /* Valence: spectral tilt (breathy = negative), jitter (high = negative) */
    r.valence = -0.3f * r.jitter * 10.0f
              + 0.2f * (r.spectral_tilt + 5.0f) / 10.0f
              + 0.1f * pitch_dev;
    if (r.valence < -1.0f) r.valence = -1.0f;
    if (r.valence > 1.0f) r.valence = 1.0f;

    /* Classify primary emotion */
    float confidence = 0.0f;

    if (r.arousal > 0.7f && r.valence > 0.2f) {
        r.primary = (r.arousal > 0.85f) ? AUDIO_EMO_EXCITED : AUDIO_EMO_HAPPY;
        confidence = r.arousal * 0.5f + r.valence * 0.3f;
    } else if (r.arousal > 0.7f && r.valence < -0.2f) {
        r.primary = (r.jitter > 0.05f) ? AUDIO_EMO_FEARFUL : AUDIO_EMO_ANGRY;
        confidence = r.arousal * 0.4f + fabsf(r.valence) * 0.3f;
    } else if (r.arousal < 0.3f && r.valence < -0.1f) {
        r.primary = AUDIO_EMO_SAD;
        confidence = (1.0f - r.arousal) * 0.4f + fabsf(r.valence) * 0.3f;
    } else if (r.arousal < 0.35f && r.valence > -0.1f) {
        r.primary = AUDIO_EMO_CALM;
        confidence = (1.0f - r.arousal) * 0.3f + 0.2f;
    } else if (r.jitter > 0.04f && rate_norm < 0.4f) {
        r.primary = AUDIO_EMO_HESITANT;
        confidence = r.jitter * 5.0f + (1.0f - rate_norm) * 0.2f;
    } else if (energy_dev > 3.0f && r.jitter > 0.03f) {
        r.primary = AUDIO_EMO_FRUSTRATED;
        confidence = energy_dev / 10.0f + r.jitter * 3.0f;
    } else {
        r.primary = AUDIO_EMO_NEUTRAL;
        confidence = 0.3f;
    }

    r.confidence = confidence;
    if (r.confidence > 1.0f) r.confidence = 1.0f;
    if (r.confidence < 0.0f) r.confidence = 0.0f;

    return r;
}

void audio_emotion_reset(AudioEmotionDetector *det) {
    if (!det) return;
    det->frame_count = 0;
    det->buf_pos = 0;
    det->buf_count = 0;
    memset(det->pitch_buf, 0, sizeof(det->pitch_buf));
    memset(det->energy_buf, 0, sizeof(det->energy_buf));
    memset(det->rate_buf, 0, sizeof(det->rate_buf));
    memset(det->tilt_buf, 0, sizeof(det->tilt_buf));
    memset(&det->smoothed, 0, sizeof(det->smoothed));
    det->smoothed.primary = AUDIO_EMO_NEUTRAL;
    det->smoothed.speaking_rate = 1.0f;
    /* Keep baseline — it persists across turns for speaker consistency */
}

int audio_emotion_describe(const AudioEmotionResult *r, char *buf, int buf_size) {
    if (!r || !buf || buf_size < 2) return 0;

    static const char *names[] = {
        "neutral", "happy", "sad", "angry", "fearful",
        "surprised", "calm", "excited", "frustrated", "hesitant"
    };
    const char *name = (r->primary < AUDIO_EMO_COUNT) ? names[r->primary] : "unknown";

    const char *valence_str = r->valence > 0.2f ? "positive"
                            : r->valence < -0.2f ? "negative" : "neutral";
    const char *arousal_str = r->arousal > 0.65f ? "high energy"
                            : r->arousal < 0.35f ? "low energy" : "moderate energy";

    int wrote = snprintf(buf, buf_size,
        "The user sounds %s (%s affect, %s). ",
        name, valence_str, arousal_str);

    if (r->speaking_rate < 0.3f && wrote < buf_size) {
        wrote += snprintf(buf + wrote, buf_size - wrote,
            "They're speaking slowly with many pauses. ");
    } else if (r->speaking_rate > 0.7f && wrote < buf_size) {
        wrote += snprintf(buf + wrote, buf_size - wrote,
            "They're speaking rapidly. ");
    }

    if (r->primary == AUDIO_EMO_FRUSTRATED && wrote < buf_size) {
        wrote += snprintf(buf + wrote, buf_size - wrote,
            "Be patient and empathetic. ");
    } else if (r->primary == AUDIO_EMO_HESITANT && wrote < buf_size) {
        wrote += snprintf(buf + wrote, buf_size - wrote,
            "They seem uncertain — be encouraging. ");
    } else if (r->primary == AUDIO_EMO_SAD && wrote < buf_size) {
        wrote += snprintf(buf + wrote, buf_size - wrote,
            "Be gentle and supportive. ");
    } else if (r->primary == AUDIO_EMO_EXCITED && wrote < buf_size) {
        wrote += snprintf(buf + wrote, buf_size - wrote,
            "Match their enthusiasm! ");
    }

    /* Cap to actual bytes written (snprintf returns "would-have-written" count) */
    return wrote < buf_size ? wrote : buf_size - 1;
}
