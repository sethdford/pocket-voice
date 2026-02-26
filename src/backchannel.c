/**
 * backchannel.c — Active listening backchannel generation.
 *
 * Detects natural backchannel timing from audio features and synthesizes
 * short confirmatory utterances ("mhm", "yeah") to signal active listening.
 *
 * Timing model: Backchannels tend to occur at:
 *   1. Pitch falls (phrase-final intonation)
 *   2. Short pauses (200-600ms) that aren't end-of-turn
 *   3. After rising intonation (questions/checks)
 *   4. Periodically during extended monologue (~3-5s intervals)
 *
 * Build: cc -O3 -shared -fPIC -framework Accelerate -o libbackchannel.dylib backchannel.c
 */

#include "backchannel.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#define BC_FRAME_SIZE  1920  /* 80ms at 24kHz */
#define BC_HIST_FRAMES 25    /* ~2s of history */
#define BC_MIN_GAP_MS  2000  /* Min 2s between backchannels */
#define BC_SYNTH_SAMPLES 4800 /* 200ms of audio at 24kHz */

/* Built-in backchannel audio (synthesized at create time) */
typedef struct {
    float *audio;
    int    len;
} BCSnippet;

struct BackchannelGen {
    int         sample_rate;
    int         enabled;

    /* Feature history */
    float       energy_hist[BC_HIST_FRAMES];
    float       pitch_hist[BC_HIST_FRAMES];
    int         hist_pos;
    int         hist_count;

    /* Timing state */
    int         frames_since_speech;
    int         frames_since_bc;
    int         speech_active;
    int         total_speech_frames;

    /* Pitch tracking (autocorrelation-based) */
    float       prev_pitch;
    int         pitch_falling;

    /* Built-in snippets */
    BCSnippet   snippets[BC_COUNT];
};

/* Simple pink noise formant synthesis for a "mhm" sound */
static void synth_mhm(float *out, int len, int sr) {
    float dt = 1.0f / (float)sr;
    for (int i = 0; i < len; i++) {
        float t = (float)i * dt;
        float env = sinf(3.14159f * t / (len * dt));
        env *= env;
        /* Nasal formant ~270Hz + ~2000Hz */
        float s = 0.6f * sinf(2.0f * 3.14159f * 270.0f * t)
                + 0.2f * sinf(2.0f * 3.14159f * 540.0f * t)
                + 0.1f * sinf(2.0f * 3.14159f * 2000.0f * t);
        /* Pitch glide: slight fall */
        float glide = 1.0f - 0.05f * t / (len * dt);
        s *= glide;
        out[i] = s * env * 0.15f;
    }
}

static void synth_uh_huh(float *out, int len, int sr) {
    float dt = 1.0f / (float)sr;
    int half = len / 2;
    for (int i = 0; i < len; i++) {
        float t = (float)i * dt;
        float local_t = (i < half) ? t : (t - half * dt);
        float local_len = (i < half) ? half * dt : (len - half) * dt;
        float env = sinf(3.14159f * local_t / local_len);
        env *= env;
        /* Two-syllable: "uh" (lower) then "huh" (higher) */
        float f0 = (i < half) ? 180.0f : 220.0f;
        float s = 0.5f * sinf(2.0f * 3.14159f * f0 * t)
                + 0.3f * sinf(2.0f * 3.14159f * f0 * 2.0f * t);
        /* Dip between syllables */
        float gap_center = (float)half;
        float gap_width = 0.08f * sr;
        float gap_dist = fabsf((float)i - gap_center) / gap_width;
        if (gap_dist < 1.0f) env *= gap_dist;
        out[i] = s * env * 0.12f;
    }
}

static float estimate_pitch(const float *audio, int n, int sr) {
    if (n < 256) return 0.0f;
    int min_lag = sr / 500;  /* 500 Hz max */
    int max_lag = sr / 60;   /* 60 Hz min */
    if (max_lag > n / 2) max_lag = n / 2;
    if (min_lag >= max_lag) return 0.0f;

    float best_corr = -1.0f;
    int best_lag = 0;
    for (int lag = min_lag; lag < max_lag; lag += 2) {
        float corr = 0.0f, e1 = 0.0f, e2 = 0.0f;
        int end = n - lag;
        if (end > 512) end = 512;
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
    return (float)sr / (float)best_lag;
}

BackchannelGen *backchannel_create(int sample_rate) {
    BackchannelGen *bc = (BackchannelGen *)calloc(1, sizeof(BackchannelGen));
    if (!bc) return NULL;

    bc->sample_rate = sample_rate;
    bc->enabled = 0;
    bc->frames_since_bc = BC_MIN_GAP_MS * sample_rate / (1000 * BC_FRAME_SIZE);

    /* Synthesize built-in snippets */
    for (int i = 0; i < BC_COUNT; i++) {
        bc->snippets[i].len = BC_SYNTH_SAMPLES;
        bc->snippets[i].audio = (float *)calloc(BC_SYNTH_SAMPLES, sizeof(float));
    }
    if (bc->snippets[BC_MHM].audio)
        synth_mhm(bc->snippets[BC_MHM].audio, BC_SYNTH_SAMPLES, sample_rate);
    if (bc->snippets[BC_UH_HUH].audio)
        synth_uh_huh(bc->snippets[BC_UH_HUH].audio, BC_SYNTH_SAMPLES, sample_rate);
    /* yeah/right/okay use mhm with pitch variation */
    if (bc->snippets[BC_YEAH].audio) {
        synth_mhm(bc->snippets[BC_YEAH].audio, BC_SYNTH_SAMPLES, sample_rate);
        for (int i = 0; i < BC_SYNTH_SAMPLES; i++)
            bc->snippets[BC_YEAH].audio[i] *= 1.1f;
    }
    if (bc->snippets[BC_RIGHT].audio) {
        synth_mhm(bc->snippets[BC_RIGHT].audio, BC_SYNTH_SAMPLES * 3 / 4, sample_rate);
        bc->snippets[BC_RIGHT].len = BC_SYNTH_SAMPLES * 3 / 4;
    }
    if (bc->snippets[BC_OKAY].audio)
        synth_uh_huh(bc->snippets[BC_OKAY].audio, BC_SYNTH_SAMPLES, sample_rate);

    return bc;
}

void backchannel_destroy(BackchannelGen *bc) {
    if (!bc) return;
    for (int i = 0; i < BC_COUNT; i++)
        free(bc->snippets[i].audio);
    free(bc);
}

BackchannelEvent backchannel_feed(BackchannelGen *bc, const float *audio,
                                  int n_samples, float stt_eou_prob) {
    BackchannelEvent ev = { BC_NONE, 0.0f, 0 };
    if (!bc || !bc->enabled || !audio || n_samples <= 0) return ev;

    /* Compute frame energy */
    float rms = 0.0f;
    vDSP_rmsqv(audio, 1, &rms, n_samples);
    float energy_db = (rms > 1e-8f) ? 20.0f * log10f(rms) : -80.0f;

    /* Compute pitch */
    float pitch = estimate_pitch(audio, n_samples, bc->sample_rate);

    /* Update history */
    int idx = bc->hist_pos % BC_HIST_FRAMES;
    bc->energy_hist[idx] = energy_db;
    bc->pitch_hist[idx] = pitch;
    bc->hist_pos++;
    if (bc->hist_count < BC_HIST_FRAMES) bc->hist_count++;

    /* Track speech activity */
    int is_speech = (energy_db > -40.0f);
    if (is_speech) {
        bc->speech_active = 1;
        bc->frames_since_speech = 0;
        bc->total_speech_frames++;
    } else {
        bc->frames_since_speech++;
    }
    bc->frames_since_bc++;

    /* Pitch fall detection */
    if (pitch > 50.0f && bc->prev_pitch > 50.0f) {
        bc->pitch_falling = (pitch < bc->prev_pitch * 0.85f);
    }
    if (pitch > 50.0f) bc->prev_pitch = pitch;

    /* Minimum gap between backchannels */
    int min_gap_frames = BC_MIN_GAP_MS * bc->sample_rate / (1000 * BC_FRAME_SIZE);
    if (bc->frames_since_bc < min_gap_frames) return ev;

    /* Don't backchannel if this looks like end-of-turn */
    if (stt_eou_prob > 0.5f) return ev;

    /* Must have heard some speech first */
    if (bc->total_speech_frames < 10) return ev;

    /* ── Backchannel Triggers ── */
    float score = 0.0f;

    /* Trigger 1: Short pause after speech (200-600ms) */
    int pause_frames = bc->frames_since_speech;
    int pause_ms = pause_frames * BC_FRAME_SIZE * 1000 / bc->sample_rate;
    if (bc->speech_active && pause_ms >= 200 && pause_ms <= 600) {
        score += 0.4f;
    }

    /* Trigger 2: Pitch fall (phrase boundary) */
    if (bc->pitch_falling && is_speech) {
        score += 0.3f;
    }

    /* Trigger 3: Extended monologue (>3s of speech without our response) */
    int speech_ms = bc->total_speech_frames * BC_FRAME_SIZE * 1000 / bc->sample_rate;
    if (speech_ms > 3000) {
        score += 0.2f;
    }

    /* Trigger 4: Moderate EOU probability (they paused but aren't done) */
    if (stt_eou_prob > 0.2f && stt_eou_prob < 0.5f) {
        score += 0.1f;
    }

    if (score >= 0.5f) {
        /* Select type based on context */
        BackchannelType type;
        if (pause_ms > 400) {
            type = BC_MHM;
        } else if (bc->pitch_falling) {
            type = BC_UH_HUH;
        } else if (speech_ms > 5000) {
            type = BC_YEAH;
        } else {
            type = BC_MHM;
        }
        ev.type = type;
        ev.confidence = score;
        ev.ready = 1;
        bc->frames_since_bc = 0;
        bc->total_speech_frames = 0;
    }

    return ev;
}

const float *backchannel_get_audio(BackchannelGen *bc, BackchannelType type, int *out_len) {
    if (!bc || type <= BC_NONE || type >= BC_COUNT || !out_len) return NULL;
    *out_len = bc->snippets[type].len;
    return bc->snippets[type].audio;
}

int backchannel_load_wav(BackchannelGen *bc, BackchannelType type, const char *wav_path) {
    if (!bc || type <= BC_NONE || type >= BC_COUNT || !wav_path) return -1;

    FILE *f = fopen(wav_path, "rb");
    if (!f) return -1;

    /* Skip WAV header (44 bytes for standard PCM) */
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 44, SEEK_SET);

    long data_bytes = size - 44;
    if (data_bytes <= 0) { fclose(f); return -1; }

    int n_samples = (int)(data_bytes / sizeof(int16_t));
    int16_t *raw = (int16_t *)malloc(data_bytes);
    if (!raw) { fclose(f); return -1; }
    fread(raw, 1, data_bytes, f);
    fclose(f);

    /* Convert int16 to float */
    float *audio = (float *)realloc(bc->snippets[type].audio,
                                     n_samples * sizeof(float));
    if (!audio) { free(raw); return -1; }

    for (int i = 0; i < n_samples; i++)
        audio[i] = (float)raw[i] / 32768.0f;
    free(raw);

    bc->snippets[type].audio = audio;
    bc->snippets[type].len = n_samples;
    return 0;
}

void backchannel_reset(BackchannelGen *bc) {
    if (!bc) return;
    memset(bc->energy_hist, 0, sizeof(bc->energy_hist));
    memset(bc->pitch_hist, 0, sizeof(bc->pitch_hist));
    bc->hist_pos = 0;
    bc->hist_count = 0;
    bc->frames_since_speech = 0;
    bc->frames_since_bc = BC_MIN_GAP_MS * bc->sample_rate / (1000 * BC_FRAME_SIZE);
    bc->speech_active = 0;
    bc->total_speech_frames = 0;
    bc->prev_pitch = 0.0f;
    bc->pitch_falling = 0;
}

void backchannel_set_enabled(BackchannelGen *bc, int enabled) {
    if (bc) bc->enabled = enabled;
}

int backchannel_is_enabled(BackchannelGen *bc) {
    return bc ? bc->enabled : 0;
}
