/**
 * fused_eou.c — Fused 3-signal End-of-Utterance detection.
 *
 * Combines energy VAD, Mimi endpointer, and STT/semantic VAD signals
 * into a single endpoint decision with adaptive weighting and early exit.
 */

#include "fused_eou.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define PROSODY_BUF_SIZE 32  /* Ring buffer for ~500ms of prosody frames */
#define CONTEXT_MAX_LEN 256

struct FusedEOU {
    /* Fusion weights (normalized to sum to 1.0) */
    float w_energy;
    float w_mimi;
    float w_stt;

    /* Solo thresholds — any signal above its solo threshold triggers alone */
    float solo_energy;
    float solo_mimi;
    float solo_stt;

    /* Fused threshold */
    float threshold;
    int   consec_required;
    float frame_ms;

    /* State */
    float smoothed_prob;    /* EMA of fused probability */
    int   consec_count;
    int   triggered;
    int   trigger_source;

    /* History for latency estimation */
    int   speech_detected;  /* Has speech been detected? */
    int   frames_since_speech;
    int   total_frames;

    /* Prosody signal (4th signal — Krisp-inspired) */
    float w_prosody;        /* Weight for prosody in fusion (default 0.0) */
    float solo_prosody;     /* Solo threshold for prosody */
    float pitch_buf[PROSODY_BUF_SIZE];
    float energy_buf[PROSODY_BUF_SIZE];
    int   prosody_write;    /* Ring buffer write index */
    int   prosody_count;    /* Total prosody frames accumulated */
    float prosody_prob;     /* Cached P(eot) from prosody */

    /* Semantic completion signal (5th signal) */
    float w_semantic;       /* Weight for semantic in fusion (default 0.0) */
    float solo_semantic;    /* Solo threshold for semantic */
    float semantic_prob;    /* Cached P(complete) from text model */

    /* Conversation context (LiveKit-inspired) */
    float context_adj;      /* Threshold adjustment: <0 = easier trigger */
    char  last_transcript[CONTEXT_MAX_LEN];
};

FusedEOU *fused_eou_create(float threshold, int consec_frames, float frame_ms) {
    FusedEOU *eou = (FusedEOU *)calloc(1, sizeof(FusedEOU));
    if (!eou) return NULL;

    eou->w_energy = 0.25f;
    eou->w_mimi   = 0.45f;
    eou->w_stt    = 0.30f;

    eou->solo_energy = 0.95f; /* Very high — energy alone rarely sufficient */
    eou->solo_mimi   = 0.90f;
    eou->solo_stt    = 0.85f;

    eou->threshold = threshold;
    eou->consec_required = consec_frames;
    eou->frame_ms = frame_ms;

    eou->w_prosody    = 0.0f;   /* Off by default — backward compatible */
    eou->solo_prosody = 0.92f;

    eou->w_semantic    = 0.0f;  /* Off by default — backward compatible */
    eou->solo_semantic = 0.88f;

    return eou;
}

void fused_eou_destroy(FusedEOU *eou) {
    free(eou);
}

/* ── Prosody probability from pitch trajectory + energy decay ──── */
static float compute_prosody_prob(const FusedEOU *eou) {
    if (eou->prosody_count < 4) return 0.5f; /* Not enough data → neutral */

    int n = eou->prosody_count < PROSODY_BUF_SIZE
          ? eou->prosody_count : PROSODY_BUF_SIZE;
    int half = n / 2;

    /* Split buffer into first half and second half */
    float pitch_first = 0, pitch_second = 0;
    float energy_first = 0, energy_second = 0;
    int voiced_first = 0, voiced_second = 0;

    for (int i = 0; i < n; i++) {
        int idx = (eou->prosody_write - n + i + PROSODY_BUF_SIZE)
                % PROSODY_BUF_SIZE;
        float p = eou->pitch_buf[idx];
        float e = eou->energy_buf[idx];
        /* Guard against NaN/Inf propagation from upstream */
        if (isnan(p) || isinf(p)) p = 0.0f;
        if (isnan(e) || isinf(e)) e = 0.0f;
        if (i < half) {
            if (p > 0.0f) {
                pitch_first += p;
                voiced_first++;
            }
            energy_first += e;
        } else {
            if (p > 0.0f) {
                pitch_second += p;
                voiced_second++;
            }
            energy_second += e;
        }
    }

    /* Pitch trajectory: falling → completion (high), rising → continuation (low) */
    float pitch_score = 0.5f;
    if (voiced_first >= 2 && voiced_second >= 2) {
        float avg1 = pitch_first / (float)voiced_first;
        float avg2 = pitch_second / (float)voiced_second;
        if (avg1 > 1.0f) {
            float ratio = avg2 / avg1;
            /* Sigmoid centered at 1.0: ratio < 1 (falling) → high score */
            float exponent = (ratio - 1.0f) * 10.0f;
            if (exponent > 88.0f) exponent = 88.0f; /* Prevent expf overflow */
            pitch_score = 1.0f / (1.0f + expf(exponent));
        }
    }

    /* Energy decay: fast drop → completion (high), increasing → continuation (low) */
    float e1 = energy_first / (float)(half > 0 ? half : 1);
    float e2 = energy_second / (float)((n - half) > 0 ? (n - half) : 1);
    float drop = e1 - e2; /* Positive = energy dropping (in dB) */
    float clamped_drop = drop * 0.3f;
    if (clamped_drop > 88.0f) clamped_drop = 88.0f;
    if (clamped_drop < -88.0f) clamped_drop = -88.0f;
    float energy_score = 1.0f / (1.0f + expf(-clamped_drop));

    /* Combine: 60% pitch, 40% energy (pitch is primary turn-taking cue) */
    return 0.6f * pitch_score + 0.4f * energy_score;
}

/* ── Context adjustment from last transcript ──────────────────── */
static float compute_context_adj(const char *transcript) {
    if (!transcript || !transcript[0]) return 0.0f;

    size_t len = strlen(transcript);

    /* Find last non-whitespace character */
    size_t end = len;
    while (end > 0 && (transcript[end - 1] == ' ' ||
                        transcript[end - 1] == '\n' ||
                        transcript[end - 1] == '\r' ||
                        transcript[end - 1] == '\t')) {
        end--;
    }
    if (end == 0) return 0.0f;

    /* Question mark → lower threshold (speaker is done asking) */
    if (transcript[end - 1] == '?') return -0.10f;

    /* Find last word boundaries */
    size_t word_end = end;
    size_t word_start = end;
    while (word_start > 0 && transcript[word_start - 1] != ' ')
        word_start--;

    size_t word_len = word_end - word_start;
    if (word_len == 0 || word_len > 16) return 0.0f;

    /* Lowercase copy of last word */
    char word[17];
    for (size_t i = 0; i < word_len; i++) {
        char c = transcript[word_start + i];
        word[i] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
    }
    word[word_len] = '\0';

    /* Conjunctions → raise threshold (speaker likely continuing) */
    static const char *conjs[] = {
        "and", "but", "or", "so", "because", "however",
        "although", "since", "while", "yet", "then"
    };
    for (int i = 0; i < (int)(sizeof(conjs) / sizeof(conjs[0])); i++) {
        if (strcmp(word, conjs[i]) == 0) return 0.10f;
    }

    /* Fillers → slight raise (speaker likely thinking/continuing) */
    static const char *fillers[] = {"um", "uh", "like", "well", "actually"};
    for (int i = 0; i < (int)(sizeof(fillers) / sizeof(fillers[0])); i++) {
        if (strcmp(word, fillers[i]) == 0) return 0.05f;
    }

    return 0.0f;
}

EOUResult fused_eou_process(FusedEOU *eou, EOUSignals sig) {
    EOUResult res = {0};
    if (!eou) return res;

    eou->total_frames++;

    /* Track speech activity based on energy VAD only — mimi_eot_prob starts at 0
     * and would falsely trigger speech_detected before any actual speech. */
    if (sig.energy_signal >= 0.3f) {
        eou->speech_detected = 1;
        eou->frames_since_speech = 0;
    } else {
        eou->frames_since_speech++;
        if (eou->frames_since_speech > 100000)
            eou->frames_since_speech = 100000;
    }

    /* ── Context-adjusted threshold ───────────────────── */
    float ctx = eou->context_adj;

    /* ── Early Exit: single-signal high-confidence ────── */
    /* Solo thresholds are offset by context_adj so that conversational
     * context (e.g. trailing conjunction → ctx > 0) makes solo triggers
     * harder, and questions (ctx < 0) make them easier. */
    int src = 0;

    if (sig.energy_signal >= eou->solo_energy + ctx && eou->speech_detected) {
        src |= EOU_SRC_ENERGY;
    }
    if (sig.mimi_eot_prob >= eou->solo_mimi + ctx && eou->speech_detected) {
        src |= EOU_SRC_MIMI;
    }
    if (sig.stt_eou_prob >= eou->solo_stt + ctx && eou->speech_detected) {
        src |= EOU_SRC_STT;
    }

    /* ── Prosody signal (4th signal) ─────────────────── */
    eou->prosody_prob = compute_prosody_prob(eou);
    if (eou->prosody_prob >= eou->solo_prosody + ctx && eou->speech_detected &&
        eou->w_prosody > 0.0f) {
        src |= EOU_SRC_PROSODY;
    }

    /* ── Semantic signal (5th signal) ────────────────── */
    if (eou->semantic_prob >= eou->solo_semantic + ctx && eou->speech_detected &&
        eou->w_semantic > 0.0f) {
        src |= EOU_SRC_SEMANTIC;
    }

    /* ── Weighted Fusion (3-signal core + optional prosody + optional semantic) ── */
    float core = eou->w_energy * sig.energy_signal
               + eou->w_mimi  * sig.mimi_eot_prob
               + eou->w_stt   * sig.stt_eou_prob;
    float fused;
    float aux_weight = eou->w_prosody + eou->w_semantic;
    if (aux_weight > 0.0f) {
        float aux = 0.0f;
        if (eou->w_prosody > 0.0f)
            aux += eou->w_prosody * eou->prosody_prob;
        if (eou->w_semantic > 0.0f)
            aux += eou->w_semantic * eou->semantic_prob;
        fused = (1.0f - aux_weight) * core + aux;
    } else {
        fused = core;
    }

    /* EMA smoothing (alpha = 0.4 for fast response) */
    eou->smoothed_prob = 0.4f * fused + 0.6f * eou->smoothed_prob;

    res.fused_prob = eou->smoothed_prob;

    /* ── Trigger Logic ────────────────────────────────── */
    int should_trigger = 0;

    /* Solo trigger: any single signal above its threshold → immediate */
    if (src && eou->speech_detected) {
        should_trigger = 1;
    }

    /* Effective fused threshold (ctx already applied to solo thresholds above) */
    float eff_threshold = eou->threshold + ctx;
    if (eff_threshold < 0.1f) eff_threshold = 0.1f;
    if (eff_threshold > 0.99f) eff_threshold = 0.99f;

    /* Silence-duration fast path: if the user has been silent for 300ms+
     * after producing speech AND we're already above the fused threshold,
     * trigger immediately without waiting for consecutive-frame consensus.
     * This mimics natural conversational turn-taking. The prob threshold
     * must match the fused path to preserve the speculative prefill zone:
     * prob in [0.55, threshold) where we send to LLM early but do not
     * commit to EOU yet. */
    float silence_ms = (float)eou->frames_since_speech * eou->frame_ms;
    if (eou->speech_detected && silence_ms >= 300.0f &&
        eou->smoothed_prob >= eff_threshold) {
        should_trigger = 1;
        src |= EOU_SRC_ENERGY | EOU_SRC_FUSED;
    }

    /* Fused trigger: combined probability above threshold for N frames */
    if (eou->smoothed_prob >= eff_threshold && eou->speech_detected) {
        eou->consec_count++;
        if (eou->consec_count >= eou->consec_required) {
            should_trigger = 1;
            src |= EOU_SRC_FUSED;
        }
    } else {
        eou->consec_count = 0;
    }

    if (should_trigger && !eou->triggered) {
        eou->triggered = 1;
        eou->trigger_source = src;
    }

    res.triggered = eou->triggered;
    res.trigger_source = eou->trigger_source;
    res.consec_frames = eou->consec_count;
    res.latency_ms = (float)eou->frames_since_speech * eou->frame_ms;

    return res;
}

void fused_eou_reset(FusedEOU *eou) {
    if (!eou) return;
    eou->smoothed_prob = 0.0f;
    eou->consec_count = 0;
    eou->triggered = 0;
    eou->trigger_source = 0;
    eou->speech_detected = 0;
    eou->frames_since_speech = 0;
    eou->total_frames = 0;

    /* Clear prosody buffers (context persists across utterances) */
    memset(eou->pitch_buf, 0, sizeof(eou->pitch_buf));
    memset(eou->energy_buf, 0, sizeof(eou->energy_buf));
    eou->prosody_write = 0;
    eou->prosody_count = 0;
    eou->prosody_prob = 0.0f;

    /* Clear semantic state */
    eou->semantic_prob = 0.0f;
}

void fused_eou_set_weights(FusedEOU *eou, float w_energy, float w_mimi, float w_stt) {
    if (!eou) return;
    float sum = w_energy + w_mimi + w_stt;
    if (sum < 1e-6f) sum = 1.0f;
    eou->w_energy = w_energy / sum;
    eou->w_mimi   = w_mimi / sum;
    eou->w_stt    = w_stt / sum;
}

void fused_eou_set_solo_thresholds(FusedEOU *eou,
                                     float energy_solo,
                                     float mimi_solo,
                                     float stt_solo) {
    if (!eou) return;
    eou->solo_energy = energy_solo;
    eou->solo_mimi   = mimi_solo;
    eou->solo_stt    = stt_solo;
}

float fused_eou_prob(const FusedEOU *eou) {
    return eou ? eou->smoothed_prob : 0.0f;
}

int fused_eou_triggered(const FusedEOU *eou) {
    return eou ? eou->triggered : 0;
}

void fused_eou_print_status(const FusedEOU *eou) {
    if (!eou) return;

    const char *src_str = "none";
    if (eou->trigger_source & EOU_SRC_FUSED)      src_str = "FUSED";
    else if (eou->trigger_source & EOU_SRC_SEMANTIC) src_str = "SEMANTIC";
    else if (eou->trigger_source & EOU_SRC_PROSODY)  src_str = "PROSODY";
    else if (eou->trigger_source & EOU_SRC_MIMI)     src_str = "MIMI";
    else if (eou->trigger_source & EOU_SRC_STT)      src_str = "STT";
    else if (eou->trigger_source & EOU_SRC_ENERGY)    src_str = "ENERGY";

    fprintf(stderr, "[fused_eou] prob=%.3f trig=%d src=%s consec=%d "
            "latency=%.0fms frames=%d prosody=%.3f semantic=%.3f ctx_adj=%+.2f\n",
            eou->smoothed_prob, eou->triggered, src_str,
            eou->consec_count,
            (float)eou->frames_since_speech * eou->frame_ms,
            eou->total_frames,
            eou->prosody_prob,
            eou->semantic_prob,
            eou->context_adj);
}

/* ── Conversation Context ──────────────────────────────── */

void fused_eou_set_context(FusedEOU *eou, const char *last_transcript) {
    if (!eou) return;
    if (last_transcript) {
        strncpy(eou->last_transcript, last_transcript, CONTEXT_MAX_LEN - 1);
        eou->last_transcript[CONTEXT_MAX_LEN - 1] = '\0';
    } else {
        eou->last_transcript[0] = '\0';
    }
    eou->context_adj = compute_context_adj(eou->last_transcript);
}

/* ── Prosodic Turn-Taking ──────────────────────────────── */

void fused_eou_feed_prosody(FusedEOU *eou, ProsodyFrame frame) {
    if (!eou) return;
    /* Sanitize inputs — NaN/Inf from upstream audio analysis must not
     * contaminate the ring buffer (would propagate through all scores). */
    float pitch = frame.pitch_hz;
    float energy = frame.energy_db;
    if (isnan(pitch) || isinf(pitch)) pitch = 0.0f;
    if (isnan(energy) || isinf(energy)) energy = 0.0f;
    eou->pitch_buf[eou->prosody_write]  = pitch;
    eou->energy_buf[eou->prosody_write] = energy;
    eou->prosody_write = (eou->prosody_write + 1) % PROSODY_BUF_SIZE;
    eou->prosody_count++;
    if (eou->prosody_count > 1000000)
        eou->prosody_count = PROSODY_BUF_SIZE; /* Prevent overflow */
}

void fused_eou_set_prosody_weight(FusedEOU *eou, float w_prosody) {
    if (!eou) return;
    if (w_prosody < 0.0f) w_prosody = 0.0f;
    if (w_prosody > 0.5f) w_prosody = 0.5f; /* Cap to preserve core signals */
    eou->w_prosody = w_prosody;
}

float fused_eou_prosody_prob(const FusedEOU *eou) {
    return eou ? eou->prosody_prob : 0.0f;
}

float fused_eou_context_adjustment(const FusedEOU *eou) {
    return eou ? eou->context_adj : 0.0f;
}

/* ── Semantic Completion Signal ────────────────────────────── */

void fused_eou_feed_semantic(FusedEOU *eou, float prob) {
    if (!eou) return;
    /* Sanitize — clamp to [0, 1] and reject NaN/Inf */
    if (isnan(prob) || isinf(prob)) prob = 0.5f;
    if (prob < 0.0f) prob = 0.0f;
    if (prob > 1.0f) prob = 1.0f;
    eou->semantic_prob = prob;
}

void fused_eou_set_semantic_weight(FusedEOU *eou, float w_semantic) {
    if (!eou) return;
    if (w_semantic < 0.0f) w_semantic = 0.0f;
    if (w_semantic > 0.4f) w_semantic = 0.4f; /* Cap to preserve core signals */
    eou->w_semantic = w_semantic;
}

float fused_eou_semantic_prob(const FusedEOU *eou) {
    return eou ? eou->semantic_prob : 0.0f;
}
