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

    return eou;
}

void fused_eou_destroy(FusedEOU *eou) {
    free(eou);
}

EOUResult fused_eou_process(FusedEOU *eou, EOUSignals sig) {
    EOUResult res = {0};
    if (!eou) return res;

    eou->total_frames++;

    /* Track speech activity (any signal indicates speech if eot probs are low) */
    if (sig.energy_signal < 0.3f || sig.mimi_eot_prob < 0.2f) {
        eou->speech_detected = 1;
        eou->frames_since_speech = 0;
    } else {
        eou->frames_since_speech++;
    }

    /* ── Early Exit: single-signal high-confidence ────── */
    int src = 0;

    if (sig.energy_signal >= eou->solo_energy && eou->speech_detected) {
        src |= EOU_SRC_ENERGY;
    }
    if (sig.mimi_eot_prob >= eou->solo_mimi && eou->speech_detected) {
        src |= EOU_SRC_MIMI;
    }
    if (sig.stt_eou_prob >= eou->solo_stt && eou->speech_detected) {
        src |= EOU_SRC_STT;
    }

    /* ── Weighted Fusion ──────────────────────────────── */
    float fused = eou->w_energy * sig.energy_signal
                + eou->w_mimi  * sig.mimi_eot_prob
                + eou->w_stt   * sig.stt_eou_prob;

    /* EMA smoothing (alpha = 0.4 for fast response) */
    eou->smoothed_prob = 0.4f * fused + 0.6f * eou->smoothed_prob;

    res.fused_prob = eou->smoothed_prob;

    /* ── Trigger Logic ────────────────────────────────── */
    int should_trigger = 0;

    /* Solo trigger: any single signal above its threshold → immediate */
    if (src && eou->speech_detected) {
        should_trigger = 1;
    }

    /* Fused trigger: combined probability above threshold for N frames */
    if (eou->smoothed_prob >= eou->threshold && eou->speech_detected) {
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
    if (eou->trigger_source & EOU_SRC_FUSED)  src_str = "FUSED";
    else if (eou->trigger_source & EOU_SRC_MIMI) src_str = "MIMI";
    else if (eou->trigger_source & EOU_SRC_STT)  src_str = "STT";
    else if (eou->trigger_source & EOU_SRC_ENERGY) src_str = "ENERGY";

    fprintf(stderr, "[fused_eou] prob=%.3f trig=%d src=%s consec=%d "
            "latency=%.0fms frames=%d\n",
            eou->smoothed_prob, eou->triggered, src_str,
            eou->consec_count,
            (float)eou->frames_since_speech * eou->frame_ms,
            eou->total_frames);
}
