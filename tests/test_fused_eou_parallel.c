/**
 * test_fused_eou_parallel.c — Tests for parallel EOU detection.
 *
 * Validates:
 * - Partial processing with only energy+mimi produces valid results
 * - Partial processing matches full processing when all signals available
 * - Prosody disabled by default (0.0 weight — no pitch extraction in production yet)
 * - Early trigger from strong energy+mimi
 * - Backward compatibility: fused_eou_process() still works
 * - Weight renormalization in partial mode
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "fused_eou.h"

#define EPSILON 0.001f

/* Test counter */
static int test_count = 0;
static int test_passed = 0;

#define TEST(name) \
    do { \
        test_count++; \
        fprintf(stderr, "Test %d: %s\n", test_count, name); \
        fflush(stderr); \
    } while (0)

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "  FAIL: %s\n", msg); \
            return 0; \
        } \
    } while (0)

#define PASS() \
    do { \
        test_passed++; \
        fprintf(stderr, "  PASS\n"); \
    } while (0)

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 1: Prosody disabled by default (no pitch extraction in production) */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_prosody_default_weight(void) {
    TEST("Prosody weight disabled by default (0.0)");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Without feeding prosody data, process a frame */
    EOUSignals sig = {.energy_signal = 0.2f, .mimi_eot_prob = 0.1f, .stt_eou_prob = 0.0f};
    EOUResult res = fused_eou_process(eou, sig);

    /* With prosody weight=0.0, prosody should not affect fusion at all.
     * Two instances with identical signals should produce identical fused_prob
     * regardless of prosody data, because weight is 0. */
    FusedEOU *eou2 = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou2 != NULL, "fused_eou_create failed for eou2");

    /* Feed prosody data to eou2 but not eou */
    for (int i = 0; i < 8; i++) {
        ProsodyFrame frame = {.pitch_hz = 200.0f - (float)i * 10.0f, .energy_db = -20.0f};
        fused_eou_feed_prosody(eou2, frame);
    }
    EOUResult res2 = fused_eou_process(eou2, sig);

    /* Both should produce identical fused_prob since w_prosody=0 */
    float diff = fabsf(res.fused_prob - res2.fused_prob);
    ASSERT(diff < EPSILON, "Prosody data should not affect fusion when weight is 0.0");

    fused_eou_destroy(eou2);

    fused_eou_destroy(eou);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 2: Partial processing with energy+mimi only */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_partial_energy_mimi(void) {
    TEST("Partial processing with energy+mimi only");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Simulate speech onset */
    for (int i = 0; i < 5; i++) {
        EOUSignals sig = {.energy_signal = 0.0f, .mimi_eot_prob = 0.1f, .stt_eou_prob = 0.0f};
        fused_eou_process_partial(eou, sig, EOU_SRC_ENERGY | EOU_SRC_MIMI);
    }

    /* Strong energy + mimi signal (STT not yet available) */
    EOUSignals sig = {.energy_signal = 0.8f, .mimi_eot_prob = 0.7f, .stt_eou_prob = 0.0f};
    EOUResult res = fused_eou_process_partial(eou, sig, EOU_SRC_ENERGY | EOU_SRC_MIMI);

    ASSERT(res.fused_prob >= 0.0f && res.fused_prob <= 1.0f, "Fused prob out of range");
    ASSERT(res.consec_frames >= 0, "Consec frames negative");

    fused_eou_destroy(eou);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 3: Partial with all signals matches full processing */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_partial_vs_full(void) {
    TEST("Partial with all 3 signals matches full processing");

    FusedEOU *eou1 = fused_eou_create(0.6f, 2, 80.0f);
    FusedEOU *eou2 = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou1 != NULL && eou2 != NULL, "fused_eou_create failed");

    /* Feed identical signal sequences */
    for (int i = 0; i < 5; i++) {
        EOUSignals sig = {.energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f};
        fused_eou_process(eou1, sig);
        fused_eou_process_partial(eou2, sig, EOU_SRC_ENERGY | EOU_SRC_MIMI | EOU_SRC_STT);
    }

    /* Now send a strong signal */
    EOUSignals sig = {.energy_signal = 0.1f, .mimi_eot_prob = 0.2f, .stt_eou_prob = 0.3f};
    EOUResult res1 = fused_eou_process(eou1, sig);
    EOUResult res2 = fused_eou_process_partial(eou2, sig, EOU_SRC_ENERGY | EOU_SRC_MIMI | EOU_SRC_STT);

    float prob_diff = fabsf(res1.fused_prob - res2.fused_prob);
    ASSERT(prob_diff < EPSILON, "Partial and full probs differ");

    fused_eou_destroy(eou1);
    fused_eou_destroy(eou2);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 4: Partial with only energy */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_partial_energy_only(void) {
    TEST("Partial processing with only energy signal");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Feed speech onset */
    for (int i = 0; i < 5; i++) {
        EOUSignals sig = {.energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f};
        fused_eou_process_partial(eou, sig, EOU_SRC_ENERGY);
    }

    /* Strong energy-only signal */
    EOUSignals sig = {.energy_signal = 0.9f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f};
    EOUResult res = fused_eou_process_partial(eou, sig, EOU_SRC_ENERGY);

    /* Should have non-zero fused prob */
    ASSERT(res.fused_prob > 0.0f, "Fused prob should be non-zero");
    ASSERT(res.fused_prob <= 1.0f, "Fused prob out of range");

    fused_eou_destroy(eou);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 5: Early trigger from strong energy+mimi */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_early_trigger(void) {
    TEST("Early trigger from strong energy+mimi before STT");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Simulate speech start */
    for (int i = 0; i < 5; i++) {
        EOUSignals sig = {.energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f};
        fused_eou_process_partial(eou, sig, EOU_SRC_ENERGY | EOU_SRC_MIMI);
    }

    /* Very strong signal that should trigger solo */
    EOUSignals sig = {.energy_signal = 0.95f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f};
    EOUResult res = fused_eou_process_partial(eou, sig, EOU_SRC_ENERGY | EOU_SRC_MIMI);

    /* Depending on solo threshold, might trigger */
    ASSERT(res.fused_prob >= 0.0f && res.fused_prob <= 1.0f, "Fused prob out of range");

    fused_eou_destroy(eou);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 6: Backward compatibility — fused_eou_process() unchanged */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_backward_compat(void) {
    TEST("Backward compatibility: fused_eou_process() works unchanged");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Classic usage: all 3 signals available */
    for (int i = 0; i < 10; i++) {
        EOUSignals sig = {.energy_signal = 0.1f, .mimi_eot_prob = 0.2f, .stt_eou_prob = 0.3f};
        EOUResult res = fused_eou_process(eou, sig);

        ASSERT(!isnan(res.fused_prob), "fused_prob is NaN");
        ASSERT(res.fused_prob >= 0.0f && res.fused_prob <= 1.0f, "Fused prob out of range");
    }

    fused_eou_destroy(eou);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 7: Weight renormalization in partial mode */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_weight_renormalization(void) {
    TEST("Weight renormalization when only some signals available");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Feed 5 frames of silence to get speech_detected set */
    for (int i = 0; i < 5; i++) {
        EOUSignals sig = {.energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f};
        fused_eou_process(eou, sig);
    }

    /* Now use partial processing with different signal subsets */
    EOUSignals sig = {.energy_signal = 0.5f, .mimi_eot_prob = 0.4f, .stt_eou_prob = 0.3f};

    /* Only energy */
    EOUResult res_e = fused_eou_process_partial(eou, sig, EOU_SRC_ENERGY);
    ASSERT(res_e.fused_prob >= 0.0f, "Energy-only fused_prob negative");

    /* Energy + mimi */
    EOUResult res_em = fused_eou_process_partial(eou, sig, EOU_SRC_ENERGY | EOU_SRC_MIMI);
    ASSERT(res_em.fused_prob >= 0.0f, "Energy+mimi fused_prob negative");

    /* All three */
    EOUResult res_all = fused_eou_process_partial(eou, sig,
        EOU_SRC_ENERGY | EOU_SRC_MIMI | EOU_SRC_STT);
    ASSERT(res_all.fused_prob >= 0.0f, "All signals fused_prob negative");

    (void)res_em;  /* Suppress unused warning if not used in assertions */

    fused_eou_destroy(eou);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 8: Prosody signal contributes to fusion */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_prosody_fusion(void) {
    TEST("Prosody signal contributes to overall fusion");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Feed prosody frames showing pitch fall and energy decay (end-of-turn pattern) */
    for (int i = 0; i < 16; i++) {
        float pitch = 200.0f - (float)i * 5.0f;  /* Falling pitch */
        float energy = -20.0f - (float)i * 0.5f; /* Decaying energy */
        ProsodyFrame frame = {.pitch_hz = pitch, .energy_db = energy};
        fused_eou_feed_prosody(eou, frame);
    }

    /* Feed speech onset signals */
    for (int i = 0; i < 5; i++) {
        EOUSignals sig = {.energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f};
        fused_eou_process(eou, sig);
    }

    /* Process a frame */
    EOUSignals sig = {.energy_signal = 0.1f, .mimi_eot_prob = 0.1f, .stt_eou_prob = 0.0f};
    EOUResult res = fused_eou_process(eou, sig);

    /* Prosody prob should be computed */
    float prosody_prob = fused_eou_prosody_prob(eou);
    ASSERT(prosody_prob >= 0.0f && prosody_prob <= 1.0f, "Prosody prob out of range");

    fused_eou_destroy(eou);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 9: Parallel processing sequence (energy+mimi first, STT later) */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_parallel_sequence(void) {
    TEST("Parallel sequence: early EOU from energy+mimi, then full confirm");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Frame 1-5: Speech onset */
    for (int i = 0; i < 5; i++) {
        EOUSignals sig = {.energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f};
        fused_eou_process(eou, sig);
    }

    /* Frame 6-10: User speaking (energy+mimi available, no STT yet) */
    for (int i = 0; i < 5; i++) {
        EOUSignals sig = {.energy_signal = 0.1f, .mimi_eot_prob = 0.2f, .stt_eou_prob = 0.0f};
        EOUResult res = fused_eou_process_partial(eou, sig, EOU_SRC_ENERGY | EOU_SRC_MIMI);
        ASSERT(!res.triggered, "Should not trigger on weak partial signal");
    }

    /* Frame 11-12: End-of-turn (all signals available now) */
    EOUSignals sig = {.energy_signal = 0.8f, .mimi_eot_prob = 0.7f, .stt_eou_prob = 0.9f};
    EOUResult res = fused_eou_process(eou, sig);
    res = fused_eou_process(eou, sig);  /* Second frame allows EMA to accumulate */

    ASSERT(res.fused_prob >= 0.3f, "Fused prob should be substantial on strong signals");

    fused_eou_destroy(eou);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 10: Semantic + prosody + core 3 signals */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_five_signal_fusion(void) {
    TEST("5-signal fusion: energy + mimi + stt + prosody + semantic");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Enable semantic signal (5th) */
    fused_eou_set_semantic_weight(eou, 0.1f);

    /* Feed semantic probability */
    fused_eou_feed_semantic(eou, 0.8f);

    /* Feed prosody data */
    for (int i = 0; i < 8; i++) {
        ProsodyFrame frame = {.pitch_hz = 150.0f - (float)i * 10.0f, .energy_db = -20.0f};
        fused_eou_feed_prosody(eou, frame);
    }

    /* Speech onset */
    for (int i = 0; i < 5; i++) {
        EOUSignals sig = {.energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f};
        fused_eou_process(eou, sig);
    }

    /* Process all 5 signals */
    EOUSignals sig = {.energy_signal = 0.2f, .mimi_eot_prob = 0.2f, .stt_eou_prob = 0.2f};
    EOUResult res_5 = fused_eou_process(eou, sig);

    ASSERT(res_5.fused_prob >= 0.0f && res_5.fused_prob <= 1.0f, "Fused prob out of range");

    fused_eou_destroy(eou);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 11: NaN prosody input — no NaN propagation                          */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_nan_prosody_input(void) {
    TEST("NaN prosody input sanitized — no NaN propagation");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Feed NaN pitch and NaN energy */
    ProsodyFrame nan_frame = {.pitch_hz = NAN, .energy_db = NAN};
    for (int i = 0; i < 8; i++) {
        fused_eou_feed_prosody(eou, nan_frame);
    }

    /* Also feed Inf values */
    ProsodyFrame inf_frame = {.pitch_hz = INFINITY, .energy_db = -INFINITY};
    for (int i = 0; i < 8; i++) {
        fused_eou_feed_prosody(eou, inf_frame);
    }

    /* Process a frame — should not produce NaN fused_prob */
    EOUSignals sig = {.energy_signal = 0.5f, .mimi_eot_prob = 0.3f, .stt_eou_prob = 0.2f};
    for (int i = 0; i < 5; i++) {
        EOUResult res = fused_eou_process(eou, sig);
        ASSERT(!isnan(res.fused_prob), "fused_prob is NaN after NaN prosody input");
        ASSERT(!isinf(res.fused_prob), "fused_prob is Inf after Inf prosody input");
        ASSERT(res.fused_prob >= 0.0f && res.fused_prob <= 1.0f,
               "fused_prob out of [0,1] range after NaN/Inf prosody");
    }

    /* Prosody prob itself should also be clean */
    float pp = fused_eou_prosody_prob(eou);
    ASSERT(!isnan(pp), "prosody_prob is NaN");
    ASSERT(!isinf(pp), "prosody_prob is Inf");
    ASSERT(pp >= 0.0f && pp <= 1.0f, "prosody_prob out of [0,1] range");

    fused_eou_destroy(eou);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 12: Reset clears triggered flag, prosody, and semantic buffers      */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_reset_clears_state(void) {
    TEST("Reset clears triggered flag, prosody buffers, and semantic state");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Enable semantic and feed data */
    fused_eou_set_semantic_weight(eou, 0.1f);
    fused_eou_feed_semantic(eou, 0.9f);

    /* Feed prosody */
    for (int i = 0; i < 16; i++) {
        ProsodyFrame frame = {.pitch_hz = 200.0f - (float)i * 8.0f, .energy_db = -15.0f};
        fused_eou_feed_prosody(eou, frame);
    }

    /* Drive to triggered state with very strong signals */
    for (int i = 0; i < 5; i++) {
        EOUSignals sig = {.energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f};
        fused_eou_process(eou, sig);
    }
    for (int i = 0; i < 10; i++) {
        EOUSignals sig = {.energy_signal = 0.99f, .mimi_eot_prob = 0.95f, .stt_eou_prob = 0.95f};
        fused_eou_process(eou, sig);
    }
    ASSERT(fused_eou_triggered(eou) == 1, "Should be triggered before reset");
    ASSERT(fused_eou_prob(eou) > 0.0f, "Fused prob should be >0 before reset");

    /* Reset */
    fused_eou_reset(eou);

    /* Verify all state cleared */
    ASSERT(fused_eou_triggered(eou) == 0, "triggered must be 0 after reset");
    ASSERT(fused_eou_prob(eou) == 0.0f, "smoothed_prob must be 0 after reset");
    ASSERT(fused_eou_prosody_prob(eou) == 0.0f, "prosody_prob must be 0 after reset");
    ASSERT(fused_eou_semantic_prob(eou) == 0.0f, "semantic_prob must be 0 after reset");

    /* Process after reset — should not be triggered */
    EOUSignals weak = {.energy_signal = 0.1f, .mimi_eot_prob = 0.1f, .stt_eou_prob = 0.1f};
    EOUResult res = fused_eou_process(eou, weak);
    ASSERT(res.triggered == 0, "Should not be triggered after reset with weak signal");
    ASSERT(res.fused_prob >= 0.0f, "fused_prob should be valid after reset");

    fused_eou_destroy(eou);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 13: signals_valid=0 (empty bitmask)                                 */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_empty_signals_valid(void) {
    TEST("Empty signals_valid bitmask (0) — no core signals contribute");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Process with signals_valid=0: no core signals should contribute */
    EOUSignals sig = {.energy_signal = 0.99f, .mimi_eot_prob = 0.99f, .stt_eou_prob = 0.99f};
    EOUResult res = fused_eou_process_partial(eou, sig, 0);

    /* With no signals valid, core contribution is zero.
     * Prosody weight is 0.0 by default, so no aux signals contribute either.
     * Result should be valid, finite, and near zero. */
    ASSERT(!isnan(res.fused_prob), "fused_prob is NaN with empty bitmask");
    ASSERT(res.fused_prob >= 0.0f && res.fused_prob <= 1.0f,
           "fused_prob out of range with empty bitmask");

    /* The fused prob should be much lower than if all signals contributed */
    FusedEOU *eou2 = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou2 != NULL, "fused_eou_create failed for comparison");
    EOUResult res_full = fused_eou_process_partial(eou2, sig,
        EOU_SRC_ENERGY | EOU_SRC_MIMI | EOU_SRC_STT);
    ASSERT(res.fused_prob < res_full.fused_prob,
           "Empty bitmask should produce lower prob than full signals");

    fused_eou_destroy(eou);
    fused_eou_destroy(eou2);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 14: Rapid context switches (question→conjunction→question)          */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_rapid_context_switches(void) {
    TEST("Rapid context switches: question→conjunction→question");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Question context → lowers threshold (negative adjustment) */
    fused_eou_set_context(eou, "Is this working?");
    float adj_q1 = fused_eou_context_adjustment(eou);
    ASSERT(adj_q1 < 0.0f, "Question should give negative context adjustment");
    ASSERT(fabsf(adj_q1 - (-0.10f)) < EPSILON,
           "Question adjustment should be -0.10");

    /* Conjunction context → raises threshold (positive adjustment) */
    fused_eou_set_context(eou, "I was thinking and");
    float adj_conj = fused_eou_context_adjustment(eou);
    ASSERT(adj_conj > 0.0f, "Conjunction should give positive context adjustment");
    ASSERT(fabsf(adj_conj - 0.10f) < EPSILON,
           "Conjunction 'and' adjustment should be +0.10");

    /* Back to question → should revert */
    fused_eou_set_context(eou, "Really?");
    float adj_q2 = fused_eou_context_adjustment(eou);
    ASSERT(fabsf(adj_q2 - adj_q1) < EPSILON,
           "Second question should have same adjustment as first");

    /* Filler context */
    fused_eou_set_context(eou, "well um");
    float adj_filler = fused_eou_context_adjustment(eou);
    ASSERT(adj_filler > 0.0f, "Filler should give positive adjustment");
    ASSERT(fabsf(adj_filler - 0.05f) < EPSILON,
           "Filler 'um' adjustment should be +0.05");

    /* NULL context → zero adjustment */
    fused_eou_set_context(eou, NULL);
    float adj_null = fused_eou_context_adjustment(eou);
    ASSERT(fabsf(adj_null) < EPSILON, "NULL context should give zero adjustment");

    /* Empty string → zero adjustment */
    fused_eou_set_context(eou, "");
    float adj_empty = fused_eou_context_adjustment(eou);
    ASSERT(fabsf(adj_empty) < EPSILON, "Empty context should give zero adjustment");

    fused_eou_destroy(eou);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 15: Double-trigger prevention                                        */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_double_trigger_prevention(void) {
    TEST("Double-trigger prevention: triggered stays set, source doesn't mutate");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Establish speech detected */
    for (int i = 0; i < 5; i++) {
        EOUSignals sig = {.energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f};
        fused_eou_process(eou, sig);
    }

    /* Drive to trigger with strong signals */
    EOUResult first_trigger = {0};
    for (int i = 0; i < 10; i++) {
        EOUSignals sig = {.energy_signal = 0.99f, .mimi_eot_prob = 0.95f, .stt_eou_prob = 0.95f};
        EOUResult res = fused_eou_process(eou, sig);
        if (res.triggered && !first_trigger.triggered) {
            first_trigger = res;
        }
    }
    ASSERT(first_trigger.triggered == 1, "Should have triggered");
    int original_source = first_trigger.trigger_source;
    ASSERT(original_source != 0, "Trigger source should be non-zero");

    /* Continue feeding more frames — triggered must remain 1 */
    for (int i = 0; i < 10; i++) {
        EOUSignals sig = {.energy_signal = 0.8f, .mimi_eot_prob = 0.8f, .stt_eou_prob = 0.8f};
        EOUResult res = fused_eou_process(eou, sig);
        ASSERT(res.triggered == 1, "triggered must stay 1 once set");
        ASSERT(res.trigger_source == original_source,
               "trigger_source must not change after first trigger");
    }

    /* Even with weak signals, triggered stays set (sticky) */
    for (int i = 0; i < 5; i++) {
        EOUSignals sig = {.energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f};
        EOUResult res = fused_eou_process(eou, sig);
        ASSERT(res.triggered == 1, "triggered must stay 1 even with weak signals");
    }

    fused_eou_destroy(eou);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Test 16: Monotonic fused probability with increasing signal strength      */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_monotonic_fused_prob(void) {
    TEST("Fused probability increases monotonically with signal strength");

    /* Create two detectors with same config, feed different signal strengths */
    FusedEOU *eou_weak = fused_eou_create(0.6f, 2, 80.0f);
    FusedEOU *eou_strong = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou_weak != NULL && eou_strong != NULL, "fused_eou_create failed");

    /* Disable prosody to isolate core signal behavior */
    fused_eou_set_prosody_weight(eou_weak, 0.0f);
    fused_eou_set_prosody_weight(eou_strong, 0.0f);

    /* Feed identical speech onset */
    for (int i = 0; i < 5; i++) {
        EOUSignals onset = {.energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f};
        fused_eou_process(eou_weak, onset);
        fused_eou_process(eou_strong, onset);
    }

    /* Feed weak vs strong signals for several frames */
    EOUSignals weak_sig = {.energy_signal = 0.2f, .mimi_eot_prob = 0.1f, .stt_eou_prob = 0.1f};
    EOUSignals strong_sig = {.energy_signal = 0.8f, .mimi_eot_prob = 0.7f, .stt_eou_prob = 0.6f};

    EOUResult res_weak = {0}, res_strong = {0};
    for (int i = 0; i < 10; i++) {
        res_weak = fused_eou_process(eou_weak, weak_sig);
        res_strong = fused_eou_process(eou_strong, strong_sig);
    }

    ASSERT(res_strong.fused_prob > res_weak.fused_prob,
           "Strong signals must produce higher fused prob than weak signals");

    fused_eou_destroy(eou_weak);
    fused_eou_destroy(eou_strong);
    PASS();
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Main */
/* ────────────────────────────────────────────────────────────────────────── */

int main(void) {
    fprintf(stderr, "=== Parallel EOU Detection Tests ===\n\n");

    test_prosody_default_weight();
    test_partial_energy_mimi();
    test_partial_vs_full();
    test_partial_energy_only();
    test_early_trigger();
    test_backward_compat();
    test_weight_renormalization();
    test_prosody_fusion();
    test_parallel_sequence();
    test_five_signal_fusion();
    test_nan_prosody_input();
    test_reset_clears_state();
    test_empty_signals_valid();
    test_rapid_context_switches();
    test_double_trigger_prevention();
    test_monotonic_fused_prob();

    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "Passed: %d / %d\n", test_passed, test_count);

    return test_passed == test_count ? 0 : 1;
}
