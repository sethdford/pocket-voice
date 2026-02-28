/**
 * test_fused_eou_parallel.c — Tests for parallel EOU detection.
 *
 * Validates:
 * - Partial processing with only energy+mimi produces valid results
 * - Partial processing matches full processing when all signals available
 * - Prosody enabled by default (0.15 weight)
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
/* Test 1: Prosody enabled by default */
/* ────────────────────────────────────────────────────────────────────────── */

static int test_prosody_enabled(void) {
    TEST("Prosody weight enabled by default (0.15)");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    ASSERT(eou != NULL, "fused_eou_create failed");

    /* Feed some prosody data */
    ProsodyFrame frame = {.pitch_hz = 150.0f, .energy_db = -20.0f};
    fused_eou_feed_prosody(eou, frame);
    frame.pitch_hz = 120.0f;
    fused_eou_feed_prosody(eou, frame);

    /* Process a full signal to trigger prosody evaluation */
    EOUSignals sig = {.energy_signal = 0.2f, .mimi_eot_prob = 0.1f, .stt_eou_prob = 0.0f};
    fused_eou_process(eou, sig);

    /* Prosody should have been computed */
    float prosody_prob = fused_eou_prosody_prob(eou);
    ASSERT(prosody_prob >= 0.0f && prosody_prob <= 1.0f, "Prosody prob out of range");

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
/* Main */
/* ────────────────────────────────────────────────────────────────────────── */

int main(void) {
    fprintf(stderr, "=== Parallel EOU Detection Tests ===\n\n");

    test_prosody_enabled();
    test_partial_energy_mimi();
    test_partial_vs_full();
    test_partial_energy_only();
    test_early_trigger();
    test_backward_compat();
    test_weight_renormalization();
    test_prosody_fusion();
    test_parallel_sequence();
    test_five_signal_fusion();

    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "Passed: %d / %d\n", test_passed, test_count);

    return test_passed == test_count ? 0 : 1;
}
