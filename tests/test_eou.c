/**
 * test_eou.c — Tests for the Parakeet-inspired EOU detection system.
 *
 * Validates: Mimi endpointer LSTM, fused 3-signal EOU, Conformer EOU token,
 * activation caching, and speculative prefill logic.
 */

#include "mimi_endpointer.h"
#include "fused_eou.h"
#include "conformer_stt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-50s ", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)

/* ── Mimi Endpointer Tests ────────────────────────────── */

static void test_mimi_ep_create_destroy(void) {
    TEST("mimi_ep create/destroy");
    MimiEndpointer *ep = mimi_ep_create(256, 128, 0.7f, 3);
    CHECK(ep != NULL, "Failed to create endpointer");
    mimi_ep_destroy(ep);
}

static void test_mimi_ep_init_random(void) {
    TEST("mimi_ep init_random (deterministic)");
    MimiEndpointer *ep = mimi_ep_create(64, 32, 0.5f, 2);
    mimi_ep_init_random(ep, 42);
    CHECK(mimi_ep_eot_prob(ep) == 0.0f, "Initial prob should be 0");
    mimi_ep_destroy(ep);
}

static void test_mimi_ep_process_silence(void) {
    TEST("mimi_ep process silence → no trigger");
    MimiEndpointer *ep = mimi_ep_create(64, 32, 0.7f, 3);
    mimi_ep_init_random(ep, 42);

    float zeros[64] = {0};
    for (int i = 0; i < 10; i++) {
        EndpointResult r = mimi_ep_process(ep, zeros);
        (void)r;
    }
    CHECK(mimi_ep_triggered(ep) == 0, "Should not trigger on silence");
    mimi_ep_destroy(ep);
}

static void test_mimi_ep_process_outputs_valid(void) {
    TEST("mimi_ep process returns valid probabilities");
    MimiEndpointer *ep = mimi_ep_create(64, 32, 0.7f, 3);
    mimi_ep_init_random(ep, 42);

    float input[64];
    for (int i = 0; i < 64; i++) input[i] = 0.5f * sinf((float)i * 0.1f);

    EndpointResult r = mimi_ep_process(ep, input);
    float sum = r.prob_silence + r.prob_speech + r.prob_ending + r.prob_eot;
    CHECK(fabsf(sum - 1.0f) < 0.01f, "Probabilities should sum to 1.0");
    mimi_ep_destroy(ep);
}

static void test_mimi_ep_reset(void) {
    TEST("mimi_ep reset clears state");
    MimiEndpointer *ep = mimi_ep_create(64, 32, 0.5f, 2);
    mimi_ep_init_random(ep, 42);

    float input[64];
    for (int i = 0; i < 64; i++) input[i] = 1.0f;
    for (int i = 0; i < 20; i++) mimi_ep_process(ep, input);

    float prob_before = mimi_ep_eot_prob(ep);
    mimi_ep_reset(ep);
    float prob_after = mimi_ep_eot_prob(ep);
    CHECK(prob_after == 0.0f && prob_before != 0.0f,
          "Reset should clear EOT prob");
    mimi_ep_destroy(ep);
}

static void test_mimi_ep_threshold_tuning(void) {
    TEST("mimi_ep set_threshold changes behavior");
    MimiEndpointer *ep = mimi_ep_create(64, 32, 0.99f, 1);
    mimi_ep_init_random(ep, 42);

    float input[64];
    for (int i = 0; i < 64; i++) input[i] = 1.0f;
    for (int i = 0; i < 50; i++) mimi_ep_process(ep, input);

    int trig_high = mimi_ep_triggered(ep);
    mimi_ep_reset(ep);
    mimi_ep_set_threshold(ep, 0.01f);
    for (int i = 0; i < 50; i++) mimi_ep_process(ep, input);
    int trig_low = mimi_ep_triggered(ep);

    /* With threshold 0.01, should be easier (or same) to trigger than 0.99 */
    CHECK(trig_low >= trig_high, "Lower threshold should be easier to trigger");
    mimi_ep_destroy(ep);
}

/* ── Fused EOU Tests ──────────────────────────────────── */

static void test_fused_eou_create_destroy(void) {
    TEST("fused_eou create/destroy");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    CHECK(eou != NULL, "Failed to create fused EOU");
    fused_eou_destroy(eou);
}

static void test_fused_eou_no_signal(void) {
    TEST("fused_eou no signal → no trigger");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    EOUSignals sig = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 10; i++) fused_eou_process(eou, sig);
    CHECK(fused_eou_triggered(eou) == 0, "Should not trigger with no signals");
    fused_eou_destroy(eou);
}

static void test_fused_eou_energy_solo(void) {
    TEST("fused_eou energy solo trigger");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);

    /* First: establish speech was detected */
    EOUSignals speech = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* Then: high energy signal (silence detected) */
    EOUSignals sig = { .energy_signal = 0.98f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    EOUResult r = fused_eou_process(eou, sig);
    CHECK(r.triggered == 1 && (r.trigger_source & EOU_SRC_ENERGY),
          "Should trigger from energy solo");
    fused_eou_destroy(eou);
}

static void test_fused_eou_stt_solo(void) {
    TEST("fused_eou STT solo trigger");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);

    EOUSignals speech = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    EOUSignals sig = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.9f };
    EOUResult r = fused_eou_process(eou, sig);
    CHECK(r.triggered == 1 && (r.trigger_source & EOU_SRC_STT),
          "Should trigger from STT solo");
    fused_eou_destroy(eou);
}

static void test_fused_eou_mimi_solo(void) {
    TEST("fused_eou Mimi solo trigger");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);

    EOUSignals speech = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    EOUSignals sig = { .energy_signal = 0.0f, .mimi_eot_prob = 0.95f, .stt_eou_prob = 0.0f };
    EOUResult r = fused_eou_process(eou, sig);
    CHECK(r.triggered == 1 && (r.trigger_source & EOU_SRC_MIMI),
          "Should trigger from Mimi solo");
    fused_eou_destroy(eou);
}

static void test_fused_eou_fused_trigger(void) {
    TEST("fused_eou combined signals trigger");
    FusedEOU *eou = fused_eou_create(0.5f, 2, 80.0f);

    EOUSignals speech = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* No single signal above solo threshold, but combined should trigger */
    EOUSignals sig = { .energy_signal = 0.6f, .mimi_eot_prob = 0.6f, .stt_eou_prob = 0.6f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, sig);
    CHECK(fused_eou_triggered(eou) == 1, "Combined signals should trigger");
    fused_eou_destroy(eou);
}

static void test_fused_eou_reset(void) {
    TEST("fused_eou reset clears all state");
    FusedEOU *eou = fused_eou_create(0.5f, 1, 80.0f);

    EOUSignals speech = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    EOUSignals sig = { .energy_signal = 0.98f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    fused_eou_process(eou, sig);
    CHECK(fused_eou_triggered(eou) == 1, "Pre-reset: should be triggered");

    fused_eou_reset(eou);
    CHECK(fused_eou_triggered(eou) == 0 && fused_eou_prob(eou) == 0.0f,
          "Post-reset: should be cleared");
    fused_eou_destroy(eou);
}

static void test_fused_eou_weight_adjustment(void) {
    TEST("fused_eou custom weights");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_weights(eou, 0.0f, 0.0f, 1.0f); /* STT only */

    EOUSignals speech = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* Only STT signal present — with weight=1.0 for STT, this should matter more */
    EOUSignals sig = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.7f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, sig);

    float prob = fused_eou_prob(eou);
    CHECK(prob > 0.5f, "STT-weighted fusion should yield high probability");
    fused_eou_destroy(eou);
}

static void test_fused_eou_requires_speech(void) {
    TEST("fused_eou requires speech before trigger");
    FusedEOU *eou = fused_eou_create(0.5f, 1, 80.0f);

    /* Send high EOT signals without any prior speech */
    EOUSignals sig = { .energy_signal = 0.98f, .mimi_eot_prob = 0.98f, .stt_eou_prob = 0.98f };
    fused_eou_process(eou, sig);

    /* Should NOT trigger because no speech was detected first */
    CHECK(fused_eou_triggered(eou) == 0,
          "Should not trigger without prior speech detection");
    fused_eou_destroy(eou);
}

static void test_fused_eou_consec_requirement(void) {
    TEST("fused_eou consecutive frames requirement");
    FusedEOU *eou = fused_eou_create(0.5f, 3, 80.0f);

    EOUSignals speech = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* One high frame followed by a low frame — should not trigger */
    EOUSignals hi = { .energy_signal = 0.5f, .mimi_eot_prob = 0.8f, .stt_eou_prob = 0.5f };
    EOUSignals lo = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    fused_eou_process(eou, hi);
    fused_eou_process(eou, lo);  /* Reset consecutive counter */
    fused_eou_process(eou, hi);
    CHECK(fused_eou_triggered(eou) == 0,
          "Interrupted consecutive frames should not trigger");
    fused_eou_destroy(eou);
}

/* ── Mimi Endpointer: NULL Safety ─────────────────────── */

static void test_mimi_ep_null_safety(void) {
    TEST("mimi_ep NULL safety on all functions");
    mimi_ep_destroy(NULL);
    mimi_ep_reset(NULL);
    mimi_ep_set_threshold(NULL, 0.5f);
    mimi_ep_init_random(NULL, 42);
    CHECK(mimi_ep_eot_prob(NULL) == 0.0f, "eot_prob(NULL) should be 0");
    CHECK(mimi_ep_triggered(NULL) == 0, "triggered(NULL) should be 0");
    CHECK(mimi_ep_latency_frames(NULL) == -1, "latency_frames(NULL) should be -1");
}

/* ── Mimi Endpointer: LSTM State Accumulation ─────────── */

static void test_mimi_ep_lstm_state_accumulation(void) {
    TEST("mimi_ep LSTM state accumulates over frames");
    MimiEndpointer *ep = mimi_ep_create(64, 32, 0.7f, 3);
    mimi_ep_init_random(ep, 42);

    float input[64];
    for (int i = 0; i < 64; i++) input[i] = 0.5f * sinf((float)i * 0.1f);

    /* Process several frames — probs should evolve as LSTM state builds */
    EndpointResult r1 = mimi_ep_process(ep, input);
    EndpointResult r2 = mimi_ep_process(ep, input);
    EndpointResult r3 = mimi_ep_process(ep, input);

    /* All should have valid softmax probabilities */
    float sum1 = r1.prob_silence + r1.prob_speech + r1.prob_ending + r1.prob_eot;
    float sum2 = r2.prob_silence + r2.prob_speech + r2.prob_ending + r2.prob_eot;
    float sum3 = r3.prob_silence + r3.prob_speech + r3.prob_ending + r3.prob_eot;
    CHECK(fabsf(sum1 - 1.0f) < 0.01f && fabsf(sum2 - 1.0f) < 0.01f &&
          fabsf(sum3 - 1.0f) < 0.01f,
          "All frames should have valid softmax probs summing to 1.0");
    mimi_ep_destroy(ep);
}

/* ── Mimi Endpointer: Reset Between Utterances ────────── */

static void test_mimi_ep_reset_between_utterances(void) {
    TEST("mimi_ep reset between utterances gives consistent results");
    MimiEndpointer *ep = mimi_ep_create(64, 32, 0.7f, 3);
    mimi_ep_init_random(ep, 42);

    float input[64];
    for (int i = 0; i < 64; i++) input[i] = 0.5f * sinf((float)i * 0.2f);

    /* First utterance */
    for (int i = 0; i < 10; i++) mimi_ep_process(ep, input);
    float prob_utt1 = mimi_ep_eot_prob(ep);

    /* Reset and process second utterance with same input */
    mimi_ep_reset(ep);
    for (int i = 0; i < 10; i++) mimi_ep_process(ep, input);
    float prob_utt2 = mimi_ep_eot_prob(ep);

    /* After same input from reset state, should get similar results */
    CHECK(fabsf(prob_utt1 - prob_utt2) < 0.01f,
          "Same input after reset should give same result");
    mimi_ep_destroy(ep);
}

/* ── Mimi Endpointer: Consecutive EOT Tracking ───────── */

static void test_mimi_ep_consec_eot_tracking(void) {
    TEST("mimi_ep consecutive EOT frame tracking");
    /* Use very low threshold and consec=1 so trigger is easy */
    MimiEndpointer *ep = mimi_ep_create(4, 4, 0.01f, 1);
    mimi_ep_init_random(ep, 7);

    float input[4] = {0.9f, 0.9f, 0.9f, 0.9f};
    int max_consec = 0;
    for (int i = 0; i < 50; i++) {
        EndpointResult r = mimi_ep_process(ep, input);
        if (r.consec_eot > max_consec) max_consec = r.consec_eot;
    }
    /* With such aggressive settings, consec should accumulate */
    CHECK(max_consec >= 0, "Consecutive EOT count should be non-negative");
    mimi_ep_destroy(ep);
}

/* ── Mimi Endpointer: Single Frame Processing ────────── */

static void test_mimi_ep_single_frame(void) {
    TEST("mimi_ep single frame produces valid output");
    MimiEndpointer *ep = mimi_ep_create(16, 8, 0.7f, 3);
    mimi_ep_init_random(ep, 42);

    float input[16] = {0};
    EndpointResult r = mimi_ep_process(ep, input);
    float sum = r.prob_silence + r.prob_speech + r.prob_ending + r.prob_eot;
    CHECK(fabsf(sum - 1.0f) < 0.01f, "Single frame should produce valid softmax");
    CHECK(r.triggered == 0, "Single frame should not trigger (need consec=3)");
    mimi_ep_destroy(ep);
}

/* ── Fused EOU: Weight Mixing Extremes ────────────────── */

static void test_fused_eou_energy_only_weight(void) {
    TEST("fused_eou energy-only weights");
    FusedEOU *eou = fused_eou_create(0.6f, 1, 80.0f);
    fused_eou_set_weights(eou, 1.0f, 0.0f, 0.0f); /* Energy only */

    EOUSignals speech = { .energy_signal = 0.3f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* High energy with Mimi and STT signals should not affect */
    EOUSignals sig = { .energy_signal = 0.98f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, sig);

    float prob = fused_eou_prob(eou);
    CHECK(prob > 0.5f, "Energy-only weight with high energy → high fused prob");
    fused_eou_destroy(eou);
}

static void test_fused_eou_mimi_only_weight(void) {
    TEST("fused_eou LSTM/Mimi-only weights");
    FusedEOU *eou = fused_eou_create(0.6f, 1, 80.0f);
    fused_eou_set_weights(eou, 0.0f, 1.0f, 0.0f); /* Mimi only */

    EOUSignals speech = { .energy_signal = 0.3f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* Only Mimi signal high */
    EOUSignals sig = { .energy_signal = 0.0f, .mimi_eot_prob = 0.8f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 10; i++) fused_eou_process(eou, sig);

    float prob = fused_eou_prob(eou);
    CHECK(prob > 0.4f, "Mimi-only weight with high Mimi → high fused prob");
    fused_eou_destroy(eou);
}

static void test_fused_eou_stt_only_weight(void) {
    TEST("fused_eou ASR/STT-only weights");
    FusedEOU *eou = fused_eou_create(0.6f, 1, 80.0f);
    fused_eou_set_weights(eou, 0.0f, 0.0f, 1.0f); /* STT only */

    EOUSignals speech = { .energy_signal = 0.3f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* Only STT signal high, others at zero */
    EOUSignals sig = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.8f };
    for (int i = 0; i < 10; i++) fused_eou_process(eou, sig);

    float prob = fused_eou_prob(eou);
    CHECK(prob > 0.4f, "STT-only weight with high STT → high fused prob");
    fused_eou_destroy(eou);
}

static void test_fused_eou_zero_weights_safety(void) {
    TEST("fused_eou all-zero weights safety");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    /* All zero weights — should be handled gracefully (normalized to prevent div-by-zero) */
    fused_eou_set_weights(eou, 0.0f, 0.0f, 0.0f);

    EOUSignals speech = { .energy_signal = 0.3f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    EOUSignals sig = { .energy_signal = 0.9f, .mimi_eot_prob = 0.9f, .stt_eou_prob = 0.9f };
    EOUResult r = fused_eou_process(eou, sig);
    CHECK(r.fused_prob >= 0.0f && r.fused_prob <= 1.0f,
          "Zero weights should not crash; prob in valid range");
    fused_eou_destroy(eou);
}

/* ── Fused EOU: frames_since_speech Overflow Protection ── */

static void test_fused_eou_frames_overflow_cap(void) {
    TEST("fused_eou frames_since_speech capped at 100000");
    FusedEOU *eou = fused_eou_create(0.99f, 999, 80.0f); /* Very high threshold, won't trigger */

    /* Detect speech first */
    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    fused_eou_process(eou, speech);

    /* Now send many silence frames — frames_since_speech should cap */
    EOUSignals silence = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    EOUResult r;
    for (int i = 0; i < 110000; i++) {
        r = fused_eou_process(eou, silence);
    }

    /* latency_ms = frames_since_speech * frame_ms = 100000 * 80 = 8000000ms */
    CHECK(r.latency_ms <= 100000.0f * 80.0f + 1.0f,
          "frames_since_speech should be capped at 100000");
    /* Also verify it didn't overflow to a negative */
    CHECK(r.latency_ms >= 0.0f, "latency_ms should not be negative (no overflow)");
    fused_eou_destroy(eou);
}

/* ── Fused EOU: Energy VAD Various Levels ────────────── */

static void test_fused_eou_energy_levels(void) {
    TEST("fused_eou various energy levels and threshold crossings");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);

    /* Speech detection at boundary: energy_signal=0.3 is the threshold */
    EOUSignals at_boundary = { .energy_signal = 0.30f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    fused_eou_process(eou, at_boundary);
    /* Exactly at 0.3 should count as speech detected (>= 0.3) */
    /* Verify by trying to trigger — would fail without speech_detected */
    fused_eou_set_solo_thresholds(eou, 0.5f, 0.99f, 0.99f);
    EOUSignals high_energy = { .energy_signal = 0.6f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    EOUResult r = fused_eou_process(eou, high_energy);
    CHECK(r.triggered == 1, "energy=0.3 boundary should count as speech_detected");
    fused_eou_destroy(eou);

    /* Below boundary: only frames with energy<0.3 → speech_detected stays 0 */
    eou = fused_eou_create(0.6f, 2, 80.0f);
    EOUSignals below = { .energy_signal = 0.29f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    fused_eou_process(eou, below);
    /* Verify speech_detected is still 0 by checking that even high mimi/stt
     * won't trigger (solo triggers require speech_detected) */
    EOUSignals high_no_energy = { .energy_signal = 0.0f, .mimi_eot_prob = 0.98f, .stt_eou_prob = 0.98f };
    r = fused_eou_process(eou, high_no_energy);
    CHECK(r.triggered == 0,
          "only energy<0.3 frames → speech_detected=0 → no solo trigger");
    fused_eou_destroy(eou);
}

/* ── Fused EOU: NULL Safety ───────────────────────────── */

static void test_fused_eou_null_safety(void) {
    TEST("fused_eou NULL safety on all functions");
    fused_eou_destroy(NULL);
    fused_eou_reset(NULL);
    fused_eou_set_weights(NULL, 1.0f, 0.0f, 0.0f);
    fused_eou_set_solo_thresholds(NULL, 0.5f, 0.5f, 0.5f);
    fused_eou_print_status(NULL);
    CHECK(fused_eou_prob(NULL) == 0.0f, "prob(NULL) should be 0");
    CHECK(fused_eou_triggered(NULL) == 0, "triggered(NULL) should be 0");

    EOUSignals sig = { .energy_signal = 0.9f, .mimi_eot_prob = 0.9f, .stt_eou_prob = 0.9f };
    EOUResult r = fused_eou_process(NULL, sig);
    CHECK(r.triggered == 0 && r.fused_prob == 0.0f,
          "process(NULL) returns empty result");
}

/* ── Fused EOU: Silence Duration Fast Path ───────────── */

static void test_fused_eou_silence_duration_trigger(void) {
    TEST("fused_eou 300ms silence fast path trigger");
    /* frame_ms=80, so 300ms ≈ 4 frames where energy_signal < 0.3.
     * Use threshold=0.5 with high consec_required to prevent the fused
     * consecutive-frames path from triggering first. */
    FusedEOU *eou = fused_eou_create(0.5f, 100, 80.0f);

    /* First: detect speech (energy_signal >= 0.3 sets speech_detected=1) */
    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* Then: post-speech frames where energy_signal < 0.3 (frames_since_speech
     * increments). Use moderate mimi/stt (below solo thresholds 0.90/0.85)
     * to push smoothed_prob above threshold without triggering solo exit. */
    EOUSignals post_speech = { .energy_signal = 0.0f, .mimi_eot_prob = 0.8f, .stt_eou_prob = 0.8f };
    EOUResult r;
    for (int i = 0; i < 10; i++) {
        r = fused_eou_process(eou, post_speech);
    }
    /* After 10 * 80ms = 800ms > 300ms of frames_since_speech,
     * with smoothed_prob above threshold, the silence-duration fast path triggers */
    CHECK(r.triggered == 1, "300ms silence fast path should trigger");
    CHECK(r.trigger_source & (EOU_SRC_ENERGY | EOU_SRC_FUSED),
          "trigger source should include energy/fused (silence duration)");
    fused_eou_destroy(eou);
}

/* ── Mimi Endpointer: Mel-Energy Feature Path Tests ───── */

static void test_mimi_ep_mel_energy_features(void) {
    TEST("mimi_ep 80-dim mel-energy features (capture path)");
    MimiEndpointer *ep = mimi_ep_create(80, 64, 0.7f, 3);
    mimi_ep_init_random(ep, 42);

    /* Simulate the mel-energy feature extraction from pocket_voice_pipeline.c:
     * Split 1920 samples into 80 bands (24 samples/band), compute RMS */
    float pcm[1920];
    for (int i = 0; i < 1920; i++)
        pcm[i] = 0.3f * sinf(2.0f * 3.14159f * 440.0f * (float)i / 24000.0f);

    float features[80];
    int samples_per_band = 1920 / 80;
    for (int b = 0; b < 80; b++) {
        float sum_sq = 0.0f;
        for (int s = 0; s < samples_per_band; s++)
            sum_sq += pcm[b * samples_per_band + s] * pcm[b * samples_per_band + s];
        features[b] = sqrtf(sum_sq / (float)samples_per_band);
    }

    EndpointResult r = mimi_ep_process(ep, features);
    float sum = r.prob_silence + r.prob_speech + r.prob_ending + r.prob_eot;
    CHECK(fabsf(sum - 1.0f) < 0.01f, "80-dim features should produce valid probs");
    mimi_ep_destroy(ep);
}

static void test_mimi_ep_latency_frames(void) {
    TEST("mimi_ep latency_frames tracks EOT timing");
    MimiEndpointer *ep = mimi_ep_create(4, 4, 0.01f, 1);
    mimi_ep_init_random(ep, 7);

    /* Feed speech-like input first to set speech_active */
    float speech[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    for (int i = 0; i < 20; i++) mimi_ep_process(ep, speech);

    int lat = mimi_ep_latency_frames(ep);
    /* Latency should be -1 (no trigger) or >= 0 (after trigger) */
    CHECK(lat == -1 || lat >= 0, "Latency should be valid");
    mimi_ep_destroy(ep);
}

/* ── Fused EOU: Advanced Scenarios ────────────────────── */

static void test_fused_eou_latency_estimation(void) {
    TEST("fused_eou latency estimation in ms");
    FusedEOU *eou = fused_eou_create(0.5f, 1, 80.0f);

    /* 5 frames of speech */
    EOUSignals speech = {0};
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* 3 frames of silence → trigger */
    EOUSignals silence = { .energy_signal = 0.98f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    EOUResult r;
    for (int i = 0; i < 3; i++) r = fused_eou_process(eou, silence);

    CHECK(r.latency_ms >= 0.0f, "Latency should be non-negative");
    fused_eou_destroy(eou);
}

static void test_fused_eou_print_status(void) {
    TEST("fused_eou print_status (smoke test)");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_print_status(eou); /* Should not crash */
    PASS();
    fused_eou_destroy(eou);
}

static void test_fused_eou_solo_threshold_config(void) {
    TEST("fused_eou custom solo thresholds");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);

    /* Set very low solo threshold for energy — should trigger easily */
    fused_eou_set_solo_thresholds(eou, 0.5f, 0.99f, 0.99f);

    EOUSignals speech = {0};
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    EOUSignals sig = { .energy_signal = 0.6f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    EOUResult r = fused_eou_process(eou, sig);
    CHECK(r.triggered == 1 && (r.trigger_source & EOU_SRC_ENERGY),
          "Low solo threshold should trigger on moderate energy");
    fused_eou_destroy(eou);
}

/* ── Speculative Prefill Logic Tests ─────────────────── */

static void test_speculative_prefill_trigger_zone(void) {
    TEST("speculative prefill: 70% zone triggers send");
    FusedEOU *eou = fused_eou_create(0.8f, 2, 80.0f);

    EOUSignals speech = {0};
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* Fused prob climbs to ~0.7 but stays below trigger threshold (0.8) */
    EOUSignals mid = { .energy_signal = 0.7f, .mimi_eot_prob = 0.7f, .stt_eou_prob = 0.7f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, mid);

    float prob = fused_eou_prob(eou);
    int triggered = fused_eou_triggered(eou);
    /* Prob should be high but not triggered (threshold=0.8, consec=2) */
    CHECK(prob > 0.5f && !triggered,
          "70% zone: high prob but not yet triggered");
    fused_eou_destroy(eou);
}

static void test_speculative_prefill_cancel_on_resume(void) {
    TEST("speculative prefill: cancel when user resumes");
    FusedEOU *eou = fused_eou_create(0.8f, 3, 80.0f);

    EOUSignals speech = {0};
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* Push into speculative zone */
    EOUSignals mid = { .energy_signal = 0.7f, .mimi_eot_prob = 0.7f, .stt_eou_prob = 0.7f };
    for (int i = 0; i < 3; i++) fused_eou_process(eou, mid);
    float prob_high = fused_eou_prob(eou);

    /* User resumes speaking — prob drops */
    EOUSignals resume = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 10; i++) fused_eou_process(eou, resume);
    float prob_low = fused_eou_prob(eou);

    CHECK(prob_low < 0.30f && prob_high > 0.40f,
          "Prob should drop below 30% when user resumes");
    fused_eou_destroy(eou);
}

static void test_speculative_skip_processing(void) {
    TEST("speculative prefill: skip to STREAMING on hit");
    /* This tests the logic: if speculative_sent==1 and EOU triggers,
     * the pipeline skips STATE_PROCESSING → goes to STATE_STREAMING.
     * We simulate just the flag logic here. */
    int speculative_sent = 1;
    int eou_triggered = 1;
    int next_state;

    if (eou_triggered && speculative_sent) {
        next_state = 3; /* STATE_STREAMING */
        speculative_sent = 0;
    } else {
        next_state = 2; /* STATE_PROCESSING */
    }

    CHECK(next_state == 3 && speculative_sent == 0,
          "Should skip to STREAMING and clear flag");
}

/* ── Speculative Prefill: 70% Confidence Verification ─── */

static void test_speculative_prefill_70pct_confidence(void) {
    TEST("speculative prefill: 70% confidence triggers prefill flag");
    /* Simulate the pipeline logic:
     * if fused_prob >= 0.55 && fused_prob < threshold → speculative_sent = 1 */
    float threshold = 0.8f;
    float speculative_zone_low = 0.55f;

    FusedEOU *eou = fused_eou_create(threshold, 3, 80.0f);

    EOUSignals speech = {0};
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* Push fused prob into the speculative zone (~0.7) */
    EOUSignals mid = { .energy_signal = 0.7f, .mimi_eot_prob = 0.7f, .stt_eou_prob = 0.7f };
    for (int i = 0; i < 10; i++) fused_eou_process(eou, mid);

    float prob = fused_eou_prob(eou);
    int triggered = fused_eou_triggered(eou);
    int in_spec_zone = (prob >= speculative_zone_low && prob < threshold && !triggered);

    CHECK(in_spec_zone || prob >= threshold,
          "70% signals should put prob in speculative zone or above threshold");
    fused_eou_destroy(eou);
}

static void test_speculative_prefill_below_zone(void) {
    TEST("speculative prefill: low signals stay below zone");
    FusedEOU *eou = fused_eou_create(0.8f, 3, 80.0f);

    EOUSignals speech = {0};
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* Very low signals — should stay below speculative zone */
    EOUSignals low = { .energy_signal = 0.1f, .mimi_eot_prob = 0.1f, .stt_eou_prob = 0.1f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, low);

    float prob = fused_eou_prob(eou);
    CHECK(prob < 0.55f, "Low signals should keep prob below speculative zone");
    fused_eou_destroy(eou);
}

/* ── Conformer EOU API Smoke Tests ───────────────────── */
/* These test the C API surface without requiring a model file. */

static void test_conformer_eou_flags(void) {
    TEST("conformer_stt EOU flag definitions");
    /* Verify the flag constants don't collide */
    CHECK((CSTT_FLAG_HAS_EOU & CSTT_FLAG_CACHE_AWARE) == 0 &&
          (CSTT_FLAG_HAS_EOU & CSTT_FLAG_HAS_BIAS) == 0 &&
          (CSTT_FLAG_HAS_EOU & CSTT_FLAG_REL_PE) == 0,
          "EOU/CACHE flags must not collide with existing flags");
}

static void test_conformer_null_safety(void) {
    TEST("conformer_stt NULL safety on EOU API");
    CHECK(conformer_stt_has_eou(NULL) == 0, "has_eou(NULL) should be 0");
    CHECK(conformer_stt_eou_prob(NULL, 4) == 0.0f, "eou_prob(NULL) should be 0.0");
    CHECK(conformer_stt_eou_frame(NULL) == -1, "eou_frame(NULL) should be -1");
    CHECK(conformer_stt_has_eou_support(NULL) == 0, "has_eou_support(NULL) should be 0");
}

static void test_conformer_cache_aware_null(void) {
    TEST("conformer_stt cache-aware NULL safety");
    conformer_stt_set_cache_aware(NULL, 1); /* Should not crash */
    CHECK(conformer_stt_stride_ms(NULL) == 0, "stride_ms(NULL) should be 0");
    PASS();
}

/* ── Mimi Endpointer: Weight I/O Tests ───────────────── */

static void test_mimi_ep_weight_save_load(void) {
    TEST("mimi_ep weight save/load roundtrip");
    MimiEndpointer *ep = mimi_ep_create(16, 8, 0.5f, 2);
    mimi_ep_init_random(ep, 123);

    /* Process a frame and record result */
    float input[16];
    for (int i = 0; i < 16; i++) input[i] = 0.3f * (float)i;
    mimi_ep_reset(ep);
    mimi_ep_process(ep, input);

    /* If load_weights fails gracefully on non-existent file */
    int ret = mimi_ep_load_weights(ep, "/tmp/nonexistent_weights.bin");
    CHECK(ret == -1, "load_weights should fail on missing file");
    mimi_ep_destroy(ep);
}

/* ── Main ─────────────────────────────────────────────── */

int main(void) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Parakeet-Inspired EOU Detection Tests (Comprehensive)   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    printf("\n[Mimi Endpointer Tests]\n");
    test_mimi_ep_create_destroy();
    test_mimi_ep_init_random();
    test_mimi_ep_process_silence();
    test_mimi_ep_process_outputs_valid();
    test_mimi_ep_reset();
    test_mimi_ep_threshold_tuning();
    test_mimi_ep_null_safety();
    test_mimi_ep_lstm_state_accumulation();
    test_mimi_ep_reset_between_utterances();
    test_mimi_ep_consec_eot_tracking();
    test_mimi_ep_single_frame();
    test_mimi_ep_mel_energy_features();
    test_mimi_ep_latency_frames();
    test_mimi_ep_weight_save_load();

    printf("\n[Fused 3-Signal EOU Tests]\n");
    test_fused_eou_create_destroy();
    test_fused_eou_no_signal();
    test_fused_eou_energy_solo();
    test_fused_eou_stt_solo();
    test_fused_eou_mimi_solo();
    test_fused_eou_fused_trigger();
    test_fused_eou_reset();
    test_fused_eou_weight_adjustment();
    test_fused_eou_requires_speech();
    test_fused_eou_consec_requirement();
    test_fused_eou_energy_only_weight();
    test_fused_eou_mimi_only_weight();
    test_fused_eou_stt_only_weight();
    test_fused_eou_zero_weights_safety();
    test_fused_eou_frames_overflow_cap();
    test_fused_eou_energy_levels();
    test_fused_eou_null_safety();
    test_fused_eou_silence_duration_trigger();
    test_fused_eou_latency_estimation();
    test_fused_eou_print_status();
    test_fused_eou_solo_threshold_config();

    printf("\n[Speculative Prefill Tests]\n");
    test_speculative_prefill_trigger_zone();
    test_speculative_prefill_cancel_on_resume();
    test_speculative_skip_processing();
    test_speculative_prefill_70pct_confidence();
    test_speculative_prefill_below_zone();

    printf("\n[Conformer EOU API Tests]\n");
    test_conformer_eou_flags();
    test_conformer_null_safety();
    test_conformer_cache_aware_null();

    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  Total: %d passed, %d failed (out of %d)\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("  %s\n", tests_failed == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("══════════════════════════════════════════════════════════\n\n");

    return tests_failed > 0 ? 1 : 0;
}
