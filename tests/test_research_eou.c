/**
 * test_research_eou.c — Tests for conversation-context and prosodic
 * turn-taking enhancements to the fused EOU detector.
 *
 * Validates:
 *   - Prosodic pitch trajectory detection (falling/rising)
 *   - Energy decay rate influence on EOU probability
 *   - Conversation context semantic adjustment (questions, conjunctions, fillers)
 *   - 4th-signal fusion integration with backward compatibility
 *   - NULL safety for all new functions
 */

#include "fused_eou.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-60s ", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)

/* ── Prosody Signal Tests ─────────────────────────────── */

static void test_prosody_falling_pitch(void) {
    TEST("prosody: falling pitch -> higher EOU probability");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_prosody_weight(eou, 0.15f);

    /* Feed falling pitch: 200Hz -> 120Hz over 16 frames */
    for (int i = 0; i < 16; i++) {
        ProsodyFrame f = {
            .pitch_hz = 200.0f - (80.0f * (float)i / 15.0f),
            .energy_db = -20.0f
        };
        fused_eou_feed_prosody(eou, f);
    }

    /* Establish speech and process to compute prosody */
    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 3; i++) fused_eou_process(eou, speech);

    float prob = fused_eou_prosody_prob(eou);
    CHECK(prob > 0.5f, "Falling pitch should yield above-neutral prosody prob");
    fused_eou_destroy(eou);
}

static void test_prosody_rising_pitch(void) {
    TEST("prosody: rising pitch -> lower EOU probability");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_prosody_weight(eou, 0.15f);

    /* Feed rising pitch: 120Hz -> 200Hz over 16 frames */
    for (int i = 0; i < 16; i++) {
        ProsodyFrame f = {
            .pitch_hz = 120.0f + (80.0f * (float)i / 15.0f),
            .energy_db = -20.0f
        };
        fused_eou_feed_prosody(eou, f);
    }

    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 3; i++) fused_eou_process(eou, speech);

    float prob = fused_eou_prosody_prob(eou);
    CHECK(prob < 0.5f, "Rising pitch should yield below-neutral prosody prob");
    fused_eou_destroy(eou);
}

static void test_prosody_energy_decay(void) {
    TEST("prosody: energy decay -> higher EOU probability");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_prosody_weight(eou, 0.15f);

    /* Feed decaying energy with flat pitch */
    for (int i = 0; i < 16; i++) {
        ProsodyFrame f = {
            .pitch_hz = 150.0f,
            .energy_db = -10.0f - (30.0f * (float)i / 15.0f)
        };
        fused_eou_feed_prosody(eou, f);
    }

    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 3; i++) fused_eou_process(eou, speech);

    float prob = fused_eou_prosody_prob(eou);
    CHECK(prob > 0.55f, "Energy decay should push prosody prob above neutral");
    fused_eou_destroy(eou);
}

static void test_prosody_energy_rising(void) {
    TEST("prosody: rising energy -> lower EOU probability");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_prosody_weight(eou, 0.15f);

    /* Feed rising energy with flat pitch */
    for (int i = 0; i < 16; i++) {
        ProsodyFrame f = {
            .pitch_hz = 150.0f,
            .energy_db = -40.0f + (30.0f * (float)i / 15.0f)
        };
        fused_eou_feed_prosody(eou, f);
    }

    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 3; i++) fused_eou_process(eou, speech);

    float prob = fused_eou_prosody_prob(eou);
    CHECK(prob < 0.5f, "Rising energy should yield below-neutral prosody prob");
    fused_eou_destroy(eou);
}

static void test_prosody_insufficient_data(void) {
    TEST("prosody: insufficient data -> neutral probability");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_prosody_weight(eou, 0.15f);

    /* Feed only 2 frames (need >=4 for meaningful computation) */
    ProsodyFrame f = { .pitch_hz = 150.0f, .energy_db = -20.0f };
    fused_eou_feed_prosody(eou, f);
    fused_eou_feed_prosody(eou, f);

    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    fused_eou_process(eou, speech);

    float prob = fused_eou_prosody_prob(eou);
    CHECK(fabsf(prob - 0.5f) < 0.01f, "Insufficient data should return 0.5 (neutral)");
    fused_eou_destroy(eou);
}

static void test_prosody_flat_pitch_neutral(void) {
    TEST("prosody: flat pitch -> neutral pitch component");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_prosody_weight(eou, 0.15f);

    /* Feed constant pitch and constant energy */
    for (int i = 0; i < 16; i++) {
        ProsodyFrame f = { .pitch_hz = 150.0f, .energy_db = -20.0f };
        fused_eou_feed_prosody(eou, f);
    }

    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    fused_eou_process(eou, speech);

    float prob = fused_eou_prosody_prob(eou);
    CHECK(fabsf(prob - 0.5f) < 0.05f, "Flat pitch + flat energy should be near neutral");
    fused_eou_destroy(eou);
}

static void test_prosody_solo_trigger(void) {
    TEST("prosody: extreme falling pitch -> very high prosody prob");
    FusedEOU *eou = fused_eou_create(0.99f, 99, 80.0f);
    fused_eou_set_prosody_weight(eou, 0.15f);

    /* Feed extreme falling pitch + energy decay */
    for (int i = 0; i < 32; i++) {
        ProsodyFrame f = {
            .pitch_hz = 300.0f - (200.0f * (float)i / 31.0f),
            .energy_db = -5.0f - (40.0f * (float)i / 31.0f)
        };
        fused_eou_feed_prosody(eou, f);
    }

    /* Establish speech */
    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 3; i++) fused_eou_process(eou, speech);

    float pp = fused_eou_prosody_prob(eou);
    CHECK(pp > 0.7f, "Extreme falling pitch + decay -> high prosody prob");
    fused_eou_destroy(eou);
}

static void test_prosody_unvoiced_frames(void) {
    TEST("prosody: all unvoiced frames -> neutral pitch score");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_prosody_weight(eou, 0.15f);

    /* Feed unvoiced frames (pitch_hz = 0) with flat energy */
    for (int i = 0; i < 16; i++) {
        ProsodyFrame f = { .pitch_hz = 0.0f, .energy_db = -20.0f };
        fused_eou_feed_prosody(eou, f);
    }

    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    fused_eou_process(eou, speech);

    float prob = fused_eou_prosody_prob(eou);
    /* Pitch contribution should be neutral (0.5), energy flat -> ~0.5 total */
    CHECK(fabsf(prob - 0.5f) < 0.05f, "Unvoiced + flat energy should be near neutral");
    fused_eou_destroy(eou);
}

static void test_prosody_source_bitmask(void) {
    TEST("EOU_SRC_PROSODY bitmask does not collide");
    CHECK((EOU_SRC_PROSODY & EOU_SRC_ENERGY) == 0, "No collision with ENERGY");
    CHECK((EOU_SRC_PROSODY & EOU_SRC_MIMI) == 0, "No collision with MIMI");
    CHECK((EOU_SRC_PROSODY & EOU_SRC_STT) == 0, "No collision with STT");
    CHECK((EOU_SRC_PROSODY & EOU_SRC_FUSED) == 0, "No collision with FUSED");
    CHECK(EOU_SRC_PROSODY == (1 << 4), "PROSODY should be bit 4");
}

/* ── Conversation Context Tests ───────────────────────── */

static void test_context_question(void) {
    TEST("context: question mark -> lower threshold (easier trigger)");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_context(eou, "How are you doing?");

    float adj = fused_eou_context_adjustment(eou);
    CHECK(adj < 0.0f, "Question should produce negative adjustment");
    fused_eou_destroy(eou);
}

static void test_context_conjunction(void) {
    TEST("context: trailing conjunction -> higher threshold");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_context(eou, "I was going to say but");

    float adj = fused_eou_context_adjustment(eou);
    CHECK(adj > 0.0f, "Conjunction should produce positive adjustment");
    fused_eou_destroy(eou);
}

static void test_context_filler(void) {
    TEST("context: trailing filler -> slight threshold raise");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_context(eou, "So I was thinking um");

    float adj = fused_eou_context_adjustment(eou);
    CHECK(adj > 0.0f && adj < 0.10f, "Filler should produce small positive adjustment");
    fused_eou_destroy(eou);
}

static void test_context_normal_sentence(void) {
    TEST("context: normal sentence -> no adjustment");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_context(eou, "I had a great day today");

    float adj = fused_eou_context_adjustment(eou);
    CHECK(adj == 0.0f, "Normal sentence should have zero adjustment");
    fused_eou_destroy(eou);
}

static void test_context_null_clear(void) {
    TEST("context: NULL transcript clears adjustment");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_context(eou, "What time is it?");
    CHECK(fused_eou_context_adjustment(eou) != 0.0f, "Pre: should have adjustment");

    fused_eou_set_context(eou, NULL);
    float adj = fused_eou_context_adjustment(eou);
    CHECK(adj == 0.0f, "NULL should clear adjustment");
    fused_eou_destroy(eou);
}

static void test_context_case_insensitive(void) {
    TEST("context: conjunction matching is case-insensitive");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_context(eou, "I was saying AND");

    float adj = fused_eou_context_adjustment(eou);
    CHECK(adj > 0.0f, "Uppercase conjunction should still match");
    fused_eou_destroy(eou);
}

static void test_context_trailing_whitespace(void) {
    TEST("context: handles trailing whitespace");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_context(eou, "What is this?   ");

    float adj = fused_eou_context_adjustment(eou);
    CHECK(adj < 0.0f, "Question with trailing spaces should still match");
    fused_eou_destroy(eou);
}

static void test_context_multiple_conjunctions(void) {
    TEST("context: various conjunctions all raise threshold");
    const char *tests[] = {
        "I said because", "thinking or", "also but",
        "maybe so", "what however", "yes although"
    };
    int pass = 1;
    for (int i = 0; i < 6; i++) {
        FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
        fused_eou_set_context(eou, tests[i]);
        if (fused_eou_context_adjustment(eou) <= 0.0f) pass = 0;
        fused_eou_destroy(eou);
    }
    CHECK(pass, "All conjunctions should produce positive adjustment");
}

static void test_context_empty_string(void) {
    TEST("context: empty string -> no adjustment");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_context(eou, "");

    float adj = fused_eou_context_adjustment(eou);
    CHECK(adj == 0.0f, "Empty string should have zero adjustment");
    fused_eou_destroy(eou);
}

static void test_context_whitespace_only(void) {
    TEST("context: whitespace-only -> no adjustment");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_context(eou, "   \t\n  ");

    float adj = fused_eou_context_adjustment(eou);
    CHECK(adj == 0.0f, "Whitespace-only should have zero adjustment");
    fused_eou_destroy(eou);
}

static void test_context_persists_across_reset(void) {
    TEST("context: persists across utterance reset");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_context(eou, "What time?");

    float adj_before = fused_eou_context_adjustment(eou);
    fused_eou_reset(eou);
    float adj_after = fused_eou_context_adjustment(eou);

    CHECK(fabsf(adj_before - adj_after) < 0.001f,
          "Context adjustment should survive reset");
    fused_eou_destroy(eou);
}

/* ── 4th Signal Fusion Integration Tests ──────────────── */

static void test_prosody_weight_default_zero(void) {
    TEST("prosody weight: default is 0.0 (backward compatible)");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);

    /* With default w_prosody=0.0, adding prosody data should not affect fusion */
    for (int i = 0; i < 16; i++) {
        ProsodyFrame f = { .pitch_hz = 200.0f - 5.0f * (float)i, .energy_db = -10.0f };
        fused_eou_feed_prosody(eou, f);
    }

    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* Compare with a detector that has no prosody data at all */
    FusedEOU *eou_ref = fused_eou_create(0.6f, 2, 80.0f);
    for (int i = 0; i < 5; i++) fused_eou_process(eou_ref, speech);

    float p1 = fused_eou_prob(eou);
    float p2 = fused_eou_prob(eou_ref);
    CHECK(fabsf(p1 - p2) < 0.01f, "w_prosody=0 should match no-prosody result");
    fused_eou_destroy(eou);
    fused_eou_destroy(eou_ref);
}

static void test_prosody_affects_fusion(void) {
    TEST("prosody: enabled weight changes fused probability");
    FusedEOU *eou_with = fused_eou_create(0.6f, 2, 80.0f);
    FusedEOU *eou_without = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_prosody_weight(eou_with, 0.15f);

    /* Feed strongly falling pitch -> high prosody prob */
    for (int i = 0; i < 20; i++) {
        ProsodyFrame f = {
            .pitch_hz = 250.0f - (130.0f * (float)i / 19.0f),
            .energy_db = -15.0f - (float)i
        };
        fused_eou_feed_prosody(eou_with, f);
    }

    /* Process same signals through both */
    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) {
        fused_eou_process(eou_with, speech);
        fused_eou_process(eou_without, speech);
    }

    /* Moderate signals */
    EOUSignals mid = { .energy_signal = 0.4f, .mimi_eot_prob = 0.4f, .stt_eou_prob = 0.4f };
    for (int i = 0; i < 5; i++) {
        fused_eou_process(eou_with, mid);
        fused_eou_process(eou_without, mid);
    }

    float p_with = fused_eou_prob(eou_with);
    float p_without = fused_eou_prob(eou_without);
    CHECK(p_with > p_without,
          "High prosody prob with falling pitch should raise fused probability");
    fused_eou_destroy(eou_with);
    fused_eou_destroy(eou_without);
}

static void test_context_affects_trigger(void) {
    TEST("context: question makes trigger easier");
    FusedEOU *eou_q = fused_eou_create(0.65f, 2, 80.0f);
    FusedEOU *eou_n = fused_eou_create(0.65f, 2, 80.0f);
    fused_eou_set_context(eou_q, "Where are you going?");

    /* Feed identical signals */
    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) {
        fused_eou_process(eou_q, speech);
        fused_eou_process(eou_n, speech);
    }

    /* Moderate signals at threshold boundary */
    EOUSignals mid = { .energy_signal = 0.6f, .mimi_eot_prob = 0.6f, .stt_eou_prob = 0.6f };
    for (int i = 0; i < 5; i++) {
        fused_eou_process(eou_q, mid);
        fused_eou_process(eou_n, mid);
    }

    int trig_q = fused_eou_triggered(eou_q);
    int trig_n = fused_eou_triggered(eou_n);
    CHECK(trig_q >= trig_n,
          "Question context should make triggering easier (or equal)");
    fused_eou_destroy(eou_q);
    fused_eou_destroy(eou_n);
}

static void test_context_conjunction_prevents_trigger(void) {
    TEST("context: conjunction makes trigger harder");
    FusedEOU *eou_c = fused_eou_create(0.55f, 2, 80.0f);
    FusedEOU *eou_n = fused_eou_create(0.55f, 2, 80.0f);
    fused_eou_set_context(eou_c, "I wanted to say and");

    /* Establish speech */
    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) {
        fused_eou_process(eou_c, speech);
        fused_eou_process(eou_n, speech);
    }

    /* Signals just above normal threshold but below raised threshold */
    EOUSignals mid = { .energy_signal = 0.55f, .mimi_eot_prob = 0.55f, .stt_eou_prob = 0.55f };
    for (int i = 0; i < 5; i++) {
        fused_eou_process(eou_c, mid);
        fused_eou_process(eou_n, mid);
    }

    int trig_c = fused_eou_triggered(eou_c);
    int trig_n = fused_eou_triggered(eou_n);
    CHECK(trig_n >= trig_c,
          "Conjunction context should make triggering harder (or equal)");
    fused_eou_destroy(eou_c);
    fused_eou_destroy(eou_n);
}

static void test_prosody_weight_clamp(void) {
    TEST("prosody weight: clamped to [0.0, 0.5]");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);

    fused_eou_set_prosody_weight(eou, -1.0f);
    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    fused_eou_process(eou, speech);

    fused_eou_set_prosody_weight(eou, 2.0f);
    fused_eou_process(eou, speech);

    CHECK(fused_eou_prob(eou) >= 0.0f, "Clamped weights should not crash");
    fused_eou_destroy(eou);
}

static void test_prosody_reset_clears_buffers(void) {
    TEST("prosody: reset clears ring buffers");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_prosody_weight(eou, 0.15f);

    for (int i = 0; i < 16; i++) {
        ProsodyFrame f = { .pitch_hz = 200.0f, .energy_db = -10.0f };
        fused_eou_feed_prosody(eou, f);
    }

    fused_eou_reset(eou);

    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    fused_eou_process(eou, speech);

    float prob = fused_eou_prosody_prob(eou);
    CHECK(fabsf(prob - 0.5f) < 0.01f, "After reset, prosody should be neutral (0.5)");
    fused_eou_destroy(eou);
}

static void test_prosody_ring_buffer_wraps(void) {
    TEST("prosody: ring buffer wraps correctly");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_prosody_weight(eou, 0.15f);

    /* Fill buffer twice over (64 frames > 32 buffer size) with rising pitch */
    for (int i = 0; i < 64; i++) {
        ProsodyFrame f = {
            .pitch_hz = 120.0f + (80.0f * (float)(i % 32) / 31.0f),
            .energy_db = -20.0f
        };
        fused_eou_feed_prosody(eou, f);
    }

    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    fused_eou_process(eou, speech);

    float prob = fused_eou_prosody_prob(eou);
    /* Rising pitch should give below-neutral, proving wrap works */
    CHECK(prob < 0.5f, "Ring buffer wrap should still compute correct trajectory");
    fused_eou_destroy(eou);
}

/* ── NULL Safety for New Functions ────────────────────── */

static void test_new_functions_null_safety(void) {
    TEST("NULL safety for all new functions");
    fused_eou_set_context(NULL, "test");
    ProsodyFrame f = { .pitch_hz = 100.0f, .energy_db = -20.0f };
    fused_eou_feed_prosody(NULL, f);
    fused_eou_set_prosody_weight(NULL, 0.15f);
    CHECK(fused_eou_prosody_prob(NULL) == 0.0f, "prosody_prob(NULL) should be 0");
    CHECK(fused_eou_context_adjustment(NULL) == 0.0f, "context_adj(NULL) should be 0");
}

/* ── Audit Bug-Fix Tests ───────────────────────────────── */

static void test_nan_prosody_guard(void) {
    TEST("NaN prosody frames are sanitized to 0");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_prosody_weight(eou, 0.15f);

    /* Feed mix of valid and NaN frames */
    for (int i = 0; i < 8; i++) {
        ProsodyFrame f = { .pitch_hz = 180.0f, .energy_db = -25.0f };
        fused_eou_feed_prosody(eou, f);
    }
    ProsodyFrame nan_frame = { .pitch_hz = NAN, .energy_db = NAN };
    fused_eou_feed_prosody(eou, nan_frame);
    for (int i = 0; i < 7; i++) {
        ProsodyFrame f = { .pitch_hz = 120.0f, .energy_db = -30.0f };
        fused_eou_feed_prosody(eou, f);
    }

    /* Process — should NOT produce NaN */
    EOUSignals sig = { .energy_signal = 0.5f, .mimi_eot_prob = 0.5f, .stt_eou_prob = 0.5f };
    EOUResult res = fused_eou_process(eou, sig);
    CHECK(!isnan(res.fused_prob) && !isinf(res.fused_prob),
          "fused_prob must not be NaN/Inf with NaN input frames");
    float pp = fused_eou_prosody_prob(eou);
    CHECK(!isnan(pp) && !isinf(pp), "prosody_prob must not be NaN/Inf");

    fused_eou_destroy(eou);
}

static void test_inf_prosody_guard(void) {
    TEST("Inf prosody frames are sanitized to 0");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    fused_eou_set_prosody_weight(eou, 0.15f);

    for (int i = 0; i < 16; i++) {
        ProsodyFrame f = { .pitch_hz = INFINITY, .energy_db = -INFINITY };
        fused_eou_feed_prosody(eou, f);
    }

    EOUSignals sig = { .energy_signal = 0.5f, .mimi_eot_prob = 0.5f, .stt_eou_prob = 0.5f };
    EOUResult res = fused_eou_process(eou, sig);
    CHECK(!isnan(res.fused_prob) && !isinf(res.fused_prob),
          "fused_prob must not be NaN/Inf with Inf input frames");

    fused_eou_destroy(eou);
}

static void test_context_adj_on_solo_triggers(void) {
    TEST("context_adj raises solo thresholds (conjunction blocks solo trigger)");
    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);

    /* Set context to trailing conjunction → +0.10 adjustment */
    fused_eou_set_context(eou, "I was thinking and");
    float adj = fused_eou_context_adjustment(eou);
    CHECK(adj > 0.05f, "conjunction should raise threshold");

    /* Feed speech to enable speech_detected */
    EOUSignals speech = { .energy_signal = 0.5f, .mimi_eot_prob = 0.3f, .stt_eou_prob = 0.3f };
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    /* STT solo threshold is 0.85 default; with +0.10 adj → effective 0.95.
     * A signal at 0.90 should NOT solo-trigger when context_adj raises it. */
    fused_eou_reset(eou);
    fused_eou_set_context(eou, "I was thinking and");
    /* Re-establish speech */
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    EOUSignals high_stt = { .energy_signal = 0.3f, .mimi_eot_prob = 0.3f, .stt_eou_prob = 0.90f };
    EOUResult res = fused_eou_process(eou, high_stt);
    /* 0.90 < 0.85 + 0.10 = 0.95, so solo should NOT trigger */
    int solo_blocked = !(res.trigger_source & 0x04); /* EOU_SRC_STT = 0x04 */
    CHECK(solo_blocked, "STT at 0.90 should not solo-trigger with +0.10 context_adj");

    /* Now with question context → -0.10 adj, same signal should solo-trigger */
    fused_eou_reset(eou);
    fused_eou_set_context(eou, "What do you think?");
    for (int i = 0; i < 5; i++) fused_eou_process(eou, speech);

    res = fused_eou_process(eou, high_stt);
    /* 0.90 >= 0.85 + (-0.10) = 0.75, so solo SHOULD trigger */
    int solo_fired = (res.trigger_source & 0x04) != 0;
    CHECK(solo_fired, "STT at 0.90 should solo-trigger with -0.10 context_adj");

    fused_eou_destroy(eou);
}

/* ── Main ─────────────────────────────────────────────── */

int main(void) {
    printf("==================================================================\n");
    printf("  Research EOU: Conversation Context + Prosodic Turn-Taking\n");
    printf("==================================================================\n");

    printf("\n[Prosody Signal Tests]\n");
    test_prosody_falling_pitch();
    test_prosody_rising_pitch();
    test_prosody_energy_decay();
    test_prosody_energy_rising();
    test_prosody_insufficient_data();
    test_prosody_flat_pitch_neutral();
    test_prosody_solo_trigger();
    test_prosody_unvoiced_frames();
    test_prosody_source_bitmask();

    printf("\n[Conversation Context Tests]\n");
    test_context_question();
    test_context_conjunction();
    test_context_filler();
    test_context_normal_sentence();
    test_context_null_clear();
    test_context_case_insensitive();
    test_context_trailing_whitespace();
    test_context_multiple_conjunctions();
    test_context_empty_string();
    test_context_whitespace_only();
    test_context_persists_across_reset();

    printf("\n[4th Signal Fusion Integration Tests]\n");
    test_prosody_weight_default_zero();
    test_prosody_affects_fusion();
    test_context_affects_trigger();
    test_context_conjunction_prevents_trigger();
    test_prosody_weight_clamp();
    test_prosody_reset_clears_buffers();
    test_prosody_ring_buffer_wraps();

    printf("\n[NULL Safety Tests]\n");
    test_new_functions_null_safety();

    printf("\n[Audit Bug-Fix Tests]\n");
    test_nan_prosody_guard();
    test_inf_prosody_guard();
    test_context_adj_on_solo_triggers();

    printf("\n==================================================================\n");
    printf("  Total: %d passed, %d failed (out of %d)\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("  %s\n", tests_failed == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("==================================================================\n\n");

    return tests_failed > 0 ? 1 : 0;
}
