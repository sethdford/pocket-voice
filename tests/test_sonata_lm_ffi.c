/*
 * test_sonata_lm_ffi.c — FFI boundary tests for the Sonata Language Model.
 *
 * Tests null-safety, constants, conversion helpers, and invalid-input
 * handling for all sonata_lm_* exported functions without requiring
 * model weights.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Sonata LM FFI ───────────────────────────────────────────────────── */

extern void *sonata_lm_create(const char *weights_path, const char *config_path);
extern void  sonata_lm_destroy(void *engine);
extern int   sonata_lm_set_text(void *engine, const unsigned int *text_ids, int n);
extern int   sonata_lm_append_text(void *engine, const unsigned int *text_ids, int n);
extern int   sonata_lm_step(void *engine, int *out_token);
extern int   sonata_lm_reset(void *engine);
extern int   sonata_lm_is_done(void *engine);
extern int   sonata_lm_sample_rate(void);
extern int   sonata_lm_frame_rate(void);
extern int   sonata_lm_samples_per_frame(void);
extern int   sonata_lm_set_params(void *engine, float temperature, int top_k,
                                   float top_p, float rep_penalty);
extern int   sonata_lm_load_draft(void *engine, const char *weights, const char *config);
extern int   sonata_lm_speculate_step(void *engine, int *out_tokens, int max_tokens, int *out_count);
extern int   sonata_lm_set_speculate_k(void *engine, int k);
extern int   sonata_lm_ms_to_frames(int ms);
extern int   sonata_lm_num_prosody_tokens(void);
extern int   sonata_lm_prosody_token_base(void *engine);
extern int   sonata_lm_inject_prosody_token(void *engine, int prosody_offset);
extern int   sonata_lm_inject_pause(void *engine, int n_frames);

/* ─── Test helpers ─────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { g_fail++; printf("  [FAIL] %s\n", msg); } \
} while(0)

#define CHECKF(cond, fmt, ...) do { \
    char _buf[256]; snprintf(_buf, sizeof(_buf), fmt, __VA_ARGS__); \
    if (cond) { g_pass++; printf("  [PASS] %s\n", _buf); } \
    else { g_fail++; printf("  [FAIL] %s\n", _buf); } \
} while(0)

/* ─── Test 1: LM constants ───────────────────────────────────────────── */

static void test_lm_constants(void) {
    printf("\n═══ Test 1: LM constants ═══\n");

    CHECKF(sonata_lm_sample_rate() == 24000,
           "sample_rate() = %d (expected 24000)", sonata_lm_sample_rate());
    CHECKF(sonata_lm_frame_rate() == 50,
           "frame_rate() = %d (expected 50)", sonata_lm_frame_rate());
    CHECKF(sonata_lm_samples_per_frame() == 480,
           "samples_per_frame() = %d (expected 480)", sonata_lm_samples_per_frame());
    CHECKF(sonata_lm_num_prosody_tokens() == 12,
           "num_prosody_tokens() = %d (expected 12)", sonata_lm_num_prosody_tokens());
}

/* ─── Test 2: ms_to_frames conversion ────────────────────────────────── */

static void test_lm_ms_to_frames(void) {
    printf("\n═══ Test 2: ms_to_frames conversion ═══\n");

    CHECKF(sonata_lm_ms_to_frames(0) == 0,
           "ms_to_frames(0) = %d (expected 0)", sonata_lm_ms_to_frames(0));
    CHECKF(sonata_lm_ms_to_frames(20) == 1,
           "ms_to_frames(20) = %d (expected 1)", sonata_lm_ms_to_frames(20));
    CHECKF(sonata_lm_ms_to_frames(100) == 5,
           "ms_to_frames(100) = %d (expected 5)", sonata_lm_ms_to_frames(100));
    CHECKF(sonata_lm_ms_to_frames(1000) == 50,
           "ms_to_frames(1000) = %d (expected 50)", sonata_lm_ms_to_frames(1000));
}

/* ─── Test 3: Comprehensive null safety ──────────────────────────────── */

static void test_lm_null_safety(void) {
    printf("\n═══ Test 3: LM null safety (all functions) ═══\n");

    unsigned int text_ids[] = {10, 20, 30};
    int out_token = -1;
    int out_tokens[16];
    int out_count = 0;

    CHECK(sonata_lm_set_text(NULL, text_ids, 3) == -1,
          "set_text(NULL) returns -1");
    CHECK(sonata_lm_append_text(NULL, text_ids, 3) == -1,
          "append_text(NULL) returns -1");
    CHECK(sonata_lm_step(NULL, &out_token) == -1,
          "step(NULL) returns -1");
    CHECK(sonata_lm_reset(NULL) == -1,
          "reset(NULL) returns -1");
    CHECK(sonata_lm_is_done(NULL) == 1,
          "is_done(NULL) returns 1 (treats NULL as done)");
    CHECK(sonata_lm_set_params(NULL, 0.8f, 50, 0.92f, 1.15f) == -1,
          "set_params(NULL) returns -1");
    CHECK(sonata_lm_load_draft(NULL, NULL, NULL) == -1,
          "load_draft(NULL) returns -1");
    CHECK(sonata_lm_speculate_step(NULL, out_tokens, 16, &out_count) == -1,
          "speculate_step(NULL) returns -1");
    CHECK(sonata_lm_set_speculate_k(NULL, 5) == -1,
          "set_speculate_k(NULL) returns -1");
    CHECK(sonata_lm_prosody_token_base(NULL) == -1,
          "prosody_token_base(NULL) returns -1");
    CHECK(sonata_lm_inject_prosody_token(NULL, 0) == -1,
          "inject_prosody_token(NULL) returns -1");
    CHECK(sonata_lm_inject_pause(NULL, 5) == -1,
          "inject_pause(NULL) returns -1");
}

/* ─── Test 4: Create with invalid paths ──────────────────────────────── */

static void test_lm_create_invalid(void) {
    printf("\n═══ Test 4: LM create with invalid paths ═══\n");

    void *engine = sonata_lm_create(
        "/nonexistent/lm.safetensors",
        "/nonexistent/lm_config.json"
    );
    CHECK(engine == NULL, "create with nonexistent paths returns NULL");
    if (engine) sonata_lm_destroy(engine);
}

/* ─── Test 5: set_params NULL ────────────────────────────────────────── */

static void test_lm_set_params_null(void) {
    printf("\n═══ Test 5: set_params(NULL) ═══\n");

    CHECK(sonata_lm_set_params(NULL, 0.8f, 50, 0.92f, 1.15f) == -1,
          "set_params(NULL, 0.8, 50, 0.92, 1.15) returns -1");
    CHECK(sonata_lm_set_params(NULL, 0.0f, 0, 0.0f, 0.0f) == -1,
          "set_params(NULL, 0, 0, 0, 0) returns -1");
}

/* ─── Test 6: set_text NULL ──────────────────────────────────────────── */

static void test_lm_set_text_null(void) {
    printf("\n═══ Test 6: set_text(NULL) ═══\n");

    unsigned int ids[] = {1, 2, 3};
    CHECK(sonata_lm_set_text(NULL, ids, 3) == -1,
          "set_text(NULL, ids, 3) returns -1");
    CHECK(sonata_lm_set_text(NULL, NULL, 0) == -1,
          "set_text(NULL, NULL, 0) returns -1");
}

/* ─── Test 7: append_text NULL ───────────────────────────────────────── */

static void test_lm_append_text_null(void) {
    printf("\n═══ Test 7: append_text(NULL) ═══\n");

    unsigned int ids[] = {10, 20};
    CHECK(sonata_lm_append_text(NULL, ids, 2) == -1,
          "append_text(NULL, ids, 2) returns -1");
    CHECK(sonata_lm_append_text(NULL, NULL, 0) == -1,
          "append_text(NULL, NULL, 0) returns -1");
}

/* ─── Test 8: step NULL ──────────────────────────────────────────────── */

static void test_lm_step_null(void) {
    printf("\n═══ Test 8: step(NULL) ═══\n");

    int token = -1;
    CHECK(sonata_lm_step(NULL, &token) == -1,
          "step(NULL, &token) returns -1");
    CHECK(sonata_lm_step(NULL, NULL) == -1,
          "step(NULL, NULL) returns -1");
}

/* ─── Test 9: reset NULL ─────────────────────────────────────────────── */

static void test_lm_reset_null(void) {
    printf("\n═══ Test 9: reset(NULL) ═══\n");

    CHECK(sonata_lm_reset(NULL) == -1,
          "reset(NULL) returns -1");
}

/* ─── Test 10: is_done NULL ──────────────────────────────────────────── */

static void test_lm_is_done_null(void) {
    printf("\n═══ Test 10: is_done(NULL) ═══\n");

    int done = sonata_lm_is_done(NULL);
    CHECKF(done == 1, "is_done(NULL) = %d (expected 1, treats NULL as done)", done);
}

/* ─── Test 11: speculate_step NULL ───────────────────────────────────── */

static void test_lm_speculate_step_null(void) {
    printf("\n═══ Test 11: speculate_step(NULL) ═══\n");

    int tokens[16];
    int count = 0;
    CHECK(sonata_lm_speculate_step(NULL, tokens, 16, &count) == -1,
          "speculate_step(NULL, tokens, 16, &count) returns -1");
    CHECK(sonata_lm_speculate_step(NULL, NULL, 0, NULL) == -1,
          "speculate_step(NULL, NULL, 0, NULL) returns -1");
}

/* ─── Test 12: inject_prosody_token NULL ─────────────────────────────── */

static void test_lm_inject_prosody_null(void) {
    printf("\n═══ Test 12: inject_prosody_token(NULL) ═══\n");

    CHECK(sonata_lm_inject_prosody_token(NULL, 0) == -1,
          "inject_prosody_token(NULL, 0) returns -1");
    CHECK(sonata_lm_inject_prosody_token(NULL, 11) == -1,
          "inject_prosody_token(NULL, 11) returns -1");
}

/* ─── Test 13: inject_pause NULL ─────────────────────────────────────── */

static void test_lm_inject_pause_null(void) {
    printf("\n═══ Test 13: inject_pause(NULL) ═══\n");

    CHECK(sonata_lm_inject_pause(NULL, 5) == -1,
          "inject_pause(NULL, 5) returns -1");
    CHECK(sonata_lm_inject_pause(NULL, 0) == -1,
          "inject_pause(NULL, 0) returns -1");
}

/* ─── Main ─────────────────────────────────────────────────────────────── */

int main(void) {
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║  Sonata LM FFI — Boundary Test Suite          ║\n");
    printf("╚════════════════════════════════════════════════╝\n");

    test_lm_constants();
    test_lm_ms_to_frames();
    test_lm_null_safety();
    test_lm_create_invalid();
    test_lm_set_params_null();
    test_lm_set_text_null();
    test_lm_append_text_null();
    test_lm_step_null();
    test_lm_reset_null();
    test_lm_is_done_null();
    test_lm_speculate_step_null();
    test_lm_inject_prosody_null();
    test_lm_inject_pause_null();

    printf("\n══════════════════════════════════════════\n");
    printf("Results: %d / %d passed\n", g_pass, g_pass + g_fail);
    if (g_fail > 0) {
        printf("FAILURES: %d\n", g_fail);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
