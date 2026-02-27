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

/* ─── Test 3: Constants consistency ───────────────────────────────────── */

static void test_lm_constants_consistency(void) {
    printf("\n═══ Test 3: LM constants consistency ═══\n");

    int sr = sonata_lm_sample_rate();
    int fr = sonata_lm_frame_rate();
    int spf = sonata_lm_samples_per_frame();

    /* samples_per_frame should equal sample_rate / frame_rate */
    CHECKF(spf == sr / fr,
           "samples_per_frame(%d) == sample_rate(%d) / frame_rate(%d)",
           spf, sr, fr);

    /* Constants should be stable */
    CHECK(sonata_lm_sample_rate() == sr, "sample_rate stable across calls");
    CHECK(sonata_lm_frame_rate() == fr, "frame_rate stable across calls");
    CHECK(sonata_lm_samples_per_frame() == spf, "samples_per_frame stable");

    /* num_prosody_tokens should be positive and reasonable */
    int npt = sonata_lm_num_prosody_tokens();
    CHECKF(npt > 0 && npt < 100,
           "num_prosody_tokens() = %d (positive and reasonable)", npt);
}

/* ─── Test 4: ms_to_frames extended ──────────────────────────────────── */

static void test_lm_ms_to_frames_extended(void) {
    printf("\n═══ Test 4: ms_to_frames extended ═══\n");

    /* Negative values */
    int neg = sonata_lm_ms_to_frames(-1);
    CHECKF(neg <= 0, "ms_to_frames(-1) = %d (<=0)", neg);

    int neg2 = sonata_lm_ms_to_frames(-100);
    CHECKF(neg2 <= 0, "ms_to_frames(-100) = %d (<=0)", neg2);

    /* Boundary values: frame_rate=50, so 1 frame = 20ms */
    CHECKF(sonata_lm_ms_to_frames(10) == 0 || sonata_lm_ms_to_frames(10) == 1,
           "ms_to_frames(10) = %d (0 or 1)", sonata_lm_ms_to_frames(10));
    CHECKF(sonata_lm_ms_to_frames(19) == 0 || sonata_lm_ms_to_frames(19) == 1,
           "ms_to_frames(19) = %d (0 or 1)", sonata_lm_ms_to_frames(19));

    /* Large values */
    CHECKF(sonata_lm_ms_to_frames(10000) == 500,
           "ms_to_frames(10000) = %d (expected 500)", sonata_lm_ms_to_frames(10000));
    CHECKF(sonata_lm_ms_to_frames(60000) == 3000,
           "ms_to_frames(60000) = %d (expected 3000)", sonata_lm_ms_to_frames(60000));

    /* Accuracy check: frames * 20ms should round-trip */
    for (int ms = 0; ms <= 200; ms += 20) {
        int frames = sonata_lm_ms_to_frames(ms);
        int expected = ms / 20;
        CHECKF(frames == expected,
               "ms_to_frames(%d) = %d (expected %d)", ms, frames, expected);
    }
}

/* ─── Test 5: Create with NULL paths ─────────────────────────────────── */

static void test_lm_create_null_paths(void) {
    printf("\n═══ Test 5: LM create with NULL paths ═══\n");

    void *engine = sonata_lm_create(NULL, NULL);
    CHECK(engine == NULL, "create(NULL, NULL) returns NULL");
    if (engine) sonata_lm_destroy(engine);

    engine = sonata_lm_create("/some/path", NULL);
    CHECK(engine == NULL, "create(path, NULL) returns NULL");
    if (engine) sonata_lm_destroy(engine);

    engine = sonata_lm_create(NULL, "/some/config");
    CHECK(engine == NULL, "create(NULL, config) returns NULL");
    if (engine) sonata_lm_destroy(engine);
}

/* ─── Test 6: Create with empty strings ──────────────────────────────── */

static void test_lm_create_empty_strings(void) {
    printf("\n═══ Test 6: LM create with empty strings ═══\n");

    void *engine = sonata_lm_create("", "");
    CHECK(engine == NULL, "create('', '') returns NULL");
    if (engine) sonata_lm_destroy(engine);
}

/* ─── Test 7: Memory lifecycle — repeated create/destroy ─────────────── */

static void test_lm_memory_lifecycle(void) {
    printf("\n═══ Test 7: Memory lifecycle ═══\n");

    for (int i = 0; i < 100; i++) {
        void *engine = sonata_lm_create(
            "/nonexistent/lm.safetensors",
            "/nonexistent/lm_config.json"
        );
        if (engine) sonata_lm_destroy(engine);
    }
    CHECK(1, "100x create(bad)/destroy cycles no crash");

    for (int i = 0; i < 100; i++) {
        sonata_lm_destroy(NULL);
    }
    CHECK(1, "100x destroy(NULL) cycles no crash");
}

/* ─── Test 8: set_text edge cases ────────────────────────────────────── */

static void test_lm_set_text_edge_cases(void) {
    printf("\n═══ Test 8: set_text edge cases ═══\n");

    /* Empty array (n=0) */
    unsigned int ids[1] = {42};
    CHECK(sonata_lm_set_text(NULL, ids, 0) == -1,
          "set_text(NULL, data, 0) returns -1");

    /* Single token */
    CHECK(sonata_lm_set_text(NULL, ids, 1) == -1,
          "set_text(NULL, data, 1) returns -1");

    /* Negative count */
    CHECK(sonata_lm_set_text(NULL, ids, -1) == -1,
          "set_text(NULL, data, -1) returns -1");

    /* NULL data with n>0 */
    CHECK(sonata_lm_set_text(NULL, NULL, 5) == -1,
          "set_text(NULL, NULL, 5) returns -1");

    /* Large token count */
    CHECK(sonata_lm_set_text(NULL, ids, 10000) == -1,
          "set_text(NULL, data, 10000) returns -1");
}

/* ─── Test 9: append_text edge cases ─────────────────────────────────── */

static void test_lm_append_text_edge_cases(void) {
    printf("\n═══ Test 9: append_text edge cases ═══\n");

    unsigned int ids[] = {10, 20, 30};

    CHECK(sonata_lm_append_text(NULL, ids, 0) == -1,
          "append_text(NULL, data, 0) returns -1");
    CHECK(sonata_lm_append_text(NULL, ids, -1) == -1,
          "append_text(NULL, data, -1) returns -1");
    CHECK(sonata_lm_append_text(NULL, NULL, 5) == -1,
          "append_text(NULL, NULL, 5) returns -1");
}

/* ─── Test 10: set_params parameter validation ───────────────────────── */

static void test_lm_set_params_validation(void) {
    printf("\n═══ Test 10: set_params parameter validation ═══\n");

    /* Typical valid ranges (all should return -1 because engine is NULL) */
    CHECK(sonata_lm_set_params(NULL, 0.8f, 50, 0.92f, 1.15f) == -1,
          "set_params(NULL, typical) returns -1");

    /* Edge temperatures */
    CHECK(sonata_lm_set_params(NULL, 0.0f, 50, 0.92f, 1.0f) == -1,
          "set_params(NULL, temp=0.0) returns -1");
    CHECK(sonata_lm_set_params(NULL, 2.0f, 50, 0.92f, 1.0f) == -1,
          "set_params(NULL, temp=2.0) returns -1");
    CHECK(sonata_lm_set_params(NULL, -1.0f, 50, 0.92f, 1.0f) == -1,
          "set_params(NULL, temp=-1.0) returns -1");

    /* Edge top_k */
    CHECK(sonata_lm_set_params(NULL, 0.8f, 0, 0.92f, 1.0f) == -1,
          "set_params(NULL, top_k=0) returns -1");
    CHECK(sonata_lm_set_params(NULL, 0.8f, 1, 0.92f, 1.0f) == -1,
          "set_params(NULL, top_k=1) returns -1");
    CHECK(sonata_lm_set_params(NULL, 0.8f, -1, 0.92f, 1.0f) == -1,
          "set_params(NULL, top_k=-1) returns -1");

    /* Edge top_p */
    CHECK(sonata_lm_set_params(NULL, 0.8f, 50, 0.0f, 1.0f) == -1,
          "set_params(NULL, top_p=0.0) returns -1");
    CHECK(sonata_lm_set_params(NULL, 0.8f, 50, 1.0f, 1.0f) == -1,
          "set_params(NULL, top_p=1.0) returns -1");

    /* Edge rep_penalty */
    CHECK(sonata_lm_set_params(NULL, 0.8f, 50, 0.92f, 0.0f) == -1,
          "set_params(NULL, rep_penalty=0.0) returns -1");
    CHECK(sonata_lm_set_params(NULL, 0.8f, 50, 0.92f, 5.0f) == -1,
          "set_params(NULL, rep_penalty=5.0) returns -1");
}

/* ─── Test 11: Error recovery after failed create ────────────────────── */

static void test_lm_error_recovery(void) {
    printf("\n═══ Test 11: Error recovery after failed create ═══\n");

    void *engine = sonata_lm_create(
        "/nonexistent/lm.bin", "/nonexistent/lm.json"
    );
    CHECK(engine == NULL, "create with bad paths returns NULL");

    /* All ops on NULL engine should be safe */
    unsigned int ids[] = {1, 2, 3};
    int token = -1;
    CHECK(sonata_lm_set_text(engine, ids, 3) == -1,
          "set_text on failed engine returns -1");
    CHECK(sonata_lm_append_text(engine, ids, 3) == -1,
          "append_text on failed engine returns -1");
    CHECK(sonata_lm_step(engine, &token) == -1,
          "step on failed engine returns -1");
    CHECK(sonata_lm_reset(engine) == -1,
          "reset on failed engine returns -1");
    CHECK(sonata_lm_is_done(engine) == 1,
          "is_done on failed engine returns 1");
    CHECK(sonata_lm_set_params(engine, 0.8f, 50, 0.92f, 1.0f) == -1,
          "set_params on failed engine returns -1");
    CHECK(sonata_lm_inject_prosody_token(engine, 0) == -1,
          "inject_prosody_token on failed engine returns -1");
    CHECK(sonata_lm_inject_pause(engine, 5) == -1,
          "inject_pause on failed engine returns -1");

    sonata_lm_destroy(engine);
    CHECK(1, "destroy(NULL) after failed create is safe");
}

/* ─── Test 12: inject_pause edge cases ───────────────────────────────── */

static void test_lm_inject_pause_edge(void) {
    printf("\n═══ Test 12: inject_pause edge cases ═══\n");

    CHECK(sonata_lm_inject_pause(NULL, 0) == -1,
          "inject_pause(NULL, 0) returns -1");
    CHECK(sonata_lm_inject_pause(NULL, -1) == -1,
          "inject_pause(NULL, -1) returns -1");
    CHECK(sonata_lm_inject_pause(NULL, 1000) == -1,
          "inject_pause(NULL, 1000) returns -1");
}

/* ─── Test 13: inject_prosody_token edge cases ───────────────────────── */

static void test_lm_inject_prosody_edge(void) {
    printf("\n═══ Test 13: inject_prosody_token edge cases ═══\n");

    int n_tokens = sonata_lm_num_prosody_tokens();

    CHECK(sonata_lm_inject_prosody_token(NULL, 0) == -1,
          "inject_prosody_token(NULL, 0) returns -1");
    CHECK(sonata_lm_inject_prosody_token(NULL, n_tokens - 1) == -1,
          "inject_prosody_token(NULL, max valid offset) returns -1");
    CHECK(sonata_lm_inject_prosody_token(NULL, -1) == -1,
          "inject_prosody_token(NULL, -1) returns -1");
    CHECK(sonata_lm_inject_prosody_token(NULL, n_tokens) == -1,
          "inject_prosody_token(NULL, out-of-range) returns -1");
}

/* ─── Test 14: speculate_step edge cases ─────────────────────────────── */

static void test_lm_speculate_step_edge(void) {
    printf("\n═══ Test 14: speculate_step edge cases ═══\n");

    int tokens[32];
    int count = 0;

    CHECK(sonata_lm_speculate_step(NULL, tokens, 0, &count) == -1,
          "speculate_step(NULL, tokens, 0, &count) returns -1");
    CHECK(sonata_lm_speculate_step(NULL, tokens, -1, &count) == -1,
          "speculate_step(NULL, tokens, -1, &count) returns -1");
    CHECK(sonata_lm_speculate_step(NULL, NULL, 16, &count) == -1,
          "speculate_step(NULL, NULL, 16, &count) returns -1");
    CHECK(sonata_lm_speculate_step(NULL, tokens, 16, NULL) == -1,
          "speculate_step(NULL, tokens, 16, NULL) returns -1");
}

/* ─── Test 15: set_speculate_k edge cases ────────────────────────────── */

static void test_lm_set_speculate_k_edge(void) {
    printf("\n═══ Test 15: set_speculate_k edge cases ═══\n");

    CHECK(sonata_lm_set_speculate_k(NULL, 0) == -1,
          "set_speculate_k(NULL, 0) returns -1");
    CHECK(sonata_lm_set_speculate_k(NULL, -1) == -1,
          "set_speculate_k(NULL, -1) returns -1");
    CHECK(sonata_lm_set_speculate_k(NULL, 100) == -1,
          "set_speculate_k(NULL, 100) returns -1");
}

/* ─── Test 16: Comprehensive null safety ─────────────────────────────── */

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

/* ─── Test 17: Create with invalid paths ─────────────────────────────── */

static void test_lm_create_invalid(void) {
    printf("\n═══ Test 17: LM create with invalid paths ═══\n");

    void *engine = sonata_lm_create(
        "/nonexistent/lm.safetensors",
        "/nonexistent/lm_config.json"
    );
    CHECK(engine == NULL, "create with nonexistent paths returns NULL");
    if (engine) sonata_lm_destroy(engine);
}

/* ─── Test 18: load_draft NULL safety ────────────────────────────────── */

static void test_lm_load_draft_null(void) {
    printf("\n═══ Test 18: load_draft NULL safety ═══\n");

    CHECK(sonata_lm_load_draft(NULL, NULL, NULL) == -1,
          "load_draft(NULL, NULL, NULL) returns -1");
    CHECK(sonata_lm_load_draft(NULL, "/some/path", "/some/config") == -1,
          "load_draft(NULL, path, config) returns -1");
}

/* ─── Test 19: prosody_token_base NULL ───────────────────────────────── */

static void test_lm_prosody_token_base_null(void) {
    printf("\n═══ Test 19: prosody_token_base(NULL) ═══\n");

    int base = sonata_lm_prosody_token_base(NULL);
    CHECKF(base == -1, "prosody_token_base(NULL) = %d (expected -1)", base);
}

/* ─── Main ─────────────────────────────────────────────────────────────── */

int main(void) {
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║  Sonata LM FFI — Boundary Test Suite          ║\n");
    printf("╚════════════════════════════════════════════════╝\n");

    test_lm_constants();
    test_lm_ms_to_frames();
    test_lm_constants_consistency();
    test_lm_ms_to_frames_extended();
    test_lm_create_null_paths();
    test_lm_create_empty_strings();
    test_lm_memory_lifecycle();
    test_lm_set_text_edge_cases();
    test_lm_append_text_edge_cases();
    test_lm_set_params_validation();
    test_lm_error_recovery();
    test_lm_inject_pause_edge();
    test_lm_inject_prosody_edge();
    test_lm_speculate_step_edge();
    test_lm_set_speculate_k_edge();
    test_lm_null_safety();
    test_lm_create_invalid();
    test_lm_load_draft_null();
    test_lm_prosody_token_base_null();

    printf("\n══════════════════════════════════════════\n");
    printf("Results: %d / %d passed\n", g_pass, g_pass + g_fail);
    if (g_fail > 0) {
        printf("FAILURES: %d\n", g_fail);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
