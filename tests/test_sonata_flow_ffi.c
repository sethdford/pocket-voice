/*
 * test_sonata_flow_ffi.c — FFI boundary tests for the Sonata Flow network.
 *
 * Tests null-safety, constants, and invalid-input handling for all
 * sonata_flow_* exported functions without requiring model weights.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Sonata Flow FFI ──────────────────────────────────────────────────── */

extern void *sonata_flow_create(const char *flow_weights, const char *flow_config,
                                 const char *decoder_weights, const char *decoder_config);
extern void  sonata_flow_destroy(void *engine);
extern int   sonata_flow_generate(void *engine, const int *semantic_tokens,
                                   int n_frames, float *out_magnitude, float *out_phase);
extern int   sonata_flow_set_speaker(void *engine, int speaker_id);
extern int   sonata_flow_set_cfg_scale(void *engine, float scale);
extern int   sonata_flow_set_n_steps(void *engine, int n_steps);
extern void  sonata_flow_reset_phase(void *engine);
extern int   sonata_flow_set_solver(void *engine, int use_heun);
extern int   sonata_flow_set_speaker_embedding(void *engine, const float *embedding, int dim);
extern void  sonata_flow_clear_speaker_embedding(void *engine);
extern int   sonata_flow_n_steps(void);
extern int   sonata_flow_acoustic_dim(void);
extern int   sonata_flow_set_causal(void *engine, int enable);
extern void  sonata_flow_reset_streaming(void *engine);
extern int   sonata_flow_decoder_type(void *engine);
extern int   sonata_flow_set_emotion(void *engine, int emotion_id);
extern int   sonata_flow_set_emotion_steering(void *engine, const float *direction, int dim, float scale);
extern void  sonata_flow_clear_emotion_steering(void *engine);
extern int   sonata_flow_set_prosody(void *engine, const float *features, int n);
extern int   sonata_flow_set_durations(void *engine, const float *durations, int n_frames);
extern int   sonata_flow_set_prosody_embedding(void *engine, const float *embedding, int dim);
extern void  sonata_flow_clear_prosody_embedding(void *engine);
extern int   sonata_flow_samples_per_frame(void *engine);

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

/* ─── Test 1: Flow constants ──────────────────────────────────────────── */

static void test_flow_constants(void) {
    printf("\n═══ Test 1: Flow constants ═══\n");

    CHECKF(sonata_flow_n_steps() == 8,
           "n_steps() = %d (expected 8)", sonata_flow_n_steps());
    CHECKF(sonata_flow_acoustic_dim() == 256,
           "acoustic_dim() = %d (expected 256)", sonata_flow_acoustic_dim());
}

/* ─── Test 2: Comprehensive null safety ──────────────────────────────── */

static void test_flow_null_safety(void) {
    printf("\n═══ Test 2: Flow null safety (all setters) ═══\n");

    float dummy[256];
    memset(dummy, 0, sizeof(dummy));

    /* Functions returning int — all should return -1 with NULL engine */
    CHECK(sonata_flow_set_speaker(NULL, 0) == -1,
          "set_speaker(NULL) returns -1");
    CHECK(sonata_flow_set_cfg_scale(NULL, 2.0f) == -1,
          "set_cfg_scale(NULL) returns -1");
    CHECK(sonata_flow_set_n_steps(NULL, 4) == -1,
          "set_n_steps(NULL) returns -1");
    CHECK(sonata_flow_set_solver(NULL, 1) == -1,
          "set_solver(NULL) returns -1");
    CHECK(sonata_flow_set_speaker_embedding(NULL, dummy, 256) == -1,
          "set_speaker_embedding(NULL engine) returns -1");
    CHECK(sonata_flow_set_emotion(NULL, 0) == -1,
          "set_emotion(NULL) returns -1");
    CHECK(sonata_flow_set_emotion_steering(NULL, dummy, 8, 1.0f) == -1,
          "set_emotion_steering(NULL) returns -1");
    CHECK(sonata_flow_set_prosody(NULL, dummy, 3) == -1,
          "set_prosody(NULL) returns -1");
    CHECK(sonata_flow_set_durations(NULL, dummy, 10) == -1,
          "set_durations(NULL) returns -1");
    CHECK(sonata_flow_set_prosody_embedding(NULL, dummy, 256) == -1,
          "set_prosody_embedding(NULL) returns -1");
    CHECK(sonata_flow_set_causal(NULL, 1) == -1,
          "set_causal(NULL) returns -1");

    /* Void functions — should handle NULL gracefully (no crash) */
    sonata_flow_reset_phase(NULL);
    CHECK(1, "reset_phase(NULL) no crash");

    sonata_flow_clear_speaker_embedding(NULL);
    CHECK(1, "clear_speaker_embedding(NULL) no crash");

    sonata_flow_clear_emotion_steering(NULL);
    CHECK(1, "clear_emotion_steering(NULL) no crash");

    sonata_flow_clear_prosody_embedding(NULL);
    CHECK(1, "clear_prosody_embedding(NULL) no crash");

    sonata_flow_reset_streaming(NULL);
    CHECK(1, "reset_streaming(NULL) no crash");
}

/* ─── Test 3: Create with invalid paths ──────────────────────────────── */

static void test_flow_create_invalid(void) {
    printf("\n═══ Test 3: Flow create with invalid paths ═══\n");

    void *engine = sonata_flow_create(
        "/nonexistent/flow.bin", "/nonexistent/flow.json",
        "/nonexistent/dec.bin", "/nonexistent/dec.json"
    );
    CHECK(engine == NULL, "create with nonexistent paths returns NULL");
    if (engine) sonata_flow_destroy(engine);
}

/* ─── Test 4: set_speaker NULL ───────────────────────────────────────── */

static void test_flow_set_speaker_null(void) {
    printf("\n═══ Test 4: set_speaker(NULL) ═══\n");

    CHECK(sonata_flow_set_speaker(NULL, 0) == -1,
          "set_speaker(NULL, 0) returns -1");
    CHECK(sonata_flow_set_speaker(NULL, 99) == -1,
          "set_speaker(NULL, 99) returns -1");
}

/* ─── Test 5: set_cfg_scale NULL ─────────────────────────────────────── */

static void test_flow_set_cfg_scale_null(void) {
    printf("\n═══ Test 5: set_cfg_scale(NULL) ═══\n");

    CHECK(sonata_flow_set_cfg_scale(NULL, 2.0f) == -1,
          "set_cfg_scale(NULL, 2.0) returns -1");
    CHECK(sonata_flow_set_cfg_scale(NULL, 0.0f) == -1,
          "set_cfg_scale(NULL, 0.0) returns -1");
}

/* ─── Test 6: set_n_steps NULL ───────────────────────────────────────── */

static void test_flow_set_n_steps_null(void) {
    printf("\n═══ Test 6: set_n_steps(NULL) ═══\n");

    CHECK(sonata_flow_set_n_steps(NULL, 4) == -1,
          "set_n_steps(NULL, 4) returns -1");
    CHECK(sonata_flow_set_n_steps(NULL, 16) == -1,
          "set_n_steps(NULL, 16) returns -1");
}

/* ─── Test 7: generate NULL ──────────────────────────────────────────── */

static void test_flow_generate_null(void) {
    printf("\n═══ Test 7: generate(NULL) ═══\n");

    int rc = sonata_flow_generate(NULL, NULL, 0, NULL, NULL);
    CHECK(rc == 0, "generate(NULL, NULL, 0, NULL, NULL) returns 0 (no frames)");

    int tokens[] = {1, 2, 3};
    float mag[256 * 3], phase[256 * 3];
    rc = sonata_flow_generate(NULL, tokens, 3, mag, phase);
    CHECK(rc == 0, "generate(NULL engine, valid args) returns 0");
}

/* ─── Test 8: set_speaker_embedding NULL ─────────────────────────────── */

static void test_flow_set_speaker_embedding_null(void) {
    printf("\n═══ Test 8: set_speaker_embedding(NULL) ═══\n");

    float data[256];
    memset(data, 0, sizeof(data));

    CHECK(sonata_flow_set_speaker_embedding(NULL, data, 256) == -1,
          "set_speaker_embedding(NULL engine, valid data, 256) returns -1");
    CHECK(sonata_flow_set_speaker_embedding(NULL, NULL, 256) == -1,
          "set_speaker_embedding(NULL, NULL, 256) returns -1");
}

/* ─── Test 9: set_emotion NULL ───────────────────────────────────────── */

static void test_flow_set_emotion_null(void) {
    printf("\n═══ Test 9: set_emotion(NULL) ═══\n");

    CHECK(sonata_flow_set_emotion(NULL, 0) == -1,
          "set_emotion(NULL, 0) returns -1");
    CHECK(sonata_flow_set_emotion(NULL, 5) == -1,
          "set_emotion(NULL, 5) returns -1");
}

/* ─── Test 10: set_prosody NULL ──────────────────────────────────────── */

static void test_flow_set_prosody_null(void) {
    printf("\n═══ Test 10: set_prosody(NULL) ═══\n");

    float features[] = {1.0f, 0.5f, 0.8f};
    CHECK(sonata_flow_set_prosody(NULL, features, 3) == -1,
          "set_prosody(NULL, data, 3) returns -1");
    CHECK(sonata_flow_set_prosody(NULL, NULL, 3) == -1,
          "set_prosody(NULL, NULL, 3) returns -1");
}

/* ─── Test 11: decoder_type NULL ─────────────────────────────────────── */

static void test_flow_decoder_type_null(void) {
    printf("\n═══ Test 11: decoder_type(NULL) ═══\n");

    int dt = sonata_flow_decoder_type(NULL);
    CHECKF(dt == 0, "decoder_type(NULL) = %d (expected 0)", dt);
}

/* ─── Test 12: samples_per_frame NULL ────────────────────────────────── */

static void test_flow_samples_per_frame_null(void) {
    printf("\n═══ Test 12: samples_per_frame(NULL) ═══\n");

    int spf = sonata_flow_samples_per_frame(NULL);
    CHECKF(spf == 0, "samples_per_frame(NULL) = %d (expected 0)", spf);
}

/* ─── Main ─────────────────────────────────────────────────────────────── */

int main(void) {
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║  Sonata Flow FFI — Boundary Test Suite        ║\n");
    printf("╚════════════════════════════════════════════════╝\n");

    test_flow_constants();
    test_flow_null_safety();
    test_flow_create_invalid();
    test_flow_set_speaker_null();
    test_flow_set_cfg_scale_null();
    test_flow_set_n_steps_null();
    test_flow_generate_null();
    test_flow_set_speaker_embedding_null();
    test_flow_set_emotion_null();
    test_flow_set_prosody_null();
    test_flow_decoder_type_null();
    test_flow_samples_per_frame_null();

    printf("\n══════════════════════════════════════════\n");
    printf("Results: %d / %d passed\n", g_pass, g_pass + g_fail);
    if (g_fail > 0) {
        printf("FAILURES: %d\n", g_fail);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
