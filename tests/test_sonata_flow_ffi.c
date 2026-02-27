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

/* ─── Test 2: Constants verification (detailed) ──────────────────────── */

static void test_flow_constants_detailed(void) {
    printf("\n═══ Test 2: Flow constants (detailed verification) ═══\n");

    int n_steps = sonata_flow_n_steps();
    CHECKF(n_steps > 0, "n_steps() = %d > 0", n_steps);
    CHECKF(n_steps <= 64, "n_steps() = %d <= 64 (reasonable upper bound)", n_steps);
    CHECKF(n_steps == 8, "n_steps() = %d == 8 (default)", n_steps);

    int adim = sonata_flow_acoustic_dim();
    CHECKF(adim > 0, "acoustic_dim() = %d > 0", adim);
    CHECKF((adim & (adim - 1)) == 0,
           "acoustic_dim() = %d is power of 2", adim);
    CHECKF(adim == 256, "acoustic_dim() = %d == 256 (default)", adim);

    /* Constants should be stable across calls */
    CHECK(sonata_flow_n_steps() == n_steps,
          "n_steps() is stable across calls");
    CHECK(sonata_flow_acoustic_dim() == adim,
          "acoustic_dim() is stable across calls");
}

/* ─── Test 3: Parameter bounds with NULL engine ──────────────────────── */

static void test_flow_parameter_bounds(void) {
    printf("\n═══ Test 3: Parameter bounds (NULL engine) ═══\n");

    /* Negative n_steps */
    CHECK(sonata_flow_set_n_steps(NULL, -1) == -1,
          "set_n_steps(NULL, -1) returns -1");
    CHECK(sonata_flow_set_n_steps(NULL, -100) == -1,
          "set_n_steps(NULL, -100) returns -1");
    CHECK(sonata_flow_set_n_steps(NULL, 0) == -1,
          "set_n_steps(NULL, 0) returns -1");

    /* Negative cfg_scale */
    CHECK(sonata_flow_set_cfg_scale(NULL, -1.0f) == -1,
          "set_cfg_scale(NULL, -1.0) returns -1");

    /* Extreme values */
    CHECK(sonata_flow_set_cfg_scale(NULL, 999999.0f) == -1,
          "set_cfg_scale(NULL, 999999.0) returns -1");
    CHECK(sonata_flow_set_n_steps(NULL, 999999) == -1,
          "set_n_steps(NULL, 999999) returns -1");

    /* Speaker with negative ID */
    CHECK(sonata_flow_set_speaker(NULL, -1) == -1,
          "set_speaker(NULL, -1) returns -1");

    /* Emotion with negative ID */
    CHECK(sonata_flow_set_emotion(NULL, -1) == -1,
          "set_emotion(NULL, -1) returns -1");

    /* Zero-dim speaker embedding */
    float dummy[1] = {0.0f};
    CHECK(sonata_flow_set_speaker_embedding(NULL, dummy, 0) == -1,
          "set_speaker_embedding(NULL, data, 0) returns -1");
    CHECK(sonata_flow_set_speaker_embedding(NULL, dummy, -1) == -1,
          "set_speaker_embedding(NULL, data, -1) returns -1");

    /* Zero-dim prosody embedding */
    CHECK(sonata_flow_set_prosody_embedding(NULL, dummy, 0) == -1,
          "set_prosody_embedding(NULL, data, 0) returns -1");

    /* Zero-dim emotion steering */
    CHECK(sonata_flow_set_emotion_steering(NULL, dummy, 0, 1.0f) == -1,
          "set_emotion_steering(NULL, data, 0, 1.0) returns -1");

    /* Prosody with zero features */
    CHECK(sonata_flow_set_prosody(NULL, dummy, 0) == -1,
          "set_prosody(NULL, data, 0) returns -1");

    /* Durations with zero frames */
    CHECK(sonata_flow_set_durations(NULL, dummy, 0) == -1,
          "set_durations(NULL, data, 0) returns -1");
}

/* ─── Test 4: Error recovery — operations after failed create ────────── */

static void test_flow_error_recovery(void) {
    printf("\n═══ Test 4: Error recovery after failed create ═══\n");

    /* Create with bad paths should fail */
    void *engine = sonata_flow_create(
        "/nonexistent/a", "/nonexistent/b",
        "/nonexistent/c", "/nonexistent/d"
    );
    CHECK(engine == NULL, "create with bad paths returns NULL");

    /* All operations on NULL should be safe */
    CHECK(sonata_flow_set_speaker(engine, 0) == -1,
          "set_speaker on failed engine returns -1");
    CHECK(sonata_flow_set_cfg_scale(engine, 1.0f) == -1,
          "set_cfg_scale on failed engine returns -1");
    CHECK(sonata_flow_set_n_steps(engine, 4) == -1,
          "set_n_steps on failed engine returns -1");

    int tokens[] = {1, 2, 3};
    float mag[256 * 3], phase[256 * 3];
    int rc = sonata_flow_generate(engine, tokens, 3, mag, phase);
    CHECK(rc == 0, "generate on failed engine returns 0");

    CHECK(sonata_flow_set_causal(engine, 1) == -1,
          "set_causal on failed engine returns -1");
    CHECK(sonata_flow_decoder_type(engine) == 0,
          "decoder_type on failed engine returns 0");
    CHECK(sonata_flow_samples_per_frame(engine) == 0,
          "samples_per_frame on failed engine returns 0");

    /* Void operations should not crash */
    sonata_flow_reset_phase(engine);
    CHECK(1, "reset_phase on failed engine no crash");
    sonata_flow_reset_streaming(engine);
    CHECK(1, "reset_streaming on failed engine no crash");
    sonata_flow_clear_speaker_embedding(engine);
    CHECK(1, "clear_speaker_embedding on failed engine no crash");
    sonata_flow_clear_emotion_steering(engine);
    CHECK(1, "clear_emotion_steering on failed engine no crash");
    sonata_flow_clear_prosody_embedding(engine);
    CHECK(1, "clear_prosody_embedding on failed engine no crash");

    /* Destroy NULL should be safe */
    sonata_flow_destroy(engine);
    CHECK(1, "destroy(NULL) after failed create is safe");
}

/* ─── Test 5: Memory — create/destroy loop (leak detection) ──────────── */

static void test_flow_memory_lifecycle(void) {
    printf("\n═══ Test 5: Memory lifecycle — repeated create/destroy ═══\n");

    /* Repeated create-with-invalid-paths/destroy should not leak */
    for (int i = 0; i < 100; i++) {
        void *engine = sonata_flow_create(
            "/nonexistent/flow.bin", "/nonexistent/flow.json",
            "/nonexistent/dec.bin", "/nonexistent/dec.json"
        );
        if (engine) {
            sonata_flow_destroy(engine);
        }
    }
    CHECK(1, "100x create(bad)/destroy cycles no crash");

    /* Repeated destroy(NULL) should be safe */
    for (int i = 0; i < 100; i++) {
        sonata_flow_destroy(NULL);
    }
    CHECK(1, "100x destroy(NULL) cycles no crash");
}

/* ─── Test 6: Solver parameter ───────────────────────────────────────── */

static void test_flow_solver_param(void) {
    printf("\n═══ Test 6: Solver parameter ═══\n");

    CHECK(sonata_flow_set_solver(NULL, 0) == -1,
          "set_solver(NULL, 0=euler) returns -1");
    CHECK(sonata_flow_set_solver(NULL, 1) == -1,
          "set_solver(NULL, 1=heun) returns -1");
    CHECK(sonata_flow_set_solver(NULL, -1) == -1,
          "set_solver(NULL, -1) returns -1");
    CHECK(sonata_flow_set_solver(NULL, 99) == -1,
          "set_solver(NULL, 99) returns -1");
}

/* ─── Test 7: Causal mode toggle ─────────────────────────────────────── */

static void test_flow_causal_toggle(void) {
    printf("\n═══ Test 7: Causal mode toggle ═══\n");

    CHECK(sonata_flow_set_causal(NULL, 0) == -1,
          "set_causal(NULL, 0=disable) returns -1");
    CHECK(sonata_flow_set_causal(NULL, 1) == -1,
          "set_causal(NULL, 1=enable) returns -1");
    CHECK(sonata_flow_set_causal(NULL, -1) == -1,
          "set_causal(NULL, -1) returns -1");
}

/* ─── Test 8: Generate with various frame counts ─────────────────────── */

static void test_flow_generate_frames(void) {
    printf("\n═══ Test 8: Generate with various frame counts ═══\n");

    int tokens[] = {1, 2, 3, 4, 5};
    float mag[256 * 5], phase[256 * 5];

    /* Zero frames */
    int rc = sonata_flow_generate(NULL, tokens, 0, mag, phase);
    CHECK(rc == 0, "generate(NULL, tokens, 0) returns 0");

    /* Negative frames */
    rc = sonata_flow_generate(NULL, tokens, -1, mag, phase);
    CHECK(rc == 0, "generate(NULL, tokens, -1) returns 0");

    /* NULL tokens with positive frames */
    rc = sonata_flow_generate(NULL, NULL, 5, mag, phase);
    CHECK(rc == 0, "generate(NULL, NULL, 5) returns 0");

    /* NULL output buffers */
    rc = sonata_flow_generate(NULL, tokens, 3, NULL, NULL);
    CHECK(rc == 0, "generate(NULL, tokens, 3, NULL, NULL) returns 0");
}

/* ─── Test 9: set_durations edge cases ───────────────────────────────── */

static void test_flow_set_durations_edge(void) {
    printf("\n═══ Test 9: set_durations edge cases ═══\n");

    float durs[] = {1.0f, 2.0f, 3.0f};
    CHECK(sonata_flow_set_durations(NULL, durs, 3) == -1,
          "set_durations(NULL, data, 3) returns -1");
    CHECK(sonata_flow_set_durations(NULL, NULL, 3) == -1,
          "set_durations(NULL, NULL, 3) returns -1");
    CHECK(sonata_flow_set_durations(NULL, durs, -1) == -1,
          "set_durations(NULL, data, -1) returns -1");
}

/* ─── Test 10: Comprehensive null safety ─────────────────────────────── */

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

/* ─── Test 11: Create with invalid paths ─────────────────────────────── */

static void test_flow_create_invalid(void) {
    printf("\n═══ Test 11: Flow create with invalid paths ═══\n");

    void *engine = sonata_flow_create(
        "/nonexistent/flow.bin", "/nonexistent/flow.json",
        "/nonexistent/dec.bin", "/nonexistent/dec.json"
    );
    CHECK(engine == NULL, "create with nonexistent paths returns NULL");
    if (engine) sonata_flow_destroy(engine);
}

/* ─── Test 12: Create with NULL paths ────────────────────────────────── */

static void test_flow_create_null_paths(void) {
    printf("\n═══ Test 12: Flow create with NULL paths ═══\n");

    void *engine = sonata_flow_create(NULL, NULL, NULL, NULL);
    CHECK(engine == NULL, "create(NULL, NULL, NULL, NULL) returns NULL");
    if (engine) sonata_flow_destroy(engine);

    engine = sonata_flow_create("/some/path", NULL, NULL, NULL);
    CHECK(engine == NULL, "create with partial NULL paths returns NULL");
    if (engine) sonata_flow_destroy(engine);
}

/* ─── Test 13: Create with empty string paths ────────────────────────── */

static void test_flow_create_empty_paths(void) {
    printf("\n═══ Test 13: Flow create with empty string paths ═══\n");

    void *engine = sonata_flow_create("", "", "", "");
    CHECK(engine == NULL, "create with empty strings returns NULL");
    if (engine) sonata_flow_destroy(engine);
}

/* ─── Test 14: decoder_type NULL ─────────────────────────────────────── */

static void test_flow_decoder_type_null(void) {
    printf("\n═══ Test 14: decoder_type(NULL) ═══\n");

    int dt = sonata_flow_decoder_type(NULL);
    CHECKF(dt == 0, "decoder_type(NULL) = %d (expected 0)", dt);
}

/* ─── Test 15: samples_per_frame NULL ────────────────────────────────── */

static void test_flow_samples_per_frame_null(void) {
    printf("\n═══ Test 15: samples_per_frame(NULL) ═══\n");

    int spf = sonata_flow_samples_per_frame(NULL);
    CHECKF(spf == 0, "samples_per_frame(NULL) = %d (expected 0)", spf);
}

/* ─── Test 16: Emotion steering edge cases ───────────────────────────── */

static void test_flow_emotion_steering_edge(void) {
    printf("\n═══ Test 16: Emotion steering edge cases ═══\n");

    float dir[8] = {1.0f, 0.0f, -1.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f};

    CHECK(sonata_flow_set_emotion_steering(NULL, dir, 8, 1.0f) == -1,
          "set_emotion_steering(NULL, data, 8, 1.0) returns -1");
    CHECK(sonata_flow_set_emotion_steering(NULL, dir, 8, 0.0f) == -1,
          "set_emotion_steering(NULL, data, 8, 0.0) returns -1");
    CHECK(sonata_flow_set_emotion_steering(NULL, dir, 8, -1.0f) == -1,
          "set_emotion_steering(NULL, data, 8, -1.0) returns -1");
    CHECK(sonata_flow_set_emotion_steering(NULL, NULL, 8, 1.0f) == -1,
          "set_emotion_steering(NULL, NULL, 8, 1.0) returns -1");
}

/* ─── Test 17: Prosody features edge cases ───────────────────────────── */

static void test_flow_prosody_features_edge(void) {
    printf("\n═══ Test 17: Prosody features edge cases ═══\n");

    float features[] = {1.0f, 0.5f, 0.8f};
    CHECK(sonata_flow_set_prosody(NULL, features, 3) == -1,
          "set_prosody(NULL, data, 3) returns -1");
    CHECK(sonata_flow_set_prosody(NULL, NULL, 3) == -1,
          "set_prosody(NULL, NULL, 3) returns -1");
    CHECK(sonata_flow_set_prosody(NULL, features, -1) == -1,
          "set_prosody(NULL, data, -1) returns -1");

    /* Large feature count */
    float big_features[1024];
    memset(big_features, 0, sizeof(big_features));
    CHECK(sonata_flow_set_prosody(NULL, big_features, 1024) == -1,
          "set_prosody(NULL, big_data, 1024) returns -1");
}

/* ─── Test 18: Speaker embedding dimension edge cases ────────────────── */

static void test_flow_speaker_embedding_dims(void) {
    printf("\n═══ Test 18: Speaker embedding dimension edge cases ═══\n");

    float data[512];
    memset(data, 0, sizeof(data));

    CHECK(sonata_flow_set_speaker_embedding(NULL, data, 256) == -1,
          "set_speaker_embedding(NULL, data, 256) returns -1");
    CHECK(sonata_flow_set_speaker_embedding(NULL, data, 512) == -1,
          "set_speaker_embedding(NULL, data, 512) returns -1");
    CHECK(sonata_flow_set_speaker_embedding(NULL, data, 1) == -1,
          "set_speaker_embedding(NULL, data, 1) returns -1");
    CHECK(sonata_flow_set_speaker_embedding(NULL, NULL, 256) == -1,
          "set_speaker_embedding(NULL, NULL, 256) returns -1");
}

/* ─── Main ─────────────────────────────────────────────────────────────── */

int main(void) {
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║  Sonata Flow FFI — Boundary Test Suite        ║\n");
    printf("╚════════════════════════════════════════════════╝\n");

    test_flow_constants();
    test_flow_constants_detailed();
    test_flow_parameter_bounds();
    test_flow_error_recovery();
    test_flow_memory_lifecycle();
    test_flow_solver_param();
    test_flow_causal_toggle();
    test_flow_generate_frames();
    test_flow_set_durations_edge();
    test_flow_null_safety();
    test_flow_create_invalid();
    test_flow_create_null_paths();
    test_flow_create_empty_paths();
    test_flow_decoder_type_null();
    test_flow_samples_per_frame_null();
    test_flow_emotion_steering_edge();
    test_flow_prosody_features_edge();
    test_flow_speaker_embedding_dims();

    printf("\n══════════════════════════════════════════\n");
    printf("Results: %d / %d passed\n", g_pass, g_pass + g_fail);
    if (g_fail > 0) {
        printf("FAILURES: %d\n", g_fail);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
