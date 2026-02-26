/*
 * test_sonata_v3.c — FFI tests for Sonata Flow v3 and Vocoder.
 *
 * Tests null-safety, parameter validation, and invalid-input handling
 * for sonata_flow_v3_* and sonata_vocoder_* without requiring model weights.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Flow v3 FFI ─────────────────────────────────────────────────────── */

extern void *sonata_flow_v3_create(const char *weights_path, const char *config_path);
extern void  sonata_flow_v3_destroy(void *engine);
extern int   sonata_flow_v3_generate(void *engine,
    const char *text_ptr, int text_len,
    const int *phoneme_ids_ptr, int phoneme_len,
    int target_frames, float *out_mel, int max_frames);
extern int   sonata_flow_v3_set_cfg_scale(void *engine, float scale);
extern int   sonata_flow_v3_set_n_steps(void *engine, int steps);
extern int   sonata_flow_v3_set_speaker(void *engine, int speaker_id);
extern int   sonata_flow_v3_set_solver(void *engine, int use_heun);

/* ─── Vocoder FFI ──────────────────────────────────────────────────────── */

extern void *sonata_vocoder_create(const char *weights_path, const char *config_path);
extern void  sonata_vocoder_destroy(void *engine);
extern int   sonata_vocoder_generate(void *engine,
    const float *mel_data, int n_frames, int mel_dim,
    float *out_audio, int max_samples);

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

/* ─── Test 1: Flow v3 null safety ──────────────────────────────────────── */

static void test_flow_v3_null_safety(void) {
    printf("\n═══ Test 1: Flow v3 null safety ═══\n");

    /* destroy(NULL) should not crash */
    sonata_flow_v3_destroy(NULL);
    CHECK(1, "destroy(NULL) no crash");

    /* generate(NULL, ...) returns -1 */
    float mel_buf[80 * 25];
    const char *text = "hello";
    int rc = sonata_flow_v3_generate(NULL, text, 5, NULL, 0, 0, mel_buf, 80 * 25);
    CHECK(rc == -1, "generate(NULL, ...) returns -1");

    /* set_cfg_scale(NULL, ...) returns -1 */
    rc = sonata_flow_v3_set_cfg_scale(NULL, 1.5f);
    CHECK(rc == -1, "set_cfg_scale(NULL, 1.5) returns -1");

    /* set_n_steps(NULL, ...) returns -1 */
    rc = sonata_flow_v3_set_n_steps(NULL, 8);
    CHECK(rc == -1, "set_n_steps(NULL, 8) returns -1");

    /* set_speaker(NULL, ...) returns -1 */
    rc = sonata_flow_v3_set_speaker(NULL, 0);
    CHECK(rc == -1, "set_speaker(NULL, 0) returns -1");

    /* set_solver(NULL, ...) returns -1 */
    rc = sonata_flow_v3_set_solver(NULL, 1);
    CHECK(rc == -1, "set_solver(NULL, 1) returns -1");
}

/* ─── Test 2: Vocoder null safety ───────────────────────────────────────── */

static void test_vocoder_null_safety(void) {
    printf("\n═══ Test 2: Vocoder null safety ═══\n");

    /* destroy(NULL) should not crash */
    sonata_vocoder_destroy(NULL);
    CHECK(1, "vocoder destroy(NULL) no crash");

    /* generate(NULL, ...) returns -1 */
    float mel_buf[80 * 10];
    float audio_buf[2400];
    int rc = sonata_vocoder_generate(NULL, mel_buf, 10, 80, audio_buf, 2400);
    CHECK(rc == -1, "vocoder generate(NULL, ...) returns -1");
}

/* ─── Test 3: Flow v3 parameter validation (requires engine) ────────────── */

static void test_flow_v3_param_validation(void) {
    printf("\n═══ Test 3: Flow v3 parameter validation ═══\n");

    void *engine = sonata_flow_v3_create(
        "models/sonata/flow_v3.safetensors",
        "models/sonata/flow_v3_config.json"
    );
    if (!engine) {
        printf("  [SKIP] No Flow v3 weights (not yet trained)\n");
        return;
    }

    /* set_n_steps(engine, 0) — invalid, should return -1 */
    int rc = sonata_flow_v3_set_n_steps(engine, 0);
    CHECK(rc == -1, "set_n_steps(engine, 0) returns -1 (invalid)");

    /* set_n_steps(engine, 100) — too high, should return -1 */
    rc = sonata_flow_v3_set_n_steps(engine, 100);
    CHECK(rc == -1, "set_n_steps(engine, 100) returns -1 (too high)");

    /* set_cfg_scale(engine, -1.0) — invalid, should return -1 */
    rc = sonata_flow_v3_set_cfg_scale(engine, -1.0f);
    CHECK(rc == -1, "set_cfg_scale(engine, -1.0) returns -1");

    sonata_flow_v3_destroy(engine);
}

/* ─── Test 4: Generate with NULL/empty text and phonemes ─────────────────── */

static void test_flow_v3_generate_null_input(void) {
    printf("\n═══ Test 4: Flow v3 generate with NULL/empty input ═══\n");

    void *engine = sonata_flow_v3_create(
        "models/sonata/flow_v3.safetensors",
        "models/sonata/flow_v3_config.json"
    );
    if (!engine) {
        printf("  [SKIP] No Flow v3 weights (not yet trained)\n");
        return;
    }

    float mel_buf[80 * 25];
    int max_frames = 80 * 25;

    /* NULL text, NULL phonemes, zero lengths — empty input → -1 */
    int rc = sonata_flow_v3_generate(engine,
        NULL, 0, NULL, 0, 0,
        mel_buf, max_frames);
    CHECK(rc == -1, "generate with NULL text + NULL phonemes returns -1");

    /* Valid output buffer but empty input */
    rc = sonata_flow_v3_generate(engine,
        "", 0, NULL, 0, 0,
        mel_buf, max_frames);
    CHECK(rc == -1, "generate with empty text returns -1");

    /* NULL out_mel — should return -1 */
    rc = sonata_flow_v3_generate(engine,
        "hi", 2, NULL, 0, 0,
        NULL, max_frames);
    CHECK(rc == -1, "generate with NULL out_mel returns -1");

    /* max_frames <= 0 — should return -1 */
    rc = sonata_flow_v3_generate(engine,
        "hi", 2, NULL, 0, 0,
        mel_buf, 0);
    CHECK(rc == -1, "generate with max_frames=0 returns -1");

    sonata_flow_v3_destroy(engine);
}

/* ─── Test 5: Vocoder with zero frames ─────────────────────────────────── */

static void test_vocoder_zero_frames(void) {
    printf("\n═══ Test 5: Vocoder with zero frames ═══\n");

    void *engine = sonata_vocoder_create(
        "models/sonata/vocoder.safetensors",
        "models/sonata/vocoder_config.json"
    );
    if (!engine) {
        printf("  [SKIP] No Vocoder weights (not yet trained)\n");
        return;
    }

    float mel_buf[80];
    float audio_buf[2400];

    /* n_frames=0 — should return -1 or 0 */
    int rc = sonata_vocoder_generate(engine, mel_buf, 0, 80, audio_buf, 2400);
    CHECK(rc == -1 || rc == 0, "vocoder generate with n_frames=0 returns -1 or 0");

    sonata_vocoder_destroy(engine);
}

/* ─── Test 6: Create with invalid paths ───────────────────────────────────── */

static void test_create_invalid_paths(void) {
    printf("\n═══ Test 6: Create with invalid paths ═══\n");

    void *flow = sonata_flow_v3_create("/nonexistent/flow.safetensors",
                                       "/nonexistent/flow.json");
    CHECK(flow == NULL, "flow_v3 create with nonexistent paths returns NULL");
    if (flow) sonata_flow_v3_destroy(flow);

    void *voc = sonata_vocoder_create("/nonexistent/vocoder.safetensors",
                                      "/nonexistent/vocoder.json");
    CHECK(voc == NULL, "vocoder create with nonexistent paths returns NULL");
    if (voc) sonata_vocoder_destroy(voc);
}

/* ─── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║  Sonata Flow v3 + Vocoder FFI Test Suite      ║\n");
    printf("╚════════════════════════════════════════════════╝\n");

    test_flow_v3_null_safety();
    test_vocoder_null_safety();
    test_flow_v3_param_validation();
    test_flow_v3_generate_null_input();
    test_vocoder_zero_frames();
    test_create_invalid_paths();

    printf("\n══════════════════════════════════════════\n");
    printf("Results: %d / %d passed\n", g_pass, g_pass + g_fail);
    if (g_fail > 0) {
        printf("FAILURES: %d\n", g_fail);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
