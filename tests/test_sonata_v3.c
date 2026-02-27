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

/* ─── Test 7: Create with NULL paths ───────────────────────────────────── */

static void test_create_null_paths(void) {
    printf("\n═══ Test 7: Create with NULL paths ═══\n");

    /* Both NULL */
    void *flow = sonata_flow_v3_create(NULL, NULL);
    CHECK(flow == NULL, "flow_v3 create(NULL, NULL) returns NULL");
    if (flow) sonata_flow_v3_destroy(flow);

    /* NULL weights, valid config path */
    flow = sonata_flow_v3_create(NULL, "models/sonata/flow_v3_config.json");
    CHECK(flow == NULL, "flow_v3 create(NULL weights, valid config) returns NULL");
    if (flow) sonata_flow_v3_destroy(flow);

    /* Valid weights path, NULL config */
    flow = sonata_flow_v3_create("models/sonata/flow_v3.safetensors", NULL);
    CHECK(flow == NULL, "flow_v3 create(valid weights, NULL config) returns NULL");
    if (flow) sonata_flow_v3_destroy(flow);

    /* Vocoder: both NULL */
    void *voc = sonata_vocoder_create(NULL, NULL);
    CHECK(voc == NULL, "vocoder create(NULL, NULL) returns NULL");
    if (voc) sonata_vocoder_destroy(voc);

    /* Vocoder: NULL weights only */
    voc = sonata_vocoder_create(NULL, "models/sonata/vocoder_config.json");
    CHECK(voc == NULL, "vocoder create(NULL weights, valid config) returns NULL");
    if (voc) sonata_vocoder_destroy(voc);

    /* Vocoder: NULL config only */
    voc = sonata_vocoder_create("models/sonata/vocoder.safetensors", NULL);
    CHECK(voc == NULL, "vocoder create(valid weights, NULL config) returns NULL");
    if (voc) sonata_vocoder_destroy(voc);
}

/* ─── Test 8: Vocoder invalid mel_dim ──────────────────────────────────── */

static void test_vocoder_invalid_mel_dim(void) {
    printf("\n═══ Test 8: Vocoder invalid mel_dim ═══\n");

    void *engine = sonata_vocoder_create(
        "models/sonata/vocoder.safetensors",
        "models/sonata/vocoder_config.json"
    );
    if (!engine) {
        printf("  [SKIP] No Vocoder weights\n");
        return;
    }

    float mel_buf[80 * 10];
    float audio_buf[24000];
    memset(mel_buf, 0, sizeof(mel_buf));

    /* mel_dim = 0 — should return -1 */
    int rc = sonata_vocoder_generate(engine, mel_buf, 10, 0, audio_buf, 24000);
    CHECK(rc == -1, "vocoder generate with mel_dim=0 returns -1");

    /* mel_dim = -1 — should return -1 */
    rc = sonata_vocoder_generate(engine, mel_buf, 10, -1, audio_buf, 24000);
    CHECK(rc == -1, "vocoder generate with mel_dim=-1 returns -1");

    /* mel_dim = 1 — unusual but may work or return -1 */
    rc = sonata_vocoder_generate(engine, mel_buf, 10, 1, audio_buf, 24000);
    CHECK(rc >= -1, "vocoder generate with mel_dim=1 doesn't crash");

    sonata_vocoder_destroy(engine);
}

/* ─── Test 9: Vocoder NULL buffer inputs ───────────────────────────────── */

static void test_vocoder_null_buffers(void) {
    printf("\n═══ Test 9: Vocoder NULL buffer inputs ═══\n");

    void *engine = sonata_vocoder_create(
        "models/sonata/vocoder.safetensors",
        "models/sonata/vocoder_config.json"
    );
    if (!engine) {
        printf("  [SKIP] No Vocoder weights\n");
        return;
    }

    float mel_buf[80 * 10];
    float audio_buf[24000];
    memset(mel_buf, 0, sizeof(mel_buf));

    /* NULL mel_data */
    int rc = sonata_vocoder_generate(engine, NULL, 10, 80, audio_buf, 24000);
    CHECK(rc == -1, "vocoder generate with NULL mel_data returns -1");

    /* NULL out_audio */
    rc = sonata_vocoder_generate(engine, mel_buf, 10, 80, NULL, 24000);
    CHECK(rc == -1, "vocoder generate with NULL out_audio returns -1");

    /* max_samples = 0 */
    rc = sonata_vocoder_generate(engine, mel_buf, 10, 80, audio_buf, 0);
    CHECK(rc == -1 || rc == 0, "vocoder generate with max_samples=0 returns -1 or 0");

    /* max_samples = -1 */
    rc = sonata_vocoder_generate(engine, mel_buf, 10, 80, audio_buf, -1);
    CHECK(rc == -1, "vocoder generate with max_samples=-1 returns -1");

    sonata_vocoder_destroy(engine);
}

/* ─── Test 10: Flow v3 negative/extreme parameter values ──────────────── */

static void test_flow_v3_extreme_params(void) {
    printf("\n═══ Test 10: Flow v3 negative/extreme parameter values ═══\n");

    /* These all operate on NULL engine — should return -1 safely */
    int rc;

    /* Negative speaker ID */
    rc = sonata_flow_v3_set_speaker(NULL, -1);
    CHECK(rc == -1, "set_speaker(NULL, -1) returns -1");

    rc = sonata_flow_v3_set_speaker(NULL, -999);
    CHECK(rc == -1, "set_speaker(NULL, -999) returns -1");

    /* Very large speaker ID */
    rc = sonata_flow_v3_set_speaker(NULL, 999999);
    CHECK(rc == -1, "set_speaker(NULL, 999999) returns -1");

    /* Extreme cfg_scale */
    rc = sonata_flow_v3_set_cfg_scale(NULL, 1e30f);
    CHECK(rc == -1, "set_cfg_scale(NULL, 1e30) returns -1");

    rc = sonata_flow_v3_set_cfg_scale(NULL, 0.0f);
    CHECK(rc == -1, "set_cfg_scale(NULL, 0.0) returns -1");

    /* Negative n_steps */
    rc = sonata_flow_v3_set_n_steps(NULL, -5);
    CHECK(rc == -1, "set_n_steps(NULL, -5) returns -1");

    /* Solver values beyond 0/1 */
    rc = sonata_flow_v3_set_solver(NULL, -1);
    CHECK(rc == -1, "set_solver(NULL, -1) returns -1");

    rc = sonata_flow_v3_set_solver(NULL, 99);
    CHECK(rc == -1, "set_solver(NULL, 99) returns -1");
}

/* ─── Test 11: Double destroy safety ───────────────────────────────────── */

static void test_double_destroy_safety(void) {
    printf("\n═══ Test 11: Double destroy safety ═══\n");

    /* Double-destroy NULL should be safe */
    sonata_flow_v3_destroy(NULL);
    sonata_flow_v3_destroy(NULL);
    CHECK(1, "flow_v3 double destroy(NULL) safe");

    sonata_vocoder_destroy(NULL);
    sonata_vocoder_destroy(NULL);
    CHECK(1, "vocoder double destroy(NULL) safe");
}

/* ─── Test 12: Generate with oversized frame request ───────────────────── */

static void test_flow_v3_oversized_frames(void) {
    printf("\n═══ Test 12: Flow v3 generate with oversized frames ═══\n");

    void *engine = sonata_flow_v3_create(
        "models/sonata/flow_v3.safetensors",
        "models/sonata/flow_v3_config.json"
    );
    if (!engine) {
        printf("  [SKIP] No Flow v3 weights\n");
        return;
    }

    /* Tiny output buffer but requesting many frames */
    float tiny_mel[80];
    int rc = sonata_flow_v3_generate(engine,
        "hello world", 11, NULL, 0, 1000000,
        tiny_mel, 1);
    CHECKF(rc >= -1, "generate with target_frames=1000000, max_frames=1 returns %d (no crash)", rc);

    /* Negative target_frames */
    float mel_buf[80 * 25];
    rc = sonata_flow_v3_generate(engine,
        "hi", 2, NULL, 0, -10,
        mel_buf, 80 * 25);
    CHECKF(rc >= -1, "generate with target_frames=-10 returns %d (no crash)", rc);

    sonata_flow_v3_destroy(engine);
}

/* ─── Test 13: Flow v3 generate with phoneme IDs ──────────────────────── */

static void test_flow_v3_generate_with_phonemes(void) {
    printf("\n═══ Test 13: Flow v3 generate with phoneme IDs ═══\n");

    void *engine = sonata_flow_v3_create(
        "models/sonata/flow_v3.safetensors",
        "models/sonata/flow_v3_config.json"
    );
    if (!engine) {
        printf("  [SKIP] No Flow v3 weights\n");
        return;
    }

    float mel_buf[80 * 50];
    int max_frames = 80 * 50;

    /* Valid phoneme IDs with NULL text */
    int phonemes[] = {1, 2, 3, 4, 5};
    int rc = sonata_flow_v3_generate(engine,
        NULL, 0, phonemes, 5, 25,
        mel_buf, max_frames);
    CHECKF(rc >= -1, "generate with phonemes only returns %d", rc);

    /* Both text and phonemes provided */
    rc = sonata_flow_v3_generate(engine,
        "hello", 5, phonemes, 5, 25,
        mel_buf, max_frames);
    CHECKF(rc >= -1, "generate with text + phonemes returns %d", rc);

    /* Zero-length phoneme array */
    rc = sonata_flow_v3_generate(engine,
        NULL, 0, phonemes, 0, 25,
        mel_buf, max_frames);
    CHECK(rc == -1, "generate with phoneme_len=0 + NULL text returns -1");

    /* Negative phoneme length */
    rc = sonata_flow_v3_generate(engine,
        NULL, 0, phonemes, -1, 25,
        mel_buf, max_frames);
    CHECK(rc == -1, "generate with phoneme_len=-1 returns -1");

    sonata_flow_v3_destroy(engine);
}

/* ─── Test 14: Create with empty string paths ─────────────────────────── */

static void test_create_empty_paths(void) {
    printf("\n═══ Test 14: Create with empty string paths ═══\n");

    void *flow = sonata_flow_v3_create("", "");
    CHECK(flow == NULL, "flow_v3 create('', '') returns NULL");
    if (flow) sonata_flow_v3_destroy(flow);

    void *voc = sonata_vocoder_create("", "");
    CHECK(voc == NULL, "vocoder create('', '') returns NULL");
    if (voc) sonata_vocoder_destroy(voc);

    /* Mixed: one empty, one valid path */
    flow = sonata_flow_v3_create("", "models/sonata/flow_v3_config.json");
    CHECK(flow == NULL, "flow_v3 create('', valid config) returns NULL");
    if (flow) sonata_flow_v3_destroy(flow);

    voc = sonata_vocoder_create("models/sonata/vocoder.safetensors", "");
    CHECK(voc == NULL, "vocoder create(valid weights, '') returns NULL");
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
    test_create_null_paths();
    test_vocoder_invalid_mel_dim();
    test_vocoder_null_buffers();
    test_flow_v3_extreme_params();
    test_double_destroy_safety();
    test_flow_v3_oversized_frames();
    test_flow_v3_generate_with_phonemes();
    test_create_empty_paths();

    printf("\n══════════════════════════════════════════\n");
    printf("Results: %d / %d passed\n", g_pass, g_pass + g_fail);
    if (g_fail > 0) {
        printf("FAILURES: %d\n", g_fail);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
