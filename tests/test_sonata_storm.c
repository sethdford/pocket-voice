/*
 * test_sonata_storm.c — Comprehensive test for the SoundStorm parallel token predictor FFI.
 *
 * Tests:
 *   1.  Constants: sample_rate=24000, frame_rate=50
 *   2.  Null safety: all FFI functions with NULL handle
 *   3.  Create with invalid paths returns NULL
 *   4.  Create with NULL paths returns NULL
 *   5.  Destroy NULL doesn't crash
 *   6.  set_text with NULL engine returns -1
 *   7.  set_text with NULL ids (model-dependent)
 *   8.  set_text with empty ids (model-dependent)
 *   9.  set_params with NULL engine returns -1
 *  10.  set_params with various ranges (model-dependent)
 *  11.  generate with NULL engine returns -1
 *  12.  generate with NULL output returns -1
 *  13.  reset with NULL engine returns -1
 *  14.  Generate without valid model — graceful failure
 *  15.  Full lifecycle: create → set_text → generate → reset → destroy
 *  16.  Create with swapped paths (config as weights, weights as config)
 *  17.  Create with very long paths (4096 chars, slash-only)
 *  18.  Create with special characters (tabs, newlines, control chars, spaces)
 *  19.  set_params with extreme temperature (negative, large, inf, NaN)
 *  20.  set_params with extreme rounds (0, negative, very large, INT_MAX)
 *  21.  generate with zero max_tokens
 *  22.  generate with negative max_tokens
 *  23.  set_text with large token array (1000 IDs)
 *  24.  set_text with max-value token IDs (UINT_MAX, zero)
 *  25.  set_text with negative count
 *  26.  Memory lifecycle stress: 100 create/destroy + 50 all-NULL rounds
 *  27.  Double destroy safety
 *  28.  Reset without prior set_text
 *  29.  Generate without set_text
 *  30.  Constants consistency across repeated calls
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── SoundStorm FFI ────────────────────────────────────────────────────── */

extern void *sonata_storm_create(const char *weights_path, const char *config_path);
extern void  sonata_storm_destroy(void *engine);
extern int   sonata_storm_set_text(void *engine, const unsigned int *text_ids, int n);
extern int   sonata_storm_generate(void *engine, int *out_tokens, int max_tokens, int *out_count);
extern int   sonata_storm_set_params(void *engine, float temperature, int n_rounds);
extern int   sonata_storm_reset(void *engine);
extern int   sonata_storm_sample_rate(void);
extern int   sonata_storm_frame_rate(void);

/* ─── Test helpers ───────────────────────────────────────────────────────── */

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

static const char *WEIGHTS_PATH = "models/sonata/sonata_storm.safetensors";
static const char *CONFIG_PATH  = "models/sonata/sonata_storm_config.json";

static int model_exists(void) {
    FILE *fw = fopen(WEIGHTS_PATH, "rb");
    FILE *fc = fopen(CONFIG_PATH, "rb");
    int ok = (fw != NULL && fc != NULL);
    if (fw) fclose(fw);
    if (fc) fclose(fc);
    return ok;
}

/* ─── Test 1: Constants ─────────────────────────────────────────────────── */

static void test_storm_constants(void) {
    printf("\n═══ Test 1: SoundStorm constants ═══\n");

    int sr = sonata_storm_sample_rate();
    CHECKF(sr == 24000, "sample_rate = %d (expected 24000)", sr);

    int fr = sonata_storm_frame_rate();
    CHECKF(fr == 50, "frame_rate = %d (expected 50)", fr);
}

/* ─── Test 2: Null safety ───────────────────────────────────────────────── */

static void test_storm_null_safety(void) {
    printf("\n═══ Test 2: Null safety — all FFI functions ═══\n");

    unsigned int dummy_ids[] = {1, 2, 3};
    int rc;

    rc = sonata_storm_set_text(NULL, dummy_ids, 3);
    CHECKF(rc == -1, "set_text(NULL, ids, 3) = %d (expected -1)", rc);

    int out_tokens[64];
    int out_count = 0;
    rc = sonata_storm_generate(NULL, out_tokens, 64, &out_count);
    CHECKF(rc == -1, "generate(NULL, buf, 64, &count) = %d (expected -1)", rc);

    rc = sonata_storm_set_params(NULL, 0.8f, 8);
    CHECKF(rc == -1, "set_params(NULL, 0.8, 8) = %d (expected -1)", rc);

    rc = sonata_storm_reset(NULL);
    CHECKF(rc == -1, "reset(NULL) = %d (expected -1)", rc);

    /* destroy(NULL) should be a safe no-op */
    sonata_storm_destroy(NULL);
    CHECK(1, "destroy(NULL) did not crash");
}

/* ─── Test 3: Create with invalid paths ──────────────────────────────────── */

static void test_storm_create_invalid_paths(void) {
    printf("\n═══ Test 3: Create with invalid paths ═══\n");

    void *engine = sonata_storm_create(
        "/nonexistent/path/weights.safetensors",
        "/nonexistent/path/config.json"
    );
    CHECK(engine == NULL, "create with nonexistent paths returns NULL");
    if (engine) sonata_storm_destroy(engine);

    engine = sonata_storm_create("", "");
    CHECK(engine == NULL, "create with empty strings returns NULL");
    if (engine) sonata_storm_destroy(engine);
}

/* ─── Test 4: Create with NULL paths ─────────────────────────────────────── */

static void test_storm_create_null_paths(void) {
    printf("\n═══ Test 4: Create with NULL paths ═══\n");

    void *engine = sonata_storm_create(NULL, NULL);
    CHECK(engine == NULL, "create(NULL, NULL) returns NULL");
    if (engine) sonata_storm_destroy(engine);

    engine = sonata_storm_create(WEIGHTS_PATH, NULL);
    CHECK(engine == NULL, "create(weights, NULL) returns NULL");
    if (engine) sonata_storm_destroy(engine);

    engine = sonata_storm_create(NULL, CONFIG_PATH);
    CHECK(engine == NULL, "create(NULL, config) returns NULL");
    if (engine) sonata_storm_destroy(engine);
}

/* ─── Test 5: Destroy NULL ───────────────────────────────────────────────── */

static void test_storm_destroy_null(void) {
    printf("\n═══ Test 5: Destroy NULL ═══\n");

    sonata_storm_destroy(NULL);
    CHECK(1, "destroy(NULL) is a safe no-op");

    sonata_storm_destroy(NULL);
    CHECK(1, "destroy(NULL) repeated is still safe");
}

/* ─── Test 6: set_text with NULL engine ──────────────────────────────────── */

static void test_storm_set_text_null_engine(void) {
    printf("\n═══ Test 6: set_text with NULL engine ═══\n");

    unsigned int ids[] = {10, 20, 30, 40, 50};
    int rc = sonata_storm_set_text(NULL, ids, 5);
    CHECKF(rc == -1, "set_text(NULL, ids, 5) = %d (expected -1)", rc);

    rc = sonata_storm_set_text(NULL, ids, 0);
    CHECKF(rc == -1, "set_text(NULL, ids, 0) = %d (expected -1)", rc);
}

/* ─── Test 7: set_text with NULL ids ─────────────────────────────────────── */

static void test_storm_set_text_null_ids(void) {
    printf("\n═══ Test 7: set_text with NULL ids ═══\n");

    if (!model_exists()) {
        printf("  [SKIP] Model weights not found — skipping NULL ids test\n");
        return;
    }

    void *engine = sonata_storm_create(WEIGHTS_PATH, CONFIG_PATH);
    if (!engine) {
        printf("  [SKIP] Failed to create engine — skipping\n");
        return;
    }

    int rc = sonata_storm_set_text(engine, NULL, 0);
    CHECKF(rc == 0 || rc == -1, "set_text(engine, NULL, 0) = %d (handled gracefully)", rc);

    rc = sonata_storm_set_text(engine, NULL, 5);
    CHECKF(rc == -1, "set_text(engine, NULL, 5) = %d (expected -1, NULL data with n>0)", rc);

    sonata_storm_destroy(engine);
}

/* ─── Test 8: set_text with empty ids ────────────────────────────────────── */

static void test_storm_set_text_empty(void) {
    printf("\n═══ Test 8: set_text with empty ids ═══\n");

    if (!model_exists()) {
        printf("  [SKIP] Model weights not found — skipping empty ids test\n");
        return;
    }

    void *engine = sonata_storm_create(WEIGHTS_PATH, CONFIG_PATH);
    if (!engine) {
        printf("  [SKIP] Failed to create engine — skipping\n");
        return;
    }

    unsigned int ids[] = {0};
    int rc = sonata_storm_set_text(engine, ids, 0);
    CHECKF(rc == 0, "set_text(engine, ids, 0) = %d (expected 0, empty input ok)", rc);

    sonata_storm_destroy(engine);
}

/* ─── Test 9: set_params with NULL engine ────────────────────────────────── */

static void test_storm_set_params_null(void) {
    printf("\n═══ Test 9: set_params with NULL engine ═══\n");

    int rc = sonata_storm_set_params(NULL, 0.8f, 8);
    CHECKF(rc == -1, "set_params(NULL, 0.8, 8) = %d (expected -1)", rc);

    rc = sonata_storm_set_params(NULL, 0.0f, 1);
    CHECKF(rc == -1, "set_params(NULL, 0.0, 1) = %d (expected -1)", rc);

    rc = sonata_storm_set_params(NULL, 2.0f, 16);
    CHECKF(rc == -1, "set_params(NULL, 2.0, 16) = %d (expected -1)", rc);
}

/* ─── Test 10: set_params ranges ─────────────────────────────────────────── */

static void test_storm_set_params_ranges(void) {
    printf("\n═══ Test 10: set_params with various ranges ═══\n");

    if (!model_exists()) {
        printf("  [SKIP] Model weights not found — skipping param range tests\n");
        return;
    }

    void *engine = sonata_storm_create(WEIGHTS_PATH, CONFIG_PATH);
    if (!engine) {
        printf("  [SKIP] Failed to create engine — skipping\n");
        return;
    }

    int rc;

    rc = sonata_storm_set_params(engine, 0.0f, 8);
    CHECKF(rc == 0, "set_params(temp=0.0 argmax, rounds=8) = %d (expected 0)", rc);

    rc = sonata_storm_set_params(engine, 0.5f, 4);
    CHECKF(rc == 0, "set_params(temp=0.5, rounds=4) = %d (expected 0)", rc);

    rc = sonata_storm_set_params(engine, 1.0f, 1);
    CHECKF(rc == 0, "set_params(temp=1.0, rounds=1) = %d (expected 0)", rc);

    rc = sonata_storm_set_params(engine, 2.0f, 16);
    CHECKF(rc == 0, "set_params(temp=2.0, rounds=16) = %d (expected 0)", rc);

    sonata_storm_destroy(engine);
}

/* ─── Test 11: generate with NULL engine ─────────────────────────────────── */

static void test_storm_generate_null(void) {
    printf("\n═══ Test 11: generate with NULL engine ═══\n");

    int out_tokens[128];
    int out_count = 0;
    int rc = sonata_storm_generate(NULL, out_tokens, 128, &out_count);
    CHECKF(rc == -1, "generate(NULL, buf, 128, &count) = %d (expected -1)", rc);
}

/* ─── Test 12: generate with NULL output buffer ──────────────────────────── */

static void test_storm_generate_null_output(void) {
    printf("\n═══ Test 12: generate with NULL output ═══\n");

    int rc = sonata_storm_generate(NULL, NULL, 0, NULL);
    CHECKF(rc == -1, "generate(NULL, NULL, 0, NULL) = %d (expected -1)", rc);

    int out_count = 0;
    rc = sonata_storm_generate(NULL, NULL, 64, &out_count);
    CHECKF(rc == -1, "generate(NULL, NULL, 64, &count) = %d (expected -1)", rc);

    int out_tokens[64];
    rc = sonata_storm_generate(NULL, out_tokens, 64, NULL);
    CHECKF(rc == -1, "generate(NULL, buf, 64, NULL) = %d (expected -1)", rc);
}

/* ─── Test 13: reset with NULL engine ────────────────────────────────────── */

static void test_storm_reset_null(void) {
    printf("\n═══ Test 13: reset with NULL engine ═══\n");

    int rc = sonata_storm_reset(NULL);
    CHECKF(rc == -1, "reset(NULL) = %d (expected -1)", rc);
}

/* ─── Test 14: Generate without valid model ──────────────────────────────── */

static void test_storm_generate_no_model(void) {
    printf("\n═══ Test 14: Generate without valid model ═══\n");

    void *engine = sonata_storm_create(
        "/tmp/fake_storm_weights.safetensors",
        "/tmp/fake_storm_config.json"
    );
    CHECK(engine == NULL, "create with fake paths returns NULL");

    if (engine) {
        int out_tokens[64];
        int out_count = 0;
        int rc = sonata_storm_generate(engine, out_tokens, 64, &out_count);
        CHECKF(rc == -1, "generate on bad engine = %d (expected -1)", rc);
        sonata_storm_destroy(engine);
    }
}

/* ─── Test 15: Full lifecycle ────────────────────────────────────────────── */

static void test_storm_lifecycle(void) {
    printf("\n═══ Test 15: Full lifecycle ═══\n");

    if (!model_exists()) {
        printf("  [SKIP] Model weights not found at:\n");
        printf("         %s\n", WEIGHTS_PATH);
        printf("         %s\n", CONFIG_PATH);
        return;
    }

    /* Create */
    void *engine = sonata_storm_create(WEIGHTS_PATH, CONFIG_PATH);
    CHECK(engine != NULL, "create engine with valid weights");
    if (!engine) return;

    /* Set params */
    int rc = sonata_storm_set_params(engine, 0.8f, 8);
    CHECKF(rc == 0, "set_params(temp=0.8, rounds=8) = %d (expected 0)", rc);

    /* Set text */
    unsigned int text_ids[] = {10, 20, 30, 40, 50, 60, 70, 80};
    rc = sonata_storm_set_text(engine, text_ids, 8);
    CHECKF(rc == 0, "set_text(8 tokens) = %d (expected 0)", rc);

    /* Generate */
    int max_tokens = 512;
    int *out_tokens = (int *)malloc(max_tokens * sizeof(int));
    int out_count = 0;

    rc = sonata_storm_generate(engine, out_tokens, max_tokens, &out_count);
    CHECKF(rc == 0, "generate() = %d (expected 0)", rc);
    CHECKF(out_count > 0, "generated %d tokens (expected > 0)", out_count);
    CHECKF(out_count <= max_tokens, "token count %d <= max %d", out_count, max_tokens);

    if (out_count > 0) {
        printf("    Generated %d semantic tokens via SoundStorm\n", out_count);
        printf("    Audio duration = %.2f s (at 50 Hz frame rate)\n", out_count / 50.0);
        printf("    First 5 tokens:");
        for (int i = 0; i < 5 && i < out_count; i++) {
            printf(" %d", out_tokens[i]);
        }
        printf("\n");
    }

    /* Reset */
    rc = sonata_storm_reset(engine);
    CHECKF(rc == 0, "reset() = %d (expected 0)", rc);

    /* Generate again after reset (should work) */
    rc = sonata_storm_set_text(engine, text_ids, 4);
    CHECKF(rc == 0, "set_text after reset = %d (expected 0)", rc);

    int out_count2 = 0;
    rc = sonata_storm_generate(engine, out_tokens, max_tokens, &out_count2);
    CHECKF(rc == 0, "generate after reset = %d (expected 0)", rc);
    CHECKF(out_count2 > 0, "generated %d tokens after reset (expected > 0)", out_count2);

    /* Destroy */
    sonata_storm_destroy(engine);
    free(out_tokens);
    CHECK(1, "full lifecycle completed without crash");
}

/* ─── Test 16: Create with swapped paths ─────────────────────────────────── */

static void test_storm_create_swapped_paths(void) {
    printf("\n═══ Test 16: Create with swapped paths ═══\n");

    /* Passing config as weights and weights as config should fail gracefully */
    void *engine = sonata_storm_create(CONFIG_PATH, WEIGHTS_PATH);
    CHECK(engine == NULL, "create with swapped (config, weights) returns NULL");
    if (engine) sonata_storm_destroy(engine);
}

/* ─── Test 17: Create with very long paths ───────────────────────────────── */

static void test_storm_create_long_paths(void) {
    printf("\n═══ Test 17: Create with very long paths ═══\n");

    /* Build a 4096-char path */
    char long_path[4097];
    memset(long_path, 'a', 4096);
    long_path[4096] = '\0';

    void *engine = sonata_storm_create(long_path, long_path);
    CHECK(engine == NULL, "create with 4096-char paths returns NULL");
    if (engine) sonata_storm_destroy(engine);

    /* Path with only slashes */
    engine = sonata_storm_create("///////////", "///////////");
    CHECK(engine == NULL, "create with slash-only paths returns NULL");
    if (engine) sonata_storm_destroy(engine);
}

/* ─── Test 18: Create with special characters ────────────────────────────── */

static void test_storm_create_special_chars(void) {
    printf("\n═══ Test 18: Create with special character paths ═══\n");

    void *engine = sonata_storm_create("weights\twith\ttabs.bin", "config\nwith\nnewlines.json");
    CHECK(engine == NULL, "create with tab/newline paths returns NULL");
    if (engine) sonata_storm_destroy(engine);

    engine = sonata_storm_create("path with spaces/weights.bin", "path with spaces/config.json");
    CHECK(engine == NULL, "create with spaces in path returns NULL");
    if (engine) sonata_storm_destroy(engine);

    engine = sonata_storm_create("\x01\x02\x03", "\x04\x05\x06");
    CHECK(engine == NULL, "create with control character paths returns NULL");
    if (engine) sonata_storm_destroy(engine);
}

/* ─── Test 19: set_params extreme temperature ────────────────────────────── */

static void test_storm_set_params_extreme_temp(void) {
    printf("\n═══ Test 19: set_params with extreme temperature values ═══\n");

    if (!model_exists()) {
        printf("  [SKIP] Model not found — skipping extreme temp tests\n");
        return;
    }

    void *engine = sonata_storm_create(WEIGHTS_PATH, CONFIG_PATH);
    if (!engine) {
        printf("  [SKIP] Failed to create engine — skipping\n");
        return;
    }

    int rc;

    /* Negative temperature */
    rc = sonata_storm_set_params(engine, -1.0f, 8);
    CHECKF(rc == 0 || rc == -1, "set_params(temp=-1.0) = %d (handled gracefully)", rc);

    /* Very large temperature */
    rc = sonata_storm_set_params(engine, 100.0f, 8);
    CHECKF(rc == 0 || rc == -1, "set_params(temp=100.0) = %d (handled gracefully)", rc);

    /* Infinity */
    float inf = 1.0f / 0.0f;
    rc = sonata_storm_set_params(engine, inf, 8);
    CHECKF(rc == 0 || rc == -1, "set_params(temp=inf) = %d (handled gracefully)", rc);

    /* NaN */
    float nan_val = 0.0f / 0.0f;
    rc = sonata_storm_set_params(engine, nan_val, 8);
    CHECKF(rc == 0 || rc == -1, "set_params(temp=NaN) = %d (handled gracefully)", rc);

    sonata_storm_destroy(engine);
}

/* ─── Test 20: set_params extreme rounds ─────────────────────────────────── */

static void test_storm_set_params_extreme_rounds(void) {
    printf("\n═══ Test 20: set_params with extreme round values ═══\n");

    if (!model_exists()) {
        printf("  [SKIP] Model not found — skipping extreme rounds tests\n");
        return;
    }

    void *engine = sonata_storm_create(WEIGHTS_PATH, CONFIG_PATH);
    if (!engine) {
        printf("  [SKIP] Failed to create engine — skipping\n");
        return;
    }

    int rc;

    /* Zero rounds */
    rc = sonata_storm_set_params(engine, 0.8f, 0);
    CHECKF(rc == 0 || rc == -1, "set_params(rounds=0) = %d (handled gracefully)", rc);

    /* Negative rounds */
    rc = sonata_storm_set_params(engine, 0.8f, -1);
    CHECKF(rc == 0 || rc == -1, "set_params(rounds=-1) = %d (handled gracefully)", rc);

    /* Very large rounds */
    rc = sonata_storm_set_params(engine, 0.8f, 999999);
    CHECKF(rc == 0 || rc == -1, "set_params(rounds=999999) = %d (handled gracefully)", rc);

    /* INT_MAX */
    rc = sonata_storm_set_params(engine, 0.8f, 2147483647);
    CHECKF(rc == 0 || rc == -1, "set_params(rounds=INT_MAX) = %d (handled gracefully)", rc);

    sonata_storm_destroy(engine);
}

/* ─── Test 21: generate with zero max_tokens ─────────────────────────────── */

static void test_storm_generate_zero_max(void) {
    printf("\n═══ Test 21: generate with zero max_tokens ═══\n");

    if (!model_exists()) {
        printf("  [SKIP] Model not found — skipping zero max_tokens test\n");
        return;
    }

    void *engine = sonata_storm_create(WEIGHTS_PATH, CONFIG_PATH);
    if (!engine) {
        printf("  [SKIP] Failed to create engine — skipping\n");
        return;
    }

    unsigned int text_ids[] = {10, 20, 30};
    sonata_storm_set_text(engine, text_ids, 3);

    int out_tokens[1];
    int out_count = 0;
    int rc = sonata_storm_generate(engine, out_tokens, 0, &out_count);
    CHECKF(rc == 0 || rc == -1, "generate(max=0) = %d (handled gracefully)", rc);
    CHECKF(out_count == 0, "generate(max=0) count = %d (expected 0)", out_count);

    sonata_storm_destroy(engine);
}

/* ─── Test 22: generate with negative max_tokens ─────────────────────────── */

static void test_storm_generate_negative_max(void) {
    printf("\n═══ Test 22: generate with negative max_tokens ═══\n");

    if (!model_exists()) {
        printf("  [SKIP] Model not found — skipping negative max_tokens test\n");
        return;
    }

    void *engine = sonata_storm_create(WEIGHTS_PATH, CONFIG_PATH);
    if (!engine) {
        printf("  [SKIP] Failed to create engine — skipping\n");
        return;
    }

    unsigned int text_ids[] = {10, 20, 30};
    sonata_storm_set_text(engine, text_ids, 3);

    int out_tokens[64];
    int out_count = 0;
    int rc = sonata_storm_generate(engine, out_tokens, -1, &out_count);
    CHECKF(rc == 0 || rc == -1, "generate(max=-1) = %d (handled gracefully)", rc);

    sonata_storm_destroy(engine);
}

/* ─── Test 23: set_text with large token count ───────────────────────────── */

static void test_storm_set_text_large(void) {
    printf("\n═══ Test 23: set_text with large token array ═══\n");

    if (!model_exists()) {
        printf("  [SKIP] Model not found — skipping large text test\n");
        return;
    }

    void *engine = sonata_storm_create(WEIGHTS_PATH, CONFIG_PATH);
    if (!engine) {
        printf("  [SKIP] Failed to create engine — skipping\n");
        return;
    }

    /* 1000 token IDs */
    unsigned int *big_ids = (unsigned int *)malloc(1000 * sizeof(unsigned int));
    for (int i = 0; i < 1000; i++) big_ids[i] = (unsigned int)(i + 1);

    int rc = sonata_storm_set_text(engine, big_ids, 1000);
    CHECKF(rc == 0 || rc == -1, "set_text(1000 tokens) = %d (handled gracefully)", rc);

    free(big_ids);
    sonata_storm_destroy(engine);
}

/* ─── Test 24: set_text with max-value IDs ───────────────────────────────── */

static void test_storm_set_text_max_ids(void) {
    printf("\n═══ Test 24: set_text with max-value token IDs ═══\n");

    if (!model_exists()) {
        printf("  [SKIP] Model not found — skipping max-value IDs test\n");
        return;
    }

    void *engine = sonata_storm_create(WEIGHTS_PATH, CONFIG_PATH);
    if (!engine) {
        printf("  [SKIP] Failed to create engine — skipping\n");
        return;
    }

    unsigned int max_ids[] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
    int rc = sonata_storm_set_text(engine, max_ids, 3);
    CHECKF(rc == 0 || rc == -1, "set_text(UINT_MAX ids) = %d (handled gracefully)", rc);

    unsigned int zero_ids[] = {0, 0, 0};
    rc = sonata_storm_set_text(engine, zero_ids, 3);
    CHECKF(rc == 0 || rc == -1, "set_text(zero ids) = %d (handled gracefully)", rc);

    sonata_storm_destroy(engine);
}

/* ─── Test 25: set_text with negative count ──────────────────────────────── */

static void test_storm_set_text_negative_count(void) {
    printf("\n═══ Test 25: set_text with negative count ═══\n");

    if (!model_exists()) {
        printf("  [SKIP] Model not found — skipping negative count test\n");
        return;
    }

    void *engine = sonata_storm_create(WEIGHTS_PATH, CONFIG_PATH);
    if (!engine) {
        printf("  [SKIP] Failed to create engine — skipping\n");
        return;
    }

    unsigned int ids[] = {1, 2, 3};
    int rc = sonata_storm_set_text(engine, ids, -1);
    CHECKF(rc == 0 || rc == -1, "set_text(n=-1) = %d (handled gracefully)", rc);

    rc = sonata_storm_set_text(engine, ids, -999);
    CHECKF(rc == 0 || rc == -1, "set_text(n=-999) = %d (handled gracefully)", rc);

    sonata_storm_destroy(engine);
}

/* ─── Test 26: Memory lifecycle stress ───────────────────────────────────── */

static void test_storm_lifecycle_stress(void) {
    printf("\n═══ Test 26: Memory lifecycle stress (100 create/destroy) ═══\n");

    /* 100 rapid create/destroy cycles with invalid paths (no model needed) */
    int ok = 1;
    for (int i = 0; i < 100; i++) {
        void *engine = sonata_storm_create(
            "/nonexistent/storm_weights.safetensors",
            "/nonexistent/storm_config.json"
        );
        if (engine != NULL) {
            ok = 0;
            sonata_storm_destroy(engine);
        }
    }
    CHECK(ok, "100 create/destroy cycles with invalid paths — all returned NULL");

    /* 100 destroy(NULL) calls */
    for (int i = 0; i < 100; i++) {
        sonata_storm_destroy(NULL);
    }
    CHECK(1, "100 destroy(NULL) calls — no crash");

    /* Alternating NULL operations */
    for (int i = 0; i < 50; i++) {
        sonata_storm_set_text(NULL, NULL, 0);
        sonata_storm_set_params(NULL, 0.5f, 4);
        sonata_storm_reset(NULL);
        int out[1]; int cnt = 0;
        sonata_storm_generate(NULL, out, 1, &cnt);
    }
    CHECK(1, "50 rounds of all-NULL operations — no crash");
}

/* ─── Test 27: Double destroy safety ─────────────────────────────────────── */

static void test_storm_double_destroy(void) {
    printf("\n═══ Test 27: Double destroy safety ═══\n");

    /* We can only test double-NULL, since real engines can't be double-freed safely
       without model files. But the NULL path is the critical one. */
    sonata_storm_destroy(NULL);
    sonata_storm_destroy(NULL);
    CHECK(1, "double destroy(NULL) is safe");
}

/* ─── Test 28: Reset without prior set_text ──────────────────────────────── */

static void test_storm_reset_without_set(void) {
    printf("\n═══ Test 28: Reset without prior set_text ═══\n");

    if (!model_exists()) {
        printf("  [SKIP] Model not found — skipping reset-without-set test\n");
        return;
    }

    void *engine = sonata_storm_create(WEIGHTS_PATH, CONFIG_PATH);
    if (!engine) {
        printf("  [SKIP] Failed to create engine — skipping\n");
        return;
    }

    /* Reset without ever calling set_text */
    int rc = sonata_storm_reset(engine);
    CHECKF(rc == 0, "reset without prior set_text = %d (expected 0)", rc);

    /* Multiple resets in a row */
    rc = sonata_storm_reset(engine);
    CHECKF(rc == 0, "second consecutive reset = %d (expected 0)", rc);

    rc = sonata_storm_reset(engine);
    CHECKF(rc == 0, "third consecutive reset = %d (expected 0)", rc);

    sonata_storm_destroy(engine);
}

/* ─── Test 29: Generate without set_text ─────────────────────────────────── */

static void test_storm_generate_without_set_text(void) {
    printf("\n═══ Test 29: Generate without set_text ═══\n");

    if (!model_exists()) {
        printf("  [SKIP] Model not found — skipping generate-without-set test\n");
        return;
    }

    void *engine = sonata_storm_create(WEIGHTS_PATH, CONFIG_PATH);
    if (!engine) {
        printf("  [SKIP] Failed to create engine — skipping\n");
        return;
    }

    /* Try to generate without calling set_text first */
    int out_tokens[64];
    int out_count = 0;
    int rc = sonata_storm_generate(engine, out_tokens, 64, &out_count);
    CHECKF(rc == 0 || rc == -1, "generate without set_text = %d (handled gracefully)", rc);

    sonata_storm_destroy(engine);
}

/* ─── Test 30: Constants are consistent ──────────────────────────────────── */

static void test_storm_constants_consistency(void) {
    printf("\n═══ Test 30: Constants consistency across calls ═══\n");

    int sr1 = sonata_storm_sample_rate();
    int sr2 = sonata_storm_sample_rate();
    int sr3 = sonata_storm_sample_rate();
    CHECK(sr1 == sr2 && sr2 == sr3, "sample_rate is consistent across 3 calls");

    int fr1 = sonata_storm_frame_rate();
    int fr2 = sonata_storm_frame_rate();
    int fr3 = sonata_storm_frame_rate();
    CHECK(fr1 == fr2 && fr2 == fr3, "frame_rate is consistent across 3 calls");

    /* Validate the mathematical relationship */
    CHECK(sr1 > 0 && fr1 > 0, "both rates are positive");
    int samples_per_frame = sr1 / fr1;
    CHECKF(samples_per_frame == 480, "samples_per_frame = %d (expected 480 = 24000/50)", samples_per_frame);
}

/* ─── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  SoundStorm Parallel Token Predictor — Test Suite    ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");

    test_storm_constants();
    test_storm_null_safety();
    test_storm_create_invalid_paths();
    test_storm_create_null_paths();
    test_storm_destroy_null();
    test_storm_set_text_null_engine();
    test_storm_set_text_null_ids();
    test_storm_set_text_empty();
    test_storm_set_params_null();
    test_storm_set_params_ranges();
    test_storm_generate_null();
    test_storm_generate_null_output();
    test_storm_reset_null();
    test_storm_generate_no_model();
    test_storm_lifecycle();

    /* New edge case tests (16-30) */
    test_storm_create_swapped_paths();
    test_storm_create_long_paths();
    test_storm_create_special_chars();
    test_storm_set_params_extreme_temp();
    test_storm_set_params_extreme_rounds();
    test_storm_generate_zero_max();
    test_storm_generate_negative_max();
    test_storm_set_text_large();
    test_storm_set_text_max_ids();
    test_storm_set_text_negative_count();
    test_storm_lifecycle_stress();
    test_storm_double_destroy();
    test_storm_reset_without_set();
    test_storm_generate_without_set_text();
    test_storm_constants_consistency();

    printf("\n═══════════════════════════════════════════\n");
    printf("Results: %d / %d passed\n", g_pass, g_pass + g_fail);
    if (g_fail > 0) {
        printf("FAILURES: %d\n", g_fail);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
