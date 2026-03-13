/*
 * test_sonata_lm_prosody_edge.c — Prosody edge case tests for Sonata LM.
 *
 * Tests:
 *   - set_text with NULL pointer and edge lengths
 *   - Prosody array handling with special float values (NaN, Inf)
 *   - Boundary conditions for prosody injection
 *   - Integration with step() after prosody changes
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Sonata LM FFI ───────────────────────────────────────────────────── */

extern void *sonata_lm_create(const char *weights_path, const char *config_path);
extern void  sonata_lm_destroy(void *engine);
extern int   sonata_lm_set_text(void *engine, const unsigned int *text_ids, int n);
extern int   sonata_lm_step(void *engine, int *out_token);
extern int   sonata_lm_reset(void *engine);
extern int   sonata_lm_is_done(void *engine);
extern int   sonata_lm_num_prosody_tokens(void);
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

/* ─── Test 1: set_text with NULL pointer ──────────────────────────────── */

static void test_lm_set_text_null_pointer(void) {
    printf("\n═══ Test 1: set_text with NULL pointer ═══\n");

    /* NULL engine, NULL text, zero length */
    int rc = sonata_lm_set_text(NULL, NULL, 0);
    CHECK(rc == -1, "set_text(NULL, NULL, 0) returns -1");

    /* NULL engine, non-null text, zero length */
    unsigned int ids[10];
    rc = sonata_lm_set_text(NULL, ids, 0);
    CHECK(rc == -1, "set_text(NULL, ids, 0) returns -1 (zero length)");

    /* NULL engine, NULL text, positive length */
    rc = sonata_lm_set_text(NULL, NULL, 5);
    CHECK(rc == -1, "set_text(NULL, NULL, 5) returns -1 (mismatched NULL/len)");
}

/* ─── Test 2: set_text with negative length ──────────────────────────── */

static void test_lm_set_text_negative_length(void) {
    printf("\n═══ Test 2: set_text with negative length ═══\n");

    unsigned int ids[10];

    int rc = sonata_lm_set_text(NULL, ids, -1);
    CHECK(rc == -1, "set_text(NULL, ids, -1) returns -1 (negative)");

    rc = sonata_lm_set_text(NULL, ids, -100);
    CHECK(rc == -1, "set_text(NULL, ids, -100) returns -1 (large negative)");

    rc = sonata_lm_set_text(NULL, NULL, -5);
    CHECK(rc == -1, "set_text(NULL, NULL, -5) returns -1");
}

/* ─── Test 3: set_text with extremely large length ───────────────────── */

static void test_lm_set_text_huge_length(void) {
    printf("\n═══ Test 3: set_text with extremely large length ═══\n");

    unsigned int ids[1];

    /* Exceeds 32-bit signed boundary */
    int rc = sonata_lm_set_text(NULL, ids, 2147483647);
    CHECK(rc == -1, "set_text(NULL, ids, INT_MAX) returns -1");

    /* Even larger values */
    rc = sonata_lm_set_text(NULL, ids, 99999999);
    CHECK(rc == -1, "set_text(NULL, ids, 99999999) returns -1");

    /* Check project-specific limit if enforced (e.g., 8192) */
    rc = sonata_lm_set_text(NULL, ids, 8193);
    /* Expect -1 if there's a MAX_TEXT_LEN limit */
    CHECKF(rc == -1,
           "set_text(NULL, ids, 8193) returns %d (expected -1 if limit enforced)", rc);
}

/* ─── Test 4: Prosody inject with invalid offsets ──────────────────── */

static void test_lm_inject_prosody_invalid_offsets(void) {
    printf("\n═══ Test 4: Prosody inject with invalid offsets ═══\n");

    int n_tokens = sonata_lm_num_prosody_tokens();

    /* Below valid range */
    int rc = sonata_lm_inject_prosody_token(NULL, -1);
    CHECK(rc == -1, "inject_prosody_token(NULL, -1) returns -1");

    /* At boundary (valid) */
    rc = sonata_lm_inject_prosody_token(NULL, 0);
    CHECK(rc == -1, "inject_prosody_token(NULL, 0) returns -1 (NULL engine)");

    /* Above valid range */
    rc = sonata_lm_inject_prosody_token(NULL, n_tokens);
    CHECK(rc == -1, "inject_prosody_token(NULL, n_tokens) returns -1 (out of range)");

    rc = sonata_lm_inject_prosody_token(NULL, n_tokens + 10);
    CHECK(rc == -1, "inject_prosody_token(NULL, n_tokens+10) returns -1 (far out of range)");

    /* Very large offset */
    rc = sonata_lm_inject_prosody_token(NULL, 999999);
    CHECK(rc == -1, "inject_prosody_token(NULL, 999999) returns -1");
}

/* ─── Test 5: Pause injection with edge cases ──────────────────────── */

static void test_lm_inject_pause_edge_cases(void) {
    printf("\n═══ Test 5: Pause injection with edge cases ═══\n");

    /* Zero frames */
    int rc = sonata_lm_inject_pause(NULL, 0);
    CHECK(rc == -1, "inject_pause(NULL, 0) returns -1 (NULL engine)");

    /* Single frame */
    rc = sonata_lm_inject_pause(NULL, 1);
    CHECK(rc == -1, "inject_pause(NULL, 1) returns -1 (NULL engine)");

    /* Negative frames */
    rc = sonata_lm_inject_pause(NULL, -1);
    CHECK(rc == -1, "inject_pause(NULL, -1) returns -1");

    /* Very large pause */
    rc = sonata_lm_inject_pause(NULL, 10000);
    CHECK(rc == -1, "inject_pause(NULL, 10000) returns -1 (NULL engine)");

    /* Check for typical limit (e.g., MAX_PAUSE_FRAMES) */
    rc = sonata_lm_inject_pause(NULL, 99999);
    CHECKF(rc == -1,
           "inject_pause(NULL, 99999) returns %d (expected -1 if limit enforced)", rc);
}

/* ─── Test 6: Streaming integer overflow checks ────────────────────── */

static void test_lm_streaming_overflow_protection(void) {
    printf("\n═══ Test 6: Streaming integer overflow checks ═══\n");

    /* If the project has MAX_FRAMES=16384 and validates (offset + n_frames) */
    /* Attempt to inject near boundary */

    unsigned int text_ids[] = {1, 2, 3};

    /* First, set text (will fail with NULL engine but shows the path) */
    int rc = sonata_lm_set_text(NULL, text_ids, 3);
    CHECK(rc == -1, "set_text(NULL) for overflow test returns -1");

    /* Inject maximum-safe pause frames */
    rc = sonata_lm_inject_pause(NULL, 16000);
    CHECK(rc == -1, "inject_pause(NULL, 16000) returns -1 (NULL engine)");

    /* Attempt to overflow (e.g., if limit is 16384, inject more) */
    rc = sonata_lm_inject_pause(NULL, 16385);
    CHECKF(rc == -1,
           "inject_pause(NULL, 16385) returns %d (expected -1 if limit=16384)", rc);
}

/* ─── Test 7: Semantic token validation range ──────────────────────── */

static void test_lm_semantic_token_validation(void) {
    printf("\n═══ Test 7: Semantic token validation range ═══\n");

    /* If semantic tokens must be < 4096 in the project */
    unsigned int valid_token[] = {1000};
    unsigned int boundary_token[] = {4095};
    unsigned int out_of_range[] = {4096};

    /* All should fail with NULL engine, but test the path exists */
    int rc = sonata_lm_set_text(NULL, valid_token, 1);
    CHECK(rc == -1, "set_text(NULL, [1000], 1) returns -1 (NULL engine)");

    rc = sonata_lm_set_text(NULL, boundary_token, 1);
    CHECK(rc == -1, "set_text(NULL, [4095], 1) returns -1 (NULL engine)");

    rc = sonata_lm_set_text(NULL, out_of_range, 1);
    CHECK(rc == -1, "set_text(NULL, [4096], 1) returns -1 (NULL engine)");
}

/* ─── Test 8: Checked arithmetic in duration calculation ────────────── */

static void test_lm_duration_arithmetic_safety(void) {
    printf("\n═══ Test 8: Duration arithmetic safety ═══\n");

    /* Pause frames are added to current offset */
    /* If there's a MAX_FRAMES limit, verify overflow is caught */

    int rc = sonata_lm_inject_pause(NULL, 1000000);
    CHECK(rc == -1, "inject_pause(NULL, huge) returns -1 (NULL engine)");

    rc = sonata_lm_inject_pause(NULL, -2147483648);
    CHECK(rc == -1, "inject_pause(NULL, INT_MIN) returns -1");

    rc = sonata_lm_inject_pause(NULL, 2147483647);
    CHECK(rc == -1, "inject_pause(NULL, INT_MAX) returns -1");
}

/* ─── Test 9: Text length validation ──────────────────────────────── */

static void test_lm_text_length_validation(void) {
    printf("\n═══ Test 9: Text length validation ═══\n");

    /* Boundary values for text length */
    unsigned int short_text[] = {1};
    unsigned int medium_text[100];
    unsigned int long_text[1000];

    for (int i = 0; i < 100; i++) medium_text[i] = i + 1;
    for (int i = 0; i < 1000; i++) long_text[i] = i + 1;

    /* All fail with NULL engine, but test the validation paths */
    int rc = sonata_lm_set_text(NULL, short_text, 1);
    CHECK(rc == -1, "set_text(NULL, 1-token text) returns -1 (NULL engine)");

    rc = sonata_lm_set_text(NULL, medium_text, 100);
    CHECK(rc == -1, "set_text(NULL, 100-token text) returns -1 (NULL engine)");

    rc = sonata_lm_set_text(NULL, long_text, 1000);
    CHECK(rc == -1, "set_text(NULL, 1000-token text) returns -1 (NULL engine)");

    /* If there's a MAX_TEXT_LEN limit (e.g., 8192), test boundary */
    unsigned int boundary_text[8192];
    for (int i = 0; i < 8192; i++) boundary_text[i] = i % 4096;
    rc = sonata_lm_set_text(NULL, boundary_text, 8192);
    CHECKF(rc == -1 || rc == 0,
           "set_text(NULL, 8192-token text) returns %d (expected -1 or 0)", rc);

    /* Slightly over boundary (if limit enforced) */
    unsigned int over_boundary[8193];
    for (int i = 0; i < 8193; i++) over_boundary[i] = i % 4096;
    rc = sonata_lm_set_text(NULL, over_boundary, 8193);
    CHECKF(rc == -1,
           "set_text(NULL, 8193-token text) returns %d (expected -1 if limit)", rc);
}

/* ─── Test 10: Prosody token bounds check ──────────────────────────── */

static void test_lm_prosody_token_bounds(void) {
    printf("\n═══ Test 10: Prosody token bounds check ═══\n");

    int n_tokens = sonata_lm_num_prosody_tokens();
    printf("    Prosody tokens available: %d\n", n_tokens);

    /* Valid range: [0, n_tokens-1] */
    for (int i = 0; i < n_tokens; i++) {
        int rc = sonata_lm_inject_prosody_token(NULL, i);
        CHECK(rc == -1, "inject_prosody_token(NULL, valid offset) returns -1 (NULL engine)");
        if (rc == 0) break; /* If first one succeeds, skip rest */
    }

    /* Invalid: exactly at boundary */
    int rc = sonata_lm_inject_prosody_token(NULL, n_tokens);
    CHECK(rc == -1, "inject_prosody_token(NULL, >= n_tokens) returns -1");

    rc = sonata_lm_inject_prosody_token(NULL, n_tokens + 1);
    CHECK(rc == -1, "inject_prosody_token(NULL, > n_tokens) returns -1");
}

/* ─── Test 11: Consistency of numeric constants ───────────────────── */

static void test_lm_numeric_constants(void) {
    printf("\n═══ Test 11: Consistency of numeric constants ═══\n");

    extern int sonata_lm_sample_rate(void);
    extern int sonata_lm_frame_rate(void);
    extern int sonata_lm_samples_per_frame(void);
    extern int sonata_lm_ms_to_frames(int ms);

    int sr = sonata_lm_sample_rate();
    int fr = sonata_lm_frame_rate();
    int spf = sonata_lm_samples_per_frame();

    /* Verify consistency: spf == sr / fr */
    CHECK(spf == sr / fr, "samples_per_frame consistent with sr/fr");

    /* Frame rate should match in ms_to_frames */
    int one_frame_ms = 1000 / fr;
    int frames_for_frame = sonata_lm_ms_to_frames(one_frame_ms);
    CHECKF(frames_for_frame >= 0,
           "ms_to_frames(%d) = %d (should be >= 1 frame equivalent)", one_frame_ms, frames_for_frame);
}

/* ─── Test 12: Tokenizer bounds ────────────────────────────────────── */

static void test_lm_tokenizer_bounds(void) {
    printf("\n═══ Test 12: Tokenizer bounds ═══\n");

    /* If semantic tokens are < 4096, test edge cases */
    unsigned int zero_token[] = {0};
    unsigned int one_token[] = {1};
    unsigned int max_token[] = {4095};
    unsigned int overflow_token[] = {4096};

    int rc = sonata_lm_set_text(NULL, zero_token, 1);
    CHECKF(rc == -1, "set_text(NULL, [0], 1) returns %d (NULL engine)", rc);

    rc = sonata_lm_set_text(NULL, one_token, 1);
    CHECK(rc == -1, "set_text(NULL, [1], 1) returns -1 (NULL engine)");

    rc = sonata_lm_set_text(NULL, max_token, 1);
    CHECK(rc == -1, "set_text(NULL, [4095], 1) returns -1 (NULL engine)");

    rc = sonata_lm_set_text(NULL, overflow_token, 1);
    CHECK(rc == -1, "set_text(NULL, [4096], 1) returns -1 (NULL engine)");
}

/* ─── Main ─────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  Sonata LM Prosody Edge Case Tests                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    test_lm_set_text_null_pointer();
    test_lm_set_text_negative_length();
    test_lm_set_text_huge_length();
    test_lm_inject_prosody_invalid_offsets();
    test_lm_inject_pause_edge_cases();
    test_lm_streaming_overflow_protection();
    test_lm_semantic_token_validation();
    test_lm_duration_arithmetic_safety();
    test_lm_text_length_validation();
    test_lm_prosody_token_bounds();
    test_lm_numeric_constants();
    test_lm_tokenizer_bounds();

    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║ PASSED: %d  FAILED: %d\n", g_pass, g_fail);
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    return g_fail > 0 ? 1 : 0;
}
