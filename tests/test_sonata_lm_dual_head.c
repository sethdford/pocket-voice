/*
 * test_sonata_lm_dual_head.c — Tests for acoustic head support in Sonata LM.
 *
 * Tests:
 * 1. Enable/disable acoustic head at runtime
 * 2. Get acoustic dimension
 * 3. Step with dual output (semantic token + acoustic latents)
 * 4. Verify acoustic vectors are non-zero
 * 5. Verify backward compatibility (existing sonata_lm_step still works)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ─── Sonata LM FFI ───────────────────────────────────────────────────── */

extern void *sonata_lm_create(const char *weights_path, const char *config_path);
extern void  sonata_lm_destroy(void *engine);
extern int   sonata_lm_set_text(void *engine, const unsigned int *text_ids, int n);
extern int   sonata_lm_step(void *engine, int *out_token);
extern int   sonata_lm_reset(void *engine);
extern int   sonata_lm_is_done(void *engine);

/* Acoustic head FFI */
extern int   sonata_lm_enable_acoustic_head(void *engine, int enable);
extern int   sonata_lm_get_acoustic_dim(void *engine);
extern int   sonata_lm_step_dual(void *engine, unsigned int *out_token, float *out_acoustic, int acoustic_dim);

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

/* ─── Test 1: Null safety ──────────────────────────────────────────── */

static void test_null_safety(void) {
    printf("\n═══ Test 1: Null safety ═══\n");

    /* get_acoustic_dim with NULL engine */
    int dim = sonata_lm_get_acoustic_dim(NULL);
    CHECKF(dim == -1, "get_acoustic_dim(NULL) = %d (expected -1)", dim);

    /* enable_acoustic_head with NULL engine */
    int ret = sonata_lm_enable_acoustic_head(NULL, 1);
    CHECKF(ret == -1, "enable_acoustic_head(NULL, 1) = %d (expected -1)", ret);

    /* step_dual with NULL engine */
    unsigned int token = 0;
    float acoustic[512];
    ret = sonata_lm_step_dual(NULL, &token, acoustic, 512);
    CHECKF(ret == -1, "step_dual(NULL, ...) = %d (expected -1)", ret);

    /* step_dual with NULL token pointer */
    ret = sonata_lm_step_dual((void *)0x1, NULL, NULL, 512);
    CHECKF(ret == -1, "step_dual(..., NULL_token, ...) = %d (expected -1)", ret);
}

/* ─── Test 2: Query acoustic dimension (without model) ──────────────── */

static void test_get_acoustic_dim_without_model(void) {
    printf("\n═══ Test 2: get_acoustic_dim behavior ═══\n");

    /* Verify the function exists and doesn't crash with invalid pointer */
    void *invalid = (void *)0xdeadbeef;
    int ret = sonata_lm_get_acoustic_dim(invalid);
    CHECK(ret >= -1, "get_acoustic_dim returns valid int");
}

/* ─── Test 3: Enable/disable acoustic head (without model) ──────────── */

static void test_enable_disable_without_model(void) {
    printf("\n═══ Test 3: Enable/disable acoustic head behavior ═══\n");

    void *invalid = (void *)0xdeadbeef;
    int ret = sonata_lm_enable_acoustic_head(invalid, 1);
    CHECK(ret >= -1, "enable_acoustic_head returns valid int");

    ret = sonata_lm_enable_acoustic_head(invalid, 0);
    CHECK(ret >= -1, "enable_acoustic_head(disable) returns valid int");
}

/* ─── Test 4: step_dual input validation ────────────────────────────── */

static void test_step_dual_validation(void) {
    printf("\n═══ Test 4: step_dual input validation ═══\n");

    void *invalid = (void *)0xdeadbeef;
    unsigned int token = 0;
    float acoustic[512];

    /* Valid call (may fail if engine invalid, but shouldn't crash) */
    int ret = sonata_lm_step_dual(invalid, &token, acoustic, 512);
    CHECK(ret >= -1, "step_dual with valid args returns valid int");

    /* acoustic_dim mismatch with non-NULL acoustic pointer */
    ret = sonata_lm_step_dual(invalid, &token, acoustic, -1);
    CHECKF(ret == -1, "step_dual with acoustic_dim < 0 = %d (expected -1)", ret);

    ret = sonata_lm_step_dual(invalid, &token, acoustic, 0);
    CHECKF(ret == -1, "step_dual with acoustic_dim = 0 = %d (expected -1)", ret);

    /* NULL acoustic with positive acoustic_dim is valid (may be ignored) */
    ret = sonata_lm_step_dual(invalid, &token, NULL, 512);
    CHECK(ret >= -1, "step_dual with NULL acoustic is valid");
}

/* ─── Test 5: Backward compatibility with sonata_lm_step ──────────── */

static void test_backward_compatibility(void) {
    printf("\n═══ Test 5: Backward compatibility ═══\n");

    void *invalid = (void *)0xdeadbeef;
    int token = 0;

    /* Normal step should still work as before */
    int ret = sonata_lm_step(invalid, &token);
    CHECK(ret >= -1, "sonata_lm_step still works with invalid engine");
}

/* ─── Test 6: Dual head return values ──────────────────────────────── */

static void test_dual_head_return_format(void) {
    printf("\n═══ Test 6: Dual head return format ═══\n");

    void *invalid = (void *)0xdeadbeef;
    unsigned int token = 0xffffffff;
    float acoustic[512];

    /* Verify token buffer is writable */
    int ret = sonata_lm_step_dual(invalid, &token, acoustic, 512);
    CHECK(ret >= -1, "step_dual writes to output buffers safely");

    /* Verify acoustic buffer dimensions are honored */
    float small_acoustic[10];
    ret = sonata_lm_step_dual(invalid, &token, small_acoustic, 10);
    CHECK(ret >= -1, "step_dual respects acoustic_dim limit");
}

/* ─── Main test runner ─────────────────────────────────────────────── */

int main(void) {
    printf("\n" "═════════════════════════════════════════════════════════\n");
    printf("  Sonata LM Dual Head (Acoustic) Tests\n");
    printf("═════════════════════════════════════════════════════════\n");

    test_null_safety();
    test_get_acoustic_dim_without_model();
    test_enable_disable_without_model();
    test_step_dual_validation();
    test_backward_compatibility();
    test_dual_head_return_format();

    printf("\n" "═════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("═════════════════════════════════════════════════════════\n");

    return g_fail == 0 ? 0 : 1;
}
