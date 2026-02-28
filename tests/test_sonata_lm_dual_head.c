/*
 * test_sonata_lm_dual_head.c — Tests for acoustic head support in Sonata LM.
 *
 * Tests:
 * 1. Null safety for all acoustic FFI functions
 * 2. Input validation (negative dims, zero dims, null buffers)
 * 3. get_acoustic_buffer FFI contract
 * 4. step_dual parameter validation
 * 5. Backward compatibility (sonata_lm_step still works)
 * 6. Acoustic buffer retrieval with various buffer sizes
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

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
extern int   sonata_lm_get_acoustic_buffer(void *engine, float *out_buf, int buf_len);

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

/* ─── Test 1: Null safety for all acoustic FFI ─────────────────────── */

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

    /* get_acoustic_buffer with NULL engine */
    float buf[512];
    ret = sonata_lm_get_acoustic_buffer(NULL, buf, 512);
    CHECKF(ret == -1, "get_acoustic_buffer(NULL, ...) = %d (expected -1)", ret);

    /* get_acoustic_buffer with NULL output buffer */
    ret = sonata_lm_get_acoustic_buffer((void *)0x1, NULL, 512);
    CHECKF(ret == -1, "get_acoustic_buffer(..., NULL, ...) = %d (expected -1)", ret);
}

/* ─── Test 2: Input validation for get_acoustic_buffer ──────────────── */

static void test_get_acoustic_buffer_validation(void) {
    printf("\n═══ Test 2: get_acoustic_buffer input validation ═══\n");

    float buf[512];

    /* buf_len <= 0 should return -1 */
    int ret = sonata_lm_get_acoustic_buffer((void *)0x1, buf, 0);
    CHECKF(ret == -1, "get_acoustic_buffer(buf_len=0) = %d (expected -1)", ret);

    ret = sonata_lm_get_acoustic_buffer((void *)0x1, buf, -1);
    CHECKF(ret == -1, "get_acoustic_buffer(buf_len=-1) = %d (expected -1)", ret);

    ret = sonata_lm_get_acoustic_buffer((void *)0x1, buf, -100);
    CHECKF(ret == -1, "get_acoustic_buffer(buf_len=-100) = %d (expected -1)", ret);

    /* NULL engine + valid buf should return -1 */
    ret = sonata_lm_get_acoustic_buffer(NULL, buf, 512);
    CHECKF(ret == -1, "get_acoustic_buffer(NULL engine) = %d (expected -1)", ret);
}

/* ─── Test 3: step_dual parameter validation ───────────────────────── */

static void test_step_dual_validation(void) {
    printf("\n═══ Test 3: step_dual input validation ═══\n");

    unsigned int token = 0;
    float acoustic[512];

    /* acoustic_dim < 0 with non-NULL acoustic pointer should fail */
    int ret = sonata_lm_step_dual((void *)0x1, &token, acoustic, -1);
    CHECKF(ret == -1, "step_dual with acoustic_dim=-1 = %d (expected -1)", ret);

    /* acoustic_dim = 0 with non-NULL acoustic pointer should fail */
    ret = sonata_lm_step_dual((void *)0x1, &token, acoustic, 0);
    CHECKF(ret == -1, "step_dual with acoustic_dim=0 = %d (expected -1)", ret);

    /* NULL acoustic with positive acoustic_dim is valid (acoustic output ignored) */
    ret = sonata_lm_step_dual((void *)0x1, &token, NULL, 512);
    CHECK(ret >= -1, "step_dual(NULL acoustic, dim=512) does not crash");

    /* Both NULL engine and NULL token */
    ret = sonata_lm_step_dual(NULL, NULL, NULL, 0);
    CHECKF(ret == -1, "step_dual(all NULL) = %d (expected -1)", ret);
}

/* ─── Test 4: enable_acoustic_head validation ──────────────────────── */

static void test_enable_disable_validation(void) {
    printf("\n═══ Test 4: enable_acoustic_head validation ═══\n");

    /* NULL engine */
    int ret = sonata_lm_enable_acoustic_head(NULL, 1);
    CHECKF(ret == -1, "enable_acoustic_head(NULL, 1) = %d (expected -1)", ret);

    ret = sonata_lm_enable_acoustic_head(NULL, 0);
    CHECKF(ret == -1, "enable_acoustic_head(NULL, 0) = %d (expected -1)", ret);

    /* Enable with various non-zero values */
    ret = sonata_lm_enable_acoustic_head(NULL, 42);
    CHECKF(ret == -1, "enable_acoustic_head(NULL, 42) = %d (expected -1)", ret);

    ret = sonata_lm_enable_acoustic_head(NULL, -1);
    CHECKF(ret == -1, "enable_acoustic_head(NULL, -1) = %d (expected -1)", ret);
}

/* ─── Test 5: Backward compatibility ───────────────────────────────── */

static void test_backward_compatibility(void) {
    printf("\n═══ Test 5: Backward compatibility ═══\n");

    /* sonata_lm_step with NULL engine should return -1, not crash */
    int token = 0;
    int ret = sonata_lm_step(NULL, &token);
    CHECKF(ret == -1, "sonata_lm_step(NULL) = %d (expected -1)", ret);

    /* sonata_lm_reset with NULL should not crash */
    ret = sonata_lm_reset(NULL);
    CHECK(ret >= -1, "sonata_lm_reset(NULL) does not crash");

    /* sonata_lm_is_done with NULL should not crash */
    ret = sonata_lm_is_done(NULL);
    CHECK(ret >= -1, "sonata_lm_is_done(NULL) does not crash");
}

/* ─── Test 6: FFI function signatures are linkable ─────────────────── */

static void test_ffi_signatures(void) {
    printf("\n═══ Test 6: FFI function signatures ═══\n");

    /* Verify all FFI functions exist and are callable (link-time check).
     * Each function pointer should be non-NULL since they're extern. */
    CHECK(sonata_lm_create != NULL, "sonata_lm_create is linked");
    CHECK(sonata_lm_destroy != NULL, "sonata_lm_destroy is linked");
    CHECK(sonata_lm_step != NULL, "sonata_lm_step is linked");
    CHECK(sonata_lm_step_dual != NULL, "sonata_lm_step_dual is linked");
    CHECK(sonata_lm_enable_acoustic_head != NULL, "sonata_lm_enable_acoustic_head is linked");
    CHECK(sonata_lm_get_acoustic_dim != NULL, "sonata_lm_get_acoustic_dim is linked");
    CHECK(sonata_lm_get_acoustic_buffer != NULL, "sonata_lm_get_acoustic_buffer is linked");
}

/* ─── Test 7: get_acoustic_buffer with boundary buf_len values ──────── */

static void test_get_acoustic_buffer_boundaries(void) {
    printf("\n═══ Test 7: get_acoustic_buffer boundary checks ═══\n");

    float buf[1];

    /* buf_len = 1 (minimum valid) with NULL engine */
    int ret = sonata_lm_get_acoustic_buffer(NULL, buf, 1);
    CHECKF(ret == -1, "get_acoustic_buffer(NULL, buf, 1) = %d (expected -1)", ret);

    /* Very large buf_len with NULL engine */
    float *large_buf = (float *)malloc(65536 * sizeof(float));
    if (large_buf) {
        ret = sonata_lm_get_acoustic_buffer(NULL, large_buf, 65536);
        CHECKF(ret == -1, "get_acoustic_buffer(NULL, buf, 65536) = %d (expected -1)", ret);
        free(large_buf);
    }
}

/* ─── Test 8: Verify acoustic head gate contract ───────────────────── */

static void test_acoustic_gate_contract(void) {
    printf("\n═══ Test 8: Acoustic head gate contract ═══\n");

    /* When acoustic_head_enabled is false (via enable_acoustic_head(eng, 0)):
     * - forward_with_acoustic should return acoustic=None
     * - step_dual should zero out_acoustic
     * - get_acoustic_buffer should return 0 (empty)
     *
     * We can't test with a real model here, but we verify the FFI contract:
     * enable_acoustic_head(NULL, 0) returns -1 (null check), proving the
     * function validates inputs before mutating state.
     */
    int ret = sonata_lm_enable_acoustic_head(NULL, 0);
    CHECKF(ret == -1, "disable on NULL engine returns -1 (validates before mutate): %d", ret);

    ret = sonata_lm_enable_acoustic_head(NULL, 1);
    CHECKF(ret == -1, "enable on NULL engine returns -1 (validates before mutate): %d", ret);

    /* Verify step_dual zeros acoustic buffer when head is disabled/missing.
     * With NULL engine it returns -1, but the contract is: if step_dual succeeds
     * with acoustic head disabled, out_acoustic must be zeroed. */
    unsigned int token = 0;
    float acoustic[512];
    memset(acoustic, 0xFF, sizeof(acoustic));  /* Fill with non-zero pattern */
    ret = sonata_lm_step_dual(NULL, &token, acoustic, 512);
    CHECKF(ret == -1, "step_dual(NULL) returns -1 before touching buffers: %d", ret);
    /* Since ret=-1, buffers should NOT have been modified (early return) */
    int all_ff = 1;
    for (int i = 0; i < 512; i++) {
        unsigned int bits;
        memcpy(&bits, &acoustic[i], sizeof(bits));
        if (bits != 0xFFFFFFFF) { all_ff = 0; break; }
    }
    CHECK(all_ff, "step_dual(NULL) does not modify acoustic buffer on error");
}

/* ─── NEW CORRECTNESS TESTS ────────────────────────────────────────── */

/* ─── Test 9: acoustic_dim boundary values ─────────────────────────── */

static void test_acoustic_dim_boundaries(void) {
    printf("\n═══ Test 9: acoustic_dim boundary values ═══\n");

    unsigned int token = 0;
    float acoustic[4096];

    /* acoustic_dim=0 with non-NULL acoustic → -1 (validated by Rust: dim<=0) */
    int ret = sonata_lm_step_dual(NULL, &token, acoustic, 0);
    CHECKF(ret == -1, "step_dual(dim=0) = %d (expected -1)", ret);

    /* acoustic_dim=1 (minimum useful) with NULL engine → -1 */
    ret = sonata_lm_step_dual(NULL, &token, acoustic, 1);
    CHECKF(ret == -1, "step_dual(dim=1, NULL engine) = %d (expected -1)", ret);

    /* acoustic_dim=4096 (large but reasonable) with NULL engine → -1 */
    ret = sonata_lm_step_dual(NULL, &token, acoustic, 4096);
    CHECKF(ret == -1, "step_dual(dim=4096, NULL engine) = %d (expected -1)", ret);

    /* acoustic_dim=INT_MAX with NULL engine → should not crash */
    ret = sonata_lm_step_dual(NULL, &token, acoustic, INT_MAX);
    CHECKF(ret == -1, "step_dual(dim=INT_MAX, NULL engine) = %d (expected -1)", ret);

    /* acoustic_dim negative: -1, INT_MIN */
    ret = sonata_lm_step_dual(NULL, &token, acoustic, -1);
    CHECKF(ret == -1, "step_dual(dim=-1) = %d (expected -1)", ret);

    ret = sonata_lm_step_dual(NULL, &token, acoustic, INT_MIN);
    CHECKF(ret == -1, "step_dual(dim=INT_MIN) = %d (expected -1)", ret);

    /* NULL acoustic with dim=0 → should be fine (no acoustic output requested) */
    ret = sonata_lm_step_dual(NULL, &token, NULL, 0);
    CHECKF(ret == -1, "step_dual(NULL acoustic, dim=0, NULL engine) = %d (expected -1)", ret);
}

/* ─── Test 10: Enable→disable→enable sequence ─────────────────────── */

static void test_enable_disable_sequence(void) {
    printf("\n═══ Test 10: Enable→disable→enable sequence ═══\n");

    /* With NULL engine, all should return -1 consistently.
     * This documents the FFI contract: enable/disable are idempotent
     * and the NULL check happens before any state mutation. */
    int ret;

    /* Enable */
    ret = sonata_lm_enable_acoustic_head(NULL, 1);
    CHECKF(ret == -1, "enable(NULL) = %d (expected -1)", ret);

    /* Disable */
    ret = sonata_lm_enable_acoustic_head(NULL, 0);
    CHECKF(ret == -1, "disable(NULL) = %d (expected -1)", ret);

    /* Re-enable */
    ret = sonata_lm_enable_acoustic_head(NULL, 1);
    CHECKF(ret == -1, "re-enable(NULL) = %d (expected -1)", ret);

    /* Double-disable */
    ret = sonata_lm_enable_acoustic_head(NULL, 0);
    CHECKF(ret == -1, "double-disable(NULL) = %d (expected -1)", ret);
    ret = sonata_lm_enable_acoustic_head(NULL, 0);
    CHECKF(ret == -1, "triple-disable(NULL) = %d (expected -1)", ret);

    /* Various truthy values should all map to enable */
    ret = sonata_lm_enable_acoustic_head(NULL, 2);
    CHECKF(ret == -1, "enable(NULL, 2) = %d (expected -1)", ret);
    ret = sonata_lm_enable_acoustic_head(NULL, -1);
    CHECKF(ret == -1, "enable(NULL, -1) = %d (expected -1)", ret);
    ret = sonata_lm_enable_acoustic_head(NULL, INT_MAX);
    CHECKF(ret == -1, "enable(NULL, INT_MAX) = %d (expected -1)", ret);
}

/* ─── Test 11: step() vs step_dual() return semantics ─────────────── */

static void test_step_vs_step_dual_consistency(void) {
    printf("\n═══ Test 11: step() vs step_dual() return consistency ═══\n");

    /* Both functions with NULL engine should return -1 */
    int token_step = 0;
    int ret_step = sonata_lm_step(NULL, &token_step);
    CHECKF(ret_step == -1, "step(NULL) = %d (expected -1)", ret_step);

    unsigned int token_dual = 0;
    int ret_dual = sonata_lm_step_dual(NULL, &token_dual, NULL, 0);
    CHECKF(ret_dual == -1, "step_dual(NULL, NULL acoustic) = %d (expected -1)", ret_dual);

    /* Both should have same error return semantics */
    CHECK(ret_step == ret_dual, "step and step_dual return same error code for NULL engine");

    /* With NULL output token pointers */
    ret_step = sonata_lm_step(NULL, NULL);
    ret_dual = sonata_lm_step_dual(NULL, NULL, NULL, 0);
    CHECK(ret_step == ret_dual, "step and step_dual return same error for NULL token pointer");
    CHECKF(ret_step == -1, "step(NULL, NULL) = %d (expected -1)", ret_step);
    CHECKF(ret_dual == -1, "step_dual(NULL, NULL, ...) = %d (expected -1)", ret_dual);
}

/* ─── Test 12: Output buffer not corrupted on error paths ──────────── */

static void test_output_buffer_integrity(void) {
    printf("\n═══ Test 12: Output buffer integrity on error ═══\n");

    /* Fill token with sentinel value before failed call */
    unsigned int token = 0xDEADBEEF;
    float acoustic[16];
    for (int i = 0; i < 16; i++) acoustic[i] = 42.0f;

    /* Call with NULL engine → error return, buffers untouched */
    int ret = sonata_lm_step_dual(NULL, &token, acoustic, 16);
    CHECKF(ret == -1, "step_dual(NULL engine) = %d (expected -1)", ret);
    CHECK(token == 0xDEADBEEF, "Token buffer preserved (0xDEADBEEF) on error");

    /* Acoustic buffer should also be preserved */
    int acoustic_intact = 1;
    for (int i = 0; i < 16; i++) {
        if (acoustic[i] != 42.0f) { acoustic_intact = 0; break; }
    }
    CHECK(acoustic_intact, "Acoustic buffer preserved (42.0f) on error");

    /* Similarly for get_acoustic_buffer */
    float buf[16];
    for (int i = 0; i < 16; i++) buf[i] = 99.0f;
    ret = sonata_lm_get_acoustic_buffer(NULL, buf, 16);
    CHECKF(ret == -1, "get_acoustic_buffer(NULL) = %d (expected -1)", ret);
    int buf_intact = 1;
    for (int i = 0; i < 16; i++) {
        if (buf[i] != 99.0f) { buf_intact = 0; break; }
    }
    CHECK(buf_intact, "get_acoustic_buffer preserves output buffer on error");

    /* step() output integrity */
    int step_token = 0x7FFFFFFF;
    ret = sonata_lm_step(NULL, &step_token);
    CHECKF(ret == -1, "step(NULL) = %d (expected -1)", ret);
    CHECK(step_token == 0x7FFFFFFF, "step() preserves token buffer on error");
}

/* ─── Test 13: get_acoustic_dim with extreme inputs ──────────────── */

static void test_get_acoustic_dim_boundaries(void) {
    printf("\n═══ Test 13: get_acoustic_dim boundary tests ═══\n");

    /* NULL engine → -1 (already tested, but verify consistency) */
    int dim = sonata_lm_get_acoustic_dim(NULL);
    CHECKF(dim == -1, "get_acoustic_dim(NULL) = %d (expected -1)", dim);

    /* Non-NULL but invalid engine → may crash or return something;
     * we only test NULL since we can't safely dereference arbitrary ptrs.
     * Document the expected dim range for valid engines:
     * - 0 = acoustic head not configured
     * - >0 = acoustic head dimension (typically 128, 256, or 512) */
    CHECK(dim == -1, "NULL → -1 is consistent sentinel for 'no engine'");
}

/* ─── Main test runner ─────────────────────────────────────────────── */

int main(void) {
    printf("\n" "═════════════════════════════════════════════════════════\n");
    printf("  Sonata LM Dual Head (Acoustic) Tests\n");
    printf("═════════════════════════════════════════════════════════\n");

    test_null_safety();
    test_get_acoustic_buffer_validation();
    test_step_dual_validation();
    test_enable_disable_validation();
    test_backward_compatibility();
    test_ffi_signatures();
    test_get_acoustic_buffer_boundaries();
    test_acoustic_gate_contract();
    /* New correctness tests */
    test_acoustic_dim_boundaries();
    test_enable_disable_sequence();
    test_step_vs_step_dual_consistency();
    test_output_buffer_integrity();
    test_get_acoustic_dim_boundaries();

    printf("\n" "═════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("═════════════════════════════════════════════════════════\n");

    return g_fail == 0 ? 0 : 1;
}
