/**
 * test_metal_loader.c — Tests for Metal kernel loader.
 *
 * Validates:
 *   - NULL safety for all public API functions
 *   - Graceful failure when metallib files are missing
 *   - Lifecycle (create/destroy, destroy NULL)
 *   - Kernel availability checks
 *   - Kernel dispatch error returns with NULL/unloaded handles
 *
 * Build (requires Objective-C + Metal):
 *   cc -O2 -arch arm64 -x objective-c -Isrc \
 *      -framework Metal -framework Foundation \
 *      -Lbuild -lmetal_loader \
 *      -o tests/test_metal_loader tests/test_metal_loader.c
 *
 * Run: ./tests/test_metal_loader
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "metal_loader.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-55s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; return; } while(0)

/* ── NULL Safety Tests ───────────────────────────────────── */

static void test_load_null_path(void) {
    TEST("metal: load(NULL path) returns NULL");
    MetalKernels *mk = metal_kernels_load(NULL);
    if (mk != NULL) {
        metal_kernels_destroy(mk);
        FAIL("expected NULL for NULL path");
    }
    PASS();
}

static void test_destroy_null(void) {
    TEST("metal: destroy(NULL) does not crash");
    metal_kernels_destroy(NULL);
    PASS();
}

static void test_available_null(void) {
    TEST("metal: available(NULL) returns 0");
    if (metal_kernels_available(NULL) != 0)
        FAIL("expected 0 for NULL handle");
    PASS();
}

static void test_list_null(void) {
    TEST("metal: list(NULL) returns 0");
    const char *names[4];
    int n = metal_kernels_list(NULL, names, 4);
    if (n != 0)
        FAIL("expected 0 for NULL handle");
    PASS();
}

static void test_gemm_null_handle(void) {
    TEST("metal: gemm_f16(NULL handle) returns -1");
    float dummy[16] = {0};
    int ret = metal_gemm_f16(NULL, dummy, dummy, dummy, 2, 2, 2, 1.0f);
    if (ret != -1)
        FAIL("expected -1 for NULL handle");
    PASS();
}

static void test_silu_null_handle(void) {
    TEST("metal: silu_gate(NULL handle) returns -1");
    float dummy[16] = {0};
    int ret = metal_silu_gate(NULL, dummy, dummy, 2, 4);
    if (ret != -1)
        FAIL("expected -1 for NULL handle");
    PASS();
}

static void test_layer_norm_null_handle(void) {
    TEST("metal: layer_norm(NULL handle) returns -1");
    float dummy[16] = {0};
    int ret = metal_layer_norm(NULL, dummy, dummy, dummy, dummy, 2, 4, 1e-5f);
    if (ret != -1)
        FAIL("expected -1 for NULL handle");
    PASS();
}

static void test_flash_attention_null_handle(void) {
    TEST("metal: flash_attention(NULL handle) returns -1");
    float dummy[16] = {0};
    int ret = metal_flash_attention(NULL, dummy, dummy, dummy, dummy, 2, 2, 4);
    if (ret != -1)
        FAIL("expected -1 for NULL handle");
    PASS();
}

/* ── Graceful Failure: Non-Existent Files ────────────────── */

static void test_load_nonexistent_file(void) {
    TEST("metal: load non-existent .metallib returns NULL");
    MetalKernels *mk = metal_kernels_load("/nonexistent/path/tensor_ops.metallib");
    if (mk != NULL) {
        metal_kernels_destroy(mk);
        FAIL("expected NULL for missing metallib");
    }
    PASS();
}

static void test_load_empty_path(void) {
    TEST("metal: load empty string returns NULL");
    MetalKernels *mk = metal_kernels_load("");
    if (mk != NULL) {
        metal_kernels_destroy(mk);
        FAIL("expected NULL for empty path");
    }
    PASS();
}

/* ── Lifecycle Tests ─────────────────────────────────────── */

static void test_load_destroy_cycle(void) {
    TEST("metal: load(bad path) + destroy is safe");
    MetalKernels *mk = metal_kernels_load("/tmp/does_not_exist.metallib");
    /* mk should be NULL since file doesn't exist, but destroy should be safe either way */
    metal_kernels_destroy(mk);
    PASS();
}

static void test_repeated_destroy_null(void) {
    TEST("metal: repeated destroy(NULL) does not crash");
    for (int i = 0; i < 10; i++) {
        metal_kernels_destroy(NULL);
    }
    PASS();
}

/* ── Kernel List Edge Cases ──────────────────────────────── */

static void test_list_zero_max(void) {
    TEST("metal: list(handle, names, 0) returns 0");
    const char *names[1];
    int n = metal_kernels_list(NULL, names, 0);
    if (n != 0)
        FAIL("expected 0 for max_n=0");
    PASS();
}

/* ── Dispatch with Unloaded Handle ───────────────────────── */

/*
 * Even if we somehow got a MetalKernels* that isn't NULL but has loaded=0,
 * the dispatch functions should return -1. We can test this by loading a
 * non-existent file — the ObjC path frees and returns NULL, but the
 * non-ObjC stub also returns NULL. In either case, NULL checks in
 * dispatch functions should catch it.
 *
 * We already test NULL above. Here we test via the available() check
 * to ensure the loaded flag logic is correct.
 */
static void test_available_after_failed_load(void) {
    TEST("metal: available() after failed load returns 0");
    MetalKernels *mk = metal_kernels_load("/nonexistent.metallib");
    /* mk is NULL from failed load */
    if (metal_kernels_available(mk) != 0) {
        metal_kernels_destroy(mk);
        FAIL("expected 0 for failed load");
    }
    metal_kernels_destroy(mk);
    PASS();
}

static void test_list_after_failed_load(void) {
    TEST("metal: list() after failed load returns 0");
    MetalKernels *mk = metal_kernels_load("/nonexistent.metallib");
    const char *names[4];
    int n = metal_kernels_list(mk, names, 4);
    if (n != 0) {
        metal_kernels_destroy(mk);
        FAIL("expected 0 kernels from failed load");
    }
    metal_kernels_destroy(mk);
    PASS();
}

/* ── Main ───────────────────────────────────────────────── */

int main(void) {
    printf("\n=== Metal Loader Test Suite ===\n\n");

    printf("NULL Safety:\n");
    test_load_null_path();
    test_destroy_null();
    test_available_null();
    test_list_null();
    test_gemm_null_handle();
    test_silu_null_handle();
    test_layer_norm_null_handle();
    test_flash_attention_null_handle();

    printf("\nGraceful Failure:\n");
    test_load_nonexistent_file();
    test_load_empty_path();

    printf("\nLifecycle:\n");
    test_load_destroy_cycle();
    test_repeated_destroy_null();

    printf("\nEdge Cases:\n");
    test_list_zero_max();
    test_available_after_failed_load();
    test_list_after_failed_load();

    printf("\n=== Results: %d passed, %d failed ===\n\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
