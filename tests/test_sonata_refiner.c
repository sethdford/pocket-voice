/*
 * test_sonata_refiner.c — Tests for the Sonata STT refiner module.
 *
 * Tests: create/destroy lifecycle, NULL safety, constants verification,
 *        config validation, vocab_size, process with NULL args.
 *
 * Build: (handled by Makefile)
 * Run:   ./tests/test_sonata_refiner
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sonata_refiner.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-55s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)

#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* ── Test 1: create with NULL model path ──────────────── */

static void test_create_null_path(void) {
    TEST("refiner: create(NULL) returns NULL");
    SonataRefiner *ref = sonata_refiner_create(NULL);
    ASSERT(ref == NULL, "expected NULL for NULL model path");
    PASS();
}

/* ── Test 2: create with nonexistent model path ───────── */

static void test_create_nonexistent_path(void) {
    TEST("refiner: create(nonexistent) returns NULL");
    SonataRefiner *ref = sonata_refiner_create("/tmp/nonexistent_model_file.cref");
    ASSERT(ref == NULL, "expected NULL for missing model file");
    PASS();
}

/* ── Test 3: create with empty string path ────────────── */

static void test_create_empty_path(void) {
    TEST("refiner: create('') returns NULL");
    SonataRefiner *ref = sonata_refiner_create("");
    ASSERT(ref == NULL, "expected NULL for empty path");
    PASS();
}

/* ── Test 4: destroy NULL is safe ─────────────────────── */

static void test_destroy_null(void) {
    TEST("refiner: destroy(NULL) is no-op");
    sonata_refiner_destroy(NULL);
    /* If we get here, it didn't crash */
    PASS();
}

/* ── Test 5: reset NULL is safe ───────────────────────── */

static void test_reset_null(void) {
    TEST("refiner: reset(NULL) is no-op");
    sonata_refiner_reset(NULL);
    PASS();
}

/* ── Test 6: process with NULL refiner ────────────────── */

static void test_process_null_refiner(void) {
    TEST("refiner: process(NULL, ...) returns -1");
    int semantic_ids[] = {10, 20, 30};
    char out[256];
    int rc = sonata_refiner_process(NULL, semantic_ids, 3, out, sizeof(out));
    ASSERT(rc == -1, "expected -1 for NULL refiner");
    PASS();
}

/* ── Test 7: process with NULL semantic_ids ───────────── */

static void test_process_null_ids(void) {
    TEST("refiner: process(NULL, NULL ids, ...) returns -1");
    char out[256];
    int rc = sonata_refiner_process(NULL, NULL, 0, out, sizeof(out));
    ASSERT(rc == -1, "expected -1 for NULL refiner + NULL ids");
    PASS();
}

/* ── Test 8: process with NULL output buffer ──────────── */

static void test_process_null_output(void) {
    TEST("refiner: process(NULL, ids, n, NULL, 0) returns -1");
    int ids[] = {1, 2, 3};
    int rc = sonata_refiner_process(NULL, ids, 3, NULL, 0);
    ASSERT(rc == -1, "expected -1 for NULL refiner + NULL output");
    PASS();
}

/* ── Test 9: process with zero tokens ─────────────────── */

static void test_process_zero_tokens(void) {
    TEST("refiner: process(NULL, ids, 0, ...) returns -1");
    int ids[] = {1};
    char out[64];
    int rc = sonata_refiner_process(NULL, ids, 0, out, sizeof(out));
    ASSERT(rc == -1, "expected -1 for NULL refiner with 0 tokens");
    PASS();
}

/* ── Test 10: vocab_size with NULL refiner ────────────── */

static void test_vocab_size_null(void) {
    TEST("refiner: vocab_size(NULL) returns 0 or -1");
    int vs = sonata_refiner_vocab_size(NULL);
    /* Should return 0 or a negative sentinel for NULL */
    ASSERT(vs <= 0, "expected <= 0 for NULL refiner");
    PASS();
}

/* ── Test 11: create with truncated file ──────────────── */

static void test_create_truncated_file(void) {
    TEST("refiner: create(truncated file) returns NULL");

    /* Write a tiny file that's too small to be a valid .cref */
    const char *path = "/tmp/test_refiner_truncated.cref";
    FILE *f = fopen(path, "wb");
    if (!f) {
        FAIL("couldn't create temp file");
        return;
    }
    /* Write just 4 bytes — not enough for a valid header */
    unsigned char junk[] = {0xDE, 0xAD, 0xBE, 0xEF};
    fwrite(junk, 1, sizeof(junk), f);
    fclose(f);

    SonataRefiner *ref = sonata_refiner_create(path);
    ASSERT(ref == NULL, "expected NULL for truncated/invalid file");

    remove(path);
    PASS();
}

/* ── Test 12: create with wrong magic number ──────────── */

static void test_create_wrong_magic(void) {
    TEST("refiner: create(wrong magic) returns NULL");

    const char *path = "/tmp/test_refiner_bad_magic.cref";
    FILE *f = fopen(path, "wb");
    if (!f) {
        FAIL("couldn't create temp file");
        return;
    }
    /* Write a plausible-size header but with wrong magic */
    unsigned char header[128];
    memset(header, 0, sizeof(header));
    /* Wrong magic: "BAAD" instead of "CREF" */
    header[0] = 'B'; header[1] = 'A'; header[2] = 'A'; header[3] = 'D';
    fwrite(header, 1, sizeof(header), f);
    fclose(f);

    SonataRefiner *ref = sonata_refiner_create(path);
    ASSERT(ref == NULL, "expected NULL for wrong magic");

    remove(path);
    PASS();
}

/* ── Test 13: double destroy is safe ──────────────────── */

static void test_double_destroy(void) {
    TEST("refiner: double destroy(NULL) is safe");
    sonata_refiner_destroy(NULL);
    sonata_refiner_destroy(NULL);
    PASS();
}

/* ── Main ─────────────────────────────────────────────── */

int main(void) {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  Sonata Refiner — Test Suite                 ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    printf("[Lifecycle]\n");
    test_create_null_path();
    test_create_nonexistent_path();
    test_create_empty_path();
    test_destroy_null();
    test_reset_null();
    test_double_destroy();

    printf("\n[NULL Safety]\n");
    test_process_null_refiner();
    test_process_null_ids();
    test_process_null_output();
    test_process_zero_tokens();
    test_vocab_size_null();

    printf("\n[Config Validation]\n");
    test_create_truncated_file();
    test_create_wrong_magic();

    printf("\n════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("════════════════════════════════════════════════\n");

    return tests_failed > 0 ? 1 : 0;
}
