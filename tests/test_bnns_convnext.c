/**
 * test_bnns_convnext.c — Tests for BNNS ConvNeXt decoder.
 *
 * Validates:
 *   - NULL safety for all public API functions
 *   - Create/destroy lifecycle
 *   - Config variations (layers, dimensions, kernel sizes)
 *   - Weight loading graceful failure (missing files, NULL args)
 *   - Forward pass with NULL inputs and edge cases
 *
 * Build:
 *   cc -O2 -arch arm64 -Isrc -DACCELERATE_NEW_LAPACK \
 *      -framework Accelerate \
 *      -Lbuild -lbnns_convnext_decoder \
 *      -o tests/test_bnns_convnext tests/test_bnns_convnext.c
 *
 * Run: ./tests/test_bnns_convnext
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "bnns_convnext_decoder.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-55s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; return; } while(0)

/* Default test config matching Sonata TTS decoder */
#define TEST_LAYERS    2
#define TEST_DEC_DIM   64
#define TEST_KERNEL    7
#define TEST_FF_MULT   4.0f
#define TEST_INPUT_DIM 320   /* 64 FSQ + 256 acoustic */
#define TEST_NFFT      256

/* ── NULL Safety Tests ───────────────────────────────────── */

static void test_destroy_null(void) {
    TEST("bnns_convnext: destroy(NULL) does not crash");
    bnns_convnext_destroy(NULL);
    PASS();
}

static void test_load_weights_null_dec(void) {
    TEST("bnns_convnext: load_weights(NULL dec) returns -1");
    int ret = bnns_convnext_load_weights(NULL, "/tmp/weights.safetensors");
    if (ret != -1)
        FAIL("expected -1 for NULL decoder");
    PASS();
}

static void test_load_weights_null_path(void) {
    TEST("bnns_convnext: load_weights(dec, NULL path) returns -1");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        TEST_LAYERS, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("create failed");

    int ret = bnns_convnext_load_weights(dec, NULL);
    if (ret != -1) {
        bnns_convnext_destroy(dec);
        FAIL("expected -1 for NULL path");
    }

    bnns_convnext_destroy(dec);
    PASS();
}

static void test_load_mlmodelc_null_dec(void) {
    TEST("bnns_convnext: load_mlmodelc(NULL dec) returns -1");
    int ret = bnns_convnext_load_mlmodelc(NULL, "/tmp/model.mlmodelc");
    if (ret != -1)
        FAIL("expected -1 for NULL decoder");
    PASS();
}

static void test_load_mlmodelc_null_path(void) {
    TEST("bnns_convnext: load_mlmodelc(dec, NULL path) returns -1");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        TEST_LAYERS, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("create failed");

    int ret = bnns_convnext_load_mlmodelc(dec, NULL);
    if (ret != -1) {
        bnns_convnext_destroy(dec);
        FAIL("expected -1 for NULL path");
    }

    bnns_convnext_destroy(dec);
    PASS();
}

static void test_forward_null_dec(void) {
    TEST("bnns_convnext: forward(NULL dec) returns 0");
    float dummy_in[320] = {0};
    float dummy_out[129] = {0};
    int ret = bnns_convnext_forward(NULL, dummy_in, dummy_in, 1,
                                     dummy_out, dummy_out);
    if (ret != 0)
        FAIL("expected 0 for NULL decoder");
    PASS();
}

static void test_forward_null_semantic(void) {
    TEST("bnns_convnext: forward(NULL semantic) returns 0");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        TEST_LAYERS, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("create failed");

    float dummy[256] = {0};
    float mag[129] = {0};
    float phase[129] = {0};
    int ret = bnns_convnext_forward(dec, NULL, dummy, 1, mag, phase);
    if (ret != 0) {
        bnns_convnext_destroy(dec);
        FAIL("expected 0 for NULL semantic input");
    }

    bnns_convnext_destroy(dec);
    PASS();
}

static void test_forward_null_acoustic(void) {
    TEST("bnns_convnext: forward(NULL acoustic) returns 0");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        TEST_LAYERS, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("create failed");

    float dummy[64] = {0};
    float mag[129] = {0};
    float phase[129] = {0};
    int ret = bnns_convnext_forward(dec, dummy, NULL, 1, mag, phase);
    if (ret != 0) {
        bnns_convnext_destroy(dec);
        FAIL("expected 0 for NULL acoustic input");
    }

    bnns_convnext_destroy(dec);
    PASS();
}

/* ── Create/Destroy Lifecycle ────────────────────────────── */

static void test_create_destroy_basic(void) {
    TEST("bnns_convnext: basic create and destroy");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        TEST_LAYERS, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("create returned NULL");
    bnns_convnext_destroy(dec);
    PASS();
}

static void test_create_destroy_repeated(void) {
    TEST("bnns_convnext: repeated create/destroy cycles");
    for (int i = 0; i < 5; i++) {
        BNNSConvNeXtDecoder *dec = bnns_convnext_create(
            TEST_LAYERS, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
            TEST_INPUT_DIM, TEST_NFFT);
        if (!dec) FAIL("create failed on iteration");
        bnns_convnext_destroy(dec);
    }
    PASS();
}

/* ── Config Variations ───────────────────────────────────── */

static void test_create_single_layer(void) {
    TEST("bnns_convnext: create with 1 layer");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        1, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("expected non-NULL for 1 layer");
    bnns_convnext_destroy(dec);
    PASS();
}

static void test_create_many_layers(void) {
    TEST("bnns_convnext: create with 8 layers");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        8, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("expected non-NULL for 8 layers");
    bnns_convnext_destroy(dec);
    PASS();
}

static void test_create_large_dim(void) {
    TEST("bnns_convnext: create with dec_dim=512");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        2, 512, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, 1024);
    if (!dec) FAIL("expected non-NULL for large dim");
    bnns_convnext_destroy(dec);
    PASS();
}

static void test_create_small_kernel(void) {
    TEST("bnns_convnext: create with conv_kernel=3");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        TEST_LAYERS, TEST_DEC_DIM, 3, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("expected non-NULL for kernel=3");
    bnns_convnext_destroy(dec);
    PASS();
}

/* ── Weight Loading: Graceful Failure ────────────────────── */

static void test_load_weights_nonexistent(void) {
    TEST("bnns_convnext: load_weights non-existent file returns -1");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        TEST_LAYERS, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("create failed");

    int ret = bnns_convnext_load_weights(dec, "/nonexistent/weights.safetensors");
    if (ret != -1) {
        bnns_convnext_destroy(dec);
        FAIL("expected -1 for missing weights file");
    }

    bnns_convnext_destroy(dec);
    PASS();
}

static void test_load_mlmodelc_nonexistent(void) {
    TEST("bnns_convnext: load_mlmodelc non-existent path returns -1");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        TEST_LAYERS, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("create failed");

    int ret = bnns_convnext_load_mlmodelc(dec, "/nonexistent/model.mlmodelc");
    if (ret != -1) {
        bnns_convnext_destroy(dec);
        FAIL("expected -1 for non-existent mlmodelc path");
    }

    bnns_convnext_destroy(dec);
    PASS();
}

/* ── Forward Pass Edge Cases ─────────────────────────────── */

static void test_forward_zero_frames(void) {
    TEST("bnns_convnext: forward with n_frames=0 returns 0");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        TEST_LAYERS, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("create failed");

    float dummy_in[320] = {0};
    float mag[129] = {0};
    float phase[129] = {0};
    int ret = bnns_convnext_forward(dec, dummy_in, dummy_in, 0, mag, phase);
    if (ret != 0) {
        bnns_convnext_destroy(dec);
        FAIL("expected 0 for n_frames=0");
    }

    bnns_convnext_destroy(dec);
    PASS();
}

static void test_forward_negative_frames(void) {
    TEST("bnns_convnext: forward with n_frames=-1 returns 0");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        TEST_LAYERS, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("create failed");

    float dummy_in[320] = {0};
    float mag[129] = {0};
    float phase[129] = {0};
    int ret = bnns_convnext_forward(dec, dummy_in, dummy_in, -1, mag, phase);
    if (ret != 0) {
        bnns_convnext_destroy(dec);
        FAIL("expected 0 for negative n_frames");
    }

    bnns_convnext_destroy(dec);
    PASS();
}

static void test_forward_single_frame(void) {
    TEST("bnns_convnext: forward single frame returns n_bins");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        TEST_LAYERS, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("create failed");

    int n_bins = TEST_NFFT / 2 + 1;  /* 129 */
    int n_frames = 1;

    /* Allocate properly sized inputs */
    int fsq_dim = TEST_INPUT_DIM - 256;
    float *semantic = (float *)calloc(n_frames * fsq_dim, sizeof(float));
    float *acoustic = (float *)calloc(n_frames * 256, sizeof(float));
    float *mag = (float *)calloc(n_frames * n_bins, sizeof(float));
    float *phase = (float *)calloc(n_frames * n_bins, sizeof(float));

    if (!semantic || !acoustic || !mag || !phase) {
        free(semantic); free(acoustic); free(mag); free(phase);
        bnns_convnext_destroy(dec);
        FAIL("allocation failed");
    }

    int ret = bnns_convnext_forward(dec, semantic, acoustic, n_frames, mag, phase);
    if (ret != n_bins) {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected %d bins, got %d", n_bins, ret);
        free(semantic); free(acoustic); free(mag); free(phase);
        bnns_convnext_destroy(dec);
        FAIL(msg);
    }

    free(semantic);
    free(acoustic);
    free(mag);
    free(phase);
    bnns_convnext_destroy(dec);
    PASS();
}

static void test_forward_output_finite(void) {
    TEST("bnns_convnext: forward output values are finite");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        TEST_LAYERS, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("create failed");

    int n_bins = TEST_NFFT / 2 + 1;
    int n_frames = 2;

    int fsq_dim = TEST_INPUT_DIM - 256;
    float *semantic = (float *)calloc(n_frames * fsq_dim, sizeof(float));
    float *acoustic = (float *)calloc(n_frames * 256, sizeof(float));
    float *mag = (float *)calloc(n_frames * n_bins, sizeof(float));
    float *phase = (float *)calloc(n_frames * n_bins, sizeof(float));

    if (!semantic || !acoustic || !mag || !phase) {
        free(semantic); free(acoustic); free(mag); free(phase);
        bnns_convnext_destroy(dec);
        FAIL("allocation failed");
    }

    int ret = bnns_convnext_forward(dec, semantic, acoustic, n_frames, mag, phase);
    if (ret != n_bins) {
        free(semantic); free(acoustic); free(mag); free(phase);
        bnns_convnext_destroy(dec);
        FAIL("forward failed");
    }

    /* Check all outputs are finite (not NaN or Inf) */
    int all_finite = 1;
    for (int i = 0; i < n_frames * n_bins; i++) {
        if (!isfinite(mag[i]) || !isfinite(phase[i])) {
            all_finite = 0;
            break;
        }
    }
    if (!all_finite) {
        free(semantic); free(acoustic); free(mag); free(phase);
        bnns_convnext_destroy(dec);
        FAIL("output contains NaN or Inf");
    }

    free(semantic);
    free(acoustic);
    free(mag);
    free(phase);
    bnns_convnext_destroy(dec);
    PASS();
}

static void test_forward_magnitude_positive(void) {
    TEST("bnns_convnext: magnitude output is positive (exp)");
    BNNSConvNeXtDecoder *dec = bnns_convnext_create(
        TEST_LAYERS, TEST_DEC_DIM, TEST_KERNEL, TEST_FF_MULT,
        TEST_INPUT_DIM, TEST_NFFT);
    if (!dec) FAIL("create failed");

    int n_bins = TEST_NFFT / 2 + 1;
    int n_frames = 1;

    int fsq_dim = TEST_INPUT_DIM - 256;
    float *semantic = (float *)calloc(n_frames * fsq_dim, sizeof(float));
    float *acoustic = (float *)calloc(n_frames * 256, sizeof(float));
    float *mag = (float *)calloc(n_frames * n_bins, sizeof(float));
    float *phase = (float *)calloc(n_frames * n_bins, sizeof(float));

    if (!semantic || !acoustic || !mag || !phase) {
        free(semantic); free(acoustic); free(mag); free(phase);
        bnns_convnext_destroy(dec);
        FAIL("allocation failed");
    }

    int ret = bnns_convnext_forward(dec, semantic, acoustic, n_frames, mag, phase);
    if (ret != n_bins) {
        free(semantic); free(acoustic); free(mag); free(phase);
        bnns_convnext_destroy(dec);
        FAIL("forward failed");
    }

    /* Magnitude goes through exp(), so all values should be > 0 */
    int all_positive = 1;
    for (int i = 0; i < n_frames * n_bins; i++) {
        if (mag[i] <= 0.0f) {
            all_positive = 0;
            break;
        }
    }
    if (!all_positive) {
        free(semantic); free(acoustic); free(mag); free(phase);
        bnns_convnext_destroy(dec);
        FAIL("magnitude has non-positive values (exp should be > 0)");
    }

    free(semantic);
    free(acoustic);
    free(mag);
    free(phase);
    bnns_convnext_destroy(dec);
    PASS();
}

/* ── Main ───────────────────────────────────────────────── */

int main(void) {
    printf("\n=== BNNS ConvNeXt Decoder Test Suite ===\n\n");

    printf("NULL Safety:\n");
    test_destroy_null();
    test_load_weights_null_dec();
    test_load_weights_null_path();
    test_load_mlmodelc_null_dec();
    test_load_mlmodelc_null_path();
    test_forward_null_dec();
    test_forward_null_semantic();
    test_forward_null_acoustic();

    printf("\nLifecycle:\n");
    test_create_destroy_basic();
    test_create_destroy_repeated();

    printf("\nConfig Variations:\n");
    test_create_single_layer();
    test_create_many_layers();
    test_create_large_dim();
    test_create_small_kernel();

    printf("\nWeight Loading:\n");
    test_load_weights_nonexistent();
    test_load_mlmodelc_nonexistent();

    printf("\nForward Pass:\n");
    test_forward_zero_frames();
    test_forward_negative_frames();
    test_forward_single_frame();
    test_forward_output_finite();
    test_forward_magnitude_positive();

    printf("\n=== Results: %d passed, %d failed ===\n\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
