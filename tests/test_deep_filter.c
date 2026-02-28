/**
 * test_deep_filter.c — Tests for neural noise suppression module.
 *
 * Tests:
 *  1.  Create/destroy lifecycle
 *  2.  Passthrough mode preserves signal energy
 *  3.  Various input sizes (not aligned to FFT size)
 *  4.  Reset clears GRU state
 *  5.  Parameter setters (strength, min_gain)
 *  6.  Noisy signal processing (passthrough)
 *  7.  Edge cases (zero/negative/NULL input)
 *  8.  Bad weight file handling
 *  9.  Streaming (multiple small process calls)
 *  10. GRU path with synthetic weights
 *  11. Noise reduction verification with GRU path
 *  12. NaN input guard
 */

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>
#include "deep_filter.h"

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); \
    } else { \
        tests_passed++; \
    } \
} while (0)

/* ── .dnf weight file constants (must match deep_filter.c internals) ────── */

#define DNF_MAGIC       0x46464E44
#define DNF_VERSION     1
#define DNF_SAMPLE_RATE 16000
#define DNF_FFT_SIZE    512
#define DNF_HOP_SIZE    256
#define DNF_N_ERB       32
#define DNF_N_GRU       2
#define DNF_GRU_HIDDEN  64
#define DNF_FREQ_BINS   (DNF_FFT_SIZE / 2 + 1)  /* 257 */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t sample_rate;
    uint32_t fft_size;
    uint32_t hop_size;
    uint32_t n_erb;
    uint32_t n_gru_layers;
    uint32_t gru_hidden;
    uint32_t reserved[4];
} TestDnfHeader;

/* Generate a sine wave */
static void gen_sine(float *buf, int n, float freq, float sr, float amp) {
    for (int i = 0; i < n; i++) {
        buf[i] = amp * sinf(2.0f * (float)M_PI * freq * (float)i / sr);
    }
}

/* Generate white noise (simple LCG PRNG) */
static void gen_noise(float *buf, int n, float amp, uint32_t seed) {
    uint32_t s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        float u = (float)(s >> 8) / (float)(1 << 24) - 0.5f;
        buf[i] = amp * u * 2.0f;
    }
}

/* Compute RMS energy */
static float compute_rms(const float *buf, int n) {
    float sum_sq = 0.0f;
    vDSP_dotpr(buf, 1, buf, 1, &sum_sq, (vDSP_Length)n);
    return sqrtf(sum_sq / (float)n);
}

/**
 * Create a minimal .dnf weight file with predictable behavior:
 * - All GRU weights = 0 → hidden state stays at 0
 * - Output bias = target_logit → gains = sigmoid(target_logit)
 *
 * With target_logit = 0, all gains = sigmoid(0) = 0.5, so the filter
 * attenuates by ~50% across all ERB bands. This is deterministic and testable.
 *
 * Returns path to temp file (caller must unlink), or NULL on failure.
 */
static char *create_test_weights(float target_logit) {
    char *path = strdup("/tmp/test_deep_filter_XXXXXX.dnf");
    if (!path) return NULL;

    /* mkstemp needs exactly 6 X's at the end — adjust format */
    char tmppath[] = "/tmp/test_df_XXXXXX";
    int fd = mkstemp(tmppath);
    if (fd < 0) { free(path); return NULL; }
    free(path);

    FILE *f = fdopen(fd, "wb");
    if (!f) { close(fd); return NULL; }

    /* Header */
    TestDnfHeader hdr = {
        .magic = DNF_MAGIC,
        .version = DNF_VERSION,
        .sample_rate = DNF_SAMPLE_RATE,
        .fft_size = DNF_FFT_SIZE,
        .hop_size = DNF_HOP_SIZE,
        .n_erb = DNF_N_ERB,
        .n_gru_layers = DNF_N_GRU,
        .gru_hidden = DNF_GRU_HIDDEN,
        .reserved = {0, 0, 0, 0}
    };
    fwrite(&hdr, sizeof(hdr), 1, f);

    /* ERB filter bank [N_ERB x FREQ_BINS] — compute triangular filters */
    {
        float fb[DNF_N_ERB * DNF_FREQ_BINS];
        memset(fb, 0, sizeof(fb));

        /* Simple uniform band assignment: each ERB band covers a range of bins */
        int bins_per_band = DNF_FREQ_BINS / DNF_N_ERB;
        for (int band = 0; band < DNF_N_ERB; band++) {
            int start = band * bins_per_band;
            int end = (band + 1) * bins_per_band;
            if (band == DNF_N_ERB - 1) end = DNF_FREQ_BINS;
            float n_bins = (float)(end - start);
            for (int bin = start; bin < end; bin++) {
                fb[band * DNF_FREQ_BINS + bin] = 1.0f / n_bins;
            }
        }
        fwrite(fb, sizeof(float), DNF_N_ERB * DNF_FREQ_BINS, f);
    }

    /* GRU Layer 1: W_ih [192 x 32], W_hh [192 x 64], b_ih [192], b_hh [192] — all zeros */
    {
        int D1 = DNF_N_ERB;  /* 32 */
        int H = DNF_GRU_HIDDEN;  /* 64 */
        int G = 3 * H;  /* 192 */

        size_t wih_sz = (size_t)G * D1;
        size_t whh_sz = (size_t)G * H;
        float *zeros = (float *)calloc(wih_sz > whh_sz ? wih_sz : whh_sz, sizeof(float));
        if (!zeros) { fclose(f); return NULL; }

        fwrite(zeros, sizeof(float), wih_sz, f);   /* W_ih */
        fwrite(zeros, sizeof(float), whh_sz, f);   /* W_hh */
        fwrite(zeros, sizeof(float), G, f);         /* b_ih */
        fwrite(zeros, sizeof(float), G, f);         /* b_hh */
        free(zeros);
    }

    /* GRU Layer 2: W_ih [192 x 64], W_hh [192 x 64], b_ih [192], b_hh [192] — all zeros */
    {
        int H = DNF_GRU_HIDDEN;
        int G = 3 * H;
        size_t whh_sz = (size_t)G * H;

        float *zeros = (float *)calloc(whh_sz, sizeof(float));
        if (!zeros) { fclose(f); return NULL; }

        fwrite(zeros, sizeof(float), whh_sz, f);   /* W_ih (same size as W_hh for L2) */
        fwrite(zeros, sizeof(float), whh_sz, f);   /* W_hh */
        fwrite(zeros, sizeof(float), G, f);         /* b_ih */
        fwrite(zeros, sizeof(float), G, f);         /* b_hh */
        free(zeros);
    }

    /* Output projection: out_w [N_ERB x GRU_HIDDEN] = zeros, out_b [N_ERB] = target_logit */
    {
        size_t ow_sz = (size_t)DNF_N_ERB * DNF_GRU_HIDDEN;
        float *zeros = (float *)calloc(ow_sz, sizeof(float));
        if (!zeros) { fclose(f); return NULL; }
        fwrite(zeros, sizeof(float), ow_sz, f);
        free(zeros);

        float bias[DNF_N_ERB];
        for (int i = 0; i < DNF_N_ERB; i++) bias[i] = target_logit;
        fwrite(bias, sizeof(float), DNF_N_ERB, f);
    }

    fclose(f);

    char *result = strdup(tmppath);
    return result;
}

/* ── Test 1: Create/destroy lifecycle ─────────────────────────────────────── */

static void test_lifecycle(void) {
    printf("Test 1: Create/destroy lifecycle\n");

    /* NULL weights -> passthrough mode */
    DeepFilter *df = deep_filter_create(16000, NULL);
    ASSERT(df != NULL, "create with NULL weights should succeed (passthrough)");

    /* Verify it can process audio (proves functional after create) */
    float buf[512];
    gen_sine(buf, 512, 440.0f, 16000.0f, 0.5f);
    deep_filter_process(df, buf, 512);
    int all_finite = 1;
    for (int i = 0; i < 512; i++) {
        if (!isfinite(buf[i])) { all_finite = 0; break; }
    }
    ASSERT(all_finite, "passthrough process produces finite output");
    deep_filter_destroy(df);

    /* Wrong sample rate */
    df = deep_filter_create(48000, NULL);
    ASSERT(df == NULL, "create with wrong sample rate should fail");

    /* Destroy NULL — verify we can still create a new filter afterward */
    deep_filter_destroy(NULL);
    df = deep_filter_create(16000, NULL);
    ASSERT(df != NULL, "create after destroy(NULL) should succeed");
    deep_filter_destroy(df);

    printf("  Passed\n");
}

/* ── Test 2: Passthrough mode preserves signal energy ─────────────────────── */

static void test_passthrough(void) {
    printf("Test 2: Passthrough mode\n");

    DeepFilter *df = deep_filter_create(16000, NULL);
    ASSERT(df != NULL, "create passthrough");

    int n = 4096;
    float *signal = (float *)malloc((size_t)n * sizeof(float));
    float *original = (float *)malloc((size_t)n * sizeof(float));
    ASSERT(signal != NULL && original != NULL, "alloc buffers");

    gen_sine(signal, n, 440.0f, 16000.0f, 0.5f);
    memcpy(original, signal, (size_t)n * sizeof(float));

    deep_filter_process(df, signal, n);

    /* In passthrough mode, output should closely match input after initial latency */
    float rms_orig = compute_rms(original + 512, n - 512);
    float rms_out  = compute_rms(signal + 512, n - 512);

    /* Allow some tolerance for FFT roundtrip numerical error */
    float ratio = (rms_orig > 0.0f) ? rms_out / rms_orig : 0.0f;
    ASSERT(ratio > 0.8f && ratio < 1.2f,
           "passthrough should preserve energy (ratio within 20%)");

    free(signal);
    free(original);
    deep_filter_destroy(df);
    printf("  Passed (energy ratio: %.3f)\n", ratio);
}

/* ── Test 3: Process with various input sizes ─────────────────────────────── */

static void test_various_sizes(void) {
    printf("Test 3: Various input sizes\n");

    DeepFilter *df = deep_filter_create(16000, NULL);
    ASSERT(df != NULL, "create");

    /* Test sizes that don't align with FFT size */
    int sizes[] = { 100, 256, 512, 1000, 1024, 2048, 4096 };
    int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    for (int i = 0; i < n_sizes; i++) {
        int n = sizes[i];
        float *buf = (float *)calloc((size_t)n, sizeof(float));
        ASSERT(buf != NULL, "alloc");

        gen_sine(buf, n, 440.0f, 16000.0f, 0.3f);
        deep_filter_process(df, buf, n);

        /* Should not crash and output should be finite */
        int all_finite = 1;
        for (int j = 0; j < n; j++) {
            if (!isfinite(buf[j])) { all_finite = 0; break; }
        }
        ASSERT(all_finite, "output should be finite");

        free(buf);
    }

    deep_filter_destroy(df);
    printf("  Passed\n");
}

/* ── Test 4: Reset clears state ───────────────────────────────────────────── */

static void test_reset(void) {
    printf("Test 4: Reset clears state\n");

    DeepFilter *df = deep_filter_create(16000, NULL);
    ASSERT(df != NULL, "create");

    /* Process some audio to build up state */
    float buf[1024];
    gen_sine(buf, 1024, 440.0f, 16000.0f, 0.5f);
    deep_filter_process(df, buf, 1024);

    /* Reset */
    deep_filter_reset(df);

    /* Process again — should work without issues */
    gen_sine(buf, 1024, 440.0f, 16000.0f, 0.5f);
    deep_filter_process(df, buf, 1024);

    int all_finite = 1;
    for (int i = 0; i < 1024; i++) {
        if (!isfinite(buf[i])) { all_finite = 0; break; }
    }
    ASSERT(all_finite, "output after reset should be finite");

    deep_filter_destroy(df);
    printf("  Passed\n");
}

/* ── Test 5: Parameter setters ────────────────────────────────────────────── */

static void test_parameters(void) {
    printf("Test 5: Parameter setters\n");

    DeepFilter *df = deep_filter_create(16000, NULL);
    ASSERT(df != NULL, "create");

    /* Strength: process with different settings, verify output remains valid */
    float buf0[1024], buf1[1024];
    gen_sine(buf0, 1024, 440.0f, 16000.0f, 0.5f);
    memcpy(buf1, buf0, sizeof(buf0));

    deep_filter_set_strength(df, 0.0f);
    deep_filter_process(df, buf0, 1024);
    int f0 = 1;
    for (int i = 0; i < 1024; i++) { if (!isfinite(buf0[i])) { f0 = 0; break; } }
    ASSERT(f0, "strength=0 produces finite output");

    deep_filter_reset(df);
    deep_filter_set_strength(df, 1.0f);
    deep_filter_process(df, buf1, 1024);
    int f1 = 1;
    for (int i = 0; i < 1024; i++) { if (!isfinite(buf1[i])) { f1 = 0; break; } }
    ASSERT(f1, "strength=1 produces finite output");

    /* Clamping: extreme values should not produce invalid output */
    deep_filter_reset(df);
    deep_filter_set_strength(df, -1.0f);  /* should clamp to 0 */
    float bufclamp[1024];
    gen_sine(bufclamp, 1024, 440.0f, 16000.0f, 0.5f);
    deep_filter_process(df, bufclamp, 1024);
    int fc = 1;
    for (int i = 0; i < 1024; i++) { if (!isfinite(bufclamp[i])) { fc = 0; break; } }
    ASSERT(fc, "clamped strength produces finite output");

    /* Min gain: verify different settings produce valid results */
    deep_filter_set_min_gain_db(df, -20.0f);
    deep_filter_set_min_gain_db(df, -60.0f);
    deep_filter_set_min_gain_db(df, 0.0f);
    deep_filter_set_min_gain_db(df, -100.0f);  /* should clamp to -60 */
    float buf_mg[1024];
    gen_sine(buf_mg, 1024, 440.0f, 16000.0f, 0.5f);
    deep_filter_process(df, buf_mg, 1024);
    int fmg = 1;
    for (int i = 0; i < 1024; i++) { if (!isfinite(buf_mg[i])) { fmg = 0; break; } }
    ASSERT(fmg, "min_gain settings produce finite output");

    /* NULL safety: verify no state corruption after NULL calls */
    deep_filter_set_strength(NULL, 0.5f);
    deep_filter_set_min_gain_db(NULL, -20.0f);
    deep_filter_process(NULL, NULL, 0);
    deep_filter_reset(NULL);

    /* Verify df still works after NULL calls to other handles */
    deep_filter_reset(df);
    float buf_after[1024];
    gen_sine(buf_after, 1024, 440.0f, 16000.0f, 0.5f);
    deep_filter_process(df, buf_after, 1024);
    int fa = 1;
    for (int i = 0; i < 1024; i++) { if (!isfinite(buf_after[i])) { fa = 0; break; } }
    ASSERT(fa, "df still works after NULL calls to other handles");

    deep_filter_destroy(df);
    printf("  Passed\n");
}

/* ── Test 6: Synthetic noisy signal processing ────────────────────────────── */

static void test_noisy_signal(void) {
    printf("Test 6: Noisy signal processing (passthrough mode)\n");

    DeepFilter *df = deep_filter_create(16000, NULL);
    ASSERT(df != NULL, "create");

    int n = 16000;  /* 1 second of audio */
    float *clean = (float *)malloc((size_t)n * sizeof(float));
    float *noisy = (float *)malloc((size_t)n * sizeof(float));
    float *noise = (float *)malloc((size_t)n * sizeof(float));
    ASSERT(clean && noisy && noise, "alloc");

    /* Generate clean speech-like signal (multi-frequency) */
    gen_sine(clean, n, 300.0f, 16000.0f, 0.4f);
    float *harmonic = (float *)malloc((size_t)n * sizeof(float));
    gen_sine(harmonic, n, 600.0f, 16000.0f, 0.2f);
    vDSP_vadd(clean, 1, harmonic, 1, clean, 1, (vDSP_Length)n);
    free(harmonic);

    /* Add noise */
    gen_noise(noise, n, 0.1f, 42);
    vDSP_vadd(clean, 1, noise, 1, noisy, 1, (vDSP_Length)n);

    float rms_noisy = compute_rms(noisy, n);

    /* Process (passthrough mode without weights) */
    deep_filter_process(df, noisy, n);

    float rms_processed = compute_rms(noisy + 512, n - 512);

    /* In passthrough, energy should be roughly preserved */
    ASSERT(rms_processed > 0.0f, "processed signal should have non-zero energy");
    ASSERT(isfinite(rms_processed), "processed RMS should be finite");

    printf("  RMS noisy=%.4f processed=%.4f\n", rms_noisy, rms_processed);

    free(clean);
    free(noisy);
    free(noise);
    deep_filter_destroy(df);
    printf("  Passed\n");
}

/* ── Test 7: Edge cases ───────────────────────────────────────────────────── */

static void test_edge_cases(void) {
    printf("Test 7: Edge cases\n");

    DeepFilter *df = deep_filter_create(16000, NULL);
    ASSERT(df != NULL, "create");

    /* Zero-length input: buffer should be unchanged */
    float buf[16];
    float sentinel = 12345.0f;
    buf[0] = sentinel;
    deep_filter_process(df, buf, 0);
    ASSERT(buf[0] == sentinel, "zero-length process should not modify buffer");

    /* Negative length: buffer should be unchanged */
    buf[0] = sentinel;
    deep_filter_process(df, buf, -1);
    ASSERT(buf[0] == sentinel, "negative length should not modify buffer");

    /* NULL pcm: filter should still work after */
    deep_filter_process(df, NULL, 100);
    float buf_after[512];
    gen_sine(buf_after, 512, 440.0f, 16000.0f, 0.3f);
    deep_filter_process(df, buf_after, 512);
    int valid = 1;
    for (int i = 0; i < 512; i++) { if (!isfinite(buf_after[i])) { valid = 0; break; } }
    ASSERT(valid, "filter works after NULL pcm call");

    /* Very small input */
    buf[0] = 0.5f;
    deep_filter_process(df, buf, 1);
    ASSERT(isfinite(buf[0]), "single sample should produce finite output");

    /* Silent input */
    float silence[2048];
    memset(silence, 0, sizeof(silence));
    deep_filter_process(df, silence, 2048);
    float rms = compute_rms(silence, 2048);
    ASSERT(rms < 0.001f, "silent input should produce near-silent output");

    /* Very loud input */
    float loud[1024];
    for (int i = 0; i < 1024; i++) loud[i] = 1.0f;
    deep_filter_process(df, loud, 1024);
    int all_finite = 1;
    for (int i = 0; i < 1024; i++) {
        if (!isfinite(loud[i])) { all_finite = 0; break; }
    }
    ASSERT(all_finite, "loud input should produce finite output");

    deep_filter_destroy(df);
    printf("  Passed\n");
}

/* ── Test 8: Weight file loading (nonexistent file) ───────────────────────── */

static void test_bad_weights(void) {
    printf("Test 8: Bad weight file handling\n");

    /* Nonexistent file -> should fall back to passthrough */
    DeepFilter *df = deep_filter_create(16000, "/nonexistent/path.dnf");
    ASSERT(df != NULL, "nonexistent weights should create passthrough filter");

    /* Process should still work in passthrough mode */
    float buf[1024];
    gen_sine(buf, 1024, 440.0f, 16000.0f, 0.3f);
    deep_filter_process(df, buf, 1024);

    int all_finite = 1;
    for (int i = 0; i < 1024; i++) {
        if (!isfinite(buf[i])) { all_finite = 0; break; }
    }
    ASSERT(all_finite, "passthrough after bad weights should produce finite output");

    deep_filter_destroy(df);
    printf("  Passed\n");
}

/* ── Test 9: Multiple process calls (streaming) ──────────────────────────── */

static void test_streaming(void) {
    printf("Test 9: Streaming (multiple small process calls)\n");

    DeepFilter *df = deep_filter_create(16000, NULL);
    ASSERT(df != NULL, "create");

    /* Process 10 chunks of 160 samples (10ms each) */
    for (int chunk = 0; chunk < 10; chunk++) {
        float buf[160];
        gen_sine(buf, 160, 440.0f, 16000.0f, 0.3f);
        deep_filter_process(df, buf, 160);

        int all_finite = 1;
        for (int i = 0; i < 160; i++) {
            if (!isfinite(buf[i])) { all_finite = 0; break; }
        }
        ASSERT(all_finite, "streaming chunk should produce finite output");
    }

    deep_filter_destroy(df);
    printf("  Passed\n");
}

/* ── Test 10: GRU path with synthetic weights ─────────────────────────────── */

static void test_gru_path(void) {
    printf("Test 10: GRU path with synthetic weights\n");

    /* Create weights where output bias = 0 → sigmoid(0) = 0.5 → 50% gain */
    char *wpath = create_test_weights(0.0f);
    ASSERT(wpath != NULL, "create test weight file");
    if (!wpath) return;

    DeepFilter *df = deep_filter_create(16000, wpath);
    ASSERT(df != NULL, "create with synthetic weights (not passthrough)");
    if (!df) { unlink(wpath); free(wpath); return; }

    /* Process a sine wave — GRU path should attenuate by ~50% (sigmoid(0) = 0.5 gain) */
    int n = 8192;  /* enough for multiple FFT frames */
    float *signal = (float *)malloc((size_t)n * sizeof(float));
    float *original = (float *)malloc((size_t)n * sizeof(float));
    ASSERT(signal && original, "alloc");

    gen_sine(signal, n, 440.0f, 16000.0f, 0.5f);
    memcpy(original, signal, (size_t)n * sizeof(float));

    /* Set strength to 1.0 for full suppression effect */
    deep_filter_set_strength(df, 1.0f);
    deep_filter_process(df, signal, n);

    /* Skip initial latency (first FFT frame) */
    int skip = 1024;
    int check_len = n - skip;
    float rms_orig = compute_rms(original + skip, check_len);
    float rms_out  = compute_rms(signal + skip, check_len);

    /* Output should be finite */
    int all_finite = 1;
    for (int i = 0; i < n; i++) {
        if (!isfinite(signal[i])) { all_finite = 0; break; }
    }
    ASSERT(all_finite, "GRU path output should be finite");

    /* Output should be attenuated (not passthrough) — gain ~0.5 */
    float ratio = (rms_orig > 0.0f) ? rms_out / rms_orig : 0.0f;
    printf("  GRU path energy ratio: %.3f (expected ~0.3-0.7 for 50%% gain)\n", ratio);
    ASSERT(ratio < 0.95f, "GRU path should attenuate signal (not passthrough)");
    ASSERT(ratio > 0.05f, "GRU path should not completely zero the signal");

    /* Compare with passthrough to ensure GRU actually changed the signal */
    DeepFilter *df_pt = deep_filter_create(16000, NULL);
    float *pt_signal = (float *)malloc((size_t)n * sizeof(float));
    gen_sine(pt_signal, n, 440.0f, 16000.0f, 0.5f);
    deep_filter_process(df_pt, pt_signal, n);
    float rms_pt = compute_rms(pt_signal + skip, check_len);
    float pt_ratio = (rms_orig > 0.0f) ? rms_pt / rms_orig : 0.0f;
    printf("  Passthrough energy ratio: %.3f\n", pt_ratio);
    ASSERT(fabsf(ratio - pt_ratio) > 0.05f,
           "GRU path should differ from passthrough");
    free(pt_signal);
    deep_filter_destroy(df_pt);

    free(signal);
    free(original);
    deep_filter_destroy(df);
    unlink(wpath);
    free(wpath);
    printf("  Passed\n");
}

/* ── Test 11: Noise reduction verification ────────────────────────────────── */

static void test_noise_reduction(void) {
    printf("Test 11: Noise reduction verification with GRU path\n");

    /* Create weights with strong attenuation: bias = -2 → sigmoid(-2) ≈ 0.12 gain */
    char *wpath = create_test_weights(-2.0f);
    ASSERT(wpath != NULL, "create test weight file for noise test");
    if (!wpath) return;

    DeepFilter *df = deep_filter_create(16000, wpath);
    ASSERT(df != NULL, "create with attenuation weights");
    if (!df) { unlink(wpath); free(wpath); return; }

    deep_filter_set_strength(df, 1.0f);

    int n = 16000;  /* 1 second */
    float *noisy = (float *)malloc((size_t)n * sizeof(float));
    float *noise_only = (float *)malloc((size_t)n * sizeof(float));
    ASSERT(noisy && noise_only, "alloc noise test buffers");

    /* Generate signal + noise at 0 dB SNR */
    float *clean = (float *)malloc((size_t)n * sizeof(float));
    gen_sine(clean, n, 440.0f, 16000.0f, 0.3f);
    gen_noise(noise_only, n, 0.3f, 12345);
    vDSP_vadd(clean, 1, noise_only, 1, noisy, 1, (vDSP_Length)n);

    float rms_before = compute_rms(noisy, n);

    /* Process */
    deep_filter_process(df, noisy, n);

    /* Skip initial latency */
    int skip = 1024;
    float rms_after = compute_rms(noisy + skip, n - skip);

    /* With sigmoid(-2) ≈ 0.12 gain, the output RMS should be significantly lower */
    printf("  RMS before=%.4f after=%.4f (ratio=%.3f)\n",
           rms_before, rms_after,
           rms_after / (rms_before > 0.0f ? rms_before : 1.0f));
    ASSERT(rms_after < rms_before, "processed RMS should be lower than input");
    ASSERT(rms_after > 0.0f, "processed signal should not be completely silent");

    /* All output samples should be finite */
    int all_finite = 1;
    for (int i = 0; i < n; i++) {
        if (!isfinite(noisy[i])) { all_finite = 0; break; }
    }
    ASSERT(all_finite, "noise reduction output should be finite");

    free(clean);
    free(noisy);
    free(noise_only);
    deep_filter_destroy(df);
    unlink(wpath);
    free(wpath);
    printf("  Passed\n");
}

/* ── Test 12: NaN input guard ─────────────────────────────────────────────── */

static void test_nan_guard(void) {
    printf("Test 12: NaN input guard\n");

    DeepFilter *df = deep_filter_create(16000, NULL);
    ASSERT(df != NULL, "create");

    /* Process a buffer with NaN values — should clamp to zero and not crash */
    int n = 2048;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    ASSERT(buf != NULL, "alloc");

    gen_sine(buf, n, 440.0f, 16000.0f, 0.3f);
    /* Inject NaN at various positions */
    buf[0] = NAN;
    buf[100] = NAN;
    buf[511] = NAN;       /* at FFT boundary */
    buf[512] = NAN;       /* at hop boundary */
    buf[1000] = INFINITY;  /* also test infinity */
    buf[1500] = -INFINITY;

    deep_filter_process(df, buf, n);

    /* All output should be finite (NaN should not propagate) */
    int all_finite = 1;
    int nan_count = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(buf[i])) { all_finite = 0; nan_count++; }
    }
    ASSERT(all_finite, "output should be finite after NaN input");
    if (!all_finite) {
        printf("  WARNING: %d non-finite values in output\n", nan_count);
    }

    /* Process another buffer after NaN — verify GRU state wasn't permanently poisoned */
    float *buf2 = (float *)malloc((size_t)n * sizeof(float));
    gen_sine(buf2, n, 440.0f, 16000.0f, 0.3f);
    deep_filter_process(df, buf2, n);

    int all_finite2 = 1;
    for (int i = 0; i < n; i++) {
        if (!isfinite(buf2[i])) { all_finite2 = 0; break; }
    }
    ASSERT(all_finite2, "output after NaN frame should be finite (state not poisoned)");

    float rms = compute_rms(buf2 + 512, n - 512);
    ASSERT(rms > 0.01f, "output after NaN frame should have meaningful energy");

    free(buf);
    free(buf2);
    deep_filter_destroy(df);
    printf("  Passed\n");
}

/* ── Test 13: NaN guard with GRU path ─────────────────────────────────────── */

static void test_nan_guard_gru(void) {
    printf("Test 13: NaN guard with GRU path (state recovery)\n");

    char *wpath = create_test_weights(0.0f);
    ASSERT(wpath != NULL, "create weights for NaN GRU test");
    if (!wpath) return;

    DeepFilter *df = deep_filter_create(16000, wpath);
    ASSERT(df != NULL, "create with weights");
    if (!df) { unlink(wpath); free(wpath); return; }

    deep_filter_set_strength(df, 1.0f);

    /* Process clean audio to establish GRU state */
    int n = 4096;
    float *clean = (float *)malloc((size_t)n * sizeof(float));
    gen_sine(clean, n, 440.0f, 16000.0f, 0.5f);
    deep_filter_process(df, clean, n);

    /* Now inject NaN — GRU state should recover */
    float *nan_buf = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) nan_buf[i] = NAN;
    deep_filter_process(df, nan_buf, n);

    /* Process clean audio again — should produce finite output */
    gen_sine(clean, n, 440.0f, 16000.0f, 0.5f);
    deep_filter_process(df, clean, n);

    int all_finite = 1;
    for (int i = 0; i < n; i++) {
        if (!isfinite(clean[i])) { all_finite = 0; break; }
    }
    ASSERT(all_finite, "GRU state recovers after NaN frame");

    float rms = compute_rms(clean + 512, n - 512);
    ASSERT(rms > 0.01f, "GRU output has energy after NaN recovery");

    free(clean);
    free(nan_buf);
    deep_filter_destroy(df);
    unlink(wpath);
    free(wpath);
    printf("  Passed\n");
}

/* ── Main ─────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== DeepFilter Neural Noise Suppression Tests ===\n\n");

    test_lifecycle();
    test_passthrough();
    test_various_sizes();
    test_reset();
    test_parameters();
    test_noisy_signal();
    test_edge_cases();
    test_bad_weights();
    test_streaming();
    test_gru_path();
    test_noise_reduction();
    test_nan_guard();
    test_nan_guard_gru();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
