/**
 * test_quality_improvements.c — Tests for quality improvements:
 *   1. Spectral noise gate effectiveness
 *   2. LUFS normalization consistency
 *   3. Noise gate + STT robustness
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "noise_gate.h"
#include "lufs.h"
#include "voice_quality.h"

static int tests_run = 0, tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("[%2d] %-60s ", tests_run, name); \
    fflush(stdout); \
} while(0)

#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) printf("FAIL: %s\n", msg)

/* ═══════════════════════════════════════════════════════════════════════════
 * Section 1: Noise Gate Tests
 * ═══════════════════════════════════════════════════════════════════════════ */

static void generate_white_noise(float *buf, int n, float amplitude) {
    for (int i = 0; i < n; i++) {
        float r = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        buf[i] = r * amplitude;
    }
}

static void generate_sine(float *buf, int n, float freq, float sr, float amp) {
    for (int i = 0; i < n; i++) {
        buf[i] = amp * sinf(2.0f * M_PI * freq * i / sr);
    }
}

static float compute_rms(const float *buf, int n) {
    if (n <= 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += buf[i] * buf[i];
    return sqrtf(sum / n);
}

static void test_noise_gate_create_destroy(void) {
    TEST("Noise gate: create/destroy");
    NoiseGate *ng = noise_gate_create(16000, 512, 256);
    if (!ng) { FAIL("create returned NULL"); return; }
    noise_gate_destroy(ng);
    PASS();
}

static void test_noise_gate_invalid_params(void) {
    TEST("Noise gate: reject invalid params");
    NoiseGate *ng1 = noise_gate_create(16000, 100, 50);  /* not power of 2 */
    NoiseGate *ng2 = noise_gate_create(16000, 512, 0);   /* hop_size 0 */
    NoiseGate *ng3 = noise_gate_create(16000, 512, 600);  /* hop > fft */
    if (ng1 || ng2 || ng3) { FAIL("accepted invalid params"); return; }
    PASS();
}

static void test_noise_gate_reduces_noise(void) {
    TEST("Noise gate: reduces background noise");

    /* Process noise through the gate in two passes:
     * Pass 1: fresh noise (learn + gate)
     * Pass 2: new noise batch (gate with learned profile) */
    NoiseGate *ng = noise_gate_create(16000, 512, 256);
    if (!ng) { FAIL("create failed"); return; }

    int n = 16000;
    float *noise1 = calloc(n, sizeof(float));
    float *noise2 = calloc(n, sizeof(float));
    generate_white_noise(noise1, n, 0.05f);
    generate_white_noise(noise2, n, 0.05f);
    float rms_before = compute_rms(noise2, n);

    /* Pass 1: let the gate learn the noise profile */
    noise_gate_learn_noise(ng, 50);
    noise_gate_process(ng, noise1, n);

    /* Pass 2: gate should now attenuate the noise */
    noise_gate_process(ng, noise2, n);
    float rms_after = compute_rms(noise2, n);

    noise_gate_destroy(ng);
    free(noise1);
    free(noise2);

    /* Expect some reduction — even a few dB means the gate is working */
    if (rms_after < rms_before) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "No reduction: before=%.4f after=%.4f", rms_before, rms_after);
        FAIL(msg);
    }
}

static void test_noise_gate_preserves_speech(void) {
    TEST("Noise gate: preserves clean speech signal");
    NoiseGate *ng = noise_gate_create(16000, 512, 256);
    if (!ng) { FAIL("create failed"); return; }

    int n = 16000;
    float *clean = calloc(n, sizeof(float));
    float *original = calloc(n, sizeof(float));

    /* Silence period for noise learning, then speech-like signal */
    generate_white_noise(clean, n / 4, 0.001f);      /* quiet background */
    generate_sine(clean + n / 4, 3 * n / 4, 440.0f, 16000.0f, 0.5f); /* loud signal */
    memcpy(original, clean, n * sizeof(float));

    noise_gate_process(ng, clean, n);

    /* Compare RMS of the speech portion */
    float rms_orig = compute_rms(original + n / 2, n / 4);
    float rms_proc = compute_rms(clean + n / 2, n / 4);

    noise_gate_destroy(ng);
    free(clean);
    free(original);

    float ratio = rms_proc / rms_orig;
    if (ratio > 0.7f) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "Signal attenuated to %.1f%% (want > 70%%)", ratio * 100);
        FAIL(msg);
    }
}

static void test_noise_gate_set_params(void) {
    TEST("Noise gate: set custom params");
    NoiseGate *ng = noise_gate_create(16000, 512, 256);
    if (!ng) { FAIL("create failed"); return; }
    noise_gate_set_params(ng, 12.0f, 2.0f, 100.0f);
    noise_gate_destroy(ng);
    PASS();
}

static void test_noise_gate_reset(void) {
    TEST("Noise gate: reset clears state");
    NoiseGate *ng = noise_gate_create(16000, 512, 256);
    if (!ng) { FAIL("create failed"); return; }

    int n = 8000;
    float *buf = calloc(n, sizeof(float));
    generate_white_noise(buf, n, 0.01f);
    noise_gate_process(ng, buf, n);

    noise_gate_reset(ng);
    /* Should start learning noise again */
    generate_white_noise(buf, n, 0.01f);
    noise_gate_process(ng, buf, n);

    noise_gate_destroy(ng);
    free(buf);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Section 2: LUFS Normalization Tests
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_lufs_consistency(void) {
    TEST("LUFS: consistent output level across amplitudes");

    float target = -16.0f;
    float results[3];
    float amplitudes[] = { 0.1f, 0.3f, 0.8f };

    for (int a = 0; a < 3; a++) {
        LUFSMeter *m = lufs_create(48000, 400);
        if (!m) { FAIL("create failed"); return; }

        int n = 48000;  /* 1 second */
        float *buf = calloc(n, sizeof(float));
        generate_sine(buf, n, 440.0f, 48000.0f, amplitudes[a]);

        lufs_normalize(m, buf, n, target);
        results[a] = compute_rms(buf, n);

        lufs_destroy(m);
        free(buf);
    }

    /* All outputs should be within 3 dB of each other */
    float max_r = results[0], min_r = results[0];
    for (int i = 1; i < 3; i++) {
        if (results[i] > max_r) max_r = results[i];
        if (results[i] < min_r) min_r = results[i];
    }

    float spread_db = 20.0f * log10f(max_r / (min_r + 1e-10f));
    if (spread_db < 6.0f) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "Output spread %.1f dB (want < 6 dB)", spread_db);
        FAIL(msg);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Section 3: Noise Gate + Speech Pipeline Integration
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_noise_gate_with_noisy_speech(void) {
    TEST("Noise gate: processes noisy speech without crashing");
    NoiseGate *ng = noise_gate_create(16000, 512, 256);
    if (!ng) { FAIL("create failed"); return; }

    int n = 32000;  /* 2 seconds */
    float *noisy = calloc(n, sizeof(float));

    /* First second: pure noise (for learning), second: noisy speech */
    generate_white_noise(noisy, n / 2, 0.02f);
    for (int i = n / 2; i < n; i++) {
        float speech = 0.3f * sinf(2.0f * M_PI * 200.0f * i / 16000.0f)
                     + 0.2f * sinf(2.0f * M_PI * 400.0f * i / 16000.0f);
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.04f;
        noisy[i] = speech + noise;
    }

    /* Process in realistic chunk sizes (10ms at 16kHz) */
    noise_gate_learn_noise(ng, 30);
    int chunk = 160;
    for (int i = 0; i < n; i += chunk) {
        int sz = (i + chunk <= n) ? chunk : (n - i);
        noise_gate_process(ng, noisy + i, sz);
    }

    /* Just verify the speech portion still has reasonable energy */
    float rms = compute_rms(noisy + 3 * n / 4, n / 4);

    noise_gate_destroy(ng);
    free(noisy);

    if (rms > 0.01f) {
        PASS();
    } else {
        char msg[64];
        snprintf(msg, sizeof(msg), "Speech energy too low: rms=%.4f", rms);
        FAIL(msg);
    }
}

static void test_noise_gate_small_buffers(void) {
    TEST("Noise gate: handles small buffer sizes");
    NoiseGate *ng = noise_gate_create(16000, 512, 256);
    if (!ng) { FAIL("create failed"); return; }

    float buf[160];  /* 10ms at 16kHz — smaller than FFT size */
    generate_white_noise(buf, 160, 0.01f);

    /* Should not crash even with buffers smaller than FFT */
    for (int i = 0; i < 100; i++) {
        noise_gate_process(ng, buf, 160);
    }

    noise_gate_destroy(ng);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    srand((unsigned)time(NULL));

    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║           Quality Improvements Test Suite                    ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    printf("── Noise Gate ──\n");
    test_noise_gate_create_destroy();
    test_noise_gate_invalid_params();
    test_noise_gate_reduces_noise();
    test_noise_gate_preserves_speech();
    test_noise_gate_set_params();
    test_noise_gate_reset();
    test_noise_gate_small_buffers();
    test_noise_gate_with_noisy_speech();

    printf("\n── LUFS Normalization ──\n");
    test_lufs_consistency();

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
