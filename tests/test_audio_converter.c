/**
 * test_audio_converter.c — Tests for HWResampler (AudioConverter wrapper).
 *
 * Verifies:
 *   - Lifecycle (create/destroy)
 *   - NULL safety on all API functions
 *   - Upsampling (24kHz → 48kHz)
 *   - Downsampling (48kHz → 24kHz)
 *   - Identity conversion (same rate)
 *   - Reset behavior
 *   - Quality levels
 *   - Zero-length input
 */

#include <AudioToolbox/AudioToolbox.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef enum {
    RESAMPLE_MIN       = 0,
    RESAMPLE_LOW       = 1,
    RESAMPLE_MEDIUM    = 2,
    RESAMPLE_HIGH      = 3,
    RESAMPLE_MAX       = 4,
} ResampleQuality;

typedef struct HWResampler HWResampler;

extern HWResampler *hw_resampler_create(uint32_t src_rate, uint32_t dst_rate,
                                         uint32_t channels, ResampleQuality quality);
extern int hw_resample(HWResampler *ctx, const float *input, uint32_t in_frames,
                       float *output, uint32_t max_out);
extern void hw_resampler_reset(HWResampler *ctx);
extern void hw_resampler_destroy(HWResampler *ctx);

static int passed = 0, failed = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); passed++; } \
    else { printf("  [FAIL] %s\n", msg); failed++; } \
} while (0)

/* ── Lifecycle tests ─────────────────────────────────────────────── */

static void test_create_destroy(void) {
    printf("\n[Create/Destroy]\n");
    HWResampler *r = hw_resampler_create(24000, 48000, 1, RESAMPLE_HIGH);
    CHECK(r != NULL, "create 24k→48k mono succeeds");
    hw_resampler_destroy(r);
    CHECK(1, "destroy does not crash");
}

static void test_create_all_qualities(void) {
    printf("\n[Create All Quality Levels]\n");
    ResampleQuality levels[] = {RESAMPLE_MIN, RESAMPLE_LOW, RESAMPLE_MEDIUM, RESAMPLE_HIGH, RESAMPLE_MAX};
    const char *names[] = {"MIN", "LOW", "MEDIUM", "HIGH", "MAX"};
    for (int i = 0; i < 5; i++) {
        HWResampler *r = hw_resampler_create(24000, 48000, 1, levels[i]);
        char msg[64];
        snprintf(msg, sizeof(msg), "create with quality %s succeeds", names[i]);
        CHECK(r != NULL, msg);
        hw_resampler_destroy(r);
    }
}

/* ── NULL safety ─────────────────────────────────────────────────── */

static void test_null_safety(void) {
    printf("\n[NULL Safety]\n");
    hw_resampler_destroy(NULL);
    CHECK(1, "destroy(NULL) does not crash");

    hw_resampler_reset(NULL);
    CHECK(1, "reset(NULL) does not crash");

    int ret = hw_resample(NULL, NULL, 0, NULL, 0);
    CHECK(ret < 0, "resample(NULL) returns error");
}

/* ── Upsample 2:1 ───────────────────────────────────────────────── */

static void test_upsample_2x(void) {
    printf("\n[Upsample 24kHz → 48kHz]\n");
    HWResampler *r = hw_resampler_create(24000, 48000, 1, RESAMPLE_HIGH);
    CHECK(r != NULL, "create for upsample test");
    if (!r) return;

    int in_frames = 480;  /* 20ms at 24kHz */
    float *input = (float *)calloc(in_frames, sizeof(float));
    for (int i = 0; i < in_frames; i++) {
        input[i] = sinf(2.0f * (float)M_PI * 440.0f * i / 24000.0f);
    }

    int out_max = 1024;
    float *output = (float *)calloc(out_max, sizeof(float));
    int out_frames = hw_resample(r, input, in_frames, output, out_max);
    CHECK(out_frames > 0, "upsample produces output");

    /* 2:1 upsample: expect approximately 2x input frames */
    float ratio = (float)out_frames / (float)in_frames;
    CHECK(ratio > 1.5f && ratio < 2.5f, "output ~2x input frames");

    free(input);
    free(output);
    hw_resampler_destroy(r);
}

/* ── Downsample 2:1 ──────────────────────────────────────────────── */

static void test_downsample_2x(void) {
    printf("\n[Downsample 48kHz → 24kHz]\n");
    HWResampler *r = hw_resampler_create(48000, 24000, 1, RESAMPLE_HIGH);
    CHECK(r != NULL, "create for downsample test");
    if (!r) return;

    int in_frames = 960;  /* 20ms at 48kHz */
    float *input = (float *)calloc(in_frames, sizeof(float));
    for (int i = 0; i < in_frames; i++) {
        input[i] = sinf(2.0f * (float)M_PI * 440.0f * i / 48000.0f);
    }

    int out_max = 1024;
    float *output = (float *)calloc(out_max, sizeof(float));
    int out_frames = hw_resample(r, input, in_frames, output, out_max);
    CHECK(out_frames > 0, "downsample produces output");

    float ratio = (float)out_frames / (float)in_frames;
    CHECK(ratio > 0.3f && ratio < 0.7f, "output ~0.5x input frames");

    free(input);
    free(output);
    hw_resampler_destroy(r);
}

/* ── Identity (same rate) ────────────────────────────────────────── */

static void test_identity_conversion(void) {
    printf("\n[Identity Conversion 48kHz → 48kHz]\n");
    HWResampler *r = hw_resampler_create(48000, 48000, 1, RESAMPLE_HIGH);
    CHECK(r != NULL, "create for identity test");
    if (!r) return;

    int in_frames = 480;
    float *input = (float *)calloc(in_frames, sizeof(float));
    for (int i = 0; i < in_frames; i++) {
        input[i] = sinf(2.0f * (float)M_PI * 1000.0f * i / 48000.0f);
    }

    int out_max = 1024;
    float *output = (float *)calloc(out_max, sizeof(float));
    int out_frames = hw_resample(r, input, in_frames, output, out_max);
    CHECK(out_frames > 0, "identity conversion produces output");
    CHECK(abs(out_frames - in_frames) <= 2, "output frames ~= input frames");

    free(input);
    free(output);
    hw_resampler_destroy(r);
}

/* ── Reset ───────────────────────────────────────────────────────── */

static void test_reset(void) {
    printf("\n[Reset]\n");
    HWResampler *r = hw_resampler_create(24000, 48000, 1, RESAMPLE_HIGH);
    CHECK(r != NULL, "create for reset test");
    if (!r) return;

    /* Process some data first */
    int in_frames = 480;
    float *input = (float *)calloc(in_frames, sizeof(float));
    for (int i = 0; i < in_frames; i++) input[i] = 0.5f;

    float *output = (float *)calloc(1024, sizeof(float));
    hw_resample(r, input, in_frames, output, 1024);

    /* Reset and process again */
    hw_resampler_reset(r);
    int out2 = hw_resample(r, input, in_frames, output, 1024);
    CHECK(out2 > 0, "resample after reset produces output");

    free(input);
    free(output);
    hw_resampler_destroy(r);
}

/* ── Zero-length input ───────────────────────────────────────────── */

static void test_zero_length_input(void) {
    printf("\n[Zero-Length Input]\n");
    HWResampler *r = hw_resampler_create(24000, 48000, 1, RESAMPLE_HIGH);
    CHECK(r != NULL, "create for zero-length test");
    if (!r) return;

    float dummy_out[64];
    float dummy_in = 0.0f;
    int out_frames = hw_resample(r, &dummy_in, 0, dummy_out, 64);
    CHECK(out_frames >= 0, "zero-length input does not crash");

    hw_resampler_destroy(r);
}

/* ── Non-standard rate ───────────────────────────────────────────── */

static void test_non_standard_rate(void) {
    printf("\n[Non-Standard Rate 16kHz → 44100Hz]\n");
    HWResampler *r = hw_resampler_create(16000, 44100, 1, RESAMPLE_MEDIUM);
    CHECK(r != NULL, "create 16k→44.1k succeeds");
    if (!r) return;

    int in_frames = 320;  /* 20ms at 16kHz */
    float *input = (float *)calloc(in_frames, sizeof(float));
    for (int i = 0; i < in_frames; i++) {
        input[i] = sinf(2.0f * (float)M_PI * 300.0f * i / 16000.0f);
    }

    float *output = (float *)calloc(2048, sizeof(float));
    int out_frames = hw_resample(r, input, in_frames, output, 2048);
    CHECK(out_frames > 0, "non-standard rate produces output");

    float ratio = (float)out_frames / (float)in_frames;
    CHECK(ratio > 2.0f && ratio < 3.5f, "output ratio ~2.76x (44100/16000)");

    free(input);
    free(output);
    hw_resampler_destroy(r);
}

/* ── Main ────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== Audio Converter Tests ===\n");

    test_create_destroy();
    test_create_all_qualities();
    test_null_safety();
    test_upsample_2x();
    test_downsample_2x();
    test_identity_conversion();
    test_reset();
    test_zero_length_input();
    test_non_standard_rate();

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed;
}
