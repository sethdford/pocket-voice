/**
 * test_spatial_audio.c — Tests for SpatialAudioEngine (HRTF binaural 3D audio).
 *
 * Verifies:
 *   - Lifecycle (create/destroy)
 *   - NULL safety on all API functions
 *   - Position setting and bounds
 *   - Single-source spatial processing
 *   - Multi-source mixing
 *   - Stereo output asymmetry from azimuth
 *   - Distance attenuation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

#define MAX_SOURCES 8

typedef struct SpatialAudioEngine SpatialAudioEngine;

extern SpatialAudioEngine *spatial_create(uint32_t sample_rate);
extern int spatial_set_position(SpatialAudioEngine *engine, int source_idx,
                                 float azimuth, float elevation, float distance);
extern int spatial_process(SpatialAudioEngine *engine, int source_idx,
                           const float *mono_input,
                           float *left_output, float *right_output,
                           int n_samples);
extern int spatial_mix(SpatialAudioEngine *engine,
                       const float **mono_inputs, int n_sources,
                       float *left_out, float *right_out,
                       int n_samples);
extern void spatial_destroy(SpatialAudioEngine *engine);

static int passed = 0, failed = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); passed++; } \
    else { printf("  [FAIL] %s\n", msg); failed++; } \
} while (0)

static float rms(const float *buf, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += buf[i] * buf[i];
    return sqrtf(sum / n);
}

/* ── Lifecycle tests ─────────────────────────────────────────────── */

static void test_create_destroy(void) {
    printf("\n[Create/Destroy]\n");
    SpatialAudioEngine *e = spatial_create(48000);
    CHECK(e != NULL, "create with 48kHz succeeds");
    spatial_destroy(e);
    CHECK(1, "destroy does not crash");
}

static void test_create_default_rate(void) {
    printf("\n[Create Default Rate]\n");
    SpatialAudioEngine *e = spatial_create(0);
    CHECK(e != NULL, "create with 0 (default 48kHz) succeeds");
    spatial_destroy(e);
}

/* ── NULL safety ─────────────────────────────────────────────────── */

static void test_null_safety(void) {
    printf("\n[NULL Safety]\n");
    spatial_destroy(NULL);
    CHECK(1, "destroy(NULL) does not crash");

    int ret = spatial_set_position(NULL, 0, 0, 0, 1.0f);
    CHECK(ret < 0, "set_position(NULL engine) returns error");

    ret = spatial_process(NULL, 0, NULL, NULL, NULL, 0);
    CHECK(ret < 0, "process(NULL engine) returns error");

    ret = spatial_mix(NULL, NULL, 0, NULL, NULL, 0);
    CHECK(ret < 0, "mix(NULL engine) returns error");
}

/* ── Position setting ────────────────────────────────────────────── */

static void test_set_position(void) {
    printf("\n[Set Position]\n");
    SpatialAudioEngine *e = spatial_create(48000);
    CHECK(e != NULL, "create for position test");
    if (!e) return;

    int ret = spatial_set_position(e, 0, -90.0f, 0.0f, 1.0f);
    CHECK(ret == 0, "set position source 0 (left) succeeds");

    ret = spatial_set_position(e, MAX_SOURCES - 1, 90.0f, 45.0f, 2.0f);
    CHECK(ret == 0, "set position last source succeeds");

    ret = spatial_set_position(e, -1, 0, 0, 1);
    CHECK(ret < 0, "set position with negative index fails");

    ret = spatial_set_position(e, MAX_SOURCES, 0, 0, 1);
    CHECK(ret < 0, "set position with out-of-range index fails");

    spatial_destroy(e);
}

/* ── Single-source processing ────────────────────────────────────── */

static void test_process_single_source(void) {
    printf("\n[Process Single Source]\n");
    SpatialAudioEngine *e = spatial_create(48000);
    CHECK(e != NULL, "create for process test");
    if (!e) return;

    spatial_set_position(e, 0, 0.0f, 0.0f, 1.0f);  /* center */

    int n = 256;
    float *mono = (float *)calloc(n, sizeof(float));
    float *left = (float *)calloc(n, sizeof(float));
    float *right = (float *)calloc(n, sizeof(float));

    for (int i = 0; i < n; i++) {
        mono[i] = sinf(2.0f * (float)M_PI * 440.0f * i / 48000.0f);
    }

    int ret = spatial_process(e, 0, mono, left, right, n);
    CHECK(ret == 0, "process returns success");

    float l_rms = rms(left, n);
    float r_rms = rms(right, n);
    CHECK(l_rms > 0.001f, "left channel has signal");
    CHECK(r_rms > 0.001f, "right channel has signal");

    free(mono);
    free(left);
    free(right);
    spatial_destroy(e);
}

/* ── Azimuth asymmetry ───────────────────────────────────────────── */

static void test_azimuth_asymmetry(void) {
    printf("\n[Azimuth Asymmetry]\n");
    SpatialAudioEngine *e = spatial_create(48000);
    CHECK(e != NULL, "create for azimuth test");
    if (!e) return;

    /* Source far left: azimuth = -90 */
    spatial_set_position(e, 0, -90.0f, 0.0f, 1.0f);

    int n = 512;
    float *mono = (float *)calloc(n, sizeof(float));
    float *left = (float *)calloc(n, sizeof(float));
    float *right = (float *)calloc(n, sizeof(float));

    for (int i = 0; i < n; i++) {
        mono[i] = 0.8f * sinf(2.0f * (float)M_PI * 440.0f * i / 48000.0f);
    }

    spatial_process(e, 0, mono, left, right, n);

    float l_rms = rms(left, n);
    float r_rms = rms(right, n);
    CHECK(l_rms > r_rms, "source at -90 azimuth: left louder than right");

    free(mono);
    free(left);
    free(right);
    spatial_destroy(e);
}

/* ── Distance attenuation ────────────────────────────────────────── */

static void test_distance_attenuation(void) {
    printf("\n[Distance Attenuation]\n");
    SpatialAudioEngine *e = spatial_create(48000);
    CHECK(e != NULL, "create for distance test");
    if (!e) return;

    int n = 512;
    float *mono = (float *)calloc(n, sizeof(float));
    float *left_near = (float *)calloc(n, sizeof(float));
    float *right_near = (float *)calloc(n, sizeof(float));
    float *left_far = (float *)calloc(n, sizeof(float));
    float *right_far = (float *)calloc(n, sizeof(float));

    for (int i = 0; i < n; i++) {
        mono[i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * i / 48000.0f);
    }

    /* Near source */
    spatial_set_position(e, 0, 0.0f, 0.0f, 0.5f);
    spatial_process(e, 0, mono, left_near, right_near, n);
    float rms_near = rms(left_near, n) + rms(right_near, n);

    /* Far source */
    spatial_set_position(e, 0, 0.0f, 0.0f, 5.0f);
    spatial_process(e, 0, mono, left_far, right_far, n);
    float rms_far = rms(left_far, n) + rms(right_far, n);

    CHECK(rms_near > rms_far, "near source louder than far source");

    free(mono);
    free(left_near);
    free(right_near);
    free(left_far);
    free(right_far);
    spatial_destroy(e);
}

/* ── Multi-source mixing ─────────────────────────────────────────── */

static void test_mix_multiple_sources(void) {
    printf("\n[Mix Multiple Sources]\n");
    SpatialAudioEngine *e = spatial_create(48000);
    CHECK(e != NULL, "create for mix test");
    if (!e) return;

    int n = 256;

    /* Two sources: one left, one right */
    spatial_set_position(e, 0, -60.0f, 0.0f, 1.5f);
    spatial_set_position(e, 1, 60.0f, 0.0f, 1.5f);

    float *mono0 = (float *)calloc(n, sizeof(float));
    float *mono1 = (float *)calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        mono0[i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * i / 48000.0f);
        mono1[i] = 0.5f * sinf(2.0f * (float)M_PI * 880.0f * i / 48000.0f);
    }

    const float *inputs[2] = { mono0, mono1 };
    float *left_out = (float *)calloc(n, sizeof(float));
    float *right_out = (float *)calloc(n, sizeof(float));

    int ret = spatial_mix(e, inputs, 2, left_out, right_out, n);
    CHECK(ret == 0, "mix 2 sources returns success");

    float l_rms = rms(left_out, n);
    float r_rms = rms(right_out, n);
    CHECK(l_rms > 0.001f, "mix left channel has signal");
    CHECK(r_rms > 0.001f, "mix right channel has signal");

    free(mono0);
    free(mono1);
    free(left_out);
    free(right_out);
    spatial_destroy(e);
}

/* ── Out-of-bounds source index ──────────────────────────────────── */

static void test_process_invalid_source(void) {
    printf("\n[Process Invalid Source Index]\n");
    SpatialAudioEngine *e = spatial_create(48000);
    CHECK(e != NULL, "create for invalid source test");
    if (!e) return;

    float mono[64] = {0};
    float left[64], right[64];

    int ret = spatial_process(e, -1, mono, left, right, 64);
    CHECK(ret < 0, "process with negative source_idx returns error");

    ret = spatial_process(e, MAX_SOURCES, mono, left, right, 64);
    CHECK(ret < 0, "process with out-of-range source_idx returns error");

    spatial_destroy(e);
}

/* ── Mix with zero/negative sources ──────────────────────────────── */

static void test_mix_edge_cases(void) {
    printf("\n[Mix Edge Cases]\n");
    SpatialAudioEngine *e = spatial_create(48000);
    CHECK(e != NULL, "create for mix edge case test");
    if (!e) return;

    float left[64], right[64];

    int ret = spatial_mix(e, NULL, 0, left, right, 64);
    CHECK(ret < 0, "mix with 0 sources returns error");

    ret = spatial_mix(e, NULL, -1, left, right, 64);
    CHECK(ret < 0, "mix with negative source count returns error");

    spatial_destroy(e);
}

/* ── Main ────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== Spatial Audio Tests ===\n");

    test_create_destroy();
    test_create_default_rate();
    test_null_safety();
    test_set_position();
    test_process_single_source();
    test_azimuth_asymmetry();
    test_distance_attenuation();
    test_mix_multiple_sources();
    test_process_invalid_source();
    test_mix_edge_cases();

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed;
}
