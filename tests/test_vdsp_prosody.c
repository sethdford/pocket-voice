/**
 * test_vdsp_prosody.c — Unit tests for vdsp_prosody.c (AMX-accelerated audio effects).
 *
 * Tests: pitch shift, time stretch, volume, soft limiter, formant EQ, crossfade.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-55s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* FFI declarations matching vdsp_prosody.c */
extern int   prosody_pitch_shift(const float *input, float *output, int n_samples,
                                  float pitch_factor, int fft_size);
typedef struct BiquadCascade BiquadCascade;
extern BiquadCascade *prosody_create_formant_eq(float pitch_factor, int sample_rate);
extern int   prosody_apply_biquad(BiquadCascade *bc, float *audio, int n_samples);
extern void  prosody_destroy_biquad(BiquadCascade *bc);
extern void  prosody_soft_limit(float *audio, int n_samples,
                                 float threshold, float knee_db);
extern void  prosody_volume(float *audio, int n_samples, float volume_db,
                             float fade_ms, int sample_rate);
extern int   prosody_time_stretch(const float *input, int in_len, float *output,
                                   float rate_factor, float window_ms, int sample_rate);
extern void  prosody_crossfade(const float *seg_a, const float *seg_b,
                                float *output, int n_samples);

/* Generate a 440 Hz sine wave at the given sample rate */
static void gen_sine(float *buf, int n, float freq, int sr) {
    for (int i = 0; i < n; i++)
        buf[i] = sinf(2.0f * 3.14159265f * freq * i / sr);
}

static float rms(const float *buf, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += buf[i] * buf[i];
    return sqrtf(sum / n);
}

/* ── Pitch Shift Tests ─────────────────────────────────── */

static void test_pitch_shift_identity(void) {
    TEST("pitch_shift: factor=1.0 produces non-silent output");
    float in[4096], out[4096];
    gen_sine(in, 4096, 440.0f, 24000);
    int rc = prosody_pitch_shift(in, out, 4096, 1.0f, 2048);
    ASSERT(rc == 0, "returned non-zero");
    float out_rms = rms(out, 4096);
    float in_rms = rms(in, 4096);
    ASSERT(out_rms > in_rms * 0.3f, "output RMS too low for identity shift");
    PASS();
}

static void test_pitch_shift_up(void) {
    TEST("pitch_shift: factor=1.5 produces output");
    float in[4096], out[4096];
    gen_sine(in, 4096, 300.0f, 24000);
    int rc = prosody_pitch_shift(in, out, 4096, 1.5f, 2048);
    ASSERT(rc == 0, "returned non-zero");
    ASSERT(rms(out, 4096) > 0.01f, "output is silence");
    PASS();
}

static void test_pitch_shift_down(void) {
    TEST("pitch_shift: factor=0.7 produces output");
    float in[4096], out[4096];
    gen_sine(in, 4096, 600.0f, 24000);
    int rc = prosody_pitch_shift(in, out, 4096, 0.7f, 2048);
    ASSERT(rc == 0, "returned non-zero");
    ASSERT(rms(out, 4096) > 0.01f, "output is silence");
    PASS();
}

static void test_pitch_shift_short(void) {
    TEST("pitch_shift: short buffer (<fft_size) handles gracefully");
    float in[512], out[512];
    memset(in, 0, sizeof(in));
    int rc = prosody_pitch_shift(in, out, 512, 1.2f, 2048);
    ASSERT(rc == 0 || rc == -1, "should return 0 or -1 for short buffer");
    PASS();
}

/* ── Time Stretch Tests ────────────────────────────────── */

static void test_time_stretch_identity(void) {
    TEST("time_stretch: factor=1.0 preserves length");
    float in[8000], out[16000];
    gen_sine(in, 8000, 440.0f, 24000);
    int n = prosody_time_stretch(in, 8000, out, 1.0f, 30.0f, 24000);
    ASSERT(n > 7000 && n < 9000, "output length should ~= input for factor 1.0");
    ASSERT(rms(out, n) > 0.1f, "output is too quiet");
    PASS();
}

static void test_time_stretch_slower(void) {
    TEST("time_stretch: factor=1.5 lengthens audio");
    float in[8000], out[16000];
    gen_sine(in, 8000, 440.0f, 24000);
    int n = prosody_time_stretch(in, 8000, out, 1.5f, 30.0f, 24000);
    ASSERT(n > 9000, "rate_factor 1.5 should produce more samples");
    PASS();
}

static void test_time_stretch_faster(void) {
    TEST("time_stretch: factor=0.75 shortens audio");
    float in[8000], out[16000];
    gen_sine(in, 8000, 440.0f, 24000);
    int n = prosody_time_stretch(in, 8000, out, 0.75f, 30.0f, 24000);
    ASSERT(n > 0 && n < 7000, "rate_factor 0.75 should produce fewer samples");
    PASS();
}

/* ── Volume Tests ──────────────────────────────────────── */

static void test_volume_gain(void) {
    TEST("volume: +6dB doubles amplitude");
    float audio[1024];
    gen_sine(audio, 1024, 440.0f, 24000);
    float before = rms(audio, 1024);
    prosody_volume(audio, 1024, 6.0f, 0.0f, 24000);
    float after = rms(audio, 1024);
    float ratio = after / before;
    ASSERT(ratio > 1.8f && ratio < 2.2f, "+6dB should roughly double RMS");
    PASS();
}

static void test_volume_cut(void) {
    TEST("volume: -6dB halves amplitude");
    float audio[1024];
    gen_sine(audio, 1024, 440.0f, 24000);
    float before = rms(audio, 1024);
    prosody_volume(audio, 1024, -6.0f, 0.0f, 24000);
    float after = rms(audio, 1024);
    float ratio = after / before;
    ASSERT(ratio > 0.4f && ratio < 0.6f, "-6dB should roughly halve RMS");
    PASS();
}

static void test_volume_zero(void) {
    TEST("volume: 0dB preserves signal");
    float audio[1024], ref[1024];
    gen_sine(audio, 1024, 440.0f, 24000);
    memcpy(ref, audio, sizeof(audio));
    prosody_volume(audio, 1024, 0.0f, 0.0f, 24000);
    float err = 0.0f;
    for (int i = 0; i < 1024; i++) err += fabsf(audio[i] - ref[i]);
    ASSERT(err < 0.001f, "0dB should not change signal");
    PASS();
}

static void test_volume_fade(void) {
    TEST("volume: fade-in ramps gain smoothly");
    float audio[4800];
    for (int i = 0; i < 4800; i++) audio[i] = 1.0f;
    prosody_volume(audio, 4800, 6.0f, 10.0f, 24000);
    ASSERT(fabsf(audio[0] - 1.0f) < 0.1f, "first sample should start near original");
    ASSERT(fabsf(audio[4799] - 2.0f) < 0.15f, "last sample should be at +6dB gain");
    PASS();
}

/* ── Soft Limiter Tests ────────────────────────────────── */

static void test_soft_limit_clamps(void) {
    TEST("soft_limit: reduces loud signal peak");
    float audio[1024];
    for (int i = 0; i < 1024; i++) audio[i] = 2.0f * sinf(2.0f * 3.14159f * 440 * i / 24000);
    float before_peak = 2.0f;
    prosody_soft_limit(audio, 1024, 0.95f, 12.0f);
    float peak = 0.0f;
    for (int i = 0; i < 1024; i++) if (fabsf(audio[i]) > peak) peak = fabsf(audio[i]);
    ASSERT(peak < before_peak, "peak should be reduced after soft limiting");
    PASS();
}

static void test_soft_limit_quiet_passthrough(void) {
    TEST("soft_limit: quiet signal passes through");
    float audio[1024], ref[1024];
    gen_sine(audio, 1024, 440.0f, 24000);
    for (int i = 0; i < 1024; i++) audio[i] *= 0.3f;
    memcpy(ref, audio, sizeof(audio));
    prosody_soft_limit(audio, 1024, 0.95f, 12.0f);
    float err = 0.0f;
    for (int i = 0; i < 1024; i++) err += fabsf(audio[i] - ref[i]);
    err /= 1024;
    ASSERT(err < 0.05f, "quiet signal should not be significantly changed");
    PASS();
}

/* ── Formant EQ Tests ──────────────────────────────────── */

static void test_formant_eq_create_destroy(void) {
    TEST("formant_eq: create/destroy lifecycle");
    BiquadCascade *eq = prosody_create_formant_eq(1.2f, 24000);
    ASSERT(eq != NULL, "create returned NULL");
    prosody_destroy_biquad(eq);
    prosody_destroy_biquad(NULL);
    PASS();
}

static void test_formant_eq_apply(void) {
    TEST("formant_eq: processing produces output");
    BiquadCascade *eq = prosody_create_formant_eq(1.3f, 24000);
    ASSERT(eq != NULL, "create returned NULL");
    float audio[2048];
    gen_sine(audio, 2048, 440.0f, 24000);
    int rc = prosody_apply_biquad(eq, audio, 2048);
    ASSERT(rc == 0, "apply returned non-zero");
    ASSERT(rms(audio, 2048) > 0.01f, "output is silence");
    prosody_destroy_biquad(eq);
    PASS();
}

static void test_formant_eq_null_safe(void) {
    TEST("formant_eq: NULL-safe apply");
    float audio[256];
    int rc = prosody_apply_biquad(NULL, audio, 256);
    ASSERT(rc == -1, "NULL eq should return -1");
    PASS();
}

/* ── Crossfade Tests ───────────────────────────────────── */

static void test_crossfade_blend(void) {
    TEST("crossfade: blends two segments");
    float a[480], b[480], out[480];
    for (int i = 0; i < 480; i++) { a[i] = 1.0f; b[i] = -1.0f; }
    prosody_crossfade(a, b, out, 480);
    ASSERT(fabsf(out[0] - 1.0f) < 0.05f, "start should be ~seg_a");
    ASSERT(fabsf(out[479] - (-1.0f)) < 0.05f, "end should be ~seg_b");
    float mid = out[240];
    ASSERT(fabsf(mid) < 0.2f, "midpoint should be near zero (blend of +1 and -1)");
    PASS();
}

/* ── Main ──────────────────────────────────────────────── */

int main(void) {
    printf("\n=== vdsp_prosody tests ===\n\n");

    test_pitch_shift_identity();
    test_pitch_shift_up();
    test_pitch_shift_down();
    test_pitch_shift_short();

    test_time_stretch_identity();
    test_time_stretch_slower();
    test_time_stretch_faster();

    test_volume_gain();
    test_volume_cut();
    test_volume_zero();
    test_volume_fade();

    test_soft_limit_clamps();
    test_soft_limit_quiet_passthrough();

    test_formant_eq_create_destroy();
    test_formant_eq_apply();
    test_formant_eq_null_safe();

    test_crossfade_blend();

    printf("\n  Results: %d passed, %d failed\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
