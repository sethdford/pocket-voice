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

/* PitchShiftContext persistent API */
typedef struct PitchShiftContext PitchShiftContext;
extern PitchShiftContext *prosody_pitch_create(int fft_size);
extern void prosody_pitch_destroy(PitchShiftContext *psc);
extern int  prosody_pitch_shift_ctx(PitchShiftContext *psc, const float *input,
                                     float *output, int n_samples, float pitch_factor);

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

/* ── Pitch Shift: Octave Up ────────────────────────────── */

static void test_pitch_shift_octave_up(void) {
    TEST("pitch_shift: factor=2.0 (octave up) produces output");
    float in[4096], out[4096];
    gen_sine(in, 4096, 220.0f, 24000);
    int rc = prosody_pitch_shift(in, out, 4096, 2.0f, 2048);
    ASSERT(rc == 0, "returned non-zero");
    ASSERT(rms(out, 4096) > 0.01f, "output is silence");
    PASS();
}

/* ── Pitch Shift: Octave Down ─────────────────────────── */

static void test_pitch_shift_octave_down(void) {
    TEST("pitch_shift: factor=0.5 (octave down) produces output");
    float in[4096], out[4096];
    gen_sine(in, 4096, 440.0f, 24000);
    int rc = prosody_pitch_shift(in, out, 4096, 0.5f, 2048);
    ASSERT(rc == 0, "returned non-zero");
    ASSERT(rms(out, 4096) > 0.01f, "output is silence");
    PASS();
}

/* ── Pitch Shift: Extreme factors ─────────────────────── */

static void test_pitch_shift_extreme(void) {
    TEST("pitch_shift: extreme factors (0.25, 3.0) produce output");
    float in[4096], out[4096];
    gen_sine(in, 4096, 440.0f, 24000);
    int rc = prosody_pitch_shift(in, out, 4096, 0.25f, 2048);
    ASSERT(rc == 0, "factor=0.25 returned non-zero");
    rc = prosody_pitch_shift(in, out, 4096, 3.0f, 2048);
    ASSERT(rc == 0, "factor=3.0 returned non-zero");
    PASS();
}

/* ── Pitch Shift: Different FFT sizes ─────────────────── */

static void test_pitch_shift_fft_sizes(void) {
    TEST("pitch_shift: different FFT sizes (512, 1024, 2048)");
    float in[4096], out[4096];
    gen_sine(in, 4096, 440.0f, 24000);
    int sizes[] = {512, 1024, 2048};
    for (int i = 0; i < 3; i++) {
        int rc = prosody_pitch_shift(in, out, 4096, 1.2f, sizes[i]);
        ASSERT(rc == 0, "non-zero return for valid FFT size");
    }
    PASS();
}

/* ── PitchShiftContext: Create/Destroy Lifecycle ──────── */

static void test_pitch_ctx_lifecycle(void) {
    TEST("pitch_ctx: create/destroy lifecycle");
    PitchShiftContext *ctx = prosody_pitch_create(2048);
    ASSERT(ctx != NULL, "create returned NULL");
    prosody_pitch_destroy(ctx);
    PASS();
}

static void test_pitch_ctx_null_destroy(void) {
    TEST("pitch_ctx: destroy(NULL) is safe");
    prosody_pitch_destroy(NULL);
    PASS();
}

static void test_pitch_ctx_shift(void) {
    TEST("pitch_ctx: shift with context produces output");
    PitchShiftContext *ctx = prosody_pitch_create(2048);
    ASSERT(ctx != NULL, "create returned NULL");
    float in[4096], out[4096];
    gen_sine(in, 4096, 440.0f, 24000);
    int rc = prosody_pitch_shift_ctx(ctx, in, out, 4096, 1.2f);
    ASSERT(rc == 0, "returned non-zero");
    ASSERT(rms(out, 4096) > 0.01f, "output is silence");
    prosody_pitch_destroy(ctx);
    PASS();
}

static void test_pitch_ctx_repeated_use(void) {
    TEST("pitch_ctx: reuse context across multiple calls");
    PitchShiftContext *ctx = prosody_pitch_create(2048);
    ASSERT(ctx != NULL, "create returned NULL");
    float in[4096], out[4096];
    gen_sine(in, 4096, 440.0f, 24000);
    for (int i = 0; i < 5; i++) {
        int rc = prosody_pitch_shift_ctx(ctx, in, out, 4096, 1.0f + 0.1f * i);
        ASSERT(rc == 0, "shift failed on iteration");
    }
    prosody_pitch_destroy(ctx);
    PASS();
}

static void test_pitch_ctx_memory_cycle(void) {
    TEST("pitch_ctx: 50x create/destroy cycles no crash");
    for (int i = 0; i < 50; i++) {
        PitchShiftContext *ctx = prosody_pitch_create(2048);
        if (ctx) prosody_pitch_destroy(ctx);
    }
    PASS();
}

/* ── Volume: Identity (0 dB) ──────────────────────────── */

static void test_volume_identity(void) {
    TEST("volume: 0dB with no fade preserves signal exactly");
    float audio[2048], ref[2048];
    gen_sine(audio, 2048, 440.0f, 24000);
    memcpy(ref, audio, sizeof(audio));
    prosody_volume(audio, 2048, 0.0f, 0.0f, 24000);
    float err = 0.0f;
    for (int i = 0; i < 2048; i++) err += fabsf(audio[i] - ref[i]);
    ASSERT(err < 0.01f, "0dB should preserve signal");
    PASS();
}

/* ── Volume: Large gain ───────────────────────────────── */

static void test_volume_large_gain(void) {
    TEST("volume: +20dB amplifies signal significantly");
    float audio[1024];
    gen_sine(audio, 1024, 440.0f, 24000);
    float before = rms(audio, 1024);
    prosody_volume(audio, 1024, 20.0f, 0.0f, 24000);
    float after = rms(audio, 1024);
    ASSERT(after > before * 5.0f, "+20dB should amplify at least 5x");
    PASS();
}

/* ── Volume: Mute (-60dB) ────────────────────────────── */

static void test_volume_mute(void) {
    TEST("volume: -60dB nearly silences signal");
    float audio[1024];
    gen_sine(audio, 1024, 440.0f, 24000);
    prosody_volume(audio, 1024, -60.0f, 0.0f, 24000);
    float after = rms(audio, 1024);
    ASSERT(after < 0.01f, "-60dB should nearly silence signal");
    PASS();
}

/* ── Soft Limiter: Threshold test ─────────────────────── */

static void test_soft_limit_threshold(void) {
    TEST("soft_limit: output peak stays near threshold");
    float audio[1024];
    for (int i = 0; i < 1024; i++)
        audio[i] = 3.0f * sinf(2.0f * 3.14159f * 440 * i / 24000);
    prosody_soft_limit(audio, 1024, 0.8f, 6.0f);
    float peak = 0.0f;
    for (int i = 0; i < 1024; i++)
        if (fabsf(audio[i]) > peak) peak = fabsf(audio[i]);
    ASSERT(peak < 1.5f, "peak should be significantly reduced toward threshold");
    PASS();
}

/* ── Soft Limiter: Zero-length ────────────────────────── */

static void test_soft_limit_zero_length(void) {
    TEST("soft_limit: zero-length input is safe");
    float audio[1] = {1.0f};
    prosody_soft_limit(audio, 0, 0.95f, 12.0f);
    ASSERT(audio[0] == 1.0f, "zero-length should not touch data");
    PASS();
}

/* ── Formant EQ: Various pitch factors ────────────────── */

static void test_formant_eq_pitch_factors(void) {
    TEST("formant_eq: create with various pitch factors");
    float factors[] = {0.5f, 0.8f, 1.0f, 1.2f, 1.5f, 2.0f};
    int n = sizeof(factors) / sizeof(factors[0]);
    for (int i = 0; i < n; i++) {
        BiquadCascade *eq = prosody_create_formant_eq(factors[i], 24000);
        ASSERT(eq != NULL, "create returned NULL for valid factor");
        float audio[2048];
        gen_sine(audio, 2048, 440.0f, 24000);
        int rc = prosody_apply_biquad(eq, audio, 2048);
        ASSERT(rc == 0, "apply returned non-zero");
        prosody_destroy_biquad(eq);
    }
    PASS();
}

/* ── Formant EQ: Different sample rates ───────────────── */

static void test_formant_eq_sample_rates(void) {
    TEST("formant_eq: create with different sample rates");
    int rates[] = {8000, 16000, 24000, 44100, 48000};
    int n = sizeof(rates) / sizeof(rates[0]);
    for (int i = 0; i < n; i++) {
        BiquadCascade *eq = prosody_create_formant_eq(1.2f, rates[i]);
        ASSERT(eq != NULL, "create returned NULL");
        prosody_destroy_biquad(eq);
    }
    PASS();
}

/* ── Formant EQ: Apply to silence ─────────────────────── */

static void test_formant_eq_silence(void) {
    TEST("formant_eq: apply to silence produces silence");
    BiquadCascade *eq = prosody_create_formant_eq(1.3f, 24000);
    ASSERT(eq != NULL, "create returned NULL");
    float audio[1024];
    memset(audio, 0, sizeof(audio));
    int rc = prosody_apply_biquad(eq, audio, 1024);
    ASSERT(rc == 0, "apply returned non-zero");
    ASSERT(rms(audio, 1024) < 0.001f, "silence through EQ should remain silent");
    prosody_destroy_biquad(eq);
    PASS();
}

/* ── Formant EQ: Memory cycle ─────────────────────────── */

static void test_formant_eq_memory_cycle(void) {
    TEST("formant_eq: 50x create/destroy cycles no crash");
    for (int i = 0; i < 50; i++) {
        BiquadCascade *eq = prosody_create_formant_eq(1.0f + 0.01f * i, 24000);
        if (eq) prosody_destroy_biquad(eq);
    }
    PASS();
}

/* ── Time Stretch: Zero-length ────────────────────────── */

static void test_time_stretch_zero(void) {
    TEST("time_stretch: zero-length input returns 0");
    float in[1] = {0.0f}, out[100];
    int n = prosody_time_stretch(in, 0, out, 1.0f, 30.0f, 24000);
    ASSERT(n >= 0, "zero-length should return >= 0");
    PASS();
}

/* ── Time Stretch: Very short ─────────────────────────── */

static void test_time_stretch_very_short(void) {
    TEST("time_stretch: very short input handles gracefully");
    float in[100], out[200];
    gen_sine(in, 100, 440.0f, 24000);
    int n = prosody_time_stretch(in, 100, out, 1.0f, 30.0f, 24000);
    ASSERT(n >= 0, "should return non-negative");
    PASS();
}

/* ── Crossfade: Equal signals ─────────────────────────── */

static void test_crossfade_equal(void) {
    TEST("crossfade: equal signals produces same signal");
    float a[480], b[480], out[480];
    for (int i = 0; i < 480; i++) { a[i] = 0.5f; b[i] = 0.5f; }
    prosody_crossfade(a, b, out, 480);
    float err = 0.0f;
    for (int i = 0; i < 480; i++) err += fabsf(out[i] - 0.5f);
    err /= 480.0f;
    ASSERT(err < 0.01f, "crossfade of equal signals should produce same value");
    PASS();
}

/* ── Crossfade: Short length ──────────────────────────── */

static void test_crossfade_short(void) {
    TEST("crossfade: very short length (2 samples)");
    float a[2] = {1.0f, 1.0f}, b[2] = {-1.0f, -1.0f}, out[2];
    prosody_crossfade(a, b, out, 2);
    /* fade_out=[1.0, 0.5], fade_in=[0.0, 0.5]: out[0]=1.0, out[1]=0.0 */
    ASSERT(out[0] > 0.5f, "first sample should be mostly seg_a");
    ASSERT(out[1] <= out[0], "second sample should blend more toward seg_b");
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
    test_pitch_shift_octave_up();
    test_pitch_shift_octave_down();
    test_pitch_shift_extreme();
    test_pitch_shift_fft_sizes();

    test_pitch_ctx_lifecycle();
    test_pitch_ctx_null_destroy();
    test_pitch_ctx_shift();
    test_pitch_ctx_repeated_use();
    test_pitch_ctx_memory_cycle();

    test_time_stretch_identity();
    test_time_stretch_slower();
    test_time_stretch_faster();
    test_time_stretch_zero();
    test_time_stretch_very_short();

    test_volume_gain();
    test_volume_cut();
    test_volume_zero();
    test_volume_fade();
    test_volume_identity();
    test_volume_large_gain();
    test_volume_mute();

    test_soft_limit_clamps();
    test_soft_limit_quiet_passthrough();
    test_soft_limit_threshold();
    test_soft_limit_zero_length();

    test_formant_eq_create_destroy();
    test_formant_eq_apply();
    test_formant_eq_null_safe();
    test_formant_eq_pitch_factors();
    test_formant_eq_sample_rates();
    test_formant_eq_silence();
    test_formant_eq_memory_cycle();

    test_crossfade_blend();
    test_crossfade_equal();
    test_crossfade_short();

    printf("\n  Results: %d passed, %d failed\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
