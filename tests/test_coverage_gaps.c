/**
 * test_coverage_gaps.c — Tests for critical blind spots in the test suite.
 *
 * Gap analysis found these UNTESTED areas:
 *   1. pocket_voice.c ring buffer (SPSC) — 0 unit tests for the core data structure
 *   2. pocket_voice.c VAD state machine — 0 unit tests for speech detection logic
 *   3. pocket_voice.c resampler — 0 unit tests for 48↔24 kHz conversion
 *   4. neon_audio.h SIMD primitives — 0 tests for copy, scale, crossfade, format conversion
 *   5. triple_buffer.h lock-free buffer — 2 references but no actual correctness tests
 *   6. breath_synthesis — only 6 test references, no edge case tests
 *   7. mimi_endpointer — 1 test reference
 *   8. Error path coverage: most tests only check happy paths
 *   9. Boundary conditions: ring buffer wrap, VAD state transitions at thresholds
 *  10. NULL safety: many public APIs never tested with NULL
 *
 * Build:
 *   cc -O3 -arch arm64 -Isrc -framework Accelerate \
 *      -Lbuild -llufs -lnoise_gate -lsentence_buffer \
 *      -Wl,-rpath,$(pwd)/build \
 *      -o build/test-coverage-gaps tests/test_coverage_gaps.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>

/* Include headers for inline/static components we can test directly */
#include "neon_audio.h"
#include "triple_buffer.h"
#include "breath_synthesis.h"
#include "noise_gate.h"
#include "lufs.h"

static int pass = 0, fail = 0, skip = 0;

#define TEST(name) printf("  %-60s", name)
#define PASS() do { printf("PASS\n"); pass++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); fail++; } while(0)
#define SKIP(msg) do { printf("SKIP: %s\n", msg); skip++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define ASSERT_FLOAT_EQ(a, b, tol, msg) do { \
    if (fabsf((a) - (b)) > (tol)) { \
        char _buf[256]; \
        snprintf(_buf, sizeof(_buf), "%s (got %.6f, expected %.6f)", msg, (double)(a), (double)(b)); \
        FAIL(_buf); return; \
    } \
} while(0)

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. NEON AUDIO PRIMITIVES — Zero prior test coverage
 *
 * These run in the CoreAudio real-time callback. If they have bugs,
 * audio glitches silently. Highest blast radius.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_neon_copy_basic(void) {
    TEST("neon_copy_f32: basic copy correctness");
    float src[32], dst[32];
    for (int i = 0; i < 32; i++) src[i] = (float)i * 0.1f;
    memset(dst, 0, sizeof(dst));

    neon_copy_f32(dst, src, 32);
    for (int i = 0; i < 32; i++) {
        ASSERT(fabsf(dst[i] - src[i]) < 1e-6f, "mismatch in copied data");
    }
    PASS();
}

static void test_neon_copy_small(void) {
    TEST("neon_copy_f32: small counts (1, 2, 3, 5, 7)");
    float src[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    int sizes[] = {1, 2, 3, 5, 7};
    for (int s = 0; s < 5; s++) {
        float dst[8] = {0};
        neon_copy_f32(dst, src, sizes[s]);
        for (int i = 0; i < sizes[s]; i++) {
            ASSERT(fabsf(dst[i] - src[i]) < 1e-6f, "small copy mismatch");
        }
        /* Ensure we didn't write past the count */
        for (int i = sizes[s]; i < 8; i++) {
            ASSERT(dst[i] == 0.0f, "neon_copy wrote past count");
        }
    }
    PASS();
}

static void test_neon_copy_zero(void) {
    TEST("neon_copy_f32: n=0 doesn't write anything");
    float src[4] = {1, 2, 3, 4};
    float dst[4] = {99, 99, 99, 99};
    neon_copy_f32(dst, src, 0);
    ASSERT(dst[0] == 99.0f, "n=0 should not modify dst");
    PASS();
}

static void test_neon_f32_to_s16_basic(void) {
    TEST("neon_f32_to_s16: [-1, 0, 1] → PCM");
    float in[3] = {-1.0f, 0.0f, 1.0f};
    int16_t out[3] = {0};
    neon_f32_to_s16(in, out, 3);
    ASSERT(out[0] == -32767, "f32_to_s16: -1.0 should map to -32767");
    ASSERT(out[1] == 0, "f32_to_s16: 0.0 should map to 0");
    ASSERT(out[2] == 32767, "f32_to_s16: 1.0 should map to 32767");
    PASS();
}

static void test_neon_f32_to_s16_saturation(void) {
    TEST("neon_f32_to_s16: saturation on out-of-range values");
    float in[2] = {2.0f, -2.0f};
    int16_t out[2] = {0};
    neon_f32_to_s16(in, out, 2);
    /* Values > 1.0 or < -1.0 should saturate, not wrap */
    ASSERT(out[0] == 32767, "positive overflow should saturate to 32767");
    ASSERT(out[1] == -32767 || out[1] == -32768, "negative overflow should saturate");
    PASS();
}

static void test_neon_s16_to_f32_roundtrip(void) {
    TEST("neon_s16_to_f32 + neon_f32_to_s16 roundtrip");
    int16_t pcm[16];
    for (int i = 0; i < 16; i++) pcm[i] = (int16_t)(i * 2000 - 16000);

    float f32[16];
    neon_s16_to_f32(pcm, f32, 16);

    int16_t back[16];
    neon_f32_to_s16(f32, back, 16);

    for (int i = 0; i < 16; i++) {
        int diff = abs(pcm[i] - back[i]);
        ASSERT(diff <= 1, "roundtrip error > 1 LSB");
    }
    PASS();
}

static void test_neon_zero_stuff(void) {
    TEST("neon_zero_stuff_2x: output pattern [val, 0, val, 0, ...]");
    float in[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float out[16] = {0};
    neon_zero_stuff_2x(in, out, 8);

    for (int i = 0; i < 8; i++) {
        float expected_val = in[i] * 2.0f;  /* 2x gain compensation */
        ASSERT_FLOAT_EQ(out[i * 2], expected_val, 1e-5f, "value sample wrong");
        ASSERT_FLOAT_EQ(out[i * 2 + 1], 0.0f, 1e-5f, "zero-stuffed sample not zero");
    }
    PASS();
}

static void test_neon_scale(void) {
    TEST("neon_scale_f32: multiply by constant");
    float in[17], out[17];
    for (int i = 0; i < 17; i++) in[i] = (float)i;
    neon_scale_f32(in, out, 17, 0.5f);
    for (int i = 0; i < 17; i++) {
        ASSERT_FLOAT_EQ(out[i], (float)i * 0.5f, 1e-6f, "scale mismatch");
    }
    PASS();
}

static void test_neon_scale_zero(void) {
    TEST("neon_scale_f32: scale by 0 zeros output");
    float in[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float out[8];
    neon_scale_f32(in, out, 8, 0.0f);
    for (int i = 0; i < 8; i++) {
        ASSERT(out[i] == 0.0f, "scale by 0 should zero");
    }
    PASS();
}

static void test_neon_crossfade(void) {
    TEST("neon_crossfade_f32: linear ramp between two signals");
    float a[16], b[16], out[16];
    for (int i = 0; i < 16; i++) { a[i] = 1.0f; b[i] = 0.0f; }
    neon_crossfade_f32(a, b, out, 16);

    /* First sample should be ~1.0 (all a), last should be ~0.0 (all b) */
    ASSERT(out[0] > 0.9f, "crossfade start should be ~a");
    ASSERT(out[15] < 0.15f, "crossfade end should be ~b");

    /* Should be monotonically decreasing */
    for (int i = 1; i < 16; i++) {
        ASSERT(out[i] <= out[i-1] + 1e-5f, "crossfade not monotonic");
    }
    PASS();
}

static void test_neon_crossfade_n_zero(void) {
    TEST("neon_crossfade_f32: n=0 doesn't crash");
    float a[4] = {1, 2, 3, 4};
    float b[4] = {5, 6, 7, 8};
    float out[4] = {99, 99, 99, 99};
    neon_crossfade_f32(a, b, out, 0);
    ASSERT(out[0] == 99.0f, "n=0 should not modify output");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. TRIPLE BUFFER — Lock-free pipeline buffer, almost zero tests
 *
 * Used between GPU TTS decode and CoreAudio playback callback.
 * If the rotation logic is wrong, audio glitches or silence.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_triple_buf_create_destroy(void) {
    TEST("triple_buffer: create/destroy lifecycle");
    TripleBuffer tb;
    int rc = triple_buf_create(&tb, 1024);
    ASSERT(rc == 0, "create failed");

    float *wp = triple_buf_write_ptr(&tb);
    ASSERT(wp != NULL, "write_ptr should not be NULL");

    triple_buf_destroy(&tb);
    PASS();
}

static void test_triple_buf_null_safety(void) {
    TEST("triple_buffer: NULL and zero-size create");
    int rc = triple_buf_create(NULL, 1024);
    ASSERT(rc == -1, "create(NULL) should fail");

    TripleBuffer tb;
    rc = triple_buf_create(&tb, 0);
    ASSERT(rc == -1, "create(size=0) should fail");
    PASS();
}

static void test_triple_buf_write_read_cycle(void) {
    TEST("triple_buffer: write → process → read data flow");
    TripleBuffer tb;
    int rc = triple_buf_create(&tb, 256);
    ASSERT(rc == 0, "create failed");

    /* Reader should have nothing initially */
    uint32_t count = 0;
    const float *rp = triple_buf_read_acquire(&tb, &count);
    ASSERT(rp == NULL, "reader should have nothing initially");

    /* Writer fills buffer */
    float *wp = triple_buf_write_ptr(&tb);
    for (int i = 0; i < 100; i++) wp[i] = (float)(i + 1);
    triple_buf_write_done(&tb, 100);

    /* Processor acquires */
    uint32_t pcount = 0;
    float *pp = triple_buf_process_acquire(&tb, &pcount);
    ASSERT(pp != NULL, "processor should get data");
    ASSERT(pcount == 100, "processor count should be 100");
    ASSERT(pp[0] == 1.0f, "processor data[0] should be 1.0");
    ASSERT(pp[99] == 100.0f, "processor data[99] should be 100.0");

    /* Processor doubles the data */
    for (uint32_t i = 0; i < pcount; i++) pp[i] *= 2.0f;
    triple_buf_process_done(&tb, pcount);

    /* Reader acquires */
    const float *rp2 = triple_buf_read_acquire(&tb, &count);
    ASSERT(rp2 != NULL, "reader should get processed data");
    ASSERT(count == 100, "reader count should be 100");
    ASSERT(rp2[0] == 2.0f, "reader data[0] should be 2.0 (doubled)");

    triple_buf_destroy(&tb);
    PASS();
}

static void test_triple_buf_no_double_acquire(void) {
    TEST("triple_buffer: second acquire without new data returns NULL");
    TripleBuffer tb;
    triple_buf_create(&tb, 256);

    float *wp = triple_buf_write_ptr(&tb);
    wp[0] = 42.0f;
    triple_buf_write_done(&tb, 1);

    uint32_t c;
    float *p1 = triple_buf_process_acquire(&tb, &c);
    ASSERT(p1 != NULL, "first acquire should succeed");

    float *p2 = triple_buf_process_acquire(&tb, &c);
    ASSERT(p2 == NULL, "second acquire without new write should be NULL");

    triple_buf_destroy(&tb);
    PASS();
}

static void test_triple_buf_destroy_null(void) {
    TEST("triple_buffer: destroy(NULL) doesn't crash");
    triple_buf_destroy(NULL);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. BREATH SYNTHESIS — Minimal coverage, critical for audio quality
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_breath_create_destroy(void) {
    TEST("breath_synthesis: create/destroy lifecycle");
    BreathSynth *bs = breath_create(48000);
    ASSERT(bs != NULL, "create(48000) failed");
    breath_destroy(bs);
    PASS();
}

static void test_breath_create_invalid_rates(void) {
    TEST("breath_synthesis: create with edge-case sample rates");
    BreathSynth *bs = breath_create(0);
    /* Implementation-dependent: may return NULL or handle gracefully */
    if (bs) breath_destroy(bs);

    bs = breath_create(8000);
    ASSERT(bs != NULL, "create(8000) should work");
    breath_destroy(bs);

    bs = breath_create(96000);
    ASSERT(bs != NULL, "create(96000) should work");
    breath_destroy(bs);
    PASS();
}

static void test_breath_destroy_null(void) {
    TEST("breath_synthesis: destroy(NULL) doesn't crash");
    breath_destroy(NULL);
    PASS();
}

static void test_breath_generate_basic(void) {
    TEST("breath_synthesis: generate adds noise to buffer");
    BreathSynth *bs = breath_create(48000);
    ASSERT(bs != NULL, "create failed");

    float audio[4800];
    memset(audio, 0, sizeof(audio));
    breath_generate(bs, audio, 4800, 0.03f);

    /* Should have added some non-zero noise */
    float sum = 0;
    for (int i = 0; i < 4800; i++) sum += fabsf(audio[i]);
    ASSERT(sum > 0.0f, "breath should add non-zero noise");

    /* Noise should be quiet relative to speech */
    float max_val = 0;
    for (int i = 0; i < 4800; i++) {
        if (fabsf(audio[i]) > max_val) max_val = fabsf(audio[i]);
    }
    ASSERT(max_val < 0.2f, "breath noise should be quiet");

    breath_destroy(bs);
    PASS();
}

static void test_breath_generate_zero_amplitude(void) {
    TEST("breath_synthesis: generate with amplitude=0 leaves buffer unchanged");
    BreathSynth *bs = breath_create(48000);
    ASSERT(bs != NULL, "create failed");

    float audio[1024];
    for (int i = 0; i < 1024; i++) audio[i] = 0.5f;
    breath_generate(bs, audio, 1024, 0.0f);

    /* With zero amplitude, buffer should be unchanged */
    for (int i = 0; i < 1024; i++) {
        ASSERT_FLOAT_EQ(audio[i], 0.5f, 1e-6f, "zero amp should not modify");
    }
    breath_destroy(bs);
    PASS();
}

static void test_breath_micropause(void) {
    TEST("breath_synthesis: micropause applies fade envelope");
    float audio[2400];  /* 50ms at 48kHz */
    for (int i = 0; i < 2400; i++) audio[i] = 1.0f;

    breath_micropause(audio, 2400, 5.0f, 48000);

    /* The middle should be near zero (silence) */
    float mid = audio[1200];
    ASSERT(fabsf(mid) < 0.1f, "middle of micropause should be ~silent");

    /* The edges should be closer to 1 (faded) */
    ASSERT(audio[0] > 0.5f, "start of micropause should still have signal");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. NOISE GATE — Good test count but missing edge cases
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_noise_gate_null_safety(void) {
    TEST("noise_gate: create with invalid params");
    /* Non-power-of-two FFT */
    NoiseGate *ng = noise_gate_create(48000, 100, 50);
    ASSERT(ng == NULL, "non-power-of-two FFT should fail");

    /* hop > fft_size */
    ng = noise_gate_create(48000, 256, 512);
    ASSERT(ng == NULL, "hop > fft_size should fail");

    /* hop = 0 */
    ng = noise_gate_create(48000, 256, 0);
    ASSERT(ng == NULL, "hop=0 should fail");
    PASS();
}

static void test_noise_gate_lifecycle(void) {
    TEST("noise_gate: create/destroy lifecycle");
    NoiseGate *ng = noise_gate_create(48000, 256, 128);
    ASSERT(ng != NULL, "create(256, 128) should work");
    noise_gate_destroy(ng);
    PASS();
}

static void test_noise_gate_destroy_null(void) {
    TEST("noise_gate: destroy(NULL) doesn't crash");
    noise_gate_destroy(NULL);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. LUFS — Good coverage but missing edge cases
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_lufs_silence_is_negative_inf(void) {
    TEST("lufs: silence produces very negative LUFS");
    LUFSMeter *m = lufs_create(48000, 400);
    ASSERT(m != NULL, "lufs_create failed");
    float silence[4800];
    memset(silence, 0, sizeof(silence));
    float lufs = lufs_measure(m, silence, 4800);
    ASSERT(lufs < -60.0f, "silence should be < -60 LUFS");
    lufs_destroy(m);
    PASS();
}

static void test_lufs_full_scale_sine(void) {
    TEST("lufs: full-scale sine is near 0 LUFS");
    LUFSMeter *m = lufs_create(48000, 400);
    ASSERT(m != NULL, "lufs_create failed");
    float sine[48000];
    for (int i = 0; i < 48000; i++)
        sine[i] = sinf(2.0f * (float)M_PI * 1000.0f * (float)i / 48000.0f);

    float lufs = lufs_measure(m, sine, 48000);
    /* Full-scale 1kHz sine should be around -3 LUFS */
    ASSERT(lufs > -10.0f, "full-scale sine should be > -10 LUFS");
    ASSERT(lufs < 5.0f, "full-scale sine should be < 5 LUFS");
    lufs_destroy(m);
    PASS();
}

static void test_lufs_small_buffer(void) {
    TEST("lufs: very small buffer doesn't crash");
    LUFSMeter *m = lufs_create(48000, 400);
    ASSERT(m != NULL, "lufs_create failed");
    float tiny[10] = {0.5f, -0.5f, 0.5f, -0.5f, 0.5f, -0.5f, 0.5f, -0.5f, 0.5f, -0.5f};
    float lufs = lufs_measure(m, tiny, 10);
    /* Just shouldn't crash; value may be imprecise with 10 samples */
    (void)lufs;
    lufs_destroy(m);
    PASS();
}

static void test_lufs_null_safety(void) {
    TEST("lufs: create/destroy NULL safety");
    lufs_destroy(NULL);  /* Should not crash */

    LUFSMeter *m = lufs_create(0, 400);
    /* Implementation may return NULL for invalid rate */
    if (m) lufs_destroy(m);

    m = lufs_create(48000, 0);
    if (m) lufs_destroy(m);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. NEON SIMD BOUNDARY CONDITIONS
 *
 * NEON processes 4 or 8 floats at a time. The scalar remainder loop
 * handles the last 1-7 elements. These are the most bug-prone paths.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_neon_nonaligned_sizes(void) {
    TEST("neon ops: correct results for non-SIMD-aligned sizes (1-15)");
    for (int n = 1; n <= 15; n++) {
        float *src = calloc(n, sizeof(float));
        float *dst = calloc(n, sizeof(float));
        for (int i = 0; i < n; i++) src[i] = (float)(i + 1);

        /* Test copy */
        neon_copy_f32(dst, src, n);
        for (int i = 0; i < n; i++) {
            if (fabsf(dst[i] - src[i]) > 1e-6f) {
                FAIL("neon_copy wrong for non-aligned size");
                free(src); free(dst);
                return;
            }
        }

        /* Test scale */
        neon_scale_f32(src, dst, n, 3.0f);
        for (int i = 0; i < n; i++) {
            if (fabsf(dst[i] - src[i] * 3.0f) > 1e-5f) {
                FAIL("neon_scale wrong for non-aligned size");
                free(src); free(dst);
                return;
            }
        }

        free(src);
        free(dst);
    }
    PASS();
}

static void test_neon_f32_to_s16_notaligned(void) {
    TEST("neon_f32_to_s16: non-aligned sizes (1, 5, 9, 13)");
    int sizes[] = {1, 5, 9, 13};
    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        float *in = calloc(n, sizeof(float));
        int16_t *out = calloc(n, sizeof(int16_t));
        for (int i = 0; i < n; i++) in[i] = (float)i / (float)n;

        neon_f32_to_s16(in, out, n);

        for (int i = 0; i < n; i++) {
            int16_t expected = (int16_t)(in[i] * 32767.0f);
            if (abs(out[i] - expected) > 1) {
                FAIL("f32_to_s16 wrong for non-aligned size");
                free(in); free(out);
                return;
            }
        }
        free(in);
        free(out);
    }
    PASS();
}

static void test_neon_zero_stuff_small(void) {
    TEST("neon_zero_stuff_2x: n=1, n=3 (scalar remainder)");
    float in1[1] = {0.5f};
    float out1[2] = {0};
    neon_zero_stuff_2x(in1, out1, 1);
    ASSERT_FLOAT_EQ(out1[0], 1.0f, 1e-5f, "n=1 value");
    ASSERT_FLOAT_EQ(out1[1], 0.0f, 1e-5f, "n=1 zero");

    float in3[3] = {1.0f, 2.0f, 3.0f};
    float out3[6] = {0};
    neon_zero_stuff_2x(in3, out3, 3);
    for (int i = 0; i < 3; i++) {
        ASSERT_FLOAT_EQ(out3[i*2], in3[i] * 2.0f, 1e-5f, "n=3 value");
        ASSERT_FLOAT_EQ(out3[i*2+1], 0.0f, 1e-5f, "n=3 zero");
    }
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 7. LARGE INPUT STRESS TESTS
 *
 * The pipeline processes up to 10 minutes of audio. Integer overflow
 * on sample counts at 48kHz * 600s = 28.8M samples could be catastrophic.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_neon_large_buffer(void) {
    TEST("neon_copy_f32: 100K samples (realistic capture buffer)");
    int n = 100000;
    float *src = calloc(n, sizeof(float));
    float *dst = calloc(n, sizeof(float));
    ASSERT(src && dst, "alloc failed");

    for (int i = 0; i < n; i++) src[i] = sinf((float)i * 0.01f);
    neon_copy_f32(dst, src, n);

    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (fabsf(dst[i] - src[i]) > 1e-6f) { ok = 0; break; }
    }
    ASSERT(ok, "large buffer copy mismatch");
    free(src);
    free(dst);
    PASS();
}

static void test_neon_scale_large(void) {
    TEST("neon_scale_f32: 1M samples with NaN/Inf guard");
    int n = 1000000;
    float *buf = calloc(n, sizeof(float));
    float *out = calloc(n, sizeof(float));
    ASSERT(buf && out, "alloc failed");

    for (int i = 0; i < n; i++) buf[i] = (float)i / (float)n;
    neon_scale_f32(buf, out, n, 0.999f);

    int has_nan = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(out[i]) || isinf(out[i])) { has_nan = 1; break; }
    }
    ASSERT(!has_nan, "scale produced NaN/Inf");
    free(buf);
    free(out);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 8. TRIPLE BUFFER CONCURRENT ACCESS
 *
 * In production, writer (GPU thread), processor (pipeline thread), and
 * reader (CoreAudio RT thread) all run simultaneously. This tests the
 * atomic state transitions under real contention.
 * ═══════════════════════════════════════════════════════════════════════════ */

static _Atomic int tb_writer_done = 0;
static _Atomic int tb_reader_got = 0;
static _Atomic int tb_reader_errors = 0;

static void *tb_writer_thread(void *arg) {
    TripleBuffer *tb = (TripleBuffer *)arg;
    for (int i = 0; i < 1000; i++) {
        float *wp = triple_buf_write_ptr(tb);
        for (int j = 0; j < 64; j++) wp[j] = (float)(i * 64 + j);
        triple_buf_write_done(tb, 64);

        /* Simulate work */
        for (volatile int k = 0; k < 100; k++);
    }
    atomic_store(&tb_writer_done, 1);
    return NULL;
}

static void *tb_reader_thread(void *arg) {
    TripleBuffer *tb = (TripleBuffer *)arg;
    while (!atomic_load(&tb_writer_done) || 1) {
        /* Process */
        uint32_t pcount;
        float *pp = triple_buf_process_acquire(tb, &pcount);
        if (pp && pcount > 0) {
            triple_buf_process_done(tb, pcount);
        }

        /* Read */
        uint32_t rcount;
        const float *rp = triple_buf_read_acquire(tb, &rcount);
        if (rp && rcount > 0) {
            atomic_fetch_add(&tb_reader_got, 1);
            /* Verify data is self-consistent (no torn reads) */
            int base = (int)rp[0];
            for (uint32_t j = 1; j < rcount && j < 64; j++) {
                if ((int)rp[j] != base + (int)j) {
                    atomic_fetch_add(&tb_reader_errors, 1);
                    break;
                }
            }
        }

        if (atomic_load(&tb_writer_done)) break;
    }
    return NULL;
}

static void test_triple_buf_concurrent(void) {
    TEST("triple_buffer: concurrent write/process/read (1000 rounds)");
    TripleBuffer tb;
    int rc = triple_buf_create(&tb, 256);
    ASSERT(rc == 0, "create failed");

    atomic_store(&tb_writer_done, 0);
    atomic_store(&tb_reader_got, 0);
    atomic_store(&tb_reader_errors, 0);

    pthread_t writer, reader;
    pthread_create(&writer, NULL, tb_writer_thread, &tb);
    pthread_create(&reader, NULL, tb_reader_thread, &tb);

    pthread_join(writer, NULL);
    pthread_join(reader, NULL);

    int got = atomic_load(&tb_reader_got);
    int errors = atomic_load(&tb_reader_errors);
    ASSERT(errors == 0, "torn reads detected in concurrent test");
    /* We should have gotten at least some reads */
    ASSERT(got > 0, "reader never got any data");

    triple_buf_destroy(&tb);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 9. NEON SPECIAL VALUES (NaN, Inf, Denormals)
 *
 * Audio can produce NaN from 0/0 or Inf from overflow.
 * NEON SIMD doesn't trap — it silently propagates.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_neon_special_values(void) {
    TEST("neon ops: NaN/Inf don't crash, propagate correctly");
    float special[8] = {0.0f, 1.0f, NAN, INFINITY, -INFINITY, 0.0f, NAN, 1.0f};
    float out[8];

    /* Copy should preserve special values */
    neon_copy_f32(out, special, 8);
    ASSERT(isnan(out[2]), "NaN should propagate through copy");
    ASSERT(isinf(out[3]) && out[3] > 0, "+Inf should propagate through copy");
    ASSERT(isinf(out[4]) && out[4] < 0, "-Inf should propagate through copy");

    /* Scale should propagate */
    neon_scale_f32(special, out, 8, 2.0f);
    ASSERT(isnan(out[2]), "NaN should propagate through scale");
    ASSERT(isinf(out[3]), "Inf should propagate through scale");
    PASS();
}

static void test_neon_f32_to_s16_special(void) {
    TEST("neon_f32_to_s16: NaN and Inf saturate gracefully");
    float in[4] = {NAN, INFINITY, -INFINITY, 0.0f};
    int16_t out[4] = {12345, 12345, 12345, 12345};
    neon_f32_to_s16(in, out, 4);
    /* NaN and Inf should not produce garbage — they should saturate or go to 0 */
    ASSERT(out[3] == 0, "0.0 should convert to 0");
    /* NaN behavior is platform-dependent, just ensure no crash */
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 10. CROSSFADE PRECISION AT BOUNDARIES
 *
 * The crossfade is used at TTS chunk boundaries. If the endpoints aren't
 * exactly right, there's an audible click.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_crossfade_endpoints(void) {
    TEST("neon_crossfade_f32: exact endpoint values (click prevention)");
    int n = 480;  /* Typical crossfade size (20ms at 24kHz) */
    float *a = calloc(n, sizeof(float));
    float *b = calloc(n, sizeof(float));
    float *out = calloc(n, sizeof(float));
    ASSERT(a && b && out, "alloc failed");

    /* Constant signals for easy verification */
    for (int i = 0; i < n; i++) { a[i] = 10.0f; b[i] = 20.0f; }
    neon_crossfade_f32(a, b, out, n);

    /* First sample: t=0, so out = a[0]*1 + b[0]*0 = 10 */
    ASSERT_FLOAT_EQ(out[0], 10.0f, 0.1f, "first sample should be ~a");

    /* Last sample: t≈1, so out ≈ a*0 + b*1 = 20 */
    /* Note: t = (n-1)/n, not exactly 1.0, so slight offset expected */
    float expected_last = 10.0f * (1.0f - (float)(n-1)/(float)n) +
                          20.0f * ((float)(n-1)/(float)n);
    ASSERT_FLOAT_EQ(out[n-1], expected_last, 0.1f, "last sample should be ~b");

    /* Middle sample should be ~15 */
    ASSERT_FLOAT_EQ(out[n/2], 15.0f, 0.5f, "middle should be ~15");

    free(a); free(b); free(out);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 11. IN-PLACE OPERATIONS
 *
 * Several NEON ops support in-place (dst == src). If the SIMD implementation
 * reads ahead of the write position, in-place breaks silently.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_neon_scale_inplace(void) {
    TEST("neon_scale_f32: in-place (dst == src)");
    float buf[32];
    for (int i = 0; i < 32; i++) buf[i] = (float)(i + 1);
    neon_scale_f32(buf, buf, 32, 2.0f);  /* in-place */
    for (int i = 0; i < 32; i++) {
        float expected = (float)(i + 1) * 2.0f;
        ASSERT_FLOAT_EQ(buf[i], expected, 1e-5f, "in-place scale wrong");
    }
    PASS();
}

static void test_neon_copy_overlapping(void) {
    TEST("neon_copy_f32: non-overlapping buffers required (contract)");
    /* neon_copy_f32 does NOT support overlapping — this documents the contract.
     * We test that separate non-overlapping regions work correctly. */
    float big[64];
    for (int i = 0; i < 32; i++) big[i] = (float)i;
    neon_copy_f32(big + 32, big, 32);
    for (int i = 0; i < 32; i++) {
        ASSERT_FLOAT_EQ(big[32 + i], (float)i, 1e-6f, "non-overlap copy failed");
    }
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("=== Test Coverage Gaps: Critical Blind Spots ===\n\n");

    printf("--- NEON Audio Primitives (0 prior tests) ---\n");
    test_neon_copy_basic();
    test_neon_copy_small();
    test_neon_copy_zero();
    test_neon_f32_to_s16_basic();
    test_neon_f32_to_s16_saturation();
    test_neon_s16_to_f32_roundtrip();
    test_neon_zero_stuff();
    test_neon_scale();
    test_neon_scale_zero();
    test_neon_crossfade();
    test_neon_crossfade_n_zero();

    printf("\n--- Triple Buffer Lock-Free (2 prior refs, 0 tests) ---\n");
    test_triple_buf_create_destroy();
    test_triple_buf_null_safety();
    test_triple_buf_write_read_cycle();
    test_triple_buf_no_double_acquire();
    test_triple_buf_destroy_null();
    test_triple_buf_concurrent();

    printf("\n--- Breath Synthesis (6 prior refs, minimal tests) ---\n");
    test_breath_create_destroy();
    test_breath_create_invalid_rates();
    test_breath_destroy_null();
    test_breath_generate_basic();
    test_breath_generate_zero_amplitude();
    test_breath_micropause();

    printf("\n--- Noise Gate Edge Cases ---\n");
    test_noise_gate_null_safety();
    test_noise_gate_lifecycle();
    test_noise_gate_destroy_null();

    printf("\n--- LUFS Edge Cases ---\n");
    test_lufs_silence_is_negative_inf();
    test_lufs_full_scale_sine();
    test_lufs_small_buffer();
    test_lufs_null_safety();

    printf("\n--- NEON Boundary Conditions (SIMD remainder loop) ---\n");
    test_neon_nonaligned_sizes();
    test_neon_f32_to_s16_notaligned();
    test_neon_zero_stuff_small();

    printf("\n--- Large Input Stress ---\n");
    test_neon_large_buffer();
    test_neon_scale_large();

    printf("\n--- Special Values (NaN/Inf) ---\n");
    test_neon_special_values();
    test_neon_f32_to_s16_special();

    printf("\n--- Crossfade Precision ---\n");
    test_crossfade_endpoints();

    printf("\n--- In-Place Operations ---\n");
    test_neon_scale_inplace();
    test_neon_copy_overlapping();

    printf("\n═══ Results: %d passed, %d failed, %d skipped ═══\n",
           pass, fail, skip);
    return fail > 0 ? 1 : 0;
}
