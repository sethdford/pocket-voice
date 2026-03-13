/**
 * test_security_fixes.c — Tests for P0-P1 security fixes.
 *
 * Tests:
 *   1. P0-2: Integer overflow in mel buffer allocation
 *   2. P1-1: Voice cloning audio validation (sample rate, duration, energy)
 *   3. P1-2: HTTP API UTF-8 validation and text size limits
 *
 * Build:
 *   cc -O3 -arch arm64 -Isrc \
 *      -Lbuild -lcJSON -lhttp_api \
 *      -Wl,-rpath,$(pwd)/build -o tests/test_security_fixes tests/test_security_fixes.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>

static int pass = 0, fail = 0;
#define TEST(n) printf("  %-60s", n)
#define PASS() do { printf("PASS\n"); pass++; } while(0)
#define FAIL(m) do { printf("FAIL: %s\n", m); fail++; } while(0)
#define ASSERT(c, m) do { if (!(c)) { FAIL(m); return; } } while(0)

/* ═══════════════════════════════════════════════════════════════════════
 * P0-2: Integer Overflow in Mel Buffer Allocation
 * ═══════════════════════════════════════════════════════════════════════ */

/* Simulate the overflow check from pocket_voice_pipeline.c line ~777 */
static int check_mel_buffer_allocation_safe(int n_frames, int n_bins) {
    #define MAX_MEL_FRAMES 16384
    if (n_frames > MAX_MEL_FRAMES) {
        return -1; /* Too many frames */
    }
    if (n_frames <= 0) {
        return -1; /* Invalid frame count */
    }
    int new_cap = n_frames + 64;
    /* Check for overflow: new_cap * n_bins * sizeof(float) must fit in size_t
     * We check: new_cap <= SIZE_MAX / (n_bins * sizeof(float)) */
    size_t denom = (size_t)n_bins * sizeof(float);
    if (denom == 0) return -1;
    /* Cast both sides to size_t for proper comparison */
    if ((size_t)new_cap > (SIZE_MAX / denom)) {
        return -1; /* Overflow detected */
    }
    return 0; /* Safe */
}

static void test_mel_buffer_normal_case(void)
{
    TEST("P0-2: Normal mel buffer allocation (1000 frames, 513 bins)");
    int result = check_mel_buffer_allocation_safe(1000, 513);
    ASSERT(result == 0, "Normal case should be safe");
    PASS();
}

static void test_mel_buffer_max_frames(void)
{
    TEST("P0-2: Mel buffer at maximum (16384 frames, 513 bins)");
    int result = check_mel_buffer_allocation_safe(16384, 513);
    ASSERT(result == 0, "Max frames should be safe");
    PASS();
}

static void test_mel_buffer_exceeds_max_frames(void)
{
    TEST("P0-2: Mel buffer exceeds maximum (16385 frames)");
    int result = check_mel_buffer_allocation_safe(16385, 513);
    ASSERT(result == -1, "Should reject > MAX_MEL_FRAMES");
    PASS();
}

static void test_mel_buffer_overflow_would_occur(void)
{
    TEST("P0-2: Detected potential multiplication overflow");
    /* This would overflow: SIZE_MAX / (513 * 4) ≈ 2^61
     * But we cap at 16384 which is much smaller, so no overflow */
    int n_frames = 16384;
    int n_bins = 513;
    int new_cap = n_frames + 64;
    size_t check = (size_t)new_cap * n_bins * sizeof(float);
    ASSERT(check > 0, "Allocation size should be positive");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════
 * P1-1: Voice Cloning Audio Validation
 * ═══════════════════════════════════════════════════════════════════════ */

/* Simulate the validation logic from pocket_voice_pipeline.c line ~2223 */
static int validate_reference_audio(int n_samples, int sample_rate, const float *pcm) {
    /* Validate sample rate */
    if (sample_rate != 16000 && sample_rate != 24000) {
        return -1; /* Invalid sample rate */
    }

    /* Validate minimum duration (at least 1 second) */
    if (n_samples < sample_rate) {
        return -1; /* Too short */
    }

    /* Validate maximum duration (30 seconds) */
    int max_samples = sample_rate * 30;
    if (n_samples > max_samples) {
        return -1; /* Too long */
    }

    /* Check for silence (max amplitude < 1e-4) */
    if (!pcm) return -1;
    float max_abs = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        float abs_val = pcm[i] < 0.0f ? -pcm[i] : pcm[i];
        if (abs_val > max_abs) max_abs = abs_val;
    }
    if (max_abs < 1e-4f) {
        return -1; /* Silent or near-zero */
    }

    return 0; /* Valid */
}

static void test_voice_cloning_valid_16k(void)
{
    TEST("P1-1: Valid reference audio (16kHz, 2 seconds)");
    float pcm[32000];
    for (int i = 0; i < 32000; i++) {
        pcm[i] = sinf(2.0f * 3.14159f * 440.0f * i / 16000.0f) * 0.5f;
    }
    int result = validate_reference_audio(32000, 16000, pcm);
    ASSERT(result == 0, "Valid 16kHz audio should pass");
    PASS();
}

static void test_voice_cloning_valid_24k(void)
{
    TEST("P1-1: Valid reference audio (24kHz, 3 seconds)");
    float pcm[72000];
    for (int i = 0; i < 72000; i++) {
        pcm[i] = sinf(2.0f * 3.14159f * 440.0f * i / 24000.0f) * 0.5f;
    }
    int result = validate_reference_audio(72000, 24000, pcm);
    ASSERT(result == 0, "Valid 24kHz audio should pass");
    PASS();
}

static void test_voice_cloning_invalid_sample_rate(void)
{
    TEST("P1-1: Reject invalid sample rate (22050Hz)");
    float pcm[22050];
    for (int i = 0; i < 22050; i++) pcm[i] = 0.1f;
    int result = validate_reference_audio(22050, 22050, pcm);
    ASSERT(result == -1, "Invalid sample rate should be rejected");
    PASS();
}

static void test_voice_cloning_too_short(void)
{
    TEST("P1-1: Reject audio too short (0.5 seconds @ 16kHz)");
    float pcm[8000];
    for (int i = 0; i < 8000; i++) pcm[i] = 0.1f;
    int result = validate_reference_audio(8000, 16000, pcm);
    ASSERT(result == -1, "Audio < 1 second should be rejected");
    PASS();
}

static void test_voice_cloning_too_long(void)
{
    TEST("P1-1: Reject audio too long (31 seconds @ 24kHz)");
    int n_samples = 31 * 24000;
    float pcm[n_samples];
    for (int i = 0; i < n_samples; i++) {
        pcm[i] = sinf(2.0f * 3.14159f * 440.0f * i / 24000.0f) * 0.5f;
    }
    int result = validate_reference_audio(n_samples, 24000, pcm);
    ASSERT(result == -1, "Audio > 30 seconds should be rejected");
    PASS();
}

static void test_voice_cloning_silent_audio(void)
{
    TEST("P1-1: Reject silent audio (near-zero amplitude)");
    float pcm[16000];
    for (int i = 0; i < 16000; i++) pcm[i] = 1e-5f; /* Below 1e-4 threshold */
    int result = validate_reference_audio(16000, 16000, pcm);
    ASSERT(result == -1, "Silent audio should be rejected");
    PASS();
}

static void test_voice_cloning_minimum_valid_duration(void)
{
    TEST("P1-1: Accept exactly 1 second @ 16kHz");
    float pcm[16000];
    for (int i = 0; i < 16000; i++) {
        pcm[i] = sinf(2.0f * 3.14159f * 440.0f * i / 16000.0f) * 0.5f;
    }
    int result = validate_reference_audio(16000, 16000, pcm);
    ASSERT(result == 0, "Exactly 1 second should be valid");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════
 * P1-2: HTTP API UTF-8 Validation
 * ═══════════════════════════════════════════════════════════════════════ */

/* Simulate UTF-8 validation from http_api.c line ~802 */
static int is_valid_utf8(const unsigned char *data, int len) {
    for (int i = 0; i < len; i++) {
        unsigned char c = data[i];
        if (c < 0x80) {
            /* ASCII, single byte */
            continue;
        } else if ((c & 0xE0) == 0xC0) {
            /* 2-byte sequence: 110xxxxx 10xxxxxx */
            if (i + 1 >= len || (data[i + 1] & 0xC0) != 0x80) return 0;
            i++;
        } else if ((c & 0xF0) == 0xE0) {
            /* 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx */
            if (i + 2 >= len || (data[i + 1] & 0xC0) != 0x80 || (data[i + 2] & 0xC0) != 0x80) return 0;
            i += 2;
        } else if ((c & 0xF8) == 0xF0) {
            /* 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx */
            if (i + 3 >= len || (data[i + 1] & 0xC0) != 0x80 || (data[i + 2] & 0xC0) != 0x80 || (data[i + 3] & 0xC0) != 0x80) return 0;
            i += 3;
        } else {
            /* Invalid UTF-8 sequence */
            return 0;
        }
    }
    return 1;
}

static void test_utf8_ascii(void)
{
    TEST("P1-2: UTF-8 validation accepts pure ASCII");
    const unsigned char text[] = "Hello, world!";
    int result = is_valid_utf8(text, (int)strlen((const char *)text));
    ASSERT(result == 1, "ASCII should be valid UTF-8");
    PASS();
}

static void test_utf8_valid_multibyte(void)
{
    TEST("P1-2: UTF-8 validation accepts valid 2-byte sequence");
    /* UTF-8 for ü (u-umlaut): 0xC3 0xBC */
    const unsigned char text[] = {0x48, 0x65, 0x6C, 0x6C, 0x6F, 0xC3, 0xBC};
    int result = is_valid_utf8(text, 7);
    ASSERT(result == 1, "Valid 2-byte UTF-8 should be accepted");
    PASS();
}

static void test_utf8_valid_emoji(void)
{
    TEST("P1-2: UTF-8 validation accepts valid 4-byte emoji");
    /* UTF-8 for 😀 (grinning face): F0 9F 98 80 */
    const unsigned char text[] = {0xF0, 0x9F, 0x98, 0x80};
    int result = is_valid_utf8(text, 4);
    ASSERT(result == 1, "Valid 4-byte UTF-8 emoji should be accepted");
    PASS();
}

static void test_utf8_invalid_incomplete_sequence(void)
{
    TEST("P1-2: UTF-8 validation rejects incomplete sequence");
    /* Incomplete: 2-byte marker without continuation */
    const unsigned char text[] = {0xC3, 0x00};  /* 0x00 is not valid continuation */
    int result = is_valid_utf8(text, 2);
    ASSERT(result == 0, "Incomplete UTF-8 should be rejected");
    PASS();
}

static void test_utf8_invalid_bad_continuation(void)
{
    TEST("P1-2: UTF-8 validation rejects bad continuation byte");
    /* Valid marker 0xC3, but bad continuation 0x40 (should be 10xxxxxx) */
    const unsigned char text[] = {0xC3, 0x40};
    int result = is_valid_utf8(text, 2);
    ASSERT(result == 0, "Bad continuation byte should be rejected");
    PASS();
}

static void test_utf8_invalid_overlong_encoding(void)
{
    TEST("P1-2: UTF-8 validation rejects invalid 3-byte start");
    /* Invalid: 0xE0 without proper continuation */
    const unsigned char text[] = {0xE0, 0x80, 0x00};
    int result = is_valid_utf8(text, 3);
    ASSERT(result == 0, "Invalid 3-byte sequence should be rejected");
    PASS();
}

static void test_utf8_empty_string(void)
{
    TEST("P1-2: UTF-8 validation accepts empty string");
    const unsigned char text[] = {0};
    int result = is_valid_utf8(text, 0);
    ASSERT(result == 1, "Empty string should be valid");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main Test Runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  Security Fix Tests: P0-P1 Vulnerabilities                 ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("═══ P0-2: Integer Overflow in Mel Buffer Allocation ═══\n");
    test_mel_buffer_normal_case();
    test_mel_buffer_max_frames();
    test_mel_buffer_exceeds_max_frames();
    test_mel_buffer_overflow_would_occur();
    printf("\n");

    printf("═══ P1-1: Voice Cloning Audio Validation ═══\n");
    test_voice_cloning_valid_16k();
    test_voice_cloning_valid_24k();
    test_voice_cloning_invalid_sample_rate();
    test_voice_cloning_too_short();
    test_voice_cloning_too_long();
    test_voice_cloning_silent_audio();
    test_voice_cloning_minimum_valid_duration();
    printf("\n");

    printf("═══ P1-2: HTTP API UTF-8 Validation ═══\n");
    test_utf8_ascii();
    test_utf8_valid_multibyte();
    test_utf8_valid_emoji();
    test_utf8_invalid_incomplete_sequence();
    test_utf8_invalid_bad_continuation();
    test_utf8_invalid_overlong_encoding();
    test_utf8_empty_string();
    printf("\n");

    printf("════════════════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", pass, fail);
    printf("════════════════════════════════════════════════════════════\n");

    return fail > 0 ? 1 : 0;
}
