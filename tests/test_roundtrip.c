/**
 * test_roundtrip.c — Round-trip intelligibility test with mock TTS/STT.
 *
 * Validates the roundtrip framework (roundtrip.c) by providing mock
 * TTS and STT callbacks that produce deterministic results. This
 * exercises the full roundtrip_test() and roundtrip_run_suite() paths
 * without requiring real TTS/STT models.
 *
 * Build: (via make test-roundtrip)
 */

#include "quality/roundtrip.h"
#include "quality/wer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-50s ", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)

/* ── Mock TTS: generates a sine wave, duration proportional to text length ── */

static float *mock_tts_synthesize(const char *text, int *out_len, int *out_sr,
                                   void *user_data) {
    (void)user_data;
    int sr = 24000;
    int duration_ms = 100 + (int)strlen(text) * 20;
    int n_samples = sr * duration_ms / 1000;

    float *audio = (float *)calloc(n_samples, sizeof(float));
    if (!audio) return NULL;

    for (int i = 0; i < n_samples; i++)
        audio[i] = 0.3f * sinf(2.0f * 3.14159f * 440.0f * (float)i / (float)sr);

    *out_len = n_samples;
    *out_sr = sr;
    return audio;
}

/* ── Mock STT: returns text with configurable error rate ── */

typedef struct {
    float error_rate; /* 0.0 = perfect, 1.0 = all wrong */
} MockSTTConfig;

static char *mock_stt_transcribe(const float *audio, int n_samples, int sr,
                                  void *user_data) {
    (void)audio;
    (void)n_samples;
    (void)sr;
    MockSTTConfig *cfg = (MockSTTConfig *)user_data;

    /* For perfect STT (error_rate=0), we echo back a known string.
     * Since we don't have the original text in the callback, we use
     * a fixed output that will be compared by the roundtrip framework. */
    if (cfg && cfg->error_rate > 0.5f) {
        return strdup("completely wrong gibberish text");
    }
    return strdup("hello world this is a test");
}

/* ── Perfect mock: echoes the original text exactly ── */

static const char *g_original_text = NULL;

static char *mock_stt_perfect(const float *audio, int n_samples, int sr,
                               void *user_data) {
    (void)audio;
    (void)n_samples;
    (void)sr;
    (void)user_data;
    if (g_original_text) return strdup(g_original_text);
    return strdup("hello world");
}

/* ── Tests ─────────────────────────────────────────────── */

static void test_roundtrip_single_perfect(void) {
    TEST("roundtrip single test: perfect STT");
    g_original_text = "hello world";
    RoundTripResult r = roundtrip_test("hello world",
                                        mock_tts_synthesize,
                                        mock_stt_perfect,
                                        NULL, 0.05f);
    CHECK(r.wer.wer < 0.01f && r.passed == 1,
          "Perfect STT should yield 0% WER");
}

static void test_roundtrip_single_bad_stt(void) {
    TEST("roundtrip single test: bad STT yields high WER");
    MockSTTConfig bad = { .error_rate = 1.0f };
    RoundTripResult r = roundtrip_test("the quick brown fox",
                                        mock_tts_synthesize,
                                        mock_stt_transcribe,
                                        &bad, 0.05f);
    CHECK(r.wer.wer > 0.5f && r.passed == 0,
          "Bad STT should yield high WER and fail");
}

static void test_roundtrip_latency_positive(void) {
    TEST("roundtrip measures positive latency");
    g_original_text = "testing latency";
    RoundTripResult r = roundtrip_test("testing latency",
                                        mock_tts_synthesize,
                                        mock_stt_perfect,
                                        NULL, 0.10f);
    CHECK(r.latency.first_chunk_ms > 0.0f,
          "TTS latency should be positive");
}

static void test_roundtrip_suite(void) {
    TEST("roundtrip full suite with mock callbacks");
    MockSTTConfig good = { .error_rate = 0.0f };
    RoundTripSuite suite = roundtrip_run_suite(mock_tts_synthesize,
                                                mock_stt_transcribe,
                                                &good);
    CHECK(suite.n_tests > 0, "Suite should run >0 tests");
    roundtrip_print_report(&suite);
    roundtrip_suite_free(&suite);
}

static void test_roundtrip_suite_free(void) {
    TEST("roundtrip suite_free on empty suite");
    RoundTripSuite empty = {0};
    roundtrip_suite_free(&empty); /* Should not crash */
    PASS();
}

static float *mock_tts_null(const char *t, int *ol, int *os, void *u) {
    (void)t; (void)u; *ol = 0; *os = 24000;
    return NULL;
}

static char *mock_stt_null(const float *a, int n, int s, void *u) {
    (void)a; (void)n; (void)s; (void)u;
    return NULL;
}

static void test_roundtrip_null_tts(void) {
    TEST("roundtrip handles NULL TTS output gracefully");
    RoundTripResult r = roundtrip_test("test", mock_tts_null, mock_stt_perfect, NULL, 0.1f);
    CHECK(r.wer.wer >= 1.0f, "NULL TTS should yield 100% WER");
}

static void test_roundtrip_null_stt(void) {
    TEST("roundtrip handles NULL STT output gracefully");
    RoundTripResult r = roundtrip_test("test", mock_tts_synthesize, mock_stt_null, NULL, 0.1f);
    CHECK(r.wer.wer >= 1.0f, "NULL STT should yield 100% WER");
}

/* ── Main ─────────────────────────────────────────────── */

int main(void) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Round-Trip Intelligibility Tests (Mock TTS/STT)         ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    test_roundtrip_single_perfect();
    test_roundtrip_single_bad_stt();
    test_roundtrip_latency_positive();
    test_roundtrip_suite();
    test_roundtrip_suite_free();
    test_roundtrip_null_tts();
    test_roundtrip_null_stt();

    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  Total: %d passed, %d failed (out of %d)\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("  %s\n", tests_failed == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("══════════════════════════════════════════════════════════\n\n");

    return tests_failed > 0 ? 1 : 0;
}
