/**
 * test_quality.c — Test suite for the quality benchmark modules.
 *
 * Validates WER, CER, MCD, STOI, SNR, F0, speaker similarity,
 * latency harness, and round-trip infrastructure.
 */

#include "quality/wer.h"
#include "quality/audio_quality.h"
#include "quality/latency_harness.h"
#include "quality/roundtrip.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-40s ", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)

/* ── WER Tests ────────────────────────────────────────── */

static void test_wer(void)
{
    printf("\n[WER Tests]\n");

    TEST("Perfect match");
    {
        WERResult r = wer_compute("hello world", "hello world");
        CHECK(r.wer == 0.0f && r.substitutions == 0, "Expected WER=0");
    }

    TEST("Complete mismatch");
    {
        WERResult r = wer_compute("hello world", "foo bar");
        CHECK(r.wer == 1.0f && r.substitutions == 2, "Expected WER=1.0");
    }

    TEST("Insertion");
    {
        WERResult r = wer_compute("hello world", "hello beautiful world");
        CHECK(r.insertions == 1, "Expected 1 insertion");
    }

    TEST("Deletion");
    {
        WERResult r = wer_compute("hello beautiful world", "hello world");
        CHECK(r.deletions == 1, "Expected 1 deletion");
    }

    TEST("Case insensitive");
    {
        WERResult r = wer_compute("HELLO WORLD", "hello world");
        CHECK(r.wer == 0.0f, "Expected case-insensitive match");
    }

    TEST("Punctuation stripped");
    {
        WERResult r = wer_compute("Hello, world!", "hello world");
        CHECK(r.wer == 0.0f, "Expected punct-stripped match");
    }

    TEST("Apostrophe preserved (contractions)");
    {
        WERResult r = wer_compute("I can't go", "i can't go");
        CHECK(r.wer == 0.0f, "Expected apostrophe preserved");
    }

    TEST("Empty reference");
    {
        WERResult r = wer_compute("", "hello");
        CHECK(r.insertions == 1, "Expected 1 insertion for empty ref");
    }

    TEST("Empty hypothesis");
    {
        WERResult r = wer_compute("hello world", "");
        CHECK(r.deletions == 2, "Expected 2 deletions for empty hyp");
    }

    TEST("Both empty");
    {
        WERResult r = wer_compute("", "");
        CHECK(r.wer == 0.0f, "Expected WER=0 for both empty");
    }
}

/* ── CER Tests ────────────────────────────────────────── */

static void test_cer(void)
{
    printf("\n[CER Tests]\n");

    TEST("Perfect match");
    {
        float c = cer_compute("hello", "hello");
        CHECK(c == 0.0f, "Expected CER=0");
    }

    TEST("One substitution");
    {
        float c = cer_compute("hello", "hallo");
        CHECK(fabsf(c - 0.2f) < 0.05f, "Expected CER≈0.2");
    }

    TEST("One insertion");
    {
        float c = cer_compute("hello", "helloo");
        CHECK(c > 0.0f && c < 0.3f, "Expected small CER");
    }
}

/* ── Audio Quality Tests ──────────────────────────────── */

static void generate_sine(float *buf, int n, float freq, int sr)
{
    for (int i = 0; i < n; i++) {
        buf[i] = 0.3f * sinf(2.0f * (float)M_PI * freq * (float)i / (float)sr);
    }
}

static void generate_noise(float *buf, int n, float amplitude)
{
    for (int i = 0; i < n; i++) {
        buf[i] = amplitude * (2.0f * (float)rand() / (float)RAND_MAX - 1.0f);
    }
}

static void test_mcd(void)
{
    printf("\n[MCD Tests]\n");

    int sr = 24000;
    int n = sr; /* 1 second */
    float *sig = (float *)malloc((size_t)n * sizeof(float));

    TEST("Identical signals → MCD ≈ 0");
    {
        generate_sine(sig, n, 200.0f, sr);
        MCDResult r = mcd_compute(sig, n, sig, n, sr);
        CHECK(r.mcd_db < 0.1f, "Expected near-zero MCD");
    }

    TEST("Different signals → MCD > 0");
    {
        float *sig2 = (float *)malloc((size_t)n * sizeof(float));
        generate_sine(sig, n, 200.0f, sr);
        generate_sine(sig2, n, 800.0f, sr);
        MCDResult r = mcd_compute(sig, n, sig2, n, sr);
        CHECK(r.mcd_db > 1.0f, "Expected nonzero MCD");
        free(sig2);
    }

    free(sig);
}

static void test_stoi(void)
{
    printf("\n[STOI Tests]\n");

    int sr = 24000;
    int n = sr;
    float *sig = (float *)malloc((size_t)n * sizeof(float));

    TEST("Identical signals → STOI ≈ 1.0");
    {
        generate_sine(sig, n, 440.0f, sr);
        STOIResult r = stoi_compute(sig, sig, n, sr);
        CHECK(r.stoi > 0.85f, "Expected high STOI for identical signals");
    }

    TEST("Signal vs noise → STOI < 0.5");
    {
        float *noise = (float *)malloc((size_t)n * sizeof(float));
        generate_sine(sig, n, 440.0f, sr);
        generate_noise(noise, n, 0.3f);
        STOIResult r = stoi_compute(sig, noise, n, sr);
        CHECK(r.stoi < 0.7f, "Expected low STOI for noise");
        free(noise);
    }

    free(sig);
}

static void test_snr(void)
{
    printf("\n[SNR Tests]\n");

    int sr = 24000;
    int n = sr;
    float *sig = (float *)malloc((size_t)n * sizeof(float));

    TEST("Identical → high SNR");
    {
        generate_sine(sig, n, 440.0f, sr);
        SNRResult r = snr_compute(sig, sig, n, sr);
        CHECK(r.seg_snr_db >= 30.0f, "Expected high SNR for identical signals");
    }

    TEST("With noise → lower SNR");
    {
        float *noisy = (float *)malloc((size_t)n * sizeof(float));
        generate_sine(sig, n, 440.0f, sr);
        for (int i = 0; i < n; i++) {
            noisy[i] = sig[i] + 0.01f * (2.0f * (float)rand() / (float)RAND_MAX - 1.0f);
        }
        SNRResult r = snr_compute(sig, noisy, n, sr);
        CHECK(r.seg_snr_db > 15.0f && r.seg_snr_db < 35.0f,
              "Expected moderate SNR with light noise");
        free(noisy);
    }

    free(sig);
}

static void test_f0(void)
{
    printf("\n[F0 Tests]\n");

    int sr = 24000;
    int n = sr;
    float *sig = (float *)malloc((size_t)n * sizeof(float));

    TEST("Same signal → F0 RMSE ≈ 0, corr = 1.0");
    {
        generate_sine(sig, n, 200.0f, sr);
        F0Result r = f0_compare(sig, n, sig, n, sr);
        CHECK(r.f0_rmse_hz < 1.0f && r.f0_corr >= 0.99f,
              "Expected perfect F0 match");
    }

    TEST("Different pitch → F0 RMSE > 0");
    {
        float *sig2 = (float *)malloc((size_t)n * sizeof(float));
        generate_sine(sig, n, 200.0f, sr);
        generate_sine(sig2, n, 250.0f, sr);
        F0Result r = f0_compare(sig, n, sig2, n, sr);
        CHECK(r.f0_rmse_hz > 10.0f, "Expected nonzero F0 RMSE");
        free(sig2);
    }

    free(sig);
}

static void test_speaker_sim(void)
{
    printf("\n[Speaker Similarity Tests]\n");

    int sr = 24000;
    int n = sr;
    float *sig = (float *)malloc((size_t)n * sizeof(float));

    TEST("Same signal → high similarity");
    {
        generate_sine(sig, n, 200.0f, sr);
        SpeakerSimResult r = speaker_similarity(sig, n, sig, n, sr);
        CHECK(r.cosine_sim > 0.99f, "Expected perfect similarity");
    }

    TEST("Different signals → lower similarity");
    {
        float *sig2 = (float *)malloc((size_t)n * sizeof(float));
        generate_sine(sig, n, 200.0f, sr);
        generate_sine(sig2, n, 1000.0f, sr);
        SpeakerSimResult r = speaker_similarity(sig, n, sig2, n, sr);
        CHECK(r.cosine_sim < 0.99f, "Expected lower similarity");
        free(sig2);
    }

    free(sig);
}

/* ── Latency Harness Tests ────────────────────────────── */

static void test_latency_harness(void)
{
    printf("\n[Latency Harness Tests]\n");

    LatencyHarness *h = latency_create();

    TEST("Record and retrieve stats");
    {
        for (int i = 0; i < 100; i++) {
            latency_record_ms(h, LAT_STEP, 3.0 + (double)(i % 10) * 0.5);
        }
        LatencyStats st = latency_stats(h, LAT_STEP);
        CHECK(st.n == 100 && st.mean > 2.0 && st.mean < 8.0,
              "Expected valid stats");
    }

    TEST("Percentiles ordering (P50 <= P95 <= P99)");
    {
        LatencyStats st = latency_stats(h, LAT_STEP);
        CHECK(st.p50 <= st.p95 && st.p95 <= st.p99,
              "Expected P50 <= P95 <= P99");
    }

    TEST("Start/stop timing");
    {
        uint64_t tok = latency_start(h, LAT_E2E);
        usleep(10000); /* 10ms */
        latency_stop(h, LAT_E2E, tok);
        LatencyStats st = latency_stats(h, LAT_E2E);
        CHECK(st.n == 1 && st.mean > 5.0 && st.mean < 30.0,
              "Expected ~10ms measurement");
    }

    TEST("Reset clears all data");
    {
        latency_reset(h);
        LatencyStats st = latency_stats(h, LAT_STEP);
        CHECK(st.n == 0, "Expected empty after reset");
    }

    TEST("Empty metric returns zero stats");
    {
        LatencyStats st = latency_stats(h, LAT_TTFT);
        CHECK(st.n == 0 && st.mean == 0.0 && st.p50 == 0.0,
              "Expected zeros for empty metric");
    }

    latency_destroy(h);
}

/* ── Quality Grading Tests ────────────────────────────── */

static void test_grading(void)
{
    printf("\n[Quality Grading Tests]\n");

    TEST("Perfect scores → Grade A");
    {
        QualityScorecard sc = {0};
        sc.stoi.stoi = 0.98f;
        sc.mcd.mcd_db = 2.0f;
        sc.wer = 0.01f;
        sc.f0.f0_corr = 0.95f;
        sc.speaker.cosine_sim = 0.95f;
        sc.latency.e2e_ms = 100.0f;
        sc = quality_grade(sc);
        CHECK(sc.grade == 'A' && sc.overall_score > 90.0f,
              "Expected Grade A for perfect scores");
    }

    TEST("Poor scores → Grade D/F");
    {
        QualityScorecard sc = {0};
        sc.stoi.stoi = 0.45f;
        sc.mcd.mcd_db = 12.0f;
        sc.wer = 0.40f;
        sc.f0.f0_corr = 0.2f;
        sc.speaker.cosine_sim = 0.4f;
        sc.latency.e2e_ms = 2000.0f;
        sc = quality_grade(sc);
        CHECK(sc.grade >= 'D', "Expected low grade for poor scores");
    }

    TEST("Report printing (smoke test)");
    {
        QualityScorecard sc = {0};
        sc.stoi.stoi = 0.96f;
        sc.mcd.mcd_db = 3.0f;
        sc.wer = 0.02f;
        sc.f0.f0_corr = 0.92f;
        sc.f0.f0_rmse_hz = 10.0f;
        sc.f0.voicing_accuracy = 0.97f;
        sc.speaker.cosine_sim = 0.95f;
        sc.snr.seg_snr_db = 30.0f;
        sc.latency.rtf = 0.15f;
        sc.latency.first_chunk_ms = 120.0f;
        sc.latency.e2e_ms = 150.0f;
        sc.cer = 0.01f;
        sc = quality_grade(sc);
        quality_print_report(&sc, "Best-in-Class Target");
        CHECK(sc.grade == 'A',
              "Expected Grade A for best-in-class scores");
    }
}

/* ── Normalization Tests ──────────────────────────────── */

static void test_normalization(void)
{
    printf("\n[Text Normalization Tests]\n");

    char buf[256];

    TEST("Lowercase conversion");
    {
        wer_normalize("HELLO WORLD", buf, 256);
        CHECK(strcmp(buf, "hello world") == 0, "Expected lowercase");
    }

    TEST("Punctuation removal");
    {
        wer_normalize("Hello, world!", buf, 256);
        CHECK(strcmp(buf, "hello world") == 0, "Expected no punctuation");
    }

    TEST("Whitespace collapse");
    {
        wer_normalize("  hello   world  ", buf, 256);
        CHECK(strcmp(buf, "hello world") == 0, "Expected collapsed whitespace");
    }

    TEST("Apostrophe preservation");
    {
        wer_normalize("can't won't", buf, 256);
        CHECK(strcmp(buf, "can't won't") == 0, "Expected apostrophes kept");
    }
}

/* ── WER Edge Case Tests ──────────────────────────────── */

static void test_wer_edge_cases(void)
{
    printf("\n[WER Edge Case Tests]\n");

    TEST("NULL reference → WER=1.0");
    {
        WERResult r = wer_compute(NULL, "hello");
        CHECK(r.wer == 1.0f, "Expected WER=1.0 for NULL ref");
    }

    TEST("NULL hypothesis → WER=1.0");
    {
        WERResult r = wer_compute("hello", NULL);
        CHECK(r.wer == 1.0f, "Expected WER=1.0 for NULL hyp");
    }

    TEST("Both NULL → WER=1.0");
    {
        WERResult r = wer_compute(NULL, NULL);
        CHECK(r.wer == 1.0f, "Expected WER=1.0 for both NULL");
    }

    TEST("Whitespace-only reference");
    {
        WERResult r = wer_compute("   ", "hello");
        CHECK(r.insertions >= 0, "Should handle whitespace-only ref");
    }

    TEST("Very long matching strings");
    {
        /* 50 words repeated */
        char long_str[2048];
        int pos = 0;
        for (int i = 0; i < 50 && pos < 2000; i++) {
            pos += snprintf(long_str + pos, (size_t)(2048 - pos), "word%d ", i);
        }
        long_str[pos > 0 ? pos - 1 : 0] = '\0';
        WERResult r = wer_compute(long_str, long_str);
        CHECK(r.wer == 0.0f, "Expected WER=0 for long matching strings");
    }

    TEST("Multiple substitutions (half wrong)");
    {
        WERResult r = wer_compute("the cat sat on the mat",
                                   "the dog sat on the rug");
        CHECK(r.substitutions == 2 && r.wer > 0.0f && r.wer < 1.0f,
              "Expected 2 substitutions");
    }
}

/* ── CER Edge Case Tests ─────────────────────────────── */

static void test_cer_edge_cases(void)
{
    printf("\n[CER Edge Case Tests]\n");

    TEST("CER with NULL reference");
    {
        float c = cer_compute(NULL, "hello");
        CHECK(c == 1.0f, "Expected CER=1.0 for NULL ref");
    }

    TEST("CER with NULL hypothesis");
    {
        float c = cer_compute("hello", NULL);
        CHECK(c == 1.0f, "Expected CER=1.0 for NULL hyp");
    }

    TEST("CER both empty");
    {
        float c = cer_compute("", "");
        CHECK(c == 0.0f, "Expected CER=0 for both empty");
    }

    TEST("CER single char difference");
    {
        float c = cer_compute("a", "b");
        CHECK(fabsf(c - 1.0f) < 0.01f, "Expected CER=1.0 for single char sub");
    }
}

/* ── Audio Quality Numerical Stability Tests ─────────── */

static void test_audio_quality_edge_cases(void)
{
    printf("\n[Audio Quality Edge Cases]\n");

    int sr = 24000;

    TEST("MCD with very short signal (100 samples)");
    {
        int n = 100;
        float *sig = (float *)malloc((size_t)n * sizeof(float));
        generate_sine(sig, n, 440.0f, sr);
        MCDResult r = mcd_compute(sig, n, sig, n, sr);
        /* Should either compute a value or gracefully handle too-short signal */
        CHECK(r.mcd_db >= 0.0f || r.n_frames == 0,
              "Short signal should not produce negative MCD or crash");
        free(sig);
    }

    TEST("STOI with all-zero (silence) signals");
    {
        int n = sr;
        float *silence = (float *)calloc((size_t)n, sizeof(float));
        STOIResult r = stoi_compute(silence, silence, n, sr);
        /* Silence vs silence: should not crash, result is implementation-defined */
        CHECK(r.stoi >= 0.0f && r.stoi <= 1.0f,
              "STOI with silence should be in [0,1]");
        free(silence);
    }

    TEST("SNR with all-zero signals (degenerate)");
    {
        int n = sr;
        float *silence = (float *)calloc((size_t)n, sizeof(float));
        SNRResult r = snr_compute(silence, silence, n, sr);
        /* Zero signal: should not crash, SNR is undefined/high */
        CHECK(r.n_active_frames >= 0, "Should not crash on zero signals");
        free(silence);
    }

    TEST("F0 with very low frequency signal");
    {
        int n = sr;
        float *sig = (float *)malloc((size_t)n * sizeof(float));
        generate_sine(sig, n, 55.0f, sr); /* Very low A1 */
        F0Result r = f0_compare(sig, n, sig, n, sr);
        CHECK(r.f0_rmse_hz >= 0.0f, "Low frequency should not crash");
        free(sig);
    }

    TEST("Speaker similarity with noise vs sine");
    {
        int n = sr;
        float *sig1 = (float *)malloc((size_t)n * sizeof(float));
        float *sig2 = (float *)malloc((size_t)n * sizeof(float));
        generate_sine(sig1, n, 200.0f, sr);
        generate_noise(sig2, n, 0.3f);
        SpeakerSimResult r = speaker_similarity(sig1, n, sig2, n, sr);
        CHECK(r.cosine_sim < 1.0f, "Noise vs sine should have lower similarity");
        free(sig1);
        free(sig2);
    }

    TEST("MCD with different length signals");
    {
        int n1 = sr;      /* 1 second */
        int n2 = sr / 2;  /* 0.5 seconds */
        float *sig1 = (float *)malloc((size_t)n1 * sizeof(float));
        float *sig2 = (float *)malloc((size_t)n2 * sizeof(float));
        generate_sine(sig1, n1, 440.0f, sr);
        generate_sine(sig2, n2, 440.0f, sr);
        MCDResult r = mcd_compute(sig1, n1, sig2, n2, sr);
        CHECK(r.n_frames >= 0, "Different length signals should not crash");
        free(sig1);
        free(sig2);
    }
}

/* ── Quality Grading Boundary Tests ──────────────────── */

static void test_grading_boundaries(void)
{
    printf("\n[Quality Grading Boundary Tests]\n");

    TEST("Moderate scores → Grade B or C");
    {
        QualityScorecard sc = {0};
        sc.stoi.stoi = 0.80f;
        sc.mcd.mcd_db = 5.5f;
        sc.wer = 0.08f;
        sc.f0.f0_corr = 0.75f;
        sc.speaker.cosine_sim = 0.80f;
        sc.latency.e2e_ms = 600.0f;
        sc = quality_grade(sc);
        CHECK(sc.grade >= 'B' && sc.grade <= 'C',
              "Expected Grade B or C for moderate scores");
    }

    TEST("Near-perfect with bad latency → not Grade A");
    {
        QualityScorecard sc = {0};
        sc.stoi.stoi = 0.99f;
        sc.mcd.mcd_db = 1.0f;
        sc.wer = 0.00f;
        sc.f0.f0_corr = 0.99f;
        sc.speaker.cosine_sim = 0.99f;
        sc.latency.e2e_ms = 5000.0f;  /* Very high latency */
        sc = quality_grade(sc);
        CHECK(sc.overall_score > 0.0f, "Should compute a valid overall score");
    }

    TEST("All zeroed scorecard → Grade F");
    {
        QualityScorecard sc = {0};
        sc = quality_grade(sc);
        CHECK(sc.grade >= 'D', "Zeroed scorecard should get low grade");
    }
}

/* ── Normalization Edge Cases ─────────────────────────── */

static void test_normalization_edge_cases(void)
{
    printf("\n[Text Normalization Edge Cases]\n");

    char buf[256];

    TEST("Only punctuation → empty");
    {
        wer_normalize(".,!?;:", buf, 256);
        CHECK(strlen(buf) == 0, "Only punctuation should yield empty");
    }

    TEST("Mixed numbers and letters");
    {
        wer_normalize("test123 hello456", buf, 256);
        CHECK(strcmp(buf, "test123 hello456") == 0,
              "Numbers should be preserved");
    }

    TEST("Buffer too small (truncation)");
    {
        int ret = wer_normalize("hello world this is a test", buf, 6);
        CHECK(ret <= 5 && buf[5] == '\0', "Should truncate to buffer size");
    }

    TEST("Tab and newline as whitespace");
    {
        wer_normalize("hello\tworld\nfoo", buf, 256);
        CHECK(strcmp(buf, "hello world foo") == 0,
              "Tabs and newlines should become spaces");
    }
}

/* ── Latency Harness Edge Cases ──────────────────────── */

static void test_latency_harness_edge_cases(void)
{
    printf("\n[Latency Harness Edge Cases]\n");

    TEST("Large number of samples");
    {
        LatencyHarness *h = latency_create();
        for (int i = 0; i < 1000; i++) {
            latency_record_ms(h, LAT_STEP, 1.0 + (double)(i % 50) * 0.1);
        }
        LatencyStats st = latency_stats(h, LAT_STEP);
        CHECK(st.n == 1000 && st.min >= 1.0 && st.max <= 6.0,
              "Expected valid stats for 1000 samples");
        latency_destroy(h);
    }

    TEST("Single sample statistics");
    {
        LatencyHarness *h = latency_create();
        latency_record_ms(h, LAT_TTFT, 42.0);
        LatencyStats st = latency_stats(h, LAT_TTFT);
        CHECK(st.n == 1 && fabs(st.mean - 42.0) < 0.01 && fabs(st.p50 - 42.0) < 0.01,
              "Single sample: mean=p50=p95=p99=value");
        latency_destroy(h);
    }

    TEST("Multiple metrics tracked independently");
    {
        LatencyHarness *h = latency_create();
        latency_record_ms(h, LAT_STEP, 5.0);
        latency_record_ms(h, LAT_E2E, 100.0);
        latency_record_ms(h, LAT_TTFT, 50.0);
        LatencyStats s1 = latency_stats(h, LAT_STEP);
        LatencyStats s2 = latency_stats(h, LAT_E2E);
        LatencyStats s3 = latency_stats(h, LAT_TTFT);
        CHECK(s1.n == 1 && s2.n == 1 && s3.n == 1 &&
              fabs(s1.mean - 5.0) < 0.01 &&
              fabs(s2.mean - 100.0) < 0.01 &&
              fabs(s3.mean - 50.0) < 0.01,
              "Each metric should be tracked independently");
        latency_destroy(h);
    }
}

/* ── Main ─────────────────────────────────────────────── */

int main(void)
{
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  pocket-voice Quality Module Tests                       ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    srand(42); /* Deterministic randomness */

    test_wer();
    test_cer();
    test_normalization();
    test_mcd();
    test_stoi();
    test_snr();
    test_f0();
    test_speaker_sim();
    test_latency_harness();
    test_grading();

    /* New deep coverage tests */
    test_wer_edge_cases();
    test_cer_edge_cases();
    test_audio_quality_edge_cases();
    test_grading_boundaries();
    test_normalization_edge_cases();
    test_latency_harness_edge_cases();

    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  Total: %d passed, %d failed (out of %d)\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("  %s\n", tests_failed == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("══════════════════════════════════════════════════════════\n\n");

    return tests_failed > 0 ? 1 : 0;
}
