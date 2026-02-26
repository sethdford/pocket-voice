/**
 * test_voice_onboard.c — Unit tests for voice onboarding / prosody transfer.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "voice_onboard.h"

static int pass_count = 0;
static int fail_count = 0;

#define TEST(cond, name) do { \
    if (cond) { printf("  [PASS] %s\n", name); pass_count++; } \
    else { printf("  [FAIL] %s (line %d)\n", name, __LINE__); fail_count++; } \
} while (0)

static void test_create_destroy(void) {
    printf("\n=== Create / Destroy ===\n");

    VoiceOnboardSession *s = voice_onboard_create(NULL, 3.0f, 16000);
    TEST(s != NULL, "create with defaults");

    float p = voice_onboard_progress(s);
    TEST(p == 0.0f, "progress starts at 0");

    voice_onboard_destroy(s);

    s = voice_onboard_create(NULL, 0.0f, 0);
    TEST(s != NULL, "create with zero params uses defaults");
    voice_onboard_destroy(s);

    voice_onboard_destroy(NULL);
    TEST(1, "destroy NULL is safe");
}

static void test_feed_and_progress(void) {
    printf("\n=== Feed + Progress ===\n");

    VoiceOnboardSession *s = voice_onboard_create(NULL, 2.0f, 16000);
    TEST(s != NULL, "session created");

    /* Feed 1 second of audio (16000 samples) */
    float *buf = (float *)calloc(16000, sizeof(float));
    for (int i = 0; i < 16000; i++)
        buf[i] = 0.3f * sinf(2.0f * 3.14159f * 200.0f * (float)i / 16000.0f);

    int done = voice_onboard_feed(s, buf, 16000);
    TEST(done == 0, "not done after 1s of 2s capture");

    float p = voice_onboard_progress(s);
    TEST(fabsf(p - 0.5f) < 0.01f, "progress ~50% after 1s");

    done = voice_onboard_feed(s, buf, 16000);
    TEST(done == 1, "done after 2s of 2s capture");

    p = voice_onboard_progress(s);
    TEST(p >= 1.0f, "progress >= 100%");

    free(buf);
    voice_onboard_destroy(s);
}

static void test_finalize_prosody(void) {
    printf("\n=== Finalize Prosody Profile ===\n");

    VoiceOnboardSession *s = voice_onboard_create(NULL, 1.5f, 16000);

    /* Generate 1.5s of 200Hz tone at 16kHz */
    int n = 24000;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++)
        buf[i] = 0.3f * sinf(2.0f * 3.14159f * 200.0f * (float)i / 16000.0f);

    voice_onboard_feed(s, buf, n);
    VoiceOnboardResult result = voice_onboard_finalize(s);

    TEST(result.success == 1, "finalize succeeded");
    TEST(result.prosody.duration_sec > 1.0f, "duration > 1s");
    TEST(result.prosody.f0_mean > 100.0f && result.prosody.f0_mean < 300.0f,
         "F0 mean in voiced range (100-300Hz)");
    TEST(result.prosody.energy_mean_db > -60.0f, "energy is audible");

    printf("    F0=%.0fHz range=%.0f energy=%.1fdB rate=%.1f wps dur=%.1fs\n",
           result.prosody.f0_mean, result.prosody.f0_range,
           result.prosody.energy_mean_db, result.prosody.speaking_rate,
           result.prosody.duration_sec);

    free(buf);
    voice_onboard_destroy(s);
}

static void test_too_short(void) {
    printf("\n=== Too Short Audio ===\n");

    VoiceOnboardSession *s = voice_onboard_create(NULL, 5.0f, 16000);

    /* Feed only 0.5s — below MIN_SPEECH_SAMPLES */
    float buf[8000];
    for (int i = 0; i < 8000; i++)
        buf[i] = 0.1f * sinf(2.0f * 3.14159f * 150.0f * (float)i / 16000.0f);

    voice_onboard_feed(s, buf, 8000);
    VoiceOnboardResult result = voice_onboard_finalize(s);
    TEST(result.success == 0, "too-short audio fails gracefully");

    voice_onboard_destroy(s);
}

static void test_f0_estimation(void) {
    printf("\n=== F0 Estimation ===\n");

    int sr = 16000;
    int n = 8000;
    float *buf = (float *)malloc((size_t)n * sizeof(float));

    /* Test 150Hz tone */
    for (int i = 0; i < n; i++)
        buf[i] = 0.5f * sinf(2.0f * 3.14159f * 150.0f * (float)i / (float)sr);

    float f0 = voice_onboard_estimate_f0(buf, n, sr);
    TEST(f0 > 130.0f && f0 < 170.0f, "150Hz tone → F0 ≈ 150Hz");

    /* Test 250Hz tone */
    for (int i = 0; i < n; i++)
        buf[i] = 0.5f * sinf(2.0f * 3.14159f * 250.0f * (float)i / (float)sr);

    f0 = voice_onboard_estimate_f0(buf, n, sr);
    TEST(f0 > 100.0f && f0 < 400.0f, "250Hz tone → F0 in voiced range");

    /* Silence should return 0 */
    memset(buf, 0, (size_t)n * sizeof(float));
    f0 = voice_onboard_estimate_f0(buf, n, sr);
    TEST(f0 == 0.0f, "silence → F0 = 0");

    /* NULL safety */
    f0 = voice_onboard_estimate_f0(NULL, 0, sr);
    TEST(f0 == 0.0f, "NULL input → F0 = 0");

    free(buf);
}

static void test_null_safety(void) {
    printf("\n=== NULL Safety ===\n");

    TEST(voice_onboard_feed(NULL, NULL, 0) == 0, "feed(NULL) safe");
    TEST(voice_onboard_progress(NULL) == 0.0f, "progress(NULL) safe");

    VoiceOnboardResult r = voice_onboard_finalize(NULL);
    TEST(r.success == 0, "finalize(NULL) safe");
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Voice Onboarding / Prosody Transfer Tests              ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    test_create_destroy();
    test_feed_and_progress();
    test_finalize_prosody();
    test_too_short();
    test_f0_estimation();
    test_null_safety();

    printf("\n════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", pass_count, fail_count);
    printf("════════════════════════════════════════════\n");

    return fail_count > 0 ? 1 : 0;
}
