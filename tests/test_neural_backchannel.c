/**
 * test_neural_backchannel.c — Unit tests for neural backchannel synthesis.
 *
 * Tests: create/destroy with NULL TTS, cache generation, get cached audio,
 * set speaker/emotion, load WAV override, fallback synthesis.
 *
 * Build: make test-neural-backchannel
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "neural_backchannel.h"

static int pass_count = 0;
static int fail_count = 0;

#define TEST(cond, name) do { \
    if (cond) { printf("  [PASS] %s\n", name); pass_count++; } \
    else { printf("  [FAIL] %s (line %d)\n", name, __LINE__); fail_count++; } \
} while (0)

#define NBC_SR 24000
#define NBC_MAX 12000

static void test_create_destroy_null_tts(void) {
    printf("\n=== Create/Destroy with NULL TTS ===\n");

    NBCConfig cfg = { NBC_SR, 500, 0 };
    NeuralBackchannel *nbc = nbc_create(&cfg, NULL);
    TEST(nbc != NULL, "nbc_create(cfg, NULL) returns non-NULL");

    nbc_destroy(nbc);
    TEST(1, "nbc_destroy(nbc) did not crash");

    nbc_destroy(NULL);
    TEST(1, "nbc_destroy(NULL) is safe");
}

static void test_create_default_config(void) {
    printf("\n=== Create with NULL config ===\n");

    NeuralBackchannel *nbc = nbc_create(NULL, NULL);
    TEST(nbc != NULL, "nbc_create(NULL, NULL) returns non-NULL");
    nbc_destroy(nbc);
}

static void test_cache_generation(void) {
    printf("\n=== Cache Generation ===\n");

    NBCConfig cfg = { NBC_SR, 500, 1 };
    NeuralBackchannel *nbc = nbc_create(&cfg, NULL);
    TEST(nbc != NULL, "create for cache test");

    int ret = nbc_warm_cache(nbc);
    TEST(ret == 0, "nbc_warm_cache returns 0");

    nbc_destroy(nbc);
}

static void test_get_cached(void) {
    printf("\n=== Get Cached Audio ===\n");

    NBCConfig cfg = { NBC_SR, 500, 1 };
    NeuralBackchannel *nbc = nbc_create(&cfg, NULL);
    nbc_warm_cache(nbc);

    int len;
    const float *audio = nbc_get_cached(nbc, NBC_MHM, &len);
    TEST(audio != NULL, "get_cached(NBC_MHM) non-NULL");
    TEST(len > 0, "get_cached returns positive length");
    TEST(len <= NBC_MAX, "cached length within max");

    audio = nbc_get_cached(nbc, NBC_YEAH, &len);
    TEST(audio != NULL, "get_cached(NBC_YEAH) non-NULL");
    TEST(len > 0, "NBC_YEAH has audio");

    audio = nbc_get_cached(nbc, NBC_LAUGH, &len);
    TEST(audio != NULL, "get_cached(NBC_LAUGH) non-NULL");

    audio = nbc_get_cached(NULL, NBC_MHM, &len);
    TEST(audio == NULL, "get_cached(NULL, ...) returns NULL");

    nbc_destroy(nbc);
}

static void test_generate_on_demand(void) {
    printf("\n=== Generate On-Demand ===\n");

    NeuralBackchannel *nbc = nbc_create(&(NBCConfig){NBC_SR, 500, 0}, NULL);
    float buf[NBC_MAX];

    int n = nbc_generate(nbc, NBC_MHM, buf, NBC_MAX);
    TEST(n > 0, "generate(NBC_MHM) returns positive");
    TEST(n <= NBC_MAX, "generate within max buffer");

    n = nbc_generate(nbc, NBC_OKAY, buf, 1000);
    TEST(n > 0 && n <= 1000, "generate with small buffer");

    n = nbc_generate(NULL, NBC_MHM, buf, NBC_MAX);
    TEST(n == -1, "generate(NULL, ...) returns -1");

    n = nbc_generate(nbc, NBC_MHM, NULL, NBC_MAX);
    TEST(n == -1, "generate(..., NULL, ...) returns -1");

    nbc_destroy(nbc);
}

static void test_set_speaker(void) {
    printf("\n=== Set Speaker ===\n");

    NeuralBackchannel *nbc = nbc_create(&(NBCConfig){NBC_SR, 500, 0}, NULL);
    float emb[192] = { 0.1f };

    int ret = nbc_set_speaker(nbc, emb, 192);
    TEST(ret == 0, "set_speaker(emb, 192) returns 0");

    ret = nbc_set_speaker(nbc, NULL, 0);
    TEST(ret == 0, "set_speaker(NULL, 0) clears");

    ret = nbc_set_speaker(NULL, emb, 192);
    TEST(ret == -1, "set_speaker(NULL, ...) returns -1");

    nbc_destroy(nbc);
}

static void test_set_emotion(void) {
    printf("\n=== Set Emotion ===\n");

    NeuralBackchannel *nbc = nbc_create(&(NBCConfig){NBC_SR, 500, 0}, NULL);
    nbc_set_emotion(nbc, 3);
    nbc_set_emotion(NULL, 0);  /* no crash */
    nbc_destroy(nbc);
}

static void test_load_wav(void) {
    printf("\n=== Load WAV Override ===\n");

    NeuralBackchannel *nbc = nbc_create(&(NBCConfig){NBC_SR, 500, 0}, NULL);

    int ret = nbc_load_wav(nbc, NBC_MHM, "/nonexistent/path.wav");
    TEST(ret == -1, "load_wav(nonexistent) returns -1");

    ret = nbc_load_wav(NULL, NBC_MHM, "/tmp/any.wav");
    TEST(ret == -1, "load_wav(NULL, ...) returns -1");

    ret = nbc_load_wav(nbc, NBC_MHM, NULL);
    TEST(ret == -1, "load_wav(..., NULL) returns -1");

    nbc_destroy(nbc);
}

static void test_fallback_audio_nonzero(void) {
    printf("\n=== Fallback Audio Non-Zero ===\n");

    NeuralBackchannel *nbc = nbc_create(&(NBCConfig){NBC_SR, 500, 0}, NULL);
    float buf[2400];

    int n = nbc_generate(nbc, NBC_MHM, buf, 2400);
    TEST(n > 0, "fallback generates samples");

    float rms = 0.0f;
    for (int i = 0; i < n; i++) rms += buf[i] * buf[i];
    rms = sqrtf(rms / n);
    TEST(rms > 1e-6f, "fallback audio has non-zero energy");

    nbc_destroy(nbc);
}

static void test_all_types_generatable(void) {
    printf("\n=== All Types Generatable ===\n");

    NeuralBackchannel *nbc = nbc_create(&(NBCConfig){NBC_SR, 500, 0}, NULL);
    float buf[2400];
    int all_ok = 1;

    for (int t = 0; t < NBC_COUNT; t++) {
        int n = nbc_generate(nbc, (NBCType)t, buf, 2400);
        if (n <= 0) all_ok = 0;
    }
    TEST(all_ok, "all NBC types generate audio");

    nbc_destroy(nbc);
}

int main(void) {
    printf("Neural Backchannel Tests\n");

    test_create_destroy_null_tts();
    test_create_default_config();
    test_cache_generation();
    test_get_cached();
    test_generate_on_demand();
    test_set_speaker();
    test_set_emotion();
    test_load_wav();
    test_fallback_audio_nonzero();
    test_all_types_generatable();

    printf("\n--- Result: %d passed, %d failed ---\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
