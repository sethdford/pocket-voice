/**
 * test_response_cache.c — Unit tests for pre-synthesized response cache.
 *
 * Tests: create/destroy, add/get, variants, has/variant_count, stats,
 * clear, save/load, warm, speaker, NULL handling, overflow.
 *
 * Build: make test-response-cache
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "response_cache.h"
#include "intent_router.h"

static int pass_count = 0;
static int fail_count = 0;

#define TEST(cond, name) do { \
    if (cond) { printf("  [PASS] %s\n", name); pass_count++; } \
    else { printf("  [FAIL] %s (line %d)\n", name, __LINE__); fail_count++; } \
} while (0)

/* Write a minimal 24kHz mono 16-bit WAV for testing */
static int write_test_wav(const char *path, const float *pcm, int n_samples, int sample_rate) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int n_bytes = n_samples * 2;
    char riff[4] = {'R','I','F','F'};
    uint32_t file_size = 36 + n_bytes;
    char wave[4] = {'W','A','V','E'};
    char fmt[4] = {'f','m','t',' '};
    uint32_t fmt_size = 16;
    uint16_t audio_fmt = 1;
    uint16_t channels = 1;
    uint32_t sr = (uint32_t)sample_rate;
    uint32_t byte_rate = sr * 2;
    uint16_t block_align = 2;
    uint16_t bits = 16;
    char data[4] = {'d','a','t','a'};
    uint32_t data_size = (uint32_t)n_bytes;

    fwrite(riff, 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite(wave, 1, 4, f);
    fwrite(fmt, 1, 4, f);
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_fmt, 2, 1, f);
    fwrite(&channels, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite(data, 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    for (int i = 0; i < n_samples; i++) {
        int16_t s = (int16_t)(pcm[i] * 32767.0f);
        if (s < -32768) s = -32768;
        if (s > 32767) s = 32767;
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
    return 0;
}

static void test_create_destroy(void) {
    printf("\n=== Create / Destroy ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);
    TEST(c != NULL, "create returns non-NULL");

    response_cache_destroy(c);
    response_cache_destroy(NULL);
    TEST(1, "destroy NULL is safe");
}

static void test_add_manual(void) {
    printf("\n=== Add Audio Manually, Get Back ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);
    float pcm[240] = {0};
    for (int i = 0; i < 240; i++) pcm[i] = 0.1f * (float)i;

    int r = response_cache_add(c, FAST_GREETING, "Hi!", pcm, 240);
    TEST(r == 0, "add returns 0");

    int len = 0;
    const float *got = response_cache_get(c, FAST_GREETING, &len);
    TEST(got != NULL, "get returns non-NULL");
    TEST(len == 240, "get returns correct length");
    TEST(memcmp(got, pcm, 240 * sizeof(float)) == 0, "get returns same audio");

    response_cache_destroy(c);
}

static void test_add_wav(void) {
    printf("\n=== Add WAV File ===\n");

    char tmppath[] = "/tmp/rcache_wav_XXXXXX";
    int fd = mkstemp(tmppath);
    TEST(fd >= 0, "mkstemp succeeds");
    close(fd);
    unlink(tmppath);

    float pcm[480] = {0};
    for (int i = 0; i < 480; i++) pcm[i] = 0.05f;
    int wrote = write_test_wav(tmppath, pcm, 480, 24000);
    TEST(wrote == 0, "write test WAV");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);
    int r = response_cache_add_wav(c, FAST_ACKNOWLEDGE, "Got it.", tmppath);
    TEST(r == 0, "add_wav returns 0");

    int len = 0;
    const float *got = response_cache_get(c, FAST_ACKNOWLEDGE, &len);
    TEST(got != NULL, "get returns non-NULL");
    TEST(len == 480, "get returns 480 samples");

    unlink(tmppath);
    response_cache_destroy(c);
}

static void test_multiple_variants(void) {
    printf("\n=== Multiple Variants, Get Returns One ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);
    float a[100], b[100], c_[100];
    for (int i = 0; i < 100; i++) { a[i] = 0.1f; b[i] = 0.2f; c_[i] = 0.3f; }
    response_cache_add(c, FAST_YES, "Yes", a, 100);
    response_cache_add(c, FAST_YES, "Absolutely", b, 100);
    response_cache_add(c, FAST_YES, "Sure", c_, 100);

    int len = 0;
    const float *got = response_cache_get(c, FAST_YES, &len);
    TEST(got != NULL, "get returns non-NULL");
    TEST(len == 100, "get returns 100 samples");
    TEST(response_cache_variant_count(c, FAST_YES) == 3, "variant_count is 3");

    response_cache_destroy(c);
}

static void test_get_variant_deterministic(void) {
    printf("\n=== Specific Variant Access ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);
    float a[50] = {0.1f}, b[50] = {0.2f};
    response_cache_add(c, FAST_NO, "No", a, 50);
    response_cache_add(c, FAST_NO, "Nope", b, 50);

    int len = 0;
    const float *v0 = response_cache_get_variant(c, FAST_NO, 0, &len);
    TEST(v0 != NULL && len == 50, "get_variant 0 works");
    TEST(v0[0] == 0.1f, "variant 0 has expected value");

    len = 0;
    const float *v1 = response_cache_get_variant(c, FAST_NO, 1, &len);
    TEST(v1 != NULL && len == 50 && v1[0] == 0.2f, "get_variant 1 works");

    len = 0;
    const float *v2 = response_cache_get_variant(c, FAST_NO, 2, &len);
    TEST(v2 == NULL, "get_variant 2 (nonexistent) returns NULL");

    response_cache_destroy(c);
}

static void test_has_variant_count(void) {
    printf("\n=== Has / Variant Count ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);
    TEST(response_cache_has(c, FAST_GREETING) == 0, "has empty returns 0");
    TEST(response_cache_variant_count(c, FAST_GREETING) == 0, "variant_count empty is 0");

    float pcm[10] = {0};
    response_cache_add(c, FAST_THANKS, "Welcome", pcm, 10);
    TEST(response_cache_has(c, FAST_THANKS) == 1, "has cached returns 1");
    TEST(response_cache_variant_count(c, FAST_THANKS) == 1, "variant_count is 1");

    response_cache_destroy(c);
}

static void test_stats(void) {
    printf("\n=== Stats ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);
    float pcm[240] = {0};
    response_cache_add(c, FAST_GREETING, "Hi", pcm, 240);
    response_cache_add(c, FAST_ACKNOWLEDGE, "Got it", pcm, 240);

    int entries = 0;
    float seconds = 0.0f;
    response_cache_stats(c, &entries, &seconds);
    TEST(entries == 2, "stats entries = 2");
    TEST(seconds > 0.019f && seconds < 0.021f, "stats seconds ~0.02");

    response_cache_destroy(c);
}

static void test_clear(void) {
    printf("\n=== Clear ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);
    float pcm[10] = {0};
    response_cache_add(c, FAST_GOODBYE, "Bye", pcm, 10);
    response_cache_clear(c);

    int len = 0;
    const float *got = response_cache_get(c, FAST_GOODBYE, &len);
    TEST(got == NULL, "get after clear returns NULL");
    TEST(response_cache_has(c, FAST_GOODBYE) == 0, "has after clear returns 0");

    response_cache_destroy(c);
}

static void test_save_load(void) {
    printf("\n=== Save / Load Round-Trip ===\n");

    char tmppath[] = "/tmp/test_rcache_save_XXXXXX";
    int fd = mkstemp(tmppath);
    TEST(fd >= 0, "mkstemp for save");
    close(fd);

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c1 = response_cache_create(&cfg);
    float pcm[120] = {0};
    for (int i = 0; i < 120; i++) pcm[i] = (float)i * 0.001f;
    response_cache_add(c1, FAST_THINKING, "One moment", pcm, 120);

    int r1 = response_cache_save(c1, tmppath);
    TEST(r1 == 0, "save returns 0");

    ResponseCache *c2 = response_cache_create(&cfg);
    int r2 = response_cache_load(c2, tmppath);
    TEST(r2 == 0, "load returns 0");

    int len = 0;
    const float *got = response_cache_get_variant(c2, FAST_THINKING, 0, &len);
    TEST(got != NULL && len == 120, "loaded audio matches");
    TEST(memcmp(got, pcm, 120 * sizeof(float)) == 0, "loaded audio identical");

    unlink(tmppath);
    response_cache_destroy(c1);
    response_cache_destroy(c2);
}

static int mock_synth(void *ctx, const char *text, float *out_pcm, int max_samples) {
    (void)ctx;
    (void)text;
    int n = max_samples < 96 ? max_samples : 96;
    for (int i = 0; i < n; i++) out_pcm[i] = 0.1f;
    return n;
}

static void test_warm(void) {
    printf("\n=== Warm with Mock TTS ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);
    int n = response_cache_warm(c, mock_synth, NULL);
    TEST(n > 0, "warm returns positive count");

    int len = 0;
    const float *got = response_cache_get(c, FAST_GREETING, &len);
    TEST(got != NULL && len == 96, "warmed greeting is cached");
    TEST(got[0] == 0.1f, "warmed audio has expected value");

    response_cache_destroy(c);
}

static void test_speaker_embedding(void) {
    printf("\n=== Speaker Embedding Set ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);
    float emb[64] = {0};
    response_cache_set_speaker(c, emb, 64);
    response_cache_set_speaker(c, NULL, 0);
    response_cache_destroy(c);
    TEST(1, "set_speaker and destroy succeed");
}

static void test_null_handling(void) {
    printf("\n=== NULL Handling ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);

    TEST(response_cache_create(NULL) == NULL, "create NULL cfg returns NULL");
    ResponseCacheConfig bad = { .sample_rate = 0 };
    TEST(response_cache_create(&bad) == NULL, "create sample_rate=0 returns NULL");

    int len = 0;
    TEST(response_cache_get(NULL, FAST_GREETING, &len) == NULL, "get NULL cache returns NULL");
    TEST(response_cache_get(c, FAST_GREETING, NULL) == NULL, "get NULL out_len returns NULL");

    float pcm[10] = {0};
    TEST(response_cache_add(NULL, FAST_GREETING, "Hi", pcm, 10) == -1, "add NULL cache returns -1");
    TEST(response_cache_add(c, FAST_GREETING, NULL, pcm, 10) == -1, "add NULL text returns -1");
    TEST(response_cache_add(c, FAST_GREETING, "Hi", NULL, 10) == -1, "add NULL pcm returns -1");

    response_cache_destroy(c);
}

static void test_empty_cache_get(void) {
    printf("\n=== Empty Cache Get Returns NULL ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);

    int len = 0;
    const float *got = response_cache_get(c, FAST_GREETING, &len);
    TEST(got == NULL, "get empty type returns NULL");
    TEST(len == 0, "out_len set to 0");

    response_cache_destroy(c);
}

static void test_overflow_protection(void) {
    printf("\n=== Overflow Protection ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 2, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);
    float pcm[10] = {0};

    response_cache_add(c, FAST_YES, "Yes", pcm, 10);
    response_cache_add(c, FAST_YES, "No", pcm, 10);
    int r = response_cache_add(c, FAST_YES, "Maybe", pcm, 10);
    TEST(r == -1, "add beyond max_variants returns -1");
    TEST(response_cache_variant_count(c, FAST_YES) == 2, "variant_count stays at 2");

    response_cache_destroy(c);
}

static void test_invalid_fast_type(void) {
    printf("\n=== Invalid Fast Type ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);
    float pcm[10] = {0};

    int r = response_cache_add(c, -1, "Hi", pcm, 10);
    TEST(r == -1, "add fast_type -1 returns -1");
    r = response_cache_add(c, FAST_COUNT, "Hi", pcm, 10);
    TEST(r == -1, "add fast_type FAST_COUNT returns -1");

    int len = 0;
    TEST(response_cache_get(c, -1, &len) == NULL, "get -1 returns NULL");
    TEST(response_cache_get(c, FAST_COUNT, &len) == NULL, "get FAST_COUNT returns NULL");

    response_cache_destroy(c);
}

static void test_load_invalid_file(void) {
    printf("\n=== Load Invalid File ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);

    int r = response_cache_load(c, "/nonexistent/path/rcache.bin");
    TEST(r == -1, "load nonexistent returns -1");

    response_cache_destroy(c);
}

static void test_save_null(void) {
    printf("\n=== Save NULL Path ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);
    float pcm[10] = {0};
    response_cache_add(c, FAST_GREETING, "Hi", pcm, 10);

    int r = response_cache_save(NULL, "/tmp/out.bin");
    TEST(r == -1, "save NULL cache returns -1");
    r = response_cache_save(c, NULL);
    TEST(r == -1, "save NULL path returns -1");

    response_cache_destroy(c);
}

static void test_get_counter_rotation(void) {
    printf("\n=== Get Variant Rotation ===\n");

    ResponseCacheConfig cfg = { .sample_rate = 24000, .max_variants = 3, .max_audio_seconds = 3 };
    ResponseCache *c = response_cache_create(&cfg);
    float a[10] = {0.1f}, b[10] = {0.2f};
    response_cache_add(c, FAST_ACKNOWLEDGE, "A", a, 10);
    response_cache_add(c, FAST_ACKNOWLEDGE, "B", b, 10);

    int len;
    const float *v0 = response_cache_get(c, FAST_ACKNOWLEDGE, &len);
    const float *v1 = response_cache_get(c, FAST_ACKNOWLEDGE, &len);
    const float *v2 = response_cache_get(c, FAST_ACKNOWLEDGE, &len);
    const float *v3 = response_cache_get(c, FAST_ACKNOWLEDGE, &len);
    TEST(v0 && v1 && v2 && v3, "four gets succeed");
    TEST(v0[0] != v1[0] || v1[0] != v2[0] || v2[0] != v3[0], "variants rotate (or at least differ)");

    response_cache_destroy(c);
}

int main(void) {
    printf("\n═══ Response Cache Tests ═══\n");

    test_create_destroy();
    test_add_manual();
    test_add_wav();
    test_multiple_variants();
    test_get_variant_deterministic();
    test_has_variant_count();
    test_stats();
    test_clear();
    test_save_load();
    test_warm();
    test_speaker_embedding();
    test_null_handling();
    test_empty_cache_get();
    test_overflow_protection();
    test_invalid_fast_type();
    test_load_invalid_file();
    test_save_null();
    test_get_counter_rotation();

    printf("\n═══ Results: %d passed, %d failed ═══\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
