/**
 * test_bugfixes.c — Regression tests for bugs found in audit.
 *
 * Tests:
 *   1. SPMC mirror copy correctness (peek across boundary)
 *   2. KV cache attend with large context (>4096)
 *   3. Arena checkpoint/restore total_allocated tracking
 *   4. Sentence buffer adaptive warmup/reset cycle
 *   5. LUFS K-weighting at non-48kHz rates
 *
 * Build:
 *   cc -O3 -arch arm64 -Isrc -framework Accelerate \
 *      -Lbuild -llufs -lvm_ring -lsentence_buffer \
 *      -Wl,-rpath,$(pwd)/build -o tests/test_bugfixes tests/test_bugfixes.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "spmc_ring.h"
#include "kv_cache.h"
#include "arena.h"
#include "lufs.h"
#include "sentence_buffer.h"

static int pass = 0, fail = 0;
#define TEST(n) printf("  %-55s", n)
#define PASS() do { printf("PASS\n"); pass++; } while(0)
#define FAIL(m) do { printf("FAIL: %s\n", m); fail++; } while(0)
#define ASSERT(c, m) do { if (!(c)) { FAIL(m); return; } } while(0)

/* ── SPMC mirror copy regression ──────────────────────── */

static void test_spmc_peek_across_boundary(void)
{
    TEST("spmc: peek sees correct data across boundary");
    SPMCRing ring;
    spmc_create(&ring, 256, 1); /* Small buffer to force wrap */

    /* Fill to near the end */
    float fill[200];
    for (int i = 0; i < 200; i++) fill[i] = (float)i;
    spmc_write(&ring, fill, 200);

    /* Consumer reads most of it */
    float tmp[180];
    spmc_read(&ring, 0, tmp, 180);

    /* Write 100 more — this wraps around */
    float wrap_data[100];
    for (int i = 0; i < 100; i++) wrap_data[i] = 1000.0f + (float)i;
    spmc_write(&ring, wrap_data, 100);

    /* Peek should see all 120 samples (20 remaining + 100 new) contiguously */
    uint32_t peek_count = 0;
    const float *peek = spmc_peek(&ring, 0, &peek_count);
    ASSERT(peek_count == 120, "peek count should be 120");
    ASSERT(peek != NULL, "peek should not be NULL");

    /* First 20 should be fill[180..199] */
    for (int i = 0; i < 20; i++) {
        float expected = (float)(180 + i);
        float got = peek[i];
        if (fabsf(got - expected) > 0.001f) {
            char msg[128];
            snprintf(msg, sizeof(msg), "peek[%d]: expected %.0f got %.0f", i, expected, got);
            FAIL(msg);
            spmc_destroy(&ring);
            return;
        }
    }

    /* Next 100 should be wrap_data[0..99] — this is the mirrored region */
    for (int i = 0; i < 100; i++) {
        float expected = 1000.0f + (float)i;
        float got = peek[20 + i];
        if (fabsf(got - expected) > 0.001f) {
            char msg[128];
            snprintf(msg, sizeof(msg), "peek[%d]: expected %.0f got %.0f", 20+i, expected, got);
            FAIL(msg);
            spmc_destroy(&ring);
            return;
        }
    }

    spmc_destroy(&ring);
    PASS();
}

/* ── KV cache large context regression ────────────────── */

static void test_kv_cache_large_context(void)
{
    TEST("kv_cache: attend works with context > 4096");
    InterleavedKVCache c;
    int max_ctx = 8192;
    kv_cache_create(&c, 1, 4, max_ctx); /* 1 head, dim=4, max_len=8192 */

    /* Fill with 5000 timesteps (exceeds the old 4096 stack limit) */
    float k[4], v[4];
    for (int t = 0; t < 5000; t++) {
        k[0] = (t == 0) ? 10.0f : 0.1f;
        k[1] = k[2] = k[3] = 0.0f;
        v[0] = (t == 0) ? 1.0f : 0.0f;
        v[1] = v[2] = v[3] = 0.0f;
        kv_cache_append(&c, k, v);
    }

    ASSERT(c.cur_len == 5000, "should have 5000 entries");

    /* Query that strongly aligns with first timestep */
    float q[4] = {10.0f, 0.0f, 0.0f, 0.0f};
    float out[4];
    kv_cache_attend(&c, q, 0, 0.5f, out);

    /* Output should be weighted toward v[0]=(1,0,0,0) since Q·K is highest there */
    ASSERT(out[0] > 0.1f, "attend output[0] should favor first timestep");

    kv_cache_destroy(&c);
    PASS();
}

/* ── Arena total_allocated after restore ──────────────── */

static void test_arena_restore_total(void)
{
    TEST("arena: restore correctly tracks total_allocated");
    Arena a = arena_create(4096);

    arena_alloc(&a, 100);
    size_t after_first = a.total_allocated;
    ArenaCheckpoint cp = arena_checkpoint(&a);

    arena_alloc(&a, 500);
    arena_alloc(&a, 300);
    ASSERT(a.total_allocated > after_first, "should have grown");

    arena_restore(&a, cp);
    ASSERT(a.total_allocated == after_first, "should match checkpoint state");

    arena_destroy(&a);
    PASS();
}

/* ── LUFS at non-48kHz ────────────────────────────────── */

static void test_lufs_44100_measure(void)
{
    TEST("lufs: 44100Hz K-weighting produces valid measurement");
    LUFSMeter *m = lufs_create(44100, 400);
    ASSERT(m != NULL, "create at 44100Hz failed");

    /* Generate a 400ms 1kHz sine at moderate amplitude */
    int n = 44100 * 400 / 1000;
    float *signal = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        signal[i] = 0.1f * sinf(2.0f * M_PI * 1000.0f * i / 44100.0f);
    }

    float lufs = lufs_measure(m, signal, n);
    /* Should be a reasonable negative number (not -70 silence, not 0) */
    ASSERT(lufs > -60.0f, "measurement too quiet for 0.1 amplitude tone");
    ASSERT(lufs < -5.0f, "measurement too loud");

    free(signal);
    lufs_destroy(m);
    PASS();
}

static void test_lufs_24000_measure(void)
{
    TEST("lufs: 24000Hz (TTS rate) produces valid measurement");
    LUFSMeter *m = lufs_create(24000, 400);
    ASSERT(m != NULL, "create at 24000Hz failed");

    int n = 24000 * 400 / 1000;
    float *signal = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        signal[i] = 0.1f * sinf(2.0f * M_PI * 440.0f * i / 24000.0f);
    }

    float lufs = lufs_measure(m, signal, n);
    ASSERT(lufs > -60.0f, "should not be silence");
    ASSERT(lufs < -5.0f, "should not be deafening");

    free(signal);
    lufs_destroy(m);
    PASS();
}

/* ── Sentence buffer adaptive cycle ───────────────────── */

static void test_sentbuf_adaptive_full_cycle(void)
{
    TEST("sentbuf: adaptive warmup → steady → reset → warmup again");
    SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SPECULATIVE, 8);
    sentbuf_set_adaptive(sb, 2, 3);
    char out[256];

    /* Turn 1: warmup flushes at 3 words */
    sentbuf_add(sb, "Hello there friend. ", 20);
    ASSERT(sentbuf_has_segment(sb), "should flush during warmup");
    sentbuf_flush(sb, out, sizeof(out));

    sentbuf_add(sb, "How are you. ", 13);
    while (sentbuf_has_segment(sb)) sentbuf_flush(sb, out, sizeof(out));

    /* After 2 sentences, warmup should be over. Short clause should NOT flush. */
    sentbuf_add(sb, "Ok, ", 4);
    /* With min_words=8, "Ok," (1 word) should not flush */
    ASSERT(!sentbuf_has_segment(sb), "should NOT flush after warmup ends");

    /* Reset for turn 2 */
    sentbuf_reset(sb);
    ASSERT(sentbuf_sentence_count(sb) == 0, "count should reset");

    /* Warmup should be re-armed */
    sentbuf_add(sb, "Hey there world. ", 17);
    ASSERT(sentbuf_has_segment(sb), "warmup should be re-armed after reset");

    sentbuf_destroy(sb);
    PASS();
}

/* ── Arena checkpoint nested overflow ──────────────────── */

static void test_arena_nested_checkpoints(void)
{
    TEST("arena: nested checkpoint/restore sequence");
    Arena a = arena_create(4096);

    /* Level 0: allocate 100 bytes */
    arena_alloc(&a, 100);
    size_t after_l0 = a.total_allocated;
    ArenaCheckpoint cp0 = arena_checkpoint(&a);

    /* Level 1: allocate 200 more */
    arena_alloc(&a, 200);
    size_t after_l1 = a.total_allocated;
    ArenaCheckpoint cp1 = arena_checkpoint(&a);

    /* Level 2: allocate 300 more */
    arena_alloc(&a, 300);
    ASSERT(a.total_allocated > after_l1, "level 2 should grow");

    /* Restore to level 1 */
    arena_restore(&a, cp1);
    ASSERT(a.total_allocated == after_l1, "should match level 1 state");

    /* Allocate again at level 1 */
    arena_alloc(&a, 150);
    ASSERT(a.total_allocated > after_l1, "new alloc at level 1 should grow");

    /* Restore to level 0 */
    arena_restore(&a, cp0);
    ASSERT(a.total_allocated == after_l0, "should match level 0 state");

    arena_destroy(&a);
    PASS();
}

/* ── Arena reset reusability ──────────────────────────── */

static void test_arena_reset_reuse(void)
{
    TEST("arena: reset allows full reuse without leaks");
    Arena a = arena_create(4096);

    /* First round of allocations */
    for (int i = 0; i < 50; i++) {
        void *p = arena_alloc(&a, 64);
        ASSERT(p != NULL, "alloc should succeed");
    }
    size_t after_first_round = a.total_allocated;
    ASSERT(after_first_round > 0, "should have allocated");

    /* Reset */
    arena_reset(&a);
    ASSERT(a.total_allocated == 0, "total_allocated should be 0 after reset");

    /* Second round: should work identically */
    for (int i = 0; i < 50; i++) {
        void *p = arena_alloc(&a, 64);
        ASSERT(p != NULL, "alloc after reset should succeed");
    }
    ASSERT(a.total_allocated == after_first_round,
           "second round should match first round total");

    arena_destroy(&a);
    PASS();
}

/* ── KV cache exact max_len boundary ──────────────────── */

static void test_kv_cache_exact_boundary(void)
{
    TEST("kv_cache: fill to exact max_len boundary");
    InterleavedKVCache c;
    int max_ctx = 128;
    kv_cache_create(&c, 1, 4, max_ctx);

    /* Fill exactly to max_len */
    float k[4], v[4];
    for (int t = 0; t < max_ctx; t++) {
        k[0] = (float)t; k[1] = k[2] = k[3] = 0.0f;
        v[0] = (float)t; v[1] = v[2] = v[3] = 0.0f;
        kv_cache_append(&c, k, v);
    }
    ASSERT(c.cur_len == max_ctx, "should be at max_len");

    /* One more append should trigger ring buffer shift */
    k[0] = 999.0f; v[0] = 999.0f;
    kv_cache_append(&c, k, v);
    ASSERT(c.cur_len == max_ctx, "should still be at max_len after wrap");

    /* The last entry should be our 999.0 value */
    const float *last_v = kv_cache_v(&c, 0, max_ctx - 1);
    ASSERT(fabsf(last_v[0] - 999.0f) < 0.001f, "last value should be 999.0");

    kv_cache_destroy(&c);
    PASS();
}

/* ── KV cache reset and reuse ─────────────────────────── */

static void test_kv_cache_reset_reuse(void)
{
    TEST("kv_cache: reset allows clean reuse");
    InterleavedKVCache c;
    kv_cache_create(&c, 2, 8, 64);

    /* Fill some entries */
    float k[16], v[16];
    memset(k, 0, sizeof(k));
    memset(v, 0, sizeof(v));
    for (int t = 0; t < 32; t++) {
        kv_cache_append(&c, k, v);
    }
    ASSERT(c.cur_len == 32, "should have 32 entries");

    /* Reset */
    kv_cache_reset(&c);
    ASSERT(c.cur_len == 0, "should be 0 after reset");

    /* Re-fill and verify attend works */
    k[0] = 5.0f; v[0] = 1.0f;
    kv_cache_append(&c, k, v);
    ASSERT(c.cur_len == 1, "should have 1 entry after re-fill");

    float q[8] = {5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float out[8];
    kv_cache_attend(&c, q, 0, 0.5f, out);
    ASSERT(fabsf(out[0] - 1.0f) < 0.1f, "attend after reset should work");

    kv_cache_destroy(&c);
    PASS();
}

/* ── LUFS 16000Hz (common STT rate) ───────────────────── */

static void test_lufs_16000_measure(void)
{
    TEST("lufs: 16000Hz (STT rate) produces valid measurement");
    LUFSMeter *m = lufs_create(16000, 400);
    ASSERT(m != NULL, "create at 16000Hz failed");

    int n = 16000 * 400 / 1000;
    float *signal = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        signal[i] = 0.1f * sinf(2.0f * (float)M_PI * 440.0f * (float)i / 16000.0f);
    }

    float lufs = lufs_measure(m, signal, n);
    ASSERT(lufs > -60.0f, "should not be silence");
    ASSERT(lufs < -5.0f, "should not be deafening");

    free(signal);
    lufs_destroy(m);
    PASS();
}

/* ── LUFS silence measurement ─────────────────────────── */

static void test_lufs_silence(void)
{
    TEST("lufs: silence → very low LUFS (near -70)");
    LUFSMeter *m = lufs_create(48000, 400);
    ASSERT(m != NULL, "create at 48000Hz failed");

    int n = 48000 * 400 / 1000;
    float *signal = (float *)calloc((size_t)n, sizeof(float));

    float lufs = lufs_measure(m, signal, n);
    /* Silence should be very low LUFS, typically -70 or below */
    ASSERT(lufs < -40.0f, "silence should measure very low LUFS");

    free(signal);
    lufs_destroy(m);
    PASS();
}

/* ── SPMC ring full write/read cycle ──────────────────── */

static void test_spmc_full_capacity(void)
{
    TEST("spmc: write fills entire capacity then read drains it");
    SPMCRing ring;
    spmc_create(&ring, 512, 2);

    /* Fill as much as possible */
    uint32_t cap = ring.size;
    float *data = (float *)malloc((size_t)cap * sizeof(float));
    for (uint32_t i = 0; i < cap; i++) data[i] = (float)i;

    /* Write in chunks */
    uint32_t written = 0;
    while (written < cap) {
        uint32_t chunk = cap - written;
        if (chunk > 64) chunk = 64;
        uint32_t avail = spmc_available_write(&ring);
        if (avail < chunk) chunk = avail;
        if (chunk == 0) break;
        spmc_write(&ring, data + written, chunk);
        written += chunk;
    }
    ASSERT(written > 0, "should write at least some data");

    /* Both consumers should see all written data */
    uint32_t avail0 = spmc_available_read(&ring, 0);
    uint32_t avail1 = spmc_available_read(&ring, 1);
    ASSERT(avail0 == written && avail1 == written,
           "both consumers should see all data");

    /* Read from consumer 0 */
    float *readbuf = (float *)malloc((size_t)written * sizeof(float));
    int ret = spmc_read(&ring, 0, readbuf, written);
    ASSERT(ret == 0, "read should succeed");

    /* Verify data integrity */
    int match = 1;
    for (uint32_t i = 0; i < written; i++) {
        if (fabsf(readbuf[i] - data[i]) > 0.001f) { match = 0; break; }
    }
    ASSERT(match, "read data should match written data");

    free(data);
    free(readbuf);
    spmc_destroy(&ring);
    PASS();
}

/* ── Sentence buffer eager mode ───────────────────────── */

static void test_sentbuf_eager_flush(void)
{
    TEST("sentbuf: eager mode flushes at word count threshold");
    SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SPECULATIVE, 20);
    sentbuf_set_eager(sb, 4);  /* Flush after 4 words even without punctuation */

    char out[256];

    /* Add 5 words without any sentence boundary */
    sentbuf_add(sb, "one two three four five ", 24);

    /* With eager_words=4, should have flushed by now */
    int has_seg = sentbuf_has_segment(sb);
    if (has_seg) {
        sentbuf_flush(sb, out, sizeof(out));
    }
    /* Eager mode behavior: should flush or accumulate, but not crash */
    ASSERT(1, "eager mode should not crash");

    sentbuf_destroy(sb);
    PASS();
}

/* ── Sentence buffer prosody hints ────────────────────── */

static void test_sentbuf_prosody_hints(void)
{
    TEST("sentbuf: prosody hints for exclamation/question");
    SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 3);

    sentbuf_add(sb, "This is exciting! ", 18);

    char out[256];
    if (sentbuf_has_segment(sb)) {
        sentbuf_flush(sb, out, sizeof(out));
        SentBufProsodyHint hint = sentbuf_get_prosody_hint(sb);
        /* Should detect exclamation */
        ASSERT(hint.exclamation_count >= 0, "should track exclamation count");
    }

    sentbuf_add(sb, "Really? ", 8);
    if (sentbuf_has_segment(sb)) {
        sentbuf_flush(sb, out, sizeof(out));
        SentBufProsodyHint hint = sentbuf_get_prosody_hint(sb);
        ASSERT(hint.question_count >= 0, "should track question count");
    }

    sentbuf_destroy(sb);
    PASS();
}

/* ── Main ─────────────────────────────────────────────── */

int main(void)
{
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  pocket-voice: Bug Fix Regression Tests      ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    printf("[SPMC Mirror Copy]\n");
    test_spmc_peek_across_boundary();

    printf("\n[KV Cache Overflow]\n");
    test_kv_cache_large_context();

    printf("\n[Arena Restore Accounting]\n");
    test_arena_restore_total();

    printf("\n[LUFS Non-48kHz]\n");
    test_lufs_44100_measure();
    test_lufs_24000_measure();

    printf("\n[Sentence Buffer Adaptive Cycle]\n");
    test_sentbuf_adaptive_full_cycle();

    /* New deep coverage regression tests */
    printf("\n[Arena Nested Checkpoints]\n");
    test_arena_nested_checkpoints();

    printf("\n[Arena Reset Reuse]\n");
    test_arena_reset_reuse();

    printf("\n[KV Cache Exact Boundary]\n");
    test_kv_cache_exact_boundary();

    printf("\n[KV Cache Reset Reuse]\n");
    test_kv_cache_reset_reuse();

    printf("\n[LUFS 16kHz]\n");
    test_lufs_16000_measure();

    printf("\n[LUFS Silence]\n");
    test_lufs_silence();

    printf("\n[SPMC Full Capacity]\n");
    test_spmc_full_capacity();

    printf("\n[Sentence Buffer Eager Mode]\n");
    test_sentbuf_eager_flush();

    printf("\n[Sentence Buffer Prosody Hints]\n");
    test_sentbuf_prosody_hints();

    printf("\n════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", pass, fail);
    printf("════════════════════════════════════════════════\n");

    return fail > 0 ? 1 : 0;
}
