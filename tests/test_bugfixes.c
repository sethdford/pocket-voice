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

    printf("\n════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", pass, fail);
    printf("════════════════════════════════════════════════\n");

    return fail > 0 ? 1 : 0;
}
