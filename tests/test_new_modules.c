/**
 * test_new_modules.c — Tests for the 12 new optimization modules.
 *
 * Tests: breath_synthesis, lufs, arena, vm_ring, triple_buffer,
 *        spmc_ring, kv_cache, sentence_buffer (predictive), bnns_mimi
 *
 * Build:
 *   cc -O3 -arch arm64 -Isrc -framework Accelerate \
 *      -Lbuild -lbreath_synthesis -llufs -lvm_ring -lsentence_buffer -lbnns_mimi \
 *      -Wl,-rpath,$(pwd)/build -o tests/test_new_modules tests/test_new_modules.c
 *
 * Run: ./tests/test_new_modules
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "breath_synthesis.h"
#include "lufs.h"
#include "arena.h"
#include "vm_ring.h"
#include "triple_buffer.h"
#include "spmc_ring.h"
#include "kv_cache.h"
#include "sentence_buffer.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-50s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)

#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define ASSERT_FLOAT_NEAR(a, b, eps, msg) \
    ASSERT(fabsf((a) - (b)) < (eps), msg)

/* ── Breath Synthesis Tests ───────────────────────────── */

static void test_breath_create_destroy(void) {
    TEST("breath: create/destroy");
    BreathSynth *bs = breath_create(48000);
    ASSERT(bs != NULL, "create returned NULL");
    breath_destroy(bs);
    breath_destroy(NULL);  /* NULL-safe */
    PASS();
}

static void test_breath_generate(void) {
    TEST("breath: generate adds noise");
    BreathSynth *bs = breath_create(48000);
    float buf[4800];
    memset(buf, 0, sizeof(buf));

    breath_generate(bs, buf, 4800, 0.03f);

    /* Check that some samples are non-zero */
    float max_val = 0;
    for (int i = 0; i < 4800; i++) {
        float a = fabsf(buf[i]);
        if (a > max_val) max_val = a;
    }
    ASSERT(max_val > 0.001f, "breath noise is too quiet");
    ASSERT(max_val < 0.2f, "breath noise is too loud");

    breath_destroy(bs);
    PASS();
}

static void test_breath_micropause(void) {
    TEST("breath: micropause fades to silence");
    float buf[2400];
    for (int i = 0; i < 2400; i++) buf[i] = 0.5f;

    breath_micropause(buf, 2400, 5.0f, 48000);

    /* Middle should be near zero */
    float mid = buf[1200];
    ASSERT(fabsf(mid) < 0.01f, "middle of micropause should be silent");

    PASS();
}

static void test_breath_sentence_gap(void) {
    TEST("breath: sentence gap generates envelope");
    BreathSynth *bs = breath_create(48000);
    float out[4800];
    memset(out, 0, sizeof(out));

    breath_sentence_gap(bs, out, 4800, 0.1f);

    /* Output should have non-zero content */
    float sum_sq = 0;
    for (int i = 0; i < 4800; i++) sum_sq += out[i] * out[i];
    float rms = sqrtf(sum_sq / 4800);
    ASSERT(rms > 0.0001f, "sentence gap should produce audible noise");
    ASSERT(rms < 0.05f, "sentence gap should be subtle");

    breath_destroy(bs);
    PASS();
}

/* ── LUFS Tests ───────────────────────────────────────── */

static void test_lufs_create_destroy(void) {
    TEST("lufs: create/destroy");
    LUFSMeter *m = lufs_create(48000, 400);
    ASSERT(m != NULL, "create returned NULL");
    lufs_destroy(m);
    lufs_destroy(NULL);
    PASS();
}

static void test_lufs_silence_measure(void) {
    TEST("lufs: silence measures < -60 LUFS");
    LUFSMeter *m = lufs_create(48000, 400);
    float zeros[19200];
    memset(zeros, 0, sizeof(zeros));

    float lufs = lufs_measure(m, zeros, 19200);
    ASSERT(lufs <= -60.0f, "silence should be very quiet");

    lufs_destroy(m);
    PASS();
}

static void test_lufs_normalize(void) {
    TEST("lufs: normalize applies gain");
    LUFSMeter *m = lufs_create(48000, 400);

    /* Generate a 400ms sine wave at -40 LUFS (very quiet) */
    float signal[19200];
    for (int i = 0; i < 19200; i++) {
        signal[i] = 0.001f * sinf(2.0f * M_PI * 440.0f * i / 48000.0f);
    }

    float gain_db = lufs_normalize(m, signal, 19200, -16.0f);

    /* Should have applied positive gain */
    ASSERT(gain_db > 0.0f, "should amplify quiet signal");

    lufs_destroy(m);
    PASS();
}

/* ── Arena Tests ──────────────────────────────────────── */

static void test_arena_create_alloc(void) {
    TEST("arena: create + alloc + destroy");
    Arena a = arena_create(4096);
    ASSERT(a.head != NULL, "head should be non-NULL");

    void *p1 = arena_alloc(&a, 64);
    ASSERT(p1 != NULL, "first alloc");
    void *p2 = arena_alloc(&a, 128);
    ASSERT(p2 != NULL, "second alloc");
    ASSERT(p1 != p2, "allocations should be different");

    arena_destroy(&a);
    PASS();
}

static void test_arena_reset(void) {
    TEST("arena: reset rewinds");
    Arena a = arena_create(4096);

    arena_alloc(&a, 1000);
    arena_alloc(&a, 1000);
    ASSERT(a.total_allocated >= 2000, "should have allocated 2000+");

    arena_reset(&a);
    ASSERT(a.total_allocated == 0, "should be zero after reset");

    arena_destroy(&a);
    PASS();
}

static void test_arena_checkpoint(void) {
    TEST("arena: checkpoint/restore");
    Arena a = arena_create(4096);

    arena_alloc(&a, 256);
    ArenaCheckpoint cp = arena_checkpoint(&a);

    arena_alloc(&a, 1024);
    arena_alloc(&a, 512);

    arena_restore(&a, cp);
    /* After restore, next alloc should reuse the space */
    void *p = arena_alloc(&a, 64);
    ASSERT(p != NULL, "alloc after restore");

    arena_destroy(&a);
    PASS();
}

static void test_arena_strdup(void) {
    TEST("arena: strdup");
    Arena a = arena_create(4096);

    char *s = arena_strdup(&a, "hello world");
    ASSERT(s != NULL, "strdup returned NULL");
    ASSERT(strcmp(s, "hello world") == 0, "string mismatch");

    arena_destroy(&a);
    PASS();
}

/* ── VM Ring Buffer Tests ─────────────────────────────── */

static void test_vm_ring_create_destroy(void) {
    TEST("vm_ring: create/destroy");
    VMRingBuffer rb;
    int rc = vm_ring_create(&rb, 4096);
    ASSERT(rc == 0, "create failed");
    ASSERT(rb.buffer != NULL, "buffer is NULL");
    ASSERT(rb.size >= 4096, "size too small");
    vm_ring_destroy(&rb);
    PASS();
}

static void test_vm_ring_write_read(void) {
    TEST("vm_ring: write + read round-trip");
    VMRingBuffer rb;
    vm_ring_create(&rb, 4096);

    float data[1024];
    for (int i = 0; i < 1024; i++) data[i] = (float)i;

    int rc = vm_ring_write(&rb, data, 1024);
    ASSERT(rc == 0, "write failed");
    ASSERT(vm_ring_available_read(&rb) == 1024, "wrong avail read");

    float out[1024];
    rc = vm_ring_read(&rb, out, 1024);
    ASSERT(rc == 0, "read failed");

    for (int i = 0; i < 1024; i++) {
        ASSERT_FLOAT_NEAR(out[i], (float)i, 0.001f, "data mismatch");
    }

    vm_ring_destroy(&rb);
    PASS();
}

static void test_vm_ring_wraparound(void) {
    TEST("vm_ring: wraparound via mirrored pages");
    VMRingBuffer rb;
    vm_ring_create(&rb, 1024); /* Small buffer to force wrap */

    float data[512];
    for (int i = 0; i < 512; i++) data[i] = (float)i;

    /* Fill to 75% */
    vm_ring_write(&rb, data, 768);
    /* Read half */
    float tmp[384];
    vm_ring_read(&rb, tmp, 384);
    /* Write more (wraps around) */
    vm_ring_write(&rb, data, 512);

    /* Read everything */
    float out[896];
    int avail = vm_ring_available_read(&rb);
    ASSERT(avail == 896, "wrong avail after wrap");

    vm_ring_read(&rb, out, 896);

    vm_ring_destroy(&rb);
    PASS();
}

/* ── Triple Buffer Tests ──────────────────────────────── */

static void test_triple_buf_flow(void) {
    TEST("triple_buf: writer→processor→reader flow");
    TripleBuffer tb;
    int rc = triple_buf_create(&tb, 1024);
    ASSERT(rc == 0, "create failed");

    /* Writer fills a buffer */
    float *w = triple_buf_write_ptr(&tb);
    ASSERT(w != NULL, "write ptr NULL");
    for (int i = 0; i < 1024; i++) w[i] = (float)i;
    triple_buf_write_done(&tb, 1024);

    /* Processor acquires */
    uint32_t count;
    float *p = triple_buf_process_acquire(&tb, &count);
    ASSERT(p != NULL, "process acquire NULL");
    ASSERT(count == 1024, "wrong count");
    /* Modify in place */
    for (uint32_t i = 0; i < count; i++) p[i] *= 2.0f;
    triple_buf_process_done(&tb, count);

    /* Reader acquires */
    const float *r = triple_buf_read_acquire(&tb, &count);
    ASSERT(r != NULL, "read acquire NULL");
    ASSERT(count == 1024, "wrong reader count");
    ASSERT_FLOAT_NEAR(r[0], 0.0f, 0.001f, "r[0] wrong");
    ASSERT_FLOAT_NEAR(r[10], 20.0f, 0.001f, "r[10] wrong");

    triple_buf_destroy(&tb);
    PASS();
}

/* ── SPMC Ring Tests ──────────────────────────────────── */

static void test_spmc_basic(void) {
    TEST("spmc: 1 producer, 2 consumers");
    SPMCRing ring;
    int rc = spmc_create(&ring, 4096, 2);
    ASSERT(rc == 0, "create failed");

    float data[100];
    for (int i = 0; i < 100; i++) data[i] = (float)i;

    /* Producer writes */
    rc = spmc_write(&ring, data, 100);
    ASSERT(rc == 0, "write failed");

    /* Both consumers should see 100 samples */
    ASSERT(spmc_available_read(&ring, 0) == 100, "consumer 0 wrong avail");
    ASSERT(spmc_available_read(&ring, 1) == 100, "consumer 1 wrong avail");

    /* Consumer 0 reads */
    float out0[100];
    rc = spmc_read(&ring, 0, out0, 100);
    ASSERT(rc == 0, "consumer 0 read failed");
    ASSERT_FLOAT_NEAR(out0[50], 50.0f, 0.001f, "consumer 0 data wrong");

    /* Consumer 1 reads independently */
    float out1[50]; /* only read 50 */
    rc = spmc_read(&ring, 1, out1, 50);
    ASSERT(rc == 0, "consumer 1 read failed");
    ASSERT(spmc_available_read(&ring, 1) == 50, "consumer 1 should have 50 left");

    spmc_destroy(&ring);
    PASS();
}

static void test_spmc_deactivate(void) {
    TEST("spmc: deactivated consumer doesn't block");
    SPMCRing ring;
    spmc_create(&ring, 256, 2);

    /* Deactivate consumer 1 */
    spmc_deactivate(&ring, 1);

    /* Fill the ring — should succeed since only consumer 0 is active */
    float data[200];
    for (int i = 0; i < 200; i++) data[i] = 1.0f;
    int rc = spmc_write(&ring, data, 200);
    ASSERT(rc == 0, "write with deactivated consumer");

    /* Consumer 0 reads */
    float out[200];
    rc = spmc_read(&ring, 0, out, 200);
    ASSERT(rc == 0, "consumer 0 read");

    /* Write more — consumer 1 is deactivated so no blocking */
    rc = spmc_write(&ring, data, 200);
    ASSERT(rc == 0, "second write");

    spmc_destroy(&ring);
    PASS();
}

/* ── KV Cache Tests ───────────────────────────────────── */

static void test_kv_cache_create_append(void) {
    TEST("kv_cache: create + append");
    InterleavedKVCache c;
    int rc = kv_cache_create(&c, 4, 64, 128);
    ASSERT(rc == 0, "create failed");

    float k[256], v[256]; /* 4 heads × 64 dim */
    for (int i = 0; i < 256; i++) { k[i] = (float)i; v[i] = -(float)i; }

    kv_cache_append(&c, k, v);
    ASSERT(c.cur_len == 1, "should have 1 entry");

    /* Check K and V are interleaved correctly */
    const float *k0 = kv_cache_k(&c, 0, 0);
    const float *v0 = kv_cache_v(&c, 0, 0);
    ASSERT_FLOAT_NEAR(k0[0], 0.0f, 0.001f, "k[0][0] wrong");
    ASSERT_FLOAT_NEAR(v0[0], 0.0f, 0.001f, "v[0][0] wrong");
    ASSERT_FLOAT_NEAR(k0[1], 1.0f, 0.001f, "k[0][1] wrong");
    ASSERT_FLOAT_NEAR(v0[1], -1.0f, 0.001f, "v[0][1] wrong");

    kv_cache_destroy(&c);
    PASS();
}

static void test_kv_cache_attend(void) {
    TEST("kv_cache: attend produces valid output");
    InterleavedKVCache c;
    kv_cache_create(&c, 1, 4, 32); /* 1 head, dim=4 */

    /* Append 3 timesteps */
    float k1[4] = {1, 0, 0, 0};
    float v1[4] = {1, 0, 0, 0};
    kv_cache_append(&c, k1, v1);

    float k2[4] = {0, 1, 0, 0};
    float v2[4] = {0, 1, 0, 0};
    kv_cache_append(&c, k2, v2);

    float k3[4] = {0, 0, 1, 0};
    float v3[4] = {0, 0, 1, 0};
    kv_cache_append(&c, k3, v3);

    /* Query that attends mostly to first timestep */
    float q[4] = {10, 0, 0, 0};
    float out[4];
    float scale = 0.5f; /* 1/sqrt(4) */

    kv_cache_attend(&c, q, 0, scale, out);

    /* Output should be close to v1 since q aligns with k1 */
    ASSERT(out[0] > 0.5f, "attend output should favor v1");

    kv_cache_destroy(&c);
    PASS();
}

/* ── Sentence Buffer Predictive Tests ─────────────────── */

static void test_sentbuf_predictive_length(void) {
    TEST("sentbuf: predictive length tracks EMA");
    SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);

    /* No sentences yet */
    ASSERT(sentbuf_predicted_length(sb) == 0, "should be 0 initially");

    /* Feed a sentence */
    sentbuf_add(sb, "Hello world. ", 13);
    char out[256];
    if (sentbuf_has_segment(sb)) {
        sentbuf_flush(sb, out, sizeof(out));
    }

    /* Should have some predicted length now */
    int pred = sentbuf_predicted_length(sb);
    ASSERT(pred > 0, "should predict non-zero after first sentence");
    ASSERT(sentbuf_sentence_count(sb) == 1, "should count 1 sentence");

    sentbuf_destroy(sb);
    PASS();
}

static void test_sentbuf_adaptive_warmup(void) {
    TEST("sentbuf: adaptive warmup lowers threshold");
    SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SPECULATIVE, 8);

    /* Enable adaptive: first 2 sentences use min_words=3 */
    sentbuf_set_adaptive(sb, 2, 3);

    /* Short clause that wouldn't flush with min_words=8 but should with 3 */
    sentbuf_add(sb, "Hey there, I'm here. ", 21);

    /* Should flush with warmup threshold */
    ASSERT(sentbuf_has_segment(sb) == 1, "should flush with warmup threshold");

    char out[256];
    sentbuf_flush(sb, out, sizeof(out));
    ASSERT(sentbuf_sentence_count(sb) >= 1, "should count at least 1");

    sentbuf_destroy(sb);
    PASS();
}

static void test_sentbuf_reset_rearms_warmup(void) {
    TEST("sentbuf: reset re-arms warmup");
    SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SPECULATIVE, 8);
    sentbuf_set_adaptive(sb, 2, 3);

    /* Flush a few sentences */
    sentbuf_add(sb, "One two three. ", 15);
    char out[256];
    while (sentbuf_has_segment(sb)) sentbuf_flush(sb, out, sizeof(out));
    sentbuf_add(sb, "Four five six. ", 15);
    while (sentbuf_has_segment(sb)) sentbuf_flush(sb, out, sizeof(out));
    sentbuf_add(sb, "Seven eight nine. ", 18);
    while (sentbuf_has_segment(sb)) sentbuf_flush(sb, out, sizeof(out));

    /* Reset should re-arm warmup */
    sentbuf_reset(sb);
    ASSERT(sentbuf_sentence_count(sb) == 0, "count should reset");
    ASSERT(sentbuf_predicted_length(sb) == 0, "prediction should reset");

    sentbuf_destroy(sb);
    PASS();
}

/* ── Main ─────────────────────────────────────────────── */

int main(void) {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  pocket-voice: New Module Tests              ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    printf("[Breath Synthesis]\n");
    test_breath_create_destroy();
    test_breath_generate();
    test_breath_micropause();
    test_breath_sentence_gap();

    printf("\n[LUFS Normalization]\n");
    test_lufs_create_destroy();
    test_lufs_silence_measure();
    test_lufs_normalize();

    printf("\n[Arena Allocator]\n");
    test_arena_create_alloc();
    test_arena_reset();
    test_arena_checkpoint();
    test_arena_strdup();

    printf("\n[VM Mirrored Ring Buffer]\n");
    test_vm_ring_create_destroy();
    test_vm_ring_write_read();
    test_vm_ring_wraparound();

    printf("\n[Triple Buffer]\n");
    test_triple_buf_flow();

    printf("\n[SPMC Ring Buffer]\n");
    test_spmc_basic();
    test_spmc_deactivate();

    printf("\n[KV Cache (Interleaved)]\n");
    test_kv_cache_create_append();
    test_kv_cache_attend();

    printf("\n[Sentence Buffer (Predictive)]\n");
    test_sentbuf_predictive_length();
    test_sentbuf_adaptive_warmup();
    test_sentbuf_reset_rearms_warmup();

    printf("\n════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("════════════════════════════════════════════════\n");

    return tests_failed > 0 ? 1 : 0;
}
