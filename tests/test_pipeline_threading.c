/**
 * test_pipeline_threading.c — Threading and concurrency stress tests for the
 * SPMC ring buffer, VM-mirrored ring, and pipeline threading primitives.
 *
 * Build:
 *   cc -O2 -Isrc tests/test_pipeline_threading.c -Lbuild -lvm_ring \
 *      -framework Accelerate -lpthread -o tests/test_pipeline_threading
 *
 * Run:
 *   ./tests/test_pipeline_threading
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

#include "spmc_ring.h"

/* ---------- test harness ---------- */

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { g_fail++; printf("  [FAIL] %s\n", msg); } \
} while(0)

#define CHECKF(cond, fmt, ...) do { \
    char _buf[256]; snprintf(_buf, sizeof(_buf), fmt, __VA_ARGS__); \
    if (cond) { g_pass++; printf("  [PASS] %s\n", _buf); } \
    else { g_fail++; printf("  [FAIL] %s\n", _buf); } \
} while(0)

/* ---------- helpers ---------- */

static double elapsed_ms(struct timespec *start, struct timespec *end) {
    double s  = (double)(end->tv_sec  - start->tv_sec)  * 1000.0;
    double ns = (double)(end->tv_nsec - start->tv_nsec) / 1e6;
    return s + ns;
}

/* ---------- 1. rapid write/read ---------- */

static void test_spmc_rapid_write_read(void) {
    printf("\n[test_spmc_rapid_write_read]\n");

    SPMCRing ring;
    int rc = spmc_create(&ring, 8192, 1);
    CHECK(rc == 0, "create 8192-float ring with 1 consumer");

    const uint32_t chunk = 480;
    float wbuf[480], rbuf[480];
    int corrupt = 0;

    for (int iter = 0; iter < 10000; iter++) {
        /* Fill write buffer with incrementing values */
        for (uint32_t i = 0; i < chunk; i++)
            wbuf[i] = (float)(iter * chunk + i);

        rc = spmc_write(&ring, wbuf, chunk);
        if (rc != 0) { corrupt++; break; }

        rc = spmc_read(&ring, 0, rbuf, chunk);
        if (rc != 0) { corrupt++; break; }

        for (uint32_t i = 0; i < chunk; i++) {
            if (rbuf[i] != (float)(iter * chunk + i)) {
                corrupt++;
                break;
            }
        }
    }

    CHECK(corrupt == 0, "10,000 write/read cycles — no corruption");
    spmc_destroy(&ring);
}

/* ---------- 2. concurrent peek ---------- */

static void test_spmc_concurrent_peek(void) {
    printf("\n[test_spmc_concurrent_peek]\n");

    SPMCRing ring;
    int rc = spmc_create(&ring, 4096, 2);
    CHECK(rc == 0, "create 4096-float ring with 2 consumers");

    /* Write known pattern */
    const uint32_t count = 1024;
    float *wbuf = malloc(count * sizeof(float));
    for (uint32_t i = 0; i < count; i++) wbuf[i] = (float)i * 3.14f;
    spmc_write(&ring, wbuf, count);

    /* Consumer 0: peek (does NOT advance tail) */
    uint32_t peek_count = 0;
    const float *peek_ptr = spmc_peek(&ring, 0, &peek_count);
    CHECK(peek_count == count, "peek returns correct available count");

    /* Verify peek data matches written data */
    int peek_ok = 1;
    for (uint32_t i = 0; i < count && peek_ok; i++) {
        if (peek_ptr[i] != wbuf[i]) peek_ok = 0;
    }
    CHECK(peek_ok, "peek data matches written data");

    /* Verify peek did NOT advance tail */
    uint32_t still_avail = spmc_available_read(&ring, 0);
    CHECK(still_avail == count, "peek does not advance tail");

    /* Consumer 1: read (advances tail) */
    float *rbuf = malloc(count * sizeof(float));
    rc = spmc_read(&ring, 1, rbuf, count);
    CHECK(rc == 0, "consumer 1 read succeeds");

    int read_ok = 1;
    for (uint32_t i = 0; i < count && read_ok; i++) {
        if (rbuf[i] != wbuf[i]) read_ok = 0;
    }
    CHECK(read_ok, "consumer 1 read data matches written data");

    /* Consumer 1 tail advanced, consumer 0 still at original position */
    uint32_t c0_avail = spmc_available_read(&ring, 0);
    uint32_t c1_avail = spmc_available_read(&ring, 1);
    CHECK(c0_avail == count, "consumer 0 still has all data after peek");
    CHECK(c1_avail == 0,     "consumer 1 has no data after read");

    /* Now advance consumer 0 after peek */
    spmc_advance(&ring, 0, count);
    c0_avail = spmc_available_read(&ring, 0);
    CHECK(c0_avail == 0, "consumer 0 has no data after advance");

    free(wbuf);
    free(rbuf);
    spmc_destroy(&ring);
}

/* ---------- 3. wrap boundary stress ---------- */

static void test_spmc_wrap_boundary_stress(void) {
    printf("\n[test_spmc_wrap_boundary_stress]\n");

    SPMCRing ring;
    int rc = spmc_create(&ring, 1024, 1);
    CHECK(rc == 0, "create 1024-float ring with 1 consumer");

    /* The ring size is page-aligned (>= 1024), use actual capacity */
    uint32_t cap = ring.size;

    /* Use chunks that force wrapping within a small ring */
    const uint32_t w1 = 900 < cap ? 900 : cap / 2;
    const uint32_t r1 = 800 < w1  ? 800 : w1 - 100;

    float *wbuf = malloc(w1 * sizeof(float));
    float *rbuf = malloc(w1 * sizeof(float));  /* large enough for any read */

    int corrupt = 0;

    for (int iter = 0; iter < 1000; iter++) {
        /* Write w1 floats with known pattern */
        for (uint32_t i = 0; i < w1; i++)
            wbuf[i] = (float)(iter * 1000 + i);

        rc = spmc_write(&ring, wbuf, w1);
        if (rc != 0) { corrupt++; break; }

        /* Read r1 floats, verify */
        rc = spmc_read(&ring, 0, rbuf, r1);
        if (rc != 0) { corrupt++; break; }
        for (uint32_t i = 0; i < r1; i++) {
            if (rbuf[i] != (float)(iter * 1000 + i)) { corrupt++; break; }
        }

        /* Read remaining (w1 - r1) floats */
        uint32_t remain = w1 - r1;
        rc = spmc_read(&ring, 0, rbuf, remain);
        if (rc != 0) { corrupt++; break; }
        for (uint32_t i = 0; i < remain; i++) {
            if (rbuf[i] != (float)(iter * 1000 + r1 + i)) { corrupt++; break; }
        }
    }

    CHECKF(corrupt == 0, "1000 wrap iterations — %d corruptions", corrupt);

    free(wbuf);
    free(rbuf);
    spmc_destroy(&ring);
}

/* ---------- 4. cancel flag (deactivate/activate) ---------- */

static void test_spmc_cancel_flag(void) {
    printf("\n[test_spmc_cancel_flag]\n");

    SPMCRing ring;
    int rc = spmc_create(&ring, 4096, 2);
    CHECK(rc == 0, "create 4096-float ring with 2 consumers");

    uint32_t cap = ring.size;

    /* Consumer 0 reads normally, consumer 1 does not read.
       Without deactivation, writes would eventually block because
       consumer 1's tail never advances. */

    /* Deactivate consumer 1 */
    spmc_deactivate(&ring, 1);

    /* Fill ring to near capacity — only consumer 0 limits write space */
    float *wbuf = calloc(480, sizeof(float));
    int writes_ok = 1;
    uint32_t total_written = 0;

    while (total_written + 480 <= cap) {
        for (uint32_t i = 0; i < 480; i++)
            wbuf[i] = (float)(total_written + i);
        rc = spmc_write(&ring, wbuf, 480);
        if (rc != 0) break;
        total_written += 480;

        /* Consumer 0 reads to free space */
        float rbuf[480];
        spmc_read(&ring, 0, rbuf, 480);
    }

    CHECK(total_written > cap / 2, "writes succeed with deactivated consumer");

    /* Reactivate consumer 1 — tail should reset to current head */
    spmc_activate(&ring, 1);
    uint32_t c1_avail = spmc_available_read(&ring, 1);
    CHECK(c1_avail == 0, "reactivated consumer starts at current head (no old data)");

    /* Write new data — both consumers should see it */
    for (uint32_t i = 0; i < 480; i++) wbuf[i] = 99.0f;
    spmc_write(&ring, wbuf, 480);

    uint32_t c0_avail = spmc_available_read(&ring, 0);
    c1_avail = spmc_available_read(&ring, 1);
    CHECK(c0_avail >= 480, "consumer 0 sees new data");
    CHECK(c1_avail == 480, "reactivated consumer 1 sees new data");

    free(wbuf);
    spmc_destroy(&ring);
}

/* ---------- 5. crossfade simulated ---------- */

typedef struct {
    float *old_buf;
    float *new_buf;
    float *out_buf;
    int    cf_len;
    pthread_mutex_t *mtx;
    int    thread_id;
} CrossfadeArgs;

static void *crossfade_worker(void *arg) {
    CrossfadeArgs *a = (CrossfadeArgs *)arg;

    pthread_mutex_lock(a->mtx);
    for (int i = 0; i < a->cf_len; i++) {
        float alpha = (float)i / (float)a->cf_len;
        a->out_buf[i] = a->old_buf[i] * (1.0f - alpha) + a->new_buf[i] * alpha;
    }
    pthread_mutex_unlock(a->mtx);

    return NULL;
}

static void test_crossfade_simulated(void) {
    printf("\n[test_crossfade_simulated]\n");

    const int cf_len = 480;
    const int n_threads = 4;

    float *old_bufs[4], *new_bufs[4], *out_bufs[4];
    pthread_mutex_t mutexes[4];
    CrossfadeArgs args[4];
    pthread_t threads[4];

    for (int t = 0; t < n_threads; t++) {
        old_bufs[t] = malloc(cf_len * sizeof(float));
        new_bufs[t] = malloc(cf_len * sizeof(float));
        out_bufs[t] = calloc(cf_len, sizeof(float));
        pthread_mutex_init(&mutexes[t], NULL);

        /* Each thread has unique data */
        for (int i = 0; i < cf_len; i++) {
            old_bufs[t][i] = (float)(t * 1000 + i);
            new_bufs[t][i] = (float)(t * 1000 + i + 10000);
        }

        args[t] = (CrossfadeArgs){
            .old_buf = old_bufs[t], .new_buf = new_bufs[t],
            .out_buf = out_bufs[t], .cf_len = cf_len,
            .mtx = &mutexes[t], .thread_id = t
        };
    }

    /* Launch all threads */
    for (int t = 0; t < n_threads; t++)
        pthread_create(&threads[t], NULL, crossfade_worker, &args[t]);
    for (int t = 0; t < n_threads; t++)
        pthread_join(threads[t], NULL);

    /* Verify each thread's output */
    int all_ok = 1;
    for (int t = 0; t < n_threads; t++) {
        for (int i = 0; i < cf_len; i++) {
            float alpha = (float)i / (float)cf_len;
            float expected = old_bufs[t][i] * (1.0f - alpha)
                           + new_bufs[t][i] * alpha;
            if (fabsf(out_bufs[t][i] - expected) > 1e-3f) {
                all_ok = 0;
                break;
            }
        }
        free(old_bufs[t]); free(new_bufs[t]); free(out_bufs[t]);
        pthread_mutex_destroy(&mutexes[t]);
    }

    CHECK(all_ok, "4 threads crossfade concurrently — results correct");
}

/* ---------- 6. barge-in timedwait ---------- */

static pthread_mutex_t barge_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  barge_cond = PTHREAD_COND_INITIALIZER;
static _Atomic int     barge_signaled = 0;

static void *barge_signaler(void *arg) {
    (void)arg;
    /* Wait 20ms then signal */
    struct timespec sl = { .tv_sec = 0, .tv_nsec = 20 * 1000000L };
    nanosleep(&sl, NULL);

    pthread_mutex_lock(&barge_mtx);
    atomic_store(&barge_signaled, 1);
    pthread_cond_signal(&barge_cond);
    pthread_mutex_unlock(&barge_mtx);
    return NULL;
}

static void test_barge_in_timedwait(void) {
    printf("\n[test_barge_in_timedwait]\n");

    /* Test 1: timeout case (no signal) */
    {
        struct timespec start, end, abstime;
        clock_gettime(CLOCK_REALTIME, &start);

        /* 50ms timeout */
        abstime = start;
        abstime.tv_nsec += 50 * 1000000L;
        if (abstime.tv_nsec >= 1000000000L) {
            abstime.tv_sec += 1;
            abstime.tv_nsec -= 1000000000L;
        }

        pthread_mutex_lock(&barge_mtx);
        int rc = pthread_cond_timedwait(&barge_cond, &barge_mtx, &abstime);
        pthread_mutex_unlock(&barge_mtx);

        clock_gettime(CLOCK_REALTIME, &end);
        double ms = elapsed_ms(&start, &end);

        CHECK(rc != 0, "timedwait returns non-zero on timeout");
        CHECKF(ms >= 35.0 && ms <= 80.0,
               "timeout elapsed %.1fms (expect 50ms ± tolerance)", ms);
    }

    /* Test 2: signal before timeout */
    {
        atomic_store(&barge_signaled, 0);

        pthread_t sig_thread;
        pthread_create(&sig_thread, NULL, barge_signaler, NULL);

        struct timespec start, end, abstime;
        clock_gettime(CLOCK_REALTIME, &start);

        /* 50ms timeout, but signal comes at ~20ms */
        abstime = start;
        abstime.tv_nsec += 50 * 1000000L;
        if (abstime.tv_nsec >= 1000000000L) {
            abstime.tv_sec += 1;
            abstime.tv_nsec -= 1000000000L;
        }

        pthread_mutex_lock(&barge_mtx);
        while (!atomic_load(&barge_signaled)) {
            int rc = pthread_cond_timedwait(&barge_cond, &barge_mtx, &abstime);
            if (rc != 0) break;  /* timed out */
        }
        pthread_mutex_unlock(&barge_mtx);

        clock_gettime(CLOCK_REALTIME, &end);
        double ms = elapsed_ms(&start, &end);

        pthread_join(sig_thread, NULL);

        CHECK(atomic_load(&barge_signaled) == 1, "signal received before timeout");
        CHECKF(ms < 45.0, "signal woke us at %.1fms (before 50ms timeout)", ms);
    }
}

/* ---------- 7. thread-local scratch ---------- */

typedef struct {
    int thread_id;
    int ok;
} ScratchArgs;

static void *scratch_worker(void *arg) {
    ScratchArgs *a = (ScratchArgs *)arg;
    const int n_frames = 10;
    const int frame_size = 480;
    const int total = n_frames * frame_size;  /* 4800 floats */

    /* Thread-local scratch buffer */
    float *scratch = malloc(total * sizeof(float));
    if (!scratch) { a->ok = 0; return NULL; }

    /* Fill with thread-specific pattern */
    float pattern = (float)(a->thread_id * 1000 + 42);
    for (int i = 0; i < total; i++)
        scratch[i] = pattern + (float)i;

    /* Simulate work — do some computation */
    for (int f = 0; f < n_frames; f++) {
        float *frame = scratch + f * frame_size;
        for (int i = 0; i < frame_size; i++)
            frame[i] *= 1.001f;
    }

    /* Verify pattern is intact (no cross-thread corruption) */
    a->ok = 1;
    for (int i = 0; i < total; i++) {
        float expected = (pattern + (float)i) * 1.001f;
        if (fabsf(scratch[i] - expected) > 0.1f) {
            a->ok = 0;
            break;
        }
    }

    free(scratch);
    return NULL;
}

static void test_thread_local_scratch(void) {
    printf("\n[test_thread_local_scratch]\n");

    const int n_threads = 4;
    pthread_t threads[4];
    ScratchArgs args[4];

    for (int t = 0; t < n_threads; t++) {
        args[t].thread_id = t;
        args[t].ok = 0;
        pthread_create(&threads[t], NULL, scratch_worker, &args[t]);
    }

    for (int t = 0; t < n_threads; t++)
        pthread_join(threads[t], NULL);

    for (int t = 0; t < n_threads; t++) {
        CHECKF(args[t].ok, "thread %d scratch buffer — no cross-thread corruption", t);
    }
}

/* ---------- 8. vm_ring mirrored peek ---------- */

static void test_vm_ring_mirrored_peek(void) {
    printf("\n[test_vm_ring_mirrored_peek]\n");

    VMRingBuffer rb;
    int rc = vm_ring_create(&rb, 1024);
    CHECK(rc == 0, "create 1024-float vm_ring");

    uint32_t cap = rb.size;

    /* Fill most of the ring to push head near the wrap point */
    uint32_t fill = cap - 128;  /* leave 128 free */
    float *filler = malloc(fill * sizeof(float));
    for (uint32_t i = 0; i < fill; i++) filler[i] = 0.0f;
    vm_ring_write(&rb, filler, fill);

    /* Read all but 64 samples — tail is near end, head near end */
    float *trash = malloc(fill * sizeof(float));
    vm_ring_read(&rb, trash, fill - 64);

    /* Now write 256 floats — this wraps around the boundary */
    float wbuf[256];
    for (int i = 0; i < 256; i++) wbuf[i] = (float)(i + 1);
    rc = vm_ring_write(&rb, wbuf, 256);
    CHECK(rc == 0, "write across wrap boundary");

    /* Peek should return contiguous pointer even across wrap */
    uint32_t peek_count = 0;
    const float *peek = vm_ring_peek(&rb, &peek_count);
    CHECK(peek_count == 64 + 256, "peek reports correct available count");

    /* The last 256 samples should match our pattern */
    int data_ok = 1;
    for (int i = 0; i < 256; i++) {
        if (peek[64 + i] != (float)(i + 1)) {
            data_ok = 0;
            break;
        }
    }
    CHECK(data_ok, "peek returns contiguous data across wrap boundary");

    free(filler);
    free(trash);
    vm_ring_destroy(&rb);
}

/* ---------- 9. multi-consumer ---------- */

static void test_spmc_multi_consumer(void) {
    printf("\n[test_spmc_multi_consumer]\n");

    SPMCRing ring;
    int rc = spmc_create(&ring, 8192, 4);
    CHECK(rc == 0, "create 8192-float ring with 4 consumers");

    /* Write 4800 floats with known pattern */
    const uint32_t total = 4800;
    float *wbuf = malloc(total * sizeof(float));
    for (uint32_t i = 0; i < total; i++) wbuf[i] = (float)i;
    rc = spmc_write(&ring, wbuf, total);
    CHECK(rc == 0, "write 4800 floats");

    /* Consumer 0: read all at once */
    float *c0 = malloc(total * sizeof(float));
    rc = spmc_read(&ring, 0, c0, total);
    CHECK(rc == 0, "consumer 0: read all at once");

    /* Consumer 1: read in 480-sample chunks */
    float *c1 = malloc(total * sizeof(float));
    int c1_ok = 1;
    for (uint32_t off = 0; off < total; off += 480) {
        uint32_t chunk = (off + 480 <= total) ? 480 : (total - off);
        rc = spmc_read(&ring, 1, c1 + off, chunk);
        if (rc != 0) { c1_ok = 0; break; }
    }
    CHECK(c1_ok, "consumer 1: read in 480-sample chunks");

    /* Consumer 2: read in 960-sample chunks */
    float *c2 = malloc(total * sizeof(float));
    int c2_ok = 1;
    for (uint32_t off = 0; off < total; off += 960) {
        uint32_t chunk = (off + 960 <= total) ? 960 : (total - off);
        rc = spmc_read(&ring, 2, c2 + off, chunk);
        if (rc != 0) { c2_ok = 0; break; }
    }
    CHECK(c2_ok, "consumer 2: read in 960-sample chunks");

    /* Consumer 3: peek then advance */
    float *c3 = malloc(total * sizeof(float));
    uint32_t c3_read = 0;
    int c3_ok = 1;
    while (c3_read < total) {
        uint32_t avail = 0;
        const float *peek = spmc_peek(&ring, 3, &avail);
        if (avail == 0) { c3_ok = 0; break; }
        uint32_t to_copy = (avail > 480) ? 480 : avail;
        memcpy(c3 + c3_read, peek, to_copy * sizeof(float));
        spmc_advance(&ring, 3, to_copy);
        c3_read += to_copy;
    }
    CHECK(c3_ok, "consumer 3: peek + advance");

    /* Verify all consumers got identical data */
    int all_match = 1;
    for (uint32_t i = 0; i < total; i++) {
        if (c0[i] != wbuf[i] || c1[i] != wbuf[i] ||
            c2[i] != wbuf[i] || c3[i] != wbuf[i]) {
            all_match = 0;
            break;
        }
    }
    CHECK(all_match, "all 4 consumers got identical data");

    free(wbuf); free(c0); free(c1); free(c2); free(c3);
    spmc_destroy(&ring);
}

/* ---------- 10. create/destroy edge cases ---------- */

static void test_spmc_create_destroy_edge_cases(void) {
    printf("\n[test_spmc_create_destroy_edge_cases]\n");

    int rc;

    /* 0 consumers → -1 */
    SPMCRing ring;
    rc = spmc_create(&ring, 4096, 0);
    CHECK(rc == -1, "create with 0 consumers returns -1");

    /* 5 consumers (> SPMC_MAX_CONSUMERS) → -1 */
    rc = spmc_create(&ring, 4096, 5);
    CHECK(rc == -1, "create with 5 consumers (> max) returns -1");

    /* NULL ring → -1 */
    rc = spmc_create(NULL, 4096, 1);
    CHECK(rc == -1, "create with NULL ring returns -1");

    /* destroy(NULL) → no crash (no-op) */
    spmc_destroy(NULL);
    CHECK(1, "destroy(NULL) is a no-op (no crash)");

    /* Exactly SPMC_MAX_CONSUMERS (4) should succeed */
    rc = spmc_create(&ring, 4096, SPMC_MAX_CONSUMERS);
    CHECK(rc == 0, "create with exactly SPMC_MAX_CONSUMERS succeeds");
    spmc_destroy(&ring);
}

/* ---------- 11. multi-reader race condition stress ---------- */

typedef struct {
    SPMCRing *ring;
    int consumer_id;
    uint32_t total_read;
    int corrupt;
} ReaderArgs;

static void *race_reader(void *arg) {
    ReaderArgs *a = (ReaderArgs *)arg;
    float rbuf[480];
    a->total_read = 0;
    a->corrupt = 0;

    while (a->total_read < 48000) {
        uint32_t avail = spmc_available_read(a->ring, a->consumer_id);
        if (avail == 0) {
            /* Busy-wait briefly */
            struct timespec sl = { .tv_sec = 0, .tv_nsec = 100000L };
            nanosleep(&sl, NULL);
            continue;
        }
        uint32_t to_read = avail < 480 ? avail : 480;
        int rc = spmc_read(a->ring, a->consumer_id, rbuf, to_read);
        if (rc != 0) { a->corrupt++; break; }

        /* Verify data is sequential based on position */
        for (uint32_t i = 0; i < to_read; i++) {
            float expected = (float)(a->total_read + i);
            if (rbuf[i] != expected) { a->corrupt++; break; }
        }
        a->total_read += to_read;
    }
    return NULL;
}

static void test_spmc_multi_reader_race(void) {
    printf("\n[test_spmc_multi_reader_race]\n");

    SPMCRing ring;
    int rc = spmc_create(&ring, 8192, 3);
    CHECK(rc == 0, "create 8192-float ring with 3 consumers");

    /* Launch 3 reader threads */
    pthread_t readers[3];
    ReaderArgs rargs[3];
    for (int i = 0; i < 3; i++) {
        rargs[i].ring = &ring;
        rargs[i].consumer_id = i;
        pthread_create(&readers[i], NULL, race_reader, &rargs[i]);
    }

    /* Producer writes 48000 floats in 480-float chunks */
    float wbuf[480];
    uint32_t total_written = 0;
    while (total_written < 48000) {
        for (uint32_t i = 0; i < 480; i++)
            wbuf[i] = (float)(total_written + i);

        while (spmc_available_write(&ring) < 480) {
            struct timespec sl = { .tv_sec = 0, .tv_nsec = 100000L };
            nanosleep(&sl, NULL);
        }
        spmc_write(&ring, wbuf, 480);
        total_written += 480;
    }

    for (int i = 0; i < 3; i++)
        pthread_join(readers[i], NULL);

    for (int i = 0; i < 3; i++) {
        CHECKF(rargs[i].corrupt == 0,
               "consumer %d: no corruption in race (%u read)", i, rargs[i].total_read);
        CHECKF(rargs[i].total_read == 48000,
               "consumer %d: read all 48000 floats (%u)", i, rargs[i].total_read);
    }

    spmc_destroy(&ring);
}

/* ---------- 12. full buffer behavior ---------- */

static void test_spmc_full_buffer(void) {
    printf("\n[test_spmc_full_buffer]\n");

    SPMCRing ring;
    int rc = spmc_create(&ring, 1024, 1);
    CHECK(rc == 0, "create 1024-float ring with 1 consumer");

    uint32_t cap = ring.size;

    /* Fill ring to exactly capacity (write cap floats without reading) */
    float *wbuf = calloc(cap, sizeof(float));
    for (uint32_t i = 0; i < cap; i++) wbuf[i] = (float)i;

    /* Write cap-1 first (should succeed — need 1 slot free in typical ring) */
    rc = spmc_write(&ring, wbuf, cap - 1);
    CHECK(rc == 0, "write cap-1 floats succeeds");

    /* Check available write space */
    uint32_t avail_w = spmc_available_write(&ring);
    CHECKF(avail_w == 1, "1 slot remaining after writing cap-1 (%u available)", avail_w);

    /* Write 1 more should succeed */
    float one = 99.0f;
    rc = spmc_write(&ring, &one, 1);
    CHECK(rc == 0, "write 1 more to fill completely");

    /* Now ring is full — further writes should fail */
    rc = spmc_write(&ring, &one, 1);
    CHECK(rc == -1, "write to full ring returns -1");

    /* Read everything back to verify integrity */
    float *rbuf = malloc(cap * sizeof(float));
    rc = spmc_read(&ring, 0, rbuf, cap);
    CHECK(rc == 0, "read full ring succeeds");

    int data_ok = 1;
    for (uint32_t i = 0; i < cap - 1; i++) {
        if (rbuf[i] != (float)i) { data_ok = 0; break; }
    }
    if (rbuf[cap - 1] != 99.0f) data_ok = 0;
    CHECK(data_ok, "data integrity after filling and draining");

    /* After drain, writes should succeed again */
    rc = spmc_write(&ring, wbuf, 480);
    CHECK(rc == 0, "write succeeds after draining full ring");

    free(wbuf);
    free(rbuf);
    spmc_destroy(&ring);
}

/* ---------- 13. zero-size write/read ---------- */

static void test_spmc_zero_size(void) {
    printf("\n[test_spmc_zero_size]\n");

    SPMCRing ring;
    int rc = spmc_create(&ring, 4096, 1);
    CHECK(rc == 0, "create ring for zero-size tests");

    /* Write 0 floats — should succeed (no-op) */
    float dummy = 1.0f;
    rc = spmc_write(&ring, &dummy, 0);
    CHECK(rc == 0, "write 0 floats succeeds");

    /* Available read should still be 0 */
    uint32_t avail = spmc_available_read(&ring, 0);
    CHECK(avail == 0, "no data available after zero-size write");

    /* Write some real data, then read 0 */
    float wbuf[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    spmc_write(&ring, wbuf, 10);

    float rbuf[10] = {0};
    rc = spmc_read(&ring, 0, rbuf, 0);
    CHECK(rc == 0, "read 0 floats succeeds");

    avail = spmc_available_read(&ring, 0);
    CHECK(avail == 10, "data still available after zero-size read");

    /* Advance 0 — should be no-op */
    spmc_advance(&ring, 0, 0);
    avail = spmc_available_read(&ring, 0);
    CHECK(avail == 10, "data still available after zero-size advance");

    spmc_destroy(&ring);
}

/* ---------- 14. wraparound exact boundary alignment ---------- */

static void test_spmc_exact_boundary_alignment(void) {
    printf("\n[test_spmc_exact_boundary_alignment]\n");

    SPMCRing ring;
    int rc = spmc_create(&ring, 1024, 1);
    CHECK(rc == 0, "create ring for boundary alignment");

    uint32_t cap = ring.size;

    /* Write and read exactly (cap-1) repeatedly to force head/tail
       to land exactly on the boundary */
    float *wbuf = malloc(cap * sizeof(float));
    float *rbuf = malloc(cap * sizeof(float));
    int corrupt = 0;

    for (int iter = 0; iter < 50; iter++) {
        /* Fill with pattern for this iteration */
        uint32_t chunk = cap - 1;
        for (uint32_t i = 0; i < chunk; i++)
            wbuf[i] = (float)(iter * 10000 + i);

        rc = spmc_write(&ring, wbuf, chunk);
        if (rc != 0) { corrupt++; break; }

        rc = spmc_read(&ring, 0, rbuf, chunk);
        if (rc != 0) { corrupt++; break; }

        for (uint32_t i = 0; i < chunk; i++) {
            if (rbuf[i] != (float)(iter * 10000 + i)) { corrupt++; break; }
        }
    }

    CHECKF(corrupt == 0, "50 boundary-aligned cycles — %d corruptions", corrupt);

    free(wbuf);
    free(rbuf);
    spmc_destroy(&ring);
}

/* ---------- 15. rapid deactivate/activate cycling ---------- */

static void test_spmc_rapid_deactivate_activate(void) {
    printf("\n[test_spmc_rapid_deactivate_activate]\n");

    SPMCRing ring;
    int rc = spmc_create(&ring, 4096, 2);
    CHECK(rc == 0, "create ring with 2 consumers");

    float wbuf[480], rbuf[480];
    int ok = 1;

    for (int cycle = 0; cycle < 100; cycle++) {
        /* Deactivate consumer 1 */
        spmc_deactivate(&ring, 1);

        /* Write data while consumer 1 is inactive */
        for (uint32_t i = 0; i < 480; i++)
            wbuf[i] = (float)(cycle * 480 + i);
        rc = spmc_write(&ring, wbuf, 480);
        if (rc != 0) { ok = 0; break; }

        /* Consumer 0 reads normally */
        rc = spmc_read(&ring, 0, rbuf, 480);
        if (rc != 0) { ok = 0; break; }
        for (uint32_t i = 0; i < 480; i++) {
            if (rbuf[i] != (float)(cycle * 480 + i)) { ok = 0; break; }
        }

        /* Reactivate consumer 1 — tail resets to head */
        spmc_activate(&ring, 1);

        /* Consumer 1 should have 0 available (just reactivated) */
        uint32_t c1_avail = spmc_available_read(&ring, 1);
        if (c1_avail != 0) { ok = 0; break; }
    }

    CHECK(ok, "100 rapid deactivate/activate cycles — no errors");
    spmc_destroy(&ring);
}

/* ---------- 16. vm_ring full capacity overflow ---------- */

static void test_vm_ring_overflow(void) {
    printf("\n[test_vm_ring_overflow]\n");

    VMRingBuffer rb;
    int rc = vm_ring_create(&rb, 1024);
    CHECK(rc == 0, "create 1024-float vm_ring");

    uint32_t cap = rb.size;

    /* Fill to capacity */
    float *wbuf = calloc(cap, sizeof(float));
    for (uint32_t i = 0; i < cap; i++) wbuf[i] = (float)i;
    rc = vm_ring_write(&rb, wbuf, cap);
    CHECK(rc == 0, "write exactly cap floats succeeds");

    /* Overflow attempt should fail */
    float one = 42.0f;
    rc = vm_ring_write(&rb, &one, 1);
    CHECK(rc == -1, "write to full vm_ring returns -1");

    /* Available counts */
    uint32_t r_avail = vm_ring_available_read(&rb);
    uint32_t w_avail = vm_ring_available_write(&rb);
    CHECK(r_avail == cap, "full ring: available_read == cap");
    CHECK(w_avail == 0, "full ring: available_write == 0");

    /* Read half, verify write works again */
    float *rbuf = malloc((cap / 2) * sizeof(float));
    rc = vm_ring_read(&rb, rbuf, cap / 2);
    CHECK(rc == 0, "read half of full ring");

    w_avail = vm_ring_available_write(&rb);
    CHECK(w_avail == cap / 2, "write space restored after partial read");

    rc = vm_ring_write(&rb, wbuf, cap / 2);
    CHECK(rc == 0, "write after partial drain succeeds");

    free(wbuf);
    free(rbuf);
    vm_ring_destroy(&rb);
}

/* ---------- 17. SPMC read from invalid consumer ---------- */

static void test_spmc_invalid_consumer_id(void) {
    printf("\n[test_spmc_invalid_consumer_id]\n");

    SPMCRing ring;
    int rc = spmc_create(&ring, 4096, 2);
    CHECK(rc == 0, "create ring with 2 consumers");

    float wbuf[480] = {0};
    spmc_write(&ring, wbuf, 480);

    /* Read from negative consumer_id */
    float rbuf[480];
    rc = spmc_read(&ring, -1, rbuf, 480);
    CHECK(rc == -1, "read from consumer -1 returns -1");

    /* Read from consumer_id >= n_consumers */
    rc = spmc_read(&ring, 2, rbuf, 480);
    CHECK(rc == -1, "read from consumer 2 (only 0,1 valid) returns -1");

    rc = spmc_read(&ring, 99, rbuf, 480);
    CHECK(rc == -1, "read from consumer 99 returns -1");

    /* Valid consumer should still work */
    rc = spmc_read(&ring, 0, rbuf, 480);
    CHECK(rc == 0, "read from valid consumer 0 succeeds");

    spmc_destroy(&ring);
}

/* ---------- main ---------- */

int main(void) {
    printf("=== Pipeline Threading Tests ===\n");

    test_spmc_rapid_write_read();
    test_spmc_concurrent_peek();
    test_spmc_wrap_boundary_stress();
    test_spmc_cancel_flag();
    test_crossfade_simulated();
    test_barge_in_timedwait();
    test_thread_local_scratch();
    test_vm_ring_mirrored_peek();
    test_spmc_multi_consumer();
    test_spmc_create_destroy_edge_cases();
    test_spmc_multi_reader_race();
    test_spmc_full_buffer();
    test_spmc_zero_size();
    test_spmc_exact_boundary_alignment();
    test_spmc_rapid_deactivate_activate();
    test_vm_ring_overflow();
    test_spmc_invalid_consumer_id();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
