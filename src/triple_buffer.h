/**
 * triple_buffer.h — Lock-free triple-buffer for GPU→CPU→CoreAudio pipeline.
 *
 * Three phase-rotated buffers allow true parallelism:
 *   Buffer A: GPU/Rust TTS writes decoded audio into it
 *   Buffer B: CPU processes (SSML prosody, LUFS normalization, breath insertion)
 *   Buffer C: CoreAudio callback reads for playback
 *
 * After each "frame" (one TTS decode step = 80ms @ 12.5 Hz), the three
 * buffers rotate atomically. This eliminates all producer-consumer blocking.
 *
 * The rotation uses a single atomic integer encoding the current phase,
 * making the buffer completely lock-free.
 */

#ifndef TRIPLE_BUFFER_H
#define TRIPLE_BUFFER_H

#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TRIPLE_BUF_SLOTS 3

typedef struct {
    float *buffers[TRIPLE_BUF_SLOTS];
    uint32_t buf_size;       /* samples per buffer */

    /*
     * Atomic state encodes which slot each role uses:
     *   bits [1:0] = writer slot index
     *   bits [3:2] = processor slot index
     *   bits [5:4] = reader slot index
     *   bit  [6]   = writer has new data for processor
     *   bit  [7]   = processor has new data for reader
     */
    _Atomic uint32_t state;

    /* Per-slot sample counts (how many valid samples in each) */
    _Atomic uint32_t counts[TRIPLE_BUF_SLOTS];
} TripleBuffer;

/**
 * Create a triple buffer.
 * @param samples_per_buf  Size of each buffer in float samples
 * @return 0 on success, -1 on failure
 */
static inline int triple_buf_create(TripleBuffer *tb, uint32_t samples_per_buf)
{
    if (!tb || samples_per_buf == 0) return -1;
    memset(tb, 0, sizeof(TripleBuffer));
    tb->buf_size = samples_per_buf;
    for (int i = 0; i < TRIPLE_BUF_SLOTS; i++) {
        tb->buffers[i] = (float *)calloc(samples_per_buf, sizeof(float));
        if (!tb->buffers[i]) {
            for (int j = 0; j < i; j++) free(tb->buffers[j]);
            return -1;
        }
    }
    /* Initial state: writer=0, processor=1, reader=2, no new data */
    atomic_store(&tb->state, 0u | (1u << 2) | (2u << 4));
    for (int i = 0; i < TRIPLE_BUF_SLOTS; i++)
        atomic_store(&tb->counts[i], 0);
    return 0;
}

static inline void triple_buf_destroy(TripleBuffer *tb)
{
    if (!tb) return;
    for (int i = 0; i < TRIPLE_BUF_SLOTS; i++) {
        free(tb->buffers[i]);
        tb->buffers[i] = NULL;
    }
}

/* Get the writer's current buffer pointer */
static inline float *triple_buf_write_ptr(TripleBuffer *tb)
{
    uint32_t s = atomic_load_explicit(&tb->state, memory_order_relaxed);
    return tb->buffers[s & 3u];
}

/**
 * Writer signals done writing. Swaps writer and processor slots atomically.
 * @param n_samples  Number of valid samples written
 */
static inline void triple_buf_write_done(TripleBuffer *tb, uint32_t n_samples)
{
    uint32_t s = atomic_load_explicit(&tb->state, memory_order_acquire);
    uint32_t w = s & 3u;
    uint32_t p = (s >> 2) & 3u;
    uint32_t r = (s >> 4) & 3u;

    atomic_store_explicit(&tb->counts[w], n_samples, memory_order_relaxed);

    /* Swap writer and processor, set "writer has data" flag */
    uint32_t ns = p | (w << 2) | (r << 4) | (1u << 6);
    atomic_store_explicit(&tb->state, ns, memory_order_release);
}

/**
 * Processor acquires its buffer (if writer has delivered new data).
 * @param out_count  Receives the number of valid samples
 * @return Buffer pointer, or NULL if no new data
 */
static inline float *triple_buf_process_acquire(TripleBuffer *tb, uint32_t *out_count)
{
    uint32_t s = atomic_load_explicit(&tb->state, memory_order_acquire);
    if (!(s & (1u << 6))) return NULL; /* no new data */

    uint32_t p = (s >> 2) & 3u;
    if (out_count)
        *out_count = atomic_load_explicit(&tb->counts[p], memory_order_relaxed);

    /* Clear "writer has data" flag */
    atomic_store_explicit(&tb->state, s & ~(1u << 6), memory_order_release);
    return tb->buffers[p];
}

/**
 * Processor signals done. Swaps processor and reader slots atomically.
 * @param n_samples  Number of valid processed samples
 */
static inline void triple_buf_process_done(TripleBuffer *tb, uint32_t n_samples)
{
    uint32_t s = atomic_load_explicit(&tb->state, memory_order_acquire);
    uint32_t w = s & 3u;
    uint32_t p = (s >> 2) & 3u;
    uint32_t r = (s >> 4) & 3u;

    atomic_store_explicit(&tb->counts[p], n_samples, memory_order_relaxed);

    /* Swap processor and reader, set "processor has data" flag */
    uint32_t ns = w | (r << 2) | (p << 4) | (s & (1u << 6)) | (1u << 7);
    atomic_store_explicit(&tb->state, ns, memory_order_release);
}

/**
 * Reader acquires its buffer (if processor has delivered new data).
 * Called from CoreAudio render callback — must be real-time safe.
 * @param out_count  Receives the number of valid samples
 * @return Buffer pointer, or NULL if no new data
 */
static inline const float *triple_buf_read_acquire(TripleBuffer *tb, uint32_t *out_count)
{
    uint32_t s = atomic_load_explicit(&tb->state, memory_order_acquire);
    if (!(s & (1u << 7))) return NULL;

    uint32_t r = (s >> 4) & 3u;
    if (out_count)
        *out_count = atomic_load_explicit(&tb->counts[r], memory_order_relaxed);

    /* Clear "processor has data" flag */
    atomic_store_explicit(&tb->state, s & ~(1u << 7), memory_order_release);
    return tb->buffers[r];
}

#ifdef __cplusplus
}
#endif

#endif /* TRIPLE_BUFFER_H */
