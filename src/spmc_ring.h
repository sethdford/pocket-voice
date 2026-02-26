/**
 * spmc_ring.h — Single-Producer, Multi-Consumer lock-free ring buffer.
 *
 * One producer (post-processor) feeds PCM data to multiple independent
 * consumers (CoreAudio speaker, Opus encoder, WebSocket streamer) without
 * any locking or data duplication.
 *
 * Each consumer maintains its own read cursor. The producer can only
 * reclaim space once ALL consumers have advanced past it.
 *
 * Uses the VM-mirrored ring buffer (vm_ring.h) as the backing store
 * for zero-copy wraparound reads (mach_vm_remap double-maps same
 * physical pages, so reads near the wrap boundary are contiguous).
 */

#ifndef SPMC_RING_H
#define SPMC_RING_H

#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include "vm_ring.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPMC_MAX_CONSUMERS 4

typedef struct {
    float *buffer;
    uint32_t size;           /* power-of-two capacity */
    uint32_t mask;

    _Alignas(64) _Atomic uint64_t head;  /* single producer write pos */

    /* Each consumer has its own read position, cache-line separated */
    struct {
        _Alignas(64) _Atomic uint64_t tail;
        _Atomic int active;
    } consumers[SPMC_MAX_CONSUMERS];

    int n_consumers;
    VMRingBuffer vm_backing;  /* VM-mirrored backing store */
} SPMCRing;

/**
 * Create a SPMC ring buffer backed by mach_vm_remap mirrored pages.
 * @param n_floats      Desired capacity (rounded up to power-of-two, page-aligned)
 * @param n_consumers   Number of consumers (max SPMC_MAX_CONSUMERS)
 * @return 0 on success, -1 on failure
 */
static inline int spmc_create(SPMCRing *ring, uint32_t n_floats, int n_consumers)
{
    if (!ring || n_consumers <= 0 || n_consumers > SPMC_MAX_CONSUMERS) return -1;

    memset(ring, 0, sizeof(SPMCRing));

    /* vm_ring_create handles power-of-two rounding and page alignment */
    if (vm_ring_create(&ring->vm_backing, n_floats) != 0) return -1;

    ring->buffer = ring->vm_backing.buffer;
    ring->size = ring->vm_backing.size;
    ring->mask = ring->vm_backing.mask;
    ring->n_consumers = n_consumers;
    atomic_store(&ring->head, 0);

    for (int i = 0; i < n_consumers; i++) {
        atomic_store(&ring->consumers[i].tail, 0);
        atomic_store(&ring->consumers[i].active, 1);
    }

    return 0;
}

static inline void spmc_destroy(SPMCRing *ring)
{
    if (!ring) return;
    vm_ring_destroy(&ring->vm_backing);
    memset(ring, 0, sizeof(SPMCRing));
}

/** Find the minimum tail across all active consumers (slowest reader). */
static inline uint64_t spmc_min_tail(const SPMCRing *ring)
{
    uint64_t min_t = UINT64_MAX;
    for (int i = 0; i < ring->n_consumers; i++) {
        if (!atomic_load_explicit(&ring->consumers[i].active, memory_order_relaxed))
            continue;
        uint64_t t = atomic_load_explicit(&ring->consumers[i].tail, memory_order_acquire);
        if (t < min_t) min_t = t;
    }
    return min_t;
}

/** Available write space (bounded by slowest consumer). */
static inline uint32_t spmc_available_write(const SPMCRing *ring)
{
    uint64_t h = atomic_load_explicit(&ring->head, memory_order_relaxed);
    uint64_t mt = spmc_min_tail(ring);
    if (h < mt) return 0; /* Guard against underflow under race conditions */
    return ring->size - (uint32_t)(h - mt);
}

/**
 * Producer: write data into ring.
 * With VM mirroring, a single memcpy always works — the hardware page table
 * makes the second half alias the first, so wraparound is transparent.
 * @return 0 on success, -1 if would overrun slowest consumer
 */
static inline int spmc_write(SPMCRing *ring, const float *data, uint32_t count)
{
    if (spmc_available_write(ring) < count) return -1;

    uint64_t h = atomic_load_explicit(&ring->head, memory_order_relaxed);
    uint32_t offset = (uint32_t)(h & ring->mask);

    /* Single contiguous write — VM mirror handles wraparound transparently */
    memcpy(ring->buffer + offset, data, (size_t)count * sizeof(float));

    atomic_store_explicit(&ring->head, h + count, memory_order_release);
    return 0;
}

/** Available data for a specific consumer. */
static inline uint32_t spmc_available_read(const SPMCRing *ring, int consumer_id)
{
    uint64_t h = atomic_load_explicit(&ring->head, memory_order_acquire);
    uint64_t t = atomic_load_explicit(&ring->consumers[consumer_id].tail,
                                       memory_order_relaxed);
    return (uint32_t)(h - t);
}

/**
 * Consumer: read data from ring.
 * Each consumer reads independently at its own pace.
 * With VM mirroring, a single memcpy always works.
 * @return 0 on success, -1 if insufficient data
 */
static inline int spmc_read(SPMCRing *ring, int consumer_id, float *out, uint32_t count)
{
    if (consumer_id < 0 || consumer_id >= ring->n_consumers) return -1;
    if (spmc_available_read(ring, consumer_id) < count) return -1;

    uint64_t t = atomic_load_explicit(&ring->consumers[consumer_id].tail,
                                       memory_order_relaxed);
    uint32_t offset = (uint32_t)(t & ring->mask);

    /* Single contiguous read — VM mirror handles wraparound transparently */
    memcpy(out, ring->buffer + offset, (size_t)count * sizeof(float));

    atomic_store_explicit(&ring->consumers[consumer_id].tail, t + count,
                           memory_order_release);
    return 0;
}

/**
 * Consumer: get a direct peek pointer (may wrap — use with VM-mirrored backing).
 * Does NOT advance the tail.
 */
static inline const float *spmc_peek(const SPMCRing *ring, int consumer_id,
                                      uint32_t *out_count)
{
    uint32_t avail = spmc_available_read(ring, consumer_id);
    if (out_count) *out_count = avail;
    uint64_t t = atomic_load_explicit(&ring->consumers[consumer_id].tail,
                                       memory_order_relaxed);
    return ring->buffer + (uint32_t)(t & ring->mask);
}

/** Consumer: advance read cursor after peek. */
static inline void spmc_advance(SPMCRing *ring, int consumer_id, uint32_t count)
{
    uint64_t t = atomic_load_explicit(&ring->consumers[consumer_id].tail,
                                       memory_order_relaxed);
    atomic_store_explicit(&ring->consumers[consumer_id].tail, t + count,
                           memory_order_release);
}

/** Deactivate a consumer (won't block producer). */
static inline void spmc_deactivate(SPMCRing *ring, int consumer_id)
{
    if (consumer_id >= 0 && consumer_id < SPMC_MAX_CONSUMERS)
        atomic_store_explicit(&ring->consumers[consumer_id].active, 0,
                               memory_order_release);
}

/** Reactivate a consumer, resetting its tail to current head. */
static inline void spmc_activate(SPMCRing *ring, int consumer_id)
{
    if (consumer_id < 0 || consumer_id >= SPMC_MAX_CONSUMERS) return;
    uint64_t h = atomic_load_explicit(&ring->head, memory_order_acquire);
    atomic_store_explicit(&ring->consumers[consumer_id].tail, h, memory_order_relaxed);
    atomic_store_explicit(&ring->consumers[consumer_id].active, 1,
                           memory_order_release);
}

#ifdef __cplusplus
}
#endif

#endif /* SPMC_RING_H */
