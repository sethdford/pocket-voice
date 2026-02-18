/**
 * vm_ring.h — Virtual memory mirrored ring buffer.
 *
 * Uses mach_vm_remap to map the same physical pages twice contiguously
 * in virtual address space. This means any read/write that wraps around
 * the buffer boundary sees the data as contiguous — zero memcpy for
 * wraparound, zero branching in the hot path.
 *
 * Memory layout:
 *   [page 0..N-1] [page 0..N-1]   ← same physical pages, mapped twice
 *    ^base          ^base+size
 *
 * Any pointer from base to base+size-1 can read up to `size` bytes
 * without wrap logic. The OS page table handles the mirroring.
 */

#ifndef VM_RING_H
#define VM_RING_H

#include <stdint.h>
#include <stdatomic.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *buffer;          /* base pointer (first mapping) */
    uint32_t size;          /* buffer size in floats (must be page-aligned) */
    uint32_t mask;          /* size - 1 */
    _Alignas(64) _Atomic uint64_t head;  /* write position */
    _Alignas(64) _Atomic uint64_t tail;  /* read position */
    void *vm_addr;          /* raw mach_vm allocated address */
    size_t vm_size;         /* total virtual mapping size (2x physical) */
} VMRingBuffer;

/**
 * Create a VM-mirrored ring buffer.
 * @param n_floats  Desired capacity in floats (rounded up to page multiple)
 * @return 0 on success, -1 on failure
 */
int vm_ring_create(VMRingBuffer *rb, uint32_t n_floats);

/** Destroy and unmap all memory. */
void vm_ring_destroy(VMRingBuffer *rb);

/** Available floats to read. */
static inline uint32_t vm_ring_available_read(const VMRingBuffer *rb)
{
    uint64_t h = atomic_load_explicit(&rb->head, memory_order_acquire);
    uint64_t t = atomic_load_explicit(&rb->tail, memory_order_relaxed);
    return (uint32_t)(h - t);
}

/** Available space to write. */
static inline uint32_t vm_ring_available_write(const VMRingBuffer *rb)
{
    return rb->size - vm_ring_available_read(rb);
}

/**
 * Write data into the ring. Zero-copy — writes directly through mirrored pages.
 * @return 0 on success, -1 if insufficient space
 */
static inline int vm_ring_write(VMRingBuffer *rb, const float *data, uint32_t count)
{
    if (vm_ring_available_write(rb) < count) return -1;

    uint64_t h = atomic_load_explicit(&rb->head, memory_order_relaxed);
    uint32_t offset = (uint32_t)(h & rb->mask);

    /* Because of VM mirroring, we can always do a single contiguous copy
       even when offset + count > size — the mirror handles wraparound. */
    memcpy(rb->buffer + offset, data, (size_t)count * sizeof(float));

    atomic_store_explicit(&rb->head, h + count, memory_order_release);
    return 0;
}

/**
 * Read data from the ring. Zero-copy contiguous read through mirrored pages.
 * @return 0 on success, -1 if insufficient data
 */
static inline int vm_ring_read(VMRingBuffer *rb, float *out, uint32_t count)
{
    if (vm_ring_available_read(rb) < count) return -1;

    uint64_t t = atomic_load_explicit(&rb->tail, memory_order_relaxed);
    uint32_t offset = (uint32_t)(t & rb->mask);

    memcpy(out, rb->buffer + offset, (size_t)count * sizeof(float));

    atomic_store_explicit(&rb->tail, t + count, memory_order_release);
    return 0;
}

/**
 * Get a direct pointer to readable data. Returns contiguous even at wrap.
 * Does NOT advance the tail — call vm_ring_advance_read() after consuming.
 */
static inline const float *vm_ring_peek(const VMRingBuffer *rb, uint32_t *out_count)
{
    uint32_t avail = vm_ring_available_read(rb);
    if (out_count) *out_count = avail;
    uint64_t t = atomic_load_explicit(&rb->tail, memory_order_relaxed);
    uint32_t offset = (uint32_t)(t & rb->mask);
    return rb->buffer + offset;
}

/** Advance read pointer after consuming data from peek. */
static inline void vm_ring_advance_read(VMRingBuffer *rb, uint32_t count)
{
    uint64_t t = atomic_load_explicit(&rb->tail, memory_order_relaxed);
    atomic_store_explicit(&rb->tail, t + count, memory_order_release);
}

#ifdef __cplusplus
}
#endif

#endif /* VM_RING_H */
