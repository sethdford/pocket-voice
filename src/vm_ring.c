/**
 * vm_ring.c — Virtual memory mirrored ring buffer implementation.
 *
 * Uses mach_vm_allocate + mach_vm_remap on macOS to create a double-mapped
 * region. The same physical pages appear twice in virtual address space,
 * eliminating all wraparound logic from read/write hot paths.
 */

#include "vm_ring.h"
#include <string.h>
#include <stdio.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_map.h>
#endif

int vm_ring_create(VMRingBuffer *rb, uint32_t n_floats)
{
    if (!rb || n_floats == 0) return -1;

    memset(rb, 0, sizeof(VMRingBuffer));

#ifdef __APPLE__
    vm_size_t page_size = vm_page_size;

    /* Round up to page-aligned float count */
    size_t byte_size = (size_t)n_floats * sizeof(float);
    byte_size = (byte_size + page_size - 1) & ~(page_size - 1);
    n_floats = (uint32_t)(byte_size / sizeof(float));

    /* Ensure power-of-two for mask-based indexing */
    uint32_t po2 = 1;
    while (po2 < n_floats) po2 <<= 1;
    n_floats = po2;
    byte_size = (size_t)n_floats * sizeof(float);
    byte_size = (byte_size + page_size - 1) & ~(page_size - 1);

    mach_vm_address_t addr = 0;
    kern_return_t kr;

    /* Allocate 2x virtual space */
    kr = mach_vm_allocate(mach_task_self(), &addr, byte_size * 2, VM_FLAGS_ANYWHERE);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "vm_ring: mach_vm_allocate failed: %s\n", mach_error_string(kr));
        return -1;
    }

    /* Deallocate the second half — we'll remap into it */
    kr = mach_vm_deallocate(mach_task_self(), addr + byte_size, byte_size);
    if (kr != KERN_SUCCESS) {
        mach_vm_deallocate(mach_task_self(), addr, byte_size * 2);
        return -1;
    }

    /* Remap the first half into the second half's address space */
    mach_vm_address_t mirror = addr + byte_size;
    vm_prot_t cur_prot, max_prot;
    kr = mach_vm_remap(mach_task_self(),
                       &mirror,
                       byte_size,
                       0,                    /* mask */
                       VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
                       mach_task_self(),
                       addr,                 /* source */
                       FALSE,                /* copy: NO — share the same pages */
                       &cur_prot,
                       &max_prot,
                       VM_INHERIT_DEFAULT);
    if (kr != KERN_SUCCESS || mirror != addr + byte_size) {
        fprintf(stderr, "vm_ring: mach_vm_remap failed: %s\n", mach_error_string(kr));
        mach_vm_deallocate(mach_task_self(), addr, byte_size);
        return -1;
    }

    rb->buffer = (float *)addr;
    rb->size = n_floats;
    rb->mask = n_floats - 1;
    rb->vm_addr = (void *)addr;
    rb->vm_size = byte_size * 2;
    atomic_store(&rb->head, 0);
    atomic_store(&rb->tail, 0);

    /* Zero the buffer */
    memset(rb->buffer, 0, byte_size);

    return 0;

#else
    /* Fallback: standard allocation (no mirroring — callers must handle wrap) */
    uint32_t po2 = 1;
    while (po2 < n_floats) po2 <<= 1;
    n_floats = po2;

    rb->buffer = (float *)calloc(n_floats * 2, sizeof(float));
    if (!rb->buffer) return -1;
    rb->size = n_floats;
    rb->mask = n_floats - 1;
    rb->vm_addr = rb->buffer;
    rb->vm_size = (size_t)n_floats * 2 * sizeof(float);
    atomic_store(&rb->head, 0);
    atomic_store(&rb->tail, 0);
    return 0;
#endif
}

void vm_ring_destroy(VMRingBuffer *rb)
{
    if (!rb || !rb->vm_addr) return;

#ifdef __APPLE__
    mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)rb->vm_addr, rb->vm_size);
#else
    free(rb->vm_addr);
#endif

    memset(rb, 0, sizeof(VMRingBuffer));
}
