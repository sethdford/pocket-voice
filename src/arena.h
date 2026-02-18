/**
 * arena.h — Bump-pointer arena allocator for zero-fragmentation per-turn memory.
 *
 * All allocations within a conversational turn come from a single contiguous
 * block. At turn end, one free() releases everything. Zero malloc/free overhead
 * during hot audio generation.
 *
 * Features:
 *   - Cache-line aligned allocations (64 bytes) by default
 *   - Checkpoint/restore for nested scopes (e.g., SSML segment processing)
 *   - Optional overflow arena chaining for large turns
 *   - Temp allocator for scratch buffers that rewind per sentence
 */

#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ARENA_DEFAULT_SIZE (1 << 20)  /* 1 MiB */
#define ARENA_ALIGNMENT    64         /* cache-line aligned */

typedef struct ArenaBlock {
    struct ArenaBlock *next;
    size_t size;
    size_t used;
    /* data follows immediately */
} ArenaBlock;

typedef struct {
    ArenaBlock *head;
    ArenaBlock *current;
    size_t total_allocated;
} Arena;

typedef struct {
    ArenaBlock *block;
    size_t used;
} ArenaCheckpoint;

/* ── Inline implementation ────────────────────────────── */

static inline ArenaBlock *arena_block_new(size_t min_size)
{
    size_t sz = min_size < ARENA_DEFAULT_SIZE ? ARENA_DEFAULT_SIZE : min_size;
    size_t total = sizeof(ArenaBlock) + sz;
    ArenaBlock *b = (ArenaBlock *)malloc(total);
    if (!b) return NULL;
    b->next = NULL;
    b->size = sz;
    b->used = 0;
    return b;
}

static inline Arena arena_create(size_t initial_size)
{
    Arena a;
    a.head = arena_block_new(initial_size > 0 ? initial_size : ARENA_DEFAULT_SIZE);
    a.current = a.head;
    a.total_allocated = 0;
    return a;
}

static inline void *arena_alloc(Arena *a, size_t size)
{
    size_t aligned = (size + (ARENA_ALIGNMENT - 1)) & ~(ARENA_ALIGNMENT - 1);
    ArenaBlock *b = a->current;

    /* Fast path: fits in current block */
    if (b && b->used + aligned <= b->size) {
        void *ptr = (char *)(b + 1) + b->used;
        b->used += aligned;
        a->total_allocated += aligned;
        return ptr;
    }

    /* Slow path: allocate new block */
    ArenaBlock *nb = arena_block_new(aligned);
    if (!nb) return NULL;
    if (b) b->next = nb;
    a->current = nb;
    if (!a->head) a->head = nb;

    void *ptr = (char *)(nb + 1);
    nb->used = aligned;
    a->total_allocated += aligned;
    return ptr;
}

static inline void *arena_calloc(Arena *a, size_t count, size_t size)
{
    size_t total = count * size;
    void *ptr = arena_alloc(a, total);
    if (ptr) memset(ptr, 0, total);
    return ptr;
}

static inline char *arena_strdup(Arena *a, const char *s)
{
    size_t len = strlen(s) + 1;
    char *p = (char *)arena_alloc(a, len);
    if (p) memcpy(p, s, len);
    return p;
}

static inline ArenaCheckpoint arena_checkpoint(Arena *a)
{
    ArenaCheckpoint cp;
    cp.block = a->current;
    cp.used = a->current ? a->current->used : 0;
    return cp;
}

static inline void arena_restore(Arena *a, ArenaCheckpoint cp)
{
    if (!cp.block) return;

    /* Compute how much to subtract from total_allocated */
    size_t freed = 0;
    ArenaBlock *b = cp.block->next;
    while (b) {
        freed += b->used;
        ArenaBlock *next = b->next;
        free(b);
        b = next;
    }
    cp.block->next = NULL;

    /* Rewind within the checkpoint block */
    freed += cp.block->used - cp.used;
    cp.block->used = cp.used;
    a->current = cp.block;

    if (freed <= a->total_allocated)
        a->total_allocated -= freed;
    else
        a->total_allocated = 0;
}

static inline void arena_reset(Arena *a)
{
    /* Free all blocks except head, rewind head */
    if (!a->head) return;
    ArenaBlock *b = a->head->next;
    while (b) {
        ArenaBlock *next = b->next;
        free(b);
        b = next;
    }
    a->head->next = NULL;
    a->head->used = 0;
    a->current = a->head;
    a->total_allocated = 0;
}

static inline void arena_destroy(Arena *a)
{
    ArenaBlock *b = a->head;
    while (b) {
        ArenaBlock *next = b->next;
        free(b);
        b = next;
    }
    a->head = NULL;
    a->current = NULL;
    a->total_allocated = 0;
}

#ifdef __cplusplus
}
#endif

#endif /* ARENA_H */
