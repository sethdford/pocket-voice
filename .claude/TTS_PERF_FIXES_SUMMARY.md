# TTS Performance Fixes Implementation Summary

**Date:** March 6, 2026  
**Scope:** Implementing top findings from `docs/TTS_PERFORMANCE_AUDIT.md`  
**Status:** All 4 fixes implemented or verified as already present

---

## Fix 1: Reduce SONATA_FIRST_CHUNK from 12 to 8

**Status:** ✅ ALREADY DONE  
**File:** `src/pocket_voice_pipeline.c`  
**Evidence:** Line 1239

```c
#define SONATA_FIRST_CHUNK_DEFAULT 8  /* configurable via --sonata-first-chunk */
```

The constant already defaults to 8 tokens, saving ~100ms TTFA (12 tokens → 8 tokens at 43 tok/s).

**Impact:** ~100ms TTFA reduction (pre-existing optimization)

---

## Fix 2: Guard Speculative TTS Warmup

**Status:** ✅ IMPLEMENTED  
**File:** `src/pocket_voice_pipeline.c`  
**Lines:** 6042-6050

**Before:**
```c
int steps = adaptive_steps_per_tick(audio, sentbuf_sentence_count(sentbuf));
for (int i = 0; i < steps; i++) {
    int step_result = tts->step(tts->engine);
    if (step_result \!= 0) break; /* done or error */
}
```

**After:**
```c
/* Only step if we have text to generate or are already processing */
if (sentbuf_sentence_count(sentbuf) > 0 || sentbuf_has_segment(sentbuf) || \!tts->is_done(tts->engine)) {
    int steps = adaptive_steps_per_tick(audio, sentbuf_sentence_count(sentbuf));
    for (int i = 0; i < steps; i++) {
        int step_result = tts->step(tts->engine);
        if (step_result \!= 0) break; /* done or error */
    }
}
```

**Rationale:** Prevents wasteful LM forward passes when there's no text to generate. The condition checks:
1. `sentbuf_sentence_count() > 0` — buffer has text tokens
2. `sentbuf_has_segment()` — buffer has accumulated a segment ready for processing
3. `\!tts->is_done()` — engine is already processing (continue draining)

**Impact:** Avoids GPU waste during silent periods before first sentence arrives (~20-50ms saved in typical startup)

---

## Fix 3: Flow First-Chunk 3-Step Mode

**Status:** ✅ ALREADY IMPLEMENTED  
**Files:**
- `src/sonata_flow/src/lib.rs` lines 23-25, 1172, 1395-1403
- `src/pocket_voice_pipeline.c` lines 1193, 6762-6763

**Evidence:**

Flow engine supports `first_chunk_steps` configuration:

```rust
pub extern "C" fn sonata_flow_set_first_chunk_steps(engine: *mut c_void, steps: c_int) {
    if engine.is_null() { return; }
    let eng = unsafe { &mut *(engine as *mut SonataFlowEngine) };
    eng.first_chunk_steps = if steps > 0 && steps <= 64 {
        Some(steps as usize)
    } else {
        None
    };
}
```

Pipeline invokes this when `tts_first_chunk_fast` is enabled:

```c
if (cfg.tts_first_chunk_fast)
    sonata_flow_set_first_chunk_steps(se->flow_engine, 3);
```

**How it works:**
- First chunk uses 3 ODE steps (Euler) instead of 4-8
- Subsequent chunks use normal quality mode (FAST=4, BALANCED=6, HIGH=8)
- Reset via `sonata_flow_reset_first_chunk()` for multi-segment synthesis

**Impact:** ~25% faster first Flow chunk (~135ms → ~100ms) with minimal quality loss on first segment

---

## Fix 4: FlowWorker Double-Buffering

**Status:** ✅ ALREADY FULLY IMPLEMENTED  
**File:** `src/pocket_voice_pipeline.c` lines 1246-1272, 1418-1438, 1816-1845

**Evidence:**

Double-buffered slot structure:

```c
typedef struct {
    struct {
        int             tokens[SONATA_MAX_FRAMES];
        int             n_tokens;
        float          *result_audio;
        int             result_len;
        int             request_pending;  /* 1 = submitted, waiting for worker */
        int             result_ready;     /* 1 = worker done, ready for collect */
    } slots[2];
    int             submit_slot;   /* next slot to submit to (0 or 1) */
    int             collect_slot;  /* next slot to collect from (0 or 1) */
    int             process_slot;  /* slot worker is processing / will process next */
    
    pthread_t       thread;
    pthread_mutex_t mutex;
    pthread_cond_t  request_cond;
    pthread_cond_t  done_cond;
    pthread_cond_t  slot_free_cond;
    // ...
} FlowWorker;
```

Non-blocking submit followed by conditional block:

```c
static void sonata_flush_chunk(SonataEngine *e) {
    // ...
    if (e->flow_worker) {
        /* Drain any ready results (non-blocking) to free slots for double-buffering */
        int len;
        while ((len = flow_worker_try_collect(e->flow_worker, e->collect_buf)) >= 0) {
            if (len > 0) {
                memcpy(&e->audio_buf[e->buf_write], e->collect_buf, len * sizeof(float));
                e->buf_write += len;
            }
        }
        /* If submit fails (both slots busy), block on collect then retry */
        while (flow_worker_submit(e->flow_worker, e->semantic_tokens, n) < 0) {
            int len = flow_worker_collect(e->flow_worker, e->collect_buf);
            // ... copy results ...
        }
    }
    // ...
}
```

**How it works:**
1. Try non-blocking collect from any ready slot (line 1829)
2. Submit to free slot; returns -1 if both busy (line 1837)
3. Only block on collect if both slots occupied (line 1838)
4. This allows LM to generate next tokens while Flow processes current chunk
5. Overlaps LM (CPU) and Flow (GPU) execution at chunk boundaries

**Impact:** Eliminates blocking wait before submit, enabling true parallelism between LM token generation and Flow audio synthesis (~50-100ms TTFA reduction once LM is optimized)

---

## Summary of Changes

| Fix | Status | Expected Impact | Implementation |
|-----|--------|-----------------|-----------------|
| 1. Reduce FIRST_CHUNK 12→8 | Already done | ~100ms TTFA | Config default = 8 |
| 2. Guard TTS warmup | Implemented | ~20-50ms | Check sentbuf before step() |
| 3. Flow first-chunk 3-step | Already done | ~25ms (135→100ms Flow) | Gated by `tts_first_chunk_fast` |
| 4. FlowWorker double-buffer | Already done | ~50-100ms potential | Non-blocking slots with conditional block |

**Cumulative Potential TTFA Reduction:** ~150-250ms (543ms → 300-400ms range)

---

## Verification

To enable all optimizations:
1. First-chunk 3-step mode: Use `--sonata-tts-first-chunk-fast` flag
2. Others: Already active by default

Test with:
```bash
make bench-sonata  # Measure TTFA, LM tok/s, Flow chunk time
make test-sonata   # Verify correctness
```

---

## Notes

- **Fix 1** was already implemented, suggesting this codebase is already partially optimized from audit recommendations.
- **Fix 3** is present but gated behind a flag; consider making this the default if quality loss is acceptable.
- **Fix 2** (warmup guard) is a quality-of-life fix that prevents unnecessary GPU work without algorithmic changes.
- **Fix 4** (double-buffering) is the most impactful for enabling true parallelism but requires the other layers (LM optimization, Flow MoE GPU routing) to reach full potential.

