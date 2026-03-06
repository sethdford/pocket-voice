# Sonata TTS Pipeline Performance Audit

**Date:** February 28, 2025  
**Scope:** Zero-Python real-time voice pipeline on Apple Silicon (C + Rust)  
**Current Benchmark:** TTFA 543–600ms, LM 43–46 tok/s, Flow first chunk ~135ms, iSTFT 1.9ms/1000 frames

---

## Executive Summary

The TTS pipeline has several optimization opportunities across **LM throughput**, **Flow/GPU efficiency**, **pipeline parallelism**, and **text preprocessing**. The highest-impact changes target TTFA (time-to-first-audio) and overall RTF.

---

## 1. High-Impact Optimizations (Expected >50ms TTFA improvement)

### 1.1 FlowWorker: Eliminate Blocking Before Submit

**File:** `src/pocket_voice_pipeline.c`  
**Lines:** 1707–1741, 1388–1395

**Issue:** `sonata_flush_chunk()` calls `sonata_collect_parallel()` before every `flow_worker_submit()`. This blocks until the previous Flow chunk completes before submitting the next one. LM and Flow therefore run serially at chunk boundaries rather than overlapping.

**Current flow:**

```c
// Line 1739–1741
if (e->flow_worker) {
    sonata_collect_parallel(e);   // BLOCKS until previous chunk done
    flow_worker_submit(e->flow_worker, e->semantic_tokens, n);
```

**Recommendation:** Use a **double-buffered FlowWorker**:

- Maintain two submission slots; submit to slot B while slot A is running.
- Only block when both slots are busy.
- Reduces TTFA by overlapping LM token generation with Flow processing.

**Implementation sketch:** Add `FlowWorkerSlot { tokens[], n, result_ready }` × 2; submit to free slot; collect from completed slot opportunistically.

---

### 1.2 Reduce First Chunk Size for Lower TTFA

**File:** `src/pocket_voice_pipeline.c`  
**Lines:** 1228–1229, 1908–1933

**Constants:**

```c
#define SONATA_FIRST_CHUNK 12
```

**Issue:** 12 tokens at 43 tok/s ≈ 279ms for LM alone. Smaller first chunk (e.g. 8) would cut LM wait before first audio, at the cost of slightly shorter first audio.

**Recommendation:**

- Make first chunk size configurable: `--sonata-first-chunk 8` (or 6 for lowest latency).
- Test 8 vs 12 for audible quality; 8 is a reasonable tradeoff and should save ~100ms to first audio.
- Benchmark: `make bench-sonata` with different `SONATA_FIRST_CHUNK` values.

---

### 1.3 Flow MoE: GPU→CPU Sync in Hot Path

**File:** `src/sonata_flow/src/lib.rs`  
**Lines:** 184–206 (MoELayer::forward)

**Issue:** MoE routing does a GPU→CPU sync to run top-k selection on CPU:

```rust
let gate_cpu = gate_f32.to_vec3::<f32>()?;  // GPU→CPU sync!
let mut indices: Vec<usize> = (0..n_experts).collect();
indices.sort_by(|&a, &b_idx| gate_cpu[bi][ti][b_idx]...);
```

Every Flow block with MoE incurs multiple synchronizations and a CPU sort per token.

**Recommendation:**

- Implement top-k routing on Metal (e.g. `select_nth` or custom shader).
- Or switch MoE models to dense-only at inference if acceptable for quality.
- Estimated impact: 10–30ms per flow forward on MoE-heavy models.

---

### 1.4 Flow Quality Mode: 3-Step First Chunk

**File:** `src/sonata_flow/src/lib.rs`  
**Lines:** 1391–1413 (quality mode), 21–23 (constants)

**Current:** FAST = 4 steps Euler. BALANCED = 6 steps. HIGH = 8 steps Heun.

**Recommendation:** Add a “first-chunk-only” mode:

- Use 3 ODE steps for the first chunk (TTFA-critical), then 4–6 for later chunks.
- Implement `sonata_flow_set_first_chunk_steps(engine, 3)` and use it in `sonata_flush_chunk()` when `e->is_first_chunk`.
- Expected: ~25% faster first Flow chunk (e.g. 135ms → ~100ms) with minor quality loss on the first segment.

---

## 2. Medium-Impact Optimizations (20–50ms TTFA or better RTF)

### 2.1 LM Throughput: Batch Forward for Speculative Decoding

**File:** `src/sonata_lm/src/lib.rs`  
**Lines:** 247–286 (forward_seq), 1343–1410 (step)

**Issue:** `sonata_lm_step()` runs a single-token forward. Speculative decoding uses `forward_seq` for batch verification but the pipeline often runs without a draft model.

**Recommendation:**

- Ensure speculative decoding is used when a draft model is available (`--sonata-draft`).
- If no draft: consider a tiny 2–4 layer draft (same architecture) for 2–3x effective tok/s from batch verification.
- AGENTS.md notes self-speculative on same GPU was 2x slower; use a smaller draft or accept single-token for now.

---

### 2.2 LM Sampling: Optimize Top-k/Top-p

**File:** `src/sonata_lm/src/lib.rs`  
**Lines:** 1427–1480 (sample_top_k_top_p)

**Issue:** `scratch.select_nth_unstable_by(k, ...)` + `sort_unstable_by` + iteration. For vocab 4096 and k=50, this is O(n) partial sort plus O(k log k) sort. `Vec::extend` and allocations in the hot path add overhead.

**Recommendation:**

- Pre-allocate `sampling_buf` at max vocab size (already done at line 1275).
- Consider a heap-based top-k (e.g. `BinaryHeap` with `Reverse`) for O(n log k) instead of full sort when k << n.
- Profile with `cargo flamegraph` to confirm sampling is not a major bottleneck before heavy tuning.

---

### 2.3 Flow Generate: Avoid Unnecessary CPU Copies

**File:** `src/sonata_flow/src/lib.rs`  
**Lines:** 1527–1531, 1626–1632, 1805, 1818–1819

**Issue:** FFI requires CPU buffers. Current pattern:

```rust
let mag_flat = mag_t.squeeze(0)?.contiguous()?.to_vec1::<f32>()?;
unsafe { std::ptr::copy_nonoverlapping(mag_flat.as_ptr(), out_magnitude, n_copy); }
```

This triggers GPU→CPU copy on every chunk. For ConvDecoder (direct audio), similar pattern for waveform.

**Recommendation:**

- For ConvDecoder path: consider keeping output on GPU and using a shared memory region (mmap, IOSurface) for zero-copy C access, if feasible.
- Short-term: ensure `contiguous()` is not causing extra copies; use `as_ptr()` on tensor storage when layouts match.
- Measure with Metal GPU capture; if transfer is significant, shared buffers are worth implementing.

---

### 2.4 Speculative TTS Warmup: Avoid Wasteful Empty Steps

**File:** `src/pocket_voice_pipeline.c`  
**Lines:** 5185–5190

**Issue:** While waiting for the first sentence, the pipeline runs:

```c
if (sentbuf_sentence_count(sentbuf) == 0 && !sentbuf_has_segment(sentbuf)) {
    tts->step(tts->engine);  // Empty step — no text fed yet!
}
```

For Sonata, `sonata_step` with no new tokens will still run LM forward (on BOS/previous state), consuming GPU time without producing useful audio.

**Recommendation:**

- Only run warmup steps when `sentbuf_has_segment(sentbuf)` is true (i.e. when there is text to generate).
- Or use a lightweight “priming” call that touches Metal buffers without full LM forward (e.g. a small dummy tensor op).
- Remove or guard the warmup if it increases GPU contention without benefit.

---

### 2.5 FlowWorker Thread: Reduce Per-Chunk Allocations

**File:** `src/pocket_voice_pipeline.c`  
**Lines:** 1258–1266, 1260–1261

**Issue:** Worker thread allocates large buffers at start:

```c
float *mag_batch = (float *)calloc(SONATA_MAX_FRAMES * SONATA_N_BINS, sizeof(float));
float *phase_batch = (float *)calloc(SONATA_MAX_FRAMES * SONATA_N_BINS, sizeof(float));
```

These are reused (good), but `SONATA_MAX_FRAMES=2000` × 513 × 4 × 2 ≈ 8MB per worker. For typical chunks of 12–80 frames, most of this is unused.

**Recommendation:** Allocate based on typical chunk size (e.g. 128 frames) or make buffer size dynamic from `req_n_tokens`. Reduces memory pressure and cache footprint.

---

## 3. Lower-Impact / Structural Optimizations

### 3.1 BNNS ConvNeXt Decoder: ANE Offload (Stub Only)

**File:** `src/bnns_convnext_decoder.c`  
**Lines:** 141–148 (load_mlmodelc), 110–139 (load_weights)

**Issue:** BNNS decoder exists but `bnns_convnext_load_weights` and `bnns_convnext_load_mlmodelc` are stubs. The pipeline uses the Rust ConvDecoder on Metal.

**Recommendation:** Complete BNNS/CoreML integration to run ConvNeXt on ANE, freeing GPU for Flow. Pipeline would become: Flow (GPU) → ConvNeXt (ANE) → iSTFT (AMX). High effort but good long-term win.

---

### 3.2 iSTFT: Batch Decode Optimization

**File:** `src/sonata_istft.c`  
**Lines:** 240–260 (sonata_istft_decode_batch)

**Issue:** `sonata_istft_decode_batch` loops over frames and calls `sonata_istft_decode_frame` per frame. Per-frame overhead (vvsincosf, vDSP_fft_zrip, etc.) is small; total 1.9ms for 1000 frames is already excellent.

**Recommendation:** iSTFT is not a bottleneck. Optional: process 2–4 frames per vDSP call if API allows batch sin/cos and FFT, but gains will be minimal. **Priority: Low.**

---

### 3.3 Phonemizer: espeak-ng Mutex and Initialization

**File:** `src/phonemizer.c`  
**Lines:** 41–43, 66–77

**Issue:** `g_espeak_mutex` serializes all phonemizer calls. `espeak_Initialize` and voice setup run under lock. First call to `phonemizer_text_to_ids` can be slow.

**Recommendation:**

- Initialize espeak-ng at pipeline startup, before any TTS segment.
- Consider caching phoneme IDs for frequent phrases (e.g. common greetings) if a cache fits the use case.
- Profile phonemizer latency; if &lt;10ms per segment, leave as is.

---

### 3.4 Text Preprocessing: process_segment

**File:** `src/pocket_voice_pipeline.c`  
**Lines:** 4468–4660 (process_segment)

**Issue:** Per segment: `text_auto_normalize`, `prosody_analyze_text`, `prosody_estimate_durations`, `prosody_conversation_adapt`, `emphasis_detect_quotes`, `emphasis_predict`, `ssml_parse`. All run sequentially on CPU.

**Recommendation:**

- Profile each stage (e.g. with `mach_absolute_time` or `clock_gettime`).
- If `prosody_estimate_durations` or `prosody_analyze_text` dominate, consider:
  - Lazy duration estimation (only when Flow uses duration conditioning).
  - Caching prosody for repeated segments.
- Ensure `sentbuf` flushes segments as early as possible to overlap preprocessing with LM generation.

---

### 3.5 SPM Tokenizer: Trie and Viterbi

**File:** `src/spm_tokenizer.c`  
**Lines:** 36–64 (trie), 102–150 (parse_sentencepiece_model)

**Issue:** Tokenization uses a trie and Viterbi DP. Load parses full protobuf into `pieces`, `scores`, `lens`. No obvious hot-path allocations after load.

**Recommendation:** Profile `sonata_tokenize` / `spm_encode`. If &lt;1ms per sentence, no change needed. For long text, consider streaming tokenization to start TTS earlier.

---

### 3.6 Post-Processing Chain: vDSP Prosody, LUFS, Breath

**Files:** `src/vdsp_prosody.c`, `src/lufs.c`, `src/breath_synthesis.c`

**Observation:** All use vDSP/AMX; LUFS and breath are applied after TTS output. They run in the playback path, not the TTFA path. No TTFA impact.

**Recommendation:** Focus on TTFA first. If overall RTF is a concern, profile the post-processing chain; vDSP should already be efficient.

---

## 4. Quantization and Model-Level Optimizations

### 4.1 INT8 Quantization for LM

**File:** `src/sonata_lm/src/lib.rs`  
**Dependencies:** `candle-core`, `candle-nn` (Cargo.toml)

**Issue:** LM uses FP16. INT8 would reduce memory bandwidth and could improve tok/s, but candle’s Metal backend may not fully support INT8 matmuls.

**Recommendation:**

- Check candle Metal INT8 support.
- If available: quantize LM to INT8 (e.g. per-channel or per-tensor); expect 1.5–2x throughput.
- Fallback: ensure FP16 is used consistently (no inadvertent F32 in hot path).

---

### 4.2 Flow ODE Solver: EPSS Schedule

**File:** `src/sonata_flow/src/lib.rs`  
**Lines:** 674–697 (sample), 661–662 (sigma_min)

**Issue:** Linear timestep schedule: `t0 = sigma_min + i * (1 - sigma_min) / steps`. EPSS (Exponential Probability Staircase) can improve quality at same step count.

**Recommendation:** Implement EPSS (see Flow v3 in `docs/codec_12hz_design.md`). May allow fewer steps for same quality, improving both TTFA and RTF.

---

## 5. Pipeline Parallelism Summary

| Stage           | Device    | Current           | Opportunity                                   |
| --------------- | --------- | ----------------- | --------------------------------------------- |
| LM              | Metal GPU | 43 tok/s          | Speculative, INT8, smaller first chunk        |
| Flow            | Metal GPU | 135ms/12 frames   | MoE on GPU, 3-step first chunk, double-buffer |
| ConvDecoder     | Metal GPU | In Flow thread    | ANE offload via BNNS                          |
| iSTFT           | AMX       | 1.9ms/1000 frames | Not a bottleneck                              |
| Post-processing | AMX       | Per-chunk         | Already efficient                             |

**Key:** FlowWorker double-buffering would let LM and Flow overlap at chunk boundaries; current design forces sequential execution there.

---

## 6. Recommended Implementation Order

1. **FlowWorker double-buffering** — High impact, moderate effort.
2. **Reduce `SONATA_FIRST_CHUNK` to 8** — Low effort, measure quality vs TTFA.
3. **Flow first-chunk 3-step mode** — Medium effort, good TTFA gain.
4. **Fix speculative TTS warmup** — Low effort, avoids wasted work.
5. **Flow MoE GPU top-k** — High effort, high impact for MoE models.
6. **BNNS ConvNeXt completion** — High effort, long-term architecture improvement.

---

## 7. Essential Files Reference

| File                          | Purpose                                                 |
| ----------------------------- | ------------------------------------------------------- |
| `src/pocket_voice_pipeline.c` | Orchestration, FlowWorker, sonata_step, process_segment |
| `src/sonata_lm/src/lib.rs`    | LM inference, sampling, warmup                          |
| `src/sonata_flow/src/lib.rs`  | Flow ODE, MoE, decoder, quality modes                   |
| `src/sonata_istft.c`          | Magnitude+phase → audio (vDSP)                          |
| `src/bnns_convnext_decoder.c` | ANE decoder (stub)                                      |
| `src/spm_tokenizer.c`         | SentencePiece tokenization                              |
| `src/phonemizer.c`            | espeak-ng IPA                                           |
| `src/sentence_buffer.c`       | LLM token accumulation                                  |
| `src/ssml_parser.c`           | SSML parsing                                            |
| `src/process_segment` path    | Text → TTS dispatch                                     |
| `tests/bench_sonata.c`        | Performance benchmarks                                  |

---

## 8. Verification

After changes, run:

```bash
make bench-sonata    # LM, Flow, iSTFT, E2E
make test-sonata    # Correctness
```

Target: TTFA &lt;450ms (from ~543ms), LM ≥48 tok/s, E2E RTF &lt;1.2x.
