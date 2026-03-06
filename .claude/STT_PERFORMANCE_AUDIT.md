# STT Pipeline Performance Audit — Sonata Project

**Audit date:** February 2025  
**Scope:** Zero-Python real-time voice pipeline for Apple Silicon (C + Rust)  
**Focus:** Performance optimization opportunities across mel → conformer → CTC → EOU path

---

## Executive Summary

The STT pipeline is well-architected with pre-allocated buffers, AMX/vDSP usage, and mmap'd weights. Several high-impact optimizations are available: **RoPE caching**, **vectorizing scalar loops** in sonata_stt/sonata_refiner, **ANE offloading** via BNNS, **silence skipping**, **conv2d malloc elimination**, and **lstm_ops vectorization**. Estimated cumulative gain: 20–40% throughput improvement and ~50–150ms latency reduction.

---

## 1. conformer_stt.c

### 1.1 [HIGH] malloc in conv2d_forward hot path (lines 824–838)

**Issue:** When `groups == 1` and `Kh > 1`, `im2col` buffer may be allocated via `malloc` if `col_need > work_sz`.

```c
// Line 824–838
size_t col_need = (size_t)col_h * out_spatial;
float *col = (col_need <= work_sz && work_buf) ? work_buf
             : (float *)malloc(col_need * sizeof(float));
```

**Impact:** Heap allocation in the forward pass causes unpredictable latency spikes and is unacceptable for real-time audio.

**Fix:** Pre-allocate a sufficiently large `sub_work` buffer in `workspace_alloc` to cover the maximal `im2col` size for Conv2D subsampling. The subsampling convs use fixed `T_in`/`F_in` from mel dimensions — compute `max_col_need` and allocate once.

---

### 1.2 [HIGH] RoPE computed per forward pass (implicit in sonata_stt — see sonata_stt)

The conformer_stt uses relative PE (Shaw-style) when `CSTT_FLAG_REL_PE` is set; sinusoidal PE is computed via `generate_rel_pe_asym`. The Sonata STT engine (sonata_stt.c) computes RoPE inline with `powf`/`cosf`/`sinf` per frame — see sonata_stt section.

---

### 1.3 [MEDIUM] depthwise_conv1d stack allocation (lines 695–696)

**Issue:** `col0..col3` and `cv0..cv3` are 8× `float[DW_PAD_T]` each, where `DW_PAD_T = MAX_SEQ_LEN + 128 = 2176`. That's ~70KB per channel group on the stack — risk of stack overflow on constrained threads.

**Fix:** Move these to workspace-allocated buffers (e.g., `ws->conv_mid` or a dedicated `float *dw_cols[8]`) allocated once in `workspace_alloc`.

---

### 1.4 [MEDIUM] per_feature_normalize_running column access (lines 336–357)

**Issue:** Inner loop iterates `t` then `f`, writing to `mel[t * n_mels + f]` — row-major layout makes column access (`col = mel + f`) stride by `n_mels`, causing cache misses.

**Fix:** Use a transposed copy or `feat_col` buffer to process columns contiguously; or batch multiple columns with vDSP for better cache behavior.

---

### 1.5 [LOW] INT8/INT4 tile size tuning

**Issue:** `INT8_TILE_N = 64` and `INT4_TILE_N = 64` are fixed. On Apple Silicon, L2 is ~16MB; tiles of 64×512 ≈ 128KB fit well, but for larger K (e.g., ff_dim 2048) consider 128 to improve GEMM efficiency.

**Fix:** Benchmark `INT8_TILE_N` ∈ {64, 128, 256} for typical Conformer dimensions.

---

## 2. sonata_stt.c

### 2.1 [HIGH] RoPE computed inline with powf/cosf/sinf (lines 272–289)

**Issue:** RoPE theta, cos, sin are recomputed for every (t, h, i) — O(T × n_heads × half_dim) transcendental calls per block.

```c
// Lines 274–287
for (int t = 0; t < T; t++) {
    for (int h = 0; h < n_heads; h++) {
        ...
        for (int i = 0; i < half_dim; i++) {
            float theta = (float)t * powf(10000.0f, -(2.0f * i) / (float)head_dim);
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
```

**Impact:** ~10–15% of MHSA time for a 4-layer 256-d model.

**Fix:** Precompute RoPE cos/sin at init (like sonata_refiner's `rope_cos`/`rope_sin`) for `MAX_FRAMES` positions. Store `[T_max][half_dim]` for cos and sin. Replace inner loop with:

```c
float cos_t = rope_cos[t * half_dim + i];
float sin_t = rope_sin[t * half_dim + i];
```

---

### 2.2 [HIGH] silu_inplace malloc fallback (lines 207–227)

**Issue:** When `n > 4096`, `silu_inplace` allocates `malloc(n * sizeof(float))`.

```c
float *t = (n <= 4096) ? tmp : (float *)malloc(n * sizeof(float));
```

**Impact:** For `T * ff_dim` (e.g., 100 × 1024 = 102,400), malloc is called every FFN block.

**Fix:** Pass a workspace buffer from the engine (e.g., `buf_ff` or a dedicated `silu_tmp` sized to `MAX_FRAMES * MAX_FF_DIM`) to avoid any malloc.

---

### 2.3 [HIGH] GLU/sigmoid in conv_module not vectorized (lines 336–341)

**Issue:** Per-element `expf` and scalar division in a tight loop.

```c
for (int t = 0; t < T; t++) {
    ...
    for (int i = 0; i < D; i++)
        dest[i] = a[i] * (1.0f / (1.0f + expf(-b_half[i])));
}
```

**Fix:** Use the same pattern as `silu_inplace`: `vDSP_vneg` + `vvexpf` + NEON/vDSP division, or a fused `glu()` helper like conformer_stt's `glu()`.

---

### 2.4 [MEDIUM] conv_module depthwise loop — col_in/col_out on stack (lines 351–365)

**Issue:** `col_in[2048+64]` and `col_out[2048]` per channel. For D=256, that's 256 × (2176 + 2048) × 4 ≈ 4.3MB stack usage in the inner loop — problematic.

**Fix:** Allocate `col_in` and `col_out` once in the engine struct (sized for `MAX_FRAMES + 2*pad` and `MAX_FRAMES`) and reuse.

---

### 2.5 [MEDIUM] Batch norm scalar loop (lines 365–371)

**Issue:** Nested loops over `i` and `t` with scalar ops; could use `vDSP_vma` with precomputed scale/shift.

**Fix:** Precompute `scale[i]` and `shift[i]` per channel, then `for (t) vDSP_vma(in+t*D, 1, scale, 1, shift, 1, out+t*D, 1, D)` — matching conformer_stt's `batch_norm_ws`.

---

### 2.6 [MEDIUM] EOU/softmax recomputation (lines 419–433, 719–751)

**Issue:** `sonata_stt_eou_probs` and `sonata_stt_eou_peak` recompute full softmax per frame from logits. If EOU is queried multiple times per utterance, this duplicates work.

**Fix:** Cache softmax probabilities when logits are produced (e.g., in `run_encoder` or `decode_with_beam`), and have `eou_probs`/`eou_peak` read from cache.

---

### 2.7 [LOW] CTC greedy decode — scalar argmax (lines 314–318)

**Issue:** Per-frame loop to find argmax is scalar.

**Fix:** Use `vDSP_maxvi` for each row to get (max_val, index) in one call.

---

### 2.8 [LOW] FP16 path incomplete (lines 739–759)

**Issue:** `sonata_stt_enable_fp16` allocates and converts weights but inference still uses FP32 GEMM. Comment states "FP16 kernels not yet implemented".

**Fix:** Implement `linear_forward_fp16` using `cblas_hgemm` (as in conformer_stt) with fp32→fp16 conversion at boundaries; expect ~2× GEMM throughput on supported systems.

---

## 3. sonata_refiner.c

### 3.1 [HIGH] silu_inplace scalar (lines 179–182)

**Issue:** No vectorization — per-element `expf` and division.

```c
static void silu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}
```

**Fix:** Use vDSP + NEON pattern from sonata_stt `silu_inplace` (or conformer_stt `silu_inplace_ws`).

---

### 3.2 [HIGH] Encoder MHSA O(T³) per-head loop (lines 215–233)

**Issue:** Triple nested loop over heads, qi, ki with `vDSP_dotpr` and manual attention accumulation. No batched GEMM.

```c
for (int h = 0; h < n_heads; h++) {
    for (int qi = 0; qi < T; qi++) {
        for (int ki = 0; ki < T; ki++) {
            vDSP_dotpr(...);
        }
        softmax_row(...);
    }
    for (int qi = 0; qi < T; qi++) {
        for (int vi = 0; vi < T; vi++) {
            /* manual += w * vr */
        }
    }
}
```

**Fix:** Restructure like sonata_stt's `mhsa_forward`: gather Q/K/V per head into contiguous blocks, use `cblas_sgemm` for scores and context. Same approach as conformer_stt MHSA.

---

### 3.3 [MEDIUM] rms_norm scalar (lines 154–159)

**Issue:** Scalar sum_sq and per-element scaling.

**Fix:** `vDSP_svesq` for sum of squares, then `vDSP_vsmul` with precomputed rms.

---

### 3.4 [MEDIUM] Decoder uses cblas_sgemv for single-step (lines 288–293)

**Issue:** sgemv is called 3× per layer (Q,K,V) — less efficient than batching for multi-token decode. For autoregressive refiner, single-step is required, so this is acceptable; but ensure no redundant conversion (e.g., sgemm with M=1 can sometimes be faster depending on batch).

**Fix:** Low priority; profile to confirm sgemv is optimal for M=1.

---

## 4. mel_spectrogram.c

### 4.1 [MEDIUM] Pre-emphasis malloc in hot path (lines 369–374)

**Issue:** When `n_samples > preemph_cap`, `mel_process` calls `free` and `malloc` to grow the buffer.

```c
if (n_samples > mel->preemph_cap) {
    free(mel->preemph_buf);
    mel->preemph_cap = n_samples + mel->cfg.sample_rate;
    mel->preemph_buf = (float *)malloc(mel->preemph_cap * sizeof(float));
```

**Impact:** First large chunk or variable-sized chunks can trigger realloc in the streaming path.

**Fix:** Pre-allocate `preemph_buf` to `pcm_cap` (or `sample_rate * 4`) at `mel_create` and never realloc. Cap `n_samples` per call to buffer size.

---

### 4.2 [MEDIUM] Ring buffer realloc (lines 298–318)

**Issue:** When `total_new > space`, the ring is grown with `malloc` + linearize + `free`. This can happen during long utterances.

**Fix:** Allocate ring to a fixed maximum (e.g., 4–10 seconds) at create. Refuse additional data beyond that, or document that ring growth is best-effort.

---

### 4.3 [LOW] vvlogf vs vvlog10f

**Issue:** `vvlogf` is used (natural log). Some models expect log10. Verify model training used log vs log10.

**Fix:** If model expects log10, switch to `vvlog10f`; otherwise keep `vvlogf` and document.

---

## 5. mimi_endpointer.c

### 5.1 [MEDIUM] layer_norm_1d variance loop not vectorized (lines 79–86)

**Issue:** Second loop over `i` for variance and output is scalar.

**Fix:** Use `vDSP_meanv`, `vDSP_measqv`, then `vDSP_vma` for the affine transform (like conformer_stt `layer_norm`).

---

### 5.2 [LOW] Softmax over 4 classes (lines 227–237)

**Issue:** Tiny loop (4 elements); vectorization overhead may exceed benefit. Optional: use `vDSP_maxv` + `vvexpf` + `vDSP_sve` + `vDSP_vsdiv` for consistency.

---

## 6. fused_eou.c

### 6.1 [LOW] compute_prosody_prob loops (lines 105–126)

**Issue:** Small buffer (PROSODY_BUF_SIZE=32); scalar loops are acceptable. Could use vDSP for sum if desired.

**Fix:** Very low priority; leave as-is unless profiling shows hotspot.

---

### 6.2 [LOW] compute_context_adj string ops

**Issue:** Called when context changes; not in per-frame hot path. No optimization needed.

---

## 7. noise_gate.c

### 7.1 [MEDIUM] memmove in overlap buffer (lines 224–227)

**Issue:** `memmove` shifts `overlap_out` by `hop_size` every frame — O(n) per frame.

**Fix:** Use a ring buffer (head/tail indices) for `overlap_out` to avoid memmove. Increment head modulo capacity and wrap reads.

---

### 7.2 [LOW] apply_aa_lowpass in speech_detector

**Note:** `apply_aa_lowpass` is in speech_detector.c (lines 79–90), not noise_gate. It's a 15-tap FIR — could use `vDSP_conv` for vectorized convolution.

---

## 8. native_vad.c

### 8.1 [MEDIUM] LSTM gates activation loop scalar (lstm_ops.h lines 81–89)

**Issue:** `lstm_step` uses scalar `lstm_sigmoid` and `tanhf` for 4H gates. With H=128, that's 512 scalar activations per step.

**Fix:** Vectorize sigmoid/tanh with vDSP or NEON: compute `-x` for i,f,g,o, then `vvexpf`, then `1/(1+exp)`. For tanh, use `vDSP_vtanh` if available, or `tanhf` in a NEON 4-wide loop.

---

### 8.2 [LOW] STFT magnitude loop (lines 236–243)

**Issue:** Per-bin magnitude sqrt(re²+im²). Could use `vDSP_zvabs` if layout allows.

**Fix:** Check if `stft` layout matches `DSPSplitComplex`; if so, use `vDSP_zvabs` for contiguous bins.

---

## 9. speech_detector.c

### 9.1 [MEDIUM] apply_aa_lowpass scalar FIR (lines 79–90)

**Issue:** 15-tap FIR implemented as nested loops — could use `vDSP_conv`.

**Fix:**

```c
vDSP_conv(pcm24, 1, aa_fir, 1, out, 1, n, AA_FIR_LEN);
```

(with boundary handling for first/last samples).

---

### 9.2 [MEDIUM] resample_24_to_16 linear interpolation (lines 96–109)

**Issue:** Per-sample linear interpolation with `(int)src` and `frac`. No vectorization.

**Fix:** Use `vDSP_vgen` for ramp generation and a vectorized lerp, or `AudioConverter` if available for higher quality.

---

### 9.3 [LOW] memmove in VAD/EP buffer consumption (lines 130–131, 166–168)

**Issue:** `memmove` to shift buffer after consuming a chunk. Same ring-buffer suggestion as noise_gate — use head/tail indices.

---

## 10. ctc_beam_decoder.cpp

### 10.1 [MEDIUM] BeamMap allocations per timestep (lines 238–239)

**Issue:** `BeamMap next_beams` is allocated every timestep; `std::unordered_map` with `vector<int>` keys causes many allocations.

**Fix:** Reuse two BeamMaps and swap; avoid rehashing. Pre-allocate `Hypothesis` objects in a pool.

---

### 10.2 [LOW] blank_skip_thresh branch

**Issue:** When `blank_skip > 0`, time steps with high P(blank) are skipped. Good for latency but ensure it doesn't hurt WER. Already present — verify tuning.

---

## 11. bnns_conformer.c

### 11.1 [HIGH] ANE offload not default

**Issue:** BNNS Conformer is only used when `--bnns-model` (or config) provides an mlmodelc path. The main Conformer path is CPU/AMX.

**Fix:** Document the export pipeline (NeMo → ONNX → CoreML → compile) and add a default model path if one is shipped. Prioritize ANE for conformer forward when available — can yield 2–5× speedup on M-series.

---

### 11.2 [LOW] Workspace realloc on shape change (lines 168–174)

**Issue:** `BNNSGraphContextGetWorkspaceSize` may change when dynamic shapes are set; buffer is reallocated.

**Fix:** Query max workspace for expected T range at init and allocate once to avoid realloc during inference.

---

## 12. Pipeline Integration (pocket_voice_pipeline.c)

### 12.1 [HIGH] Silence skipping not implemented

**Issue:** `feed_stt` always runs STT on accumulated audio. When `speech_detector_speech_active` is false for extended periods, STT still processes silence — wasting mel extraction and conformer forward.

**Fix:** Only call `stt->process_frame` when `speech_detector_speech_active(pp->speech_detector, energy_vad)` or when a minimum speech duration has been observed. Accumulate audio but defer STT until speech is detected. Reset accumulation on transition back to silence to avoid false triggers.

---

### 12.2 [MEDIUM] feed_stt buffer sizes

**Issue:** `capture_16` sized as `RESAMPLE_BUF_SIZE/3 + 64`. Ensure this is sufficient for worst-case 48→16kHz ratio; 48/16=3, so `n24/1.5` for 24→16. Verify no overflow.

**Fix:** Use a named constant (e.g., `MAX_16K_SAMPLES`) and assert `n_stt <= sizeof(capture_16)/sizeof(float)`.

---

### 12.3 [LOW] Double resample when STT is 16kHz

**Issue:** 48→24→16 involves two resample stages. Could consider 48→16 in one step if a suitable resampler exists, reducing memory traffic.

---

## 13. Rust STT (src/stt/src/lib.rs)

### 13.1 [LOW] Metal warmup

**Issue:** First inference may include Metal shader compilation. Ensure warmup is done at init (dummy forward) to avoid first-utterance latency spike.

**Fix:** Call a minimal forward pass after model load, before processing real audio.

---

## Summary Table — Optimizations Ranked by Impact

| Rank | File                       | Line(s)          | Issue                     | Impact | Effort |
| ---- | -------------------------- | ---------------- | ------------------------- | ------ | ------ |
| 1    | sonata_stt.c               | 272–289          | RoPE cached vs inline     | HIGH   | Low    |
| 2    | speech_detector / pipeline | —                | Silence skip before STT   | HIGH   | Medium |
| 3    | sonata_stt.c               | 207–227          | silu_inplace malloc       | HIGH   | Low    |
| 4    | sonata_refiner.c           | 179–182, 215–233 | silu + MHSA vectorization | HIGH   | Medium |
| 5    | conformer_stt.c            | 824–838          | conv2d malloc in hot path | HIGH   | Low    |
| 6    | sonata_stt.c               | 336–341          | GLU vectorization         | HIGH   | Low    |
| 7    | bnns_conformer             | —                | ANE default path          | HIGH   | High   |
| 8    | mel_spectrogram.c          | 369–374          | preemph realloc           | MEDIUM | Low    |
| 9    | sonata_stt.c               | 351–371          | conv stack + batchnorm    | MEDIUM | Low    |
| 10   | noise_gate.c               | 224–227          | overlap memmove → ring    | MEDIUM | Medium |
| 11   | speech_detector.c          | 79–90            | AA filter vDSP_conv       | MEDIUM | Low    |
| 12   | lstm_ops.h                 | 81–89            | LSTM gate vectorization   | MEDIUM | Medium |
| 13   | mimi_endpointer.c          | 70–86            | layer_norm vDSP           | MEDIUM | Low    |
| 14   | sonata_stt.c               | 739–759          | FP16 GEMM                 | LOW    | Medium |
| 15   | ctc_beam_decoder.cpp       | 238–239          | BeamMap pool              | MEDIUM | Medium |

---

## Essential Files for STT Optimization

1. **`src/conformer_stt.c`** — Primary Conformer engine (NeMo models)
2. **`src/sonata_stt.c`** — Sonata CTC engine (codec-based)
3. **`src/sonata_refiner.c`** — Two-pass refiner (when used)
4. **`src/mel_spectrogram.c`** — Feature extraction
5. **`src/speech_detector.c`** — VAD + EOU fusion, resampling
6. **`src/fused_eou.c`** — EOU decision logic
7. **`src/native_vad.c`** — Neural VAD
8. **`src/mimi_endpointer.c`** — LSTM endpointer
9. **`src/noise_gate.c`** — STT preprocessing
10. **`src/ctc_beam_decoder.cpp`** — Beam search
11. **`src/bnns_conformer.c`** — ANE offload
12. **`src/pocket_voice_pipeline.c`** — feed_stt, silence logic
13. **`src/lstm_ops.h`** — Shared LSTM
14. **`src/stt/src/lib.rs`** — Rust Metal STT (Kyutai)

---

_End of audit._
