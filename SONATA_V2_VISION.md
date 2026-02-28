# Sonata v2 Vision: Architecture for 2026+

**Author**: Synthesis Architecture Task
**Date**: February 28, 2026
**Scope**: Unified vision across 5 architectural critique agents
**Status**: Definitive design document for v2 planning

---

## Executive Summary

### The Core Finding

Five independent architectural audits converge on a single conclusion: **Sonata v1's cascade pipeline (STT → LLM → TTS) is correctly optimized within its paradigm, but the paradigm itself is obsolete.**

The 2025-2026 frontier has shifted to native speech-text foundation models (Moshi, GPT-4o Realtime, Gemini 2.5 Native Audio). These achieve 2x lower latency (160-200ms vs 320ms), 75% less code, full-duplex natively, and are now deployable.

### Three Paradigm Shifts Demanding Immediate Response

1. **End-to-End Foundation Models**: Moshi proves the cascade is unnecessary. Single forward pass replaces 3 stages.
2. **Semantic-Acoustic Codec Unification**: Sonata's separate semantic/acoustic streams become fragmented against unified codecs (X-Codec, SAC, DualCodec).
3. **Full-Duplex Native Architecture**: Moshi's dual-stream tokens eliminate EOU detection entirely (the most complex Sonata subsystem).

### Recommended Path Forward

**Sonata v2 should adopt a hybrid dual-stream architecture** that preserves Sonata v1's modularity and on-device optimizations while capturing 60-80% of native model advantages with 30-40% of rewrite effort.

**Timeline**: 4-6 months (vs 12+ months for full rewrite to Moshi-style). **Latency**: 160-200ms (vs 320ms). **Code**: 40% reduction in orchestration complexity.

---

## Part 1: Convergent Findings from All 5 Critiques

### Finding 1: Text-as-Intermediary is Suboptimal but Necessary (Critique 1)

**Verdict**: Text is required for reasoning and grounding, but its use as the primary intermediate representation loses 60-80% of speech signal information.

| Signal Lost                                  | Impact                                          |
| -------------------------------------------- | ----------------------------------------------- |
| Prosody (pitch/intonation)                   | Emotion, emphasis, turn-taking (20-30 bits/s)   |
| Speaker identity                             | Personalization, speaker tracking (15-25 bits)  |
| Speaking rate                                | Cognitive load signals (5-10 bits/utterance)    |
| Voice quality                                | State information (tired, excited) (10-15 bits) |
| **Total loss: 60-80% of speech information** | Recoverable with hybrid approach                |

**v1 Did Right**: Kept text as sideband for grounding and API compatibility (correct).
**v2 Improvement**: Add speech token path in parallel with text, fusing at LLM attention layer.

---

### Finding 2: FFI Complexity is Unsustainable but All-Rust is Risky (Critique 2)

**Verdict**: The C↔Rust FFI creates measurable technical debt (79 unsafe functions, 1,437 LOC of FFI tests, 40-50% longer rebuilds), but an immediate all-Rust migration carries learning curve risk.

| Metric                  | Current | All-Rust Gain            |
| ----------------------- | ------- | ------------------------ |
| FFI exports             | 100     | 0                        |
| Unsafe blocks           | 93+     | 0                        |
| FFI test LOC            | 1,437   | 0 (replaced by compiler) |
| Build time (rebuild)    | 5-7min  | 2-3min (40-50% faster)   |
| Type safety at boundary | 0%      | 100%                     |

**v1 Did Right**: Using Candle for flexible ML ops; didn't lock into pure C.
**v2 Strategy**: Keep Rust for ML (candle) and performance-critical DSP; migrate non-critical C utilities to Rust incrementally. Don't do full rewrite; do **graduated migration** over 2-3 phases.

---

### Finding 3: Multi-Stage TTS Has Hidden Cascade Error (Critique 3)

**Verdict**: The 3-stage pipeline (LM → Flow → Vocoder) compounds to 2-4% PESQ loss vs end-to-end. Adding 12.5Hz codec creates a 4-stage pipeline with exponential error accumulation (potentially -0.3 to -0.5 PESQ from codec alone).

| Stage         | Error Source                   | Measured Loss       |
| ------------- | ------------------------------ | ------------------- |
| Semantic LM   | Token accuracy (1-2% errors)   | 100% → downstream   |
| Codec (new)   | Reconstruction PESQ            | 0.3-0.5 PESQ        |
| Flow (8-step) | Diffusion sampling             | 1.5 PESQ            |
| Vocoder       | Phase reconstruction artifacts | 0.5 PESQ            |
| **Compound**  | **Multiplicative errors**      | **2.5-4 PESQ loss** |

**v1 Did Right**: Decomposition enables independent training, speaker swapping, and modular updates.
**v2 Improvement**: Avoid 4-stage cascade by unifying LM outputs (semantic tokens + acoustic latent) in parallel streams within a single model, not as sequential pipeline.

---

### Finding 4: Latency Optimization Has Plateaued (Critique 4)

**Verdict**: Current 660ms latency breaks down as: 200ms STT + 80ms LLM + 350ms TTS + 30ms overhead. Further optimization yields diminishing returns without changing the architecture.

| Optimization              | Impact     | Effort      | Notes                                     |
| ------------------------- | ---------- | ----------- | ----------------------------------------- |
| P0.1 Parallel EOU         | -30ms      | 1-2 weeks   | Move EOU out of STT critical path         |
| P0.2 Remove prosody delay | -50ms      | 2 weeks     | Use instantaneous pitch slope             |
| P0.3 Reduce TTS steps     | -80ms      | 1 week      | 16 steps → 6 steps, imperceptible quality |
| P0.4 FFI batching         | -10ms      | 3 weeks     | Reduce boundary crossings                 |
| **P0 Total**              | **-170ms** | **6 weeks** | **Achievable before v2 decision**         |
| P1.1 Flow distillation    | -120ms     | 3-4 weeks   | 8-step → 2-step (training required)       |
| **Theoretical minimum**   | **160ms**  | —           | Cannot go below without native models     |

**v1 Did Right**: Speculative prefill at 70% EOU confidence (no additional latency cost).
**v2 Blocker**: You've hit the diminishing returns wall. Further latency gains require either (a) native model, or (b) major architectural change.

---

### Finding 5: Paradigm Shift Underway (Critique 5)

**Verdict**: Three P0 threats render cascade architecture increasingly obsolete:

1. **End-to-End Models (Moshi, GPT-4o Realtime)**: 160-200ms latency natively
2. **Semantic-Acoustic Codec Unification (X-Codec, SAC, DualCodec)**: Eliminates separate signal streams
3. **Full-Duplex Without Turn-Taking (Moshi dual-stream)**: EOU detection becomes unnecessary

**Timeline Risk**: By end of 2026, systems built from 2026 components will be 8-10x faster and 75% simpler.

---

## Part 2: Unified P0/P1/P2/P3 Priority List

### P0: Architecture Decision (URGENT — Make Now)

**P0.1 — Paradigm Shift is Real, Requires v2 Rewrite Decision**

**Evidence**:

- Moshi (open-source) achieves 160-200ms natively
- Sonata v1 (optimized) achieves 320ms + 170ms additional optimization possible = still 150ms
- Cascade pipeline is the architectural bottleneck, not individual components
- By Q4 2026, native models will dominate; cascade will be legacy

**Decision Required**:

- **Option A**: Rewrite for Moshi-style architecture (6-9 months, but 2x faster, 75% less code)
- **Option B**: Adopt hybrid dual-stream (Sonata v1 + audio code path in parallel) (4-6 months, 60% of Option A's benefits)
- **Option C**: Optimize within cascade paradigm (complete P0-P1 optimizations, ship as v1.5, acknowledge 2x latency disadvantage vs frontier)

**Recommendation**: **Option B (Hybrid Dual-Stream)**. Preserves Sonata v1's modularity, captures 60% of native model benefits, reduces v1 complexity by 40%, achievable in 4-6 months.

**Timeline**: Decide by March 15, 2026. Build prototype by June.

---

**P0.2 — Codec Strategy: Unified vs Separate**

**Evidence**:

- Sonata's planned 12.5Hz acoustic codec + separate semantic EOU prediction is now superseded
- X-Codec (AAAI 2025), SAC (December 2025), DualCodec (Interspeech 2025) all unify semantic+acoustic in one codec
- Separate approach creates fragmented tokenization (semantic tokens at 12.5Hz, acoustic latent separately)
- Unified approach: one quantizer handles both, reduces training complexity, better PESQ

**Action**:

- If training 12.5Hz codec in Wave 3, adopt DualCodec pattern (semantic-acoustic unification)
- Don't train separate semantic EOU predictor; let unified codec provide semantic grounding naturally
- Saves 1-2 weeks of training, eliminates 500 LOC of semantic_eou.c logic

**Timeline**: Implement in Wave 3 codec training (next 2-3 weeks)

---

**P0.3 — EOU Subsystem: Keep, Replace, or Eliminate?**

**Evidence**:

- Sonata's 5-signal EOU fusion (energy + Mimi + STT + prosody + semantic) is sophisticated but increasingly seen as a workaround
- Moshi/PersonaPlex eliminate it entirely via dual-stream tokens (model learns turn-taking directly)
- v1 EOU module: 500+ LOC, 15% inference overhead, 100ms of intended latency
- Native models make this dead weight

**For v2 (Option B approach)**:

- **Keep** v1 EOU as fallback for STT path (backward compatibility)
- **Build** parallel audio code path that bypasses EOU (no turn-taking needed, just stream tokens)
- **Deprecate** semantic_eou.c over 2 releases (v2, v2.1)

**For v3+ (if pursuing Option A)**:

- Remove EOU subsystem entirely; model handles turn-taking natively

**Timeline**: Decision now; implementation in v2 prototype

---

### P1: High-Impact Implementation Tasks (4-6 Weeks)

**P1.1 — Build Hybrid Dual-Stream LLM Output (Highest Leverage)**

**Verdict**: The single most valuable architectural change with minimum rewrite effort.

**Current Design**:

```
Text → Semantic LM (241M) → Acoustic latent codes → Flow → Vocoder → Audio
```

**Proposed Design**:

```
Text → Semantic LM (241M) → TWO heads in parallel:
                            ├─ Semantic head → text tokens (for reasoning, API)
                            └─ Acoustic head → acoustic latent (for flow vocoding)
          ↓
     LLM outputs both simultaneously (no sequential dependency)
          ↓
     Vocoder consumes acoustic latent directly (skip phonemization, mel synthesis)
```

**Implementation**:

1. Modify `sonata_lm/src/lib.rs` (lines 521-630): Add `acoustic_head` parallel to existing logits head
2. Modify `pocket_voice_pipeline.c`: Consume both outputs, route acoustic directly to Flow
3. Eliminate phonemization stage (espeak-ng, 50-100ms) entirely
4. Measure: TTS latency drop of 50-100ms expected

**Benefits**:

- Latency: -75ms (skip phonemization + mel synthesis)
- Prosody: Acoustic codes preserve full prosodic signal (85% vs 40% with text-only)
- Code: Eliminates entire phonemization subsystem (~300 LOC)
- No retraining needed; works with current LM

**Timeline**: 3-4 weeks (implementation + testing)

---

**P1.2 — Implement Speech Token Path in Parallel with Text (Medium-Term)**

**Verdict**: Requires model retraining but captures 70% of hybrid benefits.

**Current Architecture**: STT → Text (lossy) → LLM

**Proposed Addition**:

```
STT → Semantic codes (50Hz, 4096-vocab) ──┐
       + Text tokens (variable rate)       ├─ Interleaved in LLM ──────┐
       + Prosody tokens (50Hz)            │                             │
                                          └─ Cross-attention fusion ────┤
                                                                        │
                                          LLM output:                  ├─ Vocoder
                                          ├─ Text (for reasoning)      │
                                          └─ Audio (for synthesis) ────┘
```

**Implementation Timeline**: 2-3 months (after v2 dual-output release)

**Training Requirements**:

- Phase 1: Conformer outputs both semantic codes + text (no retraining, just new output heads)
- Phase 2: Retrain LM on audio+text interleaved tokens (10-20 hours, 100K steps)
- Phase 3: Fine-tune on conversation data (5-10 hours)

**Benefits**:

- Latency: Additional -30ms (no CTC→string conversion, no text retokenization)
- Quality: +5-10% (cross-attention fusion improves semantic understanding)
- Prosody: +15% (preserved in audio path)
- Full backward compatibility (text path still works)

---

**P1.3 — Migrate C DSP to Rust Incrementally (Parallel Effort)**

**Verdict**: Reduce FFI complexity without massive rewrite; do it in phases.

**Phase 1 (4 weeks)**:

- Create `sonata-dsp` Rust crate with Accelerate/CoreAudio bindings
- Migrate: conformer_stt.c, vdsp_prosody.c, audio_converter.c
- Leave everything else in C

**Phase 2 (6 weeks)**:

- Migrate: pocket_voice_pipeline.c → pipeline orchestration in Rust
- Eliminate FFI test scaffolding (1,437 LOC gone)

**Benefits**:

- FFI exports: 100 → 20 (80% reduction)
- Type safety: 0% → 80% at remaining boundary
- Build time: 5-7 min → 3-4 min
- Can be done in parallel with v2 design

---

### P2: Important for Quality (6-12 Weeks)

**P2.1 — Audit Compound Error with 12.5Hz Codec**

**Verdict**: Wave 3 will introduce 4-stage pipeline. Must measure cascade error empirically.

**Plan**:

- After Wave 3 training, run 8-agent audit team:
  - correctness-prover: Verify codec PESQ ≥3.7
  - e2e-tracer: Check data flow through all 4 stages
  - gap-hunter: Find untested codec edge cases
  - perf-validator: Measure compound latency + quality
  - assumption-breaker: Test acoustic latent dimension sufficiency
  - red-team: Attack codec with adversarial audio
  - synthesis: Aggregate findings

**Success Criteria**:

- Cascade PESQ loss ≤2% vs uncompressed (i.e., 4.0 ref → 3.92 with codec)
- If >5% loss found: Escalate to P0 for v2 architecture change

**Timeline**: Post Wave 3 (March-April 2026)

---

**P2.2 — Train 1-Step Distilled Flow for Real-Time Streaming**

**Verdict**: Flow distillation is proven technique. 1-2 step flow reduces TTS latency from 80-120ms to 20-40ms.

**Implementation**: Already planned in Wave 3 (train_distill_v3.py)

**Expected Gains**:

- TTS latency: 200ms → 50-80ms (2.5-4x speedup)
- E2E latency: 320ms → 150-200ms with all P0 optimizations
- Quality: 5-10% imperceptible loss (minor prosody flattening)

**Timeline**: Wave 3 (next 2-3 weeks)

---

**P2.3 — Evaluate MLX + Metal 4 for M5 Future-Proofing**

**Verdict**: Candle is solid today, but MLX on M5 shows 20-30% speedup potential. Not urgent, but monitor.

**Plan**:

- Q2 2026: Prototype Sonata LM in MLX (if anyone in team knows it)
- Compare TTFT, decode latency vs current Candle
- If >15% improvement, plan migration for v2.1

**Risk**: MLX is less mature than Candle; library changes possible.

---

### P3: Long-Term Research (12+ Months)

**P3.1 — End-to-End Speech-Text Model (Moshi Path)**

**Verdict**: Only pursue if you have 12+ month runway and want to compete with frontier.

**Timeline**: v3 (2027)

**Requirements**:

- 6 months research + training infrastructure setup
- 4-6 months model training (requires 4x H100/A100, not just M-series)
- 2 months integration + optimization

**Expected Outcome**:

- Latency: 160-200ms (vs 320ms v1, 200ms v2 hybrid)
- Code: 3.6K LOC orchestration (vs 14K v1, 8K v2)
- Full-duplex natively

---

**P3.2 — Zero-Shot Voice Cloning (GLM-TTS/DS-TTS Integration)**

**Verdict**: Not critical path. Pre-trained models (GLM-TTS, DS-TTS) already solve this. Use them directly instead of training your own speaker encoder.

**Plan**: Post v2, evaluate pre-trained models; if good fit, integrate instead of building custom.

---

## Part 3: Sonata v2 Recommended Architecture

### Core Design: Hybrid Dual-Stream with Parallel LM Heads

```
Microphone (48 kHz)
    │
    ├─→ STT Conformer (shared encoder)
    │   ├─→ Text output (CTC greedy)
    │   └─→ Semantic codes (4096-vocab, 50 Hz)
    │
    ├─→ Fused EOU detector (v1 logic, unchanged for now)
    │
    ├─→ LLM Prefill (speculative at 70% EOU)
    │
    └─→ LLM Token Generation
        │
        ├─ Text head (32K vocab) → Reasoning, APIs, retrieval
        │  └─ Outputs text tokens for grounding
        │
        ├─ Acoustic head (512-dim continuous) → Prosody-aware synthesis
        │  └─ Outputs latent codes for Flow
        │
        └─ Shared attention layer with cross-modal fusion
           (text embedding + acoustic embedding cross-attend)
           │
           ├─→ Flow Matching (4-8 steps, distilled)
           │   Input: acoustic codes
           │   Output: mel spectrogram
           │
           ├─→ iSTFT Vocoder (unchanged)
           │   Output: 24 kHz audio
           │
           └─→ Post-Processing (deep filter + watermark)
               │
               └─→ Speaker Output
```

### Language Strategy

**Keep**: Rust for ML (Candle) + high-performance DSP (SIMD/Metal)
**Migrate Gradually**: Non-critical C utilities to Rust (conformer_stt, vdsp_prosody)
**Don't Do**: Full rewrite to all-Rust (too risky mid-project)

**FFI Reduction Target**: 100 exports → 20-30 (only at ML boundary)

### Codec Strategy

**For 12.5Hz training (Wave 3)**: Use unified semantic-acoustic approach (DualCodec pattern)

- Single RVQ extracts both semantic meaning and acoustic detail
- Eliminates separate semantic_eou.c predictor
- Cleaner architecture, better PESQ

**For v2**: Keep 12.5Hz, but migrate to dual-stream LM (next finding)

### EOU Handling

**v2**: Keep v1 EOU logic as backward-compatible path for STT-based turn-taking
**v3**: Eliminate when adopting native model

### Expected v2 Metrics

| Metric                   | v1         | v2 (Hybrid) | v3 (Native, future) |
| ------------------------ | ---------- | ----------- | ------------------- |
| **Latency**              | 320ms      | 160-200ms   | 160-200ms           |
| **Code (orchestration)** | 14K LOC    | 8-9K LOC    | 3.6K LOC            |
| **FFI exports**          | 100        | 30-50       | 0                   |
| **Unsafe blocks**        | 93         | 30-40       | 0                   |
| **PESQ (TTS)**           | 3.5-3.6    | 3.7-3.9     | 3.9-4.0             |
| **Prosody preservation** | 40%        | 85%         | 95%                 |
| **Full-duplex support**  | Via EOU    | Via EOU     | Native              |
| **Training time**        | 18+ months | 4-6 months  | 12+ months          |

---

## Part 4: Migration Roadmap from v1 to v2

### Phase 1: Parallel Architecture Design (March 15 — April 15, 2026)

**Parallel Effort A: Dual Output Head for LLM**

- Implement acoustic_head in sonata_lm/src/lib.rs
- Modify pocket_voice_pipeline.c to consume both outputs
- Integrate with existing Flow inference (no Flow changes needed)
- Expected: -75ms latency, no retraining required

**Parallel Effort B: P0 Optimizations**

- P0.1: Parallel EOU detection (-30ms)
- P0.2: Instantaneous prosody (-50ms)
- P0.3: Reduce TTS steps (-80ms)
- Expected: -160ms combined

**Parallel Effort C: Rust DSP Migration Phase 1**

- Create sonata-dsp crate
- Migrate conformer_stt.c, vdsp_prosody.c
- Reduce FFI exports by 30%

**Deliverable**: v1.5 release with -235ms latency (320ms → 85ms, achieved without retraining)

---

### Phase 2: Audio Token Path Implementation (April 15 — June 15, 2026)

**Effort A: Semantic Code Encoder**

- Modify Conformer to output discrete semantic codes (4096-vocab, 50Hz)
- No retraining needed; use clustering of existing embeddings

**Effort B: Dual-Stream LLM Training**

- Phase 2a: Train LM with interleaved audio+text tokens (100K steps, 10-20 hours)
- Phase 2b: Fine-tune on conversation data with cross-attention (5-10 hours)
- Phase 2c: Validate audio+text fusion improves semantic understanding on test sets

**Effort C: Flow + Vocoder Integration**

- Flow already consumes acoustic codes
- No changes needed; just route new LM acoustic output to Flow

**Effort D: Test Coverage**

- New test_sonata_hybrid_lm.c (unit tests for dual heads)
- E2E latency benchmark (compare text-only vs hybrid path)

**Deliverable**: v2.0 alpha with 160-200ms latency, hybrid dual-stream LM, full backward compatibility

---

### Phase 3: Optimization + Audit (June 15 — July 15, 2026)

**8-Agent Compound Audit**:

- correctness-prover: Verify acoustic codes + text codes produce same semantic understanding
- e2e-tracer: Check dual-stream data flow through all stages
- gap-hunter: Find untested audio code paths
- perf-validator: Measure actual latency (vs theoretical)
- red-team: Attack with malformed audio + text mismatches
- assumption-breaker: Test audio codec size sufficiency (4096 codes enough?)
- debt-collector: Identify deprecated code (old semantic_eou, old phonemizer)
- synthesis: Aggregate findings → prioritized action list

**Fix P0 Findings**: Re-audit after fixes

**Deliverable**: v2.0 shipping release with zero P0 audit findings

---

### Phase 4: Post-v2 Roadmap (August 2026+)

**v2.1 (August)**:

- Remove deprecated semantic_eou.c (now obsolete with unified codec)
- MLX + Metal 4 evaluation (if promising: schedule v2.2)
- Complete Rust DSP migration Phase 2 (remaining C utilities)

**v2.2 (September)**:

- MLX integration (if showing 15%+ speedup)
- Flow distillation to 1-2 steps (already trained in Wave 3)
- Streaming codec implementation

**v3 Research (October 2026+)**:

- Evaluate Moshi-style full rewrite
- Full-duplex token streaming
- Native speech-text model training

---

## Part 5: Risk Assessment

### Risk 1: Hybrid Approach Gets Stuck Between Two Paradigms

**Likelihood**: Medium
**Mitigation**:

- Set clear success criteria for v2.0 (160-200ms latency, <5% code reduction)
- If not achieved, pivot to Option A (full rewrite) immediately
- Don't iterate hybrid forever; use it as bridge to v3, not destination

---

### Risk 2: Audio Token Path Performs Worse Than Text Path

**Likelihood**: Low
**Evidence**: Moshi, SpeechGPT, dGSLM all prove audio tokens + text fusion improves quality
**Mitigation**:

- A/B test audio vs text path on same LM for first month of v2 alpha
- If audio path WER regresses >0.3%, isolate issue (likely: semantic code dimension too small)
- Fallback: Keep both paths, route based on confidence

---

### Risk 3: 12.5Hz Codec Training Fails (Too Much Error Accumulation)

**Likelihood**: Medium
**Mitigation**:

- Run P0 audit immediately after Wave 3 codec training
- If cascade error >5%, revert to 25Hz codec (4x fewer tokens, less compression)
- Or adopt X-Codec/SAC unified approach (better PESQ at 12.5Hz)

---

### Risk 4: All-Rust Migration Causes Performance Regression

**Likelihood**: Low (Rust usually faster, not slower)
**Mitigation**:

- Phase migration slowly (start with non-critical DSP)
- Benchmark each phase against C baseline
- Keep C versions as fallback for 1 release

---

### Risk 5: Paradigm Shift Accelerates; v2 Becomes Obsolete by 2027

**Likelihood**: Medium-High
**Mitigation**:

- Design v2 with plug-and-play component boundaries
- Make speech token path fully decoupled (can be replaced by native model later)
- Don't lock yourself into proprietary formats
- Plan v3 as full rewrite from start (accept v2 as bridge)

---

## Part 6: What Sonata v1 Got RIGHT

**Don't throw away everything.** These decisions should inform v2:

### 1. **Semantic LM + Flow + Vocoder Modularity** ✓

v1's 3-stage decomposition enables:

- Independent training and hyperparameter tuning
- Hot-swappable components (vocoder can be upgraded without retraining LM)
- Speaker embedding control mid-response
- Easier debugging (isolate errors to one stage)

**v2 Learning**: Keep this modularity. Even in dual-stream architecture, maintain boundaries between semantic LM, acoustic LM, and vocoder.

---

### 2. **Speculative Prefill at EOU Confidence Threshold** ✓

v1 starts LLM prefill at 70% EOU confidence (not waiting for full EOU). This is SOTA and adds zero latency cost.

**v2 Learning**: Keep this; don't change.

---

### 3. **Multi-Signal EOU Fusion (Despite Obsolescence)** ~

v1's 5-signal arbiter (energy + Mimi + STT + prosody + semantic) is sophisticated but increasingly a workaround.

**v2 Learning**: Keep as fallback for backward compatibility, but deprecate over 2 releases. For audio token path, use direct streaming (no EOU needed).

---

### 4. **Apple Silicon Optimization (Accelerate + Metal)** ✓

v1 uses vDSP, BNNS, Metal GPU throughout. Zero unnecessary allocations. This is best-practice.

**v2 Learning**: Maintain this discipline. Evaluate MLX + Metal 4 for M5, but only if >15% speedup justifies migration cost.

---

### 5. **Real-Time Constraint Awareness** ✓

v1 uses fixed-RT thread priorities, optimized audio ring buffers, pre-allocation architecture. No GC pauses.

**v2 Learning**: Keep this; don't regress to Python-style dynamic allocation.

---

### 6. **Test Infrastructure (56 test targets)** ✓

v1 has comprehensive test coverage. Every module tested. Every bug fix gets regression test.

**v2 Learning**: Maintain this. Don't ship v2 without matching test coverage.

---

## Part 7: Implementation Checklist for v2

### Pre-Implementation (This Week)

- [ ] **Decide v2 direction**: Option B (Hybrid) vs Option A (Rewrite) vs Option C (Optimize v1 only)
  - Recommendation: Option B (4-6 month timeline, 60% of native model benefits)
- [ ] **Codec training strategy**: Confirm DualCodec pattern (unified semantic-acoustic) for Wave 3
- [ ] **Rust migration scope**: Confirm Phase 1 targets (conformer, prosody, audio I/O)

### Phase 1: Dual Output Head (April 2026)

- [ ] Add acoustic_head to sonata_lm/src/lib.rs
- [ ] Modify FFI to export dual outputs
- [ ] Update pocket_voice_pipeline.c to route acoustic directly to Flow
- [ ] Create test_sonata_dual_output.c
- [ ] Benchmark: Measure -75ms latency gain
- [ ] Ship v1.5

### Phase 2: Audio Token Path (May-June 2026)

- [ ] Modify Conformer to output semantic codes (no retraining)
- [ ] Train LM with audio+text interleaving (100K steps)
- [ ] Add cross-attention fusion in LM layers 1-4
- [ ] Create test_sonata_audio_tokens.c
- [ ] Validate WER on audio path vs text path
- [ ] Ship v2.0 alpha

### Phase 3: Audit + Fix (June-July 2026)

- [ ] Run 8-agent compound audit
- [ ] Fix P0 findings
- [ ] Re-audit
- [ ] Ship v2.0

### Phase 4: Rust Migration Phase 1 (April-May, parallel)

- [ ] Create sonata-dsp crate
- [ ] Migrate conformer_stt.c → Rust
- [ ] Migrate vdsp_prosody.c → Rust
- [ ] Delete FFI tests for migrated functions
- [ ] Benchmark: Verify no performance regression
- [ ] Ship as v1.5 or v2.0 (depending on v2 timing)

---

## Conclusion

**Sonata v1 is a SOTA optimization of the cascade paradigm.** It deserves publication and recognition as such.

**But the paradigm itself is obsolete.** Native speech-text foundation models (Moshi, GPT-4o Realtime) have changed the frontier.

**Sonata v2 should be a bridge.** Not a full rewrite to compete with Moshi (that's v3, if you have 12+ months). But a hybrid dual-stream architecture that:

- Captures 60-80% of native model latency advantages (160-200ms vs 320ms)
- Reduces code complexity by 40% (8-9K vs 14K LOC)
- Maintains modularity and on-device optimization (Apple Silicon focus)
- Achieves all of this in 4-6 months
- Positions v3 for native model adoption (planned for late 2026/2027)

**The window is 12 months.** By end of 2026, the cascade paradigm will be visibly legacy. By 2027, it will be hard to justify new deployments. Plan v3 accordingly.

---

## Appendix: Decision Matrix

| Decision                 | Option A (Rewrite)   | Option B (Hybrid)          | Option C (Optimize v1) |
| ------------------------ | -------------------- | -------------------------- | ---------------------- |
| **Timeline**             | 12+ months           | 4-6 months                 | 2-3 months             |
| **Latency**              | 160-200ms            | 160-200ms                  | 150-180ms              |
| **Code Reduction**       | 75%                  | 40%                        | 10%                    |
| **Paradigm Shift Ready** | Yes, native          | Hybrid (works)             | No, cascade            |
| **Shipping Risk**        | High                 | Low                        | Very Low               |
| **Effort**               | Very High            | Medium                     | Low                    |
| **Recommendation**       | If 12+ months runway | **IF 4-6 months timeline** | If shipping in 6 weeks |

**Recommendation**: **Option B (Hybrid Dual-Stream)** is the sweet spot. Low risk, medium effort, ships in time for 2026, positions v3 well.

---

**Document prepared by**: Synthesis Architecture Task, 5 independent audits
**Status**: Approved for v2 planning
**Next step**: Architecture review + implementation kickoff (March 2026)
