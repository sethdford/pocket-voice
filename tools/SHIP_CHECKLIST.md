# Sonata Ship Readiness Checklist

**Version:** 1.0
**Last Updated:** 2026-03-08
**Status:** Ready for automation

This checklist ensures Sonata is production-ready before ship. Use `tools/ship_readiness.sh` to automate the technical checks (1-4). This document covers automated + manual verification steps.

## Automated Checks (Run `tools/ship_readiness.sh`)

The script validates:

- [ ] **Rust tests pass** — `cargo test --workspace` (all crates)
- [ ] **C tests pass** — `make test` (56+ test targets)
- [ ] **Python tests pass** — RVQ, codec roundtrip, duplex tests
- [ ] **Dylibs present** — `libsonata_lm.dylib`, `libsonata_flow.dylib`, `libsonata_speaker.dylib`
- [ ] **Models exported** — Distilled flow + configs valid JSON
- [ ] **No P0 security issues** — Audit reports clean
- [ ] **No undefined symbols** — nm check on all dylibs
- [ ] **No critical TODOs** — Scan for FIXME/CRITICAL in src

### Script Usage

```bash
cd /Users/sethford/Documents/pocket-voice
chmod +x tools/ship_readiness.sh
./tools/ship_readiness.sh
```

Expected output on GO:

```
═══════════════════════════════════════════════════════════
                  🚀 GO FOR SHIP 🚀
═══════════════════════════════════════════════════════════
```

Exit code: **0 = GO**, **1 = NO-GO**

---

## Section 1: Quality & Performance Validation

### 1.1 Audio Quality (Manual)

- [ ] **STT WER verified**
  - Test with LibriSpeech test-clean
  - Expected: ≤0.9% WER
  - Evidence: Run `make bench-sonata` and record RTF

- [ ] **TTS audio listened & approved**
  - Sample from 3+ different speakers
  - Listen for artifacts, naturalness, intelligibility
  - TTS TTFA: target <300ms (baseline was 543ms, needs distillation)
  - Evidence: Audio samples in `experiments/samples/`

- [ ] **E2E latency measured**
  - Mic → STT → LLM → TTS → Speaker
  - Target: <1000ms end-to-end
  - Document in: `PERFORMANCE_AUDIT.txt`

### 1.2 Voice Cloning (E2E Test)

- [ ] **Reference audio loading works**
  - Call `sonata_set_reference_audio()` with real audio
  - Verify speaker encoder produces embeddings
  - No crashes or memory leaks

- [ ] **Voice clone SIM-O ≥ 0.75**
  - Cosine similarity on speaker embeddings
  - Test 5+ different speakers
  - Evidence: Test output in `experiments/voice_cloning/results/`

- [ ] **Clone inference runs without hang**
  - Real-time inference (RTF < 1.0)
  - Generate 30s utterance end-to-end
  - No hanging or undefined behavior

### 1.3 Speculative Decoding

- [ ] **Drafter model loads & infers**
  - Load from `models/rnn_drafter.safetensors`
  - Run 100 token generation with tree guidance
  - Verify output logits shape (B, vocab)

- [ ] **Speedup measured**
  - Baseline: LM-only generation (tok/s)
  - With drafter: Speedup ≥ 1.5x
  - Evidence: Benchmark in `SPECULATIVE_DECODING_BENCH.txt`

- [ ] **No correctness loss**
  - Sample 10 generations with/without drafter
  - Output semantics unchanged (manual review or BLEU)
  - No hallucinations introduced

### 1.4 INT8 Quantization

- [ ] **Quantization pass executes**
  - `sonata_quantize_weights()` completes
  - QuantizedLinear layers load correctly

- [ ] **Inference latency improves**
  - Memory: Lower than FP32
  - Latency: ≤ FP32 (no slowdown)
  - Evidence: Benchmark in `QUANTIZATION_BENCH.txt`

- [ ] **Quality acceptable**
  - Output logits match baseline (L2 distance < 0.5%)
  - Perplexity on validation set: within 1% of FP32

---

## Section 2: Security & Hardening

### 2.1 FFI Bounds & Buffer Safety

- [ ] **All FFI entry points bounded**
  - Check: `MAX_TEXT_LEN`, `MAX_FRAMES`, `MAX_PROSODY_FRAMES`
  - All audio arrays validated before loop
  - Evidence: Audit report `.claude/SECURITY_AUDIT.md`

- [ ] **No buffer overflows**
  - Stack buffers have fixed size (valgrind/ASan clean)
  - Heap allocations checked for NULL
  - strncpy/snprintf used (not strcpy/sprintf)

- [ ] **Path traversal blocked**
  - Model load functions reject ".." in paths
  - No symlink attacks possible
  - Evidence: Review `src/**/load.c` for path validation

### 2.2 Numeric Stability

- [ ] **Softmax uses max-subtraction trick**
  - Prevents overflow/underflow in exp
  - Check: `sonata_softmax()` and all variants

- [ ] **KV cache bounded**
  - Max sequence length enforced
  - Old KV pairs evicted/pruned
  - No unbounded memory growth

- [ ] **RoPE position bounds checked**
  - Verify: `pos + seq_len <= max_seq_len`
  - No overflow in sine/cosine computation

### 2.3 Cryptography & Secrets

- [ ] **No hardcoded API keys**
  - Grep: `sk-`, `pk_`, `secret_` in src
  - All secrets from env/config files
  - Evidence: `.claude/SECURITY_AUDIT.md`

- [ ] **HTTPS used for model downloads**
  - Model URLs verified (no man-in-middle)
  - Check: `train/gce/launch.sh`, `models.py`

---

## Section 3: Build & Deployment

### 3.1 Cargo Workspace

- [ ] **All crates compile without warnings**
  - `cargo build --release --workspace 2>&1 | grep -i warning`
  - Expected: 0 warnings (or document why)

- [ ] **Dylib exports correct**
  - `nm -g target/release/libsonata_*.dylib` shows C symbols
  - No Rust mangled names exposed (use cdylib + #[no_mangle])

- [ ] **Dependencies locked**
  - Cargo.lock committed
  - All deps have version pins (e.g., "=0.9" not "0.9")

### 3.2 C Build (Makefile)

- [ ] **make clean && make test passes**
  - Full rebuild from scratch
  - No stale artifacts causing issues

- [ ] **All test binaries link**
  - No undefined symbols at runtime
  - All .dylib paths resolved (rpath set correctly)

- [ ] **Compiler flags correct**
  - `-O3 -flto` for release
  - `-march=armv8.5-a` or `-mcpu=native` for Apple Silicon
  - Security flags: `-D_FORTIFY_SOURCE=2 -fstack-protector-strong`

### 3.3 Models Exported

- [ ] **Sonata Flow Distilled**
  - `models/sonata_flow_distilled/sonata_flow_distilled.safetensors` (618 MB)
  - Config JSON valid and readable
  - Checksums match (if using GCS)

- [ ] **RNN Drafter**
  - `models/rnn_drafter.safetensors` exists and loads
  - Config specifies architecture (GRU layers, hidden dims)

- [ ] **Speaker Encoder**
  - `models/speaker_encoder.safetensors` available
  - Produces 256-dim embeddings (or documented size)

---

## Section 4: Integration & E2E Testing

### 4.1 Full Pipeline (mic→STT→LLM→TTS→speaker)

- [ ] **Pipeline initializes without crash**
  - `pocket_voice_pipeline_create()` succeeds
  - All modules load (STT, LLM, Flow, Speaker)

- [ ] **Real audio processed end-to-end**
  - Mic input → Conformer STT → Sonata LLM → Flow TTS → Vocoder → speaker
  - No hangs (all components within latency budget)
  - Audio output is audible and sensible

- [ ] **Barge-in works**
  - User can interrupt TTS mid-utterance
  - STT continues capturing while TTS stops
  - Pipeline recovers cleanly

### 4.2 Memory & Resource Usage

- [ ] **Memory footprint < 2 GB**
  - Measure: All models + buffers loaded
  - Check during real inference (not just instantiation)
  - No memory leaks (valgrind/ASan clean)

- [ ] **CPU usage acceptable**
  - STT: <1 core (Conformer 0.6B)
  - LLM: <2 cores (Sonata 500M, speculative)
  - TTS: <1.5 cores (Flow + Vocoder)
  - Total: <4 cores on Apple Silicon

- [ ] **GPU utilization good**
  - Metal GPU active during inference (check Activity Monitor)
  - No CPU fallback unless unavoidable
  - Power: Battery drain acceptable on MacBook Pro

### 4.3 Error Handling & Recovery

- [ ] **Network error handling**
  - LLM inference fails gracefully (retry w/ backoff)
  - Missing model files → clear error message
  - No null pointer dereferences

- [ ] **Audio format handling**
  - Rejects invalid WAV/PCM formats
  - Handles 8/16/32-bit PCM
  - Handles mono/stereo, 16–48 kHz

- [ ] **Concurrency safe**
  - Multiple threads can call API simultaneously
  - No race conditions in KV cache
  - Mutex protection validated (no deadlocks)

---

## Section 5: Testing & Audit Coverage

### 5.1 Test Suite

- [ ] **56+ C test targets pass**
  - `make test` completes with 0 failures
  - All outputs in `build/test_*.txt` (or logged)

- [ ] **Rust crates fully tested**
  - `cargo test --workspace` passes
  - Unit + integration tests for all modules
  - No `#[ignore]` tests remaining

- [ ] **Python tests pass**
  - `train/sonata/test_rvq_module.py`
  - `scripts/test_codec_roundtrip.py`
  - `experiments/personaplex/test_duplex.py`

### 5.2 Audit Trail (from `.claude/` reports)

- [ ] **Correctness Audit completed**
  - Algorithm correctness proven (not just tested)
  - All P0 items fixed and verified
  - Report: `.claude/CORRECTNESS_AUDIT_REPORT.md`

- [ ] **Security Audit completed**
  - FFI bounds, buffer safety, path traversal checked
  - All P0 items fixed
  - Report: `.claude/SECURITY_AUDIT.md`

- [ ] **Red Team Audit completed**
  - Malformed input handling tested
  - Fuzzing results reviewed (if applicable)
  - Report: `.claude/RED_TEAM_REPORT.md`

- [ ] **Gap Hunter completed**
  - Code coverage analyzed
  - All untested paths documented
  - Report: `.claude/GAP_HUNTER_REPORT.txt`

### 5.3 Remaining Known Issues

- [ ] **P1+ issues documented & prioritized**
  - Known limitations listed (e.g., "KV cache unbounded")
  - Not blocking ship but tracked for Phase 5
  - Document in: `KNOWN_ISSUES.md`

---

## Section 6: Documentation & Handoff

### 6.1 API Documentation

- [ ] **All C APIs documented**
  - Header comments include purpose, args, return values
  - Example usage provided in comments
  - Error codes documented

- [ ] **Rust APIs documented**
  - `/// doc comments` on all public items
  - Examples in doc tests where applicable

### 6.2 Deployment Documentation

- [ ] **Build instructions clear**
  - `README.md` covers: `make`, `cargo build`, model download
  - Tested on clean macOS VM (not just dev machine)

- [ ] **Environment variables documented**
  - `ANTHROPIC_API_KEY` required for LLM
  - Model paths, device selection documented
  - Example: `.env.example` provided

- [ ] **Known limitations documented**
  - Max input length
  - Supported languages
  - Latency targets (vs actual)
  - Documented in: `LIMITATIONS.md`

### 6.3 Operational Runbooks

- [ ] **Troubleshooting guide**
  - Common issues (e.g., "STT not detecting speech")
  - Debug steps (enable logging, check device)
  - Documented in: `TROUBLESHOOTING.md`

- [ ] **Performance tuning guide**
  - Trade-offs (quality vs latency)
  - Model size options (distilled vs full)
  - Device selection (GPU vs CPU)

---

## Section 7: Final Sign-Off

### 7.1 Pre-Ship Verification (1 week before)

**Date: ******\_\_\_\_********

- [ ] Run `tools/ship_readiness.sh` → **GO**
- [ ] All manual audits in Sections 1–5 complete
- [ ] Documentation reviewed & finalized
- [ ] No P0 issues remaining
- [ ] P1+ issues properly tracked in backlog

### 7.2 Ship Approval

**Approver: ********\_******** | Date: **\_\_\_****

- [ ] Reviewed entire checklist
- [ ] Confirmed GO status
- [ ] Approved for production deployment

---

## Appendix A: Performance Targets

| Metric                | Target  | Current | Status |
| --------------------- | ------- | ------- | ------ |
| STT WER               | <1%     | 0.9%    | ✓      |
| STT RTF               | <0.1    | 0.075   | ✓      |
| LLM latency (1 token) | <30ms   | TBD     | ?      |
| TTS TTFA              | <300ms  | 543ms   | ✗      |
| TTS RTF               | <0.25   | 0.196   | ✓      |
| E2E latency           | <1000ms | TBD     | ?      |
| Memory                | <2 GB   | TBD     | ?      |
| Speculative speedup   | >1.5x   | TBD     | ?      |

---

## Appendix B: File Locations

| Item                   | Path                                                |
| ---------------------- | --------------------------------------------------- |
| Ship readiness script  | `tools/ship_readiness.sh`                           |
| This checklist         | `tools/SHIP_CHECKLIST.md`                           |
| Rust tests             | `cargo test --workspace` (all crates)               |
| C tests                | `make test` (targets in Makefile)                   |
| Python tests           | `train/sonata/test_*.py`, `scripts/test_*.py`, etc. |
| Distilled flow model   | `models/sonata_flow_distilled/`                     |
| RNN drafter model      | `models/rnn_drafter.safetensors` (GCS or local)     |
| Security audit         | `.claude/SECURITY_AUDIT.md`                         |
| Correctness audit      | `.claude/CORRECTNESS_AUDIT_REPORT.md`               |
| Red team audit         | `.claude/RED_TEAM_REPORT.md`                        |
| Gap hunter audit       | `.claude/GAP_HUNTER_REPORT.txt`                     |
| Performance benchmarks | `PERFORMANCE_BENCHMARK_SUITE.md` in `.claude/`      |
| Known issues           | `KNOWN_ISSUES.md` (to be created)                   |
| Troubleshooting guide  | `TROUBLESHOOTING.md` (to be created)                |

---

## Appendix C: Quick Reference Commands

```bash
# Run automated checks
./tools/ship_readiness.sh

# Build from scratch
make clean && make && cargo build --release --workspace

# Run all tests
make test && cargo test --workspace

# Check for P0 security issues
grep -r "P0" .claude/SECURITY_AUDIT.md

# Validate model configs
python3 -c "import json; json.load(open('models/sonata_flow_distilled/sonata_flow_distilled_config.json'))"

# Check for undefined symbols
nm -u target/release/libsonata_*.dylib

# Measure memory usage
/usr/bin/time -l ./sonata

# Benchmark performance
make bench-sonata
```

---

**Ready for ship when all items checked.** 🚀
