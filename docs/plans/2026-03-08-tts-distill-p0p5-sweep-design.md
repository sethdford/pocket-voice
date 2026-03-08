# TTS Distillation Integration + P0-P5 Security/Architecture Sweep

**Date:** 2026-03-08
**Status:** In Progress

## Problem

Sonata's TTS TTFA is 543ms (3-4x slower than SOTA). 43 audit findings span security holes, architecture debt, unintegrated modules, and unvalidated features. The distilled Flow model (50K steps, 1-step inference) is trained and on GCS but not exported or integrated.

## Approach

5 parallel agents in isolated git worktrees, each owning distinct files:

| Agent          | Scope                                 | Files                                           | Issues             |
| -------------- | ------------------------------------- | ----------------------------------------------- | ------------------ |
| 1: TTS Distill | Export + validate distilled model     | `train/sonata/`, `models/`                      | TTS 543ms -> 150ms |
| 2: sonata_lm   | All LM security + arch fixes          | `src/sonata_lm/src/*.rs`                        | 12 fixes           |
| 3: sonata_flow | Flow security + streaming fixes       | `src/sonata_flow/src/lib.rs`                    | 3 fixes            |
| 4: C + Build   | Pipeline, Makefile, Cargo.tomls, SEGV | `pocket_voice_pipeline.c`, `Makefile`, `*.toml` | 7 fixes            |
| 5: E2E Tests   | Validation tests for untested paths   | New test files                                  | 6 test suites      |

## Issues Addressed

### P0 (6)

- Buffer overflow: `sonata_lm_set_prosody` (integer overflow in pointer arithmetic)
- Integer overflow: `sonata_flow_generate_streaming_chunk` (unbounded read)
- Path traversal: `sonata_lm_create` (no `..` validation)
- Invalid Rust edition: `stt/llm` Cargo.toml (`"2024"` -> `"2021"`)
- RoPE position overflow: no bounds check on cos/sin cache indexing
- Speaker embedding table: no size validation vs config

### P1 (8)

- Double-free: `sonata_lm_destroy` (null check)
- Unbounded slice: `sonata_lm_set_text` (MAX_TEXT_LEN)
- Softmax NaN: missing max subtraction
- KV cache: unbounded growth
- Attention scale: mixed F32/F64
- Drafter squeeze: fragile shape assumption
- INT8 dtype chain: F32->F16 scale precision loss
- C buffer overflow: `sonata_stt_get_words_wrapper`

### P2 (4)

- Config DoS: max_seq_len unbounded
- Unchecked malloc: `sonatav2_set_text_done`
- Duplicate Makefile target
- Duplicate macro definitions

### P3 (1)

- Error message info leaks (full paths in production)

### Unvalidated (6)

- Voice cloning E2E
- Flow streaming API
- Dual-head LM output
- Drafter acceptance/rejection
- Distilled flow loading
- Speaker encoder integration

## Success Criteria

- All 43 findings addressed
- `make test` passes
- Distilled flow produces audio in 1 ODE step
- No new security vulnerabilities
- E2E test coverage for all previously untested paths

## Expected Impact

- TTS TTFA: 543ms -> ~150ms (3.6x speedup)
- E2E pipeline: ~1200ms -> ~800ms
- Security: 0 P0 vulnerabilities remaining
- Test coverage: 6 new test suites
