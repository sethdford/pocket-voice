# SOTA Gap Closure — Design Document

Date: 2026-03-05
Status: Approved

## Goal

Close all 8 remaining SOTA gaps in the Sonata voice pipeline through parallel agent workstreams with isolated worktrees.

## Workstreams

### Tier 1: Code-Only (parallel agents, no GPU needed)

#### 1. Flow Distillation Inference

- **Files**: `src/sonata_flow/src/lib.rs`
- **Task**: Add distilled checkpoint loader. Distilled models use same architecture but are trained for 1-step generation. The Rust code already supports variable `n_steps` — just need to detect distilled checkpoint format and default to 1 step.
- **Key**: Distilled checkpoints from `train_distill_v3.py` use same weight layout as standard flow v3, but include a `distilled: true` flag in metadata. Loader should read this and set `n_steps=1`, `solver=Euler`.

#### 2. ReDrafter Speculative Decoding

- **Files**: `src/sonata_lm/src/lib.rs`, new `src/sonata_lm/src/drafter.rs`
- **Task**: Implement GRU draft model inference in Rust (candle). Load 3.5M param GRU drafter from safetensors. Run K draft tokens, verify against main LM, accept/reject. Tree attention already referenced in Wave 1.
- **Architecture**: `hidden_proj(1024→256) → token_emb(32000,256) → GRU(256,256,2layers) → output_head(256,32000)`
- **Reference**: `train/sonata/train_drafter.py` for exact weight names and config format.

#### 3. Denoiser Pipeline Integration

- **Files**: `src/pocket_voice_pipeline.c`, `src/deep_filter.c`, `src/deep_filter.h`
- **Task**: Wire `deep_filter_create/process/destroy` into the pipeline's audio input path. Add denoiser as optional preprocessing stage before STT. Add enable/disable config. Write test.
- **C API already exists**: `deep_filter_create()`, `deep_filter_process()`, `deep_filter_destroy()`

#### 4. Voice Cloning Pipeline

- **Files**: `src/pocket_voice_pipeline.c`, `src/sonata_flow/src/lib.rs`
- **Task**: Load speaker encoder model, extract speaker embedding from reference audio, pass embedding to flow model as conditioning. The flow v3 model already accepts speaker embeddings via cross-attention — need to wire the encoder output to the flow input.
- **Reference**: `train/sonata/voice_prompt.py` for speaker encoder architecture.

#### 5. 12.5Hz Codec C Inference

- **Files**: new `src/codec_12hz.c`, `src/codec_12hz.h`
- **Task**: C inference module for 12.5Hz codec. FSQ dequantization (8^4 codebook), ConvDecoder or iSTFT decoder, ring buffer overlap-add. Follow patterns from `sonata_istft.c`.
- **Architecture**: FSQ indices → embedding lookup → 5-stage transposed conv (upsample 1920x) → waveform. Or iSTFT path: indices → decoder → magnitude/phase → ring-buffer iSTFT.
- **Reference**: `train/sonata/codec_12hz.py` for exact architecture.

#### 6. Semantic EOU Training Script

- **Files**: new `train/sonata/train_semantic_eou.py`
- **Task**: Training script for byte-level LSTM sentence completion classifier. Architecture matches `src/semantic_eou.c`: vocab=256 (byte-level), embed_dim=64, hidden_dim=128, 1-layer LSTM, sigmoid output. Export to `.seou` binary format.
- **Data**: Generate from LibriSpeech transcripts — complete sentences = positive, truncated at random word boundaries = negative.
- **Reference**: `src/semantic_eou.c` for exact binary format (magic number, dimensions, weight layout).

#### 7. MLX Backend for Sonata LM

- **Files**: new `src/sonata_lm_mlx/` crate or feature flag in `src/sonata_lm/`
- **Task**: MLX-based inference backend. Since Rust MLX bindings are immature, practical approach is: Swift MLX wrapper exposing C FFI, or Python MLX with ctypes bridge. Evaluate feasibility first — if bindings are too rough, produce a design doc with concrete implementation plan instead.
- **Fallback**: If MLX Rust/Swift bindings aren't ready, document the exact API surface needed and create a tracking issue.

#### 8. Moshi Unified Speech-to-Speech Design

- **Files**: new `docs/plans/2026-03-05-moshi-s2s-design.md`
- **Task**: Design document for unified speech-to-speech model. Research Moshi architecture (Kyutai), analyze how it maps to Sonata's existing pipeline, propose incremental migration path. No code — design only.
- **Deliverable**: Architecture doc with component diagram, data flow, training plan, and phased implementation roadmap.

### Tier 2: Training Runs (queue after Tier 1)

| Job               | Script                  | GPU-Days | Blocked By          |
| ----------------- | ----------------------- | -------- | ------------------- |
| Flow distillation | `train_distill_v3.py`   | 3-4      | Flow v3 completion  |
| ReDrafter         | `train_drafter.py`      | 1-2      | —                   |
| Denoiser          | `train_denoiser.py`     | 1-2      | DEMAND dataset      |
| Semantic EOU      | `train_semantic_eou.py` | 2-3      | Tier 1 workstream 6 |
| 12.5Hz codec      | `codec_12hz.py`         | 4-6      | —                   |

### Tier 3: Post-Training Integration

After models are trained, wire weights into the inference code built in Tier 1.

## Agent Assignment

Each agent gets an isolated worktree. No shared file modifications.

## Audit Plan

After Tier 1 completes, run compound audit per CLAUDE.md:

- 8 agents across 8 workstreams = 6-8 auditors
- Roles: correctness-prover, e2e-tracer, gap-hunter, red-team, perf-validator, assumption-breaker
- Synthesis task aggregates P0/P1/P2/P3 findings
