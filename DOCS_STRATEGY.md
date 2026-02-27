# Sonata Documentation Strategy

> Synthesized from deep audits by 3 specialist agents across README.md (564 lines), docs/index.html (2503 lines), AGENTS.md (1513 lines), bench_output/, scripts/, and Rust crates.

---

## Executive Summary

Sonata has world-class engineering with documentation that undersells it. The core problems:

1. **Brand split** — "pocket-voice" owns the identity positions (binary name, architecture diagram, meta tags, nav logo, prose subjects) while "Sonata" is relegated to component names
2. **No on-ramp** — Requirements/Build/Run is buried at line 345 of the README; no model download guide exists at all
3. **Hidden strengths** — 1.86% WER, 11x realtime STT, OpenAI-compatible API, 5373x realtime codec — all buried in dev folders nobody will find
4. **Missing fundamentals** — No CONTRIBUTING.md, CHANGELOG, API reference, getting started tutorial, or examples directory

The strategy below is organized into 3 phases: **P0 (Rebrand & Unblock)**, **P1 (Flagship Polish)**, **P2 (Ecosystem Completeness)**.

---

## Phase 0: Rebrand & Unblock (Critical Path)

These changes are required before any public push. Without them, the project presents as "pocket-voice" — a confusing identity for a Sonata flagship launch.

### 0.1 — Rename binary from `pocket-voice` to `sonata`

**Impact:** Eliminates the single most pervasive brand conflict. Every CLI example, every `./pocket-voice` invocation (9 in README, 2 in docs site, ~20 in AGENTS.md) reinforces the old name.

**Scope:**

- Makefile: change binary target name
- README.md: all `./pocket-voice` → `./sonata`
- docs/index.html: terminal examples
- AGENTS.md: all CLI references
- Consider: symlink `pocket-voice` → `sonata` for backwards compat

### 0.2 — Fix README brand identity (3 prose sentences + diagram)

| Line | Current                                    | New                                  |
| ---- | ------------------------------------------ | ------------------------------------ |
| 59   | `pocket-voice binary`                      | `sonata`                             |
| 236  | "pocket-voice runs on all four..."         | "Sonata runs on all four..."         |
| 261  | "pocket-voice includes a comprehensive..." | "Sonata includes a comprehensive..." |
| 370  | `pocket-voice — Pipeline binary`           | `sonata — Pipeline binary`           |

### 0.3 — Fix GitHub Pages meta tags and nav

| Element               | Current                                        | New                                                                                           |
| --------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `<title>`             | `pocket-voice — Zero-Python Voice Pipeline...` | `Sonata — Real-time Voice AI for Apple Silicon`                                               |
| `og:title`            | `pocket-voice`                                 | `Sonata`                                                                                      |
| `og:description`      | "Zero-Python real-time voice pipeline..."      | "Real-time voice AI for Apple Silicon. Native C + Rust. Sub-200ms latency. Zero Python."      |
| `twitter:title`       | `pocket-voice`                                 | `Sonata`                                                                                      |
| `twitter:description` | "Mic → STT → Claude → TTS → Speaker..."        | "Native voice AI pipeline for Apple Silicon. Mic → STT → LLM → TTS → Speaker in under 200ms." |
| Nav logo              | `pocket-voice`                                 | `Sonata`                                                                                      |
| Nav aria-label        | `"pocket-voice home"`                          | `"Sonata home"`                                                                               |
| **ADD**               | —                                              | `og:image` + `twitter:image` (social share card)                                              |
| **ADD**               | —                                              | `<link rel="canonical">`                                                                      |

### 0.4 — Create model download guide

**This is the #1 blocker for new users.** Zero documentation exists on where to obtain required model files. Without models, the pipeline can't run.

Create `docs/models.md` or a "Models" section in README covering:

- Which models are required vs optional
- HuggingFace repo links for each
- Download commands (`huggingface-cli download ...`)
- Expected file locations (`models/` directory)
- Model sizes and hardware requirements

---

## Phase 1: Flagship Polish (README Restructure + Site Update)

### 1.1 — README restructure

**New section order** (current order in parentheses):

```
1. H1 + Badges + Tagline                    (was: H1 + tagline only)
2. Quick Start (3 commands)                  (was: buried at line 345)
3. "Why Sonata?" — 3-sentence elevator pitch (NEW)
4. Demo GIF / audio sample embed             (NEW)
5. Key Features (top 6)                      (was: 17 bullets at line 35)
6. Sonata TTS Architecture                   (was: line 11 — moved after features)
7. Architecture diagram + state machine      (was: line 55)
8. Performance benchmarks table              (NEW — surface hidden data)
9. Competitive comparison table              (NEW)
10. Native Libraries (keep tables)           (was: line 114)
11. Optimizations                            (was: line 232)
12. Quality Assurance                        (was: line 259)
13. Test Suite                               (was: line 291)
14. Full Feature List (remaining 11 bullets) (NEW — expanded from condensed #5)
15. Configuration & Options                  (was: line 382)
16. Project Structure (condensed)            (was: line 453 — remove file-by-file tree)
17. Contributing                             (NEW — link to CONTRIBUTING.md)
18. License                                  (was: line 561)
```

**Key changes explained:**

- **Quick Start moves to position 2**: A developer should know how to install and run within 10 seconds of landing. Format:

  ```
  brew install curl opus onnxruntime espeak-ng
  make
  ANTHROPIC_API_KEY=sk-... ./sonata
  ```

- **"Why Sonata?" paragraph**: Plain-English, non-technical. Something like: _"Most voice pipelines are Python glue connecting cloud APIs. Sonata is different — every component from speech recognition to text-to-speech runs natively on Apple Silicon, using all four hardware compute units simultaneously. The result: sub-200ms end-to-end latency with full-duplex conversation, entirely on your Mac."_

- **Features condensed to top 6**: 100% native, Full-duplex barge-in, Fused 3-signal EOU, Speculative prefill, Sonata TTS (294M params), Apple Silicon 4-unit dispatch. The other 11 go in "Full Feature List" later.

- **Performance benchmarks table** (NEW — surface hidden data):
  | Metric | Value | Source |
  |--------|-------|--------|
  | STT WER (LibriSpeech test-clean) | 1.86% | bench_output/BENCHMARK_REPORT.md |
  | STT Real-Time Factor | 11x | bench_output/BENCHMARK_REPORT.md |
  | Sonata Codec throughput | 5,373x realtime | README (existing) |
  | TTS intelligibility | 100% | bench_output/TTS_BENCHMARK_REPORT.md |
  | TTS Real-Time Factor | 67x | bench_output/TTS_BENCHMARK_REPORT.md |
  | End-to-end latency | <200ms | README (existing) |

- **Competitive comparison table** (NEW):
  | | Sonata | Whisper + XTTS | OpenAI Realtime | Moshi |
  |---|---|---|---|---|
  | Runs on-device | Yes | Partial | No (cloud) | Yes |
  | Language | C + Rust | Python | API | Python + Rust |
  | Latency | <200ms | 500ms+ | ~300ms | ~200ms |
  | Full-duplex | Yes | No | Yes | Yes |
  | Apple Silicon native | Yes (4 units) | No | N/A | No |

- **Project Structure condensed**: Remove the 100-line file-by-file annotated tree. Replace with top-level directory overview (10 lines). The Libraries tables already document every file.

### 1.2 — Badges

Add immediately under H1:

```markdown
[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)]()
[![Language](https://img.shields.io/badge/C%20%2B%20Rust-native-orange)]()
```

### 1.3 — Demo content

**Option A (preferred):** Record a terminal GIF showing: launch → speak → transcription appears → Claude responds → TTS plays back. Use `asciinema` or `vhs` for high quality.

**Option B:** Embed audio samples. Before/after comparison of Sonata TTS output. Can use `<audio>` tags on the GitHub Pages site and link from README.

**Option C (minimum viable):** Screenshot of the web dashboard showing a live conversation with latency metrics visible.

### 1.4 — GitHub Pages site updates

Beyond the P0 meta tag fixes:

1. **Hero section**: Add "Sonata" to eyebrow text and H1. Suggested: Eyebrow: `"Sonata · Native Voice AI"`, H1: `"Voice intelligence, built from scratch."`
2. **Add Sonata TTS feature strip**: Dedicated visual section showcasing the 3-stage pipeline (LM → Flow → iSTFT), parameter counts, and the "why it's different" narrative
3. **Stats section**: Add Sonata-specific stat: "294M TTS Parameters" or "3 Sonata Models"
4. **Add demo/audio section**: Between features and silicon sections
5. **Footer**: Replace or supplement the Steve Jobs quote with Sonata branding. Add copyright line mentioning Sonata.

### 1.5 — Hero section copy direction

The current site hero is:

> "100% native C and Rust on Apple Silicon"
> "Voice that thinks different."

**Proposed direction** (preserve Apple-level polish, add Sonata identity):

> "Sonata · Real-time Voice AI"
> "Intelligence at the speed of speech."
> "A native C + Rust voice pipeline for Apple Silicon. Mic to speaker in under 200ms. No cloud required."

---

## Phase 2: Ecosystem Completeness

### 2.1 — CONTRIBUTING.md

Standard sections: Code of Conduct, Development Setup, Pull Request Process, Code Style (C: K&R-ish, Rust: cargo fmt), Testing Requirements (must pass `make test`), Architecture Overview (link to docs).

### 2.2 — API Reference

The HTTP API is OpenAI-compatible — this is a major differentiator that's completely undocumented publicly. Create `docs/api.md`:

- Endpoint list (from http_api.c)
- Request/response examples
- WebSocket protocol spec
- Audio encoding options (PCM, WAV, mu-law, A-law)
- OpenAI compatibility notes

### 2.3 — Architecture deep-dive

Extract from AGENTS.md into `docs/architecture.md`:

- Pipeline state machine (with Mermaid diagram)
- Audio flow (capture → processing → playback)
- EOU detection system (3-signal fusion explanation)
- Sonata TTS pipeline (LM → Flow → iSTFT deep-dive)
- Hardware dispatch strategy (GPU + AMX + ANE + NEON)
- Memory architecture (arenas, ring buffers, zero-alloc hot path)

### 2.4 — CHANGELOG.md

Start with current state as v0.1.0. Establish semantic versioning going forward. Reference the git history for key milestones.

### 2.5 — Troubleshooting guide

Extract from AGENTS.md "gotchas" section into `docs/troubleshooting.md`:

- Common build failures
- Model file issues
- macOS version requirements
- Homebrew dependency conflicts
- CoreAudio permissions

### 2.6 — Examples directory

Create `examples/` with:

- `curl_tts.sh` — curl command to synthesize speech via REST API
- `websocket_client.py` — Python WebSocket streaming example
- `web_remote_demo.html` — Browser mic → pipeline → speaker demo

### 2.7 — Rust crate metadata

Update all 6 Cargo.toml files with:

- `description`, `repository`, `documentation`, `readme`, `license`, `keywords`, `categories`
- Add `//!` module-level docs to each `src/lib.rs`

### 2.8 — Scripts documentation

Create `scripts/README.md` documenting all 38 scripts with one-line descriptions grouped by function (benchmarking, model conversion, export, validation).

---

## Brand Voice Guidelines

All documentation should follow these principles:

1. **Lead with "Sonata"** — never "pocket-voice" in prose. The binary can remain as `sonata` (renamed). Historical references to pocket-voice should be phased out.
2. **Technical but accessible** — the first paragraph of any section should be readable by a senior developer who doesn't know ML. Deeper technical detail follows.
3. **Show, don't tell** — performance claims should always include numbers. "Fast" → "67x realtime." "Low latency" → "<200ms P95."
4. **Apple-grade polish** — the GitHub Pages site sets the tone. Docs should match that quality bar.
5. **Competitive confidence** — don't trash alternatives, but clearly articulate why native C + Rust on Apple Silicon is a different approach with concrete benefits.

---

## Implementation Priority Summary

| Priority | Task                                          | Effort | Impact                           |
| -------- | --------------------------------------------- | ------ | -------------------------------- |
| **P0**   | Rename binary to `sonata`                     | Medium | Critical — brand coherence       |
| **P0**   | Fix README prose (3 sentences + diagram)      | Tiny   | Critical — brand coherence       |
| **P0**   | Fix site meta tags + nav                      | Small  | Critical — SEO + social shares   |
| **P0**   | Create model download guide                   | Medium | Critical — unblocks new users    |
| **P1**   | Restructure README (new section order)        | Large  | High — first impression          |
| **P1**   | Add badges to README                          | Tiny   | High — trust signals             |
| **P1**   | Add Quick Start section                       | Small  | High — developer conversion      |
| **P1**   | Add performance benchmarks table              | Small  | High — surface hidden strengths  |
| **P1**   | Add competitive comparison table              | Small  | High — positioning               |
| **P1**   | Create demo content (GIF/audio)               | Medium | High — proof of quality          |
| **P1**   | Update site hero + add Sonata TTS section     | Medium | High — brand story               |
| **P1**   | Add og:image for social shares                | Small  | High — social presence           |
| **P2**   | CONTRIBUTING.md                               | Small  | Medium                           |
| **P2**   | API Reference (docs/api.md)                   | Medium | Medium — surfaces hidden feature |
| **P2**   | Architecture deep-dive (docs/architecture.md) | Large  | Medium                           |
| **P2**   | CHANGELOG.md                                  | Small  | Medium                           |
| **P2**   | Troubleshooting guide                         | Small  | Medium                           |
| **P2**   | Examples directory                            | Medium | Medium                           |
| **P2**   | Rust crate metadata                           | Small  | Low                              |
| **P2**   | Scripts README                                | Small  | Low                              |

---

## Execution Recommendation

This strategy can be executed in **3 parallel workstreams**:

1. **Binary rename + brand fix agent** — Makefile, README prose, AGENTS.md, docs site meta tags (P0.1–P0.3)
2. **README restructure agent** — New section order, Quick Start, Why Sonata, badges, benchmarks table, competitive table, condense Project Structure (P1.1–P1.2, P1.5)
3. **Docs creation agent** — Model guide, CONTRIBUTING.md, API reference, and site hero/TTS section updates (P0.4, P1.4, P2.1–P2.2)

Each workstream touches different files and can run in parallel worktrees without conflicts.
