# Sonata — Project Instructions

## Compound Engineering via Negative Critical Audit Loops

Every non-trivial task MUST conclude with a dynamically-sized audit team that applies compound engineering through negative critical feedback loops. This is not optional — it is the core engineering methodology for this project.

### The Principle

After every implementation phase, create a team of audit specialists that relentlessly ask:

- What did we get wrong?
- What did we miss?
- What did we not think through or consider?
- What did we fail to test, audit, research, validate, or prove works end-to-end?

The audit findings feed back into the next implementation cycle, creating a compound quality loop where each iteration catches what the previous one missed.

### Dynamic Team Sizing

Scale the audit team to match the scope of what was built. Do NOT use a fixed team size.

| Change Scope                  | Audit Team Size | Example                         |
| ----------------------------- | --------------- | ------------------------------- |
| 1-3 files, single concern     | 2-3 agents      | Bug fix, small feature          |
| 4-10 files, multiple concerns | 4-6 agents      | New module, security hardening  |
| 10+ files, cross-cutting      | 6-8 agents      | Architecture change, SOTA sweep |
| Full system                   | 8-10 agents     | Major refactor, new subsystem   |

### Required Audit Dimensions

Select from these specialist roles based on what was changed. Not every audit needs all roles — pick the ones relevant to the work:

| Role                   | Asks                                                                             | When to Include                                       |
| ---------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **correctness-prover** | Does the code produce mathematically/logically correct results? Can we prove it? | Any algorithmic change (DSP, ML, data structures)     |
| **e2e-tracer**         | Does data flow correctly through every module boundary? Are interfaces honored?  | Multi-module changes, pipeline modifications          |
| **gap-hunter**         | What code paths have zero test coverage? What edge cases are untested?           | After any implementation phase                        |
| **red-team**           | Can malformed input crash, hang, or exploit this? What are the attack surfaces?  | Network-facing code, parsers, user input handling     |
| **perf-validator**     | Do our optimizations actually improve performance? What are the real numbers?    | Performance claims, new algorithms, GPU/SIMD work     |
| **debt-collector**     | What dead code, broken abstractions, or inconsistencies did we create or miss?   | After large changes, refactors                        |
| **assumption-breaker** | What implicit assumptions does this code make that might be wrong?               | Complex logic, platform-specific code, numerical code |

### Audit Team Workflow

1. **Analyze scope**: Count files changed, identify concerns touched
2. **Size the team**: Use the table above to pick agent count
3. **Select roles**: Pick the relevant specialist roles for what was changed
4. **Create synthesis task**: Always include a final synthesis task blocked by all audit tasks — it aggregates findings into a prioritized action list
5. **Assign file ownership**: Each agent gets different files to avoid conflicts. Use `--worktree` for isolation when agents need to create test files
6. **Run auditors**: All specialist agents work in parallel
7. **Synthesize**: The synthesis task produces a prioritized finding report (P0/P1/P2/P3)
8. **Act on findings**: P0 items get fixed immediately. P1 items become tasks for the next cycle. P2/P3 are documented

### Audit Output Requirements

Every audit agent MUST produce:

- **Concrete evidence**: Test files, benchmark results, or code analysis — not just opinions
- **Prioritized findings**: P0 (must fix now), P1 (should fix soon), P2 (fix when convenient), P3 (note for later)
- **Reproduction steps**: Every finding must be verifiable by running a test or reading specific code

### The Compound Loop

```
Build → Audit → Fix P0s → Re-audit fixes → Ship
         ↑                        |
         └────────────────────────┘
```

If the audit finds P0 issues, fix them, then re-audit the fixes (with a smaller, focused team). This loop continues until audit produces zero P0 findings. The re-audit team should be 2-3 agents focused specifically on verifying the fixes.

### Example: After Adding a New Module

```
1. Implementation complete: new_module.c (400 LOC), test_new_module.c (200 LOC)
2. Create 4-agent audit team:
   - correctness-prover: Verify algorithm produces correct output
   - gap-hunter: Find untested paths in the new module
   - red-team: Attack the module with malformed input
   - synthesis: Aggregate findings
3. Audit finds: 2 P0 (unchecked malloc, integer overflow), 3 P1 (missing edge cases)
4. Fix P0s immediately
5. Re-audit with 2-agent team (correctness-prover + gap-hunter) on just the fixes
6. Clean audit → commit
```

## Build & Test

- Build: `make` (C) + `cargo build --release` (Rust crates)
- Test all: `make test` (runs 56+ test targets across 61 test files)
- Test specific: `make test-<name>` (e.g., `make test-conformer`, `make test-sonata`)
- Benchmarks: `make bench-sonata`, `make bench-audit`, `make bench`

## Key Conventions

- All audio DSP uses Apple Accelerate (vDSP/BNNS/cblas) — never raw loops for vector math
- Memory: pre-allocate in create/init, zero allocations in hot paths
- Error handling: check every allocation, every file open, every system call
- Security: validate all external input (network, files, user), use snprintf not sprintf, bound all buffers
- Tests: every module gets a test file, every bug fix gets a regression test
