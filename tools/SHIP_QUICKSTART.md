# Ship Readiness — Quick Start

**TL;DR**: Run one command to validate all tests, build, models, and security. Then verify manual checklist items.

## One-Command Validation

```bash
cd /Users/sethford/Documents/pocket-voice
./tools/ship_readiness.sh
```

This automates:

- ✓ Rust workspace tests (`cargo test --workspace`)
- ✓ C test suite (`make test` — 56+ targets)
- ✓ Python tests (RVQ, codec, duplex)
- ✓ Build artifacts (dylibs + checksums)
- ✓ Model files (distilled flow + configs)
- ✓ Security audit (P0 status)
- ✓ Undefined symbols (nm check)

**Expected output on GO:**

```
═══════════════════════════════════════════════════════════
                  🚀 GO FOR SHIP 🚀
═══════════════════════════════════════════════════════════
```

Exit code: `0` = GO, `1` = NO-GO

---

## If Script Fails: Debugging

### Rust tests fail

```bash
cd /Users/sethford/Documents/pocket-voice
cargo test --workspace 2>&1 | head -50
```

Check `.claude/agent-memory/` for captured failure patterns.

### C tests fail

```bash
cd /Users/sethford/Documents/pocket-voice
make test 2>&1 | tee /tmp/make_test.log
grep "FAILED" /tmp/make_test.log
```

### Models not found

```bash
ls -la /Users/sethford/Documents/pocket-voice/models/sonata_flow_distilled/
```

If missing, download from GCS or generate locally.

### Dylibs not found

```bash
cargo build --release --workspace
# or
make
```

---

## Manual Verification Checklist

After automated checks pass, verify:

### 1. Audio Quality (5 min)

- [ ] Listen to TTS samples from 3+ speakers (in `experiments/samples/`)
- [ ] Check for naturalness, artifacts, intelligibility
- [ ] Confirm STT accuracy with `make bench-sonata` output

### 2. Voice Cloning (10 min)

- [ ] Run `experiments/personaplex/test_duplex.py` (included in auto tests)
- [ ] Verify speaker encoder loads and produces embeddings
- [ ] Check SIM-O scores (target ≥0.75)

### 3. E2E Latency (5 min)

- [ ] Measure mic→STT→LLM→TTS→speaker end-to-end
- [ ] Target: <1000ms
- [ ] Document in `PERFORMANCE_AUDIT.txt`

### 4. Security Review (10 min)

- [ ] Review `.claude/SECURITY_AUDIT.md` — must be P0-free
- [ ] Review `.claude/CORRECTNESS_AUDIT_REPORT.md`
- [ ] Review `.claude/RED_TEAM_REPORT.md`

### 5. Memory & CPU (5 min)

- [ ] Run with Activity Monitor open
- [ ] Check: Memory <2GB, CPU <4 cores, GPU active
- [ ] No memory leaks (run for 60s continuous)

### 6. Error Handling (5 min)

- [ ] Kill network → graceful LLM retry
- [ ] Invalid audio file → clear error message
- [ ] Missing model → informative error

---

## Performance Targets

| Metric              | Target | Current | Pass? |
| ------------------- | ------ | ------- | ----- |
| STT WER             | <1%    | 0.9%    | ✓     |
| STT RTF             | <0.1   | 0.075   | ✓     |
| TTS RTF             | <0.25  | 0.196   | ✓     |
| E2E latency         | <1s    | TBD     | ?     |
| Memory              | <2GB   | TBD     | ?     |
| Speculative speedup | >1.5x  | TBD     | ?     |
| Voice clone SIM-O   | >0.75  | TBD     | ?     |

---

## Full Checklist

See `tools/SHIP_CHECKLIST.md` for comprehensive pre-ship verification (7 sections).

---

## Key Files

| File                     | Purpose                                     |
| ------------------------ | ------------------------------------------- |
| `ship_readiness.sh`      | Automated test runner (this file executes)  |
| `SHIP_CHECKLIST.md`      | Complete pre-ship verification (7 sections) |
| `.claude/` audit reports | Security, correctness, red team, gap hunter |
| `models/` directory      | Distilled flow + RNN drafter + speaker enc. |
| `target/release/`        | Compiled dylibs (libsonata\_\*.dylib)       |

---

## Timeline

- **Week 1**: Fix any P0 issues found in auto tests
- **Week 2**: Verify manual checklist (audio quality, latency, security)
- **Week 3**: Get sign-off from team lead
- **Week 4**: Deploy

---

## Contact

If stuck, check:

1. `.claude/ASSUMPTION_BREAKER_SUMMARY.txt` (implicit assumptions that may be wrong)
2. `.claude/agent-memory/` (failure patterns from prior builds)
3. `TROUBLESHOOTING.md` (common issues)

Ready to ship? ✓ Run the script. ✓ Verify checklist. 🚀
