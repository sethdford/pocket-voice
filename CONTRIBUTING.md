# Contributing to Sonata

Thank you for your interest in contributing to Sonata! This guide covers everything you need to get started.

## Code of Conduct

Be respectful and constructive. We welcome contributors of all backgrounds and experience levels. Focus discussions on technical merit and project goals.

## Development Setup

### Prerequisites

- **macOS** on Apple Silicon (M1/M2/M3/M4)
- **Xcode Command Line Tools**: `xcode-select --install`
- **Rust** (stable): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Homebrew dependencies**:

```bash
brew install curl opus onnxruntime espeak-ng
```

### Build

```bash
# Build everything (C shared libraries + Rust cdylibs + binary)
make

# Run the pipeline
ANTHROPIC_API_KEY=sk-ant-... ./sonata
```

The build has three stages:

1. **C shared libraries** — 43+ dylibs in `build/`, each `.c` compiles independently
2. **Rust cdylibs** — STT, LLM, Sonata LM/Flow/Storm via `cargo build --release`
3. **Pipeline binary** — links all of the above

### Run Tests

```bash
# Run the full test suite (~30 test targets)
make test

# Run a specific test suite
make test-pipeline
make test-sonata
make test-eou
```

All tests must pass before submitting a pull request.

## Pull Request Process

1. **Fork and branch** — Create a feature branch from `main`
2. **Make focused changes** — Keep PRs small and single-purpose
3. **Test thoroughly** — Run `make test` and verify no regressions
4. **Write clear commit messages** — Explain _why_, not just _what_
5. **Open a PR** — Describe the change and link any related issues

### PR Checklist

- [ ] Code compiles cleanly with `make`
- [ ] All tests pass with `make test`
- [ ] New functionality includes tests
- [ ] No new compiler warnings
- [ ] Documentation updated if behavior changes

## Code Style

### C Code

- **K&R-style** bracing with 4-space indentation
- Function names: `snake_case` (e.g., `fused_eou_feed`)
- Struct names: `PascalCase` (e.g., `AudioPostProcessor`)
- Constants: `UPPER_SNAKE_CASE`
- All Accelerate includes must define `ACCELERATE_NEW_LAPACK`
- No allocations in real-time audio paths (CoreAudio callback)
- Use `cblas_sgemv`/`vDSP_*` for vectorized math (AMX-accelerated)

### Rust Code

- Format with `cargo fmt`
- Lint with `cargo clippy`
- All public FFI functions use `#[no_mangle] extern "C"` (or `#[unsafe(no_mangle)]` in edition 2024)
- Expose C-ABI dynamic libraries via `crate-type = ["cdylib"]`

## Architecture Overview

Sonata is a real-time voice pipeline: Mic → STT → LLM → TTS → Speaker. All components run natively on Apple Silicon using three concurrent hardware units:

| Unit                | Role                                    |
| ------------------- | --------------------------------------- |
| **Metal GPU**       | Transformer inference (STT, TTS)        |
| **AMX Coprocessor** | FFT, prosody, LSTM, mel features        |
| **ARM NEON**        | PCM conversion, crossfade, ring buffers |

For a full deep-dive, see [docs/architecture.md](docs/architecture.md).

## Adding a New C Library

1. Create `src/my_lib.c` and `src/my_lib.h`
2. Add a build target in the `Makefile`:
   ```makefile
   $(BUILD)/libmy_lib.dylib: src/my_lib.c | $(BUILD)
       $(CC) $(CFLAGS) -shared -fPIC -framework Accelerate \
         -install_name @rpath/libmy_lib.dylib -o $@ $<
   ```
3. Add to the `libs:` dependency list
4. Add FFI declarations in `pocket_voice_pipeline.c`
5. Link via `-lmy_lib` in the pipeline binary target

## Adding Tests

1. Create `tests/test_my_feature.c`
2. Add a Make target:
   ```makefile
   test-my-feature: tests/test_my_feature.c $(BUILD)/libmy_lib.dylib | $(BUILD)
       $(CC) $(CFLAGS) -Isrc -L$(BUILD) -lmy_lib \
         -Wl,-rpath,@executable_path -o $(BUILD)/test-my-feature tests/test_my_feature.c
       ./$(BUILD)/test-my-feature
   ```
3. Add to `.PHONY` and to the `test:` dependencies

**Note**: Test binaries go in `build/`, so rpath is `@executable_path` (co-located with dylibs). The main binary lives in the project root and uses `@executable_path/build`.

## Questions?

Open an issue on GitHub. We're happy to help you get started.
