# Troubleshooting

Common issues and their solutions when building and running Sonata.

## Build Failures

### Homebrew curl not found

**Symptom**: Linker errors referencing curl symbols, or `brew --prefix curl` fails.

**Fix**: Install Homebrew curl and ensure it's on your path:

```bash
brew install curl
```

The Makefile auto-detects `brew --prefix curl` for the correct include and library paths. macOS system curl may lack required features.

### ONNX Runtime headers not found

**Symptom**: `#include <onnxruntime/onnxruntime_c_api.h>` fails.

**Fix**: Install via Homebrew:

```bash
brew install onnxruntime
```

Headers are at `/opt/homebrew/include/onnxruntime/`. Ensure your CFLAGS include `-I/opt/homebrew/include`.

### espeak-ng not found

**Symptom**: Linker errors for `espeak_TextToPhonemes` or missing `espeak-ng/speak_lib.h`.

**Fix**:

```bash
brew install espeak-ng
```

Headers: `/opt/homebrew/include/espeak-ng/`, library: `/opt/homebrew/lib/`.

### Rust build fails with Metal errors

**Symptom**: `candle` crate fails to compile Metal shaders.

**Fix**: Ensure Xcode Command Line Tools are installed and up to date:

```bash
xcode-select --install
# Or update:
softwareupdate --install -a
```

The Rust crates pin specific `candle` versions. Updating dependencies may break Metal kernel compatibility.

### ARM64 architecture errors

**Symptom**: Compilation errors about wrong architecture.

**Note**: Sonata is ARM64-only. All compilation targets `-arch arm64`. There is no x86/Rosetta support. You must be running natively on Apple Silicon.

### ACCELERATE_NEW_LAPACK warnings

**Symptom**: Deprecation warnings from Accelerate framework includes.

**Fix**: All files that include `<Accelerate/Accelerate.h>` should define:

```c
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
```

This is normally handled by the Makefile CFLAGS.

## Model Issues

### "Model file not found"

**Symptom**: Pipeline fails to start, error about missing model files.

**Fix**: Ensure models are in the expected locations. See [docs/models.md](models.md) for download instructions and expected paths.

### Native VAD extraction fails

**Symptom**: `extract_silero_weights.py` errors.

**Fix**: Ensure ONNX Runtime is installed in your Python environment:

```bash
pip install onnxruntime
python scripts/extract_silero_weights.py models/silero_vad.onnx models/silero_vad.nvad
```

### CoreML EP not available

**Symptom**: ONNX models fall back to CPU instead of Apple Neural Engine.

**Fix**: The Makefile auto-detects Python ONNX Runtime with CoreML EP. Install it:

```bash
pip install onnxruntime
```

The Python venv ORT is searched at `.venv/lib/python*/site-packages/onnxruntime`. If not found, Homebrew ORT (CPU-only) is used as fallback.

## macOS Issues

### CoreAudio permissions denied

**Symptom**: Pipeline starts but no audio input/output.

**Fix**: Grant microphone permission to your terminal application:

1. Open **System Settings** → **Privacy & Security** → **Microphone**
2. Enable access for your terminal (Terminal.app, iTerm2, etc.)

If running from an IDE, the IDE itself may need microphone permission.

### "VoiceProcessingIO" errors

**Symptom**: AudioUnit creation fails.

**Fix**: This usually indicates another application has exclusive access to the audio device. Close other audio applications (Zoom, FaceTime, etc.) and try again.

### macOS version requirements

Sonata requires macOS 13 (Ventura) or later for full Apple Silicon feature support. Some features (Metal 3, BNNS updates) may require macOS 14 (Sonoma).

## Runtime Issues

### High latency on first utterance

**Symptom**: First voice interaction is noticeably slower than subsequent ones.

**Explanation**: Metal shader compilation happens on first inference. Sonata mitigates this with a GPU warmup pass at load time for both Sonata LM and Flow, but some minor first-run overhead may remain.

### STT transcription quality is poor

**Try these steps**:

1. **Check your microphone**: Ensure you're using a good quality mic, not a laptop's built-in mic in a noisy room
2. **Enable noise gate**: It's on by default in the pipeline's STT path
3. **Use beam search**: Add `--beam-size 16 --lm-path models/3-gram.pruned.1e-7.bin` for language model rescoring
4. **Try Conformer 1.1B**: The larger model (`parakeet-ctc-1.1b-fp16.cstt`) provides significantly better accuracy than the lightweight Sonata STT

### TTS produces noise-like audio

**Explanation**: If you're using untrained Sonata model weights, the TTS output will be noise. The inference pipeline is verified correct — the models require training on speech data to produce intelligible audio. See the training pipeline in `train/sonata/`.

### "Barge-in" too sensitive / not sensitive enough

Sonata includes emotion-aware barge-in: empathetic content (sad, warm, calm) raises the energy threshold by 40%, while excited/angry content lowers it by 20%.

To adjust manually, modify the VAD thresholds via the `SpeechDetectorConfig`:

- Higher threshold = harder to interrupt
- Lower threshold = easier to interrupt

### Pipeline hangs during LLM response

**Check**:

1. API key is valid (`ANTHROPIC_API_KEY` or `GEMINI_API_KEY`)
2. Network connectivity to the API endpoint
3. Try `--llm local` for on-device inference (no network required)

## Homebrew Dependency Summary

```bash
# All required dependencies
brew install curl opus onnxruntime espeak-ng

# Optional (for MP3 output)
brew install lame
```

## Getting Help

If you encounter an issue not covered here:

1. Check that all dependencies are installed: `brew list | grep -E "curl|opus|onnxruntime|espeak"`
2. Run `make clean && make` for a fresh build
3. Run `make test` to verify the build is working
4. Open an issue on GitHub with your macOS version, chip (M1/M2/M3/M4), and the full error output
