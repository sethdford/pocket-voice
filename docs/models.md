# Model Download & Setup

Sonata requires several model files to run. This guide covers where to obtain them and where to place them.

## Directory Structure

All models should be placed in a `models/` directory at the project root:

```
models/
├── sonata/
│   ├── sonata_lm.safetensors          # Sonata LM weights (241M params)
│   ├── sonata_lm_config.json          # Sonata LM config
│   ├── sonata_flow.safetensors        # Sonata Flow weights (35.7M params)
│   ├── sonata_flow_config.json        # Sonata Flow config
│   ├── sonata_decoder.safetensors     # ConvNeXt/ConvDecoder weights
│   ├── sonata_decoder_config.json     # Decoder config
│   ├── sonata_stt.cstt_sonata         # Sonata STT (CTC, ~24.8MB)
│   ├── sonata_refiner.cref            # Sonata Refiner (~188MB)
│   ├── phoneme_map.json               # Phoneme-to-ID mapping
│   ├── emosteer_directions.json       # EmoSteer emotion vectors (optional)
│   ├── flow_v3.safetensors            # Flow v3 weights (optional)
│   ├── flow_v3_config.json            # Flow v3 config (optional)
│   ├── vocoder.safetensors            # BigVGAN-lite vocoder (optional)
│   └── vocoder_config.json            # Vocoder config (optional)
├── parakeet-ctc-1.1b-fp16.cstt        # Conformer STT 1.1B (alternative)
├── 3-gram.pruned.1e-7.bin             # KenLM language model (optional)
├── silero_vad.nvad                     # Native VAD weights (~1.2MB)
├── silero_vad.onnx                     # Silero VAD ONNX (for extraction)
└── ecapa_tdnn.onnx                     # Speaker encoder for voice cloning
```

## Required Models

### Sonata TTS (default engine)

| Model          | Params | Format         | Description                                                |
| -------------- | ------ | -------------- | ---------------------------------------------------------- |
| Sonata LM      | 241.7M | `.safetensors` | Llama-style transformer, predicts semantic tokens at 50 Hz |
| Sonata Flow    | 35.7M  | `.safetensors` | Conditional flow matching, generates mag+phase             |
| Sonata Decoder | 16.8M  | `.safetensors` | ConvNeXt or ConvDecoder (mag+phase → audio)                |

These are loaded automatically when using `--tts-engine sonata` (the default).

### STT Engine

You need one of the following:

| Model             | Params | Format         | Description                                     |
| ----------------- | ------ | -------------- | ----------------------------------------------- |
| Sonata STT (CTC)  | 6.2M   | `.cstt_sonata` | Lightweight CTC Conformer, reuses codec encoder |
| Sonata Refiner    | 49.3M  | `.cref`        | Optional second pass for higher accuracy        |
| Parakeet CTC 1.1B | 1.1B   | `.cstt`        | NVIDIA FastConformer, highest accuracy          |

### Voice Activity Detection

| Model      | Size   | Format  | Description                                            |
| ---------- | ------ | ------- | ------------------------------------------------------ |
| Native VAD | ~1.2MB | `.nvad` | Pure C reimplementation of Silero VAD, AMX-accelerated |

Extract from the Silero ONNX model:

```bash
python scripts/extract_silero_weights.py models/silero_vad.onnx models/silero_vad.nvad
```

## Optional Models

| Model               | Format         | Description                                                |
| ------------------- | -------------- | ---------------------------------------------------------- |
| ECAPA-TDNN          | `.onnx`        | Speaker encoder for zero-shot voice cloning                |
| KenLM 3-gram        | `.bin`         | Language model for CTC beam search rescoring               |
| EmoSteer directions | `.json`        | Emotion steering vectors for training-free emotion control |
| Flow v3 + Vocoder   | `.safetensors` | Alternative TTS pipeline (text → mel → waveform)           |

## Downloading Models

### From HuggingFace

Sonata models can be downloaded using the HuggingFace CLI:

```bash
pip install huggingface-hub

# Download Sonata model files
huggingface-cli download <repo-id> --local-dir models/sonata/
```

The Rust crates support auto-download: if you pass a HuggingFace repo ID as the weights path, models are fetched automatically via the `hf-hub` crate.

### KenLM Language Model

For STT beam search with language model rescoring:

```bash
# Download a pruned 3-gram model (LibriSpeech)
huggingface-cli download openslr/librispeech_lm --include "3-gram.pruned.1e-7.bin" --local-dir models/
```

### Silero VAD

```bash
# Download the ONNX model, then extract native weights
pip install onnxruntime
python -c "
import urllib.request
urllib.request.urlretrieve(
    'https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx',
    'models/silero_vad.onnx'
)
"
python scripts/extract_silero_weights.py models/silero_vad.onnx models/silero_vad.nvad
```

## Configuration

Models can be specified via CLI flags or JSON config:

### CLI

```bash
# Sonata TTS + Sonata STT
./sonata \
  --sonata-lm-weights models/sonata/sonata_lm.safetensors \
  --sonata-lm-config models/sonata/sonata_lm_config.json \
  --sonata-flow-weights models/sonata/sonata_flow.safetensors \
  --sonata-flow-config models/sonata/sonata_flow_config.json \
  --stt-engine sonata \
  --sonata-stt-model models/sonata/sonata_stt.cstt_sonata

# Conformer STT with beam search
./sonata \
  --stt-engine conformer \
  --cstt-model models/parakeet-ctc-1.1b-fp16.cstt \
  --beam-size 16 --lm-path models/3-gram.pruned.1e-7.bin

# Voice cloning
./sonata --clone-voice reference_voice.wav
```

### JSON Config

```json
{
  "stt": {
    "engine": "sonata",
    "sonata_model": "models/sonata/sonata_stt.cstt_sonata",
    "sonata_refiner": "models/sonata/sonata_refiner.cref"
  },
  "tts": {
    "engine": "sonata"
  },
  "sonata": {
    "lm_weights": "models/sonata/sonata_lm.safetensors",
    "lm_config": "models/sonata/sonata_lm_config.json",
    "flow_weights": "models/sonata/sonata_flow.safetensors",
    "flow_config": "models/sonata/sonata_flow_config.json"
  },
  "audio": {
    "vad": "models/silero_vad.nvad"
  }
}
```

## Hardware Requirements

All inference runs on Apple Silicon. GPU memory is shared with system RAM.

| Model                    | VRAM (approx) | Notes                 |
| ------------------------ | ------------- | --------------------- |
| Sonata LM (FP16)         | ~500MB        | Metal GPU, ~45 tok/s  |
| Sonata Flow + Decoder    | ~150MB        | Metal GPU             |
| Sonata STT (CTC)         | ~50MB         | Metal GPU             |
| Conformer 1.1B           | ~2.2GB        | Metal GPU             |
| Local LLM (Llama 3.2 3B) | ~6GB          | Metal GPU, optional   |
| Native VAD               | ~1.2MB        | CPU (AMX-accelerated) |

Minimum recommended: 16GB unified memory (M1/M2/M3 with 16GB).
