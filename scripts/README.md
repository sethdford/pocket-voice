# Scripts

Utility scripts for benchmarking, model conversion, export, and validation.

## Benchmarking

| Script               | Description                                            |
| -------------------- | ------------------------------------------------------ |
| `benchmark.sh`       | Comprehensive benchmark suite (STT, TTS, VAD, latency) |
| `benchmark_sweep.sh` | Parameter sweep across configurations                  |
| `bench_stt.py`       | STT benchmark runner                                   |
| `bench_tts.py`       | TTS benchmark runner                                   |
| `bench_e2e.py`       | End-to-end pipeline benchmark                          |
| `eval_all.sh`        | Unified evaluation pipeline (WER, quality, latency)    |
| `eval_wer.py`        | WER/CER benchmark on LibriSpeech via C engine          |
| `train_monitor.sh`   | Monitor training runs                                  |

## Model Conversion

| Script                           | Description                                  |
| -------------------------------- | -------------------------------------------- |
| `convert_nemo.py`                | Convert NVIDIA NeMo models to `.cstt` format |
| `convert_nemo_coreml.py`         | Convert NeMo models to CoreML                |
| `convert_nemo_tdt.py`            | Convert NeMo TDT models                      |
| `convert_conformer_to_coreml.py` | Export Conformer to CoreML for ANE           |
| `convert_kyutai_dsm.py`          | Convert Kyutai DSM model                     |
| `convert_tts_weights.py`         | Convert TTS model weights                    |

## Model Export

| Script                        | Description                                      |
| ----------------------------- | ------------------------------------------------ |
| `export_sonata_weights.py`    | Export Sonata model weights to safetensors       |
| `export_sonata_stt.py`        | Export Sonata STT (CTC) to `.cstt_sonata` binary |
| `export_sonata_refiner.py`    | Export Sonata Refiner to `.cref` binary          |
| `export_sonata_gguf.py`       | Export Sonata LM to GGUF (quantized)             |
| `export_flow_v2.py`           | Export Flow v2 to safetensors                    |
| `export_flow_v3.py`           | Export Flow v3 to safetensors                    |
| `export_mimi_lite_weights.py` | Export Mimi Lite weights                         |
| `export_phoneme_map.py`       | Export phoneme vocabulary for C/Rust             |
| `export_trained_models.py`    | Export all trained models for deployment         |
| `extract_silero_weights.py`   | Extract Silero ONNX → `.nvad` for native VAD     |

## Validation & Diagnostics

| Script                         | Description                              |
| ------------------------------ | ---------------------------------------- |
| `validate_model.py`            | Validate model file integrity            |
| `validate_neucodec.py`         | Validate neural codec outputs            |
| `cross_validate_sonata.py`     | Python vs Rust numerical comparison      |
| `cross_validate_phonemizer.py` | Phonemizer cross-validation              |
| `diagnose_sonata.py`           | Diagnose Sonata model issues             |
| `diagnose_stt.py`              | Diagnose STT model issues                |
| `prove_roundtrip.py`           | Prove TTS→STT round-trip intelligibility |
| `test_codec_roundtrip.py`      | Test codec encode/decode fidelity        |

## Text Processing

| Script              | Description                              |
| ------------------- | ---------------------------------------- |
| `normalize_text.py` | Python text normalizer for evaluation    |
| `nemo_reference.py` | NeMo reference implementation comparison |

## Quantization

| Script             | Description                      |
| ------------------ | -------------------------------- |
| `quantize_int8.py` | INT8 quantization for deployment |

## Dependencies

Most scripts require Python 3.10+ with:

```bash
pip install -r scripts/requirements.txt
```

Some scripts additionally require PyTorch, ONNX Runtime, or HuggingFace libraries.
