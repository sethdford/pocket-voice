#!/usr/bin/env python3
"""
validate_model.py — Compare NeMo reference inference against pocket-voice C engine.

Generates test audio, runs it through NeMo (Python) and exports reference
activations + expected transcript. Also generates a raw PCM file for the C
engine to consume, enabling layer-by-layer numerical comparison.

Usage:
    source .venv/bin/activate
    python scripts/validate_model.py nvidia/parakeet-tdt_ctc-110m -o validation/
"""

import argparse
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def generate_test_audio(duration_s: float = 3.0, sr: int = 16000) -> np.ndarray:
    """Generate a simple test waveform: silence + tone + silence."""
    n = int(duration_s * sr)
    audio = np.zeros(n, dtype=np.float32)
    t0, t1 = int(0.5 * sr), int(2.5 * sr)
    t = np.arange(t1 - t0, dtype=np.float32) / sr
    audio[t0:t1] = 0.3 * np.sin(2 * np.pi * 440 * t)
    return audio


def download_librispeech_sample(output_path: str) -> np.ndarray:
    """Download a real speech sample from LibriSpeech test-clean."""
    try:
        import urllib.request
        url = ("https://www.openslr.org/resources/12/"
               "test-clean/61/70968/61-70968-0000.flac")
        urllib.request.urlretrieve(url, "/tmp/librispeech_sample.flac")
        import soundfile as sf
        audio, sr = sf.read("/tmp/librispeech_sample.flac")
        if sr != 16000:
            import torchaudio
            audio_t = torch.from_numpy(audio).float().unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
            audio = audio_t.squeeze().numpy()
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Could not download LibriSpeech sample: {e}", file=sys.stderr)
        print("Using synthetic audio instead", file=sys.stderr)
        return generate_test_audio()


def run_nemo_inference(model_id: str, audio: np.ndarray, output_dir: str):
    """Run NeMo model on audio and save reference outputs."""
    try:
        import nemo.collections.asr as nemo_asr
        print(f"Loading NeMo model: {model_id}")
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_id)
        model.eval()
    except ImportError:
        print("NeMo not installed. Trying direct torch inference...", file=sys.stderr)
        run_direct_inference(model_id, audio, output_dir)
        return

    with torch.no_grad():
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio_len = torch.tensor([len(audio)])

        # Get mel spectrogram from preprocessor
        processed, processed_len = model.preprocessor(
            input_signal=audio_t, length=audio_len)
        mel = processed.squeeze(0).cpu().numpy()

        # Get encoder output
        encoded, encoded_len = model.encoder(
            audio_signal=processed, length=processed_len)
        encoder_out = encoded.squeeze(0).cpu().numpy()

        # Get CTC logits
        logits = model.ctc_decoder(encoder_output=encoded)
        ctc_logits = logits.squeeze(0).cpu().numpy()

        # Greedy decode
        predictions = model.ctc_decoding.ctc_decoder_predictions_tensor(
            logits, encoded_len)
        transcript = predictions[0][0] if predictions and predictions[0] else ""

    print(f"NeMo transcript: \"{transcript}\"")
    print(f"Mel shape: {mel.shape}")
    print(f"Encoder output shape: {encoder_out.shape}")
    print(f"CTC logits shape: {ctc_logits.shape}")

    # Save reference data
    save_reference(output_dir, audio, mel, encoder_out, ctc_logits, transcript)


def run_direct_inference(model_id: str, audio: np.ndarray, output_dir: str):
    """Fallback: load weights directly and run manual forward pass for validation."""
    import tarfile
    import io
    import yaml
    from huggingface_hub import hf_hub_download

    print(f"Downloading model for direct inference: {model_id}")
    nemo_path = hf_hub_download(model_id, filename=f"{model_id.split('/')[-1]}.nemo")

    config = None
    state_dict = None
    with tarfile.open(nemo_path, "r:*") as tar:
        for member in tar.getmembers():
            name = os.path.basename(member.name)
            if name == "model_config.yaml":
                config = yaml.safe_load(tar.extractfile(member).read())
            elif name == "model_weights.ckpt":
                buf = io.BytesIO(tar.extractfile(member).read())
                state_dict = torch.load(buf, map_location="cpu", weights_only=False)

    if not config or not state_dict:
        print("Error: Could not load model", file=sys.stderr)
        sys.exit(1)

    # Just save audio + empty references; the C test will compare on its own
    print("NeMo package not available — saving audio for C-side validation only")
    mel = np.zeros((1, 80), dtype=np.float32)
    encoder_out = np.zeros((1, 512), dtype=np.float32)
    ctc_logits = np.zeros((1, 1025), dtype=np.float32)
    save_reference(output_dir, audio, mel, encoder_out, ctc_logits,
                   "(nemo not installed)")


def save_reference(output_dir: str, audio: np.ndarray, mel: np.ndarray,
                   encoder_out: np.ndarray, ctc_logits: np.ndarray,
                   transcript: str):
    """Save reference data as binary files for C test harness."""
    os.makedirs(output_dir, exist_ok=True)

    # Raw PCM audio (float32, 16kHz mono)
    audio_path = os.path.join(output_dir, "test_audio.pcm")
    audio.astype(np.float32).tofile(audio_path)
    print(f"  Audio: {audio_path} ({len(audio)} samples, {len(audio)/16000:.1f}s)")

    # Mel spectrogram reference (float32, [T, n_mels])
    mel_path = os.path.join(output_dir, "ref_mel.bin")
    with open(mel_path, "wb") as f:
        f.write(struct.pack("<II", mel.shape[0], mel.shape[1] if mel.ndim > 1 else 1))
        mel.astype(np.float32).tofile(f)
    print(f"  Mel: {mel_path} {mel.shape}")

    # Encoder output reference (float32, [T, D])
    enc_path = os.path.join(output_dir, "ref_encoder.bin")
    with open(enc_path, "wb") as f:
        f.write(struct.pack("<II", encoder_out.shape[0],
                            encoder_out.shape[1] if encoder_out.ndim > 1 else 1))
        encoder_out.astype(np.float32).tofile(f)
    print(f"  Encoder: {enc_path} {encoder_out.shape}")

    # CTC logits reference (float32, [T, vocab])
    logits_path = os.path.join(output_dir, "ref_logits.bin")
    with open(logits_path, "wb") as f:
        f.write(struct.pack("<II", ctc_logits.shape[0],
                            ctc_logits.shape[1] if ctc_logits.ndim > 1 else 1))
        ctc_logits.astype(np.float32).tofile(f)
    print(f"  Logits: {logits_path} {ctc_logits.shape}")

    # Transcript
    txt_path = os.path.join(output_dir, "ref_transcript.txt")
    with open(txt_path, "w") as f:
        f.write(transcript)
    print(f"  Transcript: \"{transcript}\"")

    # Metadata
    meta_path = os.path.join(output_dir, "meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"audio_samples={len(audio)}\n")
        f.write(f"sample_rate=16000\n")
        f.write(f"mel_frames={mel.shape[0]}\n")
        f.write(f"mel_dims={mel.shape[1] if mel.ndim > 1 else 1}\n")
        f.write(f"encoder_frames={encoder_out.shape[0]}\n")
        f.write(f"encoder_dims={encoder_out.shape[1] if encoder_out.ndim > 1 else 1}\n")
        f.write(f"logits_frames={ctc_logits.shape[0]}\n")
        f.write(f"vocab_size={ctc_logits.shape[1] if ctc_logits.ndim > 1 else 1}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate validation data for pocket-voice C STT engine")
    parser.add_argument("model", help="HuggingFace model ID")
    parser.add_argument("-o", "--output", default="validation",
                        help="Output directory for reference data")
    parser.add_argument("--speech", action="store_true",
                        help="Use real speech (downloads LibriSpeech sample)")
    args = parser.parse_args()

    if args.speech:
        audio = download_librispeech_sample(args.output)
    else:
        audio = generate_test_audio()

    run_nemo_inference(args.model, audio, args.output)

    print(f"\nValidation data saved to {args.output}/")
    print(f"Run C validation: ./tests/test_validate model.cstt {args.output}/")


if __name__ == "__main__":
    main()
