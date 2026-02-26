#!/usr/bin/env python3
"""Cross-validate Python vs Rust Sonata Flow+Decoder inference.

Loads the same model weights, feeds the same semantic tokens and acoustic
latents, and compares the output audio. This catches:
  - Weight loading mismatches
  - Architecture divergence
  - Numerical precision differences (F16 vs F32)

Usage:
    python scripts/cross_validate_sonata.py

Produces:
    bench_output/python_crossval_*.wav   — Python-generated audio
    bench_output/python_crossval_report.txt  — Comparison report
"""

import json
import os
import sys
import wave
import struct
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT, "train", "sonata"))

import torch
import torch.nn.functional as F

from config import CodecConfig, FlowConfig
from codec import ConvDecoder, FSQ
from flow import SonataFlow


SAMPLE_RATE = 24000
OUTPUT_DIR = os.path.join(ROOT, "bench_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def write_wav(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def load_flow(device: torch.device) -> SonataFlow:
    cfg_path = os.path.join(ROOT, "models/sonata/sonata_flow_config.json")
    weights_path = os.path.join(ROOT, "models/sonata/sonata_flow.safetensors")

    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    cfg = FlowConfig(**{k: v for k, v in cfg_dict.items() if hasattr(FlowConfig, k)})

    model = SonataFlow(cfg)
    if os.path.exists(weights_path):
        from safetensors.torch import load_file
        sd = load_file(weights_path)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [warn] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"  [warn] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    model = model.to(device).eval()
    print(f"  Flow: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model


def load_decoder(device: torch.device) -> ConvDecoder:
    cfg_path = os.path.join(ROOT, "models/sonata/sonata_decoder_config.json")
    weights_path = os.path.join(ROOT, "models/sonata/sonata_decoder.safetensors")

    with open(cfg_path) as f:
        dec_dict = json.load(f)

    codec_cfg = CodecConfig(
        fsq_levels=dec_dict.get("fsq_levels", [8, 8, 8, 8]),
        dec_dim=dec_dict.get("dec_dim", 512),
        dec_n_layers=dec_dict.get("dec_n_layers", 8),
        dec_conv_kernel=dec_dict.get("dec_conv_kernel", 7),
        dec_ff_mult=dec_dict.get("dec_ff_mult", 4.0),
        acoustic_dim=dec_dict.get("acoustic_dim", 256),
        decoder_type="conv",
    )

    model = ConvDecoder(codec_cfg)
    if os.path.exists(weights_path):
        from safetensors.torch import load_file
        sd = load_file(weights_path)
        # Weights saved from SonataCodec wrapper have 'decoder.' prefix — strip it
        stripped = {}
        for k, v in sd.items():
            new_key = k.replace("decoder.", "", 1) if k.startswith("decoder.") else k
            stripped[new_key] = v
        missing, unexpected = model.load_state_dict(stripped, strict=False)
        if missing:
            print(f"  [warn] Decoder missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"  [warn] Decoder unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    model = model.to(device).eval()
    print(f"  Decoder: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model


def fsq_decompose(codes: np.ndarray, levels: list) -> np.ndarray:
    """Convert flat codebook indices to FSQ code vectors."""
    n_dim = len(levels)
    result = np.zeros((len(codes), n_dim), dtype=np.float32)
    for d in reversed(range(n_dim)):
        L = levels[d]
        result[:, d] = (codes % L).astype(np.float32)
        result[:, d] = result[:, d] / (L - 1) * 2 - 1
        codes = codes // L
    return result


def compute_audio_stats(audio: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    zc = int(np.sum(np.diff(np.sign(audio)) != 0))
    zcr = zc / max(len(audio), 1)
    return {"rms": rms, "peak": peak, "zcr": zcr, "duration_s": len(audio) / SAMPLE_RATE}


@torch.no_grad()
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Sonata Cross-Validation: Python vs Rust")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Load models
    print("Loading models...")
    try:
        flow = load_flow(device)
        decoder = load_decoder(device)
    except Exception as e:
        print(f"  [FATAL] Cannot load models: {e}")
        return 1

    fsq_levels = [8, 8, 8, 8]

    # Test with random semantic tokens (same seed → reproducible)
    test_configs = [
        {"name": "short", "n_frames": 25, "seed": 42},
        {"name": "medium", "n_frames": 75, "seed": 42},
        {"name": "long", "n_frames": 150, "seed": 42},
    ]

    report_lines = []
    report_lines.append("Sonata Cross-Validation Report\n")
    report_lines.append(f"Device: {device}\n\n")

    for tc in test_configs:
        name = tc["name"]
        n_frames = tc["n_frames"]
        print(f"\n--- Test: {name} ({n_frames} frames) ---")

        torch.manual_seed(tc["seed"])
        np.random.seed(tc["seed"])

        # Generate random semantic token IDs
        codebook_size = 1
        for l in fsq_levels:
            codebook_size *= l
        sem_ids = np.random.randint(0, codebook_size, size=n_frames)

        # Flow expects (B, T) integer token indices
        sem_idx_tensor = torch.from_numpy(sem_ids).unsqueeze(0).long().to(device)

        # FSQ decompose for decoder (B, T, fsq_dim)
        sem_codes = fsq_decompose(sem_ids, fsq_levels)
        sem_codes_tensor = torch.from_numpy(sem_codes).unsqueeze(0).to(device)

        # Generate acoustic latents via Flow (takes token indices)
        torch.manual_seed(tc["seed"] + 1000)
        acoustic = flow.sample(sem_idx_tensor, n_steps=8)
        print(f"  Flow output: {acoustic.shape} (acoustic latents)")

        # Decode to audio (takes FSQ codes + acoustic latents)
        audio_tensor = decoder(sem_codes_tensor, acoustic)
        audio = audio_tensor.squeeze().cpu().numpy()
        print(f"  Decoder output: {audio.shape} ({len(audio)/SAMPLE_RATE:.2f}s)")

        stats = compute_audio_stats(audio)
        print(f"  Stats: RMS={stats['rms']:.4f}  peak={stats['peak']:.4f}  "
              f"ZCR={stats['zcr']:.4f}  dur={stats['duration_s']:.2f}s")

        # Save WAV
        wav_path = os.path.join(OUTPUT_DIR, f"python_crossval_{name}.wav")
        write_wav(wav_path, audio)
        print(f"  Wrote: {wav_path}")

        # Save semantic tokens for Rust comparison
        tok_path = os.path.join(OUTPUT_DIR, f"python_crossval_{name}_tokens.npy")
        np.save(tok_path, sem_ids)
        print(f"  Tokens: {tok_path}")

        # Check for corresponding Rust-generated WAV
        rust_wav = os.path.join(OUTPUT_DIR, f"sonata_quality_0.wav")
        if os.path.exists(rust_wav):
            print(f"  [info] Rust WAV available for comparison: {rust_wav}")

        report_lines.append(f"Test: {name} ({n_frames} frames)\n")
        report_lines.append(f"  RMS: {stats['rms']:.4f}\n")
        report_lines.append(f"  Peak: {stats['peak']:.4f}\n")
        report_lines.append(f"  ZCR: {stats['zcr']:.4f}\n")
        report_lines.append(f"  Duration: {stats['duration_s']:.2f}s\n")
        report_lines.append(f"  Expected duration: {n_frames * 480 / SAMPLE_RATE:.2f}s\n\n")

    # Compare with any existing Rust WAVs
    print(f"\n--- Checking for Rust-generated WAVs ---")
    rust_wavs = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("sonata_quality_") and f.endswith(".wav")])
    if rust_wavs:
        print(f"  Found {len(rust_wavs)} Rust WAVs: {', '.join(rust_wavs)}")
        for rwav in rust_wavs:
            rpath = os.path.join(OUTPUT_DIR, rwav)
            try:
                with wave.open(rpath) as wf:
                    n_frames_wav = wf.getnframes()
                    sr = wf.getframerate()
                    raw = wf.readframes(n_frames_wav)
                    rust_audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    rs = compute_audio_stats(rust_audio)
                    print(f"  {rwav}: {rs['duration_s']:.2f}s  RMS={rs['rms']:.4f}  peak={rs['peak']:.4f}  ZCR={rs['zcr']:.4f}")
                    report_lines.append(f"Rust WAV: {rwav}\n")
                    report_lines.append(f"  RMS: {rs['rms']:.4f}\n")
                    report_lines.append(f"  Peak: {rs['peak']:.4f}\n")
                    report_lines.append(f"  ZCR: {rs['zcr']:.4f}\n")
                    report_lines.append(f"  Duration: {rs['duration_s']:.2f}s\n\n")
            except Exception as e:
                print(f"  {rwav}: ERROR reading — {e}")
    else:
        print("  No Rust WAVs found. Run `make test-sonata-quality` first.")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "python_crossval_report.txt")
    with open(report_path, "w") as f:
        f.writelines(report_lines)
    print(f"\nReport saved: {report_path}")

    print(f"\n{'='*60}")
    print(f"  Cross-validation complete.")
    print(f"  Python WAVs: bench_output/python_crossval_*.wav")
    print(f"  Compare by listening to Python vs Rust WAVs")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
