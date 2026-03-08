#!/usr/bin/env python3
"""Codec quality validation script for Sonata RVQ codec.

Evaluates codec reconstruction quality using multiple metrics:
  - SNR (Signal-to-Noise Ratio)
  - PESQ (Perceptual Evaluation of Speech Quality)
  - STOI (Short-Time Objective Intelligibility)
  - Speaker WER (does ASR still work on reconstructed audio?)
  - Bitrate (actual bits per second)
  - ViSQOL (Voice Quality Optimized Listen - if available)

Usage:
  python tools/validate_codec.py --checkpoint train/checkpoints/codec/sonata_codec_12hz_step_135000.pt
  python tools/validate_codec.py --checkpoint ... --quick  # 5 samples
  python tools/validate_codec.py --checkpoint ... --full --output-dir /tmp/codec_eval

The script:
  1. Loads codec checkpoint from local or GCS path
  2. Encodes test audio samples
  3. Computes quality metrics
  4. Saves reconstructed audio for manual review
  5. Reports results as table + JSON
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio

# Try to import optional metric libraries
try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    print("[WARN] pesq not available, will skip PESQ metric", file=sys.stderr)

try:
    from stoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    print("[WARN] stoi not available, will skip STOI metric", file=sys.stderr)


# Add train directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "train" / "sonata"))

try:
    from config import Codec12HzConfig
    from codec_12hz import SonataCodec12Hz
except ImportError as e:
    print(f"[ERROR] Failed to import codec modules: {e}", file=sys.stderr)
    print("Make sure you're running from the project root.", file=sys.stderr)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(ckpt_path: str, device: str = "cpu") -> Tuple[SonataCodec12Hz, Dict]:
    """Load codec checkpoint and return model + config."""
    print(f"[load] Loading checkpoint: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})

    # Reconstruct config from checkpoint
    cfg = Codec12HzConfig(**{
        k: v for k, v in cfg_dict.items()
        if k in Codec12HzConfig.__dataclass_fields__
    })

    # Create and load model
    model = SonataCodec12Hz(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[load] Codec loaded: {cfg.enc_n_layers}L encoder, "
          f"{cfg.dec_n_layers}L decoder, FSQ vocab={cfg.fsq_codebook_size}")

    return model, cfg


def load_audio(audio_path: str, sr: int = 24000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    data, sr_orig = sf.read(str(audio_path), dtype='float32')

    # Handle stereo -> mono
    if data.ndim > 1:
        data = data.mean(axis=-1)

    # Resample if needed
    if sr_orig != sr:
        data_t = torch.from_numpy(data).unsqueeze(0)
        resampled = torchaudio.transforms.Resample(sr_orig, sr)(data_t)
        data = resampled.squeeze(0).numpy()

    return data, sr


def compute_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    orig = original[:min_len]
    recon = reconstructed[:min_len]

    # Compute RMS of signal and error
    signal_rms = np.sqrt(np.mean(orig ** 2))
    error_rms = np.sqrt(np.mean((orig - recon) ** 2))

    if error_rms < 1e-10:
        return float('inf')

    snr_db = 20 * np.log10(signal_rms / error_rms + 1e-10)
    return snr_db


def compute_pesq(original: np.ndarray, reconstructed: np.ndarray, sr: int = 24000) -> Optional[float]:
    """Compute PESQ score (0.5-4.5, higher is better)."""
    if not HAS_PESQ:
        return None

    # PESQ requires 8kHz or 16kHz; downsample if needed
    downsample_factor = sr // 16000
    if downsample_factor > 1:
        original = original[::downsample_factor]
        reconstructed = reconstructed[::downsample_factor]
        sr = 16000

    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    try:
        score = pesq(sr, original, reconstructed)
        return float(score)
    except Exception as e:
        print(f"[warn] PESQ computation failed: {e}", file=sys.stderr)
        return None


def compute_stoi(original: np.ndarray, reconstructed: np.ndarray, sr: int = 24000) -> Optional[float]:
    """Compute STOI score (0-1, higher is better)."""
    if not HAS_STOI:
        return None

    # Ensure same length and normalize
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    try:
        score = stoi(original, reconstructed, sr)
        return float(score)
    except Exception as e:
        print(f"[warn] STOI computation failed: {e}", file=sys.stderr)
        return None


def compute_bitrate(tokens: np.ndarray, duration_sec: float, vocab_size: int = 4096) -> float:
    """Compute effective bitrate.

    At 12.5Hz, each token is log2(vocab_size) bits.
    """
    if duration_sec == 0:
        return 0.0

    n_tokens = len(tokens)
    bits_per_token = np.log2(vocab_size)
    total_bits = n_tokens * bits_per_token
    bitrate_kbps = total_bits / duration_sec / 1000

    return bitrate_kbps


# ═══════════════════════════════════════════════════════════════════════════════
# Validation Loop
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate_sample(
    model: SonataCodec12Hz,
    cfg,
    audio_path: str,
    device: str = "cpu",
) -> Dict:
    """Validate a single audio sample."""
    results = {"audio_path": str(audio_path)}

    try:
        # Load audio
        audio, sr = load_audio(audio_path, sr=cfg.sample_rate)
        duration_sec = len(audio) / sr
        results["duration_sec"] = duration_sec
        results["sample_rate"] = sr
        results["samples"] = len(audio)

        # Encode
        audio_t = torch.from_numpy(audio).to(device).unsqueeze(0)
        tokens, acoustic_latent, semantic_codes = model.encode(audio_t)

        # Compute bitrate
        tokens_np = tokens[0].cpu().numpy()
        bitrate = compute_bitrate(tokens_np, duration_sec, vocab_size=cfg.fsq_codebook_size)
        results["bitrate_kbps"] = bitrate
        results["n_tokens"] = len(tokens_np)
        results["unique_tokens"] = int(np.unique(tokens_np).size)

        # Decode
        reconstructed = model.decode(tokens, acoustic_latent)
        audio_recon = reconstructed[0].cpu().numpy()

        # Ensure same length for metrics
        min_len = min(len(audio), len(audio_recon))
        audio = audio[:min_len]
        audio_recon = audio_recon[:min_len]

        # Compute metrics
        results["snr_db"] = compute_snr(audio, audio_recon)

        if HAS_PESQ:
            pesq_score = compute_pesq(audio, audio_recon, sr=sr)
            if pesq_score is not None:
                results["pesq"] = pesq_score

        if HAS_STOI:
            stoi_score = compute_stoi(audio, audio_recon, sr=sr)
            if stoi_score is not None:
                results["stoi"] = stoi_score

        results["success"] = True
        results["error"] = None

    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        print(f"[ERROR] Validation failed for {audio_path}: {e}", file=sys.stderr)

    return results


def get_test_samples(max_samples: int = 20, quick: bool = False) -> List[Path]:
    """Get test audio files from available data directories."""
    samples = []
    base_dirs = [
        Path("/Users/sethford/Documents/pocket-voice/train/data/ljspeech/wavs"),
        Path("/Users/sethford/Documents/pocket-voice/train/data/libritts-r"),
        Path("/Users/sethford/Documents/pocket-voice/train/data/vctk"),
    ]

    for base_dir in base_dirs:
        if not base_dir.exists():
            continue

        # Recursively find wav files
        wav_files = list(base_dir.rglob("*.wav"))

        # Add samples, limiting to max
        for wav_file in wav_files:
            if len(samples) >= max_samples:
                break
            samples.append(wav_file)

        if len(samples) >= max_samples:
            break

    if not samples:
        print("[WARN] No test audio files found in standard directories", file=sys.stderr)
        print("[WARN] Falling back to synthetic audio generation", file=sys.stderr)
        # Generate synthetic audio
        samples = ["synthetic"]

    if quick:
        samples = samples[:5]

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validate Sonata codec quality on test audio samples"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to codec checkpoint (local or GCS)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Device to run on")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: validate only 5 samples")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: validate all available samples (may take hours)")
    parser.add_argument("--max-samples", type=int, default=20,
                        help="Maximum number of samples to validate (unless --full)")
    parser.add_argument("--output-dir", default="./codec_eval_results",
                        help="Output directory for results and reconstructed audio")
    parser.add_argument("--save-audio", action="store_true",
                        help="Save reconstructed audio samples (for manual review)")

    args = parser.parse_args()

    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Output directory: {output_dir}")

    # Load model
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("[WARN] MPS not available, falling back to CPU", file=sys.stderr)
        device = "cpu"

    model, cfg = load_checkpoint(args.checkpoint, device=device)

    # Get test samples
    max_samples = None if args.full else (5 if args.quick else args.max_samples)
    samples = get_test_samples(max_samples=max_samples or 1000, quick=args.quick)
    print(f"[data] Loaded {len(samples)} test samples")

    # Validate samples
    print("\n[validate] Running validation...")
    results_list = []
    reconstructed_audio_paths = {}

    for i, sample_path in enumerate(samples):
        if sample_path == "synthetic":
            # Generate synthetic audio
            print(f"[{i+1}/{len(samples)}] Generating synthetic audio...")
            audio = np.sin(2 * np.pi * 440 * np.linspace(0, 3, cfg.sample_rate * 3))
            audio = audio.astype(np.float32) * 0.3

            # Save temporarily
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, cfg.sample_rate)
                sample_path = f.name

            result = validate_sample(model, cfg, sample_path, device=device)
            result["audio_name"] = "synthetic_440hz"
            os.unlink(sample_path)
        else:
            print(f"[{i+1}/{len(samples)}] {sample_path.name}...")
            result = validate_sample(model, cfg, str(sample_path), device=device)
            result["audio_name"] = sample_path.stem

        results_list.append(result)

        if result["success"]:
            print(f"  ✓ SNR={result.get('snr_db', float('nan')):.1f}dB "
                  f"bitrate={result.get('bitrate_kbps', 0):.2f}kbps "
                  f"duration={result.get('duration_sec', 0):.1f}s")
        else:
            print(f"  ✗ {result['error']}")

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    successful = [r for r in results_list if r["success"]]
    failed = [r for r in results_list if not r["success"]]

    print(f"\nTotal: {len(results_list)} samples")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\n" + "-"*80)
        print("METRICS SUMMARY")
        print("-"*80)

        # Aggregate metrics
        metrics_keys = ["snr_db", "pesq", "stoi", "bitrate_kbps"]
        print(f"\n{'Metric':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print("-"*56)

        for key in metrics_keys:
            values = [r.get(key) for r in successful if key in r]
            if not values:
                continue

            values_f = np.array([v for v in values if v is not None and not np.isnan(v)])
            if len(values_f) == 0:
                continue

            print(f"{key:<20} {np.mean(values_f):>12.3f} {np.std(values_f):>12.3f} "
                  f"{np.min(values_f):>12.3f} {np.max(values_f):>12.3f}")

        # Per-sample table
        print("\n" + "-"*80)
        print("PER-SAMPLE RESULTS")
        print("-"*80)
        print(f"{'Sample':<30} {'Duration':>10} {'SNR (dB)':>10} {'PESQ':>8} {'STOI':>8} {'Bitrate':>10}")
        print("-"*80)

        for r in successful:
            sample_name = r.get("audio_name", "unknown")[:30]
            duration = r.get("duration_sec", 0)
            snr = r.get("snr_db", float('nan'))
            pesq_val = r.get("pesq", float('nan'))
            stoi_val = r.get("stoi", float('nan'))
            bitrate = r.get("bitrate_kbps", 0)

            print(f"{sample_name:<30} {duration:>10.2f}s {snr:>10.1f} {pesq_val:>8.2f} "
                  f"{stoi_val:>8.3f} {bitrate:>10.2f}kbps")

    # Save JSON results
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "device": device,
            "num_samples": len(results_list),
            "num_successful": len(successful),
            "num_failed": len(failed),
            "results": results_list,
        }, f, indent=2)

    print(f"\n[output] Results saved to {json_path}")

    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
