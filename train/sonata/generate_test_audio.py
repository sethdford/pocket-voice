#!/usr/bin/env python3
"""Generate test audio from partial Flow v3 checkpoints for ears-on feedback during training.

Works even when the vocoder isn't trained yet (uses Griffin-Lim fallback).

Usage:
  # Basic: Flow checkpoint only (Griffin-Lim)
  python generate_test_audio.py --checkpoint train/checkpoints/flow_v3/flow_v3_step_5000.pt

  # With trained vocoder
  python generate_test_audio.py --checkpoint flow_v3_step_5000.pt --vocoder vocoder_best.pt

  # Phoneme model + custom sentences
  python generate_test_audio.py --checkpoint flow_v3_step_5000.pt --phonemes \
    --sentences "Hello world.,Another test."

  # High quality (more steps + Heun)
  python generate_test_audio.py --checkpoint flow_v3_step_5000.pt --n-steps 16 --heun
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Add parent for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import FlowV3Config, FlowV3LargeConfig, VocoderConfig
from flow_v3 import SonataFlowV3
from modules import epss_schedule

try:
    from g2p import PhonemeFrontend
    HAS_G2P = True
except ImportError:
    HAS_G2P = False

try:
    from vocoder import SonataVocoder
except ImportError:
    SonataVocoder = None


TEST_SENTENCES = [
    "Hello, this is a test of the Sonata text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "I can't believe it's not butter!",
    "Yesterday, all my troubles seemed so far away.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck?",
    "To be or not to be, that is the question.",
    "It was a dark and stormy night.",
    "The rain in Spain falls mainly on the plain.",
    "One small step for man, one giant leap for mankind.",
]


def _mel_filterbank(n_mels: int, n_fft: int, sr: int, device: torch.device) -> torch.Tensor:
    """Build mel filterbank (n_mels, n_freqs) for forward mel transform."""
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0, sr / 2, n_freqs, device=device)
    mel_low = 2595 * math.log10(1 + 0 / 700)
    mel_high = 2595 * math.log10(1 + sr / 2 / 700)
    mels = torch.linspace(mel_low, mel_high, n_mels + 2, device=device)
    hz = 700 * (10 ** (mels / 2595) - 1)
    fb = torch.zeros(n_mels, n_freqs, device=device)
    for i in range(n_mels):
        low, center, high = hz[i].item(), hz[i + 1].item(), hz[i + 2].item()
        up = (freqs - low) / (center - low + 1e-8)
        down = (high - freqs) / (high - center + 1e-8)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0)
    return fb


def griffin_lim(mel: torch.Tensor, n_fft: int, hop_length: int, n_mels: int,
                sr: int, n_iter: int = 32, win_length: int = 1024) -> torch.Tensor:
    """Convert mel spectrogram to waveform via Griffin-Lim phase estimation.

    mel: (B, T_frames, n_mels) — log-mel spectrogram (natural log of power mel)
    Returns: (B, T_samples) waveform
    """
    device = mel.device
    B, T, _ = mel.shape
    n_freqs = n_fft // 2 + 1

    # Mel filterbank and pseudo-inverse
    mel_fb = _mel_filterbank(n_mels, n_fft, sr, device)
    inv_mel_fb = torch.linalg.pinv(mel_fb)

    # mel is log(power_mel). Convert to power: exp(mel), clamp for numerical stability
    mel_power = torch.exp(mel.clamp(max=10))

    # Inverse mel: power_mel (n_mels, T) -> power_spec (n_freqs, T)
    # mel_power: (B, T, n_mels) -> transpose to (B, n_mels, T)
    mel_t = mel_power.transpose(1, 2)
    mag_spec = torch.matmul(inv_mel_fb, mel_t)
    mag_spec = mag_spec.clamp(min=1e-8)

    window = torch.hann_window(win_length, periodic=True, device=device)
    waveforms = []

    for b in range(B):
        mag = mag_spec[b]
        phase = torch.exp(2j * math.pi * torch.rand(mag.shape, device=device))

        for _ in range(n_iter):
            spec = mag * phase
            spec_2d = spec.unsqueeze(0)
            wav = torch.istft(
                spec_2d, n_fft, hop_length, win_length,
                window=window, return_complex=False
            )
            wav = wav.squeeze(0)
            spec_new = torch.stft(
                wav, n_fft, hop_length, win_length,
                window=window, return_complex=True
            )
            phase = torch.exp(1j * torch.angle(spec_new))

        waveforms.append(wav)

    return torch.stack(waveforms, dim=0)


def compute_audio_stats(audio: np.ndarray, sr: int) -> dict:
    """Compute RMS, peak, zero-crossing rate for an audio array."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    duration = len(audio) / sr
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio + 1e-10))) > 0))
    return {
        "duration_sec": round(duration, 3),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "zcr": round(zcr, 4),
    }


def _sanitize_filename(s: str, max_chars: int = 30) -> str:
    """First words of sentence, sanitized for filename."""
    words = s.split()[:4]
    slug = "_".join(w[:12] for w in words)[:max_chars]
    return re.sub(r"[^\w\-]", "", slug) or "audio"


def main():
    parser = argparse.ArgumentParser(
        description="Generate test audio from partial Flow v3 checkpoints"
    )
    parser.add_argument("--checkpoint", required=True, help="Flow v3 checkpoint path")
    parser.add_argument("--vocoder", default="", help="Optional vocoder checkpoint for mel→waveform")
    parser.add_argument("--output-dir", default="train/checkpoints/flow_v3/test_audio",
                        help="Output directory for WAV files")
    parser.add_argument("--phonemes", action="store_true",
                        help="Use PhonemeFrontend (phoneme model)")
    parser.add_argument("--n-steps", type=int, default=8, help="ODE solver steps")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="CFG guidance scale")
    parser.add_argument("--heun", action="store_true", help="Use Heun 2nd-order ODE solver")
    parser.add_argument("--sentences", default="",
                        help="Comma-separated custom sentences (overrides default)")
    parser.add_argument("--device", default="mps", help="Device: cpu, mps, cuda")
    parser.add_argument("--no-ema", action="store_true", help="Don't use EMA weights")
    args = parser.parse_args()

    if args.device == "mps":
        try:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        except AttributeError:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Load Flow checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return 1

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})

    # Auto-detect model size from config
    d_model = cfg_dict.get("d_model", 512)
    n_layers = cfg_dict.get("n_layers", 12)
    is_large = d_model >= 768 or n_layers >= 16
    cfg_cls = FlowV3LargeConfig if is_large else FlowV3Config
    cfg = cfg_cls(**{k: v for k, v in cfg_dict.items()
                    if k in cfg_cls.__dataclass_fields__})

    # Load model state (EMA preferred)
    state = ckpt.get("ema", ckpt.get("model", ckpt)) if not args.no_ema else ckpt.get("model", ckpt)
    if state is ckpt:
        state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}

    flow = SonataFlowV3(cfg).to(device)
    flow.load_state_dict(state, strict=False)
    flow.eval()

    # Step from checkpoint
    step = ckpt.get("step", 0)
    if step == 0:
        m = re.search(r"step[_\-]?(\d+)", ckpt_path.stem, re.I)
        if m:
            step = int(m.group(1))

    # Text encoder
    g2p = None
    if args.phonemes:
        if not HAS_G2P:
            print("ERROR: --phonemes requires g2p (phonemizer). Install: pip install phonemizer")
            return 1
        g2p = PhonemeFrontend()

    def text_to_ids(text: str) -> torch.Tensor:
        if g2p is not None:
            ids = g2p.encode(text, add_bos=True, add_eos=True)
        else:
            vs = getattr(cfg, "char_vocab_size", 256)
            ids = torch.tensor([ord(c) % vs for c in text], dtype=torch.long)
        return ids.unsqueeze(0)

    # Vocoder (optional)
    vocoder = None
    sample_rate = cfg.sample_rate
    if args.vocoder and Path(args.vocoder).exists() and SonataVocoder is not None:
        voc_ckpt = torch.load(args.vocoder, map_location="cpu", weights_only=False)
        voc_cfg_dict = voc_ckpt.get("config", {})
        voc_cfg = VocoderConfig(**{k: v for k, v in voc_cfg_dict.items()
                                   if k in VocoderConfig.__dataclass_fields__})
        vocoder = SonataVocoder(voc_cfg).to(device)
        gen_state = voc_ckpt.get("generator", None)
        if gen_state:
            vocoder.generator.load_state_dict(gen_state, strict=False)
        else:
            vocoder.load_state_dict(voc_ckpt.get("model", voc_ckpt), strict=False)
        vocoder.eval()
        sample_rate = voc_cfg.sample_rate
    else:
        if args.vocoder:
            print(f"WARNING: Vocoder not found or not importable, using Griffin-Lim fallback")

    # Sentences
    if args.sentences:
        sentences = [s.strip() for s in args.sentences.split(",") if s.strip()]
    else:
        sentences = TEST_SENTENCES

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_fft = getattr(cfg, "n_fft", 1024)
    hop_length = getattr(cfg, "hop_length", 480)
    n_mels = getattr(cfg, "mel_dim", cfg.mel_dim if hasattr(cfg, "mel_dim") else 80)

    print(f"Flow v3 ({'large' if is_large else 'base'}) @ step {step}")
    print(f"Vocoder: {'SonataVocoder' if vocoder else 'Griffin-Lim'}")
    print(f"Sample rate: {sample_rate} Hz, {len(sentences)} sentences")
    print()

    step_schedule = epss_schedule(args.n_steps) if args.n_steps in (4, 5, 6, 7, 8) else None
    all_stats = []

    for idx, text in enumerate(sentences):
        char_ids = text_to_ids(text).to(device)

        with torch.no_grad():
            mel = flow.sample(
                char_ids, n_frames=0, n_steps=args.n_steps,
                speaker_ids=None, cfg_scale=args.cfg_scale,
                step_schedule=step_schedule, use_heun=args.heun,
            )

        if vocoder is not None:
            audio = vocoder.generate(mel)
            audio = audio.cpu().numpy()
        else:
            audio = griffin_lim(
                mel, n_fft, hop_length, n_mels, sample_rate,
                n_iter=32, win_length=n_fft
            )
            audio = audio.squeeze().cpu().numpy()
            if audio.ndim > 1:
                audio = audio[0]

        stats = compute_audio_stats(audio, sample_rate)
        slug = _sanitize_filename(text)
        fname = f"step_{step}_{idx}_{slug}.wav"
        out_path = out_dir / fname
        sf.write(out_path, audio, sample_rate)

        all_stats.append({
            "file": fname,
            "text": text,
            **stats,
        })

        print(f"  [{idx+1}/{len(sentences)}] {stats['duration_sec']:.2f}s "
              f"RMS={stats['rms']:.4f} peak={stats['peak']:.4f} ZCR={stats['zcr']:.3f}  {fname}")

    # Summary JSON
    summary = {
        "checkpoint": str(ckpt_path),
        "step": step,
        "vocoder": "SonataVocoder" if vocoder else "Griffin-Lim",
        "n_steps": args.n_steps,
        "cfg_scale": args.cfg_scale,
        "heun": args.heun,
        "phonemes": args.phonemes,
        "files": all_stats,
    }
    if all_stats:
        summary["mean_duration_sec"] = round(
            sum(s["duration_sec"] for s in all_stats) / len(all_stats), 3
        )
        summary["mean_rms"] = round(
            sum(s["rms"] for s in all_stats) / len(all_stats), 6
        )
        summary["mean_peak"] = round(
            sum(s["peak"] for s in all_stats) / len(all_stats), 6
        )
        summary["mean_zcr"] = round(
            sum(s["zcr"] for s in all_stats) / len(all_stats), 4
        )

    summary_path = out_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Saved {len(sentences)} WAVs to {out_dir}")
    if all_stats:
        print(f"Mean: duration={summary.get('mean_duration_sec', 0):.2f}s "
              f"RMS={summary.get('mean_rms', 0):.4f} peak={summary.get('mean_peak', 0):.4f} "
              f"ZCR={summary.get('mean_zcr', 0):.3f}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
