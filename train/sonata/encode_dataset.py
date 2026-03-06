#!/usr/bin/env python3
"""Encode audio dataset with trained Sonata Codec.

Produces (text, semantic_tokens, acoustic_latents) triplets for LM and Flow training.

Usage:
  python train/sonata/encode_dataset.py \
    --codec-ckpt train/checkpoints/codec/sonata_codec_final.pt \
    --manifest train/data/dev-clean_manifest.jsonl \
    --output train/data/encoded_dev-clean.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf

from config import CodecConfig
from codec import SonataCodec


def _dio_f0(audio_np, sr: int, hop: int, n_frames: int) -> torch.Tensor:
    """DIO F0 extraction: multi-channel pitch estimation via band-pass filters.
    Much more accurate than autocorrelation, handles unvoiced regions well.
    Falls back to autocorrelation if pyworld is not installed.
    """
    try:
        import pyworld as pw
        import numpy as np
        audio_f64 = audio_np.astype(np.float64)
        f0, _ = pw.dio(audio_f64, sr, frame_period=(hop / sr * 1000.0))
        f0 = pw.stonemask(audio_f64, f0, _, sr)
        if len(f0) > n_frames:
            f0 = f0[:n_frames]
        elif len(f0) < n_frames:
            f0 = np.pad(f0, (0, n_frames - len(f0)))
        return torch.from_numpy(f0).float()
    except ImportError:
        return None


def _autocorr_f0(audio_np, sr: int, hop: int, n_frames: int) -> torch.Tensor:
    """Fallback autocorrelation F0 estimator."""
    import numpy as np
    f0 = torch.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = min(start + 2 * hop, len(audio_np))
        if end - start < hop:
            continue
        frame = audio_np[start:end]
        if frame.max() - frame.min() < 1e-6:
            continue
        corr = np.correlate(frame, frame, mode='full')
        mid = len(corr) // 2
        corr = corr[mid:]
        min_lag = sr // 500
        max_lag = sr // 50
        search = corr[min_lag:min(max_lag, len(corr))]
        if len(search) > 0:
            peak = int(search.argmax()) + min_lag
            if peak > 0:
                f0[i] = sr / peak
    return f0


def extract_prosody(audio: torch.Tensor, sr: int, n_frames: int, hop_length: int = 480) -> torch.Tensor:
    """Extract per-frame prosody features: (log_pitch, energy, speaking_rate).

    Uses DIO (pyworld) for high-quality F0 when available, falls back to autocorrelation.
    Interpolates F0 through unvoiced regions for a smooth contour.
    Returns: (n_frames, 3) tensor.
    """
    hop = hop_length
    audio_np = audio.squeeze().cpu().numpy()

    # Energy: vectorized RMS per frame
    energy = torch.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(audio_np))
        if start < len(audio_np):
            frame = audio_np[start:end]
            energy[i] = float((frame ** 2).mean() ** 0.5)

    # Pitch: DIO (high-quality) with autocorrelation fallback
    f0 = _dio_f0(audio_np, sr, hop, n_frames)
    if f0 is None:
        f0 = _autocorr_f0(audio_np, sr, hop, n_frames)

    # Voiced/unvoiced mask and interpolation through unvoiced regions
    voiced_mask = f0 > 0
    log_pitch = torch.zeros(n_frames)
    if voiced_mask.any():
        log_pitch[voiced_mask] = torch.log(f0[voiced_mask] + 1.0)

        # Linear interpolation through unvoiced gaps for smooth contour
        voiced_indices = torch.where(voiced_mask)[0].float()
        voiced_values = log_pitch[voiced_mask]
        if len(voiced_indices) >= 2:
            all_indices = torch.arange(n_frames).float()
            interp = torch.zeros(n_frames)
            for i in range(n_frames):
                if voiced_mask[i]:
                    interp[i] = log_pitch[i]
                else:
                    left = voiced_indices[voiced_indices <= i]
                    right = voiced_indices[voiced_indices >= i]
                    if len(left) > 0 and len(right) > 0:
                        li = int(left[-1].item())
                        ri = int(right[0].item())
                        if ri > li:
                            alpha = (i - li) / (ri - li)
                            interp[i] = (1 - alpha) * log_pitch[li] + alpha * log_pitch[ri]
            log_pitch = interp

    # Speaking rate: voiced frame density in sliding window
    voiced_float = voiced_mask.float()
    rate = torch.zeros(n_frames)
    window = 25
    for i in range(n_frames):
        lo = max(0, i - window)
        hi = min(n_frames, i + window + 1)
        rate[i] = voiced_float[lo:hi].mean()

    # Normalize to [0, 1] range
    if energy.max() > 0:
        energy = energy / energy.max()
    if log_pitch.max() > 0:
        log_pitch = log_pitch / log_pitch.max()

    return torch.stack([log_pitch, energy, rate], dim=-1)


@torch.no_grad()
def encode_dataset(args):
    device = torch.device(args.device)

    ckpt = torch.load(args.codec_ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = CodecConfig(**{k: v for k, v in cfg_dict.items()
                         if k in CodecConfig.__dataclass_fields__})
    model = SonataCodec(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[encode] Loaded codec from {args.codec_ckpt}")
    print(f"[encode] Frame rate: {cfg.frame_rate} Hz")

    entries = []
    with open(args.manifest) as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"[encode] Processing {len(entries)} utterances...")

    results = []
    for i, entry in enumerate(entries):
        data, sr = sf.read(entry["audio"], dtype='float32')
        audio = torch.from_numpy(data)
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)
        if sr != cfg.sample_rate:
            ratio = cfg.sample_rate / sr
            new_len = int(audio.shape[0] * ratio)
            audio = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear',
                align_corners=False
            ).squeeze()

        if audio.shape[0] > cfg.sample_rate * args.max_duration:
            audio = audio[:cfg.sample_rate * int(args.max_duration)]

        audio_device = audio.unsqueeze(0).to(device)
        semantic_tokens, acoustic_latent, semantic_codes = model.encode(audio_device)

        n_frames = semantic_tokens.shape[1]
        hop_length = getattr(cfg, 'hop_length', 480)
        prosody = extract_prosody(audio, cfg.sample_rate, n_frames, hop_length)

        results.append({
            "text": entry["text"],
            "utt_id": entry.get("utt_id", f"utt_{i:06d}"),
            "semantic_tokens": semantic_tokens[0].cpu(),
            "acoustic_latent": acoustic_latent[0].cpu(),
            "prosody_features": prosody,
            "n_frames": n_frames,
            "n_samples": audio_device.shape[1],
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(entries)}] {entry.get('utt_id', '?')}: "
                  f"{semantic_tokens.shape[1]} frames, "
                  f"tokens range [{semantic_tokens.min()}-{semantic_tokens.max()}]")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_path)
    print(f"\n[encode] Saved {len(results)} encoded utterances to {out_path}")

    total_frames = sum(r["n_frames"] for r in results)
    total_sec = sum(r["n_samples"] for r in results) / cfg.sample_rate
    print(f"[encode] Total: {total_frames} frames, {total_sec:.1f}s audio")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec-ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-duration", type=float, default=30.0)
    args = parser.parse_args()
    encode_dataset(args)


if __name__ == "__main__":
    main()
