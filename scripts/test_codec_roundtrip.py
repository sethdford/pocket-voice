#!/usr/bin/env python3
"""Test codec encode→decode round-trip on real speech.

Proves the decoder works by encoding real audio through the trained codec
and then decoding back — should reconstruct the speech.
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train", "sonata"))

from config import CodecConfig
from codec import SonataCodec

def load_wav(path, target_sr=24000):
    """Load WAV file and resample to target sample rate."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype='float32')
    except ImportError:
        import wave
        import struct
        with wave.open(path, 'rb') as w:
            sr = w.getframerate()
            n = w.getnframes()
            ch = w.getnchannels()
            raw = w.readframes(n)
            dtype = {2: '<h', 4: '<i'}[w.getsampwidth()]
            samples = struct.unpack(f'{n * ch}{dtype}', raw)
            data = np.array(samples, dtype=np.float32) / (32768 if w.getsampwidth() == 2 else 2**31)
            if ch > 1:
                data = data.reshape(-1, ch).mean(axis=1)

    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = data.mean(axis=1)

    audio = torch.from_numpy(data).float()
    if sr != target_sr:
        ratio = target_sr / sr
        new_len = int(audio.shape[0] * ratio)
        audio = F.interpolate(
            audio.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear',
            align_corners=False
        ).squeeze()
    return audio


def save_wav(path, audio, sr=24000):
    """Save audio tensor as 16-bit WAV."""
    import struct, wave
    audio_np = audio.detach().cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    samples = (audio_np * 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Codec Encode→Decode Round-Trip Test")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    ckpt_path = os.path.join(ROOT, "train/checkpoints/codec/sonata_codec_final.pt")
    if not os.path.exists(ckpt_path):
        print(f"  ERROR: Codec checkpoint not found: {ckpt_path}")
        return 1

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = CodecConfig(**{k: v for k, v in cfg_dict.items()
                         if k in CodecConfig.__dataclass_fields__})
    model = SonataCodec(cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        # Try remapping old key names to new
        sd = ckpt["model"]
        remap = {}
        for k, v in sd.items():
            new_k = k
            if k.startswith("encoder.") and not k.startswith("encoder.acoustic") and not k.startswith("encoder.semantic"):
                new_k = "semantic_" + k
            remap[new_k] = v
        missing2, _ = model.load_state_dict(remap, strict=False)
        if len(missing2) < len(missing):
            missing = missing2
            print(f"  Remapped encoder→semantic_encoder: {len(missing)} still missing")
    model.eval()
    print(f"  Codec loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"  Config: sr={cfg.sample_rate}, n_fft={cfg.n_fft}, hop={cfg.hop_length}, "
          f"fsq={cfg.fsq_levels}, dec_dim={cfg.dec_dim}")

    out_dir = os.path.join(ROOT, "bench_output")
    os.makedirs(out_dir, exist_ok=True)

    test_files = []
    for name in ["test_speech_human.wav", "test_speech_human_16k.wav"]:
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            test_files.append(p)

    # Also try a LibriSpeech sample if available
    manifest = os.path.join(ROOT, "train/data/dev-clean_manifest.jsonl")
    if os.path.exists(manifest):
        with open(manifest) as f:
            first = json.loads(f.readline())
        audio_path = first["audio"]
        if os.path.exists(audio_path):
            test_files.append(audio_path)

    if not test_files:
        print("  No test audio files found, using synthetic chirp")
        t = torch.linspace(0, 2, 48000)
        audio = torch.sin(2 * np.pi * (200 + 600 * t) * t) * 0.5
        chirp_path = os.path.join(out_dir, "test_chirp.wav")
        save_wav(chirp_path, audio, 24000)
        test_files = [chirp_path]

    for path in test_files:
        name = os.path.basename(path)
        print(f"\n--- {name} ---")

        audio = load_wav(path, target_sr=cfg.sample_rate)
        if audio.shape[0] > cfg.sample_rate * 10:
            audio = audio[:cfg.sample_rate * 10]
        print(f"  Input: {audio.shape[0]} samples ({audio.shape[0]/cfg.sample_rate:.2f}s)")
        print(f"  Input RMS: {float(audio.pow(2).mean().sqrt()):.4f}")

        audio_in = audio.unsqueeze(0).to(device)

        with torch.no_grad():
            # Encode
            semantic_tokens, acoustic_latent, semantic_codes = model.encode(audio_in)
            n_frames = semantic_tokens.shape[1]
            print(f"  Encoded: {n_frames} frames")
            print(f"    Semantic tokens: range [{semantic_tokens.min()}, {semantic_tokens.max()}]")
            print(f"    Acoustic latent: shape {list(acoustic_latent.shape)}, "
                  f"range [{acoustic_latent.min():.3f}, {acoustic_latent.max():.3f}]")

            # Decode
            audio_out = model.decode(semantic_tokens, acoustic_latent)

        audio_out = audio_out.squeeze().cpu()
        out_rms = float(audio_out.pow(2).mean().sqrt())
        out_peak = float(audio_out.abs().max())
        out_dur = audio_out.shape[0] / cfg.sample_rate

        print(f"  Decoded: {audio_out.shape[0]} samples ({out_dur:.2f}s)")
        print(f"    RMS: {out_rms:.4f}  Peak: {out_peak:.4f}")

        # Compare
        min_len = min(audio.shape[0], audio_out.shape[0])
        ref = audio[:min_len].numpy()
        hyp = audio_out[:min_len].numpy()
        corr = float(np.corrcoef(ref, hyp)[0, 1]) if min_len > 0 else 0
        print(f"    Correlation with original: {corr:.4f}")

        if corr > 0.3:
            print(f"    → GOOD RECONSTRUCTION (correlation > 0.3)")
        elif corr > 0.1:
            print(f"    → PARTIAL RECONSTRUCTION")
        else:
            print(f"    → POOR RECONSTRUCTION")

        # Save
        stem = Path(name).stem
        out_path = os.path.join(out_dir, f"codec_roundtrip_{stem}.wav")
        save_wav(out_path, audio_out, cfg.sample_rate)
        print(f"  Saved: {out_path}")

        # Also save the input for comparison
        ref_path = os.path.join(out_dir, f"codec_reference_{stem}.wav")
        save_wav(ref_path, audio, cfg.sample_rate)

    print(f"\n{'='*60}")
    print(f"  Round-trip complete! Listen to bench_output/codec_roundtrip_*.wav")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
