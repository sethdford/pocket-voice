#!/usr/bin/env python3
"""Validate NeuCodec C decoder against Python reference.

Decodes the same token sequence used in C tests, dumps intermediate
activations as .npy files, and compares against C decoder WAV output.

Usage:
  python scripts/validate_neucodec.py [--c-wav bench_output/neucodec_c_1sec.wav]
"""

import argparse
import struct
import wave
from pathlib import Path

import numpy as np
import torch

CODES_50 = [
    2151, 43235, 56802, 4794, 18022, 27623, 37529, 2112, 2308, 7609,
    35233, 6502, 35973, 51633, 16987, 25384, 378, 36, 4221, 12329,
    2167, 22631, 64242, 47786, 65186, 59898, 39411, 35943, 50261, 9878,
    28369, 32226, 28322, 28386, 28386, 28386, 28386, 28322, 28386, 28386,
    28386, 28386, 27042, 27314, 27110, 32418, 27366, 27366, 27302, 27298,
]


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        n = wf.getnframes()
        raw = wf.readframes(n)
    pcm16 = np.frombuffer(raw, dtype=np.int16)
    return pcm16.astype(np.float32) / 32767.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-wav", default="bench_output/neucodec_c_1sec.wav")
    parser.add_argument("--output-dir", default="bench_output/neucodec_validate")
    parser.add_argument("--output-wav", default="bench_output/neucodec_py_1sec.wav")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading NeuCodec Python model...")
    from neucodec import NeuCodec
    model = NeuCodec.from_pretrained("neuphonic/neucodec")
    model.eval()

    codes_tensor = torch.tensor([CODES_50], dtype=torch.long)
    print(f"Decoding {len(CODES_50)} codes...")

    # --- Step-by-step decode with intermediate dumps ---
    with torch.no_grad():
        # 1. FSQ decode
        fsq = model.generator.quantizer.layers[0]
        fsq_emb = fsq.implicit_codebook[CODES_50]  # (50, 8)
        np.save(out_dir / "01_fsq_emb.npy", fsq_emb.numpy())
        print(f"  FSQ embedding: {fsq_emb.shape}, range [{fsq_emb.min():.4f}, {fsq_emb.max():.4f}]")

        # 2. project_out: 8 → 2048
        proj_out = model.generator.quantizer.project_out(fsq_emb)  # (50, 2048)
        np.save(out_dir / "02_proj_out.npy", proj_out.numpy())
        print(f"  project_out: {proj_out.shape}, range [{proj_out.min():.4f}, {proj_out.max():.4f}]")

        # 3. fc_post_a: 2048 → 1024
        fc_post = model.fc_post_a(proj_out)  # (50, 1024)
        np.save(out_dir / "03_fc_post.npy", fc_post.numpy())
        print(f"  fc_post_a: {fc_post.shape}, range [{fc_post.min():.4f}, {fc_post.max():.4f}]")

        # 4. VocosBackbone
        backbone = model.generator.backbone
        x = fc_post.unsqueeze(0).transpose(1, 2)  # (1, 1024, 50) — channels first for Conv1d

        # 4a. embed Conv1d
        x = backbone.embed(x)
        x_after_embed = x.squeeze(0).transpose(0, 1)  # (T, 1024)
        np.save(out_dir / "04_after_embed.npy", x_after_embed.numpy())
        print(f"  after embed: {x_after_embed.shape}, range [{x_after_embed.min():.4f}, {x_after_embed.max():.4f}]")

        # 4b. prior_net ResNet blocks
        for i, block in enumerate(backbone.prior_net):
            x = block(x)
        x_after_prior = x.squeeze(0).transpose(0, 1)  # (T, 1024)
        np.save(out_dir / "05_after_prior_net.npy", x_after_prior.numpy())
        print(f"  after prior_net: {x_after_prior.shape}, range [{x_after_prior.min():.4f}, {x_after_prior.max():.4f}]")

        # 4c. Transformer blocks
        x_t = x.transpose(1, 2)  # (1, T, 1024) for transformer
        for i, layer in enumerate(backbone.transformers):
            x_t = layer(x_t)
        x_after_xfmr = x_t.squeeze(0)  # (T, 1024)
        np.save(out_dir / "06_after_transformers.npy", x_after_xfmr.numpy())
        print(f"  after transformers: {x_after_xfmr.shape}, range [{x_after_xfmr.min():.4f}, {x_after_xfmr.max():.4f}]")

        # 4d. final_layer_norm
        x_ln = backbone.final_layer_norm(x_t)
        x_after_ln = x_ln.squeeze(0)  # (T, 1024)
        np.save(out_dir / "07_after_final_ln.npy", x_after_ln.numpy())
        print(f"  after final_ln: {x_after_ln.shape}, range [{x_after_ln.min():.4f}, {x_after_ln.max():.4f}]")

        # 4e. post_net ResNet blocks
        x_post = x_ln.transpose(1, 2)  # (1, 1024, T)
        for i, block in enumerate(backbone.post_net):
            x_post = block(x_post)
        x_after_post = x_post.squeeze(0).transpose(0, 1)  # (T, 1024)
        np.save(out_dir / "08_after_post_net.npy", x_after_post.numpy())
        print(f"  after post_net: {x_after_post.shape}, range [{x_after_post.min():.4f}, {x_after_post.max():.4f}]")

        # Full model decode for reference — expects [B, 1, F]
        print("\n  Running full model decode...")
        audio = model.decode_code(codes_tensor.unsqueeze(1))
        audio_np = audio.squeeze().numpy()
        np.save(out_dir / "09_full_audio.npy", audio_np)
        print(f"  Full audio: {audio_np.shape}, RMS={np.sqrt(np.mean(audio_np**2)):.6f}")

    # Write Python reference WAV
    pcm16 = np.clip(audio_np * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(args.output_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm16.tobytes())
    print(f"  Wrote {args.output_wav}")

    # --- Compare against C decoder ---
    print(f"\n=== Comparison ===")
    c_wav_path = Path(args.c_wav)
    if c_wav_path.exists():
        c_audio = load_wav(str(c_wav_path))
        py_audio = audio_np

        min_len = min(len(c_audio), len(py_audio))
        c_audio = c_audio[:min_len]
        py_audio = py_audio[:min_len]

        c_rms = np.sqrt(np.mean(c_audio**2))
        py_rms = np.sqrt(np.mean(py_audio**2))

        diff = c_audio - py_audio
        mae = np.mean(np.abs(diff))
        max_err = np.max(np.abs(diff))

        corr = np.corrcoef(c_audio, py_audio)[0, 1] if min_len > 1 else 0.0

        print(f"  C  audio: {len(c_audio)} samples, RMS={c_rms:.6f}")
        print(f"  Py audio: {len(py_audio)} samples, RMS={py_rms:.6f}")
        print(f"  RMS ratio (C/Py): {c_rms/py_rms:.4f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Max error: {max_err:.6f}")
        print(f"  Pearson correlation: {corr:.6f}")

        if corr > 0.9:
            print("  EXCELLENT: High correlation — C decoder closely matches Python")
        elif corr > 0.7:
            print("  GOOD: Moderate correlation — structurally similar output")
        elif corr > 0.3:
            print("  FAIR: Low correlation — some structural similarity")
        else:
            print("  POOR: Very low correlation — significant divergence")
    else:
        print(f"  C WAV not found at {c_wav_path}, run 'make test-neucodec' first")

    print("\nDone. Intermediate activations saved to", out_dir)


if __name__ == "__main__":
    main()
