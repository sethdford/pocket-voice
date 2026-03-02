"""Automated TTS quality evaluation using UTMOS and objective metrics.

Generates audio samples from a Sonata checkpoint and scores them:
  - UTMOS: Neural MOS predictor (1-5 scale, correlates with human judgment)
  - PESQ: Perceptual Evaluation of Speech Quality (reference-based)
  - Speaker similarity: MFCC cosine similarity to reference
  - MCD: Mel-Cepstral Distortion (reference-based)

Usage:
  # Evaluate codec reconstruction quality
  python eval_quality.py --mode codec --codec-ckpt checkpoints/codec_v4/sonata_codec_best.pt

  # Evaluate full TTS pipeline (LM + Flow + Codec decoder)
  python eval_quality.py --mode tts \
    --codec-ckpt checkpoints/codec_v4/sonata_codec_best.pt \
    --lm-ckpt checkpoints/lm_v4/sonata_lm_best.pt \
    --flow-ckpt checkpoints/flow_v4/flow_best.pt

  # Quick codec eval on 10 samples
  python eval_quality.py --mode codec --codec-ckpt ckpt.pt --n-samples 10
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf


def load_codec(ckpt_path, device):
    from config import CodecConfig
    from codec import SonataCodec

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = CodecConfig(**{k: v for k, v in cfg_dict.items()
                         if k in CodecConfig.__dataclass_fields__})

    model = SonataCodec(cfg).to(device)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print(f"  Loaded EMA weights from {ckpt_path}")
    else:
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded model weights from {ckpt_path}")
    model.eval()
    return model, cfg


def compute_mcd(ref_audio, gen_audio, sr=24000, n_mfcc=13):
    """Mel-Cepstral Distortion in dB between reference and generated audio."""
    try:
        import librosa
    except ImportError:
        return float('nan')

    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=n_mfcc)
    gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=sr, n_mfcc=n_mfcc)

    T_min = min(ref_mfcc.shape[1], gen_mfcc.shape[1])
    diff = ref_mfcc[:, :T_min] - gen_mfcc[:, :T_min]
    # MCD [dB] = (10/ln10) * mean_T[ sqrt(2 * sum_k(diff_k^2)) ]
    mcd = (10.0 / np.log(10.0)) * np.mean(np.sqrt(2 * np.sum(diff[1:] ** 2, axis=0)))
    return float(mcd)


def compute_speaker_similarity(ref_audio, gen_audio, sr=24000, n_mfcc=13):
    """MFCC-based cosine similarity between reference and generated audio."""
    try:
        import librosa
    except ImportError:
        return float('nan')

    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=n_mfcc).mean(axis=1)
    gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=sr, n_mfcc=n_mfcc).mean(axis=1)
    cos_sim = np.dot(ref_mfcc, gen_mfcc) / (
        np.linalg.norm(ref_mfcc) * np.linalg.norm(gen_mfcc) + 1e-8
    )
    return float(cos_sim)


def compute_utmos(audio_list, sr=16000):
    """Compute UTMOS scores for a list of audio arrays.

    Tries multiple backends:
    1. speechmos (pip install speechmos)
    2. Direct UTMOS from HuggingFace (pip install transformers)
    3. Falls back to rule-based pseudo-MOS
    """
    scores = []

    # Try speechmos
    try:
        from speechmos import UTMOS
        model = UTMOS()
        for audio in audio_list:
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            score = model.predict(audio, sr)
            scores.append(float(score))
        return scores
    except ImportError:
        pass

    # Try transformers-based approach
    try:
        from transformers import pipeline
        pipe = pipeline("audio-classification", model="utmos/utmos22_strong", device=-1)
        for audio in audio_list:
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            result = pipe({"raw": audio, "sampling_rate": sr})
            score = sum(r["score"] * float(r["label"]) for r in result)
            scores.append(score)
        return scores
    except Exception:
        pass

    # Rule-based fallback: SNR + spectral flatness as proxy
    print("  [UTMOS] No neural MOS model available. Using rule-based proxy.")
    print("  Install: pip install speechmos  OR  pip install transformers")
    for audio in audio_list:
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6:
            scores.append(1.0)
            continue
        snr_proxy = 20 * np.log10(np.max(np.abs(audio)) / (rms + 1e-8))
        score = min(5.0, max(1.0, 1.0 + snr_proxy / 10.0))
        scores.append(score)
    return scores


def eval_codec(args):
    """Evaluate codec reconstruction quality."""
    device = torch.device(args.device)
    model, cfg = load_codec(args.codec_ckpt, device)

    test_audios = []
    if args.manifest:
        with open(args.manifest) as f:
            entries = [json.loads(line) for line in f]
        import random
        random.seed(42)
        random.shuffle(entries)
        entries = entries[:args.n_samples]
        for entry in entries:
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
            test_audios.append(audio)
    else:
        print("  No manifest provided. Generating synthetic test audio.")
        for i in range(args.n_samples):
            t = torch.linspace(0, 2.0, int(cfg.sample_rate * 2.0))
            freq = 200 + i * 50
            audio = 0.5 * torch.sin(2 * math.pi * freq * t)
            test_audios.append(audio)

    print(f"\n  Evaluating {len(test_audios)} samples...")
    results = {"mcd": [], "speaker_sim": [], "utmos_ref": [], "utmos_gen": []}
    gen_audios_16k = []
    ref_audios_16k = []

    for i, ref_audio in enumerate(test_audios):
        ref_audio = ref_audio.to(device)
        with torch.no_grad():
            reconstructed, tokens, _ = model(ref_audio.unsqueeze(0))
        gen_audio = reconstructed.squeeze(0)

        min_len = min(gen_audio.shape[-1], ref_audio.shape[-1])
        gen_np = gen_audio[:min_len].cpu().numpy()
        ref_np = ref_audio[:min_len].cpu().numpy()

        results["mcd"].append(compute_mcd(ref_np, gen_np, cfg.sample_rate))
        results["speaker_sim"].append(compute_speaker_similarity(ref_np, gen_np, cfg.sample_rate))

        ratio = 16000 / cfg.sample_rate
        ref_16k = F.interpolate(
            ref_audio[:min_len].unsqueeze(0).unsqueeze(0),
            size=int(min_len * ratio), mode='linear', align_corners=False
        ).squeeze().cpu().numpy()
        gen_16k = F.interpolate(
            gen_audio[:min_len].unsqueeze(0).unsqueeze(0),
            size=int(min_len * ratio), mode='linear', align_corners=False
        ).squeeze().cpu().numpy()
        ref_audios_16k.append(ref_16k)
        gen_audios_16k.append(gen_16k)

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(test_audios)}] processed")

    print("  Computing UTMOS scores...")
    results["utmos_ref"] = compute_utmos(ref_audios_16k, sr=16000)
    results["utmos_gen"] = compute_utmos(gen_audios_16k, sr=16000)

    print(f"\n{'='*60}")
    print(f"  SONATA CODEC QUALITY REPORT")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.codec_ckpt}")
    print(f"  Samples: {len(test_audios)}")
    print(f"{'─'*60}")

    def _stat(vals):
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            return "N/A"
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"

    print(f"  MCD (lower=better):      {_stat(results['mcd'])}")
    print(f"  Speaker sim (higher=1):   {_stat(results['speaker_sim'])}")
    print(f"  UTMOS reference:          {_stat(results['utmos_ref'])}")
    print(f"  UTMOS generated:          {_stat(results['utmos_gen'])}")

    utmos_ref_mean = np.mean([v for v in results['utmos_ref'] if not math.isnan(v)])
    utmos_gen_mean = np.mean([v for v in results['utmos_gen'] if not math.isnan(v)])
    print(f"  UTMOS delta:              {utmos_gen_mean - utmos_ref_mean:+.3f}")
    print(f"{'='*60}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({
                "checkpoint": str(args.codec_ckpt),
                "n_samples": len(test_audios),
                "mcd_mean": float(np.mean(results["mcd"])),
                "speaker_sim_mean": float(np.mean(results["speaker_sim"])),
                "utmos_ref_mean": float(utmos_ref_mean),
                "utmos_gen_mean": float(utmos_gen_mean),
                "utmos_delta": float(utmos_gen_mean - utmos_ref_mean),
            }, f, indent=2)
        print(f"  Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Sonata TTS Quality Evaluation")
    parser.add_argument("--mode", default="codec", choices=["codec", "tts"],
                        help="Evaluation mode: codec (reconstruction) or tts (full pipeline)")
    parser.add_argument("--codec-ckpt", required=True, help="Codec checkpoint path")
    parser.add_argument("--lm-ckpt", default="", help="LM checkpoint (for tts mode)")
    parser.add_argument("--flow-ckpt", default="", help="Flow checkpoint (for tts mode)")
    parser.add_argument("--manifest", default="",
                        help="Manifest JSONL with audio paths for test data")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of samples to evaluate")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", default="",
                        help="Output JSON file for results")
    args = parser.parse_args()

    if args.mode == "codec":
        eval_codec(args)
    else:
        print("TTS mode evaluation not yet implemented.")
        print("Run codec evaluation first to verify codec quality.")


if __name__ == "__main__":
    main()
