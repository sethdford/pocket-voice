#!/usr/bin/env python3
"""Sonata End-to-End Benchmark Suite.

Tests the complete pipeline: Text → LM → Flow → Codec Decoder → Audio.
Also benchmarks individual component quality and speed.

Usage:
  python train/sonata/benchmark_e2e.py \
    --codec-ckpt checkpoints/codec/codec_final.pt \
    --lm-ckpt checkpoints/lm/lm_final.pt \
    --flow-ckpt checkpoints/flow/flow_final.pt \
    --audio-dir data/LibriSpeech/dev-clean
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np

from config import CodecConfig, SemanticLMConfig, FlowConfig
from codec import SonataCodec
from semantic_lm import SonataSemanticLM
from flow import SonataFlow


def compute_si_sdr(ref, est):
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    dot = np.dot(ref, est)
    s_ref_sq = np.dot(ref, ref)
    if s_ref_sq < 1e-10:
        return 0.0
    s_target = (dot / s_ref_sq) * ref
    e_noise = est - s_target
    return float(10 * np.log10(np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-10)))


def compute_stoi(ref, est, sr=24000):
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    min_len = min(len(ref), len(est))
    ref, est = ref[:min_len], est[:min_len]
    n_frames = (min_len - frame_len) // hop
    if n_frames < 1:
        return 0.0
    correlations = []
    for i in range(n_frames):
        s = i * hop
        r, e = ref[s:s+frame_len], est[s:s+frame_len]
        c = np.mean((r - np.mean(r)) * (e - np.mean(e))) / (np.std(r) * np.std(e) + 1e-10)
        correlations.append(max(0, c))
    return float(np.mean(correlations))


def compute_mcd(ref, est, sr=24000, n_mfcc=13):
    def mfcc(audio, sr, n_mfcc=13, n_fft=1024, hop=256, n_mels=80):
        padded = np.pad(audio, (n_fft // 2, n_fft // 2))
        frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
        spec = np.abs(np.fft.rfft(frames * np.hanning(n_fft)))

        mel_fb = np.zeros((n_mels, n_fft // 2 + 1))
        mel_pts = np.linspace(0, 2595 * np.log10(1 + sr / 2 / 700), n_mels + 2)
        hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
        bins = np.floor((hz_pts / sr) * n_fft).astype(int)
        for i in range(n_mels):
            lo, mid, hi = bins[i], bins[i+1], bins[i+2]
            for j in range(lo, mid):
                mel_fb[i, j] = (j - lo) / max(1, mid - lo)
            for j in range(mid, hi):
                mel_fb[i, j] = (hi - j) / max(1, hi - mid)

        from scipy.fft import dct
        return dct(np.log(np.dot(spec, mel_fb.T).clip(min=1e-7)), type=2, n=n_mfcc, axis=-1)

    try:
        r, e = mfcc(ref, sr, n_mfcc), mfcc(est, sr, n_mfcc)
    except Exception:
        return 99.0
    n = min(r.shape[0], e.shape[0])
    diff = r[:n, 1:] - e[:n, 1:]
    return float((10 / np.log(10)) * np.sqrt(2) * np.mean(np.sqrt(np.sum(diff**2, axis=-1))))


@torch.no_grad()
def generate_semantic_tokens(lm, text_ids, n_frames, temperature=0.8, top_k=50):
    """Autoregressive semantic token generation from text."""
    B = text_ids.shape[0]
    device = text_ids.device

    if text_ids.shape[1] < n_frames:
        repeats = (n_frames // text_ids.shape[1]) + 1
        text_ids = text_ids.repeat(1, repeats)[:, :n_frames]
    else:
        text_ids = text_ids[:, :n_frames]

    tokens = torch.ones(B, 1, dtype=torch.long, device=device)

    for i in range(n_frames):
        text_in = text_ids[:, :tokens.shape[1]]
        sem_in = tokens
        if text_in.shape[1] != sem_in.shape[1]:
            min_t = min(text_in.shape[1], sem_in.shape[1])
            text_in = text_in[:, :min_t]
            sem_in = sem_in[:, :min_t]

        logits, _ = lm(text_in, sem_in)
        next_logits = logits[:, -1, :] / temperature

        if top_k > 0:
            topk_vals, _ = next_logits.topk(top_k)
            next_logits[next_logits < topk_vals[:, -1:]] = float('-inf')

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens[:, 1:]


def benchmark_codec_reconstruction(model, audio_files, device, sr=24000):
    """Benchmark codec encode-decode quality."""
    results = {"si_sdr": [], "stoi": [], "mcd": [], "rtf": []}

    for f in audio_files:
        data, file_sr = sf.read(str(f), dtype='float32')
        audio = torch.from_numpy(data)
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)
        if file_sr != sr:
            new_len = int(audio.shape[0] * sr / file_sr)
            audio = F.interpolate(audio.unsqueeze(0).unsqueeze(0), size=new_len,
                                  mode='linear', align_corners=False).squeeze()

        duration = audio.shape[0] / sr
        t0 = time.perf_counter()
        rec, tokens, _ = model(audio.unsqueeze(0).to(device))
        elapsed = time.perf_counter() - t0

        rec = rec.squeeze(0).detach().cpu()
        min_len = min(rec.shape[0], audio.shape[0])
        ref, est = audio[:min_len].numpy(), rec[:min_len].numpy()

        results["si_sdr"].append(compute_si_sdr(ref, est))
        results["stoi"].append(compute_stoi(ref, est, sr))
        results["mcd"].append(compute_mcd(ref, est, sr))
        results["rtf"].append(elapsed / duration)

    return results


def benchmark_e2e_tts(codec, lm, flow, test_sentences, device, sr=24000):
    """Benchmark end-to-end TTS: text → semantic → acoustic → audio."""
    results = {"rtf": [], "gen_tokens": [], "duration_sec": []}

    for text in test_sentences:
        text_ids = torch.tensor([[ord(c) % 32000 for c in text]], device=device)
        n_frames = max(25, len(text) * 3)

        t0 = time.perf_counter()
        semantic_tokens = generate_semantic_tokens(lm, text_ids, n_frames)
        t_lm = time.perf_counter() - t0

        t1 = time.perf_counter()
        acoustic_latents = flow.sample(semantic_tokens, n_steps=8)
        t_flow = time.perf_counter() - t1

        semantic_codes = codec.fsq.indices_to_codes(semantic_tokens)
        T_min = min(semantic_codes.shape[1], acoustic_latents.shape[1])
        semantic_codes = semantic_codes[:, :T_min]
        acoustic_latents = acoustic_latents[:, :T_min]

        t2 = time.perf_counter()
        audio = codec.decoder(semantic_codes, acoustic_latents)
        t_dec = time.perf_counter() - t2

        total_time = t_lm + t_flow + t_dec
        duration = audio.shape[-1] / sr

        results["rtf"].append(total_time / max(duration, 0.01))
        results["gen_tokens"].append(n_frames)
        results["duration_sec"].append(duration)

    return results


def print_results_table(title, results, targets):
    """Print formatted results table."""
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")

    for metric, info in targets.items():
        if metric not in results or not results[metric]:
            continue
        vals = results[metric]
        mean = np.mean(vals)
        std = np.std(vals)
        p50 = np.percentile(vals, 50)
        p95 = np.percentile(vals, 95)
        target = info["target"]
        unit = info.get("unit", "")
        higher = info.get("higher_better", True)
        grade = "PASS" if (mean >= target if higher else mean <= target) else "FAIL"

        print(f"  {metric:8s}: mean={mean:8.3f} ± {std:5.3f}"
              f"  (p50={p50:7.3f}, p95={p95:7.3f})"
              f"  target={target:6.2f}{unit:2s}  [{grade}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec-ckpt", required=True)
    parser.add_argument("--lm-ckpt", default="")
    parser.add_argument("--flow-ckpt", default="")
    parser.add_argument("--audio-dir", default="train/data/LibriSpeech/dev-clean")
    parser.add_argument("--output", default="bench_output/sonata_e2e_bench.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-files", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load Codec ──
    print("Loading Sonata Codec...")
    ckpt = torch.load(args.codec_ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    codec_cfg = CodecConfig(**{k: v for k, v in cfg_dict.items()
                                if k in CodecConfig.__dataclass_fields__})
    codec = SonataCodec(codec_cfg).to(device)
    codec.load_state_dict(ckpt["model"])
    codec.eval()
    n_codec = sum(p.numel() for p in codec.parameters())
    print(f"  Codec: {n_codec/1e6:.1f}M params")

    # ── Load LM ──
    lm = None
    if args.lm_ckpt and Path(args.lm_ckpt).exists():
        print("Loading Sonata Semantic LM...")
        lm_ckpt = torch.load(args.lm_ckpt, map_location="cpu", weights_only=False)
        lm_cfg = SemanticLMConfig(**{k: v for k, v in lm_ckpt["config"].items()
                                      if k in SemanticLMConfig.__dataclass_fields__})
        lm = SonataSemanticLM(lm_cfg).to(device)
        lm.load_state_dict(lm_ckpt["model"])
        lm.eval()
        n_lm = sum(p.numel() for p in lm.parameters())
        print(f"  LM: {n_lm/1e6:.1f}M params")

    # ── Load Flow ──
    flow = None
    if args.flow_ckpt and Path(args.flow_ckpt).exists():
        print("Loading Sonata Flow...")
        flow_ckpt = torch.load(args.flow_ckpt, map_location="cpu", weights_only=False)
        flow_cfg = FlowConfig(**FlowConfig._normalize_loaded_dict(flow_ckpt["config"]))
        flow = SonataFlow(flow_cfg).to(device)
        flow.load_state_dict(flow_ckpt["model"])
        flow.eval()
        n_flow = sum(p.numel() for p in flow.parameters())
        print(f"  Flow: {n_flow/1e6:.1f}M params")

    # ── Benchmark 1: Codec Reconstruction ──
    print("\n" + "─" * 72)
    print("  BENCHMARK 1: CODEC RECONSTRUCTION")
    print("─" * 72)

    audio_dir = Path(args.audio_dir)
    audio_files = sorted(
        list(audio_dir.rglob("*.flac")) + list(audio_dir.rglob("*.wav"))
    )[:args.max_files]
    print(f"  Testing on {len(audio_files)} files...")

    codec_results = benchmark_codec_reconstruction(codec, audio_files, device)

    codec_targets = {
        "si_sdr": {"target": 15.0, "unit": "dB", "higher_better": True},
        "stoi": {"target": 0.95, "unit": "", "higher_better": True},
        "mcd": {"target": 4.0, "unit": "dB", "higher_better": False},
        "rtf": {"target": 0.2, "unit": "x", "higher_better": False},
    }
    print_results_table("CODEC RECONSTRUCTION QUALITY", codec_results, codec_targets)

    total_audio = len(audio_files) * 6  # ~6s average
    avg_rtf = np.mean(codec_results["rtf"])
    print(f"\n  Avg RTF: {avg_rtf:.4f}x ({1/avg_rtf:.0f}x realtime)")

    # ── Benchmark 2: End-to-End TTS ──
    e2e_results = None
    if lm is not None and flow is not None:
        print("\n" + "─" * 72)
        print("  BENCHMARK 2: END-TO-END TTS")
        print("─" * 72)

        test_sentences = [
            "Hello, how are you doing today?",
            "The quick brown fox jumps over the lazy dog.",
            "Speech synthesis is a fascinating technology.",
            "Pocket voice runs entirely on Apple Silicon.",
            "Zero Python, zero interpreter overhead.",
        ]

        e2e_results = benchmark_e2e_tts(codec, lm, flow, test_sentences, device)
        print(f"  Generated {len(test_sentences)} utterances")
        for i, (sent, rtf, dur) in enumerate(zip(
            test_sentences, e2e_results["rtf"], e2e_results["duration_sec"]
        )):
            print(f"  [{i+1}] \"{sent[:40]}...\" → {dur:.2f}s audio, RTF={rtf:.3f}x")

        e2e_targets = {
            "rtf": {"target": 1.0, "unit": "x", "higher_better": False},
        }
        print_results_table("END-TO-END TTS SPEED", e2e_results, e2e_targets)

    # ── Summary ──
    print(f"\n{'='*72}")
    print(f"  COMPREHENSIVE BENCHMARK SUMMARY")
    print(f"{'='*72}")

    n_pass = sum(1 for m, t in codec_targets.items()
                 if m in codec_results and
                 (np.mean(codec_results[m]) >= t["target"] if t["higher_better"]
                  else np.mean(codec_results[m]) <= t["target"]))
    n_total = len(codec_targets)

    print(f"  Codec: {n_pass}/{n_total} metrics pass")
    print(f"  Architecture: Dual encoder (Conv1d waveform + Conformer semantic)")
    print(f"  Decoder: ConvTranspose (Encodec-style)")
    print(f"  FSQ codebook: {codec_cfg.fsq_codebook_size} entries")
    print(f"  Frame rate: {codec_cfg.frame_rate} Hz")
    print(f"  Speed: {1/avg_rtf:.0f}x realtime (codec)")

    grade = "A" if n_pass >= 4 else "B" if n_pass >= 3 else "C" if n_pass >= 2 else "D"
    print(f"\n  OVERALL GRADE: {grade}")
    print(f"{'='*72}")

    # Save results
    output = {
        "codec": {
            "n_params": n_codec,
            "config": cfg_dict,
            "results": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                            "p50": float(np.percentile(v, 50)),
                            "p95": float(np.percentile(v, 95))}
                        for k, v in codec_results.items()},
            "targets": codec_targets,
        },
        "grade": grade,
    }
    if e2e_results:
        output["e2e_tts"] = {
            "results": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                        for k, v in e2e_results.items()},
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
