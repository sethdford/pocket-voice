"""Validate Sonata V3 pipeline: Flow v3 → Vocoder → audio → (optional) STT round-trip WER.

Usage:
  # Generate audio from text
  python validate_v3.py \
    --flow-ckpt checkpoints/flow_v3_ljspeech/flow_v3_best.pt \
    --vocoder-ckpt checkpoints/vocoder_ljspeech/vocoder_best.pt \
    --text "Hello, this is a test of the Sonata speech synthesis system." \
    --output output.wav

  # Round-trip WER validation (requires Whisper or Conformer STT)
  python validate_v3.py \
    --flow-ckpt checkpoints/flow_v3_ljspeech/flow_v3_best.pt \
    --vocoder-ckpt checkpoints/vocoder_ljspeech/vocoder_best.pt \
    --round-trip \
    --output report.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from config import FlowV3Config, VocoderConfig
from flow_v3 import SonataFlowV3
from vocoder import SonataVocoder


TEST_SENTENCES = [
    "Hello, this is a test of the Sonata text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "She sells seashells by the seashore.",
    "Today is a beautiful day for a walk in the park.",
]


def text_to_char_ids(text: str, vocab_size: int = 256) -> torch.Tensor:
    return torch.tensor([ord(c) % vocab_size for c in text], dtype=torch.long)


def estimate_n_frames(text: str, chars_per_sec: float = 14.0,
                      frame_rate: float = 50.0) -> int:
    """Estimate mel frames needed for text. ~14 chars/sec at 50Hz = ~3.6 frames/char."""
    duration_sec = len(text) / chars_per_sec
    return max(int(duration_sec * frame_rate), 25)


def load_flow(ckpt_path: str, device: torch.device) -> SonataFlowV3:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = FlowV3Config(**{k: v for k, v in cfg_dict.items()
                          if k in FlowV3Config.__dataclass_fields__})
    model = SonataFlowV3(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def load_vocoder(ckpt_path: str, device: torch.device) -> SonataVocoder:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = VocoderConfig(**{k: v for k, v in cfg_dict.items()
                           if k in VocoderConfig.__dataclass_fields__})
    model = SonataVocoder(cfg).to(device)
    # Vocoder checkpoints may save generator state separately (from GAN training)
    gen_state = ckpt.get("generator", None)
    if gen_state:
        model.generator.load_state_dict(gen_state, strict=False)
    else:
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    return model


def synthesize(flow: SonataFlowV3, vocoder: SonataVocoder,
               text: str, device: torch.device,
               n_steps: int = 8, cfg_scale: float = 2.0,
               speaker_id: int = None) -> np.ndarray:
    """Text → mel → waveform."""
    char_ids = text_to_char_ids(text).unsqueeze(0).to(device)

    speaker_ids = None
    if speaker_id is not None and hasattr(flow, 'cfg') and flow.cfg.n_speakers > 0:
        speaker_ids = torch.tensor([speaker_id % flow.cfg.n_speakers], device=device)

    with torch.no_grad():
        mel = flow.sample(char_ids, n_frames=0, n_steps=n_steps,
                          cfg_scale=cfg_scale, speaker_ids=speaker_ids)
        audio = vocoder.generate(mel)
    return audio.squeeze(0).cpu().numpy()


def compute_audio_stats(audio: np.ndarray, sr: int = 24000) -> dict:
    """Basic audio quality stats."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    duration = len(audio) / sr
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))
    return {
        "duration_sec": round(duration, 3),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "zero_crossing_rate": round(zcr, 4),
        "is_silence": rms < 1e-4,
        "is_clipped": peak > 0.99,
        "is_speech_like": 0.01 < zcr < 0.30 and rms > 0.001,
    }


def run_stt_roundtrip(audio: np.ndarray, sr: int = 24000) -> str:
    """Try to transcribe audio using available STT (Whisper or soundfile)."""
    try:
        import whisper
        tmp_path = "/tmp/sonata_v3_validate.wav"
        sf.write(tmp_path, audio, sr)
        model = whisper.load_model("tiny")
        result = model.transcribe(tmp_path)
        return result["text"].strip()
    except ImportError:
        return ""


def word_error_rate(ref: str, hyp: str) -> float:
    """Simple WER via Levenshtein on word sequences."""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def main():
    parser = argparse.ArgumentParser(description="Validate Sonata V3 pipeline")
    parser.add_argument("--flow-ckpt", required=True, help="Flow v3 checkpoint")
    parser.add_argument("--vocoder-ckpt", required=True, help="Vocoder checkpoint")
    parser.add_argument("--text", default="", help="Single text to synthesize")
    parser.add_argument("--output", default="validate_v3_output.wav",
                        help="Output WAV path (or .json for round-trip report)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--n-steps", type=int, default=8, help="ODE solver steps")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="CFG guidance scale")
    parser.add_argument("--round-trip", action="store_true",
                        help="Run STT round-trip on test sentences")
    parser.add_argument("--sample-rate", type=int, default=24000)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading Flow v3 from {args.flow_ckpt}...")
    flow = load_flow(args.flow_ckpt, device)
    print(f"Loading Vocoder from {args.vocoder_ckpt}...")
    vocoder = load_vocoder(args.vocoder_ckpt, device)

    if args.round_trip:
        results = []
        total_wer = 0.0
        print(f"\n{'='*60}")
        print(f"  SONATA V3 ROUND-TRIP VALIDATION")
        print(f"{'='*60}\n")

        for i, text in enumerate(TEST_SENTENCES):
            print(f"  [{i+1}/{len(TEST_SENTENCES)}] \"{text}\"")
            t0 = time.time()
            audio = synthesize(flow, vocoder, text, device,
                               n_steps=args.n_steps, cfg_scale=args.cfg_scale)
            gen_time = time.time() - t0
            stats = compute_audio_stats(audio, args.sample_rate)

            wav_path = f"/tmp/sonata_v3_test_{i}.wav"
            sf.write(wav_path, audio, args.sample_rate)

            transcript = run_stt_roundtrip(audio, args.sample_rate)
            wer = word_error_rate(text, transcript) if transcript else -1

            result = {
                "text": text,
                "transcript": transcript,
                "wer": round(wer, 4) if wer >= 0 else None,
                "gen_time_sec": round(gen_time, 3),
                "rtf": round(gen_time / stats["duration_sec"], 3) if stats["duration_sec"] > 0 else 0,
                **stats,
            }
            results.append(result)

            if wer >= 0:
                total_wer += wer
                print(f"    WER: {wer:.1%} | RTF: {result['rtf']:.3f} | "
                      f"Duration: {stats['duration_sec']:.1f}s | "
                      f"Speech-like: {stats['is_speech_like']}")
            else:
                print(f"    RTF: {result['rtf']:.3f} | Duration: {stats['duration_sec']:.1f}s | "
                      f"Speech-like: {stats['is_speech_like']} (no STT available)")

        n_with_stt = sum(1 for r in results if r["wer"] is not None)
        avg_wer = total_wer / max(n_with_stt, 1) if n_with_stt > 0 else None

        report = {
            "model": "sonata_v3",
            "flow_ckpt": args.flow_ckpt,
            "vocoder_ckpt": args.vocoder_ckpt,
            "n_steps": args.n_steps,
            "cfg_scale": args.cfg_scale,
            "avg_wer": round(avg_wer, 4) if avg_wer is not None else None,
            "speech_like_pct": sum(1 for r in results if r["is_speech_like"]) / len(results),
            "results": results,
        }

        out_path = args.output if args.output.endswith(".json") else args.output.rsplit(".", 1)[0] + ".json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n  {'='*40}")
        if avg_wer is not None:
            print(f"  Average WER: {avg_wer:.1%}")
        print(f"  Speech-like: {report['speech_like_pct']:.0%}")
        print(f"  Report: {out_path}")

    else:
        text = args.text or TEST_SENTENCES[0]
        print(f"Synthesizing: \"{text}\"")
        t0 = time.time()
        audio = synthesize(flow, vocoder, text, device,
                           n_steps=args.n_steps, cfg_scale=args.cfg_scale)
        gen_time = time.time() - t0
        stats = compute_audio_stats(audio, args.sample_rate)

        sf.write(args.output, audio, args.sample_rate)
        print(f"  Saved: {args.output}")
        print(f"  Duration: {stats['duration_sec']:.1f}s")
        print(f"  Gen time: {gen_time:.3f}s (RTF: {gen_time / max(stats['duration_sec'], 0.01):.3f})")
        print(f"  RMS: {stats['rms']:.6f}, Peak: {stats['peak']:.6f}")
        print(f"  ZCR: {stats['zero_crossing_rate']:.4f}")
        print(f"  Speech-like: {stats['is_speech_like']}")


if __name__ == "__main__":
    main()
