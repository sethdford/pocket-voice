"""Sonata TTS inference: text → Flow v3 → Vocoder → waveform.

Production-quality synthesis pipeline with speaker selection, speed control,
CFG guidance, EPSS sampling, and streaming output.

Usage:
  # Single utterance
  python synthesize.py --text "Hello world" --output hello.wav

  # With speaker selection (from LibriTTS-R speaker ID)
  python synthesize.py --text "Hello" --speaker 1089 --output hello.wav

  # Batch from file
  python synthesize.py --input sentences.txt --output-dir output/

  # High quality (more ODE steps + Heun solver)
  python synthesize.py --text "Hello" --n-steps 16 --cfg-scale 2.5 --output hq.wav

  # Fast (fewer steps + EPSS)
  python synthesize.py --text "Hello" --n-steps 4 --epss --output fast.wav
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import FlowV3Config, FlowV3LargeConfig, VocoderConfig
from flow_v3 import SonataFlowV3
from vocoder import SonataVocoder
from modules import epss_schedule

try:
    from g2p import PhonemeFrontend
    HAS_G2P = True
except ImportError:
    HAS_G2P = False


class SonataEngine:
    """End-to-end Sonata TTS engine: text → audio."""

    def __init__(self, flow_ckpt: str, vocoder_ckpt: str,
                 device: str = "mps", use_ema: bool = True,
                 model_size: str = "base", g2p: Optional["PhonemeFrontend"] = None):
        self.device = torch.device(device)
        self.sample_rate = 24000
        self.g2p = g2p
        self.flow = self._load_flow(flow_ckpt, use_ema, model_size)
        self.vocoder = self._load_vocoder(vocoder_ckpt)

    def _load_flow(self, path: str, use_ema: bool, model_size: str) -> SonataFlowV3:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config", {})
        cfg_cls = FlowV3LargeConfig if model_size == "large" else FlowV3Config
        cfg = cfg_cls(**{k: v for k, v in cfg_dict.items()
                         if k in cfg_cls.__dataclass_fields__})
        model = SonataFlowV3(cfg).to(self.device)
        state = ckpt.get("ema", ckpt.get("model", ckpt)) if use_ema else ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()
        self.flow_cfg = cfg
        return model

    def _load_vocoder(self, path: str) -> SonataVocoder:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config", {})
        cfg = VocoderConfig(**{k: v for k, v in cfg_dict.items()
                               if k in VocoderConfig.__dataclass_fields__})
        model = SonataVocoder(cfg).to(self.device)
        gen_state = ckpt.get("generator", None)
        if gen_state:
            model.generator.load_state_dict(gen_state, strict=False)
        else:
            model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        model.eval()
        self.sample_rate = cfg.sample_rate
        return model

    def text_to_char_ids(self, text: str) -> torch.Tensor:
        """Encode text to token IDs (phonemes or chars)."""
        if self.g2p is not None:
            return self.g2p.encode(text, add_bos=True, add_eos=True)
        vs = getattr(self.flow_cfg, "char_vocab_size", 256) if hasattr(self, "flow_cfg") else 256
        return torch.tensor([ord(c) % vs for c in text], dtype=torch.long)

    @torch.no_grad()
    def synthesize(self, text: str,
                   speaker_id: Optional[int] = None,
                   n_steps: int = 8,
                   cfg_scale: float = 2.0,
                   speed: float = 1.0,
                   use_epss: bool = False,
                   use_heun: bool = False) -> np.ndarray:
        """Generate audio from text.

        Returns: numpy array of audio samples at self.sample_rate.
        """
        char_ids = self.text_to_char_ids(text)
        if char_ids.dim() == 1:
            char_ids = char_ids.unsqueeze(0)
        char_ids = char_ids.to(self.device)

        speaker_ids = None
        if speaker_id is not None and self.flow_cfg.n_speakers > 0:
            sid = speaker_id % self.flow_cfg.n_speakers
            speaker_ids = torch.tensor([sid], device=self.device)

        step_schedule = epss_schedule(n_steps) if use_epss else None

        mel = self.flow.sample(
            char_ids, n_frames=0, n_steps=n_steps,
            speaker_ids=speaker_ids, cfg_scale=cfg_scale,
            speed=speed, step_schedule=step_schedule,
            use_heun=use_heun,
        )

        audio = self.vocoder.generate(mel)
        return audio.squeeze().cpu().numpy()

    @torch.no_grad()
    def synthesize_batch(self, texts: list,
                         speaker_id: Optional[int] = None,
                         n_steps: int = 8,
                         cfg_scale: float = 2.0,
                         speed: float = 1.0,
                         use_heun: bool = False) -> list:
        """Synthesize multiple utterances. Returns list of numpy arrays."""
        results = []
        for text in texts:
            audio = self.synthesize(text, speaker_id, n_steps, cfg_scale, speed,
                                    use_heun=use_heun)
            results.append(audio)
        return results


def compute_stats(audio: np.ndarray, sr: int) -> dict:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    duration = len(audio) / sr
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))
    return {
        "duration_sec": round(duration, 3),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "zcr": round(zcr, 4),
        "speech_like": 0.01 < zcr < 0.30 and rms > 0.001,
    }


def main():
    parser = argparse.ArgumentParser(description="Sonata TTS Synthesis")
    parser.add_argument("--flow-ckpt", required=True, help="Flow v3 checkpoint path")
    parser.add_argument("--vocoder-ckpt", required=True, help="Vocoder checkpoint path")
    parser.add_argument("--text", default="", help="Text to synthesize")
    parser.add_argument("--input", default="", help="File with one sentence per line")
    parser.add_argument("--output", default="output.wav", help="Output WAV file")
    parser.add_argument("--output-dir", default="", help="Output directory for batch")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--speaker", type=int, default=None, help="Speaker ID")
    parser.add_argument("--n-steps", type=int, default=8, help="ODE solver steps")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="CFG guidance scale")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor (0.5=slow, 2.0=fast)")
    parser.add_argument("--epss", action="store_true", help="Use EPSS sampling schedule")
    parser.add_argument("--no-ema", action="store_true", help="Don't use EMA weights")
    parser.add_argument("--phonemes", action="store_true",
                        help="Use PhonemeFrontend (g2p) to encode text")
    parser.add_argument("--model-size", choices=["base", "large"], default="base",
                        help="Flow config: base or large")
    parser.add_argument("--heun", action="store_true", help="Use Heun 2nd-order ODE solver")
    args = parser.parse_args()

    g2p = None
    if args.phonemes:
        if not HAS_G2P:
            print("ERROR: --phonemes requires g2p module. Install phonemizer.")
            return 1
        g2p = PhonemeFrontend()

    engine = SonataEngine(
        args.flow_ckpt, args.vocoder_ckpt,
        device=args.device, use_ema=not args.no_ema,
        model_size=args.model_size, g2p=g2p,
    )
    print(f"Loaded: Flow ({sum(p.numel() for p in engine.flow.parameters())/1e6:.1f}M) + "
          f"Vocoder ({sum(p.numel() for p in engine.vocoder.generator.parameters())/1e6:.1f}M)")
    print(f"Sample rate: {engine.sample_rate} Hz")

    if args.input:
        texts = [line.strip() for line in open(args.input) if line.strip()]
        out_dir = args.output_dir or "synth_output"
        os.makedirs(out_dir, exist_ok=True)
        total_time = 0.0
        total_audio = 0.0

        for i, text in enumerate(texts):
            t0 = time.time()
            audio = engine.synthesize(
                text, args.speaker, args.n_steps,
                args.cfg_scale, args.speed, args.epss,
                use_heun=args.heun,
            )
            gen_time = time.time() - t0
            total_time += gen_time

            out_path = os.path.join(out_dir, f"{i:04d}.wav")
            sf.write(out_path, audio, engine.sample_rate)
            stats = compute_stats(audio, engine.sample_rate)
            total_audio += stats["duration_sec"]

            print(f"  [{i+1}/{len(texts)}] {stats['duration_sec']:.1f}s | "
                  f"RTF={gen_time/max(stats['duration_sec'],0.01):.3f} | "
                  f"speech={stats['speech_like']} | {out_path}")

        print(f"\nTotal: {total_audio:.1f}s audio in {total_time:.1f}s "
              f"(RTF={total_time/max(total_audio,0.01):.3f})")

    else:
        text = args.text or "Hello, this is the Sonata text to speech system."
        print(f"Text: \"{text}\"")

        t0 = time.time()
        audio = engine.synthesize(
            text, args.speaker, args.n_steps,
            args.cfg_scale, args.speed, args.epss,
            use_heun=args.heun,
        )
        gen_time = time.time() - t0

        sf.write(args.output, audio, engine.sample_rate)
        stats = compute_stats(audio, engine.sample_rate)

        print(f"Output: {args.output}")
        print(f"Duration: {stats['duration_sec']:.2f}s")
        print(f"Gen time: {gen_time:.3f}s (RTF={gen_time/max(stats['duration_sec'],0.01):.3f})")
        print(f"RMS={stats['rms']:.6f} Peak={stats['peak']:.6f} ZCR={stats['zcr']:.4f}")
        print(f"Speech-like: {stats['speech_like']}")


if __name__ == "__main__":
    main()
