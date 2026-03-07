"""Train Sonata Vocoder (BigVGAN-lite) for mel → waveform synthesis.

Pairs with Flow v3 which outputs 80-bin mel spectrograms at 50 Hz.
The vocoder converts these to 24 kHz waveform audio.

Training data: directory of .wav files (24 kHz mono).
The training loop extracts mel spectrograms on-the-fly from audio.

Usage:
  python train_vocoder.py \
    --manifest data/libritts_r_full_manifest.jsonl \
    --device cuda \
    --epochs 50 \
    --batch-size 16 \
    --grad-accum 2 \
    --amp

Data format: directory of .wav files (mono, any sample rate — resampled to 24 kHz).
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mel_utils import mel_spectrogram
from vocoder import SonataVocoder, VocoderConfig
from modules import cosine_lr, TrainingLog


class AudioDataset(Dataset):
    """Load audio files and extract mel + waveform pairs."""

    def __init__(self, data_dir: str = None, manifest: str = None,
                 segment_length: int = 24000,
                 sample_rate: int = 24000, n_fft: int = 1024,
                 hop_length: int = 480, n_mels: int = 80):
        if manifest:
            import json
            self.files = []
            with open(manifest) as f:
                for line in f:
                    entry = json.loads(line)
                    self.files.append(Path(entry["audio"]))
            print(f"  Loaded {len(self.files)} files from manifest")
        elif data_dir:
            self.files = sorted(Path(data_dir).glob("**/*.wav"))
            if not self.files:
                self.files = sorted(Path(data_dir).glob("**/*.pt"))
        else:
            raise ValueError("Must provide --data-dir or --manifest")
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        if path.suffix == ".pt":
            data = torch.load(path, weights_only=True, map_location="cpu")
            if "audio" in data:
                audio = data["audio"].float()
            elif "waveform" in data:
                audio = data["waveform"].float()
            else:
                raise KeyError(f"Expected 'audio' or 'waveform' key in {path}")
        else:
            import soundfile as sf
            import numpy as np
            audio_np, sr = sf.read(str(path), dtype="float32")
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            audio = torch.from_numpy(audio_np)
            if sr != self.sample_rate:
                # Simple linear interpolation resampling
                ratio = self.sample_rate / sr
                new_len = int(len(audio) * ratio)
                audio = torch.nn.functional.interpolate(
                    audio.unsqueeze(0).unsqueeze(0), size=new_len, mode="linear",
                    align_corners=False).squeeze()

        # Random crop to segment_length
        if audio.shape[-1] > self.segment_length:
            start = torch.randint(0, audio.shape[-1] - self.segment_length, (1,)).item()
            audio = audio[start:start + self.segment_length]
        else:
            audio = F.pad(audio, (0, self.segment_length - audio.shape[-1]))

        mel = mel_spectrogram(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
        )
        return mel, audio


def train(args):
    device = torch.device(args.device)
    cfg = VocoderConfig(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
    )

    model = SonataVocoder(cfg).to(device)

    g_params = sum(p.numel() for p in model.generator.parameters())
    d_params = (sum(p.numel() for p in model.mpd.parameters()) +
                sum(p.numel() for p in model.msd.parameters()))
    print(f"Generator: {g_params/1e6:.1f}M params")
    print(f"Discriminators: {d_params/1e6:.1f}M params")

    opt_g = torch.optim.AdamW(model.generator.parameters(),
                               lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01)
    opt_d = torch.optim.AdamW(
        list(model.mpd.parameters()) + list(model.msd.parameters()) +
        list(model.mrstft.parameters()),
        lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01)

    # AMP setup (CUDA only — bfloat16 for GAN stability, no scaler needed)
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    if use_amp:
        print(f"  AMP: ON (bfloat16)")

    dataset = AudioDataset(data_dir=args.data_dir,
                           manifest=args.manifest,
                           segment_length=args.segment_length,
                           sample_rate=args.sample_rate,
                           n_fft=cfg.n_fft,
                           hop_length=cfg.hop_length,
                           n_mels=cfg.n_mels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), drop_last=True)

    print(f"Dataset: {len(dataset)} files")
    print(f"Batch size: {args.batch_size}, Grad accum: {args.grad_accum}, "
          f"Effective batch: {args.batch_size * args.grad_accum}")
    print(f"Epochs: {args.epochs}")

    os.makedirs(args.output_dir, exist_ok=True)
    tlog = TrainingLog(os.path.join(args.output_dir, "losses.jsonl"))
    total_steps = len(loader) * args.epochs
    step = 0
    t0 = time.time()

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt_g.load_state_dict(ckpt["opt_g"])
        if not args.reset_d_optimizer:
            opt_d.load_state_dict(ckpt["opt_d"])
        else:
            print("  Discriminator optimizer reset (--reset-d-optimizer)")
        step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.resume} at step {step}, epoch {start_epoch}")

    # Mel warmup is relative to current step (so it works after resume)
    mel_warmup_end = step + args.mel_warmup_steps
    print(f"  Mel-only warmup until step {mel_warmup_end} "
          f"(D lr ratio: {args.d_lr_ratio})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_g_loss, epoch_d_loss = 0.0, 0.0

        for batch_idx, (mel, audio) in enumerate(loader):
            mel = mel.to(device)      # (B, T_frames, n_mels)
            audio = audio.to(device)  # (B, T_samples)

            lr = cosine_lr(step, args.warmup_steps, args.lr, args.lr * 0.01, total_steps)
            for pg in opt_g.param_groups:
                pg["lr"] = lr
            for pg in opt_d.param_groups:
                pg["lr"] = lr * args.d_lr_ratio

            in_mel_warmup = step < mel_warmup_end

            # Discriminator step — skip during mel-only warmup
            if not in_mel_warmup:
                r1_w = 0.1 if step % 16 == 0 else 0.0
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    d_result = model.training_step_d(mel, audio, r1_weight=r1_w)
                if not torch.isfinite(d_result["d_loss"]):
                    print(f"  [WARN] step {step}: NaN/Inf D loss, skipping")
                    step += 1
                    continue
                d_loss_scaled = d_result["d_loss"] / args.grad_accum
                d_loss_scaled.backward()
            else:
                d_result = {"d_loss": torch.tensor(0.0), "r1": 0.0}

            # Generator step — mel-only during warmup
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                if in_mel_warmup:
                    g_result = model.training_step_g_mel_only(mel, audio)
                else:
                    g_result = model.training_step_g(mel, audio)
            if not torch.isfinite(g_result["g_loss"]):
                print(f"  [WARN] step {step}: NaN/Inf G loss, skipping")
                step += 1
                continue
            g_loss_scaled = g_result["g_loss"] / args.grad_accum
            g_loss_scaled.backward()

            # Gradient accumulation: only step every N batches
            if (batch_idx + 1) % args.grad_accum == 0:
                if not in_mel_warmup:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.mpd.parameters()) + list(model.msd.parameters()) +
                        list(model.mrstft.parameters()), 1.0)
                    opt_d.step()
                    opt_d.zero_grad()

                torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
                opt_g.step()
                opt_g.zero_grad()

            epoch_g_loss += g_result["g_loss"].item()
            d_loss_val = d_result["d_loss"].item() if hasattr(d_result["d_loss"], "item") else d_result["d_loss"]
            epoch_d_loss += d_loss_val
            step += 1

            if step <= 3 or (in_mel_warmup and step % 100 == 0):
                phase = "[mel-warmup]" if in_mel_warmup else "[warmup]"
                print(f"  {phase} step {step}: G={g_result['g_loss'].item():.4f} "
                      f"mel={g_result['mel_loss']:.4f}", flush=True)

            if step % args.log_interval == 0:
                elapsed = time.time() - t0
                sps = args.log_interval / elapsed
                d_val = d_result["d_loss"].item() if hasattr(d_result["d_loss"], "item") else d_result["d_loss"]
                print(f"  step {step}: G={g_result['g_loss'].item():.4f} "
                      f"(adv={g_result['adv_loss']:.4f} fm={g_result['fm_loss']:.4f} "
                      f"mel={g_result['mel_loss']:.4f}) "
                      f"D={d_val:.4f} lr={lr:.2e} "
                      f"| {sps:.1f} steps/s")
                tlog.log(step=step, g_loss=g_result["g_loss"].item(),
                         adv_loss=g_result["adv_loss"], fm_loss=g_result["fm_loss"],
                         mel_loss=g_result["mel_loss"],
                         d_loss=d_val, lr=lr,
                         steps_per_sec=sps)
                t0 = time.time()

            # Periodic MPS cache flush to prevent memory fragmentation
            if step % 500 == 0 and device.type == "mps":
                torch.mps.empty_cache()

            # Step-based checkpoint saving
            if args.save_steps > 0 and step % args.save_steps == 0:
                ckpt_path = os.path.join(args.output_dir, f"vocoder_step{step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "generator": model.generator.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "config": cfg.__dict__,
                }, ckpt_path)
                print(f"  Saved {ckpt_path}")

        n_batches = max(len(loader), 1)
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"G={epoch_g_loss/n_batches:.4f} D={epoch_d_loss/n_batches:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"vocoder_epoch{epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "generator": model.generator.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "step": step,
                "epoch": epoch + 1,
                "config": cfg.__dict__,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

    # Final save (generator only for inference)
    final_path = os.path.join(args.output_dir, "vocoder_generator.pt")
    torch.save({
        "generator": model.generator.state_dict(),
        "config": cfg.__dict__,
    }, final_path)
    print(f"Final generator saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sonata Vocoder")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--manifest", default=None,
                        help="JSONL manifest with 'audio' paths (faster than --data-dir glob)")
    parser.add_argument("--output-dir", default="checkpoints/vocoder")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (CUDA only)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--segment-length", type=int, default=24000)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--hop-length", type=int, default=480)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=5000,
                        help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--mel-warmup-steps", type=int, default=5000,
                        help="Mel-only warmup steps (no adversarial loss)")
    parser.add_argument("--d-lr-ratio", type=float, default=0.5,
                        help="Discriminator LR = G LR * this ratio (prevents D collapse)")
    parser.add_argument("--reset-d-optimizer", action="store_true",
                        help="Reset discriminator optimizer state on resume")
    args = parser.parse_args()
    train(args)
