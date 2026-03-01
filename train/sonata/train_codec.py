"""Train Sonata Codec with adversarial training for human-quality audio.

GAN training loop:
  1. Generator (SonataCodec): multi-scale STFT + mel + waveform + adversarial + feature matching
  2. Discriminator (MPD + MSD): hinge loss

This is what separates muffled toy codecs from crisp, natural-sounding human-quality speech.

Usage:
  python train/sonata/train_codec.py --audio-dir train/data/LibriSpeech/dev-clean --steps 50000
  python train/sonata/train_codec.py --manifest train/data/manifest_clean.jsonl --steps 200000
  python train/sonata/train_codec.py --synthetic --steps 5000  # Quick test
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import soundfile as sf

from config import CodecConfig, Codec12HzConfig
from codec import SonataCodec, MultiScaleSTFTLoss, MelReconstructionLoss, WavLMPerceptualLoss, fsq_entropy_loss
from codec_12hz import SonataCodec12Hz, Codec12HzLoss
from discriminator import (
    SonataDiscriminator,
    discriminator_loss,
    generator_adversarial_loss,
    feature_matching_loss,
    r1_gradient_penalty,
)
from ema import EMA
from modules import TrainingLog


class AudioAugmenter:
    """Audio augmentation for robust codec training.

    Speed perturbation: pitch-preserving time stretch (0.9x-1.1x)
    Additive noise: Gaussian at 15-30dB SNR
    """

    def __init__(self, speed_perturb: bool = True, additive_noise: bool = True,
                 speed_range=(0.9, 1.1), snr_range=(15.0, 30.0)):
        self.speed_perturb = speed_perturb
        self.additive_noise = additive_noise
        self.speed_range = speed_range
        self.snr_range = snr_range

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if self.speed_perturb and torch.rand(1).item() < 0.5:
            speed = self.speed_range[0] + torch.rand(1).item() * (self.speed_range[1] - self.speed_range[0])
            new_len = int(audio.shape[-1] / speed)
            audio = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear',
                align_corners=False
            ).squeeze()

        if self.additive_noise and torch.rand(1).item() < 0.3:
            snr_db = self.snr_range[0] + torch.rand(1).item() * (self.snr_range[1] - self.snr_range[0])
            rms_signal = audio.pow(2).mean().sqrt().clamp(min=1e-8)
            rms_noise = rms_signal * (10 ** (-snr_db / 20))
            noise = torch.randn_like(audio) * rms_noise
            audio = audio + noise

        return audio


class AudioDataset(Dataset):
    """Load audio files for codec training. Supports wav, flac, mp3."""

    def __init__(self, audio_dir: str, sample_rate: int = 24000,
                 segment_length: int = 24000, synthetic: bool = False,
                 n_synthetic: int = 1000, manifest: str = "",
                 augment: bool = False):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.synthetic = synthetic
        self.augmenter = AudioAugmenter() if augment else None

        if synthetic:
            self.length = n_synthetic
            self.files = []
        elif manifest:
            self.files = []
            with open(manifest) as f:
                for line in f:
                    entry = json.loads(line)
                    self.files.append(Path(entry["audio"]))
            self.length = len(self.files)
            print(f"[dataset] {self.length} utterances from manifest")
        else:
            audio_path = Path(audio_dir)
            self.files = sorted(
                list(audio_path.rglob("*.wav")) +
                list(audio_path.rglob("*.flac")) +
                list(audio_path.rglob("*.mp3"))
            )
            self.length = len(self.files)
            print(f"[dataset] Found {self.length} audio files in {audio_dir}")
        if self.augmenter:
            print(f"[dataset] Augmentation: speed perturb + additive noise")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.synthetic:
            t = torch.linspace(0, 1, self.segment_length)
            n_harmonics = torch.randint(2, 8, (1,)).item()
            audio = torch.zeros(self.segment_length)
            for _ in range(n_harmonics):
                freq = torch.randint(80, 4000, (1,)).item()
                amp = torch.rand(1).item() * 0.3
                phase = torch.rand(1).item() * 2 * math.pi
                audio += amp * torch.sin(2 * math.pi * freq * t + phase)
            audio += 0.01 * torch.randn_like(audio)
            audio = audio / (audio.abs().max() + 1e-7) * 0.9
            return audio

        data, sr = sf.read(str(self.files[idx]), dtype='float32')
        audio = torch.from_numpy(data)
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)
        if sr != self.sample_rate:
            ratio = self.sample_rate / sr
            new_len = int(audio.shape[0] * ratio)
            audio = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear',
                align_corners=False
            ).squeeze()

        if self.augmenter is not None:
            audio = self.augmenter(audio)

        if audio.shape[0] >= self.segment_length:
            start = torch.randint(0, audio.shape[0] - self.segment_length + 1, (1,)).item()
            audio = audio[start:start + self.segment_length]
        else:
            audio = F.pad(audio, (0, self.segment_length - audio.shape[0]))

        return audio


def _grads_finite(parameters) -> bool:
    """Return False if any parameter has NaN or inf in its gradient."""
    for p in parameters:
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                return False
    return True


def _check_model_nan_abort(model, step: int) -> None:
    """If model weights contain NaN/inf, abort with instructions to resume from checkpoint."""
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise RuntimeError(
                f"Model has NaN/inf in '{name}' at step {step}. "
                "Training cannot continue. Resume from the last clean checkpoint "
                "(e.g. sonata_codec_step_20000.pt) with --resume."
            )


def get_lr(step, warmup, max_lr, min_lr, total_steps):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    ratio = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))


def _save_checkpoint(ckpt_dir, name, model, disc, opt_g, opt_d, step, cfg, ema=None, best_val_loss=None):
    data = {
        "model": model.state_dict(),
        "disc": disc.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "step": step,
        "config": vars(cfg),
    }
    if ema is not None:
        data["ema"] = ema.state_dict()
    if best_val_loss is not None:
        data["best_val_loss"] = best_val_loss
    path = ckpt_dir / name
    torch.save(data, path)
    return path


@torch.no_grad()
def validate(model, val_loader, ms_stft_loss, mel_loss_fn, device, max_batches=50):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    n = 0
    for audio in val_loader:
        if n >= max_batches:
            break
        audio = audio.to(device)
        reconstructed, tokens, _ = model(audio)
        min_len = min(reconstructed.shape[-1], audio.shape[-1])
        reconstructed = reconstructed[..., :min_len]
        audio_crop = audio[..., :min_len]
        l_stft = ms_stft_loss(reconstructed, audio_crop)
        l_mel = mel_loss_fn(reconstructed, audio_crop)
        total_loss += (l_stft + l_mel).item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def train(args):
    device = torch.device(args.device)
    use_amp = args.amp and device.type in ("cuda", "mps")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    print(f"\n{'='*70}")
    print(f"  SONATA CODEC — ADVERSARIAL TRAINING")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  AMP: {'ON' if use_amp else 'OFF'}")
    print(f"  GAN training: {'ON' if args.gan_start_step > 0 else 'from step 0'}")
    print(f"  Gradient accumulation: {args.grad_accum} steps")

    # When resuming, use config from checkpoint to ensure architecture matches
    if args.resume:
        _ckpt_peek = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg_dict = _ckpt_peek.get("config", {})
        if args.codec_version == "12hz":
            cfg = Codec12HzConfig(**{k: v for k, v in cfg_dict.items()
                                     if k in Codec12HzConfig.__dataclass_fields__})
        else:
            cfg = CodecConfig(**{k: v for k, v in cfg_dict.items()
                                 if k in CodecConfig.__dataclass_fields__})
        print(f"  Config loaded from checkpoint: {args.resume}")
        del _ckpt_peek
    else:
        if args.codec_version == "12hz":
            cfg = Codec12HzConfig()
        else:
            cfg = CodecConfig(
                enc_dim=args.enc_dim,
                enc_n_layers=args.enc_n_layers,
                dec_dim=args.dec_dim,
                dec_n_layers=args.dec_n_layers,
                decoder_type=args.decoder_type,
            )

    if args.codec_version == "12hz":
        model = SonataCodec12Hz(cfg).to(device)
    else:
        model = SonataCodec(cfg).to(device)
    disc = SonataDiscriminator(use_mrd=args.use_mrd).to(device)

    n_params_g = sum(p.numel() for p in model.parameters())
    n_params_d = sum(p.numel() for p in disc.parameters())
    print(f"  Generator: {n_params_g/1e6:.1f}M params")
    print(f"  Discriminator: {n_params_d/1e6:.1f}M params")
    print(f"  Encoder: {cfg.enc_n_layers}L × d={cfg.enc_dim}")
    print(f"  Decoder: {cfg.dec_n_layers}L × d={cfg.dec_dim}")
    print(f"  FSQ: {cfg.fsq_codebook_size} entries, {cfg.fsq_dim}-dim, levels={cfg.fsq_levels}")

    # Progressive training: start with short segments, ramp to target
    initial_segment_sec = args.progressive_start_sec if args.progressive else args.segment_sec
    segment_length = int(cfg.sample_rate * initial_segment_sec)
    dataset = AudioDataset(
        args.audio_dir, cfg.sample_rate,
        segment_length=segment_length,
        synthetic=args.synthetic,
        manifest=args.manifest,
        augment=args.augment,
    )
    if args.progressive:
        print(f"  Progressive training: {args.progressive_start_sec}s → {args.segment_sec}s "
              f"over {args.progressive_steps} steps")

    # Split into train/val
    val_size = min(max(int(len(dataset) * 0.02), 100), 2000)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=True,
    )

    ms_stft_loss = MultiScaleSTFTLoss().to(device)
    mel_loss_fn = MelReconstructionLoss(cfg).to(device)
    wavlm_loss_fn = WavLMPerceptualLoss().to(device) if args.wavlm_weight > 0 else None

    opt_g = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01,
    )
    opt_d = torch.optim.AdamW(
        disc.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01,
    )

    scaler_g = GradScaler(enabled=use_amp and device.type == "cuda")
    scaler_d = GradScaler(enabled=use_amp and device.type == "cuda")

    ema_g = EMA(model, decay=args.ema_decay)
    print(f"  EMA: decay={args.ema_decay}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tlog = TrainingLog(str(ckpt_dir / "losses.jsonl"))

    start_step = 0
    best_val_loss = float("inf")
    nan_skip_count = [0]
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        for name, p in model.named_parameters():
            if not torch.isfinite(p).all():
                raise ValueError(
                    f"Checkpoint {args.resume} has NaN/inf in model param '{name}'. "
                    "Resume from an earlier clean checkpoint (e.g. step_20000.pt)."
                )
        opt_g.load_state_dict(ckpt["opt_g"])
        if "disc" in ckpt:
            disc.load_state_dict(ckpt["disc"])
            opt_d.load_state_dict(ckpt["opt_d"])
        if "ema" in ckpt:
            ema_g.load_state_dict(ckpt["ema"])
            print(f"  EMA weights restored from checkpoint")
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed from step {start_step}")

    model.train()
    disc.train()
    step = start_step
    accum_step = 0
    log = {"g_total": 0, "stft": 0, "mel": 0, "wav": 0, "adv": 0, "fm": 0, "d_loss": 0, "ent": 0, "codes": 0}
    t0 = time.time()

    print(f"\n  Training: step {step} → {args.steps}")
    print(f"  Batch size: {args.batch_size} × {args.grad_accum} accum = {args.batch_size * args.grad_accum} effective")
    print(f"  Segment: {args.segment_sec}s ({segment_length} samples)")
    print(f"  GAN starts at step: {args.gan_start_step}")
    print(f"  Train/val split: {train_size}/{val_size}")
    print()

    while step < args.steps:
        for audio in loader:
            if step >= args.steps:
                break

            # Progressive training: grow segment length
            # NOTE: with num_workers>0, worker processes have stale copies of the dataset.
            # We set it on the underlying dataset so the next epoch's workers pick it up.
            if args.progressive and step < args.progressive_steps:
                progress = min(step / max(args.progressive_steps, 1), 1.0)
                cur_sec = args.progressive_start_sec + progress * (args.segment_sec - args.progressive_start_sec)
                new_seg = int(cfg.sample_rate * cur_sec)
                base_ds = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
                if new_seg != base_ds.segment_length:
                    base_ds.segment_length = new_seg

            audio = audio.to(device)
            lr = get_lr(step, args.warmup, args.lr, args.lr * 0.1, args.steps)
            for pg in opt_g.param_groups:
                pg["lr"] = lr
            for pg in opt_d.param_groups:
                pg["lr"] = lr

            use_gan = step >= args.gan_start_step

            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                reconstructed, tokens, acoustic = model(audio)
                min_len = min(reconstructed.shape[-1], audio.shape[-1])
                reconstructed = reconstructed[..., :min_len]
                audio_crop = audio[..., :min_len]

                if not torch.isfinite(reconstructed).all():
                    opt_g.zero_grad()
                    if use_gan:
                        opt_d.zero_grad()
                    nan_skip_count[0] += 1
                    if nan_skip_count[0] % 50 == 1:
                        print(f"  [WARN] step {step}: NaN in model output, skipping batch (total skips: {nan_skip_count[0]})")
                    step += 1
                    continue

            # ── Discriminator step ──
            d_loss_val = 0.0
            if use_gan:
                with torch.no_grad():
                    fake_audio = reconstructed.detach()
                with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    disc_real = disc(audio_crop)
                    disc_fake = disc(fake_audio)
                    all_real = [r for group in disc_real for r in (group if isinstance(group, list) else [])]
                    all_fake = [r for group in disc_fake for r in (group if isinstance(group, list) else [])]
                    d_loss = discriminator_loss(all_real, all_fake) / args.grad_accum

                # Lazy R1 gradient penalty (every r1_every steps)
                if args.r1_weight > 0 and step % args.r1_every == 0:
                    r1 = r1_gradient_penalty(disc, audio_crop)
                    d_loss = d_loss + (args.r1_weight / 2.0) * r1 * args.r1_every / args.grad_accum

                scaler_d.scale(d_loss).backward() if use_amp and device.type == "cuda" else d_loss.backward()
                accum_step += 1

                if accum_step % args.grad_accum == 0:
                    if use_amp and device.type == "cuda":
                        scaler_d.unscale_(opt_d)
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), args.clip_grad)
                    if use_amp and device.type == "cuda":
                        scaler_d.step(opt_d)
                        scaler_d.update()
                    else:
                        opt_d.step()
                    opt_d.zero_grad()
                d_loss_val = d_loss.item() * args.grad_accum

            # ── Generator step ──
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                l_stft = ms_stft_loss(reconstructed, audio_crop)
                l_mel = mel_loss_fn(reconstructed, audio_crop)
                l_wav = F.l1_loss(reconstructed, audio_crop)
                l_ent = fsq_entropy_loss(tokens, cfg.fsq_codebook_size)

                g_loss = l_stft + l_mel + args.wav_weight * l_wav + args.entropy_weight * l_ent

                if wavlm_loss_fn is not None:
                    l_wavlm = wavlm_loss_fn(reconstructed, audio_crop, cfg.sample_rate)
                    g_loss = g_loss + args.wavlm_weight * l_wavlm

                if use_gan:
                    disc_fake_g = disc(reconstructed)
                    disc_real_g = disc(audio_crop.detach())
                    all_fake_g = [r for group in disc_fake_g for r in (group if isinstance(group, list) else [])]
                    all_real_g = [r for group in disc_real_g for r in (group if isinstance(group, list) else [])]
                    adv_loss = generator_adversarial_loss(all_fake_g)
                    fm_loss = feature_matching_loss(all_real_g, all_fake_g)
                    g_loss = g_loss + args.adv_weight * adv_loss + args.fm_weight * fm_loss
                    log["adv"] += adv_loss.item()
                    log["fm"] += fm_loss.item()

                g_loss = g_loss / args.grad_accum

            if not torch.isfinite(g_loss):
                opt_g.zero_grad()
                _check_model_nan_abort(model, step)
                step += 1
                continue

            scaler_g.scale(g_loss).backward() if use_amp and device.type == "cuda" else g_loss.backward()

            if (step + 1) % args.grad_accum == 0:
                grad_ok = _grads_finite(model.parameters())
                if not grad_ok:
                    opt_g.zero_grad()
                    nan_skip_count[0] += 1
                    if nan_skip_count[0] % 50 == 1:
                        print(f"  [WARN] step {step}: gradient NaN/inf detected, skipping update (total skips: {nan_skip_count[0]})")
                    _check_model_nan_abort(model, step)
                    step += 1
                    continue

                if use_amp and device.type == "cuda":
                    scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                if use_amp and device.type == "cuda":
                    scaler_g.step(opt_g)
                    scaler_g.update()
                else:
                    opt_g.step()
                opt_g.zero_grad()
                ema_g.update()

            log["g_total"] += g_loss.item() * args.grad_accum
            log["stft"] += l_stft.item()
            log["mel"] += l_mel.item()
            log["wav"] += l_wav.item()
            log["ent"] += l_ent.item()
            log["codes"] += tokens.unique().numel()
            log["d_loss"] += d_loss_val
            step += 1

            if step % args.log_every == 0:
                n = args.log_every
                elapsed = time.time() - t0
                sps = n / elapsed
                gan_str = ""
                if use_gan:
                    gan_str = (f" adv={log['adv']/n:.4f} fm={log['fm']/n:.4f}"
                               f" D={log['d_loss']/n:.4f}")
                print(f"  step {step:6d} | G={log['g_total']/n:.4f}"
                      f" (stft={log['stft']/n:.4f} mel={log['mel']/n:.4f}"
                      f" wav={log['wav']/n:.4f} ent={log['ent']/n:.3f}"
                      f" codes={log['codes']/n:.0f}{gan_str}) |"
                      f" lr={lr:.2e} | {sps:.1f} steps/s")
                tlog.log(step=step, g_total=log["g_total"]/n, stft=log["stft"]/n,
                         mel=log["mel"]/n, wav=log["wav"]/n, ent=log["ent"]/n,
                         codes=log["codes"]/n, d_loss=log["d_loss"]/n, lr=lr,
                         steps_per_sec=sps)
                log = {k: 0 for k in log}
                t0 = time.time()

            if step % args.save_every == 0:
                path = _save_checkpoint(ckpt_dir, f"sonata_codec_step_{step}.pt",
                                        model, disc, opt_g, opt_d, step, cfg, ema_g,
                                        best_val_loss=best_val_loss)
                print(f"  [ckpt] Saved {path}")

            # Validation (use EMA weights for better eval)
            if step % args.val_every == 0 and len(val_ds) > 0:
                ema_g.apply_shadow()
                val_loss = validate(model, val_loader, ms_stft_loss, mel_loss_fn, device)
                ema_g.restore()
                print(f"  [val] step {step}: loss={val_loss:.4f}"
                      f" {'(best!)' if val_loss < best_val_loss else ''}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    path = _save_checkpoint(ckpt_dir, "sonata_codec_best.pt",
                                            model, disc, opt_g, opt_d, step, cfg, ema_g,
                                            best_val_loss=best_val_loss)
                    print(f"  [best] Saved {path}")

    # Save final
    path = _save_checkpoint(ckpt_dir, "sonata_codec_final.pt",
                            model, disc, opt_g, opt_d, step, cfg, ema_g,
                            best_val_loss=best_val_loss)
    print(f"\n  [ckpt] Final checkpoint: {path}")
    print(f"  Training complete: {step} steps, best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--checkpoint-dir", default="train/checkpoints/codec")
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--segment-sec", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--enc-dim", type=int, default=256)
    parser.add_argument("--enc-n-layers", type=int, default=4)
    parser.add_argument("--dec-dim", type=int, default=512)
    parser.add_argument("--dec-n-layers", type=int, default=8)
    parser.add_argument("--decoder-type", default="conv", choices=["conv", "istft"])
    parser.add_argument("--gan-start-step", type=int, default=10000,
                        help="Step to start GAN training (warm up reconstruction first)")
    parser.add_argument("--adv-weight", type=float, default=1.0)
    parser.add_argument("--fm-weight", type=float, default=2.0)
    parser.add_argument("--entropy-weight", type=float, default=1.0)
    parser.add_argument("--wav-weight", type=float, default=1.0)
    parser.add_argument("--wavlm-weight", type=float, default=1.0,
                        help="WavLM perceptual loss weight (0=disable, requires transformers)")
    parser.add_argument("--clip-grad", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--r1-weight", type=float, default=10.0,
                        help="R1 gradient penalty weight (0=disable)")
    parser.add_argument("--r1-every", type=int, default=16,
                        help="Apply R1 penalty every N steps (lazy regularization)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision training")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate for generator weights (0=disable)")
    parser.add_argument("--use-mrd", action="store_true",
                        help="Add Multi-Resolution Discriminator (MRD) to MPD+MSD")
    parser.add_argument("--progressive", action="store_true",
                        help="Progressive training: ramp segment length from start to target")
    parser.add_argument("--progressive-start-sec", type=float, default=1.0,
                        help="Starting segment length for progressive training")
    parser.add_argument("--progressive-steps", type=int, default=50000,
                        help="Steps over which to ramp segment length to target")
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation (speed perturb + noise)")
    parser.add_argument("--codec-version", default="50hz", choices=["50hz", "12hz"],
                        help="Codec version: 50hz (default) or 12hz (12.5 Hz frame rate)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
