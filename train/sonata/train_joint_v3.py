"""Joint fine-tuning: Flow v3 + Vocoder end-to-end with waveform loss.

The key to human-level quality: let the flow model learn to produce mel
spectrograms that the vocoder can reconstruct well, and let the vocoder
learn to handle the specific mel characteristics of the flow model.

Pipeline: Text → Flow v3 → mel → Vocoder → audio → Loss(audio, real_audio)

Losses:
  - Multi-resolution mel + spectral convergence (from vocoder.MelSpecLoss)
  - Feature matching (from vocoder discriminators)
  - Adversarial (optional, GAN training)
  - Duration predictor (from flow's duration head)

The flow model is fine-tuned with a low learning rate while the vocoder
continues training — this teaches them to work together.

Usage:
  python train_joint_v3.py \
    --flow-ckpt checkpoints/flow_v3_libritts/flow_v3_best.pt \
    --vocoder-ckpt checkpoints/vocoder_libritts/vocoder_epoch50.pt \
    --manifest train/data/libritts_r_manifest.jsonl \
    --device mps --steps 50000
"""

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import FlowV3Config, FlowV3LargeConfig, VocoderConfig, VocoderLargeConfig
from flow_v3 import SonataFlowV3
from vocoder import SonataVocoder
from modules import TrainingLog

# Constants for perceptual losses
VOCODER_SR = 24000
WAVLM_SR = 16000

try:
    from perceptual_loss import PerceptualMelLoss, WavLMPerceptualLoss
    HAS_PERCEPTUAL = True
except ImportError:
    HAS_PERCEPTUAL = False

try:
    from ema import EMA
    HAS_EMA = True
except ImportError:
    HAS_EMA = False

try:
    from g2p import PhonemeFrontend
    HAS_G2P = True
except ImportError:
    HAS_G2P = False


def _config_to_dict(cfg) -> dict:
    """Build serializable config dict from dataclass or dict."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        try:
            return dataclasses.asdict(cfg)
        except TypeError:
            pass
    if isinstance(cfg, dict):
        return dict(cfg)
    return {k: getattr(cfg, k) for k in (getattr(cfg, "__dataclass_fields__", cfg.__dict__.keys()) or cfg.__dict__.keys()) if not k.startswith("_")}


class _InlineEMA:
    """Minimal inline EMA when ema module is not available."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


def cosine_lr(step, warmup, max_lr, min_lr, total):
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    ratio = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))


def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample audio. audio: (B, T) or (B, 1, T)."""
    if orig_sr == target_sr:
        return audio
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
    new_len = int(audio.shape[-1] * target_sr / orig_sr)
    out = F.interpolate(audio.float(), size=new_len, mode="linear", align_corners=False)
    return out.squeeze(1) if out.shape[1] == 1 else out


def extract_mel(audio, n_fft=1024, hop_length=480, n_mels=80, sample_rate=24000):
    """Extract 80-bin log-mel from audio tensor."""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    window = torch.hann_window(n_fft, periodic=True, device=audio.device)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    mag = stft.abs().pow(2)

    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0, sample_rate / 2, n_freqs)
    mel_low = 2595 * math.log10(1 + 0 / 700)
    mel_high = 2595 * math.log10(1 + sample_rate / 2 / 700)
    mels = torch.linspace(mel_low, mel_high, n_mels + 2)
    hz = 700 * (10 ** (mels / 2595) - 1)
    fb = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        low, center, high = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - low) / (center - low + 1e-8)
        down = (high - freqs) / (high - center + 1e-8)
        fb[i] = torch.clamp(torch.min(up, down), min=0)
    fb = fb.to(audio.device)
    mel = torch.matmul(fb, mag.squeeze(0))
    mel = torch.log(mel.clamp(min=1e-5))
    return mel.T


class JointDataset(Dataset):
    """Load audio + text pairs from JSONL manifest for joint training.

    Manifest format (JSONL): {"audio": path, "text": transcript,
    "speaker": speaker_id, "duration": seconds}
    """

    def __init__(self, manifest: str, max_frames: int = 400,
                 sample_rate: int = 24000, hop_length: int = 480,
                 g2p=None):
        self.entries = []
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_samples = max_frames * hop_length
        self.g2p = g2p

        with open(manifest) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("duration", 999) <= 10.0:
                    self.entries.append(entry)
        mode = "phonemes" if g2p else "characters"
        print(f"  Joint dataset: {len(self.entries)} utterances ({mode})")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        data, sr = sf.read(entry["audio"], dtype='float32')
        audio = torch.from_numpy(data)
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)
        if sr != self.sample_rate:
            ratio = self.sample_rate / sr
            new_len = int(audio.shape[0] * ratio)
            audio = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0), size=new_len,
                mode='linear', align_corners=False
            ).squeeze()

        if audio.shape[0] > self.max_samples:
            start = torch.randint(0, audio.shape[0] - self.max_samples, (1,)).item()
            audio = audio[start:start + self.max_samples]

        mel = extract_mel(audio)
        text = entry.get("text", "")
        precomputed = entry.get("phoneme_ids")
        if precomputed is not None:
            char_ids = torch.tensor(precomputed, dtype=torch.long)
        elif self.g2p is not None:
            char_ids = self.g2p.encode(text, add_bos=True, add_eos=True)
        else:
            char_ids = torch.tensor(
                [ord(c) % 256 for c in text],
                dtype=torch.long
            )
        speaker_id = hash(entry.get("speaker", "")) % 10000
        return char_ids, mel, audio, speaker_id


def collate_fn(batch):
    """Collate batch with mel_mask: 1=valid, 0=padded."""
    char_list, mel_list, audio_list, spk_list = zip(*batch)
    max_chars = max(c.shape[0] for c in char_list)
    max_frames = max(m.shape[0] for m in mel_list)
    max_samples = max(a.shape[0] for a in audio_list)
    mel_dim = mel_list[0].shape[1]
    B = len(batch)

    chars = torch.zeros(B, max_chars, dtype=torch.long)
    mel = torch.zeros(B, max_frames, mel_dim)
    mel_mask = torch.zeros(B, max_frames)
    audio = torch.zeros(B, max_samples)
    speakers = torch.tensor(spk_list, dtype=torch.long)

    for i, (c, m, a, _) in enumerate(zip(char_list, mel_list, audio_list, spk_list)):
        chars[i, :c.shape[0]] = c
        mel[i, :m.shape[0]] = m
        mel_mask[i, :m.shape[0]] = 1.0
        audio[i, :a.shape[0]] = a

    return chars, mel, mel_mask, audio, speakers


def train(args):
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  JOINT FINE-TUNING: Flow v3 + Vocoder")
    print(f"{'='*60}")

    g2p = None
    if getattr(args, "phonemes", False):
        if not HAS_G2P:
            print("  ERROR: --phonemes requires g2p.py and phonemizer package")
            return
        g2p = PhonemeFrontend()
        print(f"  Phoneme mode: ON ({g2p.vocab_size} tokens)")

    flow_ckpt = torch.load(args.flow_checkpoint, map_location="cpu", weights_only=False)
    flow_cfg_raw = flow_ckpt.get("config", flow_ckpt.get("flow_config", {}))
    FlowCfgCls = FlowV3LargeConfig if getattr(args, "model_size", "base") == "large" else FlowV3Config
    if isinstance(flow_cfg_raw, (FlowV3Config, FlowV3LargeConfig)):
        flow_cfg = flow_cfg_raw
    else:
        flow_cfg_dict = flow_cfg_raw if isinstance(flow_cfg_raw, dict) else {}
        flow_cfg = FlowCfgCls(**{k: v for k, v in flow_cfg_dict.items()
                                 if k in FlowCfgCls.__dataclass_fields__})
    if g2p is not None:
        flow_cfg.char_vocab_size = g2p.vocab_size
    flow = SonataFlowV3(flow_cfg).to(device)
    flow_state = flow_ckpt.get("model", flow_ckpt.get("flow", flow_ckpt))
    flow.load_state_dict(flow_state, strict=False)
    print(f"  Flow v3: {sum(p.numel() for p in flow.parameters())/1e6:.1f}M params")

    voc_ckpt = torch.load(args.vocoder_checkpoint, map_location="cpu", weights_only=False)
    voc_cfg_raw = voc_ckpt.get("config", voc_ckpt.get("vocoder_config", {}))
    VocCfgCls = VocoderLargeConfig if getattr(args, "model_size", "base") == "large" else VocoderConfig
    if isinstance(voc_cfg_raw, (VocoderConfig, VocoderLargeConfig)):
        voc_cfg = voc_cfg_raw
    else:
        voc_cfg_dict = voc_cfg_raw if isinstance(voc_cfg_raw, dict) else {}
        voc_cfg = VocCfgCls(**{k: v for k, v in voc_cfg_dict.items()
                               if k in VocCfgCls.__dataclass_fields__})
    vocoder = SonataVocoder(voc_cfg).to(device)
    voc_state = voc_ckpt.get("model", voc_ckpt.get("vocoder", voc_ckpt))
    vocoder.load_state_dict(voc_state, strict=False)
    print(f"  Vocoder: {sum(p.numel() for p in vocoder.generator.parameters())/1e6:.1f}M params")

    disc_params = (list(vocoder.mpd.parameters()) + list(vocoder.msd.parameters()) +
                  list(vocoder.mrstft.parameters()))
    flow_opt = torch.optim.AdamW(
        flow.parameters(), lr=args.flow_lr, betas=(0.9, 0.95), weight_decay=0.01)
    gen_opt = torch.optim.AdamW(
        vocoder.generator.parameters(), lr=args.vocoder_lr,
        betas=(0.8, 0.99), weight_decay=0.01)
    disc_opt = torch.optim.AdamW(
        disc_params, lr=args.vocoder_lr, betas=(0.8, 0.99), weight_decay=0.01)

    perceptual_mel_fn = None
    if HAS_PERCEPTUAL and getattr(args, "perceptual_weight", 0) > 0:
        perceptual_mel_fn = PerceptualMelLoss(n_mels=flow_cfg.mel_dim).to(device)
        print(f"  Perceptual mel loss: ON (weight={args.perceptual_weight})")

    wavlm_fn = None
    if HAS_PERCEPTUAL and getattr(args, "wavlm_weight", 0) > 0:
        wavlm_fn = WavLMPerceptualLoss().to(device)
        print(f"  WavLM loss: ON (weight={args.wavlm_weight})")

    _EMA = EMA if HAS_EMA else _InlineEMA
    flow_ema = _EMA(flow, decay=0.999) if True else None  # Always use EMA (module or inline)
    gen_ema = _EMA(vocoder.generator, decay=0.999) if True else None

    flow_cfg_dict = _config_to_dict(flow_cfg)
    voc_cfg_dict = _config_to_dict(voc_cfg)

    dataset = JointDataset(args.manifest, max_frames=args.max_frames, g2p=g2p)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, drop_last=True)

    os.makedirs(args.output_dir, exist_ok=True)
    tlog = TrainingLog(os.path.join(args.output_dir, "losses.jsonl"))

    step = 0
    accum_count = 0
    flow.train()
    vocoder.train()
    t0 = time.time()

    grad_accum = getattr(args, "grad_accum", 4)
    print(f"  Training: {args.steps} steps, batch={args.batch_size}, grad_accum={grad_accum}")
    print(f"  Flow LR: {args.flow_lr}, Vocoder LR: {args.vocoder_lr}")
    print(f"  Perceptual: {getattr(args, 'perceptual_weight', 0.1)}, WavLM: {getattr(args, 'wavlm_weight', 0.05)}")
    if flow_ema:
        print(f"  EMA: flow + vocoder generator")
    print()

    while step < args.steps:
        for chars, mel_target, mel_mask, audio_real, speakers in loader:
            if step >= args.steps:
                break

            chars = chars.to(device)
            mel_target = mel_target.to(device)
            mel_mask = mel_mask.to(device)
            audio_real = audio_real.to(device)
            speakers = speakers.to(device) if flow_cfg.n_speakers > 0 else None
            if speakers is not None and flow_cfg.n_speakers > 0:
                speakers = speakers % flow_cfg.n_speakers

            T_mel = mel_target.shape[1]
            flow_lr = cosine_lr(step, args.warmup, args.flow_lr,
                                args.flow_lr * 0.01, args.steps)
            voc_lr = cosine_lr(step, args.warmup, args.vocoder_lr,
                               args.vocoder_lr * 0.01, args.steps)
            for pg in flow_opt.param_groups:
                pg["lr"] = flow_lr
            for pg in gen_opt.param_groups:
                pg["lr"] = voc_lr
            for pg in disc_opt.param_groups:
                pg["lr"] = voc_lr

            # Flow loss + denoised mel (with mel_mask)
            flow_out = flow.compute_loss(
                mel_target, chars,
                speaker_ids=speakers, mel_mask=mel_mask, use_mas=True,
                return_denoised=True)
            flow_loss, mel_denoised = flow_out
            flow_loss_val = flow_loss.item()
            dur_loss_val = 0.0  # Already included in flow_loss
            flow_fm_val = flow_loss_val  # For logging (no separate flow FM)

            perceptual_mel_val = 0.0
            perceptual_mel_loss = torch.tensor(0.0, device=device)
            if perceptual_mel_fn is not None and getattr(args, "perceptual_weight", 0) > 0:
                perceptual_mel_loss = perceptual_mel_fn(
                    mel_denoised, mel_target, mel_mask=mel_mask)
                perceptual_mel_val = perceptual_mel_loss.item()

            # D step: use mel_denoised.detach(), R1 every 16 steps
            r1_weight = 0.1 if (step % 16 == 0) else 0.0
            d_result = vocoder.training_step_d(
                mel_denoised.detach(), audio_real, r1_weight=r1_weight)
            d_loss_val = d_result["d_loss"].item()
            if torch.isfinite(d_result["d_loss"]):
                disc_opt.zero_grad()
                d_result["d_loss"].backward()
                torch.nn.utils.clip_grad_norm_(disc_params, 1.0)
                disc_opt.step()

            # G step: end-to-end flow + vocoder with perceptual losses
            g_result = vocoder.training_step_g(mel_denoised, audio_real)
            g_loss_val = g_result["g_loss"].item()
            mel_loss_val = g_result.get("mel_loss", 0.0)

            wavlm_val = 0.0
            wavlm_loss = torch.tensor(0.0, device=device)
            if wavlm_fn is not None and getattr(args, "wavlm_weight", 0) > 0:
                audio_pred = vocoder.generate(mel_denoised)
                audio_pred_16k = resample_audio(audio_pred, VOCODER_SR, WAVLM_SR)
                audio_real_16k = resample_audio(audio_real, VOCODER_SR, WAVLM_SR)
                min_len = min(audio_pred_16k.shape[-1], audio_real_16k.shape[-1])
                audio_pred_16k = audio_pred_16k[..., :min_len]
                audio_real_16k = audio_real_16k[..., :min_len]
                wavlm_loss = wavlm_fn(audio_pred_16k, audio_real_16k)
                wavlm_val = wavlm_loss.item()

            total_g = flow_loss + g_result["g_loss"]
            if perceptual_mel_fn and getattr(args, "perceptual_weight", 0) > 0:
                total_g = total_g + args.perceptual_weight * perceptual_mel_loss
            if wavlm_fn and getattr(args, "wavlm_weight", 0) > 0:
                total_g = total_g + args.wavlm_weight * wavlm_loss

            if torch.isfinite(total_g):
                total_g = total_g / grad_accum
                if accum_count == 0:
                    flow_opt.zero_grad()
                    gen_opt.zero_grad()
                total_g.backward()
                accum_count += 1
                if accum_count >= grad_accum:
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(vocoder.generator.parameters(), 1.0)
                    flow_opt.step()
                    gen_opt.step()
                    if flow_ema:
                        flow_ema.update()
                    if gen_ema:
                        gen_ema.update()
                    accum_count = 0
                    step += 1

            if step % args.log_every == 0:
                elapsed = max(time.time() - t0, 0.001)
                print(f"  step {step:6d} | flow={flow_loss_val:.4f} dur={dur_loss_val:.4f} "
                      f"G={g_loss_val:.4f} mel={mel_loss_val:.4f} "
                      f"percep={perceptual_mel_val:.4f} wavlm={wavlm_val:.4f} "
                      f"D={d_loss_val:.4f} | {args.log_every/elapsed:.1f} steps/s")
                tlog.log(
                    step=step,
                    flow_loss=flow_loss_val,
                    dur_loss=dur_loss_val,
                    flow_fm=flow_fm_val,
                    g_loss=g_loss_val,
                    mel_loss=mel_loss_val,
                    perceptual_loss=perceptual_mel_val,
                    wavlm_loss=wavlm_val,
                    d_loss=d_loss_val,
                )
                t0 = time.time()

            if step % args.save_every == 0:
                save_dict = {
                    "flow": flow.state_dict(),
                    "vocoder": vocoder.state_dict(),
                    "step": step,
                    "flow_config": flow_cfg_dict,
                    "vocoder_config": voc_cfg_dict,
                }
                if flow_ema:
                    save_dict["flow_ema"] = flow_ema.state_dict()
                if gen_ema:
                    save_dict["vocoder_gen_ema"] = gen_ema.state_dict()
                path = os.path.join(args.output_dir, f"joint_step_{step}.pt")
                torch.save(save_dict, path)
                print(f"  [ckpt] {path}")

    final_path = os.path.join(args.output_dir, "joint_final.pt")
    save_dict = {
        "flow": flow.state_dict(),
        "vocoder": vocoder.state_dict(),
        "flow_config": flow_cfg_dict,
        "vocoder_config": voc_cfg_dict,
        "step": step,
    }
    if flow_ema:
        save_dict["flow_ema"] = flow_ema.state_dict()
    if gen_ema:
        save_dict["vocoder_gen_ema"] = gen_ema.state_dict()
    torch.save(save_dict, final_path)
    print(f"\nFinal: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint Flow v3 + Vocoder training (production)")
    parser.add_argument("--flow-ckpt", "--flow-checkpoint", dest="flow_checkpoint")
    parser.add_argument("--vocoder-ckpt", "--vocoder-checkpoint", dest="vocoder_checkpoint")
    parser.add_argument("--manifest", help="JSONL: {audio, text, speaker, duration}")
    parser.add_argument("--output-dir", default="train/checkpoints/joint_v3")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--flow-lr", type=float, default=1e-5)
    parser.add_argument("--vocoder-lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--perceptual-weight", type=float, default=0.1,
                        help="PerceptualMelLoss on flow denoised mel")
    parser.add_argument("--wavlm-weight", type=float, default=0.05,
                        help="WavLM perceptual loss on waveform")
    parser.add_argument("--model-size", choices=["base", "large"], default="base")
    parser.add_argument("--phonemes", action="store_true",
                        help="Use phoneme conditioning via espeak-ng G2P (recommended)")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=10000)
    args = parser.parse_args()

    if not args.flow_checkpoint:
        parser.error("--flow-ckpt/--flow-checkpoint required")
    if not args.vocoder_checkpoint:
        parser.error("--vocoder-ckpt/--vocoder-checkpoint required")
    if not args.manifest:
        parser.error("--manifest required")
    train(args)
