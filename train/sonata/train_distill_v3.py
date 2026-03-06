"""Consistency distillation for Sonata Flow v3.

Distills an 8-step teacher into a 1-step student using consistency distillation
(Song et al., 2023) adapted for flow matching. Enables real-time inference.

Algorithm:
  1. Sample x_0 from training data (mel spectrogram)
  2. Sample t ~ logit-normal (sway)
  3. x_t = (1-t)*noise + t*x_0
  4. Teacher: one Heun ODE step (x_t, t) → (x_{t+dt}, t+dt)
  5. Student: f_θ(x_t, t) → x_0_pred = x_t + (1-t)*v_pred
  6. EMA target: f_ema(x_{t+dt}, t+dt) → x_0_ema (stop_gradient)
  7. Loss = ||x_0_pred - sg(x_0_ema)||^2

Usage:
  python train_distill_v3.py \
    --teacher-checkpoint checkpoints/flow_v3_best.pt \
    --manifest data/manifest_clean.jsonl \
    --output-dir checkpoints/flow_v3_distilled \
    --device mps
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import FlowV3Config, FlowV3LargeConfig
from flow_v3 import SonataFlowV3, DurationPredictor
from ema import EMA

try:
    from g2p import PhonemeFrontend
    HAS_G2P = True
except ImportError:
    HAS_G2P = False

try:
    from modules import TrainingLog
except ImportError:
    TrainingLog = None


# ─── Mel extraction (matches train_flow_v3) ──────────────────────────────────

def extract_mel_from_audio(audio: torch.Tensor, n_fft=1024, hop_length=480,
                          n_mels=80, sample_rate=24000) -> torch.Tensor:
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


# ─── Datasets (same as train_flow_v3) ─────────────────────────────────────────

class PtDataset(Dataset):
    def __init__(self, data_dir: str, max_frames: int = 800, max_text_len: int = 512,
                 g2p=None, vocab_size: int = 256):
        self.files = sorted(Path(data_dir).glob("*.pt"))
        self.files = [f for f in self.files if f.name != "meta.pt"]
        self.max_frames = max_frames
        self.max_text_len = max_text_len
        self.g2p = g2p
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        text = data.get("text", "")[:self.max_text_len]
        mel = data["mel"]
        speaker_id = data.get("speaker_id", 0)
        T = min(mel.shape[0], self.max_frames)
        mel = mel[:T]
        if self.g2p is not None:
            char_ids = self.g2p.encode(text, add_bos=True, add_eos=True)
        else:
            char_ids = torch.tensor([ord(c) % self.vocab_size for c in text], dtype=torch.long)
        return char_ids, mel, speaker_id


class ManifestDataset(Dataset):
    def __init__(self, manifest: str, cfg: FlowV3Config, max_frames: int = 800,
                 g2p=None):
        self.cfg = cfg
        self.max_frames = max_frames
        self.g2p = g2p
        self.entries = []
        speakers_seen = {}
        with open(manifest) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("duration", 999) <= 15.0:
                    spk = entry.get("speaker", "")
                    if spk not in speakers_seen:
                        speakers_seen[spk] = len(speakers_seen)
                    self.entries.append(entry)
        self.speaker_map = speakers_seen
        mode = "phonemes" if g2p else "characters"
        print(f"  Manifest: {len(self.entries)} utterances, {len(speakers_seen)} speakers ({mode})")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        data, sr = sf.read(entry["audio"], dtype='float32')
        audio = torch.from_numpy(data)
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)
        if sr != self.cfg.sample_rate:
            ratio = self.cfg.sample_rate / sr
            new_len = int(audio.shape[0] * ratio)
            audio = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear',
                align_corners=False
            ).squeeze()

        if self.cfg.speed_perturb and torch.rand(1).item() < 0.5:
            speed = random.choice([0.9, 1.0, 1.1])
            if speed != 1.0:
                new_len = int(audio.shape[0] / speed)
                audio = F.interpolate(
                    audio.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear',
                    align_corners=False
                ).squeeze()

        max_samples = self.max_frames * self.cfg.hop_length
        if audio.shape[0] > max_samples:
            start = torch.randint(0, audio.shape[0] - max_samples, (1,)).item()
            audio = audio[start:start + max_samples]

        mel = extract_mel_from_audio(
            audio, self.cfg.n_fft, self.cfg.hop_length,
            self.cfg.n_mels_extract, self.cfg.sample_rate
        )

        if self.cfg.spec_augment and torch.rand(1).item() < 0.5:
            mel = self._spec_augment(mel)
        text = entry.get("text", "")
        if self.g2p is not None:
            char_ids = self.g2p.encode(text, add_bos=True, add_eos=True)
        else:
            char_ids = torch.tensor(
                [ord(c) % self.cfg.char_vocab_size for c in text],
                dtype=torch.long
            )
        speaker_id = self.speaker_map.get(entry.get("speaker", ""), 0)
        return char_ids, mel, speaker_id

    @staticmethod
    def _spec_augment(mel, n_freq_masks=2, freq_width=15, n_time_masks=2, time_width=50):
        T, D = mel.shape
        for _ in range(n_freq_masks):
            f = random.randint(0, min(freq_width, D - 1))
            f0 = random.randint(0, D - f)
            mel[..., f0:f0 + f] = 0
        for _ in range(n_time_masks):
            t = random.randint(0, min(time_width, T - 1))
            t0 = random.randint(0, T - t)
            mel[t0:t0 + t, :] = 0
        return mel


def collate_fn(batch):
    char_list, mel_list, spk_list = zip(*batch)
    max_chars = max(c.shape[0] for c in char_list)
    max_frames = max(m.shape[0] for m in mel_list)
    mel_dim = mel_list[0].shape[1]
    B = len(batch)

    chars = torch.zeros(B, max_chars, dtype=torch.long)
    mel = torch.zeros(B, max_frames, mel_dim)
    mel_mask = torch.zeros(B, max_frames)
    speakers = torch.tensor(spk_list, dtype=torch.long)

    for i, (c, m, _) in enumerate(zip(char_list, mel_list, spk_list)):
        chars[i, :c.shape[0]] = c
        mel[i, :m.shape[0]] = m
        mel_mask[i, :m.shape[0]] = 1.0

    return chars, mel, mel_mask, speakers


# ─── Text conditioning helper ─────────────────────────────────────────────────

def build_text_cond(model: SonataFlowV3, char_ids: torch.Tensor, mel_len: int) -> torch.Tensor:
    """Build text conditioning aligned to mel length (uniform alignment for distillation)."""
    if char_ids.numel() == 0 or mel_len <= 0:
        D = model.cfg.d_model
        B = char_ids.shape[0] if char_ids.dim() > 1 else 1
        return torch.zeros(B, mel_len, D, device=char_ids.device)
    text_enc = model.interleaved_enc.encode_text(char_ids)
    gt_durations = DurationPredictor.compute_gt_durations(char_ids, mel_len)
    return DurationPredictor.expand_encodings(text_enc, gt_durations, mel_len)


# ─── Consistency distillation step ────────────────────────────────────────────

def teacher_heun_step(teacher: SonataFlowV3, x_t: torch.Tensor, t: torch.Tensor,
                     text_cond: torch.Tensor, speaker_ids: Optional[torch.Tensor],
                     dt: float) -> torch.Tensor:
    """One Heun ODE step: (x_t, t) → x_{t+dt}."""
    B = x_t.shape[0]
    device = x_t.device

    v1, _ = teacher.forward(x_t, t, text_cond, speaker_ids)
    x_euler = x_t + dt * v1

    t_next = (t + dt).clamp(max=1.0)
    v2, _ = teacher.forward(x_euler, t_next, text_cond, speaker_ids)
    x_next = x_t + dt * 0.5 * (v1 + v2)

    return x_next


def velocity_to_x0(x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Convert flow velocity to x_0 prediction: x_0 = x_t + (1-t)*v."""
    t_expand = t[:, None, None]
    return x_t + (1.0 - t_expand) * v


def distill_step(
    teacher: SonataFlowV3,
    student: SonataFlowV3,
    ema_model: SonataFlowV3,
    mel: torch.Tensor,
    char_ids: torch.Tensor,
    speaker_ids: Optional[torch.Tensor],
    mel_mask: torch.Tensor,
    dt: float,
    t_min: float = 0.02,
    t_max: float = 0.98,
    sway: float = -1.0,
) -> torch.Tensor:
    """
    One consistency distillation step.

    Sample t in [t_min, min(t_max, 1-dt)] to avoid overshooting.
    Teacher takes Heun step to get x_{t+dt}. Student at (x_t,t) and EMA at (x_{t+dt},t+dt)
    predict x_0. Loss = MSE(student x_0, stop_grad(EMA x_0)).
    """
    B, T, D = mel.shape
    device = mel.device

    # Sample t: logit-normal with sway, clamped to valid range
    t_upper = min(t_max, 1.0 - dt - 1e-6)
    if t_upper <= t_min:
        # Fallback if dt too large
        t = torch.full((B,), 0.5, device=device)
    else:
        z = torch.randn(B, device=device) + sway
        t = torch.sigmoid(z).clamp(t_min, t_upper)

    t_expand = t[:, None, None]
    noise = torch.randn_like(mel, device=device)
    x_t = (1.0 - t_expand) * noise + t_expand * mel

    # Teacher: one Heun step
    text_cond = build_text_cond(teacher, char_ids, T)
    with torch.no_grad():
        x_t_next = teacher_heun_step(teacher, x_t, t, text_cond, speaker_ids, dt)
        t_next = (t + dt).clamp(max=1.0)

    # Student prediction at (x_t, t)
    v_student, _ = student.forward(x_t, t, text_cond, speaker_ids)
    x0_student = velocity_to_x0(x_t, t, v_student)

    # EMA target prediction at (x_{t+dt}, t+dt) — stop gradient
    with torch.no_grad():
        v_ema, _ = ema_model.forward(x_t_next, t_next, text_cond, speaker_ids)
        x0_ema = velocity_to_x0(x_t_next, t_next, v_ema)

    # Consistency loss: student should match EMA target
    if mel_mask is not None:
        mask = mel_mask.unsqueeze(-1)
        per_elem = F.mse_loss(x0_student, x0_ema, reduction='none') * mask
        n_valid = mask.sum() * D
        loss = per_elem.sum() / n_valid.clamp(min=1)
    else:
        loss = F.mse_loss(x0_student, x0_ema)

    return loss


# ─── Validation: 1-step vs 8-step MSE ────────────────────────────────────────

@torch.no_grad()
def validate_distillation(
    student: SonataFlowV3,
    teacher: SonataFlowV3,
    chars: torch.Tensor,
    mel: torch.Tensor,
    mel_mask: torch.Tensor,
    speakers: Optional[torch.Tensor],
    cfg,
    teacher_steps: int = 8,
) -> float:
    """Generate sample mel with 1-step student vs 8-step teacher, return MSE between them."""
    B, T, D = mel.shape
    device = mel.device

    text_cond = build_text_cond(teacher, chars, T)
    noise = torch.randn_like(mel, device=device)

    # 1-step student: t=0 → t=1 in one step
    t0 = torch.zeros(B, device=device)
    v_s, _ = student.forward(noise, t0, text_cond, speakers)
    mel_student = noise + 1.0 * v_s  # x_1 = x_0 + (1-0)*v = x_0 + v

    # 8-step teacher
    dt = 1.0 / teacher_steps
    x = noise.clone()
    for i in range(teacher_steps):
        t_val = torch.full((B,), i * dt, device=device)
        v_t, _ = teacher.forward(x, t_val, text_cond, speakers)
        if i < teacher_steps - 1:
            t_next = torch.full((B,), (i + 1) * dt, device=device)
            v_next, _ = teacher.forward(x + dt * v_t, t_next, text_cond, speakers)
            x = x + dt * 0.5 * (v_t + v_next)  # Heun
        else:
            x = x + dt * v_t

    mel_teacher = x

    # MSE over valid (unpadded) positions
    if mel_mask is not None:
        mask = mel_mask.unsqueeze(-1)
        diff = (mel_student - mel_teacher) ** 2
        per_elem = diff * mask
        n_valid = mask.sum() * D
        mse = per_elem.sum() / n_valid.clamp(min=1)
    else:
        mse = F.mse_loss(mel_student, mel_teacher)

    return mse.item()


# ─── EMA with ramp ─────────────────────────────────────────────────────────────

class EMAWithRamp:
    """EMA with decay ramping from start_decay to end_decay over total_steps."""

    def __init__(self, model: torch.nn.Module, start_decay: float = 0.95,
                 end_decay: float = 0.999, total_steps: int = 50000):
        self.model = model
        self.start_decay = start_decay
        self.end_decay = end_decay
        self.total_steps = total_steps
        self.ema = EMA(model, decay=start_decay)

    def update(self, step: int):
        # Linear ramp: decay goes from start to end over total_steps
        ratio = min(1.0, step / max(1, self.total_steps))
        decay = self.start_decay + (self.end_decay - self.start_decay) * ratio
        self.ema.decay = decay
        self.ema.update()

    def apply_shadow(self):
        return self.ema.apply_shadow()

    def restore(self):
        return self.ema.restore()

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state):
        return self.ema.load_state_dict(state)


# ─── Training ─────────────────────────────────────────────────────────────────

def load_teacher_checkpoint(path: str) -> tuple:
    """Load teacher checkpoint, return (state_dict, config_dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    cfg_dict = ckpt.get("config", {})

    # Strip "model." prefix if present
    keys = list(state.keys())
    if keys and any(k.startswith("model.") for k in keys):
        state = {k.replace("model.", ""): v for k, v in state.items()}

    return state, cfg_dict


def train(args):
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  SONATA FLOW v3 — CONSISTENCY DISTILLATION")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # Config
    g2p = None
    if args.phonemes:
        if not HAS_G2P:
            print("  ERROR: --phonemes requires g2p module. Install phonemizer.")
            return
        g2p = PhonemeFrontend()

    # Load teacher checkpoint first to infer config
    teacher_state, ckpt_cfg = load_teacher_checkpoint(args.teacher_checkpoint)

    # Infer n_speakers and speaker_dim from checkpoint config or state_dict
    n_speakers = args.n_speakers
    speaker_dim = None
    if n_speakers == 0 and ckpt_cfg.get("n_speakers", 0) > 0:
        n_speakers = ckpt_cfg["n_speakers"]
        print(f"  Inferred n_speakers={n_speakers} from teacher checkpoint config")
    if n_speakers == 0 and "speaker_emb.weight" in teacher_state:
        n_speakers = teacher_state["speaker_emb.weight"].shape[0]
        print(f"  Inferred n_speakers={n_speakers} from teacher state_dict")
    if "speaker_emb.weight" in teacher_state:
        speaker_dim = teacher_state["speaker_emb.weight"].shape[1]
        print(f"  Inferred speaker_dim={speaker_dim} from teacher state_dict")

    if args.model_size == "large":
        cfg = FlowV3LargeConfig(n_speakers=n_speakers)
    else:
        cfg = FlowV3Config(n_speakers=n_speakers)

    # Override speaker_dim to match teacher if needed
    if speaker_dim is not None and cfg.speaker_dim != speaker_dim:
        print(f"  Overriding speaker_dim: {cfg.speaker_dim} → {speaker_dim} (to match teacher)")
        cfg.speaker_dim = speaker_dim

    if g2p is not None:
        cfg.char_vocab_size = g2p.vocab_size

    # Load teacher
    teacher = SonataFlowV3(cfg, cfg_dropout_prob=0.0).to(device)
    teacher.load_state_dict(teacher_state, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student: same architecture, init from teacher
    student = SonataFlowV3(cfg, cfg_dropout_prob=0.0).to(device)
    student.load_state_dict(teacher.state_dict())

    # EMA target: copy of student, will be updated with EMA
    ema_model = SonataFlowV3(cfg, cfg_dropout_prob=0.0).to(device)
    ema_model.load_state_dict(student.state_dict())
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in student.parameters())
    print(f"  Model: {n_params/1e6:.1f}M params")
    print(f"  Teacher: 8-step Heun (frozen)")
    print(f"  Student: 1-step target (x_0 prediction)")
    print(f"  EMA decay: 0.95 → 0.999 over training")
    print(f"  dt: {1.0/args.teacher_steps:.4f}")

    # Data
    vocab_size = cfg.char_vocab_size
    if args.data_dir:
        dataset = PtDataset(args.data_dir, max_frames=args.max_frames,
                            g2p=g2p, vocab_size=vocab_size)
        print(f"  Dataset: {len(dataset)} utterances from .pt files")
    elif args.manifest:
        dataset = ManifestDataset(args.manifest, cfg, max_frames=args.max_frames, g2p=g2p)
        print(f"  Dataset: {len(dataset)} utterances from manifest")
    else:
        print("  ERROR: Provide --manifest or --data-dir")
        return

    if len(dataset) == 0:
        print("  ERROR: No data found.")
        return

    val_size = min(max(int(len(dataset) * 0.02), 20), 200)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )

    # EMA with ramp
    ema_wrapper = EMAWithRamp(student, start_decay=0.95, end_decay=0.999,
                              total_steps=args.steps)

    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tlog = TrainingLog(str(ckpt_dir / "distill_losses.jsonl")) if TrainingLog else None

    dt = 1.0 / args.teacher_steps
    step = 0
    t0 = time.time()
    running_loss = 0.0

    print(f"\n  Training: step 0 → {args.steps}, batch={args.batch_size}")
    print(f"  t in [0.02, {min(0.98, 1-dt):.2f}], sway={cfg.sway_coefficient}")

    def cosine_lr(s, warmup=500, max_lr=args.lr, min_ratio=0.01):
        if s < warmup:
            return max_lr * (s + 1) / max(1, warmup)
        ratio = (s - warmup) / max(1, args.steps - warmup)
        return max_lr * (min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * ratio)))

    student.train()
    while step < args.steps:
        for chars, mel, mel_mask_batch, speakers in loader:
            if step >= args.steps:
                break

            chars = chars.to(device)
            mel = mel.to(device)
            mel_mask_batch = mel_mask_batch.to(device)
            speakers = speakers.to(device) if cfg.n_speakers > 0 else None

            # Sync ema_model with EMA shadow (for consistency target)
            ema_wrapper.apply_shadow()
            ema_model.load_state_dict(student.state_dict())
            ema_wrapper.restore()

            lr = cosine_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            loss = distill_step(
                teacher, student, ema_model,
                mel, chars, speakers,
                mel_mask_batch,
                dt=dt,
                t_min=0.02,
                t_max=0.98,
                sway=cfg.sway_coefficient,
            )

            if torch.isnan(loss):
                print(f"  [WARN] step {step}: NaN loss, skipping")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            # Update EMA shadow from student
            ema_wrapper.update(step)

            running_loss += loss.item()
            step += 1

            if step % args.log_every == 0:
                n = args.log_every
                elapsed = time.time() - t0
                avg = running_loss / n
                print(f"  step {step:6d} | loss={avg:.4f} | lr={lr:.2e} | {n/elapsed:.1f} steps/s")
                if tlog:
                    tlog.log(step=step, loss=avg, lr=lr, steps_per_sec=n / elapsed)
                running_loss = 0.0
                t0 = time.time()

            if step % args.val_every == 0 and len(val_ds) > 0:
                # Use EMA student for validation
                ema_wrapper.apply_shadow()
                val_batch = next(iter(val_loader))
                v_chars = val_batch[0].to(device)
                v_mel = val_batch[1].to(device)
                v_mask = val_batch[2].to(device)
                v_speakers = val_batch[3].to(device) if cfg.n_speakers > 0 else None
                mse_diff = validate_distillation(
                    student, teacher, v_chars, v_mel, v_mask, v_speakers,
                    cfg, teacher_steps=args.teacher_steps,
                )
                ema_wrapper.restore()
                print(f"  [val] step {step}: 1-step vs {args.teacher_steps}-step MSE = {mse_diff:.4f}")
                if tlog:
                    tlog.log(step=step, val_mse_diff=mse_diff)

            if step % args.save_every == 0:
                ema_wrapper.apply_shadow()
                save_dict = {
                    "model": student.state_dict(),
                    "ema": ema_wrapper.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
                }
                path = ckpt_dir / f"flow_v3_distill_step_{step}.pt"
                torch.save(save_dict, path)
                ema_wrapper.restore()
                print(f"  [ckpt] {path}")

    # Final save
    ema_wrapper.apply_shadow()
    save_dict = {
        "model": student.state_dict(),
        "ema": ema_wrapper.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
    }
    path = ckpt_dir / "flow_v3_distill_final.pt"
    torch.save(save_dict, path)
    ema_wrapper.restore()
    print(f"\n  Final: {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consistency distillation for Sonata Flow v3 (8-step → 1-step)"
    )
    parser.add_argument("--teacher-checkpoint", required=True,
                        help="Path to trained teacher .pt checkpoint")
    parser.add_argument("--manifest", default="", help="Manifest JSONL")
    parser.add_argument("--data-dir", default="", help="Directory with .pt files")
    parser.add_argument("--output-dir", default="checkpoints/flow_v3_distilled")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--teacher-steps", type=int, default=8)
    parser.add_argument("--model-size", default="base", choices=["base", "large"])
    parser.add_argument("--phonemes", action="store_true",
                        help="Use PhonemeFrontend for text encoding")
    parser.add_argument("--n-speakers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-frames", type=int, default=800)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--val-every", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    train(args)
