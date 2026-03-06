"""Train Sonata STT — two-pass speech recognition.

Mode 'ctc':     Train CTC head on codec encoder (Pass 1) — with SpecAugment + speed perturb.
Mode 'refiner': Train semantic→text encoder-decoder (Pass 2).
Mode 'both':    Train CTC first, then refiner sequentially.

CTC uses (audio, text) pairs via the encoded dataset (re-extracts mel from codec).
Refiner uses (semantic_tokens, text) pairs directly from the encoded dataset.

Usage:
  # CTC on raw audio data directory (WAV files)
  python train_stt.py --mode ctc --audio-dir train/data/LibriSpeech/train-clean-100 \
    --codec-ckpt train/checkpoints/codec_v3/sonata_codec_final.pt

  # CTC with larger encoder (12L d=512, ~80M params)
  python train_stt.py --mode ctc --audio-dir ... --encoder-size large

  # Refiner on encoded data
  python train_stt.py --mode refiner --data train/data/encoded_dev-clean_v3_final.pt

  # Both sequentially
  python train_stt.py --mode both --data train/data/encoded_dev-clean_v3_final.pt \
    --audio-dir train/data/LibriSpeech/train-clean-100 \
    --codec-ckpt train/checkpoints/codec_v3/sonata_codec_final.pt
"""

import argparse
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

from config import STTConfig, STTLargeConfig, RefinerConfig
from modules import cosine_lr
from stt import (SonataCTC, SonataRefiner, SEMANTIC_PAD_ID,
                 text_to_ctc_ids, ctc_ids_to_text,
                 speed_perturb, spec_augment, apply_noise_augmentations)


# ═════════════════════════════════════════════════════════════════════════════
# Datasets
# ═════════════════════════════════════════════════════════════════════════════

class AudioCTCDataset(Dataset):
    """Raw audio (WAV) + text pairs for CTC training with speed perturbation.

    Expects LibriSpeech-style layout: speaker/chapter/speaker-chapter-id.flac
    with .trans.txt transcripts, or flat directory with (audio.wav, audio.txt).
    """

    def __init__(self, audio_dir: str, sample_rate: int = 24000,
                 max_samples: int = 24000 * 15, speed_factors=None, stt_config=None):
        self.pairs = []
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.speed_factors = speed_factors or [1.0]
        self.stt_config = stt_config
        self._scan_librispeech(audio_dir)
        if not self.pairs:
            self._scan_flat(audio_dir)
        print(f"[AudioCTC] {len(self.pairs)} utterances from {audio_dir}")

    def _scan_librispeech(self, root):
        root = Path(root)
        for trans_file in sorted(root.rglob("*.trans.txt")):
            lines = trans_file.read_text().strip().split("\n")
            for line in lines:
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                uid, text = parts
                for ext in (".flac", ".wav"):
                    audio_path = trans_file.parent / f"{uid}{ext}"
                    if audio_path.exists():
                        self.pairs.append((str(audio_path), text.strip()))
                        break

    def _scan_flat(self, root):
        root = Path(root)
        for wav in sorted(root.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if txt.exists():
                text = txt.read_text().strip()
                self.pairs.append((str(wav), text))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        audio_path, text = self.pairs[idx]
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        wav = torch.from_numpy(data)
        if sr != self.sample_rate:
            ratio = self.sample_rate / sr
            new_len = int(wav.shape[0] * ratio)
            wav = F.interpolate(wav.unsqueeze(0).unsqueeze(0), size=new_len,
                                mode="linear", align_corners=False).squeeze()

        if wav.shape[0] > self.max_samples:
            wav = wav[:self.max_samples]

        factor = random.choice(self.speed_factors)
        if abs(factor - 1.0) > 1e-4:
            wav = speed_perturb(wav.unsqueeze(0), factor).squeeze(0)

        if self.stt_config is not None:
            wav = apply_noise_augmentations(wav, self.sample_rate, self.stt_config)

        text_ids = torch.tensor(text_to_ctc_ids(text, append_eou=True), dtype=torch.long)
        return wav, text_ids


def audio_ctc_collate(batch):
    wav_list, text_list = zip(*batch)
    max_wav = max(w.shape[0] for w in wav_list)
    max_text = max(t.shape[0] for t in text_list)
    B = len(batch)

    wav_padded = torch.zeros(B, max_wav)
    text_padded = torch.zeros(B, max_text, dtype=torch.long)
    wav_lens = torch.zeros(B, dtype=torch.long)
    text_lens = torch.zeros(B, dtype=torch.long)

    for i, (w, t) in enumerate(zip(wav_list, text_list)):
        wav_padded[i, :w.shape[0]] = w
        text_padded[i, :t.shape[0]] = t
        wav_lens[i] = w.shape[0]
        text_lens[i] = t.shape[0]

    return wav_padded, text_padded, wav_lens, text_lens


class CTCDataset(Dataset):
    """Fallback: (semantic_tokens, text) pairs from encoded dataset.
    Uses semantic token count as proxy for mel frame count.
    """

    def __init__(self, data_path: str, max_audio_frames: int = 500):
        self.data = []
        for dp in data_path.split(","):
            dp = dp.strip()
            if not dp:
                continue
            chunk = torch.load(dp, weights_only=False)
            self.data.extend(chunk)
        self.max_audio_frames = max_audio_frames
        print(f"[CTC dataset] {len(self.data)} utterances (encoded)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        text = entry["text"]
        sem_tokens = entry["semantic_tokens"]
        T = min(sem_tokens.shape[0], self.max_audio_frames)
        sem_tokens = sem_tokens[:T]

        text_ids = torch.tensor(text_to_ctc_ids(text, append_eou=True), dtype=torch.long)
        return sem_tokens, text_ids, T


def ctc_collate(batch):
    sem_list, text_list, frame_lens = zip(*batch)
    max_sem = max(s.shape[0] for s in sem_list)
    max_text = max(t.shape[0] for t in text_list)
    B = len(batch)

    sem_padded = torch.zeros(B, max_sem, dtype=torch.long)
    text_padded = torch.zeros(B, max_text, dtype=torch.long)
    sem_lens = torch.zeros(B, dtype=torch.long)
    text_lens = torch.zeros(B, dtype=torch.long)

    for i, (s, t, fl) in enumerate(zip(sem_list, text_list, frame_lens)):
        sem_padded[i, :s.shape[0]] = s
        text_padded[i, :t.shape[0]] = t
        sem_lens[i] = fl
        text_lens[i] = t.shape[0]

    return sem_padded, text_padded, sem_lens, text_lens


class RefinerDataset(Dataset):
    """(semantic_tokens, text) pairs for encoder-decoder training.

    Uses character-level tokenization for simplicity (matching CTC vocab).
    BOS=1, EOS=2 are prepended/appended to text.
    """

    def __init__(self, data_path: str, max_audio_frames: int = 500,
                 max_text_len: int = 256):
        self.data = []
        for dp in data_path.split(","):
            dp = dp.strip()
            if not dp:
                continue
            chunk = torch.load(dp, weights_only=False)
            self.data.extend(chunk)
        self.max_audio_frames = max_audio_frames
        self.max_text_len = max_text_len
        print(f"[Refiner dataset] {len(self.data)} utterances")

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text: str) -> list:
        """Character-level tokenization: 0=pad, 1=BOS, 2=EOS, 3+=chars."""
        ids = [1]  # BOS
        for c in text.lower().strip():
            if c == " ":
                ids.append(3)
            elif c == "'":
                ids.append(30)
            elif "a" <= c <= "z":
                ids.append(ord(c) - ord("a") + 4)
            # skip unknown
        ids.append(2)  # EOS
        return ids

    def __getitem__(self, idx):
        entry = self.data[idx]
        sem_tokens = entry["semantic_tokens"]
        T = min(sem_tokens.shape[0], self.max_audio_frames)
        sem_tokens = sem_tokens[:T]

        text_ids = self._tokenize(entry["text"])
        text_ids = text_ids[:self.max_text_len]
        text_tensor = torch.tensor(text_ids, dtype=torch.long)

        return sem_tokens, text_tensor


def refiner_collate(batch):
    """Collate refiner batches with dedicated semantic pad and padding mask."""
    sem_list, text_list = zip(*batch)
    max_sem = max(s.shape[0] for s in sem_list)
    max_text = max(t.shape[0] for t in text_list)
    B = len(batch)

    sem_padded = torch.full((B, max_sem), SEMANTIC_PAD_ID, dtype=torch.long)
    text_padded = torch.zeros(B, max_text, dtype=torch.long)
    sem_lens = torch.zeros(B, dtype=torch.long)

    for i, (s, t) in enumerate(zip(sem_list, text_list)):
        sem_padded[i, :s.shape[0]] = s
        text_padded[i, :t.shape[0]] = t
        sem_lens[i] = s.shape[0]

    enc_padding_mask = sem_padded == SEMANTIC_PAD_ID  # (B, T_enc), True = pad
    return sem_padded, text_padded, enc_padding_mask, sem_lens


# ═════════════════════════════════════════════════════════════════════════════
# Training Loops
# ═════════════════════════════════════════════════════════════════════════════

def train_ctc(args):
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  SONATA STT — CTC TRAINING (Pass 1)")
    print(f"{'='*60}")

    if args.encoder_size == "large":
        cfg = STTLargeConfig()
        print(f"  Encoder: LARGE (12L d=512, RoPE)")
    else:
        cfg = STTConfig()
        print(f"  Encoder: BASE (4L d=256, RoPE={cfg.use_rope})")
    print(f"  Vocab: {cfg.text_vocab_size} (incl blank + <eou>)")
    print(f"  SpecAugment: {cfg.spec_augment} (F={cfg.freq_mask_width}, T={cfg.time_mask_width})")
    print(f"  Speed perturb: {cfg.speed_perturb} → {cfg.speed_factors}")
    print(f"  Noise augment: {cfg.noise_augment} (SNR={cfg.noise_snr_range} dB, RT60={cfg.reverb_rt60_range}s)")

    model = SonataCTC(cfg).to(device)

    freeze_encoder = not args.unfreeze and args.encoder_size != "large"
    if args.codec_ckpt:
        model.load_codec_encoder(args.codec_ckpt, freeze=freeze_encoder)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M total")

    use_audio = args.audio_dir and os.path.isdir(args.audio_dir)
    if use_audio:
        speed_factors = cfg.speed_factors if cfg.speed_perturb else [1.0]
        dataset = AudioCTCDataset(
            args.audio_dir, sample_rate=cfg.sample_rate,
            max_samples=int(cfg.sample_rate * 15), speed_factors=speed_factors,
            stt_config=cfg,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=audio_ctc_collate, num_workers=args.num_workers, drop_last=True,
        )
    else:
        dataset = CTCDataset(args.data, max_audio_frames=args.max_frames)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=ctc_collate, num_workers=args.num_workers, drop_last=True,
        )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )

    ckpt_dir = Path(args.output_dir) / "ctc"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"  Resumed from step {start_step}")

    model.train()
    step = start_step
    running_loss = 0.0
    t0 = time.time()
    print(f"  Training: step {step} → {args.ctc_steps}, batch={args.batch_size}\n")

    while step < args.ctc_steps:
        if use_audio:
            for wav, text_ids, wav_lens, text_lens in loader:
                if step >= args.ctc_steps:
                    break
                wav = wav.to(device)
                text_ids = text_ids.to(device)
                text_lens = text_lens.to(device)

                lr = cosine_lr(step, args.warmup, args.lr, args.lr * 0.01, args.ctc_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                logits = model(wav, augment=True)
                T_out = logits.shape[1]
                frame_lens = (wav_lens.float() / cfg.hop_length).long().clamp(max=T_out).to(device)

                log_probs = logits.transpose(0, 1).log_softmax(2)
                loss = F.ctc_loss(log_probs, text_ids, frame_lens, text_lens,
                                  blank=cfg.blank_id, zero_infinity=True)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                running_loss += loss.item()
                step += 1
                if step % args.log_every == 0:
                    n = args.log_every
                    elapsed = time.time() - t0
                    print(f"  step {step:6d} | loss={running_loss/n:.4f}"
                          f" | lr={lr:.2e} | {n/elapsed:.1f} steps/s")
                    running_loss = 0.0
                    t0 = time.time()
                if step % args.save_every == 0:
                    path = ckpt_dir / f"sonata_ctc_step_{step}.pt"
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "step": step, "config": vars(cfg)}, path)
                    print(f"  [ckpt] {path}")
        else:
            for sem_tokens, text_ids, sem_lens, text_lens in loader:
                if step >= args.ctc_steps:
                    break
                sem_tokens = sem_tokens.to(device)
                text_ids = text_ids.to(device)
                sem_lens = sem_lens.to(device)
                text_lens = text_lens.to(device)

                lr = cosine_lr(step, args.warmup, args.lr, args.lr * 0.01, args.ctc_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                B, T = sem_tokens.shape
                with torch.no_grad():
                    tok_emb = F.embedding(sem_tokens, torch.randn(32768, cfg.n_mels, device=device))
                    proxy_mel = tok_emb.transpose(1, 2)

                if cfg.use_rope:
                    x = model.encoder(proxy_mel)
                else:
                    x = proxy_mel.transpose(1, 2)
                    x = model.encoder.input_proj(x)
                    for block in model.encoder.blocks:
                        x = block(x)
                x = model.adapter(x)
                logits = model.ctc_proj(x)

                log_probs = logits.transpose(0, 1).log_softmax(2)
                loss = F.ctc_loss(log_probs, text_ids, sem_lens, text_lens,
                                  blank=cfg.blank_id, zero_infinity=True)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                running_loss += loss.item()
                step += 1
                if step % args.log_every == 0:
                    n = args.log_every
                    elapsed = time.time() - t0
                    print(f"  step {step:6d} | loss={running_loss/n:.4f}"
                          f" | lr={lr:.2e} | {n/elapsed:.1f} steps/s")
                    running_loss = 0.0
                    t0 = time.time()
                if step % args.save_every == 0:
                    path = ckpt_dir / f"sonata_ctc_step_{step}.pt"
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "step": step, "config": vars(cfg)}, path)
                    print(f"  [ckpt] {path}")

    path = ckpt_dir / "sonata_ctc_final.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "config": vars(cfg)}, path)
    print(f"\n  Final: {path}")
    return model


def train_refiner(args):
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  SONATA STT — REFINER TRAINING (Pass 2)")
    print(f"{'='*60}")

    cfg = RefinerConfig()
    model = SonataRefiner(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.1f}M params")
    print(f"  Encoder: {cfg.enc_n_layers}L × d={cfg.enc_d_model}")
    print(f"  Decoder: {cfg.dec_n_layers}L × d={cfg.dec_d_model}")

    dataset = RefinerDataset(args.data, max_audio_frames=args.max_frames,
                             max_text_len=args.max_text_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=refiner_collate, num_workers=args.num_workers, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )

    ckpt_dir = Path(args.output_dir) / "refiner"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"  Resumed from step {start_step}")

    model.train()
    step = start_step
    running_loss = 0.0
    running_acc = 0.0
    t0 = time.time()
    print(f"  Training: step {step} → {args.refiner_steps}, batch={args.batch_size}\n")

    while step < args.refiner_steps:
        for sem_tokens, text_tokens, enc_padding_mask, _sem_lens in loader:
            if step >= args.refiner_steps:
                break

            sem_tokens = sem_tokens.to(device)
            text_tokens = text_tokens.to(device)
            enc_padding_mask = enc_padding_mask.to(device)

            # Teacher forcing: input is text[:-1], target is text[1:]
            text_input = text_tokens[:, :-1]
            text_target = text_tokens[:, 1:]

            lr = cosine_lr(step, args.warmup, args.lr, args.lr * 0.01, args.refiner_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            logits, losses = model(
                sem_tokens, text_input, target_text=text_target,
                enc_padding_mask=enc_padding_mask,
            )
            loss = losses["text"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = text_target != 0
                if mask.any():
                    acc = (preds[mask] == text_target[mask]).float().mean().item()
                else:
                    acc = 0.0

            running_loss += loss.item()
            running_acc += acc
            step += 1

            if step % args.log_every == 0:
                n = args.log_every
                elapsed = time.time() - t0
                print(f"  step {step:6d} | loss={running_loss/n:.4f}"
                      f" acc={running_acc/n:.2%}"
                      f" | lr={lr:.2e} | {n/elapsed:.1f} steps/s")
                running_loss = 0.0
                running_acc = 0.0
                t0 = time.time()

            if step % args.save_every == 0:
                path = ckpt_dir / f"sonata_refiner_step_{step}.pt"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step, "config": vars(cfg)}, path)
                print(f"  [ckpt] {path}")

    path = ckpt_dir / "sonata_refiner_final.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "config": vars(cfg)}, path)
    print(f"\n  Final: {path}")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sonata STT training")
    parser.add_argument("--mode", choices=["ctc", "refiner", "both"], default="both")
    parser.add_argument("--data", default="train/data/encoded_dev-clean_v3_final.pt")
    parser.add_argument("--audio-dir", default="",
                        help="Directory of audio files for CTC (LibriSpeech layout or flat)")
    parser.add_argument("--codec-ckpt", default="")
    parser.add_argument("--output-dir", default="train/checkpoints/stt")
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--encoder-size", choices=["base", "large"], default="base",
                        help="Encoder size: base (4L d=256) or large (12L d=512)")
    parser.add_argument("--unfreeze", action="store_true",
                        help="Fine-tune full encoder (don't freeze codec weights)")

    parser.add_argument("--ctc-steps", type=int, default=10000)
    parser.add_argument("--refiner-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--max-text-len", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if args.mode in ("ctc", "both"):
        train_ctc(args)
    if args.mode in ("refiner", "both"):
        train_refiner(args)

    print("\n  STT training complete.")


if __name__ == "__main__":
    main()
