"""Prepare mel+text data for Sonata Flow v2 training.

Walks LibriSpeech/VCTK directories, extracts mel spectrograms from audio,
pairs with transcripts, and saves as .pt files for training.

Usage:
  python prepare_flow_v2_data.py \
    --audio-dir ../data/LibriSpeech/train-clean-360 \
    --output-dir ../data/flow_v2_360h \
    --sample-rate 24000 --n-mels 80
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


def extract_mel(audio: torch.Tensor, sample_rate: int = 24000,
                n_fft: int = 1024, hop_length: int = 480,
                n_mels: int = 80) -> torch.Tensor:
    """Extract log-mel spectrogram using torch operations only."""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    window = torch.hann_window(n_fft, periodic=True, device=audio.device)
    stft = torch.stft(
        audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, return_complex=True, center=True, pad_mode="reflect",
    )
    mag = stft.abs().squeeze(0).T  # (T, n_fft//2+1)

    n_bins = n_fft // 2 + 1
    fmin, fmax = 0.0, sample_rate / 2.0
    mel_lo = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_hi = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)

    fb = np.zeros((n_mels, n_bins))
    for m in range(n_mels):
        lo, mid, hi = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
        for k in range(lo, mid):
            if mid > lo:
                fb[m, k] = (k - lo) / (mid - lo)
        for k in range(mid, hi):
            if hi > mid:
                fb[m, k] = (hi - k) / (hi - mid)
    fb = torch.from_numpy(fb).float().to(audio.device)

    mel = torch.matmul(mag, fb.T)
    mel = torch.log(mel.clamp(min=1e-7))
    return mel  # (T, n_mels)


def load_and_resample(path: str, target_sr: int = 24000) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data)
    if sr != target_sr:
        ratio = target_sr / sr
        new_len = int(wav.shape[0] * ratio)
        wav = F.interpolate(
            wav.unsqueeze(0).unsqueeze(0), size=new_len,
            mode="linear", align_corners=False,
        ).squeeze()
    return wav


def process_librispeech(audio_dir: str, output_dir: str, args):
    """Process LibriSpeech/LibriTTS-R format data (speaker/chapter/utterances).

    Supports both .trans.txt (LibriSpeech) and per-file .normalized.txt (LibriTTS-R).
    """
    root = Path(audio_dir)
    os.makedirs(output_dir, exist_ok=True)

    pairs = []

    # Try LibriSpeech format first: .trans.txt files
    trans_files = sorted(root.rglob("*.trans.txt"))
    if trans_files:
        for trans_file in trans_files:
            speaker_id = trans_file.parts[-3]
            for line in trans_file.read_text().strip().split("\n"):
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                uid, text = parts
                for ext in (".flac", ".wav"):
                    audio_path = trans_file.parent / f"{uid}{ext}"
                    if audio_path.exists():
                        pairs.append((str(audio_path), text.strip().lower(), speaker_id))
                        break
    else:
        # LibriTTS-R format: per-file .normalized.txt alongside .wav
        for txt_file in sorted(root.rglob("*.normalized.txt")):
            text = txt_file.read_text().strip().lower()
            if not text:
                continue
            base = txt_file.stem.replace(".normalized", "")
            speaker_id = base.split("_")[0]
            for ext in (".wav", ".flac"):
                audio_path = txt_file.parent / f"{base}{ext}"
                if audio_path.exists():
                    pairs.append((str(audio_path), text, speaker_id))
                    break

    print(f"  Found {len(pairs)} utterances from {len(set(p[2] for p in pairs))} speakers")

    speaker_map = {}
    idx = 0
    for i, (audio_path, text, speaker_id) in enumerate(pairs):
        if speaker_id not in speaker_map:
            speaker_map[speaker_id] = len(speaker_map)

        try:
            wav = load_and_resample(audio_path, args.sample_rate)
            if wav.shape[0] < args.sample_rate * 0.5:
                continue
            if wav.shape[0] > args.sample_rate * args.max_duration:
                continue

            mel = extract_mel(wav, args.sample_rate, args.n_fft, args.hop_length, args.n_mels)

            data = {
                "text": text,
                "mel": mel,
                "speaker_id": speaker_map[speaker_id],
                "n_frames": mel.shape[0],
            }

            out_path = os.path.join(output_dir, f"utt_{idx:06d}.pt")
            torch.save(data, out_path)
            idx += 1

        except Exception as e:
            print(f"  WARN: {audio_path}: {e}")
            continue

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} ({idx} saved)")

    print(f"  Total: {idx} utterances saved to {output_dir}")

    meta = {"n_utterances": idx, "n_speakers": len(speaker_map), "speaker_map": speaker_map}
    torch.save(meta, os.path.join(output_dir, "meta.pt"))


def process_vctk(audio_dir: str, output_dir: str, args):
    """Process VCTK-format data (speaker/utterances in wav48_silence_trimmed)."""
    root = Path(audio_dir)
    wav_root = root / "wav48_silence_trimmed"
    txt_root = root / "txt"

    if not wav_root.exists():
        print(f"  VCTK wav dir not found: {wav_root}")
        return

    os.makedirs(output_dir, exist_ok=True)

    speaker_dirs = sorted([d for d in wav_root.iterdir() if d.is_dir()])
    print(f"  Found {len(speaker_dirs)} VCTK speakers")

    idx = 0
    speaker_map = {}
    for spk_dir in speaker_dirs:
        spk_id = spk_dir.name
        speaker_map[spk_id] = len(speaker_map)

        for audio_file in sorted(spk_dir.glob("*_mic1.flac")):
            base = audio_file.stem.replace("_mic1", "")
            txt_file = txt_root / spk_id / f"{base}.txt"
            if not txt_file.exists():
                continue

            text = txt_file.read_text().strip().lower()
            if not text:
                continue

            try:
                wav = load_and_resample(str(audio_file), args.sample_rate)
                if wav.shape[0] < args.sample_rate * 0.5:
                    continue
                if wav.shape[0] > args.sample_rate * args.max_duration:
                    continue

                mel = extract_mel(wav, args.sample_rate, args.n_fft, args.hop_length, args.n_mels)

                data = {
                    "text": text,
                    "mel": mel,
                    "speaker_id": speaker_map[spk_id],
                    "n_frames": mel.shape[0],
                }

                out_path = os.path.join(output_dir, f"vctk_{idx:06d}.pt")
                torch.save(data, out_path)
                idx += 1

            except Exception as e:
                continue

    print(f"  Total: {idx} VCTK utterances saved to {output_dir}")


def process_ljspeech(audio_dir: str, output_dir: str, args):
    """Process LJSpeech-format data (single speaker, metadata.csv)."""
    root = Path(audio_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Look for metadata.csv (may be nested under LJSpeech-1.1/)
    meta_path = root / "metadata.csv"
    if not meta_path.exists():
        for sub in root.iterdir():
            if (sub / "metadata.csv").exists():
                root = sub
                meta_path = root / "metadata.csv"
                break

    if not meta_path.exists():
        print(f"  ERROR: metadata.csv not found in {audio_dir}")
        return

    wav_dir = root / "wavs"
    pairs = []
    for line in meta_path.read_text().strip().split("\n"):
        parts = line.split("|")
        if len(parts) < 3:
            continue
        uid = parts[0].strip()
        text = parts[2].strip().lower()  # normalized text
        audio_path = wav_dir / f"{uid}.wav"
        if audio_path.exists():
            pairs.append((str(audio_path), text, "ljspeech"))

    print(f"  Found {len(pairs)} LJSpeech utterances")

    idx = 0
    for i, (audio_path, text, _) in enumerate(pairs):
        try:
            wav = load_and_resample(audio_path, args.sample_rate)
            if wav.shape[0] < args.sample_rate * 0.5:
                continue
            if wav.shape[0] > args.sample_rate * args.max_duration:
                continue

            mel = extract_mel(wav, args.sample_rate, args.n_fft, args.hop_length, args.n_mels)

            data = {
                "text": text,
                "mel": mel,
                "speaker_id": 0,
                "n_frames": mel.shape[0],
            }

            out_path = os.path.join(output_dir, f"lj_{idx:06d}.pt")
            torch.save(data, out_path)
            idx += 1

        except Exception as e:
            print(f"  WARN: {audio_path}: {e}")
            continue

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} ({idx} saved)")

    print(f"  Total: {idx} LJSpeech utterances saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Flow v2 training data")
    parser.add_argument("--audio-dir", required=True, help="LibriSpeech or VCTK root")
    parser.add_argument("--output-dir", required=True, help="Output directory for .pt files")
    parser.add_argument("--format", choices=["librispeech", "vctk", "ljspeech"], default="librispeech")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=480)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--max-duration", type=float, default=20.0, help="Max audio seconds")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SONATA FLOW v2 — DATA PREPARATION")
    print(f"{'='*60}")
    print(f"  Input: {args.audio_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Format: {args.format}")
    print(f"  Sample rate: {args.sample_rate}, Mel bins: {args.n_mels}")

    if args.format == "librispeech":
        process_librispeech(args.audio_dir, args.output_dir, args)
    elif args.format == "vctk":
        process_vctk(args.audio_dir, args.output_dir, args)
    elif args.format == "ljspeech":
        process_ljspeech(args.audio_dir, args.output_dir, args)


if __name__ == "__main__":
    main()
