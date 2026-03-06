#!/usr/bin/env python3
"""Prepare prosody-labeled data from public speech emotion datasets.

Extracts audio, text, emotion labels, and prosody features (F0/energy/rate)
from EmoV-DB, RAVDESS, and VCTK datasets into a unified format for:
  - compute_emosteer.py (direction vectors)
  - train_flow.py --use-emotion --use-prosody (prosody-conditioned Flow training)
  - train_lm.py --use-prosody (prosody-conditioned LM training)

Output: directory of .pt files + manifest.jsonl

Usage:
  # From EmoV-DB (4 speakers, 5 emotions, ~7000 utterances)
  python prepare_prosody_data.py --dataset emov-db --data-dir ~/data/EmoV-DB/ --output data/prosody/

  # From RAVDESS (24 speakers, 8 emotions)
  python prepare_prosody_data.py --dataset ravdess --data-dir ~/data/RAVDESS/ --output data/prosody/

  # From VCTK (109 speakers, neutral — for speaker diversity)
  python prepare_prosody_data.py --dataset vctk --data-dir ~/data/VCTK/ --output data/prosody/

  # Combine all
  python prepare_prosody_data.py --dataset all --emov-dir ~/data/EmoV-DB/ \
    --ravdess-dir ~/data/RAVDESS/ --vctk-dir ~/data/VCTK/ --output data/prosody/
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False
    print("WARNING: soundfile not installed. Install with: pip install soundfile")


EMOTION_MAP = {
    # EmoV-DB emotions
    "neutral": "neutral",
    "amused": "happy",
    "angry": "angry",
    "disgusted": "angry",
    "sleepy": "calm",
    # RAVDESS emotions (from filename coding)
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "angry",  # disgust → angry
    "08": "surprised",
}


def extract_prosody_features(audio: np.ndarray, sr: int, n_frames: int) -> torch.Tensor:
    """Extract (log_pitch, energy, speaking_rate) per frame at 50 Hz."""
    hop = sr // 50

    # Energy
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(audio))
        if start < len(audio):
            frame = audio[start:end]
            energy[i] = float(np.sqrt(np.mean(frame ** 2)))

    # F0 via autocorrelation
    f0 = np.zeros(n_frames)
    min_lag = sr // 500
    max_lag = sr // 50
    for i in range(n_frames):
        start = i * hop
        end = min(start + 2 * hop, len(audio))
        if end - start < hop:
            continue
        frame = audio[start:end]
        if frame.max() - frame.min() < 1e-6:
            continue
        corr = np.correlate(frame, frame, mode='full')
        mid = len(corr) // 2
        corr = corr[mid:]
        search_end = min(max_lag, len(corr))
        if search_end <= min_lag:
            continue
        search = corr[min_lag:search_end]
        if len(search) > 0:
            peak = int(search.argmax()) + min_lag
            if peak > 0:
                f0[i] = sr / peak

    # Log pitch with interpolation
    voiced = f0 > 0
    log_pitch = np.zeros(n_frames)
    if voiced.any():
        log_pitch[voiced] = np.log(f0[voiced] + 1.0)

    # Speaking rate (voiced density)
    voiced_float = voiced.astype(float)
    rate = np.zeros(n_frames)
    w = 25
    for i in range(n_frames):
        lo = max(0, i - w)
        hi = min(n_frames, i + w + 1)
        rate[i] = voiced_float[lo:hi].mean()

    # Normalize energy to [0, 1] per utterance
    if energy.max() > 0:
        energy = energy / energy.max()

    # Normalize log_pitch to [0, 1] using fixed human pitch range.
    # Preserves absolute scale between speakers (log(80+1)≈4.4 low male, log(300+1)≈5.7 high female)
    LOG_F0_MIN = 4.0   # ~54 Hz
    LOG_F0_MAX = 6.0   # ~402 Hz
    if log_pitch.max() > 0:
        log_pitch = (log_pitch - LOG_F0_MIN) / (LOG_F0_MAX - LOG_F0_MIN)
        log_pitch = np.clip(log_pitch, 0.0, 1.0)

    return torch.stack([
        torch.from_numpy(log_pitch).float(),
        torch.from_numpy(energy).float(),
        torch.from_numpy(rate).float(),
    ], dim=-1)


def process_emov_db(data_dir: str) -> List[dict]:
    """Process EmoV-DB dataset: speaker/emotion/utterance_id.wav structure."""
    results = []
    data_path = Path(data_dir)

    for speaker_dir in sorted(data_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name

        for emotion_dir in sorted(speaker_dir.iterdir()):
            if not emotion_dir.is_dir():
                continue
            raw_emotion = emotion_dir.name.lower()
            emotion = EMOTION_MAP.get(raw_emotion, raw_emotion)

            wavs = sorted(emotion_dir.glob("*.wav"))
            for wav in wavs:
                text = wav.stem.replace("_", " ")
                results.append({
                    "audio": str(wav),
                    "text": text,
                    "emotion": emotion,
                    "speaker": speaker,
                    "dataset": "emov-db",
                })

    print(f"  EmoV-DB: {len(results)} utterances from {data_path}")
    return results


def process_ravdess(data_dir: str) -> List[dict]:
    """Process RAVDESS dataset: Actor_XX/XX-XX-XX-XX-XX-XX-XX.wav"""
    results = []
    data_path = Path(data_dir)

    ravdess_text = {
        "01": "Kids are talking by the door.",
        "02": "Dogs are sitting by the door.",
    }

    for actor_dir in sorted(data_path.glob("Actor_*")):
        if not actor_dir.is_dir():
            continue
        speaker = actor_dir.name

        for wav in sorted(actor_dir.glob("*.wav")):
            parts = wav.stem.split("-")
            if len(parts) < 7:
                continue
            emotion_code = parts[2]
            statement = parts[4]
            emotion = EMOTION_MAP.get(emotion_code, "neutral")
            text = ravdess_text.get(statement, "Kids are talking by the door.")

            results.append({
                "audio": str(wav),
                "text": text,
                "emotion": emotion,
                "speaker": speaker,
                "dataset": "ravdess",
            })

    print(f"  RAVDESS: {len(results)} utterances from {data_path}")
    return results


def process_vctk(data_dir: str) -> List[dict]:
    """Process VCTK dataset: wav48_silence_trimmed/pXXX/pXXX_YYY_mic1.flac"""
    results = []
    data_path = Path(data_dir)
    txt_dir = data_path / "txt"
    wav_dir = data_path / "wav48_silence_trimmed"

    if not wav_dir.exists():
        wav_dir = data_path / "wav48"

    for speaker_dir in sorted(wav_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name

        for audio_file in sorted(speaker_dir.glob("*_mic1.*")):
            utt_id = audio_file.stem.replace("_mic1", "")
            txt_file = txt_dir / speaker / f"{utt_id}.txt"
            text = ""
            if txt_file.exists():
                text = txt_file.read_text().strip()

            results.append({
                "audio": str(audio_file),
                "text": text,
                "emotion": "neutral",
                "speaker": speaker,
                "dataset": "vctk",
            })

    print(f"  VCTK: {len(results)} utterances from {data_path}")
    return results


def encode_with_prosody(entries: List[dict], output_dir: str,
                        max_duration: float = 15.0, sr_target: int = 24000):
    """Encode audio files with prosody features and save as .pt files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = []
    n_skipped = 0

    for i, entry in enumerate(entries):
        try:
            audio, sr = sf.read(entry["audio"], dtype='float32')
        except Exception as e:
            n_skipped += 1
            continue

        if len(audio.shape) > 1:
            audio = audio.mean(axis=-1)

        # Resample if needed
        if sr != sr_target:
            ratio = sr_target / sr
            new_len = int(len(audio) * ratio)
            from scipy.signal import resample
            audio = resample(audio, new_len).astype(np.float32)
            sr = sr_target

        # Truncate
        max_samples = int(max_duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        n_frames = len(audio) // (sr // 50)
        if n_frames < 5:
            n_skipped += 1
            continue

        prosody = extract_prosody_features(audio, sr, n_frames)

        utt_id = f"{entry['dataset']}_{entry['speaker']}_{i:06d}"
        pt_path = out_path / f"{utt_id}.pt"

        torch.save({
            "audio_path": entry["audio"],
            "text": entry["text"],
            "emotion": entry["emotion"],
            "speaker": entry["speaker"],
            "dataset": entry["dataset"],
            "prosody_features": prosody,
            "n_frames": n_frames,
            "n_samples": len(audio),
            "sr": sr,
        }, pt_path)

        manifest.append({
            "utt_id": utt_id,
            "audio": entry["audio"],
            "text": entry["text"],
            "emotion": entry["emotion"],
            "speaker": entry["speaker"],
            "pt_file": str(pt_path),
            "n_frames": n_frames,
        })

        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{len(entries)}] {utt_id}: {n_frames} frames, "
                  f"emotion={entry['emotion']}")

    # Write manifest
    manifest_path = out_path / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")

    print(f"\n  Encoded {len(manifest)} utterances (skipped {n_skipped})")
    print(f"  Manifest: {manifest_path}")

    # Emotion distribution
    emotion_counts = {}
    for item in manifest:
        e = item["emotion"]
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    print(f"  Emotions: {json.dumps(emotion_counts, indent=2)}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Prepare prosody-labeled data for Sonata training"
    )
    parser.add_argument("--dataset", choices=["emov-db", "ravdess", "vctk", "all"],
                        default="emov-db")
    parser.add_argument("--data-dir", help="Dataset root directory")
    parser.add_argument("--emov-dir", help="EmoV-DB directory (for --dataset all)")
    parser.add_argument("--ravdess-dir", help="RAVDESS directory (for --dataset all)")
    parser.add_argument("--vctk-dir", help="VCTK directory (for --dataset all)")
    parser.add_argument("--output", default="data/prosody/", help="Output directory")
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--manifest-only", action="store_true",
                        help="Only generate manifest without encoding prosody")
    args = parser.parse_args()

    if not HAS_SF:
        print("ERROR: soundfile required. Install: pip install soundfile")
        return

    print(f"\n{'='*60}")
    print(f"  PROSODY DATA PREPARATION")
    print(f"{'='*60}")

    entries = []

    if args.dataset == "emov-db" and args.data_dir:
        entries = process_emov_db(args.data_dir)
    elif args.dataset == "ravdess" and args.data_dir:
        entries = process_ravdess(args.data_dir)
    elif args.dataset == "vctk" and args.data_dir:
        entries = process_vctk(args.data_dir)
    elif args.dataset == "all":
        if args.emov_dir:
            entries.extend(process_emov_db(args.emov_dir))
        if args.ravdess_dir:
            entries.extend(process_ravdess(args.ravdess_dir))
        if args.vctk_dir:
            entries.extend(process_vctk(args.vctk_dir))
    else:
        print("  ERROR: Provide --data-dir (or --emov-dir etc. for --dataset all)")
        return

    if not entries:
        print("  No data found. Check paths.")
        return

    print(f"\n  Total: {len(entries)} entries")

    if args.manifest_only:
        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        manifest_path = out_path / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print(f"  Manifest saved to {manifest_path}")
    else:
        encode_with_prosody(entries, args.output, args.max_duration, args.sr)

    print(f"\n{'='*60}")
    print(f"  Done. Ready for: compute_emosteer.py / train_flow.py / train_lm.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
