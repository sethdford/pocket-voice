#!/usr/bin/env python3
"""Build unified JSONL manifest for LibriTTS-R training data."""

import json
import os

import soundfile as sf

# Data roots
BASE = "/Users/sethford/Documents/pocket-voice/train/data/libritts-r/LibriTTS_R"
SPLITS = ["train-clean-100", "train-clean-360", "train-other-500"]
OUTPUT = "/Users/sethford/Documents/pocket-voice/train/data/libritts_r_full_manifest.jsonl"

# Filters
MIN_DURATION = 0.5
MAX_DURATION = 15.0
MIN_TEXT_LEN = 2


def main():
    total_files = 0
    filtered_count = 0
    total_hours = 0.0
    speakers = set()

    with open(OUTPUT, "w") as out:
        for split in SPLITS:
            split_path = os.path.join(BASE, split)
            if not os.path.isdir(split_path):
                print(f"Warning: {split_path} not found, skipping")
                continue

            for speaker_id in os.listdir(split_path):
                speaker_dir = os.path.join(split_path, speaker_id)
                if not os.path.isdir(speaker_dir):
                    continue

                for chapter_id in os.listdir(speaker_dir):
                    chapter_dir = os.path.join(speaker_dir, chapter_id)
                    if not os.path.isdir(chapter_dir):
                        continue

                    for name in os.listdir(chapter_dir):
                        if not name.endswith(".wav"):
                            continue

                        total_files += 1
                        wav_path = os.path.join(chapter_dir, name)
                        txt_path = os.path.join(
                            chapter_dir,
                            name[:-4] + ".normalized.txt",
                        )

                        if not os.path.isfile(txt_path):
                            continue

                        with open(txt_path) as f:
                            text = f.read().strip()

                        if len(text) < MIN_TEXT_LEN:
                            continue

                        try:
                            info = sf.info(wav_path)
                            duration = info.duration
                        except Exception:
                            continue

                        if duration != duration or duration <= 0:  # NaN or zero
                            continue
                        if duration < MIN_DURATION or duration > MAX_DURATION:
                            continue

                        entry = {
                            "audio": os.path.abspath(wav_path),
                            "text": text,
                            "speaker": speaker_id,
                            "duration": round(duration, 3),
                        }
                        out.write(json.dumps(entry) + "\n")

                        filtered_count += 1
                        total_hours += duration / 3600.0
                        speakers.add(speaker_id)

    print("=" * 50)
    print("LibriTTS-R Manifest Stats")
    print("=" * 50)
    print(f"Total WAV files scanned: {total_files}")
    print(f"Entries after filtering: {filtered_count}")
    print(f"Total hours: {total_hours:.2f}")
    print(f"Unique speakers: {len(speakers)}")
    print(f"Output: {OUTPUT}")
    print("=" * 50)


if __name__ == "__main__":
    main()
