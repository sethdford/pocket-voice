#!/usr/bin/env python3
"""Pre-compute phoneme IDs for LibriTTS-R manifest.

Reads manifest, encodes text to phoneme IDs via PhonemeFrontend (espeak-ng),
writes new manifest with "phoneme_ids" field. Sequential processing with
batched phonemizer calls (espeak-ng is NOT thread-safe).

Usage:
    python precompute_phonemes.py

Input:  train/data/libritts_r_full_manifest.jsonl
Output: train/data/libritts_r_full_manifest_phonemes.jsonl
"""

import json
import sys
import time
from pathlib import Path

# Add train/sonata for g2p import
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SONATA_DIR = REPO_ROOT / "train" / "sonata"
sys.path.insert(0, str(SONATA_DIR))

from g2p import PhonemeFrontend, PHONE_TO_ID

BATCH_SIZE = 256  # Texts per phonemize() call
PROGRESS_INTERVAL = 10000
INPUT_MANIFEST = REPO_ROOT / "train" / "data" / "libritts_r_full_manifest.jsonl"
OUTPUT_MANIFEST = REPO_ROOT / "train" / "data" / "libritts_r_full_manifest_phonemes.jsonl"


def phoneme_ids_from_str(phoneme_str: str, add_bos: bool = True, add_eos: bool = True) -> list:
    """Convert phoneme string to list of token IDs (same logic as PhonemeFrontend.encode)."""
    tokens = phoneme_str.split()
    ids = []
    if add_bos:
        ids.append(PHONE_TO_ID["<bos>"])
    for token in tokens:
        if token in PHONE_TO_ID:
            ids.append(PHONE_TO_ID[token])
        else:
            for ch in token:
                if ch in PHONE_TO_ID:
                    ids.append(PHONE_TO_ID[ch])
    if add_eos:
        ids.append(PHONE_TO_ID["<eos>"])
    return ids


def main():
    print(f"[precompute_phonemes] Input:  {INPUT_MANIFEST}")
    print(f"[precompute_phonemes] Output: {OUTPUT_MANIFEST}")
    print(f"[precompute_phonemes] Batch size: {BATCH_SIZE}")
    print()

    if not INPUT_MANIFEST.exists():
        print(f"ERROR: Input manifest not found: {INPUT_MANIFEST}")
        sys.exit(1)

    g2p = PhonemeFrontend()
    # Initialize backend (phonemize) eagerly
    g2p.text_to_phonemes("hello")
    phonemize_fn = g2p._phonemize
    separator = g2p._separator
    language = g2p.language

    entries = []
    with open(INPUT_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    total = len(entries)
    print(f"[precompute_phonemes] Loaded {total:,} entries")
    print()

    start = time.perf_counter()
    processed = 0
    last_progress = 0

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as out:
        for i in range(0, total, BATCH_SIZE):
            batch = entries[i : i + BATCH_SIZE]
            texts = [e["text"] for e in batch]

            # Single batched phonemize call (sequential, no threads)
            phoneme_strs = phonemize_fn(
                texts,
                language=language,
                backend="espeak",
                separator=separator,
                strip=True,
            )

            for j, (entry, phoneme_str) in enumerate(zip(batch, phoneme_strs)):
                entry_copy = dict(entry)
                entry_copy["phoneme_ids"] = phoneme_ids_from_str(phoneme_str)
                out.write(json.dumps(entry_copy, ensure_ascii=False) + "\n")

            processed += len(batch)
            if processed - last_progress >= PROGRESS_INTERVAL:
                elapsed = time.perf_counter() - start
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                print(f"  {processed:,} / {total:,} ({100 * processed / total:.1f}%) - "
                      f"{rate:.0f} entries/s, ETA {eta / 60:.1f} min")
                last_progress = processed

    elapsed = time.perf_counter() - start
    print()
    print("=" * 60)
    print("[precompute_phonemes] DONE")
    print(f"  Total entries:   {total:,}")
    print(f"  Processing time: {elapsed / 60:.2f} min ({elapsed:.1f} s)")
    print(f"  Throughput:     {total / elapsed:.1f} entries/s")
    print(f"  Output:         {OUTPUT_MANIFEST}")
    print("=" * 60)


if __name__ == "__main__":
    main()
