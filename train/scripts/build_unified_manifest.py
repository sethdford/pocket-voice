#!/usr/bin/env python3
"""Build unified multi-dataset manifest with phoneme IDs.

Merges manifests from multiple datasets, normalizes speaker IDs, filters by
duration, and optionally pre-computes phoneme IDs for training.

Usage:
  python build_unified_manifest.py --manifests m1.jsonl m2.jsonl --output train/data/unified.jsonl
  python build_unified_manifest.py --manifests m1.jsonl m2.jsonl --phonemes --workers 4

Output: train/data/unified_manifest_phonemes.jsonl (default)
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# train/scripts/ -> go up to repo root
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SONATA_DIR = os.path.join(REPO_ROOT, "train", "sonata")
sys.path.insert(0, SONATA_DIR)


def _phonemize_chunk(args):
    """Worker: process a chunk of entries, add phoneme_ids. Must be top-level for pickling."""
    chunk_id, entries = args
    try:
        from g2p import PhonemeFrontend

        g2p = PhonemeFrontend()
        g2p.text_to_phonemes("hello")  # Initialize espeak-ng backend
    except ImportError as e:
        return (chunk_id, [], str(e))

    results = []
    for e in entries:
        e2 = dict(e)
        text = (e.get("text") or "").strip()
        if not text:
            e2["phoneme_ids"] = [1, 2]  # bos, eos only for empty
            results.append(e2)
            continue
        try:
            ids = g2p.encode(text, add_bos=True, add_eos=True).tolist()
            e2["phoneme_ids"] = ids
        except Exception:
            e2["phoneme_ids"] = [1, 2]  # fallback
        results.append(e2)

    return (chunk_id, results, None)


def load_manifest(path: str) -> list:
    """Load JSONL manifest, return list of entries."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Build unified multi-dataset manifest with phoneme IDs"
    )
    parser.add_argument(
        "--manifests",
        nargs="+",
        required=True,
        help="Input manifest JSONL files",
    )
    parser.add_argument(
        "--output",
        default="train/data/unified_manifest_phonemes.jsonl",
        help="Output manifest path",
    )
    parser.add_argument(
        "--phonemes",
        action="store_true",
        help="Pre-compute phoneme IDs via g2p",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum duration in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum duration in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for phoneme computation (default: 4)",
    )
    args = parser.parse_args()

    # Load all manifests
    entries = []
    for manifest_path in args.manifests:
        if not os.path.isfile(manifest_path):
            print(f"WARNING: Manifest not found, skipping: {manifest_path}")
            continue

        dataset_name = os.path.splitext(os.path.basename(manifest_path))[0]
        raw = load_manifest(manifest_path)

        for entry in raw:
            # Normalize speaker ID (support both 'speaker' and 'speaker_id')
            speaker = entry.get("speaker_id") or entry.get("speaker")
            if speaker is not None:
                entry["speaker_id"] = f"{dataset_name}_{speaker}"
            else:
                entry["speaker_id"] = f"{dataset_name}_unknown"

            # Filter by duration
            dur = entry.get("duration")
            if dur is None:
                try:
                    import soundfile as sf

                    info = sf.info(entry.get("audio", ""))
                    dur = info.duration
                    entry["duration"] = round(dur, 3)
                except Exception:
                    continue

            if not isinstance(dur, (int, float)) or dur != dur:  # NaN check
                continue
            if dur < args.min_duration or dur > args.max_duration:
                continue

            # Skip empty text
            text = (entry.get("text") or "").strip()
            if not text:
                continue

            entries.append(entry)

    print(f"Loaded {len(entries)} entries from {len(args.manifests)} manifests")

    # Pre-compute phoneme IDs
    if args.phonemes:
        try:
            from g2p import PhonemeFrontend

            PhonemeFrontend()
        except ImportError:
            print(
                "ERROR: --phonemes requires g2p module. Install phonemizer: pip install phonemizer"
            )
            sys.exit(1)

        if args.workers <= 1:
            from g2p import PhonemeFrontend

            g2p = PhonemeFrontend()
            g2p.text_to_phonemes("hello")
            for i, entry in enumerate(entries):
                if "phoneme_ids" in entry:
                    continue
                text = entry.get("text", "").strip()
                if text:
                    try:
                        ids = g2p.encode(text, add_bos=True, add_eos=True).tolist()
                        entry["phoneme_ids"] = ids
                    except Exception:
                        entry["phoneme_ids"] = [1, 2]
                else:
                    entry["phoneme_ids"] = [1, 2]
                if (i + 1) % 10000 == 0:
                    print(f"  Phonemized {i + 1}/{len(entries)}")
            print(f"Phonemized all {len(entries)} entries")
        else:
            # Multiprocessing: split into chunks
            n = len(entries)
            chunk_size = max(1, (n + args.workers - 1) // args.workers)
            chunks = [
                (i, entries[i : i + chunk_size])
                for i in range(0, n, chunk_size)
            ]

            with Pool(processes=args.workers) as pool:
                results = pool.map(_phonemize_chunk, chunks)

            # Reassemble in order
            ordered = []
            for chunk_id, chunk_entries, err in sorted(results, key=lambda x: x[0]):
                if err:
                    print(f"WARNING: Worker error: {err}")
                    continue
                ordered.extend(chunk_entries)

            entries = ordered
            print(f"Phonemized {len(entries)} entries (workers={args.workers})")

    # Write output
    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Stats
    speakers = set(e.get("speaker_id", "unknown") for e in entries)
    total_hours = sum(e.get("duration", 0) for e in entries) / 3600
    print(
        f"Written {len(entries)} entries, {len(speakers)} speakers, "
        f"{total_hours:.1f} hours → {out_path}"
    )


if __name__ == "__main__":
    main()
