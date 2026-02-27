#!/usr/bin/env python3
"""Wrapper: MFA alignment -> interleaving -> manifest generation.

End-to-end pipeline for preparing SpeakStream-style interleaved training data.
Handles MFA alignment (if installed), falls back to uniform alignment otherwise.

Usage:
  # With MFA alignment (recommended):
  python scripts/prepare_interleaved_manifest.py \\
    --manifest data/manifest_clean.jsonl \\
    --output data/interleaved_manifest.jsonl \\
    --run-mfa --mfa-model english_mfa

  # Without MFA (uniform fallback):
  python scripts/prepare_interleaved_manifest.py \\
    --manifest data/manifest_clean.jsonl \\
    --output data/interleaved_manifest.jsonl

  # Custom chunk size:
  python scripts/prepare_interleaved_manifest.py \\
    --manifest data/manifest_clean.jsonl \\
    --output data/interleaved_manifest.jsonl \\
    --chunk-words 4 --overlap-words 0
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Add train/sonata to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "train" / "sonata"))

from config import FlowV3Config, InterleavedTrainingConfig
from prepare_interleaved import prepare_interleaved_manifest


def check_mfa_installed() -> bool:
    """Check if Montreal Forced Aligner is available."""
    return shutil.which("mfa") is not None


def run_mfa_alignment(
    manifest_path: str,
    output_dir: str,
    mfa_model: str = "english_mfa",
    sample_rate: int = 24000,
) -> str:
    """Run MFA on manifest entries, producing TextGrid alignments.

    Creates a temporary corpus directory with wav + txt pairs, runs MFA,
    and returns the path to the TextGrid output directory.
    """
    if not check_mfa_installed():
        print("  WARNING: MFA not installed. Install with: conda install -c conda-forge montreal-forced-aligner")
        return ""

    textgrid_dir = os.path.join(output_dir, "textgrids")
    os.makedirs(textgrid_dir, exist_ok=True)

    # Create temporary corpus: MFA expects wav + txt pairs in a flat directory
    corpus_dir = os.path.join(output_dir, "mfa_corpus")
    os.makedirs(corpus_dir, exist_ok=True)

    n_prepared = 0
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            audio_path = entry.get("audio", "")
            text = entry.get("text", "")
            if not audio_path or not text or not os.path.exists(audio_path):
                continue

            stem = Path(audio_path).stem
            # Symlink audio
            wav_link = os.path.join(corpus_dir, f"{stem}.wav")
            if not os.path.exists(wav_link):
                os.symlink(os.path.abspath(audio_path), wav_link)
            # Write text
            txt_path = os.path.join(corpus_dir, f"{stem}.txt")
            with open(txt_path, "w") as tf:
                tf.write(text)
            n_prepared += 1

    if n_prepared == 0:
        print("  WARNING: No entries prepared for MFA alignment")
        return ""

    print(f"  Running MFA on {n_prepared} utterances...")
    print(f"  Model: {mfa_model}")

    cmd = [
        "mfa", "align",
        corpus_dir,
        mfa_model,  # acoustic model
        mfa_model,  # dictionary (same name for pretrained)
        textgrid_dir,
        "--clean",
        "--overwrite",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            print(f"  MFA failed (exit {result.returncode}):")
            print(f"    {result.stderr[:500]}")
            return ""
        print(f"  MFA alignment complete")
    except FileNotFoundError:
        print("  ERROR: 'mfa' command not found")
        return ""
    except subprocess.TimeoutExpired:
        print("  ERROR: MFA timed out after 1 hour")
        return ""

    return textgrid_dir


def validate_output(output_path: str) -> dict:
    """Validate interleaved manifest quality.

    Checks for empty chunks, minimum duration, and data consistency.
    """
    issues = {"empty_chunks": 0, "short_chunks": 0, "total": 0, "valid": 0}

    if not os.path.exists(output_path):
        return issues

    with open(output_path) as f:
        for line in f:
            entry = json.loads(line)
            issues["total"] += 1

            text_chunks = entry.get("text_chunks", [])
            frame_counts = entry.get("speech_chunk_frames", [])

            if len(text_chunks) != len(frame_counts):
                issues["empty_chunks"] += 1
                continue

            valid = True
            for text, frames in zip(text_chunks, frame_counts):
                if not text.strip():
                    issues["empty_chunks"] += 1
                    valid = False
                if frames < 5:  # Less than ~100ms at 50Hz
                    issues["short_chunks"] += 1
                    valid = False

            if valid:
                issues["valid"] += 1

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end interleaved manifest preparation"
    )
    parser.add_argument("--manifest", required=True, help="Input manifest JSONL")
    parser.add_argument("--output", required=True, help="Output interleaved JSONL")
    parser.add_argument("--work-dir", default="", help="Working directory for MFA artifacts")
    parser.add_argument("--run-mfa", action="store_true", help="Run MFA forced alignment")
    parser.add_argument("--mfa-model", default="english_mfa", help="MFA acoustic model name")
    parser.add_argument("--alignment-dir", default="", help="Pre-computed TextGrid directory")
    parser.add_argument("--chunk-words", type=int, default=6)
    parser.add_argument("--overlap-words", type=int, default=1)
    parser.add_argument("--min-chunk-duration-ms", type=float, default=200.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SONATA — INTERLEAVED MANIFEST PIPELINE")
    print(f"{'='*60}")

    alignment_dir = args.alignment_dir

    # Step 1: MFA alignment (optional)
    if args.run_mfa:
        work_dir = args.work_dir or tempfile.mkdtemp(prefix="sonata_mfa_")
        print(f"\n  Step 1: Forced alignment (MFA)")
        print(f"  Work dir: {work_dir}")
        alignment_dir = run_mfa_alignment(
            args.manifest, work_dir, args.mfa_model, args.sample_rate,
        )
        if not alignment_dir:
            print("  Falling back to uniform alignment")
    elif alignment_dir:
        print(f"\n  Step 1: Using pre-computed alignments from {alignment_dir}")
    else:
        print(f"\n  Step 1: Using uniform alignment (no MFA)")

    # Step 2: Interleaving
    print(f"\n  Step 2: Interleaving")
    cfg = FlowV3Config(sample_rate=args.sample_rate)
    interleaved_cfg = InterleavedTrainingConfig(
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        min_chunk_duration_ms=args.min_chunk_duration_ms,
    )

    stats = prepare_interleaved_manifest(
        args.manifest,
        args.output,
        cfg,
        interleaved_cfg,
        alignment_dir=alignment_dir or None,
        max_duration=args.max_duration,
    )

    print(f"    Processed: {stats['processed']}/{stats['total_entries']}")
    print(f"    Chunks: {stats['total_chunks']} ({stats['avg_chunks_per_utt']} per utterance)")
    print(f"    Frames: {stats['total_frames']} ({stats['avg_frames_per_chunk']} per chunk)")

    # Step 3: Validation
    print(f"\n  Step 3: Validation")
    issues = validate_output(args.output)
    print(f"    Total entries: {issues['total']}")
    print(f"    Valid: {issues['valid']}")
    if issues["empty_chunks"] > 0:
        print(f"    WARNING: {issues['empty_chunks']} entries with empty chunks")
    if issues["short_chunks"] > 0:
        print(f"    WARNING: {issues['short_chunks']} chunks shorter than minimum duration")

    if issues["valid"] == 0 and issues["total"] > 0:
        print(f"\n  ERROR: No valid entries produced. Check your manifest and audio paths.")
        sys.exit(1)

    print(f"\n  Output: {args.output}")
    print(f"  Done!")


if __name__ == "__main__":
    main()
