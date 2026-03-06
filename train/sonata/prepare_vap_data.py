#!/usr/bin/env python3
"""
Prepare Voice Activity and Prediction (VAP) training data from conversational corpora.

Converts Fisher, Switchboard, CallHome, and synthetic speech datasets to a unified
VAP manifest format (jsonl). Each line contains stereo audio metadata + speaker
annotations with backchannel detection.

Output manifest (vap_manifest.jsonl):
  {
    "audio_path": "path/to/stereo.wav",
    "channels": 2,
    "duration": 300.5,
    "source": "fisher",
    "annotations": [
      {"start": 0.0, "end": 2.5, "speaker": "A", "type": "speech"},
      {"start": 1.8, "end": 2.0, "speaker": "B", "type": "backchannel"},
      ...
    ]
  }

Speaker A = channel 0 (user), Speaker B = channel 1 (system).
Type is "speech" or "backchannel" (short overlapping utterances).
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

# Try to import soundfile for SPH support; fall back to sox if unavailable
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    sf = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Backchannel keyword patterns (lowercase)
BC_PATTERNS = {
    "uh-huh", "uh huh", "yeah", "yep", "mhm", "mm-hm", "mm",
    "hmm", "right", "okay", "ok", "sure", "ah", "oh",
    "i see", "really", "wow", "huh", "mmm", "uh",
}

MAX_BACKCHANNEL_DURATION = 1.5  # seconds


@dataclass
class Annotation:
    """Single speaker turn annotation."""
    start: float
    end: float
    speaker: str  # "A" or "B"
    type: str    # "speech" or "backchannel"

    def to_dict(self):
        return asdict(self)


@dataclass
class VAPEntry:
    """Single entry in VAP manifest."""
    audio_path: str
    channels: int
    duration: float
    source: str
    annotations: List[Dict]

    def to_dict(self):
        return asdict(self)


def is_backchannel(text: str, duration: float, threshold: float = MAX_BACKCHANNEL_DURATION) -> bool:
    """
    Detect if an utterance is a backchannel.

    A backchannel is:
    - Short duration (< threshold)
    - Matches known backchannel keywords or very short phrases

    Args:
        text: Utterance text (normalized)
        duration: Duration in seconds
        threshold: Maximum duration for backchannel

    Returns:
        True if likely a backchannel
    """
    if duration > threshold:
        return False

    # Normalize text
    normalized = text.lower().strip()

    # Empty or very short utterances are backchannels
    if len(normalized) == 0 or len(normalized) < 3:
        return True

    # Check against known patterns
    if normalized in BC_PATTERNS:
        return True

    # Check if text is primarily one of the patterns (with minor variations)
    words = normalized.split()
    if len(words) == 1 and words[0] in BC_PATTERNS:
        return True

    return False


def sph_to_wav(sph_path: Path, wav_path: Path, target_sr: int = 16000) -> bool:
    """
    Convert NIST Sphere (.sph) to WAV using soundfile or sox.

    Args:
        sph_path: Path to .sph file
        wav_path: Output WAV path
        target_sr: Target sample rate (default 16kHz)

    Returns:
        True if successful, False otherwise
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Try soundfile first (requires libsndfile with SPH support)
        if HAS_SOUNDFILE and sf:
            try:
                data, sr = sf.read(str(sph_path))
                # Resample if needed (simple nearest-neighbor)
                if sr != target_sr:
                    ratio = target_sr / sr
                    n_samples = int(len(data) * ratio)
                    indices = (
                        (
                            range(n_samples) / ratio
                        ).astype(int)
                    )
                    data = data[indices]
                sf.write(str(wav_path), data, target_sr)
                return True
            except Exception as e:
                logger.warning(f"soundfile failed for {sph_path}: {e}. Trying sox...")

        # Fall back to sox
        cmd = [
            "sox",
            str(sph_path),
            "-r", str(target_sr),
            "-b", "16",
            "-c", "2",
            str(wav_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True
        else:
            logger.error(
                f"sox conversion failed for {sph_path}: {result.stderr}"
            )
            return False

    except Exception as e:
        logger.error(f"Failed to convert {sph_path}: {e}")
        return False


def get_audio_duration(wav_path: Path) -> Optional[float]:
    """Get duration of WAV file in seconds."""
    try:
        if HAS_SOUNDFILE and sf:
            info = sf.info(str(wav_path))
            return info.duration
        else:
            # Use ffprobe as fallback
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1:noescapetext=1",
                str(wav_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not get duration for {wav_path}: {e}")
    return None


def parse_fisher_transcripts(
    transcript_file: Path,
) -> List[Tuple[float, float, str, str]]:
    """
    Parse Fisher transcript file.

    Format: tab-separated
      start_time end_time speaker transcript

    Returns:
        List of (start, end, speaker, text) tuples
    """
    turns = []
    try:
        with open(transcript_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 4:
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                        speaker = parts[2].strip()
                        text = parts[3].strip()
                        turns.append((start, end, speaker, text))
                    except ValueError:
                        logger.debug(f"Skipping malformed line in {transcript_file}: {line}")
    except Exception as e:
        logger.error(f"Error parsing {transcript_file}: {e}")
    return turns


def parse_switchboard_transcripts(
    transcript_file: Path,
) -> List[Tuple[float, float, str, str]]:
    """
    Parse Switchboard transcript file (MS-State format).

    Format: comma-separated or similar, with speaker labels and times

    Fallback parser for common Switchboard formats.

    Returns:
        List of (start, end, speaker, text) tuples
    """
    turns = []
    try:
        with open(transcript_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Try to parse common Switchboard format
                # Pattern: [start-end] speaker: text
                match = re.match(
                    r'\[(\d+\.\d+)-(\d+\.\d+)\]\s+([AB]|[12]):\s*(.*)',
                    line
                )
                if match:
                    start = float(match.group(1))
                    end = float(match.group(2))
                    speaker = match.group(3)
                    text = match.group(4).strip()
                    turns.append((start, end, speaker, text))
    except Exception as e:
        logger.error(f"Error parsing {transcript_file}: {e}")
    return turns


def parse_callhome_cha(chat_file: Path) -> List[Tuple[float, float, str, str]]:
    """
    Parse CallHome CHAT format file.

    Format (CHILDES/CHAT):
      *A: utterance text
      %tim: start_end

    Or inline timestamps:
      *A: text with \\x15start_end\\x15 embedded

    Returns:
        List of (start, end, speaker, text) tuples
    """
    turns = []
    try:
        with open(chat_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Speaker turn line: *A: or *B:
            if line.startswith("*A:") or line.startswith("*B:"):
                speaker = line[1]  # A or B
                text = line[4:].strip()  # Skip "*X: "

                # Remove CHAT markup
                text = re.sub(r'[[\]<>@].*?[[\]<>@]', '', text)
                text = re.sub(r'\\x[0-9a-f]{2}', '', text, flags=re.IGNORECASE)

                # Try to extract inline timestamp
                start = None
                end = None
                timestamp_match = re.search(r'\\x15(\d+\.?\d*)_(\d+\.?\d*)\x15', text)
                if timestamp_match:
                    start = float(timestamp_match.group(1))
                    end = float(timestamp_match.group(2))
                    text = re.sub(r'\x15.*?\x15', '', text)

                # Check next line for %tim tier
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith("%tim:"):
                        tim_match = re.search(r'(\d+\.?\d*)_(\d+\.?\d*)', next_line)
                        if tim_match:
                            start = float(tim_match.group(1))
                            end = float(tim_match.group(2))
                        i += 1  # Skip timing line

                text = text.strip()
                if start is not None and end is not None and text:
                    turns.append((start, end, speaker, text))

            i += 1

    except Exception as e:
        logger.error(f"Error parsing {chat_file}: {e}")

    return turns


def detect_backchannels_in_annotations(
    annotations: List[Annotation],
    transcript_data: Dict[str, str],
) -> List[Annotation]:
    """
    Re-label short overlapping utterances as backchannels.

    Args:
        annotations: List of annotations with speaker + timing
        transcript_data: Dict mapping (speaker, start, end) → text

    Returns:
        Updated list with backchannel type labels
    """
    result = []

    for ann in annotations:
        key = (ann.speaker, ann.start, ann.end)
        text = transcript_data.get(key, "")
        duration = ann.end - ann.start

        # Check if this is a backchannel
        if is_backchannel(text, duration):
            # Verify it overlaps with other speaker's speech
            overlaps_with_other = False
            for other in annotations:
                if other.speaker != ann.speaker:
                    # Check overlap: [start1, end1] ∩ [start2, end2]
                    overlap_start = max(ann.start, other.start)
                    overlap_end = min(ann.end, other.end)
                    if overlap_start < overlap_end:
                        overlaps_with_other = True
                        break

            if overlaps_with_other:
                ann.type = "backchannel"

        result.append(ann)

    return result


def build_annotations_from_turns(
    turns: List[Tuple[float, float, str, str]],
) -> Tuple[List[Annotation], Dict[str, str]]:
    """
    Build annotation list and transcript data from parsed turns.

    Args:
        turns: List of (start, end, speaker, text) tuples

    Returns:
        (annotations list, transcript data dict)
    """
    annotations = []
    transcript_data = {}

    for start, end, speaker, text in turns:
        # Normalize speaker to A or B
        speaker_norm = "A" if speaker in ("A", "1") else "B" if speaker in ("B", "2") else speaker

        ann = Annotation(start=start, end=end, speaker=speaker_norm, type="speech")
        annotations.append(ann)
        transcript_data[(speaker_norm, start, end)] = text

    return annotations, transcript_data


def convert_fisher(
    data_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    append: bool = False,
) -> int:
    """
    Convert Fisher English corpus to VAP manifest.

    Expected structure:
      fe_03_p1_sph/
        fe_03_01_sph/  (or similar)
          *.sph
      fe_03_p1_tran/
        *.txt

    Returns:
        Number of conversations processed
    """
    logger.info(f"Converting Fisher corpus from {data_dir}")

    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.error(f"Fisher data directory not found: {data_dir}")
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find SPH directories
    sph_dirs = list(data_dir.glob("**/[fe]_*_sph"))
    if not sph_dirs:
        # Fallback: look for directories with .sph files
        sph_dirs = [d for d in data_dir.glob("*") if d.is_dir()]
        sph_dirs = [d for d in sph_dirs if list(d.glob("*.sph"))]

    entries = []
    count = 0
    total_duration = 0.0

    mode = "a" if append else "w"
    with open(manifest_path, mode) as mf:
        for sph_dir in sorted(sph_dirs):
            logger.info(f"Processing {sph_dir.name}")

            for sph_file in sorted(sph_dir.glob("*.sph")):
                basename = sph_file.stem

                # Find corresponding transcript file
                tran_file = None
                for tran_dir in data_dir.glob("**/[fe]_*_tran"):
                    candidate = tran_dir / f"{basename}.txt"
                    if candidate.exists():
                        tran_file = candidate
                        break

                if not tran_file or not tran_file.exists():
                    logger.warning(
                        f"No transcript found for {sph_file.name}, skipping"
                    )
                    continue

                # Convert SPH to WAV
                wav_file = output_dir / f"{basename}.wav"
                if not sph_to_wav(sph_file, wav_file):
                    logger.warning(f"Failed to convert {sph_file.name}, skipping")
                    continue

                # Parse transcript
                turns = parse_fisher_transcripts(tran_file)
                if not turns:
                    logger.warning(f"No turns parsed from {tran_file.name}")
                    continue

                # Build annotations
                annotations, trans_data = build_annotations_from_turns(turns)
                annotations = detect_backchannels_in_annotations(annotations, trans_data)

                # Get audio duration
                duration = get_audio_duration(wav_file)
                if duration is None:
                    logger.warning(f"Could not determine duration for {wav_file.name}")
                    continue

                # Create entry
                entry = VAPEntry(
                    audio_path=str(wav_file.relative_to(output_dir.parent)),
                    channels=2,
                    duration=duration,
                    source="fisher",
                    annotations=[a.to_dict() for a in annotations],
                )

                mf.write(json.dumps(entry.to_dict()) + "\n")
                count += 1
                total_duration += duration

                if count % 10 == 0:
                    logger.info(
                        f"Processed {count} conversations, "
                        f"total duration: {total_duration/3600:.1f}h"
                    )

    logger.info(
        f"Fisher conversion complete: {count} conversations, "
        f"total duration: {total_duration/3600:.1f}h"
    )
    return count


def convert_switchboard(
    data_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    append: bool = False,
) -> int:
    """
    Convert Switchboard corpus to VAP manifest.

    Expected structure:
      sw*/
        *.sph
      sw*.trans (transcripts)

    Returns:
        Number of conversations processed
    """
    logger.info(f"Converting Switchboard corpus from {data_dir}")

    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.error(f"Switchboard data directory not found: {data_dir}")
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sph_files = sorted(data_dir.glob("*.sph")) + sorted(data_dir.glob("*/*.sph"))
    if not sph_files:
        logger.error(f"No .sph files found in {data_dir}")
        return 0

    entries = []
    count = 0
    total_duration = 0.0

    mode = "a" if append else "w"
    with open(manifest_path, mode) as mf:
        for sph_file in sph_files:
            basename = sph_file.stem

            # Find transcript file
            tran_file = data_dir / f"{basename}.trans"
            if not tran_file.exists():
                tran_file = sph_file.parent / f"{basename}.trans"

            if not tran_file.exists():
                logger.warning(
                    f"No transcript found for {sph_file.name}, skipping"
                )
                continue

            # Convert SPH to WAV
            wav_file = output_dir / f"{basename}.wav"
            if not sph_to_wav(sph_file, wav_file):
                logger.warning(f"Failed to convert {sph_file.name}, skipping")
                continue

            # Parse transcript
            turns = parse_switchboard_transcripts(tran_file)
            if not turns:
                logger.warning(f"No turns parsed from {tran_file.name}")
                continue

            # Build annotations
            annotations, trans_data = build_annotations_from_turns(turns)
            annotations = detect_backchannels_in_annotations(annotations, trans_data)

            # Get audio duration
            duration = get_audio_duration(wav_file)
            if duration is None:
                logger.warning(f"Could not determine duration for {wav_file.name}")
                continue

            # Create entry
            entry = VAPEntry(
                audio_path=str(wav_file.relative_to(output_dir.parent)),
                channels=2,
                duration=duration,
                source="switchboard",
                annotations=[a.to_dict() for a in annotations],
            )

            mf.write(json.dumps(entry.to_dict()) + "\n")
            count += 1
            total_duration += duration

            if count % 10 == 0:
                logger.info(
                    f"Processed {count} conversations, "
                    f"total duration: {total_duration/3600:.1f}h"
                )

    logger.info(
        f"Switchboard conversion complete: {count} conversations, "
        f"total duration: {total_duration/3600:.1f}h"
    )
    return count


def convert_callhome(
    data_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    append: bool = False,
) -> int:
    """
    Convert CallHome English corpus to VAP manifest.

    Expected structure:
      *.sph (audio)
      *.cha (transcripts in CHAT format)

    Returns:
        Number of conversations processed
    """
    logger.info(f"Converting CallHome corpus from {data_dir}")

    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.error(f"CallHome data directory not found: {data_dir}")
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sph_files = sorted(data_dir.glob("*.sph")) + sorted(data_dir.glob("*/*.sph"))
    if not sph_files:
        logger.error(f"No .sph files found in {data_dir}")
        return 0

    count = 0
    total_duration = 0.0

    mode = "a" if append else "w"
    with open(manifest_path, mode) as mf:
        for sph_file in sph_files:
            basename = sph_file.stem

            # Find transcript file (CHAT format)
            cha_file = sph_file.parent / f"{basename}.cha"
            if not cha_file.exists():
                cha_file = data_dir / f"{basename}.cha"

            if not cha_file.exists():
                logger.warning(
                    f"No transcript found for {sph_file.name}, skipping"
                )
                continue

            # Convert SPH to WAV
            wav_file = output_dir / f"{basename}.wav"
            if not sph_to_wav(sph_file, wav_file):
                logger.warning(f"Failed to convert {sph_file.name}, skipping")
                continue

            # Parse transcript
            turns = parse_callhome_cha(cha_file)
            if not turns:
                logger.warning(f"No turns parsed from {cha_file.name}")
                continue

            # Build annotations
            annotations, trans_data = build_annotations_from_turns(turns)
            annotations = detect_backchannels_in_annotations(annotations, trans_data)

            # Get audio duration
            duration = get_audio_duration(wav_file)
            if duration is None:
                logger.warning(f"Could not determine duration for {wav_file.name}")
                continue

            # Create entry
            entry = VAPEntry(
                audio_path=str(wav_file.relative_to(output_dir.parent)),
                channels=2,
                duration=duration,
                source="callhome",
                annotations=[a.to_dict() for a in annotations],
            )

            mf.write(json.dumps(entry.to_dict()) + "\n")
            count += 1
            total_duration += duration

            if count % 10 == 0:
                logger.info(
                    f"Processed {count} conversations, "
                    f"total duration: {total_duration/3600:.1f}h"
                )

    logger.info(
        f"CallHome conversion complete: {count} conversations, "
        f"total duration: {total_duration/3600:.1f}h"
    )
    return count


def generate_synthetic(
    output_dir: Path,
    manifest_path: Path,
    n_conversations: int = 100,
    append: bool = False,
) -> int:
    """
    Generate synthetic test conversations.

    Creates stereo WAV files with alternating speaker noise bursts
    and realistic turn-taking patterns.

    Returns:
        Number of synthetic conversations created
    """
    try:
        import numpy as np
    except ImportError:
        logger.error("numpy required for synthetic data generation")
        return 0

    logger.info(f"Generating {n_conversations} synthetic conversations")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sr = 16000
    count = 0
    total_duration = 0.0

    mode = "a" if append else "w"
    with open(manifest_path, mode) as mf:
        for idx in range(n_conversations):
            # Random conversation length: 2-10 minutes
            duration = np.random.uniform(120, 600)
            n_samples = int(duration * sr)

            # Create stereo audio with noise bursts
            stereo = np.zeros((n_samples, 2), dtype=np.float32)

            # Generate turn-taking pattern
            current_speaker = np.random.choice([0, 1])
            t = 0
            annotations = []

            while t < n_samples:
                # Turn duration: 1-10 seconds
                turn_duration = np.random.uniform(1, 10)
                turn_samples = min(int(turn_duration * sr), n_samples - t)

                if turn_samples < sr * 0.5:  # Skip very short turns
                    break

                # Add white noise burst on current speaker's channel
                noise = np.random.normal(0, 0.1, turn_samples).astype(np.float32)
                stereo[t:t + turn_samples, current_speaker] += noise

                # Sometimes add backchannel on other speaker (20% chance)
                if np.random.random() < 0.2:
                    bc_duration = np.random.uniform(0.3, 1.0)
                    bc_samples = int(bc_duration * sr)
                    if t + turn_samples + bc_samples < n_samples:
                        other_speaker = 1 - current_speaker
                        bc_noise = np.random.normal(0, 0.05, bc_samples).astype(np.float32)
                        stereo[t + turn_samples:t + turn_samples + bc_samples, other_speaker] += bc_noise

                        # Add backchannel annotation
                        start_time = (t + turn_samples) / sr
                        end_time = (t + turn_samples + bc_samples) / sr
                        annotations.append(
                            Annotation(
                                start=start_time,
                                end=end_time,
                                speaker="B" if other_speaker == 1 else "A",
                                type="backchannel",
                            )
                        )

                # Add main turn annotation
                start_time = t / sr
                end_time = (t + turn_samples) / sr
                annotations.append(
                    Annotation(
                        start=start_time,
                        end=end_time,
                        speaker="B" if current_speaker == 1 else "A",
                        type="speech",
                    )
                )

                t += turn_samples

                # Silence between turns: 0.2-1 second
                silence_duration = np.random.uniform(0.2, 1.0)
                silence_samples = int(silence_duration * sr)
                t += silence_samples

                # Switch speaker
                current_speaker = 1 - current_speaker

            # Write synthetic audio
            wav_file = output_dir / f"synthetic_{idx:04d}.wav"
            try:
                if HAS_SOUNDFILE and sf:
                    sf.write(str(wav_file), stereo, sr)
                else:
                    logger.warning("soundfile not available, skipping synthetic WAV generation")
                    continue

                actual_duration = len(stereo) / sr

                # Create entry
                entry = VAPEntry(
                    audio_path=str(wav_file.relative_to(output_dir.parent)),
                    channels=2,
                    duration=actual_duration,
                    source="synthetic",
                    annotations=[a.to_dict() for a in annotations],
                )

                mf.write(json.dumps(entry.to_dict()) + "\n")
                count += 1
                total_duration += actual_duration

            except Exception as e:
                logger.error(f"Failed to create synthetic conversation {idx}: {e}")
                continue

            if count % 20 == 0:
                logger.info(
                    f"Generated {count} conversations, "
                    f"total duration: {total_duration/3600:.1f}h"
                )

    logger.info(
        f"Synthetic data generation complete: {count} conversations, "
        f"total duration: {total_duration/3600:.1f}h"
    )
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Prepare VAP (Voice Activity and Prediction) training data from conversational corpora.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Fisher English
  %(prog)s --source fisher --data-dir /path/to/fisher --output-dir train/data/vap --manifest train/data/vap_manifest.jsonl

  # Append Switchboard
  %(prog)s --source switchboard --data-dir /path/to/switchboard --output-dir train/data/vap --manifest train/data/vap_manifest.jsonl --append

  # Generate synthetic test data
  %(prog)s --source synthetic --output-dir train/data/vap --manifest train/data/vap_manifest.jsonl --n-synthetic 100
        """,
    )

    parser.add_argument(
        "--source",
        choices=["fisher", "switchboard", "callhome", "synthetic"],
        required=True,
        help="Corpus source",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Input data directory (not required for synthetic)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for converted WAV files",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Output manifest path (jsonl)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing manifest instead of overwriting",
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=100,
        help="Number of synthetic conversations to generate (default: 100)",
    )

    args = parser.parse_args()

    # Validate inputs
    if args.source != "synthetic" and not args.data_dir:
        parser.error(f"--data-dir required for {args.source}")

    # Run conversion
    try:
        if args.source == "fisher":
            count = convert_fisher(args.data_dir, args.output_dir, args.manifest, args.append)
        elif args.source == "switchboard":
            count = convert_switchboard(args.data_dir, args.output_dir, args.manifest, args.append)
        elif args.source == "callhome":
            count = convert_callhome(args.data_dir, args.output_dir, args.manifest, args.append)
        elif args.source == "synthetic":
            count = generate_synthetic(args.output_dir, args.manifest, args.n_synthetic, args.append)

        if count > 0:
            logger.info(f"✓ Successfully processed {count} conversations")
            logger.info(f"✓ Manifest written to {args.manifest}")
        else:
            logger.error(f"✗ No conversations processed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
