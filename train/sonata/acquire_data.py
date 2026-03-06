#!/usr/bin/env python3
"""acquire_data.py — Multi-dataset acquisition, preprocessing, and manifest generation.

Downloads, preprocesses, and unifies speech datasets into a common manifest format
for Sonata TTS training. Targets 10K–100K+ hours of diverse speech.

Supported datasets:
  ┌─────────────────┬────────┬──────────┬────────────────────────────────┐
  │ Dataset         │ Hours  │ Speakers │ Notes                          │
  ├─────────────────┼────────┼──────────┼────────────────────────────────┤
  │ LibriSpeech     │ 960    │ 2,484    │ Read English, 16kHz FLAC       │
  │ LibriTTS-R      │ 585    │ 2,456    │ Read English, 24kHz WAV        │
  │ LibriLight      │ 60,000 │ ~7,000   │ Unlabeled, needs ASR transcr.  │
  │ VCTK            │ 44     │ 110      │ Multi-accent English, 48kHz    │
  │ LJSpeech        │ 24     │ 1        │ Single speaker, 22kHz          │
  │ Common Voice 17 │ 2,000+ │ ~65,000  │ Community-recorded, MP3→WAV    │
  │ GigaSpeech      │ 10,000 │ varies   │ YouTube/podcast (needs license)│
  │ MLS             │ 50,000+│ varies   │ Multilingual read speech       │
  │ Hi-Fi TTS       │ 292    │ 10       │ Studio quality, 44.1kHz        │
  │ RAVDESS         │ 7      │ 24       │ Emotional speech, 48kHz        │
  │ EmoV-DB         │ 7      │ 5        │ Emotional speech w/ labels     │
  └─────────────────┴────────┴──────────┴────────────────────────────────┘

Manifest format (one JSON per line):
  {"audio": "/path/to/file.flac", "text": "transcript", "utt_id": "spk-chap-utt",
   "speaker": "1234", "duration": 5.2, "dataset": "librispeech"}

Usage:
  # Download and prepare LibriSpeech (960h)
  python acquire_data.py download --dataset librispeech --splits all --out-dir data/

  # Download LibriTTS-R (24kHz, ideal for Sonata's 24kHz sample rate)
  python acquire_data.py download --dataset libritts-r --splits all --out-dir data/

  # Download VCTK (multi-speaker diversity)
  python acquire_data.py download --dataset vctk --out-dir data/

  # Build unified manifest from everything downloaded
  python acquire_data.py manifest --data-dir data/ --output data/manifest_all.jsonl

  # Preprocess: resample to 24kHz, normalize, VAD trim, filter by quality
  python acquire_data.py preprocess --manifest data/manifest_all.jsonl \
    --output data/manifest_clean.jsonl --target-sr 24000

  # Show dataset statistics
  python acquire_data.py stats --manifest data/manifest_clean.jsonl
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Registry
# ═══════════════════════════════════════════════════════════════════════════════

DATASETS = {
    "librispeech": {
        "description": "LibriSpeech ASR corpus (960h read English, 16kHz FLAC)",
        "base_url": "https://openslr.trmal.net/resources/12",
        "splits": {
            "dev-clean": "dev-clean.tar.gz",
            "dev-other": "dev-other.tar.gz",
            "test-clean": "test-clean.tar.gz",
            "test-other": "test-other.tar.gz",
            "train-clean-100": "train-clean-100.tar.gz",
            "train-clean-360": "train-clean-360.tar.gz",
            "train-other-500": "train-other-500.tar.gz",
        },
        "all_train": ["train-clean-100", "train-clean-360", "train-other-500"],
        "format": "librispeech",
        "sample_rate": 16000,
    },
    "libritts-r": {
        "description": "LibriTTS-R (585h, 24kHz WAV, restored from LibriSpeech)",
        "base_url": "https://openslr.trmal.net/resources/141",
        "splits": {
            "dev-clean": "dev_clean.tar.gz",
            "dev-other": "dev_other.tar.gz",
            "test-clean": "test_clean.tar.gz",
            "test-other": "test_other.tar.gz",
            "train-clean-100": "train_clean_100.tar.gz",
            "train-clean-360": "train_clean_360.tar.gz",
            "train-other-500": "train_other_500.tar.gz",
        },
        "all_train": ["train-clean-100", "train-clean-360", "train-other-500"],
        "format": "libritts",
        "sample_rate": 24000,
    },
    "libritts": {
        "description": "LibriTTS (585h, 24kHz WAV, from LibriSpeech)",
        "base_url": "https://openslr.trmal.net/resources/60",
        "splits": {
            "dev-clean": "dev_clean.tar.gz",
            "dev-other": "dev_other.tar.gz",
            "test-clean": "test_clean.tar.gz",
            "test-other": "test_other.tar.gz",
            "train-clean-100": "train_clean_100.tar.gz",
            "train-clean-360": "train_clean_360.tar.gz",
            "train-other-500": "train_other_500.tar.gz",
        },
        "all_train": ["train-clean-100", "train-clean-360", "train-other-500"],
        "format": "libritts",
        "sample_rate": 24000,
    },
    "vctk": {
        "description": "VCTK corpus (44h, 110 speakers, multi-accent English, 48kHz)",
        "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip",
        "splits": {"all": "VCTK-Corpus-0.92.zip"},
        "format": "vctk",
        "sample_rate": 48000,
    },
    "ljspeech": {
        "description": "LJ Speech (24h, single speaker, 22kHz WAV)",
        "url": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "splits": {"all": "LJSpeech-1.1.tar.bz2"},
        "format": "ljspeech",
        "sample_rate": 22050,
    },
    "hifi-tts": {
        "description": "Hi-Fi Multi-Speaker TTS (292h, 10 speakers, 44.1kHz)",
        "base_url": "https://openslr.trmal.net/resources/109",
        "splits": {
            "all": "hi_fi_tts_v0.tar.gz",
        },
        "format": "hifi-tts",
        "sample_rate": 44100,
    },
    "common-voice": {
        "description": "Mozilla Common Voice (community-recorded, MP3)",
        "note": "Requires manual download from https://commonvoice.mozilla.org/",
        "format": "common-voice",
        "sample_rate": 48000,
    },
    "gigaspeech": {
        "description": "GigaSpeech (10Kh YouTube/podcast)",
        "note": "Requires HuggingFace access: https://huggingface.co/datasets/speechcolab/gigaspeech",
        "format": "gigaspeech",
    },
    "mls": {
        "description": "Multilingual LibriSpeech (50K+ hours)",
        "base_url": "https://dl.fbaipublicfiles.com/mls",
        "splits": {
            "english": "mls_english.tar.gz",
        },
        "format": "mls",
        "sample_rate": 16000,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Download
# ═══════════════════════════════════════════════════════════════════════════════

def _progress_hook(count, block_size, total_size):
    if total_size > 0:
        pct = count * block_size * 100 / total_size
        mb = count * block_size / 1e6
        total_mb = total_size / 1e6
        bar_len = 30
        filled = int(bar_len * min(pct, 100) / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  {bar} {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)", end="", flush=True)


def download_file(url: str, dest: Path, desc: str = ""):
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [skip] {dest.name} already exists ({dest.stat().st_size / 1e6:.0f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or url} ...")

    # Use curl for large downloads (much faster than urllib)
    try:
        result = subprocess.run(
            ["curl", "-L", "-C", "-", "--progress-bar", "-o", str(dest), url],
            check=True,
        )
        print(f"  [done] {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to urllib if curl unavailable
        try:
            urllib.request.urlretrieve(url, str(dest), reporthook=_progress_hook)
            print(f"\n  [done] {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        except Exception as e:
            print(f"\n  [error] Download failed: {e}")
            if dest.exists():
                dest.unlink()
            raise


def extract_archive(archive: Path, dest_dir: Path):
    print(f"  Extracting {archive.name} ...")
    if archive.name.endswith(".tar.gz") or archive.name.endswith(".tgz"):
        with tarfile.open(str(archive), "r:gz") as tar:
            tar.extractall(str(dest_dir))
    elif archive.name.endswith(".tar.bz2"):
        with tarfile.open(str(archive), "r:bz2") as tar:
            tar.extractall(str(dest_dir))
    elif archive.name.endswith(".zip"):
        with zipfile.ZipFile(str(archive), "r") as z:
            z.extractall(str(dest_dir))
    else:
        raise ValueError(f"Unknown archive format: {archive.name}")
    print(f"  [done] Extracted to {dest_dir}")


def cmd_download(args):
    ds_name = args.dataset.lower()
    if ds_name not in DATASETS:
        print(f"Unknown dataset: {ds_name}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return

    ds = DATASETS[ds_name]
    out_dir = Path(args.out_dir)
    archive_dir = out_dir / "_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    if "note" in ds and "base_url" not in ds and "url" not in ds:
        print(f"\n  {ds_name}: {ds['description']}")
        print(f"  {ds['note']}")
        return

    print(f"\n{'='*60}")
    print(f"  Downloading: {ds['description']}")
    print(f"{'='*60}\n")

    # Resolve splits
    if args.splits == "all":
        if "all_train" in ds:
            splits = ds["all_train"]
        else:
            splits = list(ds["splits"].keys())
    else:
        splits = [s.strip() for s in args.splits.split(",")]

    base_url = ds.get("base_url", "")
    for split in splits:
        if split not in ds["splits"]:
            print(f"  [warn] Unknown split: {split}")
            continue

        filename = ds["splits"][split]
        archive_path = archive_dir / filename

        # Build URL
        if base_url:
            url = f"{base_url}/{filename}"
        elif "url" in ds:
            url = ds["url"]
        else:
            print(f"  [skip] No URL for {split}")
            continue

        # Download
        try:
            download_file(url, archive_path, desc=f"{ds_name}/{split}")
        except Exception:
            continue

        # Extract
        extract_dir = out_dir / ds_name
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Check if already extracted
        marker = extract_dir / f".{split}_extracted"
        if marker.exists():
            print(f"  [skip] {split} already extracted")
            continue

        extract_archive(archive_path, extract_dir)
        marker.touch()

        # Optionally remove archive to save disk space
        if args.remove_archives:
            archive_path.unlink()
            print(f"  [cleanup] Removed {archive_path.name}")

    # Count files
    extract_dir = out_dir / ds_name
    if extract_dir.exists():
        n_audio = sum(1 for _ in extract_dir.rglob("*.flac")) + \
                  sum(1 for _ in extract_dir.rglob("*.wav"))
        print(f"\n  Total audio files: {n_audio}")


# ═══════════════════════════════════════════════════════════════════════════════
# Manifest Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _audio_duration(path: str) -> Optional[float]:
    """Get audio duration without loading the full file."""
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.duration
    except Exception:
        return None


def iter_librispeech_manifest(root: Path, dataset_name: str) -> Iterator[dict]:
    """LibriSpeech format: speaker/chapter/speaker-chapter-utt.flac + .trans.txt"""
    for trans_file in sorted(root.rglob("*.trans.txt")):
        transcripts = {}
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]

        for utt_id, text in transcripts.items():
            audio = trans_file.parent / f"{utt_id}.flac"
            if not audio.exists():
                continue
            speaker = utt_id.split("-")[0]
            dur = _audio_duration(str(audio))
            yield {
                "audio": str(audio.resolve()),
                "text": text.strip(),
                "utt_id": utt_id,
                "speaker": speaker,
                "duration": dur,
                "dataset": dataset_name,
            }


def iter_libritts_manifest(root: Path, dataset_name: str) -> Iterator[dict]:
    """LibriTTS format: speaker/chapter/speaker_chapter_utt.normalized.txt + .wav"""
    for wav in sorted(root.rglob("*.wav")):
        norm_txt = wav.with_suffix(".normalized.txt")
        if norm_txt.exists():
            text = norm_txt.read_text().strip()
        else:
            orig_txt = wav.with_suffix(".original.txt")
            text = orig_txt.read_text().strip() if orig_txt.exists() else ""
        if not text:
            continue
        utt_id = wav.stem
        speaker = utt_id.split("_")[0]
        dur = _audio_duration(str(wav))
        yield {
            "audio": str(wav.resolve()),
            "text": text,
            "utt_id": utt_id,
            "speaker": speaker,
            "duration": dur,
            "dataset": dataset_name,
        }


def iter_vctk_manifest(root: Path) -> Iterator[dict]:
    """VCTK: txt/pXXX/pXXX_YYY.txt + wav48_silence_trimmed/pXXX/pXXX_YYY_mic1.flac"""
    txt_dir = None
    for candidate in ["txt", "VCTK-Corpus-0.92/txt"]:
        if (root / candidate).exists():
            txt_dir = root / candidate
            break
    wav_dir = None
    for candidate in ["wav48_silence_trimmed", "VCTK-Corpus-0.92/wav48_silence_trimmed",
                       "wav48", "VCTK-Corpus-0.92/wav48"]:
        if (root / candidate).exists():
            wav_dir = root / candidate
            break

    if not txt_dir or not wav_dir:
        print(f"  [warn] VCTK structure not found in {root}")
        return

    for txt_file in sorted(txt_dir.rglob("*.txt")):
        if txt_file.name.startswith("."):
            continue
        text = txt_file.read_text().strip()
        if not text:
            continue
        speaker = txt_file.parent.name
        stem = txt_file.stem

        # Try multiple audio naming conventions
        for suffix in ["_mic1.flac", "_mic2.flac", ".wav", "_mic1.wav"]:
            audio = wav_dir / speaker / (stem + suffix)
            if audio.exists():
                dur = _audio_duration(str(audio))
                yield {
                    "audio": str(audio.resolve()),
                    "text": text,
                    "utt_id": stem,
                    "speaker": speaker,
                    "duration": dur,
                    "dataset": "vctk",
                }
                break


def iter_ljspeech_manifest(root: Path) -> Iterator[dict]:
    """LJSpeech: metadata.csv + wavs/LJXXX-YYYY.wav"""
    meta = None
    for candidate in ["metadata.csv", "LJSpeech-1.1/metadata.csv"]:
        if (root / candidate).exists():
            meta = root / candidate
            break
    if not meta:
        return

    wav_dir = meta.parent / "wavs"
    with open(meta) as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                utt_id = parts[0]
                text = parts[2]  # Normalized text
                audio = wav_dir / f"{utt_id}.wav"
                if audio.exists():
                    dur = _audio_duration(str(audio))
                    yield {
                        "audio": str(audio.resolve()),
                        "text": text,
                        "utt_id": utt_id,
                        "speaker": "lj",
                        "duration": dur,
                        "dataset": "ljspeech",
                    }


def iter_commonvoice_manifest(root: Path, splits: list) -> Iterator[dict]:
    """Common Voice TSV format."""
    clips_dir = root / "clips"
    if not clips_dir.exists():
        return

    for split in splits:
        tsv = root / f"{split}.tsv"
        if not tsv.exists():
            continue
        with open(tsv) as f:
            header = f.readline().strip().split("\t")
            try:
                path_i = header.index("path")
                text_i = header.index("sentence")
                client_i = header.index("client_id")
            except ValueError:
                continue
            for line in f:
                cols = line.strip().split("\t")
                if len(cols) <= max(path_i, text_i, client_i):
                    continue
                audio = clips_dir / cols[path_i]
                if not audio.exists():
                    # Try .wav version (if pre-converted)
                    audio = audio.with_suffix(".wav")
                    if not audio.exists():
                        continue
                dur = _audio_duration(str(audio))
                yield {
                    "audio": str(audio.resolve()),
                    "text": cols[text_i],
                    "utt_id": audio.stem,
                    "speaker": cols[client_i][:8],
                    "duration": dur,
                    "dataset": "common-voice",
                }


def iter_hifi_manifest(root: Path) -> Iterator[dict]:
    """Hi-Fi TTS: {clean,other}_XX/*.flac + manifest JSON."""
    for json_file in sorted(root.rglob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for utt_id, entry in data.items():
            if "text_normalized" not in entry:
                continue
            audio = root / entry.get("audio_filepath", "")
            if not audio.exists():
                continue
            yield {
                "audio": str(audio.resolve()),
                "text": entry["text_normalized"],
                "utt_id": utt_id,
                "speaker": str(entry.get("speaker", "")),
                "duration": entry.get("duration"),
                "dataset": "hifi-tts",
            }


def cmd_manifest(args):
    data_dir = Path(args.data_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Building unified manifest from {data_dir}")
    print(f"{'='*60}\n")

    count = 0
    total_dur = 0.0
    speakers = set()
    datasets_seen = set()

    seen_audio = set()

    def _write_entry(out, entry):
        nonlocal count, total_dur
        ap = entry.get("audio", "")
        if ap in seen_audio:
            return
        seen_audio.add(ap)
        out.write(json.dumps(entry) + "\n")
        count += 1
        total_dur += entry.get("duration") or 0
        if entry.get("speaker"):
            speakers.add(entry["speaker"])
        datasets_seen.add(entry.get("dataset", "unknown"))
        if count % 5000 == 0:
            print(f"    {count} entries, {total_dur/3600:.1f}h ...")

    with open(output, "w") as out:
        # Auto-detect datasets (case-insensitive directory matching)
        for ds_name in ["librispeech", "libritts-r", "libritts"]:
            ds_dir = data_dir / ds_name
            if not ds_dir.exists():
                # Try case-insensitive match
                for d in data_dir.iterdir():
                    if d.is_dir() and d.name.lower() == ds_name.lower():
                        ds_dir = d
                        break
            if not ds_dir.exists():
                continue
            fmt = DATASETS.get(ds_name, {}).get("format", "")
            if fmt == "librispeech":
                gen = iter_librispeech_manifest(ds_dir, ds_name)
            elif fmt == "libritts":
                gen = iter_libritts_manifest(ds_dir, ds_name)
            else:
                continue
            print(f"  Scanning {ds_name} ({ds_dir.name})...")
            for entry in gen:
                _write_entry(out, entry)

        # VCTK
        vctk_dir = data_dir / "vctk"
        if vctk_dir.exists():
            print(f"  Scanning vctk...")
            for entry in iter_vctk_manifest(vctk_dir):
                _write_entry(out, entry)

        # LJSpeech
        lj_dir = data_dir / "ljspeech"
        if lj_dir.exists():
            print(f"  Scanning ljspeech...")
            for entry in iter_ljspeech_manifest(lj_dir):
                _write_entry(out, entry)

        # Common Voice
        cv_dir = data_dir / "common-voice"
        if cv_dir.exists():
            print(f"  Scanning common-voice...")
            for entry in iter_commonvoice_manifest(cv_dir, ["validated", "train", "dev", "test"]):
                _write_entry(out, entry)

        # Hi-Fi TTS
        hifi_dir = data_dir / "hifi-tts"
        if hifi_dir.exists():
            print(f"  Scanning hifi-tts...")
            for entry in iter_hifi_manifest(hifi_dir):
                _write_entry(out, entry)

        # Custom JSONL files in data_dir root (skip known manifests)
        skip_prefixes = ("manifest_", "encoded_")
        for jsonl in sorted(data_dir.glob("*.jsonl")):
            if jsonl.name == output.name:
                continue
            if jsonl.name.startswith(skip_prefixes):
                continue
            print(f"  Including {jsonl.name}...")
            with open(jsonl) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if "audio" not in entry or "text" not in entry:
                            continue
                        audio_p = Path(entry["audio"])
                        if not audio_p.is_absolute():
                            audio_p = data_dir / audio_p
                        if not audio_p.exists():
                            continue
                        entry["audio"] = str(audio_p.resolve())
                        _write_entry(out, entry)
                    except json.JSONDecodeError:
                        pass

    print(f"\n  Manifest: {output}")
    print(f"  Utterances: {count:,}")
    print(f"  Duration:   {total_dur/3600:.1f} hours")
    print(f"  Speakers:   {len(speakers):,}")
    print(f"  Datasets:   {', '.join(sorted(datasets_seen))}")


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_preprocess(args):
    """Preprocess audio: resample, normalize, VAD trim, quality filter."""
    import numpy as np

    try:
        import soundfile as sf
    except ImportError:
        print("Install soundfile: pip install soundfile")
        return

    manifest = Path(args.manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    target_sr = args.target_sr

    resample_dir = None
    if args.resample_dir:
        resample_dir = Path(args.resample_dir)
        resample_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Preprocessing: {manifest}")
    print(f"  Target SR: {target_sr} Hz")
    print(f"  Duration:  {args.min_duration}s – {args.max_duration}s")
    print(f"  Min SNR:   {args.min_snr} dB")
    print(f"{'='*60}\n")

    total = 0
    kept = 0
    skipped_duration = 0
    skipped_snr = 0
    skipped_error = 0
    total_dur = 0.0
    t0 = time.time()

    with open(output, "w") as out:
        with open(manifest) as f:
            for line in f:
                total += 1
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    skipped_error += 1
                    continue

                audio_path = entry.get("audio", "")
                if not os.path.exists(audio_path):
                    skipped_error += 1
                    continue

                # Duration filter (fast, from metadata or file info)
                dur = entry.get("duration")
                if dur is None:
                    dur = _audio_duration(audio_path)
                if dur is None:
                    skipped_error += 1
                    continue

                if dur < args.min_duration or dur > args.max_duration:
                    skipped_duration += 1
                    continue

                # Load and check audio quality
                try:
                    audio, sr = sf.read(audio_path, dtype="float32")
                except Exception:
                    skipped_error += 1
                    continue

                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)

                # SNR estimation (simple energy-based)
                if args.min_snr > 0:
                    rms = np.sqrt(np.mean(audio ** 2))
                    if rms < 1e-6:
                        skipped_snr += 1
                        continue
                    # Estimate noise from quietest 10% of frames
                    frame_size = int(sr * 0.02)
                    n_frames = len(audio) // frame_size
                    if n_frames > 10:
                        frame_energies = np.array([
                            np.sqrt(np.mean(audio[i*frame_size:(i+1)*frame_size] ** 2))
                            for i in range(n_frames)
                        ])
                        noise_floor = np.percentile(frame_energies, 10)
                        if noise_floor > 0:
                            snr_db = 20 * np.log10(rms / noise_floor)
                            if snr_db < args.min_snr:
                                skipped_snr += 1
                                continue

                # Resample if needed
                final_path = audio_path
                if sr != target_sr and resample_dir:
                    out_name = Path(audio_path).stem + ".wav"
                    ds = entry.get("dataset", "unknown")
                    spk = entry.get("speaker", "unk")
                    resamp_path = resample_dir / ds / spk / out_name
                    resamp_path.parent.mkdir(parents=True, exist_ok=True)

                    if not resamp_path.exists():
                        # Resample with scipy
                        try:
                            from scipy.signal import resample_poly
                            gcd = math.gcd(sr, target_sr)
                            up, down = target_sr // gcd, sr // gcd
                            resampled = resample_poly(audio, up, down).astype(np.float32)
                            sf.write(str(resamp_path), resampled, target_sr)
                        except ImportError:
                            # Fallback: linear interpolation
                            ratio = target_sr / sr
                            new_len = int(len(audio) * ratio)
                            indices = np.linspace(0, len(audio) - 1, new_len)
                            resampled = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
                            sf.write(str(resamp_path), resampled, target_sr)

                    final_path = str(resamp_path)
                    dur = dur  # Duration stays the same

                entry["audio"] = final_path
                entry["duration"] = round(dur, 3)
                out.write(json.dumps(entry) + "\n")
                kept += 1
                total_dur += dur

                if kept % 2000 == 0:
                    elapsed = time.time() - t0
                    rate = total / elapsed
                    print(f"  {kept:,} kept / {total:,} total, "
                          f"{total_dur/3600:.1f}h, {rate:.0f} utt/s")

    elapsed = time.time() - t0
    print(f"\n  Results:")
    print(f"    Total:            {total:,}")
    print(f"    Kept:             {kept:,} ({100*kept/max(total,1):.1f}%)")
    print(f"    Duration filtered: {skipped_duration:,}")
    print(f"    SNR filtered:     {skipped_snr:,}")
    print(f"    Errors:           {skipped_error:,}")
    print(f"    Audio:            {total_dur/3600:.1f} hours")
    print(f"    Time:             {elapsed:.0f}s")
    print(f"    Output:           {output}")


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_stats(args):
    """Show detailed statistics for a manifest."""
    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"Manifest not found: {manifest}")
        return

    total = 0
    total_dur = 0.0
    speakers = {}
    datasets = {}
    durations = []

    with open(manifest) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            dur = entry.get("duration", 0)
            total_dur += dur
            durations.append(dur)

            spk = entry.get("speaker", "unknown")
            speakers[spk] = speakers.get(spk, 0) + 1

            ds = entry.get("dataset", "unknown")
            if ds not in datasets:
                datasets[ds] = {"count": 0, "duration": 0}
            datasets[ds]["count"] += 1
            datasets[ds]["duration"] += dur

    if total == 0:
        print("  Empty manifest")
        return

    durations.sort()

    print(f"\n{'='*60}")
    print(f"  Dataset Statistics: {manifest.name}")
    print(f"{'='*60}")
    print(f"\n  Overall:")
    print(f"    Utterances:  {total:,}")
    print(f"    Duration:    {total_dur/3600:.1f} hours ({total_dur:.0f}s)")
    print(f"    Speakers:    {len(speakers):,}")
    print(f"    Avg duration: {total_dur/total:.1f}s")
    print(f"    Median dur:  {durations[len(durations)//2]:.1f}s")
    print(f"    P5/P95 dur:  {durations[int(len(durations)*0.05)]:.1f}s / "
          f"{durations[int(len(durations)*0.95)]:.1f}s")

    print(f"\n  Per dataset:")
    print(f"    {'Dataset':<20} {'Count':>8} {'Hours':>8} {'Avg(s)':>8}")
    print(f"    {'─'*20} {'─'*8} {'─'*8} {'─'*8}")
    for ds_name in sorted(datasets.keys()):
        ds = datasets[ds_name]
        avg = ds["duration"] / max(ds["count"], 1)
        print(f"    {ds_name:<20} {ds['count']:>8,} {ds['duration']/3600:>8.1f} {avg:>8.1f}")

    print(f"\n  Speaker distribution:")
    spk_counts = sorted(speakers.values(), reverse=True)
    print(f"    Top-1 speaker:   {spk_counts[0]:,} utterances")
    if len(spk_counts) > 10:
        print(f"    Top-10 speakers: {sum(spk_counts[:10]):,} utterances")
    print(f"    Median per spk:  {spk_counts[len(spk_counts)//2]:,} utterances")
    if len(spk_counts) > 1:
        bottom = spk_counts[-1]
        print(f"    Min per speaker: {bottom} utterances")


# ═══════════════════════════════════════════════════════════════════════════════
# Transcribe (for unlabeled audio like LibriLight, podcasts, YouTube)
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_transcribe(args):
    """Transcribe unlabeled audio with Whisper and produce a manifest."""
    try:
        import whisper
    except ImportError:
        print("Install openai-whisper: pip install openai-whisper")
        return

    import numpy as np

    manifest_in = Path(args.manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Transcribing with Whisper {args.model}")
    print(f"  Input:  {manifest_in}")
    print(f"  Output: {output}")
    print(f"{'='*60}\n")

    model = whisper.load_model(args.model, device=args.device)
    print(f"  Model loaded on {args.device}")

    total = 0
    transcribed = 0
    skipped = 0
    t0 = time.time()

    with open(output, "w") as out:
        with open(manifest_in) as f:
            for line in f:
                total += 1
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                audio_path = entry.get("audio", "")
                if not os.path.exists(audio_path):
                    skipped += 1
                    continue

                # Skip if already has a transcript
                if entry.get("text", "").strip() and not args.force:
                    out.write(json.dumps(entry) + "\n")
                    transcribed += 1
                    continue

                try:
                    result = model.transcribe(
                        audio_path,
                        language=args.language,
                        fp16=(args.device != "cpu"),
                        condition_on_previous_text=False,
                    )
                    text = result["text"].strip()
                    if not text:
                        skipped += 1
                        continue

                    entry["text"] = text
                    entry["whisper_lang"] = result.get("language", args.language)
                    # Compute average log probability as a quality signal
                    if result.get("segments"):
                        avg_logprob = np.mean([s["avg_logprob"] for s in result["segments"]])
                        entry["whisper_logprob"] = round(float(avg_logprob), 3)
                        no_speech_prob = np.mean([s["no_speech_prob"] for s in result["segments"]])
                        entry["no_speech_prob"] = round(float(no_speech_prob), 3)

                        if no_speech_prob > args.max_no_speech:
                            skipped += 1
                            continue
                        if avg_logprob < args.min_logprob:
                            skipped += 1
                            continue

                    out.write(json.dumps(entry) + "\n")
                    transcribed += 1

                except Exception as e:
                    skipped += 1
                    if skipped <= 10:
                        print(f"  [error] {audio_path}: {e}")
                    continue

                if transcribed % 200 == 0:
                    elapsed = time.time() - t0
                    rate = total / elapsed
                    print(f"  {transcribed:,} transcribed / {total:,} total, "
                          f"{skipped} skipped, {rate:.1f} utt/s")

    elapsed = time.time() - t0
    print(f"\n  Results:")
    print(f"    Total:       {total:,}")
    print(f"    Transcribed: {transcribed:,}")
    print(f"    Skipped:     {skipped:,}")
    print(f"    Time:        {elapsed:.0f}s ({total/max(elapsed,1):.1f} utt/s)")
    print(f"    Output:      {output}")


# ═══════════════════════════════════════════════════════════════════════════════
# HuggingFace Hub download
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_hf_download(args):
    """Download a dataset from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        return

    out_dir = Path(args.out_dir) / args.repo_id.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Downloading from HuggingFace: {args.repo_id}")
    print(f"  Output: {out_dir}")
    if args.allow_patterns:
        print(f"  Patterns: {args.allow_patterns}")
    print(f"{'='*60}\n")

    allow_patterns = None
    if args.allow_patterns:
        allow_patterns = [p.strip() for p in args.allow_patterns.split(",")]

    try:
        path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(out_dir),
            allow_patterns=allow_patterns,
            token=args.token or os.environ.get("HF_TOKEN"),
        )
        print(f"\n  Downloaded to: {path}")
    except Exception as e:
        print(f"\n  [error] Download failed: {e}")
        print(f"  You may need to accept the dataset license at:")
        print(f"  https://huggingface.co/datasets/{args.repo_id}")


# ═══════════════════════════════════════════════════════════════════════════════
# Segment long audio into utterances via VAD
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_segment(args):
    """Segment long audio files into utterance-level chunks using energy VAD."""
    import numpy as np

    try:
        import soundfile as sf
    except ImportError:
        print("Install soundfile: pip install soundfile")
        return

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Segmenting audio from {audio_dir}")
    print(f"  Target: {args.min_seg}s – {args.max_seg}s segments")
    print(f"{'='*60}\n")

    audio_files = sorted(
        list(audio_dir.rglob("*.wav")) +
        list(audio_dir.rglob("*.flac")) +
        list(audio_dir.rglob("*.mp3")) +
        list(audio_dir.rglob("*.ogg"))
    )
    print(f"  Found {len(audio_files)} audio files")

    total_segs = 0
    total_dur = 0.0
    t0 = time.time()

    with open(manifest_path, "w") as mf:
        for fi, audio_file in enumerate(audio_files):
            try:
                audio, sr = sf.read(str(audio_file), dtype="float32")
            except Exception as e:
                print(f"  [skip] {audio_file.name}: {e}")
                continue

            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Energy-based VAD segmentation
            frame_ms = 20
            frame_size = int(sr * frame_ms / 1000)
            n_frames = len(audio) // frame_size
            if n_frames < 5:
                continue

            energies = np.array([
                np.sqrt(np.mean(audio[i*frame_size:(i+1)*frame_size] ** 2))
                for i in range(n_frames)
            ])

            # Adaptive threshold
            threshold = np.percentile(energies, 30) * 2.0
            is_speech = energies > threshold

            # Find speech segments via run-length grouping
            segments = []
            in_seg = False
            seg_start = 0
            silence_frames = 0
            min_silence = int(300 / frame_ms)  # 300ms silence to split
            min_seg_frames = int(args.min_seg * 1000 / frame_ms)
            max_seg_frames = int(args.max_seg * 1000 / frame_ms)

            for i in range(n_frames):
                if is_speech[i]:
                    if not in_seg:
                        seg_start = max(0, i - 2)
                        in_seg = True
                    silence_frames = 0
                else:
                    if in_seg:
                        silence_frames += 1
                        if silence_frames >= min_silence:
                            seg_end = i - silence_frames + 2
                            seg_len = seg_end - seg_start
                            if seg_len >= min_seg_frames:
                                segments.append((seg_start, min(seg_end, n_frames)))
                            in_seg = False
                            silence_frames = 0

                # Force split at max duration
                if in_seg and (i - seg_start) >= max_seg_frames:
                    segments.append((seg_start, i))
                    in_seg = False
                    silence_frames = 0

            if in_seg:
                seg_len = n_frames - seg_start
                if seg_len >= min_seg_frames:
                    segments.append((seg_start, n_frames))

            # Write segments
            stem = audio_file.stem
            for si, (start_f, end_f) in enumerate(segments):
                start_sample = start_f * frame_size
                end_sample = min(end_f * frame_size, len(audio))
                segment_audio = audio[start_sample:end_sample]
                dur = len(segment_audio) / sr

                seg_name = f"{stem}_{si:04d}.wav"
                seg_path = out_dir / seg_name
                sf.write(str(seg_path), segment_audio, sr)

                entry = {
                    "audio": str(seg_path),
                    "text": "",
                    "utt_id": f"{stem}_{si:04d}",
                    "speaker": stem,
                    "duration": round(dur, 3),
                    "source_file": str(audio_file),
                    "dataset": "segmented",
                }
                mf.write(json.dumps(entry) + "\n")
                total_segs += 1
                total_dur += dur

            if (fi + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  {fi+1}/{len(audio_files)} files, "
                      f"{total_segs} segments, {total_dur/3600:.1f}h")

    elapsed = time.time() - t0
    print(f"\n  Results:")
    print(f"    Files processed: {len(audio_files)}")
    print(f"    Segments:        {total_segs:,}")
    print(f"    Duration:        {total_dur/3600:.1f} hours")
    print(f"    Time:            {elapsed:.0f}s")
    print(f"    Manifest:        {manifest_path}")
    print(f"\n  Next: transcribe with Whisper:")
    print(f"    python acquire_data.py transcribe "
          f"--manifest {manifest_path} --output {manifest_path.with_suffix('.transcribed.jsonl')}")


# ═══════════════════════════════════════════════════════════════════════════════
# List available datasets
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_list(args):
    print(f"\n  Available datasets:")
    print(f"  {'─'*60}")
    for name, ds in DATASETS.items():
        auto = "auto" if ("base_url" in ds or "url" in ds) else "manual"
        sr = ds.get("sample_rate", "?")
        print(f"  {name:<18} {ds['description'][:45]:<45} {sr}Hz  [{auto}]")
    print()
    print(f"  Auto-download:    python acquire_data.py download --dataset <name> --splits all")
    print(f"  HuggingFace:      python acquire_data.py hf-download --repo-id <org/name>")
    print(f"  Segment audio:    python acquire_data.py segment --audio-dir <dir>")
    print(f"  Transcribe:       python acquire_data.py transcribe --manifest <jsonl>")
    print(f"  Build manifest:   python acquire_data.py manifest --data-dir <dir>")
    print(f"  Filter/resample:  python acquire_data.py preprocess --manifest <jsonl>")
    print(f"  Show stats:       python acquire_data.py stats --manifest <jsonl>")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-dataset acquisition and preprocessing for Sonata TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # download
    dl = sub.add_parser("download", help="Download a speech dataset")
    dl.add_argument("--dataset", required=True, help="Dataset name (see 'list')")
    dl.add_argument("--splits", default="all", help="Comma-separated splits or 'all'")
    dl.add_argument("--out-dir", default="train/data", help="Output directory")
    dl.add_argument("--remove-archives", action="store_true",
                    help="Remove .tar.gz after extraction")

    # hf-download
    hf = sub.add_parser("hf-download", help="Download from HuggingFace Hub")
    hf.add_argument("--repo-id", required=True, help="HF dataset repo (e.g. speechcolab/gigaspeech)")
    hf.add_argument("--out-dir", default="train/data", help="Output directory")
    hf.add_argument("--allow-patterns", default=None,
                    help="Comma-separated file patterns (e.g. '*.flac,*.json')")
    hf.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env)")

    # manifest
    mf = sub.add_parser("manifest", help="Build unified JSONL manifest")
    mf.add_argument("--data-dir", required=True, help="Root dir with downloaded datasets")
    mf.add_argument("--output", required=True, help="Output JSONL path")

    # preprocess
    pp = sub.add_parser("preprocess", help="Resample, normalize, filter manifest")
    pp.add_argument("--manifest", required=True, help="Input manifest JSONL")
    pp.add_argument("--output", required=True, help="Output filtered manifest JSONL")
    pp.add_argument("--target-sr", type=int, default=24000, help="Target sample rate")
    pp.add_argument("--resample-dir", default=None,
                    help="Dir to write resampled WAVs (None = skip resampling)")
    pp.add_argument("--min-duration", type=float, default=0.5, help="Min utterance seconds")
    pp.add_argument("--max-duration", type=float, default=30.0, help="Max utterance seconds")
    pp.add_argument("--min-snr", type=float, default=10.0, help="Min SNR in dB (0=skip)")

    # segment
    sg = sub.add_parser("segment", help="Segment long audio into utterances via VAD")
    sg.add_argument("--audio-dir", required=True, help="Dir with long audio files")
    sg.add_argument("--out-dir", required=True, help="Dir for output segments")
    sg.add_argument("--output", required=True, help="Output manifest JSONL")
    sg.add_argument("--min-seg", type=float, default=1.0, help="Min segment seconds")
    sg.add_argument("--max-seg", type=float, default=20.0, help="Max segment seconds")

    # transcribe
    tr = sub.add_parser("transcribe", help="Transcribe unlabeled audio with Whisper")
    tr.add_argument("--manifest", required=True, help="Input manifest JSONL")
    tr.add_argument("--output", required=True, help="Output manifest with transcripts")
    tr.add_argument("--model", default="large-v3", help="Whisper model size")
    tr.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    tr.add_argument("--language", default="en", help="Language code")
    tr.add_argument("--force", action="store_true", help="Re-transcribe even if text exists")
    tr.add_argument("--min-logprob", type=float, default=-1.0,
                    help="Min avg log probability to keep")
    tr.add_argument("--max-no-speech", type=float, default=0.6,
                    help="Max no_speech_prob to keep")

    # stats
    st = sub.add_parser("stats", help="Show manifest statistics")
    st.add_argument("--manifest", required=True, help="Manifest JSONL")

    # list
    sub.add_parser("list", help="List available datasets")

    args = parser.parse_args()
    if args.command == "download":
        cmd_download(args)
    elif args.command == "hf-download":
        cmd_hf_download(args)
    elif args.command == "manifest":
        cmd_manifest(args)
    elif args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "segment":
        cmd_segment(args)
    elif args.command == "transcribe":
        cmd_transcribe(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
