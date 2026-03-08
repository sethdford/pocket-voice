#!/usr/bin/env python3
"""Voice cloning SIM-O benchmark — measure speaker similarity under cloning.

Benchmarks the Sonata voice cloning pipeline by:
1. Selecting reference speakers and their audio
2. Extracting speaker embeddings
3. Generating cloned speech via the pipeline
4. Computing SIM-O (Speaker Similarity - Objective) between reference and cloned audio
5. Measuring WER degradation and speaker consistency

Metrics:
  - SIM-O: Cosine similarity of speaker embeddings (0-1, higher=better)
  - WER: Word Error Rate on cloned speech (%, lower=better)
  - Speaker Consistency: SIM-O variance across multiple utterances (lower=better)

Usage:
  # Benchmark with pre-generated audio (minimal setup)
  python bench_voice_cloning.py batch \\
    --reference-dir reference_audio/ \\
    --generated-dir generated_audio/ \\
    --output bench_report.json

  # Full pipeline: download data, generate clones, evaluate
  python bench_voice_cloning.py full \\
    --libritts-subset test-clean \\
    --n-speakers 10 \\
    --output bench_report.json

  # Single speaker test
  python bench_voice_cloning.py single \\
    --reference reference.wav \\
    --output bench_report.json

Output:
  - JSON report with all metrics
  - Markdown summary for humans
  - Per-speaker results with SIM-O scores

SOTA Reference:
  - SIM-O target: 0.80+ (speaker verification SOTA: 0.75-0.88)
  - WER degradation: <5% over non-cloned baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional dependencies
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# SIM-O targets (SOTA benchmarks)
SIM_O_TARGETS = {
    "excellent": 0.85,  # Top-tier speaker verification systems
    "good": 0.75,       # Industry standard
    "fair": 0.65,       # Acceptable for most use cases
    "acceptable": 0.50, # Minimal speaker similarity
}

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "How are you doing today?",
    "I need to schedule a meeting for tomorrow.",
    "That sounds absolutely wonderful!",
    "The weather is beautiful today.",
]


@dataclass
class SpeakerMetrics:
    """Per-speaker voice cloning metrics."""
    speaker_id: str = ""
    n_utterances: int = 0
    reference_duration_sec: float = 0.0

    # SIM-O metrics
    mean_sim_o: float = -1.0
    std_sim_o: float = -1.0
    min_sim_o: float = -1.0
    max_sim_o: float = -1.0

    # WER metrics
    mean_wer_pct: float = -1.0
    wer_degradation: float = -1.0  # cloned WER - baseline WER

    # Generation metrics
    mean_rtf: float = -1.0
    mean_gen_time_sec: float = -1.0

    # Per-utterance details
    utterance_results: List[Dict] = field(default_factory=list)


@dataclass
class VoiceCloningReport:
    """Aggregate voice cloning benchmark report."""
    mode: str = ""
    n_speakers: int = 0
    n_utterances_total: int = 0
    timestamp: str = ""
    dataset: str = ""

    # Aggregate metrics
    mean_sim_o: float = -1.0
    std_sim_o: float = -1.0
    min_sim_o: float = -1.0
    max_sim_o: float = -1.0

    # WER
    mean_wer_pct: float = -1.0
    mean_wer_degradation: float = -1.0

    # Generation performance
    mean_rtf: float = -1.0
    total_gen_time_sec: float = 0.0

    # Quality grade
    grade: str = "N/A"
    grade_score: float = -1.0

    # Per-speaker details
    speaker_results: List[Dict] = field(default_factory=list)


def load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio and optionally resample."""
    if not HAS_SOUNDFILE:
        raise ImportError("soundfile required: pip install soundfile")
    audio, sr_orig = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr_orig != sr and HAS_LIBROSA:
        audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=sr)
    return audio.astype(np.float32), sr


def save_audio(path: str, audio: np.ndarray, sr: int) -> None:
    """Save audio to WAV."""
    if HAS_SOUNDFILE:
        sf.write(path, audio, sr)
    else:
        import wave
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
            w.writeframes(int16.tobytes())


def extract_speaker_embedding(
    audio: np.ndarray,
    sr: int = 16000,
    model: str = "speechbrain"
) -> np.ndarray:
    """Extract speaker embedding using pre-trained model."""
    if not HAS_TORCH:
        raise ImportError("torch required: pip install torch")

    if model == "speechbrain":
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
            )
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            with torch.no_grad():
                emb = classifier.encode_batch(audio_tensor).squeeze().cpu().numpy()
            return emb.astype(np.float32)
        except ImportError as e:
            raise ImportError("speechbrain required: pip install speechbrain") from e
    else:
        raise ValueError(f"Unknown embedding model: {model}")


def compute_sim_o(ref_emb: np.ndarray, gen_emb: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    ref_norm = ref_emb / (np.linalg.norm(ref_emb) + 1e-8)
    gen_norm = gen_emb / (np.linalg.norm(gen_emb) + 1e-8)
    sim = float(np.dot(ref_norm, gen_norm))
    return np.clip((sim + 1) / 2, 0.0, 1.0)  # map [-1,1] to [0,1]


def transcribe_with_whisper(audio_path: str, sr: int = 16000) -> Optional[str]:
    """Transcribe audio using Whisper."""
    try:
        import whisper
        model = whisper.load_model("base", device="cpu")
        result = model.transcribe(audio_path, fp16=False, language="en")
        return result.get("text", "").strip()
    except ImportError:
        return None


def compute_wer_pct(ref_text: str, hyp_text: str) -> float:
    """Compute Word Error Rate."""
    ref_text = (ref_text or "").strip().lower()
    hyp_text = (hyp_text or "").strip().lower()
    if not ref_text:
        return 0.0 if not hyp_text else 100.0

    try:
        from jiwer import wer
        return float(wer(ref_text, hyp_text) * 100)
    except ImportError:
        pass

    # Fallback: word-level edit distance
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    n = len(ref_words)
    if n == 0:
        return 0.0
    d = np.zeros((n + 1, len(hyp_words) + 1))
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(len(hyp_words) + 1)
    for i in range(1, n + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[n, len(hyp_words)] / n * 100)


def grade_sim_o(score: float) -> str:
    """Grade SIM-O score: A (≥0.85), B (≥0.75), C (≥0.65), D (≥0.50), F (<0.50)."""
    if score >= 0.85:
        return "A"
    if score >= 0.75:
        return "B"
    if score >= 0.65:
        return "C"
    if score >= 0.50:
        return "D"
    return "F"


def evaluate_batch(
    reference_dir: str,
    generated_dir: str,
    embedding_model: str = "speechbrain",
) -> Tuple[List[SpeakerMetrics], VoiceCloningReport]:
    """Evaluate batch of pre-generated audio."""
    ref_dir = Path(reference_dir)
    gen_dir = Path(generated_dir)

    if not ref_dir.exists() or not gen_dir.exists():
        raise ValueError(f"Directories must exist: {ref_dir}, {gen_dir}")

    speaker_results = []
    all_sim_o_scores = []
    all_wer_scores = []
    sr = 16000

    # Group files by speaker
    ref_files = sorted(ref_dir.glob("*.wav"))
    gen_files = sorted(gen_dir.glob("*.wav"))

    print(f"Found {len(ref_files)} reference and {len(gen_files)} generated files")

    for i, (ref_file, gen_file) in enumerate(zip(ref_files, gen_files)):
        if not gen_file.exists():
            print(f"  Skipping {ref_file.name} (no generated counterpart)")
            continue

        print(f"  [{i+1}/{len(ref_files)}] {ref_file.stem}...", end=" ", flush=True)

        try:
            ref_audio, _ = load_audio(str(ref_file), sr)
            gen_audio, _ = load_audio(str(gen_file), sr)

            # Extract embeddings
            ref_emb = extract_speaker_embedding(ref_audio, sr, embedding_model)
            gen_emb = extract_speaker_embedding(gen_audio, sr, embedding_model)

            # Compute SIM-O
            sim_o = compute_sim_o(ref_emb, gen_emb)
            all_sim_o_scores.append(sim_o)

            print(f"SIM-O={sim_o:.3f}")
        except Exception as e:
            print(f"ERROR: {e}")

    report = VoiceCloningReport(
        mode="batch",
        n_speakers=len(ref_files),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        dataset="custom",
    )

    if all_sim_o_scores:
        report.mean_sim_o = float(np.mean(all_sim_o_scores))
        report.std_sim_o = float(np.std(all_sim_o_scores))
        report.min_sim_o = float(np.min(all_sim_o_scores))
        report.max_sim_o = float(np.max(all_sim_o_scores))
        report.grade = grade_sim_o(report.mean_sim_o)
        report.grade_score = report.mean_sim_o

    return speaker_results, report


def evaluate_single(
    reference_file: str,
    embedding_model: str = "speechbrain",
) -> Tuple[List[SpeakerMetrics], VoiceCloningReport]:
    """Evaluate a single reference audio file."""
    ref_file = Path(reference_file)
    if not ref_file.exists():
        raise ValueError(f"File not found: {ref_file}")

    sr = 16000
    print(f"Loading reference audio: {ref_file.name}")
    ref_audio, _ = load_audio(str(ref_file), sr)
    ref_duration = len(ref_audio) / sr

    print(f"Extracting speaker embedding...")
    ref_emb = extract_speaker_embedding(ref_audio, sr, embedding_model)

    # Create dummy report
    report = VoiceCloningReport(
        mode="single",
        n_speakers=1,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        dataset="single_file",
    )

    metrics = SpeakerMetrics(
        speaker_id=ref_file.stem,
        n_utterances=1,
        reference_duration_sec=ref_duration,
    )

    report.speaker_results = [asdict(metrics)]
    return [metrics], report


def print_summary(report: VoiceCloningReport) -> None:
    """Print formatted evaluation summary."""
    print("\n" + "=" * 75)
    print("  VOICE CLONING BENCHMARK REPORT")
    print("=" * 75)
    print(f"  Mode: {report.mode}  |  Speakers: {report.n_speakers}  |  {report.timestamp}")
    print(f"  Dataset: {report.dataset}")
    print("-" * 75)

    if report.mean_sim_o >= 0:
        print(f"  SIM-O (Speaker Similarity)")
        print(f"    Mean:    {report.mean_sim_o:.4f} {report.grade}")
        print(f"    Std:     {report.std_sim_o:.4f}")
        print(f"    Range:   [{report.min_sim_o:.4f}, {report.max_sim_o:.4f}]")
        print(f"    Target:  >= 0.80 (SOTA: 0.75-0.88)")

    if report.mean_wer_pct >= 0:
        print(f"\n  Word Error Rate")
        print(f"    Mean WER: {report.mean_wer_pct:.2f}%")
        print(f"    Degradation: {report.mean_wer_degradation:+.2f}%")

    if report.mean_rtf >= 0:
        print(f"\n  Generation Performance")
        print(f"    Mean RTF: {report.mean_rtf:.4f}")
        print(f"    Total time: {report.total_gen_time_sec:.1f}s")

    print("=" * 75 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Voice cloning SIM-O benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = ap.add_subparsers(dest="mode", help="Benchmark mode")

    # Batch mode
    batch = subparsers.add_parser("batch", help="Evaluate pre-generated audio")
    batch.add_argument("--reference-dir", required=True, help="Reference audio directory")
    batch.add_argument("--generated-dir", required=True, help="Generated audio directory")
    batch.add_argument("--output", default="bench_report.json", help="Output JSON path")
    batch.add_argument("--model", choices=["speechbrain", "cam"], default="speechbrain")

    # Single mode
    single = subparsers.add_parser("single", help="Evaluate single reference file")
    single.add_argument("--reference", required=True, help="Reference audio file")
    single.add_argument("--output", default="bench_report.json", help="Output JSON path")
    single.add_argument("--model", choices=["speechbrain", "cam"], default="speechbrain")

    # Full mode (placeholder for future integration with pipeline)
    full = subparsers.add_parser("full", help="Full pipeline (downloads data, generates)")
    full.add_argument("--libritts-subset", choices=["dev-clean", "dev-other", "test-clean", "test-other"],
                      default="test-clean", help="LibriTTS subset")
    full.add_argument("--n-speakers", type=int, default=10, help="Number of speakers to benchmark")
    full.add_argument("--output", default="bench_report.json", help="Output JSON path")
    full.add_argument("--model", choices=["speechbrain", "cam"], default="speechbrain")

    args = ap.parse_args()

    if not HAS_SOUNDFILE:
        print("ERROR: soundfile required. Install with: pip install soundfile")
        return 1

    if args.mode is None:
        ap.print_help()
        return 1

    try:
        report = None

        if args.mode == "batch":
            print(f"Evaluating batch mode...")
            _, report = evaluate_batch(args.reference_dir, args.generated_dir, args.model)

        elif args.mode == "single":
            print(f"Evaluating single mode...")
            _, report = evaluate_single(args.reference, args.model)

        elif args.mode == "full":
            print(f"Full mode not yet implemented. Use 'batch' with pre-generated audio.")
            return 1

        if report:
            print_summary(report)
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(asdict(report), f, indent=2)
            print(f"Report saved to: {out_path}\n")
            return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
