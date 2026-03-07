#!/usr/bin/env python3
"""Eval wrapper: Merge C pipeline metrics with comprehensive Python evaluation.

Reads WAVs from eval/generated/ and C report from eval/reports/c_eval_report.json,
then computes reference-based metrics (PESQ, STOI, MCD, F0, speaker sim) and
text-based metrics (WER via Whisper).

Merges C-native metrics (RTF, TTFA, prosody MOS, Conformer WER) with Python
evaluation results into a comprehensive EvalReport, adds SOTA comparison table,
and saves to eval/reports/eval_report.json.

Usage:
  python eval/run_eval.py \\
    --wav-dir eval/generated \\
    --c-report eval/reports/c_eval_report.json \\
    --ref-dir eval/reference \\
    --utmos \\
    --output eval/reports/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Add train/sonata to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "train" / "sonata"))

try:
    from eval_comprehensive import (
        EvalMetrics,
        EvalReport,
        aggregate_report,
        compute_mos_proxy,
        compute_pesq,
        compute_speaker_similarity,
        compute_stoi,
        compute_utmos,
        compute_wer_pct,
        compute_f0_metrics,
        compute_mcd,
        compute_spectral_convergence,
        load_audio,
        print_summary,
        transcribe_whisper,
        TARGETS,
    )
except ImportError as e:
    print(f"ERROR: Failed to import from eval_comprehensive: {e}", file=sys.stderr)
    sys.exit(1)


# SOTA comparison baselines (published results)
SOTA_BASELINES = {
    "rtf": {
        "sonata": None,  # Will be filled from our metrics
        "f5_tts_7step": 0.03,
    },
    "wer_pct": {
        "sonata": None,
        "f5_tts": 0.9,
    },
    "mos_proxy": {
        "sonata": None,
        "f5_tts": 4.3,
    },
    "speaker_sim": {
        "sonata": None,
        "maskgct": 0.687,
    },
    "ttfa_ms": {
        "sonata": None,
        "voxtream": 102,
    },
}


def load_c_report(report_path: str) -> Optional[dict]:
    """Load C evaluation report JSON. Returns None if not found."""
    p = Path(report_path)
    if not p.exists():
        print(f"WARNING: C report not found at {report_path}. Continuing without C metrics.")
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        print(f"WARNING: Failed to load C report: {e}. Continuing without C metrics.")
        return None


def find_ref_audio(gen_wav_path: Path, ref_dir: Path) -> Optional[Path]:
    """Find matching reference audio for a generated WAV."""
    cand = ref_dir / gen_wav_path.name
    if cand.exists():
        return cand
    refs = list(ref_dir.glob("*.wav"))
    if refs:
        idx = int(gen_wav_path.stem.replace("eval_", ""))
        if idx < len(refs):
            return refs[idx]
    return None


def evaluate_generated(
    gen_path: Path,
    ref_dir: Optional[Path] = None,
    ref_text: str = "",
    use_utmos: bool = False,
) -> EvalMetrics:
    """Evaluate a single generated WAV file."""
    gen_audio, sr_gen = load_audio(str(gen_path))
    duration = len(gen_audio) / sr_gen

    m = EvalMetrics(
        file=gen_path.name,
        text=ref_text,
        duration_sec=duration,
    )

    # Reference-based metrics (if reference audio available)
    if ref_dir:
        ref_path = find_ref_audio(gen_path, ref_dir)
        if ref_path and ref_path.exists():
            try:
                ref_audio, sr_ref = load_audio(str(ref_path))
                sr = max(sr_gen, sr_ref)

                # Resample if needed
                if sr_gen != sr:
                    from eval_comprehensive import resample_audio
                    gen_audio = resample_audio(gen_audio, sr_gen, sr)
                if sr_ref != sr:
                    from eval_comprehensive import resample_audio
                    ref_audio = resample_audio(ref_audio, sr_ref, sr)

                # Match length
                min_len = min(len(gen_audio), len(ref_audio))
                gen_audio = gen_audio[:min_len]
                ref_audio = ref_audio[:min_len]

                # Compute metrics
                m.pesq = compute_pesq(ref_audio, gen_audio, sr)
                m.stoi = compute_stoi(ref_audio, gen_audio, sr)
                m.mcd_db = compute_mcd(ref_audio, gen_audio, sr)
                m.spectral_convergence = compute_spectral_convergence(ref_audio, gen_audio)
                m.f0_rmse, m.f0_corr = compute_f0_metrics(ref_audio, gen_audio, sr)
                m.speaker_sim = compute_speaker_similarity(ref_audio, gen_audio, sr)
                m.mos_proxy = compute_mos_proxy(m.pesq, m.stoi, m.mcd_db, m.f0_corr)
                sr_gen = sr  # Use resampled sr for UTMOS
            except Exception as e:
                print(f"  WARNING: Failed to compute reference-based metrics for {gen_path.name}: {e}")

    # UTMOS (reference-free neural MOS)
    if use_utmos:
        m.utmos = compute_utmos(gen_audio, sr_gen)

    # WER (text-based) via Whisper
    if ref_text:
        try:
            import tempfile
            from eval_comprehensive import save_audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            try:
                save_audio(tmp_path, gen_audio, sr_gen)
                hyp = transcribe_whisper(tmp_path, sr_gen)
                if hyp is not None:
                    m.wer_pct = compute_wer_pct(ref_text, hyp)
            finally:
                import os
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception as e:
            print(f"  WARNING: Failed to compute WER for {gen_path.name}: {e}")
            m.wer_pct = -1.0
    else:
        m.wer_pct = -1.0

    return m


def merge_c_metrics(py_report: EvalReport, c_report: dict) -> EvalReport:
    """Merge C-native metrics into Python report."""
    # Extract C metrics if available
    c_metrics = c_report.get("metrics", {})

    # Add C metrics to the report as a sub-object
    py_report.c_metrics = c_metrics

    # If C report has RTF or TTFA, add to SOTA baselines
    if "mean_rtf_c" in c_metrics:
        SOTA_BASELINES["rtf"]["sonata"] = c_metrics["mean_rtf_c"]
    if "mean_ttfa_ms" in c_metrics:
        SOTA_BASELINES["ttfa_ms"]["sonata"] = c_metrics["mean_ttfa_ms"]

    return py_report


def print_sota_comparison(report: EvalReport) -> None:
    """Print SOTA comparison table."""
    print("\n" + "=" * 75)
    print("  SOTA COMPARISON")
    print("=" * 75)

    rows = [
        ("RTF", "Lower is better", "rtf"),
        ("WER (%)", "Lower is better", "wer_pct"),
        ("MOS Proxy", "Higher is better", "mos_proxy"),
        ("Speaker Similarity", "Higher is better", "speaker_sim"),
        ("TTFA (ms)", "Lower is better", "ttfa_ms"),
    ]

    for metric_name, direction, key in rows:
        baselines = SOTA_BASELINES.get(key, {})
        if not baselines:
            continue

        sonata_val = baselines.get("sonata")
        if sonata_val is None:
            continue

        print(f"\n  {metric_name} — {direction}")
        print("  " + "-" * 71)

        for method, val in baselines.items():
            if val is None:
                continue
            marker = " ← OUR RESULT" if method == "sonata" else ""
            print(f"    {method:<20} {val:>10.3f}{marker}")

    print("\n" + "=" * 75 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Eval wrapper: merge C pipeline metrics with Python evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--wav-dir",
        default="eval/generated",
        help="Directory with generated WAVs (named eval_NNNN.wav)",
    )
    ap.add_argument(
        "--c-report",
        default="eval/reports/c_eval_report.json",
        help="Path to C pipeline evaluation report JSON",
    )
    ap.add_argument(
        "--ref-dir",
        default="",
        help="Directory with reference audio (optional, enables reference-based metrics)",
    )
    ap.add_argument(
        "--utmos",
        action="store_true",
        help="Enable UTMOS neural MOS metric",
    )
    ap.add_argument(
        "--output",
        default="eval/reports/eval_report.json",
        help="Output path for merged evaluation report",
    )

    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    ref_dir = Path(args.ref_dir) if args.ref_dir else None
    output_path = Path(args.output)

    # Find generated WAVs
    wavs = sorted(wav_dir.glob("eval_*.wav"))
    if not wavs:
        print(f"ERROR: No WAVs found in {wav_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(wavs)} generated WAV files")

    # Load C report (if available)
    c_report = load_c_report(args.c_report)
    c_texts = []
    if c_report:
        c_texts = [r.get("text", "") for r in c_report.get("results", [])]
        print(f"Loaded C report with {len(c_texts)} samples")

    # Evaluate each WAV
    print(f"\nEvaluating generated audio (ref_dir={ref_dir}, utmos={args.utmos})...")
    results = []
    for i, wav_path in enumerate(wavs):
        ref_text = c_texts[i] if i < len(c_texts) else ""
        print(f"  [{i+1}/{len(wavs)}] {wav_path.name}...", end=" ", flush=True)

        m = evaluate_generated(
            wav_path,
            ref_dir=ref_dir,
            ref_text=ref_text,
            use_utmos=args.utmos,
        )
        results.append(m)
        print("OK")

    # Build aggregate report
    print(f"\nAggregating metrics for {len(results)} samples...")
    report = aggregate_report(results, mode="c_pipeline")

    # Merge C metrics if available
    if c_report:
        report = merge_c_metrics(report, c_report)

    # Fill SOTA baselines with our metrics
    SOTA_BASELINES["rtf"]["sonata"] = report.mean_rtf
    SOTA_BASELINES["wer_pct"]["sonata"] = report.mean_wer_pct
    SOTA_BASELINES["mos_proxy"]["sonata"] = report.mean_mos_proxy
    SOTA_BASELINES["speaker_sim"]["sonata"] = report.mean_speaker_sim

    # Print summary and SOTA comparison
    print_summary(report)
    print_sota_comparison(report)

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        report_dict = asdict(report)
        report_dict["sota_baselines"] = SOTA_BASELINES
        json.dump(report_dict, f, indent=2)

    print(f"Report saved to: {output_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
