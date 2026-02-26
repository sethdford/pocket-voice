#!/usr/bin/env python3
"""
bench_emergent_eval.py — EmergentTTS-Eval benchmark harness for pocket-voice.

Runs representative test sentences from each EmergentTTS-Eval category
through the on-device TTS engine and measures quality metrics.

Categories (from EmergentTTS-Eval):
  1. Expressiveness — emotion, emphasis, tone variation
  2. Complex Pronunciation — numbers, dates, currencies, formulas
  3. Foreign Words / Names — non-English proper nouns
  4. Paralinguistic Cues — pauses, laughter, hesitation
  5. Discourse Structure — lists, instructions, multi-sentence
  6. Edge Cases — very short/long, all caps, mixed scripts

Usage:
    python benchmarks/bench_emergent_eval.py --model models/kyutai_dsm.ctts
    python benchmarks/bench_emergent_eval.py --model models/kyutai_dsm.ctts --output bench_output/emergent.json
"""

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000

EVAL_CATEGORIES = {
    "expressiveness": [
        "I'm SO excited to tell you this — you won the grand prize!",
        "I'm deeply sorry for your loss. Please know that I'm here for you.",
        "Wait, WHAT?! That can't be right. Are you absolutely sure about that?",
        "Hmm, let me think about that for a moment... Actually, yes, I believe so.",
        "Oh come on, that's the funniest thing I've heard all week!",
        "This is critically important and requires your immediate attention.",
        "Take a deep breath. Everything is going to be just fine.",
        "You know what? That is an EXCELLENT point. I hadn't considered that angle.",
    ],
    "complex_pronunciation": [
        "The package weighs 2.5kg and measures 30cm by 20cm by 15cm.",
        "Please transfer $1,234.56 to account #4829-0071 by 12/31/2025.",
        "The temperature dropped from 72°F to -3°C overnight.",
        "Version 3.14.159 includes fixes for CVE-2025-1234 and RFC 7231.",
        "Call 1-800-555-0199 or email support@company.co.uk for assistance.",
        "The GDP grew by 3.2% in Q4, reaching $21.7 trillion.",
        "Mix 1/4 cup flour with 2/3 cup milk and 1.5 tsp vanilla extract.",
        "Dr. Smith, Ph.D., presented at the IEEE conference on Nov. 15th, 2025.",
    ],
    "foreign_names": [
        "The restaurant on Rue de Rivoli serves excellent coq au vin.",
        "Please welcome our guest speaker, Dr. Müller from Zürich.",
        "The Shinkansen from Tokyo to Osaka takes about 2 hours and 30 minutes.",
        "We visited the Louvre, Notre-Dame, and the Champs-Élysées in one day.",
        "CEO Satoshi Nakamura announced the partnership with Deutsche Telekom.",
        "The São Paulo office coordinates with the team in Buenos Aires.",
    ],
    "paralinguistic": [
        "Well... I mean... it's complicated, you know?",
        "Ha! That's rich. No, seriously though, we need to talk about this.",
        "Ugh, another Monday. But hey, at least the coffee is good, right?",
        "Shh, listen carefully... do you hear that? Nothing. Perfect silence.",
        "OK so first — and this is key — you need to preheat the oven to 350.",
        "No. No no no. Absolutely not. That is NOT what we agreed on.",
    ],
    "discourse_structure": [
        "There are three steps: first, open the app; second, tap Settings; third, toggle Dark Mode.",
        "On one hand, the price is competitive. On the other hand, the reviews are mixed. Overall, I'd recommend waiting.",
        "To summarize: the project is on track, the budget is under control, and the team morale is high.",
        "Here's what happened: I arrived at 9, the meeting started at 9:15, and by 10 we had a decision.",
        "Question: What's the capital of France? Answer: Paris. Follow-up: What's its population? About 2.1 million.",
    ],
    "edge_cases": [
        "OK.",
        "Yes? No? Maybe?",
        "ATTENTION: THIS IS AN EMERGENCY BROADCAST. SEEK SHELTER IMMEDIATELY.",
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",
        "https://en.wikipedia.org/wiki/Text-to-speech",
        "TL;DR: AI-powered TTS is getting really good, esp. with LLM-augmented prosody.",
    ],
}


def run_benchmark(model_path, categories, max_steps=200):
    """Run TTS engine on all category sentences."""
    lib_path = Path("build/libkyutai_dsm_tts.dylib")
    if not lib_path.exists():
        print(f"ERROR: {lib_path} not found. Run 'make libs' first.", file=sys.stderr)
        sys.exit(1)

    lib = ctypes.CDLL(str(lib_path))
    lib.kyutai_tts_create.restype = ctypes.c_void_p
    lib.kyutai_tts_create.argtypes = [ctypes.c_char_p]
    lib.kyutai_tts_destroy.argtypes = [ctypes.c_void_p]
    lib.kyutai_tts_set_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.kyutai_tts_set_text.restype = ctypes.c_int
    lib.kyutai_tts_set_text_done.argtypes = [ctypes.c_void_p]
    lib.kyutai_tts_set_text_done.restype = ctypes.c_int
    lib.kyutai_tts_step.argtypes = [ctypes.c_void_p]
    lib.kyutai_tts_step.restype = ctypes.c_int
    lib.kyutai_tts_get_audio.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.kyutai_tts_get_audio.restype = ctypes.c_int
    lib.kyutai_tts_is_done.argtypes = [ctypes.c_void_p]
    lib.kyutai_tts_is_done.restype = ctypes.c_int
    lib.kyutai_tts_reset.argtypes = [ctypes.c_void_p]
    lib.kyutai_tts_reset.restype = ctypes.c_int

    engine = lib.kyutai_tts_create(model_path.encode())
    if not engine:
        print(f"ERROR: Failed to create TTS engine from {model_path}", file=sys.stderr)
        sys.exit(1)

    results = {}
    total_audio_s = 0
    total_time_s = 0
    n_success = 0
    n_total = 0

    for cat_name, sentences in categories.items():
        cat_results = []
        print(f"\n  --- {cat_name} ({len(sentences)} sentences) ---")

        for idx, text in enumerate(sentences):
            lib.kyutai_tts_reset(engine)
            n_total += 1

            t0 = time.monotonic()
            lib.kyutai_tts_set_text(engine, text.encode("utf-8"))
            lib.kyutai_tts_set_text_done(engine)

            audio_chunks = []
            ttfs = None
            buf = (ctypes.c_float * 1920)()

            for step in range(max_steps):
                rc = lib.kyutai_tts_step(engine)
                n = lib.kyutai_tts_get_audio(engine, buf, 1920)
                if n > 0:
                    audio_chunks.append(np.ctypeslib.as_array(buf, shape=(n,)).copy())
                    if ttfs is None:
                        ttfs = time.monotonic() - t0
                if rc == 1:
                    break

            gen_time = time.monotonic() - t0

            if audio_chunks:
                audio = np.concatenate(audio_chunks)
            else:
                audio = np.zeros(0, dtype=np.float32)

            audio_dur = len(audio) / SAMPLE_RATE
            rtf = gen_time / audio_dur if audio_dur > 0 else float("inf")

            success = len(audio) > 0 and audio_dur > 0.1
            if success:
                n_success += 1
                total_audio_s += audio_dur
                total_time_s += gen_time

            status = "✓" if success else "✗"
            preview = text[:60] + "..." if len(text) > 60 else text
            print(f"    [{idx+1:2d}] {status} {audio_dur:.2f}s RTF={rtf:.3f} "
                  f"TTFS={ttfs*1000:.0f}ms | {preview}")

            cat_results.append({
                "text": text,
                "audio_duration_s": audio_dur,
                "generation_time_s": gen_time,
                "rtf": rtf,
                "ttfs_ms": ttfs * 1000 if ttfs else None,
                "n_samples": len(audio),
                "rms": float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0,
                "success": success,
            })

            # Save WAV
            wav_dir = Path(f"bench_output/emergent/{cat_name}")
            wav_dir.mkdir(parents=True, exist_ok=True)
            if len(audio) > 0:
                try:
                    import soundfile as sf
                    sf.write(str(wav_dir / f"sample_{idx:02d}.wav"), audio, SAMPLE_RATE)
                except ImportError:
                    pass

        results[cat_name] = cat_results

    lib.kyutai_tts_destroy(engine)

    summary = {
        "total_sentences": n_total,
        "successful": n_success,
        "success_rate": n_success / n_total if n_total > 0 else 0,
        "total_audio_s": total_audio_s,
        "total_generation_s": total_time_s,
        "overall_rtf": total_time_s / total_audio_s if total_audio_s > 0 else float("inf"),
    }

    # Per-category summary
    cat_summaries = {}
    for cat_name, cat_results in results.items():
        valid = [r for r in cat_results if r["success"]]
        cat_summaries[cat_name] = {
            "n_sentences": len(cat_results),
            "n_success": len(valid),
            "avg_rtf": np.mean([r["rtf"] for r in valid]) if valid else None,
            "avg_ttfs_ms": np.mean([r["ttfs_ms"] for r in valid if r["ttfs_ms"]]) if valid else None,
            "avg_audio_s": np.mean([r["audio_duration_s"] for r in valid]) if valid else None,
        }

    return results, summary, cat_summaries


def main():
    parser = argparse.ArgumentParser(description="EmergentTTS-Eval Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to .ctts model")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--max-steps", type=int, default=200, help="Max TTS steps")
    parser.add_argument("--categories", type=str, nargs="*", default=None,
                        help="Run only specific categories")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    categories = EVAL_CATEGORIES
    if args.categories:
        categories = {k: v for k, v in EVAL_CATEGORIES.items() if k in args.categories}
        if not categories:
            print(f"ERROR: No matching categories. Available: {list(EVAL_CATEGORIES.keys())}")
            sys.exit(1)

    total_sents = sum(len(v) for v in categories.values())
    print(f"\n{'=' * 70}")
    print(f"EmergentTTS-Eval Benchmark — pocket-voice")
    print(f"{'=' * 70}")
    print(f"  Model: {args.model}")
    print(f"  Categories: {len(categories)}")
    print(f"  Total sentences: {total_sents}")
    print(f"  Max steps: {args.max_steps}")

    results, summary, cat_summaries = run_benchmark(
        args.model, categories, args.max_steps
    )

    print(f"\n{'=' * 70}")
    print(f"Summary")
    print(f"{'=' * 70}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    print(f"\nPer-Category:")
    for cat_name, cs in cat_summaries.items():
        rtf_str = f"{cs['avg_rtf']:.3f}" if cs['avg_rtf'] else "N/A"
        ttfs_str = f"{cs['avg_ttfs_ms']:.0f}ms" if cs['avg_ttfs_ms'] else "N/A"
        print(f"  {cat_name:25s}: {cs['n_success']}/{cs['n_sentences']} ok, "
              f"RTF={rtf_str}, TTFS={ttfs_str}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "benchmark": "EmergentTTS-Eval",
            "engine": "c_kyutai_dsm",
            "model": args.model,
            "summary": summary,
            "per_category": cat_summaries,
            "detailed_results": {k: v for k, v in results.items()},
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
