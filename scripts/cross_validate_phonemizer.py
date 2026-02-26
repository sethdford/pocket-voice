#!/usr/bin/env python3
"""Cross-validate C phonemizer and Python PhonemeFrontend phoneme IDs.

Compares output from:
  - Python: train/sonata/g2p.py PhonemeFrontend (phonemizer lib → espeak-ng → PHONEME_VOCAB)
  - C: src/phonemizer.c (espeak_TextToPhonemes → phoneme_map.json lookup)

Both use espeak-ng and the same phoneme vocabulary (phoneme_map.json).
IPA and phoneme ID sequences may differ due to:
  - Python phonemizer adds | word-boundary markers; raw espeak may not
  - Minor formatting (spaces, stress marks)

Usage:
    python scripts/cross_validate_phonemizer.py

Requires: phonemizer package (pip install phonemizer), espeak-ng (brew install espeak-ng)
C phonemizer: optional, used when build/libphonemizer.dylib exists and make libs has been run.
"""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "train" / "sonata"))

# Test sentences covering edge cases
TEST_SENTENCES = [
    # Simple
    "Hello world",
    "Good morning",
    "How are you today?",
    # Numbers
    "I have 42 cats",
    "The year is 2026",
    "It costs $99.99",
    # Punctuation
    "Really? Yes! Oh...",
    "Wait — what did you say?",
    "She said, \"Hello!\"",
    # Contractions
    "I don't know, she's here",
    "We'll go there, won't we?",
    "That's 5 o'clock",
    # Proper nouns
    "John went to Paris",
    "Dr. Smith from Washington D.C.",
    "NASA launched a rocket",
    # Heteronyms
    "I will read the read book",
    "The bass swam near the bass guitar",
    "She wound the wound tightly",
    # Long
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump!",
    # Em-dash and ellipsis
    "Well — I suppose... maybe.",
    # Empty-like
    "A",
    "I",
]

PHONEME_MAP_PATHS = [
    ROOT / "models" / "sonata" / "phoneme_map.json",
    ROOT / "scripts" / "phoneme_map.json",
]
BUILD_DIR = ROOT / "build"
DYLIB_PATH = BUILD_DIR / "libphonemizer.dylib"
HOMEBREW_PREFIX = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")


def load_phoneme_map():
    """Load phoneme map from models/sonata/ or scripts/."""
    for p in PHONEME_MAP_PATHS:
        if p.exists():
            with open(p) as f:
                return json.load(f), str(p)
    return None, None


def get_python_results():
    """Get Python PhonemeFrontend results."""
    try:
        from g2p import PhonemeFrontend
    except ImportError as e:
        print(f"[ERROR] Cannot import PhonemeFrontend: {e}")
        print("  Install: pip install phonemizer")
        return None

    g2p = PhonemeFrontend(language="en-us")

    results = []
    for text in TEST_SENTENCES:
        try:
            ipa = g2p.text_to_phonemes(text)
            ids_full = g2p.encode(text, add_bos=True, add_eos=True)
            ids_core = g2p.encode(text, add_bos=False, add_eos=False)
            results.append({
                "text": text,
                "ipa": ipa,
                "ids_full": ids_full.tolist(),
                "ids_core": ids_core.tolist(),
            })
        except Exception as e:
            results.append({"text": text, "error": str(e)})
    return results


def get_c_results_via_ctypes():
    """Call C phonemizer via ctypes if available."""
    if not DYLIB_PATH.exists():
        return None

    try:
        import ctypes
    except ImportError:
        return None

    try:
        # libphonemizer.dylib has rpath to HOMEBREW_PREFIX/lib for espeak-ng
        lib = ctypes.CDLL(str(DYLIB_PATH))
    except OSError as e:
        print(f"[WARN] Cannot load libphonemizer.dylib: {e}")
        return None

    # Declare C API
    lib.phonemizer_create.argtypes = [ctypes.c_char_p]
    lib.phonemizer_create.restype = ctypes.c_void_p

    lib.phonemizer_destroy.argtypes = [ctypes.c_void_p]
    lib.phonemizer_destroy.restype = None

    lib.phonemizer_load_phoneme_map.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.phonemizer_load_phoneme_map.restype = ctypes.c_int

    lib.phonemizer_text_to_ipa.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                            ctypes.c_char_p, ctypes.c_int]
    lib.phonemizer_text_to_ipa.restype = ctypes.c_int

    lib.phonemizer_text_to_ids.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                           ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.phonemizer_text_to_ids.restype = ctypes.c_int

    phoneme_map_path = None
    for p in PHONEME_MAP_PATHS:
        if p.exists():
            phoneme_map_path = str(p)
            break
    if not phoneme_map_path:
        print("[WARN] No phoneme_map.json found for C phonemizer")
        return None

    ph = lib.phonemizer_create(b"en-us")
    if not ph:
        print("[WARN] C phonemizer_create failed")
        return None

    if lib.phonemizer_load_phoneme_map(ph, phoneme_map_path.encode()) != 0:
        lib.phonemizer_destroy(ph)
        print("[WARN] C phonemizer_load_phoneme_map failed")
        return None

    results = []
    ipa_buf = ctypes.create_string_buffer(4096)
    ids_buf = (ctypes.c_int * 512)()

    for text in TEST_SENTENCES:
        text_bytes = text.encode("utf-8")
        ipa_len = lib.phonemizer_text_to_ipa(ph, text_bytes, ipa_buf, 4096)
        n_ids = lib.phonemizer_text_to_ids(ph, text_bytes, ids_buf, 512)

        if ipa_len < 0:
            results.append({"text": text, "error": "text_to_ipa failed"})
        elif n_ids < 0:
            results.append({"text": text, "error": "text_to_ids failed", "ipa": ipa_buf.value.decode("utf-8", errors="replace")})
        else:
            ipa_str = ipa_buf.value.decode("utf-8", errors="replace")
            ids_list = list(ids_buf[:n_ids])
            results.append({
                "text": text,
                "ipa": ipa_str,
                "ids_core": ids_list,
            })

    lib.phonemizer_destroy(ph)
    return results


def compare_and_report(py_results, c_results):
    """Print comparison table and report mismatches."""
    print("\n" + "=" * 70)
    print("  Phonemizer Cross-Validation: Python vs C")
    print("=" * 70)

    n = len(TEST_SENTENCES)
    ipa_matches = 0
    id_matches = 0
    mismatches = []

    for i, text in enumerate(TEST_SENTENCES):
        py = py_results[i] if py_results else {}
        c = c_results[i] if c_results else {}

        if "error" in py:
            print(f"\n[{i+1}/{n}] ERROR (Python): {text[:50]}...")
            print(f"      {py['error']}")
            continue
        if c and "error" in c:
            print(f"\n[{i+1}/{n}] ERROR (C): {text[:50]}...")
            print(f"      {c['error']}")
            continue

        py_ipa = py.get("ipa", "")
        py_ids = py.get("ids_core", [])
        c_ipa = c.get("ipa", "") if c else ""
        c_ids = c.get("ids_core", []) if c else []

        ipa_eq = py_ipa == c_ipa
        id_eq = py_ids == c_ids

        if ipa_eq:
            ipa_matches += 1
        if id_eq:
            id_matches += 1

        if not ipa_eq or not id_eq:
            mismatches.append({
                "i": i + 1,
                "text": text,
                "py_ipa": py_ipa,
                "c_ipa": c_ipa,
                "py_ids": py_ids,
                "c_ids": c_ids,
            })

        # Print each sentence (truncate long text)
        display = text[:60] + "..." if len(text) > 60 else text
        status = "OK" if (ipa_eq and id_eq) else "DIFF"
        print(f"\n[{i+1}/{n}] [{status}] {display!r}")
        print(f"      Python IPA: {py_ipa[:80]}{'...' if len(py_ipa) > 80 else ''}")
        print(f"      Python IDs: {py_ids[:25]}{'...' if len(py_ids) > 25 else ''} (n={len(py_ids)})")
        if c:
            print(f"      C IPA:     {c_ipa[:80]}{'...' if len(c_ipa) > 80 else ''}")
            print(f"      C IDs:     {c_ids[:25]}{'...' if len(c_ids) > 25 else ''} (n={len(c_ids)})")
            if not id_eq:
                # Show first divergence
                for j, (a, b) in enumerate(zip(py_ids, c_ids)):
                    if a != b:
                        print(f"      First ID diff at index {j}: Python={a} vs C={b}")
                        break
                if len(py_ids) != len(c_ids):
                    print(f"      Length diff: Python={len(py_ids)} vs C={len(c_ids)}")

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Sentences: {n}")
    print(f"  Python results: {'OK' if py_results else 'FAILED/ missing phonemizer'}")
    print(f"  C results:      {'OK' if c_results else 'SKIPPED (libphonemizer.dylib not built)'}")
    if c_results:
        print(f"  IPA matches:   {ipa_matches}/{n}")
        print(f"  ID matches:    {id_matches}/{n}")
        if mismatches:
            print(f"  Mismatches:    {len(mismatches)}")
        else:
            print("  All phoneme ID sequences match.")
    print("=" * 70 + "\n")
    return len(mismatches) if c_results else 0


def main():
    phoneme_map, map_path = load_phoneme_map()
    if not phoneme_map:
        print("[ERROR] No phoneme_map.json found. Expected at models/sonata/phoneme_map.json")
        return 1
    print(f"[OK] Loaded phoneme map from {map_path} ({len(phoneme_map)} entries)")

    print("\n--- Python PhonemeFrontend ---")
    py_results = get_python_results()
    if not py_results:
        return 1

    print("\n--- C phonemizer (ctypes) ---")
    c_results = get_c_results_via_ctypes()
    if not c_results:
        print("  C phonemizer not available (run 'make libs' to build libphonemizer.dylib)")

    mismatches = compare_and_report(py_results, c_results)
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
