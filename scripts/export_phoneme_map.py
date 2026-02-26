#!/usr/bin/env python3
"""
Export flat phoneme map for C phonemizer inference.

Reads PHONEME_VOCAB from train/sonata/g2p.py and exports:
1. models/sonata/phoneme_map.json — flat {symbol: id} for phonemizer_load_phoneme_map
2. models/sonata/phoneme_vocab.json — full vocabulary info (vocab_size, special tokens, etc.)

The C phonemizer (phonemizer.c) expects phoneme_map.json as a JSON object where
each key is an IPA symbol and each value is the integer ID. The C code sorts
by symbol length for greedy longest-match, so order in JSON does not matter.
"""

import json
import sys
from pathlib import Path

# Add train/sonata to path so we can import g2p
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train" / "sonata"))

from g2p import PHONEME_VOCAB, PHONE_TO_ID, ID_TO_PHONE  # noqa: E402


def main():
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models" / "sonata"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. phoneme_map.json — flat {symbol: id} for C phonemizer_load_phoneme_map
    # Include ALL tokens (pad, bos, eos, etc.) — C code uses them all
    phoneme_map = {p: i for i, p in enumerate(PHONEME_VOCAB)}

    phoneme_map_path = models_dir / "phoneme_map.json"
    with open(phoneme_map_path, "w", encoding="utf-8") as f:
        json.dump(phoneme_map, f, indent=2, ensure_ascii=False)

    # 2. phoneme_vocab.json — full vocabulary info
    phoneme_vocab = {
        "vocab_size": len(PHONEME_VOCAB),
        "phone_to_id": PHONE_TO_ID,
        "id_to_phone": {str(k): v for k, v in ID_TO_PHONE.items()},
        "special_tokens": {
            "pad": 0,
            "bos": 1,
            "eos": 2,
            "word_boundary": 3,
            "silence": 4,
        },
    }

    phoneme_vocab_path = models_dir / "phoneme_vocab.json"
    with open(phoneme_vocab_path, "w", encoding="utf-8") as f:
        json.dump(phoneme_vocab, f, indent=2, ensure_ascii=False)

    print(f"Phoneme vocabulary size: {len(PHONEME_VOCAB)}")
    print(f"Exported phoneme_map.json → {phoneme_map_path}")
    print(f"Exported phoneme_vocab.json → {phoneme_vocab_path}")
    print()
    print("Examples (first 10):")
    for i, p in enumerate(PHONEME_VOCAB[:10]):
        print(f"  {p!r}: {i}")


if __name__ == "__main__":
    main()
