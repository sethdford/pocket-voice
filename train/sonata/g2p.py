"""Grapheme-to-Phoneme frontend for Sonata TTS.

Converts text to IPA phoneme sequences using espeak-ng via the phonemizer library.
This dramatically improves pronunciation quality compared to character-level input.

The phoneme vocabulary is a fixed set of ~100 IPA phones for American English,
plus special tokens (PAD, BOS, EOS, WORD_BOUNDARY, SILENCE).

Usage:
    from g2p import PhonemeFrontend
    g2p = PhonemeFrontend()
    ids = g2p.encode("Hello, world!")  # → tensor of phoneme IDs
    text = g2p.decode(ids)             # → "h ə l oʊ | w ɜː l d"
"""

from typing import List, Optional

import torch


PHONEME_VOCAB = [
    "<pad>", "<bos>", "<eos>", "|", "<sil>",
    # Vowels (monophthongs)
    "ɪ", "ɛ", "æ", "ɑː", "ɒ", "ʊ", "ʌ", "ə", "ɜː", "iː", "uː", "ɔː",
    "ɐ", "ɚ", "ɝ", "i", "u", "ɑ",
    # Diphthongs
    "eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ", "ɪɹ", "ɛɹ", "ʊɹ",
    # Consonants (plosives)
    "p", "b", "t", "d", "k", "ɡ", "ʔ",
    # Consonants (fricatives)
    "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    # Consonants (affricates)
    "tʃ", "dʒ",
    # Consonants (nasals)
    "m", "n", "ŋ",
    # Consonants (approximants)
    "l", "ɹ", "j", "w",
    # Consonants (other)
    "ɾ", "ɬ",
    # Stress/prosody markers
    "ˈ", "ˌ", "ː",
    # Punctuation-derived pauses
    ",", ".", "?", "!", ";",
    # OOV fallback (id 66)
    "<unk>",
]

PHONE_TO_ID = {p: i for i, p in enumerate(PHONEME_VOCAB)}
ID_TO_PHONE = {i: p for i, p in enumerate(PHONEME_VOCAB)}
UNK_ID = PHONE_TO_ID["<unk>"]


class PhonemeFrontend:
    """Convert text to phoneme token IDs using espeak-ng."""

    _oov_warned: set = set()  # class-level: warn once per unique OOV phone

    def __init__(self, language: str = "en-us"):
        self.language = language
        self.vocab = PHONEME_VOCAB
        self.phone_to_id = PHONE_TO_ID
        self.id_to_phone = ID_TO_PHONE
        self._backend = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _init_backend(self):
        if self._backend is not None:
            return
        try:
            from phonemizer import phonemize
            from phonemizer.separator import Separator
            self._phonemize = phonemize
            self._separator = Separator(phone=" ", word="| ", syllable="")
            self._backend = "espeak-ng"
        except ImportError:
            self._backend = "fallback"

    def text_to_phonemes(self, text: str) -> str:
        """Convert text to space-separated phoneme string."""
        self._init_backend()
        if self._backend == "espeak-ng":
            result = self._phonemize(
                [text], language=self.language, backend="espeak",
                separator=self._separator, strip=True,
            )[0]
            return result
        return " ".join(list(text.lower()))

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> torch.Tensor:
        """Convert text to tensor of phoneme token IDs."""
        phoneme_str = self.text_to_phonemes(text)
        tokens = phoneme_str.split()

        ids = []
        if add_bos:
            ids.append(self.phone_to_id["<bos>"])

        for token in tokens:
            if token in self.phone_to_id:
                ids.append(self.phone_to_id[token])
            else:
                # Unknown phone -- try to map individual characters first
                mapped_any = False
                for ch in token:
                    if ch in self.phone_to_id:
                        ids.append(self.phone_to_id[ch])
                        mapped_any = True
                if not mapped_any:
                    # OOV with no character-level match: use UNK fallback
                    ids.append(UNK_ID)
                    if token not in PhonemeFrontend._oov_warned:
                        import warnings
                        warnings.warn(f"[g2p] OOV phone '{token}' mapped to <unk>", stacklevel=2)
                        PhonemeFrontend._oov_warned.add(token)

        if add_eos:
            ids.append(self.phone_to_id["<eos>"])

        return torch.tensor(ids, dtype=torch.long)

    def encode_batch(self, texts: List[str], add_bos: bool = True,
                     add_eos: bool = True) -> List[torch.Tensor]:
        """Encode a batch of texts to phoneme token tensors."""
        return [self.encode(t, add_bos, add_eos) for t in texts]

    def decode(self, ids: torch.Tensor) -> str:
        """Convert phoneme token IDs back to readable string."""
        phones = []
        for idx in ids.tolist():
            if idx in self.id_to_phone:
                phone = self.id_to_phone[idx]
                if phone not in ("<pad>", "<bos>", "<eos>", "<unk>"):
                    phones.append(phone)
        return " ".join(phones)


def build_phoneme_vocab_json(output_path: str = "models/sonata/phoneme_vocab.json"):
    """Export phoneme vocabulary as JSON for Rust inference."""
    import json
    from pathlib import Path

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    vocab = {
        "vocab_size": len(PHONEME_VOCAB),
        "phone_to_id": PHONE_TO_ID,
        "id_to_phone": {str(k): v for k, v in ID_TO_PHONE.items()},
        "special_tokens": {
            "pad": 0, "bos": 1, "eos": 2,
            "word_boundary": 3, "silence": 4, "unk": 66,
        },
    }

    with open(out, "w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"[g2p] Exported phoneme vocab ({len(PHONEME_VOCAB)} tokens) → {out}")


if __name__ == "__main__":
    g2p = PhonemeFrontend()

    tests = [
        "Hello, this is a test of the Sonata text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "Can you believe it? That's absolutely amazing!",
        "Dr. Smith went to Washington D.C. on January 5th, 2026.",
    ]

    print(f"Phoneme vocabulary: {g2p.vocab_size} tokens\n")
    for text in tests:
        phonemes = g2p.text_to_phonemes(text)
        ids = g2p.encode(text)
        decoded = g2p.decode(ids)
        print(f"  Text:     {text}")
        print(f"  Phonemes: {phonemes}")
        print(f"  IDs:      {ids.tolist()[:20]}... ({len(ids)} tokens)")
        print(f"  Decoded:  {decoded}")
        print()

    build_phoneme_vocab_json()
