"""
Data pipeline for Mimi-Lite TTS training on Emilia-YODAS NeuCodec dataset.

Dataset: neuphonic/emilia-yodas-english-neucodec (Parquet)
Schema:
  - text: str           — transcript
  - codes: list[int]    — NeuCodec FSQ codes (50 tokens/sec)
  - duration: float     — audio duration in seconds
  - dnsmos: float       — audio quality score
  - speaker: str        — speaker identifier

Token alignment strategy:
  Text tokens are spread across the audio sequence proportionally.
  For each audio position, we compute which text token should be active.
  The model learns to pace text → audio timing.
"""

import math
from dataclasses import dataclass
from typing import Optional

import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DataConfig:
    dataset_name: str = "neuphonic/emilia-yodas-english-neucodec"
    tokenizer_path: str = "models/tokenizer.model"
    max_audio_len: int = 1000      # 20 seconds at 50 tokens/sec
    min_audio_len: int = 50        # 1 second
    min_dnsmos: float = 3.0
    max_text_len: int = 512
    pad_token: int = 0
    bos_token: int = 1
    eos_token: int = 2


class TtsDataset(Dataset):
    """Loads pre-encoded NeuCodec tokens + text for TTS training."""

    def __init__(self, hf_dataset, tokenizer: spm.SentencePieceProcessor,
                 cfg: DataConfig):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        text = sample["text"].strip()
        codes = sample["codes"]
        speaker = sample.get("speaker", "unknown")

        text_ids = self.tokenizer.encode(text)

        if len(text_ids) > self.cfg.max_text_len:
            text_ids = text_ids[:self.cfg.max_text_len]

        audio_codes = list(codes)
        if len(audio_codes) > self.cfg.max_audio_len:
            audio_codes = audio_codes[:self.cfg.max_audio_len]

        seq_len = len(audio_codes)

        # Align text tokens to audio positions proportionally.
        # Each audio position gets a text token based on its relative position.
        n_text = len(text_ids)
        aligned_text = []
        for i in range(seq_len):
            text_idx = min(int(i * n_text / seq_len), n_text - 1)
            aligned_text.append(text_ids[text_idx] + 4)  # +4 to skip special tokens

        # Audio input (shifted right): [BOS, code_0, code_1, ..., code_{T-2}]
        audio_input = [self.cfg.bos_token] + [c + 4 for c in audio_codes[:-1]]

        # Audio target: [code_0, code_1, ..., code_{T-1}] (offset by +4 for specials)
        audio_target = [c + 4 for c in audio_codes]

        return {
            "text_tokens": torch.tensor(aligned_text, dtype=torch.long),
            "audio_input": torch.tensor(audio_input, dtype=torch.long),
            "audio_target": torch.tensor(audio_target, dtype=torch.long),
            "seq_len": seq_len,
            "speaker": speaker,
        }


def collate_fn(batch):
    """Pad sequences to max length in batch."""
    max_len = max(b["seq_len"] for b in batch)

    text_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    audio_input = torch.zeros(len(batch), max_len, dtype=torch.long)
    audio_target = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["seq_len"]
        text_tokens[i, :L] = b["text_tokens"]
        audio_input[i, :L] = b["audio_input"]
        audio_target[i, :L] = b["audio_target"]

    return {
        "text_tokens": text_tokens,
        "audio_input": audio_input,
        "audio_target": audio_target,
    }


def create_dataloader(
    tokenizer_path: str,
    batch_size: int = 4,
    num_workers: int = 2,
    max_samples: Optional[int] = None,
    cfg: Optional[DataConfig] = None,
) -> DataLoader:
    """Create training DataLoader from HuggingFace dataset."""
    if cfg is None:
        cfg = DataConfig(tokenizer_path=tokenizer_path)

    from datasets import load_dataset

    print(f"Loading dataset: {cfg.dataset_name}")
    ds = load_dataset(cfg.dataset_name, split="train", streaming=False,
                      num_proc=4)

    # Filter by quality and length
    original_len = len(ds)
    ds = ds.filter(
        lambda x: (x["dnsmos"] >= cfg.min_dnsmos and
                    cfg.min_audio_len <= len(x["codes"]) <= cfg.max_audio_len),
        num_proc=4,
    )
    print(f"Filtered: {original_len:,} → {len(ds):,} samples "
          f"(dnsmos >= {cfg.min_dnsmos}, len in [{cfg.min_audio_len}, {cfg.max_audio_len}])")

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))
        print(f"Truncated to {max_samples:,} samples")

    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

    dataset = TtsDataset(ds, tokenizer, cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"DataLoader: {len(dataset):,} samples, {len(loader):,} batches, bs={batch_size}")
    return loader


def train_tokenizer(dataset_name: str, output_path: str, vocab_size: int = 32000):
    """Train a SentencePiece tokenizer on dataset transcripts."""
    from datasets import load_dataset
    import tempfile
    import os

    print(f"Loading dataset for tokenizer training...")
    ds = load_dataset(dataset_name, split="train", streaming=True)

    # Collect text samples
    texts = []
    for i, sample in enumerate(ds):
        if i >= 500000:
            break
        texts.append(sample["text"].strip())

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    for t in texts:
        tmp.write(t + "\n")
    tmp.close()

    print(f"Training SentencePiece on {len(texts):,} texts...")
    spm.SentencePieceTrainer.train(
        input=tmp.name,
        model_prefix=output_path.replace(".model", ""),
        vocab_size=vocab_size,
        model_type="unigram",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        character_coverage=0.9995,
        num_threads=8,
    )
    os.unlink(tmp.name)
    print(f"Tokenizer saved to {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train-tokenizer":
        output = sys.argv[2] if len(sys.argv) > 2 else "models/tokenizer.model"
        train_tokenizer("neuphonic/emilia-yodas-english-neucodec", output)
    else:
        print("Usage:")
        print("  python data.py train-tokenizer [output_path]")
        print()
        print("Quick test with mock tokenizer...")

        # Test with a simple character tokenizer
        class MockTokenizer:
            def encode(self, text):
                return [ord(c) % 1000 for c in text[:100]]

        from datasets import load_dataset
        ds = load_dataset("neuphonic/emilia-yodas-english-neucodec",
                         split="train", streaming=True)

        samples = []
        for i, s in enumerate(ds):
            if i >= 10:
                break
            samples.append(s)

        cfg = DataConfig()
        mock_tok = MockTokenizer()

        from torch.utils.data import Dataset as _D

        class ListDataset(_D):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        hf_ds = ListDataset(samples)
        tts_ds = TtsDataset(hf_ds, mock_tok, cfg)

        for i in range(min(3, len(tts_ds))):
            batch = tts_ds[i]
            print(f"\nSample {i}:")
            print(f"  text_tokens: {batch['text_tokens'].shape}")
            print(f"  audio_input: {batch['audio_input'].shape}")
            print(f"  audio_target: {batch['audio_target'].shape}")
            print(f"  seq_len: {batch['seq_len']}")
