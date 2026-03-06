"""Encode audio files with Kyutai's Mimi codec to produce multi-codebook RVQ tokens.

Creates training data in the format: (text_tokens, audio_codes[n_q][T]) per utterance.

Uses the pre-trained Mimi codec from kyutai/moshiko-pytorch-bf16 or similar.
Processes LibriSpeech, Emilia-YODAS, or any wav/audio dataset.

Output: .pt files with {text_ids: LongTensor, audio_codes: LongTensor[n_q, T]}
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download


def load_mimi_codec(device: str = "mps"):
    """Load Kyutai's pre-trained Mimi codec."""
    from moshi.models.loaders import get_mimi

    mimi = get_mimi(device=device)
    mimi.eval()
    print(f"[encode] Mimi codec loaded on {device}")
    print(f"[encode]   sample_rate={mimi.sample_rate}, "
          f"frame_rate={mimi.frame_rate}, "
          f"num_codebooks={mimi.num_codebooks}")
    return mimi


@torch.no_grad()
def encode_audio(mimi, audio_path: str, target_sr: int = 24000) -> torch.Tensor:
    """Encode a single audio file → (n_codebooks, T) LongTensor of RVQ codes."""
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Add batch dim: (1, 1, T)
    waveform = waveform.unsqueeze(0).to(mimi.device)

    # Encode
    codes = mimi.encode(waveform)  # (B, n_q, T_frames)
    return codes.squeeze(0).cpu()  # (n_q, T_frames)


def process_librispeech(data_dir: str, mimi, output_dir: str, max_samples: int = 0):
    """Process LibriSpeech dataset: each .flac file + .trans.txt transcript."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    trans_files = list(data_path.rglob("*.trans.txt"))
    print(f"[encode] Found {len(trans_files)} transcript files")

    count = 0
    for trans_file in sorted(trans_files):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) < 2:
                    continue
                utt_id, text = parts[0], parts[1]

                audio_file = trans_file.parent / f"{utt_id}.flac"
                if not audio_file.exists():
                    continue

                try:
                    codes = encode_audio(mimi, str(audio_file))
                    sample = {
                        "utt_id": utt_id,
                        "text": text,
                        "audio_codes": codes,
                    }
                    torch.save(sample, out_path / f"{utt_id}.pt")
                    count += 1

                    if count % 100 == 0:
                        print(f"[encode] Processed {count} utterances "
                              f"(last: {utt_id}, codes shape: {codes.shape})")

                    if max_samples > 0 and count >= max_samples:
                        print(f"[encode] Reached max_samples={max_samples}")
                        return count

                except Exception as e:
                    print(f"[encode] Error on {utt_id}: {e}")
                    continue

    print(f"[encode] Done: {count} utterances encoded")
    return count


def process_audio_dir(audio_dir: str, transcript_file: str, mimi,
                      output_dir: str, max_samples: int = 0):
    """Process a directory of audio files with a JSON/TSV transcript file.

    transcript_file format (JSON lines):
      {"audio": "path/to/file.wav", "text": "transcription"}
    or TSV:
      path/to/file.wav\ttranscription
    """
    data_path = Path(audio_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    transcripts = {}
    tf = Path(transcript_file)
    if tf.suffix == ".json" or tf.suffix == ".jsonl":
        with open(tf) as f:
            for line in f:
                entry = json.loads(line.strip())
                transcripts[entry["audio"]] = entry["text"]
    elif tf.suffix in (".tsv", ".csv"):
        sep = "\t" if tf.suffix == ".tsv" else ","
        with open(tf) as f:
            for line in f:
                parts = line.strip().split(sep, 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]

    print(f"[encode] Found {len(transcripts)} transcripts")

    count = 0
    for audio_rel, text in sorted(transcripts.items()):
        audio_file = data_path / audio_rel
        if not audio_file.exists():
            continue

        utt_id = audio_file.stem
        try:
            codes = encode_audio(mimi, str(audio_file))
            sample = {"utt_id": utt_id, "text": text, "audio_codes": codes}
            torch.save(sample, out_path / f"{utt_id}.pt")
            count += 1

            if count % 100 == 0:
                print(f"[encode] {count} done (last: {utt_id}, shape: {codes.shape})")

            if max_samples > 0 and count >= max_samples:
                return count

        except Exception as e:
            print(f"[encode] Error on {utt_id}: {e}")
            continue

    print(f"[encode] Done: {count} utterances")
    return count


def create_synthetic_data(mimi, output_dir: str, n_samples: int = 100):
    """Create synthetic training samples for initial testing.

    Generates random audio and encodes it, paired with dummy text.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    test_sentences = [
        "Hello, this is a test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "Speech synthesis converts text into natural sounding audio.",
        "Machine learning models can generate realistic speech.",
        "This voice pipeline runs entirely on Apple Silicon hardware.",
        "Real time inference requires efficient model architecture.",
        "The transformer model processes input tokens sequentially.",
        "Audio codebooks capture different frequency components.",
        "Multiple codebooks improve reconstruction quality.",
        "The depformer predicts codebook tokens in depth order.",
    ]

    for i in range(n_samples):
        text = test_sentences[i % len(test_sentences)]
        n_frames = torch.randint(20, 200, (1,)).item()
        n_q = mimi.num_codebooks if hasattr(mimi, 'num_codebooks') else 8
        audio_bins = 2048

        codes = torch.randint(0, audio_bins, (n_q, n_frames))
        sample = {"utt_id": f"synth_{i:05d}", "text": text, "audio_codes": codes}
        torch.save(sample, out_path / f"synth_{i:05d}.pt")

    print(f"[encode] Created {n_samples} synthetic samples in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Encode audio with Mimi codec")
    parser.add_argument("--mode", choices=["librispeech", "audiodir", "synthetic"],
                        default="synthetic")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--transcript", type=str, default="")
    parser.add_argument("--output", type=str, default="train/data/mimi_encoded")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    if args.mode == "synthetic":
        class FakeMimi:
            num_codebooks = 8
            sample_rate = 24000
            frame_rate = 12.5
        create_synthetic_data(FakeMimi(), args.output, n_samples=200)
    else:
        mimi = load_mimi_codec(args.device)
        if args.mode == "librispeech":
            process_librispeech(args.data_dir, mimi, args.output, args.max_samples)
        elif args.mode == "audiodir":
            process_audio_dir(args.data_dir, args.transcript, mimi, args.output, args.max_samples)


if __name__ == "__main__":
    main()
