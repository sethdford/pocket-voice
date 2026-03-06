"""Visualize text-to-mel alignment from Sonata Flow v3.

Shows predicted durations and optionally MAS alignment when reference audio
is provided. Supports both character and phoneme mode (auto-detected from
char_vocab_size in checkpoint).

Usage:
  python visualize_alignment.py --checkpoint path/to/flow.pt --text "Hello world" --output alignment.png
  python visualize_alignment.py --checkpoint flow.pt --text "Hello" --audio ref.wav  # with MAS
  python visualize_alignment.py --checkpoint flow.pt  # use built-in test sentences
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless by default
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import FlowV3Config
from flow_v3 import SonataFlowV3, InterleavedEncoder, DurationPredictor
from modules import mas_durations

try:
    from g2p import PhonemeFrontend
    HAS_G2P = True
except ImportError:
    HAS_G2P = False

DEFAULT_SENTENCES = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Sonata is a streaming text to speech system.",
]


def extract_mel(audio: torch.Tensor, n_fft=1024, hop_length=480, n_mels=80, sample_rate=24000) -> torch.Tensor:
    """Extract log-mel spectrogram from audio."""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    window = torch.hann_window(n_fft, periodic=True, device=audio.device)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    mag = stft.abs().pow(2)
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0, sample_rate / 2, n_freqs, device=audio.device)
    mel_low = 2595 * math.log10(1 + 0 / 700)
    mel_high = 2595 * math.log10(1 + sample_rate / 2 / 700)
    mels = torch.linspace(mel_low, mel_high, n_mels + 2, device=audio.device)
    hz = 700 * (10 ** (mels / 2595) - 1)
    fb = torch.zeros(n_mels, n_freqs, device=audio.device)
    for i in range(n_mels):
        low, center, high = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - low) / (center - low + 1e-8)
        down = (high - freqs) / (high - center + 1e-8)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0)
    mel = torch.matmul(fb, mag.squeeze(0))
    mel = torch.log(mel.clamp(min=1e-5))
    return mel.T  # (T_mel, n_mels)


def load_flow(ckpt_path: str, device: str = "cpu") -> tuple:
    """Load Flow v3 model and config from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = FlowV3Config(**{k: v for k, v in cfg_dict.items()
                         if k in FlowV3Config.__dataclass_fields__})
    model = SonataFlowV3(cfg).to(device)
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def text_to_ids(text: str, cfg, g2p=None) -> torch.Tensor:
    """Convert text to token IDs. Uses phoneme or character based on char_vocab_size."""
    is_phoneme = cfg.char_vocab_size < 200 and HAS_G2P and g2p is not None
    if is_phoneme:
        ids = g2p.encode(text, add_bos=True, add_eos=True)
    else:
        ids = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
    return ids


def get_token_labels(ids: torch.Tensor, cfg, g2p=None) -> list:
    """Return readable labels for each token (for plot y-axis)."""
    is_phoneme = cfg.char_vocab_size < 200 and HAS_G2P and g2p is not None
    labels = []
    for i in ids.tolist():
        if is_phoneme and g2p and i in g2p.id_to_phone:
            label = g2p.id_to_phone[i]
            if label in ("<pad>", "<bos>", "<eos>"):
                label = f"[{label[1:-1]}]"
            labels.append(label)
        else:
            c = chr(i) if 32 <= i < 127 else f"\\x{i:02x}"
            labels.append(c)
    return labels


def durations_to_alignment_matrix(durations: torch.Tensor, n_frames: int) -> np.ndarray:
    """Convert per-token durations to (n_tokens, n_frames) alignment matrix."""
    B, T_text = durations.shape
    mat = np.zeros((T_text, n_frames), dtype=np.float32)
    for b in range(B):
        pos = 0
        for i in range(T_text):
            dur = int(durations[b, i].item())
            if dur <= 0:
                continue
            end = min(pos + dur, n_frames)
            mat[i, pos:end] = 1.0
            pos = end
            if pos >= n_frames:
                break
    return mat


def main():
    parser = argparse.ArgumentParser(description="Visualize Flow v3 text-to-mel alignment")
    parser.add_argument("--checkpoint", required=True, help="Path to Flow v3 .pt checkpoint")
    parser.add_argument("--text", type=str, help="Input text (else use built-in sentences)")
    parser.add_argument("--audio", type=str, help="Optional: reference audio for MAS alignment")
    parser.add_argument("--output", type=str, help="Output PNG path (else display)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    device = args.device
    model, cfg = load_flow(args.checkpoint, device)
    g2p = PhonemeFrontend() if HAS_G2P else None

    texts = [args.text] if args.text else DEFAULT_SENTENCES
    mel_ref = None
    if args.audio and len(texts) == 1:
        try:
            import soundfile as sf
            data, sr = sf.read(args.audio, dtype="float32")
            audio = torch.from_numpy(data)
            if audio.dim() > 1:
                audio = audio.mean(dim=-1)
            if sr != cfg.sample_rate:
                ratio = cfg.sample_rate / sr
                new_len = int(audio.shape[0] * ratio)
                audio = torch.nn.functional.interpolate(
                    audio.unsqueeze(0).unsqueeze(0),
                    size=new_len,
                    mode="linear",
                    align_corners=False,
                ).squeeze()
            mel_ref = extract_mel(
                audio.unsqueeze(0).to(device),
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                n_mels=cfg.mel_dim,
                sample_rate=cfg.sample_rate,
            )
        except Exception as e:
            print(f"[warn] Could not load audio {args.audio}: {e}")
            mel_ref = None
    elif args.audio and len(texts) > 1:
        print("[warn] --audio ignored when using multiple sentences")

    n_cols = 3 if mel_ref is not None else 2
    n_plots = len(texts)
    fig, axes = plt.subplots(n_plots, n_cols, squeeze=False, figsize=(5 * n_cols, 4 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)

    for row, text in enumerate(texts):
        char_ids = text_to_ids(text, cfg, g2p).unsqueeze(0).to(device)
        with torch.no_grad():
            text_enc = model.interleaved_enc.encode_text(char_ids)
            log_dur = model.duration_predictor(text_enc)
            dur_pred = torch.exp(log_dur).clamp(min=1) * (char_ids > 0).float()
            dur_pred_int = dur_pred.round().long()
            n_frames = int(dur_pred_int.sum().item())

            if mel_ref is not None and row == 0:
                T_mel = mel_ref.shape[1]
                mel_use = mel_ref[:, :T_mel]
                gt_durations = mas_durations(text_enc, mel_use)
                n_frames = T_mel
            else:
                gt_durations = None
                n_frames = max(10, int(dur_pred_int.sum().item()))

            # Normalize durations to sum to n_frames
            dur_sum = dur_pred_int.sum().item()
            if dur_sum != n_frames:
                diff = n_frames - dur_sum
                idx = dur_pred_int.argmax(dim=-1)
                dur_pred_int[0, idx] = dur_pred_int[0, idx] + diff

        all_labels = get_token_labels(char_ids[0], cfg, g2p)
        valid = [i for i in range(len(all_labels)) if char_ids[0, i].item() > 0]
        labels = [all_labels[i] for i in valid]
        dur_pred_np = dur_pred_int[0, valid].cpu().numpy()

        # 1. Heatmap: predicted alignment
        ax_pred = axes[row, 0]
        dur_for_mat = torch.zeros_like(dur_pred_int)
        for j, idx in enumerate(valid):
            dur_for_mat[0, idx] = dur_pred_int[0, idx]
        mat_pred = durations_to_alignment_matrix(dur_for_mat.clamp(min=0), n_frames)
        # Trim mat_pred to valid rows only
        mat_pred = mat_pred[valid]
        ax_pred.imshow(mat_pred, aspect="auto", origin="lower", cmap="viridis")
        ax_pred.set_ylabel("Token")
        ax_pred.set_xlabel("Mel frame")
        ax_pred.set_title("Predicted alignment")
        ax_pred.set_yticks(range(len(labels)))
        ax_pred.set_yticklabels(labels, fontsize=8)

        # 2. Bar chart: predicted durations
        ax_bar = axes[row, 1]
        ax_bar.bar(range(len(dur_pred_np)), dur_pred_np, color="steelblue", edgecolor="navy")
        ax_bar.set_xlabel("Token index")
        ax_bar.set_ylabel("Duration (frames)")
        ax_bar.set_title("Predicted durations")
        ax_bar.set_xticks(range(len(labels)))
        ax_bar.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")

        # 3. (Optional) MAS alignment heatmap
        if gt_durations is not None and axes.shape[1] > 2:
            ax_mas = axes[row, 2]
            mat_mas = durations_to_alignment_matrix(gt_durations.clamp(min=0), n_frames)
            mat_mas = mat_mas[valid]
            ax_mas.imshow(mat_mas, aspect="auto", origin="lower", cmap="magma")
            ax_mas.set_ylabel("Token")
            ax_mas.set_xlabel("Mel frame")
            ax_mas.set_title("MAS alignment (from audio)")
            ax_mas.set_yticks(range(len(labels)))
            ax_mas.set_yticklabels(labels, fontsize=8)

    fig.suptitle(f"Flow v3 Alignment ({'phoneme' if cfg.char_vocab_size < 200 else 'character'} mode)")
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {args.output}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()
