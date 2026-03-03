"""Train Sonata STT: Conformer CTC on codec embeddings.

Architecture matches sonata-stt Rust crate:
- input_proj: Linear(512→512)
- 12 ConformerBlocks (dim=512, ffn_dim=2048, 8 heads):
    - MultiHeadSelfAttention with separate Q/K/V projections
    - Conv1d (kernel=31, padding=15) with ReLU
    - SwiGLU FFN (gate + up + down)
    - 2 LayerNorms
- output_proj: Linear(512→32000)
- CTC loss with blank=0

Input: codec embeddings [B, 512, T] from pre-trained codec.
If no codec available, can train on mel spectrograms with --mel-mode.

Usage:
    # Train on codec embeddings (requires pre-trained codec)
    python train/train_stt.py \\
      --data_dir /path/to/audio \\
      --text_dir /path/to/text \\
      --codec_checkpoint train/checkpoints/codec/codec_best.pt \\
      --epochs 100

    # Train on mel spectrograms (early experiments)
    python train/train_stt.py \\
      --data_dir /path/to/audio \\
      --text_dir /path/to/text \\
      --mel_mode \\
      --epochs 100

    # Test mode with synthetic data
    python train/train_stt.py \\
      --data_dir /tmp/dummy \\
      --text_dir /tmp/dummy \\
      --synthetic \\
      --batch_size 4 \\
      --epochs 2
"""

import argparse
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
from typing import Optional, Tuple
import string
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Constants matching sonata-common and sonata-stt
SAMPLE_RATE = 24000
MEL_BINS = 80
FFT_SIZE = 1024
WINDOW_SIZE = 600   # 25ms at 24kHz
HOP_SIZE = 240      # 10ms at 24kHz
FMIN = 0.0
FMAX = 12000.0

STT_DIM = 512
STT_FFN_DIM = 2048
STT_LAYERS = 12
STT_HEADS = 8
TEXT_VOCAB_SIZE = 32000
BLANK_TOKEN = 0


# ============================================================================
# Mel Spectrogram (matches sonata-common/src/mel.rs)
# ============================================================================

class MelSpectrogram(nn.Module):
    """Mel spectrogram matching sonata-common/src/mel.rs.

    Uses STFT with:
    - FFT_SIZE=1024, WINDOW_SIZE=600 (25ms@24kHz), HOP_SIZE=240 (10ms@24kHz)
    - 80 mel bins, 0-12kHz range
    - Log floor = 1e-10
    """

    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=MEL_BINS,
                 n_fft=FFT_SIZE, win_length=WINDOW_SIZE,
                 hop_length=HOP_SIZE, f_min=FMIN, f_max=FMAX):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(win_length))

        # Build mel filterbank
        mel_fb = self._mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer('mel_fb', mel_fb)

    @staticmethod
    def _hz_to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _mel_filterbank(self, sr, n_fft, n_mels, f_min, f_max):
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = torch.tensor([self._mel_to_hz(m) for m in mel_points])
        bin_points = (hz_points * n_fft / sr).long()

        fb = torch.zeros(n_mels, n_fft // 2 + 1)
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            for j in range(left, center):
                if center > left:
                    fb[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    fb[i, j] = (right - j) / (right - center)
        return fb

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute log mel spectrogram.

        Args:
            audio: [B, samples] or [B, 1, samples]

        Returns:
            mel: [B, n_mels, T]
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        # Pad window to FFT size
        window = F.pad(self.window, (0, self.n_fft - self.window.shape[0]))

        # STFT
        spec = torch.stft(audio, self.n_fft, self.hop_length,
                          window=window, return_complex=True)
        power = spec.abs().pow(2)  # [B, freq, T]

        # Apply mel filterbank
        mel = torch.matmul(self.mel_fb, power)  # [B, n_mels, T]

        # Log with floor
        mel = torch.log(mel.clamp(min=1e-10))
        return mel


# ============================================================================
# Tokenizer
# ============================================================================

class CharacterTokenizer:
    """Character-level tokenizer matching Rust CTC vocab.

    Uses TEXT_VOCAB_SIZE=32000 to match sonata-stt/src/ctc.rs.
    In practice, only ~100 characters are used; the rest are reserved
    for subword tokens or future expansion.

    Token 0 = CTC blank (matches BLANK_TOKEN in Rust).
    """

    def __init__(self):
        # Token 0 = blank (CTC), then printable ASCII chars
        self.blank_id = BLANK_TOKEN
        chars = string.ascii_lowercase + string.digits + " .,?!'-"
        self.char2idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx2char = {i + 1: c for i, c in enumerate(chars)}
        self._vocab_size = TEXT_VOCAB_SIZE

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list:
        text = text.lower()
        tokens = []
        for c in text:
            if c in self.char2idx:
                tokens.append(self.char2idx[c])
        return tokens

    def decode(self, token_ids: list) -> str:
        text = []
        for tid in token_ids:
            if tid in self.idx2char:
                text.append(self.idx2char[tid])
        return ''.join(text)


# ============================================================================
# Dataset
# ============================================================================

class STTDataset(Dataset):
    """Speech-to-text dataset at 24kHz."""

    def __init__(self, audio_dir: str, text_dir: str, sample_rate: int = SAMPLE_RATE,
                 segment_length: int = SAMPLE_RATE * 4,  # 4 seconds
                 tokenizer: Optional[CharacterTokenizer] = None,
                 max_files: Optional[int] = None):
        import torchaudio
        self.audio_dir = Path(audio_dir)
        self.text_dir = Path(text_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.tokenizer = tokenizer or CharacterTokenizer()

        extensions = {'.wav', '.flac', '.mp3'}
        self.files = []
        for audio_file in sorted(self.audio_dir.rglob('*')):
            if audio_file.suffix.lower() in extensions:
                text_file = self.text_dir / (audio_file.stem + '.txt')
                if text_file.exists():
                    self.files.append((audio_file, text_file))

        if max_files:
            self.files = self.files[:max_files]
        logger.info(f"Found {len(self.files)} audio-text pairs")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import torchaudio
        import torchaudio.transforms as T

        audio_file, text_file = self.files[idx]

        try:
            audio, sr = torchaudio.load(str(audio_file))
            if sr != self.sample_rate:
                audio = T.Resample(sr, self.sample_rate)(audio)
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
        except Exception as e:
            logger.warning(f"Failed to load {audio_file}: {e}")
            audio = torch.zeros(1, self.segment_length)

        # Pad or truncate
        if audio.shape[1] < self.segment_length:
            audio = F.pad(audio, (0, self.segment_length - audio.shape[1]))
        else:
            audio = audio[:, :self.segment_length]

        try:
            with open(text_file, 'r') as f:
                text = f.read().strip()
            tokens = self.tokenizer.encode(text)
        except Exception as e:
            logger.warning(f"Failed to load {text_file}: {e}")
            tokens = []

        if not tokens:
            tokens = [1]  # At least one non-blank token

        tokens = torch.tensor(tokens, dtype=torch.long)
        return audio.squeeze(0), tokens  # [samples], [seq_len]


class SyntheticSTTDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(self, num_samples: int = 100, sample_rate: int = SAMPLE_RATE,
                 segment_length: int = SAMPLE_RATE * 4):
        self.num_samples = num_samples
        self.segment_length = segment_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio = torch.randn(self.segment_length)
        tokens = torch.randint(1, 40, (torch.randint(5, 20, (1,)).item(),))
        return audio, tokens


# ============================================================================
# Model Components (matches sonata-stt Rust crate)
# ============================================================================

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network matching sonata-common/src/swiglu.rs.

    gate_proj → Swish, up_proj, element-wise multiply, down_proj.

    Args:
        dim: Input/output dimension
        ffn_dim: Hidden dimension
    """

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, ffn_dim)
        self.up = nn.Linear(dim, ffn_dim)
        self.down = nn.Linear(ffn_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention matching sonata-stt/src/conformer.rs.

    Separate Q/K/V linear projections, scaled dot-product attention.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            x: [B, T, D]

        Returns:
            out: [B, T, D]
        """
        b, t, d = x.shape

        q = self.q_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.out_proj(out)


class ConformerBlock(nn.Module):
    """Conformer block matching sonata-stt/src/conformer.rs.

    Architecture:
    1. LayerNorm → MultiHeadSelfAttention → residual
    2. Conv1d(dim, dim, kernel=31, padding=15) → ReLU → residual
    3. LayerNorm → SwiGLU FFN → residual

    Args:
        dim: Model dimension
        ffn_dim: FFN hidden dimension
        num_heads: Attention heads
    """

    def __init__(self, dim: int = STT_DIM, ffn_dim: int = STT_FFN_DIM,
                 num_heads: int = STT_HEADS):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mhsa = MultiHeadSelfAttention(dim, num_heads)

        self.conv = nn.Conv1d(dim, dim, kernel_size=31, padding=15)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Conformer block.

        Args:
            x: [B, T, D]

        Returns:
            out: [B, T, D]
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        attn_out = self.mhsa(x)
        x = residual + attn_out

        # Conv with residual
        residual = x
        x_conv = x.transpose(1, 2)  # [B, D, T]
        conv_out = F.relu(self.conv(x_conv))
        x = residual + conv_out.transpose(1, 2)

        # SwiGLU FFN with residual
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        return residual + ffn_out


class SonataSTT(nn.Module):
    """Sonata STT model matching sonata-stt/src/lib.rs.

    Architecture:
        input_proj: Linear(input_dim→512)
        12 ConformerBlocks(512, 2048, 8)
        output_proj: Linear(512→32000)

    Args:
        input_dim: Input dimension (512 for codec embeddings, 80 for mel)
        model_dim: Conformer dimension (512)
        ffn_dim: FFN dimension (2048)
        num_layers: Number of Conformer blocks (12)
        num_heads: Attention heads (8)
        vocab_size: Output vocabulary (32000)
    """

    def __init__(self, input_dim: int = STT_DIM, model_dim: int = STT_DIM,
                 ffn_dim: int = STT_FFN_DIM, num_layers: int = STT_LAYERS,
                 num_heads: int = STT_HEADS, vocab_size: int = TEXT_VOCAB_SIZE):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(model_dim, ffn_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(model_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, input_dim] (mel or codec embeddings)

        Returns:
            logits: [B, T, vocab_size]
        """
        x = self.input_proj(x)  # [B, T, model_dim]
        for block in self.conformer_blocks:
            x = block(x)
        return self.output_proj(x)  # [B, T, vocab_size]


# ============================================================================
# CTC Decoding (matches sonata-stt/src/ctc.rs)
# ============================================================================

def ctc_greedy_decode(logits: torch.Tensor, tokenizer: CharacterTokenizer) -> list:
    """Greedy CTC decoding with blank removal and dedup.

    Args:
        logits: [B, T, vocab_size]
        tokenizer: Tokenizer for decoding

    Returns:
        texts: List of decoded strings
    """
    predictions = logits.argmax(dim=-1)  # [B, T]
    texts = []
    for pred in predictions:
        # Collapse repeats and remove blanks
        collapsed = []
        prev = -1
        for token_id in pred.tolist():
            if token_id != prev and token_id != BLANK_TOKEN:
                collapsed.append(token_id)
            prev = token_id
        texts.append(tokenizer.decode(collapsed))
    return texts


def compute_wer(predicted: str, reference: str) -> float:
    """Compute Word Error Rate."""
    pred_words = predicted.split()
    ref_words = reference.split()

    if not ref_words:
        return 0.0 if not pred_words else 1.0

    # Levenshtein distance at word level
    d = [[0] * (len(ref_words) + 1) for _ in range(len(pred_words) + 1)]
    for i in range(len(pred_words) + 1):
        d[i][0] = i
    for j in range(len(ref_words) + 1):
        d[0][j] = j

    for i in range(1, len(pred_words) + 1):
        for j in range(1, len(ref_words) + 1):
            if pred_words[i - 1] == ref_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1,      # deletion
                             d[i][j - 1] + 1,        # insertion
                             d[i - 1][j - 1] + 1)    # substitution

    return d[len(pred_words)][len(ref_words)] / len(ref_words)


# ============================================================================
# Collate Function
# ============================================================================

def collate_fn(batch):
    """Collate variable-length audio and tokens."""
    audios, tokens_list = zip(*batch)

    # Stack audio (same length due to padding in dataset)
    audios = torch.stack(audios, dim=0)

    # Pad tokens to max length
    max_token_len = max(t.shape[0] for t in tokens_list)
    padded_tokens = torch.zeros(len(tokens_list), max_token_len, dtype=torch.long)
    token_lengths = torch.zeros(len(tokens_list), dtype=torch.long)
    for i, t in enumerate(tokens_list):
        padded_tokens[i, :t.shape[0]] = t
        token_lengths[i] = t.shape[0]

    return audios, padded_tokens, token_lengths


# ============================================================================
# Training
# ============================================================================

def train_epoch(model: nn.Module, mel_extractor: MelSpectrogram,
                loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device) -> dict:
    """Train one epoch with CTC loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    with tqdm(loader, desc='Training') as pbar:
        for audio, tokens, token_lengths in pbar:
            audio = audio.to(device)
            tokens = tokens.to(device)
            token_lengths = token_lengths.to(device)

            optimizer.zero_grad()

            # Compute mel spectrogram: [B, samples] → [B, mel_bins, T]
            mel = mel_extractor(audio)
            mel = mel.permute(0, 2, 1)  # [B, T, mel_bins]

            # Forward
            logits = model(mel)  # [B, T, vocab_size]

            # CTC loss: needs [T, B, V]
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            input_lengths = torch.full((audio.shape[0],), log_probs.shape[0],
                                      device=device, dtype=torch.long)

            ctc_loss = F.ctc_loss(log_probs, tokens, input_lengths, token_lengths,
                                 blank=BLANK_TOKEN, reduction='mean')

            if torch.isfinite(ctc_loss):
                ctc_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += ctc_loss.item()
                num_batches += 1

            pbar.set_postfix({'loss': f'{ctc_loss.item():.4f}'})

    return {'loss': total_loss / max(num_batches, 1)}


@torch.no_grad()
def validate(model: nn.Module, mel_extractor: MelSpectrogram,
             loader: DataLoader, device: torch.device,
             tokenizer: CharacterTokenizer) -> dict:
    """Validate with WER computation."""
    model.eval()
    total_loss = 0.0
    total_wer = 0.0
    num_batches = 0
    num_utterances = 0

    for audio, tokens, token_lengths in tqdm(loader, desc='Validating'):
        audio = audio.to(device)
        tokens = tokens.to(device)
        token_lengths = token_lengths.to(device)

        mel = mel_extractor(audio)
        mel = mel.permute(0, 2, 1)

        logits = model(mel)

        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        input_lengths = torch.full((audio.shape[0],), log_probs.shape[0],
                                  device=device, dtype=torch.long)

        ctc_loss = F.ctc_loss(log_probs, tokens, input_lengths, token_lengths,
                             blank=BLANK_TOKEN, reduction='mean')

        if torch.isfinite(ctc_loss):
            total_loss += ctc_loss.item()
            num_batches += 1

        # WER computation
        predicted_texts = ctc_greedy_decode(logits, tokenizer)
        for i, pred_text in enumerate(predicted_texts):
            ref_tokens = tokens[i, :token_lengths[i]].tolist()
            ref_text = tokenizer.decode(ref_tokens)
            total_wer += compute_wer(pred_text, ref_text)
            num_utterances += 1

    return {
        'loss': total_loss / max(num_batches, 1),
        'wer': total_wer / max(num_utterances, 1),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Sonata STT')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--text_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--checkpoint_dir', type=str, default='train/checkpoints/stt')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--mel_mode', action='store_true',
                       help='Use mel spectrograms as input (input_dim=80)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = CharacterTokenizer()
    input_dim = MEL_BINS if args.mel_mode else STT_DIM

    # Data
    logger.info("Loading data...")
    if args.synthetic:
        full_dataset = SyntheticSTTDataset(num_samples=100)
    else:
        full_dataset = STTDataset(args.data_dir, args.text_dir, tokenizer=tokenizer,
                                  max_files=args.max_files)

    if len(full_dataset) == 0:
        logger.error("No data found!")
        return

    # Split train/val
    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers,
                             pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers,
                           pin_memory=True, collate_fn=collate_fn)

    # Model
    logger.info(f"Creating STT model (dim={STT_DIM}, layers={STT_LAYERS}, "
                f"heads={STT_HEADS}, vocab={TEXT_VOCAB_SIZE}, input_dim={input_dim})...")
    model = SonataSTT(input_dim=input_dim).to(device)
    mel_extractor = MelSpectrogram().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_loss = float('inf')

    # Resume
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('loss', float('inf'))

    # Training
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_metrics = train_epoch(model, mel_extractor, train_loader, optimizer, device)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train loss: {train_metrics['loss']:.4f}")

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_metrics = validate(model, mel_extractor, val_loader, device, tokenizer)
            logger.info(f"  Val loss: {val_metrics['loss']:.4f} | "
                       f"WER: {val_metrics['wer']:.4f} ({val_metrics['wer']*100:.1f}%)")

            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                ckpt_path = Path(args.checkpoint_dir) / 'stt_best.pt'
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'loss': best_loss,
                    'wer': val_metrics['wer'],
                    'config': {
                        'input_dim': input_dim,
                        'model_dim': STT_DIM,
                        'ffn_dim': STT_FFN_DIM,
                        'num_layers': STT_LAYERS,
                        'num_heads': STT_HEADS,
                        'vocab_size': TEXT_VOCAB_SIZE,
                    },
                }, ckpt_path)
                logger.info(f"  Saved best: {ckpt_path}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = Path(args.checkpoint_dir) / f'stt_epoch{epoch+1}.pt'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
            }, ckpt_path)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        logger.info(f"  Time: {epoch_time:.1f}s")

    logger.info(f"Training complete! Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
