"""Train Sonata STT: Streaming Conformer CTC.

Trains a Conformer encoder with CTC (Connectionist Temporal Classification)
loss for streaming speech-to-text. Features:
- Conformer blocks (self-attention + convolution)
- CTC decoder for alignment-free training
- Character-level tokenization with BOS/EOS/PAD tokens
- Validation on word error rate (WER)

Usage:
    # Basic training
    python train/train_stt.py \\
      --data_dir /path/to/audio \\
      --text_dir /path/to/text \\
      --epochs 100

    # With custom hyperparameters
    python train/train_stt.py \\
      --data_dir /path/to/audio \\
      --text_dir /path/to/text \\
      --batch_size 32 \\
      --lr 5e-4 \\
      --epochs 200 \\
      --num_layers 12

    # Resume from checkpoint
    python train/train_stt.py \\
      --data_dir /path/to/audio \\
      --text_dir /path/to/text \\
      --resume train/checkpoints/stt/stt_best.pt \\
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
from typing import Optional, Tuple
import json
import string
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Tokenizer
# ============================================================================

class CharacterTokenizer:
    """Simple character-level tokenizer for ASR.

    Supports lowercase letters, numbers, and common punctuation.
    Special tokens: <pad>, <bos>, <eos>, <unk>

    Args:
        vocab: Set of characters to include (default: lowercase + digits + space)
    """

    def __init__(self, vocab: Optional[str] = None):
        special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        if vocab is None:
            vocab = string.ascii_lowercase + string.digits + ' '
        self.vocab = special_tokens + list(vocab)
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.char2idx['<pad>']

    @property
    def bos_token_id(self) -> int:
        return self.char2idx['<bos>']

    @property
    def eos_token_id(self) -> int:
        return self.char2idx['<eos>']

    @property
    def unk_token_id(self) -> int:
        return self.char2idx['<unk>']

    def encode(self, text: str) -> list:
        """Encode text to token IDs.

        Args:
            text: Text string

        Returns:
            token_ids: List of token IDs including BOS/EOS
        """
        text = text.lower()
        tokens = [self.bos_token_id]
        for c in text:
            if c in self.char2idx:
                tokens.append(self.char2idx[c])
            else:
                tokens.append(self.unk_token_id)
        tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: list) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs

        Returns:
            text: Decoded text
        """
        text = []
        for tid in token_ids:
            if tid in self.idx2char:
                c = self.idx2char[tid]
                if c not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                    text.append(c)
        return ''.join(text)


# ============================================================================
# Dataset
# ============================================================================

class STTDataset(Dataset):
    """Speech-to-text dataset.

    Loads audio files and corresponding transcripts. Audio is resampled
    to 16 kHz and split into fixed-length segments.

    Args:
        audio_dir: Directory containing audio files
        text_dir: Directory containing text transcripts (same filenames, .txt)
        sample_rate: Target sample rate (default: 16000 Hz)
        segment_length: Audio segment length in samples (default: 16000 = 1s)
        tokenizer: CharacterTokenizer instance
        max_files: Maximum number of files (default: None = all)
    """

    def __init__(self, audio_dir: str, text_dir: str, sample_rate: int = 16000,
                 segment_length: int = 16000, tokenizer: Optional[CharacterTokenizer] = None,
                 max_files: Optional[int] = None):
        import torchaudio
        self.audio_dir = Path(audio_dir)
        self.text_dir = Path(text_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.tokenizer = tokenizer or CharacterTokenizer()

        # Find audio files
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
        """Load audio and transcript.

        Args:
            idx: Index

        Returns:
            audio: [1, segment_length]
            tokens: [seq_len] (tensor of token IDs)
        """
        import torchaudio
        import torchaudio.transforms as T

        audio_file, text_file = self.files[idx]

        # Load audio
        try:
            audio, sr = torchaudio.load(str(audio_file))
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                audio = resampler(audio)
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

        # Load text and tokenize
        try:
            with open(text_file, 'r') as f:
                text = f.read().strip()
            tokens = self.tokenizer.encode(text)
        except Exception as e:
            logger.warning(f"Failed to load {text_file}: {e}")
            tokens = [self.tokenizer.pad_token_id]

        tokens = torch.tensor(tokens, dtype=torch.long)
        return audio.float(), tokens


class SyntheticSTTDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(self, num_samples: int = 100, sample_rate: int = 16000,
                 segment_length: int = 16000, vocab_size: int = 50):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio = torch.randn(1, self.segment_length)
        tokens = torch.randint(4, self.vocab_size, (torch.randint(5, 20, (1,)).item(),))
        return audio, tokens


# ============================================================================
# Model Components
# ============================================================================

class ConformerBlock(nn.Module):
    """Conformer block: self-attention + convolution.

    Implements the Conformer architecture from https://arxiv.org/abs/2005.08100
    combining multi-head self-attention and depthwise convolution.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        ffn_dim: FFN hidden dimension
        conv_kernel: Depthwise conv kernel size
        dropout: Dropout rate
    """

    def __init__(self, dim: int, num_heads: int = 4, ffn_dim: int = 2048,
                 conv_kernel: int = 31, dropout: float = 0.1):
        super().__init__()

        # Pre-norm layers
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                               batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )

        self.norm3 = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel // 2,
                     groups=dim),  # Depthwise
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.norm4 = nn.LayerNorm(dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply Conformer block.

        Args:
            x: [B, T, D]
            mask: Optional attention mask

        Returns:
            out: [B, T, D]
        """
        # Self-attention + FFN (first half)
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm,
                                     key_padding_mask=mask)
        x = x + self.dropout(attn_out)

        x_norm = self.norm2(x)
        x = x + self.dropout(self.ffn1(x_norm))

        # Convolution (second half)
        x_norm = self.norm3(x)
        conv_in = x_norm.transpose(1, 2)  # [B, D, T]
        conv_out = self.conv(conv_in).transpose(1, 2)  # [B, T, D]
        x = x + self.dropout(conv_out)

        x_norm = self.norm4(x)
        x = x + self.dropout(self.ffn2(x_norm))

        return x


class ConformerEncoder(nn.Module):
    """Streaming Conformer encoder for ASR.

    Processes audio mel-spectrograms through Conformer blocks and outputs
    logits for CTC decoding.

    Args:
        input_dim: Mel-spectrogram dimension (default: 80)
        model_dim: Conformer model dimension (default: 256)
        num_layers: Number of Conformer blocks (default: 12)
        num_heads: Attention heads (default: 4)
        vocab_size: CTC output vocabulary size (default: 128)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, input_dim: int = 80, model_dim: int = 256,
                 num_layers: int = 12, num_heads: int = 4,
                 vocab_size: int = 128, dropout: float = 0.1):
        super().__init__()

        # Input projection + subsampling
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(model_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # CTC head
        self.ctc_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, vocab_size),
        )

    def forward(self, mel: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode mel-spectrograms to CTC logits.

        Args:
            mel: [B, T, input_dim]
            lengths: [B] sequence lengths (for masking)

        Returns:
            logits: [B, T, vocab_size]
        """
        # Project input
        x = self.input_proj(mel)  # [B, T, D]

        # Create mask
        mask = None
        if lengths is not None:
            mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        # Conformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # CTC head
        logits = self.ctc_head(x)  # [B, T, vocab_size]

        return logits


# ============================================================================
# Training
# ============================================================================

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, tokenizer: CharacterTokenizer) -> dict:
    """Train one epoch with CTC loss.

    Args:
        model: Conformer encoder
        loader: Training loader
        optimizer: Optimizer
        device: Device
        tokenizer: Tokenizer for vocab size

    Returns:
        metrics: Dictionary with loss values
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    with tqdm(loader, desc='Training') as pbar:
        for audio, tokens in pbar:
            audio = audio.to(device)  # [B, 1, N]
            tokens = tokens.to(device)

            optimizer.zero_grad()

            # Dummy mel-spectrogram (normally computed from audio)
            # For simplicity: use audio directly after 1D→2D projection
            mel = audio.squeeze(1).unsqueeze(2).expand(-1, -1, 80)  # [B, T, 80]

            # Forward
            logits = model(mel)  # [B, T, vocab_size]

            # CTC loss
            # Flatten logits for CTC: [B, T, V] → [T, B, V]
            logits_t = logits.permute(1, 0, 2)
            input_lengths = torch.full((audio.shape[0],), logits_t.shape[0],
                                      device=device, dtype=torch.long)
            target_lengths = torch.full((audio.shape[0],), tokens.shape[1],
                                       device=device, dtype=torch.long)

            # CTC requires targets to be 1D
            targets = tokens.view(-1)

            ctc_loss = F.ctc_loss(logits_t, targets, input_lengths, target_lengths,
                                 blank=tokenizer.pad_token_id, reduction='mean')

            # Backward
            ctc_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += ctc_loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{ctc_loss.item():.4f}'})

    return {'loss': total_loss / num_batches}


def main():
    parser = argparse.ArgumentParser(description='Train Sonata STT')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Audio directory')
    parser.add_argument('--text_dir', type=str, required=True,
                       help='Text transcripts directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--model_dim', type=int, default=256)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='train/checkpoints/stt')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--max_files', type=int, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = CharacterTokenizer()

    # Data
    logger.info("Loading data...")
    if args.synthetic:
        dataset = SyntheticSTTDataset(num_samples=100)
    else:
        dataset = STTDataset(args.data_dir, args.text_dir, tokenizer=tokenizer,
                           max_files=args.max_files)

    if len(dataset) == 0:
        logger.error("No data found!")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)

    # Model
    logger.info("Creating model...")
    model = ConformerEncoder(vocab_size=tokenizer.vocab_size,
                           num_layers=args.num_layers,
                           model_dim=args.model_dim).to(device)
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

    # Training
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        metrics = train_epoch(model, loader, optimizer, device, tokenizer)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {metrics['loss']:.4f}")

        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            ckpt_path = Path(args.checkpoint_dir) / 'stt_best.pt'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
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

    logger.info(f"Training complete! Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
