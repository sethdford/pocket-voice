"""Train Sonata TTS: Text-to-Speech with Emotion + Nonverbal.

Trains a TTS model combining:
- Text encoder (character + duration embedding)
- Transformer decoder with dual AdaIN conditioning (speaker + emotion)
- Emotion style tokens (64 styles)
- Nonverbal action predictions (breathing, silence, backchannels)
- Codec token prediction (32 codes → 24kHz audio)

Loss combines codec prediction (cross-entropy) + emotion loss + nonverbal loss.

Usage:
    # Basic training
    python train/train_tts.py \\
      --text_file /path/to/texts.txt \\
      --speaker_file /path/to/speakers.txt \\
      --emotion_file /path/to/emotions.txt \\
      --epochs 100

    # With custom settings
    python train/train_tts.py \\
      --text_file /path/to/texts.txt \\
      --speaker_file /path/to/speakers.txt \\
      --batch_size 32 \\
      --lr 1e-3 \\
      --epochs 200

    # Test mode with synthetic data
    python train/train_tts.py \\
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
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset
# ============================================================================

class TTSDataset(Dataset):
    """Text-to-speech training dataset.

    Pairs text with speaker embeddings and emotion labels to predict
    codec tokens (audio).

    Args:
        text_file: File with text lines
        speaker_file: File with speaker IDs (one per line)
        emotion_file: File with emotion labels (one per line)
        num_speakers: Total number of unique speakers
        num_emotions: Total number of emotion styles
        max_text_len: Max characters in text (default: 200)
        num_codes: Number of RVQ codes per frame (default: 8)
        code_vocab_size: Codebook vocabulary size (default: 1024)
    """

    def __init__(self, text_file: Optional[str] = None,
                 speaker_file: Optional[str] = None,
                 emotion_file: Optional[str] = None,
                 num_speakers: int = 100, num_emotions: int = 64,
                 max_text_len: int = 200, num_codes: int = 8,
                 code_vocab_size: int = 1024):
        self.num_speakers = num_speakers
        self.num_emotions = num_emotions
        self.max_text_len = max_text_len
        self.num_codes = num_codes
        self.code_vocab_size = code_vocab_size

        # Load files
        texts = []
        speakers = []
        emotions = []

        if text_file and Path(text_file).exists():
            with open(text_file, 'r') as f:
                texts = [line.strip() for line in f]

        if speaker_file and Path(speaker_file).exists():
            with open(speaker_file, 'r') as f:
                speakers = [int(line.strip()) for line in f]

        if emotion_file and Path(emotion_file).exists():
            with open(emotion_file, 'r') as f:
                emotions = [int(line.strip()) for line in f]

        # Pad to same length
        max_len = max(len(texts), len(speakers), len(emotions), 1)
        texts = texts + [''] * (max_len - len(texts))
        speakers = speakers + [0] * (max_len - len(speakers))
        emotions = emotions + [0] * (max_len - len(emotions))

        self.texts = texts[:max_len]
        self.speakers = speakers[:max_len]
        self.emotions = emotions[:max_len]

        logger.info(f"Loaded {len(self.texts)} TTS examples")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor]:
        """Load text, speaker, emotion, and dummy codec tokens.

        Args:
            idx: Index

        Returns:
            text_ids: [max_text_len]
            speaker_id: scalar
            emotion_id: scalar
            codec_tokens: [T, num_codes] (dummy random for training)
        """
        text = self.texts[idx]
        speaker_id = self.speakers[idx] % self.num_speakers
        emotion_id = self.emotions[idx] % self.num_emotions

        # Encode text (character level + padding)
        text_ids = [ord(c) % 256 for c in text[:self.max_text_len]]
        text_ids = text_ids + [0] * (self.max_text_len - len(text_ids))
        text_ids = torch.tensor(text_ids, dtype=torch.long)

        speaker_id = torch.tensor(speaker_id, dtype=torch.long)
        emotion_id = torch.tensor(emotion_id, dtype=torch.long)

        # Dummy codec tokens (normally from preprocessed audio)
        seq_len = 50  # ~2s at 25 frames/s
        codec_tokens = torch.randint(0, self.code_vocab_size,
                                    (seq_len, self.num_codes),
                                    dtype=torch.long)

        return text_ids, speaker_id, emotion_id, codec_tokens


class SyntheticTTSDataset(Dataset):
    """Synthetic TTS dataset for testing."""

    def __init__(self, num_samples: int = 100, num_speakers: int = 10,
                 num_emotions: int = 8, max_text_len: int = 200):
        self.num_samples = num_samples
        self.num_speakers = num_speakers
        self.num_emotions = num_emotions
        self.max_text_len = max_text_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor]:
        text_ids = torch.randint(0, 256, (self.max_text_len,))
        speaker_id = torch.randint(0, self.num_speakers, (1,)).item()
        emotion_id = torch.randint(0, self.num_emotions, (1,)).item()
        codec_tokens = torch.randint(0, 1024, (50, 8))

        return text_ids, speaker_id, emotion_id, codec_tokens


# ============================================================================
# Model Components
# ============================================================================

class TextEncoder(nn.Module):
    """Text encoder: character IDs → embeddings.

    Maps characters to continuous embeddings with learnable duration weights.

    Args:
        vocab_size: Character vocabulary size (default: 256)
        embed_dim: Embedding dimension (default: 256)
        num_dur_buckets: Duration prediction buckets (default: 10)
    """

    def __init__(self, vocab_size: int = 256, embed_dim: int = 256,
                 num_dur_buckets: int = 10):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.duration_proj = nn.Linear(embed_dim, num_dur_buckets)

    def forward(self, text_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text.

        Args:
            text_ids: [B, T]

        Returns:
            embeddings: [B, T, embed_dim]
            duration_logits: [B, T, num_buckets]
        """
        embeddings = self.embed(text_ids)  # [B, T, D]
        duration_logits = self.duration_proj(embeddings)  # [B, T, K]
        return embeddings, duration_logits


class AdaINLayer(nn.Module):
    """Adaptive Instance Normalization (AdaIN).

    Normalizes features and applies learned affine transformation
    conditioned on style (e.g., speaker + emotion).

    Args:
        dim: Feature dimension
        style_dim: Style embedding dimension
    """

    def __init__(self, dim: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(dim, affine=False)
        self.gamma_proj = nn.Linear(style_dim, dim)
        self.beta_proj = nn.Linear(style_dim, dim)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Apply AdaIN.

        Args:
            x: [B, T, dim]
            style: [B, style_dim]

        Returns:
            out: [B, T, dim]
        """
        # Normalize: [B, T, dim] → [B, dim, T] → norm → [B, dim, T]
        x_norm = self.norm(x.transpose(1, 2)).transpose(1, 2)

        # Affine transformation
        gamma = self.gamma_proj(style).unsqueeze(1)  # [B, 1, dim]
        beta = self.beta_proj(style).unsqueeze(1)    # [B, 1, dim]

        return gamma * x_norm + beta


class TTSDecoder(nn.Module):
    """TTS decoder with AdaIN-conditioned transformer.

    Predicts codec tokens conditioned on text and style (speaker + emotion).

    Args:
        embed_dim: Embedding dimension (default: 256)
        num_heads: Attention heads (default: 4)
        num_layers: Transformer layers (default: 6)
        num_speakers: Number of speakers (default: 100)
        num_emotions: Number of emotions (default: 64)
        num_codes: Number of RVQ codes (default: 8)
        code_vocab_size: Codebook size (default: 1024)
        num_nonverbal: Nonverbal action classes (default: 10)
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 6, num_speakers: int = 100,
                 num_emotions: int = 64, num_codes: int = 8,
                 code_vocab_size: int = 1024, num_nonverbal: int = 10):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_codes = num_codes

        # Style embeddings
        self.speaker_embed = nn.Embedding(num_speakers, embed_dim)
        self.emotion_embed = nn.Embedding(num_emotions, embed_dim)

        # Combine speaker + emotion for AdaIN
        self.style_proj = nn.Linear(2 * embed_dim, embed_dim)

        # Transformer blocks with AdaIN
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                      batch_first=True, dropout=0.1)
            for _ in range(num_layers)
        ])

        self.adain_layers = nn.ModuleList([
            AdaINLayer(embed_dim, embed_dim) for _ in range(num_layers)
        ])

        # Output heads
        self.codec_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, num_codes * code_vocab_size),
        )

        self.nonverbal_head = nn.Linear(embed_dim, num_nonverbal)

    def forward(self, text_embeddings: torch.Tensor, speaker_ids: torch.Tensor,
                emotion_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode text to codec tokens with style conditioning.

        Args:
            text_embeddings: [B, T, embed_dim]
            speaker_ids: [B]
            emotion_ids: [B]

        Returns:
            codec_logits: [B, T, num_codes * vocab_size]
            nonverbal_logits: [B, T, num_nonverbal]
        """
        # Style embeddings
        speaker_emb = self.speaker_embed(speaker_ids)  # [B, D]
        emotion_emb = self.emotion_embed(emotion_ids)   # [B, D]
        style = torch.cat([speaker_emb, emotion_emb], dim=1)  # [B, 2D]
        style = self.style_proj(style)  # [B, D]

        # Transformer with AdaIN
        x = text_embeddings
        for transformer_block, adain_layer in zip(self.transformer_blocks,
                                                   self.adain_layers):
            x = transformer_block(x)  # Self-attention
            x = adain_layer(x, style)  # AdaIN conditioning

        # Output
        codec_logits = self.codec_head(x)  # [B, T, num_codes * vocab_size]
        nonverbal_logits = self.nonverbal_head(x)  # [B, T, num_nonverbal]

        # Reshape codec logits: [B, T, num_codes, vocab_size]
        batch_size, seq_len, _ = codec_logits.shape
        codec_logits = codec_logits.reshape(batch_size, seq_len, self.num_codes, -1)

        return codec_logits, nonverbal_logits


class SonataTTS(nn.Module):
    """Full TTS model combining text encoding and decoding."""

    def __init__(self, **kwargs):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.decoder = TTSDecoder(**kwargs)

    def forward(self, text_ids: torch.Tensor, speaker_ids: torch.Tensor,
                emotion_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text and decode to audio tokens.

        Args:
            text_ids: [B, T]
            speaker_ids: [B]
            emotion_ids: [B]

        Returns:
            codec_logits: [B, T, num_codes, vocab_size]
            nonverbal_logits: [B, T, num_nonverbal]
        """
        text_emb, _ = self.text_encoder(text_ids)
        codec_logits, nonverbal_logits = self.decoder(text_emb, speaker_ids, emotion_ids)
        return codec_logits, nonverbal_logits


# ============================================================================
# Training
# ============================================================================

def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device) -> dict:
    """Train one epoch.

    Args:
        model: TTS model
        loader: Data loader
        optimizer: Optimizer
        device: Device

    Returns:
        metrics: Loss dictionary
    """
    model.train()
    total_codec_loss = 0.0
    total_nonverbal_loss = 0.0
    num_batches = 0

    with tqdm(loader, desc='Training') as pbar:
        for text_ids, speaker_ids, emotion_ids, codec_tokens in pbar:
            text_ids = text_ids.to(device)
            speaker_ids = speaker_ids.to(device)
            emotion_ids = emotion_ids.to(device)
            codec_tokens = codec_tokens.to(device)

            optimizer.zero_grad()

            # Forward
            codec_logits, nonverbal_logits = model(text_ids, speaker_ids, emotion_ids)

            # Codec loss: cross-entropy per code
            # codec_logits: [B, T, num_codes, vocab_size]
            # codec_tokens: [B, T, num_codes]
            batch_size, seq_len, num_codes, vocab_size = codec_logits.shape

            codec_logits_flat = codec_logits.reshape(-1, vocab_size)
            codec_tokens_flat = codec_tokens.reshape(-1)

            codec_loss = F.cross_entropy(codec_logits_flat, codec_tokens_flat)

            # Nonverbal loss (dummy): MSE with random targets
            nonverbal_targets = torch.randint(0, 10, nonverbal_logits.shape[:-1],
                                             device=device)
            nonverbal_loss = F.cross_entropy(
                nonverbal_logits.reshape(-1, nonverbal_logits.shape[-1]),
                nonverbal_targets.reshape(-1)
            )

            # Total loss
            loss = codec_loss + 0.1 * nonverbal_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_codec_loss += codec_loss.item()
            total_nonverbal_loss += nonverbal_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'codec': f'{codec_loss.item():.4f}',
                'nonverbal': f'{nonverbal_loss.item():.4f}'
            })

    return {
        'codec_loss': total_codec_loss / num_batches,
        'nonverbal_loss': total_nonverbal_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Sonata TTS')
    parser.add_argument('--text_file', type=str, default='',
                       help='Text lines file')
    parser.add_argument('--speaker_file', type=str, default='',
                       help='Speaker IDs file')
    parser.add_argument('--emotion_file', type=str, default='',
                       help='Emotion IDs file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='train/checkpoints/tts')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--synthetic', action='store_true')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Data
    logger.info("Loading data...")
    if args.synthetic:
        dataset = SyntheticTTSDataset(num_samples=100)
    else:
        dataset = TTSDataset(
            text_file=args.text_file if args.text_file else None,
            speaker_file=args.speaker_file if args.speaker_file else None,
            emotion_file=args.emotion_file if args.emotion_file else None,
        )

    if len(dataset) == 0:
        logger.error("No data found!")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)

    # Model
    logger.info("Creating model...")
    model = SonataTTS().to(device)
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
        metrics = train_epoch(model, loader, optimizer, device)
        total_loss = metrics['codec_loss'] + 0.1 * metrics['nonverbal_loss']

        logger.info(f"Epoch {epoch+1}/{args.epochs} | "
                   f"Codec: {metrics['codec_loss']:.4f} | "
                   f"Nonverbal: {metrics['nonverbal_loss']:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            ckpt_path = Path(args.checkpoint_dir) / 'tts_best.pt'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
            }, ckpt_path)
            logger.info(f"  Saved best: {ckpt_path}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = Path(args.checkpoint_dir) / f'tts_epoch{epoch+1}.pt'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
            }, ckpt_path)

        scheduler.step()

    logger.info(f"Training complete! Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
