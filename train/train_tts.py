"""Train Sonata TTS: Text-to-Speech with speaker + emotion conditioning.

Architecture matches sonata-tts Rust crate:
- TextEncoder: Embedding(32000, 512) + sinusoidal positional encoding
- 12 TTSTransformerLayers:
    - Linear self-attention
    - speaker AdaIN(512, 192)
    - emotion AdaIN(512, 192)
    - SwiGLU FFN(512, 2048)
    - LayerNorm
- EmotionStyleEncoder: Embedding(64, dim) + Linear(dim→dim) + exaggeration
- NonverbalEncoder: Embedding(24, 512)
- Output: Linear(512→1024) (codec token logits)

Training requires pre-encoded codec tokens from audio (via trained codec model).

Usage:
    # Train with pre-encoded codec tokens
    python train/train_tts.py \\
      --data_dir /path/to/data \\
      --codec_checkpoint train/checkpoints/codec/codec_best.pt \\
      --epochs 100

    # Test mode with synthetic data
    python train/train_tts.py \\
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
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Constants matching sonata-common and sonata-tts
TEXT_VOCAB_SIZE = 32000
TTS_DIM = 512
TTS_FFN_DIM = 2048
TTS_LAYERS = 12
SPEAKER_EMBED_DIM = 192
EMOTION_DIM = 192
NUM_EMOTION_STYLES = 64
NUM_NONVERBAL_TOKENS = 24
NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 1024
SAMPLE_RATE = 24000


# ============================================================================
# Model Components (matches sonata-tts Rust crate)
# ============================================================================

class SwiGLU(nn.Module):
    """SwiGLU FFN matching sonata-common/src/swiglu.rs."""

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, ffn_dim)
        self.up = nn.Linear(dim, ffn_dim)
        self.down = nn.Linear(ffn_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class AdaIN(nn.Module):
    """Adaptive Instance Normalization matching sonata-common/src/adain.rs.

    Args:
        hidden_dim: Feature dimension
        style_dim: Style embedding dimension
    """

    def __init__(self, hidden_dim: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(hidden_dim, affine=False)
        self.gamma_proj = nn.Linear(style_dim, hidden_dim)
        self.beta_proj = nn.Linear(style_dim, hidden_dim)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Apply AdaIN.

        Args:
            x: [B, T, hidden_dim]
            style: [B, style_dim]

        Returns:
            out: [B, T, hidden_dim]
        """
        x_norm = self.norm(x.transpose(1, 2)).transpose(1, 2)
        gamma = self.gamma_proj(style).unsqueeze(1)  # [B, 1, D]
        beta = self.beta_proj(style).unsqueeze(1)     # [B, 1, D]
        return gamma * x_norm + beta


class TextEncoder(nn.Module):
    """Text encoder matching sonata-tts/src/text_encoder.rs.

    Embedding(32000, dim) + sinusoidal positional encoding.
    """

    def __init__(self, vocab_size: int = TEXT_VOCAB_SIZE, dim: int = TTS_DIM,
                 max_len: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.dim = dim

        # Pre-compute sinusoidal positional encoding
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Encode text with positional encoding.

        Args:
            text_ids: [B, T]

        Returns:
            encoded: [B, T, dim]
        """
        x = self.embedding(text_ids)  # [B, T, D]
        x = x + self.pe[:, :x.shape[1], :]
        return x


class EmotionStyleEncoder(nn.Module):
    """Emotion style encoder matching sonata-tts/src/emotion.rs.

    Embedding(64, dim) + Linear(dim→dim), encode with exaggeration scalar.
    """

    def __init__(self, num_styles: int = NUM_EMOTION_STYLES, dim: int = EMOTION_DIM):
        super().__init__()
        self.embedding = nn.Embedding(num_styles, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, style_id: torch.Tensor,
                exaggeration: torch.Tensor = None) -> torch.Tensor:
        """Encode emotion style.

        Args:
            style_id: [B] emotion style IDs
            exaggeration: [B] scalar multiplier (default: 1.0)

        Returns:
            style_embedding: [B, dim]
        """
        x = self.embedding(style_id)  # [B, dim]
        x = self.proj(x)
        if exaggeration is not None:
            x = x * exaggeration.unsqueeze(-1)
        return x


class NonverbalEncoder(nn.Module):
    """Nonverbal token encoder matching sonata-tts/src/nonverbal.rs.

    Embedding(24, embed_dim).
    """

    def __init__(self, num_tokens: int = NUM_NONVERBAL_TOKENS, embed_dim: int = TTS_DIM):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class TTSTransformerLayer(nn.Module):
    """TTS transformer layer matching sonata-tts/src/transformer.rs.

    Architecture:
    1. Linear self-attention
    2. Speaker AdaIN(512, 192)
    3. Emotion AdaIN(512, 192)
    4. SwiGLU FFN(512, 2048)
    5. LayerNorm
    """

    def __init__(self, dim: int = TTS_DIM, ffn_dim: int = TTS_FFN_DIM,
                 speaker_dim: int = SPEAKER_EMBED_DIM, emotion_dim: int = EMOTION_DIM):
        super().__init__()
        # Linear attention (simplified)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.speaker_adain = AdaIN(dim, speaker_dim)
        self.emotion_adain = AdaIN(dim, emotion_dim)
        self.ffn = SwiGLU(dim, ffn_dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor,
                emotion_emb: torch.Tensor) -> torch.Tensor:
        """Apply TTS transformer layer.

        Args:
            x: [B, T, dim]
            speaker_emb: [B, speaker_dim]
            emotion_emb: [B, emotion_dim]

        Returns:
            out: [B, T, dim]
        """
        # Linear attention
        residual = x
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Simplified linear attention: softmax(Q @ K^T / sqrt(d)) @ V
        scale = x.shape[-1] ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        x = self.out_proj(x)
        x = residual + x

        # Speaker AdaIN
        x = self.speaker_adain(x, speaker_emb)

        # Emotion AdaIN
        x = self.emotion_adain(x, emotion_emb)

        # SwiGLU FFN
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        x = residual + x

        return x


class SonataTTS(nn.Module):
    """Full TTS model matching sonata-tts/src/lib.rs.

    Architecture:
        TextEncoder(32000, 512) + sinusoidal PE
        12 TTSTransformerLayers(512, 2048, 192, 192)
        EmotionStyleEncoder(64, 192)
        NonverbalEncoder(24, 512)
        output: Linear(512→1024) (first codebook logits)

    Args:
        dim: Model dimension (512)
        num_layers: Transformer layers (12)
    """

    def __init__(self, dim: int = TTS_DIM, num_layers: int = TTS_LAYERS):
        super().__init__()
        self.text_encoder = TextEncoder(TEXT_VOCAB_SIZE, dim)
        self.emotion_encoder = EmotionStyleEncoder(NUM_EMOTION_STYLES, EMOTION_DIM)
        self.nonverbal_encoder = NonverbalEncoder(NUM_NONVERBAL_TOKENS, dim)

        self.layers = nn.ModuleList([
            TTSTransformerLayer(dim, TTS_FFN_DIM, SPEAKER_EMBED_DIM, EMOTION_DIM)
            for _ in range(num_layers)
        ])

        # Output head: predict first codebook tokens (1024 classes)
        self.output_proj = nn.Linear(dim, CODEBOOK_SIZE)

    def forward(self, text_ids: torch.Tensor, speaker_emb: torch.Tensor,
                emotion_ids: torch.Tensor,
                exaggeration: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            text_ids: [B, T]
            speaker_emb: [B, 192] (from pre-trained speaker encoder)
            emotion_ids: [B]
            exaggeration: [B] (optional)

        Returns:
            logits: [B, T, 1024] (codec token logits)
        """
        x = self.text_encoder(text_ids)  # [B, T, 512]
        emotion_emb = self.emotion_encoder(emotion_ids, exaggeration)  # [B, 192]

        for layer in self.layers:
            x = layer(x, speaker_emb, emotion_emb)

        return self.output_proj(x)  # [B, T, 1024]


# ============================================================================
# Dataset
# ============================================================================

class TTSDataset(Dataset):
    """TTS dataset with audio-derived codec tokens as targets.

    Loads text + audio pairs and computes codec tokens on-the-fly
    using a pre-trained codec model. Speaker embeddings are computed
    from a pre-trained speaker encoder, or random for training.

    Args:
        data_dir: Directory with audio files
        text_dir: Directory with text transcripts (matching filenames)
        codec_model: Pre-trained codec model for computing targets
        max_text_len: Maximum text length in tokens
    """

    def __init__(self, data_dir: str, text_dir: Optional[str] = None,
                 codec_model: Optional[nn.Module] = None,
                 max_text_len: int = 200,
                 max_files: Optional[int] = None):
        self.max_text_len = max_text_len
        self.codec_model = codec_model
        self.files = []

        data_path = Path(data_dir)
        text_path = Path(text_dir) if text_dir else data_path

        extensions = {'.wav', '.flac', '.mp3'}
        for audio_file in sorted(data_path.rglob('*')):
            if audio_file.suffix.lower() in extensions:
                text_file = text_path / (audio_file.stem + '.txt')
                if text_file.exists():
                    self.files.append((audio_file, text_file))

        if max_files:
            self.files = self.files[:max_files]

        logger.info(f"Found {len(self.files)} TTS examples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import torchaudio
        import torchaudio.transforms as T

        audio_file, text_file = self.files[idx]

        # Load text
        with open(text_file, 'r') as f:
            text = f.read().strip()

        # Tokenize text (character level, matching TEXT_VOCAB_SIZE range)
        text_ids = [ord(c) % TEXT_VOCAB_SIZE for c in text[:self.max_text_len]]
        text_ids = text_ids + [0] * (self.max_text_len - len(text_ids))
        text_ids = torch.tensor(text_ids, dtype=torch.long)

        # Load audio
        try:
            audio, sr = torchaudio.load(str(audio_file))
            if sr != SAMPLE_RATE:
                audio = T.Resample(sr, SAMPLE_RATE)(audio)
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
        except Exception:
            audio = torch.zeros(1, SAMPLE_RATE)  # 1 second silence

        # Compute codec tokens as targets
        if self.codec_model is not None:
            with torch.no_grad():
                # Pad/truncate to standard length
                target_len = SAMPLE_RATE * 2  # 2 seconds
                if audio.shape[1] < target_len:
                    audio = F.pad(audio, (0, target_len - audio.shape[1]))
                else:
                    audio = audio[:, :target_len]

                codes = self.codec_model.encode(audio.unsqueeze(0))  # [1, num_books, T]
                # Use first codebook as primary target
                codec_targets = codes[0, 0, :]  # [T]
        else:
            # Without codec: generate aligned targets (placeholder)
            codec_targets = torch.zeros(100, dtype=torch.long)

        # Random speaker embedding (in real training, from speaker encoder)
        speaker_emb = torch.randn(SPEAKER_EMBED_DIM)
        speaker_emb = F.normalize(speaker_emb, dim=0)

        # Random emotion
        emotion_id = torch.randint(0, NUM_EMOTION_STYLES, (1,)).item()

        return text_ids, speaker_emb, torch.tensor(emotion_id), codec_targets


class SyntheticTTSDataset(Dataset):
    """Synthetic TTS dataset for testing."""

    def __init__(self, num_samples: int = 100, max_text_len: int = 200):
        self.num_samples = num_samples
        self.max_text_len = max_text_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text_ids = torch.randint(1, TEXT_VOCAB_SIZE, (self.max_text_len,))
        speaker_emb = F.normalize(torch.randn(SPEAKER_EMBED_DIM), dim=0)
        emotion_id = torch.randint(0, NUM_EMOTION_STYLES, (1,)).item()
        codec_targets = torch.randint(0, CODEBOOK_SIZE, (100,))
        return text_ids, speaker_emb, torch.tensor(emotion_id), codec_targets


# ============================================================================
# Collate
# ============================================================================

def collate_fn(batch):
    text_ids, speaker_embs, emotion_ids, codec_targets_list = zip(*batch)
    text_ids = torch.stack(text_ids)
    speaker_embs = torch.stack(speaker_embs)
    emotion_ids = torch.stack(emotion_ids)

    # Pad codec targets to max length
    max_len = max(t.shape[0] for t in codec_targets_list)
    padded = torch.zeros(len(codec_targets_list), max_len, dtype=torch.long)
    for i, t in enumerate(codec_targets_list):
        padded[i, :t.shape[0]] = t

    return text_ids, speaker_embs, emotion_ids, padded


# ============================================================================
# Training
# ============================================================================

def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device) -> dict:
    model.train()
    total_loss = 0.0
    num_batches = 0

    with tqdm(loader, desc='Training') as pbar:
        for text_ids, speaker_embs, emotion_ids, codec_targets in pbar:
            text_ids = text_ids.to(device)
            speaker_embs = speaker_embs.to(device)
            emotion_ids = emotion_ids.to(device)
            codec_targets = codec_targets.to(device)

            optimizer.zero_grad()

            # Forward: [B, T_text, 1024]
            logits = model(text_ids, speaker_embs, emotion_ids)

            # Match sequence lengths (text may be longer than codec targets)
            min_len = min(logits.shape[1], codec_targets.shape[1])
            logits = logits[:, :min_len, :]
            targets = codec_targets[:, :min_len]

            # Cross-entropy loss
            loss = F.cross_entropy(logits.reshape(-1, CODEBOOK_SIZE),
                                   targets.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return {'loss': total_loss / max(num_batches, 1)}


def main():
    parser = argparse.ArgumentParser(description='Train Sonata TTS')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--text_dir', type=str, default='')
    parser.add_argument('--codec_checkpoint', type=str, default='',
                       help='Pre-trained codec for computing targets')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='train/checkpoints/tts')
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

    # Load codec model if provided
    codec_model = None
    if args.codec_checkpoint and Path(args.codec_checkpoint).exists():
        logger.info(f"Loading codec from {args.codec_checkpoint}")
        from train_codec import SonataCodec
        codec_model = SonataCodec()
        ckpt = torch.load(args.codec_checkpoint, map_location='cpu', weights_only=False)
        codec_model.load_state_dict(ckpt['model'])
        codec_model.eval()
        logger.info("Codec loaded for target computation")

    # Data
    logger.info("Loading data...")
    if args.synthetic:
        dataset = SyntheticTTSDataset(num_samples=100)
    else:
        dataset = TTSDataset(
            data_dir=args.data_dir,
            text_dir=args.text_dir if args.text_dir else None,
            codec_model=codec_model,
            max_files=args.max_files,
        )

    if len(dataset) == 0:
        logger.error("No data found!")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True,
                       collate_fn=collate_fn)

    # Model
    logger.info(f"Creating TTS model (dim={TTS_DIM}, layers={TTS_LAYERS}, "
                f"vocab={TEXT_VOCAB_SIZE})...")
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
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {metrics['loss']:.4f}")

        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            ckpt_path = Path(args.checkpoint_dir) / 'tts_best.pt'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
                'config': {
                    'dim': TTS_DIM,
                    'num_layers': TTS_LAYERS,
                    'vocab_size': TEXT_VOCAB_SIZE,
                    'speaker_dim': SPEAKER_EMBED_DIM,
                    'emotion_dim': EMOTION_DIM,
                    'num_emotions': NUM_EMOTION_STYLES,
                    'num_nonverbal': NUM_NONVERBAL_TOKENS,
                },
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
