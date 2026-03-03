"""Train Sonata Codec: audio encoder + RVQ + decoder.

Trains a neural audio codec with:
- 1D ConvNet encoder (24000 Hz → 50 Hz frames × 512 dim)
- Residual Vector Quantizer (8 codebooks, 1024 codes each)
- 1D transposed ConvNet decoder (512 dim → 24000 Hz)

Loss combines reconstruction (L1) + VQ commitment + codebook losses.

Usage:
    # Basic training with default settings
    python train/train_codec.py --data_dir /path/to/audio --epochs 100

    # With custom hyperparameters
    python train/train_codec.py \\
      --data_dir /path/to/audio \\
      --batch_size 32 \\
      --lr 1e-3 \\
      --epochs 200 \\
      --segment_length 48000 \\
      --save_every 5

    # Resume from checkpoint
    python train/train_codec.py \\
      --data_dir /path/to/audio \\
      --resume train/checkpoints/codec/codec_epoch50.pt \\
      --epochs 100

    # Test mode with synthetic data
    python train/train_codec.py \\
      --data_dir /tmp/dummy \\
      --synthetic \\
      --batch_size 4 \\
      --epochs 2
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import time
from tqdm import tqdm

from data.codec_dataset import AudioCodecDataset, SyntheticAudioDataset

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Model Components
# ============================================================================

class CodecEncoder(nn.Module):
    """1D ConvNet encoder: [B,1,24000] → [B,512,50].

    Applies 4 strided convolutions (8×5×4×3 = 480× downsampling) followed
    by projection to 512-dim latent space.

    Architecture:
        Conv1d(1, 64, 7, stride=1) → ReLU
        Conv1d(64, 128, 4, stride=8) → ReLU
        Conv1d(128, 256, 4, stride=5) → ReLU
        Conv1d(256, 512, 4, stride=4) → ReLU
        Conv1d(512, 512, 4, stride=3) → identity
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, 4, stride=8, padding=0),
            nn.ReLU(),
            nn.Conv1d(128, 256, 4, stride=5, padding=0),
            nn.ReLU(),
            nn.Conv1d(256, 512, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv1d(512, out_channels, 4, stride=3, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio to latent.

        Args:
            x: [B, 1, 24000]

        Returns:
            z: [B, 512, 50]
        """
        return self.layers(x)


class VectorQuantizer(nn.Module):
    """Single VQ codebook with commitment + codebook losses.

    Quantizes continuous embeddings to nearest codebook entry using
    straight-through estimator for gradient flow.

    Args:
        dim: Embedding dimension (default: 512)
        codebook_size: Number of codes per codebook (default: 1024)
        beta: Weight for commitment loss (default: 0.25)
    """

    def __init__(self, dim: int = 512, codebook_size: int = 1024, beta: float = 0.25):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.beta = beta

        # Codebook: [codebook_size, dim]
        self.codebook = nn.Embedding(codebook_size, dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> tuple:
        """Quantize latent codes.

        Args:
            z: [B, dim, T]

        Returns:
            z_q_st: Quantized with straight-through estimator [B, dim, T]
            indices: Codebook indices [B, T]
            loss: Scalar VQ loss
        """
        # Flatten: [B, dim, T] → [B*T, dim]
        z_flat = z.permute(0, 2, 1).reshape(-1, self.dim)

        # Find nearest codebook entry: [B*T, codebook_size]
        dist = torch.cdist(z_flat, self.codebook.weight)  # Euclidean distance
        indices = dist.argmin(dim=-1)  # [B*T]

        # Quantized: [B*T, dim]
        z_q_flat = self.codebook(indices)

        # Reshape back: [B, dim, T]
        z_q = z_q_flat.reshape(z.shape[0], z.shape[2], self.dim).permute(0, 2, 1)

        # Straight-through estimator: use quantized in backward
        z_q_st = z + (z_q - z).detach()

        # Losses
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = commitment_loss + self.beta * codebook_loss

        return z_q_st, indices, loss


class ResidualVQ(nn.Module):
    """Residual Vector Quantizer with K independent codebooks.

    Applies successive quantization stages, where each stage quantizes
    the residual from the previous stage. This improves reconstruction
    by allowing different parts of the signal to be represented at
    different precision levels.

    Args:
        dim: Embedding dimension (default: 512)
        num_books: Number of VQ stages (default: 8)
        codebook_size: Codes per codebook (default: 1024)
    """

    def __init__(self, dim: int = 512, num_books: int = 8, codebook_size: int = 1024):
        super().__init__()
        self.num_books = num_books
        self.quantizers = nn.ModuleList([
            VectorQuantizer(dim, codebook_size) for _ in range(num_books)
        ])

    def forward(self, z: torch.Tensor) -> tuple:
        """Quantize with residual VQ.

        Args:
            z: [B, dim, T]

        Returns:
            z_quantized: Reconstructed from all codebooks [B, dim, T]
            codes: All codebook indices [B, num_books, T]
            loss: Total loss (sum of all stages)
        """
        residual = z
        all_indices = []
        total_loss = 0.0

        for i, vq in enumerate(self.quantizers):
            z_q, codes, loss = vq(residual)
            residual = residual - z_q  # Subtract quantized from residual
            all_indices.append(codes)
            total_loss = total_loss + loss

        # z_quantized = sum of all z_q across stages
        z_quantized = z - residual

        codes_tensor = torch.stack(all_indices, dim=1)  # [B*T, num_books] → [B, T, num_books]

        return z_quantized, codes_tensor, total_loss


class CodecDecoder(nn.Module):
    """1D transposed ConvNet decoder: [B,512,50] → [B,1,24000].

    Inverts the encoder with transposed convolutions and matching
    strides/channels.

    Architecture:
        ConvTranspose1d(512, 512, 4, stride=3) → ReLU
        ConvTranspose1d(512, 256, 4, stride=4) → ReLU
        ConvTranspose1d(256, 128, 4, stride=5) → ReLU
        ConvTranspose1d(128, 64, 4, stride=8) → ReLU
        Conv1d(64, 1, 7, stride=1, padding=3) → Tanh
    """

    def __init__(self, in_channels: int = 512, out_channels: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels, 512, 4, stride=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, 4, stride=5, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, stride=8, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, out_channels, 7, stride=1, padding=3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent to audio.

        Args:
            x: [B, 512, 50]

        Returns:
            audio: [B, 1, 24000]
        """
        return self.layers(x)


class SonataCodec(nn.Module):
    """Full audio codec: encoder + RVQ + decoder.

    End-to-end codec that learns to compress audio using quantized
    embeddings and reconstruct from quantized codes.

    Args:
        sample_rate: Target sample rate (default: 24000 Hz)
        latent_dim: Latent embedding dimension (default: 512)
        num_books: Number of VQ codebooks (default: 8)
        codebook_size: Codes per codebook (default: 1024)
    """

    def __init__(self, sample_rate: int = 24000, latent_dim: int = 512,
                 num_books: int = 8, codebook_size: int = 1024):
        super().__init__()
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim

        self.encoder = CodecEncoder(out_channels=latent_dim)
        self.rvq = ResidualVQ(latent_dim, num_books, codebook_size)
        self.decoder = CodecDecoder(in_channels=latent_dim)

    def forward(self, audio: torch.Tensor) -> tuple:
        """Encode, quantize, decode.

        Args:
            audio: [B, 1, 24000]

        Returns:
            reconstructed: [B, 1, 24000]
            codes: [B, T, num_books]
            loss: Scalar VQ + reconstruction loss
        """
        # Encode
        z = self.encoder(audio)  # [B, 512, 50]

        # Quantize
        z_q, codes, vq_loss = self.rvq(z)

        # Decode
        reconstructed = self.decoder(z_q)

        return reconstructed, codes, vq_loss

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode to quantized codes (inference mode).

        Args:
            audio: [B, 1, samples]

        Returns:
            codes: [B, T, num_books]
        """
        z = self.encoder(audio)
        _, codes, _ = self.rvq(z)
        return codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from quantized codes (inference mode).

        Args:
            codes: [B, T, num_books]

        Returns:
            audio: [B, 1, samples]
        """
        # Reconstruct from codes
        z_q = torch.zeros(codes.shape[0], self.latent_dim, codes.shape[1],
                         device=codes.device)
        for i, vq in enumerate(self.rvq.quantizers):
            z_q_i = vq.codebook(codes[:, :, i])  # [B, T, dim]
            z_q = z_q + z_q_i.permute(0, 2, 1)  # Add to [B, dim, T]

        audio = self.decoder(z_q)
        return audio


# ============================================================================
# Training
# ============================================================================

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, loss_weight: float = 1.0) -> dict:
    """Train for one epoch.

    Args:
        model: Codec model
        loader: Training data loader
        optimizer: Adam optimizer
        device: Torch device
        loss_weight: Weight for reconstruction loss

    Returns:
        metrics: Dictionary with loss values
    """
    model.train()
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    num_batches = 0

    with tqdm(loader, desc='Training') as pbar:
        for batch_idx, audio in enumerate(pbar):
            audio = audio.to(device)
            optimizer.zero_grad()

            # Forward pass
            reconstructed, codes, vq_loss = model(audio)

            # Reconstruction loss (L1 for perceptual quality)
            min_len = min(audio.shape[2], reconstructed.shape[2])
            recon_loss = F.l1_loss(reconstructed[:, :, :min_len],
                                   audio[:, :, :min_len])

            # Total loss
            total_loss = loss_weight * recon_loss + vq_loss

            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Metrics
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'recon': f'{recon_loss.item():.4f}',
                'vq': f'{vq_loss.item():.4f}'
            })

    return {
        'recon_loss': total_recon_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches,
    }


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Validate on test set.

    Args:
        model: Codec model
        loader: Validation data loader
        device: Torch device

    Returns:
        metrics: Dictionary with loss values
    """
    model.eval()
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    num_batches = 0

    for audio in tqdm(loader, desc='Validating'):
        audio = audio.to(device)

        reconstructed, codes, vq_loss = model(audio)
        min_len = min(audio.shape[2], reconstructed.shape[2])
        recon_loss = F.l1_loss(reconstructed[:, :, :min_len],
                               audio[:, :, :min_len])

        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        num_batches += 1

    return {
        'recon_loss': total_recon_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Sonata Codec')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Audio directory')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--segment_length', type=int, default=24000)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--checkpoint_dir', type=str, default='train/checkpoints/codec')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--resume', type=str, default='',
                       help='Checkpoint to resume from')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data (testing)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Limit number of audio files')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    if args.synthetic:
        dataset = SyntheticAudioDataset(num_samples=100, segment_length=args.segment_length)
    else:
        dataset = AudioCodecDataset(args.data_dir, segment_length=args.segment_length,
                                   max_files=args.max_files)

    if len(dataset) == 0:
        logger.error("No data found!")
        return

    # Split train/val
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers,
                             pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers,
                           pin_memory=True)

    # Model
    logger.info("Creating model...")
    model = SonataCodec(segment_length=args.segment_length).to(device)
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
        best_loss = ckpt.get('best_loss', float('inf'))

    # Training
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | "
                   f"Train recon={train_metrics['recon_loss']:.4f} "
                   f"vq={train_metrics['vq_loss']:.4f}")

        # Validate
        if (epoch + 1) % 5 == 0:
            val_metrics = validate(model, val_loader, device)
            val_loss = val_metrics['recon_loss'] + val_metrics['vq_loss']
            logger.info(f"  Val recon={val_metrics['recon_loss']:.4f} "
                       f"vq={val_metrics['vq_loss']:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                ckpt_path = Path(args.checkpoint_dir) / 'codec_best.pt'
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_loss': best_loss,
                }, ckpt_path)
                logger.info(f"  Saved best checkpoint: {ckpt_path}")

        # Periodic save
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = Path(args.checkpoint_dir) / f'codec_epoch{epoch+1}.pt'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_loss': best_loss,
            }, ckpt_path)
            logger.info(f"  Saved checkpoint: {ckpt_path}")

        scheduler.step()
        epoch_time = time.time() - epoch_start
        logger.info(f"  Time: {epoch_time:.1f}s")

    logger.info(f"Training complete! Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
