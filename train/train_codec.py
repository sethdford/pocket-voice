"""Train Sonata Codec: audio encoder + RVQ + decoder.

Trains a neural audio codec with:
- 1D ConvNet encoder with Snake activation (24000 Hz → 50 Hz frames × 512 dim)
- Residual Vector Quantizer (8 codebooks, 1024 codes each, 128-dim codebook vectors)
- 1D transposed ConvNet decoder with Snake activation (512 dim → 24000 Hz)

Architecture matches Rust inference in sonata-codec crate:
- Encoder: 4 blocks with strides [8,5,4,3], kernel=stride*2, padding=stride//2
- Snake activation: x + (1/alpha) * sin^2(alpha * x)
- RVQ: project_in(512→128) → 8 codebooks(1024, 128) → project_out(128→512)
- Decoder: 4 blocks with strides [3,4,5,8], kernel=stride*2, padding=stride//2

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

try:
    from data.codec_dataset import AudioCodecDataset, SyntheticAudioDataset
except ImportError:
    from train.data.codec_dataset import AudioCodecDataset, SyntheticAudioDataset

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Model Components
# ============================================================================

class Snake(nn.Module):
    """Snake activation: x + (1/alpha) * sin^2(alpha * x).

    Learnable periodic activation function for audio, matching
    sonata-codec/src/snake.rs. Alpha is per-channel.

    Args:
        channels: Number of channels (alpha has shape [channels])
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Snake activation.

        Args:
            x: [B, channels, T]

        Returns:
            activated: [B, channels, T]
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
        return x + (1.0 / alpha) * torch.sin(alpha * x).pow(2)


class CodecEncoder(nn.Module):
    """1D ConvNet encoder: [B,1,24000] → [B,512,50].

    Matches sonata-codec/src/encoder.rs:
    4 EncoderBlocks with strides [8,5,4,3] (480× downsampling).
    channels [1, 64, 128, 256, 512], kernel=stride*2, padding=stride//2.
    Each block: Conv1d → Snake activation.
    """

    STRIDES = [8, 5, 4, 3]
    CHANNELS = [1, 64, 128, 256, 512]

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.snake_layers = nn.ModuleList()
        for i, stride in enumerate(self.STRIDES):
            in_ch = self.CHANNELS[i]
            out_ch = self.CHANNELS[i + 1]
            kernel = stride * 2
            padding = stride // 2
            self.conv_layers.append(nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding))
            self.snake_layers.append(Snake(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio to latent.

        Args:
            x: [B, 1, 24000]

        Returns:
            z: [B, 512, 50]
        """
        for conv, snake in zip(self.conv_layers, self.snake_layers):
            x = snake(conv(x))
        return x


class ResidualVQ(nn.Module):
    """Residual Vector Quantizer with shared project_in/out and K codebooks.

    Matches sonata-codec/src/quantizer.rs ResidualVQ:
    - Shared project_in: Linear(input_dim→codebook_dim)
    - Shared project_out: Linear(codebook_dim→input_dim)
    - K codebooks as nn.ModuleList (each nn.Embedding)

    State dict keys:
        rvq.project_in.weight, rvq.project_out.weight
        rvq.codebooks.{i}.embeddings  (renamed from .weight)

    Args:
        input_dim: Input embedding dimension (default: 512)
        num_books: Number of VQ stages (default: 8)
        codebook_size: Codes per codebook (default: 1024)
        codebook_dim: Codebook vector dimension (default: 128)
        beta: Commitment loss weight (default: 0.25)
    """

    def __init__(self, input_dim: int = 512, num_books: int = 8,
                 codebook_size: int = 1024, codebook_dim: int = 128,
                 beta: float = 0.25):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.num_books = num_books
        self.beta = beta

        # Shared projections (matches Rust: single project_in/out)
        self.project_in = nn.Linear(input_dim, codebook_dim)
        self.project_out = nn.Linear(codebook_dim, input_dim)

        # Codebooks as ModuleList (keys: codebooks.{i}.embeddings)
        self.codebooks = nn.ModuleList([
            self._make_codebook(codebook_size, codebook_dim)
            for _ in range(num_books)
        ])

    @staticmethod
    def _make_codebook(size, dim):
        """Create a codebook module with 'embeddings' parameter name."""
        codebook = nn.Module()
        codebook.embeddings = nn.Parameter(
            torch.empty(size, dim).uniform_(-1.0 / size, 1.0 / size)
        )
        return codebook

    def _quantize_one(self, z_proj: torch.Tensor, codebook_idx: int):
        """Quantize using one codebook (operates in codebook space).

        Args:
            z_proj: [B, T, codebook_dim] (already projected)
            codebook_idx: Which codebook to use

        Returns:
            z_q: [B, input_dim, T] quantized back in input space
            indices: [B, T] codebook indices
            loss: scalar VQ loss
        """
        b, t, d = z_proj.shape
        cb_weight = self.codebooks[codebook_idx].embeddings  # [codebook_size, codebook_dim]

        z_flat = z_proj.reshape(-1, d)  # [B*T, codebook_dim]
        dist = torch.cdist(z_flat, cb_weight)  # [B*T, codebook_size]
        indices = dist.argmin(dim=-1)  # [B*T]

        z_q_flat = cb_weight[indices]  # [B*T, codebook_dim]
        z_q_proj = z_q_flat.reshape(b, t, d)

        # Straight-through estimator in codebook space
        z_q_st = z_proj + (z_q_proj - z_proj).detach()

        # Losses in codebook space
        commitment_loss = F.mse_loss(z_q_proj.detach(), z_proj)
        codebook_loss = F.mse_loss(z_q_proj, z_proj.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # Project back to input dimension
        z_q_out = self.project_out(z_q_st).permute(0, 2, 1)  # [B, input_dim, T]

        return z_q_out, indices.reshape(b, t), loss

    def forward(self, z: torch.Tensor) -> tuple:
        """Quantize with residual VQ.

        Args:
            z: [B, input_dim, T]

        Returns:
            z_quantized: Reconstructed from all codebooks [B, input_dim, T]
            codes: All codebook indices [B, num_books, T]
            loss: Total loss (sum of all stages)
        """
        residual = z
        all_indices = []
        total_loss = 0.0

        for i in range(self.num_books):
            # Project residual to codebook space
            r_perm = residual.permute(0, 2, 1)  # [B, T, input_dim]
            r_proj = self.project_in(r_perm)  # [B, T, codebook_dim]

            z_q, codes, loss = self._quantize_one(r_proj, i)
            residual = residual - z_q
            all_indices.append(codes)
            total_loss = total_loss + loss

        z_quantized = z - residual
        codes_tensor = torch.stack(all_indices, dim=1)  # [B, num_books, T]

        return z_quantized, codes_tensor, total_loss


class CodecDecoder(nn.Module):
    """1D transposed ConvNet decoder: [B,512,50] → [B,1,24000].

    Matches sonata-codec/src/decoder.rs:
    4 DecoderBlocks with strides [3,4,5,8] (480× upsampling).
    channels [512, 256, 128, 64, 1], kernel=stride*2, padding=stride//2.
    Each block: ConvTranspose1d → Snake activation (except final → Tanh).
    """

    STRIDES = [3, 4, 5, 8]
    CHANNELS = [512, 256, 128, 64, 1]

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.snake_layers = nn.ModuleList()
        for i, stride in enumerate(self.STRIDES):
            in_ch = self.CHANNELS[i]
            out_ch = self.CHANNELS[i + 1]
            kernel = stride * 2
            padding = stride // 2
            self.conv_layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride, padding=padding))
            if i < len(self.STRIDES) - 1:
                self.snake_layers.append(Snake(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent to audio.

        Args:
            x: [B, 512, 50]

        Returns:
            audio: [B, 1, ~24000]
        """
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i < len(self.snake_layers):
                x = self.snake_layers[i](x)
            else:
                x = torch.tanh(x)
        return x


class SonataCodec(nn.Module):
    """Full audio codec: encoder + RVQ + decoder.

    Matches sonata-codec/src/lib.rs architecture.

    Args:
        sample_rate: Target sample rate (default: 24000 Hz)
        latent_dim: Latent embedding dimension (default: 512)
        num_books: Number of VQ codebooks (default: 8)
        codebook_size: Codes per codebook (default: 1024)
        codebook_dim: Codebook vector dimension (default: 128)
    """

    def __init__(self, sample_rate: int = 24000, latent_dim: int = 512,
                 num_books: int = 8, codebook_size: int = 1024,
                 codebook_dim: int = 128):
        super().__init__()
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim

        self.encoder = CodecEncoder()
        self.rvq = ResidualVQ(latent_dim, num_books, codebook_size, codebook_dim)
        self.decoder = CodecDecoder()

    def forward(self, audio: torch.Tensor) -> tuple:
        """Encode, quantize, decode.

        Args:
            audio: [B, 1, 24000]

        Returns:
            reconstructed: [B, 1, ~24000]
            codes: [B, num_books, T]
            loss: Scalar VQ + reconstruction loss
        """
        z = self.encoder(audio)
        z_q, codes, vq_loss = self.rvq(z)
        reconstructed = self.decoder(z_q)
        return reconstructed, codes, vq_loss

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode to quantized codes (inference mode).

        Args:
            audio: [B, 1, samples]

        Returns:
            codes: [B, num_books, T]
        """
        z = self.encoder(audio)
        _, codes, _ = self.rvq(z)
        return codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from quantized codes (inference mode).

        Args:
            codes: [B, num_books, T]

        Returns:
            audio: [B, 1, samples]
        """
        b, num_books, t = codes.shape
        z_q = torch.zeros(b, self.latent_dim, t, device=codes.device)
        for i in range(num_books):
            book_codes = codes[:, i, :]  # [B, T]
            cb_weight = self.rvq.codebooks[i].embeddings  # [codebook_size, dim]
            z_q_cb = cb_weight[book_codes]  # [B, T, codebook_dim]
            z_q_out = self.rvq.project_out(z_q_cb)  # [B, T, latent_dim]
            z_q = z_q + z_q_out.permute(0, 2, 1)

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

    # Model (no segment_length arg — matches SonataCodec constructor)
    logger.info("Creating model...")
    model = SonataCodec().to(device)
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
