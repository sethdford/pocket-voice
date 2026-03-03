"""Train Sonata CFM (Conditional Flow Matching) decoder.

Converts codec token embeddings to mel spectrograms using straight-line flow matching.
Architecture matches crates/sonata-cfm/ exactly:
  - 12 DiT (Diffusion Transformer) blocks
  - Sinusoidal time embedding (dim=256) + 2-layer MLP with ReLU
  - Speaker conditioning via AdaIN
  - SwiGLU feed-forward networks
  - Euler ODE solver for inference

Usage:
    python train/train_cfm.py \
      --data_dir /path/to/libritts_r \
      --speaker_checkpoint train/checkpoints/speaker_encoder_best.pt \
      --epochs 100 \
      --batch_size 16
"""

import argparse
import logging
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Constants (match sonata-common/src/lib.rs) ---
SAMPLE_RATE = 24000
MEL_BINS = 80
HOP_LENGTH = 256
FFT_SIZE = 1024

# --- CFM constants (match sonata-cfm) ---
CFM_DIM = 512
CFM_FFN_DIM = 2048
CFM_LAYERS = 12
CFM_HEADS = 8
TIME_DIM = 256

# --- Speaker encoder (match sonata-cam) ---
SPEAKER_EMBED_DIM = 192


# ============================================================
#  MelSpectrogram — matches crates/sonata-common/src/mel.rs
# ============================================================
class MelSpectrogram(nn.Module):
    """Compute mel spectrogram matching Rust mel.rs exactly.

    Uses FFT_SIZE=1024, WINDOW_SIZE=600, HOP_SIZE=240, 80 mel bins, 0-12kHz.
    """

    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=FFT_SIZE, n_mels=MEL_BINS,
                 win_length=600, hop_length=240, f_min=0.0, f_max=12000.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Pre-compute Hann window
        self.register_buffer('window', torch.hann_window(win_length))

        # Pre-compute mel filterbank (triangular filters)
        mel_fb = self._create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer('mel_fb', mel_fb)

    @staticmethod
    def _hz_to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_filterbank(self, sr, n_fft, n_mels, f_min, f_max):
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = torch.tensor([self._mel_to_hz(m.item()) for m in mel_points])
        bin_points = torch.floor((n_fft + 1) * hz_points / sr).long()

        n_freqs = n_fft // 2 + 1
        fb = torch.zeros(n_mels, n_freqs)
        for i in range(n_mels):
            left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
            for j in range(left, center):
                if center > left:
                    fb[i, j] = (j - left).float() / (center - left).float()
            for j in range(center, right):
                if right > center:
                    fb[i, j] = (right - j).float() / (right - center).float()
        return fb  # [n_mels, n_freqs]

    def forward(self, audio):
        """Compute mel spectrogram from audio waveform.

        Args:
            audio: [B, T] waveform at 24kHz

        Returns:
            mel: [B, MEL_BINS, frames]
        """
        # Pad window to n_fft size
        pad_amount = self.n_fft - self.win_length
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        window = F.pad(self.window, (pad_left, pad_right))

        stft = torch.stft(audio, self.n_fft, hop_length=self.hop_length,
                          window=window, return_complex=True)
        power = stft.abs().pow(2)  # [B, n_freqs, frames]

        # Apply mel filterbank: [n_mels, n_freqs] x [B, n_freqs, frames]
        mel = torch.matmul(self.mel_fb, power)  # [B, n_mels, frames]

        # Log scale with floor
        mel = torch.log(mel.clamp(min=1e-5))
        return mel


# ============================================================
#  SwiGLU — matches crates/sonata-common/src/swiglu.rs
# ============================================================
class SwiGLU(nn.Module):
    """SwiGLU FFN: gate → SiLU(Swish), up, element-wise multiply, down.

    Weight names: w_gate, w_up, w_down match Rust SwiGLU field names.
    VarBuilder paths: ffn.gate.{weight,bias}, ffn.up.{weight,bias}, ffn.down.{weight,bias}
    """

    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.gate = nn.Linear(dim, ffn_dim)
        self.up = nn.Linear(dim, ffn_dim)
        self.down = nn.Linear(ffn_dim, dim)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ============================================================
#  AdaIN — matches crates/sonata-common/src/adain.rs
# ============================================================
class AdaIN(nn.Module):
    """Adaptive Instance Normalization.

    AdaIN(x, style) = gamma(style) * InstanceNorm(x) + beta(style)

    VarBuilder paths: gamma_proj.{weight,bias}, beta_proj.{weight,bias}
    """

    def __init__(self, hidden_dim, style_dim):
        super().__init__()
        self.gamma_proj = nn.Linear(style_dim, hidden_dim)
        self.beta_proj = nn.Linear(style_dim, hidden_dim)

    def forward(self, x, style):
        """
        Args:
            x: [B, T, hidden_dim]
            style: [B, style_dim]
        """
        # Instance normalization across hidden_dim
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + 1e-5).sqrt()

        # Affine transform from style
        gamma = self.gamma_proj(style).unsqueeze(1)  # [B, 1, hidden_dim]
        beta = self.beta_proj(style).unsqueeze(1)     # [B, 1, hidden_dim]

        return gamma * x_norm + beta


# ============================================================
#  TimeEmbedding — matches sonata-cfm/src/lib.rs TimeEmbedding
# ============================================================
class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding + 2-layer MLP for diffusion timestep.

    Uses log-spaced frequencies: freq_i = exp(-i/half_dim * ln(10000))
    Then MLP: Linear(dim, dim) → ReLU → Linear(dim, dim)

    VarBuilder paths: time_mlp1.{weight,bias}, time_mlp2.{weight,bias}
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.time_mlp1 = nn.Linear(dim, dim)
        self.time_mlp2 = nn.Linear(dim, dim)

    def sinusoidal_embedding(self, t, device):
        """Create sinusoidal embedding for timestep(s) t.

        Matches Rust: freq = exp(-i/half_dim * ln(10000)), first half sin, second half cos.

        Args:
            t: [B] or scalar timestep(s) in [0, 1]
            device: torch device

        Returns:
            [B, dim] embedding tensor
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        b = t.shape[0]
        half_dim = self.dim // 2

        # Vectorized: freqs [half_dim], angles [B, half_dim]
        freqs = torch.exp(
            -torch.arange(half_dim, device=device, dtype=torch.float32)
            / half_dim * math.log(10000.0)
        )
        angles = t.unsqueeze(1) * freqs.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [B, dim]
        return emb

    def forward(self, t, device):
        """
        Args:
            t: [B] or scalar timestep(s) in [0, 1]
            device: torch device

        Returns:
            [B, dim] time embedding
        """
        emb = self.sinusoidal_embedding(t, device)
        h = F.relu(self.time_mlp1(emb))
        return self.time_mlp2(h)


# ============================================================
#  DiTBlock — matches crates/sonata-cfm/src/dit.rs
# ============================================================
class DiTBlock(nn.Module):
    """Diffusion Transformer block with speaker and time conditioning.

    Architecture (matches dit.rs exactly):
      norm → attn (Linear) + residual → time_adain → speaker_adain → ffn (SwiGLU) + residual

    VarBuilder paths per block:
      attn.{weight,bias}
      time_adain.gamma_proj.{weight,bias}, time_adain.beta_proj.{weight,bias}
      spk_adain.gamma_proj.{weight,bias}, spk_adain.beta_proj.{weight,bias}
      ffn.gate.{weight,bias}, ffn.up.{weight,bias}, ffn.down.{weight,bias}
      norm.{weight,bias}
    """

    def __init__(self, dim, ffn_dim, time_dim, speaker_dim):
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.time_adain = AdaIN(dim, time_dim)
        self.spk_adain = AdaIN(dim, speaker_dim)
        self.ffn = SwiGLU(dim, ffn_dim)
        self.norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x, time_emb, speaker_emb):
        """
        Args:
            x: [B, T, dim]
            time_emb: [B, time_dim] or [1, time_dim] (broadcast)
            speaker_emb: [B, speaker_dim]

        Returns:
            [B, T, dim]
        """
        # Normalize → attention + residual
        x_norm = self.norm(x)
        attn_out = self.attn(x_norm)
        x = x + attn_out

        # Condition on timestep via AdaIN
        x = self.time_adain(x, time_emb)

        # Condition on speaker via AdaIN
        x = self.spk_adain(x, speaker_emb)

        # SwiGLU FFN + residual
        x = x + self.ffn(x)
        return x


# ============================================================
#  SonataCFM — matches crates/sonata-cfm/src/lib.rs
# ============================================================
class SonataCFM(nn.Module):
    """Conditional Flow Matching decoder: noise → mel spectrogram.

    Architecture:
      input_proj: Linear(MEL_BINS=80, CFM_DIM=512)
      12x DiTBlock(512, 2048, 256, 192)
      output_proj: Linear(CFM_DIM=512, MEL_BINS=80)
      time_embed: TimeEmbedding(256)

    VarBuilder paths:
      input.{weight,bias}
      blocks.{i}.attn.{weight,bias}
      blocks.{i}.time_adain.{gamma_proj,beta_proj}.{weight,bias}
      blocks.{i}.spk_adain.{gamma_proj,beta_proj}.{weight,bias}
      blocks.{i}.ffn.{gate,up,down}.{weight,bias}
      blocks.{i}.norm.{weight,bias}
      output.{weight,bias}
      time_embed.time_mlp1.{weight,bias}
      time_embed.time_mlp2.{weight,bias}
    """

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(MEL_BINS, CFM_DIM)
        self.blocks = nn.ModuleList([
            DiTBlock(CFM_DIM, CFM_FFN_DIM, TIME_DIM, SPEAKER_EMBED_DIM)
            for _ in range(CFM_LAYERS)
        ])
        self.output_proj = nn.Linear(CFM_DIM, MEL_BINS)
        self.time_embed = TimeEmbedding(TIME_DIM)

    def velocity(self, x, t, speaker_emb):
        """Compute velocity field v(x, t, speaker).

        Args:
            x: noisy mel [B, MEL_BINS, T]
            t: [B] or scalar timestep(s) in [0, 1]
            speaker_emb: [B, SPEAKER_EMBED_DIM]

        Returns:
            velocity [B, MEL_BINS, T]
        """
        time_emb = self.time_embed(t, x.device)  # [B, TIME_DIM]

        # x is [B, mel, T] → transpose to [B, T, mel] for Linear layers
        h = x.transpose(1, 2)          # [B, T, mel]
        h = self.input_proj(h)          # [B, T, CFM_DIM]

        for block in self.blocks:
            h = block(h, time_emb, speaker_emb)

        out = self.output_proj(h)       # [B, T, mel]
        return out.transpose(1, 2)      # [B, mel, T]

    def generate(self, speaker_emb, num_frames, steps=8):
        """Generate mel spectrogram via Euler ODE solver.

        Matches ode.rs euler_solve: integrates from t=0 to t=1.

        Args:
            speaker_emb: [B, SPEAKER_EMBED_DIM]
            num_frames: number of output time frames
            steps: number of Euler integration steps (typically 4-8)

        Returns:
            mel spectrogram [B, MEL_BINS, num_frames]
        """
        batch = speaker_emb.shape[0]
        device = speaker_emb.device

        # Start from random noise
        x = torch.randn(batch, MEL_BINS, num_frames, device=device)

        # Euler solve from t=0 to t=1
        dt = 1.0 / steps
        for step in range(steps):
            t = step * dt
            v = self.velocity(x, t, speaker_emb)
            x = x + v * dt

        return x


# ============================================================
#  Dataset
# ============================================================
class CFMDataset(Dataset):
    """Dataset for CFM training: mel spectrograms + speaker embeddings.

    Loads audio files, computes mel spectrograms as targets.
    Speaker embeddings are extracted using a pre-trained CAM++ encoder.
    """

    def __init__(self, data_dir, speaker_model=None, max_frames=200):
        self.data_dir = data_dir
        self.speaker_model = speaker_model
        self.max_frames = max_frames
        self.mel_transform = MelSpectrogram()

        # Collect audio files
        self.audio_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.wav') or f.endswith('.flac'):
                    self.audio_files.append(os.path.join(root, f))
        self.audio_files.sort()
        logger.info(f"CFMDataset: {len(self.audio_files)} audio files from {data_dir}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]

        # Load and resample audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio = waveform.squeeze(0)  # [T]

        # Compute mel spectrogram
        mel = self.mel_transform(audio.unsqueeze(0)).squeeze(0)  # [MEL_BINS, frames]

        # Truncate to max_frames
        if mel.shape[1] > self.max_frames:
            start = random.randint(0, mel.shape[1] - self.max_frames)
            mel = mel[:, start:start + self.max_frames]

        # Extract speaker embedding (or use zero placeholder)
        if self.speaker_model is not None:
            with torch.no_grad():
                # Resample to 16kHz for speaker encoder
                audio_16k = torchaudio.functional.resample(audio.unsqueeze(0), SAMPLE_RATE, 16000)
                spk_emb = self.speaker_model(audio_16k).squeeze(0)  # [SPEAKER_EMBED_DIM]
        else:
            spk_emb = torch.zeros(SPEAKER_EMBED_DIM)

        return mel, spk_emb


def collate_fn(batch):
    """Pad mel spectrograms to same length within batch."""
    mels, spk_embs = zip(*batch)

    max_frames = max(m.shape[1] for m in mels)
    padded_mels = []
    for mel in mels:
        if mel.shape[1] < max_frames:
            pad = torch.zeros(MEL_BINS, max_frames - mel.shape[1])
            mel = torch.cat([mel, pad], dim=1)
        padded_mels.append(mel)

    return torch.stack(padded_mels), torch.stack(spk_embs)


# ============================================================
#  Training Loop
# ============================================================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load pre-trained speaker encoder (optional)
    speaker_model = None
    if args.speaker_checkpoint and os.path.exists(args.speaker_checkpoint):
        logger.info(f"Loading speaker encoder: {args.speaker_checkpoint}")
        from train_speaker_encoder import CamPlusPlus
        speaker_model = CamPlusPlus()
        ckpt = torch.load(args.speaker_checkpoint, map_location='cpu', weights_only=True)
        speaker_model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
        speaker_model.eval()
        speaker_model.to(device)

    # Dataset and loader
    dataset = CFMDataset(args.data_dir, speaker_model=speaker_model, max_frames=args.max_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # Model
    model = SonataCFM().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"SonataCFM: {total_params:,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for mel_target, spk_emb in loader:
            mel_target = mel_target.to(device)  # [B, MEL_BINS, T]
            spk_emb = spk_emb.to(device)        # [B, SPEAKER_EMBED_DIM]

            # --- Conditional Flow Matching loss ---
            # Sample random noise x0
            x0 = torch.randn_like(mel_target)

            # Sample random timestep t ~ Uniform(0, 1)
            t = random.random()

            # Straight-line interpolation: x_t = (1-t)*x0 + t*x1
            x_t = (1 - t) * x0 + t * mel_target

            # Target velocity: v = x1 - x0
            v_target = mel_target - x0

            # Model prediction
            v_pred = model.velocity(x_t, t, spk_emb)

            # MSE loss on velocity
            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | lr={lr:.6f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.checkpoint_dir, 'cfm_best.pt'))
            logger.info(f"  Saved best checkpoint (loss={avg_loss:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.checkpoint_dir, f'cfm_epoch{epoch+1}.pt'))

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")


def export_safetensors(checkpoint_path, output_path):
    """Export trained CFM model to safetensors format."""
    from safetensors.torch import save_file

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state_dict = ckpt.get('model', ckpt)

    # Ensure all tensors are f32 and contiguous
    export_dict = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            export_dict[key] = tensor.float().contiguous()

    save_file(export_dict, output_path)
    logger.info(f"Exported {len(export_dict)} tensors to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Sonata CFM decoder')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory with audio files (LibriTTS-R)')
    parser.add_argument('--speaker_checkpoint', type=str, default=None,
                        help='Path to pre-trained CAM++ speaker encoder checkpoint')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Max mel frames per sample')
    parser.add_argument('--checkpoint_dir', type=str, default='train/checkpoints/cfm')
    parser.add_argument('--export', type=str, default=None,
                        help='Export checkpoint to safetensors path')
    args = parser.parse_args()

    if args.export:
        export_safetensors(args.export, args.export.replace('.pt', '.safetensors'))
    else:
        train(args)
