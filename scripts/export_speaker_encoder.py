#!/usr/bin/env python3
"""
Export speaker encoder v2 checkpoint to .safetensors and .onnx formats.

Loads a trained ECAPA-TDNN checkpoint (PyTorch with AAM-Softmax classifier)
and exports the encoder portion to both safetensors (for Rust) and ONNX
(for pocket-voice C inference via ONNX Runtime).

Usage:
    python3 scripts/export_speaker_encoder.py \
      --checkpoint checkpoints/speaker_encoder_v2/speaker_encoder_best.pt \
      --output-dir models/speaker_encoder/
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model Architecture Classes (ECAPA-TDNN)
# ============================================================================


class BatchNorm1d(nn.Module):
    """Wrapper around nn.BatchNorm1d for consistency."""

    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, 128, kernel_size=1, bias=True)
        self.conv2 = nn.Conv1d(128, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling
        s = x.mean(dim=2, keepdim=True)
        # Squeeze-excitation
        s = F.relu(self.conv1(s))
        s = torch.sigmoid(self.conv2(s))
        # Scale input
        return x * s


class Res2NetBlock(nn.Module):
    """Multi-branch residual block with dilated convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
    ):
        super().__init__()
        self.scale = scale
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Split channels across branches
        width = out_channels // scale

        self.branches = nn.ModuleList([
            nn.Conv1d(
                width if i > 0 else in_channels,
                width,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2,
                bias=True,
            )
            for i in range(scale)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input across branches
        xs = torch.chunk(x, self.scale, dim=1)
        out = []

        for i, branch in enumerate(self.branches):
            if i == 0:
                # First branch processes original input
                out.append(branch(xs[i]))
            else:
                # Other branches process output of previous branch + original input
                out.append(branch(xs[i] + out[-1]))

        return torch.cat(out, dim=1)


class SERes2NetBlock(nn.Module):
    """Residual block with Res2Net + SE attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bn1 = BatchNorm1d(out_channels)
        self.res2net = Res2NetBlock(
            out_channels, out_channels, kernel_size, dilation, scale
        )
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True)
        self.bn2 = BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)

        # Shortcut projection if needed
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True),
                BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.res2net(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.se(out)

        return F.relu(out + residual)


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling with learnable attention."""

    def __init__(self, in_channels: int, bottleneck_dim: int = 128):
        super().__init__()
        self.attention_conv = nn.Conv1d(
            in_channels, 1, kernel_size=1, padding=0, bias=True
        )
        self.attention_bn = BatchNorm1d(1)
        self.attention_proj = nn.Linear(in_channels, bottleneck_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, time]
        Returns:
            pooled: [batch, channels * 2]
        """
        # Compute attention weights
        attn = self.attention_conv(x)  # [batch, 1, time]
        attn = self.attention_bn(attn)
        attn = torch.tanh(attn)
        attn = F.softmax(attn, dim=2)  # [batch, 1, time]

        # Compute weighted mean
        mean = torch.sum(x * attn, dim=2)  # [batch, channels]

        # Compute weighted variance
        var = torch.sum(((x - mean.unsqueeze(2)) ** 2) * attn, dim=2)  # [batch, channels]
        std = torch.sqrt(torch.clamp(var, min=1e-10))

        # Concatenate mean and std
        pooled = torch.cat([mean, std], dim=1)  # [batch, channels * 2]
        return pooled


class EcapaTdnn(nn.Module):
    """ECAPA-TDNN speaker embedding model."""

    def __init__(
        self,
        n_mels: int = 80,
        embedding_dim: int = 256,
        channels: List[int] = None,
    ):
        super().__init__()
        if channels is None:
            channels = [1024, 1024, 1024, 1024, 1536]

        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.channels = channels

        # Initial convolution
        self.conv1 = nn.Conv1d(n_mels, channels[0], kernel_size=5, padding=2, bias=True)
        self.bn1 = BatchNorm1d(channels[0])

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    dilation=2,
                    scale=8,
                )
            )

        # Multi-layer feature aggregation (concatenate all block outputs)
        feature_dim = sum(channels)

        # Attentive statistics pooling
        self.pool = AttentiveStatisticsPooling(channels[-1], bottleneck_dim=128)

        # Projection to embedding space
        self.linear = nn.Linear(channels[-1] * 2, embedding_dim, bias=True)
        self.bn_linear = BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_mels, time]
        Returns:
            embedding: [batch, embedding_dim] with L2 normalization
        """
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Statistics pooling
        x = self.pool(x)

        # Linear projection
        x = self.linear(x)
        x = self.bn_linear(x)

        # L2 normalization
        x = F.normalize(x, p=2, dim=1)

        return x


# ============================================================================
# Export Functions
# ============================================================================


def load_checkpoint(checkpoint_path: str) -> Tuple[EcapaTdnn, Dict]:
    """Load checkpoint and reconstruct model."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract config
    config = checkpoint.get("config", {})
    n_mels = config.get("n_mels", 80)
    embedding_dim = config.get("embedding_dim", 256)
    channels = config.get("channels", [1024, 1024, 1024, 1024, 1536])

    # Reconstruct model
    model = EcapaTdnn(n_mels=n_mels, embedding_dim=embedding_dim, channels=channels)

    # Load only encoder weights (skip classifier)
    encoder_state_dict = {}
    for key, value in checkpoint["model"].items():
        if not key.startswith("classifier."):
            encoder_state_dict[key] = value

    model.load_state_dict(encoder_state_dict, strict=False)
    model.eval()

    # Extract metadata
    metadata = {
        "n_mels": n_mels,
        "embedding_dim": embedding_dim,
        "channels": channels,
        "num_speakers": checkpoint.get("num_speakers", 0),
        "best_eer": checkpoint.get("best_eer", 0.0),
    }

    return model, metadata


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_safetensors(
    model: EcapaTdnn, metadata: Dict, output_path: str
) -> None:
    """Export model to safetensors format."""
    print(f"Exporting to safetensors: {output_path}...")

    try:
        from safetensors.torch import save_file
    except ImportError:
        print("ERROR: safetensors not installed. Install with: pip install safetensors")
        return

    # Get state dict
    state_dict = model.state_dict()

    # Create metadata JSON
    metadata_json = json.dumps(metadata)

    # Save with metadata
    save_file(state_dict, output_path, metadata={"config": metadata_json})

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Saved: {file_size:.2f} MB")


def export_onnx(
    model: EcapaTdnn, metadata: Dict, output_path: str, n_mels: int = 80
) -> None:
    """Export model to ONNX format."""
    print(f"Exporting to ONNX: {output_path}...")

    # Create dummy input: [batch=1, n_mels=80, time=16000] (1 second at 16kHz)
    dummy_input = torch.randn(1, n_mels, 16000)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={
                "input": {2: "time"},  # Allow variable time dimension
                "embedding": {},  # Fixed output shape [1, 256]
            },
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
        )
        print(f"  Saved: {Path(output_path).stat().st_size / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"ERROR exporting to ONNX: {e}")
        return

    # Validate ONNX model
    try:
        import onnxruntime as ort
        import onnx

        # Load and check model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX model structure valid")

        # Test inference
        sess = ort.InferenceSession(output_path)

        # Test with different time lengths
        for time_length in [8000, 16000, 24000]:
            test_input = np.random.randn(1, n_mels, time_length).astype(np.float32)
            outputs = sess.run(None, {"input": test_input})
            embedding = outputs[0]

            if embedding.shape == (1, metadata["embedding_dim"]):
                print(f"  ✓ Inference test passed (time={time_length}): output shape {embedding.shape}")
            else:
                print(f"  ✗ Output shape mismatch: expected (1, {metadata['embedding_dim']}), got {embedding.shape}")

    except ImportError:
        print("  WARNING: onnxruntime not installed for validation. Install with: pip install onnxruntime")
    except Exception as e:
        print(f"  ERROR validating ONNX model: {e}")


def print_summary(model: EcapaTdnn, metadata: Dict, output_dir: str) -> None:
    """Print export summary."""
    print("\n" + "=" * 70)
    print("SPEAKER ENCODER EXPORT SUMMARY")
    print("=" * 70)

    # Model info
    num_params = count_parameters(model)
    print(f"Model Architecture: ECAPA-TDNN")
    print(f"  Input features: {metadata['n_mels']} mel-spectrogram bins")
    print(f"  Embedding dimension: {metadata['embedding_dim']}")
    print(f"  Channel layers: {metadata['channels']}")
    print(f"  Total parameters: {num_params:,}")

    # Training info
    print(f"\nTraining Info:")
    print(f"  Number of speakers: {metadata['num_speakers']}")
    print(f"  Best EER: {metadata['best_eer']:.4f}")

    # File sizes
    output_dir_path = Path(output_dir)
    safetensors_path = output_dir_path / "speaker_encoder.safetensors"
    onnx_path = output_dir_path / "speaker_encoder.onnx"

    if safetensors_path.exists():
        size_mb = safetensors_path.stat().st_size / (1024 * 1024)
        print(f"\nExported Files:")
        print(f"  safetensors: {safetensors_path.name} ({size_mb:.2f} MB)")
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX: {onnx_path.name} ({size_mb:.2f} MB)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Export speaker encoder to safetensors and ONNX formats."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to speaker encoder checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for exported models",
    )
    parser.add_argument(
        "--skip-safetensors",
        action="store_true",
        help="Skip safetensors export",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    model, metadata = load_checkpoint(args.checkpoint)

    # Export formats
    if not args.skip_safetensors:
        safetensors_path = output_dir / "speaker_encoder.safetensors"
        export_safetensors(model, metadata, str(safetensors_path))

    if not args.skip_onnx:
        onnx_path = output_dir / "speaker_encoder.onnx"
        export_onnx(model, metadata, str(onnx_path), n_mels=metadata["n_mels"])

    # Print summary
    print_summary(model, metadata, str(output_dir))


if __name__ == "__main__":
    main()
