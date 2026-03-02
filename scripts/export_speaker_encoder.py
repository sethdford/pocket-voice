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
# Must exactly match train/train_speaker_encoder.py for checkpoint loading.
# ============================================================================


class BatchNorm1d(nn.Module):
    """Wrapper around nn.BatchNorm1d with consistent initialization."""

    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""

    def __init__(self, channels: int, se_channels: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(channels, se_channels)
        self.fc2 = nn.Linear(se_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        se = x.mean(dim=2)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se.unsqueeze(2)


class Res2NetBlock(nn.Module):
    """Res2Net block with multi-branch dilated convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, scale: int = 8):
        super().__init__()
        self.scale = scale
        self.branch_channels = out_channels // scale

        self.conv1x1_in = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn1 = BatchNorm1d(out_channels)

        self.branches = nn.ModuleList()
        for i in range(scale - 1):
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(self.branch_channels, self.branch_channels, kernel_size,
                              padding=(kernel_size - 1) // 2 * dilation, dilation=dilation),
                    BatchNorm1d(self.branch_channels),
                    nn.ReLU(inplace=True)
                )
            )

        self.conv1x1_out = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn2 = BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1x1_in(x)
        x = self.bn1(x)
        x = F.relu(x)

        xs = torch.chunk(x, self.scale, dim=1)
        ys = [xs[0]]

        for i, branch in enumerate(self.branches):
            y = branch(xs[i + 1] + ys[-1])
            ys.append(y)

        x = torch.cat(ys, dim=1)
        x = self.conv1x1_out(x)
        x = self.bn2(x)

        if residual.shape[1] != x.shape[1]:
            residual = F.pad(residual, (0, 0, 0, x.shape[1] - residual.shape[1]))

        return F.relu(x + residual)


class SERes2NetBlock(nn.Module):
    """SE-Res2Net block combining Res2Net with squeeze-and-excitation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, scale: int = 8, se_channels: int = 128):
        super().__init__()
        self.res2net = Res2NetBlock(in_channels, out_channels, kernel_size,
                                     dilation, scale)
        self.se = SEBlock(out_channels, se_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res2net(x)
        x = self.se(x)
        return x


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling with channel-wise attention."""

    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(attention_channels, channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = torch.softmax(self.attention(x), dim=2)
        weighted_x = x * attention
        mean = weighted_x.sum(dim=2)
        weighted_var = ((x ** 2) * attention).sum(dim=2) - mean ** 2
        std = torch.sqrt(weighted_var.clamp(min=1e-9))
        pooled = torch.cat([mean, std], dim=1)
        return pooled


class EcapaTdnn(nn.Module):
    """ECAPA-TDNN speaker encoder with multi-layer feature aggregation."""

    def __init__(self, n_mels: int = 80, embedding_dim: int = 256,
                 channels: List[int] = None, kernel_sizes: List[int] = None,
                 dilations: List[int] = None, scale: int = 8,
                 se_channels: int = 128, attention_channels: int = 128):
        super().__init__()

        if channels is None:
            channels = [1024, 1024, 1024, 1024, 1536]
        if kernel_sizes is None:
            kernel_sizes = [5, 3, 3, 3, 1]
        if dilations is None:
            dilations = [1, 2, 3, 4, 1]

        self.embedding_dim = embedding_dim

        # Input convolution
        self.input_conv = nn.Conv1d(n_mels, channels[0], kernel_size=5,
                                     padding=2, stride=1)
        self.input_bn = BatchNorm1d(channels[0])

        # SE-Res2Net blocks (N-1 blocks for N channel specs)
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(channels[i], channels[i + 1], kernel_sizes[i + 1],
                              dilations[i + 1], scale=scale, se_channels=se_channels)
            )

        # Multi-layer feature aggregation
        total_channels = sum(channels)
        self.mfa_conv = nn.Conv1d(total_channels, channels[-1], kernel_size=1)
        self.mfa_bn = BatchNorm1d(channels[-1])

        # Attentive statistics pooling
        self.asp = AttentiveStatisticsPooling(channels[-1],
                                               attention_channels=attention_channels)

        # Final linear layer to embedding
        self.final_linear = nn.Linear(channels[-1] * 2, embedding_dim)
        self.final_bn = BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input processing
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)

        # Collect features from all blocks (multi-layer feature aggregation)
        features = [x]
        for block in self.blocks:
            x = block(x)
            features.append(x)

        # Concatenate all features and reduce
        cat = torch.cat(features, dim=1)
        x = self.mfa_conv(cat)
        x = self.mfa_bn(x)
        x = F.relu(x)

        # Attentive pooling
        x = self.asp(x)

        # Final linear projection
        x = self.final_linear(x)
        x = self.final_bn(x)

        # L2 normalization
        return F.normalize(x, p=2, dim=1)


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
    kernel_sizes = config.get("kernel_sizes", [5, 3, 3, 3, 1])
    dilations = config.get("dilations", [1, 2, 3, 4, 1])
    scale = config.get("scale", 8)
    se_channels = config.get("se_channels", 128)
    attention_channels = config.get("attention_channels", 128)

    # Reconstruct model with full config
    model = EcapaTdnn(
        n_mels=n_mels, embedding_dim=embedding_dim, channels=channels,
        kernel_sizes=kernel_sizes, dilations=dilations, scale=scale,
        se_channels=se_channels, attention_channels=attention_channels,
    )

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

    # Create dummy input: [batch=1, n_mels=80, time=100] (100 mel frames)
    dummy_input = torch.randn(1, n_mels, 100)

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
        for time_length in [50, 100, 200]:
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
