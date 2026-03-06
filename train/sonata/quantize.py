"""BitTTS-style quantization-aware training (QAT) for Sonata models.

Ternary quantization: weights to {-1, 0, +1} × scale_factor.
~83% size reduction (log2(3) ≈ 1.58 bits/weight) with minimal quality loss.
Reference: BitTTS (arXiv:2506.03515)

Usage:
  QAT with Sonata LM:
    python train_lm.py --data ... --qat --qat-exclude "embedding,semantic_head"

  QAT with Sonata Flow:
    python train_flow.py --data ... --qat --qat-exclude "semantic_emb,output_proj"

Import:
  from quantize import quantize_model, compute_compression_stats, export_ternary
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Ternary Quantization ────────────────────────────────────────────────────

class TernaryQuantizer(nn.Module):
    """BitTTS-style ternary quantization with STE (Straight-Through Estimator).

    For each weight tensor: w_q = scale * sign(w) * (|w| > threshold)
    where scale = mean(|w|), threshold = 0.7 * scale (TWN-style adaptive threshold).
    STE: forward uses quantized, backward flows through full precision.
    """

    @staticmethod
    def quantize(w: torch.Tensor) -> torch.Tensor:
        """Quantize weights to ternary {-1, 0, +1} with STE."""
        scale = w.abs().mean().clamp(min=1e-8)
        threshold = 0.7 * scale
        w_ternary = torch.sign(w) * (w.abs() > threshold).float()
        w_q = scale * w_ternary
        return w + (w_q - w).detach()

    @staticmethod
    def quantize_with_scale(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize and return (quantized_tensor, scale) for export."""
        scale = w.abs().mean().clamp(min=1e-8)
        threshold = 0.7 * scale
        w_ternary = torch.sign(w) * (w.abs() > threshold).float()
        w_q = scale * w_ternary
        return w_q, scale


# ─── Quantized Linear Layer ──────────────────────────────────────────────────

class QuantizedLinear(nn.Module):
    """Linear layer with ternary weight quantization during forward pass."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = TernaryQuantizer.quantize(self.weight)
        return F.linear(x, w_q, self.bias)


# ─── Model Conversion ───────────────────────────────────────────────────────

def quantize_model(
    model: nn.Module,
    exclude_patterns: Optional[List[str]] = None,
) -> nn.Module:
    """Replace nn.Linear with QuantizedLinear for QAT.

    exclude_patterns: list of name substrings to skip (e.g. ["embedding", "head"]).
    Matching is case-insensitive; a module is excluded if any pattern appears in its
    full qualified name.
    """
    exclude_patterns = exclude_patterns or []
    exclude_lower = [p.lower() for p in exclude_patterns]

    def _should_exclude(name: str) -> bool:
        return any(pat in name.lower() for pat in exclude_lower)

    replacements: List[Tuple[nn.Module, str, nn.Linear]] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not _should_exclude(name):
            replacements.append((module, name, module))

    # Replace from leaf to root so we don't double-replace
    for child_module, full_name, linear in replacements:
        parent_name = full_name.rsplit(".", 1)[0] if "." in full_name else ""
        child_name = full_name.split(".")[-1]

        qlinear = QuantizedLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
        )
        qlinear.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            qlinear.bias.data = linear.bias.data.clone()

        if parent_name:
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, qlinear)
        else:
            setattr(model, child_name, qlinear)

    return model


# ─── Compression Statistics ────────────────────────────────────────────────

def compute_compression_stats(model: nn.Module) -> Dict[str, Any]:
    """Compute storage savings from ternary quantization.

    Returns:
        original_mb: FP32 size in MB
        quantized_mb: Ternary size in MB (2 bits/weight + float32 scales)
        compression_ratio: original_mb / quantized_mb
    """
    bytes_per_param = 4  # float32
    bits_per_ternary = 1.5849625  # log2(3)
    bytes_per_ternary = bits_per_ternary / 8
    bytes_per_scale = 4  # float32 per layer

    original_bytes = 0
    quantized_bytes = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, QuantizedLinear)):
            w = module.weight
            n = w.numel()
            original_bytes += n * bytes_per_param
            quantized_bytes += n * bytes_per_ternary + bytes_per_scale
            if module.bias is not None:
                b = module.bias.numel()
                original_bytes += b * bytes_per_param
                quantized_bytes += b * bytes_per_param  # bias stays fp32

    return {
        "original_mb": original_bytes / (1024 * 1024),
        "quantized_mb": quantized_bytes / (1024 * 1024),
        "compression_ratio": original_bytes / quantized_bytes if quantized_bytes > 0 else 1.0,
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
    }


# ─── Export ─────────────────────────────────────────────────────────────────

def _pack_ternary(values: torch.Tensor) -> bytes:
    """Pack ternary values {-1, 0, +1} as 2 bits each. 4 values per byte."""
    # Map: -1 -> 0, 0 -> 1, +1 -> 2
    v = values.flatten().to(torch.int64)
    codes = (v + 1).clamp(0, 2)
    n = codes.numel()
    padded = (n + 3) // 4 * 4
    codes = F.pad(codes, (0, padded - n), value=1)  # pad with 0 (code 1)
    codes = codes.view(-1, 4)
    packed = (codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6))
    return packed.to(torch.uint8).numpy().tobytes()


def _unpack_ternary(data: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    """Unpack 2-bit ternary values to tensor."""
    arr = bytearray(data)
    n_total = (len(arr) * 4 + 3) // 4 * 4
    codes = []
    for b in arr:
        for shift in range(0, 8, 2):
            c = (b >> shift) & 3
            codes.append(c)
    codes = codes[:shape[0] * shape[1]]
    v = torch.tensor(codes, dtype=torch.long)
    v = v - 1
    return v.reshape(shape).float()


def export_ternary(
    model: nn.Module,
    path: Union[str, Path],
) -> None:
    """Export ternary weights in compact format.

    Format per layer:
      - scale: float32 (4 bytes)
      - packed ternary: 2 bits per weight, 4 weights per byte

    Saves a directory with:
      - manifest.json: layer names, shapes, offsets
      - weights.bin: concatenated binary data
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {"layers": {}}
    chunks: List[bytes] = []
    offset = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, QuantizedLinear)):
            w = module.weight.data
            w_q, scale = TernaryQuantizer.quantize_with_scale(w)
            w_ternary = torch.sign(w_q) * (w_q.abs() > 1e-8).float()
            packed = _pack_ternary(w_ternary)
            scale_bytes = struct.pack("f", float(scale))

            layer_data = scale_bytes + packed
            chunks.append(layer_data)

            manifest["layers"][name] = {
                "shape": list(w.shape),
                "offset": offset,
                "scale_size": 4,
                "packed_size": len(packed),
            }
            offset += len(layer_data)

    with open(path / "weights.bin", "wb") as f:
        f.write(b"".join(chunks))

    with open(path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
