"""Post-training quantization for Sonata models.

Stores actual quantized integer values + scale factors (not dequantized f32),
matching the Rust QuantConfig convention in sonata-common/src/quantization.rs.

Output format per quantized tensor:
  - "{name}.qweight": int8 tensor (8-bit) or packed uint8 tensor (4-bit)
  - "{name}.scale": f32 scale factors per group
  - Metadata: bits, group_size, symmetric

Unquantized tensors (biases, norms, embeddings <group_size) are kept as f32.

Usage:
    python train/quantize.py \
      --input models/sonata-codec.safetensors \
      --output models/sonata-codec-q4.safetensors \
      --bits 4

    python train/quantize.py \
      --input models/sonata-cfm.safetensors \
      --output models/sonata-cfm-q8.safetensors \
      --bits 8
"""
import argparse
import json
import torch
import numpy as np
from safetensors.torch import load_file, save_file


def quantize_tensor_int8(tensor: torch.Tensor, group_size: int = 128) -> dict:
    """Quantize a 2D+ tensor to int8 with group-wise symmetric scaling.

    Args:
        tensor: f32 tensor (must be 2D+)
        group_size: number of elements per quantization group

    Returns:
        dict with 'qweight' (int8) and 'scale' (f32)
    """
    original_shape = tensor.shape
    flat = tensor.reshape(-1)
    numel = flat.numel()

    # Pad to multiple of group_size
    pad_size = (group_size - numel % group_size) % group_size
    if pad_size > 0:
        flat = torch.cat([flat, torch.zeros(pad_size)])

    groups = flat.reshape(-1, group_size)  # [num_groups, group_size]

    # Symmetric quantization: scale = max_abs / 127
    max_val = groups.abs().max(dim=-1, keepdim=True).values
    scale = max_val / 127.0
    scale = scale.clamp(min=1e-8)

    # Quantize to int8 range [-127, 127]
    quantized = (groups / scale).round().clamp(-127, 127).to(torch.int8)

    # Remove padding
    quantized = quantized.reshape(-1)[:numel].reshape(original_shape)

    return {
        'qweight': quantized,
        'scale': scale.squeeze(-1),
    }


def quantize_tensor_int4(tensor: torch.Tensor, group_size: int = 128) -> dict:
    """Quantize a 2D+ tensor to int4 packed into uint8.

    Two int4 values are packed into each uint8 byte:
      byte = (high_nibble << 4) | (low_nibble & 0xF)

    Signed int4 range: [-8, 7], stored as unsigned [0, 15] with offset.

    Args:
        tensor: f32 tensor (must be 2D+)
        group_size: number of elements per quantization group

    Returns:
        dict with 'qweight' (uint8, packed) and 'scale' (f32)
    """
    original_shape = tensor.shape
    flat = tensor.reshape(-1)
    numel = flat.numel()

    # Pad to multiple of group_size (and even number for packing)
    pad_to = group_size
    pad_size = (pad_to - numel % pad_to) % pad_to
    if numel % 2 != 0:
        pad_size = max(pad_size, 1)
    if (numel + pad_size) % 2 != 0:
        pad_size += 1
    if pad_size > 0:
        flat = torch.cat([flat, torch.zeros(pad_size)])

    padded_numel = flat.numel()
    groups = flat.reshape(-1, group_size)

    # Symmetric quantization: scale = max_abs / 7
    max_val = groups.abs().max(dim=-1, keepdim=True).values
    scale = max_val / 7.0
    scale = scale.clamp(min=1e-8)

    # Quantize to signed int4 [-8, 7]
    quantized = (groups / scale).round().clamp(-8, 7)

    # Convert to unsigned [0, 15] for packing
    unsigned = (quantized + 8).to(torch.uint8).reshape(-1)  # [padded_numel]

    # Pack two int4 values into each uint8
    packed = torch.zeros(padded_numel // 2, dtype=torch.uint8)
    packed = (unsigned[0::2] << 4) | (unsigned[1::2] & 0x0F)

    return {
        'qweight': packed,
        'scale': scale.squeeze(-1),
        'original_shape': list(original_shape),
        'original_numel': numel,
    }


def quantize_model(input_path: str, output_path: str, bits: int = 4,
                   group_size: int = 128):
    """Quantize all eligible tensors in a safetensors file.

    Eligible: 2D+ tensors with numel >= group_size (typically weight matrices).
    Skipped: 1D tensors (biases, norms), small tensors, embeddings.
    """
    state_dict = load_file(input_path)

    quantized_dict = {}
    metadata = {
        'quantization.bits': str(bits),
        'quantization.group_size': str(group_size),
        'quantization.symmetric': 'true',
    }

    total_original = 0
    total_quantized = 0
    quantized_keys = []

    for name, tensor in state_dict.items():
        total_original += tensor.numel() * 4  # f32 = 4 bytes

        if tensor.ndim >= 2 and tensor.numel() >= group_size:
            if bits == 8:
                result = quantize_tensor_int8(tensor, group_size)
                quantized_dict[f"{name}.qweight"] = result['qweight']
                quantized_dict[f"{name}.scale"] = result['scale']
                total_quantized += tensor.numel()  # 1 byte per element
                total_quantized += result['scale'].numel() * 4  # f32 scales
            elif bits == 4:
                result = quantize_tensor_int4(tensor, group_size)
                quantized_dict[f"{name}.qweight"] = result['qweight']
                quantized_dict[f"{name}.scale"] = result['scale']
                # Store original shape in metadata for unpacking
                metadata[f"shape.{name}"] = json.dumps(result['original_shape'])
                total_quantized += result['qweight'].numel()  # packed bytes
                total_quantized += result['scale'].numel() * 4  # f32 scales
            quantized_keys.append(name)
            print(f"  Quantized: {name} {list(tensor.shape)} → {bits}-bit")
        else:
            quantized_dict[name] = tensor
            total_quantized += tensor.numel() * 4
            print(f"  Kept f32:  {name} {list(tensor.shape)}")

    metadata['quantized_keys'] = json.dumps(quantized_keys)

    save_file(quantized_dict, output_path, metadata=metadata)
    ratio = total_original / max(total_quantized, 1)
    print(f"\nCompression: {total_original/1e6:.1f}MB → {total_quantized/1e6:.1f}MB ({ratio:.1f}x)")
    print(f"Quantized {len(quantized_keys)} tensors, kept {len(state_dict) - len(quantized_keys)} as f32")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantize Sonata model weights')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--bits', type=int, default=4, choices=[4, 8])
    parser.add_argument('--group_size', type=int, default=128)
    args = parser.parse_args()
    quantize_model(args.input, args.output, args.bits, args.group_size)
