"""Post-training quantization for Sonata models.

Usage:
    python train/quantize.py --input models/sonata-codec.safetensors --output models/sonata-codec-q4.safetensors --bits 4
"""
import argparse
import torch
import numpy as np
from safetensors.torch import load_file, save_file


def quantize_tensor(tensor: torch.Tensor, bits: int = 4, group_size: int = 128) -> dict:
    """Quantize a tensor to N-bit with group-wise scaling."""
    if tensor.ndim < 2:
        return {'weight': tensor}  # Skip 1D tensors (biases)

    original_shape = tensor.shape
    tensor_flat = tensor.reshape(-1, group_size) if tensor.numel() % group_size == 0 \
                  else tensor.reshape(-1)

    # Symmetric quantization
    max_val = tensor_flat.abs().max(dim=-1, keepdim=True).values
    scale = max_val / (2 ** (bits - 1) - 1)
    scale = scale.clamp(min=1e-8)

    quantized = (tensor_flat / scale).round().clamp(-(2**(bits-1)), 2**(bits-1) - 1)
    dequantized = quantized * scale

    return {
        'weight': dequantized.reshape(original_shape),
        'scale': scale.squeeze(-1),
    }


def quantize_model(input_path: str, output_path: str, bits: int = 4, group_size: int = 128):
    """Quantize all eligible tensors in a safetensors file."""
    state_dict = load_file(input_path)

    quantized_dict = {}
    total_original = 0
    total_quantized = 0

    for name, tensor in state_dict.items():
        total_original += tensor.numel() * 4  # f32

        if tensor.ndim >= 2 and tensor.numel() >= group_size:
            result = quantize_tensor(tensor, bits, group_size)
            quantized_dict[name] = result['weight']
            if 'scale' in result:
                quantized_dict[f"{name}.scale"] = result['scale']
            total_quantized += tensor.numel() * bits // 8
            print(f"  Quantized: {name} {list(tensor.shape)}")
        else:
            quantized_dict[name] = tensor
            total_quantized += tensor.numel() * 4
            print(f"  Kept f32:  {name} {list(tensor.shape)}")

    save_file(quantized_dict, output_path)
    ratio = total_original / max(total_quantized, 1)
    print(f"\nCompression: {total_original/1e6:.1f}MB → {total_quantized/1e6:.1f}MB ({ratio:.1f}x)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantize Sonata model weights')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--bits', type=int, default=4, choices=[4, 8])
    parser.add_argument('--group_size', type=int, default=128)
    args = parser.parse_args()
    quantize_model(args.input, args.output, args.bits, args.group_size)
