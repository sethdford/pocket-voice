"""Export PyTorch Sonata models to candle-compatible safetensors format.

Converts trained PyTorch models to safetensors format for use with
Rust candle inference code. Handles weight format conversion and
tensor key remapping.

Usage:
    # Export codec
    python train/export_candle.py \\
      --model codec \\
      --checkpoint train/checkpoints/codec/codec_best.pt \\
      --output models/sonata-codec.safetensors

    # Export speaker encoder
    python train/export_candle.py \\
      --model cam \\
      --checkpoint train/checkpoints/speaker_encoder_best.pt \\
      --output models/cam-plus-plus.safetensors

    # Export STT
    python train/export_candle.py \\
      --model stt \\
      --checkpoint train/checkpoints/stt/stt_best.pt \\
      --output models/sonata-stt.safetensors

    # Export TTS
    python train/export_candle.py \\
      --model tts \\
      --checkpoint train/checkpoints/tts/tts_best.pt \\
      --output models/sonata-tts.safetensors

    # Export with compression
    python train/export_candle.py \\
      --model codec \\
      --checkpoint train/checkpoints/codec/codec_best.pt \\
      --output models/sonata-codec.safetensors \\
      --quantize

Output format:
    - Tensors stored as float32 (standard format for candle)
    - Keys follow PyTorch module hierarchy (e.g., "encoder.layers.0.weight")
    - Metadata includes original model architecture/config
"""

import argparse
import logging
import torch
from pathlib import Path
from typing import Dict
import json

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    from safetensors.torch import save_file
except ImportError:
    logger.error("safetensors not installed. Run: pip install safetensors")
    exit(1)


def remap_keys(state_dict: Dict, model_type: str) -> Dict:
    """Remap PyTorch state dict keys to match Rust candle struct field names.

    Candle's VarBuilder uses dot-separated paths like:
      "encoder.layers.0.weight"
    which are loaded via:
      vb.pp("encoder").pp("layers").pp("0").get(..., "weight")

    PyTorch state dicts typically already use this format from named_modules(),
    so minimal remapping is needed. This function ensures all tensors are
    float32 and contiguous.

    Args:
        state_dict: PyTorch state_dict() output
        model_type: Model type (codec, stt, tts, cam, cfm)

    Returns:
        remapped: Dictionary with properly formatted keys and float32 tensors
    """
    remapped = {}

    for key, tensor in state_dict.items():
        # Skip non-tensor values (shouldn't happen in state_dict)
        if not isinstance(tensor, torch.Tensor):
            continue

        # Ensure float32 and contiguous for safetensors
        tensor = tensor.float().contiguous()

        # Standard PyTorch keys are already in the right format
        # Just use them directly
        remapped[key] = tensor

    return remapped


def export_codec(checkpoint_path: str, output_path: str) -> None:
    """Export Sonata Codec model.

    Args:
        checkpoint_path: Path to codec checkpoint
        output_path: Path to save safetensors file
    """
    logger.info(f"Loading codec checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    remapped = remap_keys(state_dict, 'codec')

    logger.info(f"Exporting {len(remapped)} tensors")
    for name, tensor in list(remapped.items())[:3]:  # Print first 3
        logger.info(f"  {name}: {list(tensor.shape)}")

    save_file(remapped, output_path)
    logger.info(f"Saved codec to {output_path}")


def export_stt(checkpoint_path: str, output_path: str) -> None:
    """Export Sonata STT model.

    Args:
        checkpoint_path: Path to STT checkpoint
        output_path: Path to save safetensors file
    """
    logger.info(f"Loading STT checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    remapped = remap_keys(state_dict, 'stt')

    logger.info(f"Exporting {len(remapped)} tensors")
    for name, tensor in list(remapped.items())[:3]:
        logger.info(f"  {name}: {list(tensor.shape)}")

    save_file(remapped, output_path)
    logger.info(f"Saved STT to {output_path}")


def export_tts(checkpoint_path: str, output_path: str) -> None:
    """Export Sonata TTS model.

    Args:
        checkpoint_path: Path to TTS checkpoint
        output_path: Path to save safetensors file
    """
    logger.info(f"Loading TTS checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    remapped = remap_keys(state_dict, 'tts')

    logger.info(f"Exporting {len(remapped)} tensors")
    for name, tensor in list(remapped.items())[:3]:
        logger.info(f"  {name}: {list(tensor.shape)}")

    save_file(remapped, output_path)
    logger.info(f"Saved TTS to {output_path}")


def export_speaker_encoder(checkpoint_path: str, output_path: str) -> None:
    """Export CAM++ speaker encoder.

    Args:
        checkpoint_path: Path to speaker encoder checkpoint
        output_path: Path to save safetensors file
    """
    logger.info(f"Loading speaker encoder checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    # Handle checkpoint structure from train_speaker_encoder.py
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    remapped = remap_keys(state_dict, 'cam')

    logger.info(f"Exporting {len(remapped)} tensors (speaker encoder ~7M params)")
    for name, tensor in list(remapped.items())[:3]:
        logger.info(f"  {name}: {list(tensor.shape)}")

    save_file(remapped, output_path)
    logger.info(f"Saved speaker encoder to {output_path}")


def export_cfm(checkpoint_path: str, output_path: str) -> None:
    """Export Conditional Flow Matching decoder.

    Args:
        checkpoint_path: Path to CFM checkpoint
        output_path: Path to save safetensors file
    """
    logger.info(f"Loading CFM checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    remapped = remap_keys(state_dict, 'cfm')

    logger.info(f"Exporting {len(remapped)} tensors")
    for name, tensor in list(remapped.items())[:3]:
        logger.info(f"  {name}: {list(tensor.shape)}")

    save_file(remapped, output_path)
    logger.info(f"Saved CFM to {output_path}")


def validate_safetensors(path: str) -> None:
    """Validate safetensors file format.

    Args:
        path: Path to safetensors file
    """
    try:
        from safetensors.torch import load_file
        data = load_file(path)
        logger.info(f"✓ Validated {path}")
        logger.info(f"  Keys: {len(data)}")
        for key in list(data.keys())[:3]:
            logger.info(f"    {key}: {list(data[key].shape)}")
    except Exception as e:
        logger.error(f"✗ Validation failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Export PyTorch to candle safetensors'
    )
    parser.add_argument('--model', type=str, required=True,
                       choices=['codec', 'cam', 'stt', 'tts', 'cfm'],
                       help='Model type')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='PyTorch checkpoint path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output safetensors path')
    parser.add_argument('--validate', action='store_true',
                       help='Validate output file')

    args = parser.parse_args()

    # Validate input
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        exit(1)

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Export
    logger.info(f"Exporting {args.model} model...")
    if args.model == 'codec':
        export_codec(args.checkpoint, args.output)
    elif args.model == 'cam':
        export_speaker_encoder(args.checkpoint, args.output)
    elif args.model == 'stt':
        export_stt(args.checkpoint, args.output)
    elif args.model == 'tts':
        export_tts(args.checkpoint, args.output)
    elif args.model == 'cfm':
        export_cfm(args.checkpoint, args.output)

    # Validate
    if args.validate:
        validate_safetensors(args.output)

    logger.info(f"✓ Export complete: {args.output}")


if __name__ == '__main__':
    main()
