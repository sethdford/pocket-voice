"""Export PyTorch Sonata models to candle-compatible safetensors format.

Converts trained PyTorch models to safetensors format for use with
Rust candle inference code. Handles weight format conversion, tensor
key remapping, and shape verification against expected Rust architectures.

Usage:
    # Export codec
    python train/export_candle.py \
      --model codec \
      --checkpoint train/checkpoints/codec/codec_best.pt \
      --output models/sonata-codec.safetensors

    # Export with verification
    python train/export_candle.py \
      --model codec \
      --checkpoint train/checkpoints/codec/codec_best.pt \
      --output models/sonata-codec.safetensors \
      --verify
"""

import argparse
import logging
import re
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    from safetensors.torch import save_file, load_file
except ImportError:
    logger.error("safetensors not installed. Run: pip install safetensors")
    exit(1)


# ============================================================
#  Expected key shapes per model type
#  Derived from Rust VarBuilder paths in each crate
# ============================================================

def get_expected_shapes_codec() -> Dict[str, Optional[List[int]]]:
    """Expected tensor shapes for sonata-codec.

    Based on crates/sonata-codec/src/{encoder,decoder,snake,quantizer}.rs
    Encoder strides: [8,5,4,3], channels: [1,64,128,256,512]
    Decoder strides: [3,4,5,8], channels: [512,256,128,64,1]
    RVQ: 8 codebooks, 1024 entries, 128-dim with project_in/out (512↔128)
    """
    shapes = {}

    # Encoder: 4 conv layers with Snake activation (no norms)
    enc_channels = [1, 64, 128, 256, 512]
    enc_strides = [8, 5, 4, 3]
    for i in range(4):
        c_in, c_out = enc_channels[i], enc_channels[i + 1]
        k = enc_strides[i] * 2
        shapes[f'encoder.conv_layers.{i}.weight'] = [c_out, c_in, k]
        shapes[f'encoder.conv_layers.{i}.bias'] = [c_out]
        shapes[f'encoder.snake_layers.{i}.alpha'] = [c_out]

    # Decoder: 4 transposed conv layers, Snake on first 3, Tanh on final
    dec_channels = [512, 256, 128, 64, 1]
    dec_strides = [3, 4, 5, 8]
    for i in range(4):
        c_in, c_out = dec_channels[i], dec_channels[i + 1]
        k = dec_strides[i] * 2
        shapes[f'decoder.conv_layers.{i}.weight'] = [c_in, c_out, k]
        shapes[f'decoder.conv_layers.{i}.bias'] = [c_out]
        if i < 3:  # No snake on final layer (uses Tanh)
            shapes[f'decoder.snake_layers.{i}.alpha'] = [c_out]

    # RVQ: shared project_in/out + 8 codebooks
    shapes['rvq.project_in.weight'] = [128, 512]
    shapes['rvq.project_in.bias'] = [128]
    shapes['rvq.project_out.weight'] = [512, 128]
    shapes['rvq.project_out.bias'] = [512]
    for i in range(8):
        shapes[f'rvq.codebooks.{i}.embeddings'] = [1024, 128]

    return shapes


def get_expected_shapes_cam() -> Dict[str, Optional[List[int]]]:
    """Expected tensor shapes for CAM++ speaker encoder.

    Based on crates/sonata-cam/src/{lib,frontend,cam_block,pooling}.rs
    Frontend: 80→64→128→256 (3 Conv1d layers, kernel=3, padding=1)
    6 CAMBlocks: dim=256, heads=8
    AttentiveStatsPooling → Linear(512→192)
    """
    shapes = {}

    # MultiScaleFrontend: 3 conv layers
    front_channels = [80, 64, 128, 256]
    for i in range(3):
        c_in, c_out = front_channels[i], front_channels[i + 1]
        shapes[f'frontend.convs.{i}.weight'] = [c_out, c_in, 3]
        shapes[f'frontend.convs.{i}.bias'] = [c_out]
        shapes[f'frontend.norms.{i}.weight'] = [c_out]
        shapes[f'frontend.norms.{i}.bias'] = [c_out]

    # 6 CAM blocks
    dim = 256
    for i in range(6):
        # Attention
        shapes[f'blocks.{i}.q_proj.weight'] = [dim, dim]
        shapes[f'blocks.{i}.q_proj.bias'] = [dim]
        shapes[f'blocks.{i}.k_proj.weight'] = [dim, dim]
        shapes[f'blocks.{i}.k_proj.bias'] = [dim]
        shapes[f'blocks.{i}.v_proj.weight'] = [dim, dim]
        shapes[f'blocks.{i}.v_proj.bias'] = [dim]
        shapes[f'blocks.{i}.out_proj.weight'] = [dim, dim]
        shapes[f'blocks.{i}.out_proj.bias'] = [dim]
        # FFN (Conv1d: dim→4*dim→dim)
        shapes[f'blocks.{i}.ffn.0.weight'] = [dim * 4, dim, 3]
        shapes[f'blocks.{i}.ffn.0.bias'] = [dim * 4]
        shapes[f'blocks.{i}.ffn.2.weight'] = [dim, dim * 4, 3]
        shapes[f'blocks.{i}.ffn.2.bias'] = [dim]
        # Layer norms
        shapes[f'blocks.{i}.norm1.weight'] = [dim]
        shapes[f'blocks.{i}.norm1.bias'] = [dim]
        shapes[f'blocks.{i}.norm2.weight'] = [dim]
        shapes[f'blocks.{i}.norm2.bias'] = [dim]

    # AttentiveStatsPooling
    shapes['pooling.attention.weight'] = [1, dim]
    shapes['pooling.attention.bias'] = [1]
    shapes['pooling.output_proj.weight'] = [192, dim * 2]
    shapes['pooling.output_proj.bias'] = [192]

    return shapes


def get_expected_shapes_stt() -> Dict[str, Optional[List[int]]]:
    """Expected tensor shapes for Sonata STT (Conformer).

    Based on crates/sonata-stt/src/{lib,conformer,ctc}.rs
    input_proj: Linear(80, 512), 12 ConformerBlocks, output_proj: Linear(512, 32000)
    Each block: MHSA (Q/K/V separate) + Conv1d(31) + SwiGLU FFN
    """
    shapes = {}
    dim = 512
    ffn_dim = 2048

    shapes['input_proj.weight'] = [dim, 80]
    shapes['input_proj.bias'] = [dim]

    for i in range(12):
        # MHSA: separate Q, K, V projections
        shapes[f'conformer_blocks.{i}.mhsa.q_proj.weight'] = [dim, dim]
        shapes[f'conformer_blocks.{i}.mhsa.q_proj.bias'] = [dim]
        shapes[f'conformer_blocks.{i}.mhsa.k_proj.weight'] = [dim, dim]
        shapes[f'conformer_blocks.{i}.mhsa.k_proj.bias'] = [dim]
        shapes[f'conformer_blocks.{i}.mhsa.v_proj.weight'] = [dim, dim]
        shapes[f'conformer_blocks.{i}.mhsa.v_proj.bias'] = [dim]
        shapes[f'conformer_blocks.{i}.mhsa.out_proj.weight'] = [dim, dim]
        shapes[f'conformer_blocks.{i}.mhsa.out_proj.bias'] = [dim]
        # Conv1d (kernel=31, padding=15)
        shapes[f'conformer_blocks.{i}.conv.weight'] = [dim, dim, 31]
        shapes[f'conformer_blocks.{i}.conv.bias'] = [dim]
        # SwiGLU FFN
        shapes[f'conformer_blocks.{i}.ffn.gate.weight'] = [ffn_dim, dim]
        shapes[f'conformer_blocks.{i}.ffn.gate.bias'] = [ffn_dim]
        shapes[f'conformer_blocks.{i}.ffn.up.weight'] = [ffn_dim, dim]
        shapes[f'conformer_blocks.{i}.ffn.up.bias'] = [ffn_dim]
        shapes[f'conformer_blocks.{i}.ffn.down.weight'] = [dim, ffn_dim]
        shapes[f'conformer_blocks.{i}.ffn.down.bias'] = [dim]
        # LayerNorms
        shapes[f'conformer_blocks.{i}.norm1.weight'] = [dim]
        shapes[f'conformer_blocks.{i}.norm1.bias'] = [dim]
        shapes[f'conformer_blocks.{i}.norm2.weight'] = [dim]
        shapes[f'conformer_blocks.{i}.norm2.bias'] = [dim]

    shapes['output_proj.weight'] = [32000, dim]
    shapes['output_proj.bias'] = [32000]

    return shapes


def get_expected_shapes_tts() -> Dict[str, Optional[List[int]]]:
    """Expected tensor shapes for Sonata TTS.

    Based on crates/sonata-tts/src/{lib,text_encoder,transformer,emotion,nonverbal}.rs
    TextEncoder: Embedding(32000, 512) + sinusoidal PE
    EmotionStyleEncoder: Embedding(64, 192) + Linear(192, 192)
    NonverbalEncoder: Embedding(24, 512)
    12 TTSTransformerLayers: attention + speaker_adain(512,192) + emotion_adain(512,192) + SwiGLU(512,2048)
    output_proj: Linear(512, 1024)
    """
    shapes = {}
    dim = 512
    ffn_dim = 2048

    # TextEncoder
    shapes['text_encoder.embedding.weight'] = [32000, dim]

    # EmotionStyleEncoder
    shapes['emotion_encoder.embedding.weight'] = [64, 192]
    shapes['emotion_encoder.proj.weight'] = [192, 192]
    shapes['emotion_encoder.proj.bias'] = [192]

    # NonverbalEncoder
    shapes['nonverbal_encoder.embedding.weight'] = [24, dim]

    # 12 TTSTransformerLayers
    for i in range(12):
        # Attention (Q/K/V)
        shapes[f'layers.{i}.q_proj.weight'] = [dim, dim]
        shapes[f'layers.{i}.q_proj.bias'] = [dim]
        shapes[f'layers.{i}.k_proj.weight'] = [dim, dim]
        shapes[f'layers.{i}.k_proj.bias'] = [dim]
        shapes[f'layers.{i}.v_proj.weight'] = [dim, dim]
        shapes[f'layers.{i}.v_proj.bias'] = [dim]
        shapes[f'layers.{i}.out_proj.weight'] = [dim, dim]
        shapes[f'layers.{i}.out_proj.bias'] = [dim]
        # Speaker AdaIN
        shapes[f'layers.{i}.speaker_adain.gamma_proj.weight'] = [dim, 192]
        shapes[f'layers.{i}.speaker_adain.gamma_proj.bias'] = [dim]
        shapes[f'layers.{i}.speaker_adain.beta_proj.weight'] = [dim, 192]
        shapes[f'layers.{i}.speaker_adain.beta_proj.bias'] = [dim]
        # Emotion AdaIN
        shapes[f'layers.{i}.emotion_adain.gamma_proj.weight'] = [dim, 192]
        shapes[f'layers.{i}.emotion_adain.gamma_proj.bias'] = [dim]
        shapes[f'layers.{i}.emotion_adain.beta_proj.weight'] = [dim, 192]
        shapes[f'layers.{i}.emotion_adain.beta_proj.bias'] = [dim]
        # SwiGLU FFN
        shapes[f'layers.{i}.ffn.gate.weight'] = [ffn_dim, dim]
        shapes[f'layers.{i}.ffn.gate.bias'] = [ffn_dim]
        shapes[f'layers.{i}.ffn.up.weight'] = [ffn_dim, dim]
        shapes[f'layers.{i}.ffn.up.bias'] = [ffn_dim]
        shapes[f'layers.{i}.ffn.down.weight'] = [dim, ffn_dim]
        shapes[f'layers.{i}.ffn.down.bias'] = [dim]
        # LayerNorm
        shapes[f'layers.{i}.norm.weight'] = [dim]
        shapes[f'layers.{i}.norm.bias'] = [dim]

    shapes['output_proj.weight'] = [1024, dim]
    shapes['output_proj.bias'] = [1024]

    return shapes


def get_expected_shapes_cfm() -> Dict[str, Optional[List[int]]]:
    """Expected tensor shapes for Sonata CFM.

    Based on crates/sonata-cfm/src/{lib,dit,ode}.rs
    input_proj: Linear(80, 512)
    12 DiTBlocks: attn + time_adain(512,256) + spk_adain(512,192) + SwiGLU(512,2048)
    output_proj: Linear(512, 80)
    TimeEmbedding: mlp1(256,256) + mlp2(256,256)
    """
    shapes = {}
    dim = 512
    ffn_dim = 2048
    time_dim = 256

    shapes['input_proj.weight'] = [dim, 80]
    shapes['input_proj.bias'] = [dim]

    for i in range(12):
        # Attention (Linear)
        shapes[f'blocks.{i}.attn.weight'] = [dim, dim]
        shapes[f'blocks.{i}.attn.bias'] = [dim]
        # Time AdaIN
        shapes[f'blocks.{i}.time_adain.gamma_proj.weight'] = [dim, time_dim]
        shapes[f'blocks.{i}.time_adain.gamma_proj.bias'] = [dim]
        shapes[f'blocks.{i}.time_adain.beta_proj.weight'] = [dim, time_dim]
        shapes[f'blocks.{i}.time_adain.beta_proj.bias'] = [dim]
        # Speaker AdaIN
        shapes[f'blocks.{i}.spk_adain.gamma_proj.weight'] = [dim, 192]
        shapes[f'blocks.{i}.spk_adain.gamma_proj.bias'] = [dim]
        shapes[f'blocks.{i}.spk_adain.beta_proj.weight'] = [dim, 192]
        shapes[f'blocks.{i}.spk_adain.beta_proj.bias'] = [dim]
        # SwiGLU FFN
        shapes[f'blocks.{i}.ffn.gate.weight'] = [ffn_dim, dim]
        shapes[f'blocks.{i}.ffn.gate.bias'] = [ffn_dim]
        shapes[f'blocks.{i}.ffn.up.weight'] = [ffn_dim, dim]
        shapes[f'blocks.{i}.ffn.up.bias'] = [ffn_dim]
        shapes[f'blocks.{i}.ffn.down.weight'] = [dim, ffn_dim]
        shapes[f'blocks.{i}.ffn.down.bias'] = [dim]
        # LayerNorm
        shapes[f'blocks.{i}.norm.weight'] = [dim]
        shapes[f'blocks.{i}.norm.bias'] = [dim]

    shapes['output_proj.weight'] = [80, dim]
    shapes['output_proj.bias'] = [80]

    # TimeEmbedding
    shapes['time_embed.time_mlp1.weight'] = [time_dim, time_dim]
    shapes['time_embed.time_mlp1.bias'] = [time_dim]
    shapes['time_embed.time_mlp2.weight'] = [time_dim, time_dim]
    shapes['time_embed.time_mlp2.bias'] = [time_dim]

    return shapes


EXPECTED_SHAPES = {
    'codec': get_expected_shapes_codec,
    'cam': get_expected_shapes_cam,
    'stt': get_expected_shapes_stt,
    'tts': get_expected_shapes_tts,
    'cfm': get_expected_shapes_cfm,
}


# ============================================================
#  Key Verification
# ============================================================

def verify_keys(state_dict: Dict, model_type: str) -> Tuple[int, int, int]:
    """Verify exported keys match expected Rust VarBuilder paths and shapes.

    Returns:
        (matched, missing, unexpected) counts
    """
    if model_type not in EXPECTED_SHAPES:
        logger.warning(f"No shape verification available for model type: {model_type}")
        return 0, 0, 0

    expected = EXPECTED_SHAPES[model_type]()
    exported_keys = set(state_dict.keys())
    expected_keys = set(expected.keys())

    matched = 0
    shape_mismatches = 0

    # Check for missing expected keys
    missing = expected_keys - exported_keys
    for key in sorted(missing):
        logger.error(f"  MISSING: {key} (expected shape {expected[key]})")

    # Check for unexpected keys (in export but not in expected)
    unexpected = exported_keys - expected_keys
    for key in sorted(unexpected):
        tensor = state_dict[key]
        logger.warning(f"  UNEXPECTED: {key} shape={list(tensor.shape)}")

    # Check shapes of matched keys
    common = exported_keys & expected_keys
    for key in sorted(common):
        tensor = state_dict[key]
        exp_shape = expected[key]
        if exp_shape is not None and list(tensor.shape) != exp_shape:
            logger.error(f"  SHAPE MISMATCH: {key}: got {list(tensor.shape)}, expected {exp_shape}")
            shape_mismatches += 1
        else:
            matched += 1

    logger.info(f"Verification: {matched} matched, {len(missing)} missing, "
                f"{len(unexpected)} unexpected, {shape_mismatches} shape mismatches")

    return matched, len(missing), len(unexpected)


# ============================================================
#  Export Functions
# ============================================================

def remap_keys(state_dict: Dict, model_type: str) -> Dict:
    """Remap PyTorch state dict keys to match Rust candle VarBuilder paths.

    Ensures all tensors are float32 and contiguous for safetensors.

    Args:
        state_dict: PyTorch state_dict() output
        model_type: Model type (codec, stt, tts, cam, cfm)

    Returns:
        remapped: Dictionary with properly formatted keys and float32 tensors
    """
    remapped = {}

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        # Ensure float32 and contiguous for safetensors
        tensor = tensor.float().contiguous()

        # PyTorch keys from our training scripts are designed to match
        # Rust VarBuilder paths, so no remapping is needed
        remapped[key] = tensor

    return remapped


def export_model(checkpoint_path: str, output_path: str, model_type: str,
                 verify: bool = False) -> None:
    """Export a Sonata model checkpoint to safetensors.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save safetensors file
        model_type: Model type (codec, cam, stt, tts, cfm)
        verify: If True, verify keys against expected Rust shapes
    """
    logger.info(f"Loading {model_type} checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    remapped = remap_keys(state_dict, model_type)

    logger.info(f"Exporting {len(remapped)} tensors")
    for name, tensor in list(remapped.items())[:5]:
        logger.info(f"  {name}: {list(tensor.shape)}")
    if len(remapped) > 5:
        logger.info(f"  ... and {len(remapped) - 5} more")

    if verify:
        logger.info(f"\nVerifying keys against expected Rust architecture...")
        matched, missing, unexpected = verify_keys(remapped, model_type)
        if missing > 0:
            logger.error(f"\n{missing} expected keys are MISSING. "
                         f"Rust inference will fail to load these weights!")

    save_file(remapped, output_path)
    logger.info(f"Saved {model_type} to {output_path}")


def validate_safetensors(path: str, model_type: str = None) -> None:
    """Validate safetensors file format and optionally verify against expected shapes.

    Args:
        path: Path to safetensors file
        model_type: Optional model type for shape verification
    """
    try:
        data = load_file(path)
        logger.info(f"Validated {path}")
        logger.info(f"  Keys: {len(data)}")

        total_params = 0
        for key in sorted(data.keys())[:10]:
            logger.info(f"    {key}: {list(data[key].shape)}")
            total_params += data[key].numel()
        if len(data) > 10:
            for key in data:
                total_params += data[key].numel()
            total_params = sum(t.numel() for t in data.values())
        logger.info(f"  Total parameters: {total_params:,}")

        if model_type:
            verify_keys(data, model_type)
    except Exception as e:
        logger.error(f"Validation failed: {e}")


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
    parser.add_argument('--verify', action='store_true',
                       help='Verify keys against expected Rust architecture shapes')
    parser.add_argument('--validate', action='store_true',
                       help='Validate output file after export')

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    export_model(args.checkpoint, args.output, args.model, verify=args.verify)

    if args.validate:
        validate_safetensors(args.output, args.model)

    logger.info(f"Export complete: {args.output}")


if __name__ == '__main__':
    main()
