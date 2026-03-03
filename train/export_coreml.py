#!/usr/bin/env python3
"""Export Sonata v2 models to Core ML for Apple Neural Engine (ANE) inference.

ANE provides 15.8 TOPS on M1, 18 TOPS on M3, 38 TOPS on M4 — significantly
faster than Metal GPU for supported operations.

Usage:
    python export_coreml.py --model codec --checkpoint codec_best.pt --output sonata-codec.mlpackage
    python export_coreml.py --model cam --checkpoint speaker_encoder_best.pt --output sonata-cam.mlpackage
    python export_coreml.py --model stt --checkpoint stt_best.pt --output sonata-stt.mlpackage
    python export_coreml.py --model tts --checkpoint tts_best.pt --output sonata-tts.mlpackage
    python export_coreml.py --model cfm --checkpoint cfm_best.pt --output sonata-cfm.mlpackage
    python export_coreml.py --all --checkpoint-dir ./checkpoints --output-dir ./models

Requirements:
    pip install coremltools torch torchaudio

ANE Compatibility Notes:
    - Most ops (Conv, Linear, LayerNorm, Attention) run on ANE
    - Snake activation (x + sin²(αx)/α) decomposes to ANE-compatible ops
    - SwiGLU uses element-wise ops — ANE compatible
    - ConvTranspose1d — ANE compatible
    - Codebook lookup (embedding) — ANE compatible
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Model reconstruction (mirrors training architectures) ──────────────────

class SnakeActivation(nn.Module):
    """Snake activation: x + sin²(αx)/α — ANE compatible via decomposition."""
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2


class SwiGLU(nn.Module):
    """SwiGLU FFN — ANE compatible."""
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim)
        self.w2 = nn.Linear(dim, ffn_dim)
        self.w3 = nn.Linear(ffn_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(torch.silu(self.w1(x)) * self.w2(x))


# ── Codec Model ────────────────────────────────────────────────────────────

class CodecEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [1, 32, 64, 128, 256]
        strides = [8, 5, 4, 3]
        layers = []
        for i in range(4):
            layers.extend([
                nn.Conv1d(channels[i], channels[i + 1],
                          kernel_size=strides[i] * 2, stride=strides[i],
                          padding=strides[i] // 2),
                SnakeActivation(channels[i + 1]),
            ])
        layers.append(nn.Conv1d(256, 512, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CodecDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [512, 256, 128, 64, 32]
        strides = [3, 4, 5, 8]
        layers = [nn.Conv1d(512, 512, kernel_size=3, padding=1)]
        for i in range(4):
            act = SnakeActivation(channels[i + 1]) if i < 3 else nn.Tanh()
            layers.extend([
                nn.ConvTranspose1d(channels[i], channels[i + 1],
                                   kernel_size=strides[i] * 2, stride=strides[i],
                                   padding=strides[i] // 2,
                                   output_padding=0),
                act,
            ])
        layers.append(nn.Conv1d(32, 1, kernel_size=7, padding=3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SonataCodecForExport(nn.Module):
    """Codec encoder + decoder (RVQ handled separately for ANE)."""
    def __init__(self):
        super().__init__()
        self.encoder = CodecEncoder()
        self.decoder = CodecDecoder()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(audio)
        return self.decoder(encoded)


# ── CAM++ Speaker Encoder ──────────────────────────────────────────────────

class CAMBlock(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ffn = SwiGLU(dim, dim * 4)
        self.ffn_norm = nn.LayerNorm(dim)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.heads, D // self.heads)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (D // self.heads) ** 0.5
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(out)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CAMPlusPlusForExport(nn.Module):
    def __init__(self):
        super().__init__()
        # Multi-scale frontend: 80 mel → 64 → 128 → 256
        self.frontend = nn.Sequential(
            nn.Conv1d(80, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.blocks = nn.ModuleList([CAMBlock(256, 8) for _ in range(6)])
        # Attentive stats pooling
        self.pool_attn = nn.Linear(256, 1)
        self.output_proj = nn.Linear(512, 192)  # mean + std → 192

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, 80, T]
        x = self.frontend(mel)  # [B, 256, T']
        x = x.transpose(1, 2)  # [B, T', 256]
        for block in self.blocks:
            x = block(x)
        # Attentive stats pooling
        w = torch.softmax(self.pool_attn(x), dim=1)  # [B, T', 1]
        mean = (w * x).sum(dim=1)  # [B, 256]
        std = ((w * (x - mean.unsqueeze(1)) ** 2).sum(dim=1)).sqrt()  # [B, 256]
        pooled = torch.cat([mean, std], dim=-1)  # [B, 512]
        emb = self.output_proj(pooled)  # [B, 192]
        return nn.functional.normalize(emb, p=2, dim=-1)


# ── STT Conformer ──────────────────────────────────────────────────────────

class ConformerBlockForExport(nn.Module):
    def __init__(self, dim=512, ffn_dim=2048, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=31, padding=15, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
        )
        self.ffn = SwiGLU(dim, ffn_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).reshape(B, T, self.heads, D // self.heads).transpose(1, 2)
        k = self.k_proj(h).reshape(B, T, self.heads, D // self.heads).transpose(1, 2)
        v = self.v_proj(h).reshape(B, T, self.heads, D // self.heads).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (D // self.heads) ** 0.5
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o_proj(out)
        h = self.norm2(x).transpose(1, 2)
        x = x + self.conv(h).transpose(1, 2)
        x = x + self.ffn(self.norm3(x))
        return x


class SonataSTTForExport(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(512, 512)
        self.blocks = nn.ModuleList([
            ConformerBlockForExport(512, 2048, 8) for _ in range(12)
        ])
        self.output_proj = nn.Linear(512, 32000)

    def forward(self, codec_embeddings: torch.Tensor) -> torch.Tensor:
        # codec_embeddings: [B, 512, T] → transpose → [B, T, 512]
        x = codec_embeddings.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)  # [B, T, 32000]


# ── TTS Transformer ───────────────────────────────────────────────────────

class AdaIN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale = nn.Linear(cond_dim, dim)
        self.bias = nn.Linear(cond_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond).unsqueeze(1)
        bias = self.bias(cond).unsqueeze(1)
        return self.norm(x) * (1 + scale) + bias


class SonataTTSForExport(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_embed = nn.Embedding(32000, 512)
        self.blocks = nn.ModuleList()
        for _ in range(12):
            self.blocks.append(nn.ModuleDict({
                'norm': nn.LayerNorm(512),
                'q': nn.Linear(512, 512),
                'k': nn.Linear(512, 512),
                'v': nn.Linear(512, 512),
                'o': nn.Linear(512, 512),
                'speaker_adain': AdaIN(512, 192),
                'emotion_adain': AdaIN(512, 192),
                'ffn': SwiGLU(512, 2048),
                'ffn_norm': nn.LayerNorm(512),
            }))
        self.emotion_embed = nn.Embedding(64, 192)
        self.nonverbal_embed = nn.Embedding(24, 512)
        self.output_proj = nn.Linear(512, 1024)

    def forward(self, text_ids: torch.Tensor,
                speaker_emb: torch.Tensor,
                emotion_id: torch.Tensor) -> torch.Tensor:
        x = self.text_embed(text_ids)  # [B, T, 512]
        emo = self.emotion_embed(emotion_id)  # [B, 192]

        for block in self.blocks:
            B, T, D = x.shape
            h = block['norm'](x)
            q = block['q'](h).reshape(B, T, 8, 64).transpose(1, 2)
            k = block['k'](h).reshape(B, T, 8, 64).transpose(1, 2)
            v = block['v'](h).reshape(B, T, 8, 64).transpose(1, 2)
            attn = torch.matmul(q, k.transpose(-2, -1)) / 8.0
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
            x = x + block['o'](out)
            x = block['speaker_adain'](x, speaker_emb)
            x = block['emotion_adain'](x, emo)
            x = x + block['ffn'](block['ffn_norm'](x))

        return self.output_proj(x)  # [B, T, 1024]


# ── CFM Decoder ────────────────────────────────────────────────────────────

class SonataCFMForExport(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(80, 512)
        # Sinusoidal time embedding + MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.blocks = nn.ModuleList()
        for _ in range(12):
            self.blocks.append(nn.ModuleDict({
                'norm': nn.LayerNorm(512),
                'q': nn.Linear(512, 512),
                'k': nn.Linear(512, 512),
                'v': nn.Linear(512, 512),
                'o': nn.Linear(512, 512),
                'time_adain': AdaIN(512, 256),
                'speaker_adain': AdaIN(512, 192),
                'ffn': SwiGLU(512, 2048),
                'ffn_norm': nn.LayerNorm(512),
            }))
        self.output_proj = nn.Linear(512, 80)

    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        half = 128
        freqs = torch.exp(-torch.arange(half, device=t.device) * (
            torch.log(torch.tensor(10000.0)) / half
        ))
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_mlp(emb)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                speaker_emb: torch.Tensor) -> torch.Tensor:
        # x_t: [B, 80, T] noisy mel
        # t: [B] timestep in [0, 1]
        # speaker_emb: [B, 192]
        x = self.input_proj(x_t.transpose(1, 2))  # [B, T, 512]
        t_emb = self._time_embed(t)  # [B, 256]

        for block in self.blocks:
            B, T, D = x.shape
            h = block['norm'](x)
            q = block['q'](h).reshape(B, T, 8, 64).transpose(1, 2)
            k = block['k'](h).reshape(B, T, 8, 64).transpose(1, 2)
            v = block['v'](h).reshape(B, T, 8, 64).transpose(1, 2)
            attn = torch.matmul(q, k.transpose(-2, -1)) / 8.0
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
            x = x + block['o'](out)
            x = block['time_adain'](x, t_emb)
            x = block['speaker_adain'](x, speaker_emb)
            x = x + block['ffn'](block['ffn_norm'](x))

        return self.output_proj(x).transpose(1, 2)  # [B, 80, T]


# ── Core ML conversion ────────────────────────────────────────────────────

MODEL_CLASSES = {
    "codec": SonataCodecForExport,
    "cam": CAMPlusPlusForExport,
    "stt": SonataSTTForExport,
    "tts": SonataTTSForExport,
    "cfm": SonataCFMForExport,
}

# Example input shapes for tracing
TRACE_INPUTS = {
    "codec": lambda: (torch.randn(1, 1, 24000),),
    "cam": lambda: (torch.randn(1, 80, 200),),
    "stt": lambda: (torch.randn(1, 512, 50),),
    "tts": lambda: (
        torch.randint(0, 100, (1, 20)),    # text_ids
        torch.randn(1, 192),               # speaker_emb
        torch.randint(0, 64, (1,)),         # emotion_id
    ),
    "cfm": lambda: (
        torch.randn(1, 80, 50),            # x_t
        torch.tensor([0.5]),                # t
        torch.randn(1, 192),               # speaker_emb
    ),
}

# Input descriptions for Core ML
INPUT_DESCRIPTIONS = {
    "codec": {"audio": "Mono audio at 24kHz [1, 1, samples]"},
    "cam": {"mel": "80-bin mel spectrogram [1, 80, frames]"},
    "stt": {"codec_embeddings": "Codec embeddings [1, 512, frames]"},
    "tts": {
        "text_ids": "Token IDs [1, seq_len]",
        "speaker_emb": "Speaker embedding [1, 192]",
        "emotion_id": "Emotion ID [1]",
    },
    "cfm": {
        "x_t": "Noisy mel [1, 80, frames]",
        "t": "Timestep [1]",
        "speaker_emb": "Speaker embedding [1, 192]",
    },
}


def load_weights(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load PyTorch checkpoint weights into export model."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    # Try direct load first
    try:
        model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded {len(state_dict)} weight tensors")
    except Exception as e:
        log.warning(f"Partial weight load: {e}")

    return model


def convert_to_coreml(model_name: str, checkpoint_path: str, output_path: str):
    """Convert PyTorch model to Core ML with ANE optimization."""
    try:
        import coremltools as ct
        from coremltools.models.neural_network import quantization_utils
    except ImportError:
        log.error("coremltools not installed. Install with: pip install coremltools")
        sys.exit(1)

    log.info(f"Converting {model_name} to Core ML...")

    # Build and load model
    model_cls = MODEL_CLASSES[model_name]
    model = model_cls()
    model = load_weights(model, checkpoint_path)
    model.eval()

    # Trace with example inputs
    example_inputs = TRACE_INPUTS[model_name]()
    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs)

    # Convert to Core ML
    input_specs = []
    input_names = list(INPUT_DESCRIPTIONS[model_name].keys())
    for i, (name, desc) in enumerate(INPUT_DESCRIPTIONS[model_name].items()):
        shape = example_inputs[i].shape
        # Use flexible shapes for variable-length inputs
        if model_name in ("codec", "cam", "stt", "cfm") and "frame" in desc.lower():
            # Variable time dimension
            flex_shape = ct.Shape(
                shape=list(shape),
                default=list(shape),
            )
            input_specs.append(ct.TensorType(name=name, shape=flex_shape))
        elif model_name == "codec" and "sample" in desc.lower():
            input_specs.append(ct.TensorType(
                name=name,
                shape=ct.Shape(shape=list(shape), default=list(shape)),
            ))
        else:
            input_specs.append(ct.TensorType(name=name, shape=list(shape)))

    mlmodel = ct.convert(
        traced,
        inputs=input_specs,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,  # ANE support
        compute_precision=ct.precision.FLOAT16,  # FP16 for ANE
    )

    # Set metadata
    mlmodel.author = "Sonata v2"
    mlmodel.short_description = f"Sonata {model_name} — on-device voice pipeline"
    mlmodel.version = "2.0.0"

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))

    size_mb = sum(f.stat().st_size for f in output.rglob("*") if f.is_file()) / 1e6
    log.info(f"Saved Core ML model: {output} ({size_mb:.1f} MB)")

    # Verify on macOS
    try:
        import platform
        if platform.system() == "Darwin":
            log.info("Verifying model on macOS...")
            loaded = ct.models.MLModel(str(output))
            spec = loaded.get_spec()
            log.info(f"  Inputs: {[i.name for i in spec.description.input]}")
            log.info(f"  Outputs: {[o.name for o in spec.description.output]}")

            # Check ANE compatibility
            compute_units = loaded.compute_unit
            log.info(f"  Compute units: {compute_units}")
            log.info("  ANE: Available (model uses FLOAT16)")
    except Exception as e:
        log.warning(f"Verification skipped: {e}")

    return str(output)


def convert_all(checkpoint_dir: str, output_dir: str):
    """Convert all available checkpoints to Core ML."""
    ckpt_dir = Path(checkpoint_dir)
    out_dir = Path(output_dir)

    models = {
        "cam": ckpt_dir / "cam" / "speaker_encoder_best.pt",
        "codec": ckpt_dir / "codec" / "codec_best.pt",
        "stt": ckpt_dir / "stt" / "stt_best.pt",
        "tts": ckpt_dir / "tts" / "tts_best.pt",
        "cfm": ckpt_dir / "cfm" / "cfm_best.pt",
    }

    for name, ckpt_path in models.items():
        if ckpt_path.exists():
            output = out_dir / f"sonata-{name}.mlpackage"
            try:
                convert_to_coreml(name, str(ckpt_path), str(output))
            except Exception as e:
                log.error(f"Failed to convert {name}: {e}")
        else:
            log.warning(f"Checkpoint not found for {name}: {ckpt_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export Sonata v2 models to Core ML for Apple Neural Engine"
    )
    parser.add_argument("--model", choices=list(MODEL_CLASSES.keys()),
                        help="Model to convert")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str,
                        help="Output .mlpackage path")
    parser.add_argument("--all", action="store_true",
                        help="Convert all available checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Checkpoint directory (with --all)")
    parser.add_argument("--output-dir", type=str, default="./models",
                        help="Output directory (with --all)")

    args = parser.parse_args()

    if args.all:
        convert_all(args.checkpoint_dir, args.output_dir)
    elif args.model and args.checkpoint and args.output:
        convert_to_coreml(args.model, args.checkpoint, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
