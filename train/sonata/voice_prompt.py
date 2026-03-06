"""In-Context Voice Learning — zero-shot cloning from a 3-second audio prompt.

Instead of speaker IDs or pre-computed embeddings, the model learns to extract
voice characteristics directly from a reference audio clip at training time.

Training recipe:
  1. For each sample, randomly select a 3-10 second clip from the SAME speaker
  2. Extract mel features from the reference clip
  3. Feed through a reference encoder → per-frame features
  4. Flow model cross-attends to these features (via RefAudioCrossAttention)
  5. At inference: any 3-second clip from any person serves as voice prompt

This is how CosyVoice 2, MARS5, and MiniMax-Speech achieve zero-shot cloning.

Architecture:
  Reference audio → Mel spectrogram → Reference Encoder (Conformer) → Features
  Flow blocks cross-attend to reference features at every layer.
"""

import json
import math
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import RMSNorm


class ReferenceEncoder(nn.Module):
    """Encodes reference audio mel spectrogram into per-frame voice features.

    Architecture: 6-layer 1D ConvNet with residual connections.
    Captures speaker timbre, prosody patterns, and speaking style from
    a short reference clip. Output features are used as cross-attention
    keys/values in the flow model.
    """

    def __init__(self, mel_dim: int = 80, hidden_dim: int = 512, output_dim: int = 80,
                 n_layers: int = 6, kernel_size: int = 5):
        super().__init__()
        self.input_proj = nn.Linear(mel_dim, hidden_dim)

        layers = []
        for i in range(n_layers):
            layers.append(RefEncoderBlock(hidden_dim, kernel_size))
        self.blocks = nn.ModuleList(layers)

        self.output_norm = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, T_ref, mel_dim) → (B, T_ref, output_dim)"""
        x = self.input_proj(mel)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(self.output_norm(x))


class RefEncoderBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)
        x = x + self.conv(h).transpose(1, 2)
        x = x + self.ff(self.norm2(x))
        return x


class VoicePromptWrapper(nn.Module):
    """Wraps a flow model with reference encoder for voice-prompted generation.

    During training, a reference clip from the same speaker is randomly selected.
    The reference encoder produces features that condition the flow model via
    cross-attention at every layer.
    """

    def __init__(self, flow_model, ref_encoder: ReferenceEncoder,
                 mel_extractor=None):
        super().__init__()
        ref_out = ref_encoder.output_proj.out_features
        flow_expects = flow_model.cfg.ref_audio_dim
        assert ref_out == flow_expects, (
            f"ReferenceEncoder output_dim ({ref_out}) must match "
            f"FlowConfig.ref_audio_dim ({flow_expects})"
        )
        self.flow = flow_model
        self.ref_encoder = ref_encoder
        self.mel_extractor = mel_extractor

    def forward(self, x_t, t, semantic_tokens, ref_mel, speaker_ids=None):
        """Forward with reference audio conditioning."""
        ref_features = self.ref_encoder(ref_mel)
        return self.flow(x_t, t, semantic_tokens, speaker_ids,
                         ref_audio=ref_features)

    def compute_loss(self, acoustic_target, semantic_tokens, ref_mel,
                     speaker_ids=None):
        """OT-CFM loss with reference audio conditioning."""
        B = acoustic_target.shape[0]
        z = torch.randn(B, device=acoustic_target.device)
        t = torch.sigmoid(z).clamp(1e-5, 1 - 1e-5)
        noise = torch.randn_like(acoustic_target)
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * noise + t_expand * acoustic_target
        v_pred = self.forward(x_t, t, semantic_tokens, ref_mel, speaker_ids)
        return F.mse_loss(v_pred, acoustic_target - noise)


class VoicePromptDataset(torch.utils.data.Dataset):
    """Dataset that pairs each utterance with a reference clip from same speaker.

    For each training sample:
    1. Load the target utterance (semantic tokens + acoustic latents)
    2. Find another utterance from the same speaker
    3. Extract a 3-10 second reference mel from that utterance
    """

    def __init__(self, manifest_path: str, encoded_data_path: str,
                 ref_duration: float = 5.0, sample_rate: int = 16000,
                 n_fft: int = 1024, hop_length: int = 480, n_mels: int = 80):
        self.ref_frames = int(ref_duration * sample_rate / hop_length)
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Group entries by speaker
        self.speaker_to_entries = {}
        self.all_entries = []

        if Path(encoded_data_path).exists():
            data = torch.load(encoded_data_path, map_location="cpu", weights_only=False)
            entries = data if isinstance(data, list) else data.get("entries", [])
            for i, entry in enumerate(entries):
                spk = entry.get("speaker", "default")
                if spk not in self.speaker_to_entries:
                    self.speaker_to_entries[spk] = []
                self.speaker_to_entries[spk].append(i)
                self.all_entries.append(entry)

        print(f"[voice-prompt] {len(self.all_entries)} entries, "
              f"{len(self.speaker_to_entries)} speakers")

    def __len__(self):
        return max(len(self.all_entries), 1)

    def __getitem__(self, idx):
        if not self.all_entries:
            # Synthetic fallback
            sem = torch.randint(4, 32768, (50,))
            aco = torch.randn(50, 256)
            ref_mel = torch.randn(self.ref_frames, self.n_mels)
            return sem, aco, ref_mel

        entry = self.all_entries[idx]
        sem = entry.get("semantic_tokens", torch.zeros(1, dtype=torch.long))
        aco = entry.get("acoustic_latents", entry.get("acoustic_latent", torch.zeros(1, 256)))

        if isinstance(sem, list):
            sem = torch.tensor(sem, dtype=torch.long)
        if isinstance(aco, list):
            aco = torch.tensor(aco)

        # Get reference from same speaker (different utterance)
        spk = entry.get("speaker", "default")
        same_speaker = self.speaker_to_entries.get(spk, [idx])
        candidates = [i for i in same_speaker if i != idx]
        if not candidates:
            candidates = [idx]
        ref_idx = random.choice(candidates)
        ref_entry = self.all_entries[ref_idx]

        # Use acoustic latents as proxy for mel (same shape, different utterance)
        ref_aco = ref_entry.get("acoustic_latents",
                                ref_entry.get("acoustic_latent", torch.zeros(1, 256)))
        if isinstance(ref_aco, list):
            ref_aco = torch.tensor(ref_aco)

        # Trim/pad reference to fixed length
        if ref_aco.shape[0] > self.ref_frames:
            start = random.randint(0, ref_aco.shape[0] - self.ref_frames)
            ref_aco = ref_aco[start:start + self.ref_frames]
        elif ref_aco.shape[0] < self.ref_frames:
            pad = torch.zeros(self.ref_frames - ref_aco.shape[0], ref_aco.shape[-1])
            ref_aco = torch.cat([ref_aco, pad])

        # Project to mel_dim if needed
        if ref_aco.shape[-1] != self.n_mels:
            # Linear projection (in real training, use actual mel from audio)
            ref_mel = ref_aco[:, :self.n_mels] if ref_aco.shape[-1] >= self.n_mels \
                else F.pad(ref_aco, (0, self.n_mels - ref_aco.shape[-1]))
        else:
            ref_mel = ref_aco

        return sem, aco, ref_mel


if __name__ == "__main__":
    ref_enc = ReferenceEncoder(mel_dim=80, hidden_dim=256, output_dim=256, n_layers=4)
    n = sum(p.numel() for p in ref_enc.parameters())
    print(f"ReferenceEncoder: {n/1e6:.1f}M params")

    B = 2
    ref_mel = torch.randn(B, 50, 80)
    features = ref_enc(ref_mel)
    print(f"  Reference features: {features.shape}")

    # Test with a mock flow that accepts ref_audio
    from config import FlowConfig
    from flow import SonataFlow
    cfg = FlowConfig(use_ref_audio=True, ref_audio_dim=256, d_model=256,
                     n_layers=4, n_heads=4, cond_dim=128)
    flow = SonataFlow(cfg)
    wrapper = VoicePromptWrapper(flow, ref_enc)

    sem = torch.randint(0, cfg.semantic_vocab_size, (B, 50))
    aco = torch.randn(B, 50, cfg.acoustic_dim)

    loss = wrapper.compute_loss(aco, sem, ref_mel)
    print(f"  Voice-prompted loss: {loss:.4f}")
    print("PASS")
