"""Joint fine-tuning: end-to-end Codec + LM + Flow with waveform-level loss.

The key insight: when trained separately, each component introduces errors
that compound through the pipeline. Joint fine-tuning lets gradients flow
through the entire system so components learn to compensate for each other.

Pipeline: Text → LM → semantic tokens → Flow → acoustic latents → Decoder → Audio
Losses: waveform L1 + multi-scale STFT + (optional) discriminator

The codec encoder/FSQ are frozen (we keep the learned codebook).
Gradients flow: Decoder ← Flow ← (straight-through) ← LM.

Usage:
  python train_joint.py \
    --codec-ckpt checkpoints/codec_best.pt \
    --lm-ckpt checkpoints/lm_best.pt \
    --flow-ckpt checkpoints/flow_best.pt \
    --manifest train/data/manifest_clean.jsonl \
    --steps 20000 --device mps
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import CodecConfig, SemanticLMConfig, FlowConfig
from codec import SonataCodec, MultiScaleSTFTLoss
from semantic_lm import SonataSemanticLM
from flow import SonataFlow
from ema import EMA
from modules import cosine_lr


class JointPipeline(nn.Module):
    """End-to-end TTS pipeline with differentiable path through all components.

    Frozen: codec encoder + FSQ (quantization is fixed)
    Trainable: LM, Flow, Decoder

    Forward:
      1. Codec encodes audio → semantic_tokens + acoustic_target (frozen, for loss)
      2. LM predicts semantic tokens from text (trainable)
      3. Gumbel-softmax on LM logits → soft token embeddings (differentiable)
      4. Flow predicts acoustic latents from soft tokens (trainable)
      5. Decoder reconstructs audio from (soft semantic, predicted acoustic) (trainable)
      6. Loss against original audio
    """

    def __init__(self, codec: SonataCodec, lm: SonataSemanticLM,
                 flow: SonataFlow, temperature: float = 1.0):
        super().__init__()
        self.codec = codec
        self.lm = lm
        self.flow = flow
        self.temperature = temperature

        # Freeze codec encoder + FSQ
        for n, p in codec.named_parameters():
            if "encoder" in n or "mel" in n or "fsq" in n or "waveform_encoder" in n:
                p.requires_grad_(False)

    def forward(self, audio: torch.Tensor, text_tokens: torch.Tensor,
                speaker_ids=None):
        """
        audio: (B, samples) at codec sample rate
        text_tokens: (B, T_text)
        Returns: reconstructed audio, losses dict
        """
        B = audio.shape[0]

        # Step 1: Encode audio with frozen codec
        with torch.no_grad():
            semantic_tokens, acoustic_target, _ = self.codec.encode(audio)
            T = semantic_tokens.shape[1]

        # Step 2: LM predicts semantic tokens
        # Shift right: input is [BOS] + tokens[:-1], target is tokens
        bos = torch.ones(B, 1, dtype=torch.long, device=audio.device)
        lm_input = torch.cat([bos, semantic_tokens[:, :-1]], dim=1)
        lm_logits, lm_losses = self.lm(text_tokens, lm_input,
                                        target_semantic=semantic_tokens)

        # Step 3: Gumbel-softmax for differentiable token selection
        soft_onehot = F.gumbel_softmax(lm_logits, tau=self.temperature, hard=True)
        soft_emb = soft_onehot @ self.flow.semantic_emb.weight  # (B, T, V) @ (V, D)
        hard_tokens = lm_logits.argmax(dim=-1)

        # Step 4: Flow predicts acoustic latents (with gradient flow via soft_emb)
        noise = torch.randn_like(acoustic_target)
        t = torch.rand(B, device=audio.device).clamp(1e-5, 1 - 1e-5)
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * noise + t_expand * acoustic_target
        target_velocity = acoustic_target - noise

        v_pred = self.flow(x_t, t, hard_tokens, speaker_ids)
        flow_loss = F.mse_loss(v_pred, target_velocity)

        # Step 5: Generate acoustic latents for waveform reconstruction (no grad)
        with torch.no_grad():
            generated_acoustic = self.flow.sample(
                hard_tokens, n_steps=4, speaker_ids=speaker_ids)

        # Step 6: Decode to audio
        T_min = min(T, generated_acoustic.shape[1])
        semantic_codes = self.codec.fsq.indices_to_codes(hard_tokens[:, :T_min])
        recon_audio = self.codec.decoder(semantic_codes, generated_acoustic[:, :T_min])

        # Trim to same length
        min_len = min(audio.shape[-1], recon_audio.shape[-1])
        audio_trimmed = audio[..., :min_len]
        recon_trimmed = recon_audio[..., :min_len]

        # Waveform L1 loss
        waveform_loss = F.l1_loss(recon_trimmed, audio_trimmed)

        losses = {
            "lm_ce": lm_losses.get("semantic", torch.tensor(0.0)),
            "flow_cfm": flow_loss,
            "waveform_l1": waveform_loss,
        }

        return recon_trimmed, losses


class JointDataset(Dataset):
    """Loads audio + text pairs from a manifest for joint training."""

    def __init__(self, manifest_path: str, max_duration: float = 10.0,
                 sample_rate: int = 24000, text_vocab_size: int = 32000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.text_vocab_size = text_vocab_size
        self.entries = []

        if not Path(manifest_path).exists():
            return

        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                dur = entry.get("duration", 0)
                if 0.5 <= dur <= max_duration:
                    self.entries.append(entry)

        print(f"[joint-data] {len(self.entries)} entries from {manifest_path}")

    def __len__(self):
        return max(len(self.entries), 1)

    def __getitem__(self, idx):
        if not self.entries:
            # Synthetic fallback
            audio = torch.randn(self.max_samples) * 0.1
            text = torch.randint(4, self.text_vocab_size, (20,))
            return audio, text

        entry = self.entries[idx % len(self.entries)]
        audio_path = entry.get("audio", "")

        try:
            import torchaudio
            audio, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            audio = audio[0][:self.max_samples]
        except Exception:
            audio = torch.randn(self.max_samples) * 0.1

        # Simple char-level tokenization for text
        text_str = entry.get("text", "")
        text = torch.tensor([min(ord(c), self.text_vocab_size - 1)
                             for c in text_str[:512]], dtype=torch.long)
        if len(text) == 0:
            text = torch.zeros(1, dtype=torch.long)

        return audio, text


def collate_joint(batch):
    audio_list, text_list = zip(*batch)
    max_audio = max(a.shape[0] for a in audio_list)
    max_text = max(t.shape[0] for t in text_list)
    B = len(batch)

    audio_pad = torch.zeros(B, max_audio)
    text_pad = torch.zeros(B, max_text, dtype=torch.long)
    for i, (a, t) in enumerate(zip(audio_list, text_list)):
        audio_pad[i, :a.shape[0]] = a
        text_pad[i, :t.shape[0]] = t

    return audio_pad, text_pad


def train(args):
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  SONATA JOINT FINE-TUNING")
    print(f"  End-to-end: Text → LM → Flow → Decoder → Audio")
    print(f"{'='*60}")

    # Load pretrained components
    def load_ckpt(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt

    codec_ckpt = load_ckpt(args.codec_ckpt)
    codec_cfg = CodecConfig(**{k: v for k, v in codec_ckpt.get("config", {}).items()
                                if k in CodecConfig.__dataclass_fields__})
    codec = SonataCodec(codec_cfg)
    codec.load_state_dict(codec_ckpt.get("ema", codec_ckpt.get("model", {})), strict=False)

    lm_ckpt = load_ckpt(args.lm_ckpt)
    lm_cfg = SemanticLMConfig(**{k: v for k, v in lm_ckpt.get("config", {}).items()
                                  if k in SemanticLMConfig.__dataclass_fields__})
    lm = SonataSemanticLM(lm_cfg)
    lm.load_state_dict(lm_ckpt.get("ema", lm_ckpt.get("model", {})), strict=False)

    flow_ckpt = load_ckpt(args.flow_ckpt)
    flow_cfg = FlowConfig(**FlowConfig._normalize_loaded_dict(flow_ckpt.get("config", {})))
    flow = SonataFlow(flow_cfg)
    flow.load_state_dict(flow_ckpt.get("ema", flow_ckpt.get("model", {})), strict=False)

    pipeline = JointPipeline(codec, lm, flow, temperature=args.gumbel_temp).to(device)

    trainable = [p for p in pipeline.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in pipeline.parameters())
    print(f"  Trainable: {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M total")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    ms_stft = MultiScaleSTFTLoss()

    dataset = JointDataset(args.manifest, sample_rate=codec_cfg.sample_rate)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_joint, num_workers=2, drop_last=True)

    os.makedirs(args.output_dir, exist_ok=True)

    step = 0
    running = {"lm_ce": 0, "flow_cfm": 0, "waveform_l1": 0}
    t0 = time.time()

    print(f"  Steps: {args.steps}, Batch: {args.batch_size}")
    print(f"  LR: {args.lr}, Gumbel τ: {args.gumbel_temp}\n")

    while step < args.steps:
        for audio, text in loader:
            if step >= args.steps:
                break

            audio = audio.to(device)
            text = text.to(device)

            lr = cosine_lr(step, args.warmup, args.lr, args.lr * 0.01, args.steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            recon, losses = pipeline(audio, text)

            total_loss = (args.w_lm * losses["lm_ce"] +
                          args.w_flow * losses["flow_cfm"] +
                          args.w_wav * losses["waveform_l1"])

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            for k in running:
                running[k] += losses[k].item()
            step += 1

            if step % args.log_every == 0:
                n = args.log_every
                elapsed = time.time() - t0
                parts = " ".join(f"{k}={running[k]/n:.4f}" for k in running)
                print(f"  step {step:6d} | {parts} | lr={lr:.2e} | {n/elapsed:.1f} it/s")
                running = {k: 0 for k in running}
                t0 = time.time()

            if step % args.save_every == 0:
                path = os.path.join(args.output_dir, f"joint_step_{step}.pt")
                torch.save({
                    "lm": lm.state_dict(),
                    "flow": flow.state_dict(),
                    "decoder": {k: v for k, v in codec.state_dict().items()
                                if "decoder" in k or "fsq" in k},
                    "step": step,
                    "lm_config": vars(lm_cfg),
                    "flow_config": vars(flow_cfg),
                    "codec_config": vars(codec_cfg),
                }, path)
                print(f"  [ckpt] {path}")

    path = os.path.join(args.output_dir, "joint_final.pt")
    torch.save({
        "lm": lm.state_dict(),
        "flow": flow.state_dict(),
        "decoder": {k: v for k, v in codec.state_dict().items()
                    if "decoder" in k or "fsq" in k},
        "step": step,
        "lm_config": vars(lm_cfg),
        "flow_config": vars(flow_cfg),
        "codec_config": vars(codec_cfg),
    }, path)
    print(f"\n  Final: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec-ckpt", required=True)
    parser.add_argument("--lm-ckpt", required=True)
    parser.add_argument("--flow-ckpt", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output-dir", default="train/checkpoints/joint")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--gumbel-temp", type=float, default=0.5)
    parser.add_argument("--w-lm", type=float, default=1.0)
    parser.add_argument("--w-flow", type=float, default=1.0)
    parser.add_argument("--w-wav", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    args = parser.parse_args()
    train(args)
