"""Consistency distillation for Sonata Flow.

Distills an 8-step Sonata Flow teacher into a 1-2 step student model,
following the progressive distillation approach from:
  - Salimans & Ho (2022): "Progressive Distillation for Fast Sampling"
  - Lipman et al. (2024): "Flow Matching for Generative Modeling"

Usage:
  python train_flow_distill.py \
    --teacher models/sonata/flow.safetensors \
    --teacher-config models/sonata/sonata_flow_config.json \
    --data-dir data/sonata_pairs/ \
    --output-dir checkpoints/flow_distilled/ \
    --device mps

Data format: directory of .pt files, each containing:
  { "semantic_tokens": LongTensor(T,), "acoustic_latents": FloatTensor(T, 256) }
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import FlowConfig
from flow import SonataFlow
from ema import EMA


# ─── Dataset ────────────────────────────────────────────────────────────────

class SonataFlowDataset(Dataset):
    """Loads pre-extracted (semantic_tokens, acoustic_latents) pairs.

    Supports three data formats:
    1. Directory of per-utterance .pt files
    2. Sharded .pt files (list of dicts)
    3. Shard index file (.shards.txt or .txt pointing to shard files)
    """

    def __init__(self, data_path: str, max_frames: int = 500):
        self.max_frames = max_frames
        self.entries = []
        self._per_file_mode = False

        p = Path(data_path)
        if p.is_dir():
            self._per_file_mode = True
            self.files = sorted(p.glob("*.pt"))
        elif p.suffix == ".txt" and p.exists():
            for line in p.read_text().strip().split("\n"):
                line = line.strip()
                if line and Path(line).exists():
                    shard = torch.load(line, map_location="cpu", weights_only=False)
                    self._load_shard(shard)
        elif p.exists():
            shard = torch.load(str(p), map_location="cpu", weights_only=False)
            self._load_shard(shard)

        if not self._per_file_mode:
            print(f"[distill-data] {len(self.entries)} entries loaded")

    def _load_shard(self, shard):
        if isinstance(shard, list):
            self.entries.extend(shard)
        elif isinstance(shard, dict) and "entries" in shard:
            self.entries.extend(shard["entries"])

    def __len__(self) -> int:
        return len(self.files) if self._per_file_mode else len(self.entries)

    def __getitem__(self, idx: int) -> tuple:
        if self._per_file_mode:
            data = torch.load(self.files[idx], weights_only=True)
        else:
            data = self.entries[idx]

        sem = data.get("semantic_tokens", data.get("semantic", torch.zeros(1, dtype=torch.long)))
        aco = data.get("acoustic_latents", data.get("acoustic_latent", torch.zeros(1, 256)))
        if isinstance(sem, list):
            sem = torch.tensor(sem, dtype=torch.long)
        if isinstance(aco, list):
            aco = torch.tensor(aco)
        return sem[:self.max_frames], aco[:self.max_frames]


def collate_fn(batch: list) -> tuple:
    """Pad to max length in batch."""
    sem_list, aco_list = zip(*batch)
    max_t = max(s.shape[0] for s in sem_list)
    B = len(sem_list)
    acoustic_dim = aco_list[0].shape[-1]

    sem_padded = torch.zeros(B, max_t, dtype=torch.long)
    aco_padded = torch.zeros(B, max_t, acoustic_dim)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, (s, a) in enumerate(zip(sem_list, aco_list)):
        T = s.shape[0]
        sem_padded[i, :T] = s
        aco_padded[i, :T] = a
        lengths[i] = T

    return sem_padded, aco_padded, lengths


# ─── Consistency Distillation ───────────────────────────────────────────────

class ConsistencyDistillation:
    """Progressive consistency distillation for Sonata Flow."""

    def __init__(
        self,
        teacher: SonataFlow,
        student: SonataFlow,
        cfg: FlowConfig,
        device: torch.device,
        aux_loss_weight: float = 0.1,
    ):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device).train()
        self.cfg = cfg
        self.device = device
        self.aux_loss_weight = aux_loss_weight

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def teacher_generate(
        self,
        noise: torch.Tensor,
        semantic_tokens: torch.Tensor,
        n_steps: int,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run teacher's multi-step ODE to get target output."""
        x = noise.clone()
        dt = 1.0 / n_steps
        B = noise.shape[0]

        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=self.device, dtype=noise.dtype)
            v = self.teacher(x, t, semantic_tokens, speaker_ids)
            x = x + dt * v

        return x

    def student_generate(
        self,
        noise: torch.Tensor,
        semantic_tokens: torch.Tensor,
        n_steps: int,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run student's fewer-step ODE."""
        x = noise.clone()
        dt = 1.0 / n_steps
        B = noise.shape[0]

        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=self.device, dtype=noise.dtype)
            v = self.student(x, t, semantic_tokens, speaker_ids)
            x = x + dt * v

        return x

    def distill_step(
        self,
        semantic_tokens: torch.Tensor,
        acoustic_target: torch.Tensor,
        teacher_steps: int,
        student_steps: int,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One distillation training step."""
        B, T, D = acoustic_target.shape

        # Sample noise
        noise = torch.randn_like(acoustic_target, device=acoustic_target.device)

        # Teacher target (detached, no grad)
        teacher_output = self.teacher_generate(
            noise, semantic_tokens, teacher_steps, speaker_ids
        )

        # Student prediction
        student_output = self.student_generate(
            noise, semantic_tokens, student_steps, speaker_ids
        )

        # Huber loss (more stable than MSE for distillation)
        loss = F.huber_loss(student_output, teacher_output, delta=1.0)

        # Optional: add auxiliary OT-CFM loss against real data (stabilizes training)
        aux_loss = self.student.compute_loss(acoustic_target, semantic_tokens, speaker_ids)
        return loss + self.aux_loss_weight * aux_loss

    def consistency_step(
        self,
        semantic_tokens: torch.Tensor,
        acoustic_target: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DSFlow-style consistency training: endpoint matching + velocity alignment.

        Instead of progressive distillation, directly train the student to be
        consistent at ANY point along the ODE trajectory — meaning the student
        can generate correct output from any timestep t in a single step.

        Two loss terms:
        1. Endpoint matching: student(x_t, t) should reach x_1 in one step
        2. Velocity alignment: student velocity should match mean ODE velocity
        """
        B, T, D = acoustic_target.shape
        noise = torch.randn_like(acoustic_target)

        # Sample random timestep
        t = torch.rand(B, device=acoustic_target.device).clamp(1e-5, 1.0 - 1e-5)
        t_expand = t[:, None, None]

        # Interpolate: x_t = (1-t)*noise + t*target
        x_t = (1 - t_expand) * noise + t_expand * acoustic_target

        # Student predicts velocity at x_t
        v_pred = self.student(x_t, t, semantic_tokens, speaker_ids)

        # --- Loss 1: Endpoint matching ---
        # From x_t, one step of size (1-t) should reach x_1
        x_pred = x_t + (1 - t_expand) * v_pred
        endpoint_loss = F.huber_loss(x_pred, acoustic_target, delta=1.0)

        # --- Loss 2: Velocity alignment ---
        # The true mean velocity is (x_1 - x_0) = (target - noise)
        true_velocity = acoustic_target - noise
        velocity_loss = F.mse_loss(v_pred, true_velocity)

        # --- Loss 3: Self-consistency ---
        # The student at two different points on the same trajectory should agree
        # on the endpoint. Pick t2 > t, compute x_t2, predict from there.
        # Only t2/x_t2 need no_grad; student forward must retain gradients.
        with torch.no_grad():
            t2 = t + (1 - t) * torch.rand_like(t) * 0.5  # t2 in [t, midpoint to 1]
            t2_expand = t2[:, None, None]
            x_t2 = (1 - t2_expand) * noise + t2_expand * acoustic_target

        v2 = self.student(x_t2, t2, semantic_tokens, speaker_ids)
        x_pred2 = x_t2 + (1 - t2_expand) * v2
        consistency_loss = F.mse_loss(x_pred.detach(), x_pred2)

        return endpoint_loss + 0.5 * velocity_loss + 0.1 * consistency_loss


# ─── Training ───────────────────────────────────────────────────────────────

def load_flow_config(path: str) -> FlowConfig:
    """Load FlowConfig from JSON, filtering to valid FlowConfig fields."""
    with open(path) as f:
        cfg_dict = json.load(f)
    valid = FlowConfig._normalize_loaded_dict(cfg_dict)
    return FlowConfig(**valid)


def load_teacher_weights(path: str, device: torch.device) -> dict:
    """Load teacher weights from .pt or .safetensors."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path, device=str(device))
    else:
        state = torch.load(path, weights_only=True, map_location=device)
    # Strip "model." prefix if present (e.g. from full training checkpoints)
    keys = list(state.keys())
    if keys and keys[0].startswith("model."):
        state = {k.replace("model.", ""): v for k, v in state.items()}
    return state


def train(args: argparse.Namespace) -> None:
    # Load config
    cfg = load_flow_config(args.teacher_config)
    device = torch.device(args.device)

    # Create teacher and student (same architecture)
    teacher = SonataFlow(cfg)
    student = SonataFlow(cfg)

    # Load teacher weights
    teacher_state = load_teacher_weights(args.teacher, device)
    teacher.load_state_dict(teacher_state, strict=False)

    # Initialize student from teacher
    student.load_state_dict(teacher.state_dict())

    # Dataset (supports directory, shard file, or shard index)
    dataset = SonataFlowDataset(args.data_dir, max_frames=args.max_frames)
    if len(dataset) == 0:
        raise ValueError(f"No data found at {args.data_dir}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(args.device != "cpu"),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    ema = EMA(student, decay=0.999)

    distiller = ConsistencyDistillation(
        teacher, student, cfg, device, aux_loss_weight=args.aux_loss_weight
    )
    os.makedirs(args.output_dir, exist_ok=True)

    if args.consistency:
        # --- DSFlow-style consistency training (novel) ---
        print(f"\n{'='*60}")
        print(f"CONSISTENCY FLOW TRAINING (1-step target)")
        print(f"Endpoint matching + velocity alignment + self-consistency")
        print(f"{'='*60}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.consistency_steps, eta_min=args.lr * 0.01
        )
        global_step = 0
        t0 = time.time()
        running_loss = 0.0

        for epoch in range(9999):
            for sem_tokens, aco_latents, lengths in loader:
                if global_step >= args.consistency_steps:
                    break
                sem_tokens = sem_tokens.to(device)
                aco_latents = aco_latents.to(device)

                loss = distiller.consistency_step(sem_tokens, aco_latents)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                ema.update()
                scheduler.step()

                running_loss += loss.item()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg = running_loss / args.log_every
                    elapsed = time.time() - t0
                    lr = optimizer.param_groups[0]["lr"]
                    print(f"  step {global_step:6d} | loss {avg:.4f} | "
                          f"lr {lr:.2e} | {args.log_every/elapsed:.1f} steps/s")
                    running_loss = 0.0
                    t0 = time.time()

                if global_step % (args.save_every * len(loader)) == 0:
                    ckpt_path = os.path.join(args.output_dir, f"flow_consistency_step_{global_step}.pt")
                    ema.apply_shadow()
                    torch.save(student.state_dict(), ckpt_path)
                    ema.restore()
                    print(f"  Saved: {ckpt_path}")

            if global_step >= args.consistency_steps:
                break

    else:
        # --- Progressive distillation (original) ---
        phases = [
            {"teacher_steps": 8, "student_steps": 4, "epochs": args.phase_epochs},
            {"teacher_steps": 4, "student_steps": 2, "epochs": args.phase_epochs},
            {"teacher_steps": 2, "student_steps": 1, "epochs": args.phase_epochs},
        ]

        global_step = 0
        for phase_idx, phase in enumerate(phases):
            t_steps = phase["teacher_steps"]
            s_steps = phase["student_steps"]
            n_epochs = phase["epochs"]

            print(f"\n{'='*60}")
            print(f"Phase {phase_idx+1}/3: Teacher={t_steps} steps → Student={s_steps} steps")
            print(f"{'='*60}")

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs * len(loader), eta_min=args.lr * 0.01
            )

            for epoch in range(n_epochs):
                epoch_loss = 0.0
                n_batches = 0
                t0 = time.time()

                for sem_tokens, aco_latents, lengths in loader:
                    sem_tokens = sem_tokens.to(device)
                    aco_latents = aco_latents.to(device)

                    loss = distiller.distill_step(
                        sem_tokens, aco_latents, t_steps, s_steps
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    ema.update()
                    scheduler.step()

                    epoch_loss += loss.item()
                    n_batches += 1
                    global_step += 1

                    if global_step % args.log_every == 0:
                        avg = epoch_loss / n_batches
                        lr = optimizer.param_groups[0]["lr"]
                        print(f"  step {global_step:6d} | loss {avg:.4f} | lr {lr:.2e}")

                elapsed = time.time() - t0
                avg_loss = epoch_loss / max(n_batches, 1)
                print(f"  Epoch {epoch+1}/{n_epochs} | loss {avg_loss:.4f} | {elapsed:.1f}s")

                if (epoch + 1) % args.save_every == 0 or epoch == n_epochs - 1:
                    ckpt_path = os.path.join(
                        args.output_dir,
                        f"flow_distilled_phase{phase_idx+1}_epoch{epoch+1}.pt",
                    )
                    ema.apply_shadow()
                    torch.save(student.state_dict(), ckpt_path)
                    ema.restore()
                    print(f"  Saved: {ckpt_path}")

            if phase_idx < len(phases) - 1:
                print(f"\n  Promoting student → teacher for next phase")
                distiller.teacher.load_state_dict(student.state_dict())

    # Export final model with EMA weights
    ema.apply_shadow()
    suffix = "consistency" if args.consistency else "distilled"
    final_path = os.path.join(args.output_dir, f"flow_{suffix}_1step.pt")
    torch.save(student.state_dict(), final_path)
    ema.restore()
    print(f"\nFinal 1-step model saved: {final_path}")

    # Also export as safetensors if available
    try:
        from safetensors.torch import save_file
        st_path = os.path.join(args.output_dir, "flow_distilled_1step.safetensors")
        save_file(student.state_dict(), st_path)
        print(f"SafeTensors export: {st_path}")
    except ImportError:
        pass


# ─── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consistency distillation for Sonata Flow"
    )
    parser.add_argument("--teacher", required=True, help="Path to teacher flow weights")
    parser.add_argument(
        "--teacher-config", required=True, help="Path to flow config JSON"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory of .pt training pairs"
    )
    parser.add_argument(
        "--output-dir", default="checkpoints/flow_distilled"
    )
    parser.add_argument(
        "--device", default="mps", choices=["mps", "cuda", "cpu"]
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--phase-epochs", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--aux-loss-weight",
        type=float,
        default=0.1,
        help="Weight for auxiliary OT-CFM loss against real data",
    )
    parser.add_argument(
        "--consistency", action="store_true",
        help="Use DSFlow-style consistency training instead of progressive distillation"
    )
    parser.add_argument(
        "--consistency-steps", type=int, default=50000,
        help="Total training steps for consistency mode"
    )
    args = parser.parse_args()

    train(args)
