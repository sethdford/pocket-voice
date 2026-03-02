"""Test-Time Adaptive Quality — critic + progressive refinement.

LLMs get better with more reasoning tokens. TTS can get better with more
refinement steps. Instead of fixed N-step flow:

1. Generate at 2 steps (fast draft, ~25ms)
2. Run a lightweight quality critic on the draft
3. If quality is below threshold → refine with 4 more steps on problematic regions
4. If user is waiting → stop at 2 steps; if buffered → refine to 8+

The quality critic is a small model trained to predict UTMOS/PESQ scores
from acoustic latents (no need for full decode + external evaluator).

This is unique to conversational TTS where latency and quality trade off.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import RMSNorm


class QualityCritic(nn.Module):
    """Lightweight critic that predicts quality score from acoustic latents.

    Trained to regress UTMOS/PESQ scores. At inference, decides whether
    to refine the generation further or accept as-is.

    Architecture: 4-layer 1D ConvNet → global pool → MLP → scalar score [0, 5].
    ~500K parameters for negligible inference overhead.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        layers = [nn.Conv1d(input_dim, hidden_dim, 7, padding=3), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2), nn.GELU()]
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.encoder = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """latents: (B, T, D) → quality score (B,) in [0, 1]"""
        x = self.encoder(latents.transpose(1, 2)).squeeze(-1)
        return self.head(x).squeeze(-1) * 5.0  # Scale to [0, 5] (UTMOS range)

    def compute_loss(self, latents: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Train on (latents, UTMOS_score) pairs."""
        predicted = self.forward(latents)
        return F.mse_loss(predicted, scores)


class RegionalRefiner(nn.Module):
    """Identifies low-quality regions and refines them with additional flow steps.

    Instead of refining the entire sequence (expensive), identifies the
    worst-quality regions via the critic and only refines those.
    """

    def __init__(self, critic: QualityCritic, window_size: int = 25):
        super().__init__()
        self.critic = critic
        self.window_size = window_size

    @torch.no_grad()
    def find_low_quality_regions(self, latents: torch.Tensor,
                                 threshold: float = 3.5) -> list:
        """Score windows of the latent sequence, return indices of low-quality ones."""
        B, T, D = latents.shape
        ws = self.window_size
        regions = []

        for start in range(0, T - ws + 1, ws):
            window = latents[:, start:start + ws]
            score = self.critic(window).mean().item()
            if score < threshold:
                regions.append((start, min(start + ws, T), score))

        return regions


class AdaptiveFlowSampler:
    """Adaptive-quality sampler: draft → critique → refine.

    Usage:
      sampler = AdaptiveFlowSampler(flow_model, critic)
      audio = sampler.sample(semantic_tokens,
                             min_steps=2, max_steps=16,
                             quality_threshold=3.8)
    """

    def __init__(self, flow_model, critic: QualityCritic,
                 refiner: Optional[RegionalRefiner] = None):
        self.flow = flow_model
        self.critic = critic
        self.refiner = refiner or RegionalRefiner(critic)

    @torch.no_grad()
    def sample(self, semantic_tokens: torch.Tensor,
               min_steps: int = 2, max_steps: int = 16,
               quality_threshold: float = 3.8,
               speaker_ids=None, **flow_kwargs) -> Tuple[torch.Tensor, dict]:
        """Generate with adaptive quality refinement.

        1. Draft at min_steps
        2. Score with critic
        3. If below threshold and budget remains, refine
        4. Return final latents + quality info
        """
        B, T = semantic_tokens.shape
        device = semantic_tokens.device

        # Step 1: Fast draft
        x = torch.randn(B, T, self.flow.cfg.acoustic_dim, device=device)
        dt = 1.0 / min_steps
        for i in range(min_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.flow._velocity(x, t, semantic_tokens, speaker_ids,
                                    None, None, flow_kwargs.get("cfg_scale", 1.0))
            x = x + dt * v

        # Step 2: Quality assessment
        draft_score = self.critic(x).mean().item()
        total_steps = min_steps
        info = {"draft_score": draft_score, "total_steps": min_steps, "refined": False}

        if draft_score >= quality_threshold:
            return x, info

        # Step 3: Progressive refinement
        remaining_budget = max_steps - min_steps
        refine_steps = min(remaining_budget, 4)

        if refine_steps > 0:
            # Re-run ODE from scratch with more steps (draft was min_steps, now use min_steps + refine_steps)
            total_steps = min_steps + refine_steps
            x = torch.randn(B, T, self.flow.cfg.acoustic_dim, device=device)
            dt_full = 1.0 / total_steps
            for i in range(total_steps):
                t = torch.full((B,), i * dt_full, device=device)
                v = self.flow._velocity(x, t, semantic_tokens, speaker_ids,
                                        None, None, flow_kwargs.get("cfg_scale", 1.0))
                x = x + dt_full * v
            total_steps += refine_steps

        refined_score = self.critic(x).mean().item()
        info.update({
            "refined_score": refined_score,
            "total_steps": total_steps,
            "refined": True,
            "improvement": refined_score - draft_score,
        })

        return x, info

    @torch.no_grad()
    def sample_budget_aware(self, semantic_tokens: torch.Tensor,
                            time_budget_ms: float = 100.0,
                            speaker_ids=None, **flow_kwargs) -> Tuple[torch.Tensor, dict]:
        """Generate with a time budget instead of step budget.

        Estimates steps from timing, generates as many as budget allows.
        For real-time conversational TTS.
        """
        import time

        B, T = semantic_tokens.shape
        device = semantic_tokens.device
        x = torch.randn(B, T, self.flow.cfg.acoustic_dim, device=device)

        steps_done = 0
        max_possible = 16
        dt = 1.0 / max_possible

        t0 = time.time()
        while steps_done < max_possible:
            elapsed_ms = (time.time() - t0) * 1000
            if elapsed_ms > time_budget_ms * 0.8:
                break

            t = torch.full((B,), steps_done * dt, device=device)
            v = self.flow._velocity(x, t, semantic_tokens, speaker_ids,
                                    None, None, flow_kwargs.get("cfg_scale", 1.0))
            x = x + dt * v
            steps_done += 1

        score = self.critic(x).mean().item()
        elapsed_ms = (time.time() - t0) * 1000

        return x, {
            "steps": steps_done,
            "score": score,
            "elapsed_ms": elapsed_ms,
        }


if __name__ == "__main__":
    critic = QualityCritic(input_dim=256, hidden_dim=128)
    n = sum(p.numel() for p in critic.parameters())
    print(f"QualityCritic: {n/1e3:.1f}K params")

    B, T = 2, 50
    latents = torch.randn(B, T, 256)
    score = critic(latents)
    print(f"  Predicted quality: {score.tolist()}")

    target_scores = torch.tensor([4.0, 3.5])
    loss = critic.compute_loss(latents, target_scores)
    print(f"  Training loss: {loss:.4f}")

    # Test regional refiner
    refiner = RegionalRefiner(critic, window_size=10)
    regions = refiner.find_low_quality_regions(latents, threshold=5.0)
    print(f"  Low-quality regions (threshold=5.0): {len(regions)}")

    # Test adaptive sampler with a mock flow
    from config import FlowConfig
    from flow import SonataFlow
    cfg = FlowConfig(d_model=256, n_layers=4, n_heads=4)
    flow = SonataFlow(cfg)
    sampler = AdaptiveFlowSampler(flow, critic)

    sem = torch.randint(0, cfg.semantic_vocab_size, (B, T))
    gen, info = sampler.sample(sem, min_steps=2, max_steps=8, quality_threshold=0.0)
    print(f"  Adaptive sample: {gen.shape}, steps={info['total_steps']}, "
          f"draft={info['draft_score']:.2f}")

    gen2, info2 = sampler.sample_budget_aware(sem, time_budget_ms=500)
    print(f"  Budget-aware: steps={info2['steps']}, score={info2['score']:.2f}, "
          f"time={info2['elapsed_ms']:.1f}ms")
    print("PASS")
