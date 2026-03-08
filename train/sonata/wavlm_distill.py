"""WavLM semantic distillation for first RVQ codebook.

The first codebook should capture semantic content (like Mimi codec).
We distill from WavLM-Large hidden states using contrastive + MSE loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WavLMDistillLoss(nn.Module):
    """Distillation loss between codec first-codebook embeddings and WavLM features.

    Uses a projection head to map codebook embeddings to WavLM feature space,
    then applies MSE + contrastive loss.
    """
    def __init__(self, codebook_dim: int = 128, wavlm_dim: int = 1024,
                 temperature: float = 0.07, mse_weight: float = 1.0,
                 contrastive_weight: float = 0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(codebook_dim, wavlm_dim),
            nn.GELU(),
            nn.Linear(wavlm_dim, wavlm_dim),
        )
        self.temperature = temperature
        self.mse_weight = mse_weight
        self.contrastive_weight = contrastive_weight

    def forward(self, codec_emb, wavlm_features):
        """
        Args:
            codec_emb: (batch, codebook_dim, time) — first codebook embeddings
            wavlm_features: (batch, wavlm_dim, time) — WavLM hidden states (downsampled to codec rate)
        Returns:
            loss: scalar
        """
        # Project codec embeddings to WavLM space
        codec_proj = self.proj(codec_emb.permute(0, 2, 1))  # (batch, time, wavlm_dim)
        target = wavlm_features.permute(0, 2, 1)  # (batch, time, wavlm_dim)

        # MSE loss
        mse = F.mse_loss(codec_proj, target.detach())

        # Contrastive loss (InfoNCE)
        # Normalize
        codec_norm = F.normalize(codec_proj.reshape(-1, codec_proj.shape[-1]), dim=-1)
        target_norm = F.normalize(target.reshape(-1, target.shape[-1]), dim=-1)

        # Similarity matrix
        logits = codec_norm @ target_norm.t() / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        contrastive = F.cross_entropy(logits, labels)

        return self.mse_weight * mse + self.contrastive_weight * contrastive
