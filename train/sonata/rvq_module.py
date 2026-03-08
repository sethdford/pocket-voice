import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        self.codebook_size = codebook_size

    def forward(self, z):
        # z: (batch, dim, time) -> (batch, time, dim)
        z_t = z.permute(0, 2, 1)
        # L2 distance: ||z - e||^2
        d = (z_t.pow(2).sum(-1, keepdim=True)
             + self.codebook.weight.pow(2).sum(-1)
             - 2 * z_t @ self.codebook.weight.t())
        codes = d.argmin(-1)  # (batch, time)
        quantized = self.codebook(codes).permute(0, 2, 1)  # (batch, dim, time)
        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()
        commit_loss = F.mse_loss(quantized.detach(), z)
        return codes, quantized_st, commit_loss

    def decode(self, codes):
        return self.codebook(codes).permute(0, 2, 1)


class ResidualVQ(nn.Module):
    def __init__(self, input_dim: int, n_codebooks: int = 8,
                 codebook_size: int = 2048, codebook_dim: int = 128):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.project_in = nn.Linear(input_dim, codebook_dim) if input_dim != codebook_dim else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, input_dim) if input_dim != codebook_dim else nn.Identity()
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, codebook_dim)
            for _ in range(n_codebooks)
        ])

    def forward(self, z):
        # z: (batch, channels, time)
        # Project to codebook dim
        z_t = z.permute(0, 2, 1)  # (batch, time, channels)
        z_proj = self.project_in(z_t).permute(0, 2, 1)  # (batch, codebook_dim, time)

        residual = z_proj
        all_codes = []
        total_commit = 0.0
        quantized_sum = torch.zeros_like(z_proj)

        for vq in self.quantizers:
            codes, quantized, commit = vq(residual)
            all_codes.append(codes.unsqueeze(1))
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized.detach()
            total_commit = total_commit + commit

        codes = torch.cat(all_codes, dim=1)  # (batch, n_codebooks, time)

        # Project back to input dim
        out_t = quantized_sum.permute(0, 2, 1)  # (batch, time, codebook_dim)
        out = self.project_out(out_t).permute(0, 2, 1)  # (batch, channels, time)

        return codes, out, total_commit / self.n_codebooks

    def decode(self, codes):
        # codes: (batch, n_codebooks, time)
        quantized_sum = None
        for i, vq in enumerate(self.quantizers):
            book_codes = codes[:, i, :]  # (batch, time)
            q = vq.decode(book_codes)
            quantized_sum = q if quantized_sum is None else quantized_sum + q
        out_t = quantized_sum.permute(0, 2, 1)
        return self.project_out(out_t).permute(0, 2, 1)
