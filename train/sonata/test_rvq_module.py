import torch
from rvq_module import ResidualVQ

def test_rvq_output_shape():
    rvq = ResidualVQ(input_dim=512, n_codebooks=8, codebook_size=2048, codebook_dim=128)
    z = torch.randn(2, 512, 50)  # (batch, channels, time)
    codes, quantized, commit_loss = rvq(z)
    assert codes.shape == (2, 8, 50), f"Expected (2,8,50), got {codes.shape}"
    assert quantized.shape == z.shape
    assert commit_loss.item() >= 0

def test_rvq_codebook_usage():
    rvq = ResidualVQ(input_dim=512, n_codebooks=8, codebook_size=2048, codebook_dim=128)
    z = torch.randn(4, 512, 100)
    codes, _, _ = rvq(z)
    # All codes should be valid indices
    assert (codes >= 0).all() and (codes < 2048).all()

def test_rvq_decode():
    rvq = ResidualVQ(input_dim=512, n_codebooks=8, codebook_size=2048, codebook_dim=128)
    z = torch.randn(1, 512, 10)
    codes, _, _ = rvq(z)
    z_hat = rvq.decode(codes)
    assert z_hat.shape == z.shape

if __name__ == "__main__":
    test_rvq_output_shape()
    test_rvq_codebook_usage()
    test_rvq_decode()
    print("All RVQ tests passed!")
