import torch

from src.encoder import ProjectionHead, SimCLRModel, SmallEncoder


def test_encoder_output_shape_grayscale():
    enc = SmallEncoder(in_channels=1, embedding_dim=64)
    x = torch.rand(5, 1, 28, 28)
    out = enc(x)
    assert out.shape == (5, 64)


def test_encoder_output_shape_rgb():
    enc = SmallEncoder(in_channels=3, embedding_dim=32)
    x = torch.rand(4, 3, 28, 28)
    out = enc(x)
    assert out.shape == (4, 32)


def test_simclr_model_returns_embedding_and_projection():
    enc = SmallEncoder(in_channels=1, embedding_dim=48)
    head = ProjectionHead(in_dim=48, out_dim=16)
    model = SimCLRModel(enc, head)
    x = torch.rand(3, 1, 16, 16)
    emb, proj = model(x)
    assert emb.shape == (3, 48)
    assert proj.shape == (3, 16)


def test_encoder_is_differentiable():
    enc = SmallEncoder(in_channels=1, embedding_dim=8)
    x = torch.rand(2, 1, 16, 16, requires_grad=True)
    out = enc(x).sum()
    out.backward()
    # Gradients should flow back to the input and to the first conv weights.
    assert x.grad is not None
    first_conv = enc.features[0][0]
    assert first_conv.weight.grad is not None
    assert torch.isfinite(first_conv.weight.grad).all()
