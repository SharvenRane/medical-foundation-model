"""A small convolutional encoder used as the domain foundation backbone.

The encoder maps an image to a fixed length embedding vector. It is deliberately
small so that self supervised pretraining runs quickly on CPU. A projection head
sits on top of the encoder during contrastive pretraining and is discarded
afterwards, leaving the backbone embeddings for downstream linear probing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallEncoder(nn.Module):
    """A compact CNN backbone producing a flat embedding vector.

    Args:
        in_channels: number of input image channels (1 for grayscale MedMNIST,
            3 for the RGB variants).
        embedding_dim: size of the output embedding vector.
        widths: channel widths for the successive convolutional stages.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 128,
        widths: tuple[int, ...] = (16, 32, 64),
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        blocks = []
        prev = in_channels
        for w in widths:
            blocks.append(_conv_block(prev, w))
            prev = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(prev, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)


class ProjectionHead(nn.Module):
    """A two layer MLP head used only during contrastive pretraining."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """Encoder plus projection head, the full network optimised during SimCLR."""

    def __init__(self, encoder: SmallEncoder, projection: ProjectionHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection = projection

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        projected = self.projection(embedding)
        return embedding, projected
