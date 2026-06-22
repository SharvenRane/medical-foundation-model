"""SimCLR style self supervised pretraining loop for the small encoder.

The loop draws batches of unlabeled images, builds two augmented views of each,
passes them through the encoder and projection head, and minimises the NT-Xent
loss. It returns the trained model and the per epoch loss history so callers
(and tests) can confirm the loss actually decreased.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, TensorDataset

from .augment import make_two_views
from .encoder import ProjectionHead, SimCLRModel, SmallEncoder
from .loss import nt_xent_loss


@dataclass
class PretrainResult:
    model: SimCLRModel
    loss_history: list[float] = field(default_factory=list)


def pretrain_simclr(
    images: torch.Tensor,
    *,
    epochs: int = 5,
    batch_size: int = 32,
    embedding_dim: int = 128,
    temperature: float = 0.5,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> PretrainResult:
    """Run SimCLR pretraining on a tensor of unlabeled images.

    Args:
        images: tensor shaped (N, C, H, W) in [0, 1].
        epochs: number of passes over the data.
        batch_size: SimCLR batch size (drives the number of negatives).
        embedding_dim: encoder output width.
        temperature: NT-Xent temperature.
        lr: Adam learning rate.
        seed: RNG seed for reproducibility.
        device: torch device.

    Returns:
        A :class:`PretrainResult` with the trained model and loss history.
    """
    torch.manual_seed(seed)
    device = torch.device(device)

    in_channels = images.shape[1]
    encoder = SmallEncoder(in_channels=in_channels, embedding_dim=embedding_dim)
    projection = ProjectionHead(in_dim=embedding_dim)
    model = SimCLRModel(encoder, projection).to(device)

    dataset = TensorDataset(images)
    # drop_last keeps every batch the same size so NT-Xent indexing stays valid.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) >= batch_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history: list[float] = []
    model.train()
    for _ in range(epochs):
        epoch_losses = []
        for (batch,) in loader:
            if batch.shape[0] < 2:
                continue
            batch = batch.to(device)
            view_a, view_b = make_two_views(batch)
            _, z_a = model(view_a)
            _, z_b = model(view_b)
            loss = nt_xent_loss(z_a, z_b, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if epoch_losses:
            loss_history.append(sum(epoch_losses) / len(epoch_losses))

    return PretrainResult(model=model, loss_history=loss_history)
