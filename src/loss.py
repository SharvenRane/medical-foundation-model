"""The normalized temperature scaled cross entropy loss used by SimCLR (NT-Xent).

Given two batches of projected embeddings from two views of the same images,
the loss pulls matching views together and pushes every other pairing apart in
the embedding space.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Compute the NT-Xent contrastive loss.

    Args:
        z_a: projected embeddings of the first view, shaped (B, D).
        z_b: projected embeddings of the second view, shaped (B, D).
        temperature: softmax temperature.

    Returns:
        A scalar loss tensor.
    """
    if z_a.shape != z_b.shape:
        raise ValueError("the two view embeddings must share the same shape")
    batch_size = z_a.shape[0]

    z = torch.cat([z_a, z_b], dim=0)          # (2B, D)
    z = F.normalize(z, dim=1)

    similarity = z @ z.t()                     # (2B, 2B)
    similarity = similarity / temperature

    # Mask out self comparisons on the diagonal.
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    similarity = similarity.masked_fill(self_mask, float("-inf"))

    # For row i in [0, B), its positive is row i + B, and vice versa.
    targets = torch.arange(2 * batch_size, device=z.device)
    targets = (targets + batch_size) % (2 * batch_size)

    return F.cross_entropy(similarity, targets)
