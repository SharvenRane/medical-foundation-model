"""Stochastic augmentations that build the two correlated views SimCLR needs.

Each call produces a pair of randomly transformed versions of the same image.
The transforms stay light so they run fast and remain sensible for medical
grayscale or RGB images: random crop and resize, horizontal flip, small
brightness and contrast jitter, and Gaussian noise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _random_resized_crop(img: torch.Tensor, scale_min: float = 0.6) -> torch.Tensor:
    """Crop a random square region covering at least ``scale_min`` of the area,
    then resize back to the original spatial size.

    Args:
        img: tensor shaped (C, H, W).
    """
    c, h, w = img.shape
    area = h * w
    target_area = (scale_min + (1.0 - scale_min) * torch.rand(1).item()) * area
    side = int(round(target_area ** 0.5))
    side = max(1, min(side, min(h, w)))
    top = torch.randint(0, h - side + 1, (1,)).item()
    left = torch.randint(0, w - side + 1, (1,)).item()
    crop = img[:, top:top + side, left:left + side]
    crop = crop.unsqueeze(0)
    resized = F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)
    return resized.squeeze(0)


def _maybe_hflip(img: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    if torch.rand(1).item() < p:
        return torch.flip(img, dims=[2])
    return img


def _brightness_contrast(img: torch.Tensor, jitter: float = 0.3) -> torch.Tensor:
    brightness = 1.0 + (torch.rand(1).item() * 2 - 1) * jitter
    contrast = 1.0 + (torch.rand(1).item() * 2 - 1) * jitter
    mean = img.mean()
    out = (img - mean) * contrast + mean * brightness
    return out.clamp(0.0, 1.0)


def _gaussian_noise(img: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    return (img + torch.randn_like(img) * sigma).clamp(0.0, 1.0)


def augment_once(img: torch.Tensor) -> torch.Tensor:
    """Apply the full random augmentation pipeline to a single image (C, H, W)."""
    out = _random_resized_crop(img)
    out = _maybe_hflip(out)
    out = _brightness_contrast(out)
    out = _gaussian_noise(out)
    return out


def make_two_views(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build two independently augmented views for every image in a batch.

    Args:
        batch: tensor shaped (B, C, H, W) with values in [0, 1].

    Returns:
        A pair of tensors, each shaped like ``batch``.
    """
    if batch.dim() != 4:
        raise ValueError(f"expected a 4D batch, got shape {tuple(batch.shape)}")
    view_a = torch.stack([augment_once(img) for img in batch])
    view_b = torch.stack([augment_once(img) for img in batch])
    return view_a, view_b
