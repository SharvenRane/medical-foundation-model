"""Data helpers for MedMNIST and for synthetic tensors used in tests.

The MedMNIST loader downloads a chosen 2D dataset (PathMNIST by default) and
returns plain float tensors in [0, 1] together with integer labels. The
synthetic helper builds class structured random data so tests stay offline and
fast while still being separable enough for a linear probe to learn from.
"""

from __future__ import annotations

import os

import numpy as np
import torch


def medmnist_cache_path(flag: str = "pathmnist") -> str:
    """Return the expected local cache path for a 2D MedMNIST dataset .npz file."""
    root = os.environ.get(
        "MEDMNIST_ROOT", os.path.join(os.path.expanduser("~"), ".medmnist")
    )
    return os.path.join(root, f"{flag}.npz")


def medmnist_is_cached(flag: str = "pathmnist") -> bool:
    """True when the dataset is already downloaded locally, so loading is offline."""
    return os.path.exists(medmnist_cache_path(flag))


def _to_tensor_images(images: np.ndarray) -> torch.Tensor:
    """Convert a uint8 MedMNIST image array to a float tensor (N, C, H, W)."""
    arr = images.astype(np.float32) / 255.0
    t = torch.from_numpy(arr)
    if t.dim() == 3:            # (N, H, W) grayscale
        t = t.unsqueeze(1)
    elif t.dim() == 4:         # (N, H, W, C)
        t = t.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"unexpected MedMNIST image array shape {images.shape}")
    return t


def load_medmnist(
    flag: str = "pathmnist",
    split: str = "train",
    limit: int | None = None,
):
    """Load a 2D MedMNIST split as ``(images, labels)`` tensors.

    Args:
        flag: MedMNIST dataset key, for example ``"pathmnist"`` or ``"bloodmnist"``.
        split: one of ``"train"``, ``"val"``, ``"test"``.
        limit: if given, keep only the first ``limit`` samples.

    Returns:
        images shaped (N, C, H, W) in [0, 1], labels shaped (N,) as int64.
    """
    import medmnist
    from medmnist import INFO

    info = INFO[flag]
    dataset_class = getattr(medmnist, info["python_class"])
    # Avoid a network fetch when the dataset is already cached locally.
    download = not medmnist_is_cached(flag)
    dataset = dataset_class(split=split, download=download)

    images = _to_tensor_images(dataset.imgs)
    labels = torch.from_numpy(dataset.labels.astype(np.int64)).squeeze(-1)
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)

    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]
    return images, labels


def make_synthetic_dataset(
    n_per_class: int = 40,
    n_classes: int = 3,
    channels: int = 1,
    size: int = 16,
    seed: int = 0,
):
    """Build a small class structured synthetic image dataset.

    Each class gets a distinct spatial bias pattern plus noise, so the images
    carry learnable structure without any download. Returns ``(images, labels)``
    with images in [0, 1].
    """
    g = torch.Generator().manual_seed(seed)
    images = []
    labels = []
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, size),
        torch.linspace(0, 1, size),
        indexing="ij",
    )

    # Each class carries a distinct oriented stripe texture plus a low frequency
    # gradient, with a random phase shift and additive noise per sample. This
    # gives the encoder learnable structure for the contrastive objective while
    # keeping every sample offline and cheap to generate.
    freq = max(3, size // 4)
    for c in range(n_classes):
        angle = np.pi * c / max(1, n_classes)
        proj = np.cos(angle) * xx + np.sin(angle) * yy
        for _ in range(n_per_class):
            shift = torch.rand(1, generator=g).item()
            pattern = 0.5 + 0.5 * torch.sin(2 * np.pi * freq * proj + 2 * np.pi * shift)
            pattern = pattern.unsqueeze(0).repeat(channels, 1, 1)
            noise = 0.12 * torch.randn(channels, size, size, generator=g)
            img = (pattern + noise).clamp(0.0, 1.0)
            images.append(img)
            labels.append(c)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.int64)

    perm = torch.randperm(images.shape[0], generator=g)
    return images[perm], labels[perm]
