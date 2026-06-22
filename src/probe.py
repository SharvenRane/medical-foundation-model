"""Linear probing utilities to evaluate the pretrained encoder.

Linear probing freezes the encoder, extracts embeddings for a small labeled
split, and fits a plain logistic regression on top. The accuracy of that probe
measures how much useful structure the self supervised encoder captured. A
random (untrained) encoder gives the baseline to beat.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .encoder import SmallEncoder


@torch.no_grad()
def extract_features(
    encoder: SmallEncoder,
    images: torch.Tensor,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Run images through a frozen encoder and return embeddings as a numpy array."""
    device = torch.device(device)
    encoder = encoder.to(device).eval()
    feats = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start:start + batch_size].to(device)
        feats.append(encoder(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def linear_probe_accuracy(
    encoder: SmallEncoder,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    max_iter: int = 1000,
    seed: int = 0,
) -> float:
    """Fit a logistic regression on frozen encoder features and return test accuracy."""
    x_train = extract_features(encoder, train_images, device=device)
    x_test = extract_features(encoder, test_images, device=device)
    y_train = train_labels.cpu().numpy()
    y_test = test_labels.cpu().numpy()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(x_train, y_train)
    return float(clf.score(x_test, y_test))


def random_encoder_like(encoder: SmallEncoder, seed: int = 123) -> SmallEncoder:
    """Build a fresh, randomly initialised encoder with the same architecture."""
    torch.manual_seed(seed)
    return SmallEncoder(
        in_channels=encoder.in_channels,
        embedding_dim=encoder.embedding_dim,
    )
