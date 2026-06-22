"""Behavioural tests for the two headline claims of the project.

1. Self supervised pretraining drives the contrastive loss down. This runs on a
   small synthetic dataset so it is fast and fully offline.
2. The pretrained encoder's features beat a random encoder under a linear probe.
   This runs on real MedMNIST images, since randomly initialised CNNs are
   surprisingly strong feature extractors on clean synthetic textures and the
   honest gap shows up on real medical data. The test is skipped when the
   dataset is not already cached locally, so the suite never triggers a
   download.
"""

import pytest
import torch

from src.data import (
    load_medmnist,
    make_synthetic_dataset,
    medmnist_is_cached,
)
from src.pretrain import pretrain_simclr
from src.probe import linear_probe_accuracy, random_encoder_like

REAL_FLAG = "pathmnist"


def test_pretraining_loss_decreases():
    images, _ = make_synthetic_dataset(
        n_per_class=40, n_classes=3, channels=1, size=16, seed=0
    )
    result = pretrain_simclr(
        images,
        epochs=8,
        batch_size=32,
        embedding_dim=64,
        lr=1e-3,
        seed=0,
    )
    history = result.loss_history
    assert len(history) >= 3
    # The average of the last two epochs should sit clearly below the first.
    early = history[0]
    late = sum(history[-2:]) / 2
    assert late < early, f"loss did not decrease: {history}"


def test_probe_runs_on_synthetic_three_channel_input():
    """The probe pipeline should run end to end on multi channel input and return
    a valid accuracy. This is offline and does not assert a random gap."""
    train_images, train_labels = make_synthetic_dataset(
        n_per_class=20, n_classes=2, channels=3, size=16, seed=1
    )
    test_images, test_labels = make_synthetic_dataset(
        n_per_class=10, n_classes=2, channels=3, size=16, seed=2
    )
    result = pretrain_simclr(
        train_images, epochs=3, batch_size=16, embedding_dim=32, seed=0
    )
    acc = linear_probe_accuracy(
        result.model.encoder, train_images, train_labels, test_images, test_labels
    )
    assert 0.0 <= acc <= 1.0


@pytest.mark.skipif(
    not medmnist_is_cached(REAL_FLAG),
    reason=f"{REAL_FLAG} not cached locally; skipping to avoid a download",
)
def test_pretrained_beats_random_under_linear_probe_on_real_data():
    torch.manual_seed(0)
    pre_images, _ = load_medmnist(REAL_FLAG, split="train", limit=1500)
    tr_images, tr_labels = load_medmnist(REAL_FLAG, split="train", limit=400)
    te_images, te_labels = load_medmnist(REAL_FLAG, split="test", limit=400)

    result = pretrain_simclr(
        pre_images,
        epochs=6,
        batch_size=64,
        embedding_dim=128,
        lr=1e-3,
        seed=0,
    )
    pretrained_encoder = result.model.encoder
    random_encoder = random_encoder_like(pretrained_encoder, seed=123)

    pretrained_acc = linear_probe_accuracy(
        pretrained_encoder, tr_images, tr_labels, te_images, te_labels
    )
    random_acc = linear_probe_accuracy(
        random_encoder, tr_images, tr_labels, te_images, te_labels
    )

    # The contrastively pretrained encoder should beat the random baseline, and
    # both should clear the nine class chance level of about 0.11.
    assert pretrained_acc > random_acc, (
        f"pretrained {pretrained_acc:.3f} did not beat random {random_acc:.3f}"
    )
    assert pretrained_acc > 0.2
