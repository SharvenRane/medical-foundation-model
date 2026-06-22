import torch

from src.data import make_synthetic_dataset


def test_synthetic_shapes_and_balance():
    images, labels = make_synthetic_dataset(
        n_per_class=10, n_classes=4, channels=1, size=12, seed=3
    )
    assert images.shape == (40, 1, 12, 12)
    assert labels.shape == (40,)
    # All four classes present, balanced.
    counts = torch.bincount(labels, minlength=4)
    assert (counts == 10).all()


def test_synthetic_values_in_unit_range():
    images, _ = make_synthetic_dataset(n_per_class=5, n_classes=2, channels=3, size=8)
    assert images.min() >= 0.0
    assert images.max() <= 1.0
    assert images.shape[1] == 3


def test_synthetic_is_deterministic():
    a_img, a_lab = make_synthetic_dataset(seed=7)
    b_img, b_lab = make_synthetic_dataset(seed=7)
    assert torch.allclose(a_img, b_img)
    assert torch.equal(a_lab, b_lab)
