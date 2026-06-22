import torch

from src.augment import augment_once, make_two_views


def test_augment_preserves_shape_and_range():
    img = torch.rand(1, 28, 28)
    out = augment_once(img)
    assert out.shape == img.shape
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_two_views_differ():
    torch.manual_seed(0)
    batch = torch.rand(8, 1, 28, 28)
    view_a, view_b = make_two_views(batch)
    assert view_a.shape == batch.shape
    assert view_b.shape == batch.shape
    # Two independent augmentations of the same batch should not be identical.
    assert not torch.allclose(view_a, view_b)


def test_make_two_views_rejects_non_4d():
    bad = torch.rand(1, 28, 28)
    try:
        make_two_views(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised
