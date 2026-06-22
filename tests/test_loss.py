import torch
import torch.nn.functional as F

from src.loss import nt_xent_loss


def test_nt_xent_is_scalar_and_finite():
    z_a = torch.randn(8, 16)
    z_b = torch.randn(8, 16)
    loss = nt_xent_loss(z_a, z_b)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_nt_xent_lower_when_views_match():
    """Identical matched views should give a much lower loss than random pairs."""
    torch.manual_seed(0)
    base = torch.randn(16, 32)

    # Perfectly aligned positives: view B equals view A.
    aligned = nt_xent_loss(base, base.clone(), temperature=0.5)

    # Misaligned: view B is unrelated noise.
    misaligned = nt_xent_loss(base, torch.randn(16, 32), temperature=0.5)

    assert aligned < misaligned


def test_nt_xent_matches_manual_for_orthogonal_case():
    """When positives align and negatives are near orthogonal, loss approaches
    the analytic floor log(1 + something small). We check it is small."""
    torch.manual_seed(1)
    # Build clearly separated, mutually distinct unit vectors.
    z = F.normalize(torch.eye(8), dim=1)  # 8 orthogonal vectors, dim 8
    loss = nt_xent_loss(z, z.clone(), temperature=0.1)
    assert loss.item() < 0.5


def test_nt_xent_rejects_shape_mismatch():
    z_a = torch.randn(4, 8)
    z_b = torch.randn(5, 8)
    try:
        nt_xent_loss(z_a, z_b)
        raised = False
    except ValueError:
        raised = True
    assert raised
