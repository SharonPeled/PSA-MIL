"""Shape, gradient, and schedule checks. No dataset required."""

import torch

from psa_mil.models.decay import compute_theta_from_local_k, solve_for_local_k
from psa_mil.models.psa import PSAMIL
from psa_mil.training.schedule import piecewise_linear


def test_decay_radius_roundtrip():
    for kind in ("Gaussian", "Exponential", "InverseQuadratic"):
        theta = compute_theta_from_local_k(kind, 7.0, 0.001)
        radius = solve_for_local_k(kind, torch.tensor(theta), 0.001)
        assert abs(float(radius) - 7.0) < 1e-3, (kind, float(radius))


def test_forward_and_backward():
    torch.manual_seed(0)
    model = PSAMIL(
        embed_dim=32,
        num_classes=2,
        num_heads=3,
        depth=2,
        attn_dim=8,
        num_residual_layers=2,
        attention_chunk=8,
        alpha=0.001,
    )
    features = torch.randn(2, 20, 32)
    coords = torch.randint(0, 12, (2, 20, 2)).float()
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[1, 14:] = False
    logits = model(features, coords, mask)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    loss = logits.sum() - model.alpha * model.diversity_loss()
    loss.backward()
    grad = model.blocks[0].attn.decay.lambda_p.grad
    assert grad is not None
    assert torch.isfinite(grad).all()

    single = model(features[0, :14], coords[0, :14])
    assert single.shape == (2,)
    assert torch.isfinite(single).all()


def test_schedule_covers_every_step():
    segments = [
        {"from": 1e-6, "to": 1e-4, "steps": 0.1},
        {"from": 1e-4, "to": 1e-4, "steps": 0.9},
        {"from": 1e-4, "to": 1e-6, "steps": -1},
    ]
    values = piecewise_linear(segments, 100)
    assert len(values) == 100
    assert abs(values[0] - 1e-6) < 1e-12
    assert values[-1] < 1e-4


if __name__ == "__main__":
    test_decay_radius_roundtrip()
    test_forward_and_backward()
    test_schedule_covers_every_step()
    print("ok")
