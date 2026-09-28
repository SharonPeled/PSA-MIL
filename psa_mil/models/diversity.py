"""Entropy diversity on per-head decay rates."""

from __future__ import annotations

import torch


def diversity_entropy(rates: torch.Tensor, kind: str) -> torch.Tensor:
    """Entropy of head decay rates. ``rates`` has shape (depth, heads), values in [0, 1]."""
    if rates.ndim == 1:
        rates = rates.unsqueeze(0)
    if rates.shape[-1] < 2:
        return rates.new_zeros(())
    if kind == "gaussian_binning":
        return _gaussian_binning(rates)
    if kind == "monte_carlo":
        return _kde_monte_carlo(rates.reshape(-1))
    raise ValueError(f"Unknown diversity loss: {kind}")


def _gaussian_binning(rates: torch.Tensor) -> torch.Tensor:
    heads = rates.shape[-1]
    centers = torch.linspace(0, 1, steps=heads, device=rates.device, dtype=rates.dtype)
    sigma = 0.1
    weights = torch.exp(-((rates.unsqueeze(-1) - centers) ** 2) / (2 * sigma ** 2))
    weights = weights / weights.sum(dim=-1, keepdim=True)
    probabilities = weights.mean(dim=1)
    entropy = -(probabilities * torch.log2(probabilities.clamp_min(1e-10))).sum(dim=-1)
    return entropy.mean()


def _kde_monte_carlo(rates: torch.Tensor, bandwidth: float = 0.1, num_samples: int = 1000) -> torch.Tensor:
    count = rates.shape[0]
    if count < 2:
        return rates.new_zeros(())
    picked = rates[torch.randint(0, count, (num_samples,), device=rates.device)]
    samples = picked + torch.randn_like(picked) * bandwidth
    pairwise = (samples.unsqueeze(0) - rates.unsqueeze(1)) ** 2
    kernel = torch.exp(-pairwise / (2 * bandwidth ** 2)) / (torch.sqrt(torch.tensor(2 * torch.pi, device=rates.device)) * bandwidth)
    density = kernel.mean(dim=0)
    return -torch.mean(torch.log(density.clamp_min(1e-10)))
