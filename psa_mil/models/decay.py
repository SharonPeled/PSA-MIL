"""Learnable distance-decay priors, one parameter per attention head."""

from __future__ import annotations

import torch
import torch.nn as nn


class DecayNetwork(nn.Module):
    def __init__(
        self,
        decay_type: str,
        num_heads: int,
        decay_clip: float = 1e-3,
        min_local_k: float = 1.0,
        max_local_k: float = 25.0,
        init_k: float = 7.0,
    ):
        super().__init__()
        if decay_type == "Cauchy":
            decay_type = "InverseQuadratic"
        self.decay_type = decay_type
        self.decay_clip = float(decay_clip)
        self.min_local_k = float(min_local_k)
        self.max_local_k = float(max_local_k)

        theta_lo = compute_theta_from_local_k(decay_type, self.min_local_k, self.decay_clip)
        theta_hi = compute_theta_from_local_k(decay_type, self.max_local_k, self.decay_clip)
        self.theta_min = min(theta_lo, theta_hi)
        self.theta_max = max(theta_lo, theta_hi)
        self.lambda_p = nn.Parameter(self._init_rates(init_k, num_heads))

    def _init_rates(self, target_k: float, num_heads: int) -> torch.Tensor:
        low = compute_theta_from_local_k(self.decay_type, target_k - 1, self.decay_clip)
        high = compute_theta_from_local_k(self.decay_type, target_k + 1, self.decay_clip)
        theta_lo, theta_hi = min(low, high), max(low, high)
        span = theta_hi - theta_lo
        theta_lo += 0.25 * span
        theta_hi -= 0.25 * span
        rate_lo = (theta_lo - self.theta_min) / (self.theta_max - self.theta_min)
        rate_hi = (theta_hi - self.theta_min) / (self.theta_max - self.theta_min)
        if num_heads == 1:
            return torch.tensor([(rate_lo + rate_hi) / 2])
        return torch.linspace(rate_lo, rate_hi, num_heads)

    def reparam(self, rates: torch.Tensor | None = None) -> torch.Tensor:
        rates = self.lambda_p if rates is None else rates
        rates = torch.clamp(rates, 0.0, 1.0)
        return self.theta_min + rates * (self.theta_max - self.theta_min)

    def rates(self) -> torch.Tensor:
        return torch.clamp(self.lambda_p, 0.0, 1.0)

    def local_k(self) -> torch.Tensor:
        return solve_for_local_k(self.decay_type, self.reparam(), self.decay_clip)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Distance (B, N, K) -> prior f(d | theta_h) with shape (B, H, N, K)."""
        theta = self.reparam().to(device=distance.device, dtype=distance.dtype)
        dist = distance.unsqueeze(1)
        theta = theta.view(1, -1, 1, 1)
        if self.decay_type == "Gaussian":
            return torch.exp(-(dist ** 2) / (2 * theta ** 2))
        if self.decay_type == "Exponential":
            return torch.exp(-theta * dist)
        if self.decay_type == "InverseQuadratic":
            return 1.0 / (1.0 + (dist / theta) ** 2)
        raise ValueError(f"Unknown decay type: {self.decay_type}")


def compute_theta_from_local_k(decay_type: str, local_k: float, decay_clip: float) -> float:
    clip = torch.tensor(float(decay_clip))
    radius = torch.tensor(float(local_k))
    if decay_type == "Cauchy":
        decay_type = "InverseQuadratic"
    if decay_type == "Gaussian":
        theta = torch.sqrt(radius ** 2 / (-2 * torch.log(clip)))
    elif decay_type == "Exponential":
        theta = -torch.log(clip) / radius
    elif decay_type == "InverseQuadratic":
        theta = radius / torch.sqrt((1 - clip) / clip)
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")
    return float(theta)


def solve_for_local_k(decay_type: str, param: torch.Tensor, decay_clip: float) -> torch.Tensor:
    clip = torch.tensor(float(decay_clip), dtype=param.dtype, device=param.device)
    if decay_type == "Cauchy":
        decay_type = "InverseQuadratic"
    if decay_type == "Gaussian":
        return torch.sqrt(-torch.log(clip) * 2 * param ** 2)
    if decay_type == "Exponential":
        return -torch.log(clip) / param
    if decay_type == "InverseQuadratic":
        return torch.sqrt(((1 - clip) / clip) * param ** 2)
    raise ValueError(f"Unknown decay type: {decay_type}")
