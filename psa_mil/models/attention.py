"""Vectorized spatial multi-head attention with per-head distance pruning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from psa_mil.models.decay import DecayNetwork


class SpatialAttention(nn.Module):
    """Gaussian posterior attention restricted to each head's learned neighborhood.

    Logits match the released implementation::

        q·k / sqrt(d) - 0.5 (||q||^2 + ||k||^2) / sqrt(d) + log f(d | theta_h)

    which is -||q - k||^2 / (2 sqrt(d)) plus the log distance prior.
    Heads use different radii. Neighbors are gathered up to the largest radius
    and the extra positions are masked, so the head axis stays batched.
    Queries are chunked only to bound memory; the result is the same as one full pass.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        decay_type: str,
        decay_clip: float,
        min_local_k: float,
        max_local_k: float,
        init_k: float,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        chunk_size: int = 256,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.decay = DecayNetwork(
            decay_type,
            num_heads,
            decay_clip=decay_clip,
            min_local_k=min_local_k,
            max_local_k=max_local_k,
            init_k=init_k,
        )

    def forward(self, x: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        counts = self._neighbor_counts()
        k_max = max(1, min(length, int(counts.max().item())))
        scale = math.sqrt(self.head_dim)
        pieces = []
        for start in range(0, length, self.chunk_size):
            end = min(length, start + self.chunk_size)
            pieces.append(self._attend_queries(q[:, :, start:end], k, v, dist[:, start:end], counts, k_max, scale))
        out = torch.cat(pieces, dim=2).transpose(1, 2).reshape(batch, length, channels)
        return self.proj_drop(self.proj(out))

    def _neighbor_counts(self) -> torch.Tensor:
        radius = self.decay.local_k().detach()
        return torch.round(math.pi * radius ** 2).clamp(min=1)

    def _attend_queries(self, q, k, v, dist, counts, k_max, scale):
        dist_sel, index = torch.topk(dist, k=k_max, dim=-1, largest=False)
        keys = _gather_heads(k, index)
        values = _gather_heads(v, index)
        q_norm = (q ** 2).sum(dim=-1, keepdim=True)
        k_norm = (keys ** 2).sum(dim=-1)
        logits = torch.einsum("bhnd,bhnkd->bhnk", q, keys) / scale
        logits = logits - 0.5 * (q_norm + k_norm) / scale
        # Padded neighbors are at infinity. The prior is never used there (they are
        # masked below), but exp(-inf^2) backpropagates as NaN, which poisons theta.
        finite = torch.isfinite(dist_sel)
        safe_dist = torch.where(finite, dist_sel, torch.zeros_like(dist_sel))
        logits = logits + torch.log(self.decay(safe_dist) + 1e-6)
        positions = torch.arange(k_max, device=logits.device)
        within_radius = positions.view(1, 1, 1, k_max) < counts.view(1, -1, 1, 1)
        logits = logits.masked_fill(~(within_radius & finite.unsqueeze(1)), float("-inf"))
        weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        weights = self.attn_drop(weights)
        return torch.einsum("bhnk,bhnkd->bhnd", weights, values)


class SpatialBlock(nn.Module):
    def __init__(self, dim: int, attn: SpatialAttention, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = attn
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), dist)
        x = x + self.mlp(self.norm2(x))
        return x


def pairwise_distance(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Euclidean tile-grid distance. Padded tokens are at infinity so they are never neighbors."""
    dist = torch.cdist(coords.float(), coords.float())
    invalid = ~mask
    dist = dist.masked_fill(invalid[:, None, :], torch.inf)
    dist = dist.masked_fill(invalid[:, :, None], torch.inf)
    return dist


def _gather_heads(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """tokens (B, H, N, D), index (B, n, K) -> (B, H, n, K, D)."""
    batch, heads, _, dim = tokens.shape
    queries, k = index.shape[1], index.shape[2]
    expanded = tokens[:, :, None, :, :].expand(batch, heads, queries, tokens.shape[2], dim)
    gather_index = index[:, None, :, :, None].expand(batch, heads, queries, k, dim)
    return torch.gather(expanded, 3, gather_index)
