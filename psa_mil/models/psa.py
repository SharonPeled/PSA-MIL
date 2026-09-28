"""PSA-MIL classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from psa_mil.models.attention import SpatialAttention, SpatialBlock, pairwise_distance
from psa_mil.models.diversity import diversity_entropy


class ResidualAdapter(nn.Module):
    """Linear map plus residual blocks. ``num_layers`` counts the first projection."""

    def __init__(self, in_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        if num_layers <= 0:
            self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
            self.blocks = nn.ModuleList()
            return
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(out_dim, out_dim, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim, bias=False),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_layers - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers <= 0:
            return self.proj(x)
        x = self.proj(x)
        for block in self.blocks:
            x = x + block(x)
        return x


class PSAMIL(nn.Module):
    """Probabilistic spatial-attention MIL.

    Args:
        features: ``(N, D)`` or ``(B, N, D)`` tile embeddings.
        coords: matching ``(N, 2)`` or ``(B, N, 2)`` coordinates in tile-grid units
            (TRIDENT level-0 pixels divided by ``patch_size_level0``).
        mask: optional ``(B, N)`` bool, True for real tiles. Required when a batch is padded.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_heads: int = 3,
        depth: int = 1,
        attn_dim: int = 96,
        num_residual_layers: int = 2,
        pool_type: str = "attention",
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_chunk: int = 256,
        decay_type: str = "Gaussian",
        decay_clip: float = 1e-3,
        min_local_k: float = 1.0,
        max_local_k: float = 25.0,
        init_k: float = 7.0,
        div_loss: str = "gaussian_binning",
        alpha: float = 0.001,
    ):
        super().__init__()
        if pool_type not in {"attention", "mean"}:
            raise ValueError(f"pool_type must be 'attention' or 'mean', got {pool_type!r}")
        model_dim = attn_dim * num_heads
        self.pool_type = pool_type
        self.div_loss = div_loss
        self.alpha = float(alpha)
        self.model_kwargs = {
            "embed_dim": embed_dim,
            "num_classes": num_classes,
            "num_heads": num_heads,
            "depth": depth,
            "attn_dim": attn_dim,
            "num_residual_layers": num_residual_layers,
            "pool_type": pool_type,
            "qkv_bias": qkv_bias,
            "mlp_ratio": mlp_ratio,
            "dropout": dropout,
            "attention_chunk": attention_chunk,
            "decay_type": decay_type,
            "decay_clip": decay_clip,
            "min_local_k": min_local_k,
            "max_local_k": max_local_k,
            "init_k": init_k,
            "div_loss": div_loss,
            "alpha": alpha,
        }
        self.adapter = ResidualAdapter(embed_dim, model_dim, num_residual_layers)
        self.blocks = nn.ModuleList(
            SpatialBlock(
                model_dim,
                SpatialAttention(
                    model_dim,
                    num_heads,
                    decay_type=decay_type,
                    decay_clip=decay_clip,
                    min_local_k=min_local_k,
                    max_local_k=max_local_k,
                    init_k=init_k,
                    qkv_bias=qkv_bias,
                    attn_drop=dropout,
                    proj_drop=dropout,
                    chunk_size=attention_chunk,
                ),
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(model_dim, eps=1e-6)
        self.attention_pool = nn.Linear(model_dim, 1) if pool_type == "attention" else None
        self.head = nn.Linear(model_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squeezed = features.dim() == 2
        if squeezed:
            features = features.unsqueeze(0)
            coords = coords.unsqueeze(0)
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        elif squeezed:
            mask = mask.unsqueeze(0)

        x = self.adapter(features)
        dist = pairwise_distance(coords, mask)
        for block in self.blocks:
            x = block(x, dist)
        x = self.norm(x)
        pooled = self._pool(x, mask)
        logits = self.head(self.dropout(pooled))
        return logits.squeeze(0) if squeezed else logits

    def _pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
            return (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        scores = self.attention_pool(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)

    def decay_rates(self) -> torch.Tensor:
        """Clamped decay rates, shape (depth, heads)."""
        return torch.stack([block.attn.decay.rates() for block in self.blocks])

    def local_ks(self) -> torch.Tensor:
        return torch.stack([block.attn.decay.local_k() for block in self.blocks])

    def diversity_loss(self) -> torch.Tensor:
        if self.alpha <= 0:
            return self.decay_rates().new_zeros(())
        return diversity_entropy(self.decay_rates(), self.div_loss)

    @classmethod
    def from_checkpoint(cls, path: str, map_location: str = "cpu") -> "PSAMIL":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def export_state(self) -> dict:
        return {"model_kwargs": self.model_kwargs, "state_dict": self.state_dict()}
