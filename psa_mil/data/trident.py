"""Read TRIDENT patch-feature HDF5 files.

Each slide is ``{trident_dir}/features_{encoder}/{slide_id}.h5`` with datasets
``features`` (N, D) and ``coords`` (N, 2). Coordinates are level-0 pixels.
PSA-MIL decay is defined in tile steps, so coordinates are divided by the
``patch_size_level0`` attribute.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


class TridentReader:
    def __init__(self, trident_dir: str | Path, encoder: str):
        self.root = Path(trident_dir)
        self.encoder = encoder
        self.feature_dir = self.root / f"features_{encoder}"
        if not self.feature_dir.is_dir():
            raise FileNotFoundError(
                f"TRIDENT feature directory not found: {self.feature_dir}. "
                f"Expected features_{encoder}/ under {self.root}."
            )

    def path_for(self, slide_id: str) -> Path:
        return self.feature_dir / f"{slide_id}.h5"

    def has_slide(self, slide_id: str) -> bool:
        return self.path_for(slide_id).is_file()

    def embed_dim(self, slide_id: str) -> int:
        features, _ = self.load(slide_id)
        return int(features.shape[-1])

    def load(self, slide_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.path_for(slide_id)
        if not path.is_file():
            raise FileNotFoundError(f"No TRIDENT features for slide {slide_id}: {path}")
        with h5py.File(path, "r") as handle:
            if "features" not in handle or "coords" not in handle:
                raise KeyError(f"{path} must contain 'features' and 'coords' datasets")
            features = _drop_leading_one(np.asarray(handle["features"]))
            coords = _drop_leading_one(np.asarray(handle["coords"]))
            attrs = dict(handle["coords"].attrs)
        if features.ndim != 2:
            raise ValueError(f"{path} features should be (N, D), got {features.shape}")
        if coords.ndim != 2 or coords.shape[-1] != 2:
            raise ValueError(f"{path} coords should be (N, 2), got {coords.shape}")
        if features.shape[0] != coords.shape[0]:
            raise ValueError(
                f"{path} has {features.shape[0]} features and {coords.shape[0]} coordinates"
            )
        patch_size = _patch_size_level0(attrs, path)
        grid = coords.astype(np.float32) / patch_size
        return torch.from_numpy(features.astype(np.float32)), torch.from_numpy(grid)


def _drop_leading_one(array: np.ndarray) -> np.ndarray:
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    return array


def _patch_size_level0(attrs: dict, path: Path) -> float:
    if "patch_size_level0" in attrs:
        size = float(attrs["patch_size_level0"])
    elif "patch_size" in attrs and "level0_magnification" in attrs and "target_magnification" in attrs:
        scale = float(attrs["level0_magnification"]) / float(attrs["target_magnification"])
        size = float(attrs["patch_size"]) * scale
    else:
        raise KeyError(
            f"{path} coords are missing patch_size_level0, so tile-grid distances cannot be computed"
        )
    if size <= 0:
        raise ValueError(f"{path} has non-positive patch size at level 0: {size}")
    return size
