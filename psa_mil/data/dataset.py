"""Slide bags from a task table and a TRIDENT feature directory."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from psa_mil.data.trident import TridentReader


class SlideDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        reader: TridentReader,
        max_tiles: int | None = None,
        fixed_subsample: bool = False,
    ):
        self.frame = frame.reset_index(drop=True)
        self.reader = reader
        self.max_tiles = max_tiles
        self.fixed_subsample = fixed_subsample

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        features, coords = self.reader.load(row.slide_id)
        if self.max_tiles is not None and features.shape[0] > self.max_tiles:
            if self.fixed_subsample:
                generator = torch.Generator()
                generator.manual_seed(_stable_seed(str(row.slide_id)))
                choice = torch.randperm(features.shape[0], generator=generator)[: self.max_tiles]
            else:
                choice = torch.randperm(features.shape[0])[: self.max_tiles]
            features = features[choice]
            coords = coords[choice]
        return {
            "features": features,
            "coords": coords,
            "label": int(row.label),
            "slide_id": str(row.slide_id),
            "case_id": str(row.case_id),
        }


def collate_slides(samples: list[dict]) -> dict:
    width = samples[0]["features"].shape[-1]
    length = max(sample["features"].shape[0] for sample in samples)
    features = torch.zeros(len(samples), length, width)
    coords = torch.zeros(len(samples), length, 2)
    mask = torch.zeros(len(samples), length, dtype=torch.bool)
    labels = []
    for i, sample in enumerate(samples):
        n = sample["features"].shape[0]
        features[i, :n] = sample["features"]
        coords[i, :n] = sample["coords"]
        mask[i, :n] = True
        labels.append(sample["label"])
    return {
        "features": features,
        "coords": coords,
        "mask": mask,
        "label": torch.tensor(labels, dtype=torch.long),
        "slide_id": [sample["slide_id"] for sample in samples],
        "case_id": [sample["case_id"] for sample in samples],
    }


def _stable_seed(slide_id: str) -> int:
    return int.from_bytes(slide_id.encode(), "little") % (2**31 - 1)


def attach_features(frame: pd.DataFrame, reader: TridentReader) -> tuple[pd.DataFrame, list[str]]:
    present = frame["slide_id"].map(reader.has_slide)
    missing = frame.loc[~present, "slide_id"].tolist()
    return frame.loc[present].reset_index(drop=True), missing
