"""Load a task folder: config.yaml plus a fold table (k=all.tsv).

The folder layout matches Patho-Bench tasks: a YAML description of the label
column and a TSV with one row per slide, predefined ``fold_*`` assignments,
``case_id``, and ``slide_id``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_task(task_dir: str | Path, splits_name: str = "k=all.tsv") -> dict:
    task_dir = Path(task_dir)
    config_path = task_dir / "config.yaml"
    splits_path = task_dir / splits_name
    if not config_path.is_file():
        raise FileNotFoundError(f"Task config not found: {config_path}")
    if not splits_path.is_file():
        raise FileNotFoundError(f"Task splits not found: {splits_path}")

    with config_path.open() as handle:
        meta = yaml.safe_load(handle) or {}
    task_type = meta.get("task_type", "classification")
    if task_type != "classification":
        raise NotImplementedError(
            f"This trainer supports classification tasks. {config_path} has task_type={task_type!r}."
        )
    task_col = meta.get("task_col")
    sample_col = meta.get("sample_col", "case_id")
    if not task_col:
        raise ValueError(f"{config_path} is missing task_col")

    frame = pd.read_csv(splits_path, sep="\t")
    for column in (sample_col, "slide_id", task_col):
        if column not in frame.columns:
            raise ValueError(f"{splits_path} is missing column {column!r}")

    fold_cols = [c for c in frame.columns if c.startswith("fold_")]
    if not fold_cols:
        raise ValueError(f"{splits_path} has no fold_* columns")
    fold_ids = sorted(int(c.split("_", 1)[1]) for c in fold_cols)

    labels = pd.to_numeric(frame[task_col], errors="raise").astype(int)
    classes = sorted(labels.unique().tolist())
    remap = {label: index for index, label in enumerate(classes)}
    label_dict = {int(k): str(v) for k, v in (meta.get("label_dict") or {}).items()}
    class_names = [label_dict.get(label, str(label)) for label in classes]

    out = pd.DataFrame(
        {
            "case_id": frame[sample_col].astype(str),
            "slide_id": frame["slide_id"].astype(str),
            "label": labels.map(remap).astype(int),
        }
    )
    for fold in fold_ids:
        out[f"fold_{fold}"] = frame[f"fold_{fold}"].astype(str).str.lower()

    return {
        "frame": out,
        "fold_ids": fold_ids,
        "num_classes": len(classes),
        "class_names": class_names,
        "task_col": task_col,
        "metrics": list(meta.get("metrics") or ["macro-ovr-auc"]),
        "task_dir": str(task_dir),
    }
