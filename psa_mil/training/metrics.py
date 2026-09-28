"""Slide predictions aggregated to the case, then scored."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def predictions_frame(chunks: list[dict], class_names: list[str]) -> pd.DataFrame:
    rows = []
    for chunk in chunks:
        probs = chunk["probs"].numpy()
        labels = chunk["label"].numpy()
        for i, (case_id, slide_id) in enumerate(zip(chunk["case_id"], chunk["slide_id"])):
            row = {
                "case_id": case_id,
                "slide_id": slide_id,
                "y_true": int(labels[i]),
                "y_pred": int(probs[i].argmax()),
            }
            for class_index, name in enumerate(class_names):
                row[f"prob_{class_index}"] = float(probs[i, class_index])
                row[f"prob_{name}"] = float(probs[i, class_index])
            rows.append(row)
    return pd.DataFrame(rows)


def case_metrics(slides: pd.DataFrame, num_classes: int) -> dict[str, float]:
    prob_cols = [f"prob_{i}" for i in range(num_classes)]
    cases = (
        slides.groupby("case_id", as_index=False)
        .agg({**{col: "mean" for col in prob_cols}, "y_true": "first"})
    )
    y_true = cases["y_true"].to_numpy()
    probs = cases[prob_cols].to_numpy()
    y_pred = probs.argmax(axis=1)
    scores = probs[:, 1] if num_classes == 2 else probs
    return {
        "n_cases": int(len(cases)),
        "n_slides": int(len(slides)),
        "auc": _safe_auc(y_true, scores, multiclass=num_classes > 2),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary" if num_classes == 2 else "macro", zero_division=0)),
    }


def _safe_auc(y_true: np.ndarray, scores: np.ndarray, multiclass: bool) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if multiclass:
        return float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro"))
    return float(roc_auc_score(y_true, scores))
