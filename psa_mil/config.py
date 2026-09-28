"""YAML config loading."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "experiment": {
        "name": "psa_mil",
        "run_name": "run",
        "description": "",
        "save_dir": "runs/psa_mil",
        "seed": 1234,
        "mlflow_dir": None,
    },
    "data": {
        "task_dir": None,
        "trident_dir": None,
        "encoder": "conch_v15",
        "max_tiles": None,
    },
    "model": {
        "embed_dim": None,
        "num_heads": 3,
        "depth": 1,
        "attn_dim": 96,
        "num_residual_layers": 2,
        "pool_type": "attention",
        "qkv_bias": True,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "attention_chunk": 256,
        "decay": {
            "type": "Gaussian",
            "alpha": 0.001,
            "clip": 0.001,
            "lr_scale": 100.0,
            "div_loss": "gaussian_binning",
            "min_local_k": 1.0,
            "max_local_k": 25.0,
            "init_k": 7.0,
        },
    },
    "train": {
        "num_epochs": 15,
        "batch_size": 8,
        "num_workers": 4,
        "folds": None,
        "accelerator": "auto",
        "devices": 1,
        "class_balance": True,
        "lr": [
            {"from": 1.0e-6, "to": 1.0e-4, "steps": 0.1},
            {"from": 1.0e-4, "to": 1.0e-4, "steps": 0.9},
            {"from": 1.0e-4, "to": 1.0e-6, "steps": -1},
        ],
        "weight_decay": [
            {"from": 1.0e-6, "to": 1.0e-5, "steps": 0.1},
            {"from": 1.0e-5, "to": 5.0e-5, "steps": 0.9},
            {"from": 5.0e-5, "to": 1.0e-4, "steps": -1},
        ],
    },
}


def deep_merge(base: dict, override: dict | None) -> dict:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as handle:
        user = yaml.safe_load(handle) or {}
    if not isinstance(user, dict):
        raise ValueError(f"Config must be a mapping, got {type(user).__name__}: {path}")
    cfg = deep_merge(DEFAULTS, user)
    cfg["_config_path"] = str(path.resolve())
    _validate(cfg, path)
    return cfg


def _validate(cfg: dict, path: Path) -> None:
    missing = [
        key
        for key in ("data.task_dir", "data.trident_dir", "experiment.save_dir")
        if _get(cfg, key) in (None, "")
    ]
    if missing:
        raise ValueError(f"{path} is missing required keys: {', '.join(missing)}")
    pool = cfg["model"]["pool_type"]
    if pool not in {"attention", "mean"}:
        raise ValueError(f"model.pool_type must be 'attention' or 'mean', got {pool!r}")
    decay = cfg["model"]["decay"]["type"]
    if decay not in {"Gaussian", "Exponential", "InverseQuadratic", "Cauchy"}:
        raise ValueError(
            "model.decay.type must be Gaussian, Exponential, InverseQuadratic, or Cauchy, "
            f"got {decay!r}"
        )
    div = cfg["model"]["decay"]["div_loss"]
    if div not in {"gaussian_binning", "monte_carlo"}:
        raise ValueError(f"model.decay.div_loss must be gaussian_binning or monte_carlo, got {div!r}")
    if cfg["model"]["num_heads"] < 1 or cfg["model"]["depth"] < 1:
        raise ValueError("model.num_heads and model.depth must be >= 1")


def _get(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node
