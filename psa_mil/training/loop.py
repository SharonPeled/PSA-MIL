"""Cross-validation training from a YAML config."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader

from psa_mil.config import load_config
from psa_mil.data.dataset import SlideDataset, attach_features, collate_slides
from psa_mil.data.task import load_task
from psa_mil.data.trident import TridentReader
from psa_mil.models.psa import PSAMIL
from psa_mil.training.metrics import case_metrics, predictions_frame
from psa_mil.training.module import PSAModule

log = logging.getLogger("psa_mil")


def train_from_config(config_path: str) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(config_path)
    pl.seed_everything(int(cfg["experiment"]["seed"]), workers=True)

    task = load_task(cfg["data"]["task_dir"])
    reader = TridentReader(cfg["data"]["trident_dir"], cfg["data"]["encoder"])
    slides, missing = attach_features(task["frame"], reader)
    if missing:
        log.warning("Skipping %d slides with no features. First few: %s", len(missing), missing[:5])
    if slides.empty:
        raise RuntimeError("No task slides matched TRIDENT feature files.")
    log.info(
        "Task %s: %d slides, %d cases, classes %s",
        task["task_col"],
        len(slides),
        slides["case_id"].nunique(),
        task["class_names"],
    )

    embed_dim = cfg["model"]["embed_dim"]
    if embed_dim is None:
        embed_dim = reader.embed_dim(slides.iloc[0].slide_id)
        log.info("Inferred embed_dim=%d from %s", embed_dim, cfg["data"]["encoder"])

    folds = cfg["train"]["folds"] if cfg["train"]["folds"] is not None else task["fold_ids"]
    if isinstance(folds, int):
        folds = [folds]
    cfg["train"]["folds"] = [int(fold) for fold in folds]
    save_dir = Path(cfg["experiment"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    rows = []
    for fold in folds:
        rows.append(_run_fold(cfg, task, reader, slides, int(fold), int(embed_dim), save_dir))

    summary = pd.DataFrame(rows)
    summary_path = save_dir / "metrics.csv"
    summary.to_csv(summary_path, index=False)
    _print_summary(summary)
    log.info("Wrote %s", summary_path)
    return summary


def _run_fold(cfg, task, reader, slides, fold, embed_dim, save_dir: Path) -> dict:
    column = f"fold_{fold}"
    if column not in slides.columns:
        raise ValueError(f"Fold {fold} is not in the task splits")
    train_df = slides[slides[column] == "train"].reset_index(drop=True)
    test_df = slides[slides[column] == "test"].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise RuntimeError(f"Fold {fold} has {len(train_df)} train and {len(test_df)} test slides")
    log.info("Fold %d: %d train slides, %d test slides", fold, len(train_df), len(test_df))

    max_tiles = cfg["data"]["max_tiles"]
    train_loader = _loader(train_df, reader, max_tiles, cfg, shuffle=True, fixed_subsample=False)
    test_loader = _loader(test_df, reader, max_tiles, cfg, shuffle=False, fixed_subsample=True)
    class_weight = _class_weights(train_df) if cfg["train"]["class_balance"] else {}

    model = _build_model(cfg, embed_dim, task["num_classes"])
    module = PSAModule(
        model,
        cfg["train"],
        class_weight,
        decay_lr_scale=float(cfg["model"]["decay"]["lr_scale"]),
    )
    fold_dir = save_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    trainer = pl.Trainer(
        accelerator=cfg["train"]["accelerator"],
        devices=cfg["train"]["devices"],
        max_epochs=int(cfg["train"]["num_epochs"]),
        logger=_mlflow_logger(cfg),
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        enable_model_summary=fold == (cfg["train"]["folds"] or task["fold_ids"])[0],
        default_root_dir=str(fold_dir),
        # Do not infer a multi-node job from the surrounding SLURM allocation.
        plugins=[LightningEnvironment()],
    )
    trainer.fit(module, train_loader)
    trainer.test(module, test_loader)

    predictions = predictions_frame(module._predictions, task["class_names"])
    predictions.to_csv(fold_dir / "predictions.csv", index=False)
    torch.save(model.export_state(), fold_dir / "psa_mil.pt")

    metrics = case_metrics(predictions, task["num_classes"])
    metrics["fold"] = fold
    log.info(
        "Fold %d  auc=%.4f  balanced_accuracy=%.4f  f1=%.4f  accuracy=%.4f",
        fold,
        metrics["auc"],
        metrics["balanced_accuracy"],
        metrics["f1"],
        metrics["accuracy"],
    )
    return metrics


def _build_model(cfg: dict, embed_dim: int, num_classes: int) -> PSAMIL:
    model_cfg = cfg["model"]
    decay = model_cfg["decay"]
    return PSAMIL(
        embed_dim=int(embed_dim),
        num_classes=int(num_classes),
        num_heads=int(model_cfg["num_heads"]),
        depth=int(model_cfg["depth"]),
        attn_dim=int(model_cfg["attn_dim"]),
        num_residual_layers=int(model_cfg["num_residual_layers"]),
        pool_type=model_cfg["pool_type"],
        qkv_bias=bool(model_cfg["qkv_bias"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
        dropout=float(model_cfg["dropout"]),
        attention_chunk=int(model_cfg["attention_chunk"]),
        decay_type=decay["type"],
        decay_clip=float(decay["clip"]),
        min_local_k=float(decay["min_local_k"]),
        max_local_k=float(decay["max_local_k"]),
        init_k=float(decay["init_k"]),
        div_loss=decay["div_loss"],
        alpha=float(decay["alpha"]),
    )


def _loader(frame, reader, max_tiles, cfg, shuffle: bool, fixed_subsample: bool) -> DataLoader:
    dataset = SlideDataset(frame, reader, max_tiles=max_tiles, fixed_subsample=fixed_subsample)
    workers = int(cfg["train"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_slides,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=_worker_init if workers > 0 else None,
    )


def _worker_init(_worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")


def _class_weights(frame: pd.DataFrame) -> dict[int, float]:
    counts = frame["label"].value_counts().to_dict()
    return {int(label): 1.0 / float(count) for label, count in counts.items()}


def _mlflow_logger(cfg: dict):
    mlflow_dir = cfg["experiment"]["mlflow_dir"]
    if not mlflow_dir:
        return False
    from pytorch_lightning.loggers import MLFlowLogger

    return MLFlowLogger(
        experiment_name=cfg["experiment"]["name"],
        run_name=cfg["experiment"]["run_name"],
        save_dir=mlflow_dir,
        tags={"mlflow.note.content": cfg["experiment"].get("description") or ""},
    )


def _print_summary(summary: pd.DataFrame) -> None:
    print("\nFold results")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(summary) > 1:
        for column in ("auc", "balanced_accuracy", "f1", "accuracy"):
            print(f"{column}: {summary[column].mean():.4f} ± {summary[column].std():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train PSA-MIL from a YAML config")
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
