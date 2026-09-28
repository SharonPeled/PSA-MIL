"""Lightning wrapper around PSAMIL."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from psa_mil.models.psa import PSAMIL
from psa_mil.training.schedule import piecewise_linear


class PSAModule(LightningModule):
    def __init__(
        self,
        model: PSAMIL,
        train_cfg: dict,
        class_weight: dict[int, float],
        decay_lr_scale: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.train_cfg = train_cfg
        self.class_weight = class_weight
        self.decay_lr_scale = float(decay_lr_scale)
        self._predictions: list[dict] = []

    def forward(self, features, coords, mask=None):
        return self.model(features, coords, mask)

    def training_step(self, batch, batch_idx):
        logits = self.model(batch["features"], batch["coords"], batch["mask"])
        task_loss = self._classification_loss(logits, batch["label"])
        diversity = self.model.diversity_loss()
        loss = task_loss - self.model.alpha * diversity
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=logits.shape[0])
        self.log("task_loss", task_loss, on_step=False, on_epoch=True, batch_size=logits.shape[0])
        if self.model.alpha > 0:
            self.log("div_loss", diversity, on_step=False, on_epoch=True, batch_size=logits.shape[0])
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        if not hasattr(self, "lr_schedule"):
            return
        step = min(self.global_step, len(self.lr_schedule) - 1)
        optimizer = self.optimizers()
        base_lr = float(self.lr_schedule[step])
        weight_decay = float(self.wd_schedule[step])
        optimizer.param_groups[0]["lr"] = base_lr
        optimizer.param_groups[0]["weight_decay"] = weight_decay
        optimizer.param_groups[1]["lr"] = base_lr * self.decay_lr_scale
        optimizer.param_groups[1]["weight_decay"] = weight_decay
        self.log("lr", base_lr, on_step=True, on_epoch=False, prog_bar=False)

    def on_train_epoch_end(self):
        rates = self.model.decay_rates().detach()
        local_k = self.model.local_ks().detach()
        for layer in range(rates.shape[0]):
            for head in range(rates.shape[1]):
                self.log(f"rate/l{layer}h{head}", rates[layer, head], on_epoch=True)
                self.log(f"local_k/l{layer}h{head}", local_k[layer, head], on_epoch=True)

    def on_test_epoch_start(self):
        self._predictions = []

    def test_step(self, batch, batch_idx):
        logits = self.model(batch["features"], batch["coords"], batch["mask"])
        self._predictions.append(
            {
                "probs": torch.softmax(logits, dim=-1).detach().cpu(),
                "label": batch["label"].detach().cpu(),
                "slide_id": list(batch["slide_id"]),
                "case_id": list(batch["case_id"]),
            }
        )

    def on_train_start(self):
        steps_per_epoch = int(self.trainer.num_training_batches)
        total = steps_per_epoch * int(self.trainer.max_epochs)
        self.lr_schedule = piecewise_linear(self.train_cfg["lr"], total)
        self.wd_schedule = piecewise_linear(self.train_cfg["weight_decay"], total)

    def configure_optimizers(self):
        decay, other = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("lambda_p"):
                decay.append(param)
            else:
                other.append(param)
        base = float(self.train_cfg["lr"][0]["from"])
        return torch.optim.Adam(
            [
                {"params": other, "lr": base, "weight_decay": float(self.train_cfg["weight_decay"][0]["from"])},
                {
                    "params": decay,
                    "lr": base * self.decay_lr_scale,
                    "weight_decay": float(self.train_cfg["weight_decay"][0]["from"]),
                },
            ]
        )

    def _classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        per_sample = F.cross_entropy(logits, labels, reduction="none")
        if not self.class_weight:
            return per_sample.mean()
        weights = torch.tensor(
            [self.class_weight[int(label)] for label in labels],
            device=logits.device,
            dtype=logits.dtype,
        )
        weights = weights / weights.sum().clamp_min(1e-12)
        return torch.sum(per_sample * weights)
