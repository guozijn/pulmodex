"""Training loop compatible with both UNet3D and HybridNet."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


class Trainer:
    """Generic trainer for segmentation models.

    Args:
        model: nn.Module with forward() → dict containing "logits" and optionally "ds_logits"
        loss_fn: callable(logits, mask, ds_logits=None) → scalar tensor
        optimizer: torch optimizer
        scheduler: optional LR scheduler (step called per epoch)
        cfg: OmegaConf trainer config node
        device: torch device string
        checkpoint_dir: where to save checkpoints
        experiment_name: prefix for checkpoint filenames
        wandb_run: optional W&B run object for logging
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        cfg,
        run_config: dict[str, Any],
        device: str,
        checkpoint_dir: str,
        experiment_name: str,
        task_type: str = "segmentation",
        data_dir: str | None = None,
        wandb_run=None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.run_config = run_config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.task_type = task_type
        self.data_dir = data_dir
        self.wandb_run = wandb_run

        self.monitor_mode = getattr(self.cfg, "monitor_mode", "max")
        if self.monitor_mode not in {"max", "min"}:
            raise ValueError(f"Unsupported monitor_mode: {self.monitor_mode}")

        self._best_metric = (
            -float("inf") if self.monitor_mode == "max" else float("inf")
        )
        self._saved_checkpoints: list[tuple[float, Path]] = []
        self.dice_metric = (
            DiceMetric(include_background=True, reduction="mean")
            if self.task_type == "segmentation"
            else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        for epoch in range(1, self.cfg.max_epochs + 1):
            if self.task_type == "classification":
                train_metrics = self._train_classification_epoch(train_loader, epoch)
            else:
                train_metrics = self._train_segmentation_epoch(train_loader, epoch)
            log.info(f"Epoch {epoch}/{self.cfg.max_epochs} — {train_metrics}")

            if epoch % self.cfg.val_every_n_epochs == 0:
                if self.task_type == "classification":
                    val_metrics = self._val_classification_epoch(val_loader, epoch)
                else:
                    val_metrics = self._val_segmentation_epoch(val_loader, epoch)
                log.info(f"  Validation — {val_metrics}")
                monitor = val_metrics.get(self.cfg.monitor_metric, 0.0)
                self._maybe_save_checkpoint(epoch, monitor)

            if self.scheduler is not None:
                self.scheduler.step()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_segmentation_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader, 1):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(images)
            logits = out["logits"]
            ds_logits = out.get("ds_logits")

            if ds_logits is not None:
                loss = self.loss_fn(logits, masks, ds_logits)
            else:
                loss = self.loss_fn(logits, masks)
            loss.backward()

            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.optimizer.step()
            total_loss += loss.item()

            if step % self.cfg.log_every_n_steps == 0:
                avg = total_loss / step
                log.info(f"  Step {step}/{len(loader)} loss={avg:.4f}")
                if self.wandb_run:
                    self.wandb_run.log({"train/loss": avg, "epoch": epoch, "step": step})

        return {"train_loss": total_loss / len(loader)}

    @torch.no_grad()
    def _val_segmentation_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        from src.evaluation.froc import compute_froc

        self.model.eval()
        total_loss = 0.0
        pred_list: list[dict] = []
        assert self.dice_metric is not None
        self.dice_metric.reset()

        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            out = self.model(images)
            logits = out["logits"]
            loss = self.loss_fn(logits, masks)
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            self.dice_metric(y_pred=preds, y=masks)

            probs = out["seg"].cpu().numpy()
            for i, uid in enumerate(batch["seriesuid"]):
                pred_list.append({
                    "seriesuid": uid,
                    "prob": float(probs[i].max()),
                    "coord_xyz": batch["coord_xyz"][i].numpy(),
                })

        # FROC requires annotation_df — load lazily
        try:
            import pandas as pd

            if self.data_dir is None:
                raise ValueError("data_dir must be set for segmentation validation")

            ann_df = pd.read_csv(Path(self.data_dir) / "annotations.csv")
            froc = compute_froc(pred_list, ann_df)
            cpm = froc["cpm"]
        except Exception:
            log.warning("FROC computation failed; setting val_cpm=0.0", exc_info=True)
            cpm = 0.0

        mean_dice = float(self.dice_metric.aggregate().item())
        metrics = {"val_loss": total_loss / len(loader), "val_cpm": cpm, "val_dice": mean_dice}
        if self.wandb_run:
            self.wandb_run.log(
                {
                    "val/loss": metrics["val_loss"],
                    "val/cpm": cpm,
                    "val/dice": mean_dice,
                    "epoch": epoch,
                }
            )
        return metrics

    def _train_classification_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for step, batch in enumerate(loader, 1):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(images)
            logits = out["logits"]
            loss = self.loss_fn(logits, labels)
            loss.backward()

            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_examples += int(labels.numel())

            if step % self.cfg.log_every_n_steps == 0:
                avg_loss = total_loss / step
                acc = total_correct / max(total_examples, 1)
                log.info(f"  Step {step}/{len(loader)} loss={avg_loss:.4f} acc={acc:.4f}")
                if self.wandb_run:
                    self.wandb_run.log(
                        {
                            "train/loss": avg_loss,
                            "train/acc": acc,
                            "epoch": epoch,
                            "step": step,
                        }
                    )

        return {
            "train_loss": total_loss / len(loader),
            "train_acc": total_correct / max(total_examples, 1),
        }

    @torch.no_grad()
    def _val_classification_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            out = self.model(images)
            logits = out["logits"]
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_examples += int(labels.numel())

        metrics = {
            "val_loss": total_loss / len(loader),
            "val_acc": total_correct / max(total_examples, 1),
        }
        if self.wandb_run:
            self.wandb_run.log(
                {
                    "val/loss": metrics["val_loss"],
                    "val/acc": metrics["val_acc"],
                    "epoch": epoch,
                }
            )
        return metrics

    def _maybe_save_checkpoint(self, epoch: int, metric: float) -> None:
        path = (
            self.checkpoint_dir
            / f"{self.experiment_name}_epoch{epoch:04d}_{self.cfg.monitor_metric}{metric:.4f}.ckpt"
        )
        torch.save(
            {
                "epoch": epoch,
                "experiment_name": self.experiment_name,
                "model_name": self.run_config["model"]["name"],
                "task_type": self.task_type,
                "config": self.run_config,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metric": metric,
            },
            path,
        )
        self._saved_checkpoints.append((metric, path))
        self._saved_checkpoints.sort(
            key=lambda x: x[0],
            reverse=self.monitor_mode == "max",
        )

        # Prune to top-k
        top_k = getattr(self.cfg, "save_top_k", 3)
        while len(self._saved_checkpoints) > top_k:
            _, old_path = self._saved_checkpoints.pop()
            old_path.unlink(missing_ok=True)

        is_better = (
            metric > self._best_metric
            if self.monitor_mode == "max"
            else metric < self._best_metric
        )
        if is_better:
            self._best_metric = metric
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.ckpt"
            shutil.copy(path, best_path)
            log.info(f"  New best checkpoint: {metric:.4f} → {best_path}")
