"""Training loop compatible with both UNet3D and HybridNet."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
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
        device: str,
        checkpoint_dir: str,
        experiment_name: str,
        wandb_run=None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.wandb_run = wandb_run

        self._best_metric = -float("inf")
        self._saved_checkpoints: list[tuple[float, Path]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        for epoch in range(1, self.cfg.max_epochs + 1):
            train_metrics = self._train_epoch(train_loader, epoch)
            log.info(f"Epoch {epoch}/{self.cfg.max_epochs} — {train_metrics}")

            if epoch % self.cfg.val_every_n_epochs == 0:
                val_metrics = self._val_epoch(val_loader, epoch)
                log.info(f"  Validation — {val_metrics}")
                monitor = val_metrics.get(self.cfg.monitor_metric, 0.0)
                self._maybe_save_checkpoint(epoch, monitor)

            if self.scheduler is not None:
                self.scheduler.step()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader, 1):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(images)
            logits = out["logits"]
            ds_logits = out.get("ds_logits")

            loss = self.loss_fn(logits, masks, ds_logits) if ds_logits is not None else self.loss_fn(logits, masks)
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
    def _val_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        from src.evaluation.froc import compute_froc
        self.model.eval()
        total_loss = 0.0
        pred_list: list[dict] = []

        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            out = self.model(images)
            logits = out["logits"]
            loss = self.loss_fn(logits, masks)
            total_loss += loss.item()

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
            ann_df = pd.read_csv(
                os.path.join(os.environ.get("DATA_DIR", "data/processed"), "annotations.csv")
            )
            froc = compute_froc(pred_list, ann_df)
            cpm = froc["cpm"]
        except Exception:
            log.warning("FROC computation failed; setting val_cpm=0.0", exc_info=True)
            cpm = 0.0

        metrics = {"val_loss": total_loss / len(loader), "val_cpm": cpm}
        if self.wandb_run:
            self.wandb_run.log({"val/loss": metrics["val_loss"], "val/cpm": cpm, "epoch": epoch})
        return metrics

    def _maybe_save_checkpoint(self, epoch: int, metric: float) -> None:
        path = self.checkpoint_dir / f"{self.experiment_name}_epoch{epoch:04d}_cpm{metric:.4f}.ckpt"
        torch.save(
            {
                "epoch": epoch,
                "model": self.experiment_name,  # identifies architecture for loading
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metric": metric,
            },
            path,
        )
        self._saved_checkpoints.append((metric, path))
        self._saved_checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Prune to top-k
        top_k = getattr(self.cfg, "save_top_k", 3)
        while len(self._saved_checkpoints) > top_k:
            _, old_path = self._saved_checkpoints.pop()
            old_path.unlink(missing_ok=True)

        if metric > self._best_metric:
            self._best_metric = metric
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.ckpt"
            import shutil
            shutil.copy(path, best_path)
            log.info(f"  New best checkpoint: {metric:.4f} → {best_path}")
