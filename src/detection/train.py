"""Training helpers for MONAI 3D RetinaNet lung nodule detection."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from monai.data.utils import no_collation

from .data import (
    build_monai_detection_train_dataset,
    build_monai_detection_val_dataset,
    monai_detection_collate,
)
from .evaluate import evaluate_detection_model
from .io import load_prepared_split
from .model import build_detection_detector, save_detection_checkpoint

log = logging.getLogger(__name__)


def _default_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _loss_total(detector: Any, loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return loss_dict[detector.cls_key] + loss_dict[detector.box_reg_key]


def _autocast_device_type(device: str) -> str:
    return "cuda" if str(device).startswith("cuda") else "cpu"


def _last_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_last{path.suffix}")


def _freeze_batchnorm_stats(module: torch.nn.Module) -> None:
    if isinstance(module, _BatchNorm):
        module.eval()


def _initial_best_value(selection_metric: str) -> float:
    return float("-inf") if selection_metric == "cpm" else float("inf")


def train_detection_model(
    fold: int = 0,
    prepared_dir: str | Path = "data/monai_detection_nifti_prepared",
    checkpoint_path: str | Path = "checkpoints/monai_detection_fold0.pt",
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 1e-4,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    target_spacing: float = 1.0,
    samples_per_image: int = 4,
    num_workers: int = 0,
    device: str | None = None,
    val_interval: int = 5,
    selection_metric: str = "cpm",
    amp: bool = True,
    warmup_epochs: int = 1,
    min_lr: float = 1e-6,
    resume_from: str | Path | None = None,
    score_thresh: float = 0.15,
    nms_thresh: float = 0.1,
    detections_per_img: int = 150,
    annotations_path: str | Path = "data/evaluationScript/annotations/annotations.csv",
    excluded_annotations_path: str | Path = "data/evaluationScript/annotations/annotations_excluded.csv",
) -> Path:
    device = _default_device(device)
    device_type = _autocast_device_type(device)
    use_amp = bool(amp and device_type == "cuda")
    split = load_prepared_split(fold, prepared_dir)
    if not split.get("training"):
        raise ValueError(
            f"Prepared split fold {fold} has no training items under {prepared_dir}. "
            "Standardize more than one LUNA16 subset or choose a different fold."
        )
    if not split.get("validation"):
        raise ValueError(
            f"Prepared split fold {fold} has no validation items under {prepared_dir}. "
            "Standardize the matching subset for this fold or choose a different fold."
        )
    train_ds = build_monai_detection_train_dataset(
        split["training"],
        patch_size=patch_size,
        target_spacing=target_spacing,
        samples_per_image=samples_per_image,
    )
    val_ds = build_monai_detection_val_dataset(
        split["validation"],
        target_spacing=target_spacing,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=no_collation,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=no_collation,
    )

    detector = build_detection_detector(
        patch_size=patch_size,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
    )
    if resume_from is not None:
        payload = torch.load(resume_from, map_location=device, weights_only=False)
        detector.network.load_state_dict(payload["model_state_dict"])
        log.info(
            "Loaded detection weights from %s for resume",
            resume_from,
        )
    detector.to(device)
    optimizer = torch.optim.AdamW(detector.network.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = None
    total_decay_epochs = max(epochs - max(warmup_epochs, 0), 1)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=max(min_lr / max(lr, 1e-12), 1e-3),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_decay_epochs,
            eta_min=min_lr,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=min_lr,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    config = {
        "fold": int(fold),
        "patch_size": list(patch_size),
        "target_spacing": float(target_spacing),
        "score_thresh": float(score_thresh),
        "nms_thresh": float(nms_thresh),
        "detections_per_img": int(detections_per_img),
        "val_interval": int(max(val_interval, 1)),
        "selection_metric": selection_metric,
        "amp": bool(use_amp),
        "warmup_epochs": int(max(warmup_epochs, 0)),
        "min_lr": float(min_lr),
        "annotations_path": str(annotations_path),
        "excluded_annotations_path": str(excluded_annotations_path),
    }
    best_selection_value = _initial_best_value(selection_metric)
    metric_output_root = Path("outputs") / f"monai_detection_fold{fold}_validation"
    start_epoch = 1
    annotations_path = Path(annotations_path)
    excluded_annotations_path = Path(excluded_annotations_path)
    resumed_training_state: dict[str, Any] = {}

    if resume_from is not None:
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if "scaler_state_dict" in payload:
            scaler.load_state_dict(payload["scaler_state_dict"])
        training_state = payload.get("training_state", {})
        resumed_training_state = dict(training_state)
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_val_loss = float(training_state.get("best_val_loss", training_state.get("best_patch_val_loss", best_val_loss)))
        best_selection_value = float(training_state.get("best_selection_value", best_selection_value))
        log.info(
            "Resumed detection training from %s at epoch %d",
            resume_from,
            start_epoch,
        )

    for epoch in range(start_epoch, epochs + 1):
        detector.train()
        train_loss = 0.0
        for batch in train_loader:
            images, targets, _ = monai_detection_collate(batch)
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                losses = detector(images, targets)
                loss = _loss_total(detector, losses)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())

        val_loss = 0.0
        with torch.no_grad():
            detector.train()
            detector.network.apply(_freeze_batchnorm_stats)
            for batch in val_loader:
                images, targets, _ = monai_detection_collate(batch)
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    losses = detector(images, targets)
                    loss = _loss_total(detector, losses)
                val_loss += float(loss.item())
        detector.eval()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        log.info(
            "Epoch %d/%d train_loss=%.4f val_loss=%.4f lr=%.6g",
            epoch,
            epochs,
            train_loss,
            val_loss,
            optimizer.param_groups[0]["lr"],
        )

        should_run_detection_eval = epoch % max(val_interval, 1) == 0 or epoch == epochs
        cpm = None
        if should_run_detection_eval:
            if annotations_path.exists() and excluded_annotations_path.exists():
                metric_output_dir = metric_output_root / f"epoch_{epoch:03d}"
                eval_results = evaluate_detection_model(
                    detector=detector,
                    fold=fold,
                    prepared_dir=prepared_dir,
                    output_path=metric_output_root / f"epoch_{epoch:03d}.json",
                    inference_output_dir=metric_output_dir,
                    device=device,
                    score_thresh=float(config["score_thresh"]),
                    target_spacing=target_spacing,
                    annotations_path=annotations_path,
                    excluded_annotations_path=excluded_annotations_path,
                )
                cpm = float(eval_results["cpm"])
                log.info("Epoch %d/%d validation CPM=%.4f", epoch, epochs, cpm)
            else:
                log.warning(
                    "Skipping CPM evaluation because LUNA16 annotation files were not found at %s and %s.",
                    annotations_path,
                    excluded_annotations_path,
                )

        best_val_loss = min(best_val_loss, val_loss)
        effective_selection_metric = "cpm" if selection_metric == "cpm" and cpm is not None else "val_loss"
        if epoch == start_epoch and "best_selection_value" not in resumed_training_state:
            best_selection_value = _initial_best_value(effective_selection_metric)
        elif not math.isfinite(best_selection_value) and effective_selection_metric == "val_loss":
            best_selection_value = float("inf")
        candidate_metric = cpm if effective_selection_metric == "cpm" else val_loss
        is_better = False
        if candidate_metric is not None:
            is_better = (
                candidate_metric >= best_selection_value
                if effective_selection_metric == "cpm"
                else candidate_metric <= best_selection_value
            )
        if is_better:
            best_selection_value = candidate_metric
            save_detection_checkpoint(
                checkpoint_path,
                detector=detector,
                config=config,
                epoch=epoch,
                best_metric=float(candidate_metric),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                training_state={
                    "best_val_loss": float(best_val_loss),
                    "best_selection_value": float(best_selection_value),
                    "selection_metric": selection_metric,
                    "effective_selection_metric": effective_selection_metric,
                },
            )
            if effective_selection_metric == "cpm" and cpm is not None:
                log.info("Saved best detection checkpoint to %s (CPM %.4f)", checkpoint_path, cpm)
            else:
                log.info("Saved best detection checkpoint to %s (val_loss %.4f)", checkpoint_path, val_loss)

        last_checkpoint_metric = best_selection_value
        if not math.isfinite(last_checkpoint_metric):
            last_checkpoint_metric = val_loss
        save_detection_checkpoint(
            _last_checkpoint_path(checkpoint_path),
            detector=detector,
            config=config,
            epoch=epoch,
            best_metric=float(last_checkpoint_metric),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            training_state={
                "best_val_loss": float(best_val_loss),
                "best_selection_value": float(best_selection_value),
                "selection_metric": selection_metric,
                "effective_selection_metric": effective_selection_metric,
            },
        )

        if scheduler is not None:
            scheduler.step()

    return Path(checkpoint_path)
