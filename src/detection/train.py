"""Training helpers for MONAI 3D RetinaNet lung nodule detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import LUNA16DetectionDataset, detection_collate
from .io import load_prepared_split
from .model import build_detection_detector, save_detection_checkpoint

log = logging.getLogger(__name__)


def _default_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _loss_total(detector: Any, loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return loss_dict[detector.cls_key] + loss_dict[detector.box_reg_key]


def train_detection_model(
    fold: int = 0,
    prepared_dir: str | Path = "data/monai_detection",
    checkpoint_path: str | Path = "checkpoints/monai_detection_fold0.pt",
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 1e-4,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    target_spacing: float = 1.0,
    samples_per_image: int = 4,
    num_workers: int = 0,
    device: str | None = None,
) -> Path:
    device = _default_device(device)
    split = load_prepared_split(fold, prepared_dir)
    train_ds = LUNA16DetectionDataset(
        split["training"],
        patch_size=patch_size,
        target_spacing=target_spacing,
        samples_per_image=samples_per_image,
        training=True,
        seed=fold + 17,
    )
    val_ds = LUNA16DetectionDataset(
        split["validation"],
        patch_size=patch_size,
        target_spacing=target_spacing,
        samples_per_image=1,
        positive_fraction=1.0,
        training=False,
        seed=fold + 31,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate,
    )

    detector = build_detection_detector(patch_size=patch_size)
    detector.to(device)
    optimizer = torch.optim.AdamW(detector.network.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    config = {
        "fold": int(fold),
        "patch_size": list(patch_size),
        "target_spacing": float(target_spacing),
        "score_thresh": 0.15,
        "nms_thresh": 0.1,
        "detections_per_img": 150,
    }

    for epoch in range(1, epochs + 1):
        detector.train()
        train_loss = 0.0
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            losses = detector(images, targets)
            loss = _loss_total(detector, losses)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        val_loss = 0.0
        with torch.no_grad():
            detector.train()
            for images, targets, _ in val_loader:
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
                losses = detector(images, targets)
                loss = _loss_total(detector, losses)
                val_loss += float(loss.item())
        detector.eval()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        log.info(
            "Epoch %d/%d train_loss=%.4f val_loss=%.4f",
            epoch,
            epochs,
            train_loss,
            val_loss,
        )

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_detection_checkpoint(
                checkpoint_path,
                detector=detector,
                config=config,
                epoch=epoch,
                best_metric=best_val_loss,
            )
            log.info("Saved best detection checkpoint to %s", checkpoint_path)

    return Path(checkpoint_path)
