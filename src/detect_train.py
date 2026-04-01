"""Train the MONAI 3D detection model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.detection.config import merge_cli_with_config
from src.detection import train_detection_model


def _configure_train_logging(log_file: str | None) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    if not any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in root.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)
    else:
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.INFO)
                handler.setFormatter(formatter)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = log_path.resolve()
        if not any(isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == resolved for handler in root.handlers):
            file_handler = logging.FileHandler(resolved, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)


def main() -> None:
    defaults = {
        "fold": 0,
        "prepared_dir": "data/monai_detection_nifti_prepared",
        "checkpoint": "checkpoints/monai_detection_fold0.pt",
        "epochs": 10,
        "batch_size": 2,
        "lr": 1e-4,
        "patch_size": [96, 96, 96],
        "target_spacing": 1.0,
        "samples_per_image": 4,
        "num_workers": 0,
        "device": None,
        "val_interval": 5,
        "selection_metric": "cpm",
        "amp": True,
        "warmup_epochs": 1,
        "min_lr": 1e-6,
        "resume_from": None,
        "score_thresh": 0.15,
        "nms_thresh": 0.1,
        "detections_per_img": 150,
        "log_file": "logs/monai_detection_train.log",
        "annotations": "data/evaluationScript/annotations/annotations.csv",
        "excluded_annotations": "data/evaluationScript/annotations/annotations_excluded.csv",
    }
    parser = argparse.ArgumentParser(description="Train a MONAI 3D RetinaNet detector on LUNA16.")
    parser.add_argument("--config", default="configs/detection/train.yaml")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--prepared_dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patch_size", type=int, nargs=3)
    parser.add_argument("--target_spacing", type=float)
    parser.add_argument("--samples_per_image", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--val_interval", type=int)
    parser.add_argument("--selection_metric", choices=("cpm", "val_loss"))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--min_lr", type=float)
    parser.add_argument("--resume_from")
    parser.add_argument("--score_thresh", type=float)
    parser.add_argument("--nms_thresh", type=float)
    parser.add_argument("--detections_per_img", type=int)
    parser.add_argument("--log_file")
    parser.add_argument("--annotations")
    parser.add_argument("--excluded_annotations")
    args = parser.parse_args()
    config = merge_cli_with_config(
        config_path=args.config,
        defaults=defaults,
        cli_values={k: v for k, v in vars(args).items() if k != "config"},
    )
    if config["device"] is None:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    _configure_train_logging(config.get("log_file"))

    train_detection_model(
        fold=int(config["fold"]),
        prepared_dir=config["prepared_dir"],
        checkpoint_path=config["checkpoint"],
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        lr=float(config["lr"]),
        patch_size=tuple(int(v) for v in config["patch_size"]),
        target_spacing=float(config["target_spacing"]),
        samples_per_image=int(config["samples_per_image"]),
        num_workers=int(config["num_workers"]),
        device=config["device"],
        val_interval=int(config["val_interval"]),
        selection_metric=str(config["selection_metric"]),
        amp=bool(config["amp"]),
        warmup_epochs=int(config["warmup_epochs"]),
        min_lr=float(config["min_lr"]),
        resume_from=config["resume_from"],
        score_thresh=float(config["score_thresh"]),
        nms_thresh=float(config["nms_thresh"]),
        detections_per_img=int(config["detections_per_img"]),
        annotations_path=config["annotations"],
        excluded_annotations_path=config["excluded_annotations"],
    )


if __name__ == "__main__":
    main()
