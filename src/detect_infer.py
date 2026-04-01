"""Run MONAI 3D detection inference."""

from __future__ import annotations

import argparse
import logging

import torch

from src.detection.config import merge_cli_with_config
from src.detection import infer_detection_directory, load_detection_checkpoint


def main() -> None:
    defaults = {
        "checkpoint": "checkpoints/monai_detection_fold0.pt",
        "input_dir": "data/monai_detection_nifti/images",
        "output_dir": "outputs/monai_detection",
        "score_thresh": 0.15,
        "device": None,
    }
    parser = argparse.ArgumentParser(description="Run MONAI 3D detection inference on NIfTI scans.")
    parser.add_argument("--config", default="configs/detection/infer.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--score_thresh", type=float)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = merge_cli_with_config(
        config_path=args.config,
        defaults=defaults,
        cli_values={k: v for k, v in vars(args).items() if k != "config"},
    )
    if config["device"] is None:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    detector, payload = load_detection_checkpoint(config["checkpoint"], config["device"])
    checkpoint_config = payload.get("config", {})
    infer_detection_directory(
        detector=detector,
        input_dir=config["input_dir"],
        output_dir=config["output_dir"],
        device=config["device"],
        score_thresh=float(config["score_thresh"]),
        target_spacing=float(checkpoint_config.get("target_spacing", 1.0)),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
