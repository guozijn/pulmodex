"""Evaluate MONAI 3D detection checkpoints on LUNA16."""

from __future__ import annotations

import argparse
import json
import logging

import torch

from src.detection.config import merge_cli_with_config
from src.detection import evaluate_detection_model, load_detection_checkpoint


def main() -> None:
    defaults = {
        "checkpoint": "checkpoints/monai_detection_fold0.pt",
        "fold": 0,
        "prepared_dir": "data/monai_detection_nifti_prepared",
        "output": "outputs/detection_eval_fold0.json",
        "inference_output_dir": "outputs/detection_eval_cases",
        "annotations": "data/evaluationScript/annotations/annotations.csv",
        "excluded_annotations": "data/evaluationScript/annotations/annotations_excluded.csv",
        "score_thresh": 0.15,
        "device": None,
    }
    parser = argparse.ArgumentParser(description="Evaluate a MONAI 3D detection checkpoint on LUNA16.")
    parser.add_argument("--config", default="configs/detection/evaluate.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--prepared_dir")
    parser.add_argument("--output")
    parser.add_argument("--inference_output_dir")
    parser.add_argument("--annotations")
    parser.add_argument("--excluded_annotations")
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
    results = evaluate_detection_model(
        detector=detector,
        fold=int(config["fold"]),
        prepared_dir=config["prepared_dir"],
        output_path=config["output"],
        annotations_path=config["annotations"],
        excluded_annotations_path=config["excluded_annotations"],
        inference_output_dir=config["inference_output_dir"],
        device=config["device"],
        score_thresh=float(config["score_thresh"]),
        target_spacing=float(checkpoint_config.get("target_spacing", 1.0)),
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
