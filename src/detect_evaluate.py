"""Evaluate MONAI 3D detection checkpoints on LUNA16."""

from __future__ import annotations

import argparse
import json
import logging

import torch

from src.detection import evaluate_detection_model, load_detection_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MONAI 3D detection checkpoint on LUNA16.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--prepared_dir", default="data/monai_detection")
    parser.add_argument("--output", default="outputs/detection_eval_fold0.json")
    parser.add_argument("--inference_output_dir", default="outputs/detection_eval_cases")
    parser.add_argument("--annotations", default="data/evaluationScript/annotations/annotations.csv")
    parser.add_argument(
        "--excluded_annotations",
        default="data/evaluationScript/annotations/annotations_excluded.csv",
    )
    parser.add_argument("--score_thresh", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    detector, _ = load_detection_checkpoint(args.checkpoint, args.device)
    results = evaluate_detection_model(
        detector=detector,
        fold=args.fold,
        prepared_dir=args.prepared_dir,
        output_path=args.output,
        annotations_path=args.annotations,
        excluded_annotations_path=args.excluded_annotations,
        inference_output_dir=args.inference_output_dir,
        device=args.device,
        score_thresh=args.score_thresh,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
