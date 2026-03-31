"""Run MONAI 3D detection inference."""

from __future__ import annotations

import argparse
import logging

import torch

from src.detection import infer_detection_directory, load_detection_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MONAI 3D detection inference on MHD scans.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_dir", default="data/orig_datasets")
    parser.add_argument("--output_dir", default="outputs/monai_detection")
    parser.add_argument("--score_thresh", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    detector, _ = load_detection_checkpoint(args.checkpoint, args.device)
    infer_detection_directory(
        detector=detector,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
        score_thresh=args.score_thresh,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
