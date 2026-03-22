"""Inference entry point.

Usage:
    python src/inference.py --checkpoint checkpoints/best.ckpt \
        --fp_checkpoint checkpoints/best_fp.ckpt \
        --input_dir data/processed \
        --output_dir outputs
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def load_seg_model(path: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model_name = ckpt.get("model", "unet3d")
    if model_name == "hybrid_net":
        from src.models.hybrid import HybridNet
        m = HybridNet()
    else:
        from src.models.baseline import UNet3D
        m = UNet3D()
    m.load_state_dict(ckpt["model_state_dict"])
    return m


def load_fp_model(path: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    from src.fp_reduction import FPClassifier
    m = FPClassifier()
    m.load_state_dict(ckpt["model_state_dict"])
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Segmentation model checkpoint")
    parser.add_argument("--fp_checkpoint", required=True, help="FP classifier checkpoint")
    parser.add_argument("--input_dir", default="data/processed")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--fp_threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seg_model = load_seg_model(args.checkpoint, args.device)
    fp_model = load_fp_model(args.fp_checkpoint, args.device)

    from src.inference.pipeline import InferencePipeline
    pipeline = InferencePipeline(
        seg_model=seg_model,
        fp_model=fp_model,
        fp_threshold=args.fp_threshold,
        device=args.device,
    )

    # Find all .mhd files in input_dir
    mhd_files = list(Path(args.input_dir).rglob("*.mhd"))
    log.info(f"Found {len(mhd_files)} scans in {args.input_dir}")

    for mhd_path in mhd_files:
        seriesuid = mhd_path.stem
        report = pipeline.run(str(mhd_path), args.output_dir, seriesuid)
        log.info(f"  {seriesuid}: {report['n_candidates_final']} nodule(s)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
