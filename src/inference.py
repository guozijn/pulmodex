"""Inference entry point.

Usage:
    pulmodex infer --checkpoint checkpoints/hybrid_best.ckpt \
        --fp_checkpoint checkpoints/fp_reduction_best.ckpt \
        --input_dir data/processed \
        --output_dir outputs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.loading import load_checkpoint_model  # noqa: E402

log = logging.getLogger(__name__)


def load_primary_model(path: str, device: str) -> torch.nn.Module:
    model, _ = load_checkpoint_model(path, device)
    return model


def load_fp_model(path: str, device: str) -> torch.nn.Module:
    model, _ = load_checkpoint_model(path, device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Primary model source: project checkpoint or MONAI bundle directory",
    )
    parser.add_argument("--fp_checkpoint", required=True, help="FP classifier checkpoint")
    parser.add_argument("--input_dir", default="data/processed")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--candidate_threshold", type=float, default=0.5)
    parser.add_argument("--min_candidate_voxels", type=int, default=10)
    parser.add_argument("--primary_patch_size", type=int, default=256)
    parser.add_argument("--fp_threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    fp_model = load_fp_model(args.fp_checkpoint, args.device)

    from src.inference import InferencePipeline, MONAIBundleDetectionPipeline, is_monai_bundle_path

    if is_monai_bundle_path(args.checkpoint):
        pipeline = MONAIBundleDetectionPipeline(
            bundle_dir=args.checkpoint,
            fp_model=fp_model,
            fp_threshold=args.fp_threshold,
            device=args.device,
        )
    else:
        primary_model = load_primary_model(args.checkpoint, args.device)
        pipeline = InferencePipeline(
            primary_model=primary_model,
            fp_model=fp_model,
            fp_threshold=args.fp_threshold,
            candidate_threshold=args.candidate_threshold,
            min_candidate_voxels=args.min_candidate_voxels,
            device=args.device,
            primary_patch_size=args.primary_patch_size,
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
