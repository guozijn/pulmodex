"""Prepare local LUNA16 data for the MONAI detection workflow."""

from __future__ import annotations

import argparse
import logging

from src.detection import prepare_luna16_detection_splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create LUNA16 fold manifests from standardized NIfTI scans and annotations."
    )
    parser.add_argument("--standardized_dir", default="data/monai_detection_nifti")
    parser.add_argument("--annotations", default="data/evaluationScript/annotations/annotations.csv")
    parser.add_argument("--output_dir", default="data/monai_detection_nifti_prepared")
    args = parser.parse_args()

    written = prepare_luna16_detection_splits(
        standardized_dir=args.standardized_dir,
        annotations_path=args.annotations,
        output_dir=args.output_dir,
    )
    logging.info("Prepared %d MONAI detection split file(s) in %s", len(written), args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
