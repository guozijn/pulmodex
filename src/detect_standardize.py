"""Standardize raw detection inputs into compressed NIfTI volumes."""

from __future__ import annotations

import argparse
import logging

from src.detection import prepare_detection_inputs_as_nifti


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standardize raw MHD or DICOM detection inputs into NIfTI volumes."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="data/monai_detection_nifti")
    parser.add_argument("--source_format", choices=("auto", "mhd", "dicom"), default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    written = prepare_detection_inputs_as_nifti(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        source_format=args.source_format,
        limit=args.limit,
    )
    logging.info("Prepared %d NIfTI file(s) in %s", len(written), args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
