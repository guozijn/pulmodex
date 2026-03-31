"""Prepare local LUNA16 data for the MONAI detection workflow."""

from __future__ import annotations

import argparse
import logging

from src.detection import prepare_luna16_detection_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local LUNA16 splits for MONAI detection.")
    parser.add_argument("--raw_data_dir", default="data/orig_datasets")
    parser.add_argument("--split_dir", default="data/LUNA16_datasplit/mhd_original")
    parser.add_argument("--output_dir", default="data/monai_detection")
    args = parser.parse_args()

    written = prepare_luna16_detection_splits(
        raw_data_dir=args.raw_data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
    )
    logging.info("Prepared %d MONAI detection split file(s) in %s", len(written), args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
