"""Generate a tiny LUNA16-style mock dataset for local development.

Usage:
    python scripts/generate_mock_luna16.py
    python scripts/generate_mock_luna16.py --output_dir data/mock_luna16 --scans_per_fold 1
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def add_blob(
    volume: np.ndarray,
    centre_zyx: np.ndarray,
    radius_mm: float,
    peak_hu: float,
) -> None:
    """Add a smooth spherical blob to a CT volume."""
    zz, yy, xx = np.ogrid[: volume.shape[0], : volume.shape[1], : volume.shape[2]]
    dist_sq = (
        (zz - centre_zyx[0]) ** 2
        + (yy - centre_zyx[1]) ** 2
        + (xx - centre_zyx[2]) ** 2
    )
    sigma_sq = max((radius_mm / 2.0) ** 2, 1.0)
    blob = np.exp(-dist_sq / (2.0 * sigma_sq)).astype(np.float32)
    volume += blob * peak_hu


def world_xyz_from_voxel(centre_zyx: np.ndarray, origin_zyx: np.ndarray) -> np.ndarray:
    """Convert voxel zyx to world xyz under 1 mm isotropic spacing."""
    world_zyx = origin_zyx + centre_zyx.astype(np.float32)
    return world_zyx[::-1]


def build_scan(
    rng: np.random.Generator,
    shape: tuple[int, int, int],
    origin_zyx: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]], list[dict[str, float]]]:
    """Create one synthetic CT plus matching annotation/candidate rows."""
    volume = rng.normal(loc=-850.0, scale=35.0, size=shape).astype(np.float32)

    annotations: list[dict[str, float]] = []
    candidates: list[dict[str, float]] = []

    n_nodules = int(rng.integers(1, 3))
    for _ in range(n_nodules):
        margin = 12
        centre_zyx = np.array(
            [
                rng.integers(margin, shape[0] - margin),
                rng.integers(margin, shape[1] - margin),
                rng.integers(margin, shape[2] - margin),
            ],
            dtype=np.int32,
        )
        diameter_mm = float(rng.uniform(6.0, 12.0))
        add_blob(volume, centre_zyx, radius_mm=diameter_mm / 2.0, peak_hu=900.0)

        coord_xyz = world_xyz_from_voxel(centre_zyx, origin_zyx)
        annotations.append(
            {
                "coordX": float(coord_xyz[0]),
                "coordY": float(coord_xyz[1]),
                "coordZ": float(coord_xyz[2]),
                "diameter_mm": diameter_mm,
            }
        )
        candidates.append(
            {
                "coordX": float(coord_xyz[0]),
                "coordY": float(coord_xyz[1]),
                "coordZ": float(coord_xyz[2]),
                "class": 1,
            }
        )

    n_negatives = int(rng.integers(2, 5))
    for _ in range(n_negatives):
        for _attempt in range(50):
            centre_zyx = np.array(
                [
                    rng.integers(8, shape[0] - 8),
                    rng.integers(8, shape[1] - 8),
                    rng.integers(8, shape[2] - 8),
                ],
                dtype=np.int32,
            )
            far_enough = True
            for ann in annotations:
                ann_zyx = np.array([ann["coordZ"], ann["coordY"], ann["coordX"]], dtype=np.float32)
                ann_voxel = ann_zyx - origin_zyx
                if np.linalg.norm(centre_zyx.astype(np.float32) - ann_voxel) < 10.0:
                    far_enough = False
                    break
            if far_enough:
                coord_xyz = world_xyz_from_voxel(centre_zyx, origin_zyx)
                candidates.append(
                    {
                        "coordX": float(coord_xyz[0]),
                        "coordY": float(coord_xyz[1]),
                        "coordZ": float(coord_xyz[2]),
                        "class": 0,
                    }
                )
                break

    volume = np.clip(volume, -1000.0, 400.0).astype(np.int16)
    return volume, annotations, candidates


def write_volume(path: Path, volume: np.ndarray, origin_zyx: np.ndarray) -> None:
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin(tuple(origin_zyx[::-1].tolist()))
    sitk.WriteImage(image, str(path), useCompression=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock LUNA16 data")
    parser.add_argument("--output_dir", default="data/mock_luna16")
    parser.add_argument("--scans_per_fold", type=int, default=1)
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    shape = (args.shape, args.shape, args.shape)

    annotation_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for fold in range(10):
        subset_dir = output_dir / f"subset{fold}"
        subset_dir.mkdir(parents=True, exist_ok=True)

        for scan_idx in range(args.scans_per_fold):
            seriesuid = f"mock_fold{fold}_scan{scan_idx:02d}"
            origin_zyx = np.array(
                [
                    rng.integers(-30, 30),
                    rng.integers(-30, 30),
                    rng.integers(-30, 30),
                ],
                dtype=np.float32,
            )
            volume, annotations, candidates = build_scan(rng, shape, origin_zyx)
            write_volume(subset_dir / f"{seriesuid}.mhd", volume, origin_zyx)

            for row in annotations:
                annotation_rows.append({"seriesuid": seriesuid, **row})
            for row in candidates:
                candidate_rows.append({"seriesuid": seriesuid, **row})

    with (output_dir / "annotations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"],
        )
        writer.writeheader()
        writer.writerows(annotation_rows)

    with (output_dir / "candidates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seriesuid", "coordX", "coordY", "coordZ", "class"],
        )
        writer.writeheader()
        writer.writerows(candidate_rows)

    total_scans = 10 * args.scans_per_fold
    positives = sum(int(row["class"]) == 1 for row in candidate_rows)
    negatives = sum(int(row["class"]) == 0 for row in candidate_rows)
    print(
        f"Wrote {total_scans} scans to {output_dir} "
        f"with {len(annotation_rows)} annotations, "
        f"{positives} positive candidates, {negatives} negative candidates."
    )


if __name__ == "__main__":
    main()
