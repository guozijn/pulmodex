from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.detection.evaluate import _filter_excluded_predictions
from src.detection.io import (
    prepare_luna16_detection_splits,
    voxel_corners_to_world_box,
    world_box_to_voxel_corners,
)


def test_prepare_luna16_detection_splits_resolves_absolute_paths(tmp_path: Path) -> None:
    standardized_dir = tmp_path / "standardized"
    images_dir = standardized_dir / "images"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "case_a.nii.gz"
    image_path.write_text("fake nii")
    payload = {
        "items": [
            {
                "seriesuid": "case_a",
                "source_path": str((tmp_path / "subset0" / "case_a.mhd").resolve()),
                "image": str(image_path.resolve()),
            }
        ]
    }
    (standardized_dir / "dataset_index.json").write_text(json.dumps(payload))

    annotations_path = tmp_path / "annotations.csv"
    annotations_path.write_text("seriesuid,coordX,coordY,coordZ,diameter_mm\n")

    written = prepare_luna16_detection_splits(
        standardized_dir=standardized_dir,
        annotations_path=annotations_path,
        output_dir=tmp_path / "out",
    )

    assert len(written) == 1
    prepared = json.loads((tmp_path / "out" / "dataset_fold0.json").read_text())
    assert prepared["validation"]
    assert Path(prepared["validation"][0]["image"]).is_absolute()
    assert prepared["validation"][0]["image"].endswith("case_a.nii.gz")


def test_world_box_round_trip_preserves_geometry() -> None:
    spacing_zyx = np.array([1.25, 0.8, 0.7], dtype=np.float32)
    origin_zyx = np.array([-200.0, -120.0, 30.0], dtype=np.float32)
    box_world = np.array([10.0, 20.0, -30.0, 8.0, 10.0, 12.0], dtype=np.float32)

    voxel_box = world_box_to_voxel_corners(box_world, spacing_zyx, origin_zyx)
    restored = voxel_corners_to_world_box(voxel_box, spacing_zyx, origin_zyx)

    assert np.allclose(restored, box_world, atol=1e-4)


def test_filter_excluded_predictions_drops_nearby_marks() -> None:
    import pandas as pd

    predictions = [
        {"seriesuid": "case1", "coordX": 10.0, "coordY": 10.0, "coordZ": 10.0, "prob": 0.9},
        {"seriesuid": "case1", "coordX": 80.0, "coordY": 80.0, "coordZ": 80.0, "prob": 0.7},
    ]
    excluded = pd.DataFrame(
        [{"seriesuid": "case1", "coordX": 12.0, "coordY": 9.0, "coordZ": 10.0, "diameter_mm": 8.0}]
    )

    kept = _filter_excluded_predictions(predictions, excluded)

    assert len(kept) == 1
    assert kept[0]["coordX"] == 80.0
