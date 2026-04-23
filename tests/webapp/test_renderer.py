from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.webapp.renderer import render_slices


def _make_volume(tmp_path: Path, shape: tuple[int, int, int]) -> None:
    vol = np.zeros(shape, dtype=np.float32)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(vol, affine), str(tmp_path / "ct_volume.nii.gz"))


def _make_candidate(
    vox_z: int = 5,
    vox_y: int = 8,
    vox_x: int = 8,
    diameter_mm: float = 6.0,
    prob: float = 0.9,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "voxel_z": vox_z,
        "voxel_y": vox_y,
        "voxel_x": vox_x,
        "coordZ": float(vox_z),
        "coordY": float(vox_y),
        "coordX": float(vox_x),
        "prob": prob,
        "fp_prob": prob,
        "diameter_mm": diameter_mm,
    }])


def test_render_slices_only_writes_nodule_axial_slices(tmp_path):
    """Only axial slices within the nodule radius are written."""
    _make_volume(tmp_path, shape=(12, 16, 16))

    # Candidate at axial slice 5 with radius 3 voxels -> slices 2..8 visible.
    _make_candidate(vox_z=5, diameter_mm=6.0).to_csv(tmp_path / "candidates.csv", index=False)

    written = render_slices(str(tmp_path))

    axial_base_indices = sorted(
        int(Path(p).stem.split("_")[-1])
        for p in written
        if Path(p).stem.startswith("axial_")
    )
    assert axial_base_indices, "Expected at least one axial slice to be written"
    for idx in axial_base_indices:
        assert 2 <= idx <= 8, f"Axial slice {idx} written but should not contain the candidate"


def test_render_slices_writes_nothing_without_candidates(tmp_path):
    """No PNG files are written when there are no candidates."""
    _make_volume(tmp_path, shape=(10, 16, 16))
    # No candidates.csv

    written = render_slices(str(tmp_path))
    assert written == []


def test_render_slices_writes_nothing_with_empty_candidates_csv(tmp_path):
    """No PNG files are written when candidates.csv exists but is empty."""
    _make_volume(tmp_path, shape=(10, 16, 16))
    pd.DataFrame(columns=["voxel_z", "voxel_y", "voxel_x", "prob", "diameter_mm"]).to_csv(
        tmp_path / "candidates.csv", index=False
    )

    written = render_slices(str(tmp_path))
    assert written == []


def test_render_slices_only_writes_single_boxed_image_per_slice(tmp_path):
    """Each written slice produces a single boxed CT PNG."""
    _make_volume(tmp_path, shape=(10, 16, 16))
    _make_candidate(vox_z=5, diameter_mm=6.0).to_csv(tmp_path / "candidates.csv", index=False)

    written = render_slices(str(tmp_path))

    stems = [Path(p).stem for p in written]
    assert all(len(stem.split("_")) == 2 for stem in stems)
    assert all(Path(p).exists() for p in written)


def test_render_slices_multiple_candidates(tmp_path):
    """All slices from both candidates are written with no duplicates."""
    _make_volume(tmp_path, shape=(20, 16, 16))

    pd.DataFrame([
        {"voxel_z": 3, "voxel_y": 8, "voxel_x": 8,
         "coordZ": 3.0, "coordY": 8.0, "coordX": 8.0,
         "prob": 0.8, "fp_prob": 0.8, "diameter_mm": 4.0},
        {"voxel_z": 15, "voxel_y": 8, "voxel_x": 8,
         "coordZ": 15.0, "coordY": 8.0, "coordX": 8.0,
         "prob": 0.7, "fp_prob": 0.7, "diameter_mm": 4.0},
    ]).to_csv(tmp_path / "candidates.csv", index=False)

    written = render_slices(str(tmp_path))

    axial_base_indices = sorted(
        int(Path(p).stem.split("_")[-1])
        for p in written
        if Path(p).stem.startswith("axial_")
    )
    # No duplicates.
    assert len(axial_base_indices) == len(set(axial_base_indices))
    # Slice 3 visible region: 1..5; slice 15 visible region: 13..17.
    for idx in axial_base_indices:
        assert (1 <= idx <= 5) or (13 <= idx <= 17), (
            f"Axial slice {idx} written but neither candidate covers it"
        )
