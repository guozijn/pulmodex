"""Data preparation and geometry helpers for MONAI lung nodule detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.preprocessing import load_mhd


def _image_map(raw_data_dir: str | Path) -> dict[str, Path]:
    raw_data_dir = Path(raw_data_dir)
    return {path.name: path.resolve() for path in raw_data_dir.rglob("*.mhd")}


def _resolve_item(item: dict[str, Any], image_lookup: dict[str, Path]) -> dict[str, Any]:
    resolved = dict(item)
    image_name = Path(item["image"]).name
    if image_name not in image_lookup:
        raise FileNotFoundError(f"Could not resolve {image_name} under raw data directory.")
    resolved["image"] = str(image_lookup[image_name])
    return resolved


def prepare_luna16_detection_splits(
    raw_data_dir: str | Path = "data/orig_datasets",
    split_dir: str | Path = "data/LUNA16_datasplit/mhd_original",
    output_dir: str | Path = "data/monai_detection",
) -> list[Path]:
    """Resolve raw MHD paths in the existing LUNA16 split files.

    The upstream MONAI detection tutorial uses env files that point to the raw
    LUNA16 scans and fold JSONs. This helper makes those fold files directly
    usable from the current project by replacing relative image names with
    absolute local paths.
    """

    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_lookup = _image_map(raw_data_dir)

    written: list[Path] = []
    for source in sorted(split_dir.glob("dataset_fold*.json")):
        payload = json.loads(source.read_text())
        resolved = {
            key: [_resolve_item(item, image_lookup) for item in value]
            for key, value in payload.items()
        }
        dest = output_dir / source.name
        dest.write_text(json.dumps(resolved, indent=2))
        written.append(dest)

    metadata = {
        "raw_data_dir": str(Path(raw_data_dir).resolve()),
        "source_split_dir": str(split_dir.resolve()),
        "prepared_split_dir": str(output_dir.resolve()),
        "num_split_files": len(written),
    }
    (output_dir / "dataset_index.json").write_text(json.dumps(metadata, indent=2))
    return written


def load_prepared_split(
    fold: int,
    prepared_dir: str | Path = "data/monai_detection",
) -> dict[str, list[dict[str, Any]]]:
    path = Path(prepared_dir) / f"dataset_fold{fold}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared split {path} was not found. Run `pulmodex detect prepare` first."
        )
    return json.loads(path.read_text())


def world_box_to_voxel_corners(
    box_world_xyzwhd: np.ndarray,
    spacing_zyx: np.ndarray,
    origin_zyx: np.ndarray,
) -> np.ndarray:
    """Convert world-space center-size box to voxel xyzxyz."""

    spacing_xyz = np.asarray(spacing_zyx[::-1], dtype=np.float32)
    origin_xyz = np.asarray(origin_zyx[::-1], dtype=np.float32)
    centre_xyz = np.asarray(box_world_xyzwhd[:3], dtype=np.float32)
    size_xyz = np.asarray(box_world_xyzwhd[3:], dtype=np.float32)
    centre_voxel_xyz = (centre_xyz - origin_xyz) / spacing_xyz
    half_size_voxel_xyz = (size_xyz / spacing_xyz) / 2.0
    mins = centre_voxel_xyz - half_size_voxel_xyz
    maxs = centre_voxel_xyz + half_size_voxel_xyz
    return np.concatenate([mins, maxs]).astype(np.float32)


def voxel_corners_to_world_box(
    voxel_box_xyzxyz: np.ndarray,
    spacing_zyx: np.ndarray,
    origin_zyx: np.ndarray,
) -> np.ndarray:
    """Convert voxel xyzxyz box to world-space center-size xyzwhd."""

    spacing_xyz = np.asarray(spacing_zyx[::-1], dtype=np.float32)
    origin_xyz = np.asarray(origin_zyx[::-1], dtype=np.float32)
    mins = np.asarray(voxel_box_xyzxyz[:3], dtype=np.float32)
    maxs = np.asarray(voxel_box_xyzxyz[3:], dtype=np.float32)
    centre_voxel_xyz = (mins + maxs) / 2.0
    size_voxel_xyz = np.maximum(maxs - mins, 0.0)
    centre_world_xyz = origin_xyz + centre_voxel_xyz * spacing_xyz
    size_world_xyz = size_voxel_xyz * spacing_xyz
    return np.concatenate([centre_world_xyz, size_world_xyz]).astype(np.float32)


def load_detection_case(
    image_path: str | Path,
    boxes_world: list[list[float]],
) -> dict[str, Any]:
    """Load a raw LUNA16 case and convert world-space boxes to voxel corners."""

    vol, spacing_zyx, origin_zyx = load_mhd(str(image_path))
    boxes_voxel = np.asarray(
        [world_box_to_voxel_corners(np.asarray(box), spacing_zyx, origin_zyx) for box in boxes_world],
        dtype=np.float32,
    )
    return {
        "image_path": str(image_path),
        "image": vol.astype(np.float32),
        "spacing_zyx": spacing_zyx.astype(np.float32),
        "origin_zyx": origin_zyx.astype(np.float32),
        "boxes_world": np.asarray(boxes_world, dtype=np.float32),
        "boxes_voxel": boxes_voxel,
    }
