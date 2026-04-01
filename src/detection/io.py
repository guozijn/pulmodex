"""Data preparation and geometry helpers for MONAI lung nodule detection."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from src.data.preprocessing import load_image_volume

_SUPPORTED_IMAGE_SUFFIXES = (".nii", ".nii.gz")
_SUBSET_PATTERN = re.compile(r"subset(\d+)", re.IGNORECASE)
log = logging.getLogger(__name__)


def _is_supported_image(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(_SUPPORTED_IMAGE_SUFFIXES)


def _resolve_standardized_root(path: str | Path) -> Path:
    path = Path(path)
    if (path / "dataset_index.json").exists():
        return path
    if path.name == "images" and (path.parent / "dataset_index.json").exists():
        return path.parent
    raise FileNotFoundError(
        f"Could not find dataset_index.json under {path}. Run `pulmodex detect standardize` first."
    )


def _load_annotations_by_seriesuid(annotations_path: str | Path) -> dict[str, list[list[float]]]:
    grouped: dict[str, list[list[float]]] = {}
    with Path(annotations_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{annotations_path} is missing required columns: {sorted(required)}")
        for row in reader:
            diameter = float(row["diameter_mm"])
            grouped.setdefault(str(row["seriesuid"]), []).append(
                [
                    float(row["coordX"]),
                    float(row["coordY"]),
                    float(row["coordZ"]),
                    diameter,
                    diameter,
                    diameter,
                ]
            )
    return grouped


def _subset_id_from_source_path(source_path: str | Path) -> int | None:
    source_path = Path(source_path)
    for part in source_path.parts:
        match = _SUBSET_PATTERN.fullmatch(part)
        if match:
            return int(match.group(1))
    return None


def _build_detection_item(image_path: str, boxes_world: list[list[float]]) -> dict[str, Any]:
    return {
        "image": image_path,
        "box": boxes_world,
        "label": [0 for _ in boxes_world],
    }


def prepare_luna16_detection_splits(
    standardized_dir: str | Path = "data/monai_detection_nifti",
    annotations_path: str | Path = "data/evaluationScript/annotations/annotations.csv",
    output_dir: str | Path = "data/monai_detection_nifti_prepared",
) -> list[Path]:
    """Create fold manifests from standardized LUNA16 NIfTI volumes and annotations.

    This mirrors the subset-based split logic used by LUNA16/nnDetection:
    validation fold ``i`` corresponds to scans from ``subset{i}``, and the
    training split is the union of the remaining discovered subsets.
    """

    standardized_root = _resolve_standardized_root(standardized_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads((standardized_root / "dataset_index.json").read_text())
    annotations_by_series = _load_annotations_by_seriesuid(annotations_path)

    items_by_subset: dict[int, list[dict[str, Any]]] = {}
    skipped: list[str] = []
    for item in payload.get("items", []):
        image_path = Path(item["image"])
        if not _is_supported_image(image_path):
            continue
        subset_id = _subset_id_from_source_path(item.get("source_path", ""))
        if subset_id is None:
            skipped.append(str(item.get("source_path", image_path)))
            continue
        seriesuid = str(item["seriesuid"])
        boxes_world = annotations_by_series.get(seriesuid, [])
        items_by_subset.setdefault(subset_id, []).append(
            _build_detection_item(str(image_path.resolve()), boxes_world)
        )

    if not items_by_subset:
        raise ValueError(
            "No subset-aware standardized images were found. Expected source paths to contain subset0..subset9."
        )
    if skipped:
        log.warning("Skipped %d standardized image(s) without subset information.", len(skipped))

    written: list[Path] = []
    subset_ids = sorted(items_by_subset)
    for fold in subset_ids:
        validation = list(items_by_subset.get(fold, []))
        training = [item for subset_id, subset_items in items_by_subset.items() if subset_id != fold for item in subset_items]
        resolved = {"training": training, "validation": validation}
        dest = output_dir / f"dataset_fold{fold}.json"
        dest.write_text(json.dumps(resolved, indent=2))
        written.append(dest)

    metadata = {
        "standardized_dir": str(standardized_root.resolve()),
        "annotations_path": str(Path(annotations_path).resolve()),
        "prepared_split_dir": str(output_dir.resolve()),
        "num_split_files": len(written),
        "subset_ids": subset_ids,
        "num_images": int(sum(len(items) for items in items_by_subset.values())),
        "num_annotation_series": int(len(annotations_by_series)),
        "skipped_without_subset": skipped,
    }
    (output_dir / "dataset_index.json").write_text(json.dumps(metadata, indent=2))
    return written


def _discover_mhd_sources(input_dir: str | Path) -> list[Path]:
    return sorted(Path(input_dir).rglob("*.mhd"))


def _discover_dicom_series_dirs(input_dir: str | Path) -> list[Path]:
    series_dirs: list[Path] = []
    for path in sorted(Path(input_dir).rglob("*")):
        if not path.is_dir():
            continue
        try:
            files = [child for child in path.iterdir() if child.is_file()]
        except PermissionError:
            continue
        if files and any(child.suffix.lower() == ".dcm" for child in files):
            series_dirs.append(path)
    return series_dirs


def _load_dicom_series_image(series_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = list(reader.GetGDCMSeriesIDs(str(series_dir)))
    if series_ids:
        file_names = reader.GetGDCMSeriesFileNames(str(series_dir), series_ids[0])
        reader.SetFileNames(file_names)
        return reader.Execute()

    file_names = sorted(str(path) for path in series_dir.glob("*.dcm"))
    if not file_names:
        raise FileNotFoundError(f"No DICOM files found under {series_dir}")
    reader.SetFileNames(file_names)
    return reader.Execute()


def _sanitize_seriesuid(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "series"


def _dicom_series_uid(series_dir: Path) -> str:
    reader = sitk.ImageSeriesReader()
    series_ids = list(reader.GetGDCMSeriesIDs(str(series_dir)))
    if series_ids:
        return _sanitize_seriesuid(str(series_ids[0]))
    digest = hashlib.sha1(str(series_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return _sanitize_seriesuid(f"{series_dir.name}_{digest}")


def _seriesuid_for_source(source: Path, source_format: str) -> str:
    if source_format == "mhd":
        return _sanitize_seriesuid(source.stem)
    return _dicom_series_uid(source)


def _unique_seriesuid(seriesuid: str, source: Path, used_seriesuids: set[str]) -> str:
    if seriesuid not in used_seriesuids:
        used_seriesuids.add(seriesuid)
        return seriesuid
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    unique = _sanitize_seriesuid(f"{seriesuid}_{digest}")
    used_seriesuids.add(unique)
    return unique


def prepare_detection_inputs_as_nifti(
    input_dir: str | Path,
    output_dir: str | Path = "data/monai_detection_nifti",
    source_format: str = "auto",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Standardize raw MHD or DICOM inputs into compressed NIfTI volumes.

    The output directory contains:
    - ``images/<seriesuid>.nii.gz`` for each discovered scan
    - ``dataset_index.json`` with a manifest of converted files
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if source_format not in {"auto", "mhd", "dicom"}:
        raise ValueError("source_format must be one of: auto, mhd, dicom")

    resolved_format = source_format
    if source_format == "auto":
        mhd_sources = _discover_mhd_sources(input_dir)
        dicom_sources = _discover_dicom_series_dirs(input_dir)
        if mhd_sources:
            resolved_format = "mhd"
            sources: list[Path] = mhd_sources
        elif dicom_sources:
            resolved_format = "dicom"
            sources = dicom_sources
        else:
            raise FileNotFoundError(f"No .mhd files or DICOM series were found under {input_dir}")
    elif source_format == "mhd":
        sources = _discover_mhd_sources(input_dir)
    else:
        sources = _discover_dicom_series_dirs(input_dir)

    if not sources:
        raise FileNotFoundError(f"No {resolved_format} sources were found under {input_dir}")

    if limit is not None:
        sources = sources[: max(int(limit), 0)]

    manifest: list[dict[str, Any]] = []
    used_seriesuids: set[str] = set()
    for source in sources:
        seriesuid = _unique_seriesuid(_seriesuid_for_source(source, resolved_format), source, used_seriesuids)
        output_path = images_dir / f"{seriesuid}.nii.gz"
        if resolved_format == "mhd":
            image = sitk.ReadImage(str(source))
        else:
            image = _load_dicom_series_image(source)
        sitk.WriteImage(image, str(output_path), useCompression=True)
        manifest.append(
            {
                "seriesuid": seriesuid,
                "source_format": resolved_format,
                "source_path": str(source.resolve()),
                "image": str(output_path.resolve()),
                "spacing_xyz": [float(v) for v in image.GetSpacing()],
                "origin_xyz": [float(v) for v in image.GetOrigin()],
                "direction": [float(v) for v in image.GetDirection()],
                "size_xyz": [int(v) for v in image.GetSize()],
            }
        )

    index = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "source_format": resolved_format,
        "num_images": len(manifest),
        "items": manifest,
    }
    (output_dir / "dataset_index.json").write_text(json.dumps(index, indent=2))
    return manifest


def seriesuid_from_image_path(image_path: str | Path) -> str:
    image_path = Path(image_path)
    name = image_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return image_path.stem


def load_prepared_split(
    fold: int,
    prepared_dir: str | Path = "data/monai_detection_nifti_prepared",
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

    vol, spacing_zyx, origin_zyx = load_image_volume(str(image_path))
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
