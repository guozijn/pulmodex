"""FastAPI application.

Endpoints:
  POST /predict          — upload zipped DICOM series, enqueue Celery job → {job_id}
  GET  /status/{job_id}  — poll job status → {state, progress, result}
  GET  /slices/{uid}/{view} → list of PNG slice paths or streamed image
"""

from __future__ import annotations

import os
import shutil
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydicom.errors import InvalidDicomError

from src.webapp.tasks import celery_app, predict_task

app = FastAPI(title="Pulmodex API", version="0.1.0")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
UPLOAD_DIR = "/tmp/pulmodex_uploads"
_ALLOWED_SUFFIXES = {".zip"}


def _validate_upload_name(filename: str | None) -> str:
    if not filename:
        raise HTTPException(400, "Uploaded file must have a filename")

    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(400, "Only .zip uploads are supported")
    return safe_name


def _extract_upload_zip(zip_path: Path, dest_dir: Path) -> Path:
    extract_dir = dest_dir / "dicom_series"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    return extract_dir


def _slice_normal(dataset: pydicom.dataset.FileDataset) -> np.ndarray | None:
    orientation = getattr(dataset, "ImageOrientationPatient", None)
    if orientation is None or len(orientation) != 6:
        return None
    row = np.asarray([float(v) for v in orientation[:3]], dtype=np.float64)
    col = np.asarray([float(v) for v in orientation[3:]], dtype=np.float64)
    normal = np.cross(row, col)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None
    return normal / norm


def _slice_position(dataset: pydicom.dataset.FileDataset, path: Path) -> tuple[int, float | int | str]:
    if hasattr(dataset, "ImagePositionPatient"):
        position = np.asarray([float(v) for v in dataset.ImagePositionPatient], dtype=np.float64)
        normal = _slice_normal(dataset)
        if normal is not None:
            return (0, float(np.dot(position, normal)))
        return (1, float(position[2]))
    if hasattr(dataset, "InstanceNumber"):
        return (2, int(dataset.InstanceNumber))
    return (3, path.name)


def _find_series_files(scan_root: Path) -> list[Path]:
    groups: dict[str, list[tuple[pydicom.dataset.FileDataset, Path]]] = defaultdict(list)

    for path in scan_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                specific_tags=["SeriesInstanceUID", "ImagePositionPatient", "InstanceNumber"],
            )
        except (InvalidDicomError, FileNotFoundError, IsADirectoryError):
            continue

        series_uid = str(getattr(dataset, "SeriesInstanceUID", "default"))
        groups[series_uid].append((dataset, path))

    if not groups:
        raise HTTPException(400, "No DICOM files found in uploaded zip")

    def _series_rank(items: list[tuple[pydicom.dataset.FileDataset, Path]]) -> tuple[int, int, int]:
        sample = items[0][0]
        is_ct = int(str(getattr(sample, "Modality", "")).upper() == "CT")
        rows = int(getattr(sample, "Rows", 0) or 0)
        cols = int(getattr(sample, "Columns", 0) or 0)
        return (is_ct, len(items), rows * cols)

    selected = max(groups.values(), key=_series_rank)
    return [path for _, path in sorted(selected, key=lambda item: _slice_position(item[0], item[1]))]


def _resolve_slice_thickness(
    slices: list[pydicom.dataset.FileDataset],
    fallback: float = 1.0,
) -> float:
    if len(slices) > 1 and hasattr(slices[0], "ImagePositionPatient") and hasattr(slices[-1], "ImagePositionPatient"):
        normal = _slice_normal(slices[0])
        if normal is not None:
            projections = [
                float(np.dot(np.asarray([float(v) for v in s.ImagePositionPatient], dtype=np.float64), normal))
                for s in slices
            ]
            diffs = np.diff(sorted(projections))
            non_zero = np.abs(diffs[np.abs(diffs) > 1e-6])
            if non_zero.size > 0:
                return float(np.median(non_zero))

    for attr in ("SliceThickness", "SpacingBetweenSlices"):
        value = getattr(slices[0], attr, None)
        if value is not None:
            resolved = abs(float(value))
            if resolved > 0:
                return resolved
    return fallback


def _body_crop_image(
    image: sitk.Image,
    body_threshold_hu: float = -900.0,
    margin_mm: float = 12.0,
) -> sitk.Image:
    """Crop away obvious air/background while preserving scan geometry.

    The crop is intentionally conservative: threshold the body region, keep the
    largest connected component, then expand the bounding box by a small margin.
    If no stable foreground is found, return the original image unchanged.
    """
    # Cast to float32 so threshold bounds don't overflow integer pixel types.
    float_image = sitk.Cast(image, sitk.sitkFloat32)
    body_mask = sitk.BinaryThreshold(
        float_image,
        lowerThreshold=body_threshold_hu,
        upperThreshold=3072.0,  # above any physical HU value
        insideValue=1,
        outsideValue=0,
    )
    body_mask = sitk.BinaryMorphologicalClosing(body_mask, [2, 2, 1])
    body_mask = sitk.BinaryFillhole(body_mask)

    components = sitk.ConnectedComponent(body_mask)
    relabelled = sitk.RelabelComponent(components, sortByObjectSize=True)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(relabelled)
    if not stats.HasLabel(1):
        return image

    bbox = stats.GetBoundingBox(1)  # (x, y, z, size_x, size_y, size_z)
    size = list(image.GetSize())
    spacing = image.GetSpacing()
    margin_vox = [
        max(1, int(round(margin_mm / float(spacing_dim))))
        for spacing_dim in spacing
    ]

    start = [0, 0, 0]
    crop_size = [0, 0, 0]
    for axis in range(3):
        axis_start = max(0, int(bbox[axis]) - margin_vox[axis])
        axis_end = min(size[axis], int(bbox[axis] + bbox[axis + 3]) + margin_vox[axis])
        start[axis] = axis_start
        crop_size[axis] = max(1, axis_end - axis_start)

    return sitk.RegionOfInterest(image, size=crop_size, index=start)


def _convert_dicom_series_to_mhd(series_files: list[Path], out_path: Path) -> Path:
    slices = [pydicom.dcmread(str(path), stop_before_pixels=True) for path in series_files]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in series_files])
    image = reader.Execute()

    # Preserve physical geometry. Some series readers fall back to 1 mm slice
    # spacing, so correct it from the DICOM metadata when needed.
    spacing = list(image.GetSpacing())
    resolved_thickness = _resolve_slice_thickness(slices, fallback=spacing[2] if len(spacing) > 2 else 1.0)
    if len(spacing) == 3 and abs(float(spacing[2]) - resolved_thickness) > 1e-3:
        image.SetSpacing([float(spacing[0]), float(spacing[1]), float(resolved_thickness)])
    image = _body_crop_image(image)
    sitk.WriteImage(image, str(out_path), useCompression=False)
    return out_path


def _prepare_scan_input(upload_path: Path, stored_file: Path, seriesuid: str) -> Path:
    extract_dir = _extract_upload_zip(stored_file, upload_path)
    series_files = _find_series_files(extract_dir)
    return _convert_dicom_series_to_mhd(series_files, upload_path / f"{seriesuid}.mhd")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload a zipped DICOM series and enqueue inference.

    Returns:
        {"job_id": str}
    """
    seriesuid = str(uuid.uuid4())
    upload_path = Path(UPLOAD_DIR) / seriesuid
    upload_path.mkdir(parents=True, exist_ok=True)

    safe_name = _validate_upload_name(file.filename)
    dest = upload_path / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    scan_path = _prepare_scan_input(upload_path, dest, seriesuid)

    # Persist scan metadata so history survives restarts
    import json, datetime
    out_dir = Path(OUTPUT_DIR) / seriesuid
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "seriesuid": seriesuid,
        "filename": safe_name,
        "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))

    # Enqueue
    task = predict_task.delay(str(scan_path), OUTPUT_DIR, seriesuid)
    return {"job_id": task.id, "seriesuid": seriesuid}


@app.get("/status/{job_id}")
async def status(job_id: str):
    """Poll Celery task status.

    Returns:
        {"state": str, "progress": dict|None, "result": dict|None, "error": str|None}
    """
    result = celery_app.AsyncResult(job_id)
    info = result.info if result.state == "PROGRESS" else None
    error = None
    if result.state == "FAILURE":
        error = str(result.info or result.result or "Inference failed")
    return {
        "state": result.state,
        "progress": info,
        "result": result.result if result.successful() else None,
        "error": error,
    }


@app.get("/slices/{uid}/{view}")
async def get_slices(uid: str, view: str, idx: int = 0, layer: str = "composite"):
    """Return a rendered PNG slice.

    Args:
        uid: seriesuid
        view: axial | coronal | sagittal
        idx: slice index
        layer: composite | base | overlay

    Returns:
        PNG image response
    """
    if view not in ("axial", "coronal", "sagittal"):
        raise HTTPException(400, "view must be axial|coronal|sagittal")
    if layer not in ("composite", "base", "overlay"):
        raise HTTPException(400, "layer must be composite|base|overlay")

    filename = (
        f"{view}_{idx:04d}.png"
        if layer == "composite"
        else f"{layer}_{view}_{idx:04d}.png"
    )
    img_path = Path(OUTPUT_DIR) / uid / "slices" / filename
    if not img_path.exists():
        raise HTTPException(404, f"Slice not found: {img_path}")

    return FileResponse(str(img_path), media_type="image/png")


@app.get("/slices/{uid}/{view}/index")
async def list_slices(uid: str, view: str):
    """List all available slice indices for a view."""
    if view not in ("axial", "coronal", "sagittal"):
        raise HTTPException(400, "view must be axial|coronal|sagittal")

    slice_dir = Path(OUTPUT_DIR) / uid / "slices"
    if not slice_dir.exists():
        raise HTTPException(404, "Slices not yet generated")

    files = sorted(slice_dir.glob(f"base_{view}_*.png"))
    if not files:
        files = sorted(slice_dir.glob(f"{view}_*.png"))
        indices = [int(f.stem.split("_")[1]) for f in files]
    else:
        indices = [int(f.stem.split("_")[2]) for f in files]
    return {"view": view, "indices": indices, "count": len(indices)}


@app.get("/scans")
async def list_scans():
    """Return all completed scans ordered newest-first."""
    import json
    scans = []
    for scan_dir in sorted(Path(OUTPUT_DIR).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not scan_dir.is_dir():
            continue
        meta_path = scan_dir / "meta.json"
        report_path = scan_dir / "report.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if report_path.exists():
            meta["report"] = json.loads(report_path.read_text())
            meta["status"] = "done"
        else:
            meta["status"] = "pending"
        scans.append(meta)
    return scans


@app.get("/report/{uid}")
async def get_report(uid: str):
    """Return the inference report JSON for a scan."""
    report_path = Path(OUTPUT_DIR) / uid / "report.json"
    if not report_path.exists():
        raise HTTPException(404, "Report not found")
    import json
    return JSONResponse(json.loads(report_path.read_text()))
