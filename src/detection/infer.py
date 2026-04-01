"""Inference helpers for MONAI 3D detection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from .data import load_preprocessed_detection_case
from .io import voxel_corners_to_world_box

log = logging.getLogger(__name__)


def _iter_supported_detection_images(input_dir: str | Path) -> list[Path]:
    input_dir = Path(input_dir)
    paths = [path for path in input_dir.rglob("*") if path.is_file()]
    supported = []
    for path in paths:
        name = path.name.lower()
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            supported.append(path)
    return sorted(supported)


def _should_use_inferer(detector: Any, vol_shape: tuple[int, int, int]) -> bool:
    inferer = getattr(detector, "inferer", None)
    roi_size = getattr(inferer, "roi_size", None)
    if roi_size is None:
        return False
    roi_size = tuple(int(v) for v in roi_size)
    return any(dim > roi for dim, roi in zip(vol_shape, roi_size))


def _minimum_input_shape(detector: Any) -> tuple[int, int, int] | None:
    network = getattr(detector, "network", None)
    size_divisible = getattr(network, "size_divisible", None)
    if size_divisible is None:
        return None
    return tuple(int(v) for v in size_divisible)


def _pad_volume_to_min_shape(
    vol: np.ndarray,
    min_shape_zyx: tuple[int, int, int] | None,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    original_shape = tuple(int(v) for v in vol.shape)
    if min_shape_zyx is None:
        return vol, original_shape
    target_shape = tuple(max(dim, minimum) for dim, minimum in zip(original_shape, min_shape_zyx))
    if target_shape == original_shape:
        return vol, original_shape
    pad_width = [(0, target - dim) for dim, target in zip(original_shape, target_shape)]
    padded = np.pad(vol, pad_width, mode="constant", constant_values=0.0)
    return padded.astype(np.float32), original_shape


def _clip_boxes_to_image_shape(
    boxes: np.ndarray,
    image_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if boxes.size == 0:
        return boxes.reshape(0, 6).astype(np.float32), np.zeros((0,), dtype=bool)
    max_xyz = np.asarray(
        [image_shape_zyx[2], image_shape_zyx[1], image_shape_zyx[0], image_shape_zyx[2], image_shape_zyx[1], image_shape_zyx[0]],
        dtype=np.float32,
    )
    clipped = np.clip(boxes, 0.0, max_xyz)
    valid = np.all(clipped[:, 3:] - clipped[:, :3] >= 1.0, axis=1)
    return clipped[valid].astype(np.float32), valid


def _candidate_payload(box_world: np.ndarray, score: float, box_voxel: np.ndarray) -> dict[str, Any]:
    centre_zyx = np.array(
        [
            (box_voxel[2] + box_voxel[5]) / 2.0,
            (box_voxel[1] + box_voxel[4]) / 2.0,
            (box_voxel[0] + box_voxel[3]) / 2.0,
        ],
        dtype=np.float32,
    )
    payload = {
        "coordX": float(box_world[0]),
        "coordY": float(box_world[1]),
        "coordZ": float(box_world[2]),
        "diameter_mm": float(max(box_world[3:])),
        "bbox_mm": [float(v) for v in box_world[3:]],
        "prob": float(score),
        "centre_zyx": centre_zyx,
        "slice_indices": {
            "axial": int(round(float(centre_zyx[0]))),
            "coronal": int(round(float(centre_zyx[1]))),
            "sagittal": int(round(float(centre_zyx[2]))),
        },
    }
    return payload


def _build_detection_maps(
    vol_shape: tuple[int, int, int],
    candidates: list[dict[str, Any]],
    spacing_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    seg_mask = np.zeros(vol_shape, dtype=np.uint8)
    confidence_map = np.zeros(vol_shape, dtype=np.float32)
    for cand in candidates:
        centre = cand["centre_zyx"].astype(int)
        radius_vox = max(2, int(round((cand["diameter_mm"] / 2.0) / float(spacing_zyx.min()))))
        z0 = max(0, centre[0] - radius_vox)
        z1 = min(vol_shape[0], centre[0] + radius_vox + 1)
        y0 = max(0, centre[1] - radius_vox)
        y1 = min(vol_shape[1], centre[1] + radius_vox + 1)
        x0 = max(0, centre[2] - radius_vox)
        x1 = min(vol_shape[2], centre[2] + radius_vox + 1)
        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        sphere = (
            (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2
        ) <= radius_vox**2
        seg_mask[z0:z1, y0:y1, x0:x1][sphere] = 1
        confidence_map[z0:z1, y0:y1, x0:x1][sphere] = np.maximum(
            confidence_map[z0:z1, y0:y1, x0:x1][sphere],
            float(cand["prob"]),
        )
    return seg_mask, confidence_map


def infer_detection_case(
    detector: Any,
    image_path: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    score_thresh: float = 0.15,
    target_spacing: float = 1.0,
) -> dict[str, Any]:
    image_path = Path(image_path)
    seriesuid = image_path.name
    if seriesuid.endswith(".nii.gz"):
        seriesuid = seriesuid[:-7]
    else:
        seriesuid = image_path.stem
    out_dir = Path(output_dir) / seriesuid
    out_dir.mkdir(parents=True, exist_ok=True)

    case = load_preprocessed_detection_case(image_path, boxes_world=None, target_spacing=target_spacing)
    vol = np.asarray(case["image"][0], dtype=np.float32)
    spacing_zyx = np.asarray(case["spacing_zyx"], dtype=np.float32)
    origin_zyx = np.asarray(case["origin_zyx"], dtype=np.float32)
    vol, original_shape_zyx = _pad_volume_to_min_shape(vol, _minimum_input_shape(detector))

    image = torch.from_numpy(vol[None, None, ...]).to(device)
    use_inferer = _should_use_inferer(detector, tuple(vol.shape))
    detector.eval()
    with torch.inference_mode():
        try:
            outputs = detector(image, use_inferer=use_inferer)[0]
        except RuntimeError as exc:
            if use_inferer and "split_with_sizes expects split_sizes to sum exactly" in str(exc):
                log.warning(
                    "Sliding-window inference failed for %s with shape %s; retrying without inferer.",
                    seriesuid,
                    tuple(vol.shape),
                )
                outputs = detector(image, use_inferer=False)[0]
            else:
                raise

    boxes = outputs.get(detector.target_box_key)
    scores = outputs.get(detector.pred_score_key)
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    else:
        boxes = np.asarray(boxes if boxes is not None else [])
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    else:
        scores = np.asarray(scores if scores is not None else [])
    boxes, valid_box_mask = _clip_boxes_to_image_shape(np.asarray(boxes), original_shape_zyx)
    scores = scores[valid_box_mask] if len(valid_box_mask) else np.asarray(scores[:0])

    candidates: list[dict[str, Any]] = []
    for box_voxel, score in zip(boxes, scores):
        if float(score) < score_thresh:
            continue
        box_world = voxel_corners_to_world_box(np.asarray(box_voxel), spacing_zyx, origin_zyx)
        candidates.append(_candidate_payload(box_world, float(score), np.asarray(box_voxel)))

    candidates.sort(key=lambda item: item["prob"], reverse=True)
    output_vol = vol[: original_shape_zyx[0], : original_shape_zyx[1], : original_shape_zyx[2]]
    seg_mask, confidence_map = _build_detection_maps(original_shape_zyx, candidates, spacing_zyx)
    affine = np.diag(list(spacing_zyx[::-1]) + [1.0])

    nib.save(nib.Nifti1Image(output_vol.astype(np.float32), affine), str(out_dir / "ct_volume.nii.gz"))
    nib.save(nib.Nifti1Image(seg_mask, affine), str(out_dir / "seg_mask.nii.gz"))
    nib.save(nib.Nifti1Image(confidence_map, affine), str(out_dir / "confidence_map.nii.gz"))
    nib.save(nib.Nifti1Image(np.zeros_like(vol, dtype=np.float32), affine), str(out_dir / "saliency_map.nii.gz"))

    rows = []
    for cand in candidates:
        rows.append(
            {
                "seriesuid": seriesuid,
                "coordX": cand["coordX"],
                "coordY": cand["coordY"],
                "coordZ": cand["coordZ"],
                "prob": cand["prob"],
                "diameter_mm": cand["diameter_mm"],
                "slice_axial": cand["slice_indices"]["axial"],
                "slice_coronal": cand["slice_indices"]["coronal"],
                "slice_sagittal": cand["slice_indices"]["sagittal"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "candidates.csv", index=False)

    report = {
        "seriesuid": seriesuid,
        "n_candidates_final": len(candidates),
        "top_candidates": rows[:5],
        "candidates": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def infer_detection_directory(
    detector: Any,
    input_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    score_thresh: float = 0.15,
    target_spacing: float = 1.0,
) -> list[dict[str, Any]]:
    reports = []
    for image_path in _iter_supported_detection_images(input_dir):
        log.info("Running 3D detection inference for %s", image_path.name)
        reports.append(
            infer_detection_case(
                detector=detector,
                image_path=image_path,
                output_dir=output_dir,
                device=device,
                score_thresh=score_thresh,
                target_spacing=target_spacing,
            )
        )
    return reports
