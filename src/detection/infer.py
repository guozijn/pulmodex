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

from src.data.preprocessing import load_mhd, normalise_hu, resample_to_isotropic

from .io import voxel_corners_to_world_box

log = logging.getLogger(__name__)


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
    mhd_path: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    score_thresh: float = 0.15,
) -> dict[str, Any]:
    mhd_path = Path(mhd_path)
    seriesuid = mhd_path.stem
    out_dir = Path(output_dir) / seriesuid
    out_dir.mkdir(parents=True, exist_ok=True)

    vol, spacing_zyx, origin_zyx = load_mhd(str(mhd_path))
    vol, spacing_zyx = resample_to_isotropic(vol, spacing_zyx, target_spacing=1.0)
    vol = normalise_hu(vol)

    image = torch.from_numpy(vol[None, None, ...]).to(device)
    detector.eval()
    with torch.inference_mode():
        outputs = detector(image, use_inferer=True)[0]

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

    candidates: list[dict[str, Any]] = []
    for box_voxel, score in zip(boxes, scores):
        if float(score) < score_thresh:
            continue
        box_world = voxel_corners_to_world_box(np.asarray(box_voxel), spacing_zyx, origin_zyx)
        candidates.append(_candidate_payload(box_world, float(score), np.asarray(box_voxel)))

    candidates.sort(key=lambda item: item["prob"], reverse=True)
    seg_mask, confidence_map = _build_detection_maps(tuple(vol.shape), candidates, spacing_zyx)
    affine = np.diag(list(spacing_zyx[::-1]) + [1.0])

    nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), str(out_dir / "ct_volume.nii.gz"))
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
) -> list[dict[str, Any]]:
    reports = []
    for mhd_path in sorted(Path(input_dir).rglob("*.mhd")):
        log.info("Running 3D detection inference for %s", mhd_path.name)
        reports.append(
            infer_detection_case(
                detector=detector,
                mhd_path=mhd_path,
                output_dir=output_dir,
                device=device,
                score_thresh=score_thresh,
            )
        )
    return reports
