"""Slice renderer for base CT slices and transparent visual overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import nibabel as nib
import numpy as np
import pandas as pd

VIEW = Literal["axial", "coronal", "sagittal"]


def _apply_lung_window(
    vol: np.ndarray,
    window_level: int = -600,
    window_width: int = 1500,
) -> np.ndarray:
    """Convert HU or [0,1] normalised volume to 8-bit with lung window.

    If vol is in [0,1] (already normalised), convert back to HU first.
    """
    # Detect if normalised
    if vol.max() <= 1.01 and vol.min() >= -0.01:
        hu_min, hu_max = -1000.0, 400.0
        vol = vol * (hu_max - hu_min) + hu_min

    low = window_level - window_width / 2
    high = window_level + window_width / 2
    vol_clipped = np.clip(vol, low, high)
    vol_8bit = ((vol_clipped - low) / (high - low) * 255).astype(np.uint8)
    return vol_8bit


def _draw_candidates(
    img: np.ndarray,
    candidates_on_slice: list[dict],
    spacing_yx: tuple[float, float],
    fp_threshold: float,
    confident_color: tuple[int, int, int] = (0, 255, 0),
    uncertain_color: tuple[int, int, int] = (0, 180, 0),
) -> np.ndarray:
    """Draw candidate square boxes and confidence scores.

    Args:
        img: (H, W, 4) uint8 BGRA
        candidates_on_slice: list of {cy, cx, prob, diameter_mm}
        spacing_yx: (mm/pixel_y, mm/pixel_x)
        fp_threshold: threshold for confident vs uncertain colour
    """
    for cand in candidates_on_slice:
        cy = int(round(cand["cy"]))
        cx = int(round(cand["cx"]))
        prob = float(cand["prob"])
        diam_mm = float(cand["diameter_mm"])

        # Half box size in pixels — keep a modest visible minimum for tiny nodules.
        half_h_px = max(8, int(round((diam_mm / 2.0) / spacing_yx[0])))
        half_w_px = max(8, int(round((diam_mm / 2.0) / spacing_yx[1])))
        top_left = (cx - half_w_px, cy - half_h_px)
        bottom_right = (cx + half_w_px, cy + half_h_px)

        color = confident_color if prob >= fp_threshold else uncertain_color
        # Main square box
        cv2.rectangle(img, top_left, bottom_right, (*color, 255), thickness=1, lineType=cv2.LINE_AA)
        # Score + diameter label with dark shadow for readability
        prob_label = f"{prob * 100:.0f}%"
        diam_label = f"{diam_mm:.1f}mm"
        lx, ly = cx + half_w_px + 4, cy + 4
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        line_gap = 14
        for row, text in enumerate((prob_label, diam_label)):
            y = ly + row * line_gap
            cv2.putText(img, text, (lx + 1, y + 1), font, scale, (0, 0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, text, (lx, y), font, scale, (*color, 255), thick, cv2.LINE_AA)
    return img


def render_slices(
    scan_output_dir: str,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    saliency_alpha: float = 0.4,
    fp_threshold: float = 0.5,
    window_level: int = -600,
    window_width: int = 1500,
    confident_color: tuple[int, int, int] = (0, 255, 0),
    uncertain_color: tuple[int, int, int] = (0, 180, 0),
) -> list[str]:
    """Render annotated PNG slices for all three views.

    Reads:
      - ct_volume.nii.gz or confidence_map.nii.gz as CT proxy
      - saliency_map.nii.gz
      - candidates.csv

    Writes boxed CT PNG slices under scan_output_dir/slices/.

    Args:
        scan_output_dir: output directory for a single scan
        spacing_mm: (z, y, x) voxel spacing in mm
    Returns:
        list of written file paths
    """
    base = Path(scan_output_dir)
    slice_dir = base / "slices"
    slice_dir.mkdir(exist_ok=True)
    for old_png in slice_dir.glob("*.png"):
        old_png.unlink()

    ct_proxy_path = base / "ct_volume.nii.gz"
    if not ct_proxy_path.exists():
        ct_proxy_path = base / "confidence_map.nii.gz"

    ct_proxy_img = nib.load(str(ct_proxy_path))
    spacing_from_affine = tuple(
        float(v)
        for v in np.linalg.norm(ct_proxy_img.affine[:3, :3], axis=0)[::-1]
    )
    spacing_mm = spacing_from_affine

    conf_map: np.ndarray = ct_proxy_img.get_fdata().astype(np.float32)

    cand_df = (
        pd.read_csv(base / "candidates.csv")
        if (base / "candidates.csv").exists()
        else pd.DataFrame()
    )

    if cand_df.empty:
        return []

    written: list[str] = []
    view_configs: list[tuple[str, int]] = [("axial", 0), ("coronal", 1), ("sagittal", 2)]

    for view_name, axis in view_configs:
        spacing_yx = (spacing_mm[1], spacing_mm[2]) if axis == 0 else (
            (spacing_mm[0], spacing_mm[2]) if axis == 1 else (spacing_mm[0], spacing_mm[1])
        )
        relevant_indices = sorted(_candidate_slice_indices(cand_df, axis, conf_map.shape[axis], spacing_mm))
        if not relevant_indices:
            continue

        for idx in relevant_indices:
            sl = [slice(None)] * 3
            sl[axis] = idx

            conf_slice = conf_map[tuple(sl)]
            grey = _apply_lung_window(conf_slice, window_level, window_width)
            base_bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

            # Draw nodule boxes directly onto the base image so they are
            # always visible and are the only persisted image output.
            on_slice = _candidates_on_slice(cand_df, axis, idx, spacing_mm)
            if on_slice:
                base_bgra = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2BGRA)
                base_bgra = _draw_candidates(
                    base_bgra,
                    on_slice,
                    spacing_yx,
                    fp_threshold,
                    confident_color,
                    uncertain_color,
                )
                base_bgr = cv2.cvtColor(base_bgra, cv2.COLOR_BGRA2BGR)

            base_path = str(slice_dir / f"{view_name}_{idx:04d}.png")
            cv2.imwrite(base_path, base_bgr)
            written.append(base_path)

    return written


def _candidates_on_slice(
    cand_df: pd.DataFrame,
    axis: int,
    slice_idx: int,
    spacing_mm: tuple[float, float, float],
) -> list[dict]:
    """Return candidates whose centroid lies within ±radius voxels of slice_idx."""
    result = []
    for _, row in cand_df.iterrows():
        if {"voxel_z", "voxel_y", "voxel_x"}.issubset(cand_df.columns):
            vox_z = float(row["voxel_z"])
            vox_y = float(row["voxel_y"])
            vox_x = float(row["voxel_x"])
        else:
            # Fallback for older artefacts without voxel columns.
            vox_z = row["coordZ"] / spacing_mm[0]
            vox_y = row["coordY"] / spacing_mm[1]
            vox_x = row["coordX"] / spacing_mm[2]
        radius_vox = row["diameter_mm"] / 2.0 / spacing_mm[axis]

        centres = [vox_z, vox_y, vox_x]
        if abs(centres[axis] - slice_idx) > radius_vox:
            continue

        plane_coords = [c for i, c in enumerate(centres) if i != axis]
        result.append({
            "cy": plane_coords[0],
            "cx": plane_coords[1],
            "prob": row.get("fp_prob", row.get("prob", 0.0)),
            "diameter_mm": row["diameter_mm"],
        })
    return result


def _candidate_slice_indices(
    cand_df: pd.DataFrame,
    axis: int,
    axis_length: int,
    spacing_mm: tuple[float, float, float],
) -> set[int]:
    """Return the slice indices that intersect at least one candidate."""
    indices: set[int] = set()
    for _, row in cand_df.iterrows():
        if {"voxel_z", "voxel_y", "voxel_x"}.issubset(cand_df.columns):
            centres = [
                float(row["voxel_z"]),
                float(row["voxel_y"]),
                float(row["voxel_x"]),
            ]
        else:
            centres = [
                float(row["coordZ"]) / spacing_mm[0],
                float(row["coordY"]) / spacing_mm[1],
                float(row["coordX"]) / spacing_mm[2],
            ]

        radius_vox = float(row["diameter_mm"]) / 2.0 / spacing_mm[axis]
        start = max(0, int(np.ceil(centres[axis] - radius_vox)))
        end = min(axis_length - 1, int(np.floor(centres[axis] + radius_vox)))
        if end < start:
            rounded = int(round(centres[axis]))
            if 0 <= rounded < axis_length:
                indices.add(rounded)
            continue
        indices.update(range(start, end + 1))
    return indices
