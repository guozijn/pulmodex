"""Slice renderer: annotated PNG slices for each CT view.

Rendering pipeline per slice:
  1. Apply lung window (WL −600, WW 1500) → 8-bit grey
  2. Overlay jet-coloured saliency map (configurable opacity)
  3. Draw circle per candidate (radius ∝ diameter_mm in pixels)
     - red   (255,   0,   0) if prob ≥ fp_threshold
     - orange(255, 165,   0) if prob <  fp_threshold
  4. Burn confidence score text next to each circle

Writes slices/<view>_<idx>.png under the scan's output directory.
Views: axial (z), coronal (y), sagittal (x).
"""

from __future__ import annotations

import json
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


def _overlay_saliency(
    grey: np.ndarray,
    saliency_slice: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay jet saliency on greyscale slice.

    Args:
        grey: (H, W) uint8
        saliency_slice: (H, W) float32 in [0, 1]
        alpha: saliency blend weight

    Returns:
        rgb: (H, W, 3) uint8
    """
    rgb = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    sal_8bit = (saliency_slice * 255).astype(np.uint8)
    jet = cv2.applyColorMap(sal_8bit, cv2.COLORMAP_JET)
    return cv2.addWeighted(rgb, 1.0 - alpha, jet, alpha, 0)


def _draw_candidates(
    img: np.ndarray,
    candidates_on_slice: list[dict],
    spacing_yx: tuple[float, float],
    fp_threshold: float,
    confident_color: tuple[int, int, int] = (255, 0, 0),
    uncertain_color: tuple[int, int, int] = (255, 165, 0),
) -> np.ndarray:
    """Draw candidate circles and confidence scores.

    Args:
        img: (H, W, 3) uint8 BGR
        candidates_on_slice: list of {cy, cx, prob, diameter_mm}
        spacing_yx: (mm/pixel_y, mm/pixel_x)
        fp_threshold: threshold for confident vs uncertain colour
    """
    for cand in candidates_on_slice:
        cy = int(round(cand["cy"]))
        cx = int(round(cand["cx"]))
        prob = float(cand["prob"])
        diam_mm = float(cand["diameter_mm"])

        # Radius in pixels (use y-spacing as approximation)
        radius_px = max(3, int(round((diam_mm / 2.0) / spacing_yx[0])))

        color = confident_color if prob >= fp_threshold else uncertain_color
        cv2.circle(img, (cx, cy), radius_px, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(
            img,
            f"{prob:.2f}",
            (cx + radius_px + 2, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return img


def render_slices(
    scan_output_dir: str,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    saliency_alpha: float = 0.4,
    fp_threshold: float = 0.5,
    window_level: int = -600,
    window_width: int = 1500,
    confident_color: tuple[int, int, int] = (255, 0, 0),
    uncertain_color: tuple[int, int, int] = (255, 165, 0),
) -> list[str]:
    """Render annotated PNG slices for all three views.

    Reads:
      - confidence_map.nii.gz  (as CT proxy if vol not available)
      - saliency_map.nii.gz
      - candidates.csv

    Writes slices to scan_output_dir/slices/<view>_<idx:04d>.png.

    Args:
        scan_output_dir: output directory for a single scan
        spacing_mm: (z, y, x) voxel spacing in mm
        saliency_alpha: saliency overlay opacity

    Returns:
        list of written file paths
    """
    base = Path(scan_output_dir)
    slice_dir = base / "slices"
    slice_dir.mkdir(exist_ok=True)

    conf_map: np.ndarray = nib.load(str(base / "confidence_map.nii.gz")).get_fdata().astype(np.float32)
    sal_map: np.ndarray = nib.load(str(base / "saliency_map.nii.gz")).get_fdata().astype(np.float32)
    cand_df = pd.read_csv(base / "candidates.csv") if (base / "candidates.csv").exists() else pd.DataFrame()

    written: list[str] = []
    view_configs: list[tuple[str, int]] = [("axial", 0), ("coronal", 1), ("sagittal", 2)]

    for view_name, axis in view_configs:
        n_slices = conf_map.shape[axis]
        spacing_yx = (spacing_mm[1], spacing_mm[2]) if axis == 0 else (
            (spacing_mm[0], spacing_mm[2]) if axis == 1 else (spacing_mm[0], spacing_mm[1])
        )

        for idx in range(n_slices):
            sl = [slice(None)] * 3
            sl[axis] = idx

            conf_slice = conf_map[tuple(sl)]
            sal_slice = sal_map[tuple(sl)]

            grey = _apply_lung_window(conf_slice, window_level, window_width)
            rgb = _overlay_saliency(grey, sal_slice, alpha=saliency_alpha)

            # Find candidates on this slice (within ±diameter/2 voxels)
            if not cand_df.empty:
                on_slice = _candidates_on_slice(cand_df, axis, idx, spacing_mm)
                if on_slice:
                    rgb = _draw_candidates(rgb, on_slice, spacing_yx, fp_threshold, confident_color, uncertain_color)

            out_path = str(slice_dir / f"{view_name}_{idx:04d}.png")
            cv2.imwrite(out_path, rgb)
            written.append(out_path)

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
        # coordX/Y/Z are world mm (LPS); convert to voxel
        # Simplified: assume origin=0 (artefacts already resampled to isotropic)
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
