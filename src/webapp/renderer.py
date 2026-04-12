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


def _saliency_rgba(
    saliency_slice: np.ndarray,
    alpha: float = 0.4,  # kept for signature compatibility, no longer baked in
) -> np.ndarray:
    """Return an RGBA saliency overlay for a slice.

    Colour is a warm yellow-orange gradient tuned for CT readability.
    Per-pixel alpha follows saliency intensity so strong responses appear
    denser while weak responses remain translucent. The frontend CSS opacity
    slider still controls the final global blend level across the full
    0–100 % range.
    """
    sal = np.clip(saliency_slice.astype(np.float32), 0.0, 1.0)
    visible = sal >= 0.05
    sal_gamma = np.power(sal, 0.85)

    rgba = np.zeros((*sal.shape, 4), dtype=np.uint8)
    # BGRA channels: warm amber -> yellow without rainbow artefacts.
    rgba[..., 1] = np.where(visible, (140.0 + 90.0 * sal_gamma).astype(np.uint8), 0)
    rgba[..., 2] = np.where(visible, (220.0 + 35.0 * sal_gamma).astype(np.uint8), 0)
    rgba[..., 3] = np.where(visible, np.clip(sal_gamma * 255.0, 0, 255).astype(np.uint8), 0)
    return rgba


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
        # Score label with dark shadow for readability
        label = f"{prob:.2f}"
        lx, ly = cx + half_w_px + 4, cy + 4
        cv2.putText(img, label, (lx + 1, ly + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (*color, 255), 1, cv2.LINE_AA)
    return img


def _composite_base_and_overlay(base_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
    """Composite a transparent BGRA overlay onto a BGR base image."""
    out = base_bgr.astype(np.float32).copy()
    alpha = (overlay_bgra[..., 3:4].astype(np.float32)) / 255.0
    overlay_rgb = overlay_bgra[..., :3].astype(np.float32)
    out = overlay_rgb * alpha + out * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


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

    Writes raw, base, overlay, and composite PNG slices under scan_output_dir/slices/.

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

    sal_map: np.ndarray = (
        nib.load(str(base / "saliency_map.nii.gz")).get_fdata().astype(np.float32)
    )
    # When saliency is all zeros (e.g. MONAI bundle pipeline), fall back to the
    # confidence map so the heatmap overlay actually shows detection signal.
    if sal_map.max() < 1e-6:
        conf_map_path = base / "confidence_map.nii.gz"
        if conf_map_path.exists():
            sal_map = nib.load(str(conf_map_path)).get_fdata().astype(np.float32)
            if sal_map.max() > 0:
                sal_map = sal_map / sal_map.max()
    cand_df = (
        pd.read_csv(base / "candidates.csv")
        if (base / "candidates.csv").exists()
        else pd.DataFrame()
    )

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
            raw_bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
            base_bgr = raw_bgr.copy()
            overlay_bgra = _saliency_rgba(sal_slice, alpha=saliency_alpha)

            # Draw nodule boxes directly onto the base image so they are
            # always visible at full opacity regardless of the overlay toggle.
            if not cand_df.empty:
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

            composite_bgr = _composite_base_and_overlay(base_bgr, overlay_bgra)

            raw_path = str(slice_dir / f"raw_{view_name}_{idx:04d}.png")
            base_path = str(slice_dir / f"base_{view_name}_{idx:04d}.png")
            overlay_path = str(slice_dir / f"overlay_{view_name}_{idx:04d}.png")
            composite_path = str(slice_dir / f"{view_name}_{idx:04d}.png")
            cv2.imwrite(raw_path, raw_bgr)
            cv2.imwrite(base_path, base_bgr)
            cv2.imwrite(overlay_path, overlay_bgra)
            cv2.imwrite(composite_path, composite_bgr)
            written.extend([raw_path, base_path, overlay_path, composite_path])

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
