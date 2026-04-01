"""CT preprocessing utilities shared across pipeline stages."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

# HU window used for lung nodule detection
_HU_MIN = -1000.0
_HU_MAX = 400.0


def load_image_volume(image_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a medical image volume via SimpleITK.

    Returns:
        vol:     float32 array (Z, Y, X) in Hounsfield units
        spacing: float32 array (Z, Y, X) voxel spacing in mm
        origin:  float32 array (Z, Y, X) world origin in mm
    """
    image = sitk.ReadImage(str(image_path))
    vol_xyz = sitk.GetArrayFromImage(image).astype(np.float32)  # (Z, Y, X)
    spacing_xyz = np.array(image.GetSpacing(), dtype=np.float32)  # (X, Y, Z)
    origin_xyz = np.array(image.GetOrigin(), dtype=np.float32)   # (X, Y, Z)

    # SimpleITK GetArrayFromImage already returns (Z, Y, X); spacing/origin are (X,Y,Z)
    spacing_zyx = spacing_xyz[::-1].copy()
    origin_zyx = origin_xyz[::-1].copy()
    return vol_xyz, spacing_zyx, origin_zyx


def load_mhd(mhd_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a .mhd/.raw CT volume via SimpleITK."""

    return load_image_volume(mhd_path)


def normalise_hu(vol: np.ndarray) -> np.ndarray:
    """Clip to lung window and rescale to [0, 1].

    Args:
        vol: float32 array in Hounsfield units

    Returns:
        float32 array in [0, 1]
    """
    vol = np.clip(vol, _HU_MIN, _HU_MAX)
    return ((vol - _HU_MIN) / (_HU_MAX - _HU_MIN)).astype(np.float32)


def resample_to_isotropic(
    vol: np.ndarray,
    spacing: np.ndarray,
    target_spacing: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample volume to isotropic voxel spacing.

    Args:
        vol:            float32 (Z, Y, X) volume
        spacing:        (Z, Y, X) current voxel spacing in mm
        target_spacing: desired isotropic spacing; defaults to min(spacing)

    Returns:
        resampled volume and new isotropic spacing array (Z, Y, X)
    """
    if target_spacing is None:
        target_spacing = float(spacing.min())

    zoom_factors = (spacing / target_spacing).astype(np.float64)
    if np.allclose(zoom_factors, 1.0, atol=1e-3):
        return vol, np.full(3, target_spacing, dtype=np.float32)

    resampled = zoom(vol, zoom_factors, order=1, prefilter=False).astype(np.float32)
    new_spacing = np.full(3, target_spacing, dtype=np.float32)
    return resampled, new_spacing


def extract_patch(
    vol: np.ndarray,
    centre_zyx: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """Extract a cubic patch centred at ``centre_zyx``, zero-padding at boundaries.

    Args:
        vol:         float32 (Z, Y, X) volume
        centre_zyx:  integer centre coordinate (Z, Y, X)
        patch_size:  side length of the cubic patch

    Returns:
        float32 array of shape (patch_size, patch_size, patch_size)
    """
    half = patch_size // 2
    D, H, W = vol.shape
    patch = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)

    cz, cy, cx = int(centre_zyx[0]), int(centre_zyx[1]), int(centre_zyx[2])

    # Source slice in the volume
    z0_v = max(0, cz - half)
    y0_v = max(0, cy - half)
    x0_v = max(0, cx - half)
    z1_v = min(D, cz + half + (patch_size % 2))
    y1_v = min(H, cy + half + (patch_size % 2))
    x1_v = min(W, cx + half + (patch_size % 2))

    # Destination slice in the patch
    z0_p = z0_v - (cz - half)
    y0_p = y0_v - (cy - half)
    x0_p = x0_v - (cx - half)
    z1_p = z0_p + (z1_v - z0_v)
    y1_p = y0_p + (y1_v - y0_v)
    x1_p = x0_p + (x1_v - x0_v)

    patch[z0_p:z1_p, y0_p:y1_p, x0_p:x1_p] = vol[z0_v:z1_v, y0_v:y1_v, x0_v:x1_v]
    return patch
