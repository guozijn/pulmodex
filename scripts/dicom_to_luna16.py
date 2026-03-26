"""Convert raw DICOM scans to LUNA16-compatible format.

Stages:
  1. Parse DICOM series → 3D HU volume (pydicom)
  2. Resample to 1 mm isotropic spacing (SimpleITK)
  3. Lung segmentation (threshold + morphological ops via skimage)
  4. Export .mhd / .raw (SimpleITK)
  5. Write candidates.csv and annotations.csv stubs

Usage:
    python scripts/dicom_to_luna16.py \
        --input_dir data/raw_dicom \
        --output_dir data/processed

Input layout:
    data/raw_dicom/
        <PatientID>/
            *.dcm

Output layout (LUNA16-compatible):
    data/processed/
        subset0/          ← all converted scans go into subset0
            <seriesuid>.mhd
            <seriesuid>.raw
        candidates.csv    (seriesuid, coordX, coordY, coordZ, class)
        annotations.csv   (seriesuid, coordX, coordY, coordZ, diameter_mm)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure, morphology

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DICOM loading
# ---------------------------------------------------------------------------

def load_dicom_series(series_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a DICOM series directory, returning (volume_HU, spacing_zyx, origin_zyx).

    Slices are sorted by ImagePositionPatient[2] (z-position).
    """
    dcm_files = sorted(
        [f for f in Path(series_dir).glob("*.dcm")],
        key=lambda f: float(
            pydicom.dcmread(
                str(f),
                stop_before_pixels=True,
            ).ImagePositionPatient[2]
        ),
    )
    if not dcm_files:
        raise ValueError(f"No .dcm files in {series_dir}")

    slices = [pydicom.dcmread(str(f)) for f in dcm_files]
    ref = slices[0]

    # Pixel spacing (row, col) + slice thickness
    pixel_spacing = list(map(float, ref.PixelSpacing))
    slice_thickness = float(getattr(ref, "SliceThickness", 1.0))
    spacing_zyx = np.array([slice_thickness, pixel_spacing[0], pixel_spacing[1]])

    origin_xyz = list(map(float, ref.ImagePositionPatient))
    origin_zyx = np.array(origin_xyz[::-1])

    # Build volume
    vol = np.stack([
        s.pixel_array.astype(np.float32) * float(s.RescaleSlope) + float(s.RescaleIntercept)
        for s in slices
    ]).astype(np.float32)  # (D, H, W) HU

    return vol, spacing_zyx, origin_zyx


# ---------------------------------------------------------------------------
# Lung segmentation
# ---------------------------------------------------------------------------

def segment_lung_mask(vol_hu: np.ndarray) -> np.ndarray:
    """Simple threshold-based lung segmentation.

    Returns binary mask (D, H, W) with 1 = lung, 0 = background.
    Air is ~−1000 HU; soft tissue ~40 HU; bone ~400+ HU.
    """
    # Threshold: lung air + parenchyma
    binary = vol_hu < -400
    binary = binary.astype(np.uint8)

    # Clear border artifacts slice by slice
    for i in range(binary.shape[0]):
        binary[i] = morphology.clear_border(binary[i])

    # Label connected components; keep the two largest (left + right lung)
    labelled = measure.label(binary)
    props = sorted(measure.regionprops(labelled), key=lambda r: r.area, reverse=True)
    mask = np.zeros_like(binary)
    for prop in props[:2]:
        mask[labelled == prop.label] = 1

    # Fill holes and dilate slightly
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    mask = morphology.binary_dilation(mask, morphology.ball(3)).astype(np.uint8)

    return mask


# ---------------------------------------------------------------------------
# MHD export
# ---------------------------------------------------------------------------

def export_mhd(
    vol: np.ndarray,
    spacing_zyx: np.ndarray,
    origin_zyx: np.ndarray,
    out_path: str,
) -> None:
    """Write (D, H, W) float32 volume to .mhd/.raw."""
    sitk_img = sitk.GetImageFromArray(vol)
    sitk_img.SetSpacing(spacing_zyx[::-1].tolist())  # (x, y, z)
    sitk_img.SetOrigin(origin_zyx[::-1].tolist())
    sitk.WriteImage(sitk_img, out_path, useCompression=False)
    log.info(f"  Written {out_path}")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def write_csv_row(path: str, row: list, header: list | None = None) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists and header:
            writer.writerow(header)
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_patient(
    patient_dir: str,
    output_dir: str,
    subset: str = "subset0",
) -> str | None:
    """Process one patient directory.

    Returns seriesuid on success, None on failure.
    """
    seriesuid = Path(patient_dir).name
    out_subset = Path(output_dir) / subset
    out_subset.mkdir(parents=True, exist_ok=True)
    out_mhd = str(out_subset / f"{seriesuid}.mhd")

    if os.path.exists(out_mhd):
        log.info(f"  Already exists, skipping: {seriesuid}")
        return seriesuid

    log.info(f"Processing {seriesuid} …")

    try:
        vol, spacing, origin = load_dicom_series(patient_dir)
        log.info(f"  Loaded {vol.shape} HU, spacing={spacing}")

        # Resample to 1 mm isotropic
        from src.data.preprocessing import resample_to_isotropic
        vol_iso, spacing_iso = resample_to_isotropic(vol, spacing, target_spacing=1.0)
        log.info(f"  Resampled to {vol_iso.shape}")

        # Lung mask (applied to zero out background, optional)
        lung_mask = segment_lung_mask(vol_iso)
        vol_masked = vol_iso * lung_mask  # keep HU in lung, zero outside

        export_mhd(vol_masked, spacing_iso, origin, out_mhd)

        # Stub candidate row (class=0 placeholder; real labels added manually)
        cands_path = str(Path(output_dir) / "candidates.csv")
        ann_path = str(Path(output_dir) / "annotations.csv")

        cx, cy, cz = (np.array(vol_masked.shape) // 2 * spacing_iso[::-1] + origin[::-1]).tolist()
        write_csv_row(
            cands_path,
            [seriesuid, cx, cy, cz, 0],
            header=["seriesuid", "coordX", "coordY", "coordZ", "class"],
        )
        write_csv_row(
            ann_path,
            [seriesuid, cx, cy, cz, 0.0],
            header=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"],
        )

    except Exception as e:
        log.error(f"  Failed: {e}")
        return None

    return seriesuid


def main() -> None:
    parser = argparse.ArgumentParser(description="DICOM → LUNA16 converter")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Root dir with per-patient DICOM subdirs",
    )
    parser.add_argument("--output_dir", required=True, help="Output root (LUNA16-compatible)")
    parser.add_argument("--subset", default="subset0", help="Output subset folder name")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    patient_dirs = [d for d in Path(args.input_dir).iterdir() if d.is_dir()]
    log.info(f"Found {len(patient_dirs)} patient directories")

    success, fail = 0, 0
    for d in patient_dirs:
        uid = process_patient(str(d), args.output_dir, args.subset)
        if uid:
            success += 1
        else:
            fail += 1

    log.info(f"Done. {success} converted, {fail} failed.")
    log.info("NOTE: Edit candidates.csv and annotations.csv to add ground-truth labels.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
