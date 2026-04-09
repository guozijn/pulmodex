"""Visualize a LUNA16 nodule with a cubic bounding box on axial/coronal/sagittal slices."""

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Configuration -----------------------------------------------------------
SERIES_UID = "1.3.6.1.4.1.14519.5.2.1.6279.6001.511347030803753100045216493273"
COORD_X = 60.775061    # LPS mm
COORD_Y = 74.123970
COORD_Z = -214.782347
DIAMETER_MM = 25.233202
SUBSET_DIR = "data/subset0"
OUTPUT_PATH = "outputs/nodule_bbox_viz.png"
# -----------------------------------------------------------------------------

import os, pathlib
pathlib.Path("outputs").mkdir(exist_ok=True)

mhd_path = os.path.join(SUBSET_DIR, SERIES_UID + ".mhd")
print(f"Loading {mhd_path}")
image = sitk.ReadImage(mhd_path)

# World (LPS mm) -> voxel index
# SimpleITK TransformPhysicalPointToIndex expects the same coordinate system as the image origin/direction.
world_point = [COORD_X, COORD_Y, COORD_Z]
voxel_idx = image.TransformPhysicalPointToIndex(world_point)
cx, cy, cz = voxel_idx           # col, row, slice  (x=col, y=row, z=slice in SimpleITK)
print(f"Nodule centre voxel (col, row, slice): ({cx}, {cy}, {cz})")

# Pixel spacing: (x_spacing, y_spacing, z_spacing) in mm/voxel
spacing = image.GetSpacing()     # (sx, sy, sz)
print(f"Spacing (mm): {spacing}")

# Half-side in voxels for each axis
half_x = (DIAMETER_MM / 2.0) / spacing[0]
half_y = (DIAMETER_MM / 2.0) / spacing[1]
half_z = (DIAMETER_MM / 2.0) / spacing[2]

# Convert SimpleITK image to numpy array: shape = (z, y, x)
volume = sitk.GetArrayFromImage(image)   # (slices, rows, cols)
print(f"Volume shape (z, y, x): {volume.shape}")

# Lung window HU -> display range
HU_MIN, HU_MAX = -1000, 400

def to_display(arr):
    return np.clip(arr, HU_MIN, HU_MAX)

# ---- Helper: draw rectangle on axes ----------------------------------------
def draw_rect(ax, center_h, center_v, half_h, half_v, color="lime", lw=1.5):
    """Draw a rectangle (2D projection of the bounding box face) on matplotlib axes."""
    left   = center_h - half_h
    bottom = center_v - half_v
    w = 2 * half_h
    h = 2 * half_v
    rect = patches.Rectangle(
        (left, bottom), w, h,
        linewidth=lw, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)

# ---- Three orthogonal views -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor("black")

# --- Axial (Z fixed = cz, axes: col=x, row=y) ---
ax = axes[0]
slice_data = to_display(volume[cz, :, :])
ax.imshow(slice_data, cmap="gray", vmin=HU_MIN, vmax=HU_MAX, origin="upper")
draw_rect(ax, cx, cy, half_x, half_y)
ax.set_title(f"Axial  z={cz}", color="white")
ax.axhline(cy, color="red", lw=0.5, alpha=0.5)
ax.axvline(cx, color="red", lw=0.5, alpha=0.5)

# --- Coronal (Y fixed = cy, axes: col=x, row=z) ---
ax = axes[1]
# volume[:,cy,:] has shape (z, x) but we want z increasing downward naturally
slice_data = to_display(volume[:, cy, :])
ax.imshow(slice_data, cmap="gray", vmin=HU_MIN, vmax=HU_MAX, origin="upper")
draw_rect(ax, cx, cz, half_x, half_z)
ax.set_title(f"Coronal  y={cy}", color="white")
ax.axhline(cz, color="red", lw=0.5, alpha=0.5)
ax.axvline(cx, color="red", lw=0.5, alpha=0.5)

# --- Sagittal (X fixed = cx, axes: col=y, row=z) ---
ax = axes[2]
slice_data = to_display(volume[:, :, cx])
ax.imshow(slice_data, cmap="gray", vmin=HU_MIN, vmax=HU_MAX, origin="upper")
draw_rect(ax, cy, cz, half_y, half_z)
ax.set_title(f"Sagittal  x={cx}", color="white")
ax.axhline(cz, color="red", lw=0.5, alpha=0.5)
ax.axvline(cy, color="red", lw=0.5, alpha=0.5)

for ax in axes:
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

plt.suptitle(
    f"LUNA16 nodule bbox  |  diameter={DIAMETER_MM:.1f} mm  |  {SERIES_UID[-12:]}",
    color="white", fontsize=11
)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="black")
print(f"Saved -> {OUTPUT_PATH}")
