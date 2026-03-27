"""Two-stage inference pipeline: candidate generation → FP reduction.

Stage 1: Slide 256³ patch across the full CT volume in a sliding window.
         Collect all candidate centroids where model probability ≥ 0.5.
Stage 2: Extract 32³ patch around each candidate, run FP classifier.
         Keep candidates where FP classifier P(nodule) ≥ fp_threshold.

Produces:
  - seg_mask.nii.gz       (full-resolution binary segmentation)
  - confidence_map.nii.gz (full-resolution probability map)
  - saliency_map.nii.gz   (full-resolution Grad-CAM / Swin attention)
  - candidates.csv        (seriesuid, coordX, coordY, coordZ, prob, diameter_mm)
  - report.json           (summary)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import (
    extract_patch,
    load_mhd,
    normalise_hu,
    resample_to_isotropic,
)

log = logging.getLogger(__name__)

# Sliding window stride (50% overlap)
_STRIDE_RATIO = 0.5


def _patch_coords(D: int, H: int, W: int, patch_size: int, stride: int):
    """Yield (zs, ys, xs) corner coordinates for a sliding window."""
    for z in range(0, max(D - patch_size + 1, 1), stride):
        for y in range(0, max(H - patch_size + 1, 1), stride):
            for x in range(0, max(W - patch_size + 1, 1), stride):
                ze = min(z + patch_size, D)
                ye = min(y + patch_size, H)
                xe = min(x + patch_size, W)
                yield ze - patch_size, ye - patch_size, xe - patch_size


def _sliding_window_inference(
    vol: np.ndarray,
    model: torch.nn.Module,
    patch_size: int,
    device: str,
    batch_size: int = 2,
) -> np.ndarray:
    """Run sliding-window primary model inference over the full volume.

    Patches are generated lazily in batches to avoid materialising all
    patches in RAM simultaneously.

    Returns probability map (D, H, W) in [0, 1].
    """
    D, H, W = vol.shape
    stride = int(patch_size * _STRIDE_RATIO)

    prob_map = np.zeros_like(vol)
    count_map = np.zeros_like(vol)

    model.eval()
    coord_iter = _patch_coords(D, H, W, patch_size, stride)

    with torch.no_grad():
        exhausted = False
        while not exhausted:
            batch_patches: list[torch.Tensor] = []
            batch_coords: list[tuple[int, int, int]] = []
            for _ in range(batch_size):
                try:
                    zs, ys, xs = next(coord_iter)
                    patch = vol[zs : zs + patch_size, ys : ys + patch_size, xs : xs + patch_size]
                    batch_patches.append(torch.from_numpy(patch).unsqueeze(0))
                    batch_coords.append((zs, ys, xs))
                except StopIteration:
                    exhausted = True
                    break

            if not batch_patches:
                break

            batch = torch.stack(batch_patches).to(device)
            out = model(batch)
            probs = out["seg"].cpu().numpy()  # (B, 1, ps, ps, ps)
            for j, (zs, ys, xs) in enumerate(batch_coords):
                p = probs[j, 0]
                prob_map[zs : zs + patch_size, ys : ys + patch_size, xs : xs + patch_size] += p
                count_map[zs : zs + patch_size, ys : ys + patch_size, xs : xs + patch_size] += 1.0

    count_map = np.maximum(count_map, 1)
    return (prob_map / count_map).astype(np.float32)


def _extract_candidates(
    prob_map: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    threshold: float = 0.5,
    min_voxels: int = 10,
) -> list[dict]:
    """Connected-component analysis on thresholded probability map.

    Returns list of dicts: {coordX, coordY, coordZ, prob, diameter_mm, centre_zyx}
    """
    from skimage.measure import label, regionprops

    binary = (prob_map >= threshold).astype(np.uint8)
    labelled = label(binary)
    candidates = []

    for region in regionprops(labelled, intensity_image=prob_map):
        if region.area < min_voxels:
            continue
        cz, cy, cx = region.centroid
        # World coordinates (LPS)
        world_zyx = origin + np.array([cz, cy, cx]) * spacing
        diameter_mm = float((region.area * np.prod(spacing)) ** (1 / 3) * 2)

        candidates.append(
            {
                "coordX": float(world_zyx[2]),
                "coordY": float(world_zyx[1]),
                "coordZ": float(world_zyx[0]),
                "prob": float(region.mean_intensity),
                "diameter_mm": diameter_mm,
                "centre_zyx": np.array([cz, cy, cx]),
            }
        )

    return sorted(candidates, key=lambda c: c["prob"], reverse=True)


def _candidate_payload(candidate: dict) -> dict:
    """Return frontend-safe candidate payload with per-view slice indices."""
    centre_zyx = candidate.get("centre_zyx")
    payload = {k: v for k, v in candidate.items() if k != "centre_zyx"}

    if centre_zyx is not None:
        voxel_z = int(round(float(centre_zyx[0])))
        voxel_y = int(round(float(centre_zyx[1])))
        voxel_x = int(round(float(centre_zyx[2])))
        payload["slice_indices"] = {
            "axial": voxel_z,
            "coronal": voxel_y,
            "sagittal": voxel_x,
        }
        payload["voxel_z"] = voxel_z
        payload["voxel_y"] = voxel_y
        payload["voxel_x"] = voxel_x
        payload["slice_axial"] = voxel_z
        payload["slice_coronal"] = voxel_y
        payload["slice_sagittal"] = voxel_x

    return payload


class InferencePipeline:
    """Two-stage inference pipeline.

    Args:
        primary_model: primary candidate-generation model (UNet3D or HybridNet)
        fp_model: FPClassifier
        fp_threshold: probability threshold for FP classifier
        candidate_threshold: probability threshold for generated candidates
        min_candidate_voxels: connected-component size floor for candidates
        device: torch device string
        primary_patch_size: sliding-window patch size (256 for inference)
        use_swin: if True, use SwinAttentionExtractor; else GradCAM
    """

    def __init__(
        self,
        primary_model: torch.nn.Module,
        fp_model: torch.nn.Module | None,
        fp_threshold: float = 0.5,
        candidate_threshold: float = 0.5,
        min_candidate_voxels: int = 10,
        device: str = "cpu",
        primary_patch_size: int = 256,
        use_swin: bool = False,
    ):
        self.primary_model = primary_model.to(device).eval()
        self.fp_model = fp_model.to(device).eval() if fp_model is not None else None
        self.fp_threshold = fp_threshold
        self.candidate_threshold = candidate_threshold
        self.min_candidate_voxels = min_candidate_voxels
        self.device = device
        self.primary_patch_size = primary_patch_size

        if use_swin:
            from src.interpretability import SwinAttentionExtractor
            self.saliency_fn = SwinAttentionExtractor(primary_model)
        else:
            from src.interpretability import GradCAM
            self.saliency_fn = GradCAM(primary_model)

    def run(self, mhd_path: str, output_dir: str, seriesuid: str) -> dict:
        """Run full inference on a single CT scan.

        Args:
            mhd_path: path to .mhd file
            output_dir: where to write artefacts
            seriesuid: scan identifier

        Returns:
            report dict (also written to report.json)
        """
        out_path = Path(output_dir) / seriesuid
        out_path.mkdir(parents=True, exist_ok=True)

        log.info(f"Loading {mhd_path}")
        vol, spacing, origin = load_mhd(mhd_path)
        vol, spacing = resample_to_isotropic(vol, spacing)
        vol = normalise_hu(vol)

        # --- Stage 1: candidate generation ---
        log.info("Running sliding-window candidate generation …")
        prob_map = _sliding_window_inference(vol, self.primary_model, self.primary_patch_size, self.device)
        detection_mask = (prob_map >= self.candidate_threshold).astype(np.uint8)

        candidates = _extract_candidates(
            prob_map,
            spacing,
            origin,
            threshold=self.candidate_threshold,
            min_voxels=self.min_candidate_voxels,
        )
        log.info(f"  {len(candidates)} candidates from primary model")

        # --- Saliency on the full volume (patch-by-patch) ---
        log.info("Computing saliency …")
        saliency_map = self._compute_saliency(vol)

        # --- Stage 2: FP reduction ---
        log.info("Running FP reduction …")
        final_candidates = self._fp_filter(vol, candidates)
        log.info(f"  {len(final_candidates)} candidates after FP reduction")

        # --- Save artefacts ---
        affine = np.diag(list(spacing[::-1]) + [1.0])  # approx affine (RAS)
        nib.save(nib.Nifti1Image(detection_mask, affine), str(out_path / "seg_mask.nii.gz"))
        nib.save(nib.Nifti1Image(prob_map, affine), str(out_path / "confidence_map.nii.gz"))
        nib.save(nib.Nifti1Image(saliency_map, affine), str(out_path / "saliency_map.nii.gz"))

        cand_df = pd.DataFrame([_candidate_payload(c) for c in final_candidates])
        cand_df.insert(0, "seriesuid", seriesuid)
        cand_df.to_csv(out_path / "candidates.csv", index=False)

        report = {
            "seriesuid": seriesuid,
            "n_candidates_stage1": len(candidates),
            "n_candidates_final": len(final_candidates),
            "top_candidates": [
                _candidate_payload(c)
                for c in final_candidates[:5]
            ],
        }
        (out_path / "report.json").write_text(json.dumps(report, indent=2))

        log.info(f"Artefacts written to {out_path}")
        return report

    # ------------------------------------------------------------------

    def _fp_filter(self, vol: np.ndarray, candidates: list[dict]) -> list[dict]:
        if self.fp_model is None:
            return candidates

        kept = []
        with torch.no_grad():
            for cand in candidates:
                patch = extract_patch(vol, cand["centre_zyx"].astype(int), patch_size=32)
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
                fp_out = self.fp_model(t)
                prob = float(fp_out["prob"].item())
                if prob >= self.fp_threshold:
                    cand = dict(cand)
                    cand["fp_prob"] = prob
                    kept.append(cand)
        return kept

    def _compute_saliency(self, vol: np.ndarray) -> np.ndarray:
        """Compute per-slice saliency by taking max over patches."""
        D, H, W = vol.shape
        saliency = np.zeros_like(vol)
        patch_size = min(128, D, H, W)
        half = patch_size // 2
        centre = np.array([D // 2, H // 2, W // 2])
        patch = extract_patch(vol, centre, patch_size)
        t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
        sal_patch = self.saliency_fn(t)  # (patch, patch, patch)

        # Place into full volume
        zs = centre[0] - half
        ys = centre[1] - half
        xs = centre[2] - half
        ze, ye, xe = zs + patch_size, ys + patch_size, xs + patch_size
        zs, ys, xs = max(0, zs), max(0, ys), max(0, xs)
        ze, ye, xe = min(D, ze), min(H, ye), min(W, xe)
        saliency[zs:ze, ys:ye, xs:xe] = sal_patch[: ze - zs, : ye - ys, : xe - xs]
        return saliency.astype(np.float32)
