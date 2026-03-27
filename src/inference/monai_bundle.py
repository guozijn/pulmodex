"""MONAI bundle-backed detection pipeline adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.bundle import ConfigParser

from src.data.preprocessing import extract_patch

from .pipeline import _candidate_payload


def is_monai_bundle_path(path: str | Path) -> bool:
    """Return True when path looks like a MONAI bundle directory."""
    bundle_dir = Path(path)
    return bundle_dir.is_dir() and (bundle_dir / "configs" / "inference.json").exists()


class MONAIBundleDetectionPipeline:
    """Run MONAI bundle detection, then apply the local FP classifier."""

    def __init__(
        self,
        bundle_dir: str | Path,
        fp_model: torch.nn.Module | None,
        fp_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()
        self.fp_model = fp_model.to(device).eval() if fp_model is not None else None
        self.fp_threshold = fp_threshold
        self.device = torch.device(device)

        sys.path.insert(0, str(self.bundle_dir))
        sys.path.insert(0, str(self.bundle_dir / "scripts"))

        parser = ConfigParser()
        parser.read_config(str(self.bundle_dir / "configs" / "inference.json"))
        parser["bundle_root"] = str(self.bundle_dir)
        parser["device"] = self.device
        parser["amp"] = False
        # We feed the bundle a temporary .mhd volume converted from the uploaded
        # DICOM series, so force the LUNA16/raw branch that uses ITK image I/O.
        parser["whether_raw_luna16"] = True
        parser["whether_resampled_luna16"] = False

        self.preprocessing = parser.get_parsed_content("preprocessing")
        self.network = parser.get_parsed_content("network")
        self.detector = parser.get_parsed_content("detector")
        self.postprocessing = parser.get_parsed_content("postprocessing")
        self.inferer = parser.get_parsed_content("inferer")
        _ = parser.get_parsed_content("detector_ops")

        state_dict = torch.load(
            self.bundle_dir / "models" / "model.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.network.load_state_dict(state_dict)
        self.network.to(self.device).eval()
        self.detector.network = self.network
        self.detector.eval()

    def run(self, mhd_path: str, output_dir: str, seriesuid: str) -> dict[str, Any]:
        out_path = Path(output_dir) / seriesuid
        out_path.mkdir(parents=True, exist_ok=True)

        item = self.preprocessing({"image": mhd_path})
        vol_zyx, affine_ras_xyz, spacing_zyx = self._preprocessed_volume(item)

        detected_candidates = self._detect_candidates(item, affine_ras_xyz)
        final_candidates = self._fp_filter(vol_zyx, detected_candidates)

        seg_mask, confidence_map = self._build_detection_maps(
            vol_shape=vol_zyx.shape,
            candidates=final_candidates,
            spacing=spacing_zyx,
        )
        saliency_map = np.zeros_like(vol_zyx, dtype=np.float32)

        affine = np.diag(list(spacing_zyx[::-1]) + [1.0])
        nib.save(nib.Nifti1Image(vol_zyx.astype(np.float32), affine), str(out_path / "ct_volume.nii.gz"))
        nib.save(nib.Nifti1Image(seg_mask, affine), str(out_path / "seg_mask.nii.gz"))
        nib.save(nib.Nifti1Image(confidence_map, affine), str(out_path / "confidence_map.nii.gz"))
        nib.save(nib.Nifti1Image(saliency_map, affine), str(out_path / "saliency_map.nii.gz"))

        cand_df = pd.DataFrame([_candidate_payload(c) for c in final_candidates])
        cand_df.insert(0, "seriesuid", seriesuid)
        cand_df.to_csv(out_path / "candidates.csv", index=False)

        report = {
            "seriesuid": seriesuid,
            "n_candidates_stage1": len(detected_candidates),
            "n_candidates_final": len(final_candidates),
            "top_candidates": [_candidate_payload(c) for c in final_candidates[:5]],
        }
        (out_path / "report.json").write_text(json.dumps(report, indent=2))
        return report

    def _preprocessed_volume(
        self,
        item: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = item["image"]
        vol_xyz = image.detach().cpu().numpy()[0].astype(np.float32)
        affine_ras_xyz = np.asarray(image.affine.detach().cpu().numpy(), dtype=np.float64)
        spacing_xyz = np.linalg.norm(affine_ras_xyz[:3, :3], axis=0).astype(np.float32)
        vol_zyx = np.transpose(vol_xyz, (2, 1, 0))
        return vol_zyx, affine_ras_xyz, spacing_xyz[::-1]

    def _detect_candidates(
        self,
        item: dict[str, Any],
        affine_ras_xyz: np.ndarray,
    ) -> list[dict[str, Any]]:
        image = item["image"].to(self.device)

        self.network.eval()
        self.detector.eval()
        self.detector.training = False
        if hasattr(self.inferer, "detector"):
            self.inferer.detector.eval()
            self.inferer.detector.training = False

        with torch.inference_mode():
            prediction = self.inferer(inputs=[image], network=self.network, targets=None)[0]

        # Grab raw voxel-space boxes BEFORE postprocessing converts them to world
        # coordinates.  The postprocessing chain (ClipBoxToImaged →
        # AffineBoxToWorldCoordinated → ConvertBoxModed) does not reorder boxes,
        # so raw_boxes[i] corresponds to processed boxes[i].
        raw_boxes_np = (
            prediction["box"].detach().cpu().numpy()
            if hasattr(prediction.get("box"), "detach")
            else np.asarray(prediction.get("box", []))
        )  # shape (N, 6), xyzxyz voxel format [x1,y1,z1,x2,y2,z2]

        processed = self.postprocessing({**prediction, "image": item["image"]})
        boxes = processed.get("box")
        scores = processed.get("label_scores")

        if boxes is None or scores is None:
            return []

        boxes = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
        scores = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)

        candidates: list[dict[str, Any]] = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            coord_x, coord_y, coord_z, width_mm, height_mm, depth_mm = [float(v) for v in box.tolist()]

            # Derive voxel-space centre directly from the raw (pre-postprocessing)
            # box to avoid the double LPS↔RAS flip that corrupts the y coordinate.
            if i < len(raw_boxes_np):
                rb = raw_boxes_np[i]  # [x1,y1,z1,x2,y2,z2] in voxel (X,Y,Z)
                vox_x = (float(rb[0]) + float(rb[3])) / 2.0
                vox_y = (float(rb[1]) + float(rb[4])) / 2.0
                vox_z = (float(rb[2]) + float(rb[5])) / 2.0
            else:
                # Fallback: invert the affine (may be inaccurate for y axis)
                voxel_xyz = nib.affines.apply_affine(
                    np.linalg.inv(affine_ras_xyz),
                    np.array([coord_x, coord_y, coord_z], dtype=np.float32),
                )
                vox_x, vox_y, vox_z = float(voxel_xyz[0]), float(voxel_xyz[1]), float(voxel_xyz[2])

            centre_zyx = np.array([vox_z, vox_y, vox_x], dtype=np.float32)
            candidates.append(
                {
                    "coordX": coord_x,
                    "coordY": coord_y,
                    "coordZ": coord_z,
                    "prob": float(score),
                    "diameter_mm": float(max(width_mm, height_mm, depth_mm)),
                    "bbox_mm": [width_mm, height_mm, depth_mm],
                    "centre_zyx": centre_zyx,
                }
            )

        return sorted(candidates, key=lambda c: c["prob"], reverse=True)

    def _fp_filter(self, vol: np.ndarray, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.fp_model is None:
            return candidates

        kept: list[dict[str, Any]] = []
        with torch.no_grad():
            for cand in candidates:
                patch = extract_patch(vol, cand["centre_zyx"].astype(int), patch_size=32)
                tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
                fp_out = self.fp_model(tensor)
                prob = float(fp_out["prob"].item())
                if prob >= self.fp_threshold:
                    item = dict(cand)
                    item["fp_prob"] = prob
                    kept.append(item)
        return kept

    def _build_detection_maps(
        self,
        vol_shape: tuple[int, int, int],
        candidates: list[dict[str, Any]],
        spacing: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        seg_mask = np.zeros(vol_shape, dtype=np.uint8)
        confidence_map = np.zeros(vol_shape, dtype=np.float32)

        for cand in candidates:
            centre = cand["centre_zyx"].astype(int)
            radius_vox = max(2, int(round((cand["diameter_mm"] / 2.0) / float(spacing.min()))))

            z0 = max(0, centre[0] - radius_vox)
            z1 = min(vol_shape[0], centre[0] + radius_vox + 1)
            y0 = max(0, centre[1] - radius_vox)
            y1 = min(vol_shape[1], centre[1] + radius_vox + 1)
            x0 = max(0, centre[2] - radius_vox)
            x1 = min(vol_shape[2], centre[2] + radius_vox + 1)

            zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
            sphere = (
                (zz - centre[0]) ** 2
                + (yy - centre[1]) ** 2
                + (xx - centre[2]) ** 2
                <= radius_vox**2
            )
            seg_mask[z0:z1, y0:y1, x0:x1][sphere] = 1
            current = confidence_map[z0:z1, y0:y1, x0:x1]
            current[sphere] = np.maximum(
                current[sphere],
                float(cand.get("fp_prob", cand.get("prob", 0.0))),
            )

        return seg_mask, confidence_map
