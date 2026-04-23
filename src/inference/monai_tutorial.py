"""Adapter for standalone MONAI tutorial RetinaNet TorchScript models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToWorldCoordinated,
    ClipBoxToImaged,
    ConvertBoxModed,
)
from monai.apps.detection.transforms.array import ClipBoxToImage
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

from src.data.preprocessing import extract_patch

from .pipeline import _candidate_payload

_DEFAULT_RETURNED_LAYERS = [1, 2]
_DEFAULT_BASE_ANCHOR_SHAPES = [[6, 8, 4], [8, 6, 5], [10, 10, 6]]
_DEFAULT_VAL_PATCH_SIZE = [512, 512, 208]
_DEFAULT_SPACING = [0.703125, 0.703125, 1.25]
_DEFAULT_SCORE_THRESH = 0.02
_DEFAULT_NMS_THRESH = 0.22
_DEFAULT_GT_BOX_MODE = "cccwhd"


def is_monai_tutorial_model_path(path: str | Path) -> bool:
    """Return True when path looks like a standalone tutorial TorchScript model."""
    candidate = Path(path)
    return candidate.is_file() and candidate.suffix == ".pt" and candidate.parent.name != "models"


class MONAITutorialDetectionPipeline:
    """Run MONAI tutorial RetinaNet TorchScript inference, then local FP reduction."""

    def __init__(
        self,
        model_path: str | Path,
        fp_model: torch.nn.Module | None,
        fp_threshold: float = 0.5,
        device: str = "cpu",
        returned_layers: list[int] | None = None,
        base_anchor_shapes: list[list[int]] | None = None,
        val_patch_size: list[int] | None = None,
        spacing: list[float] | None = None,
        score_thresh: float = _DEFAULT_SCORE_THRESH,
        nms_thresh: float = _DEFAULT_NMS_THRESH,
        gt_box_mode: str = _DEFAULT_GT_BOX_MODE,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.fp_model = fp_model.to(device).eval() if fp_model is not None else None
        self.fp_threshold = fp_threshold
        self.device = torch.device(device)
        self.returned_layers = returned_layers or list(_DEFAULT_RETURNED_LAYERS)
        self.base_anchor_shapes = base_anchor_shapes or [list(shape) for shape in _DEFAULT_BASE_ANCHOR_SHAPES]
        self.val_patch_size = val_patch_size or list(_DEFAULT_VAL_PATCH_SIZE)
        self.spacing = spacing or list(_DEFAULT_SPACING)
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.gt_box_mode = gt_box_mode
        self.amp = self.device.type == "cuda"
        self.raw_box_clipper = ClipBoxToImage(remove_empty=True)

        self.preprocessing, self.postprocessing = self._build_transforms()
        self.detector = self._build_detector()

    def _build_transforms(self) -> tuple[Compose, Compose]:
        compute_dtype = torch.float16 if self.amp else torch.float32
        intensity_transform = ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024.0,
            a_max=300.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )

        preprocessing = Compose(
            [
                LoadImaged(
                    keys=["image"],
                    image_only=False,
                    meta_key_postfix="meta_dict",
                    reader="itkreader",
                    affine_lps_to_ras=True,
                ),
                EnsureChannelFirstd(keys=["image"]),
                EnsureTyped(keys=["image"], dtype=torch.float32),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=self.spacing, padding_mode="border"),
                intensity_transform,
                EnsureTyped(keys=["image"], dtype=compute_dtype),
            ]
        )
        postprocessing = Compose(
            [
                ClipBoxToImaged(
                    box_keys=["pred_box"],
                    label_keys=["pred_label", "pred_score"],
                    box_ref_image_keys="image",
                    remove_empty=True,
                ),
                AffineBoxToWorldCoordinated(
                    box_keys=["pred_box"],
                    box_ref_image_keys="image",
                    image_meta_key_postfix="meta_dict",
                    affine_lps_to_ras=False,
                ),
                ConvertBoxModed(box_keys=["pred_box"], src_mode="xyzxyz", dst_mode=self.gt_box_mode),
                DeleteItemsd(keys=["image"]),
            ]
        )
        return preprocessing, postprocessing

    def _build_detector(self) -> RetinaNetDetector:
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=[2**level for level in range(len(self.returned_layers) + 1)],
            base_anchor_shapes=self.base_anchor_shapes,
        )

        network = torch.jit.load(str(self.model_path), map_location=self.device).to(self.device)
        detector = RetinaNetDetector(network=network, anchor_generator=anchor_generator, debug=False)
        detector.set_box_selector_parameters(
            score_thresh=self.score_thresh,
            topk_candidates_per_level=1000,
            nms_thresh=self.nms_thresh,
            detections_per_img=100,
        )
        detector.set_sliding_window_inferer(
            roi_size=self.val_patch_size,
            overlap=0.25,
            sw_batch_size=1,
            mode="gaussian",
            device="cpu",
        )
        detector.eval()
        return detector

    def run(self, scan_path: str, output_dir: str, seriesuid: str) -> dict[str, Any]:
        out_path = Path(output_dir) / seriesuid
        out_path.mkdir(parents=True, exist_ok=True)

        item = self.preprocessing({"image": scan_path})
        vol_zyx, spacing_zyx = self._preprocessed_volume(item)

        detected_candidates = self._detect_candidates(item)
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
            "coordinate_system": "RAS",
            "n_candidates_stage1": len(detected_candidates),
            "n_candidates_final": len(final_candidates),
            "candidates": [_candidate_payload(c) for c in final_candidates],
        }
        (out_path / "report.json").write_text(json.dumps(report, indent=2))
        return report

    def _preprocessed_volume(self, item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        image = item["image"]
        vol_xyz = image.detach().cpu().numpy()[0].astype(np.float32)
        affine_ras_xyz = np.asarray(image.affine.detach().cpu().numpy(), dtype=np.float64)
        spacing_xyz = np.linalg.norm(affine_ras_xyz[:3, :3], axis=0).astype(np.float32)
        vol_zyx = np.transpose(vol_xyz, (2, 1, 0))
        return vol_zyx, spacing_xyz[::-1]

    def _detect_candidates(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        image_for_detection = item["image"].as_tensor() if hasattr(item["image"], "as_tensor") else item["image"]
        image = image_for_detection.to(self.device)
        use_inferer = bool(image[0, ...].numel() >= int(np.prod(self.val_patch_size)))

        with torch.inference_mode():
            if self.amp:
                with torch.autocast("cuda"):
                    prediction = self.detector([image], use_inferer=use_inferer)[0]
            else:
                prediction = self.detector([image], use_inferer=use_inferer)[0]

        raw_boxes = (
            prediction[self.detector.target_box_key].detach().cpu().numpy()
            if hasattr(prediction[self.detector.target_box_key], "detach")
            else np.asarray(prediction[self.detector.target_box_key])
        )
        raw_scores = (
            prediction[self.detector.pred_score_key].detach().cpu().numpy()
            if hasattr(prediction[self.detector.pred_score_key], "detach")
            else np.asarray(prediction[self.detector.pred_score_key])
        )
        raw_labels = (
            prediction[self.detector.target_label_key].detach().cpu().numpy()
            if hasattr(prediction[self.detector.target_label_key], "detach")
            else np.asarray(prediction[self.detector.target_label_key])
        )
        aligned_raw_boxes = self._clip_raw_boxes_to_image(item, raw_boxes)

        processed = self.postprocessing(
            {
                **item,
                "pred_box": prediction[self.detector.target_box_key].to(torch.float32),
                "pred_label": prediction[self.detector.target_label_key],
                "pred_score": prediction[self.detector.pred_score_key].to(torch.float32),
            }
        )
        boxes = processed.get("pred_box")
        scores = processed.get("pred_score")
        labels = processed.get("pred_label")

        if boxes is None or scores is None or labels is None:
            return []

        boxes = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
        scores = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
        labels = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)

        candidates: list[dict[str, Any]] = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            coord_x, coord_y, coord_z, width_mm, height_mm, depth_mm = [float(v) for v in box.tolist()]
            if i < len(aligned_raw_boxes):
                rb = aligned_raw_boxes[i]
                vox_x = (float(rb[0]) + float(rb[3])) / 2.0
                vox_y = (float(rb[1]) + float(rb[4])) / 2.0
                vox_z = (float(rb[2]) + float(rb[5])) / 2.0
            else:
                vox_x = coord_x
                vox_y = coord_y
                vox_z = coord_z

            candidates.append(
                {
                    "coordX": coord_x,
                    "coordY": coord_y,
                    "coordZ": coord_z,
                    "prob": float(score),
                    "diameter_mm": float(max(width_mm, height_mm, depth_mm)),
                    "bbox_mm": [width_mm, height_mm, depth_mm],
                    "centre_zyx": np.array([vox_z, vox_y, vox_x], dtype=np.float32),
                }
            )

        return sorted(candidates, key=lambda c: c["prob"], reverse=True)

    def _clip_raw_boxes_to_image(self, item: dict[str, Any], raw_boxes: np.ndarray) -> np.ndarray:
        raw_boxes = np.asarray(raw_boxes)
        if raw_boxes.size == 0:
            return raw_boxes.reshape(0, 6)
        if raw_boxes.ndim == 1:
            raw_boxes = raw_boxes.reshape(1, -1)

        image = item["image"]
        spatial_size = image.shape[1:]
        dummy_labels = np.zeros(raw_boxes.shape[0], dtype=np.int64)
        clipped_boxes, _ = self.raw_box_clipper(raw_boxes, dummy_labels, spatial_size)
        if hasattr(clipped_boxes, "detach"):
            return clipped_boxes.detach().cpu().numpy()
        return np.asarray(clipped_boxes)

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
            ) <= radius_vox**2

            seg_mask[z0:z1, y0:y1, x0:x1][sphere] = 1
            confidence_map[z0:z1, y0:y1, x0:x1][sphere] = np.maximum(
                confidence_map[z0:z1, y0:y1, x0:x1][sphere],
                float(cand["prob"]),
            )

        return seg_mask, confidence_map
