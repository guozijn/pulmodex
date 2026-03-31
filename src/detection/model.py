"""MONAI RetinaNet construction and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from monai.apps.detection.networks.retinanet_detector import retinanet_resnet50_fpn_detector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape


def build_detection_detector(
    patch_size: tuple[int, int, int] = (96, 96, 96),
    score_thresh: float = 0.15,
    nms_thresh: float = 0.1,
    detections_per_img: int = 150,
    pretrained_backbone: bool = False,
) -> Any:
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=(1, 2, 4, 8),
        base_anchor_shapes=(
            (6, 6, 6),
            (10, 10, 10),
            (16, 16, 16),
            (24, 24, 24),
        ),
    )
    detector = retinanet_resnet50_fpn_detector(
        num_classes=1,
        anchor_generator=anchor_generator,
        spatial_dims=3,
        n_input_channels=1,
        conv1_t_size=7,
        conv1_t_stride=(2, 2, 2),
        pretrained=pretrained_backbone,
    )
    detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
    detector.set_hard_negative_sampler(
        batch_size_per_image=64,
        positive_fraction=0.5,
        min_neg=16,
        pool_size=4.0,
    )
    detector.set_box_selector_parameters(
        score_thresh=score_thresh,
        topk_candidates_per_level=300,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
        apply_sigmoid=True,
    )
    detector.set_sliding_window_inferer(roi_size=patch_size, sw_batch_size=1, overlap=0.25)
    return detector


def save_detection_checkpoint(
    path: str | Path,
    detector: Any,
    config: dict[str, Any],
    epoch: int,
    best_metric: float,
) -> None:
    payload = {
        "model_type": "monai_detection",
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "config": config,
        "model_state_dict": detector.network.state_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_detection_checkpoint(path: str | Path, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("config", {})
    detector = build_detection_detector(
        patch_size=tuple(config.get("patch_size", (96, 96, 96))),
        score_thresh=float(config.get("score_thresh", 0.15)),
        nms_thresh=float(config.get("nms_thresh", 0.1)),
        detections_per_img=int(config.get("detections_per_img", 150)),
        pretrained_backbone=False,
    )
    detector.network.load_state_dict(payload["model_state_dict"])
    detector.network.to(device)
    detector.to(device)
    detector.eval()
    return detector, payload
