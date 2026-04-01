"""Dataset utilities for MONAI 3D detection training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.apps.detection.transforms.box_ops import convert_box_to_mask
from monai.apps.detection.transforms.dictionary import (
    BoxToMaskd,
    ClipBoxToImaged,
    MaskToBoxd,
    StandardizeEmptyBoxd,
)
from monai.data import Dataset as MonaiDataset
from monai.data.box_utils import clip_boxes_to_image
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureTyped,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
)
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd, ConvertPointsToBoxesd
from monai.transforms.transform import MapTransform
from monai.transforms.utility.dictionary import ApplyTransformToPointsd

from src.data.preprocessing import normalise_hu, resample_to_isotropic

from .io import load_detection_case


def load_preprocessed_detection_case(
    image_path: str | Path,
    boxes_world: list[list[float]] | None = None,
    target_spacing: float = 1.0,
) -> dict[str, Any]:
    """Load, resample, normalize, and convert world-space boxes to voxel xyzxyz."""

    case = load_detection_case(image_path, boxes_world or [])
    vol, spacing = resample_to_isotropic(
        case["image"],
        case["spacing_zyx"],
        target_spacing=target_spacing,
    )
    scale_xyz = case["spacing_zyx"][::-1] / spacing[::-1]
    boxes_voxel = case["boxes_voxel"].copy()
    if boxes_voxel.size:
        boxes_voxel[:, [0, 3]] *= scale_xyz[0]
        boxes_voxel[:, [1, 4]] *= scale_xyz[1]
        boxes_voxel[:, [2, 5]] *= scale_xyz[2]
    return {
        **case,
        "image": normalise_hu(vol)[None, ...].astype(np.float32),
        "spacing_zyx": spacing.astype(np.float32),
        "boxes_voxel": boxes_voxel.astype(np.float32).reshape(-1, 6),
        "seriesuid": Path(str(image_path)).stem,
    }


class LoadDetectionCased(MapTransform):
    """Load raw LUNA16 scans, resample, normalize, and convert boxes to voxel xyzxyz."""

    def __init__(
        self,
        image_key: str = "image",
        box_key: str = "box",
        label_key: str = "label",
        target_spacing: float = 1.0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__([image_key], allow_missing_keys=allow_missing_keys)
        self.image_key = image_key
        self.box_key = box_key
        self.label_key = label_key
        self.target_spacing = float(target_spacing)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        d = dict(data)
        image_path = str(d[self.image_key])
        case = load_preprocessed_detection_case(
            image_path=image_path,
            boxes_world=d.get(self.box_key, []),
            target_spacing=self.target_spacing,
        )
        d[self.image_key] = case["image"]
        d[self.box_key] = case["boxes_voxel"]
        d[self.label_key] = np.zeros((len(d[self.box_key]),), dtype=np.int64)
        d["seriesuid"] = case["seriesuid"]
        return d


class GenerateExtendedBoxMaskd(MapTransform):
    """Generate a foreground mask of valid crop centers from boxes."""

    def __init__(
        self,
        keys: str | list[str],
        image_key: str,
        spatial_size: tuple[int, int, int],
        whole_box: bool = True,
        mask_image_key: str = "mask_image",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.image_key = image_key
        self.spatial_size = tuple(int(v) for v in spatial_size)
        self.whole_box = whole_box
        self.mask_image_key = mask_image_key

    def _generate_fg_center_boxes(self, boxes: np.ndarray, image_size: tuple[int, int, int]) -> np.ndarray:
        spatial_dims = len(image_size)
        extended_boxes = np.zeros_like(boxes, dtype=int)
        boxes_start = np.ceil(boxes[:, :spatial_dims]).astype(int)
        boxes_stop = np.floor(boxes[:, spatial_dims:]).astype(int)
        for axis in range(spatial_dims):
            if not self.whole_box:
                extended_boxes[:, axis] = boxes_start[:, axis] - self.spatial_size[axis] // 2 + 1
                extended_boxes[:, axis + spatial_dims] = boxes_stop[:, axis] + self.spatial_size[axis] // 2 - 1
            else:
                extended_boxes[:, axis] = boxes_stop[:, axis] - self.spatial_size[axis] // 2 - 1
                extended_boxes[:, axis] = np.minimum(extended_boxes[:, axis], boxes_start[:, axis])
                extended_boxes[:, axis + spatial_dims] = extended_boxes[:, axis] + self.spatial_size[axis] // 2
                extended_boxes[:, axis + spatial_dims] = np.maximum(
                    extended_boxes[:, axis + spatial_dims], boxes_stop[:, axis]
                )
        clipped, _ = clip_boxes_to_image(extended_boxes, image_size, remove_empty=True)
        return np.asarray(clipped, dtype=np.int32).reshape(-1, spatial_dims * 2)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[self.image_key]
            boxes = np.asarray(d[key], dtype=np.float32).reshape(-1, 6)
            image_size = tuple(int(v) for v in image.shape[1:])
            if boxes.size == 0:
                d[self.mask_image_key] = np.zeros((1, *image_size), dtype=np.uint8)
                continue
            extended_boxes = self._generate_fg_center_boxes(boxes, image_size)
            mask = convert_box_to_mask(
                extended_boxes,
                np.ones((len(extended_boxes),), dtype=np.int64),
                image_size,
                bg_label=0,
                ellipse_mask=False,
            )
            d[self.mask_image_key] = np.amax(mask, axis=0, keepdims=True)[0:1, ...].astype(np.uint8)
        return d


def _detection_train_transforms(
    patch_size: tuple[int, int, int],
    target_spacing: float,
    samples_per_image: int,
) -> Compose:
    return Compose(
        [
            LoadDetectionCased(image_key="image", box_key="box", label_key="label", target_spacing=target_spacing),
            EnsureTyped(keys=["image", "box"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.long),
            StandardizeEmptyBoxd(box_keys=["box"], box_ref_image_keys="image"),
            ConvertBoxToPointsd(keys=["box"], point_key="points"),
            GenerateExtendedBoxMaskd(keys="box", image_key="image", spatial_size=patch_size, whole_box=True),
            RandCropByPosNegLabeld(
                keys=["image"],
                label_key="mask_image",
                spatial_size=patch_size,
                num_samples=samples_per_image,
                pos=1.0,
                neg=1.0,
            ),
            RandZoomd(
                keys=["image"],
                prob=0.2,
                min_zoom=0.7,
                max_zoom=1.4,
                padding_mode="constant",
                keep_size=True,
            ),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image"], prob=0.75, max_k=3, spatial_axes=(0, 1)),
            ApplyTransformToPointsd(keys=["points"], refer_keys="image", affine_lps_to_ras=False),
            ConvertPointsToBoxesd(keys=["points"], box_key="box"),
            ClipBoxToImaged(
                box_keys=["box"],
                label_keys=["label"],
                box_ref_image_keys="image",
                remove_empty=True,
            ),
            BoxToMaskd(
                box_keys=["box"],
                label_keys=["label"],
                box_mask_keys=["box_mask"],
                box_ref_image_keys="image",
                min_fg_label=0,
                ellipse_mask=True,
            ),
            RandRotated(
                keys=["image", "box_mask"],
                mode=["bilinear", "nearest"],
                prob=0.2,
                range_x=np.pi / 6,
                range_y=np.pi / 6,
                range_z=np.pi / 6,
                keep_size=True,
                padding_mode="zeros",
            ),
            MaskToBoxd(
                box_keys=["box"],
                label_keys=["label"],
                box_mask_keys=["box_mask"],
                min_fg_label=0,
            ),
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.03),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.1,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys=["image"], prob=0.15, factors=0.25),
            RandShiftIntensityd(keys=["image"], prob=0.15, offsets=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            DeleteItemsd(keys=["mask_image", "points", "box_mask"]),
            EnsureTyped(keys=["image", "box"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.long),
        ]
    )


def _detection_val_transforms(target_spacing: float) -> Compose:
    return Compose(
        [
            LoadDetectionCased(image_key="image", box_key="box", label_key="label", target_spacing=target_spacing),
            EnsureTyped(keys=["image", "box"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.long),
            StandardizeEmptyBoxd(box_keys=["box"], box_ref_image_keys="image"),
        ]
    )


def build_monai_detection_train_dataset(
    items: list[dict[str, Any]],
    patch_size: tuple[int, int, int] = (96, 96, 96),
    target_spacing: float = 1.0,
    samples_per_image: int = 4,
) -> MonaiDataset:
    return MonaiDataset(
        data=items,
        transform=_detection_train_transforms(
            patch_size=patch_size,
            target_spacing=target_spacing,
            samples_per_image=samples_per_image,
        ),
    )


def build_monai_detection_val_dataset(
    items: list[dict[str, Any]],
    target_spacing: float = 1.0,
) -> MonaiDataset:
    return MonaiDataset(
        data=items,
        transform=_detection_val_transforms(target_spacing=target_spacing),
    )


def _flatten_monai_batch(batch: list[Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for item in batch:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def monai_detection_collate(
    batch: list[Any],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]], list[str]]:
    samples = _flatten_monai_batch(batch)
    images = [sample["image"] for sample in samples]
    targets = [
        {
            "boxes": sample["box"],
            "labels": sample["label"],
        }
        for sample in samples
    ]
    seriesuids = [str(sample.get("seriesuid", "")) for sample in samples]
    return images, targets, seriesuids
