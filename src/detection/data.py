"""Dataset utilities for MONAI 3D detection training and inference."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import normalise_hu, resample_to_isotropic

from .io import load_detection_case


def clip_boxes_to_patch(boxes_xyzxyz: np.ndarray, patch_size_zyx: tuple[int, int, int]) -> np.ndarray:
    if boxes_xyzxyz.size == 0:
        return boxes_xyzxyz.reshape(0, 6).astype(np.float32)
    max_xyz = np.asarray(
        [patch_size_zyx[2], patch_size_zyx[1], patch_size_zyx[0], patch_size_zyx[2], patch_size_zyx[1], patch_size_zyx[0]],
        dtype=np.float32,
    )
    clipped = np.clip(boxes_xyzxyz, 0.0, max_xyz)
    valid = np.all(clipped[:, 3:] - clipped[:, :3] >= 1.0, axis=1)
    return clipped[valid].astype(np.float32)


def crop_patch(
    vol: np.ndarray,
    boxes_xyzxyz: np.ndarray,
    start_zyx: tuple[int, int, int],
    patch_size_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    patch = np.zeros(patch_size_zyx, dtype=np.float32)
    z0, y0, x0 = start_zyx
    pd, ph, pw = patch_size_zyx
    z1, y1, x1 = z0 + pd, y0 + ph, x0 + pw

    src_z0, src_y0, src_x0 = max(z0, 0), max(y0, 0), max(x0, 0)
    src_z1, src_y1, src_x1 = min(z1, vol.shape[0]), min(y1, vol.shape[1]), min(x1, vol.shape[2])
    dst_z0, dst_y0, dst_x0 = src_z0 - z0, src_y0 - y0, src_x0 - x0
    dst_z1 = dst_z0 + (src_z1 - src_z0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    patch[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = vol[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]

    if boxes_xyzxyz.size == 0:
        return patch, boxes_xyzxyz.reshape(0, 6).astype(np.float32)

    offset_xyz = np.asarray([x0, y0, z0, x0, y0, z0], dtype=np.float32)
    adjusted = boxes_xyzxyz - offset_xyz
    adjusted = clip_boxes_to_patch(adjusted, patch_size_zyx)
    return patch, adjusted


class LUNA16DetectionDataset(Dataset):
    """Random crop dataset for 3D RetinaNet training on LUNA16."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        patch_size: tuple[int, int, int] = (96, 96, 96),
        target_spacing: float = 1.0,
        samples_per_image: int = 4,
        positive_fraction: float = 0.75,
        training: bool = True,
        seed: int = 42,
    ) -> None:
        self.items = items
        self.patch_size = tuple(int(v) for v in patch_size)
        self.target_spacing = float(target_spacing)
        self.samples_per_image = int(samples_per_image)
        self.positive_fraction = float(positive_fraction)
        self.training = training
        self.rng = random.Random(seed)
        self._cache: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        if self.training:
            return len(self.items) * max(self.samples_per_image, 1)
        return len(self.items)

    def _load_case(self, item: dict[str, Any]) -> dict[str, Any]:
        key = item["image"]
        if key not in self._cache:
            case = load_detection_case(item["image"], item.get("box", []))
            vol, spacing = resample_to_isotropic(
                case["image"],
                case["spacing_zyx"],
                target_spacing=self.target_spacing,
            )
            scale_xyz = case["spacing_zyx"][::-1] / spacing[::-1]
            boxes_voxel = case["boxes_voxel"].copy()
            if boxes_voxel.size:
                boxes_voxel[:, [0, 3]] *= scale_xyz[0]
                boxes_voxel[:, [1, 4]] *= scale_xyz[1]
                boxes_voxel[:, [2, 5]] *= scale_xyz[2]
            self._cache[key] = {
                **case,
                "image": normalise_hu(vol),
                "spacing_zyx": spacing.astype(np.float32),
                "boxes_voxel": boxes_voxel.astype(np.float32),
            }
        return self._cache[key]

    def _sample_start(self, shape_zyx: tuple[int, int, int], boxes_xyzxyz: np.ndarray) -> tuple[int, int, int]:
        pd, ph, pw = self.patch_size
        depth, height, width = shape_zyx

        if not self.training:
            if boxes_xyzxyz.size:
                chosen = boxes_xyzxyz[0]
                centre_xyz = (chosen[:3] + chosen[3:]) / 2.0
                x0 = int(round(float(centre_xyz[0] - pw / 2.0)))
                y0 = int(round(float(centre_xyz[1] - ph / 2.0)))
                z0 = int(round(float(centre_xyz[2] - pd / 2.0)))
            else:
                x0 = max((width - pw) // 2, 0)
                y0 = max((height - ph) // 2, 0)
                z0 = max((depth - pd) // 2, 0)
            x0 = min(max(x0, 0), max(width - pw, 0))
            y0 = min(max(y0, 0), max(height - ph, 0))
            z0 = min(max(z0, 0), max(depth - pd, 0))
            return z0, y0, x0

        if self.training and boxes_xyzxyz.size and self.rng.random() < self.positive_fraction:
            chosen = boxes_xyzxyz[self.rng.randrange(len(boxes_xyzxyz))]
            centre_xyz = (chosen[:3] + chosen[3:]) / 2.0
            jitter_xyz = np.asarray(
                [
                    self.rng.uniform(-pw * 0.15, pw * 0.15),
                    self.rng.uniform(-ph * 0.15, ph * 0.15),
                    self.rng.uniform(-pd * 0.15, pd * 0.15),
                ],
                dtype=np.float32,
            )
            centre_xyz = centre_xyz + jitter_xyz
            x0 = int(round(float(centre_xyz[0] - pw / 2.0)))
            y0 = int(round(float(centre_xyz[1] - ph / 2.0)))
            z0 = int(round(float(centre_xyz[2] - pd / 2.0)))
        else:
            x0 = self.rng.randint(0, max(width - pw, 0)) if width > pw else 0
            y0 = self.rng.randint(0, max(height - ph, 0)) if height > ph else 0
            z0 = self.rng.randint(0, max(depth - pd, 0)) if depth > pd else 0

        x0 = min(max(x0, 0), max(width - pw, 0))
        y0 = min(max(y0, 0), max(height - ph, 0))
        z0 = min(max(z0, 0), max(depth - pd, 0))
        return z0, y0, x0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx % len(self.items)] if self.training else self.items[idx]
        case = self._load_case(item)
        vol = case["image"]
        boxes = case["boxes_voxel"]
        start_zyx = self._sample_start(tuple(vol.shape), boxes)
        patch, patch_boxes = crop_patch(vol, boxes, start_zyx, self.patch_size)

        labels = np.zeros((len(patch_boxes),), dtype=np.int64)
        return {
            "image": torch.from_numpy(patch[None, ...].astype(np.float32)),
            "target": {
                "boxes": torch.as_tensor(patch_boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
            },
            "seriesuid": Path(item["image"]).stem,
        }


def detection_collate(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], list[str]]:
    images = torch.stack([sample["image"] for sample in batch])
    targets = [sample["target"] for sample in batch]
    seriesuids = [sample["seriesuid"] for sample in batch]
    return images, targets, seriesuids
