"""Grad-CAM for the baseline 3D U-Net.

Hooks the final encoder layer (enc4.block) to extract gradients and
activations, then produces a saliency map resampled to input resolution.

Usage:
    gcam = GradCAM(model)
    saliency = gcam(image_tensor)  # (D, H, W) in [0, 1]
    gcam.remove_hooks()
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM saliency extractor for UNet3D.

    Args:
        model: UNet3D instance (in eval mode)
        target_layer: nn.Module to hook (default: model.enc4.block)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        if target_layer is None:
            target_layer = model.enc4.block  # type: ignore[attr-defined]

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """Compute saliency map.

        Args:
            x: (1, 1, D, H, W) input patch, on same device as model

        Returns:
            saliency: (D, H, W) float32 in [0, 1]
        """
        self._activations = None
        self._gradients = None
        self.model.zero_grad()
        out = self.model(x)
        # Scalar target: sum of segmentation probabilities
        score = out["seg"].sum()
        score.backward()

        assert self._gradients is not None and self._activations is not None

        # Global average pooling of gradients → channel weights
        weights = self._gradients.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, d, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()
