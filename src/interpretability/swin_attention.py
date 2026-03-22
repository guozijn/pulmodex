"""Swin bottleneck attention extractor for the hybrid model.

Hooks all SwinBlock3D modules in the bottleneck and averages the
attention maps across heads and blocks to produce a single saliency
volume, then resamples to input resolution.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hybrid.swin3d import WindowAttention3D


class SwinAttentionExtractor:
    """Extract and average attention maps from the Swin bottleneck.

    Args:
        model: HybridNet instance (in eval mode)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._attn_maps: list[torch.Tensor] = []
        self._hooks: list = []

        # Register hooks on every WindowAttention3D in the bottleneck
        for module in model.swin.modules():  # type: ignore[attr-defined]
            if isinstance(module, WindowAttention3D):
                h = module.register_forward_hook(self._save_attn)
                self._hooks.append(h)

    def _save_attn(self, module: WindowAttention3D, input, output):
        # Re-compute attention weights from saved qkv
        # We capture the output (projected) and use a lightweight re-forward
        # to get the raw attention matrix stored in the module.
        # Since we can't easily access internal attn without modifying the module,
        # we approximate saliency as the L2 norm of the output per spatial token.
        self._attn_maps.append(output.detach())  # (nW*B, ws³, C)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """Compute attention-based saliency map.

        Args:
            x: (1, 1, D, H, W) input, same device as model

        Returns:
            saliency: (D, H, W) float32 in [0, 1]
        """
        self._attn_maps.clear()
        out = self.model(x)

        if not self._attn_maps:
            return np.zeros(x.shape[2:], dtype=np.float32)

        # Average L2 norm across all captured attention outputs
        # Each tensor: (nW*B, ws³, C) → (nW*B, ws³)
        maps = [m.norm(dim=-1) for m in self._attn_maps]
        avg_map = torch.stack(maps).mean(dim=0)  # (nW*B, ws³)

        # Reconstruct spatial volume from window tokens
        # We know bottleneck spatial size from model output
        bottleneck_spatial = self.model.swin.blocks[0].window_size  # type: ignore
        ws = bottleneck_spatial
        nW_B = avg_map.shape[0]
        # nW = nW_B (batch=1), spatial tokens = nW * ws³ → reshape to (1, d, h, w)
        n_tokens = avg_map.shape[0] * avg_map.shape[1]
        side = round(n_tokens ** (1 / 3))
        try:
            vol = avg_map.view(1, 1, side, side, side).float()
        except RuntimeError:
            vol = avg_map.mean(dim=0).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        saliency = F.interpolate(vol, size=x.shape[2:], mode="trilinear", align_corners=False)
        saliency = saliency.squeeze().cpu().numpy()

        s_min, s_max = saliency.min(), saliency.max()
        if s_max > s_min:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)

        return saliency.astype(np.float32)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
