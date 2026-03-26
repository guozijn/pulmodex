"""MONAI-backed 3D U-Net baseline model for lung nodule segmentation."""

import torch
import torch.nn as nn
from monai.networks.nets import UNet


class UNet3D(nn.Module):
    """3D U-Net wrapper that preserves the project's output contract."""

    def __init__(self, in_ch: int = 1, base_ch: int = 32):
        super().__init__()
        self.net = UNet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=1,
            channels=(base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.net(x)
        return {"seg": torch.sigmoid(logits), "logits": logits}
