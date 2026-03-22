"""Shared building blocks: residual blocks and attention modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Sequential):
    """3D Conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    """3D residual block with optional projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBnRelu(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return self.relu(out + self.shortcut(x))


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1, 1)
        return x * w


class ResidualBlockSE(nn.Module):
    """Residual block with Squeeze-and-Excitation attention."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, reduction: int = 16):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, stride)
        self.se = ChannelAttention(out_ch, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.res(x))
