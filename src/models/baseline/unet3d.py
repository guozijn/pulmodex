"""3D U-Net baseline model for lung nodule segmentation.

Architecture: encoder-decoder with skip connections.
Input: (B, 1, D, H, W) CT patch, 128³ at training.
Output: {"seg": sigmoid mask (B,1,D,H,W), "logits": raw (B,1,D,H,W)}
"""

import torch
import torch.nn as nn

from src.models.shared.blocks import ConvBnRelu, ResidualBlock


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ResidualBlock(in_ch, out_ch),
            ResidualBlock(out_ch, out_ch),
        )
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.block(x)
        return self.pool(feat), feat  # (pooled, skip)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ResidualBlock(in_ch // 2 + skip_ch, out_ch),
            ResidualBlock(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNet3D(nn.Module):
    """3D U-Net with four encoder and four decoder levels.

    Channels: 1 → 32 → 64 → 128 → 256 (bottleneck) → 128 → 64 → 32 → 1
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 32):
        super().__init__()
        c = base_ch
        self.enc1 = EncoderBlock(in_ch, c)
        self.enc2 = EncoderBlock(c, c * 2)
        self.enc3 = EncoderBlock(c * 2, c * 4)
        self.enc4 = EncoderBlock(c * 4, c * 8)

        self.bottleneck = nn.Sequential(
            ResidualBlock(c * 8, c * 16),
            ResidualBlock(c * 16, c * 16),
        )

        self.dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.dec1 = DecoderBlock(c * 2, c, c)

        self.head = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)

        b = self.bottleneck(x4)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        logits = self.head(d1)
        return {"seg": torch.sigmoid(logits), "logits": logits}
