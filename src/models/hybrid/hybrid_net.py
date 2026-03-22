"""Hybrid Res-U-Net + Swin Transformer model for lung nodule segmentation.

Architecture:
  - Residual CNN encoder (4 levels, SE attention)
  - Swin Transformer bottleneck (2 blocks)
  - Transposed-conv decoder with deep supervision
  - Dice-Focal loss (configured separately)

Input:  (B, 1, D, H, W) CT patch, 128³ at training.
Output: {"seg": sigmoid mask (B,1,D,H,W), "logits": raw (B,1,D,H,W),
         "ds_logits": list of lower-res raw outputs (for deep supervision)}

References:
  - Hybrid U-Net–Transformer / Dice-Focal loss (Nature Sci. Reports, Jan 2026)
  - AutoLungDx Res-U-Net + ViT on LUNA16 (arXiv, 2025)
  - Dual-attention CNN (Nature Sci. Reports, 2024)
"""

import torch
import torch.nn as nn

from src.models.shared.blocks import ResidualBlockSE
from src.models.hybrid.swin3d import SwinBottleneck


class ResEncoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ResidualBlockSE(in_ch, out_ch),
            ResidualBlockSE(out_ch, out_ch),
        )
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.block(x)
        return self.pool(feat), feat  # (pooled, skip)


class ResDecoder(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ResidualBlockSE(in_ch // 2 + skip_ch, out_ch),
            ResidualBlockSE(out_ch, out_ch),
        )
        self.ds_head = nn.Conv3d(out_ch, 1, kernel_size=1)  # deep supervision

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        feat = self.block(x)
        ds = self.ds_head(feat)
        return feat, ds


class HybridNet(nn.Module):
    """Hybrid Residual CNN Encoder + Swin Transformer Bottleneck + Decoder."""

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        swin_depth: int = 2,
        swin_heads: int = 8,
        swin_window: int = 4,
    ):
        super().__init__()
        c = base_ch

        # Encoder
        self.enc1 = ResEncoder(in_ch, c)
        self.enc2 = ResEncoder(c, c * 2)
        self.enc3 = ResEncoder(c * 2, c * 4)
        self.enc4 = ResEncoder(c * 4, c * 8)

        # Bottleneck: CNN bridge → Swin → CNN bridge
        self.pre_swin = nn.Sequential(
            ResidualBlockSE(c * 8, c * 16),
        )
        self.swin = SwinBottleneck(c * 16, depth=swin_depth, num_heads=swin_heads, window_size=swin_window)
        self.post_swin = ResidualBlockSE(c * 16, c * 16)

        # Decoder
        self.dec4 = ResDecoder(c * 16, c * 8, c * 8)
        self.dec3 = ResDecoder(c * 8, c * 4, c * 4)
        self.dec2 = ResDecoder(c * 4, c * 2, c * 2)
        self.dec1 = ResDecoder(c * 2, c, c)

        self.head = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Encode
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)

        # Bottleneck
        b = self.pre_swin(x4)
        b = self.swin(b)
        b = self.post_swin(b)

        # Decode with deep supervision
        d4, ds4 = self.dec4(b, s4)
        d3, ds3 = self.dec3(d4, s3)
        d2, ds2 = self.dec2(d3, s2)
        d1, ds1 = self.dec1(d2, s1)

        logits = self.head(d1)
        return {
            "seg": torch.sigmoid(logits),
            "logits": logits,
            # ds1 is full-resolution (same as main head) — excluded from deep supervision
            "ds_logits": [ds4, ds3, ds2],
        }
