"""False-positive reduction classifier.

3D CNN on 32³ patches extracted around each candidate detected by the
segmentation model. Binary classification: 0 = FP, 1 = TP nodule.

Training uses online hard negative mining (OHEM): within each batch,
keep the hardest `hard_neg_ratio * n_positives` negatives.

Target: ≤1 FP/scan at sensitivity ≥0.85. Threshold tuned on val fold
to maximise CPM; stored in configs/experiment/fp_reduction.yaml.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.shared.blocks import ResidualBlock, ConvBnRelu


class FPClassifier(nn.Module):
    """Compact 3D CNN for FP reduction.

    Input:  (B, 1, 32, 32, 32) normalised CT patch
    Output: {"logits": (B, 2), "prob": (B,) — P(nodule)}
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 16):
        super().__init__()
        c = base_ch
        self.encoder = nn.Sequential(
            ConvBnRelu(in_ch, c),          # 32³
            ResidualBlock(c, c),
            nn.MaxPool3d(2),               # 16³
            ResidualBlock(c, c * 2),
            ResidualBlock(c * 2, c * 2),
            nn.MaxPool3d(2),               # 8³
            ResidualBlock(c * 2, c * 4),
            ResidualBlock(c * 4, c * 4),
            nn.MaxPool3d(2),               # 4³
            ResidualBlock(c * 4, c * 8),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(c * 8, c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(c * 4, 2),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(x)
        logits = self.head(feat)
        prob = torch.softmax(logits, dim=1)[:, 1]  # P(nodule)
        return {"logits": logits, "prob": prob}


class OHEMLoss(nn.Module):
    """Online Hard Example Mining cross-entropy loss.

    Keeps `hard_neg_ratio` hardest negatives per positive in each batch.
    """

    def __init__(self, hard_neg_ratio: float = 2.0):
        super().__init__()
        self.hard_neg_ratio = hard_neg_ratio
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses = self.ce(logits, labels)

        pos_mask = labels == 1
        neg_mask = labels == 0
        n_pos = int(pos_mask.sum().item())

        if n_pos == 0:
            return losses[neg_mask].mean() if neg_mask.any() else losses.mean()

        pos_loss = losses[pos_mask]
        neg_losses = losses[neg_mask]

        # Keep top-k hardest negatives
        k = min(int(n_pos * self.hard_neg_ratio), len(neg_losses))
        if k > 0:
            hard_neg_loss, _ = torch.topk(neg_losses, k)
            return torch.cat([pos_loss, hard_neg_loss]).mean()
        return pos_loss.mean()
