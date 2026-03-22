"""Loss functions: Dice+BCE and Dice-Focal."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss. pred and target are (B, C, D, H, W) or (B, D, H, W) float tensors."""
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1).float()
    intersection = (pred * target).sum(dim=1)
    return 1.0 - (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)


class DiceBCELoss(nn.Module):
    """Dice + Binary Cross-Entropy loss for the baseline model."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: raw model output (B, 1, D, H, W)
            target: binary mask (B, 1, D, H, W) or (B, D, H, W)
        """
        if target.dim() == logits.dim() - 1:
            target = target.unsqueeze(1)
        bce = self.bce(logits, target.float())
        pred = torch.sigmoid(logits)
        dl = dice_loss(pred, target).mean()
        return self.bce_weight * bce + self.dice_weight * dl


class FocalLoss(nn.Module):
    """Binary Focal loss."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class DiceFocalLoss(nn.Module):
    """Dice + Focal loss for the hybrid model.

    Reference: hybrid U-Net–Transformer / Dice-Focal loss
    (Nature Sci. Reports, Jan 2026).
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        gamma: float = 2.0,
        alpha: float = 0.25,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal = FocalLoss(gamma, alpha)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        deep_supervision_logits: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, D, H, W) main output
            target: (B, 1, D, H, W) or (B, D, H, W) binary mask
            deep_supervision_logits: list of lower-resolution outputs
        """
        if target.dim() == logits.dim() - 1:
            target = target.unsqueeze(1)

        pred = torch.sigmoid(logits)
        loss = self.focal_weight * self.focal(logits, target) + self.dice_weight * dice_loss(
            pred, target
        ).mean()

        if deep_supervision_logits:
            for ds_logit in deep_supervision_logits:
                ds_target = F.interpolate(target.float(), size=ds_logit.shape[2:], mode="nearest")
                ds_pred = torch.sigmoid(ds_logit)
                loss = loss + 0.5 * (
                    self.focal_weight * self.focal(ds_logit, ds_target)
                    + self.dice_weight * dice_loss(ds_pred, ds_target).mean()
                )

        return loss
