"""MONAI-backed loss wrappers with the project's existing call signatures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss as MonaiDiceCELoss
from monai.losses import DiceFocalLoss as MonaiDiceFocalLoss
from monai.losses import FocalLoss as MonaiFocalLoss


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Per-sample soft Dice loss for binary segmentation tensors."""
    if target.dim() == pred.dim() - 1:
        target = target.unsqueeze(1)

    pred = pred.float()
    target = target.float()

    reduce_dims = tuple(range(1, pred.dim()))
    intersection = (pred * target).sum(dim=reduce_dims)
    denom = pred.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Dice + BCE using MONAI's DiceCELoss."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.loss = MonaiDiceCELoss(
            sigmoid=True,
            squared_pred=False,
            lambda_dice=dice_weight,
            lambda_ce=bce_weight,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == logits.dim() - 1:
            target = target.unsqueeze(1)
        return self.loss(logits, target.float())


class FocalLoss(nn.Module):
    """Binary focal loss using MONAI's implementation."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.loss = MonaiFocalLoss(
            include_background=True,
            use_softmax=False,
            gamma=gamma,
            alpha=alpha,
            reduction="mean",
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == logits.dim() - 1:
            target = target.unsqueeze(1)
        return self.loss(logits, target.float())


class DiceFocalLoss(nn.Module):
    """Dice + focal loss with optional deep supervision."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        gamma: float = 2.0,
        alpha: float = 0.25,
    ):
        super().__init__()
        self.loss = MonaiDiceFocalLoss(
            sigmoid=True,
            squared_pred=False,
            lambda_dice=dice_weight,
            lambda_focal=focal_weight,
            gamma=gamma,
            alpha=alpha,
        )
        self.deep_supervision_weight = 0.5

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        deep_supervision_logits: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if target.dim() == logits.dim() - 1:
            target = target.unsqueeze(1)

        target = target.float()
        loss = self.loss(logits, target)

        if deep_supervision_logits:
            for ds_logit in deep_supervision_logits:
                ds_target = F.interpolate(target, size=ds_logit.shape[2:], mode="nearest")
                loss = loss + self.deep_supervision_weight * self.loss(ds_logit, ds_target)

        return loss
