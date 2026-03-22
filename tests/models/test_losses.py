"""Tests for loss functions."""

import torch
import pytest

from src.models.shared.losses import DiceBCELoss, DiceFocalLoss, FocalLoss, dice_loss


def test_dice_loss_perfect():
    pred = torch.ones(2, 1, 8, 8, 8)
    target = torch.ones(2, 1, 8, 8, 8)
    loss = dice_loss(pred, target)
    assert loss.mean().item() == pytest.approx(0.0, abs=1e-5)


def test_dice_loss_zeros():
    pred = torch.zeros(2, 1, 8, 8, 8)
    target = torch.ones(2, 1, 8, 8, 8)
    loss = dice_loss(pred, target)
    assert loss.mean().item() > 0.5


def test_dice_bce_loss():
    loss_fn = DiceBCELoss()
    logits = torch.randn(2, 1, 8, 8, 8)
    target = torch.randint(0, 2, (2, 1, 8, 8, 8)).float()
    loss = loss_fn(logits, target)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_dice_focal_loss_with_deep_supervision():
    loss_fn = DiceFocalLoss()
    logits = torch.randn(2, 1, 32, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
    ds = [torch.randn(2, 1, 16, 16, 16), torch.randn(2, 1, 8, 8, 8)]
    loss = loss_fn(logits, target, deep_supervision_logits=ds)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
