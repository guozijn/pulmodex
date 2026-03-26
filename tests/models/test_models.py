"""Smoke tests for model forward passes."""

import pytest
import torch

from src.fp_reduction import FPClassifier
from src.models.baseline import UNet3D
from src.models.hybrid import HybridNet


@pytest.mark.parametrize("patch_size", [32, 64])
def test_unet3d_forward(patch_size):
    model = UNet3D(in_ch=1, base_ch=8)  # small for speed
    x = torch.randn(1, 1, patch_size, patch_size, patch_size)
    out = model(x)
    assert "seg" in out and "logits" in out
    assert out["seg"].shape == x.shape
    assert out["logits"].shape == x.shape
    assert out["seg"].min() >= 0.0 and out["seg"].max() <= 1.0


@pytest.mark.parametrize("patch_size", [32])
def test_hybrid_net_forward(patch_size):
    model = HybridNet(in_ch=1, base_ch=4, swin_depth=1, swin_heads=2, swin_window=4)
    x = torch.randn(1, 1, patch_size, patch_size, patch_size)
    out = model(x)
    assert "seg" in out and "logits" in out and "ds_logits" in out
    assert out["seg"].shape == x.shape
    assert isinstance(out["ds_logits"], list) and len(out["ds_logits"]) > 0


def test_fp_classifier_forward():
    model = FPClassifier(in_ch=1, base_ch=8)
    x = torch.randn(2, 1, 32, 32, 32)
    out = model(x)
    assert "logits" in out and "prob" in out
    assert out["logits"].shape == (2, 2)
    assert out["prob"].shape == (2,)
    assert (out["prob"] >= 0).all() and (out["prob"] <= 1).all()
