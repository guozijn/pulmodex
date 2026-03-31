"""Tests for the training pipeline: build_model, build_loss, and Trainer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.fp_reduction.classifier import FPClassifier
from src.models.baseline import UNet3D
from src.models.hybrid import HybridNet
from src.train import build_loss, build_model
from src.training.trainer import Trainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEG_PATCH = 32   # UNet3D needs ≥16 per dimension (5-level encoder, 2× pool each)
_CLS_PATCH = 32   # FPClassifier expects 32³


def _seg_cfg(monitor_mode: str = "max"):
    """Minimal OmegaConf config for a segmentation experiment."""
    return OmegaConf.create(
        {
            "model": {"name": "unet3d", "in_ch": 1, "base_ch": 8},
            "loss": {"name": "dice_bce", "dice_weight": 0.5, "bce_weight": 0.5},
            "trainer": {
                "max_epochs": 1,
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "grad_clip": 1.0,
                "log_every_n_steps": 1,
                "val_every_n_epochs": 1,
                "save_top_k": 3,
                "monitor_metric": "val_dice",
                "monitor_mode": monitor_mode,
            },
            "data_dir": "tests/data",
        }
    )


def _cls_cfg():
    """Minimal OmegaConf config for a classification experiment."""
    return OmegaConf.create(
        {
            "model": {"name": "fp_classifier", "in_ch": 1, "base_ch": 8},
            "data": {"hard_neg_ratio": 2.0},
            "trainer": {
                "max_epochs": 1,
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "grad_clip": 1.0,
                "log_every_n_steps": 1,
                "val_every_n_epochs": 1,
                "save_top_k": 3,
                "monitor_metric": "val_acc",
                "monitor_mode": "max",
            },
        }
    )


def _seg_loader(n: int = 2):
    """Synthetic segmentation DataLoader returning dict batches."""
    images = torch.randn(n, 1, _SEG_PATCH, _SEG_PATCH, _SEG_PATCH)
    masks = torch.randint(0, 2, (n, 1, _SEG_PATCH, _SEG_PATCH, _SEG_PATCH)).float()

    class SegBatchDataset(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, idx):
            return {
                "image": images[idx],
                "mask": masks[idx],
                "seriesuid": f"uid-{idx}",
                "coord_xyz": torch.zeros(3),
            }

    return DataLoader(SegBatchDataset(), batch_size=n)


def _cls_loader(n: int = 4):
    """Synthetic classification DataLoader returning dict batches."""
    images = torch.randn(n, 1, _CLS_PATCH, _CLS_PATCH, _CLS_PATCH)
    labels = torch.randint(0, 2, (n,))

    class ClsBatchDataset(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, idx):
            return {"image": images[idx], "label": labels[idx]}

    return DataLoader(ClsBatchDataset(), batch_size=n)


def _make_trainer(cfg, task_type: str, checkpoint_dir: str, model=None, loss_fn=None):
    if model is None:
        model = build_model(cfg)
    if loss_fn is None:
        loss_fn = build_loss(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=None,
        cfg=cfg.trainer,
        run_config=OmegaConf.to_container(cfg, resolve=True),
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        experiment_name="test_exp",
        task_type=task_type,
        data_dir=cfg.get("data_dir"),
    )


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


def test_build_model_unet3d():
    cfg = _seg_cfg()
    model = build_model(cfg)
    assert isinstance(model, UNet3D)


def test_build_model_hybrid_net():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "hybrid_net",
                "in_ch": 1,
                "base_ch": 4,
                "swin_depth": 1,
                "swin_heads": 2,
                "swin_window": 4,
            }
        }
    )
    model = build_model(cfg)
    assert isinstance(model, HybridNet)


def test_build_model_fp_classifier():
    cfg = _cls_cfg()
    model = build_model(cfg)
    assert isinstance(model, FPClassifier)


def test_build_model_unknown_raises():
    cfg = OmegaConf.create({"model": {"name": "nonexistent"}})
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(cfg)


# ---------------------------------------------------------------------------
# build_loss
# ---------------------------------------------------------------------------


def test_build_loss_dice_bce():
    from src.models.shared.losses import DiceBCELoss

    cfg = _seg_cfg()
    loss_fn = build_loss(cfg)
    assert isinstance(loss_fn, DiceBCELoss)


def test_build_loss_dice_focal():
    from src.models.shared.losses import DiceFocalLoss

    cfg = OmegaConf.create(
        {
            "model": {"name": "unet3d"},
            "loss": {
                "name": "dice_focal",
                "dice_weight": 0.5,
                "focal_weight": 0.5,
                "gamma": 2.0,
                "alpha": 0.25,
            },
        }
    )
    loss_fn = build_loss(cfg)
    assert isinstance(loss_fn, DiceFocalLoss)


def test_build_loss_ohem_for_fp_classifier():
    from src.fp_reduction import OHEMLoss

    cfg = _cls_cfg()
    loss_fn = build_loss(cfg)
    assert isinstance(loss_fn, OHEMLoss)


def test_build_loss_unknown_raises():
    cfg = OmegaConf.create(
        {"model": {"name": "unet3d"}, "loss": {"name": "bad_loss"}}
    )
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss(cfg)


# ---------------------------------------------------------------------------
# Trainer — segmentation
# ---------------------------------------------------------------------------


def test_trainer_segmentation_one_epoch_runs():
    """Trainer completes 1 segmentation epoch without error."""
    cfg = _seg_cfg()
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(cfg, "segmentation", tmpdir)
        train_loader = _seg_loader()
        val_loader = _seg_loader()
        trainer.fit(train_loader, val_loader)  # should not raise


def test_trainer_segmentation_creates_checkpoint():
    cfg = _seg_cfg()
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(cfg, "segmentation", tmpdir)
        trainer.fit(_seg_loader(), _seg_loader())
        ckpts = list(Path(tmpdir).glob("*.ckpt"))
        assert len(ckpts) >= 1


def test_trainer_segmentation_best_checkpoint_exists():
    cfg = _seg_cfg()
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(cfg, "segmentation", tmpdir)
        trainer.fit(_seg_loader(), _seg_loader())
        best = Path(tmpdir) / "test_exp_best.ckpt"
        assert best.exists()


def test_trainer_checkpoint_contains_expected_keys():
    cfg = _seg_cfg()
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(cfg, "segmentation", tmpdir)
        trainer.fit(_seg_loader(), _seg_loader())
        ckpt_path = next(Path(tmpdir).glob("test_exp_epoch*.ckpt"))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for key in ("epoch", "model_state_dict", "optimizer_state_dict", "metric"):
            assert key in ckpt, f"Missing key: {key}"


def test_trainer_top_k_pruning():
    """With save_top_k=1, only 1 epoch checkpoint should remain after 3 epochs."""
    cfg = OmegaConf.create(
        {
            "model": {"name": "unet3d", "in_ch": 1, "base_ch": 8},
            "loss": {"name": "dice_bce", "dice_weight": 0.5, "bce_weight": 0.5},
            "trainer": {
                "max_epochs": 3,
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "grad_clip": 1.0,
                "log_every_n_steps": 1,
                "val_every_n_epochs": 1,
                "save_top_k": 1,
                "monitor_metric": "val_dice",
                "monitor_mode": "max",
            },
            "data_dir": "tests/data",
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(cfg, "segmentation", tmpdir)
        trainer.fit(_seg_loader(), _seg_loader())
        epoch_ckpts = list(Path(tmpdir).glob("test_exp_epoch*.ckpt"))
        assert len(epoch_ckpts) == 1


# ---------------------------------------------------------------------------
# Trainer — classification
# ---------------------------------------------------------------------------


def test_trainer_classification_one_epoch_runs():
    """Trainer completes 1 classification epoch without error."""
    cfg = _cls_cfg()
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(cfg, "classification", tmpdir)
        trainer.fit(_cls_loader(), _cls_loader())


def test_trainer_classification_creates_checkpoint():
    cfg = _cls_cfg()
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(cfg, "classification", tmpdir)
        trainer.fit(_cls_loader(), _cls_loader())
        ckpts = list(Path(tmpdir).glob("*.ckpt"))
        assert len(ckpts) >= 1


# ---------------------------------------------------------------------------
# Trainer — monitor_mode
# ---------------------------------------------------------------------------


def test_trainer_invalid_monitor_mode_raises():
    cfg = OmegaConf.create(
        {
            "model": {"name": "unet3d", "in_ch": 1, "base_ch": 8},
            "loss": {"name": "dice_bce", "dice_weight": 0.5, "bce_weight": 0.5},
            "trainer": {
                "max_epochs": 1,
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "grad_clip": 1.0,
                "log_every_n_steps": 1,
                "val_every_n_epochs": 1,
                "save_top_k": 3,
                "monitor_metric": "val_dice",
                "monitor_mode": "invalid",
            },
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="monitor_mode"):
            _make_trainer(cfg, "segmentation", tmpdir)
