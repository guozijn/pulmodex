"""Training entry point.

Usage:
    python src/train.py experiment=baseline
    python src/train.py experiment=hybrid
    python src/train.py experiment=fp_reduction

Mac CPU smoke test:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train.py experiment=baseline \
        trainer.max_epochs=1 data.patch_size=[32,32,32] data.batch_size=1
"""

import logging
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from monai.utils import set_determinism
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Allow `python src/train.py ...` from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)


def build_model(cfg: DictConfig) -> torch.nn.Module:
    name = cfg.model.name
    if name == "unet3d":
        from src.models.baseline import UNet3D
        return UNet3D(in_ch=cfg.model.in_ch, base_ch=cfg.model.base_ch)
    if name == "hybrid_net":
        from src.models.hybrid import HybridNet
        return HybridNet(
            in_ch=cfg.model.in_ch,
            base_ch=cfg.model.base_ch,
            swin_depth=cfg.model.swin_depth,
            swin_heads=cfg.model.swin_heads,
            swin_window=cfg.model.swin_window,
        )
    if name == "fp_classifier":
        from src.fp_reduction.classifier import FPClassifier
        return FPClassifier(in_ch=cfg.model.in_ch, base_ch=cfg.model.base_ch)
    raise ValueError(f"Unknown model: {name}")


def build_loss(cfg: DictConfig) -> torch.nn.Module:
    if cfg.model.name == "fp_classifier":
        from src.fp_reduction import OHEMLoss

        return OHEMLoss(hard_neg_ratio=cfg.data.hard_neg_ratio)

    from src.models.shared.losses import DiceBCELoss, DiceFocalLoss, FocalLoss
    name = cfg.loss.name
    if name == "dice_bce":
        return DiceBCELoss(dice_weight=cfg.loss.dice_weight, bce_weight=cfg.loss.bce_weight)
    if name == "dice_focal":
        return DiceFocalLoss(
            dice_weight=cfg.loss.dice_weight,
            focal_weight=cfg.loss.focal_weight,
            gamma=cfg.loss.gamma,
            alpha=cfg.loss.alpha,
        )
    if name == "focal":
        return FocalLoss(gamma=cfg.loss.gamma, alpha=cfg.loss.alpha)
    raise ValueError(f"Unknown loss: {name}")


def build_datasets(cfg: DictConfig):
    from src.data import LUNA16Dataset
    cache_dir = cfg.data.get("cache_dir")
    train_ds = LUNA16Dataset(
        data_dir=cfg.data_dir,
        folds=list(cfg.data.train_folds),
        patch_size=cfg.data.patch_size,
        augment=cfg.data.augment,
        cache_dir=cache_dir,
    )
    val_ds = LUNA16Dataset(
        data_dir=cfg.data_dir,
        folds=list(cfg.data.val_folds),
        patch_size=cfg.data.patch_size,
        augment=False,
        cache_dir=cache_dir,
    )
    return train_ds, val_ds


def get_task_type(cfg: DictConfig) -> str:
    if cfg.model.name == "fp_classifier":
        return "classification"
    return "segmentation"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    set_determinism(seed=cfg.seed)

    device = str(cfg.trainer.device)
    log.info(f"Starting experiment '{cfg.experiment_name}' on device={device}")

    # W&B (optional)
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(project="pulmodex", name=cfg.experiment_name, config=dict(cfg))
    except Exception:
        log.warning("W&B not available; running without logging.")

    model = build_model(cfg)
    loss_fn = build_loss(cfg)
    train_ds, val_ds = build_datasets(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=cfg.data.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.trainer.lr,
        weight_decay=cfg.trainer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.trainer.max_epochs
    )

    from src.training.trainer import Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg.trainer,
        run_config=OmegaConf.to_container(cfg, resolve=True),
        device=device,
        checkpoint_dir=cfg.checkpoint_dir,
        experiment_name=cfg.experiment_name,
        task_type=get_task_type(cfg),
        data_dir=cfg.data_dir,
        wandb_run=wandb_run,
    )
    trainer.fit(train_loader, val_loader)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
