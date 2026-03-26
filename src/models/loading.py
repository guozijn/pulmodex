"""Checkpoint-aware model construction helpers."""

from __future__ import annotations

from typing import Any

import torch


def build_model_from_config(model_cfg: dict[str, Any]) -> torch.nn.Module:
    """Instantiate a model from a serialized config dictionary."""
    name = model_cfg["name"]
    if name == "unet3d":
        from src.models.baseline import UNet3D

        return UNet3D(
            in_ch=model_cfg.get("in_ch", 1),
            base_ch=model_cfg.get("base_ch", 32),
        )
    if name == "hybrid_net":
        from src.models.hybrid import HybridNet

        return HybridNet(
            in_ch=model_cfg.get("in_ch", 1),
            base_ch=model_cfg.get("base_ch", 32),
            swin_depth=model_cfg.get("swin_depth", 2),
            swin_heads=model_cfg.get("swin_heads", 8),
            swin_window=model_cfg.get("swin_window", 4),
        )
    if name == "fp_classifier":
        from src.fp_reduction import FPClassifier

        return FPClassifier(
            in_ch=model_cfg.get("in_ch", 1),
            base_ch=model_cfg.get("base_ch", 16),
        )
    raise ValueError(f"Unknown model: {name}")


def load_checkpoint_model(path: str, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a model and checkpoint payload from disk."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model_cfg = ckpt.get("config", {}).get("model")
    if model_cfg is None:
        legacy_name = ckpt.get("model_name") or ckpt.get("model") or "unet3d"
        model_cfg = {"name": legacy_name}

    model = build_model_from_config(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt
