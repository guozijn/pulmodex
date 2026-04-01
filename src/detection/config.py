"""Shared config loading helpers for detection CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def merge_cli_with_config(
    config_path: str | Path,
    defaults: dict[str, Any],
    cli_values: dict[str, Any],
) -> dict[str, Any]:
    """Merge defaults, YAML config, and explicit CLI values."""

    merged = dict(defaults)
    cfg_path = Path(config_path)
    if cfg_path.exists():
        loaded = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Detection config {cfg_path} must be a mapping.")
        merged.update(loaded)
    else:
        raise FileNotFoundError(f"Detection config file not found: {cfg_path}")

    for key, value in cli_values.items():
        if value is not None:
            merged[key] = value
    return merged
