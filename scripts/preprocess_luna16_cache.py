"""Precompute cached isotropic, normalized LUNA16 volumes for faster training.

Usage:
    python scripts/preprocess_luna16_cache.py \
        --data_dir data/processed \
        --cache_dir data/processed/.cache/luna16_iso
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.preprocessing import load_mhd, normalise_hu, resample_to_isotropic


def cache_paths(cache_dir: Path, seriesuid: str) -> tuple[Path, Path]:
    return cache_dir / f"{seriesuid}_vol.npy", cache_dir / f"{seriesuid}_meta.json"


def write_cache(
    cache_dir: Path,
    seriesuid: str,
    vol: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
) -> None:
    vol_path, meta_path = cache_paths(cache_dir, seriesuid)
    tmp_vol = vol_path.with_suffix(".tmp.npy")
    tmp_meta = meta_path.with_suffix(".tmp.json")

    np.save(tmp_vol, vol.astype(np.float32, copy=False))
    tmp_meta.write_text(
        json.dumps(
            {
                "spacing_zyx": spacing.tolist(),
                "origin_zyx": origin.tolist(),
            }
        )
    )
    tmp_vol.replace(vol_path)
    tmp_meta.replace(meta_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute LUNA16 training cache")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir or data_dir / ".cache" / "luna16_iso")
    cache_dir.mkdir(parents=True, exist_ok=True)

    series = sorted({p.stem for p in data_dir.glob("subset*/*.mhd")})
    if not series:
        raise FileNotFoundError(f"No .mhd files found under {data_dir}")

    print(f"Found {len(series)} scans. Writing cache to {cache_dir}")
    for i, seriesuid in enumerate(series, 1):
        vol_path, meta_path = cache_paths(cache_dir, seriesuid)
        if vol_path.exists() and meta_path.exists():
            if i % 20 == 0 or i == len(series):
                print(f"[{i}/{len(series)}] cached {seriesuid}")
            continue

        matches = list(data_dir.glob(f"subset*/{seriesuid}.mhd"))
        if not matches:
            raise FileNotFoundError(f"Missing .mhd for {seriesuid}")

        vol, spacing, origin = load_mhd(str(matches[0]))
        vol, spacing = resample_to_isotropic(vol, spacing)
        vol = normalise_hu(vol)
        write_cache(cache_dir, seriesuid, vol, spacing, origin)
        print(f"[{i}/{len(series)}] cached {seriesuid}")


if __name__ == "__main__":
    main()
