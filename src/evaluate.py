"""Evaluation entry point.

Usage:
    python src/evaluate.py --checkpoint checkpoints/best.ckpt --split test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import LUNA16Dataset
from src.evaluation.froc import compute_froc
from src.evaluation.metrics import dice_coefficient

log = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Detect model type from checkpoint key (saved by Trainer)
    # Fall back to UNet3D if not determinable
    try:
        from src.models.baseline import UNet3D
        model = UNet3D()
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception:
        from src.models.hybrid import HybridNet
        model = HybridNet()
        model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/eval_results.json")
    args = parser.parse_args()

    fold_map = {"val": [8], "test": [9]}
    folds = fold_map[args.split]

    ds = LUNA16Dataset(data_dir=args.data_dir, folds=folds, patch_size=256, augment=False)
    loader = DataLoader(ds, batch_size=1, num_workers=4)

    model = load_model(args.checkpoint, args.device)
    ann_df = pd.read_csv(f"{args.data_dir}/annotations.csv")

    pred_list = []
    dice_scores = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(args.device)
            out = model(images)
            prob = out["seg"].cpu().numpy()[0, 0]  # (D, H, W)
            mask = batch["mask"].numpy()[0, 0]

            dice_scores.append(dice_coefficient(prob, mask))

            pred_list.append({
                "seriesuid": batch["seriesuid"][0],
                "prob": float(prob.max()),
                "coord_xyz": batch["coord_xyz"][0].numpy(),
            })

    froc = compute_froc(pred_list, ann_df)
    results = {
        "split": args.split,
        "cpm": froc["cpm"],
        "sensitivity_at_fps": dict(zip([str(f) for f in froc["fps"]], froc["sensitivity"])),
        "mean_dice": float(np.mean(dice_scores)),
        "checkpoint": args.checkpoint,
    }

    print(json.dumps(results, indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    log.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
