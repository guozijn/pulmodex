"""LUNA16 evaluation helpers for MONAI 3D detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.froc import compute_froc

from .infer import infer_detection_case
from .io import load_prepared_split, seriesuid_from_image_path


def _filter_excluded_predictions(
    predictions: list[dict[str, Any]],
    excluded_df: pd.DataFrame,
    default_radius_mm: float = 5.0,
) -> list[dict[str, Any]]:
    kept = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for _, row in excluded_df.iterrows():
        grouped.setdefault(str(row["seriesuid"]), []).append(
            {
                "coord": np.array([row["coordX"], row["coordY"], row["coordZ"]], dtype=np.float32),
                "radius": float(row["diameter_mm"]) / 2.0 if float(row["diameter_mm"]) > 0 else default_radius_mm,
            }
        )

    for pred in predictions:
        coord = np.array([pred["coordX"], pred["coordY"], pred["coordZ"]], dtype=np.float32)
        drop = False
        for excluded in grouped.get(pred["seriesuid"], []):
            if float(np.linalg.norm(coord - excluded["coord"])) <= excluded["radius"]:
                drop = True
                break
        if not drop:
            kept.append(pred)
    return kept


def evaluate_detection_model(
    detector: Any,
    fold: int = 0,
    prepared_dir: str | Path = "data/monai_detection_nifti_prepared",
    output_path: str | Path = "outputs/detection_eval_fold0.json",
    annotations_path: str | Path = "data/evaluationScript/annotations/annotations.csv",
    excluded_annotations_path: str | Path = "data/evaluationScript/annotations/annotations_excluded.csv",
    inference_output_dir: str | Path = "outputs/detection_eval_cases",
    device: str = "cpu",
    score_thresh: float = 0.15,
    target_spacing: float = 1.0,
) -> dict[str, Any]:
    split = load_prepared_split(fold, prepared_dir)
    validation_items = split["validation"]

    predictions: list[dict[str, Any]] = []
    for item in validation_items:
        report = infer_detection_case(
            detector=detector,
            image_path=item["image"],
            output_dir=inference_output_dir,
            device=device,
            score_thresh=score_thresh,
            target_spacing=target_spacing,
        )
        seriesuid = seriesuid_from_image_path(item["image"])
        for cand in report["candidates"]:
            predictions.append(
                {
                    "seriesuid": seriesuid,
                    "coordX": cand["coordX"],
                    "coordY": cand["coordY"],
                    "coordZ": cand["coordZ"],
                    "prob": cand["prob"],
                }
            )

    ann_df = pd.read_csv(annotations_path)
    ann_df = ann_df[ann_df["seriesuid"].isin([seriesuid_from_image_path(item["image"]) for item in validation_items])]
    excluded_df = pd.read_csv(excluded_annotations_path)
    predictions = _filter_excluded_predictions(predictions, excluded_df)
    pred_list = [
        {
            "seriesuid": pred["seriesuid"],
            "prob": pred["prob"],
            "coord_xyz": np.array([pred["coordX"], pred["coordY"], pred["coordZ"]], dtype=np.float32),
        }
        for pred in predictions
    ]
    froc = compute_froc(pred_list, ann_df)
    results = {
        "fold": int(fold),
        "checkpoint_type": "monai_detection",
        "num_validation_scans": len(validation_items),
        "num_predictions": len(predictions),
        "cpm": froc["cpm"],
        "sensitivity_at_fps": dict(zip([str(v) for v in froc["fps"]], froc["sensitivity"])),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    return results
