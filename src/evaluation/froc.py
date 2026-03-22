"""FROC evaluation for LUNA16.

Standard LUNA16 metric:
  CPM = mean sensitivity at FP/scan ∈ [0.125, 0.25, 0.5, 1, 2, 4, 8]

Matching rules (from LUNA16 challenge):
  - TP if prediction centroid is within diameter_mm / 2 of annotation centroid
  - Greedy matching by descending confidence score
  - Per-scan sensitivity, then average across scans
  - Never mix folds

Usage:
    result = compute_froc(pred_list, ann_df)
    # result = {"cpm": float, "sensitivity": list[float], "fps": list[float]}

Sanity checks:
    all-zeros predictions → CPM 0.0
    perfect predictions   → CPM 1.0
    sensitivity is non-decreasing with FP rate
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FROC_FPS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def compute_froc(
    pred_list: list[dict],
    ann_df: pd.DataFrame,
    fps_per_scan: list[float] | None = None,
) -> dict[str, object]:
    """Compute FROC curve and CPM.

    Args:
        pred_list: list of dicts with keys:
            seriesuid (str), prob (float), coord_xyz (np.ndarray shape [3])
        ann_df: DataFrame with columns: seriesuid, coordX, coordY, coordZ, diameter_mm
        fps_per_scan: FP/scan thresholds (default LUNA16 standard)

    Returns:
        {"cpm": float, "sensitivity": list[float], "fps": list[float]}
    """
    if fps_per_scan is None:
        fps_per_scan = FROC_FPS

    if not pred_list:
        return {"cpm": 0.0, "sensitivity": [0.0] * len(fps_per_scan), "fps": fps_per_scan}

    # ---- Group annotations by scan ----
    ann_by_scan: dict[str, list[dict]] = {}
    for _, row in ann_df.iterrows():
        uid = row["seriesuid"]
        ann_by_scan.setdefault(uid, []).append(
            {
                "coord": np.array([row["coordX"], row["coordY"], row["coordZ"]], dtype=np.float64),
                "radius": float(row["diameter_mm"]) / 2.0,
            }
        )

    scans_in_preds = {p["seriesuid"] for p in pred_list}
    all_scans = list(scans_in_preds | set(ann_by_scan.keys()))
    n_scans = len(all_scans)

    # ---- Sort predictions by descending confidence ----
    preds_sorted = sorted(pred_list, key=lambda p: p["prob"], reverse=True)

    # Track per-scan TP / FP counts as we descend the ranked list
    # Each annotation can only be matched once per scan
    matched: dict[str, list[bool]] = {uid: [False] * len(anns) for uid, anns in ann_by_scan.items()}
    tp_total = np.zeros(len(preds_sorted), dtype=float)
    fp_total = np.zeros(len(preds_sorted), dtype=float)

    total_anns = sum(len(v) for v in ann_by_scan.values())

    for i, pred in enumerate(preds_sorted):
        uid = pred["seriesuid"]
        coord = np.asarray(pred["coord_xyz"], dtype=np.float64)
        is_tp = False

        if uid in ann_by_scan:
            for j, ann in enumerate(ann_by_scan[uid]):
                if matched[uid][j]:
                    continue
                dist = float(np.linalg.norm(coord - ann["coord"]))
                if dist <= ann["radius"]:
                    matched[uid][j] = True
                    is_tp = True
                    break

        tp_total[i] = float(is_tp)
        fp_total[i] = float(not is_tp)

    cum_tp = np.cumsum(tp_total)
    cum_fp = np.cumsum(fp_total)

    sensitivity_curve = cum_tp / max(total_anns, 1)
    fps_per_scan_curve = cum_fp / max(n_scans, 1)

    # ---- Interpolate sensitivity at each FP/scan threshold ----
    sensitivities = []
    for threshold in fps_per_scan:
        # Find indices where fps_per_scan_curve ≤ threshold
        mask = fps_per_scan_curve <= threshold
        if mask.any():
            sensitivities.append(float(sensitivity_curve[mask][-1]))
        else:
            sensitivities.append(0.0)

    cpm = float(np.mean(sensitivities))

    # Sanity: sensitivity should be non-decreasing across FP/scan thresholds
    if not all(
        sensitivities[i] <= sensitivities[i + 1] + 1e-9 for i in range(len(sensitivities) - 1)
    ):
        log.warning("FROC sensitivity is not non-decreasing: %s — check prediction scores.", sensitivities)

    return {"cpm": cpm, "sensitivity": sensitivities, "fps": fps_per_scan}
