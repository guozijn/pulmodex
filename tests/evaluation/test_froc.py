"""Tests for FROC evaluation (sanity checks from spec)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.froc import compute_froc


def _make_ann_df(entries):
    """entries: list of (uid, x, y, z, diam)"""
    return pd.DataFrame(entries, columns=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])


def test_all_zeros_predictions():
    """All-zero probabilities → CPM 0.0."""
    ann_df = _make_ann_df([("uid1", 0.0, 0.0, 0.0, 6.0)])
    preds = [{"seriesuid": "uid1", "prob": 0.0, "coord_xyz": np.array([99.0, 99.0, 99.0])}]
    result = compute_froc(preds, ann_df)
    assert result["cpm"] == 0.0


def test_perfect_predictions():
    """Perfect predictions (TP at annotation centre, no FPs) → CPM 1.0."""
    ann_df = _make_ann_df([("uid1", 0.0, 0.0, 0.0, 6.0)])
    preds = [{"seriesuid": "uid1", "prob": 1.0, "coord_xyz": np.array([0.0, 0.0, 0.0])}]
    result = compute_froc(preds, ann_df)
    assert result["cpm"] == pytest.approx(1.0)


def test_sensitivity_non_decreasing():
    """Sensitivity must be non-decreasing across FP/scan thresholds."""
    ann_df = _make_ann_df([
        ("uid1", 0.0, 0.0, 0.0, 6.0),
        ("uid2", 10.0, 10.0, 10.0, 8.0),
    ])
    preds = [
        {"seriesuid": "uid1", "prob": 0.9, "coord_xyz": np.array([0.0, 0.0, 0.0])},
        {"seriesuid": "uid1", "prob": 0.3, "coord_xyz": np.array([50.0, 50.0, 50.0])},  # FP
        {"seriesuid": "uid2", "prob": 0.6, "coord_xyz": np.array([10.0, 10.0, 10.0])},
    ]
    result = compute_froc(preds, ann_df)
    sens = result["sensitivity"]
    for i in range(len(sens) - 1):
        assert sens[i] <= sens[i + 1] + 1e-9


def test_empty_preds():
    ann_df = _make_ann_df([("uid1", 0.0, 0.0, 0.0, 6.0)])
    result = compute_froc([], ann_df)
    assert result["cpm"] == 0.0


def test_match_within_radius():
    """Prediction just inside radius → TP; just outside → FP."""
    ann_df = _make_ann_df([("uid1", 0.0, 0.0, 0.0, 10.0)])  # radius = 5 mm

    inside = [{"seriesuid": "uid1", "prob": 1.0, "coord_xyz": np.array([4.9, 0.0, 0.0])}]
    assert compute_froc(inside, ann_df)["cpm"] == pytest.approx(1.0)

    outside = [{"seriesuid": "uid1", "prob": 1.0, "coord_xyz": np.array([5.1, 0.0, 0.0])}]
    assert compute_froc(outside, ann_df)["cpm"] == 0.0
