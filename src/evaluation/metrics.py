"""Segmentation metrics: Dice coefficient and sensitivity."""

from __future__ import annotations

import numpy as np


def dice_coefficient(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Binary Dice coefficient.

    Args:
        pred: float array, probabilities in [0, 1]
        target: binary array
        threshold: binarisation threshold for pred

    Returns:
        Dice score in [0, 1]
    """
    pred_bin = (pred >= threshold).astype(np.float32)
    target_bin = target.astype(np.float32)
    intersection = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return float(2.0 * intersection / (denom + 1e-8))


def sensitivity_at_specificity(
    y_true: np.ndarray, y_score: np.ndarray, specificity: float = 0.95
) -> tuple[float, float]:
    """Find (sensitivity, threshold) at a given specificity.

    Useful for tuning FP classifier threshold on validation fold.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    spec = 1.0 - fpr
    idx = np.argmin(np.abs(spec - specificity))
    return float(tpr[idx]), float(thresholds[idx])
