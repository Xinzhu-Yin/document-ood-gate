from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class BinaryOODMetrics:
    auroc: float
    fpr95_interpolated: float
    fpr95_first_ge: float
    n_count: int
    o_count: int


def fpr_at_tpr(
    labels: np.ndarray,
    normality_scores: np.ndarray,
    target_tpr: float = 0.95,
) -> tuple[float, float]:
    """Return interpolated and first-achieved FPR at the target ID TPR.

    Labels must be 1 for the pass/ID side (N) and 0 for the reject/OOD side (O).
    Higher scores must mean more normal. Both common FPR95 conventions are retained
    because the 2022 paper did not publish its evaluation code.
    """

    y = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(normality_scores, dtype=np.float64)
    if y.shape != scores.shape or y.ndim != 1:
        raise ValueError("labels and normality_scores must be same-length 1D arrays")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("labels must contain both 0 (O) and 1 (N)")
    if not np.isfinite(scores).all():
        raise ValueError("normality_scores contain NaN or infinity")

    fpr, tpr, _ = roc_curve(y, scores, pos_label=1, drop_intermediate=False)
    interpolated = float(np.interp(target_tpr, tpr, fpr))
    eligible = np.flatnonzero(tpr >= target_tpr)
    first_ge = float(fpr[eligible[0]]) if eligible.size else 1.0
    return interpolated, first_ge


def binary_ood_metrics(
    n_scores: np.ndarray,
    o_scores: np.ndarray,
    target_tpr: float = 0.95,
) -> BinaryOODMetrics:
    n = np.asarray(n_scores, dtype=np.float64).reshape(-1)
    o = np.asarray(o_scores, dtype=np.float64).reshape(-1)
    labels = np.concatenate([np.ones(n.size, dtype=np.int8), np.zeros(o.size, dtype=np.int8)])
    scores = np.concatenate([n, o])
    interpolated, first_ge = fpr_at_tpr(labels, scores, target_tpr)
    return BinaryOODMetrics(
        auroc=float(roc_auc_score(labels, scores)),
        fpr95_interpolated=interpolated,
        fpr95_first_ge=first_ge,
        n_count=int(n.size),
        o_count=int(o.size),
    )


def macro_auroc_by_predicted_class(
    n_scores: np.ndarray,
    o_scores: np.ndarray,
    n_predicted: np.ndarray,
    o_predicted: np.ndarray,
) -> tuple[float, dict[int, float]]:
    """Match the paper's macro AUROC: average within predicted-label buckets."""

    per_class: dict[int, float] = {}
    classes = sorted(set(np.asarray(n_predicted).tolist()) | set(np.asarray(o_predicted).tolist()))
    for class_id in classes:
        n_mask = np.asarray(n_predicted) == class_id
        o_mask = np.asarray(o_predicted) == class_id
        if not n_mask.any() or not o_mask.any():
            continue
        per_class[int(class_id)] = binary_ood_metrics(
            np.asarray(n_scores)[n_mask], np.asarray(o_scores)[o_mask]
        ).auroc
    if not per_class:
        raise ValueError("No predicted class contains both N and O samples")
    return float(np.mean(list(per_class.values()))), per_class


def macro_auroc_released_code(
    n_scores: np.ndarray,
    o_scores: np.ndarray,
    n_true_labels: np.ndarray,
    o_predicted_labels: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Reproduce the released script: N true-label buckets versus O predicted buckets.

    This differs from the paper text, which says both sides are grouped by predicted
    class. Both definitions are retained so the discrepancy remains visible.
    """

    n_values = np.asarray(n_scores)
    o_values = np.asarray(o_scores)
    n_labels = np.asarray(n_true_labels).astype(str)
    o_labels = np.asarray(o_predicted_labels).astype(str)
    per_class: dict[str, float] = {}
    for label in sorted(set(o_labels.tolist())):
        n_mask = n_labels == label
        o_mask = o_labels == label
        if not n_mask.any() or not o_mask.any():
            continue
        per_class[label] = binary_ood_metrics(n_values[n_mask], o_values[o_mask]).auroc
    if not per_class:
        raise ValueError("No released-code class contains both N and O samples")
    return float(np.mean(list(per_class.values()))), per_class
