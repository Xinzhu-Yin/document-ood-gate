from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .metrics import binary_ood_metrics
from .scores import feature_norm_normality, output_normality_scores


def _path_key(value: object) -> str:
    return os.path.normcase(os.path.abspath(str(value)))


def development_indices(
    cache_paths: np.ndarray,
    assignments: pd.DataFrame,
    *,
    exclude_cross_label_conflicts: bool = False,
) -> np.ndarray:
    required = {"path", "split"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"Split assignments are missing columns: {', '.join(missing)}")
    if assignments["path"].duplicated().any():
        raise ValueError("Split assignments contain duplicate paths")

    assignment_by_path = {
        _path_key(path): str(split)
        for path, split in zip(assignments["path"], assignments["split"], strict=True)
    }
    cache_keys = [_path_key(path) for path in cache_paths]
    if len(set(cache_keys)) != len(cache_keys):
        raise ValueError("Feature cache contains duplicate paths")
    if set(cache_keys) != set(assignment_by_path):
        missing_from_split = len(set(cache_keys) - set(assignment_by_path))
        missing_from_cache = len(set(assignment_by_path) - set(cache_keys))
        raise ValueError(
            "Feature cache and split path sets differ: "
            f"cache_only={missing_from_split}, split_only={missing_from_cache}"
        )
    excluded_paths: set[str] = set()
    if exclude_cross_label_conflicts:
        conflict_required = {"subset", "split_group_id"}
        conflict_missing = sorted(conflict_required - set(assignments.columns))
        if conflict_missing:
            raise ValueError(
                "Cannot exclude cross-label conflicts; assignments are missing: "
                + ", ".join(conflict_missing)
            )
        conflict_groups = set(
            assignments.groupby("split_group_id", observed=True)["subset"]
            .nunique()
            .loc[lambda values: values > 1]
            .index
        )
        excluded_paths = {
            _path_key(path)
            for path in assignments.loc[
                assignments["split_group_id"].isin(conflict_groups), "path"
            ]
        }
    return np.asarray(
        [
            index
            for index, key in enumerate(cache_keys)
            if assignment_by_path[key] == "development" and key not in excluded_paths
        ],
        dtype=np.int64,
    )


def _operating_point(
    n_scores: np.ndarray,
    o_scores: np.ndarray,
    target_tpr: float,
) -> dict[str, float]:
    n = np.asarray(n_scores, dtype=np.float64)
    o = np.asarray(o_scores, dtype=np.float64)
    labels = np.concatenate([np.ones(len(n), dtype=np.int8), np.zeros(len(o), dtype=np.int8)])
    scores = np.concatenate([n, o])
    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
        drop_intermediate=False,
    )
    eligible = np.flatnonzero(tpr >= target_tpr)
    index = int(eligible[0]) if eligible.size else len(thresholds) - 1
    threshold = float(thresholds[index])
    return {
        "threshold": threshold,
        "n_acceptance": float((n >= threshold).mean()),
        "o_pass_rate": float((o >= threshold).mean()),
    }


def bootstrap_intervals(
    n_scores: np.ndarray,
    o_scores: np.ndarray,
    *,
    target_tpr: float,
    iterations: int,
    seed: int,
) -> dict[str, list[float]]:
    if iterations < 1:
        raise ValueError("bootstrap iterations must be at least 1")
    n = np.asarray(n_scores, dtype=np.float64)
    o = np.asarray(o_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    auroc = np.empty(iterations, dtype=np.float64)
    fpr95 = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        n_sample = n[rng.integers(0, len(n), size=len(n))]
        o_sample = o[rng.integers(0, len(o), size=len(o))]
        labels = np.concatenate(
            [np.ones(len(n_sample), dtype=np.int8), np.zeros(len(o_sample), dtype=np.int8)]
        )
        values = np.concatenate([n_sample, o_sample])
        auroc[iteration] = roc_auc_score(labels, values)
        fpr, tpr, _ = roc_curve(labels, values, pos_label=1, drop_intermediate=False)
        eligible = np.flatnonzero(tpr >= target_tpr)
        fpr95[iteration] = fpr[eligible[0]] if eligible.size else 1.0
    return {
        "auroc_95ci": [float(value) for value in np.quantile(auroc, [0.025, 0.975])],
        "fpr95_95ci": [float(value) for value in np.quantile(fpr95, [0.025, 0.975])],
    }


def evaluate_development_single_signals(
    cache: dict[str, np.ndarray],
    assignments: pd.DataFrame,
    *,
    target_tpr: float = 0.95,
    bootstrap_iterations: int = 2000,
    seed: int = 20260812,
    exclude_cross_label_conflicts: bool = True,
) -> dict[str, Any]:
    required = {"paths", "subsets", "logits", "features"}
    missing = sorted(required - set(cache))
    if missing:
        raise ValueError(f"Feature cache is missing arrays: {', '.join(missing)}")
    all_development = development_indices(cache["paths"], assignments)
    indices = development_indices(
        cache["paths"],
        assignments,
        exclude_cross_label_conflicts=exclude_cross_label_conflicts,
    )
    subsets = np.asarray(cache["subsets"])[indices].astype(str)
    if set(subsets) != {"N", "O"}:
        raise ValueError("Development split must contain both N and O")
    logits = np.asarray(cache["logits"])[indices]
    features = np.asarray(cache["features"])[indices]
    signals = output_normality_scores(logits)
    signals["feature_norm"] = feature_norm_normality(features)

    results: dict[str, Any] = {}
    for signal_index, (name, values) in enumerate(signals.items()):
        n_scores = np.asarray(values)[subsets == "N"]
        o_scores = np.asarray(values)[subsets == "O"]
        metrics = binary_ood_metrics(n_scores, o_scores, target_tpr=target_tpr)
        result = {
            "auroc": metrics.auroc,
            "fpr95": metrics.fpr95_first_ge,
            "fpr95_interpolated": metrics.fpr95_interpolated,
            **_operating_point(n_scores, o_scores, target_tpr),
            **bootstrap_intervals(
                n_scores,
                o_scores,
                target_tpr=target_tpr,
                iterations=bootstrap_iterations,
                seed=seed + signal_index,
            ),
        }
        results[name] = result

    return {
        "scope": "development_only",
        "sealed_metrics_computed": False,
        "seed": int(seed),
        "bootstrap_iterations": int(bootstrap_iterations),
        "target_n_acceptance": float(target_tpr),
        "counts": {
            "N": int((subsets == "N").sum()),
            "O": int((subsets == "O").sum()),
            "total": int(len(subsets)),
            "cross_label_conflict_rows_excluded": int(len(all_development) - len(indices)),
        },
        "cross_label_conflicts_excluded": bool(exclude_cross_label_conflicts),
        "signals": results,
    }
