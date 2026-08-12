from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from .metrics import binary_ood_metrics
from .reference_scores import ShrinkageMahalanobis, knn_cosine_normality
from .scores import feature_norm_normality, output_normality_scores
from .single_signal import bootstrap_intervals, development_indices


def _path_key(value: object) -> str:
    return os.path.normcase(os.path.abspath(str(value)))


def _cache_manifest_rows(
    cache: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    purpose: str,
) -> pd.DataFrame:
    selected = manifest[manifest["purpose"] == purpose].copy()
    selected_by_path = {_path_key(path): index for index, path in selected["local_path"].items()}
    keys = [_path_key(path) for path in cache["paths"]]
    if set(keys) != set(selected_by_path):
        raise ValueError(f"{purpose} cache paths do not match extracted sample manifest")
    return selected.loc[[selected_by_path[key] for key in keys]].reset_index(drop=True)


def validation_threshold(scores: np.ndarray, target_acceptance: float) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not 0.0 < target_acceptance < 1.0:
        raise ValueError("target_acceptance must be strictly between 0 and 1")
    if not np.isfinite(values).all() or not len(values):
        raise ValueError("validation scores must be finite and non-empty")
    index = min(len(values) - 1, int(np.floor((1.0 - target_acceptance) * len(values))))
    return float(np.sort(values)[index])


def _signal_report(
    n_scores: np.ndarray,
    o_scores: np.ndarray,
    validation_scores: np.ndarray,
    t_scores: np.ndarray,
    *,
    target_n_tpr: float,
    target_validation_acceptance: float,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    metrics = binary_ood_metrics(n_scores, o_scores, target_tpr=target_n_tpr)
    threshold = validation_threshold(validation_scores, target_validation_acceptance)
    return {
        "auroc": metrics.auroc,
        "fpr95": metrics.fpr95_first_ge,
        "fpr95_interpolated": metrics.fpr95_interpolated,
        **bootstrap_intervals(
            n_scores,
            o_scores,
            target_tpr=target_n_tpr,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "validation_threshold": threshold,
        "validation_acceptance": float((validation_scores >= threshold).mean()),
        "t_false_rejection_rate": float((t_scores < threshold).mean()),
        "development_n_acceptance_at_validation_threshold": float((n_scores >= threshold).mean()),
        "development_o_pass_rate_at_validation_threshold": float((o_scores >= threshold).mean()),
    }


def evaluate_reference_signals(
    ood_cache: dict[str, np.ndarray],
    split_assignments: pd.DataFrame,
    reference_cache: dict[str, np.ndarray],
    validation_cache: dict[str, np.ndarray],
    t_cache: dict[str, np.ndarray],
    extracted_manifest: pd.DataFrame,
    *,
    reference_sizes: tuple[int, ...] = (8000, 32000, 80000),
    k: int = 10,
    target_n_tpr: float = 0.95,
    target_validation_acceptance: float = 0.99,
    bootstrap_iterations: int = 2000,
    seed: int = 20260812,
    device: str = "auto",
) -> dict[str, Any]:
    all_development = development_indices(ood_cache["paths"], split_assignments)
    development = development_indices(
        ood_cache["paths"],
        split_assignments,
        exclude_cross_label_conflicts=True,
    )
    development_subsets = np.asarray(ood_cache["subsets"])[development].astype(str)
    development_features = np.asarray(ood_cache["features"])[development]
    development_logits = np.asarray(ood_cache["logits"])[development]
    n_mask = development_subsets == "N"
    o_mask = development_subsets == "O"
    if not n_mask.any() or not o_mask.any():
        raise ValueError("Development split must contain both N and O")

    reference_rows = _cache_manifest_rows(reference_cache, extracted_manifest, "reference")
    validation_rows = _cache_manifest_rows(
        validation_cache, extracted_manifest, "validation_threshold"
    )
    t_rows = _cache_manifest_rows(t_cache, extracted_manifest, "t_guardrail")
    reference_features = np.asarray(reference_cache["features"])
    reference_labels = reference_rows["label"].to_numpy(dtype=np.int64)
    validation_features = np.asarray(validation_cache["features"])
    t_features = np.asarray(t_cache["features"])
    signals: dict[str, Any] = {}

    output_development = output_normality_scores(development_logits)
    output_validation = output_normality_scores(np.asarray(validation_cache["logits"]))
    output_t = output_normality_scores(np.asarray(t_cache["logits"]))
    output_development["feature_norm"] = feature_norm_normality(development_features)
    output_validation["feature_norm"] = feature_norm_normality(validation_features)
    output_t["feature_norm"] = feature_norm_normality(t_features)
    for signal_index, name in enumerate(output_development):
        signals[name] = _signal_report(
            output_development[name][n_mask],
            output_development[name][o_mask],
            output_validation[name],
            output_t[name],
            target_n_tpr=target_n_tpr,
            target_validation_acceptance=target_validation_acceptance,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed + signal_index,
        )

    for bank_index, size in enumerate(reference_sizes):
        bank_mask = reference_rows["class_sample_rank"].to_numpy(dtype=np.int64) < size // 16
        if int(bank_mask.sum()) != size:
            raise ValueError(f"Reference manifest did not produce the requested nested bank {size}")
        bank_features = reference_features[bank_mask]
        bank_labels = reference_labels[bank_mask]

        knn_development = knn_cosine_normality(
            bank_features,
            development_features,
            k=k,
            device=device,
        )
        knn_validation = knn_cosine_normality(
            bank_features,
            validation_features,
            k=k,
            device=device,
        )
        knn_t = knn_cosine_normality(bank_features, t_features, k=k, device=device)
        signals[f"knn_k{k}_bank_{size}"] = _signal_report(
            knn_development[n_mask],
            knn_development[o_mask],
            knn_validation,
            knn_t,
            target_n_tpr=target_n_tpr,
            target_validation_acceptance=target_validation_acceptance,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed + 100 + bank_index,
        )

        mahalanobis = ShrinkageMahalanobis().fit(bank_features, bank_labels)
        mahalanobis_development = mahalanobis.normality(development_features)
        mahalanobis_validation = mahalanobis.normality(validation_features)
        mahalanobis_t = mahalanobis.normality(t_features)
        signals[f"mahalanobis_bank_{size}"] = _signal_report(
            mahalanobis_development[n_mask],
            mahalanobis_development[o_mask],
            mahalanobis_validation,
            mahalanobis_t,
            target_n_tpr=target_n_tpr,
            target_validation_acceptance=target_validation_acceptance,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed + 200 + bank_index,
        )

    t_predictions = np.asarray(t_cache["logits"]).argmax(axis=1)
    validation_predictions = np.asarray(validation_cache["logits"]).argmax(axis=1)
    return {
        "scope": "development_only_with_rvl_guardrails",
        "sealed_metrics_computed": False,
        "reference_sizes": [int(value) for value in reference_sizes],
        "knn_k": int(k),
        "target_n_tpr_for_fpr95": float(target_n_tpr),
        "target_validation_acceptance": float(target_validation_acceptance),
        "counts": {
            "development_N": int(n_mask.sum()),
            "development_O": int(o_mask.sum()),
            "reference": int(len(reference_rows)),
            "validation": int(len(validation_rows)),
            "T_guardrail": int(len(t_rows)),
            "cross_label_conflict_rows_excluded": int(
                len(all_development) - len(development)
            ),
        },
        "cross_label_conflicts_excluded": True,
        "classifier_sanity": {
            "validation_accuracy": float(
                (validation_predictions == validation_rows["label"].to_numpy()).mean()
            ),
            "t_accuracy": float((t_predictions == t_rows["label"].to_numpy()).mean()),
            "note": "Sanity check only; downstream classification accuracy is not the Gate target.",
        },
        "signals": signals,
    }
