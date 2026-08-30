from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .layer_evaluation import _evaluate_feature_bank, exact_vim_decomposition
from .metrics import binary_ood_metrics
from .reference_scores import ShrinkageMahalanobis, knn_cosine_normality
from .scores import output_normality_scores
from .single_signal import bootstrap_intervals


FEATURE_DIM = 768
LAYERS = tuple(range(1, 13))
PRIMARY_D = 512
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260812


def _finite_features(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM:
        raise ValueError(f"{name} must have shape [rows, {FEATURE_DIM}]")
    if not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite numeric values")
    return matrix


def layer_norm_free(values: np.ndarray, *, eps: float) -> np.ndarray:
    """Apply per-sample LayerNorm geometry without learned affine parameters."""

    matrix = _finite_features(values, name="values").astype(np.float32, copy=False)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    mean = matrix.mean(axis=1, keepdims=True, dtype=np.float32)
    centered = matrix - mean
    variance = np.mean(centered * centered, axis=1, keepdims=True, dtype=np.float32)
    result = centered / np.sqrt(variance + np.float32(eps))
    return np.asarray(result, dtype=np.float32)


def layer_norm_pooler(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    """Apply the checkpoint's final pooler LayerNorm to any pooled representation."""

    normalized = layer_norm_free(values, eps=eps)
    scale = np.asarray(weight, dtype=np.float32)
    offset = np.asarray(bias, dtype=np.float32)
    if scale.shape != (FEATURE_DIM,) or offset.shape != (FEATURE_DIM,):
        raise ValueError("pooler LayerNorm weight and bias must each contain 768 values")
    result = normalized * scale + offset
    if not np.isfinite(result).all():
        raise ValueError("LN-pooler output contains NaN or infinity")
    return np.asarray(result, dtype=np.float32)


def _flatten_metric_row(row: dict[str, Any], *, split: str, principal_dim: int) -> dict[str, Any]:
    auroc_ci = row.pop("auroc_95ci")
    fpr95_ci = row.pop("fpr95_95ci")
    return {
        **row,
        "split": split,
        "D": int(principal_dim) if "residual" in str(row["method"]) else None,
        "k": 10 if row["method"] == "knn" else None,
        "auroc_ci_low": float(auroc_ci[0]),
        "auroc_ci_high": float(auroc_ci[1]),
        "fpr95_ci_low": float(fpr95_ci[0]),
        "fpr95_ci_high": float(fpr95_ci[1]),
    }


def _metric_values(
    scores: np.ndarray,
    is_n: np.ndarray,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(is_n, dtype=bool)
    if values.shape != mask.shape or not mask.any() or mask.all():
        raise ValueError("scores and is_n must align and contain both N and O")
    metrics = binary_ood_metrics(values[mask], values[~mask], target_tpr=0.95)
    intervals = bootstrap_intervals(
        values[mask],
        values[~mask],
        target_tpr=0.95,
        iterations=bootstrap_iterations,
        seed=seed,
    )
    return {
        "auroc": float(metrics.auroc),
        "auroc_ci_low": float(intervals["auroc_95ci"][0]),
        "auroc_ci_high": float(intervals["auroc_95ci"][1]),
        "fpr95": float(metrics.fpr95_first_ge),
        "fpr95_ci_low": float(intervals["fpr95_95ci"][0]),
        "fpr95_ci_high": float(intervals["fpr95_95ci"][1]),
    }


def evaluate_normalization_grid(
    *,
    reference_raw: np.ndarray,
    query_raw: np.ndarray,
    reference_labels: np.ndarray,
    query_is_n: np.ndarray,
    layernorm_weight: np.ndarray,
    layernorm_bias: np.ndarray,
    layernorm_eps: float,
    split: str,
    experiment: str,
    k: int = 10,
    principal_dim: int = PRIMARY_D,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    knn_device: str = "auto",
) -> list[dict[str, Any]]:
    """Evaluate raw, LN-free, and LN-pooler at all 12 layers with a fixed 32k bank."""

    reference = np.asarray(reference_raw)
    query = np.asarray(query_raw)
    if reference.shape[1:] != (12, FEATURE_DIM) or query.shape[1:] != (12, FEATURE_DIM):
        raise ValueError("raw caches must have shape [rows, 12, 768]")
    labels = np.asarray(reference_labels)
    mask = np.asarray(query_is_n)
    if labels.shape != (len(reference),) or mask.shape != (len(query),):
        raise ValueError("labels/mask must align with reference/query features")

    rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(LAYERS):
        raw_reference = np.asarray(reference[:, layer_index, :], dtype=np.float32)
        raw_query = np.asarray(query[:, layer_index, :], dtype=np.float32)
        representations = (
            ("raw_patch_mean", raw_reference, raw_query),
            (
                "ln_free",
                layer_norm_free(raw_reference, eps=layernorm_eps),
                layer_norm_free(raw_query, eps=layernorm_eps),
            ),
            (
                "ln_pooler",
                layer_norm_pooler(
                    raw_reference,
                    layernorm_weight,
                    layernorm_bias,
                    eps=layernorm_eps,
                ),
                layer_norm_pooler(
                    raw_query,
                    layernorm_weight,
                    layernorm_bias,
                    eps=layernorm_eps,
                ),
            ),
        )
        for representation, reference_features, query_features in representations:
            evaluated = _evaluate_feature_bank(
                reference_features,
                labels,
                query_features,
                mask,
                experiment=experiment,
                layer=layer,
                representation=representation,
                bank_role="primary_32k",
                k=k,
                principal_dim=principal_dim,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
                knn_device=knn_device,
            )
            rows.extend(
                _flatten_metric_row(dict(row), split=split, principal_dim=principal_dim)
                for row in evaluated
            )
    return rows


def _pca_residual_scores_by_d(
    reference_features: np.ndarray,
    query_features: np.ndarray,
    dimensions: Iterable[int],
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    reference = _finite_features(reference_features, name="reference_features").astype(
        np.float64, copy=False
    )
    query = _finite_features(query_features, name="query_features").astype(
        np.float64, copy=False
    )
    dims = tuple(int(value) for value in dimensions)
    if len(set(dims)) != len(dims) or any(value <= 0 or value >= FEATURE_DIM for value in dims):
        raise ValueError("dimensions must be unique integers between 1 and 767")
    origin = reference.mean(axis=0)
    centered_reference = reference - origin
    covariance = (centered_reference.T @ centered_reference) / len(centered_reference)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total_variance = float(eigenvalues.sum())
    if not np.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError("reference features must have positive total variance")
    projected_query = (query - origin) @ eigenvectors
    scores: dict[int, np.ndarray] = {}
    explained: dict[int, float] = {}
    for dimension in dims:
        residual_dim = FEATURE_DIM - dimension
        scores[dimension] = -np.linalg.norm(projected_query[:, :residual_dim], axis=1)
        explained[dimension] = float(eigenvalues[-dimension:].sum() / total_variance)
    return scores, explained


def evaluate_residual_d_sensitivity(
    *,
    reference_raw: np.ndarray,
    query_raw: np.ndarray,
    query_is_n: np.ndarray,
    layers: tuple[int, ...] = (3, 7, 9, 11),
    dimensions: tuple[int, ...] = (128, 256, 384, 512),
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Run the fixed development-only diagnostic D sensitivity panel."""

    reference = np.asarray(reference_raw)
    query = np.asarray(query_raw)
    if reference.shape[1:] != (12, FEATURE_DIM) or query.shape[1:] != (12, FEATURE_DIM):
        raise ValueError("raw caches must have shape [rows, 12, 768]")
    if tuple(layers) != (3, 7, 9, 11) or tuple(dimensions) != (128, 256, 384, 512):
        raise ValueError("the diagnostic layer/D grid is frozen")
    rows: list[dict[str, Any]] = []
    for layer in layers:
        scores_by_d, explained_by_d = _pca_residual_scores_by_d(
            reference[:, layer - 1, :], query[:, layer - 1, :], dimensions
        )
        for dimension in dimensions:
            rows.append(
                {
                    "experiment": "B",
                    "split": "development",
                    "layer": layer,
                    "representation": "raw_patch_mean",
                    "method": "mean_centered_pca_residual",
                    "residual_centering_convention": "reference_mean",
                    "bank_role": "primary_32k",
                    "bank_size": int(len(reference)),
                    "D": dimension,
                    "k": None,
                    **_metric_values(
                        scores_by_d[dimension],
                        query_is_n,
                        bootstrap_iterations=bootstrap_iterations,
                        seed=seed,
                    ),
                    "top_d_explained_variance_ratio": explained_by_d[dimension],
                    "bootstrap_iterations": bootstrap_iterations,
                    "bootstrap_seed": seed,
                }
            )
    return rows


def evaluate_sealed_baselines(
    *,
    reference_h_doc: np.ndarray,
    reference_logits: np.ndarray,
    reference_labels: np.ndarray,
    sealed_h_doc: np.ndarray,
    sealed_logits: np.ndarray,
    sealed_is_n: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    k: int = 10,
    principal_dim: int = PRIMARY_D,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    knn_device: str = "auto",
) -> list[dict[str, Any]]:
    """Evaluate the frozen D1 baseline panel on the once-opened sealed features."""

    output_scores = output_normality_scores(sealed_logits)
    vim_scores = exact_vim_decomposition(
        reference_h_doc,
        reference_logits,
        sealed_h_doc,
        sealed_logits,
        classifier_weight,
        classifier_bias,
        principal_dim=principal_dim,
    )
    scores = {
        "msp": output_scores["msp"],
        "energy": output_scores["energy"],
        "mahalanobis": ShrinkageMahalanobis()
        .fit(reference_h_doc, reference_labels)
        .normality(sealed_h_doc),
        "knn": knn_cosine_normality(
            reference_h_doc, sealed_h_doc, k=k, device=knn_device
        ),
        "vim_origin_residual": vim_scores["vim_origin_residual"],
        "vim": vim_scores["vim"],
    }
    rows: list[dict[str, Any]] = []
    for method in (
        "msp",
        "energy",
        "mahalanobis",
        "knn",
        "vim_origin_residual",
        "vim",
    ):
        is_output = method in {"msp", "energy"}
        is_vim = method in {"vim_origin_residual", "vim"}
        rows.append(
            {
                "experiment": "D1",
                "split": "sealed_test",
                "layer": None if is_output else 12,
                "representation": "logits" if is_output else "h_doc",
                "method": method,
                "residual_centering_convention": (
                    "vim_classifier_origin" if is_vim else None
                ),
                "bank_role": "primary_32k" if not is_output else None,
                "bank_size": int(len(reference_h_doc)) if not is_output else None,
                "D": principal_dim if is_vim else None,
                "k": k if method == "knn" else None,
                **_metric_values(
                    scores[method],
                    sealed_is_n,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed,
                ),
                "bootstrap_iterations": bootstrap_iterations,
                "bootstrap_seed": seed,
            }
        )
    return rows


def evaluate_replication_predictions(
    baseline_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate frozen P1-P5 without changing their wording or thresholds."""

    def select(rows: list[dict[str, Any]], **criteria: Any) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if all(row.get(key) == expected for key, expected in criteria.items())
        ]

    primary_methods = {"mahalanobis", "knn", "mean_centered_pca_residual"}
    p1_rows = [
        row
        for row in grid_rows
        if row["representation"] == "raw_patch_mean"
        and row["layer"] in {2, 3, 4, 5}
        and row["method"] in primary_methods
    ]
    if len(p1_rows) != 12:
        raise ValueError("P1 requires exactly 12 raw Layer 2-5 primary-probe rows")

    p2_free = select(grid_rows, layer=12, representation="ln_free", method="mahalanobis")
    p2_raw = select(
        grid_rows, layer=12, representation="raw_patch_mean", method="mahalanobis"
    )
    p3_rows = select(baseline_rows, representation="h_doc", method="mahalanobis")
    if len(p2_free) != 1 or len(p2_raw) != 1 or len(p3_rows) != 1:
        raise ValueError("P2/P3 require unique frozen result rows")
    p2_delta = float(p2_free[0]["auroc"] - p2_raw[0]["auroc"])
    p3_value = float(p3_rows[0]["auroc"])

    all_rows = [*baseline_rows, *grid_rows]
    best_auroc = max(float(row["auroc"]) for row in all_rows)
    lowest_fpr95 = min(float(row["fpr95"]) for row in all_rows)
    best_rows = [row for row in all_rows if np.isclose(row["auroc"], best_auroc)]
    lowest_rows = [row for row in all_rows if np.isclose(row["fpr95"], lowest_fpr95)]

    def identity(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row["experiment"], row["layer"], row["representation"], row["method"])

    best_ids = {identity(row) for row in best_rows}
    lowest_ids = {identity(row) for row in lowest_rows}
    return [
        {
            "prediction": "P1",
            "status": "supported" if max(float(row["auroc"]) for row in p1_rows) < 0.60 else "not supported",
            "observed": {"maximum_auroc_across_12_rows": max(float(row["auroc"]) for row in p1_rows)},
        },
        {
            "prediction": "P2",
            "status": "supported" if p2_delta >= 0.05 else "not supported",
            "observed": {"l12_ln_free_minus_raw_mahalanobis_auroc": p2_delta},
        },
        {
            "prediction": "P3",
            "status": "supported" if 0.71 <= p3_value <= 0.78 else "not supported",
            "observed": {"h_doc_mahalanobis_32k_auroc": p3_value},
        },
        {
            "prediction": "P4",
            "status": "supported" if all(float(row["fpr95"]) > 0.75 for row in best_rows) else "not supported",
            "observed": {
                "highest_auroc": best_auroc,
                "fpr95_at_highest_auroc": [float(row["fpr95"]) for row in best_rows],
                "configurations": [identity(row) for row in best_rows],
            },
        },
        {
            "prediction": "P5",
            "status": "supported" if best_ids.isdisjoint(lowest_ids) else "not supported",
            "observed": {
                "highest_auroc": best_auroc,
                "lowest_fpr95": lowest_fpr95,
                "highest_auroc_configurations": [identity(row) for row in best_rows],
                "lowest_fpr95_configurations": [identity(row) for row in lowest_rows],
            },
        },
    ]
