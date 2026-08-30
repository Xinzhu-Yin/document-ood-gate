from __future__ import annotations

from typing import Any

import numpy as np

from .layer_scores import MeanCenteredPCAResidual
from .metrics import binary_ood_metrics
from .reference_scores import ShrinkageMahalanobis, knn_cosine_normality
from .single_signal import bootstrap_intervals
from .supplementary_scores import ViMNormality, logsumexp_normality


LAYER_COUNT = 12
G2_TARGET_AUROC = 0.7416556
G2_TOLERANCE = 0.005


class G2ReproductionError(RuntimeError):
    """Raised before the layer study when the frozen Mahalanobis gate fails."""


def _finite_matrix(
    values: np.ndarray,
    *,
    name: str,
    rows: int | None = None,
    columns: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if rows is not None and len(matrix) != rows:
        raise ValueError(f"{name} must contain {rows} rows")
    if columns is not None and matrix.shape[1] != columns:
        raise ValueError(f"{name} must contain {columns} columns")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _layer_tensor(
    values: np.ndarray,
    *,
    name: str,
    rows: int,
    feature_dim: int,
) -> np.ndarray:
    tensor = np.asarray(values)
    expected = (LAYER_COUNT, rows, feature_dim)
    if tensor.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, found {tensor.shape}")
    if not np.issubdtype(tensor.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    return tensor


def _balanced_labels(values: np.ndarray, *, name: str, rows: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) != rows:
        raise ValueError(f"{name} must be a one-dimensional array with {rows} rows")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError(f"{name} must contain at least two classes")
    if len(set(counts.tolist())) != 1:
        raise ValueError(f"{name} must be class-balanced")
    return labels


def _metric_fields(
    scores: np.ndarray,
    development_is_n: np.ndarray,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(development_is_n):
        raise ValueError("normality scores must align with the development mask")
    if not np.isfinite(values).all():
        raise ValueError("normality scores must contain only finite values")
    n_scores = values[development_is_n]
    o_scores = values[~development_is_n]
    metrics = binary_ood_metrics(n_scores, o_scores, target_tpr=0.95)
    return {
        "auroc": metrics.auroc,
        "fpr95": metrics.fpr95_first_ge,
        **bootstrap_intervals(
            n_scores,
            o_scores,
            target_tpr=0.95,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
    }


def _evaluate_feature_bank(
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    development_features: np.ndarray,
    development_is_n: np.ndarray,
    *,
    experiment: str,
    layer: int,
    representation: str,
    bank_role: str,
    k: int,
    principal_dim: int,
    bootstrap_iterations: int,
    seed: int,
    knn_device: str,
) -> list[dict[str, Any]]:
    reference = _finite_matrix(reference_features, name="reference_features")
    development = _finite_matrix(
        development_features,
        name="development_features",
        columns=reference.shape[1],
    )
    if len(reference_labels) != len(reference):
        raise ValueError("reference labels and features must be aligned")

    residual = MeanCenteredPCAResidual(principal_dim=principal_dim).fit(reference)
    variance_ratio = float(residual.top_d_explained_variance_ratio_)
    method_scores = (
        (
            "mahalanobis",
            ShrinkageMahalanobis().fit(reference, reference_labels).normality(development),
            None,
        ),
        (
            "knn",
            knn_cosine_normality(
                reference,
                development,
                k=k,
                device=knn_device,
            ),
            None,
        ),
        (
            "mean_centered_pca_residual",
            residual.normality(development),
            "reference_mean",
        ),
    )

    rows: list[dict[str, Any]] = []
    for method, scores, centering in method_scores:
        rows.append(
            {
                "experiment": experiment,
                "layer": int(layer),
                "representation": representation,
                "method": method,
                "residual_centering_convention": centering,
                "bank_role": bank_role,
                "bank_size": int(len(reference)),
                **_metric_fields(
                    scores,
                    development_is_n,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed,
                ),
                # The frozen production setting is D=512. The value is repeated
                # across methods for the same representation/bank as a geometry
                # diagnostic; it is never used for model selection.
                "top_512_variance_ratio": variance_ratio,
                "bootstrap_iterations": int(bootstrap_iterations),
                "bootstrap_seed": int(seed),
            }
        )
    return rows


def exact_vim_decomposition(
    reference_h_doc: np.ndarray,
    reference_logits: np.ndarray,
    development_h_doc: np.ndarray,
    development_logits: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    *,
    principal_dim: int = 512,
) -> dict[str, np.ndarray]:
    """Return Energy, ViM-origin residual-only, and full ViM from one ViM fit."""

    vim = ViMNormality(principal_dim=principal_dim).fit(
        reference_h_doc,
        reference_logits,
        classifier_weight,
        classifier_bias,
    )
    energy = logsumexp_normality(development_logits)
    residual_only = vim.residual_normality(development_h_doc)
    full_vim = vim.normality(development_h_doc, development_logits)
    return {
        "energy": energy,
        "vim_origin_residual": residual_only,
        "vim": full_vim,
    }


def evaluate_layer_study(
    *,
    reference_raw_patch_mean: np.ndarray,
    reference_cls: np.ndarray,
    reference_h_doc: np.ndarray,
    reference_logits: np.ndarray,
    reference_labels: np.ndarray,
    reference_8k_indices: np.ndarray,
    reference_8k_labels: np.ndarray,
    development_raw_patch_mean: np.ndarray,
    development_cls: np.ndarray,
    development_h_doc: np.ndarray,
    development_logits: np.ndarray,
    development_is_n: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    k: int = 10,
    principal_dim: int = 512,
    bootstrap_iterations: int = 2000,
    seed: int = 20260812,
    g2_target_auroc: float = G2_TARGET_AUROC,
    g2_tolerance: float = G2_TOLERANCE,
    knn_device: str = "auto",
) -> dict[str, Any]:
    """Run the frozen G2, S1, S2, and S1b development-only array evaluation."""

    h_doc_reference = _finite_matrix(reference_h_doc, name="reference_h_doc")
    h_doc_development = _finite_matrix(
        development_h_doc,
        name="development_h_doc",
        columns=h_doc_reference.shape[1],
    )
    reference_count = len(h_doc_reference)
    development_count = len(h_doc_development)
    feature_dim = int(h_doc_reference.shape[1])
    if not 0 < principal_dim < feature_dim:
        raise ValueError("principal_dim must be between zero and the feature dimension")
    if not 1 <= k:
        raise ValueError("k must be positive")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")
    if not np.isfinite(g2_target_auroc) or not 0.0 <= g2_target_auroc <= 1.0:
        raise ValueError("g2_target_auroc must be between zero and one")
    if not np.isfinite(g2_tolerance) or g2_tolerance <= 0.0:
        raise ValueError("g2_tolerance must be finite and positive")

    logits_reference = _finite_matrix(
        reference_logits,
        name="reference_logits",
        rows=reference_count,
    )
    logits_development = _finite_matrix(
        development_logits,
        name="development_logits",
        rows=development_count,
        columns=logits_reference.shape[1],
    )
    weight = _finite_matrix(
        classifier_weight,
        name="classifier_weight",
        rows=logits_reference.shape[1],
        columns=feature_dim,
    )
    bias = np.asarray(classifier_bias)
    if bias.shape != (weight.shape[0],) or not np.issubdtype(bias.dtype, np.number):
        raise ValueError("classifier_bias must align with classifier_weight")
    if not np.isfinite(bias).all():
        raise ValueError("classifier_bias must contain only finite values")

    labels_32k = _balanced_labels(
        reference_labels,
        name="reference_labels",
        rows=reference_count,
    )
    indices_8k = np.asarray(reference_8k_indices)
    if indices_8k.ndim != 1 or not np.issubdtype(indices_8k.dtype, np.integer):
        raise ValueError("reference_8k_indices must be a one-dimensional integer array")
    if not len(indices_8k) or len(np.unique(indices_8k)) != len(indices_8k):
        raise ValueError("reference_8k_indices must be non-empty and unique")
    if indices_8k.min() < 0 or indices_8k.max() >= reference_count:
        raise ValueError("reference_8k_indices contain an out-of-range index")
    labels_8k = _balanced_labels(
        reference_8k_labels,
        name="reference_8k_labels",
        rows=len(indices_8k),
    )
    if not np.array_equal(labels_32k[indices_8k], labels_8k):
        raise ValueError("reference_8k_labels do not exactly match the indexed frozen 32k labels")
    if k > len(indices_8k):
        raise ValueError("k cannot exceed the 8k sensitivity-bank size")

    is_n = np.asarray(development_is_n)
    if is_n.dtype != np.bool_ or is_n.shape != (development_count,):
        raise ValueError("development_is_n must be a one-dimensional boolean mask")
    if not is_n.any() or is_n.all():
        raise ValueError("development_is_n must contain both N and O rows")

    raw_reference = _layer_tensor(
        reference_raw_patch_mean,
        name="reference_raw_patch_mean",
        rows=reference_count,
        feature_dim=feature_dim,
    )
    cls_reference = _layer_tensor(
        reference_cls,
        name="reference_cls",
        rows=reference_count,
        feature_dim=feature_dim,
    )
    raw_development = _layer_tensor(
        development_raw_patch_mean,
        name="development_raw_patch_mean",
        rows=development_count,
        feature_dim=feature_dim,
    )
    cls_development = _layer_tensor(
        development_cls,
        name="development_cls",
        rows=development_count,
        feature_dim=feature_dim,
    )

    # G2 is intentionally completed before any S1/S2/S1b estimator is fitted.
    h_doc_8k = h_doc_reference[indices_8k]
    g2_scores = ShrinkageMahalanobis().fit(h_doc_8k, labels_8k).normality(
        h_doc_development
    )
    g2_metrics = binary_ood_metrics(g2_scores[is_n], g2_scores[~is_n], target_tpr=0.95)
    g2_delta = float(g2_metrics.auroc - g2_target_auroc)
    if not abs(g2_delta) < g2_tolerance:
        raise G2ReproductionError(
            "G2 Mahalanobis-8k reproduction failed: "
            f"expected={g2_target_auroc:.7f}, observed={g2_metrics.auroc:.7f}, "
            f"delta={g2_delta:+.7f}, required_abs_delta<{g2_tolerance:.7f}"
        )

    rows: list[dict[str, Any]] = []
    s1_raw_rows: list[dict[str, Any]] = []
    s1_h_doc_32k_variance_ratio: float | None = None
    final_layer_representations = (
        ("raw_patch_mean", raw_reference[-1], raw_development[-1]),
        ("h_doc", h_doc_reference, h_doc_development),
        ("cls", cls_reference[-1], cls_development[-1]),
    )
    for representation, reference_features, development_features in final_layer_representations:
        for bank_role, bank_features, bank_labels in (
            ("primary_32k", reference_features, labels_32k),
            ("sensitivity_8k", reference_features[indices_8k], labels_8k),
        ):
            feature_rows = _evaluate_feature_bank(
                bank_features,
                bank_labels,
                development_features,
                is_n,
                experiment="S1",
                layer=12,
                representation=representation,
                bank_role=bank_role,
                k=k,
                principal_dim=principal_dim,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
                knn_device=knn_device,
            )
            rows.extend(feature_rows)
            if representation == "raw_patch_mean":
                s1_raw_rows.extend(feature_rows)
            if representation == "h_doc" and bank_role == "primary_32k":
                s1_h_doc_32k_variance_ratio = float(
                    feature_rows[0]["top_512_variance_ratio"]
                )

    # S2 reports every layer. Reuse the already-computed L12 raw-mean metrics
    # from S1 rather than refitting the identical estimators.
    for layer_index in range(LAYER_COUNT):
        layer_number = layer_index + 1
        if layer_number == 12:
            rows.extend([{**row, "experiment": "S2"} for row in s1_raw_rows])
            continue
        for bank_role, bank_features, bank_labels in (
            ("primary_32k", raw_reference[layer_index], labels_32k),
            (
                "sensitivity_8k",
                raw_reference[layer_index][indices_8k],
                labels_8k,
            ),
        ):
            rows.extend(
                _evaluate_feature_bank(
                    bank_features,
                    bank_labels,
                    raw_development[layer_index],
                    is_n,
                    experiment="S2",
                    layer=layer_number,
                    representation="raw_patch_mean",
                    bank_role=bank_role,
                    k=k,
                    principal_dim=principal_dim,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed,
                    knn_device=knn_device,
                )
            )

    if s1_h_doc_32k_variance_ratio is None:  # pragma: no cover - loop invariant
        raise AssertionError("S1 h_doc 32k geometry diagnostic was not computed")
    vim_scores = exact_vim_decomposition(
        h_doc_reference,
        logits_reference,
        h_doc_development,
        logits_development,
        weight,
        bias,
        principal_dim=principal_dim,
    )
    vim_centering = {
        "energy": None,
        "vim_origin_residual": "vim_classifier_origin",
        "vim": "vim_classifier_origin",
    }
    for method in ("energy", "vim_origin_residual", "vim"):
        rows.append(
            {
                "experiment": "S1b",
                "layer": 12,
                "representation": "h_doc",
                "method": method,
                "residual_centering_convention": vim_centering[method],
                "bank_role": "primary_32k",
                "bank_size": int(reference_count),
                **_metric_fields(
                    vim_scores[method],
                    is_n,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed,
                ),
                "top_512_variance_ratio": s1_h_doc_32k_variance_ratio,
                "bootstrap_iterations": int(bootstrap_iterations),
                "bootstrap_seed": int(seed),
            }
        )

    return {
        "scope": "development_only_layer_study",
        "sealed_metrics_computed": False,
        "g2": {
            "passed": True,
            "method": "existing_shrinkage_mahalanobis",
            "bank_size": int(len(indices_8k)),
            "expected_auroc": float(g2_target_auroc),
            "observed_auroc": g2_metrics.auroc,
            "delta_auroc": g2_delta,
            "absolute_tolerance_exclusive": float(g2_tolerance),
        },
        "protocol": {
            "layers": LAYER_COUNT,
            "feature_dim": feature_dim,
            "k": int(k),
            "principal_dim": int(principal_dim),
            "bootstrap_iterations": int(bootstrap_iterations),
            "bootstrap_seed": int(seed),
            "hyperparameter_tuning": False,
        },
        "counts": {
            "reference_32k": int(reference_count),
            "reference_8k": int(len(indices_8k)),
            "development_N": int(is_n.sum()),
            "development_O": int((~is_n).sum()),
            "rows": int(len(rows)),
        },
        "rows": rows,
    }
