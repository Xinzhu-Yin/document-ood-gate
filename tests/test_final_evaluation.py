from __future__ import annotations

import numpy as np

from document_ood_gate.final_evaluation import (
    evaluate_replication_predictions,
    layer_norm_free,
    layer_norm_pooler,
)


def test_layer_norm_free_is_per_sample_and_has_no_affine() -> None:
    rng = np.random.default_rng(20260812)
    values = rng.normal(size=(7, 768)).astype(np.float32)
    normalized = layer_norm_free(values, eps=1e-12)
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized.mean(axis=1), 0.0, atol=2e-7)
    np.testing.assert_allclose(normalized.var(axis=1), 1.0, atol=2e-6)


def test_layer_norm_pooler_adds_learned_gamma_beta() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(5, 768)).astype(np.float32)
    weight = np.linspace(0.5, 1.5, 768, dtype=np.float32)
    bias = np.linspace(-0.2, 0.2, 768, dtype=np.float32)
    expected = layer_norm_free(values, eps=1e-5) * weight + bias
    observed = layer_norm_pooler(values, weight, bias, eps=1e-5)
    np.testing.assert_array_equal(observed, expected)


def _row(
    *,
    experiment: str,
    layer: int | None,
    representation: str,
    method: str,
    auroc: float,
    fpr95: float,
) -> dict[str, object]:
    return {
        "experiment": experiment,
        "layer": layer,
        "representation": representation,
        "method": method,
        "auroc": auroc,
        "fpr95": fpr95,
    }


def test_replication_predictions_use_the_frozen_definitions() -> None:
    grid: list[dict[str, object]] = []
    methods = ("mahalanobis", "knn", "mean_centered_pca_residual")
    for layer in range(2, 6):
        for method in methods:
            grid.append(
                _row(
                    experiment="D2",
                    layer=layer,
                    representation="raw_patch_mean",
                    method=method,
                    auroc=0.59,
                    fpr95=0.91,
                )
            )
    grid.extend(
        [
            _row(
                experiment="D2",
                layer=12,
                representation="raw_patch_mean",
                method="mahalanobis",
                auroc=0.65,
                fpr95=0.90,
            ),
            _row(
                experiment="D2",
                layer=12,
                representation="ln_free",
                method="mahalanobis",
                auroc=0.71,
                fpr95=0.85,
            ),
            _row(
                experiment="D2",
                layer=12,
                representation="ln_pooler",
                method="knn",
                auroc=0.80,
                fpr95=0.81,
            ),
            _row(
                experiment="D2",
                layer=11,
                representation="ln_free",
                method="knn",
                auroc=0.77,
                fpr95=0.70,
            ),
        ]
    )
    baselines = [
        _row(
            experiment="D1",
            layer=12,
            representation="h_doc",
            method="mahalanobis",
            auroc=0.74,
            fpr95=0.82,
        )
    ]
    results = evaluate_replication_predictions(baselines, grid)
    assert [row["prediction"] for row in results] == ["P1", "P2", "P3", "P4", "P5"]
    assert [row["status"] for row in results] == ["supported"] * 5
