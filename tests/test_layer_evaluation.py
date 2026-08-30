from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import document_ood_gate.layer_evaluation as layer_evaluation
from document_ood_gate.layer_evaluation import (
    G2ReproductionError,
    evaluate_layer_study,
    exact_vim_decomposition,
)
from document_ood_gate.metrics import binary_ood_metrics
from document_ood_gate.reference_scores import ShrinkageMahalanobis


def _inputs() -> dict[str, Any]:
    rng = np.random.default_rng(41)
    feature_dim = 6
    class_count = 4
    reference_count = 32
    development_count = 20
    labels = np.repeat(np.arange(class_count), reference_count // class_count)
    centers = rng.normal(scale=1.5, size=(class_count, feature_dim))
    reference_h_doc = centers[labels] + rng.normal(scale=0.3, size=(reference_count, feature_dim))

    development_is_n = np.zeros(development_count, dtype=bool)
    development_is_n[: development_count // 2] = True
    n_labels = np.arange(development_count // 2) % class_count
    development_h_doc = np.empty((development_count, feature_dim))
    development_h_doc[development_is_n] = centers[n_labels] + rng.normal(
        scale=0.45,
        size=(development_is_n.sum(), feature_dim),
    )
    development_h_doc[~development_is_n] = rng.normal(
        loc=3.0,
        scale=1.2,
        size=((~development_is_n).sum(), feature_dim),
    )

    reference_raw = np.stack(
        [
            reference_h_doc + rng.normal(scale=0.02 * layer, size=reference_h_doc.shape)
            for layer in range(1, 13)
        ]
    )
    development_raw = np.stack(
        [
            development_h_doc
            + rng.normal(scale=0.02 * layer, size=development_h_doc.shape)
            for layer in range(1, 13)
        ]
    )
    reference_cls = np.stack(
        [
            0.8 * reference_h_doc
            + rng.normal(scale=0.03 * layer, size=reference_h_doc.shape)
            for layer in range(1, 13)
        ]
    )
    development_cls = np.stack(
        [
            0.8 * development_h_doc
            + rng.normal(scale=0.03 * layer, size=development_h_doc.shape)
            for layer in range(1, 13)
        ]
    )

    weight = rng.normal(scale=0.5, size=(class_count, feature_dim))
    bias = np.array([2.0, 1.5, 1.0, 0.5])
    reference_logits = reference_h_doc @ weight.T + bias
    development_logits = development_h_doc @ weight.T + bias
    indices_8k = np.concatenate(
        [np.arange(label * 8, label * 8 + 4) for label in range(class_count)]
    )
    labels_8k = labels[indices_8k]

    g2_scores = (
        ShrinkageMahalanobis()
        .fit(reference_h_doc[indices_8k], labels_8k)
        .normality(development_h_doc)
    )
    g2_target = binary_ood_metrics(
        g2_scores[development_is_n],
        g2_scores[~development_is_n],
    ).auroc
    return {
        "reference_raw_patch_mean": reference_raw.astype(np.float32),
        "reference_cls": reference_cls.astype(np.float32),
        "reference_h_doc": reference_h_doc.astype(np.float32),
        "reference_logits": reference_logits.astype(np.float32),
        "reference_labels": labels,
        "reference_8k_indices": indices_8k,
        "reference_8k_labels": labels_8k,
        "development_raw_patch_mean": development_raw.astype(np.float32),
        "development_cls": development_cls.astype(np.float32),
        "development_h_doc": development_h_doc.astype(np.float32),
        "development_logits": development_logits.astype(np.float32),
        "development_is_n": development_is_n,
        "classifier_weight": weight.astype(np.float32),
        "classifier_bias": bias.astype(np.float32),
        "k": 3,
        "principal_dim": 3,
        "bootstrap_iterations": 5,
        "seed": 20260812,
        "g2_target_auroc": g2_target,
        "knn_device": "cpu",
    }


def test_g2_passes_before_complete_s1_s2_and_s1b_table_is_returned() -> None:
    result = evaluate_layer_study(**_inputs())

    assert result["g2"]["passed"] is True
    assert result["sealed_metrics_computed"] is False
    assert result["counts"]["rows"] == 93
    assert len(result["rows"]) == 93
    assert sum(row["experiment"] == "S1" for row in result["rows"]) == 18
    assert sum(row["experiment"] == "S2" for row in result["rows"]) == 72
    assert sum(row["experiment"] == "S1b" for row in result["rows"]) == 3
    required = {
        "experiment",
        "layer",
        "representation",
        "method",
        "residual_centering_convention",
        "bank_size",
        "auroc",
        "auroc_95ci",
        "fpr95",
        "fpr95_95ci",
        "top_512_variance_ratio",
    }
    assert all(required <= set(row) for row in result["rows"])
    assert all(row["bootstrap_seed"] == 20260812 for row in result["rows"])
    assert {row["layer"] for row in result["rows"] if row["experiment"] == "S2"} == set(
        range(1, 13)
    )

    geometry_groups: dict[tuple[object, ...], set[float]] = {}
    for row in result["rows"]:
        if row["experiment"] in {"S1", "S2"}:
            key = (
                row["experiment"],
                row["layer"],
                row["representation"],
                row["bank_size"],
            )
            geometry_groups.setdefault(key, set()).add(row["top_512_variance_ratio"])
    assert all(len(values) == 1 for values in geometry_groups.values())


def test_g2_failure_stops_before_any_layer_method_runs(monkeypatch) -> None:
    inputs = _inputs()
    target = inputs["g2_target_auroc"]
    inputs["g2_target_auroc"] = target - 0.1 if target >= 0.1 else target + 0.1

    def forbidden_layer_method(*_args, **_kwargs):
        raise AssertionError("layer methods must not run after G2 failure")

    monkeypatch.setattr(layer_evaluation, "_evaluate_feature_bank", forbidden_layer_method)
    with pytest.raises(G2ReproductionError, match="G2 Mahalanobis-8k reproduction failed"):
        evaluate_layer_study(**inputs)


def test_exact_vim_decomposition_uses_one_origin_residual_and_energy_sum() -> None:
    inputs = _inputs()
    scores = exact_vim_decomposition(
        inputs["reference_h_doc"],
        inputs["reference_logits"],
        inputs["development_h_doc"],
        inputs["development_logits"],
        inputs["classifier_weight"],
        inputs["classifier_bias"],
        principal_dim=inputs["principal_dim"],
    )

    np.testing.assert_allclose(
        scores["vim"],
        scores["energy"] + scores["vim_origin_residual"],
        atol=1e-12,
    )
    assert np.all(scores["vim_origin_residual"] <= 0.0)


def test_centering_labels_distinguish_mean_centered_residual_from_vim() -> None:
    rows = evaluate_layer_study(**_inputs())["rows"]
    mean_residual = [
        row for row in rows if row["method"] == "mean_centered_pca_residual"
    ]
    assert mean_residual
    assert {row["residual_centering_convention"] for row in mean_residual} == {
        "reference_mean"
    }

    s1b = {row["method"]: row for row in rows if row["experiment"] == "S1b"}
    assert s1b["energy"]["residual_centering_convention"] is None
    assert s1b["vim_origin_residual"]["residual_centering_convention"] == (
        "vim_classifier_origin"
    )
    assert s1b["vim"]["residual_centering_convention"] == "vim_classifier_origin"
    assert len({row["top_512_variance_ratio"] for row in s1b.values()}) == 1
