import numpy as np
import pandas as pd
import pytest

from document_ood_gate.single_signal import (
    development_indices,
    evaluate_development_single_signals,
)


def _cache() -> dict[str, np.ndarray]:
    logits = np.zeros((8, 16), dtype=np.float32)
    logits[:4, 0] = [8.0, 7.0, 6.0, 5.0]
    logits[4:, 0] = [1.0, 0.5, 0.25, 0.0]
    return {
        "paths": np.asarray([f"page-{index}.tif" for index in range(8)]),
        "subsets": np.asarray(["N"] * 4 + ["O"] * 4),
        "logits": logits,
        "features": np.arange(8 * 4, dtype=np.float32).reshape(8, 4),
    }


def _assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path": [f"page-{index}.tif" for index in range(8)],
            "split": ["development"] * 8,
            "subset": ["N"] * 4 + ["O"] * 4,
            "split_group_id": [f"singleton-{index}" for index in range(8)],
        }
    )


def test_development_evaluation_never_reports_sealed_metrics() -> None:
    result = evaluate_development_single_signals(
        _cache(),
        _assignments(),
        bootstrap_iterations=20,
    )
    assert result["scope"] == "development_only"
    assert result["sealed_metrics_computed"] is False
    assert result["counts"] == {
        "N": 4,
        "O": 4,
        "total": 8,
        "cross_label_conflict_rows_excluded": 0,
    }
    assert set(result["signals"]) == {
        "msp",
        "energy",
        "max_logit",
        "negative_entropy",
        "logit_margin",
        "feature_norm",
    }
    assert result["signals"]["msp"]["auroc"] == 1.0


def test_path_sets_must_match_exactly() -> None:
    assignments = _assignments().iloc[:-1]
    with pytest.raises(ValueError, match="path sets differ"):
        development_indices(_cache()["paths"], assignments)


def test_cross_label_duplicate_group_can_be_excluded_without_scores() -> None:
    assignments = _assignments()
    assignments.loc[[0, 4], "split_group_id"] = "pixel-identical-conflict"
    indices = development_indices(
        _cache()["paths"],
        assignments,
        exclude_cross_label_conflicts=True,
    )
    assert set(indices.tolist()) == {1, 2, 3, 5, 6, 7}
