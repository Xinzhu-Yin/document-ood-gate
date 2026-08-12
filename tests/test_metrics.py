import numpy as np

from document_ood_gate.metrics import (
    binary_ood_metrics,
    macro_auroc_by_predicted_class,
    macro_auroc_released_code,
)


def test_perfect_separation() -> None:
    result = binary_ood_metrics(np.array([0.8, 0.9, 1.0]), np.array([0.0, 0.1, 0.2]))
    assert result.auroc == 1.0
    assert result.fpr95_interpolated == 0.0
    assert result.fpr95_first_ge == 0.0


def test_reverse_and_tied_rankings() -> None:
    reverse = binary_ood_metrics(np.array([0.0, 0.1]), np.array([0.8, 0.9]))
    tied = binary_ood_metrics(np.ones(20), np.ones(20))
    assert reverse.auroc == 0.0
    assert reverse.fpr95_first_ge == 1.0
    assert tied.auroc == 0.5
    assert tied.fpr95_first_ge == 1.0


def test_auc_flips_when_score_direction_flips() -> None:
    n = np.array([0.2, 0.7, 0.9])
    o = np.array([0.1, 0.4, 0.8])
    forward = binary_ood_metrics(n, o).auroc
    backward = binary_ood_metrics(-n, -o).auroc
    assert np.isclose(forward + backward, 1.0)


def test_macro_uses_predicted_buckets() -> None:
    macro, per_class = macro_auroc_by_predicted_class(
        np.array([0.9, 0.8]),
        np.array([0.2, 0.1]),
        np.array([0, 1]),
        np.array([0, 1]),
    )
    assert macro == 1.0
    assert per_class == {0: 1.0, 1: 1.0}


def test_released_macro_uses_n_true_and_o_predicted_buckets() -> None:
    macro, per_class = macro_auroc_released_code(
        np.array([0.9, 0.8]),
        np.array([0.2, 0.1]),
        np.array(["a", "b"]),
        np.array(["a", "b"]),
    )
    assert macro == 1.0
    assert per_class == {"a": 1.0, "b": 1.0}
