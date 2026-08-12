import numpy as np

from document_ood_gate.metrics import binary_ood_metrics, macro_auroc_by_predicted_class


def test_perfect_separation() -> None:
    result = binary_ood_metrics(np.array([0.8, 0.9, 1.0]), np.array([0.0, 0.1, 0.2]))
    assert result.auroc == 1.0
    assert result.fpr95_interpolated == 0.0


def test_macro_uses_predicted_buckets() -> None:
    macro, per_class = macro_auroc_by_predicted_class(
        np.array([0.9, 0.8]),
        np.array([0.2, 0.1]),
        np.array([0, 1]),
        np.array([0, 1]),
    )
    assert macro == 1.0
    assert per_class == {0: 1.0, 1: 1.0}

