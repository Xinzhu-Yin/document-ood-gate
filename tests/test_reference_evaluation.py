import numpy as np
import pandas as pd

from document_ood_gate.reference_evaluation import (
    evaluate_reference_signals,
    validation_threshold,
)


def test_validation_threshold_meets_requested_acceptance() -> None:
    scores = np.arange(100, dtype=np.float64)
    threshold = validation_threshold(scores, 0.95)
    assert (scores >= threshold).mean() >= 0.95
    assert (scores >= threshold).mean() < 0.97


def test_reference_signal_evaluation_stays_development_only() -> None:
    rng = np.random.default_rng(7)

    def cache(prefix: str, count: int, feature_dim: int = 4) -> dict[str, np.ndarray]:
        return {
            "paths": np.asarray([f"{prefix}-{index}.tif" for index in range(count)]),
            "subsets": np.asarray([prefix] * count),
            "logits": rng.normal(size=(count, 16)).astype(np.float32),
            "features": rng.normal(size=(count, feature_dim)).astype(np.float32),
        }

    ood = cache("ood", 8)
    ood["subsets"] = np.asarray(["N"] * 4 + ["O"] * 4)
    split = pd.DataFrame(
        {
            "path": ood["paths"],
            "subset": ood["subsets"],
            "split": ["development"] * 8,
            "split_group_id": [f"singleton-{index}" for index in range(8)],
        }
    )
    reference = cache("reference", 32)
    validation = cache("validation", 32)
    t_guardrail = cache("t", 32)
    rows = []
    for purpose, values in (
        ("reference", reference),
        ("validation_threshold", validation),
        ("t_guardrail", t_guardrail),
    ):
        for index, path in enumerate(values["paths"]):
            rows.append(
                {
                    "purpose": purpose,
                    "local_path": path,
                    "label": index % 16,
                    "class_sample_rank": index // 16,
                }
            )
    result = evaluate_reference_signals(
        ood,
        split,
        reference,
        validation,
        t_guardrail,
        pd.DataFrame(rows),
        reference_sizes=(16, 32),
        k=1,
        bootstrap_iterations=5,
        device="cpu",
    )
    assert result["sealed_metrics_computed"] is False
    assert result["counts"]["development_N"] == 4
    assert "knn_k1_bank_32" in result["signals"]
    assert "mahalanobis_bank_32" in result["signals"]
    assert "feature_norm" in result["signals"]
