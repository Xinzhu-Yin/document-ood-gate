import numpy as np
import pandas as pd

from document_ood_gate.supplementary_evaluation import evaluate_supplementary_baselines


def test_supplementary_panel_is_development_only_and_contains_four_methods() -> None:
    rng = np.random.default_rng(31)
    feature_dim = 6
    weight = rng.normal(size=(16, feature_dim))
    bias = rng.normal(scale=0.1, size=16)

    def cache(prefix: str, count: int) -> dict[str, np.ndarray]:
        features = rng.normal(size=(count, feature_dim)).astype(np.float32)
        logits = (features @ weight.T + bias).astype(np.float32)
        return {
            "paths": np.asarray([f"{prefix}-{index}.tif" for index in range(count)]),
            "subsets": np.asarray([prefix] * count),
            "features": features,
            "logits": logits,
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

    result = evaluate_supplementary_baselines(
        ood,
        split,
        reference,
        validation,
        t_guardrail,
        pd.DataFrame(rows),
        weight,
        bias,
        reference_size=32,
        vim_principal_dim=3,
        bootstrap_iterations=5,
        enforce_frozen_counts=False,
        enforce_baseline_gate=False,
    )
    assert result["status"] == "completed"
    assert result["sealed_metrics_computed"] is False
    assert result["counts"]["development_cache_rows_loaded"] == 8
    assert result["counts"]["sealed_feature_or_logit_rows_loaded"] == 0
    assert result["counts"]["sealed_rows_inferred_or_scored"] == 0
    assert set(result["supplementary_signals"]) == {
        "gen_gamma_0.1_m16",
        "relative_mahalanobis_bank_32",
        "vim_dim_3_bank_32",
        "react_p90_energy_bank_32",
    }
