from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .metrics import binary_ood_metrics, macro_auroc_by_predicted_class
from .scores import output_normality_scores


PUBLISHED_TARGETS = {
    "msp": {"auroc": 0.728, "fpr95": 0.847, "macro_auroc": 0.780},
    "energy": {"auroc": 0.753, "fpr95": 0.843, "macro_auroc": 0.792},
}


def classify_reproduction_delta(auroc_delta: float, fpr95_delta: float) -> str:
    a = abs(auroc_delta)
    f = abs(fpr95_delta)
    if a <= 0.01 and f <= 0.02:
        return "green"
    if a > 0.03 or f > 0.05:
        return "red"
    return "yellow"


def reproduce_from_logits(logits: np.ndarray, subsets: np.ndarray) -> dict[str, object]:
    subset_values = np.asarray(subsets).astype(str)
    n_mask = subset_values == "N"
    o_mask = subset_values == "O"
    if int(n_mask.sum()) != 1002 or int(o_mask.sum()) != 3415:
        raise ValueError(f"Expected N=1002/O=3415, got N={n_mask.sum()}/O={o_mask.sum()}")

    scores = output_normality_scores(logits)
    predicted = np.asarray(logits).argmax(axis=1)
    result: dict[str, object] = {"n_count": int(n_mask.sum()), "o_count": int(o_mask.sum()), "signals": {}}
    for signal in ("msp", "energy"):
        metric = binary_ood_metrics(scores[signal][n_mask], scores[signal][o_mask])
        macro, per_class = macro_auroc_by_predicted_class(
            scores[signal][n_mask],
            scores[signal][o_mask],
            predicted[n_mask],
            predicted[o_mask],
        )
        target = PUBLISHED_TARGETS[signal]
        auroc_delta = metric.auroc - target["auroc"]
        # The strict comparator is the first ROC point reaching N TPR >= 95%.
        # The interpolated convention remains in the result as a sensitivity check.
        fpr95_delta = metric.fpr95_first_ge - target["fpr95"]
        result["signals"][signal] = {
            **asdict(metric),
            "macro_auroc": macro,
            "macro_by_predicted_class": {str(k): v for k, v in per_class.items()},
            "published": target,
            "auroc_delta": auroc_delta,
            "fpr95_delta": fpr95_delta,
            "gate": classify_reproduction_delta(auroc_delta, fpr95_delta),
        }
    gates = [entry["gate"] for entry in result["signals"].values()]
    result["overall_gate"] = "red" if "red" in gates else "yellow" if "yellow" in gates else "green"
    return result
