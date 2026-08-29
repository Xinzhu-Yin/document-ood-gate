from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import binary_ood_metrics
from .reference_evaluation import _cache_manifest_rows, _signal_report
from .reference_scores import ShrinkageMahalanobis, ShrinkageRelativeMahalanobis
from .scores import output_normality_scores
from .single_signal import development_indices
from .supplementary_scores import (
    ViMNormality,
    generalized_entropy_normality,
    linear_classifier_logits,
    react_clip_threshold,
    react_energy_normality,
)

EXPECTED_DEVELOPMENT_BASELINES = {
    "energy": {
        "auroc": 0.7282381124934574,
        "fpr95": 0.8613963039014374,
        "t_false_rejection_rate": 0.0113,
    },
    "mahalanobis_bank_32000": {
        "auroc": 0.7447457422394008,
        "fpr95": 0.8162217659137577,
        "t_false_rejection_rate": 0.0089,
    },
}


def paired_bootstrap_deltas(
    candidate_n: np.ndarray,
    candidate_o: np.ndarray,
    baseline_n: np.ndarray,
    baseline_o: np.ndarray,
    *,
    target_tpr: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    candidate_n = np.asarray(candidate_n, dtype=np.float64)
    candidate_o = np.asarray(candidate_o, dtype=np.float64)
    baseline_n = np.asarray(baseline_n, dtype=np.float64)
    baseline_o = np.asarray(baseline_o, dtype=np.float64)
    if candidate_n.shape != baseline_n.shape or candidate_o.shape != baseline_o.shape:
        raise ValueError("Candidate and baseline scores must be row-aligned")
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")

    candidate_point = binary_ood_metrics(candidate_n, candidate_o, target_tpr=target_tpr)
    baseline_point = binary_ood_metrics(baseline_n, baseline_o, target_tpr=target_tpr)
    rng = np.random.default_rng(seed)
    auroc_deltas = np.empty(iterations, dtype=np.float64)
    fpr95_deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        n_indices = rng.integers(0, len(candidate_n), size=len(candidate_n))
        o_indices = rng.integers(0, len(candidate_o), size=len(candidate_o))
        candidate = binary_ood_metrics(
            candidate_n[n_indices], candidate_o[o_indices], target_tpr=target_tpr
        )
        baseline = binary_ood_metrics(
            baseline_n[n_indices], baseline_o[o_indices], target_tpr=target_tpr
        )
        auroc_deltas[iteration] = candidate.auroc - baseline.auroc
        fpr95_deltas[iteration] = candidate.fpr95_first_ge - baseline.fpr95_first_ge
    return {
        "delta_auroc": float(candidate_point.auroc - baseline_point.auroc),
        "delta_auroc_95ci": [
            float(value) for value in np.quantile(auroc_deltas, [0.025, 0.975])
        ],
        "delta_fpr95": float(
            candidate_point.fpr95_first_ge - baseline_point.fpr95_first_ge
        ),
        "delta_fpr95_95ci": [
            float(value) for value in np.quantile(fpr95_deltas, [0.025, 0.975])
        ],
        "orientation": "positive delta AUROC and negative delta FPR95 favor candidate",
    }


def _classifier_equivalence(
    features: np.ndarray,
    cached_logits: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> float:
    recomputed = linear_classifier_logits(features, weight, bias)
    error = float(np.max(np.abs(recomputed - np.asarray(cached_logits, dtype=np.float64))))
    if error > 2e-5:
        raise ValueError(f"Cached feature/logit classifier equivalence failed: max error {error}")
    return error


def _react_diagnostics(
    features: np.ndarray,
    original_logits: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    clip_threshold: float,
) -> dict[str, float]:
    values = np.asarray(features, dtype=np.float64)
    clipped_logits = linear_classifier_logits(
        np.minimum(values, clip_threshold), weight, bias
    )
    return {
        "clip_fraction": float((values > clip_threshold).mean()),
        "mean_excess_over_clip_all_dimensions": float(
            np.maximum(values - clip_threshold, 0.0).mean()
        ),
        "prediction_flip_rate_vs_original_logits": float(
            (clipped_logits.argmax(axis=1) != np.asarray(original_logits).argmax(axis=1)).mean()
        ),
    }


def _report(
    development_scores: np.ndarray,
    validation_scores: np.ndarray,
    t_scores: np.ndarray,
    n_mask: np.ndarray,
    o_mask: np.ndarray,
    *,
    target_n_tpr: float,
    target_validation_acceptance: float,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    n_scores = np.asarray(development_scores)[n_mask]
    o_scores = np.asarray(development_scores)[o_mask]
    result = _signal_report(
        n_scores,
        o_scores,
        validation_scores,
        t_scores,
        target_n_tpr=target_n_tpr,
        target_validation_acceptance=target_validation_acceptance,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    return result, (n_scores, o_scores)


def _baseline_reproduction_gate(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tolerance = 1e-8
    details: dict[str, Any] = {}
    passed = True
    for name, expected in EXPECTED_DEVELOPMENT_BASELINES.items():
        observed = {key: float(reports[name][key]) for key in expected}
        deltas = {key: observed[key] - expected[key] for key in expected}
        method_passed = all(abs(value) <= tolerance for value in deltas.values())
        details[name] = {
            "passed": method_passed,
            "expected": expected,
            "observed": observed,
            "deltas": deltas,
        }
        passed = passed and method_passed
    return {
        "passed": passed,
        "absolute_tolerance": tolerance,
        "methods": details,
        "action_if_failed": "Stop before supplementary methods and audit cached inputs/scoring.",
    }


def evaluate_supplementary_baselines(
    ood_cache: dict[str, np.ndarray],
    split_assignments: pd.DataFrame,
    reference_cache: dict[str, np.ndarray],
    validation_cache: dict[str, np.ndarray],
    t_cache: dict[str, np.ndarray],
    extracted_manifest: pd.DataFrame,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    *,
    reference_size: int = 32000,
    gen_gamma: float = 0.1,
    gen_top_m: int = 100,
    react_percentile: float = 90.0,
    vim_principal_dim: int = 512,
    target_n_tpr: float = 0.95,
    target_validation_acceptance: float = 0.99,
    bootstrap_iterations: int = 2000,
    seed: int = 20260812,
    enforce_frozen_counts: bool = True,
    enforce_baseline_gate: bool = True,
) -> dict[str, Any]:
    all_development = development_indices(ood_cache["paths"], split_assignments)
    development = development_indices(
        ood_cache["paths"],
        split_assignments,
        exclude_cross_label_conflicts=True,
    )
    subsets = np.asarray(ood_cache["subsets"])[development].astype(str)
    n_mask = subsets == "N"
    o_mask = subsets == "O"
    if not n_mask.any() or not o_mask.any():
        raise ValueError("Development split must contain both N and O")

    reference_rows = _cache_manifest_rows(reference_cache, extracted_manifest, "reference")
    validation_rows = _cache_manifest_rows(
        validation_cache, extracted_manifest, "validation_threshold"
    )
    t_rows = _cache_manifest_rows(t_cache, extracted_manifest, "t_guardrail")
    per_class = reference_size // 16
    if reference_size % 16:
        raise ValueError("reference_size must be divisible by 16")
    bank_mask = reference_rows["class_sample_rank"].to_numpy(dtype=np.int64) < per_class
    if int(bank_mask.sum()) != reference_size:
        raise ValueError("Reference manifest did not produce the requested nested bank")

    development_features = np.asarray(ood_cache["features"])[development]
    development_logits = np.asarray(ood_cache["logits"])[development]
    reference_features = np.asarray(reference_cache["features"])[bank_mask]
    reference_logits = np.asarray(reference_cache["logits"])[bank_mask]
    reference_labels = reference_rows.loc[bank_mask, "label"].to_numpy(dtype=np.int64)
    validation_features = np.asarray(validation_cache["features"])
    validation_logits = np.asarray(validation_cache["logits"])
    t_features = np.asarray(t_cache["features"])
    t_logits = np.asarray(t_cache["logits"])

    if enforce_frozen_counts:
        expected_counts = (306, 974, 32000, 8000, 10000)
        actual_counts = (
            int(n_mask.sum()),
            int(o_mask.sum()),
            len(reference_features),
            len(validation_features),
            len(t_features),
        )
        if actual_counts != expected_counts:
            raise ValueError(f"Frozen count gate failed: expected {expected_counts}, found {actual_counts}")

    classifier_errors = {
        "development": _classifier_equivalence(
            development_features,
            development_logits,
            classifier_weight,
            classifier_bias,
        ),
        "reference": _classifier_equivalence(
            reference_features,
            reference_logits,
            classifier_weight,
            classifier_bias,
        ),
        "validation": _classifier_equivalence(
            validation_features,
            validation_logits,
            classifier_weight,
            classifier_bias,
        ),
        "T_guardrail": _classifier_equivalence(
            t_features,
            t_logits,
            classifier_weight,
            classifier_bias,
        ),
    }

    output_development = output_normality_scores(development_logits)
    output_validation = output_normality_scores(validation_logits)
    output_t = output_normality_scores(t_logits)
    reports: dict[str, dict[str, Any]] = {}
    score_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("energy", "negative_entropy"):
        reports[name], score_pairs[name] = _report(
            output_development[name],
            output_validation[name],
            output_t[name],
            n_mask,
            o_mask,
            target_n_tpr=target_n_tpr,
            target_validation_acceptance=target_validation_acceptance,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )

    mahalanobis = ShrinkageMahalanobis().fit(reference_features, reference_labels)
    reports["mahalanobis_bank_32000"], score_pairs["mahalanobis_bank_32000"] = _report(
        mahalanobis.normality(development_features),
        mahalanobis.normality(validation_features),
        mahalanobis.normality(t_features),
        n_mask,
        o_mask,
        target_n_tpr=target_n_tpr,
        target_validation_acceptance=target_validation_acceptance,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    gate = _baseline_reproduction_gate(reports)
    base_result: dict[str, Any] = {
        "scope": "development_only_supplementary_posthoc_baselines",
        "sealed_metrics_computed": False,
        "protocol": {
            "model": "frozen microsoft/dit-base-finetuned-rvlcdip",
            "OOD_cache": "physically isolated development-only logits/features",
            "reference_bank": f"same nested {reference_size}-row RVL bank",
            "development_metrics": "micro AUROC and FPR95 with N positive/high-normality",
            "threshold": (
                f"each score independently calibrated for {target_validation_acceptance:.0%} "
                "acceptance on the same 8k RVL validation set"
            ),
            "T_guardrail": "reported only; never used for method or hyperparameter selection",
            "sealed_test": "not loaded, inferred, or scored",
            "bootstrap": f"{bootstrap_iterations} stratified resamples; seed {seed}",
        },
        "counts": {
            "development_cache_rows_loaded": int(len(ood_cache["paths"])),
            "development_split_manifest_rows_loaded": int(len(split_assignments)),
            "development_N": int(n_mask.sum()),
            "development_O": int(o_mask.sum()),
            "cross_label_conflict_rows_excluded": int(
                len(all_development) - len(development)
            ),
            "reference": int(len(reference_features)),
            "validation_threshold": int(len(validation_rows)),
            "T_guardrail": int(len(t_rows)),
            "sealed_image_pixel_rows_loaded": 0,
            "sealed_feature_or_logit_rows_loaded": 0,
            "sealed_rows_inferred_or_scored": 0,
        },
        "classifier_feature_equivalence_max_abs_error": classifier_errors,
        "internal_baseline_reproduction_gate": gate,
        "comparison_baselines": reports,
    }
    if enforce_baseline_gate and not gate["passed"]:
        base_result["supplementary_signals"] = {}
        base_result["status"] = "stopped_on_internal_baseline_gate"
        return base_result

    supplementary_reports: dict[str, dict[str, Any]] = {}
    supplementary_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    effective_gen_m = min(int(gen_top_m), development_logits.shape[1])
    gen_name = f"gen_gamma_{gen_gamma:g}_m{effective_gen_m}"
    supplementary_reports[gen_name], supplementary_pairs[gen_name] = _report(
        generalized_entropy_normality(
            development_logits, gamma=gen_gamma, top_m=gen_top_m
        ),
        generalized_entropy_normality(validation_logits, gamma=gen_gamma, top_m=gen_top_m),
        generalized_entropy_normality(t_logits, gamma=gen_gamma, top_m=gen_top_m),
        n_mask,
        o_mask,
        target_n_tpr=target_n_tpr,
        target_validation_acceptance=target_validation_acceptance,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )

    relative = ShrinkageRelativeMahalanobis().fit(reference_features, reference_labels)
    rmd_name = f"relative_mahalanobis_bank_{reference_size}"
    supplementary_reports[rmd_name], supplementary_pairs[rmd_name] = _report(
        relative.normality(development_features),
        relative.normality(validation_features),
        relative.normality(t_features),
        n_mask,
        o_mask,
        target_n_tpr=target_n_tpr,
        target_validation_acceptance=target_validation_acceptance,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )

    vim = ViMNormality(principal_dim=vim_principal_dim).fit(
        reference_features,
        reference_logits,
        classifier_weight,
        classifier_bias,
    )
    vim_name = f"vim_dim_{vim_principal_dim}_bank_{reference_size}"
    supplementary_reports[vim_name], supplementary_pairs[vim_name] = _report(
        vim.normality(development_features, development_logits),
        vim.normality(validation_features, validation_logits),
        vim.normality(t_features, t_logits),
        n_mask,
        o_mask,
        target_n_tpr=target_n_tpr,
        target_validation_acceptance=target_validation_acceptance,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )

    clip = react_clip_threshold(reference_features, percentile=react_percentile)
    if clip <= 0.0:
        raise ValueError("Frozen ReAct p90 clip threshold must be positive")
    react_name = f"react_p{react_percentile:g}_energy_bank_{reference_size}"
    supplementary_reports[react_name], supplementary_pairs[react_name] = _report(
        react_energy_normality(
            development_features,
            classifier_weight,
            classifier_bias,
            clip_threshold=clip,
        ),
        react_energy_normality(
            validation_features,
            classifier_weight,
            classifier_bias,
            clip_threshold=clip,
        ),
        react_energy_normality(
            t_features,
            classifier_weight,
            classifier_bias,
            clip_threshold=clip,
        ),
        n_mask,
        o_mask,
        target_n_tpr=target_n_tpr,
        target_validation_acceptance=target_validation_acceptance,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )

    parent_baselines = {
        gen_name: "negative_entropy",
        rmd_name: "mahalanobis_bank_32000",
        vim_name: "energy",
        react_name: "energy",
    }
    paired: dict[str, Any] = {}
    for name, baseline_name in parent_baselines.items():
        candidate_n, candidate_o = supplementary_pairs[name]
        baseline_n, baseline_o = score_pairs[baseline_name]
        paired[name] = {
            "baseline": baseline_name,
            **paired_bootstrap_deltas(
                candidate_n,
                candidate_o,
                baseline_n,
                baseline_o,
                target_tpr=target_n_tpr,
                iterations=bootstrap_iterations,
                seed=seed,
            ),
        }

    current_best = reports["mahalanobis_bank_32000"]
    point_dominators = [
        name
        for name, metrics in supplementary_reports.items()
        if metrics["auroc"] > current_best["auroc"]
        and metrics["fpr95"] < current_best["fpr95"]
    ]
    vim_eigenvalues = np.asarray(vim.eigenvalues_)
    vim_residual_dim = reference_features.shape[1] - vim_principal_dim
    react_diagnostics = {
        "reference": _react_diagnostics(
            reference_features,
            reference_logits,
            classifier_weight,
            classifier_bias,
            clip,
        ),
        "development": _react_diagnostics(
            development_features,
            development_logits,
            classifier_weight,
            classifier_bias,
            clip,
        ),
        "development_N": _react_diagnostics(
            development_features[n_mask],
            development_logits[n_mask],
            classifier_weight,
            classifier_bias,
            clip,
        ),
        "development_O": _react_diagnostics(
            development_features[o_mask],
            development_logits[o_mask],
            classifier_weight,
            classifier_bias,
            clip,
        ),
        "validation": _react_diagnostics(
            validation_features,
            validation_logits,
            classifier_weight,
            classifier_bias,
            clip,
        ),
        "T_guardrail": _react_diagnostics(
            t_features,
            t_logits,
            classifier_weight,
            classifier_bias,
            clip,
        ),
    }
    base_result.update(
        {
            "status": "completed",
            "method_configuration": {
                gen_name: {
                    "formula": "-sum(top-M p^gamma * (1-p)^gamma)",
                    "gamma": float(gen_gamma),
                    "requested_M": int(gen_top_m),
                    "effective_M": effective_gen_m,
                    "fit_data": "none",
                },
                rmd_name: {
                    "formula": "background squared distance - minimum class squared distance",
                    "class_model": "16 class means + shared Ledoit-Wolf covariance",
                    "background_model": "one mean + Ledoit-Wolf covariance",
                    "fit_data": f"same {reference_size}-row RVL bank only",
                },
                vim_name: {
                    "principal_dim": int(vim_principal_dim),
                    "residual_dim": int(vim_residual_dim),
                    "alpha": float(vim.alpha_),
                    "origin_logit_max_abs_error": float(
                        np.max(
                            np.abs(
                                np.asarray(classifier_weight, dtype=np.float64)
                                @ np.asarray(vim.origin_)
                                + np.asarray(classifier_bias, dtype=np.float64)
                            )
                        )
                    ),
                    "null_space_orthogonality_max_abs_error": float(
                        np.max(
                            np.abs(
                                np.asarray(vim.null_space_).T @ np.asarray(vim.null_space_)
                                - np.eye(vim_residual_dim)
                            )
                        )
                    ),
                    "eigenvalue_at_residual_boundary": float(
                        vim_eigenvalues[vim_residual_dim - 1]
                    ),
                    "eigenvalue_at_principal_boundary": float(
                        vim_eigenvalues[vim_residual_dim]
                    ),
                    "boundary_eigenvalue_gap": float(
                        vim_eigenvalues[vim_residual_dim]
                        - vim_eigenvalues[vim_residual_dim - 1]
                    ),
                    "fit_data": f"same {reference_size}-row RVL features/logits only",
                },
                react_name: {
                    "percentile": float(react_percentile),
                    "clip_threshold": float(clip),
                    "energy_temperature": 1.0,
                    "fit_data": f"percentile from same {reference_size}-row RVL bank only",
                    "adaptation_note": (
                        "Applied before the linear classifier to signed LayerNorm h_doc features; "
                        "the original ReAct evidence was primarily on pooled CNN activations."
                    ),
                    "diagnostics": react_diagnostics,
                },
            },
            "supplementary_signals": supplementary_reports,
            "paired_deltas_vs_parent_baseline": paired,
            "point_estimate_pareto_check_vs_mahalanobis_32000": {
                "methods_with_higher_auroc_and_lower_fpr95": point_dominators,
                "note": "Descriptive development check only; confidence intervals govern interpretation.",
            },
        }
    )
    return base_result


def supplementary_summary_markdown(result: dict[str, Any]) -> str:
    gate = result["internal_baseline_reproduction_gate"]
    lines = [
        "# Supplementary post-hoc baselines — development checkpoint",
        "",
        "> Scope: 306 N / 974 O development rows. The sealed 70% test was not loaded, "
        "inferred, or scored.",
        "",
        "## Reproducibility gate",
        "",
        f"Status: **{'PASS' if gate['passed'] else 'FAIL'}**",
        "",
    ]
    if result.get("status") != "completed":
        lines.append("Supplementary methods were not run because the internal baseline gate failed.")
        return "\n".join(lines) + "\n"

    display_names = {
        "energy": "Energy (existing)",
        "negative_entropy": "Negative entropy (existing)",
        "mahalanobis_bank_32000": "Mahalanobis 32k (existing)",
        "gen_gamma_0.1_m16": "GEN (gamma=0.1, M=16)",
        "relative_mahalanobis_bank_32000": "Relative Mahalanobis 32k",
        "vim_dim_512_bank_32000": "ViM 32k (DIM=512)",
        "react_p90_energy_bank_32000": "ReAct p90 + Energy",
    }
    rows = {
        **result["comparison_baselines"],
        **result["supplementary_signals"],
    }
    lines.extend(
        [
            "## Results",
            "",
            "| Signal | AUROC ↑ | AUROC 95% CI | FPR95 ↓ | FPR95 95% CI | T false reject ↓ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in rows.items():
        auroc_ci = metrics["auroc_95ci"]
        fpr_ci = metrics["fpr95_95ci"]
        lines.append(
            f"| {display_names[name]} | {metrics['auroc']:.4f} | "
            f"[{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}] | {metrics['fpr95']:.4f} | "
            f"[{fpr_ci[0]:.4f}, {fpr_ci[1]:.4f}] | "
            f"{metrics['t_false_rejection_rate']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Paired comparison against each method's parent baseline",
            "",
            "| Candidate | Baseline | ΔAUROC | ΔAUROC 95% CI | ΔFPR95 | ΔFPR95 95% CI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name, delta in result["paired_deltas_vs_parent_baseline"].items():
        auroc_ci = delta["delta_auroc_95ci"]
        fpr_ci = delta["delta_fpr95_95ci"]
        lines.append(
            f"| {display_names[name]} | {display_names[delta['baseline']]} | "
            f"{delta['delta_auroc']:+.4f} | [{auroc_ci[0]:+.4f}, {auroc_ci[1]:+.4f}] | "
            f"{delta['delta_fpr95']:+.4f} | [{fpr_ci[0]:+.4f}, {fpr_ci[1]:+.4f}] |"
        )

    lines.extend(
        [
            "",
            "Positive ΔAUROC and negative ΔFPR95 favor the candidate. Each method uses its own "
            "99%-acceptance threshold calibrated on the same 8k RVL validation set; T is report-only.",
            "",
            "ReAct is an explicit transfer of the published clipping rule to signed, "
            "LayerNorm-normalized DiT `h_doc` features, so that result should be interpreted "
            "as an architecture-transfer baseline rather than an exact CNN reproduction.",
        ]
    )
    return "\n".join(lines) + "\n"


def supplementary_results_frame(result: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, methods in (
        ("existing_comparator", result.get("comparison_baselines", {})),
        ("supplementary", result.get("supplementary_signals", {})),
    ):
        for name, metrics in methods.items():
            rows.append(
                {
                    "group": group,
                    "signal": name,
                    "auroc": metrics["auroc"],
                    "auroc_ci_low": metrics["auroc_95ci"][0],
                    "auroc_ci_high": metrics["auroc_95ci"][1],
                    "fpr95": metrics["fpr95"],
                    "fpr95_ci_low": metrics["fpr95_95ci"][0],
                    "fpr95_ci_high": metrics["fpr95_95ci"][1],
                    "t_false_rejection_rate": metrics["t_false_rejection_rate"],
                }
            )
    return pd.DataFrame(rows)
