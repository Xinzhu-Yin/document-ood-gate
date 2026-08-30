from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .final_cache import extract_sealed_feature_cache, load_sealed_cache
from .final_evaluation import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    PRIMARY_D,
    evaluate_normalization_grid,
    evaluate_replication_predictions,
    evaluate_residual_d_sensitivity,
    evaluate_sealed_baselines,
    layer_norm_pooler,
)
from .final_reporting import write_development_reports, write_sealed_reports
from .layer_cache import load_layer_cache_metadata
from .layer_study import _cache_arrays, reconstruct_h_doc


class FinalProtocolError(RuntimeError):
    """Raised when the final frozen protocol differs from the requested design."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_final_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise FinalProtocolError("Final experiment config must be a YAML mapping")
    expected = {
        "model": "microsoft/dit-base-finetuned-rvlcdip",
        "reference_rows": 32000,
        "development_n": 306,
        "development_o": 974,
        "sealed_n": 694,
        "sealed_o": 2324,
        "k": 10,
        "primary_D": 512,
        "bootstrap_iterations": 2000,
        "bootstrap_seed": 20260812,
        "cache_dtype": "float32",
        "sealed_reference_reextract": False,
        "post_sealed_iteration": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise FinalProtocolError(f"Frozen config {key} must be {value!r}")
    if config.get("normalizations") != ["raw_patch_mean", "ln_free", "ln_pooler"]:
        raise FinalProtocolError("Frozen normalization grid changed")
    if config.get("methods") != [
        "mahalanobis",
        "knn",
        "mean_centered_pca_residual",
    ]:
        raise FinalProtocolError("Frozen primary detector grid changed")
    if config.get("diagnostic_layers") != [3, 7, 9, 11]:
        raise FinalProtocolError("Frozen diagnostic layers changed")
    if config.get("diagnostic_D") != [128, 256, 384, 512]:
        raise FinalProtocolError("Frozen diagnostic D values changed")
    return config


def _load_layer_arrays(cache_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = load_layer_cache_metadata(cache_dir, require_complete=True)
    arrays = _cache_arrays(cache_dir, metadata)
    return metadata, arrays


def _records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise FinalProtocolError(f"Result JSON has no records list: {path}")
    return records


def run_final_development(
    *,
    config_path: Path,
    layer_cache_dir: Path,
    existing_layer_results: Path,
    output_dir: Path,
    knn_device: str = "cuda",
) -> dict[str, Any]:
    """Run A+B without loading a split manifest, image, or sealed feature."""

    validate_final_config(config_path)
    started_at = _utc_now()
    clock = time.perf_counter()
    cache_metadata, arrays = _load_layer_arrays(layer_cache_dir)
    if cache_metadata.get("sealed") != {"loaded": 0, "scored": 0}:
        raise FinalProtocolError("Layer cache does not prove zero sealed access")
    eps = float(cache_metadata["pooler_layernorm_eps"])
    development_is_n = arrays["development_subsets"].astype(str) == "N"
    if int(development_is_n.sum()) != 306 or int((~development_is_n).sum()) != 974:
        raise FinalProtocolError("Development cache does not contain frozen 306 N + 974 O")

    normalization_rows = evaluate_normalization_grid(
        reference_raw=arrays["reference_raw"],
        query_raw=arrays["development_raw"],
        reference_labels=arrays["reference_labels"],
        query_is_n=development_is_n,
        layernorm_weight=arrays["layernorm_weight"],
        layernorm_bias=arrays["layernorm_bias"],
        layernorm_eps=eps,
        split="development",
        experiment="A",
        k=10,
        principal_dim=PRIMARY_D,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
        knn_device=knn_device,
    )
    sensitivity_rows = evaluate_residual_d_sensitivity(
        reference_raw=arrays["reference_raw"],
        query_raw=arrays["development_raw"],
        query_is_n=development_is_n,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
    )

    reference_h_doc = reconstruct_h_doc(
        arrays["reference_raw"][:, -1, :],
        arrays["layernorm_weight"],
        arrays["layernorm_bias"],
        eps=eps,
    )
    development_h_doc = reconstruct_h_doc(
        arrays["development_raw"][:, -1, :],
        arrays["layernorm_weight"],
        arrays["layernorm_bias"],
        eps=eps,
    )
    ln_pooler_reference = layer_norm_pooler(
        arrays["reference_raw"][:, -1, :],
        arrays["layernorm_weight"],
        arrays["layernorm_bias"],
        eps=eps,
    )
    ln_pooler_development = layer_norm_pooler(
        arrays["development_raw"][:, -1, :],
        arrays["layernorm_weight"],
        arrays["layernorm_bias"],
        eps=eps,
    )
    feature_error = max(
        float(np.max(np.abs(reference_h_doc - ln_pooler_reference))),
        float(np.max(np.abs(development_h_doc - ln_pooler_development))),
    )
    if feature_error >= 1.0e-5:
        raise FinalProtocolError(f"L12 LN-pooler/h_doc feature sanity failed: {feature_error}")

    existing = pd.read_csv(existing_layer_results)
    existing_h_doc = existing[
        (existing["experiment"] == "S1")
        & (existing["representation"] == "h_doc")
        & (existing["bank_size"] == 32000)
    ].set_index("method")
    current_h_doc = pd.DataFrame(normalization_rows)
    current_h_doc = current_h_doc[
        (current_h_doc["layer"] == 12)
        & (current_h_doc["representation"] == "ln_pooler")
    ].set_index("method")
    metrics = ["auroc", "fpr95", "auroc_ci_low", "auroc_ci_high", "fpr95_ci_low", "fpr95_ci_high"]
    detector_error = float(
        np.max(
            np.abs(
                current_h_doc.loc[existing_h_doc.index, metrics].to_numpy(dtype=float)
                - existing_h_doc.loc[:, metrics].to_numpy(dtype=float)
            )
        )
    )
    if detector_error >= 1.0e-10:
        raise FinalProtocolError(
            f"L12 LN-pooler detector results differ from existing h_doc: {detector_error}"
        )

    sanity = {
        "l12_ln_pooler_vs_h_doc_max_abs_error": feature_error,
        "l12_detector_result_max_abs_error": detector_error,
        "required_feature_tolerance": 1.0e-5,
        "passed": True,
    }
    metadata = {
        "scope": "development_only_final_ablation",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": time.perf_counter() - clock,
        "reference_rows": 32000,
        "development_N": 306,
        "development_O": 974,
        "feature_extraction_images": 0,
        "reused_layer_cache": str(layer_cache_dir),
        "sealed_rows_loaded": 0,
        "sealed_rows_scored": 0,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_D": PRIMARY_D,
        "D_sensitivity_role": "diagnostic_not_hyperparameter_optimization",
        "diagnostic_layer_selection": (
            "post_hoc representative early, transition, and late depth regions; not model selection"
        ),
    }
    paths = write_development_reports(
        normalization_rows,
        sensitivity_rows,
        sanity=sanity,
        metadata=metadata,
        output_dir=output_dir,
    )
    return {"metadata": metadata, "sanity": sanity, "paths": {k: str(v) for k, v in paths.items()}}


def run_final_sealed(
    *,
    config_path: Path,
    split_manifest_path: Path,
    layer_cache_dir: Path,
    sealed_cache_dir: Path,
    development_results_dir: Path,
    output_dir: Path,
    freeze_commit_sha: str,
    freeze_commit_timestamp: str,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """Open the sealed split once, run the frozen D1/D2 grid, report, and stop."""

    validate_final_config(config_path)
    started_at = _utc_now()
    clock = time.perf_counter()
    extraction_metadata = extract_sealed_feature_cache(
        split_manifest_path=split_manifest_path,
        output_dir=sealed_cache_dir,
        freeze_commit_sha=freeze_commit_sha,
        freeze_commit_timestamp=freeze_commit_timestamp,
        batch_size=batch_size,
        device=device,
    )
    cache_metadata, arrays = _load_layer_arrays(layer_cache_dir)
    sealed = load_sealed_cache(sealed_cache_dir)
    eps = float(cache_metadata["pooler_layernorm_eps"])
    sealed_is_n = sealed["subsets"].astype(str) == "N"
    if int(sealed_is_n.sum()) != 694 or int((~sealed_is_n).sum()) != 2324:
        raise FinalProtocolError("Sealed cache does not contain frozen 694 N + 2,324 O")
    reference_h_doc = reconstruct_h_doc(
        arrays["reference_raw"][:, -1, :],
        arrays["layernorm_weight"],
        arrays["layernorm_bias"],
        eps=eps,
    )
    reconstructed_sealed_h_doc = reconstruct_h_doc(
        sealed["raw"][:, -1, :],
        arrays["layernorm_weight"],
        arrays["layernorm_bias"],
        eps=eps,
    )
    h_doc_error = float(np.max(np.abs(reconstructed_sealed_h_doc - sealed["h_doc"])))
    if h_doc_error >= 1.0e-5:
        raise FinalProtocolError(f"Sealed h_doc reconstruction failed: {h_doc_error}")

    baseline_rows = evaluate_sealed_baselines(
        reference_h_doc=reference_h_doc,
        reference_logits=arrays["reference_logits"],
        reference_labels=arrays["reference_labels"],
        sealed_h_doc=sealed["h_doc"],
        sealed_logits=sealed["logits"],
        sealed_is_n=sealed_is_n,
        classifier_weight=arrays["classifier_weight"],
        classifier_bias=arrays["classifier_bias"],
        k=10,
        principal_dim=PRIMARY_D,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
        knn_device=device,
    )
    grid_rows = evaluate_normalization_grid(
        reference_raw=arrays["reference_raw"],
        query_raw=sealed["raw"],
        reference_labels=arrays["reference_labels"],
        query_is_n=sealed_is_n,
        layernorm_weight=arrays["layernorm_weight"],
        layernorm_bias=arrays["layernorm_bias"],
        layernorm_eps=eps,
        split="sealed_test",
        experiment="D2",
        k=10,
        principal_dim=PRIMARY_D,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
        knn_device=device,
    )
    development_rows = _records(
        development_results_dir / "development_normalization_results.json"
    )
    sensitivity_rows = _records(
        development_results_dir / "development_residual_d_sensitivity.json"
    )
    prediction_rows = evaluate_replication_predictions(baseline_rows, grid_rows)
    metadata = {
        "scope": "final_sealed_test_once",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": time.perf_counter() - clock,
        "freeze_commit_sha": freeze_commit_sha,
        "freeze_commit_timestamp": freeze_commit_timestamp,
        "sealed_access_started_at": extraction_metadata["started_at"],
        "freeze_predates_sealed_access": (
            datetime.fromisoformat(freeze_commit_timestamp)
            < datetime.fromisoformat(extraction_metadata["started_at"])
        ),
        "sealed_extracted_rows": extraction_metadata["extracted_rows"],
        "sealed_image_open_count": extraction_metadata["image_open_count"],
        "sealed_forward_count_per_image": extraction_metadata["forward_count_per_image"],
        "sealed_N": 694,
        "sealed_O": 2324,
        "reference_cache_reused": extraction_metadata["reference_cache_reused"],
        "reference_images_reextracted": extraction_metadata["reference_images_extracted"],
        "sealed_h_doc_reconstruction_max_abs_error": h_doc_error,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_D": PRIMARY_D,
        "forbidden_sealed_runs": {
            "D_sensitivity": 0,
            "8k_sensitivity": 0,
            "CLS": 0,
            "new_methods": 0,
        },
        "post_sealed_experimental_iteration": False,
    }
    if metadata["freeze_predates_sealed_access"] is not True:
        raise FinalProtocolError("Freeze commit timestamp does not predate sealed access")
    paths = write_sealed_reports(
        baseline_rows,
        grid_rows,
        development_rows,
        prediction_rows,
        sensitivity_rows,
        metadata=metadata,
        output_dir=output_dir,
    )
    return {
        "metadata": metadata,
        "predictions": prediction_rows,
        "paths": {key: str(value) for key, value in paths.items()},
    }
