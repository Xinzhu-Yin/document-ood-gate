from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .io import write_json
from .layer_cache import (
    DEVELOPMENT_N,
    DEVELOPMENT_O,
    DEVELOPMENT_ROWS,
    FEATURE_DIM,
    NUM_LAYERS,
    REFERENCE_8K_ROWS,
    REFERENCE_ROWS,
    build_layer_feature_cache,
    load_layer_cache_metadata,
)
from .layer_evaluation import G2ReproductionError, evaluate_layer_study
from .layer_reporting import write_layer_scan_reports


class LayerStudyProtocolError(ValueError):
    """Raised before execution when a setting differs from the frozen protocol."""


def _nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise LayerStudyProtocolError(f"Layer-scan config is missing {'.'.join(keys)}")
        value = value[key]
    return value


def validate_frozen_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise LayerStudyProtocolError("Layer-scan config must be a YAML mapping")
    expected = {
        ("study", "scope"): "development_only",
        ("study", "seed"): 20260812,
        ("study", "sealed_test_access"): "forbidden",
        ("model", "id"): "microsoft/dit-base-finetuned-rvlcdip",
        ("model", "revision"): "23f8b03d130fb66bbc9a15df3c75d753e49240eb",
        ("model", "frozen"): True,
        ("model", "inference_dtype"): "float32",
        ("model", "feature_dim"): FEATURE_DIM,
        ("model", "layers"): NUM_LAYERS,
        ("data", "one_pass_images"): REFERENCE_ROWS + DEVELOPMENT_ROWS,
        ("data", "reference_32k", "size"): REFERENCE_ROWS,
        ("data", "nested_reference_8k", "size"): REFERENCE_8K_ROWS,
        ("data", "development", "N"): DEVELOPMENT_N,
        ("data", "development", "O"): DEVELOPMENT_O,
        ("data", "development", "total"): DEVELOPMENT_ROWS,
        ("data", "sealed_test", "policy"): "never_read_or_score",
        ("hard_gates", "G1", "max_abs_error"): 1.0e-5,
        ("hard_gates", "G2", "target_auroc"): 0.7416556,
        ("hard_gates", "G2", "absolute_tolerance"): 0.005,
        ("methods", "knn", "k"): 10,
        ("methods", "mean_centered_pca_residual", "principal_dim"): 512,
        ("methods", "vim", "principal_dim"): 512,
        ("evaluation", "bootstrap_iterations"): 2000,
        ("evaluation", "bootstrap_seed"): 20260812,
        ("evaluation", "cache_dtype"): "float32",
        ("evaluation", "hyperparameter_tuning"): False,
    }
    for keys, expected_value in expected.items():
        observed = _nested(config, *keys)
        if observed != expected_value:
            raise LayerStudyProtocolError(
                f"Frozen setting {'.'.join(keys)} must be {expected_value!r}; "
                f"found {observed!r}"
            )
    return config


def reconstruct_h_doc(
    raw_patch_mean: np.ndarray,
    layernorm_weight: np.ndarray,
    layernorm_bias: np.ndarray,
    *,
    eps: float,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Reconstruct pooler h_doc from cached L12 raw means in float32 chunks."""

    values = np.asarray(raw_patch_mean)
    weight = np.asarray(layernorm_weight)
    bias = np.asarray(layernorm_bias)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM:
        raise ValueError(f"raw_patch_mean must have shape [rows, {FEATURE_DIM}]")
    if values.dtype != np.float32:
        raise ValueError("raw_patch_mean must be float32")
    if weight.shape != (FEATURE_DIM,) or bias.shape != (FEATURE_DIM,):
        raise ValueError("LayerNorm weight and bias must each contain 768 values")
    if weight.dtype != np.float32 or bias.dtype != np.float32:
        raise ValueError("LayerNorm parameters must be float32")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("LayerNorm eps must be finite and positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    output = np.empty(values.shape, dtype=np.float32)
    epsilon = np.float32(eps)
    for start in range(0, len(values), chunk_size):
        end = min(len(values), start + chunk_size)
        batch = np.asarray(values[start:end], dtype=np.float32)
        mean = batch.mean(axis=1, keepdims=True, dtype=np.float32)
        centered = batch - mean
        variance = np.mean(centered * centered, axis=1, keepdims=True, dtype=np.float32)
        output[start:end] = centered / np.sqrt(variance + epsilon) * weight + bias
        if not np.isfinite(output[start:end]).all():
            raise ValueError("Reconstructed h_doc contains NaN or infinity")
    return output


def _load_array(cache_dir: Path, name: str, *, mmap: bool = True) -> np.ndarray:
    path = cache_dir / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def _cache_arrays(cache_dir: Path, metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    if metadata.get("complete") is not True:
        raise LayerStudyProtocolError("Layer feature cache is not complete")
    if metadata.get("sealed") != {"loaded": 0, "scored": 0}:
        raise LayerStudyProtocolError("Layer feature cache does not prove zero sealed access")

    arrays = {
        "reference_raw": _load_array(cache_dir, "reference_raw_patch_mean"),
        "reference_cls": _load_array(cache_dir, "reference_cls"),
        "reference_logits": _load_array(cache_dir, "reference_logits"),
        "development_raw": _load_array(cache_dir, "development_raw_patch_mean"),
        "development_cls": _load_array(cache_dir, "development_cls"),
        "development_logits": _load_array(cache_dir, "development_logits"),
        "reference_labels": _load_array(cache_dir, "reference_labels", mmap=False),
        "reference_8k_indices": _load_array(cache_dir, "reference_8k_indices", mmap=False),
        "development_subsets": _load_array(cache_dir, "development_subsets", mmap=False),
        "classifier_weight": _load_array(cache_dir, "classifier_weight", mmap=False),
        "classifier_bias": _load_array(cache_dir, "classifier_bias", mmap=False),
        "layernorm_weight": _load_array(cache_dir, "pooler_layernorm_weight", mmap=False),
        "layernorm_bias": _load_array(cache_dir, "pooler_layernorm_bias", mmap=False),
    }
    expected_shapes = {
        "reference_raw": (REFERENCE_ROWS, NUM_LAYERS, FEATURE_DIM),
        "reference_cls": (REFERENCE_ROWS, NUM_LAYERS, FEATURE_DIM),
        "reference_logits": (REFERENCE_ROWS, 16),
        "development_raw": (DEVELOPMENT_ROWS, NUM_LAYERS, FEATURE_DIM),
        "development_cls": (DEVELOPMENT_ROWS, NUM_LAYERS, FEATURE_DIM),
        "development_logits": (DEVELOPMENT_ROWS, 16),
        "reference_labels": (REFERENCE_ROWS,),
        "reference_8k_indices": (REFERENCE_8K_ROWS,),
        "development_subsets": (DEVELOPMENT_ROWS,),
        "classifier_weight": (16, FEATURE_DIM),
        "classifier_bias": (16,),
        "layernorm_weight": (FEATURE_DIM,),
        "layernorm_bias": (FEATURE_DIM,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise LayerStudyProtocolError(
                f"Cache array {name} must have shape {shape}; found {arrays[name].shape}"
            )
    for name in (
        "reference_raw",
        "reference_cls",
        "reference_logits",
        "development_raw",
        "development_cls",
        "development_logits",
        "classifier_weight",
        "classifier_bias",
        "layernorm_weight",
        "layernorm_bias",
    ):
        if arrays[name].dtype != np.float32:
            raise LayerStudyProtocolError(f"Cache array {name} must be float32")
    return arrays


def _duration_from_metadata(metadata: dict[str, Any]) -> float | None:
    try:
        start = datetime.fromisoformat(str(metadata["started_at"]))
        finish = datetime.fromisoformat(str(metadata["finished_at"]))
    except (KeyError, TypeError, ValueError):
        return None
    return max(0.0, (finish - start).total_seconds())


def run_frozen_layer_study(
    *,
    config_path: Path,
    reference_manifest_path: Path,
    development_manifest_path: Path,
    cache_dir: Path,
    results_dir: Path,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """Execute extraction, G1/G2, fixed S1/S1b/S2, and reporting in order."""

    started = datetime.now(UTC)
    total_clock = time.perf_counter()
    config = validate_frozen_config(config_path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    extraction_clock = time.perf_counter()
    build_layer_feature_cache(
        reference_manifest_path,
        development_manifest_path,
        cache_dir,
        batch_size=batch_size,
        device=device,
    )
    extraction_call_seconds = time.perf_counter() - extraction_clock
    cache_metadata = load_layer_cache_metadata(cache_dir, require_complete=True)
    g1 = dict(cache_metadata["g1"])
    if g1.get("passed") is not True:
        raise RuntimeError("G1 did not produce a passing cache record")

    arrays = _cache_arrays(cache_dir, cache_metadata)
    eps = float(cache_metadata["pooler_layernorm_eps"])
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

    evaluation_clock = time.perf_counter()
    try:
        evaluation = evaluate_layer_study(
            reference_raw_patch_mean=np.moveaxis(arrays["reference_raw"], 1, 0),
            reference_cls=np.moveaxis(arrays["reference_cls"], 1, 0),
            reference_h_doc=reference_h_doc,
            reference_logits=arrays["reference_logits"],
            reference_labels=arrays["reference_labels"],
            reference_8k_indices=arrays["reference_8k_indices"],
            reference_8k_labels=arrays["reference_labels"][arrays["reference_8k_indices"]],
            development_raw_patch_mean=np.moveaxis(arrays["development_raw"], 1, 0),
            development_cls=np.moveaxis(arrays["development_cls"], 1, 0),
            development_h_doc=development_h_doc,
            development_logits=arrays["development_logits"],
            development_is_n=arrays["development_subsets"].astype(str) == "N",
            classifier_weight=arrays["classifier_weight"],
            classifier_bias=arrays["classifier_bias"],
            k=10,
            principal_dim=512,
            bootstrap_iterations=2000,
            seed=20260812,
            g2_target_auroc=0.7416556,
            g2_tolerance=0.005,
            knn_device=device,
        )
    except G2ReproductionError as error:
        results_dir.mkdir(parents=True, exist_ok=True)
        failed_gates = {"G1": g1, "G2": {"passed": False, "detail": str(error)}}
        write_json(results_dir / "gates.json", failed_gates)
        write_json(
            results_dir / "run_metadata.json",
            {
                "scope": "development_only",
                "status": "stopped_at_G2",
                "sealed_rows_loaded": 0,
                "sealed_rows_scored": 0,
                "started_at": started.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
            },
        )
        raise
    evaluation_seconds = time.perf_counter() - evaluation_clock

    gates = {"G1": g1, "G2": evaluation["g2"]}
    reporting_metadata = {
        "scope": "development_only",
        "extraction_image_count": REFERENCE_ROWS + DEVELOPMENT_ROWS,
        "reference_image_count": REFERENCE_ROWS,
        "reference_8k_image_count": REFERENCE_8K_ROWS,
        "development_n_count": DEVELOPMENT_N,
        "development_o_count": DEVELOPMENT_O,
        "excluded_cross_label_conflicts": int(cache_metadata["excluded_cross_label_conflicts"]),
        "sealed_rows_loaded": 0,
        "sealed_rows_scored": 0,
        "sealed_metrics_computed": False,
        "t_guardrail_rescored": False,
        "cache_dtype": "float32",
        "cache_layout": cache_metadata["layout"],
        "reference_id_sha256": cache_metadata["reference_id_sha256"],
        "reference_8k_id_sha256": cache_metadata["reference_8k_id_sha256"],
        "development_id_sha256": cache_metadata["development_id_sha256"],
        "model_id": _nested(config, "model", "id"),
        "model_revision": _nested(config, "model", "revision"),
    }
    reporting_clock = time.perf_counter()
    rows = evaluation["rows"]
    report_paths = write_layer_scan_reports(
        rows,
        [row for row in rows if row["experiment"] == "S1b"],
        gates=gates,
        metadata=reporting_metadata,
        output_dir=results_dir,
    )
    reporting_seconds = time.perf_counter() - reporting_clock

    finished = datetime.now(UTC)
    run_metadata = {
        **reporting_metadata,
        "status": "completed",
        "branch": "exp/layer-scan",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "feature_extraction_elapsed_seconds": _duration_from_metadata(cache_metadata),
        "feature_extraction_call_seconds": extraction_call_seconds,
        "evaluation_seconds": evaluation_seconds,
        "reporting_seconds": reporting_seconds,
        "total_command_seconds": time.perf_counter() - total_clock,
        "result_row_count": int(len(rows)),
        "output_files": {name: str(path) for name, path in report_paths.items()},
    }
    write_json(results_dir / "gates.json", gates)
    write_json(results_dir / "run_metadata.json", run_metadata)
    write_json(
        results_dir / "study_result.json",
        {"metadata": run_metadata, "gates": gates, "evaluation": evaluation},
    )
    return {
        "gates": gates,
        "metadata": run_metadata,
        "evaluation": evaluation,
        "report_paths": report_paths,
    }


__all__ = [
    "LayerStudyProtocolError",
    "reconstruct_h_doc",
    "run_frozen_layer_study",
    "validate_frozen_config",
]
