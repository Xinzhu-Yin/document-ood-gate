from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image


MODEL_ID = "microsoft/dit-base-finetuned-rvlcdip"
MODEL_REVISION = "23f8b03d130fb66bbc9a15df3c75d753e49240eb"
DEVELOPMENT_MANIFEST_NAME = "research_split_development_only.csv"
REFERENCE_ROWS = 32_000
REFERENCE_8K_ROWS = 8_000
DEVELOPMENT_ROWS = 1_280
DEVELOPMENT_N = 306
DEVELOPMENT_O = 974
NUM_CLASSES = 16
NUM_LAYERS = 12
FEATURE_DIM = 768
CACHE_DTYPE = np.dtype("float32")
CACHE_VERSION = 1


class LayerCacheBoundaryError(ValueError):
    """Raised before an unsafe manifest, image, or cache boundary is crossed."""


class LayerCacheGateError(RuntimeError):
    """Raised when a hard structural gate fails."""


@dataclass(frozen=True)
class LayerCachePlan:
    reference_ids: tuple[str, ...]
    reference_paths: tuple[Path, ...]
    reference_labels: tuple[int, ...]
    reference_8k_ids: frozenset[str]
    reference_8k_indices: tuple[int, ...]
    development_ids: tuple[str, ...]
    development_paths: tuple[Path, ...]
    development_subsets: tuple[str, ...]
    excluded_cross_label_conflicts: int
    reference_manifest_path: Path
    development_manifest_path: Path

    @property
    def reference_id_sha256(self) -> str:
        return _ordered_id_hash(self.reference_ids)

    @property
    def reference_8k_id_sha256(self) -> str:
        return _set_id_hash(self.reference_8k_ids)

    @property
    def development_id_sha256(self) -> str:
        return _ordered_id_hash(self.development_ids)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _ordered_id_hash(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _set_id_hash(values: Sequence[str] | frozenset[str]) -> str:
    return _ordered_id_hash(tuple(sorted(str(value) for value in values)))


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LayerCacheBoundaryError(f"Expected a JSON object in {path}")
    return value


def _assert_balanced(labels: pd.Series, *, total: int, per_class: int, name: str) -> None:
    numeric = pd.to_numeric(labels, errors="raise").astype(np.int64)
    counts = numeric.value_counts().sort_index().to_dict()
    expected = {label: per_class for label in range(NUM_CLASSES)}
    if len(numeric) != total or counts != expected:
        raise LayerCacheBoundaryError(
            f"{name} must contain {total} rows balanced over 16 classes; found {counts}"
        )


def select_frozen_reference_manifest(
    manifest_path: Path,
    *,
    expected_8k_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the frozen nested banks by IDs/ranks, never by row position."""

    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)
    required = {
        "purpose",
        "relative_path",
        "local_path",
        "label",
        "class_sample_rank",
        "minimum_reference_size",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise LayerCacheBoundaryError(
            "Reference manifest is missing required columns: " + ", ".join(missing)
        )

    reference = frame.loc[frame["purpose"].astype(str).eq("reference")].copy()
    reference["class_sample_rank"] = pd.to_numeric(
        reference["class_sample_rank"], errors="raise"
    ).astype(np.int64)
    if (reference["class_sample_rank"] < 0).any():
        raise LayerCacheBoundaryError("Reference class_sample_rank must be non-negative")
    if reference["relative_path"].astype(str).duplicated().any():
        raise LayerCacheBoundaryError("Reference manifest contains duplicate image IDs")
    if reference["local_path"].astype(str).duplicated().any():
        raise LayerCacheBoundaryError("Reference manifest contains duplicate local paths")

    bank_32k = reference.loc[reference["class_sample_rank"] < 2_000].copy()
    bank_8k = reference.loc[reference["class_sample_rank"] < 500].copy()
    _assert_balanced(bank_32k["label"], total=REFERENCE_ROWS, per_class=2_000, name="32k")
    _assert_balanced(
        bank_8k["label"], total=REFERENCE_8K_ROWS, per_class=500, name="frozen 8k"
    )

    bank_8k_ids = frozenset(bank_8k["relative_path"].astype(str))
    minimum_size = pd.to_numeric(reference["minimum_reference_size"], errors="coerce")
    marked_8k_ids = frozenset(
        reference.loc[minimum_size.eq(REFERENCE_8K_ROWS), "relative_path"].astype(str)
    )
    if len(marked_8k_ids) != REFERENCE_8K_ROWS or marked_8k_ids != bank_8k_ids:
        raise LayerCacheBoundaryError(
            "Rank-selected 8k IDs do not exactly match the existing frozen 8k manifest IDs"
        )
    if expected_8k_ids is not None:
        expected = frozenset(str(value) for value in expected_8k_ids)
        if expected != bank_8k_ids:
            raise LayerCacheBoundaryError(
                "Rank-selected 8k IDs do not exactly match the supplied frozen 8k IDs"
            )

    # Preserve manifest order. The masks above, rather than iloc/first rows, define membership.
    return bank_32k.reset_index(drop=True), bank_8k.reset_index(drop=True)


def _validate_development_manifest_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.name != DEVELOPMENT_MANIFEST_NAME:
        raise LayerCacheBoundaryError(
            f"Only {DEVELOPMENT_MANIFEST_NAME} may be opened; refused {candidate.name}"
        )
    if any("sealed" in part.casefold() for part in candidate.parts):
        raise LayerCacheBoundaryError("Refused a development manifest beneath a sealed path")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise LayerCacheBoundaryError(f"Development-only manifest does not exist: {candidate}") from error
    if resolved.name != DEVELOPMENT_MANIFEST_NAME:
        raise LayerCacheBoundaryError(
            "Development-only manifest resolves to a different split file; refusing before open"
        )
    return resolved


def load_development_only_manifest(path: Path) -> tuple[pd.DataFrame, int]:
    """Open only the physically isolated development manifest and remove conflicts."""

    safe_path = _validate_development_manifest_path(path)
    frame = pd.read_csv(safe_path)
    required = {"path", "subset", "split", "split_group_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise LayerCacheBoundaryError(
            "Development-only manifest is missing required columns: " + ", ".join(missing)
        )
    splits = set(frame["split"].astype(str))
    if splits != {"development"}:
        raise LayerCacheBoundaryError(
            f"Development-only manifest must contain only development rows; found {splits}"
        )
    if not set(frame["subset"].astype(str)).issubset({"N", "O"}):
        raise LayerCacheBoundaryError("Development-only manifest contains a non-N/O subset")
    if frame["path"].astype(str).duplicated().any():
        raise LayerCacheBoundaryError("Development-only manifest contains duplicate paths")

    grouped = (
        frame.dropna(subset=["split_group_id"])
        .groupby("split_group_id", observed=True)["subset"]
        .nunique()
    )
    conflict_groups = set(grouped.loc[grouped > 1].index)
    conflict_mask = frame["split_group_id"].isin(conflict_groups)
    filtered = frame.loc[~conflict_mask].copy().reset_index(drop=True)
    counts = filtered["subset"].astype(str).value_counts().to_dict()
    if len(filtered) != DEVELOPMENT_ROWS or counts != {"N": DEVELOPMENT_N, "O": DEVELOPMENT_O}:
        raise LayerCacheBoundaryError(
            "Frozen development rows must be 306 N + 974 O after conflict removal; "
            f"found {counts} ({len(filtered)} total)"
        )
    return filtered, int(conflict_mask.sum())


def prepare_layer_cache_plan(
    reference_manifest_path: Path,
    development_manifest_path: Path,
    *,
    expected_8k_ids: Sequence[str] | None = None,
) -> LayerCachePlan:
    reference, reference_8k = select_frozen_reference_manifest(
        reference_manifest_path,
        expected_8k_ids=expected_8k_ids,
    )
    development, conflicts = load_development_only_manifest(development_manifest_path)
    reference_ids = tuple(reference["relative_path"].astype(str))
    frozen_ids = frozenset(reference_8k["relative_path"].astype(str))
    reference_8k_indices = tuple(
        index for index, image_id in enumerate(reference_ids) if image_id in frozen_ids
    )
    if len(reference_8k_indices) != REFERENCE_8K_ROWS:
        raise LayerCacheBoundaryError("Could not map all frozen 8k IDs into the 32k bank")

    plan = LayerCachePlan(
        reference_ids=reference_ids,
        reference_paths=tuple(Path(value) for value in reference["local_path"].astype(str)),
        reference_labels=tuple(pd.to_numeric(reference["label"]).astype(int)),
        reference_8k_ids=frozen_ids,
        reference_8k_indices=reference_8k_indices,
        development_ids=tuple(development["path"].astype(str)),
        development_paths=tuple(Path(value) for value in development["path"].astype(str)),
        development_subsets=tuple(development["subset"].astype(str)),
        excluded_cross_label_conflicts=conflicts,
        reference_manifest_path=Path(reference_manifest_path).resolve(),
        development_manifest_path=_validate_development_manifest_path(development_manifest_path),
    )
    _validate_plan(plan, strict_counts=True)
    _validate_production_paths(plan)
    return plan


def _validate_plan(plan: LayerCachePlan, *, strict_counts: bool) -> None:
    if not (
        len(plan.reference_ids)
        == len(plan.reference_paths)
        == len(plan.reference_labels)
    ):
        raise LayerCacheBoundaryError("Reference IDs, paths, and labels are not aligned")
    if not (
        len(plan.development_ids)
        == len(plan.development_paths)
        == len(plan.development_subsets)
    ):
        raise LayerCacheBoundaryError("Development IDs, paths, and subsets are not aligned")
    if len(set(plan.reference_ids)) != len(plan.reference_ids):
        raise LayerCacheBoundaryError("Reference plan contains duplicate IDs")
    if len(set(plan.development_ids)) != len(plan.development_ids):
        raise LayerCacheBoundaryError("Development plan contains duplicate IDs")
    if not plan.reference_8k_ids.issubset(set(plan.reference_ids)):
        raise LayerCacheBoundaryError("Frozen 8k IDs are not a subset of the 32k reference IDs")
    mapped_ids = {plan.reference_ids[index] for index in plan.reference_8k_indices}
    if mapped_ids != set(plan.reference_8k_ids):
        raise LayerCacheBoundaryError("Frozen 8k indices do not exactly map to frozen 8k IDs")
    if set(plan.reference_labels) - set(range(NUM_CLASSES)):
        raise LayerCacheBoundaryError("Reference labels must be in [0, 15]")
    if set(plan.development_subsets) - {"N", "O"}:
        raise LayerCacheBoundaryError("Development subsets must contain only N and O")
    if strict_counts:
        if len(plan.reference_ids) != REFERENCE_ROWS:
            raise LayerCacheBoundaryError("Strict plan must contain exactly 32,000 reference rows")
        if len(plan.reference_8k_ids) != REFERENCE_8K_ROWS:
            raise LayerCacheBoundaryError("Strict plan must contain exactly 8,000 frozen IDs")
        reference_counts = {
            label: plan.reference_labels.count(label) for label in range(NUM_CLASSES)
        }
        if reference_counts != {label: 2_000 for label in range(NUM_CLASSES)}:
            raise LayerCacheBoundaryError("Strict 32k plan is not class-balanced")
        development_counts = {
            subset: plan.development_subsets.count(subset) for subset in ("N", "O")
        }
        if development_counts != {"N": DEVELOPMENT_N, "O": DEVELOPMENT_O}:
            raise LayerCacheBoundaryError("Strict development plan is not 306 N + 974 O")


def _validate_production_paths(plan: LayerCachePlan) -> None:
    normalized: set[str] = set()
    for path in (*plan.reference_paths, *plan.development_paths):
        _assert_not_sealed_image_path(path)
        if not Path(path).is_file():
            raise LayerCacheBoundaryError(f"Frozen input image is missing or not a file: {path}")
        key = os.path.normcase(os.path.abspath(str(path)))
        if key in normalized:
            raise LayerCacheBoundaryError(f"Frozen 32k + development inputs repeat a path: {path}")
        normalized.add(key)
    expected = len(plan.reference_paths) + len(plan.development_paths)
    if len(normalized) != expected:
        raise LayerCacheBoundaryError("Frozen input paths are not unique")


def _assert_not_sealed_image_path(path: Path) -> None:
    if any("sealed" in part.casefold() for part in Path(path).parts):
        raise LayerCacheBoundaryError(f"Refused sealed image path before opening: {path}")


def _default_image_loader(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _close_images(images: Sequence[Any]) -> None:
    for image in images:
        close = getattr(image, "close", None)
        if callable(close):
            close()


def _copy_pooled_vectors_to_cpu(value: torch.Tensor, *, batch_rows: int) -> np.ndarray:
    expected = (batch_rows, NUM_LAYERS, FEATURE_DIM)
    if tuple(value.shape) != expected:
        raise LayerCacheGateError(
            f"Only pooled [batch, 12, 768] vectors may move to CPU; found {tuple(value.shape)}"
        )
    array = value.detach().to(dtype=torch.float32).cpu().numpy()
    if array.dtype != CACHE_DTYPE or array.shape != expected:
        raise LayerCacheGateError("Pooled CPU cache conversion changed shape or dtype")
    return array


def _copy_logits_to_cpu(value: torch.Tensor, *, batch_rows: int) -> np.ndarray:
    expected = (batch_rows, NUM_CLASSES)
    if tuple(value.shape) != expected:
        raise LayerCacheGateError(f"Classifier logits must have shape {expected}")
    array = value.detach().to(dtype=torch.float32).cpu().numpy()
    if array.dtype != CACHE_DTYPE:
        raise LayerCacheGateError("Logits cache must be float32")
    return array


def _forward_pooled_vectors(
    model: Any,
    pixel_values: torch.Tensor,
    *,
    run_g1: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    output = model.beit(
        pixel_values=pixel_values,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = output.hidden_states
    if hidden_states is None or len(hidden_states) != NUM_LAYERS + 1:
        found = None if hidden_states is None else len(hidden_states)
        raise LayerCacheGateError(f"Expected embedding + 12 hidden states; found {found}")

    batch_rows = int(pixel_values.shape[0])
    cls_by_layer: list[torch.Tensor] = []
    raw_mean_by_layer: list[torch.Tensor] = []
    for layer_index in range(1, NUM_LAYERS + 1):
        hidden = hidden_states[layer_index]
        if hidden.ndim != 3:
            raise LayerCacheGateError(f"Layer {layer_index} hidden state must be rank 3")
        if hidden.shape[0] != batch_rows or hidden.shape[1] < 2 or hidden.shape[2] != FEATURE_DIM:
            raise LayerCacheGateError(
                f"Layer {layer_index} has invalid shape {tuple(hidden.shape)}"
            )
        if hidden.dtype != torch.float32:
            raise LayerCacheGateError(
                f"Layer {layer_index} must be float32; found {hidden.dtype}"
            )
        # Pool on the model device. No full 197x768 hidden state crosses to CPU.
        cls_by_layer.append(hidden[:, 0, :])
        raw_mean_by_layer.append(hidden[:, 1:, :].mean(dim=1))

    cls_gpu = torch.stack(cls_by_layer, dim=1)
    raw_mean_gpu = torch.stack(raw_mean_by_layer, dim=1)
    g1: dict[str, Any] | None = None
    if run_g1:
        pooler = getattr(getattr(model, "beit", None), "pooler", None)
        layernorm = getattr(pooler, "layernorm", None)
        if layernorm is None:
            raise LayerCacheGateError("G1 requires model.beit.pooler.layernorm")
        reconstructed = layernorm(raw_mean_by_layer[-1])
        pooler_output = output.pooler_output
        if pooler_output is None or tuple(pooler_output.shape) != (batch_rows, FEATURE_DIM):
            raise LayerCacheGateError("G1 requires a [batch, 768] pooler_output")
        max_abs_error = float(
            (reconstructed.float() - pooler_output.float()).abs().max().item()
        )
        g1 = {
            "passed": max_abs_error < 1.0e-5,
            "max_abs_error": max_abs_error,
            "tolerance": 1.0e-5,
            "batch_rows": batch_rows,
        }
        if not g1["passed"]:
            raise LayerCacheGateError(
                f"G1 failed: reconstructed h_doc max_abs_error={max_abs_error:.9g}"
            )

    logits_gpu = model.classifier(output.pooler_output)
    cls_cpu = _copy_pooled_vectors_to_cpu(cls_gpu, batch_rows=batch_rows)
    raw_mean_cpu = _copy_pooled_vectors_to_cpu(raw_mean_gpu, batch_rows=batch_rows)
    logits_cpu = _copy_logits_to_cpu(logits_gpu, batch_rows=batch_rows)
    return cls_cpu, raw_mean_cpu, logits_cpu, g1


def _parameter_array(value: torch.Tensor, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    array = value.detach().to(dtype=torch.float32).cpu().numpy()
    if array.shape != expected_shape or array.dtype != CACHE_DTYPE:
        raise LayerCacheGateError(
            f"{name} must be float32 with shape {expected_shape}; found {array.shape}/{array.dtype}"
        )
    return array


def _model_parameters(model: Any) -> dict[str, np.ndarray]:
    classifier = getattr(model, "classifier", None)
    pooler = getattr(getattr(model, "beit", None), "pooler", None)
    layernorm = getattr(pooler, "layernorm", None)
    if classifier is None or layernorm is None:
        raise LayerCacheGateError("Expected classifier and model.beit.pooler.layernorm")
    return {
        "classifier_weight": _parameter_array(
            classifier.weight, (NUM_CLASSES, FEATURE_DIM), "classifier_weight"
        ),
        "classifier_bias": _parameter_array(
            classifier.bias, (NUM_CLASSES,), "classifier_bias"
        ),
        "pooler_layernorm_weight": _parameter_array(
            layernorm.weight, (FEATURE_DIM,), "pooler_layernorm_weight"
        ),
        "pooler_layernorm_bias": _parameter_array(
            layernorm.bias, (FEATURE_DIM,), "pooler_layernorm_bias"
        ),
    }


def _pooler_layernorm_eps(model: Any) -> float:
    pooler = getattr(getattr(model, "beit", None), "pooler", None)
    layernorm = getattr(pooler, "layernorm", None)
    if layernorm is None or not hasattr(layernorm, "eps"):
        raise LayerCacheGateError("Expected model.beit.pooler.layernorm.eps")
    eps = float(layernorm.eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise LayerCacheGateError(f"pooler LayerNorm eps must be positive and finite; found {eps}")
    return eps


def _cache_shapes(plan: LayerCachePlan) -> dict[str, tuple[int, ...]]:
    return {
        "reference_raw_patch_mean": (len(plan.reference_ids), NUM_LAYERS, FEATURE_DIM),
        "reference_cls": (len(plan.reference_ids), NUM_LAYERS, FEATURE_DIM),
        "reference_logits": (len(plan.reference_ids), NUM_CLASSES),
        "development_raw_patch_mean": (
            len(plan.development_ids),
            NUM_LAYERS,
            FEATURE_DIM,
        ),
        "development_cls": (len(plan.development_ids), NUM_LAYERS, FEATURE_DIM),
        "development_logits": (len(plan.development_ids), NUM_CLASSES),
    }


def _cache_spec(plan: LayerCachePlan) -> dict[str, Any]:
    shapes = _cache_shapes(plan)
    return {
        "cache_version": CACHE_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": CACHE_DTYPE.name,
        "layout": "[rows, 12, 768]",
        "hidden_state_indices": list(range(1, NUM_LAYERS + 1)),
        "reference_rows": len(plan.reference_ids),
        "development_rows": len(plan.development_ids),
        "reference_id_sha256": plan.reference_id_sha256,
        "reference_8k_id_sha256": plan.reference_8k_id_sha256,
        "development_id_sha256": plan.development_id_sha256,
        "arrays": {
            name: {"file": f"{name}.npy", "shape": list(shape), "dtype": "float32"}
            for name, shape in shapes.items()
        },
        "sealed": {"loaded": 0, "scored": 0},
    }


def _write_static_plan_files(output_dir: Path, plan: LayerCachePlan) -> None:
    _atomic_write_text(
        output_dir / "reference_ids.jsonl",
        "".join(json.dumps(value) + "\n" for value in plan.reference_ids),
    )
    _atomic_write_text(
        output_dir / "development_ids.jsonl",
        "".join(json.dumps(value) + "\n" for value in plan.development_ids),
    )
    np.save(output_dir / "reference_labels.npy", np.asarray(plan.reference_labels, dtype=np.int64))
    np.save(
        output_dir / "reference_8k_indices.npy",
        np.asarray(plan.reference_8k_indices, dtype=np.int64),
    )
    np.save(
        output_dir / "development_subsets.npy",
        np.asarray(plan.development_subsets, dtype="<U1"),
    )


def _open_or_create_cache(
    output_dir: Path,
    plan: LayerCachePlan,
    *,
    requested_batch_size: int,
) -> tuple[dict[str, np.memmap], dict[str, Any]]:
    output_dir = Path(output_dir)
    spec_path = output_dir / "cache_spec.json"
    progress_path = output_dir / "progress.json"
    expected_spec = _cache_spec(plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_entries = list(output_dir.iterdir())

    if spec_path.exists():
        found_spec = _read_json(spec_path)
        if found_spec != expected_spec:
            raise LayerCacheBoundaryError(
                "Existing layer cache spec does not match the frozen plan; refusing overwrite"
            )
        if not progress_path.exists():
            raise LayerCacheBoundaryError("Existing layer cache has no resumable progress record")
        progress = _read_json(progress_path)
        mode = "r+"
    else:
        if existing_entries:
            raise LayerCacheBoundaryError(
                "Output directory is non-empty but has no matching cache spec; refusing overwrite"
            )
        _atomic_write_json(spec_path, expected_spec)
        progress = {
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
            "completed_rows": {"reference": 0, "development": 0},
            "requested_batch_size": requested_batch_size,
            "effective_batch_size": requested_batch_size,
            "g1": {"passed": False, "max_abs_error": None, "tolerance": 1.0e-5},
            "sealed": {"loaded": 0, "scored": 0},
        }
        _atomic_write_json(progress_path, progress)
        _write_static_plan_files(output_dir, plan)
        mode = "w+"

    completed = progress.get("completed_rows", {})
    for split, total in (
        ("reference", len(plan.reference_ids)),
        ("development", len(plan.development_ids)),
    ):
        value = completed.get(split)
        if not isinstance(value, int) or not 0 <= value <= total:
            raise LayerCacheBoundaryError(f"Invalid resumable completed_rows for {split}: {value}")
    if completed["reference"] > 0 and not progress.get("g1", {}).get("passed", False):
        raise LayerCacheBoundaryError("Reference rows exist without a passed G1 record")
    if progress.get("sealed") != {"loaded": 0, "scored": 0}:
        raise LayerCacheBoundaryError("Existing cache reports sealed access; refusing resume")

    arrays: dict[str, np.memmap] = {}
    for name, shape in _cache_shapes(plan).items():
        path = output_dir / f"{name}.npy"
        if mode == "w+":
            arrays[name] = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=CACHE_DTYPE,
                shape=shape,
            )
        else:
            array = np.lib.format.open_memmap(path, mode="r+")
            if array.dtype != CACHE_DTYPE or tuple(array.shape) != shape:
                raise LayerCacheBoundaryError(f"Existing cache array mismatch: {name}")
            arrays[name] = array
    return arrays, progress


def _write_or_validate_parameters(output_dir: Path, model: Any, *, resume: bool) -> None:
    for name, values in _model_parameters(model).items():
        path = output_dir / f"{name}.npy"
        if resume:
            if not path.exists():
                raise LayerCacheBoundaryError(f"Resume is missing frozen parameter file: {name}")
            existing = np.load(path, allow_pickle=False)
            if not np.array_equal(existing, values):
                raise LayerCacheBoundaryError(
                    f"Frozen model parameter changed across resume: {name}"
                )
        else:
            np.save(path, values)


def _validate_loaded_model(model: Any) -> None:
    config = getattr(model, "config", None)
    if config is not None:
        if int(getattr(config, "hidden_size", FEATURE_DIM)) != FEATURE_DIM:
            raise LayerCacheGateError("Loaded model hidden_size is not 768")
        if int(getattr(config, "num_hidden_layers", NUM_LAYERS)) != NUM_LAYERS:
            raise LayerCacheGateError("Loaded model does not have 12 transformer layers")
        if int(getattr(config, "num_labels", NUM_CLASSES)) != NUM_CLASSES:
            raise LayerCacheGateError("Loaded model does not have 16 classifier outputs")
    if not hasattr(model, "beit") or not hasattr(model, "classifier"):
        raise LayerCacheGateError("Expected a BEiT/DiT image-classification model")


def _metadata_snapshot(
    plan: LayerCachePlan,
    progress: dict[str, Any],
    *,
    complete: bool,
) -> dict[str, Any]:
    return {
        **_cache_spec(plan),
        "complete": complete,
        "completed_rows": dict(progress["completed_rows"]),
        "requested_batch_size": int(progress["requested_batch_size"]),
        "effective_batch_size": int(progress["effective_batch_size"]),
        "g1": dict(progress["g1"]),
        "excluded_cross_label_conflicts": plan.excluded_cross_label_conflicts,
        "reference_8k_indices_file": "reference_8k_indices.npy",
        "reference_labels_file": "reference_labels.npy",
        "development_subsets_file": "development_subsets.npy",
        "classifier_weight_file": "classifier_weight.npy",
        "classifier_bias_file": "classifier_bias.npy",
        "pooler_layernorm_weight_file": "pooler_layernorm_weight.npy",
        "pooler_layernorm_bias_file": "pooler_layernorm_bias.npy",
        "pooler_layernorm_eps": float(progress["pooler_layernorm_eps"]),
        "sealed": {"loaded": 0, "scored": 0},
        "started_at": progress["started_at"],
        "updated_at": progress["updated_at"],
        "finished_at": _utc_now() if complete else None,
    }


def _extract_layer_feature_cache(
    plan: LayerCachePlan,
    output_dir: Path,
    *,
    model: Any,
    processor: Any,
    device: str | torch.device,
    batch_size: int,
    image_loader: Callable[[Path], Any],
    strict_counts: bool,
    require_cuda: bool,
) -> dict[str, Any]:
    _validate_plan(plan, strict_counts=strict_counts)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for path in (*plan.reference_paths, *plan.development_paths):
        _assert_not_sealed_image_path(path)

    target_device = torch.device(device)
    if require_cuda and target_device.type != "cuda":
        raise LayerCacheBoundaryError("Production layer extraction must pool on CUDA")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise LayerCacheBoundaryError("CUDA extraction requested but CUDA is unavailable")

    _validate_loaded_model(model)
    model.eval()
    model.requires_grad_(False)
    model.to(device=target_device, dtype=torch.float32)

    output_dir = Path(output_dir)
    existed = (output_dir / "cache_spec.json").exists()
    arrays, progress = _open_or_create_cache(
        output_dir,
        plan,
        requested_batch_size=batch_size,
    )
    _write_or_validate_parameters(output_dir, model, resume=existed)
    layernorm_eps = _pooler_layernorm_eps(model)
    previous_eps = progress.get("pooler_layernorm_eps")
    if previous_eps is not None and float(previous_eps) != layernorm_eps:
        raise LayerCacheBoundaryError("Pooler LayerNorm eps changed across resume")
    progress["pooler_layernorm_eps"] = layernorm_eps
    progress["updated_at"] = _utc_now()
    _atomic_write_json(output_dir / "progress.json", progress)
    effective_batch_size = min(
        int(progress.get("effective_batch_size", batch_size)), batch_size
    )

    split_inputs = (
        ("reference", plan.reference_paths),
        ("development", plan.development_paths),
    )
    progress_path = output_dir / "progress.json"
    metadata_path = output_dir / "metadata.json"
    for split, paths in split_inputs:
        start = int(progress["completed_rows"][split])
        while start < len(paths):
            end = min(len(paths), start + effective_batch_size)
            batch_paths = paths[start:end]
            images: list[Any] = []
            try:
                for path in batch_paths:
                    _assert_not_sealed_image_path(path)
                    images.append(image_loader(path))
                encoded = processor(images=images, return_tensors="pt")
                if "pixel_values" not in encoded:
                    raise LayerCacheGateError("Image processor did not return pixel_values")
                pixel_values = encoded["pixel_values"].to(
                    device=target_device,
                    dtype=torch.float32,
                )
                with torch.inference_mode():
                    cls, raw_mean, logits, g1 = _forward_pooled_vectors(
                        model,
                        pixel_values,
                        run_g1=not progress["g1"]["passed"],
                    )
            except torch.cuda.OutOfMemoryError:
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()
                if effective_batch_size == 1:
                    raise
                effective_batch_size = max(1, effective_batch_size // 2)
                progress["effective_batch_size"] = effective_batch_size
                progress["updated_at"] = _utc_now()
                _atomic_write_json(progress_path, progress)
                continue
            finally:
                _close_images(images)

            arrays[f"{split}_cls"][start:end] = cls
            arrays[f"{split}_raw_patch_mean"][start:end] = raw_mean
            arrays[f"{split}_logits"][start:end] = logits
            for name in (
                f"{split}_cls",
                f"{split}_raw_patch_mean",
                f"{split}_logits",
            ):
                arrays[name].flush()
            if g1 is not None:
                progress["g1"] = g1
            progress["completed_rows"][split] = end
            progress["effective_batch_size"] = effective_batch_size
            progress["updated_at"] = _utc_now()
            _atomic_write_json(progress_path, progress)
            _atomic_write_json(
                metadata_path,
                _metadata_snapshot(plan, progress, complete=False),
            )
            start = end

    metadata = _metadata_snapshot(plan, progress, complete=True)
    _atomic_write_json(metadata_path, metadata)
    return metadata


def extract_layer_feature_cache_from_loaded_model(
    plan: LayerCachePlan,
    output_dir: Path,
    *,
    model: Any,
    processor: Any,
    device: str | torch.device = "cuda",
    batch_size: int = 32,
    image_loader: Callable[[Path], Any] = _default_image_loader,
) -> dict[str, Any]:
    """Run the strict frozen extraction with an already loaded model and processor."""

    return _extract_layer_feature_cache(
        plan,
        output_dir,
        model=model,
        processor=processor,
        device=device,
        batch_size=batch_size,
        image_loader=image_loader,
        strict_counts=True,
        require_cuda=True,
    )


def build_layer_feature_cache(
    reference_manifest_path: Path,
    development_manifest_path: Path,
    output_dir: Path,
    *,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
    expected_8k_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Prepare the frozen plan, load the pinned DiT, and build/resume its layer cache."""

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    plan = prepare_layer_cache_plan(
        reference_manifest_path,
        development_manifest_path,
        expected_8k_ids=expected_8k_ids,
    )
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=False,
    )
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    return extract_layer_feature_cache_from_loaded_model(
        plan,
        output_dir,
        model=model,
        processor=processor,
        device=device,
        batch_size=batch_size,
    )


def load_layer_cache_metadata(output_dir: Path, *, require_complete: bool = True) -> dict[str, Any]:
    metadata_path = Path(output_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = _read_json(metadata_path)
    if metadata.get("sealed") != {"loaded": 0, "scored": 0}:
        raise LayerCacheBoundaryError("Layer cache metadata does not prove zero sealed access")
    if require_complete and not metadata.get("complete", False):
        raise LayerCacheBoundaryError("Layer cache is incomplete")
    return metadata


__all__ = [
    "LayerCacheBoundaryError",
    "LayerCacheGateError",
    "LayerCachePlan",
    "build_layer_feature_cache",
    "extract_layer_feature_cache_from_loaded_model",
    "load_development_only_manifest",
    "load_layer_cache_metadata",
    "prepare_layer_cache_plan",
    "select_frozen_reference_manifest",
]
