from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEVELOPMENT = "development"
SEALED = "sealed"
_MISSING_GROUP_MARKERS = {"", "na", "n/a", "nan", "none", "null"}


def _stable_singleton_id(path: str) -> str:
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
    return f"singleton:{digest}"


def _effective_group_id(value: object, path: str) -> str:
    if pd.isna(value):
        return _stable_singleton_id(path)
    text = str(value).strip()
    if text.lower() in _MISSING_GROUP_MARKERS:
        return _stable_singleton_id(path)
    return f"confirmed:{text}"


def _validate_manifest(
    manifest: pd.DataFrame,
    *,
    group_column: str,
    source_column: str,
    n_class_column: str,
    expected_n_classes: int | None,
) -> pd.DataFrame:
    required = {"path", "subset", group_column, source_column, n_class_column}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    if manifest.empty:
        raise ValueError("Manifest is empty")

    frame = manifest.copy()
    frame["path"] = frame["path"].astype("string").str.strip()
    if frame["path"].isna().any() or frame["path"].eq("").any():
        raise ValueError("Every manifest row must have a non-empty path")
    if frame["path"].duplicated().any():
        duplicates = frame.loc[frame["path"].duplicated(keep=False), "path"].unique()
        preview = ", ".join(str(value) for value in duplicates[:3])
        raise ValueError(f"Each image path must occur exactly once; duplicates include: {preview}")

    frame["subset"] = frame["subset"].astype("string").str.strip().str.upper()
    invalid_subsets = sorted(set(frame["subset"].dropna()) - {"N", "O"})
    if frame["subset"].isna().any() or invalid_subsets:
        raise ValueError(f"subset must contain only N or O; found invalid values: {invalid_subsets}")

    frame[source_column] = frame[source_column].astype("string").str.strip()
    if frame[source_column].isna().any() or frame[source_column].eq("").any():
        raise ValueError(f"Every row must have a non-empty {source_column!r}")

    n_mask = frame["subset"].eq("N")
    if not n_mask.any() or frame["subset"].eq("O").sum() == 0:
        raise ValueError("Manifest must contain both N and O rows")
    frame[n_class_column] = frame[n_class_column].astype("string").str.strip()
    if frame.loc[n_mask, n_class_column].isna().any() or frame.loc[n_mask, n_class_column].eq("").any():
        raise ValueError(f"Every N row must have a non-empty {n_class_column!r}")

    actual_n_classes = int(frame.loc[n_mask, n_class_column].nunique())
    if expected_n_classes is not None and actual_n_classes != expected_n_classes:
        raise ValueError(
            f"Expected {expected_n_classes} N classes in {n_class_column!r}, "
            f"found {actual_n_classes}"
        )

    frame["split_group_id"] = [
        _effective_group_id(group, str(path))
        for group, path in zip(frame[group_column], frame["path"], strict=True)
    ]
    frame["_stratum"] = np.where(
        n_mask,
        "N|class="
        + frame[n_class_column].fillna("")
        + "|source="
        + frame[source_column],
        "O|source=" + frame[source_column],
    )
    return frame


def _score_candidate(
    development_counts: np.ndarray,
    stratum_totals: np.ndarray,
    splittable_strata: np.ndarray,
    subset_masks: tuple[np.ndarray, np.ndarray],
    *,
    development_rows: int,
    total_rows: int,
    development_fraction: float,
) -> tuple[float, float, float, float, float]:
    fractions = development_counts / stratum_totals
    errors = np.abs(fractions - development_fraction)
    missing_side = ((development_counts == 0) | (development_counts == stratum_totals)) & splittable_strata
    coverage_penalty = float(missing_side.sum())
    subset_error = max(
        abs(
            float(development_counts[mask].sum() / stratum_totals[mask].sum())
            - development_fraction
        )
        for mask in subset_masks
    )
    weights = np.sqrt(stratum_totals)
    weighted_mean_error = float(np.average(errors, weights=weights))
    worst_error = float(errors[splittable_strata].max()) if splittable_strata.any() else float(errors.max())
    overall_error = abs(development_rows / total_rows - development_fraction)
    return coverage_penalty, worst_error, subset_error, weighted_mean_error, overall_error


def _choose_development_groups(
    group_stratum_counts: np.ndarray,
    group_sizes: np.ndarray,
    *,
    development_fraction: float,
    seed: int,
    search_trials: int,
    strata: list[str],
) -> np.ndarray:
    n_groups = len(group_sizes)
    if n_groups < 2:
        raise ValueError("At least two duplicate groups are required for a development/sealed split")
    if search_trials < 1:
        raise ValueError("search_trials must be at least 1")

    total_rows = int(group_sizes.sum())
    target_rows = development_fraction * total_rows
    stratum_totals = group_stratum_counts.sum(axis=0)
    splittable_strata = (group_stratum_counts > 0).sum(axis=0) >= 2
    subset_masks = (
        np.array([name.startswith("N|") for name in strata], dtype=bool),
        np.array([name.startswith("O|") for name in strata], dtype=bool),
    )
    rng = np.random.default_rng(seed)

    best_mask: np.ndarray | None = None
    best_score: tuple[float, float, float, float, float] | None = None
    for _ in range(search_trials):
        order = rng.permutation(n_groups)
        cumulative_rows = np.cumsum(group_sizes[order])
        insertion = int(np.searchsorted(cumulative_rows, target_rows, side="left"))
        candidate_cuts = {max(1, min(n_groups - 1, insertion))}
        candidate_cuts.add(max(1, min(n_groups - 1, insertion + 1)))
        for cut in sorted(candidate_cuts):
            selected = order[:cut]
            counts = group_stratum_counts[selected].sum(axis=0)
            rows = int(group_sizes[selected].sum())
            score = _score_candidate(
                counts,
                stratum_totals,
                splittable_strata,
                subset_masks,
                development_rows=rows,
                total_rows=total_rows,
                development_fraction=development_fraction,
            )
            if best_score is None or score < best_score:
                mask = np.zeros(n_groups, dtype=bool)
                mask[selected] = True
                best_mask = mask
                best_score = score

    if best_mask is None:  # pragma: no cover - guarded by search_trials validation
        raise RuntimeError("Unable to construct a split")
    return best_mask


def create_research_split(
    manifest: pd.DataFrame,
    *,
    development_fraction: float = 0.30,
    seed: int = 20260812,
    group_column: str = "confirmed_group_id",
    source_column: str = "source",
    n_class_column: str = "n_class",
    expected_n_classes: int | None = 12,
    search_trials: int = 4096,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a metadata-only grouped development/sealed assignment.

    ``group_column`` must contain only exact or manually confirmed duplicate groups.
    Blank group IDs are treated as singleton images. Model outputs and OOD scores are
    deliberately not accepted or used by this function.
    """

    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be strictly between 0 and 1")

    frame = _validate_manifest(
        manifest,
        group_column=group_column,
        source_column=source_column,
        n_class_column=n_class_column,
        expected_n_classes=expected_n_classes,
    )
    group_ids = sorted(frame["split_group_id"].unique())
    strata = sorted(frame["_stratum"].unique())
    group_index = {value: index for index, value in enumerate(group_ids)}
    stratum_index = {value: index for index, value in enumerate(strata)}
    group_stratum_counts = np.zeros((len(group_ids), len(strata)), dtype=np.int64)
    for group, stratum in zip(frame["split_group_id"], frame["_stratum"], strict=True):
        group_stratum_counts[group_index[group], stratum_index[stratum]] += 1
    group_sizes = group_stratum_counts.sum(axis=1)

    development_mask = _choose_development_groups(
        group_stratum_counts,
        group_sizes,
        development_fraction=development_fraction,
        seed=seed,
        search_trials=search_trials,
        strata=strata,
    )
    group_assignment = {
        group_id: DEVELOPMENT if development_mask[index] else SEALED
        for index, group_id in enumerate(group_ids)
    }
    frame["split"] = frame["split_group_id"].map(group_assignment)

    output_columns = [
        "path",
        "subset",
        source_column,
        n_class_column,
        group_column,
        "split_group_id",
        "split",
    ]
    assignments = frame[output_columns].sort_values(["subset", "path"]).reset_index(drop=True)
    _assert_split_invariants(assignments)
    summary = _metadata_summary(
        assignments,
        development_fraction=development_fraction,
        seed=seed,
        source_column=source_column,
        n_class_column=n_class_column,
    )
    return assignments, summary


def _assert_split_invariants(assignments: pd.DataFrame) -> None:
    if assignments["path"].duplicated().any():
        raise AssertionError("An image was assigned more than once")
    if set(assignments["split"]) != {DEVELOPMENT, SEALED}:
        raise AssertionError("Both development and sealed splits must be non-empty")
    group_split_counts = assignments.groupby("split_group_id", sort=False)["split"].nunique()
    if int(group_split_counts.max()) != 1:
        raise AssertionError("A confirmed duplicate group crosses development and sealed")


def _metadata_summary(
    assignments: pd.DataFrame,
    *,
    development_fraction: float,
    seed: int,
    source_column: str,
    n_class_column: str,
) -> dict[str, Any]:
    split_counts = assignments.groupby(["split", "subset"], observed=True).size()
    strata_frame = assignments.copy()
    strata_frame["stratum"] = np.where(
        strata_frame["subset"].eq("N"),
        "N|class="
        + strata_frame[n_class_column].fillna("")
        + "|source="
        + strata_frame[source_column],
        "O|source=" + strata_frame[source_column],
    )
    stratum_counts = (
        strata_frame.groupby(["stratum", "split"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[DEVELOPMENT, SEALED], fill_value=0)
    )
    strata = []
    for name, row in stratum_counts.iterrows():
        development = int(row[DEVELOPMENT])
        sealed = int(row[SEALED])
        total = development + sealed
        strata.append(
            {
                "stratum": str(name),
                "total": total,
                "development": development,
                "sealed": sealed,
                "development_fraction": development / total,
            }
        )

    return {
        "seed": int(seed),
        "requested_development_fraction": float(development_fraction),
        "rows": int(len(assignments)),
        "groups": int(assignments["split_group_id"].nunique()),
        "split_counts": {
            split: {
                subset: int(split_counts.get((split, subset), 0)) for subset in ("N", "O")
            }
            for split in (DEVELOPMENT, SEALED)
        },
        "strata": strata,
        "invariants": {
            "each_path_exactly_once": True,
            "confirmed_group_crosses_split": False,
            "model_outputs_or_scores_used": False,
        },
    }


def read_manifest(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Manifest must be a .csv or .parquet file")


def write_assignments(assignments: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        assignments.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        assignments.to_csv(path, index=False)
        return
    raise ValueError("Split output must be a .csv or .parquet file")
