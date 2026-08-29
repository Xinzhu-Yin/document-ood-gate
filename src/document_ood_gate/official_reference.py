from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metadata import normalize_official_filename
from .reproduction import reproduce_from_logits
from .scores import output_normality_scores


def load_official_output(path: Path) -> dict[str, dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Official output must be a JSON list")
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        filename = normalize_official_filename(str(row["filename"]))
        if filename in output:
            raise ValueError(f"Duplicate official filename: {filename}")
        logits = np.asarray(row["logits"], dtype=np.float64)
        if logits.shape != (16,) or not np.isfinite(logits).all():
            raise ValueError(f"Invalid official logits for {filename}")
        output[filename] = row
    return output


def compare_logits_to_official(
    image_paths: np.ndarray,
    logits: np.ndarray,
    official_rows: dict[str, dict[str, object]],
) -> dict[str, object]:
    paths = np.asarray(image_paths).astype(str)
    actual = np.asarray(logits, dtype=np.float64)
    if actual.shape != (len(paths), 16):
        raise ValueError("logits must have shape [number of paths, 16]")

    official_logits: list[np.ndarray] = []
    actual_logits: list[np.ndarray] = []
    missing_from_official: list[str] = []
    for path, row_logits in zip(paths, actual, strict=True):
        filename = normalize_official_filename(path)
        reference = official_rows.get(filename)
        if reference is None:
            missing_from_official.append(filename)
            continue
        official_logits.append(np.asarray(reference["logits"], dtype=np.float64))
        actual_logits.append(row_logits)
    if not official_logits:
        raise ValueError("No cache paths matched the official output")

    expected = np.stack(official_logits)
    observed = np.stack(actual_logits)
    difference = observed - expected
    expected_scores = output_normality_scores(expected)
    observed_scores = output_normality_scores(observed)
    return {
        "cache_rows": int(len(paths)),
        "official_rows": int(len(official_rows)),
        "matched_rows": int(len(expected)),
        "missing_from_official": sorted(missing_from_official),
        "logit_max_abs_error": float(np.abs(difference).max()),
        "logit_mean_abs_error": float(np.abs(difference).mean()),
        "logit_rmse": float(np.sqrt(np.mean(difference**2))),
        "prediction_agreement": float(np.mean(observed.argmax(axis=1) == expected.argmax(axis=1))),
        "msp_mean_abs_error": float(
            np.abs(observed_scores["msp"] - expected_scores["msp"]).mean()
        ),
        "energy_mean_abs_error": float(
            np.abs(observed_scores["energy"] - expected_scores["energy"]).mean()
        ),
    }


def hybrid_reproduction_report(
    cache: dict[str, np.ndarray],
    official_n: dict[str, dict[str, object]],
    official_o: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Use our rerun where an image exists and author logits only for missing images."""

    cache_paths = np.asarray(cache["paths"]).astype(str)
    cache_logits = np.asarray(cache["logits"], dtype=np.float64)
    if cache_logits.shape != (len(cache_paths), 16):
        raise ValueError("Cache logits must have shape [rows, 16]")
    observed: dict[str, np.ndarray] = {}
    for path, logits in zip(cache_paths, cache_logits, strict=True):
        filename = normalize_official_filename(path)
        if filename in observed:
            raise ValueError(f"Duplicate normalized cache filename: {filename}")
        observed[filename] = logits

    logits_rows: list[np.ndarray] = []
    subsets: list[str] = []
    true_labels: list[str] = []
    author_fallback: list[str] = []
    for subset, official in (("N", official_n), ("O", official_o)):
        for filename, row in official.items():
            if filename in observed:
                logits_rows.append(observed.pop(filename))
            else:
                logits_rows.append(np.asarray(row["logits"], dtype=np.float64))
                author_fallback.append(filename)
            subsets.append(subset)
            true_labels.append(str(row["true_label"]))
    if observed:
        raise ValueError(f"Cache contains {len(observed)} rows absent from official outputs")

    report = reproduce_from_logits(
        np.stack(logits_rows),
        np.asarray(subsets),
        np.asarray(true_labels),
    )
    report["provenance"] = {
        "actual_image_reruns": int(len(cache_paths)),
        "author_logit_fallback": int(len(author_fallback)),
        "fallback_filenames": sorted(author_fallback),
        "independent_full_reproduction": False,
        "note": (
            "The current author release is missing these source images. Their author-provided "
            "logits are used only for full-set metric sensitivity; internal-feature methods "
            "never use these rows."
        ),
    }
    return report
