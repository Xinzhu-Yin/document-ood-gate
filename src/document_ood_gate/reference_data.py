from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_official_split(path: Path, split: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.strip().split()
        if len(fields) != 2:
            raise ValueError(f"Invalid official split row at {path}:{line_number}")
        relative_path, label_text = fields
        label = int(label_text)
        if not 0 <= label < 16:
            raise ValueError(f"Invalid RVL-CDIP label {label} at {path}:{line_number}")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative image path at {path}:{line_number}")
        rows.append({"split": split, "relative_path": relative.as_posix(), "label": label})
    frame = pd.DataFrame(rows)
    if frame["relative_path"].duplicated().any():
        raise ValueError(f"Official {split} split contains duplicate image paths")
    return frame


def _balanced_sample(
    frame: pd.DataFrame,
    *,
    per_class: int,
    seed: int,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for label in range(16):
        class_rows = frame[frame["label"] == label].sort_values("relative_path")
        if len(class_rows) < per_class:
            raise ValueError(
                f"Class {label} has {len(class_rows)} rows, fewer than requested {per_class}"
            )
        class_rng = np.random.default_rng(seed + label)
        positions = class_rng.permutation(len(class_rows))[:per_class]
        sample = class_rows.iloc[positions].copy()
        sample["class_sample_rank"] = np.arange(per_class, dtype=np.int64)
        selected.append(sample)
    return pd.concat(selected, ignore_index=True).sort_values(
        ["label", "class_sample_rank"]
    ).reset_index(drop=True)


def create_reference_sample_manifest(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    reference_sizes: tuple[int, ...] = (8000, 32000, 80000),
    validation_size: int = 8000,
    test_size: int = 10000,
    seed: int = 20260812,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if tuple(sorted(reference_sizes)) != reference_sizes:
        raise ValueError("reference_sizes must be sorted from smallest to largest")
    requested = (*reference_sizes, validation_size, test_size)
    if any(size % 16 != 0 for size in requested):
        raise ValueError("Every requested sample size must be divisible by 16")

    maximum_reference = reference_sizes[-1]
    train_sample = _balanced_sample(train, per_class=maximum_reference // 16, seed=seed)
    train_sample["purpose"] = "reference"
    train_sample["minimum_reference_size"] = maximum_reference
    for size in reversed(reference_sizes[:-1]):
        train_sample.loc[
            train_sample["class_sample_rank"] < size // 16,
            "minimum_reference_size",
        ] = size

    validation_sample = _balanced_sample(
        validation,
        per_class=validation_size // 16,
        seed=seed + 10_000,
    )
    validation_sample["purpose"] = "validation_threshold"
    validation_sample["minimum_reference_size"] = pd.NA

    test_sample = _balanced_sample(test, per_class=test_size // 16, seed=seed + 20_000)
    test_sample["purpose"] = "t_guardrail"
    test_sample["minimum_reference_size"] = pd.NA

    manifest = pd.concat(
        [train_sample, validation_sample, test_sample],
        ignore_index=True,
    )
    if manifest["relative_path"].duplicated().any():
        raise AssertionError("Official train/validation/test splits unexpectedly overlap")
    summary: dict[str, Any] = {
        "seed": int(seed),
        "reference_sizes": [int(value) for value in reference_sizes],
        "validation_size": int(validation_size),
        "test_guardrail_size": int(test_size),
        "total_selected": int(len(manifest)),
        "class_balanced": True,
        "reference_banks_nested": True,
        "counts_by_purpose": {
            str(key): int(value)
            for key, value in manifest.groupby("purpose", observed=True).size().items()
        },
    }
    return manifest, summary


def extract_selected_from_tar(
    archive_path: Path,
    manifest: pd.DataFrame,
    output_root: Path,
) -> pd.DataFrame:
    required = {"relative_path", "purpose"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Sample manifest is missing columns: {', '.join(missing)}")
    selected = {str(value): index for index, value in manifest["relative_path"].items()}
    if len(selected) != len(manifest):
        raise ValueError("Sample manifest contains duplicate relative paths")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    extracted_paths: dict[str, str] = {}

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            normalized = member.name.replace("\\", "/").lstrip("./")
            relative_path = normalized if normalized in selected else None
            if relative_path is None:
                # The official archive wraps split-list paths such as ``imagesq/...``
                # inside a top-level ``images/`` directory.
                archive_unwrapped = (
                    normalized[len("images/") :] if normalized.startswith("images/") else None
                )
                if archive_unwrapped in selected:
                    relative_path = archive_unwrapped
            if relative_path is None:
                images_marker = normalized.find("images")
                if images_marker >= 0 and (
                    images_marker == 0 or normalized[images_marker - 1] == "/"
                ):
                    candidate = normalized[images_marker:]
                    if candidate in selected:
                        relative_path = candidate
            if relative_path is None:
                continue
            purpose = str(manifest.loc[selected[relative_path], "purpose"])
            destination = (output_root / purpose / relative_path).resolve()
            if output_root not in destination.parents:
                raise ValueError(f"Unsafe output path derived from archive: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Could not read archive member: {member.name}")
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1 << 20)
            extracted_paths[relative_path] = str(destination)
            if len(extracted_paths) == len(selected):
                break

    missing_paths = sorted(set(selected) - set(extracted_paths))
    if missing_paths:
        raise FileNotFoundError(
            f"Archive did not contain {len(missing_paths)} selected images; first={missing_paths[0]}"
        )
    output = manifest.copy()
    output["local_path"] = output["relative_path"].map(extracted_paths)
    return output
