from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402


FULL_RESULT_COLUMNS = (
    "experiment",
    "layer",
    "representation",
    "method",
    "residual_centering_convention",
    "bank_size",
    "auroc",
    "auroc_ci_low",
    "auroc_ci_high",
    "fpr95",
    "fpr95_ci_low",
    "fpr95_ci_high",
    "top_512_explained_variance_ratio",
)

VIM_RESULT_COLUMNS = (
    "experiment",
    "method",
    "representation",
    "residual_centering_convention",
    "bank_size",
    "auroc",
    "auroc_ci_low",
    "auroc_ci_high",
    "fpr95",
    "fpr95_ci_low",
    "fpr95_ci_high",
)

_METHOD_ORDER = ("mahalanobis", "knn", "residual")
_METHOD_LABELS = {
    "mahalanobis": "Mahalanobis",
    "knn": "kNN",
    "residual": "Mean-centered PCA Residual",
}
_METHOD_COLORS = {
    "mahalanobis": "#2166ac",
    "knn": "#1b9e77",
    "residual": "#d95f02",
}
_REPRESENTATION_ORDER = ("raw_patch_mean", "h_doc", "cls")
_REPRESENTATION_LABELS = {
    "raw_patch_mean": "L12 raw patch mean",
    "h_doc": "h_doc",
    "cls": "L12 CLS",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _records_frame(
    records: pd.DataFrame | Iterable[Mapping[str, Any]],
    required_columns: tuple[str, ...],
    *,
    name: str,
) -> pd.DataFrame:
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    aliases = {
        "residual_centering": "residual_centering_convention",
        "top512_variance_ratio": "top_512_explained_variance_ratio",
        "top_512_variance_ratio": "top_512_explained_variance_ratio",
    }
    frame = frame.rename(
        columns={source: target for source, target in aliases.items() if target not in frame}
    )
    for metric in ("auroc", "fpr95"):
        interval_column = f"{metric}_95ci"
        low_column = f"{metric}_ci_low"
        high_column = f"{metric}_ci_high"
        if interval_column in frame and low_column not in frame and high_column not in frame:
            intervals = frame[interval_column].map(list)
            if not intervals.map(lambda values: len(values) == 2).all():
                raise ValueError(f"{name} {interval_column} values must contain two endpoints")
            frame[low_column] = intervals.map(lambda values: values[0])
            frame[high_column] = intervals.map(lambda values: values[1])
        if interval_column in frame and low_column in frame and high_column in frame:
            frame = frame.drop(columns=interval_column)

    missing = [column for column in required_columns if column not in frame]
    if missing:
        raise ValueError(f"{name} records are missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{name} records must not be empty")

    numeric_columns = [
        column
        for column in required_columns
        if column
        not in {"experiment", "representation", "method", "residual_centering_convention"}
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    metric_columns = [
        "auroc",
        "auroc_ci_low",
        "auroc_ci_high",
        "fpr95",
        "fpr95_ci_low",
        "fpr95_ci_high",
    ]
    if not np.isfinite(frame[metric_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{name} metric values must be finite")
    if not ((frame[metric_columns] >= 0.0) & (frame[metric_columns] <= 1.0)).all().all():
        raise ValueError(f"{name} metric values must be in [0, 1]")
    if not (
        (frame["auroc_ci_low"] <= frame["auroc"])
        & (frame["auroc"] <= frame["auroc_ci_high"])
        & (frame["fpr95_ci_low"] <= frame["fpr95"])
        & (frame["fpr95"] <= frame["fpr95_ci_high"])
    ).all():
        raise ValueError(f"{name} point estimates must lie inside their confidence intervals")
    if "top_512_explained_variance_ratio" in required_columns:
        variance_ratio = frame["top_512_explained_variance_ratio"].to_numpy(dtype=np.float64)
        if not np.isfinite(variance_ratio).all() or not (
            (variance_ratio >= 0.0) & (variance_ratio <= 1.0)
        ).all():
            raise ValueError(f"{name} top-512 variance ratios must be finite and in [0, 1]")
    return frame


def _method_key(value: object) -> str | None:
    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if "mahalanobis" in normalized:
        return "mahalanobis"
    if normalized in {"knn", "k nn", "k nearest neighbor", "k nearest neighbours"}:
        return "knn"
    if "residual" in normalized and "vim" not in normalized:
        return "residual"
    return None


def _representation_key(value: object) -> str | None:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"raw_patch_mean", "raw_mean", "l12_raw_patch_mean"}:
        return "raw_patch_mean"
    if normalized in {"h_doc", "hdoc"}:
        return "h_doc"
    if normalized in {"cls", "l12_cls", "cls_token"}:
        return "cls"
    return None


def _plot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    plotted = frame.copy()
    plotted["_experiment"] = plotted["experiment"].astype(str).str.strip().str.upper()
    plotted["_method"] = plotted["method"].map(_method_key)
    plotted["_representation"] = plotted["representation"].map(_representation_key)
    plotted["layer"] = pd.to_numeric(plotted["layer"], errors="coerce")
    plotted["bank_size"] = pd.to_numeric(plotted["bank_size"], errors="coerce")
    return plotted


def _require_plot_grid(frame: pd.DataFrame) -> None:
    residual_rows = frame[frame["_method"] == "residual"]
    residual_centering = (
        residual_rows["residual_centering_convention"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    if not (residual_centering == "reference_mean").all():
        raise ValueError("S1/S2 Residual rows must use reference_mean centering")

    layer_scan = frame[
        (frame["_experiment"] == "S2")
        & (frame["_representation"] == "raw_patch_mean")
        & frame["_method"].isin(_METHOD_ORDER)
        & frame["layer"].isin(range(1, 13))
        & frame["bank_size"].isin((8000, 32000))
    ]
    observed = {
        (int(layer), str(method), int(bank))
        for layer, method, bank in layer_scan[
            ["layer", "_method", "bank_size"]
        ].itertuples(index=False, name=None)
    }
    expected = {
        (layer, method, bank)
        for layer in range(1, 13)
        for method in _METHOD_ORDER
        for bank in (8000, 32000)
    }
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"Layer-curve records are incomplete; missing {missing[:5]}")

    final_layer = frame[
        (frame["_experiment"] == "S1")
        & (frame["layer"] == 12)
        & frame["_representation"].isin(_REPRESENTATION_ORDER)
        & frame["_method"].isin(_METHOD_ORDER)
        & frame["bank_size"].isin((8000, 32000))
    ]
    final_observed = {
        (str(representation), str(method), int(bank))
        for representation, method, bank in final_layer[
            ["_representation", "_method", "bank_size"]
        ].itertuples(index=False, name=None)
    }
    final_expected = {
        (representation, method, bank)
        for representation in _REPRESENTATION_ORDER
        for method in _METHOD_ORDER
        for bank in (8000, 32000)
    }
    final_missing = sorted(final_expected - final_observed)
    if final_missing:
        raise ValueError(f"Final-layer records are incomplete; missing {final_missing[:5]}")


def _require_vim_decomposition(frame: pd.DataFrame) -> None:
    experiments = frame["experiment"].astype(str).str.strip().str.upper()
    if not (experiments == "S1B").all():
        raise ValueError("ViM decomposition rows must be labelled experiment S1b")
    methods = frame["method"].astype(str).str.strip().str.lower()
    has_energy = (methods == "energy").any()
    has_residual = methods.str.contains("vim") & methods.str.contains("residual")
    has_full_vim = methods.isin({"vim", "full vim"}).any()
    if not has_energy or not has_residual.any() or not has_full_vim:
        raise ValueError("ViM decomposition must contain Energy, ViM-origin Residual, and ViM")
    residual_rows = frame[has_residual]
    conventions = residual_rows["residual_centering_convention"].astype(str).str.lower()
    if not conventions.str.contains("classifier_origin").all():
        raise ValueError("ViM-origin Residual must use classifier_origin centering")


def _single_row(frame: pd.DataFrame, description: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"Expected one {description} row, found {len(frame)}")
    return frame.iloc[0]


def _save_final_layer_plot(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    x = np.arange(len(_REPRESENTATION_ORDER), dtype=np.float64)
    for axis, metric, title in zip(
        axes,
        ("auroc", "fpr95"),
        ("AUROC (higher is better)", "FPR95 (lower is better)"),
        strict=True,
    ):
        for method_index, method in enumerate(_METHOD_ORDER):
            for bank in (32000, 8000):
                rows = []
                for representation in _REPRESENTATION_ORDER:
                    rows.append(
                        _single_row(
                            frame[
                                (frame["_experiment"] == "S1")
                                & (frame["layer"] == 12)
                                & (frame["_representation"] == representation)
                                & (frame["_method"] == method)
                                & (frame["bank_size"] == bank)
                            ],
                            f"L12 {representation}/{method}/{bank}",
                        )
                    )
                y = np.asarray([row[metric] for row in rows], dtype=np.float64)
                offset = (method_index - 1) * 0.035
                axis.plot(
                    x + offset,
                    y,
                    color=_METHOD_COLORS[method],
                    linestyle="-" if bank == 32000 else "--",
                    marker="o" if bank == 32000 else "s",
                    linewidth=2.0 if bank == 32000 else 1.4,
                    markersize=5,
                    label=(
                        f"{_METHOD_LABELS[method]} · "
                        f"{'32k primary' if bank == 32000 else '8k sensitivity'}"
                    ),
                )
        axis.set_title(title)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.set_xticks(x, [_REPRESENTATION_LABELS[key] for key in _REPRESENTATION_ORDER])
    axes[0].set_ylabel("Development-set metric")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Final-layer representation comparison")
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_layer_curve(frame: pd.DataFrame, path: Path, metric: str) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 6.2))
    for method in _METHOD_ORDER:
        for bank in (32000, 8000):
            rows = frame[
                (frame["_experiment"] == "S2")
                & (frame["_representation"] == "raw_patch_mean")
                & (frame["_method"] == method)
                & (frame["bank_size"] == bank)
                & frame["layer"].isin(range(1, 13))
            ].sort_values("layer")
            axis.plot(
                rows["layer"],
                rows[metric],
                color=_METHOD_COLORS[method],
                linestyle="-" if bank == 32000 else "--",
                marker="o" if bank == 32000 else "s",
                linewidth=2.2 if bank == 32000 else 1.5,
                markersize=4.5,
                label=(
                    f"{_METHOD_LABELS[method]} · "
                    f"{'32k primary' if bank == 32000 else '8k sensitivity'}"
                ),
            )
    axis.set_xlim(1, 12)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(range(1, 13))
    axis.set_xlabel("Transformer layer (raw patch mean)")
    if metric == "auroc":
        axis.set_ylabel("AUROC")
        axis.set_title("Development AUROC across Layers 1–12 (higher is better)")
    else:
        axis.set_ylabel("FPR95 — Lower is better")
        axis.set_title("Development FPR95 across Layers 1–12 (Lower is better)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _format_metric(row: pd.Series) -> str:
    return (
        f"{row['auroc']:.4f} [{row['auroc_ci_low']:.4f}, {row['auroc_ci_high']:.4f}] / "
        f"{row['fpr95']:.4f} [{row['fpr95_ci_low']:.4f}, {row['fpr95_ci_high']:.4f}]"
    )


def _gate_text(gates: Mapping[str, Any], name: str) -> str:
    value = gates.get(name, gates.get(name.lower()))
    if isinstance(value, Mapping):
        passed = value.get("passed")
        detail = value.get("detail") or value.get("message")
    else:
        passed = value
        detail = None
    status = "PASS" if passed is True else "FAIL" if passed is False else "NOT REPORTED"
    return f"{status}{f' — {detail}' if detail else ''}"


def _development_summary(
    full_frame: pd.DataFrame,
    plotted: pd.DataFrame,
    vim_frame: pd.DataFrame,
    gates: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    extraction_count = int(metadata["extraction_image_count"])
    reference_count = int(metadata["reference_image_count"])
    development_n = int(metadata["development_n_count"])
    development_o = int(metadata["development_o_count"])
    sealed_loaded = int(metadata["sealed_rows_loaded"])
    sealed_scored = int(metadata["sealed_rows_scored"])

    lines = [
        "# Development-only layer scan",
        "",
        "## Protocol and hard gates",
        "",
        f"- G1 structure gate: **{_gate_text(gates, 'G1')}**",
        f"- G2 existing-pipeline reproduction gate: **{_gate_text(gates, 'G2')}**",
        (
            f"- Extraction: **{extraction_count:,} images** "
            f"({reference_count:,} reference + {development_n:,} N + {development_o:,} O)."
        ),
        f"- Sealed rows loaded: **{sealed_loaded}**; sealed rows scored: **{sealed_scored}**.",
        (
            "- This run extracted only 33,280 images, so the existing T guardrail was not "
            "re-extracted or rescored; its prior setting remains unchanged."
        ),
        "",
        "## Residual definitions",
        "",
        (
            "- **Mean-centered PCA Residual (S1/S2):** PCA is fitted after subtracting the "
            "reference-feature mean; D=512 and the score is the negative residual norm."
        ),
        (
            "- **ViM-origin Residual (S1b):** uses the classifier-defined origin "
            "`u = -pinv(W) @ b`, the existing ViM null space and alpha, and the score "
            "`-alpha * residual_norm`. It is not the mean-centered PCA Residual."
        ),
        "",
        "## Final-layer representation comparison",
        "",
        "AUROC [95% CI] / FPR95 [95% CI]:",
        "",
        "| Representation | Method | Bank | AUROC / FPR95 |",
        "|---|---|---:|---|",
    ]
    final_rows = plotted[
        (plotted["_experiment"] == "S1")
        & (plotted["layer"] == 12)
        & plotted["_representation"].isin(_REPRESENTATION_ORDER)
        & plotted["_method"].isin(_METHOD_ORDER)
        & plotted["bank_size"].isin((8000, 32000))
    ]
    for representation in _REPRESENTATION_ORDER:
        for method in _METHOD_ORDER:
            for bank in (32000, 8000):
                row = _single_row(
                    final_rows[
                        (final_rows["_representation"] == representation)
                        & (final_rows["_method"] == method)
                        & (final_rows["bank_size"] == bank)
                    ],
                    f"summary L12 {representation}/{method}/{bank}",
                )
                lines.append(
                    f"| {_REPRESENTATION_LABELS[representation]} | {_METHOD_LABELS[method]} "
                    f"| {bank // 1000}k | {_format_metric(row)} |"
                )

    lines.extend(
        [
            "",
            "## ViM decomposition",
            "",
            "| Component | Centering convention | AUROC / FPR95 |",
            "|---|---|---|",
        ]
    )
    for row in vim_frame.itertuples(index=False):
        series = pd.Series(row._asdict())
        lines.append(
            f"| {series['method']} | {series['residual_centering_convention']} "
            f"| {_format_metric(series)} |"
        )

    lines.extend(
        [
            "",
            "## Complete Layer 1–12 curves",
            "",
            "The complete curves are reported below; this report does not select an argmax layer.",
            "Any layer-specific pattern is a development observation; not confirmed on held-out data.",
            "",
            "Each cell is AUROC / FPR95. FPR95: lower is better.",
            "",
            (
                "| Layer | Mahalanobis 32k | Mahalanobis 8k | kNN 32k | kNN 8k "
                "| Mean-centered Residual 32k | Mean-centered Residual 8k |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for layer in range(1, 13):
        cells = []
        for method in _METHOD_ORDER:
            for bank in (32000, 8000):
                row = _single_row(
                    plotted[
                        (plotted["_experiment"] == "S2")
                        & (plotted["layer"] == layer)
                        & (plotted["_representation"] == "raw_patch_mean")
                        & (plotted["_method"] == method)
                        & (plotted["bank_size"] == bank)
                    ],
                    f"summary Layer {layer}/{method}/{bank}",
                )
                cells.append(f"{row['auroc']:.4f} / {row['fpr95']:.4f}")
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "The CSV/JSON files contain all point estimates, confidence intervals, bank sizes, "
            "representations, methods, centering conventions, and variance-ratio diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def write_layer_scan_reports(
    result_records: pd.DataFrame | Iterable[Mapping[str, Any]],
    vim_decomposition_records: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    metadata: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write development-only layer-scan tables, plots, and a concise summary.

    The function consumes in-memory result records only. It does not load images,
    feature caches, manifests, split assignments, or any sealed-test artifact.
    """

    required_metadata = {
        "extraction_image_count": 33280,
        "reference_image_count": 32000,
        "development_n_count": 306,
        "development_o_count": 974,
        "sealed_rows_loaded": 0,
        "sealed_rows_scored": 0,
        "t_guardrail_rescored": False,
    }
    missing_metadata = sorted(set(required_metadata) - set(metadata))
    if missing_metadata:
        raise ValueError(f"metadata is missing fields: {', '.join(missing_metadata)}")
    for key, expected in required_metadata.items():
        observed = metadata[key]
        matches = observed is expected if isinstance(expected, bool) else int(observed) == expected
        if not matches:
            raise ValueError(f"metadata {key} must be {expected}, found {metadata[key]}")

    full_frame = _records_frame(
        result_records,
        FULL_RESULT_COLUMNS,
        name="full layer-scan",
    )
    vim_frame = _records_frame(
        vim_decomposition_records,
        VIM_RESULT_COLUMNS,
        name="ViM decomposition",
    )
    _require_vim_decomposition(vim_frame)
    plotted = _plot_frame(full_frame)
    _require_plot_grid(plotted)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "full_csv": output / "full_results.csv",
        "full_json": output / "full_results.json",
        "vim_csv": output / "vim_decomposition.csv",
        "vim_json": output / "vim_decomposition.json",
        "summary": output / "development_summary.md",
        "final_layer_plot": output / "final_layer_comparison.png",
        "auroc_plot": output / "layer_auroc_curves.png",
        "fpr95_plot": output / "layer_fpr95_curves.png",
    }

    ordered_full = full_frame.loc[:, [*FULL_RESULT_COLUMNS, *sorted(set(full_frame) - set(FULL_RESULT_COLUMNS))]]
    ordered_vim = vim_frame.loc[:, [*VIM_RESULT_COLUMNS, *sorted(set(vim_frame) - set(VIM_RESULT_COLUMNS))]]
    ordered_full.to_csv(paths["full_csv"], index=False)
    ordered_vim.to_csv(paths["vim_csv"], index=False)
    paths["full_json"].write_text(
        json.dumps(
            _json_safe(
                {
                    "metadata": metadata,
                    "gates": gates,
                    "records": ordered_full.to_dict(orient="records"),
                }
            ),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["vim_json"].write_text(
        json.dumps(
            _json_safe(
                {
                    "metadata": metadata,
                    "gates": gates,
                    "records": ordered_vim.to_dict(orient="records"),
                }
            ),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    _save_final_layer_plot(plotted, paths["final_layer_plot"])
    _save_layer_curve(plotted, paths["auroc_plot"], "auroc")
    _save_layer_curve(plotted, paths["fpr95_plot"], "fpr95")
    paths["summary"].write_text(
        _development_summary(full_frame, plotted, vim_frame, gates, metadata),
        encoding="utf-8",
    )
    return paths
