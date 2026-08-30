from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPRESENTATIONS = ("raw_patch_mean", "ln_free", "ln_pooler")
METHODS = ("mahalanobis", "knn", "mean_centered_pca_residual")
REP_LABELS = {
    "raw_patch_mean": "Raw",
    "ln_free": "LN-free",
    "ln_pooler": "LN-pooler",
}
METHOD_LABELS = {
    "mahalanobis": "Mahalanobis",
    "knn": "kNN (k=10)",
    "mean_centered_pca_residual": "Mean-centered Residual",
}
COLORS = {"raw_patch_mean": "#4c78a8", "ln_free": "#f58518", "ln_pooler": "#54a24b"}
LIMITATION = (
    "A full shortcut audit was not performed. The development/sealed split preserves source "
    "composition, reducing source-distribution mismatch across splits, but this does not rule "
    "out within-task acquisition, rendering, or source-related shortcuts. In particular, 395 "
    "recovered O pages are calibrated rerenders rather than byte-identical original TIFFs."
)


def _frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    required = {
        "experiment",
        "split",
        "layer",
        "representation",
        "method",
        "residual_centering_convention",
        "bank_size",
        "D",
        "auroc",
        "auroc_ci_low",
        "auroc_ci_high",
        "fpr95",
        "fpr95_ci_low",
        "fpr95_ci_high",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Result rows are missing columns: " + ", ".join(missing))
    metric_columns = [
        "auroc",
        "auroc_ci_low",
        "auroc_ci_high",
        "fpr95",
        "fpr95_ci_low",
        "fpr95_ci_high",
    ]
    values = frame[metric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("All reported metrics and intervals must be finite and in [0, 1]")
    return frame


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _write_table(frame: pd.DataFrame, csv_path: Path, json_path: Path, metadata: dict[str, Any]) -> None:
    frame.to_csv(csv_path, index=False)
    payload = {
        "metadata": metadata,
        "records": [_json_value(row) for row in frame.to_dict(orient="records")],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalization_curve(frame: pd.DataFrame, path: Path, metric: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)
    for axis, method in zip(axes, METHODS, strict=True):
        for representation in REPRESENTATIONS:
            rows = frame[
                (frame["method"] == method) & (frame["representation"] == representation)
            ].sort_values("layer")
            axis.plot(
                rows["layer"],
                rows[metric],
                marker="o",
                markersize=4,
                linewidth=2,
                color=COLORS[representation],
                label=REP_LABELS[representation],
            )
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlim(1, 12)
        axis.set_xticks(range(1, 13))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("AUROC (higher is better)" if metric == "auroc" else "FPR95 (lower is better)")
    for axis in axes:
        axis.set_xlabel("Transformer layer")
    axes[1].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    fig.suptitle("Development normalization ablation")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _l12_decomposition(frame: pd.DataFrame, path: Path) -> None:
    l12 = frame[frame["layer"] == 12]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(3)
    width = 0.24
    for method_index, method in enumerate(METHODS):
        rows = [
            l12[(l12["method"] == method) & (l12["representation"] == representation)].iloc[0]
            for representation in REPRESENTATIONS
        ]
        for axis, metric in zip(axes, ("auroc", "fpr95"), strict=True):
            axis.bar(
                x + (method_index - 1) * width,
                [row[metric] for row in rows],
                width=width,
                label=METHOD_LABELS[method],
            )
    for axis, title in zip(axes, ("AUROC (higher is better)", "FPR95 (lower is better)"), strict=True):
        axis.set_xticks(x, ["Raw", "LN-free", "LN-pooler\n= h_doc"])
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Layer 12: raw → pure normalization → learned γ/β")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _d_sensitivity(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    for layer in (3, 7, 9, 11):
        rows = frame[frame["layer"] == layer].sort_values("D")
        axes[0].plot(rows["D"], rows["auroc"], marker="o", label=f"Layer {layer}")
        axes[1].plot(rows["D"], rows["fpr95"], marker="o", label=f"Layer {layer}")
    axes[0].set_ylabel("AUROC (higher is better)")
    axes[1].set_ylabel("FPR95 (lower is better)")
    for axis in axes:
        axis.set_xlabel("PCA principal dimension D")
        axis.set_xticks((128, 256, 384, 512))
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Development-only Residual D sensitivity (diagnostic, not tuning)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_development_reports(
    normalization_rows: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]],
    *,
    sanity: dict[str, Any],
    metadata: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    normalization = _frame(normalization_rows)
    sensitivity = _frame(sensitivity_rows)
    if len(normalization) != 108 or len(sensitivity) != 16:
        raise ValueError("Expected 108 normalization rows and 16 D-sensitivity rows")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "normalization_csv": output / "development_normalization_results.csv",
        "normalization_json": output / "development_normalization_results.json",
        "sensitivity_csv": output / "development_residual_d_sensitivity.csv",
        "sensitivity_json": output / "development_residual_d_sensitivity.json",
        "auroc_plot": output / "development_normalization_auroc.png",
        "fpr95_plot": output / "development_normalization_fpr95.png",
        "l12_plot": output / "l12_normalization_decomposition.png",
        "sensitivity_plot": output / "development_residual_d_sensitivity.png",
        "summary": output / "development_summary.md",
        "metadata": output / "run_metadata.json",
    }
    _write_table(normalization, paths["normalization_csv"], paths["normalization_json"], metadata)
    _write_table(sensitivity, paths["sensitivity_csv"], paths["sensitivity_json"], metadata)
    _normalization_curve(normalization, paths["auroc_plot"], "auroc")
    _normalization_curve(normalization, paths["fpr95_plot"], "fpr95")
    _l12_decomposition(normalization, paths["l12_plot"])
    _d_sensitivity(sensitivity, paths["sensitivity_plot"])

    l12 = normalization[normalization["layer"] == 12]
    lines = [
        "# Final development ablations",
        "",
        "## Protocol boundary",
        "",
        "- 32k reference and 306 N + 974 O development caches were reused; DiT was not rerun.",
        "- All AUROC/FPR95 intervals use 2,000 bootstrap resamples and seed `20260812`.",
        "- Residual D sensitivity is diagnostic only; the frozen primary/sealed value remains D=512.",
        "- Sealed rows loaded/scored during A+B: **0 / 0**.",
        "",
        "## Layer 12 decomposition",
        "",
        "| Representation | Method | AUROC | FPR95 |",
        "|---|---|---:|---:|",
    ]
    for representation in REPRESENTATIONS:
        for method in METHODS:
            row = l12[
                (l12["representation"] == representation) & (l12["method"] == method)
            ].iloc[0]
            lines.append(
                f"| {REP_LABELS[representation]} | {METHOD_LABELS[method]} | "
                f"{row['auroc']:.4f} | {row['fpr95']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Mandatory sanity checks",
            "",
            f"- L12 LN-pooler vs existing h_doc max absolute feature error: "
            f"`{sanity['l12_ln_pooler_vs_h_doc_max_abs_error']:.9g}`.",
            f"- L12 detector-result maximum absolute error vs existing h_doc rows: "
            f"`{sanity['l12_detector_result_max_abs_error']:.9g}`.",
            "",
            "Layers 1–11 LN-pooler results are analytical probes, not the model's real forward path.",
            "",
        ]
    )
    paths["summary"].write_text("\n".join(lines), encoding="utf-8")
    paths["metadata"].write_text(
        json.dumps(_json_value({**metadata, "sanity": sanity}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def _development_vs_sealed(
    development: pd.DataFrame,
    sealed: pd.DataFrame,
    path: Path,
    metric: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)
    for axis, method in zip(axes, METHODS, strict=True):
        for representation in REPRESENTATIONS:
            for frame, linestyle, split_label in (
                (development, "-", "Development"),
                (sealed, "--", "Sealed"),
            ):
                rows = frame[
                    (frame["method"] == method)
                    & (frame["representation"] == representation)
                ].sort_values("layer")
                axis.plot(
                    rows["layer"],
                    rows[metric],
                    color=COLORS[representation],
                    linestyle=linestyle,
                    marker="o" if split_label == "Development" else "s",
                    markersize=3.5,
                    label=f"{REP_LABELS[representation]} · {split_label}",
                )
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlim(1, 12)
        axis.set_xticks(range(1, 13))
        axis.grid(alpha=0.25)
        axis.set_xlabel("Transformer layer")
    axes[0].set_ylabel("AUROC (higher is better)" if metric == "auroc" else "FPR95 (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False)
    fig.suptitle("Development vs sealed normalization/depth curves")
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_sealed_reports(
    baseline_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    baseline = _frame(baseline_rows)
    grid = _frame(grid_rows)
    development = _frame(development_rows)
    sensitivity = _frame(sensitivity_rows)
    if len(baseline) != 6 or len(grid) != 108 or len(development) != 108:
        raise ValueError("Expected 6 D1, 108 sealed D2, and 108 development A rows")
    residual = pd.concat([baseline, grid], ignore_index=True)
    bad_residual = residual[
        residual["method"].astype(str).str.contains("residual", case=False, na=False)
        & ~residual["residual_centering_convention"].isin(
            ["reference_mean", "vim_classifier_origin"]
        )
    ]
    if not bad_residual.empty:
        raise ValueError("Residual centering convention is missing or invalid")
    if not grid.loc[
        grid["method"] == "mean_centered_pca_residual", "residual_centering_convention"
    ].eq("reference_mean").all():
        raise ValueError("D2 Residual must use reference_mean centering")
    if not baseline.loc[
        baseline["method"].isin(["vim_origin_residual", "vim"]),
        "residual_centering_convention",
    ].eq("vim_classifier_origin").all():
        raise ValueError("D1 ViM rows must use vim_classifier_origin")
    if grid["D"].dropna().astype(int).ne(512).any():
        raise ValueError("Sealed D2 contains a forbidden non-512 D value")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "baseline_csv": output / "sealed_baselines.csv",
        "baseline_json": output / "sealed_baselines.json",
        "grid_csv": output / "sealed_layer_grid.csv",
        "grid_json": output / "sealed_layer_grid.json",
        "all_csv": output / "complete_results.csv",
        "all_json": output / "complete_results.json",
        "prediction_json": output / "replication_predictions_evaluation.json",
        "prediction_md": output / "replication_predictions_evaluation.md",
        "auroc_plot": output / "development_vs_sealed_auroc.png",
        "fpr95_plot": output / "development_vs_sealed_fpr95.png",
        "summary": output / "final_summary.md",
        "metadata": output / "run_metadata.json",
    }
    _write_table(baseline, paths["baseline_csv"], paths["baseline_json"], metadata)
    _write_table(grid, paths["grid_csv"], paths["grid_json"], metadata)
    complete = pd.concat([development, sensitivity, baseline, grid], ignore_index=True, sort=False)
    _write_table(complete, paths["all_csv"], paths["all_json"], metadata)
    _development_vs_sealed(development, grid, paths["auroc_plot"], "auroc")
    _development_vs_sealed(development, grid, paths["fpr95_plot"], "fpr95")

    predictions_payload = {"metadata": metadata, "predictions": prediction_rows}
    paths["prediction_json"].write_text(
        json.dumps(_json_value(predictions_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prediction_lines = [
        "# Pre-specified replication predictions: sealed evaluation",
        "",
        "The original predictions remain unchanged; each is recorded only as supported or not supported.",
        "",
        "| Prediction | Status |",
        "|---|---|",
    ]
    for row in prediction_rows:
        prediction_lines.append(f"| {row['prediction']} | **{row['status']}** |")
    paths["prediction_md"].write_text("\n".join(prediction_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Final sealed-test results",
        "",
        f"- Freeze commit: `{metadata['freeze_commit_sha']}`.",
        "- Sealed extraction: exactly 3,018 images (694 N + 2,324 O), one forward each.",
        "- Existing 32k reference cache reused; reference images re-extracted: 0.",
        "- No post-sealed experimental iteration was performed.",
        "",
        "## Core sealed baseline panel",
        "",
        "| Method | AUROC [95% CI] | FPR95 [95% CI] |",
        "|---|---|---|",
    ]
    for row in baseline.itertuples(index=False):
        summary_lines.append(
            f"| {row.method} | {row.auroc:.4f} [{row.auroc_ci_low:.4f}, "
            f"{row.auroc_ci_high:.4f}] | {row.fpr95:.4f} [{row.fpr95_ci_low:.4f}, "
            f"{row.fpr95_ci_high:.4f}] |"
        )
    summary_lines.extend(
        [
            "",
            "## Replication predictions",
            "",
            *[f"- {row['prediction']}: **{row['status']}**" for row in prediction_rows],
            "",
            "## Limitation",
            "",
            LIMITATION,
            "",
        ]
    )
    paths["summary"].write_text("\n".join(summary_lines), encoding="utf-8")
    paths["metadata"].write_text(
        json.dumps(_json_value(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths
