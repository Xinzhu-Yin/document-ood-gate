"""Print committed aggregate results without inference or third-party dependencies."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results/tables/07_sealed_test/sealed_baselines.csv"
METADATA = ROOT / "results/supporting/07_sealed_test/run_metadata.json"

LABELS = {
    "msp": "MSP",
    "energy": "Energy",
    "mahalanobis": "Mahalanobis (32k)",
    "knn": "kNN (32k, k=10)",
    "vim_origin_residual": "ViM-origin Residual",
    "vim": "ViM (32k)",
}


def interval(row: dict[str, str], metric: str) -> str:
    value = float(row[metric])
    low = float(row[f"{metric}_ci_low"])
    high = float(row[f"{metric}_ci_high"])
    return f"{value:.4f} [{low:.4f}, {high:.4f}]"


def main() -> int:
    try:
        metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        with TABLE.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 6 or {row["method"] for row in rows} != set(LABELS):
            raise ValueError("Expected the six recorded core baseline methods.")
        if any(row["split"] != "sealed_test" or row["experiment"] != "D1" for row in rows):
            raise ValueError("Expected only the recorded sealed D1 baseline panel.")
        lines = [
            "Document OOD Gate | saved results preview",
            "Reads aggregate artifacts only; no inference or metric recomputation.",
            "",
            f"Sealed set: {metadata['sealed_N']} N + {metadata['sealed_O']} O",
            f"95% CIs: {metadata['bootstrap_iterations']} stratified bootstrap resamples",
            "",
            f"{'Method':<24} {'AUROC (higher is better)':<27} FPR95 (lower is better)",
            "-" * 80,
        ]
        for row in rows:
            lines.append(
                f"{LABELS[row['method']]:<24} {interval(row, 'auroc'):<27} "
                f"{interval(row, 'fpr95')}"
            )
    except (OSError, ValueError, KeyError) as error:
        print(f"Cannot read the saved result bundle: {error}", file=sys.stderr)
        return 1

    lines.extend(
        [
            "",
            "FPR95 is the fraction of O accepted when at least 95% of N is accepted.",
            "These results do not establish a deployable gate.",
            f"Source: {TABLE.relative_to(ROOT).as_posix()}",
            "Full layer grid, limitations, and provenance: results/README.md",
        ]
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
