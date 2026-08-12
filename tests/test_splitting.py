from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest

from document_ood_gate.cli import build_parser, split_command
from document_ood_gate.splitting import create_research_split


def _manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_index in range(12):
        for item in range(20):
            rows.append(
                {
                    "path": f"N/class-{class_index:02d}/image-{item:02d}.png",
                    "subset": "N",
                    "source": f"n-source-{item % 2}",
                    "n_class": f"class-{class_index:02d}",
                    "confirmed_group_id": (
                        f"n-pair-{class_index:02d}-{item // 2:02d}" if item < 4 else None
                    ),
                    "score_must_not_leak": float(item),
                }
            )
    for source in ("documentcloud", "web-search"):
        for item in range(60):
            rows.append(
                {
                    "path": f"O/{source}/image-{item:02d}.png",
                    "subset": "O",
                    "source": source,
                    "n_class": None,
                    "confirmed_group_id": f"o-pair-{source}-{item // 2}" if item < 4 else None,
                    "score_must_not_leak": float(item),
                }
            )
    return pd.DataFrame(rows)


def test_grouped_split_is_reproducible_balanced_and_metadata_only() -> None:
    manifest = _manifest()
    assignments, summary = create_research_split(manifest, search_trials=1024)
    shuffled, shuffled_summary = create_research_split(
        manifest.sample(frac=1.0, random_state=91), search_trials=1024
    )

    pd.testing.assert_frame_equal(assignments, shuffled)
    assert summary == shuffled_summary
    assert len(assignments) == len(manifest)
    assert assignments["path"].is_unique
    assert set(assignments["path"]) == set(manifest["path"])
    assert set(assignments["split"]) == {"development", "sealed"}
    assert "score_must_not_leak" not in assignments.columns

    groups_per_split = assignments.groupby("split_group_id")["split"].nunique()
    assert groups_per_split.max() == 1

    n_fractions = (
        assignments[assignments["subset"] == "N"]
        .groupby(["n_class", "source"])["split"]
        .apply(lambda values: (values == "development").mean())
    )
    o_fractions = (
        assignments[assignments["subset"] == "O"]
        .groupby("source")["split"]
        .apply(lambda values: (values == "development").mean())
    )
    assert (n_fractions.sub(0.30).abs() <= 0.20).all()
    assert (o_fractions.sub(0.30).abs() <= 0.10).all()
    assert abs((assignments["split"] == "development").mean() - 0.30) <= 0.02
    assert summary["invariants"] == {
        "each_path_exactly_once": True,
        "confirmed_group_crosses_split": False,
        "model_outputs_or_scores_used": False,
    }


def test_blank_group_ids_are_independent_singletons() -> None:
    manifest = _manifest()
    blank_rows = manifest[manifest["confirmed_group_id"].isna()]
    assignments, _ = create_research_split(manifest, search_trials=256)
    singleton_ids = assignments[assignments["path"].isin(blank_rows["path"])]["split_group_id"]
    assert singleton_ids.str.startswith("singleton:").all()
    assert singleton_ids.is_unique


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="confirmed_group_id"), "missing required columns"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "exactly once"),
        (
            lambda frame: frame.assign(
                n_class=frame["n_class"].where(frame["path"] != "N/class-00/image-00.png")
            ),
            "Every N row",
        ),
    ],
)
def test_manifest_validation(mutation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        create_research_split(mutation(_manifest()), search_trials=8)


def test_expected_twelve_n_classes_is_enforced() -> None:
    manifest = _manifest()
    manifest = manifest[manifest["n_class"] != "class-11"].copy()
    with pytest.raises(ValueError, match="Expected 12 N classes"):
        create_research_split(manifest, search_trials=8)


def test_split_cli_writes_assignments_and_count_only_summary(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "split.csv"
    summary_path = tmp_path / "summary.json"
    _manifest().to_csv(manifest_path, index=False)
    args = argparse.Namespace(
        manifest=str(manifest_path),
        output=str(output_path),
        summary=str(summary_path),
        development_fraction=0.30,
        seed=20260812,
        group_column="confirmed_group_id",
        source_column="source",
        n_class_column="n_class",
        expected_n_classes=12,
        search_trials=128,
    )

    split_command(args)

    written = pd.read_csv(output_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(written) == len(_manifest())
    assert summary["rows"] == len(_manifest())
    assert summary["invariants"]["model_outputs_or_scores_used"] is False


def test_parser_exposes_split_research_command() -> None:
    args = build_parser().parse_args(
        [
            "split-research",
            "--manifest",
            "manifest.parquet",
            "--output",
            "split.parquet",
            "--summary",
            "summary.json",
        ]
    )
    assert args.func is split_command
    assert args.seed == 20260812
    assert args.development_fraction == 0.30
