from argparse import Namespace

import pandas as pd
import pytest

from document_ood_gate import cli


def test_supplementary_cli_refuses_mixed_split_before_opening_cache(
    tmp_path, monkeypatch
) -> None:
    split_path = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "path": ["development.tif", "sealed.tif"],
            "subset": ["N", "O"],
            "split": ["development", "sealed"],
        }
    ).to_csv(split_path, index=False)

    def fail_if_loaded(_path):
        raise AssertionError("feature cache was opened before the split boundary check")

    monkeypatch.setattr(cli, "load_feature_cache", fail_if_loaded)
    with pytest.raises(SystemExit, match="physically isolated development-only"):
        cli.supplementary_baselines_command(
            Namespace(split=str(split_path), ood_features="must-not-open.npz")
        )


def test_parser_exposes_development_only_extraction() -> None:
    args = cli.build_parser().parse_args(
        [
            "extract-development-ood",
            "--split",
            "research-split.csv",
            "--output-features",
            "development.npz",
            "--output-split",
            "development.csv",
        ]
    )
    assert args.func is cli.extract_development_ood_command
    assert args.expected == 1294
    assert args.dtype == "float32"
