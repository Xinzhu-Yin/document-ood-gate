import json

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from document_ood_gate.layer_reporting import FULL_RESULT_COLUMNS, write_layer_scan_reports


def _metric_record(
    experiment: str,
    layer: int,
    representation: str,
    method: str,
    bank_size: int,
    *,
    offset: float,
) -> dict[str, object]:
    auroc = 0.60 + layer * 0.01 + offset + (0.01 if bank_size == 32000 else 0.0)
    fpr95 = 0.92 - layer * 0.01 - offset - (0.01 if bank_size == 32000 else 0.0)
    return {
        "experiment": experiment,
        "layer": layer,
        "representation": representation,
        "method": method,
        "residual_centering_convention": (
            "reference_mean" if "residual" in method.lower() else "not_applicable"
        ),
        "bank_size": bank_size,
        "auroc": auroc,
        "auroc_95ci": [auroc - 0.01, auroc + 0.01],
        "fpr95": fpr95,
        "fpr95_95ci": [fpr95 - 0.01, fpr95 + 0.01],
        "top_512_explained_variance_ratio": 0.80 + layer * 0.005,
    }


def _synthetic_records() -> list[dict[str, object]]:
    methods = (
        ("Mahalanobis", 0.00),
        ("kNN", 0.02),
        ("mean_centered_pca_residual", 0.04),
    )
    records = [
        _metric_record("S2", layer, "raw_patch_mean", method, bank, offset=offset)
        for layer in range(1, 13)
        for method, offset in methods
        for bank in (8000, 32000)
    ]
    records.extend(
        _metric_record("S1", 12, representation, method, bank, offset=offset - 0.01)
        for representation in ("raw_patch_mean", "h_doc", "CLS")
        for method, offset in methods
        for bank in (8000, 32000)
    )
    return records


def _vim_records() -> list[dict[str, object]]:
    rows = []
    for index, (method, centering) in enumerate(
        (
            ("Energy", "not_applicable"),
            ("ViM-origin Residual", "classifier_origin"),
            ("ViM", "classifier_origin_plus_energy"),
        )
    ):
        auroc = 0.70 + index * 0.02
        fpr95 = 0.85 - index * 0.03
        rows.append(
            {
                "experiment": "S1b",
                "layer": 12,
                "method": method,
                "representation": "h_doc",
                "residual_centering_convention": centering,
                "bank_size": 32000,
                "auroc": auroc,
                "auroc_95ci": [auroc - 0.01, auroc + 0.01],
                "fpr95": fpr95,
                "fpr95_95ci": [fpr95 - 0.02, fpr95 + 0.02],
                "top_512_explained_variance_ratio": 0.91,
            }
        )
    return rows


def _metadata() -> dict[str, object]:
    return {
        "extraction_image_count": 33280,
        "reference_image_count": 32000,
        "development_n_count": 306,
        "development_o_count": 974,
        "sealed_rows_loaded": 0,
        "sealed_rows_scored": 0,
        "t_guardrail_rescored": False,
        "cache_dtype": "float32",
    }


def test_layer_reporting_writes_complete_tables_plots_and_disclaimers(tmp_path) -> None:
    paths = write_layer_scan_reports(
        [*_synthetic_records(), *_vim_records()],
        _vim_records(),
        gates={"G1": {"passed": True}, "G2": {"passed": True}},
        metadata=_metadata(),
        output_dir=tmp_path,
    )

    assert set(paths) == {
        "full_csv",
        "full_json",
        "vim_csv",
        "vim_json",
        "summary",
        "final_layer_plot",
        "auroc_plot",
        "fpr95_plot",
    }
    assert all(path.is_file() for path in paths.values())
    for key in ("final_layer_plot", "auroc_plot", "fpr95_plot"):
        assert paths[key].stat().st_size > 1_000

    full_frame = pd.read_csv(paths["full_csv"])
    assert set(FULL_RESULT_COLUMNS).issubset(full_frame.columns)
    assert len(full_frame) == 93
    assert set(full_frame["layer"]) == set(range(1, 13))
    assert set(full_frame["bank_size"]) == {8000, 32000}
    assert set(full_frame["experiment"]) == {"S1", "S1b", "S2"}

    full_json = json.loads(paths["full_json"].read_text(encoding="utf-8"))
    assert full_json["metadata"]["sealed_rows_loaded"] == 0
    assert full_json["metadata"]["sealed_rows_scored"] == 0
    assert full_json["gates"]["G1"]["passed"] is True
    assert len(full_json["records"]) == 93
    assert len(json.loads(paths["vim_json"].read_text(encoding="utf-8"))["records"]) == 3

    summary = paths["summary"].read_text(encoding="utf-8")
    assert "33,280 images" in summary
    assert "Sealed rows loaded: **0**; sealed rows scored: **0**" in summary
    assert "T guardrail was not re-extracted or rescored" in summary
    assert "Mean-centered PCA Residual (S1/S2)" in summary
    assert "ViM-origin Residual (S1b)" in summary
    assert "development observation; not confirmed on held-out data" in summary
    assert "does not select an argmax layer" in summary
    for layer in range(1, 13):
        assert f"| {layer} |" in summary


def test_layer_reporting_refuses_nonzero_sealed_access(tmp_path) -> None:
    metadata = _metadata()
    metadata["sealed_rows_scored"] = 1
    with pytest.raises(ValueError, match="sealed_rows_scored must be 0"):
        write_layer_scan_reports(
            [*_synthetic_records(), *_vim_records()],
            _vim_records(),
            gates={"G1": True, "G2": True},
            metadata=metadata,
            output_dir=tmp_path,
        )
