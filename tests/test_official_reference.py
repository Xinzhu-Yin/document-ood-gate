import numpy as np

from document_ood_gate.official_reference import (
    compare_logits_to_official,
    hybrid_reproduction_report,
)


def test_compare_logits_to_official_aligns_prefixed_names() -> None:
    paths = np.array(["folder_documentcloud_4_idood__a_budget.tif"])
    expected = np.arange(16, dtype=np.float64)
    observed = expected + 0.1
    official = {
        "idood__a_budget.tif": {
            "filename": "idood__a_budget.tif",
            "logits": expected.tolist(),
        }
    }
    result = compare_logits_to_official(paths, observed[None, :], official)
    assert result["matched_rows"] == 1
    assert np.isclose(result["logit_mean_abs_error"], 0.1)
    assert result["prediction_agreement"] == 1.0


def test_hybrid_reproduction_uses_fallback_only_for_missing_rows() -> None:
    n = {
        f"n-{index}.tif": {
            "filename": f"n-{index}.tif",
            "true_label": "letter",
            "logits": [1.0] + [0.0] * 15,
        }
        for index in range(1002)
    }
    o = {
        f"o-{index}.tif": {
            "filename": f"o-{index}.tif",
            "true_label": "ood",
            "logits": [0.1] + [0.0] * 15,
        }
        for index in range(3415)
    }
    paths = [*list(n)[:1002], *list(o)[:3414]]
    logits = np.asarray(
        [n[path]["logits"] if path in n else o[path]["logits"] for path in paths]
    )
    report = hybrid_reproduction_report(
        {"paths": np.asarray(paths), "logits": logits},
        n,
        o,
    )
    assert report["provenance"]["actual_image_reruns"] == 4416
    assert report["provenance"]["author_logit_fallback"] == 1
    assert report["provenance"]["fallback_filenames"] == ["o-3414.tif"]
