from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from document_ood_gate.final_cache import (
    SealedBoundaryError,
    _sealed_subset_array,
    validate_sealed_manifest_frame,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path": ["dev.png", "n1.png", "n2.png", "o1.png", "o2.png", "o3.png"],
            "subset": ["N", "N", "N", "O", "O", "O"],
            "split": ["development", "sealed", "sealed", "sealed", "sealed", "sealed"],
            "split_group_id": ["d", "n1", "n2", "o1", "o2", "o3"],
        }
    )


def test_sealed_manifest_selection_is_metadata_only_and_exact() -> None:
    sealed = validate_sealed_manifest_frame(_manifest(), expected_n=2, expected_o=3)
    assert sealed["path"].tolist() == ["n1.png", "n2.png", "o1.png", "o2.png", "o3.png"]


def test_sealed_manifest_rejects_count_drift() -> None:
    with pytest.raises(SealedBoundaryError, match="Expected 6 sealed rows"):
        validate_sealed_manifest_frame(_manifest(), expected_n=3, expected_o=3)


def test_sealed_manifest_rejects_duplicate_paths() -> None:
    frame = _manifest()
    frame.loc[5, "path"] = "o2.png"
    with pytest.raises(SealedBoundaryError, match="duplicate image paths"):
        validate_sealed_manifest_frame(frame, expected_n=2, expected_o=3)


def test_sealed_subset_cache_is_pickle_free(tmp_path) -> None:
    labels = _sealed_subset_array(pd.Series(["N", "O", "N"], dtype="string"))
    assert labels.dtype.kind == "U"
    cache = tmp_path / "subsets.npy"
    np.save(cache, labels, allow_pickle=False)
    np.testing.assert_array_equal(np.load(cache, allow_pickle=False), labels)
