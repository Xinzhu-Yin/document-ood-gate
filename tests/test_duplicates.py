import json
from pathlib import Path

import pandas as pd
from PIL import Image

from document_ood_gate.duplicates import (
    build_fingerprint_manifest,
    exact_duplicate_groups,
    finalize_duplicate_audit,
    measure_candidate_similarity,
    phash_candidate_pairs,
)


def _save_page(path: Path, dark_box: tuple[int, int, int, int]) -> None:
    image = Image.new("L", (64, 64), 255)
    pixels = image.load()
    for x in range(dark_box[0], dark_box[2]):
        for y in range(dark_box[1], dark_box[3]):
            pixels[x, y] = 0
    image.save(path)


def test_exact_hashes_and_phash_are_kept_separate(tmp_path: Path) -> None:
    original = tmp_path / "original_budget.tif"
    copied = tmp_path / "copied_budget.tif"
    reencoded = tmp_path / "reencoded_budget.png"
    different = tmp_path / "different.tif"
    _save_page(original, (8, 8, 24, 24))
    copied.write_bytes(original.read_bytes())
    with Image.open(original) as image:
        image.save(reencoded)
    _save_page(different, (40, 40, 56, 56))

    manifest = build_fingerprint_manifest([original, reencoded], [copied, different])
    exact = exact_duplicate_groups(manifest)
    assert (exact["hash_type"] == "raw_sha256").any()
    assert (exact["hash_type"] == "pixel_sha256").any()

    candidates = phash_candidate_pairs(manifest, max_distance=64)
    assert not candidates.empty
    assert set(candidates["review_status"]) == {"pending"}


def test_finalize_groups_only_exact_and_explicitly_confirmed_pairs(tmp_path: Path) -> None:
    exact_a = tmp_path / "exact_a_budget.tif"
    exact_b = tmp_path / "exact_b_budget.tif"
    near_a = tmp_path / "near_a_budget.tif"
    near_b = tmp_path / "near_b_budget.tif"
    unrelated = tmp_path / "unrelated.tif"
    _save_page(exact_a, (8, 8, 24, 24))
    exact_b.write_bytes(exact_a.read_bytes())
    _save_page(near_a, (30, 30, 45, 45))
    _save_page(near_b, (31, 30, 46, 45))
    _save_page(unrelated, (45, 45, 60, 60))

    manifest = build_fingerprint_manifest([exact_a, exact_b, near_a], [near_b, unrelated])
    candidates = phash_candidate_pairs(manifest, max_distance=64)
    measured = measure_candidate_similarity(manifest, candidates)
    assert {"gray_correlation_256", "gray_mae_256", "pixel_sha256_equal"} <= set(
        measured.columns
    )

    near_left = int(manifest.index[manifest["path"].str.endswith("near_a_budget.tif")][0])
    near_right = int(manifest.index[manifest["path"].str.endswith("near_b_budget.tif")][0])
    left_index, right_index = sorted((near_left, near_right))
    decision = {
        "confirmed_pairs": [
            {
                "left_index": left_index,
                "right_index": right_index,
                "left_file": Path(manifest.iloc[left_index]["path"]).name,
                "right_file": Path(manifest.iloc[right_index]["path"]).name,
                "note": "test visual confirmation",
            }
        ]
    }
    decision_path = tmp_path / "decisions.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    finalized, reviewed, summary = finalize_duplicate_audit(
        manifest,
        candidates,
        decision_path,
    )

    exact_groups = finalized.loc[
        finalized["path"].str.contains("exact_"), "confirmed_group_id"
    ]
    near_groups = finalized.loc[
        finalized["path"].str.contains("near_"), "confirmed_group_id"
    ]
    unrelated_group = finalized.loc[
        finalized["path"].str.endswith("unrelated.tif"), "confirmed_group_id"
    ].iloc[0]
    assert exact_groups.nunique() == 1 and exact_groups.notna().all()
    assert near_groups.nunique() == 1 and near_groups.notna().all()
    assert pd.isna(unrelated_group)
    assert "confirmed_near_visual" in set(reviewed["review_status"])
    assert summary["confirmed_near_visual_pairs"] == 1
