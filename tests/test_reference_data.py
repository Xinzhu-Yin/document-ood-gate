import io
import tarfile
from pathlib import Path

import pandas as pd

from document_ood_gate.reference_data import (
    create_reference_sample_manifest,
    extract_selected_from_tar,
    read_official_split,
)


def _split_frame(split: str, per_class: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"split": split, "relative_path": f"images/{split}/{label}/{index}.tif", "label": label}
            for label in range(16)
            for index in range(per_class)
        ]
    )


def test_balanced_reference_samples_are_nested_and_reproducible() -> None:
    train = _split_frame("train", 6)
    validation = _split_frame("validation", 3)
    test = _split_frame("test", 3)
    manifest, summary = create_reference_sample_manifest(
        train,
        validation,
        test,
        reference_sizes=(32, 64),
        validation_size=32,
        test_size=32,
        seed=9,
    )
    repeated, repeated_summary = create_reference_sample_manifest(
        train,
        validation,
        test,
        reference_sizes=(32, 64),
        validation_size=32,
        test_size=32,
        seed=9,
    )
    pd.testing.assert_frame_equal(manifest, repeated)
    assert summary == repeated_summary
    assert summary["total_selected"] == 128
    reference = manifest[manifest["purpose"] == "reference"]
    assert (reference.groupby("label").size() == 4).all()
    assert (reference["minimum_reference_size"] == 32).sum() == 32


def test_selected_tar_extraction_does_not_unpack_unselected(tmp_path: Path) -> None:
    archive_path = tmp_path / "pages.tar.gz"
    selected_path = "images/a/b/selected.tif"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, content in ((selected_path, b"selected"), ("images/other.tif", b"other")):
            info = tarfile.TarInfo(f"images/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    manifest = pd.DataFrame(
        [{"relative_path": selected_path, "purpose": "reference", "label": 0}]
    )
    output = extract_selected_from_tar(archive_path, manifest, tmp_path / "output")
    local_path = Path(output.iloc[0]["local_path"])
    assert local_path.read_bytes() == b"selected"
    assert not (tmp_path / "output" / "images" / "other.tif").exists()


def test_official_split_parser(tmp_path: Path) -> None:
    path = tmp_path / "train.txt"
    path.write_text("images/a.tif 0\nimages/b.tif 15\n", encoding="utf-8")
    frame = read_official_split(path, "train")
    assert frame.to_dict("records") == [
        {"split": "train", "relative_path": "images/a.tif", "label": 0},
        {"split": "train", "relative_path": "images/b.tif", "label": 15},
    ]
