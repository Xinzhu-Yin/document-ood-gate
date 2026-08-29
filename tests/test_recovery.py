from pathlib import Path

import pytest

from document_ood_gate.recovery import recovered_tiff_name, target_size


def test_target_size_matches_calibrated_letter_page() -> None:
    assert target_size(1700, 2200) == (772, 1000)
    assert target_size(2200, 1700) == (1000, 772)
    assert target_size(1692, 2208) == (766, 1000)


def test_recovered_name_matches_official_output() -> None:
    assert (
        recovered_tiff_name(Path("documentcloud_00749.pdf"))
        == "ood__documentcloud_00749.tif"
    )


def test_recovered_name_rejects_unexpected_source() -> None:
    with pytest.raises(ValueError, match="Unexpected"):
        recovered_tiff_name(Path("mystery_00749.pdf"))
