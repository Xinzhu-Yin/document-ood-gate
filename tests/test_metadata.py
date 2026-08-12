from pathlib import Path

import pytest

from document_ood_gate.metadata import normalize_official_filename, parse_official_image_metadata


@pytest.mark.parametrize(
    ("path", "source", "n_class"),
    [
        (
            "documentcloud_4/idood__pdf3_documentcloud_00002_questionnaire.tif",
            "documentcloud",
            "questionnaire",
        ),
        ("websearch_5/idood__00017_news_article.tif", "web_search", "news_article"),
        ("idood__00018_scientific_publication.tif", "web_search", "scientific_publication"),
    ],
)
def test_parse_n_metadata(path: str, source: str, n_class: str) -> None:
    actual = parse_official_image_metadata(Path(path), "N")
    assert actual.source == source
    assert actual.n_class == n_class


def test_parse_o_metadata() -> None:
    dc = parse_official_image_metadata(Path("ood__documentcloud_00749.tif"), "O")
    web = parse_official_image_metadata(Path("ood__03019.tif"), "O")
    assert (dc.source, dc.n_class) == ("documentcloud", None)
    assert (web.source, web.n_class) == ("web_search", None)


def test_unknown_n_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="Could not determine"):
        parse_official_image_metadata(Path("idood__mystery.tif"), "N")


def test_normalize_official_filename() -> None:
    assert (
        normalize_official_filename("documentcloud_4_idood__pdf3_00002_budget.tif")
        == "idood__pdf3_00002_budget.tif"
    )
    assert normalize_official_filename("web_7_ood__00123.tif") == "ood__00123.tif"
