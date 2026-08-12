from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


N_CLASSES = (
    "budget",
    "email",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo",
    "news_article",
    "questionnaire",
    "resume",
    "scientific_publication",
    "specification",
)

DIT_ID2LABEL = {
    0: "letter",
    1: "form",
    2: "email",
    3: "handwritten",
    4: "advertisement",
    5: "scientific_report",
    6: "scientific_publication",
    7: "specification",
    8: "file_folder",
    9: "news_article",
    10: "budget",
    11: "invoice",
    12: "presentation",
    13: "questionnaire",
    14: "resume",
    15: "memo",
}


@dataclass(frozen=True)
class OfficialImageMetadata:
    source: str
    n_class: str | None


def normalize_class_name(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "article": "news_article",
        "publication": "scientific_publication",
        "scientific_pub": "scientific_publication",
    }
    return aliases.get(normalized, normalized)


def normalize_official_filename(path: Path | str) -> str:
    """Remove download-folder prefixes while preserving the author's sample filename."""

    name = Path(path).name
    for marker in ("idood__", "ood__"):
        index = name.find(marker)
        if index >= 0:
            return name[index:]
    return name


def parse_official_image_metadata(path: Path, subset: str) -> OfficialImageMetadata:
    """Parse source and N class from the author's published filename conventions."""

    text = path.as_posix().lower()
    source = "documentcloud" if "documentcloud" in text else "web_search"
    subset_value = subset.upper()
    if subset_value == "O":
        return OfficialImageMetadata(source=source, n_class=None)
    if subset_value != "N":
        raise ValueError("subset must be N or O")

    stem = path.stem.lower().replace("-", "_")
    matches = [name for name in N_CLASSES if stem.endswith(f"_{name}") or stem == name]
    if not matches:
        # The released JSON uses short true-label aliases for these two categories.
        if stem.endswith("_article"):
            matches = ["news_article"]
        elif stem.endswith("_publication"):
            matches = ["scientific_publication"]
    if len(matches) != 1:
        raise ValueError(f"Could not determine one N class from filename: {path}")
    return OfficialImageMetadata(source=source, n_class=matches[0])
