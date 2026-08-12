from __future__ import annotations

import json
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def find_images(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def save_feature_cache(
    path: Path,
    image_paths: list[Path],
    subsets: list[str],
    logits: np.ndarray,
    features: np.ndarray,
    true_labels: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "paths": np.asarray([str(p.resolve()) for p in image_paths]),
        "subsets": np.asarray(subsets),
        "logits": np.asarray(logits, dtype=np.float32),
        "features": np.asarray(features, dtype=np.float32),
    }
    if true_labels is not None:
        if len(true_labels) != len(image_paths):
            raise ValueError("true_labels must match image_paths length")
        payload["true_labels"] = np.asarray(true_labels)
    np.savez_compressed(
        path,
        **payload,
    )


def load_feature_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
