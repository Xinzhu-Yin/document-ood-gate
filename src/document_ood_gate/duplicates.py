from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .metadata import parse_official_image_metadata


@dataclass(frozen=True)
class ImageFingerprint:
    path: str
    subset: str
    source: str
    n_class: str | None
    confirmed_group_id: str | None
    size_bytes: int
    width: int
    height: int
    mode: str
    image_format: str
    raw_sha256: str
    pixel_sha256: str
    phash_hex: str


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path, subset: str) -> ImageFingerprint:
    metadata = parse_official_image_metadata(path, subset)
    with Image.open(path) as image:
        image.load()
        mode = str(image.mode)
        image_format = str(image.format or "unknown")
        width, height = image.size
        normalized = image.convert("L")
        pixel_digest = hashlib.sha256()
        pixel_digest.update(f"L:{width}x{height}:".encode("ascii"))
        pixel_digest.update(normalized.tobytes())
        phash_hex = str(imagehash.phash(normalized, hash_size=8))
    return ImageFingerprint(
        path=str(path.resolve()),
        subset=subset,
        source=metadata.source,
        n_class=metadata.n_class,
        confirmed_group_id=None,
        size_bytes=path.stat().st_size,
        width=width,
        height=height,
        mode=mode,
        image_format=image_format,
        raw_sha256=_sha256_file(path),
        pixel_sha256=pixel_digest.hexdigest(),
        phash_hex=phash_hex,
    )


def build_fingerprint_manifest(n_paths: list[Path], o_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset, paths in (("N", n_paths), ("O", o_paths)):
        for path in tqdm(paths, desc=f"Fingerprint {subset}"):
            rows.append(asdict(fingerprint(path, subset)))
    return pd.DataFrame(rows).sort_values(["subset", "path"]).reset_index(drop=True)


def exact_duplicate_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for hash_type in ("raw_sha256", "pixel_sha256"):
        duplicated = manifest[manifest.duplicated(hash_type, keep=False)].copy()
        if duplicated.empty:
            continue
        duplicated.insert(0, "hash_type", hash_type)
        duplicated.insert(1, "duplicate_key", duplicated[hash_type])
        parts.append(duplicated[["hash_type", "duplicate_key", "subset", "path"]])
    if not parts:
        return pd.DataFrame(columns=["hash_type", "duplicate_key", "subset", "path"])
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def phash_candidate_pairs(manifest: pd.DataFrame, max_distance: int = 8, block: int = 256) -> pd.DataFrame:
    """Generate candidates only; pHash hits are not automatic duplicate groups."""

    hashes = np.array([int(value, 16) for value in manifest["phash_hex"]], dtype=np.uint64)
    popcount = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    pairs: list[dict[str, object]] = []
    total = len(hashes)
    for start in tqdm(range(0, total, block), desc="pHash candidate search"):
        stop = min(start + block, total)
        xor = np.bitwise_xor(hashes[start:stop, None], hashes[None, :])
        byte_view = xor.view(np.uint8).reshape(stop - start, total, 8)
        distances = popcount[byte_view].sum(axis=2)
        local_i, j = np.where(distances <= max_distance)
        i = local_i + start
        keep = i < j
        for left, right, distance in zip(i[keep], j[keep], distances[local_i[keep], j[keep]], strict=True):
            pairs.append(
                {
                    "left_index": int(left),
                    "right_index": int(right),
                    "left_subset": manifest.iloc[left]["subset"],
                    "right_subset": manifest.iloc[right]["subset"],
                    "left_path": manifest.iloc[left]["path"],
                    "right_path": manifest.iloc[right]["path"],
                    "phash_distance": int(distance),
                    "review_status": "pending",
                    "review_note": "",
                }
            )
    return pd.DataFrame(pairs)


def _normalized_gray(path: str, size: int) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L").resize((size, size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / np.float32(255.0)


def measure_candidate_similarity(
    manifest: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    comparison_size: int = 256,
) -> pd.DataFrame:
    """Add conservative review measurements without deciding that a pair is duplicate."""

    if comparison_size < 32:
        raise ValueError("comparison_size must be at least 32")
    required = {
        "left_index",
        "right_index",
        "left_path",
        "right_path",
        "review_status",
        "review_note",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate table is missing columns: {', '.join(missing)}")

    cache: dict[str, np.ndarray] = {}

    def normalized(path: str) -> np.ndarray:
        if path not in cache:
            cache[path] = _normalized_gray(path, comparison_size)
        return cache[path]

    rows: list[dict[str, object]] = []
    for candidate in candidates.itertuples(index=False):
        left_index = int(candidate.left_index)
        right_index = int(candidate.right_index)
        left = manifest.iloc[left_index]
        right = manifest.iloc[right_index]
        if str(left["path"]) != str(candidate.left_path) or str(right["path"]) != str(
            candidate.right_path
        ):
            raise ValueError("Candidate indices and paths do not match the fingerprint manifest")

        left_array = normalized(str(candidate.left_path))
        right_array = normalized(str(candidate.right_path))
        left_flat = left_array.ravel()
        right_flat = right_array.ravel()
        left_std = float(left_flat.std())
        right_std = float(right_flat.std())
        if left_std <= 1e-8 or right_std <= 1e-8:
            correlation = float(left_std <= 1e-8 and right_std <= 1e-8)
        else:
            correlation = float(np.corrcoef(left_flat, right_flat)[0, 1])

        row = candidate._asdict()
        row.update(
            {
                "raw_sha256_equal": bool(left["raw_sha256"] == right["raw_sha256"]),
                "pixel_sha256_equal": bool(left["pixel_sha256"] == right["pixel_sha256"]),
                "gray_correlation_256": correlation,
                "gray_mae_256": float(np.abs(left_array - right_array).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _union_equal_hashes(manifest: pd.DataFrame, union_find: _UnionFind) -> None:
    for column in ("raw_sha256", "pixel_sha256"):
        for _, indices in manifest.groupby(column, sort=False).groups.items():
            group = [int(index) for index in indices]
            for index in group[1:]:
                union_find.union(group[0], index)


def _load_manual_confirmations(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("confirmed_pairs")
    if not isinstance(pairs, list):
        raise ValueError("Manual decision JSON must contain a confirmed_pairs list")
    return pairs


def finalize_duplicate_audit(
    manifest: pd.DataFrame,
    candidates: pd.DataFrame,
    manual_decisions_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply exact hashes plus explicit visual decisions to create safe split groups."""

    reviewed = measure_candidate_similarity(manifest, candidates)
    reviewed["review_status"] = "not_confirmed"
    reviewed["review_note"] = "pHash candidate only; not used as a duplicate group"
    exact_mask = reviewed["raw_sha256_equal"] | reviewed["pixel_sha256_equal"]
    reviewed.loc[exact_mask, "review_status"] = "confirmed_exact"
    reviewed.loc[exact_mask, "review_note"] = "identical raw bytes or decoded grayscale pixels"

    confirmed_pairs: set[tuple[int, int]] = set()
    for decision in _load_manual_confirmations(manual_decisions_path):
        left_index = int(decision["left_index"])
        right_index = int(decision["right_index"])
        key = (min(left_index, right_index), max(left_index, right_index))
        matches = (reviewed["left_index"] == key[0]) & (reviewed["right_index"] == key[1])
        if int(matches.sum()) != 1:
            raise ValueError(f"Manual pair {key} does not identify exactly one pHash candidate")
        row = reviewed.loc[matches].iloc[0]
        expected_left = str(decision["left_file"])
        expected_right = str(decision["right_file"])
        if os.path.basename(str(row["left_path"])) != expected_left or os.path.basename(
            str(row["right_path"])
        ) != expected_right:
            raise ValueError(f"Manual pair {key} filename check failed")
        if bool(row["pixel_sha256_equal"]):
            raise ValueError(f"Manual near-duplicate pair {key} is already an exact pixel duplicate")
        reviewed.loc[matches, "review_status"] = "confirmed_near_visual"
        reviewed.loc[matches, "review_note"] = str(
            decision.get("note", "same page confirmed by side-by-side visual review")
        )
        confirmed_pairs.add(key)

    union_find = _UnionFind(len(manifest))
    _union_equal_hashes(manifest, union_find)
    for left_index, right_index in confirmed_pairs:
        union_find.union(left_index, right_index)

    components: dict[int, list[int]] = {}
    for index in range(len(manifest)):
        components.setdefault(union_find.find(index), []).append(index)

    output = manifest.copy()
    output["confirmed_group_id"] = None
    confirmed_components = [indices for indices in components.values() if len(indices) > 1]
    for indices in confirmed_components:
        digest_input = "|".join(sorted(str(output.iloc[index]["pixel_sha256"]) for index in indices))
        group_id = "dup:" + hashlib.sha256(digest_input.encode("ascii")).hexdigest()[:16]
        output.loc[indices, "confirmed_group_id"] = group_id

    cross_label_groups = 0
    for indices in confirmed_components:
        if output.iloc[indices]["subset"].nunique() > 1:
            cross_label_groups += 1
    summary: dict[str, Any] = {
        "images": int(len(output)),
        "phash_candidates": int(len(reviewed)),
        "confirmed_exact_pairs_in_candidates": int(exact_mask.sum()),
        "confirmed_near_visual_pairs": int(len(confirmed_pairs)),
        "confirmed_duplicate_groups": int(len(confirmed_components)),
        "images_in_confirmed_groups": int(sum(len(indices) for indices in confirmed_components)),
        "cross_label_confirmed_groups": int(cross_label_groups),
        "not_confirmed_phash_candidates": int((reviewed["review_status"] == "not_confirmed").sum()),
        "grouping_rule": (
            "Only byte/pixel-identical images and explicitly side-by-side-confirmed near duplicates "
            "share a group. pHash alone never creates a group."
        ),
    }
    return output, reviewed, summary
