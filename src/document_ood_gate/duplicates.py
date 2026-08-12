from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class ImageFingerprint:
    path: str
    subset: str
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

