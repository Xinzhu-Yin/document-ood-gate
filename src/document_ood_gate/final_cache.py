from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .layer_cache import FEATURE_DIM, MODEL_ID, MODEL_REVISION, NUM_CLASSES, NUM_LAYERS


SEALED_N = 694
SEALED_O = 2324
SEALED_ROWS = SEALED_N + SEALED_O


class SealedBoundaryError(RuntimeError):
    """Raised before a sealed image is opened when the frozen boundary is invalid."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _ordered_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def validate_sealed_manifest_frame(
    frame: pd.DataFrame,
    *,
    expected_n: int = SEALED_N,
    expected_o: int = SEALED_O,
) -> pd.DataFrame:
    """Select and validate the frozen sealed rows without opening image files."""

    required = {"path", "subset", "split", "split_group_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SealedBoundaryError("Split manifest is missing: " + ", ".join(missing))
    sealed = frame.loc[frame["split"].astype(str).eq("sealed")].copy()
    if len(sealed) != expected_n + expected_o:
        raise SealedBoundaryError(
            f"Expected {expected_n + expected_o} sealed rows; found {len(sealed)}"
        )
    subsets = sealed["subset"].astype(str)
    counts = subsets.value_counts().to_dict()
    if counts != {"O": expected_o, "N": expected_n}:
        raise SealedBoundaryError(
            f"Expected sealed N={expected_n}, O={expected_o}; found {counts}"
        )
    if sealed["path"].astype(str).duplicated().any():
        raise SealedBoundaryError("Sealed manifest contains duplicate image paths")
    if sealed["split_group_id"].astype(str).str.strip().eq("").any():
        raise SealedBoundaryError("Sealed manifest contains a blank split_group_id")
    return sealed.reset_index(drop=True)


def load_sealed_manifest(path: Path) -> pd.DataFrame:
    """This is the explicit sealed-access boundary; callers must invoke it only after freeze."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return validate_sealed_manifest_frame(pd.read_csv(manifest_path))


def _forward_sealed(
    model: Any,
    pixel_values: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    output = model.beit(
        pixel_values=pixel_values,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = output.hidden_states
    if hidden_states is None or len(hidden_states) != NUM_LAYERS + 1:
        raise RuntimeError("Expected embedding output plus 12 hidden states")
    batch_rows = int(pixel_values.shape[0])
    raw_by_layer: list[torch.Tensor] = []
    for layer in range(1, NUM_LAYERS + 1):
        hidden = hidden_states[layer]
        if tuple(hidden.shape[:1]) != (batch_rows,) or hidden.shape[1:] != (197, FEATURE_DIM):
            raise RuntimeError(f"Layer {layer} has unexpected shape {tuple(hidden.shape)}")
        if hidden.dtype != torch.float32:
            raise RuntimeError(f"Layer {layer} must be float32; found {hidden.dtype}")
        raw_by_layer.append(hidden[:, 1:, :].mean(dim=1))
    raw_gpu = torch.stack(raw_by_layer, dim=1)
    h_doc_gpu = output.pooler_output
    if h_doc_gpu is None or tuple(h_doc_gpu.shape) != (batch_rows, FEATURE_DIM):
        raise RuntimeError("pooler_output must have shape [batch, 768]")
    logits_gpu = model.classifier(h_doc_gpu)
    reconstructed = model.beit.pooler.layernorm(raw_by_layer[-1])
    max_abs_error = float((reconstructed - h_doc_gpu).abs().max().item())
    if max_abs_error >= 1.0e-5:
        raise RuntimeError(f"Sealed structure check failed: max_abs_error={max_abs_error}")
    raw = raw_gpu.detach().to(dtype=torch.float32).cpu().numpy()
    h_doc = h_doc_gpu.detach().to(dtype=torch.float32).cpu().numpy()
    logits = logits_gpu.detach().to(dtype=torch.float32).cpu().numpy()
    if raw.shape != (batch_rows, NUM_LAYERS, FEATURE_DIM):
        raise RuntimeError("Only pooled [batch, 12, 768] raw means may cross to CPU")
    if h_doc.shape != (batch_rows, FEATURE_DIM) or logits.shape != (batch_rows, NUM_CLASSES):
        raise RuntimeError("Final representation or logits have an invalid shape")
    return raw, h_doc, logits, max_abs_error


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def extract_sealed_feature_cache(
    *,
    split_manifest_path: Path,
    output_dir: Path,
    freeze_commit_sha: str,
    freeze_commit_timestamp: str,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """Open exactly the 3,018 frozen sealed images once and cache only fixed outputs."""

    if len(freeze_commit_sha) != 40 or any(ch not in "0123456789abcdef" for ch in freeze_commit_sha):
        raise SealedBoundaryError("freeze_commit_sha must be a lowercase 40-character Git SHA")
    try:
        freeze_time = datetime.fromisoformat(freeze_commit_timestamp)
    except ValueError as error:
        raise SealedBoundaryError("freeze_commit_timestamp must be ISO-8601") from error
    if freeze_time.tzinfo is None:
        raise SealedBoundaryError("freeze_commit_timestamp must include a timezone")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    output = Path(output_dir)
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("complete") is True and metadata.get("extracted_rows") == SEALED_ROWS:
            return metadata
        raise SealedBoundaryError(
            "An incomplete sealed cache already exists; refusing an uncertain second access"
        )
    if output.exists() and any(output.iterdir()):
        raise SealedBoundaryError("Sealed output directory is non-empty before first access")
    output.mkdir(parents=True, exist_ok=True)

    # This call is intentionally the first operation that reads the full split manifest.
    sealed = load_sealed_manifest(split_manifest_path)
    paths = [Path(value) for value in sealed["path"].astype(str)]
    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        raise SealedBoundaryError(
            f"Sealed manifest contains {len(missing_paths)} missing image files"
        )

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=False,
    )
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.eval()
    model.requires_grad_(False)
    model.to(target_device, dtype=torch.float32)

    raw_cache = np.lib.format.open_memmap(
        output / "sealed_raw_patch_mean.npy",
        mode="w+",
        dtype=np.float32,
        shape=(SEALED_ROWS, NUM_LAYERS, FEATURE_DIM),
    )
    h_doc_cache = np.lib.format.open_memmap(
        output / "sealed_h_doc.npy",
        mode="w+",
        dtype=np.float32,
        shape=(SEALED_ROWS, FEATURE_DIM),
    )
    logits_cache = np.lib.format.open_memmap(
        output / "sealed_logits.npy",
        mode="w+",
        dtype=np.float32,
        shape=(SEALED_ROWS, NUM_CLASSES),
    )
    started_at = _utc_now()
    max_structure_error = 0.0
    with torch.inference_mode():
        for start in tqdm(range(0, SEALED_ROWS, batch_size), desc="One-time sealed extraction"):
            end = min(SEALED_ROWS, start + batch_size)
            images: list[Image.Image] = []
            try:
                for image_path in paths[start:end]:
                    with Image.open(image_path) as image:
                        images.append(image.convert("RGB"))
                encoded = processor(images=images, return_tensors="pt")
                pixel_values = encoded["pixel_values"].to(target_device, dtype=torch.float32)
                raw, h_doc, logits, structure_error = _forward_sealed(model, pixel_values)
            finally:
                for image in images:
                    image.close()
            raw_cache[start:end] = raw
            h_doc_cache[start:end] = h_doc
            logits_cache[start:end] = logits
            max_structure_error = max(max_structure_error, structure_error)
    raw_cache.flush()
    h_doc_cache.flush()
    logits_cache.flush()
    np.save(output / "sealed_subsets.npy", sealed["subset"].astype("U1").to_numpy())
    ids = sealed["path"].astype(str).tolist()
    (output / "sealed_ids.jsonl").write_text(
        "".join(json.dumps({"path": value}) + "\n" for value in ids),
        encoding="utf-8",
    )
    metadata = {
        "complete": True,
        "scope": "sealed_test_exactly_once",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "freeze_commit_sha": freeze_commit_sha,
        "freeze_commit_timestamp": freeze_commit_timestamp,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "extracted_rows": SEALED_ROWS,
        "N": SEALED_N,
        "O": SEALED_O,
        "image_open_count": SEALED_ROWS,
        "forward_count_per_image": 1,
        "reference_images_extracted": 0,
        "reference_cache_reused": "data/features/layer_scan/reference_*",
        "cache_dtype": "float32",
        "raw_cache_shape": [SEALED_ROWS, NUM_LAYERS, FEATURE_DIM],
        "h_doc_cache_shape": [SEALED_ROWS, FEATURE_DIM],
        "logits_cache_shape": [SEALED_ROWS, NUM_CLASSES],
        "sealed_path_order_sha256": _ordered_hash(ids),
        "max_l12_pooler_reconstruction_error": max_structure_error,
    }
    _write_json(metadata_path, metadata)
    return metadata


def load_sealed_cache(output_dir: Path) -> dict[str, np.ndarray]:
    output = Path(output_dir)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("complete") is not True or metadata.get("extracted_rows") != SEALED_ROWS:
        raise SealedBoundaryError("Sealed cache is not a complete 3,018-row extraction")
    arrays = {
        "raw": np.load(output / "sealed_raw_patch_mean.npy", mmap_mode="r"),
        "h_doc": np.load(output / "sealed_h_doc.npy", mmap_mode="r"),
        "logits": np.load(output / "sealed_logits.npy", mmap_mode="r"),
        "subsets": np.load(output / "sealed_subsets.npy", allow_pickle=False),
    }
    expected = {
        "raw": (SEALED_ROWS, NUM_LAYERS, FEATURE_DIM),
        "h_doc": (SEALED_ROWS, FEATURE_DIM),
        "logits": (SEALED_ROWS, NUM_CLASSES),
        "subsets": (SEALED_ROWS,),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise SealedBoundaryError(f"Sealed {name} cache has shape {arrays[name].shape}")
    return arrays
