from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from document_ood_gate import layer_cache
from document_ood_gate.layer_cache import (
    LayerCacheBoundaryError,
    LayerCacheGateError,
    LayerCachePlan,
    load_development_only_manifest,
    load_layer_cache_metadata,
    select_frozen_reference_manifest,
)


def _reference_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "purpose": "reference",
                "relative_path": f"images/train/{label}/{rank}.tif",
                "local_path": f"reference/{label}/{rank}.tif",
                "label": label,
                "class_sample_rank": rank,
                "minimum_reference_size": 8_000 if rank < 500 else 32_000,
            }
            for label in range(16)
            for rank in range(2_000)
        ]
    )


def _development_manifest() -> pd.DataFrame:
    records = [
        {
            "path": f"development/N/{index}.tif",
            "subset": "N",
            "split": "development",
            "split_group_id": f"n-{index}",
        }
        for index in range(306)
    ]
    records.extend(
        {
            "path": f"development/O/{index}.tif",
            "subset": "O",
            "split": "development",
            "split_group_id": f"o-{index}",
        }
        for index in range(974)
    )
    records.extend(
        {
            "path": f"development/conflict-N/{index}.tif",
            "subset": "N",
            "split": "development",
            "split_group_id": "cross-label-conflict",
        }
        for index in range(2)
    )
    records.extend(
        {
            "path": f"development/conflict-O/{index}.tif",
            "subset": "O",
            "split": "development",
            "split_group_id": "cross-label-conflict",
        }
        for index in range(12)
    )
    return pd.DataFrame(records)


def test_reference_banks_are_selected_by_rank_and_exact_frozen_ids(tmp_path: Path) -> None:
    path = tmp_path / "rvl_selected_extracted.csv"
    frame = _reference_manifest()
    frame.to_csv(path, index=False)
    expected_ids = {
        f"images/train/{label}/{rank}.tif" for label in range(16) for rank in range(500)
    }

    bank_32k, bank_8k = select_frozen_reference_manifest(
        path,
        expected_8k_ids=sorted(expected_ids),
    )

    assert len(bank_32k) == 32_000
    assert len(bank_8k) == 8_000
    assert set(bank_8k["relative_path"]) == expected_ids
    assert bank_32k.groupby("label").size().to_dict() == {
        label: 2_000 for label in range(16)
    }
    assert bank_8k.groupby("label").size().to_dict() == {label: 500 for label in range(16)}
    # With label-major manifest order, the first 8k rows are not the frozen balanced bank.
    assert set(frame.iloc[:8_000]["relative_path"]) != expected_ids


def test_reference_selection_rejects_changed_frozen_8k_ids(tmp_path: Path) -> None:
    path = tmp_path / "rvl_selected_extracted.csv"
    frame = _reference_manifest()
    frame.loc[0, "minimum_reference_size"] = 32_000
    frame.to_csv(path, index=False)
    with pytest.raises(LayerCacheBoundaryError, match="exactly match"):
        select_frozen_reference_manifest(path)


def test_only_development_only_filename_is_allowed_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong_path = tmp_path / "research_split.csv"
    wrong_path.write_text("must not be opened", encoding="utf-8")
    opened = False

    def forbidden_read(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("unsafe split was opened")

    monkeypatch.setattr(layer_cache.pd, "read_csv", forbidden_read)
    with pytest.raises(LayerCacheBoundaryError, match="Only research_split_development_only"):
        load_development_only_manifest(wrong_path)
    assert not opened


def test_development_manifest_filters_conflicts_to_frozen_counts(tmp_path: Path) -> None:
    path = tmp_path / "research_split_development_only.csv"
    _development_manifest().to_csv(path, index=False)
    filtered, excluded = load_development_only_manifest(path)
    assert excluded == 14
    assert filtered["subset"].value_counts().to_dict() == {"O": 974, "N": 306}
    assert len(filtered) == 1_280


def test_development_manifest_beneath_sealed_path_is_refused_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sealed" / "research_split_development_only.csv"
    opened = False

    def forbidden_read(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("sealed manifest was opened")

    monkeypatch.setattr(layer_cache.pd, "read_csv", forbidden_read)
    with pytest.raises(LayerCacheBoundaryError, match="sealed path"):
        load_development_only_manifest(path)
    assert not opened


class _FakeBeit(nn.Module):
    def __init__(self, *, pooler_offset: float = 0.0, oom_above: int | None = None) -> None:
        super().__init__()
        self.pooler = nn.Module()
        self.pooler.layernorm = nn.LayerNorm(768)
        self.pooler_offset = pooler_offset
        self.oom_above = oom_above

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert output_hidden_states
        assert return_dict
        batch = len(pixel_values)
        if self.oom_above is not None and batch > self.oom_above:
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        feature = torch.arange(768, device=pixel_values.device, dtype=torch.float32)
        feature = feature.view(1, 1, 768) / 768.0
        positions = torch.arange(197, device=pixel_values.device, dtype=torch.float32)
        positions = positions.view(1, 197, 1) / 1_000.0
        hidden_states = []
        for layer_index in range(13):
            hidden = feature + positions + float(layer_index)
            hidden_states.append(hidden.expand(batch, -1, -1))
        raw_mean = hidden_states[12][:, 1:, :].mean(dim=1)
        pooler_output = self.pooler.layernorm(raw_mean) + self.pooler_offset
        return SimpleNamespace(
            hidden_states=tuple(hidden_states),
            pooler_output=pooler_output,
        )


class _FakeModel(nn.Module):
    def __init__(self, *, pooler_offset: float = 0.0, oom_above: int | None = None) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=768, num_hidden_layers=12, num_labels=16)
        self.beit = _FakeBeit(pooler_offset=pooler_offset, oom_above=oom_above)
        self.classifier = nn.Linear(768, 16)


class _FakeProcessor:
    def __call__(self, *, images, return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros(len(images), 3, 4, 4, dtype=torch.float32)}


class _FakeImage:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _small_plan(tmp_path: Path, *, sealed: bool = False, ref_rows: int = 2) -> LayerCachePlan:
    reference_ids = tuple(f"reference-{index}" for index in range(ref_rows))
    reference_dir = tmp_path / ("sealed" if sealed else "reference")
    reference_paths = tuple(reference_dir / f"{index}.tif" for index in range(ref_rows))
    development_ids = ("development-N", "development-O")
    return LayerCachePlan(
        reference_ids=reference_ids,
        reference_paths=reference_paths,
        reference_labels=tuple(index % 16 for index in range(ref_rows)),
        reference_8k_ids=frozenset({reference_ids[0]}),
        reference_8k_indices=(0,),
        development_ids=development_ids,
        development_paths=(tmp_path / "development" / "N.tif", tmp_path / "development" / "O.tif"),
        development_subsets=("N", "O"),
        excluded_cross_label_conflicts=0,
        reference_manifest_path=tmp_path / "reference.csv",
        development_manifest_path=tmp_path / "research_split_development_only.csv",
    )


def _run_small_cache(
    plan: LayerCachePlan,
    output_dir: Path,
    *,
    model: nn.Module,
    batch_size: int = 2,
    loader=None,
) -> dict:
    if loader is None:
        def default_loader(_path: Path) -> _FakeImage:
            return _FakeImage()

        loader = default_loader
    return layer_cache._extract_layer_feature_cache(
        plan,
        output_dir,
        model=model,
        processor=_FakeProcessor(),
        device="cpu",
        batch_size=batch_size,
        image_loader=loader,
        strict_counts=False,
        require_cuda=False,
    )


def test_extraction_writes_float32_row_major_memmaps_and_g1_metadata(tmp_path: Path) -> None:
    plan = _small_plan(tmp_path)
    output_dir = tmp_path / "cache"
    opened: list[Path] = []
    images: list[_FakeImage] = []

    def loader(path: Path) -> _FakeImage:
        opened.append(path)
        image = _FakeImage()
        images.append(image)
        return image

    metadata = _run_small_cache(plan, output_dir, model=_FakeModel(), loader=loader)

    assert opened == [*plan.reference_paths, *plan.development_paths]
    assert all(image.closed for image in images)
    for split, rows in (("reference", 2), ("development", 2)):
        for representation in ("raw_patch_mean", "cls"):
            values = np.load(output_dir / f"{split}_{representation}.npy", mmap_mode="r")
            assert values.shape == (rows, 12, 768)
            assert values.dtype == np.float32
        logits = np.load(output_dir / f"{split}_logits.npy", mmap_mode="r")
        assert logits.shape == (rows, 16)
        assert logits.dtype == np.float32
    assert metadata["layout"] == "[rows, 12, 768]"
    assert metadata["completed_rows"] == {"reference": 2, "development": 2}
    assert metadata["g1"]["passed"] is True
    assert metadata["g1"]["max_abs_error"] == 0.0
    assert metadata["pooler_layernorm_eps"] == _FakeModel().beit.pooler.layernorm.eps
    assert metadata["sealed"] == {"loaded": 0, "scored": 0}
    assert load_layer_cache_metadata(output_dir) == metadata


def test_full_hidden_state_shape_is_refused_by_cpu_copy_guard() -> None:
    with pytest.raises(LayerCacheGateError, match="Only pooled"):
        layer_cache._copy_pooled_vectors_to_cpu(
            torch.zeros(2, 197, 768),
            batch_rows=2,
        )


def test_g1_failure_stops_before_any_completed_row(tmp_path: Path) -> None:
    plan = _small_plan(tmp_path)
    output_dir = tmp_path / "cache"
    with pytest.raises(LayerCacheGateError, match="G1 failed"):
        _run_small_cache(plan, output_dir, model=_FakeModel(pooler_offset=0.1))
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_rows"] == {"reference": 0, "development": 0}
    assert progress["g1"]["passed"] is False


def test_sealed_image_path_is_refused_before_loader(tmp_path: Path) -> None:
    plan = _small_plan(tmp_path, sealed=True)
    opened = False

    def loader(_path: Path) -> _FakeImage:
        nonlocal opened
        opened = True
        return _FakeImage()

    with pytest.raises(LayerCacheBoundaryError, match="Refused sealed image path"):
        _run_small_cache(plan, tmp_path / "cache", model=_FakeModel(), loader=loader)
    assert not opened


def test_production_path_preflight_rejects_missing_file_without_opening(tmp_path: Path) -> None:
    plan = _small_plan(tmp_path)
    with pytest.raises(LayerCacheBoundaryError, match="missing or not a file"):
        layer_cache._validate_production_paths(plan)


def test_oom_halves_batch_without_changing_rows_or_dtype(tmp_path: Path) -> None:
    plan = _small_plan(tmp_path, ref_rows=3)
    output_dir = tmp_path / "cache"
    metadata = _run_small_cache(
        plan,
        output_dir,
        model=_FakeModel(oom_above=1),
        batch_size=2,
    )
    assert metadata["effective_batch_size"] == 1
    assert metadata["completed_rows"] == {"reference": 3, "development": 2}
    assert np.load(output_dir / "reference_cls.npy", mmap_mode="r").dtype == np.float32


def test_resume_skips_already_flushed_rows(tmp_path: Path) -> None:
    plan = _small_plan(tmp_path, ref_rows=3)
    output_dir = tmp_path / "cache"
    calls = 0

    class FailsAfterFirstBatch(_FakeBeit):
        def forward(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic interruption")
            return super().forward(*args, **kwargs)

    interrupted_model = _FakeModel()
    interrupted_model.beit = FailsAfterFirstBatch()
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _run_small_cache(plan, output_dir, model=interrupted_model, batch_size=2)
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_rows"] == {"reference": 2, "development": 0}

    reopened: list[Path] = []
    resumed_model = _FakeModel()
    resumed_model.load_state_dict(interrupted_model.state_dict())
    metadata = _run_small_cache(
        plan,
        output_dir,
        model=resumed_model,
        batch_size=2,
        loader=lambda path: reopened.append(path) or _FakeImage(),
    )
    assert plan.reference_paths[0] not in reopened
    assert plan.reference_paths[1] not in reopened
    assert reopened == [plan.reference_paths[2], *plan.development_paths]
    assert metadata["complete"] is True
