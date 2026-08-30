from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from document_ood_gate.layer_study import (
    LayerStudyProtocolError,
    reconstruct_h_doc,
    validate_frozen_config,
)


def test_reconstruct_h_doc_matches_torch_layer_norm() -> None:
    rng = np.random.default_rng(20260812)
    raw = rng.normal(size=(7, 768)).astype(np.float32)
    weight = rng.normal(size=768).astype(np.float32)
    bias = rng.normal(size=768).astype(np.float32)
    expected = torch.nn.functional.layer_norm(
        torch.from_numpy(raw),
        (768,),
        torch.from_numpy(weight),
        torch.from_numpy(bias),
        eps=1.0e-12,
    ).numpy()
    observed = reconstruct_h_doc(raw, weight, bias, eps=1.0e-12, chunk_size=3)
    np.testing.assert_allclose(observed, expected, atol=2.0e-6, rtol=2.0e-6)
    assert observed.dtype == np.float32


def test_validate_frozen_config_accepts_repository_config() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = validate_frozen_config(repository / "configs" / "layer_scan.yaml")
    assert config["study"]["sealed_test_access"] == "forbidden"
    assert config["data"]["one_pass_images"] == 33280


def test_validate_frozen_config_rejects_tuning(tmp_path: Path) -> None:
    path = tmp_path / "layer_scan.yaml"
    repository = Path(__file__).resolve().parents[1]
    text = (repository / "configs" / "layer_scan.yaml").read_text(encoding="utf-8")
    path.write_text(
        text.replace("hyperparameter_tuning: false", "hyperparameter_tuning: true"),
        encoding="utf-8",
    )
    with pytest.raises(LayerStudyProtocolError, match="hyperparameter_tuning"):
        validate_frozen_config(path)
