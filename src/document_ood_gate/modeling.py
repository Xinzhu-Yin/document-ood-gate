from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_REVISION = "23f8b03d130fb66bbc9a15df3c75d753e49240eb"


@dataclass(frozen=True)
class BatchOutput:
    logits: np.ndarray
    features: np.ndarray


class FrozenDiT:
    def __init__(
        self,
        model_id: str = "microsoft/dit-base-finetuned-rvlcdip",
        revision: str = MODEL_REVISION,
        device: str = "auto",
        dtype: str = "float32",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # Match the original PIL/BILINEAR processor path rather than a newer fast path.
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            revision=revision,
            use_fast=False,
        )
        self.model = AutoModelForImageClassification.from_pretrained(model_id, revision=revision)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.compute_dtype = torch.float16 if dtype == "float16" and self.device.type == "cuda" else torch.float32

        if not hasattr(self.model, "beit") or not hasattr(self.model, "classifier"):
            raise TypeError(f"Expected a BEiT/DiT classifier, got {type(self.model).__name__}")

    @torch.inference_mode()
    def infer_images(self, images: list[Image.Image]) -> BatchOutput:
        encoded = self.processor(images=images, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(self.device, dtype=self.compute_dtype)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.compute_dtype,
            enabled=self.compute_dtype == torch.float16,
        ):
            base_output = self.model.beit(pixel_values=pixel_values, return_dict=True)
            features = base_output.pooler_output
            logits_from_features = self.model.classifier(features)
            direct_logits = self.model(pixel_values=pixel_values, return_dict=True).logits
        max_error = (direct_logits.float() - logits_from_features.float()).abs().max().item()
        tolerance = 1e-5 if self.compute_dtype == torch.float32 else 1e-3
        if max_error > tolerance:
            raise RuntimeError(f"Classifier-input feature failed logit equivalence: max error {max_error}")
        return BatchOutput(
            logits=direct_logits.float().cpu().numpy(),
            features=features.float().cpu().numpy(),
        )

    def infer_paths(self, paths: Iterable[Path], batch_size: int = 4) -> BatchOutput:
        path_list = [Path(p) for p in paths]
        logits_parts: list[np.ndarray] = []
        feature_parts: list[np.ndarray] = []
        for start in range(0, len(path_list), batch_size):
            images: list[Image.Image] = []
            for path in path_list[start : start + batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            output = self.infer_images(images)
            logits_parts.append(output.logits)
            feature_parts.append(output.features)
        return BatchOutput(
            logits=np.concatenate(logits_parts, axis=0),
            features=np.concatenate(feature_parts, axis=0),
        )
