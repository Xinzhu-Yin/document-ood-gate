from __future__ import annotations

import numpy as np
import torch
from sklearn.covariance import LedoitWolf


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def knn_cosine_normality(
    reference: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    batch_size: int = 128,
    device: str = "auto",
) -> np.ndarray:
    """k-th-nearest cosine similarity; higher means closer to the normal reference bank."""

    if k < 1 or k > len(reference):
        raise ValueError("k must be between 1 and the reference-bank size")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ref = torch.from_numpy(_l2_normalize(reference)).to(device)
    query_np = _l2_normalize(queries)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(query_np), batch_size):
            query = torch.from_numpy(query_np[start : start + batch_size]).to(device)
            similarity = query @ ref.T
            kth = torch.topk(similarity, k=k, dim=1).values[:, -1]
            outputs.append(kth.cpu().numpy())
    return np.concatenate(outputs)


class ShrinkageMahalanobis:
    """Class-conditional means with one Ledoit-Wolf shared covariance."""

    def __init__(self) -> None:
        self.means_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "ShrinkageMahalanobis":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels)
        classes = np.unique(y)
        means = np.stack([x[y == cls].mean(axis=0) for cls in classes])
        mean_lookup = {cls: means[i] for i, cls in enumerate(classes)}
        centered = x - np.stack([mean_lookup[value] for value in y])
        covariance = LedoitWolf(assume_centered=True).fit(centered)
        self.means_ = means
        self.precision_ = covariance.precision_
        return self

    def normality(self, queries: np.ndarray, batch_size: int = 256) -> np.ndarray:
        if self.means_ is None or self.precision_ is None:
            raise RuntimeError("fit must be called before normality")
        q = np.asarray(queries, dtype=np.float64)
        outputs: list[np.ndarray] = []
        for start in range(0, len(q), batch_size):
            delta = q[start : start + batch_size, None, :] - self.means_[None, :, :]
            squared = np.einsum("bcd,de,bce->bc", delta, self.precision_, delta, optimize=True)
            outputs.append(-squared.min(axis=1))
        return np.concatenate(outputs)
