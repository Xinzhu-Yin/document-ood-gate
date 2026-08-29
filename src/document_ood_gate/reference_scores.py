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
        mean_precision = self.means_ @ self.precision_
        mean_quadratic = np.einsum("cd,cd->c", mean_precision, self.means_)
        outputs: list[np.ndarray] = []
        for start in range(0, len(q), batch_size):
            batch = q[start : start + batch_size]
            batch_precision = batch @ self.precision_
            batch_quadratic = np.einsum("bd,bd->b", batch_precision, batch)
            squared = (
                batch_quadratic[:, None]
                - 2.0 * (batch_precision @ self.means_.T)
                + mean_quadratic[None, :]
            )
            outputs.append(-squared.min(axis=1))
        return np.concatenate(outputs)


class ShrinkageRelativeMahalanobis:
    """Relative Mahalanobis with shrinkage on both class and background models.

    The class-conditional side exactly matches ``ShrinkageMahalanobis``.  The
    background Gaussian is fit to the same reference bank while ignoring class.
    Higher returned values mean more ID-like:

        background_distance - minimum_class_distance
    """

    def __init__(self) -> None:
        self.means_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.background_mean_: np.ndarray | None = None
        self.background_precision_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "ShrinkageRelativeMahalanobis":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("features must be 2D and aligned with labels")
        if not len(x) or not np.isfinite(x).all():
            raise ValueError("features must be finite and non-empty")
        classes = np.unique(y)
        means = np.stack([x[y == cls].mean(axis=0) for cls in classes])
        mean_lookup = {cls: means[index] for index, cls in enumerate(classes)}
        class_centered = x - np.stack([mean_lookup[value] for value in y])
        class_covariance = LedoitWolf(assume_centered=True).fit(class_centered)

        background_mean = x.mean(axis=0)
        background_centered = x - background_mean
        background_covariance = LedoitWolf(assume_centered=True).fit(background_centered)
        self.means_ = means
        self.precision_ = class_covariance.precision_
        self.background_mean_ = background_mean
        self.background_precision_ = background_covariance.precision_
        return self

    def normality(self, queries: np.ndarray, batch_size: int = 256) -> np.ndarray:
        if any(
            value is None
            for value in (
                self.means_,
                self.precision_,
                self.background_mean_,
                self.background_precision_,
            )
        ):
            raise RuntimeError("fit must be called before normality")
        means = np.asarray(self.means_)
        precision = np.asarray(self.precision_)
        background_mean = np.asarray(self.background_mean_)
        background_precision = np.asarray(self.background_precision_)
        q = np.asarray(queries, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != means.shape[1] or not np.isfinite(q).all():
            raise ValueError("queries must be a finite 2D array with the fitted dimension")
        mean_precision = means @ precision
        mean_quadratic = np.einsum("cd,cd->c", mean_precision, means)
        outputs: list[np.ndarray] = []
        for start in range(0, len(q), batch_size):
            batch = q[start : start + batch_size]
            batch_precision = batch @ precision
            batch_quadratic = np.einsum("bd,bd->b", batch_precision, batch)
            class_squared = (
                batch_quadratic[:, None]
                - 2.0 * (batch_precision @ means.T)
                + mean_quadratic[None, :]
            )
            background_delta = batch - background_mean
            background_squared = np.einsum(
                "bi,ij,bj->b",
                background_delta,
                background_precision,
                background_delta,
                optimize=True,
            )
            outputs.append(background_squared - class_squared.min(axis=1))
        return np.concatenate(outputs)
