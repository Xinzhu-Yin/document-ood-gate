from __future__ import annotations

import numpy as np
from sklearn.covariance import EmpiricalCovariance

from .scores import softmax


def _validate_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("logits must have shape [n, classes] with at least two classes")
    if not np.isfinite(values).all():
        raise ValueError("logits contain NaN or infinity")
    return values


def logsumexp_normality(logits: np.ndarray) -> np.ndarray:
    """Temperature-one Energy orientation: higher means more ID-like."""

    values = _validate_logits(logits)
    maximum = values.max(axis=1)
    return maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))


def generalized_entropy_normality(
    logits: np.ndarray,
    *,
    gamma: float = 0.1,
    top_m: int = 100,
) -> np.ndarray:
    """Official GEN score, oriented so higher means more ID-like.

    The paper's default ``M=100`` is capped at the available class count.  For
    this 16-class checkpoint that means all 16 probabilities participate.
    """

    if not 0.0 < gamma < 1.0:
        raise ValueError("GEN gamma must be strictly between zero and one")
    if top_m < 1:
        raise ValueError("GEN top_m must be positive")
    probabilities = softmax(_validate_logits(logits))
    effective_m = min(int(top_m), probabilities.shape[1])
    selected = np.sort(probabilities, axis=1)[:, -effective_m:]
    generalized_entropy = np.sum(
        np.power(selected, gamma) * np.power(1.0 - selected, gamma),
        axis=1,
    )
    return -generalized_entropy


def linear_classifier_logits(
    features: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    if x.ndim != 2 or w.ndim != 2 or b.shape != (w.shape[0],):
        raise ValueError("Invalid feature or classifier parameter shapes")
    if x.shape[1] != w.shape[1]:
        raise ValueError("Feature dimension does not match classifier weight")
    return x @ w.T + b


def react_clip_threshold(reference_features: np.ndarray, *, percentile: float = 90.0) -> float:
    values = np.asarray(reference_features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("reference_features must be a finite 2D array")
    if not 0.0 < percentile < 100.0:
        raise ValueError("ReAct percentile must be strictly between 0 and 100")
    return float(np.percentile(values, percentile))


def react_energy_normality(
    features: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    clip_threshold: float,
) -> np.ndarray:
    if not np.isfinite(clip_threshold):
        raise ValueError("ReAct clip threshold must be finite")
    clipped = np.minimum(np.asarray(features, dtype=np.float64), clip_threshold)
    return logsumexp_normality(linear_classifier_logits(clipped, weight, bias))


class ViMNormality:
    """Official ViM residual/virtual-logit score with deterministic eigensolver."""

    def __init__(self, principal_dim: int = 512) -> None:
        self.principal_dim = int(principal_dim)
        self.origin_: np.ndarray | None = None
        self.null_space_: np.ndarray | None = None
        self.alpha_: float | None = None
        self.eigenvalues_: np.ndarray | None = None

    def fit(
        self,
        reference_features: np.ndarray,
        reference_logits: np.ndarray,
        classifier_weight: np.ndarray,
        classifier_bias: np.ndarray,
    ) -> "ViMNormality":
        features = np.asarray(reference_features, dtype=np.float64)
        logits = _validate_logits(reference_logits)
        weight = np.asarray(classifier_weight, dtype=np.float64)
        bias = np.asarray(classifier_bias, dtype=np.float64)
        if features.ndim != 2 or len(features) != len(logits):
            raise ValueError("Reference features and logits must be aligned 2D arrays")
        if weight.shape != (logits.shape[1], features.shape[1]):
            raise ValueError("Classifier weight does not match features/logits")
        if bias.shape != (logits.shape[1],):
            raise ValueError("Classifier bias does not match logits")
        if not 0 < self.principal_dim < features.shape[1]:
            raise ValueError("ViM principal_dim must be between zero and feature dimension")

        origin = -np.linalg.pinv(weight) @ bias
        shifted = features - origin
        covariance = EmpiricalCovariance(assume_centered=True).fit(shifted).covariance_
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        residual_dimensions = features.shape[1] - self.principal_dim
        null_space = np.ascontiguousarray(eigenvectors[:, :residual_dimensions])
        residual_norm = np.linalg.norm(shifted @ null_space, axis=1)
        mean_residual = float(residual_norm.mean())
        if mean_residual <= 0.0:
            raise ValueError("ViM reference residual norm must be positive")
        alpha = float(logits.max(axis=1).mean() / mean_residual)
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("ViM alpha must be finite and positive")

        self.origin_ = origin
        self.null_space_ = null_space
        self.alpha_ = alpha
        self.eigenvalues_ = eigenvalues
        return self

    def normality(self, features: np.ndarray, logits: np.ndarray) -> np.ndarray:
        if self.origin_ is None or self.null_space_ is None or self.alpha_ is None:
            raise RuntimeError("fit must be called before normality")
        values = np.asarray(features, dtype=np.float64)
        logit_values = _validate_logits(logits)
        if values.ndim != 2 or len(values) != len(logit_values):
            raise ValueError("Features and logits must be aligned 2D arrays")
        residual = np.linalg.norm((values - self.origin_) @ self.null_space_, axis=1)
        return logsumexp_normality(logit_values) - self.alpha_ * residual
