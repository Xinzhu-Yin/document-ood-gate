from __future__ import annotations

import numpy as np


def _feature_matrix(
    values: np.ndarray,
    *,
    name: str,
    minimum_rows: int,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D feature matrix")
    if matrix.shape[0] < minimum_rows:
        raise ValueError(f"{name} must contain at least {minimum_rows} rows")
    if matrix.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two feature dimensions")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


class MeanCenteredPCAResidual:
    """PCA residual normality around the reference-feature mean.

    The largest ``principal_dim`` covariance eigenvectors define the principal
    subspace. The remaining, smallest-eigenvalue directions define the residual
    subspace. Higher returned scores mean more reference-like because normality
    is the negative residual norm.

    This estimator deliberately uses the reference mean as its origin. It is
    distinct from ViM's classifier-derived origin and calibrated residual.
    """

    def __init__(self, principal_dim: int = 512) -> None:
        if isinstance(principal_dim, bool) or not isinstance(
            principal_dim, (int, np.integer)
        ):
            raise ValueError("principal_dim must be an integer")
        if int(principal_dim) < 1:
            raise ValueError("principal_dim must be positive")
        self.principal_dim = int(principal_dim)
        self.reference_mean_: np.ndarray | None = None
        self.residual_space_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.top_d_explained_variance_ratio_: float | None = None
        self.feature_dim_: int | None = None

    def fit(self, reference_features: np.ndarray) -> "MeanCenteredPCAResidual":
        reference = _feature_matrix(
            reference_features,
            name="reference_features",
            minimum_rows=2,
        )
        feature_dim = int(reference.shape[1])
        if self.principal_dim >= feature_dim:
            raise ValueError("principal_dim must be smaller than the feature dimension")

        reference_mean = reference.mean(axis=0)
        centered = reference - reference_mean
        covariance = (centered.T @ centered) / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        # Symmetric eigensolvers can return tiny negative values for a PSD matrix.
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        total_variance = float(eigenvalues.sum())
        if not np.isfinite(total_variance) or total_variance <= 0.0:
            raise ValueError("reference_features must have positive total variance")

        residual_dim = feature_dim - self.principal_dim
        residual_space = np.ascontiguousarray(eigenvectors[:, :residual_dim])
        explained = float(eigenvalues[-self.principal_dim :].sum() / total_variance)

        self.reference_mean_ = reference_mean
        self.residual_space_ = residual_space
        self.eigenvalues_ = eigenvalues
        self.top_d_explained_variance_ratio_ = explained
        self.feature_dim_ = feature_dim
        return self

    def residual_norm(self, features: np.ndarray) -> np.ndarray:
        if (
            self.reference_mean_ is None
            or self.residual_space_ is None
            or self.feature_dim_ is None
        ):
            raise RuntimeError("fit must be called before residual_norm")
        values = _feature_matrix(features, name="features", minimum_rows=1)
        if values.shape[1] != self.feature_dim_:
            raise ValueError("features do not match the fitted feature dimension")
        centered = values - self.reference_mean_
        return np.linalg.norm(centered @ self.residual_space_, axis=1)

    def normality(self, features: np.ndarray) -> np.ndarray:
        """Return negative residual norm so higher scores are more reference-like."""

        return -self.residual_norm(features)
