import numpy as np
import pytest

from document_ood_gate.layer_scores import MeanCenteredPCAResidual


def _axis_aligned_reference() -> np.ndarray:
    return np.array(
        [
            [-3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_mean_centered_residual_is_translation_invariant() -> None:
    reference = _axis_aligned_reference()
    queries = np.array([[1.5, -0.5, 0.25], [-2.0, 1.0, 3.0]])
    translation = np.array([17.0, -23.0, 101.0])

    original = MeanCenteredPCAResidual(principal_dim=2).fit(reference)
    translated = MeanCenteredPCAResidual(principal_dim=2).fit(reference + translation)

    np.testing.assert_allclose(
        original.normality(queries),
        translated.normality(queries + translation),
        atol=1e-12,
    )


def test_normality_is_negative_residual_norm_and_higher_is_more_normal() -> None:
    estimator = MeanCenteredPCAResidual(principal_dim=2).fit(_axis_aligned_reference())
    along_principal_space = np.array([[10.0, -4.0, 0.0]])
    displaced_in_residual_space = np.array([[10.0, -4.0, 2.5]])

    scores = estimator.normality(
        np.concatenate([along_principal_space, displaced_in_residual_space])
    )

    np.testing.assert_allclose(scores, np.array([0.0, -2.5]), atol=1e-12)
    assert scores[0] > scores[1]


def test_residual_dimension_and_top_d_explained_variance_ratio() -> None:
    reference = _axis_aligned_reference()
    estimator = MeanCenteredPCAResidual(principal_dim=2).fit(reference)

    assert estimator.residual_space_.shape == (3, 1)
    np.testing.assert_allclose(
        estimator.residual_space_.T @ estimator.residual_space_,
        np.eye(1),
        atol=1e-12,
    )
    expected_ratio = (3.0 + 4.0 / 3.0) / (3.0 + 4.0 / 3.0 + 1.0 / 3.0)
    assert estimator.top_d_explained_variance_ratio_ == pytest.approx(expected_ratio)


def test_input_validation_rejects_invalid_dimensions_and_values() -> None:
    with pytest.raises(ValueError, match="integer"):
        MeanCenteredPCAResidual(principal_dim=1.5)
    with pytest.raises(ValueError, match="positive"):
        MeanCenteredPCAResidual(principal_dim=0)
    with pytest.raises(ValueError, match="smaller"):
        MeanCenteredPCAResidual(principal_dim=3).fit(_axis_aligned_reference())
    with pytest.raises(ValueError, match="2D"):
        MeanCenteredPCAResidual(principal_dim=1).fit(np.ones(4))
    with pytest.raises(ValueError, match="finite"):
        MeanCenteredPCAResidual(principal_dim=1).fit(
            np.array([[0.0, 1.0], [np.nan, 2.0]])
        )
    with pytest.raises(ValueError, match="positive total variance"):
        MeanCenteredPCAResidual(principal_dim=1).fit(np.ones((3, 2)))


def test_scoring_requires_a_fitted_estimator_and_matching_finite_features() -> None:
    estimator = MeanCenteredPCAResidual(principal_dim=2)
    with pytest.raises(RuntimeError, match="fit must be called"):
        estimator.normality(np.zeros((1, 3)))

    estimator.fit(_axis_aligned_reference())
    with pytest.raises(ValueError, match="fitted feature dimension"):
        estimator.normality(np.zeros((1, 4)))
    with pytest.raises(ValueError, match="finite"):
        estimator.normality(np.array([[0.0, 0.0, np.inf]]))
    with pytest.raises(ValueError, match="at least 1 rows"):
        estimator.normality(np.empty((0, 3)))
