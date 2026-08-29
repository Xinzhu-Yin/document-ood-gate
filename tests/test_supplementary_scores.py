import numpy as np

from document_ood_gate.supplementary_scores import (
    ViMNormality,
    generalized_entropy_normality,
    linear_classifier_logits,
    logsumexp_normality,
    react_clip_threshold,
    react_energy_normality,
)


def test_gen_matches_official_formula_and_is_shift_invariant() -> None:
    logits = np.array([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    shifted = logits + 17.0
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    expected = -np.sum(probabilities**0.1 * (1.0 - probabilities) ** 0.1, axis=1)
    actual = generalized_entropy_normality(logits, gamma=0.1, top_m=100)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(
        actual,
        generalized_entropy_normality(shifted, gamma=0.1, top_m=100),
    )
    assert actual[0] > actual[1]


def test_react_clips_only_the_upper_tail_before_the_linear_head() -> None:
    reference = np.array([[-2.0, 0.0], [1.0, 3.0]])
    assert react_clip_threshold(reference, percentile=75.0) == 1.5
    features = np.array([[-2.0, 3.0]])
    weight = np.eye(2)
    bias = np.array([0.5, -0.5])
    expected_logits = np.array([[-1.5, 1.0]])
    expected = logsumexp_normality(expected_logits)
    actual = react_energy_normality(
        features,
        weight,
        bias,
        clip_threshold=1.5,
    )
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(
        linear_classifier_logits(features, weight, bias),
        np.array([[-1.5, 2.5]]),
    )


def test_vim_fits_official_origin_residual_and_normality_formula() -> None:
    rng = np.random.default_rng(23)
    features = rng.normal(size=(80, 6))
    weight = rng.normal(size=(3, 6))
    bias = np.array([0.4, -0.2, 0.1])
    logits = linear_classifier_logits(features, weight, bias)
    estimator = ViMNormality(principal_dim=3).fit(features, logits, weight, bias)
    assert estimator.null_space_.shape == (6, 3)
    np.testing.assert_allclose(
        estimator.null_space_.T @ estimator.null_space_, np.eye(3), atol=1e-12
    )
    np.testing.assert_allclose(weight @ estimator.origin_ + bias, 0.0, atol=1e-10)

    queries = rng.normal(size=(5, 6))
    query_logits = linear_classifier_logits(queries, weight, bias)
    residual = np.linalg.norm(
        (queries - estimator.origin_) @ estimator.null_space_, axis=1
    )
    expected = logsumexp_normality(query_logits) - estimator.alpha_ * residual
    np.testing.assert_allclose(estimator.normality(queries, query_logits), expected)
