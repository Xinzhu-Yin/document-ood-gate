import numpy as np

from document_ood_gate.reference_scores import ShrinkageMahalanobis, knn_cosine_normality


def test_knn_matches_brute_force_and_ignores_positive_scale() -> None:
    reference = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0], [0.2, 0.8]], dtype=np.float32)
    actual = knn_cosine_normality(reference, queries, k=2, device="cpu")

    ref_norm = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    query_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    expected = np.sort(query_norm @ ref_norm.T, axis=1)[:, -2]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    scaled = knn_cosine_normality(reference * 3.0, queries * 7.0, k=2, device="cpu")
    np.testing.assert_allclose(actual, scaled, atol=1e-6)


def test_mahalanobis_prefers_class_centres_and_is_translation_invariant() -> None:
    features = np.array(
        [[-1.1, 0.0], [-0.9, 0.1], [0.9, -0.1], [1.1, 0.0]], dtype=np.float64
    )
    labels = np.array([0, 0, 1, 1])
    queries = np.array([[-1.0, 0.05], [0.0, 3.0]])
    base = ShrinkageMahalanobis().fit(features, labels).normality(queries)
    shifted = (
        ShrinkageMahalanobis()
        .fit(features + 4.0, labels)
        .normality(queries + 4.0)
    )
    assert np.isfinite(base).all()
    assert base[0] > base[1]
    np.testing.assert_allclose(base, shifted, rtol=1e-8, atol=1e-8)
