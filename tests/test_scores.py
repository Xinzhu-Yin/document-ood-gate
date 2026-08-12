import numpy as np

from document_ood_gate.scores import output_normality_scores, softmax


def test_softmax_rows_sum_to_one() -> None:
    logits = np.zeros((3, 16), dtype=np.float32)
    probabilities = softmax(logits)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_confident_logits_raise_normality_scores() -> None:
    flat = np.zeros(16)
    confident = np.zeros(16)
    confident[0] = 10.0
    scores = output_normality_scores(np.stack([flat, confident]))
    assert scores["msp"][1] > scores["msp"][0]
    assert scores["energy"][1] > scores["energy"][0]
    assert scores["negative_entropy"][1] > scores["negative_entropy"][0]
    assert scores["logit_margin"][1] > scores["logit_margin"][0]

