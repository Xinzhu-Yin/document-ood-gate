from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def output_normality_scores(logits: np.ndarray) -> dict[str, np.ndarray]:
    """Return scalar scores oriented so that higher means more normal/ID-like."""

    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 16:
        raise ValueError(f"Expected [n, 16] logits, got {x.shape}")
    probs = softmax(x)
    sorted_logits = np.sort(x, axis=1)
    max_logits = sorted_logits[:, -1]
    margin = sorted_logits[:, -1] - sorted_logits[:, -2]
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    max_x = np.max(x, axis=1)
    logsumexp = max_x + np.log(np.exp(x - max_x[:, None]).sum(axis=1))
    return {
        "msp": probs.max(axis=1),
        # The paper's raw energy is -logsumexp at T=1. We negate it so higher is ID.
        "energy": logsumexp,
        "max_logit": max_logits,
        "negative_entropy": -entropy,
        "logit_margin": margin,
    }


def feature_norm_normality(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be 2D")
    return np.linalg.norm(x, axis=1)

