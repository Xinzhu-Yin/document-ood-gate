# Pre-specified replication predictions committed before sealed-test access

These are replication predictions, not blind predictions. Each is derived from the development
results and states what we expect to hold on independent held-out data. They are committed before
any sealed access so that the record of what we expected provably precedes what we observed.

## P1 — Early-layer weakness

For raw patch mean, at each of Layers 2–5, all three primary probes are predicted to have sealed
AUROC < 0.60. The three probes are Mahalanobis-32k, kNN-32k (`k=10`), and mean-centered PCA
Residual-32k (D=512).

## P2 — Pure normalization effect

At Layer 12, LN-free normalization is predicted to improve Mahalanobis AUROC by at least +0.05
over raw mean. The primary comparison is:

`L12 LN-free Mahalanobis AUROC - L12 raw Mahalanobis AUROC`.

## P3 — h_doc Mahalanobis

Sealed h_doc Mahalanobis-32k AUROC is predicted to fall within 0.71–0.78. This is a
pre-specified prediction range, not a statistical prediction interval.

## P4 — Deployment limitation

The best pre-specified sealed configuration is predicted to retain FPR95 > 0.75. “Best” means
the configuration with the highest sealed AUROC in the frozen D1 + D2 set defined in
`sealed_test_protocol.md`.

## P5 — AUROC / operating-point mismatch

The configuration with the highest sealed AUROC is predicted not to simultaneously achieve the
lowest sealed FPR95 in the same frozen D1 + D2 set.

After sealed evaluation, each prediction will be recorded only as `supported` or
`not supported`; the original predictions and thresholds will not be modified.
