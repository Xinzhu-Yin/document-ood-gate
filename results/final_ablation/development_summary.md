# Final development ablations

## Protocol boundary

- 32k reference and 306 N + 974 O development caches were reused; DiT was not rerun.
- All AUROC/FPR95 intervals use 2,000 bootstrap resamples and seed `20260812`.
- Residual D sensitivity is diagnostic only; the frozen primary/sealed value remains D=512.
- Sealed rows loaded/scored during A+B: **0 / 0**.

## Layer 12 decomposition

| Representation | Method | AUROC | FPR95 |
|---|---|---:|---:|
| Raw | Mahalanobis | 0.6567 | 0.9579 |
| Raw | kNN (k=10) | 0.7336 | 0.9086 |
| Raw | Mean-centered Residual | 0.6671 | 0.9435 |
| LN-free | Mahalanobis | 0.7447 | 0.8152 |
| LN-free | kNN (k=10) | 0.7337 | 0.9086 |
| LN-free | Mean-centered Residual | 0.7456 | 0.8429 |
| LN-pooler | Mahalanobis | 0.7447 | 0.8162 |
| LN-pooler | kNN (k=10) | 0.7354 | 0.8676 |
| LN-pooler | Mean-centered Residual | 0.7432 | 0.8203 |

## Mandatory sanity checks

- L12 LN-pooler vs existing h_doc max absolute feature error: `0`.
- L12 detector-result maximum absolute error vs existing h_doc rows: `1.11022302e-16`.

Layers 1–11 LN-pooler results are analytical probes, not the model's real forward path.
