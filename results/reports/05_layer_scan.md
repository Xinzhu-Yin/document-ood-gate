# Development-only layer scan

## Protocol and hard gates

- G1 structure gate: **PASS** (first float32 batch, `n=32`; max error `0 < 1e-5`).
- G2 existing-pipeline reproduction gate: **PASS**
- Extraction: **33,280 images** (32,000 reference + 306 N + 974 O).
- Sealed rows loaded: **0**; sealed rows scored: **0**.
- This run extracted only 33,280 images, so the existing T guardrail was not re-extracted or rescored; its prior setting remains unchanged.

## Residual definitions

- **Mean-centered PCA Residual (S1/S2):** PCA is fitted after subtracting the reference-feature mean; D=512 and the score is the negative residual norm.
- **ViM-origin Residual (S1b):** uses the classifier-defined origin `u = -pinv(W) @ b`, the existing ViM null space and alpha, and the score `-alpha * residual_norm`. It is not the mean-centered PCA Residual.

## Final-layer representation comparison

AUROC [95% CI] / FPR95 [95% CI]:

| Representation | Method | Bank | AUROC / FPR95 |
|---|---|---:|---|
| L12 raw patch mean | Mahalanobis | 32k | 0.6567 [0.6195, 0.6943] / 0.9579 [0.9025, 0.9795] |
| L12 raw patch mean | Mahalanobis | 8k | 0.6479 [0.6107, 0.6859] / 0.9466 [0.8943, 0.9692] |
| L12 raw patch mean | kNN | 32k | 0.7336 [0.6990, 0.7688] / 0.9086 [0.8275, 0.9620] |
| L12 raw patch mean | kNN | 8k | 0.7245 [0.6896, 0.7590] / 0.9014 [0.8326, 0.9415] |
| L12 raw patch mean | Mean-centered PCA Residual | 32k | 0.6671 [0.6298, 0.7052] / 0.9435 [0.8891, 0.9713] |
| L12 raw patch mean | Mean-centered PCA Residual | 8k | 0.6598 [0.6218, 0.6998] / 0.9353 [0.8830, 0.9743] |
| h_doc | Mahalanobis | 32k | 0.7447 [0.7111, 0.7784] / 0.8162 [0.7485, 0.9261] |
| h_doc | Mahalanobis | 8k | 0.7417 [0.7074, 0.7752] / 0.8296 [0.7577, 0.9107] |
| h_doc | kNN | 32k | 0.7354 [0.7014, 0.7700] / 0.8676 [0.8101, 0.9076] |
| h_doc | kNN | 8k | 0.7244 [0.6900, 0.7585] / 0.8747 [0.8162, 0.9138] |
| h_doc | Mean-centered PCA Residual | 32k | 0.7432 [0.7105, 0.7771] / 0.8203 [0.7433, 0.9076] |
| h_doc | Mean-centered PCA Residual | 8k | 0.7385 [0.7042, 0.7730] / 0.8357 [0.7813, 0.9251] |
| L12 CLS | Mahalanobis | 32k | 0.6633 [0.6283, 0.6995] / 0.8943 [0.8182, 0.9569] |
| L12 CLS | Mahalanobis | 8k | 0.6543 [0.6189, 0.6902] / 0.8932 [0.8039, 0.9312] |
| L12 CLS | kNN | 32k | 0.6315 [0.5948, 0.6694] / 0.8932 [0.8573, 0.9425] |
| L12 CLS | kNN | 8k | 0.6058 [0.5673, 0.6460] / 0.9127 [0.8758, 0.9394] |
| L12 CLS | Mean-centered PCA Residual | 32k | 0.6828 [0.6485, 0.7182] / 0.9035 [0.8111, 0.9723] |
| L12 CLS | Mean-centered PCA Residual | 8k | 0.6731 [0.6380, 0.7082] / 0.9076 [0.8347, 0.9713] |

## ViM decomposition

| Component | Centering convention | AUROC / FPR95 |
|---|---|---|
| energy | None | 0.7282 [0.6941, 0.7588] / 0.8614 [0.7895, 0.8871] |
| vim_origin_residual | vim_classifier_origin | 0.7424 [0.7095, 0.7769] / 0.8080 [0.7659, 0.9107] |
| vim | vim_classifier_origin | 0.7451 [0.7121, 0.7791] / 0.7988 [0.7608, 0.9127] |

## Complete Layer 1–12 curves

The complete curves are reported below; this report does not select an argmax layer.
Any layer-specific pattern is a development observation; not confirmed on held-out data.

Each cell is AUROC / FPR95. FPR95: lower is better.

| Layer | Mahalanobis 32k | Mahalanobis 8k | kNN 32k | kNN 8k | Mean-centered Residual 32k | Mean-centered Residual 8k |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5191 / 0.9394 | 0.5170 / 0.9394 | 0.5286 / 0.9579 | 0.5317 / 0.9435 | 0.5586 / 0.8727 | 0.5577 / 0.8727 |
| 2 | 0.5001 / 0.9394 | 0.5000 / 0.9394 | 0.5153 / 0.9209 | 0.5143 / 0.9302 | 0.5119 / 0.9240 | 0.5073 / 0.9343 |
| 3 | 0.4993 / 0.9425 | 0.5001 / 0.9487 | 0.5068 / 0.9251 | 0.5066 / 0.9230 | 0.4942 / 0.9374 | 0.4931 / 0.9435 |
| 4 | 0.4911 / 0.9281 | 0.4919 / 0.9302 | 0.4923 / 0.9035 | 0.4942 / 0.9148 | 0.4862 / 0.9312 | 0.4875 / 0.9312 |
| 5 | 0.4946 / 0.9292 | 0.4952 / 0.9322 | 0.4941 / 0.9066 | 0.4928 / 0.9107 | 0.4873 / 0.9220 | 0.4874 / 0.9322 |
| 6 | 0.5248 / 0.9117 | 0.5246 / 0.9086 | 0.5283 / 0.9045 | 0.5292 / 0.9014 | 0.5122 / 0.9179 | 0.5107 / 0.9179 |
| 7 | 0.5620 / 0.8747 | 0.5622 / 0.8706 | 0.5486 / 0.9004 | 0.5444 / 0.8912 | 0.5482 / 0.8871 | 0.5493 / 0.8891 |
| 8 | 0.6091 / 0.8501 | 0.6086 / 0.8491 | 0.6402 / 0.8070 | 0.6321 / 0.8203 | 0.5948 / 0.8583 | 0.5915 / 0.8593 |
| 9 | 0.6191 / 0.8409 | 0.6206 / 0.8306 | 0.6999 / 0.8306 | 0.6737 / 0.8368 | 0.6168 / 0.8665 | 0.6180 / 0.8429 |
| 10 | 0.5793 / 0.9209 | 0.5806 / 0.9086 | 0.7489 / 0.9425 | 0.7378 / 0.9476 | 0.5846 / 0.9117 | 0.5824 / 0.9220 |
| 11 | 0.6736 / 0.9548 | 0.6678 / 0.9343 | 0.7408 / 0.8789 | 0.7315 / 0.8922 | 0.6814 / 0.9138 | 0.6769 / 0.9189 |
| 12 | 0.6567 / 0.9579 | 0.6479 / 0.9466 | 0.7336 / 0.9086 | 0.7245 / 0.9014 | 0.6671 / 0.9435 | 0.6598 / 0.9353 |

## Files

The CSV/JSON files contain all point estimates, confidence intervals, bank sizes, representations, methods, centering conventions, and variance-ratio diagnostics.
