# Frozen DiT representation layer scan

## Scope

This development-only study keeps the checkpoint and OOD scoring rules fixed and changes only
the representation readout or Transformer layer. It does not select a deployable Gate and does
not access the sealed 694 N + 2,324 O test split.

## Frozen inputs

- Model: `microsoft/dit-base-finetuned-rvlcdip`, revision
  `23f8b03d130fb66bbc9a15df3c75d753e49240eb`, frozen float32 inference.
- Development data: 306 N + 974 O after the existing 14-row cross-label pixel-conflict group is
  excluded from the physically isolated development manifest.
- Reference data: the existing balanced 32k RVL bank and its exact manifest-ID-matched 8k
  subset; the 8k bank is never selected by row position or resampled.
- Feature dimension: 768; kNN `k=10`; PCA principal dimension `D=512`.
- Metrics: micro AUROC and first-achieved FPR95, with 2,000 stratified bootstrap resamples and
  seed `20260812` for every reported row.

Exactly 32,000 reference images and 1,280 development images are passed through the backbone
once. For each of Layers 1–12, CLS and raw patch mean are pooled on the GPU and only those two
float32 768-dimensional vectors are transferred to the CPU cache. Full hidden-state tensors are
never cached on the CPU.

## Hard execution gates

1. **G1 — representation structure.** On the first real float32 batch, applying the model's
   pooler LayerNorm to Layer-12 raw patch mean must reproduce `pooler_output` with maximum
   absolute error below `1e-5`.
2. **G2 — existing pipeline reproduction.** Reconstructed `h_doc`, the frozen 8k manifest IDs,
   and the existing Mahalanobis implementation must reproduce development AUROC `0.7416556`
   within absolute tolerance `0.005`.

Failure of either gate stops the run before the layer study. The gates may not be relaxed.

## Experiments

### S1 — final-layer readout

Compare Layer-12 raw patch mean, `h_doc = LayerNorm(raw mean)`, and Layer-12 CLS. Apply the same
Mahalanobis, cosine kNN, and mean-centered PCA Residual methods to every representation. The 32k
bank is primary and the nested 8k bank is a sensitivity analysis.

### S1b — exact ViM decomposition

On `h_doc` with the 32k bank, compare Energy, ViM-origin Residual-only, and full ViM. The
Residual-only score is exactly `-alpha * residual_norm` from the same fitted ViM instance; only
the logit/Energy term is removed.

### S2 — Layer 1–12 scan

For each layer, read only raw patch mean and apply the same three S1 methods with both 32k and 8k
banks. The full curves are reported directly; no best-layer rerun or tuning is performed.

S1/S2 Residual always centers on the corresponding reference-feature mean. It is distinct from
the classifier-origin Residual used only in S1b. Top-512 explained variance ratio is diagnostic
metadata and is never used to change `D`, discard a layer, or choose a method.

## Evaluation boundary

The one-pass extraction is fixed at 33,280 images and therefore does not re-extract the existing
8k threshold-validation or 10k T-guardrail images for new representations. Existing `h_doc`
guardrail results remain unchanged; this layer scan reports development AUROC/FPR95 only. This
does not authorize reading, inference, or scoring of the sealed test split.
