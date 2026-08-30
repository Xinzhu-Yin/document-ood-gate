# Frozen sealed-test protocol

Prepared before sealed access: `2026-08-30T13:57:25.8193875Z`.

This protocol freezes the project's final evaluation. The commit containing this file must be
created and pushed before the first sealed sample is opened. Its SHA and timestamp are recorded
in the final sealed metadata. The sealed split contains exactly 694 N + 2,324 O = 3,018 images.

## Inputs and one-time boundary

- Reuse the existing balanced 32k RVL layer-scan reference cache; do not re-extract reference
  images.
- Open only the 3,018 sealed images. Each image receives one frozen DiT forward pass, saving
  float32 logits, h_doc, and Layer 1–12 raw patch means. Full token hidden states are not cached.
- Derive LN-free and LN-pooler offline from the saved raw means.
- Metrics are AUROC and first-achieved FPR95 with 2,000 stratified bootstrap resamples and seed
  `20260812`.

## D1 — core baseline panel

Run MSP, Energy, h_doc Mahalanobis-32k, h_doc kNN-32k (`k=10`), ViM-32k, and ViM-origin
Residual-only. ViM and its Residual-only component use `vim_classifier_origin`,
`u = -pinv(W) @ b`, and D=512.

## D2 — full replication grid

For every Layer 1–12, evaluate raw patch mean, LN-free, and LN-pooler with Mahalanobis-32k,
kNN-32k (`k=10`), and mean-centered PCA Residual (`reference_mean`, D=512). Layer 1–11
LN-pooler remains an analytical probe rather than the model's real forward path.

For P4 and P5, the set of “pre-specified sealed configurations” is the union of all D1 and D2
rows above. The highest-AUROC and lowest-FPR95 configurations are determined from this full
frozen set; no configuration is added after access.

## Freeze rules

- No new methods.
- No new layer selection.
- No D tuning.
- No k tuning.
- No new readout.
- No new normalization.
- No 8k sealed sensitivity.
- No CLS sealed scan.
- No post-test redesign.
- No development redesign after sealed results are observed.

The development D=128/256/384 sensitivity panel is diagnostic only. The sealed Residual setting
remains D=512 regardless of development results. Once sealed results exist, experiments stop.
