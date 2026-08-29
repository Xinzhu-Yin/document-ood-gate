# Frozen supplementary post-hoc baseline protocol

## Purpose

Complete one bounded baseline panel before deciding whether to deepen the Mahalanobis
representation study. This addendum evaluates exactly four methods: GEN, shrinkage Relative
Mahalanobis, ViM, and ReAct + Energy. It does not run layer readout, layer scanning, fusion, or
the sealed test.

## Common evaluation contract

- Frozen checkpoint and cached float32 `h_doc`/logits from protocol v1.1.
- Same duplicate-clean development rows: 306 N and 974 O. Before this panel, the frozen
  assignment file is filtered once into a 1,294-row development-only manifest and DiT inference
  is rerun only for those image paths, creating a physically separate feature cache. The 14
  cross-label exact-pixel rows are then excluded exactly as in protocol v1.1.
- Same nested 32,000-row class-balanced RVL reference bank.
- Same 8,000-row RVL validation set; each score independently receives a threshold giving at
  least 99% validation acceptance.
- Same 10,000-row T guardrail, used only for reporting and never for method selection.
- Micro AUROC and FPR95 with N as positive and higher score meaning more normal.
- 2,000 stratified bootstrap resamples with seed 20260812; paired deltas use identical sampled
  N and O row indices.
- The 3,018-row sealed test must not be loaded, inferred, scored, or used for configuration.
  The extraction step may read the pre-existing split column as metadata to select development
  paths, but it must not open sealed image pixels, logits, or features. The evaluation command
  refuses any split manifest that contains sealed rows before opening its OOD feature cache.

Before the new methods run, cached Energy and 32k Mahalanobis must reproduce their frozen
development AUROC, FPR95, and T false-rejection rate within `1e-8`. Failure stops the panel.

## Frozen methods

### GEN

Use the [authors' score](https://github.com/XixiLiu95/GEN/blob/main/benchmark.py):

`-sum(p^gamma * (1-p)^gamma)` over the largest M probabilities. Freeze `gamma=0.1`, requested
`M=100`, and temperature 1. With only 16 classes, the official slicing behavior makes the
effective M equal to 16. No M or gamma sweep is allowed.

### Shrinkage Relative Mahalanobis

Fit the existing 16-class Mahalanobis model on the 32k bank. Separately fit one background
Gaussian to the same features while ignoring labels. Return background squared distance minus
minimum class squared distance. Both covariance estimates use Ledoit-Wolf so the only change
from the existing baseline is the relative background correction. No normalization or bank-size
sweep is allowed. The definition follows [Relative Mahalanobis](https://arxiv.org/abs/2106.09022).

### ViM

Use the actual classifier-input `h_doc`, its original 16-class logits, and the frozen linear
classifier. Follow the [official ViM implementation](https://github.com/haoqiwang/vim/blob/master/benchmark.py):
the classifier-defined origin, an origin-centered second moment, principal dimension 512 for a
768-dimensional feature, a 256-dimensional residual space, and alpha estimated only from the
32k bank. Return `logsumexp(logits) - alpha * residual_norm` so higher means more normal. No
dimension sensitivity sweep is allowed.

### ReAct + Energy

Compute one global upper clipping threshold at the 90th percentile of all 32k reference `h_doc`
coordinates. Clip only values above it, retain negative values, pass the clipped vector through
the unchanged linear classifier, and use temperature-one Energy normality. The downstream
classifier continues to use the original logits; clipped logits are only an OOD score.

This is a controlled architecture-transfer baseline. The original
[ReAct](https://proceedings.neurips.cc/paper_files/paper/2021/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html)
evidence primarily concerns nonnegative pooled CNN activations, whereas DiT `h_doc` is a signed
LayerNorm representation. The clip threshold and per-split clipping diagnostics must therefore
be reported explicitly.

## Decision boundary

This panel reports point estimates, individual confidence intervals, and paired deltas against
each method's direct parent baseline: GEN versus negative entropy, Relative Mahalanobis versus
Mahalanobis, and ViM/ReAct versus Energy. It does not automatically choose the next experiment.
The results are reviewed before any layer study or sealed evaluation is authorized.
