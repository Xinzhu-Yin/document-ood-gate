# Results

This directory contains the complete aggregate evidence for the project. Raw document images,
model weights, and extracted feature matrices are intentionally excluded from Git.

## Directory layout

```text
results/
├── tables/       # CSV/JSON metric tables, ordered by research stage
├── figures/      # Publication-ready plots, ordered by research stage
├── reports/      # Human-readable stage summaries
└── supporting/   # Data audits, provenance, gates, and run metadata
```

Curated GitHub artifacts live only in these four locations. Re-running the command-line pipeline
writes mixed raw outputs to the ignored `results/runs/<stage>/` workspace, so generated files do
not make this directory untidy again.

## Reading order

| Stage | Purpose | Human-readable result | Machine-readable evidence |
|---|---|---|---|
| 1. Published-baseline reproduction | Verify MSP/Energy before new experiments | Summary below | [`tables/01_reproduction/`](tables/01_reproduction/) |
| 2. Data and duplicate audit | Record release gaps, duplicates, and the frozen split | Summary below | [`supporting/02_data_audit/`](supporting/02_data_audit/) |
| 3. Main single-signal panel | Compare output and representation scores | [`reports/03_single_signals.md`](reports/03_single_signals.md) | [`tables/03_single_signals/`](tables/03_single_signals/) |
| 4. Supplementary panel | Evaluate GEN, RMD, ViM, and ReAct | [`reports/04_supplementary_baselines.md`](reports/04_supplementary_baselines.md) | [`tables/04_supplementary_baselines/`](tables/04_supplementary_baselines/) |
| 5. Representation layer scan | Compare final-layer readouts and full Layer 1–12 curves | [`reports/05_layer_scan.md`](reports/05_layer_scan.md) | [`tables/05_layer_scan/`](tables/05_layer_scan/), [`figures/05_layer_scan/`](figures/05_layer_scan/) |
| 6. Final development ablations | Isolate LayerNorm and diagnose fixed Residual dimensions | [`reports/06_final_ablation.md`](reports/06_final_ablation.md) | [`tables/06_final_ablation/`](tables/06_final_ablation/), [`figures/06_final_ablation/`](figures/06_final_ablation/) |
| 7. One-time sealed replication | Evaluate the frozen D1+D2 panel and P1–P5 | [`reports/07_sealed_test.md`](reports/07_sealed_test.md) | [`tables/07_sealed_test/`](tables/07_sealed_test/), [`figures/07_sealed_test/`](figures/07_sealed_test/), [`supporting/07_sealed_test/`](supporting/07_sealed_test/) |

## 1. Reproduction gate

| Scope | MSP AUROC | MSP FPR95 | Energy AUROC | Energy FPR95 |
|---|---:|---:|---:|---:|
| Published, 4,417 rows | 0.7280 | 0.8470 | 0.7530 | 0.8430 |
| Independent image inference, 4,312 images | 0.7286 | 0.8465 | 0.7533 | 0.8444 |
| Full output-score sensitivity, 4,417 rows | 0.7279 | 0.8466 | 0.7535 | 0.8436 |

Decision: **GREEN**. The full sensitivity result fills only the 105 unavailable O rows with the
authors' logits and is not described as a complete independent image rerun.

## 2. Audit and split

| Item | Result |
|---|---:|
| Images independently available/recovered | 4,312 |
| pHash candidate pairs | 583 |
| Confirmed duplicate groups | 53 |
| Images in confirmed groups | 120 |
| Confirmed group crossing development/test | 0 |
| Initial development split | 308 N + 986 O = 1,294 |
| Sealed test split | 694 N + 2,324 O = 3,018 |
| Primary development comparison | 306 N + 974 O = 1,280 |

The difference between initial and primary development counts is one 14-row, identical-pixel
group carrying contradictory N/O labels. It remains in the paper reproduction and is excluded
only from the primary research comparison.

## 3. Main development checkpoint

| Signal | Reference | AUROC ↑ | FPR95 ↓ | T false rejection ↓ |
|---|---:|---:|---:|---:|
| MSP | — | 0.7103 | 0.8552 | 1.10% |
| Energy | — | 0.7282 | 0.8614 | 1.13% |
| kNN, k=10 | 80k | 0.7433 | 0.8511 | 0.89% |
| Mahalanobis | 32k | 0.7447 | **0.8162** | 0.89% |
| Mahalanobis | 80k | **0.7463** | 0.8306 | 0.92% |

The linked main summary contains all 12 signals and their 95% bootstrap confidence intervals.

## 4. Supplementary development checkpoint

| Method | AUROC ↑ | FPR95 ↓ | T false rejection ↓ |
|---|---:|---:|---:|
| GEN | 0.7173 | 0.8624 | 1.06% |
| Relative Mahalanobis | 0.7328 | 0.8655 | 0.93% |
| ViM | **0.7451** | **0.7988** | 1.00% |
| ReAct + Energy | 0.5436 | 0.8819 | 1.08% |

ViM improves the point estimates over Energy, but its paired ΔAUROC and ΔFPR95 confidence
intervals both cross zero. Relative Mahalanobis reduces AUROC relative to Mahalanobis-32k, and
ReAct substantially reduces AUROC in this signed DiT representation setting.

## 5. Frozen representation layer scan

| Gate | Required | Observed | Status |
|---|---:|---:|---|
| G1: first float32 batch (`n=32`), `LayerNorm(L12 raw mean)` vs `pooler_output` | max error < 1e-5 | 0 | PASS |
| G2: Mahalanobis-8k reproduction | \|ΔAUROC\| < 0.005 (exclusive) | 0.00000335 | PASS |

| Fixed 32k development result | AUROC ↑ | FPR95 ↓ |
|---|---:|---:|
| L12 raw mean + Mahalanobis | 0.6567 | 0.9579 |
| `h_doc` + Mahalanobis | 0.7447 | 0.8162 |
| L12 CLS + Mahalanobis | 0.6633 | 0.8943 |
| Layer 10 raw mean + kNN | 0.7489 | 0.9425 |
| ViM-origin Residual-only | 0.7424 | 0.8080 |
| Full ViM | 0.7451 | 0.7988 |

The complete 12-layer 32k primary and 8k sensitivity curves are reported rather than only an
argmax. Any layer-specific pattern is a development observation; not confirmed on held-out data.
S1/S2 use reference-mean PCA Residual, while S1b uses the existing classifier-origin ViM
Residual. See the linked summary for all confidence intervals and plots.

## 6. Final development ablations

The final development round reused cached 32k reference and 306 N + 974 O development features;
sealed rows loaded/scored were 0/0. At Layer 12, Mahalanobis AUROC changed from 0.6567 on raw
patch mean to 0.7447 after either LN-free or LN-pooler normalization. The existing `h_doc`
pipeline was reproduced to maximum feature error 0 and maximum detector-result error
`1.11e-16`. The fixed D=128/256/384/512 panel was diagnostic only; D=512 remained frozen for
sealed evaluation.

## 7. Final sealed replication

Exactly 694 N + 2,324 O = 3,018 sealed images were opened once and received one frozen DiT
forward pass each. The existing 32k reference cache was reused; reference images re-extracted:
0. Forbidden sealed runs (8k sensitivity, CLS, D sensitivity, new methods): all 0.

| Frozen sealed configuration | AUROC ↑ | FPR95 ↓ |
|---|---:|---:|
| MSP | 0.7372 | 0.8468 |
| Energy | **0.7650** | 0.8369 |
| `h_doc` Mahalanobis-32k | 0.7474 | 0.8546 |
| `h_doc` kNN-32k (`k=10`) | 0.7405 | 0.8795 |
| ViM-origin Residual-only | 0.7433 | 0.8808 |
| ViM | 0.7473 | 0.8731 |

P1–P5 were all supported. Energy had the highest AUROC in the complete frozen set. Layer 9
LN-pooler Mahalanobis had the lowest FPR95 (0.8171) but AUROC 0.6607; therefore the highest-AUROC
and lowest-FPR95 configurations were different. The complete 108-row sealed layer grid is in
[`tables/07_sealed_test/sealed_layer_grid.csv`](tables/07_sealed_test/sealed_layer_grid.csv), and the full curves are
reported rather than only an argmax.

The immutable local freeze commit `451fbdb4e902e75843e8e4ef5a31745cc95f1875` was created before
sealed access and its timestamp predates the first image access. Protocol deviation: it could not
be published before access because command-line GitHub credentials were unavailable. During
final sync, its byte-identical tree (`7d12bfd0dd46c5055ec2428bf93094b6188bb157`) was published as
remote commit `d0d058bdb7abd3ff719a0553aca58580823bb714`; the commit SHA differs because the GitHub API
generated new commit metadata. No score was available when the implementation-only string-array
serialization repair was made, and no experimental setting changed.

## Metric note

FPR95 uses the threshold that accepts at least 95% of N. T false rejection uses a different,
score-specific threshold fixed to accept 99% of an independent 8k RVL validation set. The two
columns are not measurements at the same threshold.

## Evaluation boundary

The sealed test was used once for final replication only, after the local freeze commit. Its
results were not used to add methods, select layers, tune D/k, change readouts, or redesign the
development experiment. The final metadata records `post_sealed_experimental_iteration=false`.

A full shortcut audit was not performed. The development/sealed split preserves source
composition, reducing source-distribution mismatch across splits, but this does not rule out
within-task acquisition, rendering, or source-related shortcuts. In particular, 395 recovered O
pages are calibrated rerenders rather than byte-identical original TIFFs.

