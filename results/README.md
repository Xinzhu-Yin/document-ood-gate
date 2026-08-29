# Results

This directory contains the complete aggregate evidence for the project. Raw document images,
model weights, and extracted feature matrices are intentionally excluded from Git.

## Reading order

| Stage | Purpose | Human-readable result | Machine-readable evidence |
|---|---|---|---|
| 1. Published-baseline reproduction | Verify MSP/Energy before new experiments | Summary below | [`reproduction/`](reproduction/) |
| 2. Data and duplicate audit | Record release gaps, duplicates, and the frozen split | Summary below | [`data_release_status.json`](data_release_status.json), [`duplicate_audit/`](duplicate_audit/) |
| 3. Main single-signal panel | Compare output and representation scores | [`single_signals/development_summary.md`](single_signals/development_summary.md) | [`single_signals/development_all_single_signals.json`](single_signals/development_all_single_signals.json) |
| 4. Supplementary panel | Evaluate GEN, RMD, ViM, and ReAct | [`supplementary_baselines/development_summary.md`](supplementary_baselines/development_summary.md) | [`supplementary_baselines/development_results.csv`](supplementary_baselines/development_results.csv), [`development_results.json`](supplementary_baselines/development_results.json) |

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

## Metric note

FPR95 uses the threshold that accepts at least 95% of N. T false rejection uses a different,
score-specific threshold fixed to accept 99% of an independent 8k RVL validation set. The two
columns are not measurements at the same threshold.

## Evaluation boundary

The 3,018-row sealed test has not been read, inferred, scored, or used for method selection. All
new-method results in this directory are development observations, not final test claims.
