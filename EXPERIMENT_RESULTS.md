# Experimental Results

All values below are measured results. New methods have been evaluated only on development data;
the 3,018-image sealed test remains untouched.

## 1. Published-baseline reproduction

| Scope | MSP AUROC | MSP FPR95 | Energy AUROC | Energy FPR95 |
|---|---:|---:|---:|---:|
| Published, 4,417 rows | 0.7280 | 0.8470 | 0.7530 | 0.8430 |
| Independent image inference, 4,312 images | 0.7286 | 0.8465 | 0.7533 | 0.8444 |
| Full sensitivity, 4,417 rows | 0.7279 | 0.8466 | 0.7535 | 0.8436 |

**Reproduction decision: GREEN.**

The independent rerun uses 1,002 N and 3,310 O images. The full sensitivity result uses author
logits only for the 105 unavailable O images, so it is not presented as a complete independent
image rerun.

## 2. Duplicate audit and split

| Audit item | Result |
|---|---:|
| Images audited | 4,312 |
| pHash candidate pairs | 583 |
| Confirmed duplicate groups | 53 |
| Images in confirmed groups | 120 |
| Cross-label confirmed groups | 1 |
| Confirmed group crossing development/test | 0 |

The frozen split contains 1,294 development rows and 3,018 sealed rows. Excluding the 14-row
identical-pixel label conflict leaves 306 N and 974 O for primary development comparisons.

## 3. Single-signal development panel

All intervals are 95% confidence intervals from 2,000 stratified bootstrap resamples.

| Signal | AUROC [95% CI] ↑ | FPR95 [95% CI] ↓ | T false rejection ↓ |
|---|---:|---:|---:|
| MSP | 0.7103 [0.6739, 0.7432] | 0.8552 [0.7598, 0.9230] | 1.10% |
| Energy | 0.7282 [0.6938, 0.7613] | 0.8614 [0.7916, 0.8891] | 1.13% |
| Maximum logit | 0.7203 [0.6874, 0.7534] | 0.8450 [0.7577, 0.8994] | 1.10% |
| Negative entropy | 0.7148 [0.6788, 0.7502] | 0.8480 [0.7854, 0.8963] | 1.20% |
| Top-2 margin | 0.7029 [0.6662, 0.7391] | 0.8686 [0.8141, 0.9168] | 1.03% |
| Feature norm | 0.4883 [0.4501, 0.5261] | 0.9928 [0.9856, 0.9979] | 0.96% |
| kNN, 8k | 0.7244 [0.6873, 0.7592] | 0.8747 [0.8152, 0.9138] | 0.90% |
| kNN, 32k | 0.7354 [0.6992, 0.7717] | 0.8676 [0.8070, 0.9066] | 0.97% |
| kNN, 80k | 0.7433 [0.7073, 0.7770] | 0.8511 [0.7813, 0.9138] | 0.89% |
| Mahalanobis, 8k | 0.7417 [0.7070, 0.7749] | 0.8296 [0.7597, 0.9097] | 1.01% |
| Mahalanobis, 32k | 0.7447 [0.7112, 0.7785] | **0.8162** [0.7495, 0.9261] | 0.89% |
| Mahalanobis, 80k | **0.7463** [0.7139, 0.7788] | 0.8306 [0.7464, 0.9271] | 0.92% |

Observed on this development split:

- Mahalanobis-80k has the highest AUROC point estimate.
- Mahalanobis-32k has the lowest FPR95 point estimate in the main panel.
- kNN AUROC rises from 0.7244 to 0.7433 as the bank grows from 8k to 80k.
- Feature norm is close to random ranking.
- Confidence intervals overlap substantially; these are candidate-selection observations, not
  final generalization claims.

## 4. Supplementary development panel

| Method | AUROC [95% CI] ↑ | FPR95 [95% CI] ↓ | T false rejection ↓ |
|---|---:|---:|---:|
| GEN | 0.7173 [0.6826, 0.7494] | 0.8624 [0.7834, 0.9117] | 1.06% |
| Relative Mahalanobis | 0.7328 [0.6981, 0.7663] | 0.8655 [0.7608, 0.9230] | 0.93% |
| ViM | **0.7451** [0.7121, 0.7791] | **0.7988** [0.7608, 0.9127] | 1.00% |
| ReAct + Energy | 0.5436 [0.5051, 0.5797] | 0.8819 [0.8285, 0.9025] | 1.08% |

Paired changes against each method's direct baseline follow. Positive ΔAUROC and negative
ΔFPR95 favor the candidate.

| Candidate | Baseline | ΔAUROC [95% CI] | ΔFPR95 [95% CI] |
|---|---|---:|---:|
| GEN | Negative entropy | +0.0026 [-0.0029, +0.0081] | +0.0144 [-0.0349, +0.0606] |
| Relative Mahalanobis | Mahalanobis-32k | -0.0119 [-0.0230, -0.0015] | +0.0493 [-0.0493, +0.1027] |
| ViM | Energy | +0.0169 [-0.0081, +0.0402] | -0.0626 [-0.0965, +0.0637] |
| ReAct + Energy | Energy | -0.1847 [-0.2200, -0.1513] | +0.0205 [-0.0318, +0.0832] |

The ViM point estimates improve on Energy, but both paired confidence intervals cross zero.
Relative Mahalanobis decreases AUROC relative to Mahalanobis-32k. ReAct transfers poorly to the
signed, LayerNorm-normalized DiT representation used here.

## 5. Threshold reality check

The following operating points use each score's separate threshold calibrated for 99% acceptance
on 8,000 RVL validation pages:

| Score | Development N accepted | Development O incorrectly accepted | T false rejection |
|---|---:|---:|---:|
| Energy | 97.39% | 89.32% | 1.13% |
| Mahalanobis-32k | 84.64% | 64.27% | 0.89% |
| ViM | 86.93% | 66.43% | 1.00% |

About 1% T false rejection does not imply 99% acceptance of shifted N. At the separate FPR95
operating point, the best observed value is 0.7988, meaning 79.88% of O is still incorrectly
accepted while 95% of N is accepted.

## 6. Current limits

- New methods use only 306 N + 974 O development images.
- N is small, and FPR95 confidence intervals are wide.
- The 3,018-image sealed test has not been read or scored.
- The 395 reconstructed O pages are rerenders; 105 missing O pages cannot enter feature methods.
- A complete shortcut audit has not yet been run.
- No learned fusion Gate or Transformer layer scan has been run.

The defensible current result is therefore limited: internal representation geometry has slightly
better development point estimates than common output-confidence scores, while no evaluated score
yet provides an effective operational Gate.
