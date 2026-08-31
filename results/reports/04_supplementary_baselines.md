# Supplementary post-hoc baselines — development checkpoint

> Scope: 306 N / 974 O development rows. The sealed 70% test was not loaded, inferred, or scored.

## Reproducibility gate

Status: **PASS**

## Results

| Signal | AUROC ↑ | AUROC 95% CI | FPR95 ↓ | FPR95 95% CI | T false reject ↓ |
|---|---:|---:|---:|---:|---:|
| Energy (existing) | 0.7282 | [0.6941, 0.7588] | 0.8614 | [0.7895, 0.8871] | 1.13% |
| Negative entropy (existing) | 0.7148 | [0.6800, 0.7476] | 0.8480 | [0.7906, 0.8994] | 1.20% |
| Mahalanobis 32k (existing) | 0.7447 | [0.7111, 0.7784] | 0.8162 | [0.7485, 0.9261] | 0.89% |
| GEN (gamma=0.1, M=16) | 0.7173 | [0.6826, 0.7494] | 0.8624 | [0.7834, 0.9117] | 1.06% |
| Relative Mahalanobis 32k | 0.7328 | [0.6981, 0.7663] | 0.8655 | [0.7608, 0.9230] | 0.93% |
| ViM 32k (DIM=512) | 0.7451 | [0.7121, 0.7791] | 0.7988 | [0.7608, 0.9127] | 1.00% |
| ReAct p90 + Energy | 0.5436 | [0.5051, 0.5797] | 0.8819 | [0.8285, 0.9025] | 1.08% |

## Paired comparison against each method's parent baseline

| Candidate | Baseline | ΔAUROC | ΔAUROC 95% CI | ΔFPR95 | ΔFPR95 95% CI |
|---|---|---:|---:|---:|---:|
| GEN (gamma=0.1, M=16) | Negative entropy (existing) | +0.0026 | [-0.0029, +0.0081] | +0.0144 | [-0.0349, +0.0606] |
| Relative Mahalanobis 32k | Mahalanobis 32k (existing) | -0.0119 | [-0.0230, -0.0015] | +0.0493 | [-0.0493, +0.1027] |
| ViM 32k (DIM=512) | Energy (existing) | +0.0169 | [-0.0081, +0.0402] | -0.0626 | [-0.0965, +0.0637] |
| ReAct p90 + Energy | Energy (existing) | -0.1847 | [-0.2200, -0.1513] | +0.0205 | [-0.0318, +0.0832] |

Positive ΔAUROC and negative ΔFPR95 favor the candidate. Each method uses its own 99%-acceptance threshold calibrated on the same 8k RVL validation set; T is report-only.

ReAct is an explicit transfer of the published clipping rule to signed, LayerNorm-normalized DiT `h_doc` features, so that result should be interpreted as an architecture-transfer baseline rather than an exact CNN reproduction.
