# Document OOD Gate for Frozen DiT

A reproducible study of post-hoc out-of-distribution signals for a frozen document
classifier. DiT is never retrained; the project asks which score best accepts supported
documents while rejecting documents outside the classifier's 16 categories.

## Research question

- `T`: familiar RVL-CDIP documents used only as a false-rejection guardrail.
- `N`: shifted documents that still belong to one of the 16 supported categories.
- `O`: documents outside those 16 categories.

The Gate should accept `T` and `N`, reject `O`, and remain independent of whether DiT later
predicts the correct in-distribution class. This repository currently covers single scores and
four fixed post-hoc additions; learned score fusion is not yet included.

## Experiment at a glance

| Item | Frozen setting |
|---|---|
| Classifier | [`microsoft/dit-base-finetuned-rvlcdip`](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip), inference only |
| Development N/O | 306 N + 974 O after duplicate-conflict exclusion |
| Reference bank | Nested 8k/32k/80k RVL banks in the main panel; fixed 32k in the supplementary panel |
| Threshold calibration | 8k independent RVL pages, fixed at 99% acceptance per score |
| T guardrail | 10k balanced RVL test pages, report-only |
| Metrics | AUROC ↑, FPR95 ↓, T false-rejection rate ↓ |
| Unseen evaluation | 3,018-row sealed N/O test, not evaluated or used for model selection |

## Results

The mandatory reproduction gate passed before development experiments began.

| Scope | MSP AUROC / FPR95 | Energy AUROC / FPR95 |
|---|---:|---:|
| Published target, 4,417 rows | 0.728 / 0.847 | 0.753 / 0.843 |
| Independent image rerun, 4,312 rows | 0.729 / 0.847 | 0.753 / 0.844 |
| Full sensitivity result, 4,417 rows | 0.728 / 0.847 | 0.753 / 0.844 |

The full sensitivity result uses author-provided logits only for 105 source images that are no
longer available; their provenance is recorded in the result file.

Selected development results:

| Signal | AUROC ↑ | FPR95 ↓ | T false reject ↓ |
|---|---:|---:|---:|
| Energy | 0.7282 | 0.8614 | 1.13% |
| Mahalanobis 32k | 0.7447 | 0.8162 | 0.89% |
| Mahalanobis 80k | **0.7460** | 0.8310 | 0.92% |
| GEN | 0.7173 | 0.8624 | 1.06% |
| Relative Mahalanobis 32k | 0.7328 | 0.8655 | 0.93% |
| ViM 32k | 0.7451 | **0.7988** | 1.00% |
| ReAct p90 + Energy | 0.5436 | 0.8819 | 1.08% |

These are development-set observations, not final claims. ViM has the lowest observed FPR95;
its paired confidence interval versus Energy still crosses zero.

FPR95 uses the threshold that accepts 95% of N. T false rejection uses a different,
score-specific threshold calibrated to accept 99% of the 8k RVL validation set.

## Repository map

| Path | Contents |
|---|---|
| [`results/README.md`](results/README.md) | Result index: reproduction, main study, and supplementary panel |
| [`docs/protocol.md`](docs/protocol.md) | Main frozen protocol |
| [`docs/supplementary_baselines_protocol.md`](docs/supplementary_baselines_protocol.md) | Four-method frozen addendum |
| [`configs/`](configs/) | Machine-readable experiment settings |
| [`src/document_ood_gate/`](src/document_ood_gate/) | Reproducible implementation |
| [`tests/`](tests/) | Automated checks |

Start with the [main development summary](results/single_signals/development_summary.md) and
the [supplementary result summary](results/supplementary_baselines/development_summary.md).

## Current boundary

- No metric has been computed on the sealed partition, and it has not been used for method
  selection or configuration.
- Learned combination Gates and Transformer layer scans have not been run.
- Raw documents, extracted features, model weights, credentials, and personal data are excluded
  from Git.

## Primary sources

- [RVL-CDIP dataset paper](https://adamharley.com/icdar15/harley_convnet_icdar15.pdf)
- [DiT paper and code](https://github.com/microsoft/unilm/tree/master/dit)
- [N/O benchmark paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4c0986bd04d747745beba3752bdf4d9d-Abstract-Datasets_and_Benchmarks.html)
- [Author N/O data release](https://github.com/gxlarson/RVL-CDIP-OOD)

## Local verification

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

All experiment commands are data-path driven; the source datasets remain outside the repository.
