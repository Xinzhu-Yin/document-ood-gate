# Document OOD Gate for a Frozen DiT Classifier

Can a lightweight gate accept supported documents and reject documents outside a
classifier's 16 known categories—without retraining the classifier?

This project freezes
[`microsoft/dit-base-finetuned-rvlcdip`](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip)
and compares post-hoc OOD scores.

- **T** — familiar RVL-CDIP documents; used only to monitor false rejection.
- **N** — shifted documents that still belong to one of the 16 supported classes; should pass.
- **O** — documents outside all 16 supported classes; should be rejected.

## Current status

- Reproduced the published MSP and Energy N-vs-O baselines.
- Audited duplicates and created a duplicate-group-aware 30% development / 70% sealed split.
- Compared 12 single signals and four fixed supplementary methods on development data.
- Kept the 3,018-image sealed test untouched; no final test claim has been made.

## Headline results

Published-baseline reproduction:

| Scope | MSP AUROC / FPR95 | Energy AUROC / FPR95 |
|---|---:|---:|
| Published, 4,417 rows | 0.7280 / 0.8470 | 0.7530 / 0.8430 |
| Independent image rerun, 4,312 images | 0.7286 / 0.8465 | 0.7533 / 0.8444 |

Selected development results on 306 N + 974 O:

| Signal | AUROC ↑ | FPR95 ↓ | T false rejection ↓ |
|---|---:|---:|---:|
| Energy | 0.7282 | 0.8614 | 1.13% |
| Mahalanobis, 32k reference | 0.7447 | 0.8162 | 0.89% |
| Mahalanobis, 80k reference | **0.7463** | 0.8306 | 0.92% |
| ViM, 32k reference | 0.7451 | **0.7988** | 1.00% |

The lowest observed FPR95 is still 0.7988: at the threshold that accepts 95% of development
N, about 79.88% of O also passes. The current methods therefore show measurable ranking
signal, but not yet a deployable Gate.

## Read the project

1. [Experimental design](EXPERIMENT_DESIGN.md)
2. [Complete results](EXPERIMENT_RESULTS.md)

## Primary sources

- [RVL-CDIP dataset paper](https://adamharley.com/icdar15/harley_convnet_icdar15.pdf)
- [DiT paper and code](https://github.com/microsoft/unilm/tree/master/dit)
- [N/O benchmark paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4c0986bd04d747745beba3752bdf4d9d-Abstract-Datasets_and_Benchmarks.html)
- [Author N/O data release](https://github.com/gxlarson/RVL-CDIP-OOD)

Raw documents, model weights, extracted features, and the reproducible implementation are kept
in the local research archive rather than on this presentation-focused branch.
