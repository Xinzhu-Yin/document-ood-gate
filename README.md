# Document OOD Gate for a Frozen DiT Classifier

## Abstract

Modern document classifiers perform well on familiar benchmarks but can still be highly
confident on unsupported documents. This project adds a post-hoc Gate in front of a frozen
16-class DiT classifier: supported documents should pass, while documents outside the supported
classes should be rejected. DiT is never retrained.

The study first reproduces the published MSP and Energy baselines, then compares output-level
confidence scores with representation-level methods such as kNN, Mahalanobis, and ViM under one
duplicate-aware, development-only protocol.

## Problem definition

| Set | Meaning | Desired Gate action |
|---|---|---|
| **T** | Familiar RVL-CDIP documents | Pass; used only as a false-rejection guardrail |
| **N** | Shifted documents that still belong to one of the 16 supported classes | Pass |
| **O** | Documents outside all 16 supported classes | Reject |

The Gate decides only whether an input should enter the downstream classifier. Whether DiT then
predicts the correct one of the 16 classes is a separate downstream task.

## Experimental design

| Item | Frozen setting |
|---|---|
| Classifier | [`microsoft/dit-base-finetuned-rvlcdip`](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip), revision `23f8b03d...`, float32 inference |
| N/O release | 1,002 N + 3,415 O target rows |
| Independent image rerun | 1,002 N + 3,310 O = 4,312 available/recovered images |
| Development comparison | 306 N + 974 O after removing one exact-pixel label-conflict group |
| Sealed test | 694 N + 2,324 O = 3,018 rows; not evaluated |
| RVL reference banks | Nested 8k / 32k / 80k, balanced over 16 classes |
| Threshold validation | Independent 8k RVL pages, 99% acceptance per score |
| T guardrail | Independent balanced 10k RVL pages, report-only |
| Primary metrics | AUROC ↑, FPR95 ↓, T false-rejection rate ↓ |
| Uncertainty | 2,000 stratified bootstrap resamples; seed `20260812` |

Execution order:

1. **Reproduction gate:** reproduce the paper's full-set MSP and Energy results before any new
   method is evaluated.
2. **Data audit:** detect byte-identical, pixel-identical, and confirmed near-duplicate images;
   keep each confirmed group within one split.
3. **Frozen split:** use approximately 30% for development and leave 70% sealed.
4. **Single signals:** compare five output scores, feature norm, kNN, and Mahalanobis.
5. **Supplementary panel:** evaluate fixed GEN, Relative Mahalanobis, ViM, and ReAct settings.

Exact parameter choices and leakage controls are recorded in
[`configs/protocol.md`](configs/protocol.md) and
[`configs/supplementary_protocol.md`](configs/supplementary_protocol.md).

## Latest results

### Published-baseline reproduction

| Scope | MSP AUROC / FPR95 | Energy AUROC / FPR95 |
|---|---:|---:|
| Published target, 4,417 rows | 0.7280 / 0.8470 | 0.7530 / 0.8430 |
| Independent rerun, 4,312 images | 0.7286 / 0.8465 | 0.7533 / 0.8444 |

**Reproduction gate: passed.** The remaining 105 O source images are unavailable in the live
author release. Author-provided logits are used only in a separately marked 4,417-row
output-score sensitivity result, never in representation-based experiments.

### Selected development results

| Signal | AUROC ↑ | FPR95 ↓ | T false rejection ↓ |
|---|---:|---:|---:|
| Energy | 0.7282 | 0.8614 | 1.13% |
| kNN, 80k reference | 0.7433 | 0.8511 | 0.89% |
| Mahalanobis, 32k reference | 0.7447 | 0.8162 | 0.89% |
| Mahalanobis, 80k reference | **0.7463** | 0.8306 | 0.92% |
| ViM, 32k reference | 0.7451 | **0.7988** | 1.00% |

Representation-based scores have slightly better development point estimates than the output
confidence baselines. However, the best observed FPR95 is still 0.7988: when 95% of N is
accepted, approximately 79.88% of O is also accepted. The current methods therefore do not yet
form a deployable Gate, and overlapping confidence intervals prevent a final winner claim.

See [`results/`](results/) for the complete tables, confidence intervals, paired comparisons,
duplicate audit, provenance, and machine-readable outputs.

## Repository structure

```text
.
├── README.md          # Background, design, headline results, and reproduction entry point
├── configs/           # Frozen YAML settings and executable experimental protocols
├── results/           # Complete human-readable and machine-readable experiment outputs
├── src/               # Reusable Python implementation and command-line pipeline
├── tests/             # Metric, data-isolation, and pipeline regression tests
├── pyproject.toml     # Exact Python package and dependency specification
└── .gitignore         # Keeps raw data, features, model weights, and credentials out of Git
```

## Reproduction

### 1. Environment

Python 3.12 is required. From the repository root:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

The dependency versions, including PyTorch, Transformers, scikit-learn, Pillow, and pytest, are
pinned in [`pyproject.toml`](pyproject.toml).

### 2. Data

Download RVL-CDIP and the authors' RVL-CDIP-N/O release from the links in the References section.
Raw images are not redistributed here because of size and source-distribution constraints. All
commands accept explicit input paths; no personal path is hard-coded in the source.

### 3. Pipeline

The installed `doc-ood` command exposes every experimental stage:

```bash
doc-ood --help
doc-ood <stage> --help
```

| Stage | CLI command |
|---|---|
| Extract frozen DiT N/O logits and features | `extract-ood` |
| Reproduce MSP and Energy | `reproduce` / `reproduce-hybrid` |
| Audit and confirm duplicates | `audit-duplicates` / `finalize-duplicate-audit` |
| Create the grouped 30/70 split | `split-research` |
| Sample and extract RVL references | `sample-rvl` / `extract-rvl-sample` / `extract-rvl-features` |
| Evaluate the main single-signal panel | `evaluate-development` / `evaluate-reference-signals` |
| Evaluate the fixed supplementary panel | `extract-development-ood` / `evaluate-supplementary-baselines` |

Use [`configs/experiment.yaml`](configs/experiment.yaml) and
[`configs/supplementary_baselines.yaml`](configs/supplementary_baselines.yaml) for the frozen
values. Each command validates counts and refuses incompatible inputs; the supplementary command
also rejects a manifest containing sealed rows before opening its OOD feature cache.

## Current limits

- New methods have been compared only on the 306 N + 974 O development set.
- The 3,018-row sealed test has not been loaded, inferred, or scored.
- The 395 recovered O pages are calibrated rerenders, not byte-identical original TIFFs.
- A full shortcut audit, learned fusion Gate, and Transformer layer scan have not been run.

## References

- [Harley et al. — RVL-CDIP (ICDAR 2015)](https://adamharley.com/icdar15/harley_convnet_icdar15.pdf)
- [Li et al. — DiT: Self-Supervised Pre-Training for Document Image Transformer](https://github.com/microsoft/unilm/tree/master/dit)
- [Larson et al. — Evaluating Out-of-Distribution Performance on Document Image Classifiers](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4c0986bd04d747745beba3752bdf4d9d-Abstract-Datasets_and_Benchmarks.html)
- [Larson et al. — RVL-CDIP-N/O data and released logits](https://github.com/gxlarson/RVL-CDIP-OOD)
- [Hendrycks & Gimpel — MSP baseline](https://arxiv.org/abs/1610.02136)
- [Liu et al. — Energy-based OOD Detection](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html)
- [Lee et al. — Mahalanobis OOD Detection](https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html)
- [Wang et al. — ViM](https://github.com/haoqiwang/vim)
- [Ren et al. — Relative Mahalanobis](https://arxiv.org/abs/2106.09022)
- [Sun et al. — ReAct](https://proceedings.neurips.cc/paper_files/paper/2021/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html)
- [Liu et al. — GEN implementation](https://github.com/XixiLiu95/GEN)
