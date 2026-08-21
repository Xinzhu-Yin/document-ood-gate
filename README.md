# Frozen DiT Document OOD Gate

[![CI](https://github.com/Xinzhu-Yin/document-ood-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/Xinzhu-Yin/document-ood-gate/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Study status](https://img.shields.io/badge/study-development%20checkpoint-2F80ED)

A reproducible study of post-hoc OOD Gates for a frozen document classifier. The goal is
to pass supported documents (`T` and `N`) while rejecting unsupported documents (`O`),
without retraining DiT.

## Project status — 2026-08-21

| Milestone | Status |
|---|---|
| Published MSP/Energy reproduction | ✅ Passed |
| Duplicate audit and grouped 30/70 split | ✅ Complete |
| Development-only single-signal comparison | ✅ Complete |
| Shortcut/source audit | ⏸ Pending; not part of this presentation-only step |
| Combination Gate | ⏳ Awaiting a separate decision; no parameters frozen |
| 3,018-row sealed test | 🔒 Not scored |

The repository is at a development checkpoint. No final method has been selected and no
sealed-test claim has been made.

## Research question

Which single output or representation signal separates `N` from `O` more reliably than
MSP and Energy, while keeping ordinary `T` false rejections low?

The classifier is fixed at
[`microsoft/dit-base-finetuned-rvlcdip`](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip)
revision `23f8b03d130fb66bbc9a15df3c75d753e49240eb`. Only the Gate changes.

## T, N, and O at a glance

| Set | What it contains | Expected Gate action | Project use |
|---|---|---|---:|
| `T` | Standard RVL-CDIP test pages from the supported 16-class task | Pass | 10,000-page guardrail |
| `N` | New-source or new-style pages that still belong to supported classes | Pass | 1,002 pages |
| `O` | Pages outside the supported 16 classes | Reject | 3,415 expected; 3,310 images available/recovered |

The primary comparison is `N` versus `O`. `T` checks whether a Gate disrupts familiar
traffic; downstream 16-class classification accuracy is not the Gate target.

## Experimental flow

```text
Published baseline reproduction
        ↓
Duplicate audit and grouped 30/70 split
        ↓
30% development: compare single signals
        ↓
Freeze the next protocol decision
        ↓
70% sealed test: evaluate once
```

The final step has not been run.

## Reproduction checkpoint

The author release currently lacks 105 of the expected O images. The independent rerun
therefore uses 1,002 N and 3,310 O images. A separate 4,417-row sensitivity analysis uses
author-provided logits only for the missing 105 rows and labels that provenance explicitly.

| Scope | Signal | AUROC ↑ | FPR95 ↓ |
|---|---|---:|---:|
| Published target | MSP | 0.728 | 0.847 |
| 4,312-image rerun | MSP | 0.729 | 0.847 |
| Published target | Energy | 0.753 | 0.843 |
| 4,312-image rerun | Energy | 0.753 | 0.844 |

The mandatory reproduction gate passed its frozen green tolerance.

## Development-only checkpoint

These results use 306 N and 974 O development rows. The 3,018-row sealed partition was
not scored.

| Signal | Reference bank | AUROC ↑ | AUROC 95% CI | FPR95 ↓ | T false rejection* ↓ |
|---|---:|---:|---:|---:|---:|
| Energy | — | 0.728 | 0.694–0.761 | 0.861 | 1.13% |
| kNN, `k=10` | 80k | 0.743 | 0.707–0.777 | 0.851 | 0.89% |
| Mahalanobis | 32k | 0.745 | 0.711–0.778 | **0.816** | 0.89% |
| Mahalanobis | 80k | **0.746** | 0.714–0.779 | 0.831 | 0.92% |

The improvements are modest and the confidence intervals overlap. These are candidate-
selection results, not final claims.

\* `FPR95` and `T false rejection` use different thresholds. FPR95 asks how much O passes
when 95% of N is accepted. The T guardrail uses a separate threshold calibrated to accept
99% of 8,000 RVL validation pages.

## Quick links

- [Results index](results/README.md)
- [Complete 12-row development table](results/single_signals/development_summary.md)
- [Independent 4,312-image reproduction](results/reproduction/actual_images_4312.json)
- [Data release and recovery status](results/data_release_status.json)
- [Duplicate and split audit](results/duplicate_audit/duplicate_audit_summary.json)
- [Frozen protocol v1.1](docs/protocol.md)
- [Current decision checkpoint](docs/decisions/2026-08-21-development-checkpoint.md)

## Reproducibility and safety

- Reference-bank sizes are nested at 8k, 32k, and 80k RVL training pages.
- Threshold calibration uses 8,000 separate RVL validation pages.
- The `T` guardrail uses 10,000 balanced RVL test pages and is never used for selection.
- Raw documents, model weights, extracted features, PII, and secrets are excluded from Git.
- Curated aggregate results, hashes, seeds, and provenance records are committed.

## Local checks

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest -q
```

GitHub Actions runs the same lint and unit-test checks without accessing private research
data.

## Primary sources

- [RVL-CDIP dataset paper](https://adamharley.com/icdar15/harley_convnet_icdar15.pdf)
- [Frozen DiT checkpoint](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip)
- [NeurIPS 2022 N/O benchmark paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4c0986bd04d747745beba3752bdf4d9d-Abstract-Datasets_and_Benchmarks.html)
- [Author RVL-CDIP-OOD release](https://github.com/gxlarson/RVL-CDIP-OOD)
