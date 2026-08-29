# Frozen DiT Document OOD Gate

Reproducible evaluation of post-hoc OOD signals for a frozen document classifier.

## Research question

Which single signal can pass familiar documents (`T`) and novel-but-supported documents (`N`), while rejecting unsupported documents (`O`), more reliably than MSP and Energy—without retraining DiT?

The primary study compares single signals. A learned combination Gate is deliberately out of scope until the single-signal results have been reviewed and a separate protocol has been frozen.

## Hard execution gate

Before any main experiment, the project must reproduce the published full-set DiT N–O baselines on all 1,002 N and 3,415 O documents:

| Metric | MSP | Energy |
|---|---:|---:|
| micro AUROC | 0.728 | 0.753 |
| micro FPR95 | 0.847 | 0.843 |

Green: `|ΔAUROC| <= 0.01` and `|ΔFPR95| <= 0.02`.  
Yellow: larger but explainable implementation differences within the frozen protocol bounds.  
Red: `|ΔAUROC| > 0.02` or `|ΔFPR95| > 0.04`; stop and diagnose.

The current author Drive is incomplete: 2,915 of 3,415 O TIFFs remain. We recovered
395 pages from author-hosted source PDFs using a calibrated first-page rendering path;
105 images have no available source. The strict image rerun therefore covers 4,312 of
4,417 N/O pages. A full-set sensitivity result uses the author-provided logits only for
those final 105 rows and marks that provenance explicitly.

## Current results

The mandatory reproduction gate passed green. The independent 4,312-image rerun and
the 4,417-row provenance-marked sensitivity analysis both reproduce the published MSP
and Energy results within the frozen tolerance.

| Scope | Signal | AUROC | FPR95 |
|---|---|---:|---:|
| Published target | MSP | 0.728 | 0.847 |
| Published target | Energy | 0.753 | 0.843 |
| 4,312-image rerun | MSP | 0.729 | 0.847 |
| 4,312-image rerun | Energy | 0.753 | 0.844 |
| 4,417-row sensitivity | MSP | 0.728 | 0.847 |
| 4,417-row sensitivity | Energy | 0.753 | 0.844 |

The duplicate-aware research split contains 30% development and 70% sealed test. The
sealed-test scores have not been computed. On the development partition, the strongest
single-signal AUROC is Mahalanobis with an 80k RVL reference bank (0.746); the lowest
FPR95 is Mahalanobis with a 32k bank (0.816). For comparison, development Energy is
0.728 AUROC / 0.861 FPR95. All tested signals reject about 0.9%--1.2% of the independent
10k T guardrail at thresholds fixed for 99% RVL validation acceptance.

These are model-selection results, not final claims. The combination Gate remains
unimplemented and the sealed test remains untouched.

### Supplementary post-hoc panel (development only)

Four frozen additions were run on the same 306 N / 974 O development rows. The run uses
a physically isolated 1,294-row development cache, the same 32k RVL reference bank, the
same 8k threshold set, and the report-only 10k T guardrail. The 3,018-row sealed test was
not loaded, inferred, scored, or used for configuration.

| Signal | AUROC | FPR95 | T false reject |
|---|---:|---:|---:|
| Energy (existing comparator) | 0.7282 | 0.8614 | 1.13% |
| Negative entropy (existing comparator) | 0.7148 | 0.8480 | 1.20% |
| Mahalanobis 32k (existing comparator) | 0.7447 | 0.8162 | 0.89% |
| GEN (gamma=0.1, M=16) | 0.7173 | 0.8624 | 1.06% |
| Relative Mahalanobis 32k | 0.7328 | 0.8655 | 0.93% |
| ViM 32k (DIM=512) | 0.7451 | 0.7988 | 1.00% |
| ReAct p90 + Energy | 0.5436 | 0.8819 | 1.08% |

The internal Energy/Mahalanobis reproduction gate passed exactly before the four additions
ran. See the [readable result summary](results/supplementary_baselines/development_summary.md),
[CSV table](results/supplementary_baselines/development_results.csv),
[complete JSON record](results/supplementary_baselines/development_results.json), and
[frozen supplementary protocol](docs/supplementary_baselines_protocol.md).

## Frozen scope

- Model: `microsoft/dit-base-finetuned-rvlcdip`, inference only.
- Reproduction: full official N/O, no augmentation.
- Duplicate audit: raw SHA256, normalized-pixel SHA256, perceptual hash candidates, manual confirmation before the research split.
- Research split: approximately 30% development / 70% sealed test, grouped by confirmed duplicate cluster.
- Primary signals: MSP, Energy, maximum logit, negative entropy, top-2 margin, feature norm, kNN similarity, and Mahalanobis distance.
- Guardrail: 10,000 balanced RVL-CDIP test pages; never used for model selection.
- No raw document image, model weight, PII, or secret is committed.

See [`docs/protocol.md`](docs/protocol.md) for the main executable protocol and
[`docs/supplementary_baselines_protocol.md`](docs/supplementary_baselines_protocol.md) for
the four-method addendum.

## Local setup

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

Commands are intentionally data-path driven; source datasets remain outside Git.
