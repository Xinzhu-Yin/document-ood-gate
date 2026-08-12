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
Red: `|ΔAUROC| > 0.03` or `|ΔFPR95| > 0.05`; stop and diagnose.

## Frozen scope

- Model: `microsoft/dit-base-finetuned-rvlcdip`, inference only.
- Reproduction: full official N/O, no augmentation.
- Duplicate audit: raw SHA256, normalized-pixel SHA256, perceptual hash candidates, manual confirmation before the research split.
- Research split: approximately 30% development / 70% sealed test, grouped by confirmed duplicate cluster.
- Primary signals: MSP, Energy, maximum logit, negative entropy, top-2 margin, feature norm, kNN similarity, and Mahalanobis distance.
- Guardrail: 10,000 balanced RVL-CDIP test pages; never used for model selection.
- No raw document image, model weight, PII, or secret is committed.

See [`docs/protocol.md`](docs/protocol.md) for the executable protocol.

## Local setup

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest
```

Commands are intentionally data-path driven; source datasets remain outside Git.

