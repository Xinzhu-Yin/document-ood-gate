# Experimental Design

## 1. Research question

With the downstream DiT classifier frozen, which post-hoc score best separates N (pass) from O
(reject) while keeping false rejection on familiar T documents low?

The Gate decides only whether an input should enter the downstream system. Correct 16-class
classification remains the downstream model's responsibility.

## 2. Frozen setup

| Item | Setting |
|---|---|
| Classifier | `microsoft/dit-base-finetuned-rvlcdip` |
| Checkpoint revision | `23f8b03d130fb66bbc9a15df3c75d753e49240eb` |
| Model use | Frozen, evaluation only, float32 |
| Seed | `20260812` |
| Score direction | Higher means more normal / more likely to pass |
| Primary metrics | Micro AUROC and FPR95, with N as positive |
| Uncertainty | 2,000 paired stratified bootstrap resamples |
| Final evaluation | One use of the sealed split after the protocol is frozen |

## 3. Data

### N/O benchmark

The target release contains 1,002 N and 3,415 O images. The live author release currently
contains 2,915 O TIFFs. Another 395 pages were reconstructed from author-hosted source PDFs,
leaving 105 O source images unavailable.

| Use | N | O | Total |
|---|---:|---:|---:|
| Independent image inference | 1,002 | 3,310 | 4,312 |
| Full output-score sensitivity | 1,002 | 3,415 | 4,417 |

Only the full output-score sensitivity result fills the unavailable 105 rows with the authors'
published logits. Those rows never enter representation-based experiments. The 395 reconstructed
pages are close rerenders, not byte-identical copies of the original TIFFs.

### RVL-CDIP reference data

The 16 classes are sampled evenly, with no overlap between purposes:

| Purpose | Images | Role |
|---|---:|---|
| Nested reference banks | 80,000 | 8k, 32k, and 80k banks for representation scores |
| Threshold validation | 8,000 | Set each score's 99%-T-style acceptance threshold |
| T guardrail | 10,000 | Report false rejection only; never select a method |

## 4. Execution order

### Stage A — mandatory reproduction gate

First reproduce the published full-set MSP and Energy targets:

| Signal | Published AUROC | Published FPR95 |
|---|---:|---:|
| MSP | 0.728 | 0.847 |
| Energy | 0.753 | 0.843 |

- Green: `|ΔAUROC| ≤ 0.01` and `|ΔFPR95| ≤ 0.02`.
- Red: `|ΔAUROC| > 0.02` or `|ΔFPR95| > 0.04`.
- An intermediate difference may continue only after a concrete implementation cause is recorded.

No main experiment proceeds after a red result.

### Stage B — duplicate audit and frozen split

1. Raw-file SHA256 finds byte-identical files.
2. Decoded-pixel SHA256 finds images with identical visual pixels.
3. pHash proposes near-duplicate candidates.
4. pHash-only candidates require confirmation before grouping.
5. Confirmed groups remain entirely within one split.

The duplicate-group-aware split contains:

| Split | N | O | Total |
|---|---:|---:|---:|
| Initial development | 308 | 986 | 1,294 |
| Sealed test | 694 | 2,324 | 3,018 |

One identical-pixel group carries contradictory N and O labels (14 rows: 2 N and 12 O). It
remains in the paper reproduction but is excluded from the primary development comparison,
which therefore uses 306 N + 974 O.

### Stage C — single-signal comparison

- Output scores: MSP, Energy, maximum logit, negative entropy, and top-2 margin.
- Representation scores: feature norm, kNN (`k=10`), and shrinkage Mahalanobis.
- kNN and Mahalanobis use the same frozen 768-dimensional `h_doc` representation and the same
  nested 8k/32k/80k reference banks.

### Stage D — bounded supplementary panel

Exactly four post-hoc additions were run on the same development data and a fixed 32k bank:

| Method | Frozen setting |
|---|---|
| GEN | `gamma=0.1`; requested `M=100`, effective `M=16` for 16 classes |
| Relative Mahalanobis | Ledoit-Wolf class and background covariance |
| ViM | 512-dimensional principal space and 256-dimensional residual space |
| ReAct + Energy | Global 90th-percentile upper clipping; temperature 1 |

Before this panel, cached Energy and Mahalanobis-32k results had to reproduce the earlier
development values within `1e-8`.

## 5. Metrics and thresholds

- **AUROC:** ranking quality over every possible threshold; higher is better. A value near 0.5
  is approximately random ranking.
- **FPR95:** choose the threshold that accepts at least 95% of N, then report the fraction of O
  incorrectly accepted; lower is better.
- **T false rejection:** for each score, set a separate threshold to accept 99% of the 8,000
  RVL validation pages, then report rejection on the independent 10,000-page T guardrail.

FPR95 and T false rejection use different thresholds and must not be interpreted as the same
operating point.

## 6. Leakage controls and current boundary

- Model outputs and Gate scores were not used to create duplicate groups or the split.
- The supplementary panel used a physically isolated development-only feature cache.
- The 3,018-row sealed split was not loaded, inferred, scored, or used for configuration.
- T is report-only and does not select a method.
- File format, dimensions, source, and other metadata are not Gate inputs.
- A full shortcut audit, learned score fusion, Transformer layer scans, and sealed evaluation
  have not yet been run.
