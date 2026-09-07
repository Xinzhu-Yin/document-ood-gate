# Study design and interpretation

[Project overview](../README.md) · [Complete evidence](../results/README.md)

## Research question

Can post-hoc normality scores distinguish shifted documents that still belong to
the classifier's supported categories (N) from unsupported documents (O), while
limiting false rejection of familiar RVL-CDIP pages (T)?

The backbone is a frozen, 16-class DiT. This study evaluates existing OOD methods
and representation choices; it does not propose a new pretrained model. The
reusable implementation covers feature extraction, duplicate auditing, grouped
splitting, score fitting, uncertainty estimation, and result reporting.

## Recorded experimental design

| Component | Setting |
|---|---|
| Backbone | microsoft/dit-base-finetuned-rvlcdip |
| Model revision | `23f8b03d130fb66bbc9a15df3c75d753e49240eb` |
| Inference | Frozen weights, evaluation mode, float32 |
| Available/recovered N/O images | 1,002 N + 3,310 O = 4,312 |
| Initial development partition | 308 N + 986 O = 1,294 |
| Primary development comparison | 306 N + 974 O = 1,280 |
| Sealed test | 694 N + 2,324 O = 3,018 |
| Reference banks | Nested, class-balanced 8k / 32k / 80k RVL pages |
| Threshold validation | Independent balanced 8k RVL pages |
| T guardrail | Independent balanced 10k RVL pages |
| Bootstrap | 2,000 stratified resamples, seed 20260812 |

One 14-row identical-pixel group had contradictory N/O labels. It was excluded
from the primary development comparison by a label/pixel rule, while remaining
in the full-set reproduction and a separate sensitivity result. Confirmed
duplicate groups do not cross the development/sealed boundary.

## From reproduction to held-out evaluation

1. **Reproduce the baseline.** Compare MSP and Energy with the published targets.
   The 4,312-image rerun passed the frozen tolerance checks. A separate
   4,417-row sensitivity table uses author logits only for the 105 missing images.
2. **Audit and split.** Check byte and pixel identity, review perceptual-hash
   candidates, and preserve confirmed groups within the approximately 30/70 split.
3. **Compare single signals on development.** Evaluate output confidence and
   feature norm, kNN, and shrinkage Mahalanobis. Add a fixed supplementary panel
   of GEN, Relative Mahalanobis, ViM, and ReAct.
4. **Inspect representations.** Validate the feature path and a Mahalanobis
   reproduction gate, then compare raw patch mean, classifier-input feature
   `h_doc`, and CLS; report full Layer 1–12 curves.
5. **Isolate normalization.** Reuse cached development features to separate raw
   mean, LN-free, and LN-pooler effects. Residual-D sensitivity remains diagnostic.
6. **Freeze and evaluate once.** Record five replication predictions, then
   evaluate six D1 baselines plus the full 108-row D2 grid on the sealed set.

The D2 grid is 12 layers × 3 representations × 3 methods at a 32k reference size,
with k=10 and Residual D=512. Intermediate-layer LN-pooler features are analytical
probes using the final pooler's normalization, not the model's actual forward path.

## What the results support

| Observation | Interpretation | Evidence |
|---|---|---|
| Development ViM AUROC 0.7451 vs Energy 0.7282 | A development point-estimate gain; paired difference intervals cross zero | [Supplementary report](../results/reports/04_supplementary_baselines.md) |
| L12 Mahalanobis development AUROC 0.6567 raw vs 0.7447 normalized | Feature normalization materially changes the measured signal | [Ablation report](../results/reports/06_final_ablation.md) |
| Sealed Energy AUROC 0.7650 vs ViM 0.7473 | The development ordering does not persist in the sealed core panel | [Sealed table](../results/tables/07_sealed_test/sealed_baselines.csv) |
| Best sealed AUROC and best FPR95 belong to different configurations | Global ranking and a high-acceptance operating point answer different questions | [Prediction checks](../results/supporting/07_sealed_test/replication_predictions_evaluation.json) |

All five pre-specified replication predictions are recorded as supported.
They are development-informed replication checks, not blind predictions.
The complete grid is reported, including weak methods and layers; the best
sealed rows describe this fixed panel and are not evidence of a newly tuned gate.

## Metric interpretation

- **AUROC:** ranking separation between N and O, with N as the positive class and
  larger scores indicating greater normality. It is not 16-class accuracy.
- **FPR95:** the fraction of O accepted at the first achieved ROC point accepting
  at least 95% of N. It is a benchmark operating-point statistic, not a threshold
  learned solely on development and proven ready for deployment.
- **T false rejection:** measured at a different threshold, calibrated to accept
  99% of an independent RVL validation set. It should not be read as a measurement
  at the FPR95 threshold.

The reproduction code retains both interpolated and first-achieved FPR95
conventions. Final sealed results use first-achieved FPR95. Bootstrap intervals
describe uncertainty under the recorded resampling scheme; differences in point
estimates alone do not establish statistical superiority.

## Protocol record and deviations

The original protocols are retained as historical records. Statements such as
"the sealed partition has not been scored" describe their dated checkpoint,
not the present repository. The final sealed round is governed by the
[sealed protocol](../configs/sealed_test_protocol.md), with
[predictions](../configs/sealed_test_predictions.md) fixed beforehand.

- **Freeze publication:** local commit
  `451fbdb4e902e75843e8e4ef5a31745cc95f1875` was recorded before sealed
  access. The planned pre-access GitHub push did not happen because command-line
  credentials were unavailable. Its byte-identical tree
  `7d12bfd0dd46c5055ec2428bf93094b6188bb157` was later published as
  [remote freeze commit](https://github.com/Xinzhu-Yin/document-ood-gate/commit/d0d058bdb7abd3ff719a0553aca58580823bb714).
  This is a local freeze with later publication, not prior public registration.
- **Implementation repair:** the record describes a sealed-label string-array
  serialization repair before scores existed, with no experimental-setting change.
- **Shortcut audit:** the original protocol called for source/rendering
  diagnostics, but a full shortcut audit was not completed. Source-preserving
  splitting does not rule out acquisition or rendering shortcuts.
- **Threshold and guardrail scope:** existing `h_doc` T results remain available.
  New layer probes did not re-extract validation/T pages; the final round reports
  the frozen D1/D2 AUROC/FPR95 panel rather than a full deployment-threshold study.

See the [run metadata](../results/supporting/07_sealed_test/run_metadata.json)
and [results record](../results/README.md) for timestamps and execution counts.
The remote commit supports the published tree identity; local timing is reported
from the retained project record.

## Limits

No evaluated configuration meets the intended reliable-rejection goal: even the
lowest sealed FPR95 is 0.8171. There is no validated production threshold, routing
service, or deployment evaluation in this repository.

The [data audit](DATA.md) records 395 calibrated rerenders and 105 unavailable O
images. Model weights, raw data, and feature caches are not bundled, so the repository
supports inspection of saved aggregate evidence without claiming turnkey,
bit-for-bit reproduction from a fresh clone.

The final panel excludes new methods, 8k sealed sensitivity, a sealed CLS scan,
and D tuning. No new experiment is part of the presentation cleanup.

## Original protocol map

| Stage | Record |
|---|---|
| Baselines, audit, split, single signals | [Protocol](../configs/protocol.md), [settings](../configs/experiment.yaml) |
| Supplementary methods | [Protocol](../configs/supplementary_protocol.md), [settings](../configs/supplementary_baselines.yaml) |
| Layer scan | [Protocol](../configs/layer_scan_protocol.md), [settings](../configs/layer_scan.yaml) |
| Final evaluation | [Protocol](../configs/sealed_test_protocol.md), [settings](../configs/final_experiment.yaml) |

[References and acknowledgements](REFERENCES.md)
