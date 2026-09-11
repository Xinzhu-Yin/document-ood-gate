# Study design and evidence

[Project overview](../README.md) · [Results](../results/README.md) · [Reproduction guide](REPRODUCIBILITY.md)

## Research question

Can scores from a frozen document classifier distinguish unfamiliar pages that
still belong to its supported categories from pages outside those categories?
The study evaluates existing OOD methods and representation choices. Its
contribution is the data audit, evaluation pipeline, and empirical analysis.

| Group | Documents | Evaluation role |
|---|---|---|
| T | Familiar RVL-CDIP pages | Monitor false rejection |
| N | Shifted pages in supported categories | Should be accepted |
| O | Pages outside all 16 supported categories | Should be rejected |

The proposed gate scores model outputs or features after DiT inference. Correctness
of an accepted 16-class prediction is a separate classification question.

## Experimental design

| Component | Recorded setting |
|---|---|
| Backbone | `microsoft/dit-base-finetuned-rvlcdip`, frozen, evaluation mode, float32 |
| Model revision | `23f8b03d130fb66bbc9a15df3c75d753e49240eb` |
| Available/recovered images | 1,002 N + 3,310 O = 4,312 |
| Initial development partition | 308 N + 986 O = 1,294 |
| Primary development comparison | 306 N + 974 O = 1,280 |
| Sealed test | 694 N + 2,324 O = 3,018 |
| Reference banks | Nested, class-balanced 8k / 32k / 80k RVL pages |
| Threshold validation / T guardrail | Independent balanced 8k / 10k RVL pages |
| Bootstrap | 2,000 stratified resamples, seed 20260812 |

Byte/pixel hashes and reviewed perceptual-hash candidates identify duplicates;
confirmed groups stay within one partition. A 14-row identical-pixel group with
contradictory N/O labels was removed from the primary development comparison by
a label/pixel rule. It remains in the full-set reproduction and a separate sensitivity result.

The study first reproduced MSP/Energy baselines, then compared single signals,
supplementary methods, and layer/readout choices on development data. Fixed
normalization ablations preceded one sealed evaluation of **six D1 baselines and
108 D2 configurations**: 12 layers × 3 readouts × 3 methods, using 32k reference
pages, kNN k=10, and residual D=512.

The readouts are raw patch mean, **LN-free** (LayerNorm without learned affine
parameters), and **LN-pooler** (the model's final pooler normalization). Applying
LN-pooler at intermediate layers is an analytical probe, not the actual forward
path. The grid's residual score uses reference-mean PCA centring; the separate
ViM-origin residual baseline uses the classifier-origin convention.

## Interpretation

The layer curves support stronger OOD separation in middle-to-late layers.
Normalization accounts for a substantial late-layer gain: sealed Layer 12
Mahalanobis AUROC rises from 0.6458 to 0.7455 with LN-free, and mean-centred PCA
residual rises from 0.6571 to 0.7449. This is a change in feature-based detection,
not evidence of improved category accuracy.

Development gains do not guarantee the same held-out ordering: ViM exceeds Energy
on development (0.7451 vs 0.7282 AUROC), but its paired difference intervals cross
zero. On the sealed set, Energy has the highest AUROC point estimate across the
complete panel (0.7650). The lowest FPR95 belongs to Layer 9 LN-pooler Mahalanobis
(0.8171), whose AUROC is 0.6607. Ranking quality and a high-acceptance operating
point therefore favour different configurations.

All five [pre-specified replication checks](../configs/sealed_test_predictions.md)
were recorded as supported. They were informed by development results. The
[complete tables and curves](../results/README.md) include weak layers and methods;
best sealed rows describe the frozen panel and were not used for further tuning.

## Metrics

- **AUROC:** ranking separation of N and O, with N positive and larger scores
  indicating greater normality. It is not classification accuracy.
- **FPR95:** fraction of O accepted at the first achieved ROC point accepting at
  least 95% of N. This is a benchmark statistic, not a development-calibrated
  deployment threshold. The reproduction also retains an interpolated convention.
- **T false rejection:** measured at a separate threshold calibrated to accept
  99% of the independent RVL validation set, not at the FPR95 threshold.

The final results use first-achieved FPR95 and stratified bootstrap intervals.
Differences in point estimates alone do not establish statistical superiority.

## Data and provenance

| Material | Source and recorded identifier |
|---|---|
| DiT checkpoint | [Microsoft](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip); revision above |
| RVL-CDIP | [Dataset repository](https://huggingface.co/datasets/aharley/rvl_cdip); revision `03f14a4ad0a32413eff51ca10f9f511545f2bd5b` |
| RVL-CDIP-N/O and baseline logits | [Authors' release](https://github.com/gxlarson/RVL-CDIP-OOD); audited 2026-08-12 |

The release targets 1,002 N and 3,415 O pages. At the audit, all N pages and 2,915 O
TIFFs were available; 395 O pages were recovered from author-hosted PDFs using
calibrated rendering, leaving 105 O images unavailable. Recovered pages are close
reconstructions, not byte-identical original TIFFs. See the
[release audit](../results/supporting/02_data_audit/data_release_status.json).

The independent image rerun therefore contains **4,312 images**. Author logits
fill the 105 missing rows only in the separately labelled
[4,417-row output-score sensitivity](../results/tables/01_reproduction/hybrid_full_4417.json);
those rows never enter representation experiments. The
[RVL provenance record](../results/supporting/02_data_audit/reference_provenance.json)
retains archive size, checksums, manifest checksum, and sampling seed. These
identify recorded inputs without guaranteeing future upstream availability.

The repository includes code, frozen configurations, aggregate results, plots,
metadata, and duplicate-review decisions. Duplicate records retain source identifiers
but no document pixels. Raw images/PDFs, per-image features/logits, full runtime
manifests, and weights require separate access and preparation.

Third-party datasets, models, and methods remain subject to their creators' terms;
this project grants no redistribution rights over them. The repository currently
has no project-wide license file.

## Protocol record and deviations

- **Freeze publication:** local commit `451fbdb4e902e75843e8e4ef5a31745cc95f1875`
  was recorded before sealed access. The planned pre-access GitHub push failed
  because command-line credentials were unavailable. Its byte-identical tree
  `7d12bfd0dd46c5055ec2428bf93094b6188bb157` was later published as
  [remote commit d0d058b](https://github.com/Xinzhu-Yin/document-ood-gate/commit/d0d058bdb7abd3ff719a0553aca58580823bb714).
  This was a local freeze with later publication, not prior public registration.
- **Implementation repair:** a sealed-label string-array serialization repair
  preceded score creation, with no experimental-setting change, according to the record.
- **Shortcut audit:** a full planned source/rendering audit was not completed.
  Balancing source composition and grouping confirmed duplicates does not rule
  out acquisition or rendering shortcuts.
- **Guardrail scope:** existing classifier-input (`h_doc`) T results are retained.
  New layer probes did not re-extract validation/T pages; the final round reports
  AUROC/FPR95 rather than a full deployment-threshold evaluation.

The [run metadata](../results/supporting/07_sealed_test/run_metadata.json) records
one frozen forward pass per sealed image, no reference re-extraction, and no
post-sealed experimental iteration. Local timing comes from the retained project
record; the remote commit establishes the published tree identity. The final panel
excludes new methods, 8k sealed sensitivity, a sealed CLS scan, and residual-D tuning.

Even the lowest sealed FPR95 is 81.71%, so the study does not establish a reliable
automatic rejection rule. A production routing service and deployment validation
are outside the delivered evaluation pipeline's scope.

## Original protocol map

These files preserve dated research decisions. Statements such as "the sealed
partition has not been scored" refer to the checkpoint when they were written.

| Stage | Protocol | Settings |
|---|---|---|
| Baselines, audit, split, single signals | [Protocol](../configs/protocol.md) | [YAML](../configs/experiment.yaml) |
| Supplementary methods | [Protocol](../configs/supplementary_protocol.md) | [YAML](../configs/supplementary_baselines.yaml) |
| Layer scan | [Protocol](../configs/layer_scan_protocol.md) | [YAML](../configs/layer_scan.yaml) |
| Final evaluation | [Protocol](../configs/sealed_test_protocol.md) | [YAML](../configs/final_experiment.yaml) |

## References

The model, benchmark, and OOD methods build on the following prior work:

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
