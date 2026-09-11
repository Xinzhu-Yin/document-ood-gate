# Results and evidence

[Project overview](../README.md) · [Study and metric definitions](../docs/STUDY.md)

The final evaluation is in stage 7. Earlier stages retain the development evidence
that led to its fixed design. All tables, plots, and stage reports below are saved
outputs from the completed study.

## Held-out evaluation

The sealed set contains **694 supported, shifted pages (N) and 2,324 unsupported
pages (O)**. The frozen panel comprises six core baselines and 108 layer/readout/method
configurations, with a 32k reference bank for feature methods.

| Core method | AUROC ↑ [95% CI] | FPR95 ↓ |
|---|---:|---:|
| MSP | 0.7372 [0.7160, 0.7605] | 0.8468 |
| Energy | 0.7650 [0.7452, 0.7862] | 0.8369 |
| Mahalanobis | 0.7474 [0.7253, 0.7697] | 0.8546 |
| kNN (`k=10`) | 0.7405 [0.7171, 0.7642] | 0.8795 |
| ViM-origin Residual | 0.7433 [0.7213, 0.7658] | 0.8808 |
| ViM | 0.7473 [0.7254, 0.7698] | 0.8731 |

Sources: [baseline CSV](tables/07_sealed_test/sealed_baselines.csv) and
[sealed report](reports/07_sealed_test.md), including FPR95 intervals.
Intervals use 2,000 stratified bootstrap resamples. AUROC is ranking separation;
FPR95 is the fraction of O accepted at the first ROC point accepting at least
95% of N. These are distinct from category accuracy and from T false rejection
at the independently calibrated 99%-acceptance threshold.

The core Mahalanobis and kNN scores use the model's classifier-input feature
(`h_doc`, equivalent to Layer 12 LN-pooler). The homepage's LN-free comparison
isolates normalization without learned affine parameters and comes from the
separate layer grid. ViM-origin Residual and the grid's mean-centred PCA residual
also use different centring conventions.

For the main representation findings, inspect the
[108-row layer grid](tables/07_sealed_test/sealed_layer_grid.csv),
[AUROC curves](figures/07_sealed_test/development_vs_sealed_auroc.png), and
[FPR95 curves](figures/07_sealed_test/development_vs_sealed_fpr95.png).
All five [replication checks](supporting/07_sealed_test/replication_predictions_evaluation.json)
were recorded as supported. Energy has the highest AUROC point estimate (0.7650)
in the complete panel; Layer 9 LN-pooler Mahalanobis has the lowest FPR95 (0.8171),
with AUROC 0.6607. Neither establishes a reliable deployment threshold.

## Evidence index

| Stage | Report or record | Tables and figures |
|---|---|---|
| **7. Sealed evaluation** | [Final report](reports/07_sealed_test.md), [run metadata](supporting/07_sealed_test/run_metadata.json) | [Tables](tables/07_sealed_test/), [figures](figures/07_sealed_test/) |
| 6. Normalization and residual ablations | [Development report](reports/06_final_ablation.md) | [Tables](tables/06_final_ablation/), [figures](figures/06_final_ablation/) |
| 5. Representation and layer scan | [Development report](reports/05_layer_scan.md) | [Tables](tables/05_layer_scan/), [figures](figures/05_layer_scan/) |
| 4. Supplementary methods | [Development report](reports/04_supplementary_baselines.md) | [Tables](tables/04_supplementary_baselines/) |
| 3. Single-signal comparison | [Development report](reports/03_single_signals.md) | [Tables](tables/03_single_signals/) |
| 2. Data audit and grouped split | [Audit and provenance records](supporting/02_data_audit/) | [Data explanation](../docs/STUDY.md#data-and-provenance) |
| 1. Published-baseline reproduction | [Independent 4,312-image rerun](tables/01_reproduction/actual_images_4312.json) | [All reproduction tables](tables/01_reproduction/) |

Stage reports and protocols describe their dated checkpoints. The
[study record](../docs/STUDY.md#protocol-record-and-deviations) explains freeze
publication, source/rendering limitations, and guardrail scope. The separately
labelled [4,417-row sensitivity](tables/01_reproduction/hybrid_full_4417.json)
uses author logits for 105 unavailable images; it is not a full independent image rerun.

## Local preview

```bash
python scripts/preview_results.py
```

Run from the repository root to print the saved core table without model or data
access. Curated evidence lives in `tables/`, `figures/`, `reports/`, and `supporting/`.
Independent reruns belong under the ignored `results/runs/<stage>/` directory;
see the [reproduction guide](../docs/REPRODUCIBILITY.md).
