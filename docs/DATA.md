# Data access, provenance, and reuse

[Project overview](../README.md) · [Research record](STUDY.md)

## Source materials

| Material | Source | Recorded identifier |
|---|---|---|
| Frozen DiT classifier | [Microsoft checkpoint](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip) | Revision `23f8b03d130fb66bbc9a15df3c75d753e49240eb` |
| RVL-CDIP | [Dataset repository](https://huggingface.co/datasets/aharley/rvl_cdip) | Revision `03f14a4ad0a32413eff51ca10f9f511545f2bd5b` |
| RVL-CDIP-N/O and baseline logits | [Authors' repository](https://github.com/gxlarson/RVL-CDIP-OOD) | Release status recorded on 2026-08-12 |

The retained [RVL provenance record](../results/supporting/02_data_audit/reference_provenance.json)
includes archive size, split-file checksums, manifest checksum, and sampling seed.
These are identifiers of the original study inputs, not a claim that every upstream
download is still available.

## Availability recorded in the study

| N/O input | Count |
|---|---:|
| N target and available images | 1,002 |
| O target rows | 3,415 |
| O TIFFs available in the audited release | 2,915 |
| O pages recovered from author-hosted PDFs | 395 |
| O images unavailable at the audit | 105 |
| Independent image rerun | 4,312 |

The 395 recovered pages were rendered with the calibrated pipeline described in
the [release audit](../results/supporting/02_data_audit/data_release_status.json).
They are close reconstructions, not byte-identical original TIFFs.

Author-provided logits fill the 105 unavailable rows only in the separately named
[4,417-row output-score sensitivity result](../results/tables/01_reproduction/hybrid_full_4417.json).
Those rows do not participate in representation experiments. The
[4,312-image rerun](../results/tables/01_reproduction/actual_images_4312.json) is the
independent image-based result.

## What is included

Source code, frozen configurations, aggregate metrics, plots, reports, run metadata,
and duplicate-review records are included. Duplicate records retain source image
identifiers and grouping decisions for auditability; they contain no document pixels.

Raw document images, original PDFs, extracted features/logits, complete runtime
manifests, and pretrained weights are not bundled. Full reproduction requires
separate source access and local preparation. The saved results can be inspected
without those materials.

## Attribution and reuse status

The pretrained model, datasets, and referenced methods belong to their respective
creators. See [references](REFERENCES.md). This project does not grant rights to
redistribute third-party documents or weights; obtain them from their source and
follow the terms provided there.

This repository currently has no project-wide license file.
