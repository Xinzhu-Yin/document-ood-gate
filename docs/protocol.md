# Frozen experimental protocol (v1.1)

## Primary question

With the DiT classifier frozen, compare single output- and representation-level normality signals for separating N (pass) from O (reject). T is used only as a false-rejection guardrail. Downstream 16-class correctness is secondary.

## Stage 1 — mandatory reproduction

1. Pin the public DiT checkpoint and processor revision to
   `23f8b03d130fb66bbc9a15df3c75d753e49240eb`.
2. Use the official, unaugmented TIFF N/O release: 1,002 N and 3,415 O.
3. Run every image once in evaluation mode and cache logits, probabilities, predicted class, and the 768-dimensional classifier-input feature.
4. Compute micro AUROC and FPR95 with N as the positive/in-domain class and a higher score meaning more normal.
5. Compare MSP and Energy against the published targets in `configs/experiment.yaml`.
6. Stop on a red result. A yellow result may proceed only after a concrete implementation difference is isolated with an A/B check and documented.

No RVL-CDIP reference data is downloaded or processed before this gate passes.

## Duplicate audit and split

The original full-set reproduction always retains all images. Duplicate handling affects only the research split.

1. Hash raw bytes with SHA256.
2. Hash normalized decoded pixels with SHA256.
3. Generate pHash candidate pairs at the frozen Hamming-distance threshold.
4. Confirm pHash-only candidates before clustering; same-template/different-content pages are not automatic duplicates.
5. Group confirmed duplicates and create an approximately 30/70 development/sealed-test split while preserving N category and collection-source proportions as closely as possible.

## Stage 2 — single-signal study

- Output signals: MSP, Energy confidence, maximum logit, negative predictive entropy, and top-2 logit margin.
- Representation signals: feature norm, normalized kNN similarity, and shrinkage Mahalanobis normality.
- RVL reference-bank sizes are nested: 8k, 32k, and at most 80k.
- Hyperparameters and score orientation are frozen without using the sealed test.
- MSP and Energy are recomputed on exactly the same sealed test as every new signal.

Primary metrics are micro AUROC and FPR95. The fixed operating threshold is selected on development/RVL validation data and then applied once to sealed N/O and the 10k T guardrail. Final comparisons use paired bootstrap confidence intervals.

## Shortcut audit

File/source metadata are never Gate inputs. Mandatory diagnostics are uniform re-encoding, within-DocumentCloud and within-Web-Search evaluation, and a nuisance-only baseline using low-level image statistics.

## Combination Gate

Not part of protocol v1.1. After single-signal results are reviewed, any combination experiment requires a separate frozen addendum before the sealed test is touched.
