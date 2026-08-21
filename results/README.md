# Results index

This directory contains aggregate results and audit evidence only. Raw documents,
extracted features, model weights, and per-page sensitive previews are not committed.

## Primary results

| Artifact | Purpose |
|---|---|
| [`single_signals/development_summary.md`](single_signals/development_summary.md) | Human-readable 12-row development-only comparison |
| [`single_signals/development_all_single_signals.json`](single_signals/development_all_single_signals.json) | Complete machine-readable scores, confidence intervals, and fixed-threshold operating points |
| [`reproduction/actual_images_4312.json`](reproduction/actual_images_4312.json) | Independent reproduction using every currently available or recovered image |
| [`reproduction/hybrid_full_4417.json`](reproduction/hybrid_full_4417.json) | Full-row sensitivity check using author logits only for the 105 unavailable images |

## Data and split audit

| Artifact | Purpose |
|---|---|
| [`data_release_status.json`](data_release_status.json) | Expected, available, recovered, and unavailable image counts |
| [`duplicate_audit/duplicate_audit_summary.json`](duplicate_audit/duplicate_audit_summary.json) | Exact and manually confirmed near-duplicate summary |
| [`duplicate_audit/phash_review.csv`](duplicate_audit/phash_review.csv) | Candidate-level pHash review evidence; pHash alone never created a group |
| [`duplicate_audit/manual_confirmed_pairs.json`](duplicate_audit/manual_confirmed_pairs.json) | Manually confirmed visual duplicate pairs |
| [`duplicate_audit/review_notes.json`](duplicate_audit/review_notes.json) | Review procedure and notes |
| [`duplicate_audit/split_summary.json`](duplicate_audit/split_summary.json) | Group-isolated 30/70 split counts and integrity checks |

## Reference-data provenance

| Artifact | Purpose |
|---|---|
| [`reference_data/provenance.json`](reference_data/provenance.json) | RVL source archive, hashes, and extraction provenance |
| [`reference_data/sample_summary.json`](reference_data/sample_summary.json) | Reference, validation, and T-guardrail sample counts |

## Supplementary development checks

| Artifact | Purpose |
|---|---|
| [`single_signals/development_output_signals.json`](single_signals/development_output_signals.json) | Output-level signals after the frozen cross-label conflict rule |
| [`single_signals/development_output_signals_including_label_conflict.json`](single_signals/development_output_signals_including_label_conflict.json) | Sensitivity result retaining the contradictory exact-pixel N/O group |

All development artifacts precede the one-time sealed evaluation. No sealed metrics are
present in this directory as of 2026-08-21.

