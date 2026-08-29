# Result index

All values in this directory are aggregate experiment outputs; raw document images and cached
features are excluded from Git.

| Stage | Read first | Machine-readable evidence |
|---|---|---|
| Published-baseline reproduction | [`reproduction/`](reproduction/) | `actual_images_4312.json`, `hybrid_full_4417.json` |
| Duplicate-aware single signals | [`single_signals/development_summary.md`](single_signals/development_summary.md) | `development_all_single_signals.json` |
| Four supplementary methods | [`supplementary_baselines/development_summary.md`](supplementary_baselines/development_summary.md) | `development_results.csv`, `development_results.json` |

Supporting audit records are retained in [`duplicate_audit/`](duplicate_audit/) and
[`reference_data/`](reference_data/). The label-conflict sensitivity file is kept under
`single_signals/` as supporting evidence, but it is not the primary result.

The 3,018-row sealed test has not been evaluated. Development outputs must not be interpreted as
final test claims.
