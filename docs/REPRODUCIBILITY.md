# Inspecting results and reproducing the environment

[Project overview](../README.md) · [Data requirements](DATA.md)

## Start with the saved results

From the repository root:

```bash
python scripts/preview_results.py
```

This standard-library reader prints the committed sealed baseline table and its
confidence intervals. It only reads aggregate CSV/JSON files. It does not download
data, fit a detector, run inference, bootstrap results, or access sealed images.
The documented research environment is Python 3.12.

The complete evidence is browsable in [results/](../results/README.md).

## Research environment

Use **Python 3.12**. The commands below explicitly use the virtual environment's
interpreter, so activation is unnecessary. Installation requires network access
and downloads the research dependencies; the environment includes PyTorch.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\doc-ood.exe --help
```

If the Windows Python launcher is unavailable, use `python -m venv .venv`
after checking that `python --version` reports Python 3.12.

### Linux / macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/doc-ood --help
```

The [package specification](../pyproject.toml) preserves the existing research
requirements. Most direct dependencies are pinned, but NumPy and Pillow use bounded
ranges and build/transitive dependencies are not fully locked. It is not a complete
environment lockfile. The model revision is separately fixed in
[modeling.py](../src/document_ood_gate/modeling.py).

The existing tests cover numerical behavior, duplicate grouping, split isolation,
caches, and pipeline handling. Running these tests is software validation; it does
not reproduce the recorded full-image study. This presentation cleanup was checked
without installing the full research environment; it does not certify a fresh
full-suite run or platform-specific GPU compatibility.

## Historical pipeline map

These commands document the existing research implementation. They are not needed
to explore the portfolio. After activating the environment, inspect each command
with `doc-ood <command> --help`; otherwise use the full executable path shown above.

| Stage | Commands | Inputs needed |
|---|---|---|
| Recover available source PDFs | `recover-pdf-pages` | Author PDFs, Poppler, recovery paths |
| Extract N/O features | `extract-ood` | Local N/O image directories, model |
| Reproduce output baselines | `reproduce`, `reproduce-hybrid` | Feature cache; author logits for hybrid |
| Audit duplicates | `audit-duplicates`, `finalize-duplicate-audit` | Images, candidate pairs, manual decisions |
| Create grouped split | `split-research` | Confirmed duplicate manifest |
| Build RVL reference inputs | `sample-rvl`, `extract-rvl-sample`, `extract-rvl-features` | Official RVL archive/splits and manifests |
| Main signal panel | `evaluate-development`, `evaluate-reference-signals` | Split and relevant feature caches |
| Supplementary panel | `extract-development-ood`, `evaluate-supplementary-baselines` | Development manifest and reference/validation/T caches |
| Layer study | `run-layer-scan` | Isolated development manifest and RVL reference manifest |
| Final development ablations | `run-final-development` | Existing layer caches |
| Frozen final evaluation | `run-final-sealed` | Sealed manifest, caches, and freeze metadata |

The original sealed evaluation is complete. Its command is retained for provenance
and independent reproduction, not as a suggested next experiment or tuning step.
Any independent rerun must be labelled separately from the historical sealed result.

### Count and configuration details

The original release has 3,415 O target rows, while the recorded independent rerun
has 3,310 available/recovered O images. The early CLI defaults use the target
counts; an independent rerun of the recorded available-image subset therefore
needs `extract-ood --o-expected 3310` and `reproduce --expected-o 3310`, alongside
their required path arguments. Do not invent features for the 105 unavailable rows.

The initial development manifest has 1,294 rows. The primary comparison excludes
14 contradictory-label duplicates and has 1,280 rows. These are different stages,
not interchangeable input counts.

The early [experiment settings](../configs/experiment.yaml) are a protocol reference;
the early commands expose explicit arguments rather than a universal YAML loader.
The layer and final commands accept their own `--config` files. Use each command's
help and the [protocol map](STUDY.md#original-protocol-map) together.

## Artifacts and hardware

The repository includes aggregate tables, plots, audit summaries, and source code.
Raw document images, per-image feature caches, complete runtime manifests, and
model weights must be obtained or regenerated separately.

Model access may download the pinned Hugging Face checkpoint. The PDF-recovery
stage additionally needs Poppler. The layer/final commands default to CUDA;
no new CPU/GPU runtime or hardware-performance claim was measured during cleanup.

Keep local data under the ignored `data/raw/`, `data/cache/`, and
`data/features/` locations. Write independent rerun outputs under
`results/runs/<stage>/` (ignored) rather than overwriting the curated evidence in
`results/tables/`, `results/figures/`, `results/reports/`, or `results/supporting/`.
