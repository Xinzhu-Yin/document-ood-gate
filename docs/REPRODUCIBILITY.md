# Setup and reproduction

[Project overview](../README.md) · [Saved results](../results/README.md) · [Data requirements](STUDY.md#data-and-provenance)

## Preview saved results

From the repository root, run `python scripts/preview_results.py`. This
standard-library reader prints the committed sealed baseline table and confidence
intervals without installing dependencies, downloading data, or running inference.

## Research environment

Use **Python 3.12**. These commands use the virtual environment's interpreter
directly, so activation is unnecessary. Installation downloads the research
dependencies, including PyTorch.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\doc-ood.exe --help
```

If the Windows launcher is unavailable, use `python -m venv .venv` after checking
that `python --version` reports Python 3.12.

### Linux / macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/doc-ood --help
```

The [package specification](../pyproject.toml) pins most direct dependencies;
NumPy and Pillow use bounded ranges, and build/transitive dependencies are not
fully locked. The checkpoint revision is fixed in
[modeling.py](../src/document_ood_gate/modeling.py).

Tests cover numerical scoring, duplicate grouping, split isolation, feature
extraction, and cache recovery. They validate the implementation; reproducing
the full study additionally requires the source data and model.

## Research pipeline

Use `doc-ood <command> --help` after activation, or the full executable path above.
Commands require locally prepared inputs; this table maps the recorded workflow.

| Stage | Commands | Required inputs |
|---|---|---|
| Recover source PDFs | `recover-pdf-pages` | Author PDFs, Poppler, recovery paths |
| Extract N/O features | `extract-ood` | N/O image directories, model |
| Reproduce output baselines | `reproduce`, `reproduce-hybrid` | Feature cache; author logits for hybrid |
| Audit duplicates | `audit-duplicates`, `finalize-duplicate-audit` | Images, candidate pairs, manual decisions |
| Create grouped split | `split-research` | Confirmed duplicate manifest |
| Build RVL reference inputs | `sample-rvl`, `extract-rvl-sample`, `extract-rvl-features` | RVL archive/splits and manifests |
| Main signal panel | `evaluate-development`, `evaluate-reference-signals` | Split and feature caches |
| Supplementary panel | `extract-development-ood`, `evaluate-supplementary-baselines` | Development manifest; reference/validation/T caches |
| Layer study | `run-layer-scan` | Isolated development and RVL reference manifests |
| Development ablations | `run-final-development` | Existing layer caches |
| Frozen final evaluation | `run-final-sealed` | Sealed manifest, caches, and freeze metadata |

The original sealed evaluation is complete. An independent reproduction must be
labelled separately from that historical result.

## Input and configuration details

- **Available-image counts:** the release targets 3,415 O rows; the independent
  rerun used 3,310 available/recovered O images. Early CLI defaults use target
  counts. Reproducing the available-image subset needs `extract-ood --o-expected
  3310` and `reproduce --expected-o 3310`, plus required paths. The 105 missing
  rows have no independently extracted features.
- **Development counts:** the initial manifest has 1,294 rows; the primary
  comparison has 1,280 after excluding 14 contradictory-label duplicates.
- **Configuration loading:** early commands use explicit arguments, with
  [experiment.yaml](../configs/experiment.yaml) as a protocol reference. Layer
  and final commands accept their own `--config` files. Consult command help and
  the [protocol map](STUDY.md#original-protocol-map).

Raw images/PDFs, full runtime manifests, feature caches, and weights are obtained
or regenerated separately. Model access may download the pinned checkpoint;
PDF recovery requires Poppler, and layer/final commands default to CUDA.

Keep local data in ignored `data/raw/`, `data/cache/`, and `data/features/`
locations. Write independent reruns to ignored `results/runs/<stage>/` to preserve
the committed aggregate evidence. See [data provenance and reuse](STUDY.md#data-and-provenance)
for recorded source revisions, availability gaps, and attribution.
