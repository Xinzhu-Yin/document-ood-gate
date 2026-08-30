from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModelForImageClassification

from .duplicates import (
    build_fingerprint_manifest,
    exact_duplicate_groups,
    finalize_duplicate_audit,
    phash_candidate_pairs,
)
from .io import find_images, load_feature_cache, save_feature_cache, write_json
from .layer_study import run_frozen_layer_study
from .metadata import parse_official_image_metadata
from .modeling import MODEL_REVISION, FrozenDiT
from .official_reference import hybrid_reproduction_report, load_official_output
from .reproduction import reproduce_from_logits
from .recovery import recover_first_pages
from .reference_evaluation import evaluate_reference_signals
from .reference_data import (
    create_reference_sample_manifest,
    extract_selected_from_tar,
    read_official_split,
)
from .single_signal import evaluate_development_single_signals
from .splitting import create_research_split, read_manifest, write_assignments
from .supplementary_evaluation import (
    evaluate_supplementary_baselines,
    supplementary_results_frame,
    supplementary_summary_markdown,
)


def extract_command(args: argparse.Namespace) -> None:
    n_paths = find_images(Path(args.n_dir))
    o_directories = args.o_dir if isinstance(args.o_dir, list) else [args.o_dir]
    o_paths = sorted(
        path for directory in o_directories for path in find_images(Path(directory))
    )
    if len({path.resolve() for path in o_paths}) != len(o_paths):
        raise SystemExit("The supplied O directories contain duplicate paths")
    if len(n_paths) != args.n_expected or len(o_paths) != args.o_expected:
        raise SystemExit(
            f"Count gate failed: expected N={args.n_expected}/O={args.o_expected}, "
            f"found N={len(n_paths)}/O={len(o_paths)}"
        )
    model = FrozenDiT(device=args.device, dtype=args.dtype)
    paths = n_paths + o_paths
    output = model.infer_paths(paths, batch_size=args.batch_size)
    save_feature_cache(
        Path(args.output),
        paths,
        ["N"] * len(n_paths) + ["O"] * len(o_paths),
        output.logits,
        output.features,
        [
            parse_official_image_metadata(path, subset).n_class or "ood"
            for path, subset in zip(
                paths,
                ["N"] * len(n_paths) + ["O"] * len(o_paths),
                strict=True,
            )
        ],
    )


def extract_development_ood_command(args: argparse.Namespace) -> None:
    """Create a physically isolated development-only OOD feature cache."""

    assignments = read_manifest(Path(args.split))
    required = {"path", "subset", "split"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise SystemExit(f"Research split is missing: {', '.join(missing)}")
    development = assignments.loc[assignments["split"] == "development"].copy()
    if len(development) != args.expected:
        raise SystemExit(
            f"Development count gate failed: expected {args.expected}, found {len(development)}"
        )
    if set(development["subset"].astype(str)) != {"N", "O"}:
        raise SystemExit("Development rows must contain exactly N and O")
    paths = [Path(value) for value in development["path"]]
    missing_files = [path for path in paths if not path.is_file()]
    if missing_files:
        raise SystemExit(f"Missing development image: {missing_files[0]}")

    model = FrozenDiT(device=args.device, dtype=args.dtype)
    output = model.infer_paths(paths, batch_size=args.batch_size)
    true_labels = [
        str(value) if subset == "N" else "ood"
        for value, subset in zip(
            development.get("n_class", [""] * len(development)),
            development["subset"],
            strict=True,
        )
    ]
    save_feature_cache(
        Path(args.output_features),
        paths,
        development["subset"].astype(str).tolist(),
        output.logits,
        output.features,
        true_labels,
    )
    write_assignments(development, Path(args.output_split))
    print(
        f"Cached {len(development)} development-only N/O rows; "
        "sealed image pixels, logits, and features were not opened"
    )


def reproduce_command(args: argparse.Namespace) -> None:
    cache = load_feature_cache(Path(args.features))
    result = reproduce_from_logits(
        cache["logits"],
        cache["subsets"],
        cache.get("true_labels"),
        expected_n=args.expected_n,
        expected_o=args.expected_o,
    )
    write_json(Path(args.output), result)
    print(f"Reproduction gate: {result['overall_gate']}")
    for name, metrics in result["signals"].items():
        print(
            f"{name}: AUROC={metrics['auroc']:.6f}, "
            f"FPR95={metrics['fpr95_first_ge']:.6f}, gate={metrics['gate']}"
        )


def hybrid_reproduce_command(args: argparse.Namespace) -> None:
    report = hybrid_reproduction_report(
        load_feature_cache(Path(args.features)),
        load_official_output(Path(args.official_n)),
        load_official_output(Path(args.official_o)),
    )
    write_json(Path(args.output), report)
    print(
        f"Hybrid full-set reproduction gate: {report['overall_gate']}; "
        f"author-logit fallbacks={report['provenance']['author_logit_fallback']}"
    )


def duplicates_command(args: argparse.Namespace) -> None:
    n_paths = find_images(Path(args.n_dir))
    o_directories = args.o_dir if isinstance(args.o_dir, list) else [args.o_dir]
    o_paths = sorted(
        path for directory in o_directories for path in find_images(Path(directory))
    )
    if len({path.resolve() for path in o_paths}) != len(o_paths):
        raise SystemExit("The supplied O directories contain duplicate paths")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_fingerprint_manifest(n_paths, o_paths)
    manifest.to_parquet(output_dir / "fingerprints.parquet", index=False)
    exact_duplicate_groups(manifest).to_csv(output_dir / "exact_duplicates.csv", index=False)
    phash_candidate_pairs(manifest, max_distance=args.phash_distance).to_csv(
        output_dir / "phash_candidates.csv", index=False
    )


def split_command(args: argparse.Namespace) -> None:
    manifest = read_manifest(Path(args.manifest))
    assignments, summary = create_research_split(
        manifest,
        development_fraction=args.development_fraction,
        seed=args.seed,
        group_column=args.group_column,
        source_column=args.source_column,
        n_class_column=args.n_class_column,
        expected_n_classes=args.expected_n_classes,
        search_trials=args.search_trials,
    )
    write_assignments(assignments, Path(args.output))
    write_json(Path(args.summary), summary)
    development_rows = int((assignments["split"] == "development").sum())
    print(
        f"Metadata-only split: development={development_rows}, "
        f"sealed={len(assignments) - development_rows}; no sealed metrics computed"
    )


def finalize_duplicates_command(args: argparse.Namespace) -> None:
    manifest = read_manifest(Path(args.manifest))
    candidates = read_manifest(Path(args.candidates))
    finalized, reviewed, summary = finalize_duplicate_audit(
        manifest,
        candidates,
        Path(args.manual_decisions),
    )
    write_assignments(finalized, Path(args.output_manifest))
    reviewed_export = reviewed.copy()
    reviewed_export["left_path"] = reviewed_export["left_path"].map(
        lambda value: Path(value).name
    )
    reviewed_export["right_path"] = reviewed_export["right_path"].map(
        lambda value: Path(value).name
    )
    write_assignments(reviewed_export, Path(args.output_candidates))
    write_json(Path(args.summary), summary)
    print(
        f"Confirmed duplicate groups={summary['confirmed_duplicate_groups']}; "
        f"cross-label groups={summary['cross_label_confirmed_groups']}"
    )


def recover_command(args: argparse.Namespace) -> None:
    pdf_paths = sorted(Path(args.pdf_dir).glob("documentcloud_*.pdf"))
    if len(pdf_paths) != args.expected:
        raise SystemExit(
            f"Recovery count gate failed: expected {args.expected} PDFs, found {len(pdf_paths)}"
        )
    manifest = recover_first_pages(
        pdf_paths,
        Path(args.output_dir),
        pdftoppm=Path(args.pdftoppm),
        cache_dir=Path(args.cache_dir),
        dpi=args.dpi,
        maximum_dimension=args.maximum_dimension,
    )
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"Recovered {len(manifest)} first-page TIFFs")


def evaluate_development_command(args: argparse.Namespace) -> None:
    cache = load_feature_cache(Path(args.features))
    assignments = read_manifest(Path(args.split))
    result = evaluate_development_single_signals(
        cache,
        assignments,
        target_tpr=args.target_n_acceptance,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        exclude_cross_label_conflicts=not args.include_cross_label_conflicts,
    )
    write_json(Path(args.output), result)
    print(
        f"Development-only evaluation: N={result['counts']['N']}, "
        f"O={result['counts']['O']}; sealed metrics were not computed"
    )
    for name, metrics in result["signals"].items():
        print(f"{name}: AUROC={metrics['auroc']:.6f}, FPR95={metrics['fpr95']:.6f}")


def sample_rvl_command(args: argparse.Namespace) -> None:
    train = read_official_split(Path(args.train), "train")
    validation = read_official_split(Path(args.validation), "validation")
    test = read_official_split(Path(args.test), "test")
    manifest, summary = create_reference_sample_manifest(
        train,
        validation,
        test,
        reference_sizes=tuple(args.reference_sizes),
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    write_assignments(manifest, Path(args.output))
    write_json(Path(args.summary), summary)
    print(f"Selected {len(manifest)} class-balanced RVL-CDIP images")


def extract_rvl_sample_command(args: argparse.Namespace) -> None:
    manifest = read_manifest(Path(args.manifest))
    extracted = extract_selected_from_tar(
        Path(args.archive),
        manifest,
        Path(args.output_root),
    )
    write_assignments(extracted, Path(args.output_manifest))
    print(f"Extracted {len(extracted)} selected images; full archive was not unpacked")


def extract_rvl_features_command(args: argparse.Namespace) -> None:
    manifest = read_manifest(Path(args.manifest))
    required = {"local_path", "purpose", "label"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise SystemExit(f"Extracted RVL manifest is missing: {', '.join(missing)}")
    selected = manifest[manifest["purpose"] == args.purpose].copy()
    if selected.empty:
        raise SystemExit(f"No rows found for purpose={args.purpose!r}")
    paths = [Path(value) for value in selected["local_path"]]
    missing_files = [path for path in paths if not path.is_file()]
    if missing_files:
        raise SystemExit(f"Missing extracted image: {missing_files[0]}")
    model = FrozenDiT(device=args.device, dtype=args.dtype)
    output = model.infer_paths(paths, batch_size=args.batch_size)
    save_feature_cache(
        Path(args.output),
        paths,
        [str(args.purpose)] * len(paths),
        output.logits,
        output.features,
        [str(int(value)) for value in selected["label"]],
    )
    print(f"Cached logits/features for {len(paths)} {args.purpose} images")


def evaluate_reference_signals_command(args: argparse.Namespace) -> None:
    result = evaluate_reference_signals(
        load_feature_cache(Path(args.ood_features)),
        read_manifest(Path(args.split)),
        load_feature_cache(Path(args.reference_features)),
        load_feature_cache(Path(args.validation_features)),
        load_feature_cache(Path(args.t_features)),
        read_manifest(Path(args.manifest)),
        reference_sizes=tuple(args.reference_sizes),
        k=args.knn_k,
        target_n_tpr=args.target_n_acceptance,
        target_validation_acceptance=args.target_validation_acceptance,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        device=args.device,
    )
    write_json(Path(args.output), result)
    print("Development-only single-signal evaluation complete; sealed metrics were not computed")
    for name, metrics in result["signals"].items():
        print(
            f"{name}: AUROC={metrics['auroc']:.6f}, FPR95={metrics['fpr95']:.6f}, "
            f"T false reject={metrics['t_false_rejection_rate']:.6f}"
        )


def supplementary_baselines_command(args: argparse.Namespace) -> None:
    assignments = read_manifest(Path(args.split))
    if "split" not in assignments or set(assignments["split"].astype(str)) != {"development"}:
        raise SystemExit(
            "Supplementary evaluation requires a physically isolated development-only "
            "split manifest; refusing to open the OOD feature cache"
        )
    ood_cache = load_feature_cache(Path(args.ood_features))
    model = AutoModelForImageClassification.from_pretrained(
        args.model_id,
        revision=args.model_revision,
    )
    if not hasattr(model, "classifier"):
        raise SystemExit(f"Expected a classifier head, got {type(model).__name__}")
    weight = model.classifier.weight.detach().cpu().numpy()
    bias = model.classifier.bias.detach().cpu().numpy()
    result = evaluate_supplementary_baselines(
        ood_cache,
        assignments,
        load_feature_cache(Path(args.reference_features)),
        load_feature_cache(Path(args.validation_features)),
        load_feature_cache(Path(args.t_features)),
        read_manifest(Path(args.manifest)),
        weight,
        bias,
        reference_size=args.reference_size,
        gen_gamma=args.gen_gamma,
        gen_top_m=args.gen_top_m,
        react_percentile=args.react_percentile,
        vim_principal_dim=args.vim_principal_dim,
        target_n_tpr=args.target_n_acceptance,
        target_validation_acceptance=args.target_validation_acceptance,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    write_json(Path(args.output), result)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(supplementary_summary_markdown(result), encoding="utf-8")
    table_path = Path(args.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    supplementary_results_frame(result).to_csv(table_path, index=False)
    print(
        "Supplementary development-only baseline panel: "
        f"status={result['status']}; sealed metrics were not computed"
    )
    for name, metrics in result.get("supplementary_signals", {}).items():
        print(
            f"{name}: AUROC={metrics['auroc']:.6f}, FPR95={metrics['fpr95']:.6f}, "
            f"T false reject={metrics['t_false_rejection_rate']:.6f}"
        )
    if result["status"] != "completed":
        raise SystemExit("Internal baseline reproduction gate failed; supplementary run stopped")


def layer_scan_command(args: argparse.Namespace) -> None:
    result = run_frozen_layer_study(
        config_path=Path(args.config),
        reference_manifest_path=Path(args.reference_manifest),
        development_manifest_path=Path(args.development_manifest),
        cache_dir=Path(args.cache_dir),
        results_dir=Path(args.results_dir),
        batch_size=args.batch_size,
        device=args.device,
    )
    g1 = result["gates"]["G1"]
    g2 = result["gates"]["G2"]
    print(
        "Frozen development-only layer scan complete: "
        f"G1 max_abs_error={g1['max_abs_error']:.9g}; "
        f"G2 AUROC={g2['observed_auroc']:.7f}; "
        "sealed rows loaded/scored=0/0"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doc-ood")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract-ood", help="extract frozen DiT logits/features for official N/O")
    extract.add_argument("--n-dir", required=True)
    extract.add_argument(
        "--o-dir",
        required=True,
        action="append",
        help="O image directory; repeat for author TIFF plus recovered TIFF directories",
    )
    extract.add_argument("--output", required=True)
    extract.add_argument("--n-expected", type=int, default=1002)
    extract.add_argument("--o-expected", type=int, default=3415)
    extract.add_argument("--batch-size", type=int, default=4)
    extract.add_argument("--device", default="auto")
    extract.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    extract.set_defaults(func=extract_command)

    extract_development = sub.add_parser(
        "extract-development-ood",
        help="extract frozen DiT features only for rows assigned to development",
    )
    extract_development.add_argument("--split", required=True)
    extract_development.add_argument("--output-features", required=True)
    extract_development.add_argument("--output-split", required=True)
    extract_development.add_argument("--expected", type=int, default=1294)
    extract_development.add_argument("--batch-size", type=int, default=16)
    extract_development.add_argument("--device", default="auto")
    extract_development.add_argument(
        "--dtype", choices=["float16", "float32"], default="float32"
    )
    extract_development.set_defaults(func=extract_development_ood_command)

    reproduce = sub.add_parser("reproduce", help="evaluate published MSP/Energy baselines")
    reproduce.add_argument("--features", required=True)
    reproduce.add_argument("--output", required=True)
    reproduce.add_argument("--expected-n", type=int, default=1002)
    reproduce.add_argument("--expected-o", type=int, default=3415)
    reproduce.set_defaults(func=reproduce_command)

    hybrid_reproduce = sub.add_parser(
        "reproduce-hybrid",
        help="full N/O metrics using reruns plus author logits only for unavailable images",
    )
    hybrid_reproduce.add_argument("--features", required=True)
    hybrid_reproduce.add_argument("--official-n", required=True)
    hybrid_reproduce.add_argument("--official-o", required=True)
    hybrid_reproduce.add_argument("--output", required=True)
    hybrid_reproduce.set_defaults(func=hybrid_reproduce_command)

    duplicates = sub.add_parser("audit-duplicates", help="fingerprint N/O and produce review candidates")
    duplicates.add_argument("--n-dir", required=True)
    duplicates.add_argument("--o-dir", required=True, action="append")
    duplicates.add_argument("--output-dir", required=True)
    duplicates.add_argument("--phash-distance", type=int, default=8)
    duplicates.set_defaults(func=duplicates_command)

    finalize_duplicates = sub.add_parser(
        "finalize-duplicate-audit",
        help="apply exact hashes and explicit visual review decisions",
    )
    finalize_duplicates.add_argument("--manifest", required=True)
    finalize_duplicates.add_argument("--candidates", required=True)
    finalize_duplicates.add_argument("--manual-decisions", required=True)
    finalize_duplicates.add_argument("--output-manifest", required=True)
    finalize_duplicates.add_argument("--output-candidates", required=True)
    finalize_duplicates.add_argument("--summary", required=True)
    finalize_duplicates.set_defaults(func=finalize_duplicates_command)

    split = sub.add_parser(
        "split-research",
        help="create a grouped metadata-only development/sealed assignment",
    )
    split.add_argument("--manifest", required=True, help="confirmed .csv/.parquet manifest")
    split.add_argument("--output", required=True, help="assignment .csv/.parquet path")
    split.add_argument("--summary", required=True, help="metadata balance summary JSON")
    split.add_argument("--development-fraction", type=float, default=0.30)
    split.add_argument("--seed", type=int, default=20260812)
    split.add_argument("--group-column", default="confirmed_group_id")
    split.add_argument("--source-column", default="source")
    split.add_argument("--n-class-column", default="n_class")
    split.add_argument("--expected-n-classes", type=int, default=12)
    split.add_argument("--search-trials", type=int, default=4096)
    split.set_defaults(func=split_command)

    recover = sub.add_parser(
        "recover-pdf-pages",
        help="recover the 395 missing DocumentCloud TIFFs from author-hosted PDFs",
    )
    recover.add_argument("--pdf-dir", required=True)
    recover.add_argument("--output-dir", required=True)
    recover.add_argument("--manifest", required=True)
    recover.add_argument("--pdftoppm", required=True)
    recover.add_argument("--cache-dir", default="tmp/pdfs")
    recover.add_argument("--expected", type=int, default=395)
    recover.add_argument("--dpi", type=int, default=200)
    recover.add_argument("--maximum-dimension", type=int, default=1000)
    recover.set_defaults(func=recover_command)

    evaluate = sub.add_parser(
        "evaluate-development",
        help="evaluate individual output/feature-norm signals on the development split only",
    )
    evaluate.add_argument("--features", required=True)
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--target-n-acceptance", type=float, default=0.95)
    evaluate.add_argument("--bootstrap-iterations", type=int, default=2000)
    evaluate.add_argument("--seed", type=int, default=20260812)
    evaluate.add_argument("--include-cross-label-conflicts", action="store_true")
    evaluate.set_defaults(func=evaluate_development_command)

    sample_rvl = sub.add_parser(
        "sample-rvl",
        help="create fixed class-balanced RVL-CDIP reference/validation/T samples",
    )
    sample_rvl.add_argument("--train", required=True)
    sample_rvl.add_argument("--validation", required=True)
    sample_rvl.add_argument("--test", required=True)
    sample_rvl.add_argument("--output", required=True)
    sample_rvl.add_argument("--summary", required=True)
    sample_rvl.add_argument("--reference-sizes", type=int, nargs="+", default=[8000, 32000, 80000])
    sample_rvl.add_argument("--validation-size", type=int, default=8000)
    sample_rvl.add_argument("--test-size", type=int, default=10000)
    sample_rvl.add_argument("--seed", type=int, default=20260812)
    sample_rvl.set_defaults(func=sample_rvl_command)

    extract_rvl = sub.add_parser(
        "extract-rvl-sample",
        help="stream through the official archive and extract only selected images",
    )
    extract_rvl.add_argument("--archive", required=True)
    extract_rvl.add_argument("--manifest", required=True)
    extract_rvl.add_argument("--output-root", required=True)
    extract_rvl.add_argument("--output-manifest", required=True)
    extract_rvl.set_defaults(func=extract_rvl_sample_command)

    extract_rvl_features = sub.add_parser(
        "extract-rvl-features",
        help="run frozen DiT over one selected RVL purpose and cache logits/features",
    )
    extract_rvl_features.add_argument("--manifest", required=True)
    extract_rvl_features.add_argument(
        "--purpose",
        required=True,
        choices=["reference", "validation_threshold", "t_guardrail"],
    )
    extract_rvl_features.add_argument("--output", required=True)
    extract_rvl_features.add_argument("--batch-size", type=int, default=16)
    extract_rvl_features.add_argument("--device", default="auto")
    extract_rvl_features.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    extract_rvl_features.set_defaults(func=extract_rvl_features_command)

    evaluate_reference = sub.add_parser(
        "evaluate-reference-signals",
        help="evaluate output, kNN, and Mahalanobis signals without opening the sealed split",
    )
    evaluate_reference.add_argument("--ood-features", required=True)
    evaluate_reference.add_argument("--split", required=True)
    evaluate_reference.add_argument("--manifest", required=True)
    evaluate_reference.add_argument("--reference-features", required=True)
    evaluate_reference.add_argument("--validation-features", required=True)
    evaluate_reference.add_argument("--t-features", required=True)
    evaluate_reference.add_argument("--output", required=True)
    evaluate_reference.add_argument("--reference-sizes", type=int, nargs="+", default=[8000, 32000, 80000])
    evaluate_reference.add_argument("--knn-k", type=int, default=10)
    evaluate_reference.add_argument("--target-n-acceptance", type=float, default=0.95)
    evaluate_reference.add_argument("--target-validation-acceptance", type=float, default=0.99)
    evaluate_reference.add_argument("--bootstrap-iterations", type=int, default=2000)
    evaluate_reference.add_argument("--seed", type=int, default=20260812)
    evaluate_reference.add_argument("--device", default="auto")
    evaluate_reference.set_defaults(func=evaluate_reference_signals_command)

    supplementary = sub.add_parser(
        "evaluate-supplementary-baselines",
        help="evaluate GEN, RMD, ViM, and ReAct+Energy without opening the sealed split",
    )
    supplementary.add_argument("--ood-features", required=True)
    supplementary.add_argument("--split", required=True)
    supplementary.add_argument("--manifest", required=True)
    supplementary.add_argument("--reference-features", required=True)
    supplementary.add_argument("--validation-features", required=True)
    supplementary.add_argument("--t-features", required=True)
    supplementary.add_argument("--output", required=True)
    supplementary.add_argument("--summary", required=True)
    supplementary.add_argument("--table", required=True)
    supplementary.add_argument("--model-id", default="microsoft/dit-base-finetuned-rvlcdip")
    supplementary.add_argument("--model-revision", default=MODEL_REVISION)
    supplementary.add_argument("--reference-size", type=int, default=32000)
    supplementary.add_argument("--gen-gamma", type=float, default=0.1)
    supplementary.add_argument("--gen-top-m", type=int, default=100)
    supplementary.add_argument("--react-percentile", type=float, default=90.0)
    supplementary.add_argument("--vim-principal-dim", type=int, default=512)
    supplementary.add_argument("--target-n-acceptance", type=float, default=0.95)
    supplementary.add_argument("--target-validation-acceptance", type=float, default=0.99)
    supplementary.add_argument("--bootstrap-iterations", type=int, default=2000)
    supplementary.add_argument("--seed", type=int, default=20260812)
    supplementary.set_defaults(func=supplementary_baselines_command)

    layer_scan = sub.add_parser(
        "run-layer-scan",
        help="run the frozen G1/G2-gated DiT representation layer study",
    )
    layer_scan.add_argument("--config", default="configs/layer_scan.yaml")
    layer_scan.add_argument(
        "--reference-manifest",
        default="data/raw/manifests/rvl_selected_extracted.csv",
    )
    layer_scan.add_argument(
        "--development-manifest",
        default="data/features/research_split_development_only.csv",
    )
    layer_scan.add_argument("--cache-dir", default="data/features/layer_scan")
    layer_scan.add_argument("--results-dir", default="results/layer_scan")
    layer_scan.add_argument("--batch-size", type=int, default=32)
    layer_scan.add_argument("--device", default="cuda")
    layer_scan.set_defaults(func=layer_scan_command)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
