from __future__ import annotations

import argparse
from pathlib import Path

from .duplicates import build_fingerprint_manifest, exact_duplicate_groups, phash_candidate_pairs
from .io import find_images, load_feature_cache, save_feature_cache, write_json
from .modeling import FrozenDiT
from .reproduction import reproduce_from_logits


def extract_command(args: argparse.Namespace) -> None:
    n_paths = find_images(Path(args.n_dir))
    o_paths = find_images(Path(args.o_dir))
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
    )


def reproduce_command(args: argparse.Namespace) -> None:
    cache = load_feature_cache(Path(args.features))
    result = reproduce_from_logits(cache["logits"], cache["subsets"])
    write_json(Path(args.output), result)
    print(f"Reproduction gate: {result['overall_gate']}")
    for name, metrics in result["signals"].items():
        print(
            f"{name}: AUROC={metrics['auroc']:.6f}, "
            f"FPR95={metrics['fpr95_first_ge']:.6f}, gate={metrics['gate']}"
        )


def duplicates_command(args: argparse.Namespace) -> None:
    n_paths = find_images(Path(args.n_dir))
    o_paths = find_images(Path(args.o_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_fingerprint_manifest(n_paths, o_paths)
    manifest.to_parquet(output_dir / "fingerprints.parquet", index=False)
    exact_duplicate_groups(manifest).to_csv(output_dir / "exact_duplicates.csv", index=False)
    phash_candidate_pairs(manifest, max_distance=args.phash_distance).to_csv(
        output_dir / "phash_candidates.csv", index=False
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doc-ood")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract-ood", help="extract frozen DiT logits/features for official N/O")
    extract.add_argument("--n-dir", required=True)
    extract.add_argument("--o-dir", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--n-expected", type=int, default=1002)
    extract.add_argument("--o-expected", type=int, default=3415)
    extract.add_argument("--batch-size", type=int, default=4)
    extract.add_argument("--device", default="auto")
    extract.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    extract.set_defaults(func=extract_command)

    reproduce = sub.add_parser("reproduce", help="evaluate published MSP/Energy baselines")
    reproduce.add_argument("--features", required=True)
    reproduce.add_argument("--output", required=True)
    reproduce.set_defaults(func=reproduce_command)

    duplicates = sub.add_parser("audit-duplicates", help="fingerprint N/O and produce review candidates")
    duplicates.add_argument("--n-dir", required=True)
    duplicates.add_argument("--o-dir", required=True)
    duplicates.add_argument("--output-dir", required=True)
    duplicates.add_argument("--phash-distance", type=int, default=8)
    duplicates.set_defaults(func=duplicates_command)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
