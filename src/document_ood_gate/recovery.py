from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_size(width: int, height: int, maximum_dimension: int = 1000) -> tuple[int, int]:
    if width <= 0 or height <= 0 or maximum_dimension <= 0:
        raise ValueError("Image dimensions and maximum_dimension must be positive")
    if width >= height:
        return maximum_dimension, max(1, int(height * maximum_dimension / width))
    return max(1, int(width * maximum_dimension / height)), maximum_dimension


def recovered_tiff_name(pdf_path: Path) -> str:
    if not pdf_path.stem.startswith("documentcloud_"):
        raise ValueError(f"Unexpected recovery PDF filename: {pdf_path.name}")
    return f"ood__{pdf_path.stem}.tif"


def recover_first_pages(
    pdf_paths: list[Path],
    output_dir: Path,
    *,
    pdftoppm: Path,
    cache_dir: Path,
    dpi: int = 200,
    maximum_dimension: int = 1000,
) -> pd.DataFrame:
    """Recover author-style first-page TIFFs from the still-available source PDFs.

    Calibration against an intact PDF/TIFF pair identified the closest public pipeline as
    Poppler at 200 DPI, integer-floor scaling to a 1000-pixel maximum dimension, Pillow
    bicubic resize, RGB conversion, and uncompressed TIFF output.
    """

    executable = Path(pdftoppm)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for pdf_path in tqdm(sorted(map(Path, pdf_paths)), desc="Recover PDF first pages"):
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)
        output_path = output_dir / recovered_tiff_name(pdf_path)
        with tempfile.TemporaryDirectory(prefix="recover_", dir=cache_dir) as temporary:
            prefix = Path(temporary) / "page"
            command = [
                str(executable),
                "-f",
                "1",
                "-l",
                "1",
                "-singlefile",
                "-r",
                str(dpi),
                "-png",
                str(pdf_path),
                str(prefix),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"pdftoppm failed for {pdf_path}: {completed.stderr.strip()}"
                )
            rendered_path = prefix.with_suffix(".png")
            if not rendered_path.is_file():
                raise RuntimeError(f"pdftoppm produced no first page for {pdf_path}")
            with Image.open(rendered_path) as rendered:
                rendered.load()
                source_width, source_height = rendered.size
                size = target_size(source_width, source_height, maximum_dimension)
                recovered = rendered.convert("RGB").resize(size, Image.Resampling.BICUBIC)
                recovered.save(output_path, format="TIFF", compression="raw", dpi=(1, 1))

        with Image.open(output_path) as check:
            check.load()
            if check.mode != "RGB" or max(check.size) != maximum_dimension or check.n_frames != 1:
                raise RuntimeError(f"Recovered TIFF invariant failed: {output_path}")
            width, height = check.size
        records.append(
            {
                "source_pdf": str(pdf_path.resolve()),
                "source_pdf_sha256": _sha256(pdf_path),
                "output_tiff": str(output_path.resolve()),
                "output_tiff_sha256": _sha256(output_path),
                "render_dpi": dpi,
                "rendered_width": source_width,
                "rendered_height": source_height,
                "output_width": width,
                "output_height": height,
                "resize": "Pillow BICUBIC with integer-floor dimensions",
                "format": "TIFF",
                "mode": "RGB",
                "compression": "raw",
                "provenance": "recovered_from_author_source_pdf",
            }
        )
    return pd.DataFrame(records)
