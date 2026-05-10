"""Convert a PDF-corpus to markdown, using various implementations."""

import shutil
from pathlib import Path

from ctm_ai_eval.pdf.pdfer import PDF_CONVERTERS
from ctm_ai_eval.utils.rich_print import CONS

CORPUS_PDF_PATH = Path("./tmp/corpora/lectures_subset")

OUT_ROOT = Path("./tmp/converted_corpora")


def _convert(indir: Path, convert_fn: str):
    outdir = OUT_ROOT / f"{indir.stem}_{convert_fn}"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    files = indir.rglob("*.pdf")
    if not files:
        raise FileNotFoundError("no pdf files found")

    for f in files:
        text_md = PDF_CONVERTERS[convert_fn](f)
        p = outdir / f"{f.stem}.md"
        p.write_text(text_md)
        print(f"saved {p}")


def _main():
    """Convert corpus, using each converter."""

    assert CORPUS_PDF_PATH.is_dir(), "expects a directory of PDFs"

    CONS.print(f"Converting corpus ({CORPUS_PDF_PATH}) with:", style="bold")
    for k in PDF_CONVERTERS:
        CONS.print(f"    {k}")

    # convert all
    for k in PDF_CONVERTERS:
        CONS.print(f"converting with: {k}")
        try:
            _convert(CORPUS_PDF_PATH, k)
        except Exception as err:
            print(f"[ERROR] {k} failed: {err}")


_main()
