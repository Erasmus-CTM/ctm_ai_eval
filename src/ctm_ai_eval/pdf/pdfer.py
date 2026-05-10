"""Dispatcher — selects a PDF converter based on Config.PDF_CONVERTER."""

from collections.abc import Callable
from pathlib import Path

from . import pdfer_docling, pdfer_marker, pdfer_pymupdf

# Registry of each pdf-to-markdown implementation.
PDF_CONVERTERS: dict[str, Callable[[Path], str]] = {
    # "pymu": pdfer_pymupdf.pdf_to_markdown,
    # "docling": pdfer_docling.pdf_to_markdown,
    "marker": pdfer_marker.pdf_to_markdown,
}
