"""PDF to Markdown conversion using Docling (DS4SD/docling)."""

from pathlib import Path

_converter = None  # module-level cache — model loading is expensive


def pdf_to_markdown(pdf_path: Path) -> str:
    global _converter
    from docling.document_converter import DocumentConverter

    if _converter is None:
        print("[pdfer_docling] Loading Docling models (first call only)...")
        _converter = DocumentConverter()

    result = _converter.convert(str(pdf_path))
    text = result.document.export_to_markdown()
    return text
