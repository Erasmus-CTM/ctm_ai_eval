"""PDF to Markdown conversion using PyMuPDF4LLM with layout mode."""

from pathlib import Path
from typing import Any


def pdf_to_markdown(pdf_path: Path) -> str:
    """
    Convert PDF to markdown using PyMuPDF4LLM with layout mode.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of dicts with keys:
          - page: page number (1-indexed)
          - text: markdown text from that page

    Raises:
        Exception: If PDF conversion fails
    """
    # Activate layout mode for better structure detection
    import pymupdf.layout  # noqa: F401
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    # Debug: print structure of first chunk if chunks exist
    if chunks:
        print(f"[pdfer] First chunk keys: {list(chunks[0].keys())}")
        if "metadata" in chunks[0]:
            print(f"[pdfer] Metadata keys: {list(chunks[0]['metadata'].keys())}")

    result: list[str] = []
    for chunk in chunks:
        # Handle different possible structures
        if isinstance(chunk, dict):
            if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                # PyMuPDF4LLM uses 'page_number' in metadata
                page = chunk["metadata"].get("page_number", 1)
            elif "page" in chunk:
                page = chunk["page"]
            else:
                page = 1  # Default to page 1 if not found

            text = chunk.get("text", "")
            if text:
                assert isinstance(text, str)
                result.append(text)

    return "\n\n".join(result)
