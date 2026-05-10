"""PDF to Markdown conversion using Marker (datalab-to/marker)."""

from pathlib import Path

# Models are expensive to load; cache them for the lifetime of the process.
_models = None


def pdf_to_markdown(pdf_path: Path) -> str:
    global _models
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    if _models is None:
        print("[pdfer_marker] Loading marker models (first call only)...")
        _models = create_model_dict()

    converter = PdfConverter(artifact_dict=_models)
    rendered = converter(str(pdf_path))
    text, _, _ = text_from_rendered(rendered)

    return text
