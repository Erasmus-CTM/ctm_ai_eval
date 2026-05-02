from ctm_ai_eval.rag import text_processing
from ctm_ai_eval.rag.chunkers.basic_chunking import TokenChunker
from ctm_ai_eval.rag.chunkers.chunk_markdown import MarkdownChunker
from ctm_ai_eval.rag.config import ChunkerConfig


def resolve_chunker(cfg: ChunkerConfig):
    """Get the chnker for a config."""
    if cfg.type == "word":
        return TokenChunker(cfg.length, cfg.overlap, text_processing.tokenize_words)
    elif cfg.type == "sentence":
        return TokenChunker(cfg.length, cfg.overlap, text_processing.tokenize_sentences)
    elif cfg.type == "markdown":
        return MarkdownChunker(cfg.length, cfg.overlap)
    else:
        raise ValueError(f"unknown type: {cfg.type}")
