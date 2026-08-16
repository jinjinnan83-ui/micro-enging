"""Document parsing, psychoanalytic metadata extraction, and chunking."""

from src.ingestion.chunker import chunk_document
from src.ingestion.parser import ParsedDocument, parse_document

__all__ = [
    "ParsedDocument",
    "parse_document",
    "chunk_document",
    "chunk_source",
    "IngestionPipeline",
]


def __getattr__(name: str):
    if name == "IngestionPipeline":
        from src.ingestion.pipeline import IngestionPipeline

        return IngestionPipeline
    if name == "chunk_source":
        from src.ingestion.chunking import chunk_source

        return chunk_source
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
