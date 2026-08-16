"""End-to-end ingestion: parse → chunk → persist metadata → upsert to Qdrant."""

from __future__ import annotations

import json
from pathlib import Path

from typing import Any

from src.config import Settings, get_settings
from src.ingestion.chunker import TextChunk, chunk_document
from src.ingestion.parser import parse_document


class IngestionPipeline:
    """Load raw psychoanalytic files and write chunks into the vector store."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or self._default_store()

    def _default_store(self):
        from src.vectorstore.qdrant_manager import QdrantStoreManager

        return QdrantStoreManager(self.settings)

    def ingest_path(self, path: str | Path) -> list[TextChunk]:
        source = Path(path)
        files = [source] if source.is_file() else _iter_source_files(source)
        chunks: list[TextChunk] = []
        for file_path in files:
            parsed = parse_document(file_path)
            file_chunks = chunk_document(parsed)
            chunks.extend(file_chunks)
        self._write_processed(chunks)
        self.store.upsert_chunks(chunks)
        return chunks

    def to_llama_documents(self, chunks: list[TextChunk]):
        """Convert internal chunks to LlamaIndex documents for optional graph/index use."""
        from llama_index.core import Document

        return [
            Document(text=chunk.text, metadata=chunk.metadata, doc_id=f"{chunk.metadata['source_document']}:{chunk.chunk_index}")
            for chunk in chunks
        ]

    def _write_processed(self, chunks: list[TextChunk]) -> None:
        output_dir = self.settings.data_processed_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = [chunk.to_dict() for chunk in chunks]
        (output_dir / "chunks.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in payload),
            encoding="utf-8",
        )


def _iter_source_files(directory: Path) -> list[Path]:
    suffixes = {".md", ".txt", ".pdf"}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
