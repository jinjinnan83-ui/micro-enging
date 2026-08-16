"""Chunk psychoanalytic texts while keeping metaphoric context and metadata.

Dense theoretical prose is split by paragraph first, then by size, with
overlap so a concept is not severed from its surrounding metaphor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.config import get_settings
from src.ingestion.parser import ParsedDocument

PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class TextChunk:
    """A retrieval unit that always carries psychoanalytic metadata."""

    text: str
    metadata: dict[str, Any]
    chunk_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            **self.metadata,
        }


def chunk_document(
    document: ParsedDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """Split a parsed document and attach Author/School/Concepts/Source to every chunk."""
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    paragraphs = [p.strip() for p in PARAGRAPH_SPLIT.split(document.text) if p.strip()]
    windows = _window_paragraphs(paragraphs, size=size, overlap=overlap)

    chunks: list[TextChunk] = []
    for index, text in enumerate(windows):
        metadata = document.metadata()
        metadata["chunk_index"] = index
        chunks.append(TextChunk(text=text, metadata=metadata, chunk_index=index))
    return chunks


def _window_paragraphs(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    if not paragraphs:
        return []

    windows: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for paragraph in paragraphs:
        if buffer and buffer_len + len(paragraph) > size:
            windows.append("\n\n".join(buffer))
            buffer, buffer_len = _overlap_tail(buffer, overlap)
        buffer.append(paragraph)
        buffer_len += len(paragraph)

    if buffer:
        windows.append("\n\n".join(buffer))
    return windows


def _overlap_tail(buffer: list[str], overlap: int) -> tuple[list[str], int]:
    if overlap <= 0 or not buffer:
        return [], 0
    tail: list[str] = []
    length = 0
    for paragraph in reversed(buffer):
        tail.insert(0, paragraph)
        length += len(paragraph)
        if length >= overlap:
            break
    return tail, length
