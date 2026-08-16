"""Qdrant collection management for the psychoanalysis knowledge base."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import Settings, get_settings
from src.ingestion.chunker import TextChunk


class QdrantStoreManager:
    """Create, inspect, upsert, and query the psychoanalysis collection."""

    def __init__(self, settings: Settings | None = None, client: QdrantClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or _connect_qdrant(self.settings)
        self.collection = self.settings.qdrant_collection

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(
                size=self.settings.embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        for field_name in ("author", "school", "source_document"):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

    def upsert_chunks(self, chunks: list[TextChunk], vectors: list[list[float]] | None = None) -> int:
        """Upsert chunks. If vectors are omitted, embeddings are computed lazily."""
        if not chunks:
            return 0
        self.ensure_collection()
        embeddings = vectors or self._embed([chunk.text for chunk in chunks])
        points = [
            qmodels.PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload=chunk.to_dict(),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_collection()
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k or self.settings.dense_top_k,
            query_filter=_to_qdrant_filter(filters),
            with_payload=True,
        )
        hits = []
        for point in response.points:
            payload = dict(point.payload or {})
            hits.append(
                {
                    "id": str(point.id),
                    "score": float(point.score),
                    "text": payload.get("text", ""),
                    "metadata": payload,
                }
            )
        return hits

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        from src.retrieval.embeddings import EmbeddingClient

        return EmbeddingClient(self.settings).embed(texts)


def _connect_qdrant(settings: Settings) -> QdrantClient:
    """Prefer Docker Qdrant; fall back to a local embedded store."""
    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=2,
        )
        client.get_collections()
        return client
    except Exception:
        local_dir = settings.data_processed_dir.parent / "qdrant_storage"
        local_dir.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(local_dir))


def _to_qdrant_filter(filters: dict[str, Any] | None) -> qmodels.Filter | None:
    if not filters:
        return None
    conditions = [
        qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
        for key, value in filters.items()
        if value is not None
    ]
    return qmodels.Filter(must=conditions) if conditions else None
