"""Hybrid search: dense Qdrant vectors + BM25, fused by reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import Settings, get_settings


@dataclass
class RetrievalHit:
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "hybrid"

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
        }


class HybridRetriever:
    """Dense vector retrieval + lexical BM25, then RRF fusion."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: Any | None = None,
        embedder: Any | None = None,
        corpus: list[dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or self._default_store()
        self.embedder = embedder or self._default_embedder()
        self._corpus = corpus or []
        self._bm25 = None
        if self._corpus:
            self._rebuild_bm25()

    def set_corpus(self, corpus: list[dict[str, Any]]) -> None:
        self._corpus = corpus
        self._rebuild_bm25()

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        dense_hits = self._dense_search(query, filters=filters)
        bm25_hits = self._bm25_search(query)
        fused = reciprocal_rank_fusion(
            [dense_hits, bm25_hits],
            weight=self.settings.hybrid_fusion_weight,
        )
        limit = top_k or (self.settings.dense_top_k + self.settings.bm25_top_k)
        return fused[:limit]

    def _dense_search(self, query: str, filters: dict[str, Any] | None) -> list[RetrievalHit]:
        vector = self.embedder.embed_query(query)
        raw = self.store.search(vector, top_k=self.settings.dense_top_k, filters=filters)
        return [
            RetrievalHit(
                text=item["text"],
                score=item["score"],
                metadata=item.get("metadata", {}),
                source="dense",
            )
            for item in raw
        ]

    def _bm25_search(self, query: str) -> list[RetrievalHit]:
        if self._bm25 is None or not self._corpus:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        hits: list[RetrievalHit] = []
        for index, score in ranked[: self.settings.bm25_top_k]:
            if score <= 0:
                continue
            row = self._corpus[index]
            hits.append(
                RetrievalHit(
                    text=row.get("text", ""),
                    score=float(score),
                    metadata=row,
                    source="bm25",
                )
            )
        return hits

    def _default_store(self):
        from src.vectorstore.qdrant_manager import QdrantStoreManager

        return QdrantStoreManager(self.settings)

    def _default_embedder(self):
        from src.retrieval.embeddings import EmbeddingClient

        return EmbeddingClient(self.settings)

    def _rebuild_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        tokenized = [_tokenize(row.get("text", "")) for row in self._corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalHit]],
    k: int = 60,
    weight: float = 0.6,
) -> list[RetrievalHit]:
    """Fuse dense (weighted) and BM25 lists by text identity."""
    scores: dict[str, float] = {}
    payloads: dict[str, RetrievalHit] = {}
    for list_index, hits in enumerate(ranked_lists):
        list_weight = weight if list_index == 0 else (1.0 - weight)
        for rank, hit in enumerate(hits, start=1):
            key = hit.text
            scores[key] = scores.get(key, 0.0) + list_weight * (1.0 / (k + rank))
            if key not in payloads or hit.score > payloads[key].score:
                payloads[key] = hit
    fused = [
        RetrievalHit(
            text=payloads[key].text,
            score=score,
            metadata=payloads[key].metadata,
            source="hybrid",
        )
        for key, score in scores.items()
    ]
    fused.sort(key=lambda item: item.score, reverse=True)
    return fused


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().replace("，", " ").replace("。", " ").split() if token]
