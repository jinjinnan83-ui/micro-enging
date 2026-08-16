"""Retrieval chain used by the Agent-facing API."""

from __future__ import annotations

from typing import Any

from src.config import Settings, get_settings
from src.retrieval.hybrid import RetrievalHit


class RetrievalChain:
    """Delegate to HybridEngine: Qdrant + BM25 + BGE rerank (Top-20 → 3-5)."""

    def __init__(
        self,
        settings: Settings | None = None,
        engine: Any | None = None,
        retriever: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if engine is not None:
            self.engine = engine
        else:
            from src.retrieval.hybrid_engine import HybridEngine

            self.engine = HybridEngine(
                settings=self.settings,
                store=getattr(retriever, "store", None),
                embedder=getattr(retriever, "embedder", None),
                reranker=reranker,
                corpus=getattr(retriever, "_corpus", None),
            )

    def query(
        self,
        question: str,
        school: str | None = None,
        author: str | None = None,
        top_n: int | None = None,
        rerank: bool = True,
    ) -> list[RetrievalHit]:
        return self.engine.query(
            question,
            school=school,
            author=author,
            top_n=top_n,
            rerank=rerank,
        )

    def as_agent_payload(self, hits: list[RetrievalHit]) -> list[dict[str, Any]]:
        return [hit.as_dict() for hit in hits]
