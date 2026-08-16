"""BGE cross-encoder reranker applied after hybrid fusion."""

from __future__ import annotations

from functools import lru_cache

from src.config import Settings, get_settings
from src.retrieval.hybrid import RetrievalHit


class BGEReranker:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def rerank(self, query: str, hits: list[RetrievalHit], top_n: int | None = None) -> list[RetrievalHit]:
        if not hits:
            return []
        model = _load_reranker(self.settings.reranker_model)
        pairs = [(query, hit.text) for hit in hits]
        scores = model.predict(pairs)
        ranked = sorted(
            zip(hits, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        limit = top_n or self.settings.rerank_top_n
        reranked: list[RetrievalHit] = []
        for hit, score in ranked[:limit]:
            reranked.append(
                RetrievalHit(
                    text=hit.text,
                    score=float(score),
                    metadata=hit.metadata,
                    source="rerank",
                )
            )
        return reranked


@lru_cache(maxsize=1)
def _load_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)
