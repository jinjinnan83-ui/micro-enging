"""Lazy BGE embedding client used by ingestion and dense retrieval."""

from __future__ import annotations

from functools import lru_cache

from src.config import Settings, get_settings


class EmbeddingClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = _load_embedding_model(self.settings.embedding_model)
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]


@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
