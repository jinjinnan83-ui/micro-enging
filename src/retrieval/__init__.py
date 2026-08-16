"""Hybrid retrieval chain: dense vector + BM25 + BGE reranker."""

__all__ = ["HybridRetriever", "HybridEngine", "RetrievalChain", "index_documents", "query"]


def __getattr__(name: str):
    if name == "HybridRetriever":
        from src.retrieval.hybrid import HybridRetriever

        return HybridRetriever
    if name == "HybridEngine":
        from src.retrieval.hybrid_engine import HybridEngine

        return HybridEngine
    if name in {"index_documents", "query"}:
        from src.retrieval import hybrid_engine

        return getattr(hybrid_engine, name)
    if name == "RetrievalChain":
        from src.retrieval.chain import RetrievalChain

        return RetrievalChain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
