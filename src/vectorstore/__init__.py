"""Qdrant collection lifecycle and chunk upsert."""

__all__ = ["QdrantStoreManager"]


def __getattr__(name: str):
    if name == "QdrantStoreManager":
        from src.vectorstore.qdrant_manager import QdrantStoreManager

        return QdrantStoreManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
