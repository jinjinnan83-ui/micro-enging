"""Hybrid retrieval engine for the psychoanalysis knowledge base.

Dense vectors (Qdrant + bge-large-zh-v1.5) capture metaphoric wording.
BM25 with a domain-term tokenizer captures exact theoretical terms
such as 死本能 and 投射性认同. BGE-Reranker-v2-m3 then reranks the
fused Top-20 and returns the best 3-5 passages.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import get_settings
from src.ingestion.chunker import TextChunk
from src.retrieval.hybrid import RetrievalHit, reciprocal_rank_fusion

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
CANDIDATE_K = 20
RERANK_MIN = 3
RERANK_MAX = 5

# Longest-first so 投射性认同 is not split into 投射 + 认同.
DOMAIN_TERMS = (
    "投射性认同",
    "偏执-分裂位置",
    "抑郁位置",
    "过渡客体",
    "部分客体",
    "镜像阶段",
    "死亡驱力",
    "死亡本能",
    "死本能",
    "梦的工作",
    "反移情",
    "俄狄浦斯",
    "对象a",
    "大他者",
    "实在界",
    "想象界",
    "象征界",
    "无意识",
    "移情",
    "压抑",
    "阉割",
    "自恋",
    "能指",
    "欲望",
    "抱持",
    "容纳",
    "分裂",
    "projective identification",
    "death instinct",
    "death drive",
    "todestrieb",
    "mirror stage",
    "transference",
    "countertransference",
    "unconscious",
    "repression",
)

TERM_ALIASES = {
    "死本能": ("死本能", "死亡本能", "死亡驱力", "death drive", "death instinct", "todestrieb"),
    "死亡本能": ("死本能", "死亡本能", "死亡驱力", "death drive", "death instinct", "todestrieb"),
    "死亡驱力": ("死本能", "死亡本能", "死亡驱力", "death drive", "death instinct", "todestrieb"),
    "death drive": ("死本能", "死亡本能", "死亡驱力", "death drive", "death instinct", "todestrieb"),
    "death instinct": ("死本能", "死亡本能", "死亡驱力", "death drive", "death instinct", "todestrieb"),
    "todestrieb": ("死本能", "死亡本能", "死亡驱力", "death drive", "death instinct", "todestrieb"),
    "投射性认同": ("投射性认同", "projective identification"),
    "projective identification": ("投射性认同", "projective identification"),
    "镜像阶段": ("镜像阶段", "mirror stage"),
    "mirror stage": ("镜像阶段", "mirror stage"),
}

SORTED_TERMS = tuple(sorted(DOMAIN_TERMS, key=len, reverse=True))


def tokenize_psychoanalytic(text: str) -> list[str]:
    """Tokenize for BM25: keep domain terms intact, then CJK n-grams + Latin words."""
    if not text:
        return []
    haystack = text
    tokens: list[str] = []
    for term in SORTED_TERMS:
        count = _count_term(haystack, term)
        if not count:
            continue
        tokens.extend([term.lower() if term.isascii() else term] * count)
        haystack = _mask_term(haystack, term)
    tokens.extend(word.lower() for word in LATIN_RE.findall(haystack))
    chars = CJK_RE.findall(haystack)
    tokens.extend(chars)
    tokens.extend("".join(chars[i : i + 2]) for i in range(len(chars) - 1))
    return tokens


def expand_query_tokens(tokens: list[str]) -> list[str]:
    """Add theoretical aliases so 死本能 also matches 死亡驱力 / death drive."""
    expanded = list(tokens)
    seen = set(tokens)
    for token in tokens:
        for alias in TERM_ALIASES.get(token, ()):
            key = alias.lower() if alias.isascii() else alias
            if key not in seen:
                expanded.append(key)
                seen.add(key)
    return expanded


class HybridEngine:
    """Index chunks into Qdrant and run hybrid search + BGE rerank."""

    def __init__(
        self,
        settings: Any | None = None,
        store: Any | None = None,
        embedder: Any | None = None,
        reranker: Any | None = None,
        corpus: list[dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self._corpus = list(corpus or [])
        self._bm25 = None
        if self._corpus:
            self._rebuild_bm25()

    @property
    def collection(self) -> str:
        return self.settings.qdrant_collection

    @property
    def embedding_model(self) -> str:
        return self.settings.embedding_model

    @property
    def reranker_model(self) -> str:
        return self.settings.reranker_model

    def index_documents(
        self,
        sources: str | Path | Sequence[Any] | None = None,
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Embed chunks with bge-large-zh-v1.5 and upsert them into Qdrant.

        `sources` may be a file, a directory, TextChunk objects, or dicts
        with `text` plus psychoanalytic metadata.
        """
        chunks = self._normalize_sources(sources)
        if not chunks:
            return {"indexed": 0, "collection": collection or self.collection, "embedding_model": self.embedding_model}

        texts = [chunk.text for chunk in chunks]
        vectors = self._get_embedder().embed(texts)
        store = self._get_store()
        if collection:
            store.collection = collection
        stored = store.upsert_chunks(chunks, vectors=vectors)

        records = [self._chunk_to_record(chunk, index) for index, chunk in enumerate(chunks)]
        self._corpus = records
        self._rebuild_bm25()
        self._write_corpus(records)
        return {
            "indexed": stored,
            "collection": store.collection,
            "embedding_model": self.embedding_model,
            "bm25_docs": len(records),
        }

    def query(
        self,
        question: str,
        *,
        top_n: int | None = None,
        school: str | None = None,
        author: str | None = None,
        rerank: bool = True,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        """Hybrid search over metaphor + exact terms, then rerank Top-20 to 3-5 hits."""
        if not question or not question.strip():
            return []
        self._ensure_bm25()
        merged_filters = {**(filters or {})}
        if school:
            merged_filters["school"] = school
        if author:
            merged_filters["author"] = author

        dense_hits = self._dense_search(question, filters=merged_filters or None)
        bm25_hits = self._bm25_search(question)
        fused = reciprocal_rank_fusion(
            [dense_hits, bm25_hits],
            weight=getattr(self.settings, "hybrid_fusion_weight", 0.6),
        )[:CANDIDATE_K]

        limit = _clamp_top_n(top_n or getattr(self.settings, "rerank_top_n", RERANK_MAX))
        if rerank:
            return self._get_reranker().rerank(question, fused, top_n=limit)
        return fused[:limit]

    def _dense_search(self, question: str, filters: dict[str, Any] | None) -> list[RetrievalHit]:
        vector = self._get_embedder().embed_query(question)
        raw = self._get_store().search(vector, top_k=CANDIDATE_K, filters=filters)
        return [
            RetrievalHit(
                text=item.get("text", ""),
                score=float(item.get("score", 0.0)),
                metadata=item.get("metadata", {}),
                source="dense",
            )
            for item in raw
            if item.get("text")
        ]

    def _bm25_search(self, question: str) -> list[RetrievalHit]:
        if self._bm25 is None or not self._corpus:
            return []
        tokens = expand_query_tokens(tokenize_psychoanalytic(question))
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        hits: list[RetrievalHit] = []
        for index, score in ranked[:CANDIDATE_K]:
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

    def _normalize_sources(self, sources: str | Path | Sequence[Any] | None) -> list[TextChunk]:
        if sources is None:
            sources = self.settings.data_raw_dir
        if isinstance(sources, (str, Path)):
            return self._chunks_from_path(Path(sources))
        chunks: list[TextChunk] = []
        for item in sources:
            if isinstance(item, TextChunk):
                chunks.append(item)
            elif isinstance(item, (str, Path)):
                chunks.extend(self._chunks_from_path(Path(item)))
            elif isinstance(item, dict):
                chunks.append(_dict_to_chunk(item, index=len(chunks)))
            else:
                raise TypeError(f"Unsupported index source: {type(item)!r}")
        return chunks

    def _chunks_from_path(self, path: Path) -> list[TextChunk]:
        from src.ingestion.chunking import SUPPORTED_SUFFIXES, chunk_source, iter_source_files

        files = [path] if path.is_file() else iter_source_files(path)
        chunks: list[TextChunk] = []
        for file_path in files:
            if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            chunks.extend(chunk_source(file_path))
        return chunks

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        records = self._read_corpus()
        if not records:
            records = self._scroll_store_corpus()
        self._corpus = records
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        tokenized = [tokenize_psychoanalytic(row.get("text", "")) for row in self._corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _corpus_path(self) -> Path:
        path = Path(self.settings.data_processed_dir) / "bm25_corpus.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _write_corpus(self, records: list[dict[str, Any]]) -> None:
        self._corpus_path().write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in records),
            encoding="utf-8",
        )

    def _read_corpus(self) -> list[dict[str, Any]]:
        path = self._corpus_path()
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def _scroll_store_corpus(self) -> list[dict[str, Any]]:
        store = self.store
        client = getattr(store, "client", None)
        collection = getattr(store, "collection", None)
        if client is None or collection is None:
            return []
        try:
            points, _offset = client.scroll(collection_name=collection, limit=2048, with_payload=True)
        except Exception:
            return []
        records = []
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            if payload.get("text"):
                records.append(payload)
        return records

    def _get_store(self):
        if self.store is None:
            from src.vectorstore.qdrant_manager import QdrantStoreManager

            self.store = QdrantStoreManager(self.settings)
        return self.store

    def _get_embedder(self):
        if self.embedder is None:
            from src.retrieval.embeddings import EmbeddingClient

            self.embedder = EmbeddingClient(self.settings)
        return self.embedder

    def _get_reranker(self):
        if self.reranker is None:
            from src.retrieval.reranker import BGEReranker

            self.reranker = BGEReranker(self.settings)
        return self.reranker

    @staticmethod
    def _chunk_to_record(chunk: TextChunk, index: int) -> dict[str, Any]:
        record = chunk.to_dict()
        record.setdefault("chunk_index", index)
        return record


def index_documents(
    sources: str | Path | Sequence[Any] | None = None,
    *,
    collection: str | None = None,
    engine: HybridEngine | None = None,
) -> dict[str, Any]:
    """Index psychoanalysis chunks into Qdrant (bge-large-zh-v1.5) and refresh BM25."""
    engine = engine or HybridEngine()
    return engine.index_documents(sources, collection=collection)


def query(
    question: str,
    *,
    top_n: int | None = None,
    school: str | None = None,
    author: str | None = None,
    rerank: bool = True,
    engine: HybridEngine | None = None,
) -> list[dict[str, Any]]:
    """Query the hybrid index and return the top 3-5 reranked passages as dicts."""
    engine = engine or HybridEngine()
    hits = engine.query(
        question,
        top_n=top_n,
        school=school,
        author=author,
        rerank=rerank,
    )
    return [hit.as_dict() for hit in hits]


def _clamp_top_n(value: int) -> int:
    return max(RERANK_MIN, min(RERANK_MAX, int(value)))


def _count_term(text: str, term: str) -> int:
    if term.isascii():
        return len(re.findall(rf"\b{re.escape(term)}\b", text, flags=re.I))
    return text.count(term)


def _mask_term(text: str, term: str) -> str:
    if term.isascii():
        return re.sub(rf"\b{re.escape(term)}\b", " ", text, flags=re.I)
    return text.replace(term, " ")


def _dict_to_chunk(item: dict[str, Any], index: int) -> TextChunk:
    text = str(item.get("text") or "").strip()
    if not text:
        raise ValueError("Index record is missing `text`.")
    metadata = {
        "author": item.get("author", "unknown"),
        "school": item.get("school", "精神分析"),
        "core_concepts": item.get("core_concepts") or item.get("concepts") or [],
        "file_name": item.get("file_name") or item.get("source_document") or "",
    }
    extra = {
        key: value
        for key, value in item.items()
        if key not in {"text", "author", "school", "core_concepts", "concepts", "file_name", "chunk_index"}
    }
    metadata.update(extra)
    return TextChunk(text=text, metadata=metadata, chunk_index=item.get("chunk_index", index))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index or query the psychoanalysis hybrid engine.")
    sub = parser.add_subparsers(dest="command", required=True)
    index_cmd = sub.add_parser("index", help="Index PDF/TXT files or a directory")
    index_cmd.add_argument("--path", default=None, help="File or directory. Defaults to data/raw/")
    query_cmd = sub.add_parser("query", help="Hybrid query + BGE rerank")
    query_cmd.add_argument("question", help="例如：什么是投射性认同 / 死本能")
    query_cmd.add_argument("--top-n", type=int, default=5)
    query_cmd.add_argument("--school", default=None)
    query_cmd.add_argument("--author", default=None)
    args = parser.parse_args(argv)

    if args.command == "index":
        result = index_documents(args.path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    hits = query(args.question, top_n=args.top_n, school=args.school, author=args.author)
    print(json.dumps(hits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
