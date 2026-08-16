#!/usr/bin/env python3
"""Retrieve Bandura knowledge-base passages for coaching dialogue.

Default path is fast BM25 over data/processed/chunks.jsonl.
Use --hybrid only when dense + rerank is explicitly needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        chunks = parent / "data" / "processed" / "chunks.jsonl"
        if chunks.exists():
            return parent
        if (parent / "src" / "retrieval" / "hybrid_engine.py").exists():
            return parent
    return Path.cwd()


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.hybrid_engine import expand_query_tokens, tokenize_bandura


def load_corpus(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("text"):
            rows.append(row)
    return rows


def bm25_search(
    query: str,
    corpus: list[dict[str, Any]],
    *,
    top_n: int,
    school: str | None,
    author: str | None,
) -> list[dict[str, Any]]:
    from rank_bm25 import BM25Okapi

    filtered = corpus
    if school:
        filtered = [row for row in filtered if row.get("school") == school]
    if author:
        filtered = [row for row in filtered if author in str(row.get("author") or "")]
    if not filtered:
        return []

    tokenized = [tokenize_bandura(row.get("text", "")) for row in filtered]
    bm25 = BM25Okapi(tokenized)
    tokens = expand_query_tokens(tokenize_bandura(query))
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    hits: list[dict[str, Any]] = []
    for index, score in ranked[:top_n]:
        if score <= 0:
            continue
        hits.append(_to_hit(filtered[index], float(score), source="bm25"))
    return hits


def hybrid_search(
    query: str,
    *,
    top_n: int,
    school: str | None,
    author: str | None,
) -> list[dict[str, Any]]:
    from src.retrieval.hybrid_engine import HybridEngine

    hits = HybridEngine().query(query, top_n=top_n, school=school, author=author, rerank=True)
    return [
        {
            "text": hit.text,
            "score": hit.score,
            "source": hit.source,
            "author": hit.metadata.get("author"),
            "school": hit.metadata.get("school"),
            "core_concepts": hit.metadata.get("core_concepts") or hit.metadata.get("concepts") or [],
            "file_name": hit.metadata.get("file_name") or hit.metadata.get("source_document"),
        }
        for hit in hits
    ]


def _to_hit(row: dict[str, Any], score: float, source: str) -> dict[str, Any]:
    return {
        "text": row.get("text", ""),
        "score": score,
        "source": source,
        "author": row.get("author"),
        "school": row.get("school"),
        "core_concepts": row.get("core_concepts") or row.get("concepts") or [],
        "file_name": row.get("file_name") or row.get("source_document"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query the Bandura knowledge base.")
    parser.add_argument("query", help="用户原话，或改写成的理论检索句")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--school",
        default=None,
        help="社会认知理论 / 自我效能 / 观察学习 / 道德疏离 / 目标与自我调节 / 干预与测量",
    )
    parser.add_argument("--author", default=None)
    parser.add_argument("--hybrid", action="store_true", help="dense + BM25 + BGE rerank（较慢）")
    args = parser.parse_args(argv)

    query = (args.query or "").strip()
    if not query:
        print(json.dumps({"error": "empty query"}, ensure_ascii=False))
        return 1

    chunks_path = ROOT / "data" / "processed" / "chunks.jsonl"
    payload: dict[str, Any] = {
        "query": query,
        "mode": "hybrid" if args.hybrid else "bm25",
        "project_root": str(ROOT),
        "hits": [],
    }

    try:
        if args.hybrid:
            payload["hits"] = hybrid_search(
                query, top_n=args.top_n, school=args.school, author=args.author
            )
        else:
            if not chunks_path.exists():
                payload["error"] = f"missing corpus: {chunks_path}"
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 1
            corpus = load_corpus(chunks_path)
            payload["corpus_size"] = len(corpus)
            payload["hits"] = bm25_search(
                query,
                corpus,
                top_n=args.top_n,
                school=args.school,
                author=args.author,
            )
    except Exception as exc:  # noqa: BLE001
        payload["error"] = str(exc)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
