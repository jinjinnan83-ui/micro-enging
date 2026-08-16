"""Fast local BM25 retrieval for psychodynamic dialogue context."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.config import Settings
from src.dialogue.models import KnowledgePassage
from src.retrieval.hybrid_engine import expand_query_tokens, tokenize_psychoanalytic


class DialogueKnowledgeBase:
    def __init__(self, settings: Settings) -> None:
        self.corpus_path = Path(settings.data_processed_dir) / "chunks.jsonl"
        self._rows: list[dict[str, Any]] | None = None

    def query(
        self,
        query: str,
        *,
        top_n: int = 3,
        school: str | None = None,
    ) -> list[KnowledgePassage]:
        rows = self._load_rows()
        if not rows:
            return []

        tokenized = [tokenize_psychoanalytic(str(row.get("text") or "")) for row in rows]
        query_tokens = expand_query_tokens(tokenize_psychoanalytic(query))
        scores = BM25Okapi(tokenized).get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        passages: list[KnowledgePassage] = []
        query_token_set = set(query_tokens)
        for index, score in ranked:
            row = rows[index]
            if school and row.get("school") != school:
                continue
            if not query_token_set.intersection(tokenized[index]):
                continue
            passages.append(
                KnowledgePassage(
                    query=query,
                    text=str(row.get("text") or ""),
                    score=float(score),
                    author=_optional_string(row.get("author")),
                    school=_optional_string(row.get("school")),
                    core_concepts=list(row.get("core_concepts") or row.get("concepts") or []),
                    source_document=_optional_string(
                        row.get("source_document") or row.get("file_name")
                    ),
                )
            )
            if len(passages) >= top_n:
                break
        return passages

    def _load_rows(self) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Knowledge corpus not found: {self.corpus_path}")
        self._rows = [
            json.loads(line)
            for line in self.corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return self._rows


def _optional_string(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None
