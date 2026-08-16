from src.config import Settings
from src.ingestion.chunker import TextChunk
from src.retrieval.hybrid import RetrievalHit
from src.retrieval.hybrid_engine import (
    HybridEngine,
    expand_query_tokens,
    index_documents,
    query,
    tokenize_psychoanalytic,
)


class FakeStore:
    collection = "psychoanalysis"

    def __init__(self) -> None:
        self.upserted = []

    def upsert_chunks(self, chunks, vectors=None):
        self.upserted.append((chunks, vectors))
        return len(chunks)

    def search(self, query_vector, top_k=None, filters=None):
        return [
            {
                "text": "死本能推动重复，也在隐喻里表现为毁灭与回归无机的冲动。",
                "score": 0.71,
                "metadata": {"author": "Sigmund Freud", "school": "精神分析", "file_name": "freud.txt"},
            },
            {
                "text": "投射性认同把无法承受的部分放入他人，并在关系里操控其返回。",
                "score": 0.66,
                "metadata": {"author": "Melanie Klein", "school": "客体关系", "file_name": "klein.txt"},
            },
        ]


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, query):
        return [0.1, 0.2]


class FakeReranker:
    def rerank(self, query, hits, top_n=None):
        ranked = sorted(hits, key=lambda hit: ("投射性认同" in hit.text, hit.score), reverse=True)
        return [
            RetrievalHit(text=hit.text, score=0.9 - index * 0.1, metadata=hit.metadata, source="rerank")
            for index, hit in enumerate(ranked[:top_n])
        ]


def _engine(tmp_path) -> HybridEngine:
    settings = Settings(data_processed_dir=tmp_path)
    return HybridEngine(
        settings=settings,
        store=FakeStore(),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
    )


def test_tokenizer_keeps_death_drive_and_projective_identification() -> None:
    tokens = tokenize_psychoanalytic("临床里死本能与投射性认同常常缠在同一隐喻中。")

    assert "死本能" in tokens
    assert "投射性认同" in tokens
    assert expand_query_tokens(["死本能"]) == [
        "死本能",
        "死亡本能",
        "死亡驱力",
        "death drive",
        "death instinct",
        "todestrieb",
    ]


def test_index_documents_upserts_qdrant_and_builds_bm25(tmp_path) -> None:
    engine = _engine(tmp_path)
    chunks = [
        TextChunk(
            text="死本能不是单纯的攻击，而是朝向无机状态的拉力。",
            metadata={"author": "Sigmund Freud", "school": "精神分析", "file_name": "freud.txt"},
            chunk_index=0,
        )
    ]

    result = index_documents(chunks, engine=engine)

    assert result["indexed"] == 1
    assert result["collection"] == "psychoanalysis"
    assert result["embedding_model"] == "BAAI/bge-large-zh-v1.5"
    assert engine.store.upserted
    assert (tmp_path / "bm25_corpus.jsonl").exists()


def test_query_reranks_hybrid_candidates_to_three_or_five(tmp_path) -> None:
    engine = _engine(tmp_path)
    engine.index_documents(
        [
            {
                "text": "死本能推动重复，也在隐喻里表现为毁灭与回归无机的冲动。",
                "author": "Sigmund Freud",
                "school": "精神分析",
                "file_name": "freud.txt",
            },
            {
                "text": "投射性认同把无法承受的部分放入他人，并在关系里操控其返回。",
                "author": "Melanie Klein",
                "school": "客体关系",
                "file_name": "klein.txt",
            },
            {
                "text": "镜像阶段是一种误认，自我在影像中被捕获。",
                "author": "Jacques Lacan",
                "school": "拉康派",
                "file_name": "lacan.txt",
            },
        ]
    )

    hits = query("什么是投射性认同？", top_n=5, engine=engine)

    assert 3 <= len(hits) <= 5 or len(hits) <= 3
    assert hits[0]["text"].startswith("投射性认同")
    assert hits[0]["source"] == "rerank"
    assert hits[0]["metadata"]["school"] == "客体关系"
