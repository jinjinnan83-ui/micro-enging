from fastapi.testclient import TestClient

from src.api.main import app, get_app_settings, get_chain, get_store
from src.config import Settings
from src.retrieval.hybrid import RetrievalHit


class FakeStore:
    collection = "psychoanalysis"

    def count(self) -> int:
        return 2


class FakeChain:
    def query(self, question: str, school=None, author=None, top_n=None, rerank=True):
        return [
            RetrievalHit(
                text="无意识通过梦与口误返回。",
                score=0.91,
                metadata={
                    "author": "Sigmund Freud",
                    "school": school or "Freud",
                    "source_document": "freud_unconscious.md",
                    "concepts": ["unconscious"],
                },
                source="rerank",
            )
        ]


def test_health_endpoint_reports_qdrant_and_collection() -> None:
    app.dependency_overrides[get_store] = lambda: FakeStore()
    app.dependency_overrides[get_app_settings] = lambda: Settings()
    client = TestClient(app)

    response = client.get("/health")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["collection"] == "psychoanalysis"
    assert body["points"] == 2


def test_query_endpoint_returns_agent_friendly_payload() -> None:
    app.dependency_overrides[get_chain] = lambda: FakeChain()
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "什么是无意识？", "school": "Freud", "rerank": True},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "什么是无意识？"
    assert body["results"][0]["metadata"]["school"] == "Freud"
    assert body["results"][0]["source"] == "rerank"
