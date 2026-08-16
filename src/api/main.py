"""FastAPI application: health, ingest, and hybrid retrieval for Agents."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from src.api.schemas import (
    DialogueRequest,
    DialogueResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievalItem,
)
from src.config import Settings, get_settings
from src.dialogue.llm import LLMError


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = get_settings()
    yield


app = FastAPI(
    title="Psychoanalysis Knowledge Base",
    description="Agent-callable RAG API: hybrid search (dense + BM25) + BGE rerank.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_chain():
    from src.retrieval.chain import RetrievalChain

    if not getattr(app.state, "chain", None):
        app.state.chain = RetrievalChain(get_app_settings())
    return app.state.chain


def get_store():
    from src.vectorstore.qdrant_manager import QdrantStoreManager

    if not getattr(app.state, "store", None):
        app.state.store = QdrantStoreManager(get_app_settings())
    return app.state.store


def get_dialogue_service():
    from src.dialogue.service import DialogueService

    if not getattr(app.state, "dialogue_service", None):
        app.state.dialogue_service = DialogueService(get_app_settings())
    return app.state.dialogue_service


def get_app_settings() -> Settings:
    return getattr(app.state, "settings", None) or get_settings()


@app.get("/health", response_model=HealthResponse)
def health(
    store=Depends(get_store),
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    try:
        points = store.count()
        qdrant_status = "up"
    except Exception as exc:  # noqa: BLE001 — surface connectivity for operators
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    return HealthResponse(
        status="ok",
        qdrant=qdrant_status,
        collection=settings.qdrant_collection,
        points=points,
    )


@app.post("/query", response_model=QueryResponse)
def query_knowledge_base(
    payload: QueryRequest,
    chain=Depends(get_chain),
) -> QueryResponse:
    """Agent function: retrieve psychoanalytic passages with optional school/author filters."""
    hits = chain.query(
        question=payload.query,
        school=payload.school,
        author=payload.author,
        top_n=payload.top_n,
        rerank=payload.rerank,
    )
    return QueryResponse(
        query=payload.query,
        results=[RetrievalItem(**hit.as_dict()) for hit in hits],
    )


@app.post("/dialogue", response_model=DialogueResponse)
def psychodynamic_dialogue(
    payload: DialogueRequest,
    service=Depends(get_dialogue_service),
) -> DialogueResponse:
    """Generate a skill-grounded psychodynamic reply with optional trace data."""
    try:
        result = service.respond(
            payload.letter,
            preferred_school=payload.preferred_school,
            include_trace=payload.include_trace,
        )
    except (LLMError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return DialogueResponse.model_validate(result.model_dump())


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents(
    payload: IngestRequest,
    settings: Settings = Depends(get_app_settings),
) -> IngestResponse:
    """Parse, chunk (with metadata), and upsert files from data/raw or a given path."""
    from src.ingestion.pipeline import IngestionPipeline

    source = payload.path or str(settings.data_raw_dir)
    pipeline = IngestionPipeline(settings)
    chunks = pipeline.ingest_path(source)
    return IngestResponse(
        ingested_chunks=len(chunks),
        collection=settings.qdrant_collection,
    )


@app.get("/collections/{name}")
def collection_info(
    name: str,
    store=Depends(get_store),
) -> dict[str, Any]:
    if name != store.collection:
        raise HTTPException(status_code=404, detail="Unknown collection")
    return {"name": name, "points": store.count()}
