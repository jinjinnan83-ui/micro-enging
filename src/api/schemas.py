"""Request / response models for Agent function calling and REST clients."""

from typing import Any

from pydantic import BaseModel, Field

from src.dialogue.models import DialogueResult


class QueryRequest(BaseModel):
    query: str = Field(..., description="精神分析问题或概念，例如：拉康的镜像阶段")
    school: str | None = Field(None, description="流派过滤：Freud / Lacan / Jung 等")
    author: str | None = Field(None, description="作者过滤")
    top_n: int | None = Field(5, ge=3, le=5, description="BGE 重排后返回 3-5 条")
    rerank: bool = Field(True, description="是否使用 BGE reranker")


class RetrievalItem(BaseModel):
    text: str
    score: float
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    results: list[RetrievalItem]


class IngestRequest(BaseModel):
    path: str | None = Field(
        None,
        description="相对或绝对路径。默认读取 data/raw/",
    )


class IngestResponse(BaseModel):
    ingested_chunks: int
    collection: str


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    collection: str
    points: int


class DialogueRequest(BaseModel):
    letter: str = Field(..., min_length=1, max_length=30000, description="用户来信")
    preferred_school: str | None = Field(None, description="可选的精神分析流派")
    include_trace: bool = Field(False, description="返回基础/解释性检索词、两类命中文献和 token usage")


class DialogueResponse(DialogueResult):
    pass
