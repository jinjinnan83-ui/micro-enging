"""Runtime configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Project settings. Values come from `.env` or process environment."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_collection: str = "psychoanalysis"
    qdrant_api_key: str = ""

    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    chunk_size: int = 512
    chunk_overlap: int = 80

    dense_top_k: int = 20
    bm25_top_k: int = 20
    hybrid_candidate_k: int = 20
    rerank_top_n: int = 5
    hybrid_fusion_weight: float = 0.6

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "info"

    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"

    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-v4-flash"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_thinking_type: str = "disabled"
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 2

    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
