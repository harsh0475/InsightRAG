"""Application settings and environment configuration."""
from functools import lru_cache
from typing import Any, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """InsightRAG system configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Metadata
    ENVIRONMENT: str = Field(default="development", description="Runtime environment: development, staging, production")
    APP_NAME: str = Field(default="InsightRAG", description="Application display name")
    APP_VERSION: str = Field(default="0.1.0", description="SemVer application version")
    DEBUG: bool = Field(default=True, description="Enable debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR")

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v: Any) -> bool:
        """Safely parse boolean for DEBUG even if OS env sets strings like 'release'."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "t", "on", "dev", "development")
        return bool(v)

    # API Server
    API_HOST: str = Field(default="0.0.0.0", description="Host to bind API server")
    API_PORT: int = Field(default=8000, description="Port to bind API server")
    API_V1_PREFIX: str = Field(default="/api/v1", description="Prefix for API version 1 routes")
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origin URLs",
    )

    # Database & Storage
    POSTGRES_USER: str = Field(default="insightrag_user")
    POSTGRES_PASSWORD: str = Field(default="insightrag_secret_change_in_production")
    POSTGRES_DB: str = Field(default="insightrag_db")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    DATABASE_URL: str = Field(
        default="postgresql+psycopg://insightrag_user:insightrag_secret_change_in_production@localhost:5432/insightrag_db",
        description="PostgreSQL connection string with pgvector support",
    )

    # LLM Provider Configuration
    LLM_PROVIDER: str = Field(default="openai", description="LLM provider: openai, anthropic, or custom")
    LLM_API_KEY: str = Field(default="", description="API key for LLM provider")
    LLM_BASE_URL: str = Field(default="https://api.openai.com/v1", description="Base URL for OpenAI-compatible LLM endpoint")
    LLM_MODEL: str = Field(default="gpt-4o-mini", description="Model identifier for generation")
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature for generation")
    LLM_MAX_TOKENS: int = Field(default=1024, gt=0, description="Max tokens for LLM generation")

    # Embedding Provider Configuration
    EMBEDDING_PROVIDER: str = Field(default="openai", description="Embedding provider: openai, fastembed, huggingface")
    EMBEDDING_API_KEY: str = Field(default="", description="API key for embedding provider")
    EMBEDDING_BASE_URL: str = Field(default="https://api.openai.com/v1", description="Base URL for embedding endpoint")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="Embedding model name")
    EMBEDDING_DIMENSION: int = Field(default=1536, gt=0, description="Vector dimension of embeddings")

    # Ingestion & Chunking Defaults
    DEFAULT_CHUNK_SIZE: int = Field(default=500, gt=0, description="Target chunk size in tokens/words")
    DEFAULT_CHUNK_OVERLAP: int = Field(default=50, ge=0, description="Overlap between consecutive chunks")

    # Retrieval & Reranker Configuration
    RETRIEVAL_MODE: str = Field(default="hybrid", description="Default retrieval mode: vector, bm25, hybrid")
    DEFAULT_TOP_K: int = Field(default=5, gt=0, description="Default number of chunks returned for generation")
    BM25_K1: float = Field(default=1.5, ge=0.0, description="BM25 term frequency saturation parameter")
    BM25_B: float = Field(default=0.75, ge=0.0, le=1.0, description="BM25 document length normalization parameter")
    HYBRID_RETRIEVAL_ALPHA: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight between vector (alpha) and BM25 (1-alpha)")
    RRF_K: int = Field(default=60, gt=0, description="Reciprocal Rank Fusion smoothing parameter")
    RERANKER_PROVIDER: str = Field(default="cross-encoder", description="Reranker provider: cross-encoder, cohere, none")
    RERANKER_MODEL: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder model name")
    RERANK_TOP_N: int = Field(default=5, gt=0, description="Number of candidates to pass to LLM after reranking")
    CANDIDATE_POOL_SIZE: int = Field(default=20, gt=0, description="Number of candidates retrieved before reranking")


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
