"""Schemas for reranking candidates in two-stage retrieval."""
from typing import List, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.document import ChunkMetadata
from backend.app.schemas.retrieval import VectorSearchResult


class RerankedResult(BaseModel):
    """Candidate chunk rescaled and reordered by cross-encoder reranker."""
    chunk_id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Chunk content text")
    document_id: str = Field(description="Parent document identifier")
    document_name: str = Field(description="Source document name")
    metadata: ChunkMetadata = Field(description="Provenance metadata (page, section, position)")
    retrieval_score: float = Field(description="Initial retrieval score from bi-encoder or hybrid search")
    rerank_score: float = Field(description="Relevance score assigned by cross-encoder model")
    initial_rank: int = Field(ge=1, description="1-based candidate rank prior to reranking")
    final_rank: int = Field(ge=1, description="1-based candidate rank after reranking")
    rank_delta: int = Field(
        description="Position shift: positive indicates promotion, negative indicates demotion"
    )


class RerankRequest(BaseModel):
    """Payload for standalone reranking service."""
    query: str = Field(min_length=1, description="Search query against which chunks are reranked")
    candidates: List[VectorSearchResult] = Field(description="Candidate chunks retrieved by Stage 1")
    top_n: int = Field(default=5, gt=0, description="Number of top candidates to retain after reranking")


class RerankResponse(BaseModel):
    """Response returned by reranking service."""
    query: str = Field(description="Search query")
    results: List[RerankedResult] = Field(description="Reranked and truncated chunk list")
    total_candidates: int = Field(description="Number of candidate chunks evaluated")
    model_name: str = Field(description="Name of reranker model used")
    execution_time_ms: float = Field(description="Latency of reranking stage in milliseconds")

