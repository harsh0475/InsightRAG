"""Schemas for embedding generation, vector storage, BM25, and hybrid retrieval."""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.document import ChunkMetadata, DocumentChunk


class RetrievalMode(str, Enum):
    """Retrieval execution strategies."""
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


class VectorSearchResult(BaseModel):
    """Result item returned by vector similarity retrieval."""
    chunk_id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Content of the retrieved text chunk")
    document_id: str = Field(description="Parent document identifier")
    document_name: str = Field(description="Name of the source document")
    metadata: ChunkMetadata = Field(description="Provenance metadata (page, section, position)")
    score: float = Field(description="Similarity or relevance score")


class HybridSearchResult(BaseModel):
    """Result item returned by hybrid retrieval with fusion breakdown."""
    chunk_id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Content of the retrieved text chunk")
    document_id: str = Field(description="Parent document identifier")
    document_name: str = Field(description="Name of the source document")
    metadata: ChunkMetadata = Field(description="Provenance metadata")
    score: float = Field(description="Final merged relevance score (e.g. RRF score)")
    vector_score: Optional[float] = Field(default=None, description="Original dense vector similarity score")
    bm25_score: Optional[float] = Field(default=None, description="Original sparse BM25 score")
    vector_rank: Optional[int] = Field(default=None, description="1-based rank in dense retriever")
    bm25_rank: Optional[int] = Field(default=None, description="1-based rank in BM25 retriever")
    retrieval_method: str = Field(default="hybrid", description="Method used: vector, bm25, or hybrid")


class VectorSearchRequest(BaseModel):
    """Payload for retrieval query."""
    query: str = Field(min_length=1, description="Natural language search query")
    top_k: int = Field(default=5, gt=0, le=100, description="Number of top results to retrieve")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional document ID filters")
    mode: RetrievalMode = Field(default=RetrievalMode.HYBRID, description="Retrieval mode: vector, bm25, or hybrid")


class IndexChunksRequest(BaseModel):
    """Payload to index document chunks into the vector and lexical stores."""
    chunks: List[DocumentChunk] = Field(description="List of document chunks to embed and store")


class IndexChunksResponse(BaseModel):
    """Response confirming chunk indexing."""
    indexed_count: int = Field(description="Number of chunks successfully embedded and indexed")
    document_id: str = Field(description="Parent document identifier")
    model: str = Field(description="Embedding model used")
    dimension: int = Field(description="Embedding vector dimensionality")
