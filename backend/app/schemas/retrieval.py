"""Schemas for embedding generation, vector storage, and semantic retrieval."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.document import ChunkMetadata, DocumentChunk


class VectorSearchResult(BaseModel):
    """Result item returned by vector similarity retrieval."""
    chunk_id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Content of the retrieved text chunk")
    document_id: str = Field(description="Parent document identifier")
    document_name: str = Field(description="Name of the source document")
    metadata: ChunkMetadata = Field(description="Provenance metadata (page, section, position)")
    score: float = Field(description="Cosine similarity score between 0.0 and 1.0 (higher = more similar)")


class VectorSearchRequest(BaseModel):
    """Payload for semantic vector search."""
    query: str = Field(min_length=1, description="Natural language search query")
    top_k: int = Field(default=5, gt=0, le=100, description="Number of top results to retrieve")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional document ID filters")


class IndexChunksRequest(BaseModel):
    """Payload to index document chunks into the vector database."""
    chunks: List[DocumentChunk] = Field(description="List of document chunks to embed and store")


class IndexChunksResponse(BaseModel):
    """Response confirming chunk indexing."""
    indexed_count: int = Field(description="Number of chunks successfully embedded and indexed")
    document_id: str = Field(description="Parent document identifier")
    model: str = Field(description="Embedding model used")
    dimension: int = Field(description="Embedding vector dimensionality")

