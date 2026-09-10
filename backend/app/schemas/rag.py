"""Schemas for Baseline RAG generation, citations, and query requests."""
from typing import List, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.retrieval import VectorSearchResult


class Citation(BaseModel):
    """Source provenance citation verifying a claim in the generated answer."""
    chunk_id: str = Field(description="Unique identifier of the cited chunk")
    document_id: str = Field(description="Parent document identifier")
    document_name: str = Field(description="Filename of the source document")
    page_number: Optional[int] = Field(default=None, description="Page number if applicable (e.g. for PDFs)")
    section: Optional[str] = Field(default=None, description="Section heading if applicable")
    snippet: str = Field(description="Text snippet supporting the cited assertion")


class RAGQueryRequest(BaseModel):
    """Payload for RAG question-answering query."""
    query: str = Field(min_length=1, description="The user question to answer")
    top_k: int = Field(default=5, gt=0, le=50, description="Number of context chunks to retrieve")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional document IDs filter")


class RAGResponse(BaseModel):
    """Complete RAG response containing the grounded answer, citations, and retrieved evidence."""
    query: str = Field(description="The user's original query")
    answer: str = Field(description="The generated answer, grounded strictly in retrieved context")
    citations: List[Citation] = Field(default_factory=list, description="List of source citations extracted from answer")
    retrieved_chunks: List[VectorSearchResult] = Field(description="Raw candidate chunks retrieved for context")
    has_sufficient_context: bool = Field(description="False if context lacked sufficient evidence to answer")
    execution_time_ms: float = Field(description="End-to-end latency in milliseconds")
    model_name: str = Field(description="Name of the generative LLM model used")

