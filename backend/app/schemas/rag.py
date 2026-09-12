"""Schemas for RAG generation, citations, query rewriting, and reranking."""
from typing import List, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.chat import ChatMessage
from backend.app.schemas.reranker import RerankedResult
from backend.app.schemas.retrieval import RetrievalMode, VectorSearchResult


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
    top_k: int = Field(default=5, gt=0, le=50, description="Number of context chunks to use for generation")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional document IDs filter")
    retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.HYBRID, description="Retrieval strategy: vector, bm25, or hybrid"
    )
    enable_reranking: bool = Field(
        default=True, description="Whether to apply cross-encoder reranking on retrieved candidate pool"
    )
    candidate_pool_size: int = Field(
        default=20, gt=0, le=100, description="Number of candidate chunks retrieved prior to reranking"
    )
    chat_history: Optional[List[ChatMessage]] = Field(
        default=None, description="Optional previous messages in conversation for conversational query rewriting"
    )


class RAGResponse(BaseModel):
    """Complete RAG response containing the grounded answer, citations, and retrieved evidence."""
    query: str = Field(description="The user's original query")
    rewritten_query: Optional[str] = Field(
        default=None, description="Standalone rewritten query formulated by conversational rewriter"
    )
    answer: str = Field(description="The generated answer, grounded strictly in retrieved context")
    citations: List[Citation] = Field(default_factory=list, description="List of source citations extracted from answer")
    retrieved_chunks: List[VectorSearchResult] = Field(description="Candidate chunks used for generation context")
    reranked_chunks: Optional[List[RerankedResult]] = Field(
        default=None, description="Detailed reranking results with score shifts and rank deltas"
    )
    has_sufficient_context: bool = Field(description="False if context lacked sufficient evidence to answer")
    execution_time_ms: float = Field(description="End-to-end latency in milliseconds")
    model_name: str = Field(description="Name of the generative LLM model used")
    retrieval_mode: str = Field(default="hybrid", description="Retrieval strategy executed")
    reranker_applied: bool = Field(default=False, description="Whether reranker was applied to candidate pool")
    reranker_latency_ms: Optional[float] = Field(default=None, description="Latency spent in reranking stage in ms")

