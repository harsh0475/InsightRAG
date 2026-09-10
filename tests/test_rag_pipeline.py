"""Unit and integration tests for BaselineRAGPipeline."""
import pytest
from backend.app.schemas.document import ChunkMetadata, DocumentChunk
from backend.app.services.embeddings.mock_provider import MockEmbeddingProvider
from backend.app.services.llm.mock_provider import MockLLMProvider
from backend.app.services.rag.context_builder import ContextBuilder
from backend.app.services.rag.pipeline import BaselineRAGPipeline
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.vector_store.in_memory_store import InMemoryVectorStore


def setup_test_pipeline():
    """Build an isolated in-memory RAG pipeline."""
    emb = MockEmbeddingProvider(dimension=128)
    store = InMemoryVectorStore()
    retrieval = RetrievalService(embedding_provider=emb, vector_store=store)
    llm = MockLLMProvider()
    builder = ContextBuilder(max_context_tokens=1000)

    # Index sample chunk
    meta = ChunkMetadata(
        document_id="doc_oauth",
        document_name="oauth.txt",
        source="knowledge_base/oauth.txt",
        page_number=1,
        section="Token Expiration",
        chunk_id="doc_oauth_c0001",
        chunk_position=0,
        token_count=25,
        char_count=120,
    )
    chunk = DocumentChunk(
        chunk_id="doc_oauth_c0001",
        text="OAuth 2.0 access tokens have a 15-minute expiration window and require refresh token rotation.",
        metadata=meta,
    )
    retrieval.index_chunks([chunk])

    pipeline = BaselineRAGPipeline(
        retrieval_service=retrieval,
        llm_provider=llm,
        context_builder=builder,
    )
    return pipeline


def test_baseline_rag_successful_answer_with_citations():
    """Verify grounded answer generation and citation resolution."""
    pipeline = setup_test_pipeline()
    response = pipeline.run(query="What is the expiration window for OAuth access tokens?", top_k=2)

    assert response.has_sufficient_context is True
    assert "OAuth 2.0" in response.answer or "15-minute" in response.answer
    assert response.execution_time_ms > 0

    # Verify citation was parsed and resolved
    assert len(response.citations) >= 1
    citation = response.citations[0]
    assert citation.chunk_id == "doc_oauth_c0001"
    assert citation.document_name == "oauth.txt"
    assert citation.page_number == 1
    assert citation.section == "Token Expiration"


def test_baseline_rag_refusal_on_insufficient_context():
    """Verify pipeline acknowledges insufficient evidence when context is empty."""
    emb = MockEmbeddingProvider(dimension=128)
    empty_store = InMemoryVectorStore()  # empty store
    retrieval = RetrievalService(embedding_provider=emb, vector_store=empty_store)
    pipeline = BaselineRAGPipeline(retrieval_service=retrieval, llm_provider=MockLLMProvider())

    response = pipeline.run(query="What is the recipe for baking chocolate brownies?")
    assert response.has_sufficient_context is False
    assert "I do not have sufficient information" in response.answer
    assert len(response.citations) == 0

