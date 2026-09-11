"""Unit and integration tests for HybridRetriever."""
import pytest
from backend.app.schemas.document import ChunkMetadata, DocumentChunk
from backend.app.schemas.retrieval import RetrievalMode
from backend.app.services.embeddings.mock_provider import MockEmbeddingProvider
from backend.app.services.retrieval.bm25 import BM25Retriever
from backend.app.services.retrieval.hybrid import HybridRetriever
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.vector_store.in_memory_store import InMemoryVectorStore


def make_chunk(chunk_id: str, doc_id: str, text: str) -> DocumentChunk:
    meta = ChunkMetadata(
        document_id=doc_id,
        document_name=f"{doc_id}.txt",
        source=f"knowledge_base/{doc_id}.txt",
        page_number=1,
        section="Overview",
        chunk_id=chunk_id,
        chunk_position=0,
        token_count=len(text.split()),
        char_count=len(text),
    )
    return DocumentChunk(chunk_id=chunk_id, text=text, metadata=meta)


def setup_hybrid_retriever() -> HybridRetriever:
    emb = MockEmbeddingProvider(dimension=128)
    store = InMemoryVectorStore()
    dense_service = RetrievalService(embedding_provider=emb, vector_store=store)
    bm25 = BM25Retriever()
    retriever = HybridRetriever(dense_service=dense_service, bm25_retriever=bm25)

    chunks = [
        make_chunk("c1", "doc_oauth", "OAuth 2.0 PKCE requires code_challenge and 43 character code_verifier."),
        make_chunk("c2", "doc_k8s", "The Kubernetes control plane uses etcd as its key-value backing store."),
        make_chunk("c3", "doc_redis", "Redis caching prevents stampedes using jittered TTL expiration."),
    ]
    retriever.index_chunks(chunks)
    return retriever


def test_hybrid_retrieval_modes():
    """Verify retriever supports VECTOR, BM25, and HYBRID modes."""
    retriever = setup_hybrid_retriever()

    # 1. Vector only
    res_vec = retriever.retrieve("code_verifier PKCE", top_k=2, mode=RetrievalMode.VECTOR)
    assert len(res_vec) > 0
    assert res_vec[0].retrieval_method == "vector"

    # 2. BM25 only
    res_bm25 = retriever.retrieve("code_verifier PKCE", top_k=2, mode=RetrievalMode.BM25)
    assert len(res_bm25) > 0
    assert res_bm25[0].retrieval_method == "bm25"
    assert res_bm25[0].chunk_id == "c1"

    # 3. Hybrid (fused)
    res_hybrid = retriever.retrieve("code_verifier PKCE", top_k=2, mode=RetrievalMode.HYBRID)
    assert len(res_hybrid) > 0
    assert res_hybrid[0].retrieval_method == "hybrid"
    assert res_hybrid[0].chunk_id == "c1"
    assert res_hybrid[0].score > 0.0


def test_hybrid_retrieval_document_filter():
    """Verify document filtering works in hybrid mode."""
    retriever = setup_hybrid_retriever()
    results = retriever.retrieve(
        "caching",
        top_k=5,
        mode=RetrievalMode.HYBRID,
        document_ids=["doc_redis"],
    )
    assert len(results) == 1
    assert results[0].document_id == "doc_redis"

