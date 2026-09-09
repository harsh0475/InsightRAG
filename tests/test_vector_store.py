"""Unit tests for vector storage and similarity search."""
import pytest
from backend.app.schemas.document import ChunkMetadata, DocumentChunk
from backend.app.services.embeddings.mock_provider import MockEmbeddingProvider
from backend.app.services.vector_store.in_memory_store import InMemoryVectorStore


def create_test_chunk(chunk_id: str, doc_id: str, text: str, pos: int = 0) -> DocumentChunk:
    """Helper to construct DocumentChunk."""
    meta = ChunkMetadata(
        document_id=doc_id,
        document_name=f"{doc_id}.txt",
        source=f"knowledge_base/{doc_id}.txt",
        page_number=1,
        section="Overview",
        chunk_id=chunk_id,
        chunk_position=pos,
        token_count=len(text.split()),
        char_count=len(text),
    )
    return DocumentChunk(chunk_id=chunk_id, text=text, metadata=meta)


def test_vector_store_indexing_and_search():
    """Verify chunks can be indexed and searched by cosine similarity."""
    store = InMemoryVectorStore()
    emb_provider = MockEmbeddingProvider(dimension=256)

    chunks = [
        create_test_chunk("c1", "doc1", "Kubernetes control plane manages pods and nodes."),
        create_test_chunk("c2", "doc1", "Redis replication provides high-availability distributed caching."),
        create_test_chunk("c3", "doc2", "OAuth 2.0 refresh token rotation protects against token compromise."),
    ]
    texts = [c.text for c in chunks]
    embeddings = emb_provider.embed_texts(texts)

    count = store.index_chunks(chunks, embeddings)
    assert count == 3
    assert store.count_chunks() == 3

    # Search for Redis caching
    q_vec = emb_provider.embed_query("Redis caching and replication failover")
    results = store.search(q_vec, top_k=2)

    assert len(results) == 2
    # The top result should be chunk c2 (Redis)
    assert results[0].chunk_id == "c2"
    assert "Redis replication" in results[0].text
    assert 0.0 <= results[0].score <= 1.0


def test_vector_store_document_filtering():
    """Verify search respects document_ids filter."""
    store = InMemoryVectorStore()
    emb_provider = MockEmbeddingProvider(dimension=128)

    chunks = [
        create_test_chunk("c1", "doc_k8s", "Kubernetes cluster security and RBAC."),
        create_test_chunk("c2", "doc_oauth", "OAuth security and access tokens."),
    ]
    store.index_chunks(chunks, emb_provider.embed_texts([c.text for c in chunks]))

    q_vec = emb_provider.embed_query("security")
    # Filter to only doc_oauth
    results = store.search(q_vec, top_k=5, document_ids=["doc_oauth"])

    assert len(results) == 1
    assert results[0].document_id == "doc_oauth"


def test_vector_store_delete_document():
    """Verify document deletion removes all associated chunks."""
    store = InMemoryVectorStore()
    emb_provider = MockEmbeddingProvider(dimension=64)

    chunks = [
        create_test_chunk("c1", "doc_del", "First chunk of document to delete", 0),
        create_test_chunk("c2", "doc_del", "Second chunk of document to delete", 1),
        create_test_chunk("c3", "doc_keep", "Chunk to retain", 0),
    ]
    store.index_chunks(chunks, emb_provider.embed_texts([c.text for c in chunks]))

    assert store.count_chunks() == 3
    deleted = store.delete_document("doc_del")
    assert deleted == 2
    assert store.count_chunks() == 1
    assert store.get_chunk("c3") is not None
    assert store.get_chunk("c1") is None

