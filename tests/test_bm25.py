"""Unit tests for Okapi BM25 sparse keyword retriever."""
import pytest
from backend.app.schemas.document import ChunkMetadata, DocumentChunk
from backend.app.services.retrieval.bm25 import BM25Retriever


def make_chunk(chunk_id: str, text: str, doc_name: str = "doc.txt") -> DocumentChunk:
    meta = ChunkMetadata(
        document_id=doc_name,
        document_name=doc_name,
        source=f"knowledge_base/{doc_name}",
        page_number=1,
        section="Section",
        chunk_id=chunk_id,
        chunk_position=0,
        token_count=len(text.split()),
        char_count=len(text),
    )
    return DocumentChunk(chunk_id=chunk_id, text=text, metadata=meta)


def test_bm25_tokenization():
    """Verify tokenizer extracts words, hyphens, and dotted paths."""
    retriever = BM25Retriever()
    tokens = retriever.tokenize("The kube-apiserver runs spec.clusterIP on port 6443.")
    assert "kube-apiserver" in tokens
    assert "spec.clusterip" in tokens
    assert "6443" in tokens
    assert "port" in tokens


def test_bm25_indexing_and_scoring():
    """Verify exact keyword matching ranks relevant chunk first."""
    retriever = BM25Retriever()
    chunks = [
        make_chunk("c1", "OAuth 2.0 PKCE code_verifier requires 43 characters entropy."),
        make_chunk("c2", "Kubernetes pods communicate via ClusterIP service."),
        make_chunk("c3", "Distributed Redis cache stampedes use jittered TTL."),
    ]
    retriever.index_chunks(chunks)
    assert retriever.count_chunks() == 3

    # Query targeting exact terms in c1
    results = retriever.retrieve("code_verifier PKCE characters", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].score > 0.0


def test_bm25_document_length_normalization():
    """Verify document length normalization parameter b penalizes bloated documents."""
    retriever = BM25Retriever(k1=1.5, b=0.75)
    # Short document containing target term 'etcd'
    short_chunk = make_chunk("c_short", "etcd is a consistent key value store.")
    # Long document containing target term 'etcd' once, but surrounded by 50 filler words
    filler = " " + " ".join([f"filler{i}" for i in range(50)])
    long_chunk = make_chunk("c_long", "etcd is a consistent key value store." + filler)

    retriever.index_chunks([short_chunk, long_chunk])

    # Search for 'etcd'
    results = retriever.retrieve("etcd", top_k=2)
    assert len(results) == 2
    # Short document should score higher due to higher term density
    assert results[0].chunk_id == "c_short"
    assert results[0].score > results[1].score


def test_bm25_no_matches():
    """Verify empty result when query terms do not appear in index."""
    retriever = BM25Retriever()
    retriever.index_chunks([make_chunk("c1", "Kubernetes cluster architecture.")])

    results = retriever.retrieve("nonexistentterm12345")
    assert results == []

