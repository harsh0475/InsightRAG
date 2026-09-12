"""Tests for Reranker subsystem and rank shift computation."""
import pytest
from backend.app.schemas.document import ChunkMetadata
from backend.app.schemas.retrieval import VectorSearchResult
from backend.app.services.reranker import (
    BaseRerankerProvider,
    CrossEncoderReranker,
    MockRerankerProvider,
    get_reranker_provider,
)


def _make_candidate(chunk_id: str, text: str, score: float = 0.5) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        document_id="doc-1",
        document_name="test.md",
        metadata=ChunkMetadata(
            document_id="doc-1",
            document_name="test.md",
            source="knowledge_base/test.md",
            chunk_id=chunk_id,
            chunk_position=1,
            token_count=20,
            char_count=len(text),
        ),
        score=score,
    )


def test_mock_reranker_empty_candidates():
    """Mock reranker should return empty list on empty input."""
    reranker = MockRerankerProvider()
    results = reranker.rerank(query="test query", candidates=[], top_n=5)
    assert results == []


def test_mock_reranker_scoring_relevance():
    """Mock reranker should score chunks with exact query terms significantly higher."""
    reranker = MockRerankerProvider()
    query = "Redis sentinel failover quorum"

    candidates = [
        _make_candidate("c1", "Kubernetes pods manage container deployments across cluster nodes.", score=0.9),
        _make_candidate("c2", "Redis sentinel failover requires quorum configuration among nodes.", score=0.6),
        _make_candidate("c3", "Distributed caching strategies using memory stores.", score=0.7),
    ]

    # Stage 1 ranks: c1 (rank 1), c2 (rank 2), c3 (rank 3)
    reranked = reranker.rerank(query=query, candidates=candidates, top_n=3)

    assert len(reranked) == 3
    # c2 contains exact keywords 'Redis sentinel failover requires quorum'
    # It must be promoted to rank 1
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].final_rank == 1
    assert reranked[0].initial_rank == 2
    assert reranked[0].rank_delta == 1  # 2 - 1 = +1 promotion


def test_rank_delta_calculation():
    """Ensure rank_delta correctly measures promotions and demotions."""
    reranker = MockRerankerProvider()
    query = "OAuth PKCE authorization code"

    candidates = [
        _make_candidate("c1", "Irrelevant passage about database indexing.", score=0.85),
        _make_candidate("c2", "Another loosely related passage about HTTP APIs.", score=0.80),
        _make_candidate("c3", "OAuth PKCE protects authorization code exchange against interception.", score=0.75),
    ]

    reranked = reranker.rerank(query=query, candidates=candidates, top_n=2)
    assert len(reranked) == 2
    top = reranked[0]
    assert top.chunk_id == "c3"
    assert top.initial_rank == 3
    assert top.final_rank == 1
    assert top.rank_delta == 2  # Promoted 2 spots


def test_cross_encoder_fallback():
    """CrossEncoderReranker should seamlessly fall back to deterministic provider if torch is missing."""
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    candidates = [
        _make_candidate("c1", "PostgreSQL pgvector cosine distance."),
        _make_candidate("c2", "FastAPI dependency injection patterns."),
    ]
    results = reranker.rerank("pgvector cosine", candidates, top_n=2)
    assert len(results) == 2
    assert results[0].chunk_id == "c1"


def test_reranker_factory():
    """get_reranker_provider should return instances adhering to BaseRerankerProvider."""
    provider_mock = get_reranker_provider("mock")
    assert isinstance(provider_mock, BaseRerankerProvider)
    assert isinstance(provider_mock, MockRerankerProvider)

    provider_cross = get_reranker_provider("cross-encoder")
    assert isinstance(provider_cross, BaseRerankerProvider)
    assert isinstance(provider_cross, CrossEncoderReranker)

