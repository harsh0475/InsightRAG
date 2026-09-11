"""Unit tests for Reciprocal Rank Fusion (RRF)."""
from backend.app.schemas.document import ChunkMetadata
from backend.app.schemas.retrieval import VectorSearchResult
from backend.app.services.retrieval.fusion import reciprocal_rank_fusion


def make_result(chunk_id: str, score: float = 0.9) -> VectorSearchResult:
    meta = ChunkMetadata(
        document_id="doc1",
        document_name="doc1.txt",
        source="doc1.txt",
        page_number=1,
        section="Overview",
        chunk_id=chunk_id,
        chunk_position=0,
        token_count=10,
        char_count=50,
    )
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=f"Content for {chunk_id}",
        document_id="doc1",
        document_name="doc1.txt",
        metadata=meta,
        score=score,
    )


def test_rrf_joint_candidate_boost():
    """Verify chunk appearing at rank 1 in both retrievers scores higher than single-retriever candidates."""
    # c1 is rank 1 in both
    dense_list = [make_result("c1", 0.95), make_result("c2", 0.85)]
    sparse_list = [make_result("c1", 12.5), make_result("c3", 10.0)]

    k = 60
    fused = reciprocal_rank_fusion(dense_list, sparse_list, k=k, top_n=3)

    assert len(fused) == 3
    # c1 gets 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.032787
    assert fused[0].chunk_id == "c1"
    expected_c1_score = (1.0 / (k + 1)) + (1.0 / (k + 1))
    assert pytest.approx(fused[0].score, rel=1e-3) == expected_c1_score

    # Check rank tracking
    assert fused[0].vector_rank == 1
    assert fused[0].bm25_rank == 1
    assert fused[0].vector_score == 0.95
    assert fused[0].bm25_score == 12.5


def test_rrf_single_list_candidate():
    """Verify items appearing in only one list receive proper score."""
    dense_list = [make_result("c_dense_only", 0.9)]
    sparse_list = [make_result("c_sparse_only", 8.0)]

    k = 60
    fused = reciprocal_rank_fusion(dense_list, sparse_list, k=k, top_n=2)
    assert len(fused) == 2

    # Both are at rank 1 in their respective lists, so both have score 1/(60+1)
    expected_score = 1.0 / (k + 1)
    assert pytest.approx(fused[0].score, rel=1e-3) == expected_score
    assert pytest.approx(fused[1].score, rel=1e-3) == expected_score


import pytest

