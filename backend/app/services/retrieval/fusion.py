"""Reciprocal Rank Fusion (RRF) algorithm for combining disparate retrieval rankings."""
import logging
from typing import Dict, List, Optional

from backend.app.schemas.retrieval import HybridSearchResult, VectorSearchResult

logger = logging.getLogger("insightrag.retrieval.fusion")


def reciprocal_rank_fusion(
    dense_results: List[VectorSearchResult],
    sparse_results: List[VectorSearchResult],
    k: int = 60,
    top_n: int = 5,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> List[HybridSearchResult]:
    """Fuse dense vector rankings and sparse BM25 rankings using Reciprocal Rank Fusion.
    
    Formula:
        RRF_Score(d) = w_dense / (k + r_dense(d)) + w_sparse / (k + r_sparse(d))
    
    Args:
        dense_results: Ranked list from dense vector similarity retriever.
        sparse_results: Ranked list from sparse BM25 retriever.
        k: Smoothing parameter (default 60 from Cormack et al.).
        top_n: Number of fused results to return.
        dense_weight: Relative weight for dense retriever (default 1.0).
        sparse_weight: Relative weight for sparse BM25 retriever (default 1.0).
        
    Returns:
        List of HybridSearchResult objects sorted by descending RRF score,
        with detailed score and rank provenance for debugging.
    """
    scores: Dict[str, float] = {}
    chunk_meta_map: Dict[str, VectorSearchResult] = {}
    dense_ranks: Dict[str, int] = {}
    dense_scores: Dict[str, float] = {}
    sparse_ranks: Dict[str, int] = {}
    sparse_scores: Dict[str, float] = {}

    # 1. Process dense vector results (1-based rank)
    for rank_idx, result in enumerate(dense_results, start=1):
        cid = result.chunk_id
        chunk_meta_map[cid] = result
        dense_ranks[cid] = rank_idx
        dense_scores[cid] = result.score
        rrf_contribution = dense_weight / (k + rank_idx)
        scores[cid] = scores.get(cid, 0.0) + rrf_contribution

    # 2. Process sparse BM25 results (1-based rank)
    for rank_idx, result in enumerate(sparse_results, start=1):
        cid = result.chunk_id
        if cid not in chunk_meta_map:
            chunk_meta_map[cid] = result
        sparse_ranks[cid] = rank_idx
        sparse_scores[cid] = result.score
        rrf_contribution = sparse_weight / (k + rank_idx)
        scores[cid] = scores.get(cid, 0.0) + rrf_contribution

    # 3. Sort unique candidates by descending RRF score
    sorted_cids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    fused_results: List[HybridSearchResult] = []
    for cid, rrf_score in sorted_cids:
        source_chunk = chunk_meta_map[cid]
        fused_results.append(
            HybridSearchResult(
                chunk_id=source_chunk.chunk_id,
                text=source_chunk.text,
                document_id=source_chunk.document_id,
                document_name=source_chunk.document_name,
                metadata=source_chunk.metadata,
                score=round(rrf_score, 6),
                vector_score=dense_scores.get(cid),
                bm25_score=sparse_scores.get(cid),
                vector_rank=dense_ranks.get(cid),
                bm25_rank=sparse_ranks.get(cid),
                retrieval_method="hybrid",
            )
        )

    logger.debug(f"Fused {len(dense_results)} dense and {len(sparse_results)} sparse candidates -> {len(fused_results)} top results")
    return fused_results

