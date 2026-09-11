"""Hybrid retrieval coordinator integrating dense vector search and sparse BM25 with RRF."""
import logging
from typing import List, Optional

from backend.app.core.config import get_settings
from backend.app.schemas.document import DocumentChunk
from backend.app.schemas.retrieval import HybridSearchResult, RetrievalMode, VectorSearchResult
from backend.app.services.retrieval.bm25 import BM25Retriever, get_bm25_retriever
from backend.app.services.retrieval.fusion import reciprocal_rank_fusion
from backend.app.services.retrieval_service import RetrievalService

logger = logging.getLogger("insightrag.retrieval.hybrid")


class HybridRetriever:
    """Combines dense semantic vector search and sparse lexical BM25 matching."""

    def __init__(
        self,
        dense_service: Optional[RetrievalService] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
    ):
        settings = get_settings()
        self.dense_service = dense_service or RetrievalService()
        self.bm25_retriever = bm25_retriever or get_bm25_retriever()
        self.default_mode = RetrievalMode(settings.RETRIEVAL_MODE)
        self.candidate_pool_size = settings.CANDIDATE_POOL_SIZE
        self.rrf_k = settings.RRF_K
        self.alpha = settings.HYBRID_RETRIEVAL_ALPHA

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Index chunks in both the dense vector store and the BM25 inverted index."""
        if not chunks:
            return 0

        # 1. Index in dense vector store
        dense_count = self.dense_service.index_chunks(chunks)

        # 2. Index in BM25 inverted index
        bm25_count = self.bm25_retriever.index_chunks(chunks)

        logger.info(f"Hybrid index updated: {dense_count} dense vectors, {bm25_count} BM25 documents.")
        return dense_count

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: Optional[RetrievalMode] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[HybridSearchResult]:
        """Execute retrieval under the specified mode: vector, bm25, or hybrid.
        
        Args:
            query: Natural language query string.
            top_k: Number of final results to return.
            mode: RetrievalMode (defaults to config setting: hybrid).
            document_ids: Optional document filters.
            
        Returns:
            List of HybridSearchResult objects sorted by relevance score.
        """
        active_mode = mode or self.default_mode
        logger.info(f"Executing retrieval in [{active_mode.value.upper()}] mode for query: '{query}'")

        # --- Mode 1: Dense Vector Only ---
        if active_mode == RetrievalMode.VECTOR:
            dense_results = self.dense_service.retrieve(
                query=query,
                top_k=top_k,
                document_ids=document_ids,
            )
            return [
                HybridSearchResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    document_id=r.document_id,
                    document_name=r.document_name,
                    metadata=r.metadata,
                    score=r.score,
                    vector_score=r.score,
                    vector_rank=idx + 1,
                    retrieval_method="vector",
                )
                for idx, r in enumerate(dense_results)
            ]

        # --- Mode 2: Sparse BM25 Only ---
        if active_mode == RetrievalMode.BM25:
            bm25_results = self.bm25_retriever.retrieve(
                query=query,
                top_k=top_k,
                document_ids=document_ids,
            )
            return [
                HybridSearchResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    document_id=r.document_id,
                    document_name=r.document_name,
                    metadata=r.metadata,
                    score=r.score,
                    bm25_score=r.score,
                    bm25_rank=idx + 1,
                    retrieval_method="bm25",
                )
                for idx, r in enumerate(bm25_results)
            ]

        # --- Mode 3: Hybrid (Dense + BM25 via Reciprocal Rank Fusion) ---
        pool_size = max(top_k * 2, self.candidate_pool_size)

        dense_candidates = self.dense_service.retrieve(
            query=query,
            top_k=pool_size,
            document_ids=document_ids,
        )

        bm25_candidates = self.bm25_retriever.retrieve(
            query=query,
            top_k=pool_size,
            document_ids=document_ids,
        )

        fused = reciprocal_rank_fusion(
            dense_results=dense_candidates,
            sparse_results=bm25_candidates,
            k=self.rrf_k,
            top_n=top_k,
            dense_weight=self.alpha,
            sparse_weight=(1.0 - self.alpha),
        )

        return fused

