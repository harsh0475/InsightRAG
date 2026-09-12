"""Abstract base class for two-stage retrieval reranker providers."""
from abc import ABC, abstractmethod
from typing import List

from backend.app.schemas.reranker import RerankedResult
from backend.app.schemas.retrieval import VectorSearchResult


class BaseRerankerProvider(ABC):
    """Abstract interface defining the contract for reranker models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or HuggingFace ID of the reranker model."""
        pass

    @abstractmethod
    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """Compute cross-attention relevance scores for (query, text) pairs.
        
        Args:
            query: The user query string.
            texts: List of candidate chunk texts.
            
        Returns:
            List of float relevance scores where higher means more relevant.
        """
        pass

    def rerank(
        self,
        query: str,
        candidates: List[VectorSearchResult],
        top_n: int = 5,
    ) -> List[RerankedResult]:
        """Rerank candidate chunks using cross-attention relevance scoring.
        
        Args:
            query: The natural language search query.
            candidates: List of initial candidate chunks retrieved by Stage 1.
            top_n: Number of highest-scoring candidates to return.
            
        Returns:
            List of RerankedResult items sorted by rerank_score descending.
        """
        if not candidates:
            return []

        # Extract texts for scoring
        texts = [c.text for c in candidates]
        scores = self.score_pairs(query, texts)

        # Pair each candidate with its initial 1-based rank and its reranker score
        scored_candidates = []
        for idx, (candidate, score) in enumerate(zip(candidates, scores)):
            scored_candidates.append({
                "candidate": candidate,
                "initial_rank": idx + 1,
                "rerank_score": float(score),
            })

        # Sort descending by reranker score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Truncate to top_n and compute rank deltas
        results: List[RerankedResult] = []
        for final_idx, item in enumerate(scored_candidates[:top_n]):
            cand: VectorSearchResult = item["candidate"]
            final_rank = final_idx + 1
            initial_rank = item["initial_rank"]
            rank_delta = initial_rank - final_rank  # Positive = promoted, negative = demoted

            results.append(
                RerankedResult(
                    chunk_id=cand.chunk_id,
                    text=cand.text,
                    document_id=cand.document_id,
                    document_name=cand.document_name,
                    metadata=cand.metadata,
                    retrieval_score=cand.score,
                    rerank_score=round(item["rerank_score"], 4),
                    initial_rank=initial_rank,
                    final_rank=final_rank,
                    rank_delta=rank_delta,
                )
            )

        return results

