"""First-principles Okapi BM25 sparse keyword retrieval engine."""
import logging
import math
import re
from typing import Dict, List, Optional, Set, Tuple

from backend.app.core.config import get_settings
from backend.app.schemas.document import DocumentChunk
from backend.app.schemas.retrieval import VectorSearchResult

logger = logging.getLogger("insightrag.retrieval.bm25")


class BM25Retriever:
    """Okapi BM25 sparse lexical retriever with inverted indexing and length normalization."""

    # Matches words, hyphenated identifiers, and dotted paths (e.g. 'kube-apiserver', 'spec.clusterIP')
    _TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_\-\.]+\b")

    def __init__(self, k1: Optional[float] = None, b: Optional[float] = None):
        settings = get_settings()
        self.k1 = k1 if k1 is not None else settings.BM25_K1
        self.b = b if b is not None else settings.BM25_B

        # Storage & Inverted Index
        self._chunks: Dict[str, DocumentChunk] = {}
        self._doc_lengths: Dict[str, int] = {}
        # Inverted index: term -> {chunk_id: term_frequency}
        self._inverted_index: Dict[str, Dict[str, int]] = {}
        # Document frequencies: term -> count of chunks containing term
        self._doc_frequencies: Dict[str, int] = {}
        self._total_docs: int = 0
        self._avg_doc_length: float = 0.0

    def tokenize(self, text: str) -> List[str]:
        """Normalize text into lowercase alphanumeric tokens."""
        if not text:
            return []
        return [t.lower() for t in self._TOKEN_PATTERN.findall(text) if len(t) > 1]

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Add or update document chunks in the BM25 inverted index."""
        if not chunks:
            return 0

        for chunk in chunks:
            tokens = self.tokenize(chunk.text)
            doc_len = len(tokens)
            cid = chunk.chunk_id

            # Remove old term frequencies if chunk was previously indexed
            if cid in self._chunks:
                self._remove_chunk_from_index(cid)

            self._chunks[cid] = chunk
            self._doc_lengths[cid] = doc_len

            # Count term frequencies in this chunk
            tf_map: Dict[str, int] = {}
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            for term, freq in tf_map.items():
                if term not in self._inverted_index:
                    self._inverted_index[term] = {}
                    self._doc_frequencies[term] = 0
                self._inverted_index[term][cid] = freq
                self._doc_frequencies[term] += 1

        self._total_docs = len(self._chunks)
        total_tokens = sum(self._doc_lengths.values())
        self._avg_doc_length = total_tokens / self._total_docs if self._total_docs > 0 else 0.0

        logger.info(
            f"Indexed {len(chunks)} chunks in BM25 index. "
            f"Total docs: {self._total_docs}, Avg length: {self._avg_doc_length:.1f} terms."
        )
        return len(chunks)

    def _remove_chunk_from_index(self, chunk_id: str) -> None:
        """Remove a chunk from inverted index and update doc frequencies."""
        for term, postings in list(self._inverted_index.items()):
            if chunk_id in postings:
                del postings[chunk_id]
                self._doc_frequencies[term] -= 1
                if self._doc_frequencies[term] <= 0:
                    del self._inverted_index[term]
                    del self._doc_frequencies[term]
        if chunk_id in self._chunks:
            del self._chunks[chunk_id]
        if chunk_id in self._doc_lengths:
            del self._doc_lengths[chunk_id]

    def _compute_idf(self, term: str) -> float:
        """Compute Robertson-Spärck Jones Inverse Document Frequency with smoothing."""
        df = self._doc_frequencies.get(term, 0)
        if df == 0:
            return 0.0
        # Formula: ln(1 + (N - df + 0.5) / (df + 0.5))
        numerator = self._total_docs - df + 0.5
        denominator = df + 0.5
        return math.log(1.0 + (numerator / denominator))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        """Execute Okapi BM25 scoring over inverted index and return top-K candidates.
        
        Args:
            query: User search query.
            top_k: Number of highest-scoring chunks to return.
            document_ids: Optional list of document IDs to filter by.
            
        Returns:
            List of VectorSearchResult objects ranked in descending order of BM25 score.
        """
        if not query or not query.strip() or self._total_docs == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        filter_set: Optional[Set[str]] = set(document_ids) if document_ids else None
        scores: Dict[str, float] = {}

        for token in query_tokens:
            postings = self._inverted_index.get(token)
            if not postings:
                continue

            idf = self._compute_idf(token)
            if idf <= 0.0:
                continue

            for cid, tf in postings.items():
                chunk = self._chunks[cid]
                if filter_set and chunk.metadata.document_id not in filter_set:
                    continue

                doc_len = self._doc_lengths[cid]
                # Length normalization term
                len_norm = 1.0 - self.b + (self.b * (doc_len / self._avg_doc_length))
                # TF saturation term
                tf_weight = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)

                scores[cid] = scores.get(cid, 0.0) + (idf * tf_weight)

        if not scores:
            return []

        # Sort candidate chunks in descending order of BM25 score
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[VectorSearchResult] = []
        for cid, score in sorted_candidates:
            chunk = self._chunks[cid]
            results.append(
                VectorSearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    document_id=chunk.metadata.document_id,
                    document_name=chunk.metadata.document_name,
                    metadata=chunk.metadata,
                    score=round(score, 4),
                )
            )

        return results

    def count_chunks(self) -> int:
        """Return total indexed chunks."""
        return self._total_docs

    def clear(self) -> None:
        """Reset index."""
        self._chunks.clear()
        self._doc_lengths.clear()
        self._inverted_index.clear()
        self._doc_frequencies.clear()
        self._total_docs = 0
        self._avg_doc_length = 0.0


# Shared singleton instance for the session
_SHARED_BM25_RETRIEVER: Optional[BM25Retriever] = None


def get_bm25_retriever(force_new: bool = False) -> BM25Retriever:
    """Return shared BM25Retriever instance."""
    global _SHARED_BM25_RETRIEVER
    if _SHARED_BM25_RETRIEVER is None or force_new:
        _SHARED_BM25_RETRIEVER = BM25Retriever()
    return _SHARED_BM25_RETRIEVER

