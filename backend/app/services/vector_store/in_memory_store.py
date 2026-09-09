"""In-memory vector store implementation using NumPy for exact cosine similarity."""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from backend.app.schemas.document import DocumentChunk
from backend.app.schemas.retrieval import VectorSearchResult
from backend.app.services.vector_store.base import BaseVectorStore

logger = logging.getLogger("insightrag.vector_store.in_memory")


class InMemoryVectorStore(BaseVectorStore):
    """High-performance in-memory vector store backed by NumPy array operations."""

    def __init__(self):
        self._chunks: Dict[str, DocumentChunk] = {}
        # Stores mapping: chunk_id -> np.ndarray (float32)
        self._embeddings: Dict[str, np.ndarray] = {}

    def index_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
        """Store chunks and normalized vector representations."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        stored = 0
        for chunk, emb in zip(chunks, embeddings):
            vec = np.array(emb, dtype=np.float32)
            # Store normalized vector for rapid dot-product cosine similarity
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            self._chunks[chunk.chunk_id] = chunk
            self._embeddings[chunk.chunk_id] = vec
            stored += 1

        logger.info(f"Indexed {stored} chunks in InMemoryVectorStore (Total: {len(self._chunks)})")
        return stored

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        """Compute cosine similarity over all indexed vectors and return top-K."""
        if not self._chunks:
            return []

        # Convert and normalize query vector
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        filter_set = set(document_ids) if document_ids else None

        candidate_ids: List[str] = []
        candidate_vectors: List[np.ndarray] = []

        for cid, chunk in self._chunks.items():
            if filter_set and chunk.metadata.document_id not in filter_set:
                continue
            candidate_ids.append(cid)
            candidate_vectors.append(self._embeddings[cid])

        if not candidate_ids:
            return []

        # Stack into matrix (N, D) and compute matrix-vector dot product (N,)
        matrix = np.vstack(candidate_vectors)
        scores = np.dot(matrix, q_vec)

        # Clip scores to [-1.0, 1.0] to guard against floating-point epsilon
        scores = np.clip(scores, -1.0, 1.0)

        # Map cosine similarity from [-1, 1] to [0, 1] scale: (cos + 1) / 2 or standard cosine
        # In RAG, standard cosine similarity for normalized vectors is simply dot product
        # Ensure values stay in [0.0, 1.0] range
        scores = np.maximum(0.0, scores)

        # Get top-K indices
        top_indices = np.argsort(-scores)[:top_k]

        results: List[VectorSearchResult] = []
        for idx in top_indices:
            cid = candidate_ids[idx]
            score = float(scores[idx])
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

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self._chunks.get(chunk_id)

    def delete_document(self, document_id: str) -> int:
        to_delete = [
            cid for cid, chunk in self._chunks.items()
            if chunk.metadata.document_id == document_id
        ]
        for cid in to_delete:
            del self._chunks[cid]
            del self._embeddings[cid]
        logger.info(f"Deleted {len(to_delete)} chunks for document '{document_id}'")
        return len(to_delete)

    def count_chunks(self) -> int:
        return len(self._chunks)

