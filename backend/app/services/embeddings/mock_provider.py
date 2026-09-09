"""Deterministic, zero-dependency mock embedding provider for tests and offline development."""
import hashlib
import math
from typing import List
from backend.app.services.embeddings.base import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Generates deterministic, L2-normalized dense embeddings without external API calls."""

    def __init__(self, dimension: int = 1536, model_name: str = "mock-embedding-v1"):
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _hash_token_to_index(self, token: str) -> int:
        """Map a token string to a pseudo-random dimension index [0, dimension - 1]."""
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self._dimension

    def _generate_vector(self, text: str) -> List[float]:
        """Produce a deterministic normalized vector based on text token frequencies and hash seeding."""
        vec = [0.0] * self._dimension
        tokens = text.lower().split()
        
        if not tokens:
            return vec

        # Project token occurrences into the high-dimensional space
        for token in tokens:
            idx = self._hash_token_to_index(token)
            # Add signed weight derived from hash
            weight = 1.0 + (int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:4], 16) % 10) / 10.0
            vec[idx] += weight

        # Add a document-level hash signature for smooth distribution
        doc_hash = hashlib.sha256(text.encode("utf-8")).digest()
        for i in range(min(len(doc_hash), self._dimension)):
            vec[i] += (doc_hash[i] / 255.0) * 0.1

        # L2-normalize vector: v / ||v||_2
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]

        return vec

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic normalized vectors for a batch of texts."""
        return [self._generate_vector(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")
        return self._generate_vector(query)

