"""Abstract Base Embedding Provider interface."""
from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Contract for text embedding model providers (OpenAI, local models, mock)."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Compute dense vector embeddings for a batch of text chunks.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of float vectors, each of length `dimension`.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Compute dense vector embedding for a single search query string.
        
        Args:
            query: The user query string.
            
        Returns:
            Float vector of length `dimension`.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the vector space (e.g. 1536 for text-embedding-3-small)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier name of the underlying embedding model."""
        pass

