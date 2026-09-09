"""Abstract Base Vector Store interface."""
from abc import ABC, abstractmethod
from typing import List, Optional
from backend.app.schemas.document import DocumentChunk
from backend.app.schemas.retrieval import VectorSearchResult


class BaseVectorStore(ABC):
    """Contract for vector databases (pgvector, in-memory, etc.)."""

    @abstractmethod
    def index_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
        """Store document chunks and their associated embedding vectors.
        
        Args:
            chunks: List of DocumentChunk objects with text and metadata.
            embeddings: Corresponding dense float vectors.
            
        Returns:
            Count of successfully stored chunks.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        """Perform nearest-neighbor vector similarity search.
        
        Args:
            query_embedding: Dense vector representation of search query.
            top_k: Maximum number of closest matches to return.
            document_ids: Optional list of document_ids to constrain search space.
            
        Returns:
            List of VectorSearchResult objects sorted by descending similarity score.
        """
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Fetch a specific chunk by its ID."""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document ID."""
        pass

    @abstractmethod
    def count_chunks(self) -> int:
        """Return total number of chunks currently indexed."""
        pass

