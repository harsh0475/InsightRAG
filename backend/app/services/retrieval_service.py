"""Retrieval coordination service combining EmbeddingProvider and VectorStore."""
import logging
from typing import List, Optional

from backend.app.core.config import get_settings
from backend.app.schemas.document import DocumentChunk
from backend.app.schemas.retrieval import VectorSearchResult
from backend.app.services.embeddings import BaseEmbeddingProvider, get_embedding_provider
from backend.app.services.vector_store import BaseVectorStore, get_vector_store

logger = logging.getLogger("insightrag.services.retrieval")


class RetrievalService:
    """Orchestrates dense semantic search: query embedding -> vector similarity -> Top-K."""

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        settings = get_settings()
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.vector_store = vector_store or get_vector_store()
        self.default_top_k = settings.DEFAULT_TOP_K

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Embed a batch of document chunks and index them in the vector store."""
        if not chunks:
            return 0

        logger.info(f"Generating dense embeddings for {len(chunks)} chunks using {self.embedding_provider.model_name}")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_provider.embed_texts(texts)

        logger.info(f"Indexing {len(chunks)} vectors into vector store")
        indexed_count = self.vector_store.index_chunks(chunks, embeddings)

        # Also synchronize BM25 index for sparse and hybrid search
        try:
            from backend.app.services.retrieval.bm25 import get_bm25_retriever
            get_bm25_retriever().index_chunks(chunks)
        except Exception as e:
            logger.warning(f"Could not index into BM25 retriever: {e}")

        return indexed_count

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        """Perform semantic retrieval for a natural language query.
        
        Args:
            query: User search query.
            top_k: Number of nearest chunks to retrieve.
            document_ids: Optional list of document_ids to filter by.
            
        Returns:
            List of VectorSearchResult objects sorted by descending cosine similarity.
        """
        k = top_k or self.default_top_k
        if not query or not query.strip():
            return []

        logger.info(f"Embedding query: '{query}'")
        query_embedding = self.embedding_provider.embed_query(query)

        logger.info(f"Searching top {k} nearest neighbors in vector store")
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            document_ids=document_ids,
        )
        return results

